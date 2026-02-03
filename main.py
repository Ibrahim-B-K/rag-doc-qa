import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os

# LlamaIndex Settings
from llama_index.core import Settings

# We still import these for the embedding/DB logic
from data_loader import embed_texts 
from vector_db import QdrantStorage
from custom_types import RAGUpsertResult, RAGSearchResult

load_dotenv()

# --- CONFIGURATION ---
render_url = os.getenv("RENDER_EXTERNAL_URL")
if render_url:
    os.environ["INNGEST_SERVE_ORIGIN"] = render_url
    print(f"🚀 PRODUCTION MODE: Set INNGEST_SERVE_ORIGIN to {render_url}")

my_signing_key = os.getenv("INNGEST_SIGNING_KEY")
is_prod = my_signing_key is not None

# Initialize Client
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=is_prod,
    signing_key=my_signing_key,
    event_key=os.getenv("INNGEST_EVENT_KEY"),
    serializer=inngest.PydanticSerializer()
)

# --- Function 1: Ingest TEXT (Replaces the PDF File Handler) ---
@inngest_client.create_function(
    fn_id="RAG: Ingest Text",
    trigger=inngest.TriggerEvent(event="rag/ingest_text") # Listening for the new event
)
async def rag_ingest_text(ctx: inngest.Context):
    
    # Simple chunking logic to replace the file loader
    def _chunk_text(text: str, chunk_size=1000, overlap=100):
        if not text:
            return []
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    def _process_text(ctx: inngest.Context) -> RAGUpsertResult:
        raw_text = ctx.event.data["text"]
        filename = ctx.event.data["filename"]
        
        print(f"📄 Processing text from '{filename}' ({len(raw_text)} chars)...")

        chunks = _chunk_text(raw_text)
        
        if not chunks:
            print(f"⚠️ No text content found for {filename}")
            return RAGUpsertResult(inngested=0)

        # Create vectors
        vecs = embed_texts(chunks)
        
        # Create deterministic IDs
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": filename, "text": chunk} for chunk in chunks]
        
        # Save to Qdrant
        QdrantStorage().upsert(ids, vecs, payloads)
        print(f"✅ Indexed {len(chunks)} chunks for {filename}")
        
        return RAGUpsertResult(inngested=len(chunks))

    # We skip the "load" step since we already have text, just process it
    result = await ctx.step.run("chunk-embed-upsert", lambda: _process_text(ctx), output_type=RAGUpsertResult)
    return result.model_dump()


# --- Function 2: Query PDF (Unchanged) ---
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf(ctx: inngest.Context):
    
    def _search(question: str, top_k: int = 5):
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    async def _generate_answer(prompt: str):
        response = await Settings.llm.acomplete(prompt)
        return str(response)

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    answer_text = await ctx.step.run("llm-generate", lambda: _generate_answer(user_content))
    answer_text = answer_text.strip()

    return {"answer": answer_text, "sources": found.sources, "num_contexts": len(found.contexts)}


app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "alive", "message": "RAG Backend is running correctly"}

inngest.fast_api.serve(
    app, 
    inngest_client, 
    functions=[rag_ingest_text, rag_query_pdf] # <--- Updated to use the text function
)