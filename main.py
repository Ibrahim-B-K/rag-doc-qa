import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime

# LlamaIndex Settings
from llama_index.core import Settings

from data_loader import load_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()

# --- FIX 1: Set the Serve Origin via Environment Variable ---
# This tells Inngest: "My public URL is https://rag-doc-qa-w2xn.onrender.com"
# The SDK automatically appends "/api/inngest" to this.
render_url = os.getenv("RENDER_EXTERNAL_URL")
if render_url:
    os.environ["INNGEST_SERVE_ORIGIN"] = render_url
    print(f"🚀 PRODUCTION MODE: Set INNGEST_SERVE_ORIGIN to {render_url}")

# --- FIX 2: Detect Environment ---
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

# --- Function 1: Ingest PDF ---
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_inngest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
    
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        
        if not chunks:
            print(f"⚠️ PDF '{source_id}' contained no text! Skipping.")
            return RAGUpsertResult(inngested=0)

        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(inngested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    inngested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return inngested.model_dump()


# --- Function 2: Query PDF ---
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

# Health Check
@app.get("/")
def read_root():
    return {"status": "alive", "message": "RAG Backend is running correctly"}

# --- SERVER REGISTRATION ---
inngest.fast_api.serve(
    app, 
    inngest_client, 
    functions=[rag_inngest_pdf, rag_query_pdf]
    # REMOVED the invalid 'serve_url' argument
)