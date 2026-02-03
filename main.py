import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
# Remove 'from inngest.experimental import ai' - we don't need this wrapper for Gemini
from dotenv import load_dotenv
import uuid
import os
import datetime

# LlamaIndex Settings (This holds our Gemini configuration)
from llama_index.core import Settings

from data_loader import load_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

# --- Function 1: Ingest PDF ---
@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_inngest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_chunk_pdf(pdf_path)
        # Fix: Ensure source_id is passed if your updated custom_types expects it
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
    
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        
        # Check for empty chunks to prevent Qdrant 400 Error
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


# --- Function 2: Query PDF (The New Part) ---
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf(ctx: inngest.Context):
    
    # 1. Search the Vector DB
    def _search(question: str, top_k: int = 5):
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    # 2. Generate Answer with Gemini
    async def _generate_answer(prompt: str):
        # CHANGE 2: Use 'acomplete' (Async) instead of 'complete' (Sync)
        # This uses the existing event loop instead of trying to start a new one
        response = await Settings.llm.acomplete(prompt)
        return str(response)

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # Step 1: Get Context
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # Prepare the prompt
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    # Step 2: Ask Gemini (Replaces the OpenAI Adapter)
    answer_text = await ctx.step.run("llm-generate", lambda: _generate_answer(user_content))
    
    # Clean up the string (Replaces .stril())
    answer_text = answer_text.strip()

    return {"answer": answer_text, "sources": found.sources, "num_contexts": len(found.contexts)}


app = FastAPI()

# IMPORTANT: You must register BOTH functions here!
inngest.fast_api.serve(app, inngest_client, functions=[rag_inngest_pdf, rag_query_pdf])