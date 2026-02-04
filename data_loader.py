import os
from dotenv import load_dotenv
from llama_index.core import Settings

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


Settings.llm = Gemini(
    model="models/gemini-flash-latest", 
    api_key=api_key
)


Settings.embed_model = GeminiEmbedding(
    model_name="models/text-embedding-004",
    api_key=api_key
)

# ---  CONFIGURATION ---
EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# --- LOAD PDF ---
def load_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

# --- EMBED TEXTS ---
def embed_texts(texts: list[str]) -> list[list[float]]:
    return Settings.embed_model.get_text_embedding_batch(texts)