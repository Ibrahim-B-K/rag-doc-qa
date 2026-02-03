import os
from dotenv import load_dotenv
from llama_index.core import Settings
# We are using the stable Gemini driver
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# --- 1. SETUP LLM & EMBEDDING ---
# CRITICAL FIX: Use the simple alias "gemini-1.5-flash". 
# Do NOT add "models/" and do NOT add "-001".
Settings.llm = Gemini(
    model="models/gemini-flash-latest", 
    api_key=api_key
)

# For embeddings, we also strip the prefix to be safe with this driver
Settings.embed_model = GeminiEmbedding(
    model_name="models/text-embedding-004",
    api_key=api_key
)

# --- 2. CONFIGURATION ---
EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# --- 3. LOAD PDF ---
def load_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

# --- 4. EMBED TEXTS ---
def embed_texts(texts: list[str]) -> list[list[float]]:
    return Settings.embed_model.get_text_embedding_batch(texts)