import asyncio
from pathlib import Path
import time
import os

import streamlit as st
import inngest
from dotenv import load_dotenv
import requests
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

# --- CONFIGURATION ---
IS_CLOUD = st.secrets.get("INNGEST_EVENT_KEY") is not None

if IS_CLOUD:
    inngest_key = st.secrets["INNGEST_EVENT_KEY"]
    inngest_url = "https://api.inngest.com/v1"
    is_prod = True
else:
    inngest_key = "test"
    inngest_url = "http://127.0.0.1:8288/v1"
    is_prod = False

def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        is_production=is_prod,
        event_key=inngest_key,
    )

# --- ASYNC EVENT SENDERS ---

async def send_rag_ingest_text_event(filename: str, text: str) -> None:
    """
    Async function to send the text event. 
    Must be called with asyncio.run() from the main script.
    """
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_text", # <--- Updated Event Name
            data={
                "filename": filename,
                "text": text, # <--- Sending actual text
            },
        )
    )

async def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    ids = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return ids[0]

# --- HELPER FUNCTIONS ---

def _inngest_api_base() -> str:
    return inngest_url

def fetch_runs(event_id: str) -> list[dict]:
    base_url = _inngest_api_base()
    url = f"{base_url}/events/{event_id}/runs"
    
    headers = {}
    if "api.inngest.com" in base_url:
        signing_key = st.secrets.get("INNGEST_SIGNING_KEY") or os.getenv("INNGEST_SIGNING_KEY")
        if signing_key:
            headers["Authorization"] = f"Bearer {signing_key}"
        else:
            st.error("Missing INNGEST_SIGNING_KEY in secrets!")

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])

def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 1.0) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        
        time.sleep(poll_interval_s)

# --- UI LAYOUT ---

st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    # Unique ID to prevent re-processing on every redraw
    file_id = f"processed_{uploaded.name}_{uploaded.size}"

    if file_id not in st.session_state:
        with st.spinner("Extracting text and triggering ingestion..."):
            
            # 1. Extract Text Locally
            try:
                reader = PdfReader(uploaded)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
                
                # 2. Send Event
                # asyncio.run is required because we are calling async from sync context
                asyncio.run(send_rag_ingest_text_event(uploaded.name, text_content))
                
                # 3. Mark as done
                st.session_state[file_id] = True
                st.success(f"Successfully extracted {len(text_content)} characters and sent to backend!")
                
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
            
    else:
        st.info(f"Already processed: {uploaded.name}")

st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            try:
                event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
                output = wait_for_run_output(event_id)
                answer = output.get("answer", "")
                sources = output.get("sources", [])

                st.subheader("Answer")
                st.write(answer or "(No answer received)")
                
                if sources:
                    st.caption("Sources")
                    for s in sources:
                        st.write(f"- {s}")
            except Exception as e:
                st.error(f"Error during query: {e}")