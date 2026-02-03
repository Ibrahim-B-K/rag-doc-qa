import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

# Check if we are running on Streamlit Cloud by looking for secrets
IS_CLOUD = st.secrets.get("INNGEST_EVENT_KEY") is not None

if IS_CLOUD:
    # PRODUCTION SETTINGS (Streamlit Cloud)
    inngest_key = st.secrets["INNGEST_EVENT_KEY"]
    inngest_url = "https://api.inngest.com/v1"
    is_prod = True
else:
    # LOCAL SETTINGS (Your Laptop)
    # This assumes you are running 'npx inngest-cli dev' locally
    inngest_key = "test"  # Local dev server doesn't check keys
    inngest_url = "http://127.0.0.1:8288/v1"
    is_prod = False


def get_inngest_client() -> inngest.Inngest:
    # UPDATED: Use the Event Key from secrets (we will set this up next)
    return inngest.Inngest(
        app_id="rag_app",
        is_production=is_prod, # <--- IMPORTANT: We are now in Production!
        event_key=inngest_key,
        
    )


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    # Generate a unique ID for this file so we don't process it twice
    file_id = f"processed_{uploaded.name}_{uploaded.size}"

    if file_id not in st.session_state:
        with st.spinner("Uploading and triggering ingestion..."):
            path = save_uploaded_pdf(uploaded)
            asyncio.run(send_rag_ingest_event(path))
            time.sleep(0.3)
            # Mark as done
            st.session_state[file_id] = True
            
        st.success(f"Triggered ingestion for: {path.name}")
    else:
        st.info(f"Already processed: {uploaded.name}")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]


def _inngest_api_base() -> str:
    # UPDATED: Point to the real Inngest Cloud API
    return inngest_url


def fetch_runs(event_id: str) -> list[dict]:
    base_url = _inngest_api_base()
    url = f"{base_url}/events/{event_id}/runs"
    
    headers = {}
    
    # If we are talking to the real Inngest Cloud, we MUST have the Signing Key
    if "api.inngest.com" in base_url:
        # Try to get key from Streamlit secrets (Cloud) or env vars (Local)
        signing_key = st.secrets.get("INNGEST_SIGNING_KEY") or os.getenv("INNGEST_SIGNING_KEY")
        
        if signing_key:
            headers["Authorization"] = f"Bearer {signing_key}"
        else:
            st.error("Missing INNGEST_SIGNING_KEY in secrets!")

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
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


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            # Poll the local Inngest API for the run's output
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
