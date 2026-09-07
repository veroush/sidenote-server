"""
Sidenote transcription server.

Run this on your laptop, then open it from your phone's browser while
on the same wifi network (e.g. http://192.168.1.42:8000).

Setup (first time only):
    pip install -r requirements.txt

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000

Then find your laptop's local IP address:
    Mac:     System Settings -> Wi-Fi -> Details -> IP Address
    Windows: run `ipconfig` in Command Prompt, look for "IPv4 Address"
And on your phone (same wifi), visit: http://THAT_IP:8000
"""

from dotenv import load_dotenv

load_dotenv()

import io
import re
from pathlib import Path
from groq_answer import get_concise_answer, get_list_tool_response
from groq_speak import synthesize_speech

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from faster_whisper import WhisperModel

from collections import Counter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI()

# Not strictly needed since everything is served from one origin, but harmless
# and useful if you ever want to call this from a different host/page.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# "base" is what's actually been tested for real end-to-end latency this
# session (~2.5-3s transcribe time). "small" measured far slower and far
# less consistent on this hardware (9-35s) despite being marketed as only
# a modest step up â€” not worth it here. If accuracy ever needs revisiting,
# retest candidates properly with latency_test.py before switching this.
MODEL_SIZE = "base"
print("Loading Whisper model (\"" + MODEL_SIZE + "\")... on a first run this downloads the model and can take a few minutes. Nothing wrong if this line sits here a while.")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded. Starting server...")
DATA_DIR = Path(__file__).parent / "data"

print("Loading PDFs from data/ and building retrieval index...")
_pdf_docs = []
for pdf_path in DATA_DIR.glob("*.pdf"):
    print(f"  - {pdf_path.name}")
    _pdf_docs.extend(PyPDFLoader(str(pdf_path)).load())

if _pdf_docs:
    _splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    doc_chunks = _splitter.split_documents(_pdf_docs)
    print(f"Indexed {len(doc_chunks)} chunk(s) from {len(_pdf_docs)} page(s).")
    _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(doc_chunks, _embeddings)
else:
    print("Warning: no PDFs found in data/ â€” retrieval will return nothing.")
    doc_chunks = []
    vectorstore = None

print("Index ready.")

LANG_MAP = {"en-US": "en", "nl-NL": "nl"}

_HOTWORD_STOPWORDS = {
    "the","a","an","is","are","was","were","and","or","to","of","in","on","for",
    "with","as","at","by","from","that","this","these","those","be","been",
    "being","will","would","could","should","not","no","yes","so","but","it",
    "its","if","when","where","which","who","whom","their","they","he","she",
    "you","your","our","we","i","also","can","may","might","must","one","two",
    "than","then","there","here","such","into","over","under","more","most",
}

def build_hotwords_from_docs() -> str:
    """Pull frequent, meaningful terms out of the indexed PDF content to bias
    Whisper toward domain vocabulary â€” replaces the old topics-keyword
    version now that topics are going away."""
    counts = Counter()
    for chunk in doc_chunks:
        words = re.findall(r"[a-zA-Z]{4,}", chunk.page_content.lower())
        for w in words:
            if w not in _HOTWORD_STOPWORDS:
                counts[w] += 1
    return ", ".join(w for w, _ in counts.most_common(40))

HOTWORDS = build_hotwords_from_docs()
print(f"Hotwords built from PDF content: {HOTWORDS[:120]}{'...' if len(HOTWORDS) > 120 else ''}")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str = Form("en-US")):
    audio_bytes = await file.read()
    print(f"DEBUG: received {len(audio_bytes)} bytes of audio")

    hotwords = HOTWORDS

    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language="en",
        beam_size=5,
        vad_filter=True,  # built-in silence/noise detection â€” skips dead air
                           # and cuts down on hallucinated text on quiet audio
        hotwords=hotwords if hotwords else None,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"DEBUG: transcribed text = {text!r}")
    return {"text": text}


class AnswerRequest(BaseModel):
    question: str
    notes: str | None = None  # no longer used -- kept optional so the current
                               # frontend (which still sends it) doesn't break


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


@app.post("/answer")
async def answer(req: AnswerRequest):
    try:
        list_answer = get_list_tool_response(req.question)
        if list_answer is not None:
            return {"answer": list_answer, "source": "list"}
    except Exception as e:
        print(f"List tool check failed: {e}")
        # fall through to RAG path below

    if vectorstore is None:
        raise HTTPException(status_code=503, detail="No documents indexed in data/ -- nothing to answer from.")
    try:
        retrieved = vectorstore.similarity_search(req.question, k=5)
        context = format_docs(retrieved)
        return {"answer": get_concise_answer(req.question, context), "source": "groq"}
    except Exception as e:
        print(f"RAG answer failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))


class SpeakRequest(BaseModel):
    text: str


@app.post("/speak")
async def speak(req: SpeakRequest):
    try:
        audio_bytes = synthesize_speech(req.text)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        print(f"TTS failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))

# Serves static/index.html (and everything else in static/) at the root.
# This must be mounted last â€” routes above take priority over static files.
app.mount("/", StaticFiles(directory="static", html=True), name="static")