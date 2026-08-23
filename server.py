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

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

from whisperlivekit import AudioProcessor, TranscriptionEngine

# ---------------------------------------------------------------------------
# OLD backend (faster-whisper) — kept only as a fallback while we confirm the
# new /asr WebSocket (Voxtral) works. Safe to delete once /asr is verified.
# ---------------------------------------------------------------------------
MODEL_SIZE = "small"
print("Loading faster-whisper model (\"" + MODEL_SIZE + "\") [fallback /transcribe route]...")
old_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("faster-whisper model loaded.")

LANG_MAP = {"en-US": "en", "nl-NL": "nl"}

# ---------------------------------------------------------------------------
# NEW backend (Voxtral via WhisperLiveKit) — real-time streaming, per-chunk
# language re-detection instead of a manual toggle.
# ---------------------------------------------------------------------------
transcription_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    print(
        "Loading Voxtral model (backend='voxtral') via WhisperLiveKit... "
        "on a first run this downloads several GB from Hugging Face and can "
        "take a while. Nothing wrong if this line sits here a long time."
    )
    transcription_engine = TranscriptionEngine(
        backend="voxtral",   # HF Transformers Voxtral Mini — works on CPU/CUDA/MPS
        lan="auto",          # per-chunk language auto-detection (EN <-> NL switching)
        pcm_input=True,      # we send raw 16kHz PCM directly, no ffmpeg needed
    )
    print("Voxtral model loaded. Server ready.")
    yield


app = FastAPI(lifespan=lifespan)

# Not strictly needed since everything is served from one origin, but harmless
# and useful if you ever want to call this from a different host/page.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str = Form("en-US")):
    """OLD fallback route — batch faster-whisper, one shot per 4s chunk."""
    audio_bytes = await file.read()

    segments, _info = old_model.transcribe(
        io.BytesIO(audio_bytes),
        language=LANG_MAP.get(lang, "en"),
        beam_size=5,
        vad_filter=True,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": text}


@app.websocket("/asr")
async def websocket_asr(websocket: WebSocket):
    """NEW route — persistent streaming connection, Voxtral backend.

    Client sends raw 16-bit PCM, 16kHz, mono audio bytes continuously.
    Server sends back JSON messages as transcription updates arrive.
    """
    global transcription_engine

    audio_processor = AudioProcessor(transcription_engine=transcription_engine)
    await websocket.accept()
    print("WebSocket /asr connection opened.")

    results_generator = await audio_processor.create_tasks()

    async def send_results():
        try:
            async for response in results_generator:
                await websocket.send_json(response.to_dict())
            await websocket.send_json({"type": "ready_to_stop"})
        except WebSocketDisconnect:
            pass

    import asyncio
    sender_task = asyncio.create_task(send_results())

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except WebSocketDisconnect:
        print("WebSocket /asr disconnected by client.")
    except Exception as e:
        print(f"WebSocket /asr error: {e}")
    finally:
        if not sender_task.done():
            sender_task.cancel()
        await audio_processor.cleanup()
        print("WebSocket /asr cleaned up.")


# Serves static/index.html (and everything else in static/) at the root.
# This must be mounted last — routes above take priority over static files.
app.mount("/", StaticFiles(directory="static", html=True), name="static")