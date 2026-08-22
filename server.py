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

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app = FastAPI()

# Not strictly needed since everything is served from one origin, but harmless
# and useful if you ever want to call this from a different host/page.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# "small" is a solid accuracy/speed balance on a modern laptop CPU.
# If it's too slow: try "base". If your laptop can handle more and you
# want even better accuracy: try "medium".
MODEL_SIZE = "small"
print("Loading Whisper model (\"" + MODEL_SIZE + "\")... on a first run this downloads the model and can take a few minutes. Nothing wrong if this line sits here a while.")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded. Starting server...")

LANG_MAP = {"en-US": "en", "nl-NL": "nl"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str = Form("en-US")):
    audio_bytes = await file.read()

    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=LANG_MAP.get(lang, "en"),
        beam_size=5,
        vad_filter=True,  # built-in silence/noise detection — skips dead air
                           # and cuts down on hallucinated text on quiet audio
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": text}


# Serves static/index.html (and everything else in static/) at the root.
# This must be mounted last — routes above take priority over static files.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
