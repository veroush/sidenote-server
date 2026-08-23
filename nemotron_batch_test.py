"""
Standalone BATCH (non-streaming) smoke test for Nemotron 3.5 ASR
(nvidia/nemotron-3.5-asr-streaming-0.6b).

Records one short burst of live mic audio, then runs a single non-streaming
transcribe call over the whole burst -- no chunked generator, no cache-aware
streaming complexity. Tests whether "record short burst, transcribe after a
pause" is viable on this CPU, now that true live streaming (nemotron_smoke_test.py)
measured ~2.3x-4.1x slower than real-time at both lookahead settings tried.

No server integration yet -- standalone script only.

Usage (PowerShell, from inside sidenote-server\\):
    python nemotron_batch_test.py en-US
    python nemotron_batch_test.py nl-NL 8
    (second arg optional: burst length in seconds, default 5)
"""

import sys
import time

import numpy as np
import sounddevice as sd
from transformers import AutoModelForRNNT, AutoProcessor

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
DEFAULT_BURST_SECONDS = 5


def main():
    locale = sys.argv[1] if len(sys.argv) > 1 else "en-US"
    burst_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BURST_SECONDS

    print(f"Loading processor and model ({MODEL_ID})...")
    t_load_start = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID, device_map="cpu")
    print(f"Model loaded in {time.time() - t_load_start:.1f}s")

    sampling_rate = processor.feature_extractor.sampling_rate

    print(f"\nRecording {burst_seconds}s burst (locale={locale}) -- speak now...")
    recording = sd.rec(
        int(burst_seconds * sampling_rate),
        samplerate=sampling_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = recording[:, 0]
    print("Recording finished.")

    print("Transcribing (single batch call, no streaming)...")
    t_transcribe_start = time.time()
    inputs = processor(
        audio,
        sampling_rate=sampling_rate,
        language=locale,
        return_tensors="pt",
    )
    generated_ids = model.generate(**inputs)
    print(f"\n[DEBUG] type(generated_ids): {type(generated_ids)}")
    print(f"[DEBUG] generated_ids: {generated_ids}")
    transcript = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
    t_transcribe_end = time.time()

    transcribe_time = t_transcribe_end - t_transcribe_start
    rtf = transcribe_time / burst_seconds

    print("\n--- RESULT ---")
    print(f"Transcript: {transcript}")
    print(f"Burst length: {burst_seconds}s")
    print(f"Transcription time: {transcribe_time:.2f}s")
    print(f"Real-time factor: {rtf:.2f}x "
          f"({'FASTER than real-time' if rtf < 1 else 'SLOWER than real-time'})")


if __name__ == "__main__":
    main()