"""
Standalone Kokoro TTS smoke test - no FastAPI, no server integration.
Just the library turning text into audio, to measure real synthesis
latency on this laptop's CPU before wiring it into server.py.

Run with:
    python kokoro_smoke_test.py

Produces a .wav file per text sample and prints how long each one took.
"""

import time
from kokoro import KPipeline
import soundfile as sf

# 'a' = American English. This matches Section 88's decision to ship
# English-only for now (Dutch TTS is out of scope until Dutch
# transcription itself comes back).
LANG_CODE = "a"

# af_heart is Kokoro's default/flagship voice - good baseline to test
# with before comparing others.
VOICE = "af_heart"

# Two samples: a short one (roughly the length of a real /answer
# response) and a slightly longer one, so we get a sense of whether
# synthesis time scales roughly linearly with text length on this CPU -
# the same kind of question asked of faster-whisper/Nemotron before it.
SAMPLES = {
    "short": "A pawn can promote to a queen, rook, bishop, or knight when it reaches the far side of the board.",
    "longer": (
        "Castling is a special move that involves the king and one of the rooks. "
        "The king moves two squares toward the rook, and the rook moves to the square "
        "the king crossed. Castling is only legal if neither piece has moved before, "
        "there are no pieces between them, and the king is not in or passing through check."
    ),
}


def main():
    print(f"Loading Kokoro pipeline (lang_code='{LANG_CODE}')... this downloads the model on first run.")
    t_load_start = time.time()
    pipeline = KPipeline(lang_code=LANG_CODE)
    t_load_elapsed = time.time() - t_load_start
    print(f"Pipeline loaded in {t_load_elapsed:.2f}s.\n")

    for name, text in SAMPLES.items():
        word_count = len(text.split())
        print(f"--- Sample: {name} ({word_count} words) ---")
        print(f'"{text}"')

        t0 = time.time()
        generator = pipeline(text, voice=VOICE)
        chunk_count = 0
        for i, (graphemes, phonemes, audio) in enumerate(generator):
            out_path = f"kokoro_test_{name}_{i}.wav"
            sf.write(out_path, audio, 24000)
            chunk_count += 1
        elapsed = time.time() - t0

        print(f"Synthesis time: {elapsed:.2f}s across {chunk_count} audio chunk(s).")
        print(f"Saved to kokoro_test_{name}_*.wav\n")

    print("Done. Play the .wav files to confirm audio quality, and compare")
    print("the synthesis times above against how long a real /answer")
    print("response takes to arrive - that combined delay is what the")
    print("user will actually experience.")


if __name__ == "__main__":
    main()
