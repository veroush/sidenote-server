"""
Standalone LIVE MIC smoke test for Nemotron 3.5 ASR
(nvidia/nemotron-3.5-asr-streaming-0.6b), streaming mode.

Captures real microphone audio, feeds it chunk-by-chunk into the model's
streaming generate() following the official cache-aware streaming pattern,
and prints transcribed text as it comes in, with rough per-chunk timing.

No server integration yet -- this is purely to confirm streaming works live
on this Windows/CPU laptop before touching server.py.

NOTE: HuggingFace's docs mark this streaming API "experimental, subject to
change." If output looks wrong, that's the first place to suspect, not
necessarily this script's buffering logic.

Setup (PowerShell, from inside sidenote-server\\):
    pip install sounddevice --break-system-packages

Usage:
    python nemotron_smoke_test.py en-US
    python nemotron_smoke_test.py nl-NL
"""

import sys
import time
import queue
import threading
from threading import Thread

import numpy as np
import sounddevice as sd
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# How long to listen before stopping. Bump this up once the basic plumbing
# is confirmed working -- 20s is just enough to say a sentence or two.
RECORD_SECONDS = 20

# One of the model's supported presets: {3: 320ms, 0: 80ms, 6: 560ms, 13: 1120ms}
# Higher = more accurate but higher latency. Testing 0 (80ms) here to see whether
# less lookahead meaningfully speeds up CPU inference, since 6 (560ms) measured
# ~2.5x slower than real-time on this laptop.
NUM_LOOKAHEAD_TOKENS = 0


def main():
    locale = sys.argv[1] if len(sys.argv) > 1 else "en-US"

    print(f"Loading processor and model ({MODEL_ID}) -- ~5.66GB, first run downloads it...")
    t_load_start = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID, device_map="cpu")
    processor.set_num_lookahead_tokens(NUM_LOOKAHEAD_TOKENS)
    print(f"Model loaded in {time.time() - t_load_start:.1f}s")
    print(f"Streaming latency setting: {processor.streaming_latency_ms} ms "
          f"(available: {processor.supported_streaming_latencies_ms})")

    sampling_rate = processor.feature_extractor.sampling_rate
    audio_queue = queue.Queue()
    stop_event = threading.Event()

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[mic status] {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    chunk_send_times = []
    text_receive_times = []

    def safe_slice(buf, start, end):
        # start_idx can go negative for small first-chunk configs (e.g. 80ms lookahead),
        # since it's computed as mel_frame_idx * hop_length - n_fft // 2. A plain buf[start:end]
        # would silently wrap around to the wrong (and much smaller) slice in that case --
        # that was the actual bug behind the earlier "stft expected n_fft < 16" crash.
        # Zero-pad the missing leading context instead, mirroring the "centered" padding
        # already applied automatically to the first chunk.
        if start < 0:
            pad = np.zeros(-start, dtype=np.float32)
            return np.concatenate([pad, buf[:end]])
        return buf[start:end]

    def input_features_generator():
        buffer = np.zeros((0,), dtype=np.float32)

        def wait_for(n_samples):
            nonlocal buffer
            while buffer.shape[0] < n_samples and not stop_event.is_set():
                try:
                    piece = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                buffer = np.concatenate([buffer, piece])

        # -- first chunk --
        wait_for(processor.num_samples_first_audio_chunk)
        if stop_event.is_set():
            return
        chunk_send_times.append(time.time())
        first_chunk_inputs = processor(
            buffer[: processor.num_samples_first_audio_chunk],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=locale,
            return_tensors="pt",
        )
        first_chunk_inputs.to(model.device, dtype=model.dtype)
        yield first_chunk_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]

        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        start_idx = mel_frame_idx * hop_length - n_fft // 2

        while not stop_event.is_set():
            end_idx = start_idx + processor.num_samples_per_audio_chunk
            wait_for(end_idx)
            if stop_event.is_set():
                break
            chunk_send_times.append(time.time())
            inputs = processor(
                safe_slice(buffer, start_idx, end_idx),
                sampling_rate=sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=locale,
                return_tensors="pt",
            )
            inputs.to(model.device, dtype=model.dtype)
            yield inputs.input_features

            mel_frame_idx += processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * hop_length - n_fft // 2

        return  # store first_chunk_inputs for generate_kwargs below (see closure)

    # We need first_chunk_inputs' non-input_features fields (e.g. num_lookahead_tokens,
    # attention_mask) available *before* the generator starts, per the documented pattern.
    # Do a tiny synchronous priming read for those fields, then let the real generator run.
    print(f"\nListening for {RECORD_SECONDS}s (locale={locale}) -- speak now...")

    with sd.InputStream(samplerate=sampling_rate, channels=1, dtype="float32", callback=mic_callback):
        gen = input_features_generator()
        first_features = next(gen)  # blocks until first chunk of real mic audio arrives

        streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)

        def feed():
            yield first_features
            yield from gen

        # Build generate_kwargs: reuse the processor's non-audio fields (num_lookahead_tokens
        # etc.) from a fresh first-chunk call, since the priming read above already consumed
        # the generator's first yield.
        priming_inputs = processor(
            np.zeros((processor.num_samples_first_audio_chunk,), dtype=np.float32),
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=locale,
            return_tensors="pt",
        )
        priming_inputs.to(model.device, dtype=model.dtype)

        generate_kwargs = {
            **priming_inputs,
            "input_features": feed(),
            "streamer": streamer,
        }

        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        t_start = time.time()
        thread.start()

        def stop_after_timeout():
            time.sleep(RECORD_SECONDS)
            stop_event.set()

        Thread(target=stop_after_timeout, daemon=True).start()

        print("\n--- LIVE TRANSCRIPT ---")
        full_text = ""
        for text_chunk in streamer:
            text_receive_times.append(time.time())
            print(text_chunk, end="", flush=True)
            full_text += text_chunk

        thread.join(timeout=5)

    print("\n\n--- SUMMARY ---")
    print(f"Full transcript: {full_text}")
    print(f"Audio chunks sent: {len(chunk_send_times)}")
    print(f"Text chunks received: {len(text_receive_times)}")
    if chunk_send_times and text_receive_times:
        print(f"Time from listen-start to first text: {text_receive_times[0] - t_start:.2f}s")
        print(f"Time from listen-start to last text: {text_receive_times[-1] - t_start:.2f}s")
    if len(chunk_send_times) > 1:
        print("\nGap between consecutive audio chunks being sent to the model (seconds):")
        for i in range(1, len(chunk_send_times)):
            gap = chunk_send_times[i] - chunk_send_times[i - 1]
            print(f"  chunk {i-1} -> {i}: {gap:.2f}s")
        print(f"(Each chunk represents ~0.58s of real audio -- gaps much larger than that "
              f"mean the model is slower than real-time on this CPU.)")


if __name__ == "__main__":
    main()