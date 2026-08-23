import sys
from transformers import AutoProcessor

lookahead = int(sys.argv[1]) if len(sys.argv) > 1 else 6

processor = AutoProcessor.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b")
processor.set_num_lookahead_tokens(lookahead)

sr = processor.feature_extractor.sampling_rate
print(f"NUM_LOOKAHEAD_TOKENS: {lookahead}")
print(f"sampling_rate: {sr}")
print(f"num_samples_first_audio_chunk: {processor.num_samples_first_audio_chunk} "
      f"({processor.num_samples_first_audio_chunk / sr:.3f}s)")
print(f"num_samples_per_audio_chunk: {processor.num_samples_per_audio_chunk} "
      f"({processor.num_samples_per_audio_chunk / sr:.3f}s)")
print(f"num_mel_frames_first_audio_chunk: {processor.num_mel_frames_first_audio_chunk}")
print(f"num_mel_frames_per_audio_chunk: {processor.num_mel_frames_per_audio_chunk}")
print(f"hop_length: {processor.feature_extractor.hop_length}")
print(f"n_fft: {processor.feature_extractor.n_fft}")