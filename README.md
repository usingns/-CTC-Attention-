# -CTC-Attention-
A streaming ASR system combining CTC prefix beam search with triggered attention rescoring for real-time subtitle generation. Runs on CPU via ESPnet2 pretrained Conformer. Includes WASAPI loopback capture and always-on-top subtitle overlay for Windows 11.

# Combined CTC/Attention Streaming Decoding for Subtitles

Real-time subtitle generation from any system audio source on Windows 11,
using a joint CTC/Attention streaming ASR pipeline built on ESPnet2.

CTC prefix beam search runs continuously on audio chunks; a triggered
attention rescorer fires at confident boundaries to improve accuracy.
No GPU required — runs entirely on CPU using a pretrained LibriSpeech
Conformer model. Captures system audio via WASAPI loopback (browser,
media player, live streams) with no virtual cable software needed.

## Quick start
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install espnet espnet_model_zoo soundfile librosa soundcard
python download_model.py
python live_subtitle.py
