# CTC-Attention
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

# ESPnet2 Install Guide (CPU / No Dedicated GPU)
## Combined CTC/Attention Streaming ASR Project

This version skips Kaldi, skips GPU drivers, and skips all training.
You will install just the Python packages, download a pretrained model,
and go straight to experimenting with decoding and the subtitle engine.

---

## What you need

- Python 3.9 or 3.10 (3.11+ may have package conflicts — see troubleshooting)
- ~4GB free disk space for the pretrained model
- Internet connection for the one-time model download

No CUDA, no Kaldi, no make, no GPU required.

---

## Step 1 — Check your Python version

```bash
python3 --version
```

- **3.9 or 3.10**: perfect, proceed.
- **3.11 or 3.12**: use pyenv to install 3.10 alongside your system Python (see troubleshooting).
- **3.8 or below**: upgrade first — some dependencies require 3.9+.

---

## Step 2 — Create a virtual environment

Always work inside a virtual environment to avoid breaking other projects:

```bash
python3 -m venv espnet_env
source espnet_env/bin/activate        # Linux / macOS
# espnet_env\Scripts\activate         # Windows (PowerShell)
```

You should see `(espnet_env)` at the start of your terminal prompt.
Run this activation command every time you open a new terminal session.

---

## Step 3 — Install PyTorch (CPU build)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchaudio; print(torchaudio.__version__)"
```

Both should print a version number without errors.

---

## Step 4 — Install ESPnet and utilities

```bash
pip install espnet espnet_model_zoo
pip install soundfile librosa
```

Verify:

```bash
python -c "import espnet; print('ESPnet OK')"
python -c "from espnet_model_zoo.downloader import ModelDownloader; print('Model zoo OK')"
```

---

## Step 5 — Download the pretrained model

```bash
python download_model.py
```

This downloads a pretrained LibriSpeech Conformer (~460MB) trained with joint
CTC/Attention and caches it in `~/.cache/espnet/`. Happens once; future runs
use the cache.

Expected output:
```
Downloading pretrained model...
Model ready. Vocabulary size: 5000
```

---

## Step 6 — Confirm everything works

Run the inference script on any short WAV file (5–30 seconds, 16kHz mono):

```bash
python inference.py --audio your_audio.wav
```

Expected output:
```
Transcription: the quick brown fox jumps over the lazy dog
```

---

## Step 7 — Run the streaming decoder

```bash
python streaming_decode.py --audio your_audio.wav --chunk_ms 320 --output subtitles.srt
```

This is the core of the project: CTC prefix beam search over audio chunks,
triggered attention rescoring, and subtitle output. See streaming_decode.py
for the parameters to tune.

---

## Performance expectations on CPU

| Audio length | Approximate inference time |
|---|---|
| 5 seconds | 3–8 seconds |
| 30 seconds | 20–50 seconds |
| 5 minutes | 3–8 minutes |

CPU inference is not real-time. That is fine for learning and experimentation.
The decoding logic and subtitle output are identical to what would run on GPU;
only the speed differs.

---

## Troubleshooting

**"No module named espnet"** — the virtual environment is not active:
```bash
source espnet_env/bin/activate
```

**pip install fails on Python 3.11/3.12** — install Python 3.10 via pyenv:
```bash
curl https://pyenv.run | bash
pyenv install 3.10.13
pyenv local 3.10.13
python -m venv espnet_env
source espnet_env/bin/activate
# Then repeat steps 3–4
```

**Model download times out** — the model is hosted on Hugging Face; retry or use:
```bash
pip install huggingface_hub
huggingface-cli download espnet/librispeech_asr_train_asr_conformer
```

**"could not load audio file"** — install ffmpeg for broader format support:
```bash
sudo apt-get install ffmpeg      # Ubuntu/Debian
brew install ffmpeg              # macOS
# Windows: https://ffmpeg.org/download.html
```

**Inference is very slow** — reduce beam size for faster iteration:
```bash
python inference.py --audio test.wav --beam_size 5
```
Use beam_size=5 while developing, beam_size=60 for final WER measurements.
