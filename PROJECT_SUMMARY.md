# Combined CTC/Attention Streaming Decoding — Project Summary
### Personal Learning Project · CPU Setup · Windows 11

---

## What this project is

An end-to-end ASR (Automatic Speech Recognition) system that generates
real-time subtitles from any audio source. The core research contribution
is a streaming decoding algorithm that combines two complementary approaches:

- **CTC** (Connectionist Temporal Classification) — fast, always-on,
  naturally left-to-right. Good for streaming but less accurate.
- **Attention decoder** — more accurate, but needs context. Used selectively.

The system runs CTC continuously and triggers the attention decoder only at
confident boundary points, getting the best of both: low latency from CTC,
accuracy from attention.

---

## System architecture

```
Audio stream
    │
    ▼
┌─────────────────────────────┐
│     Streaming encoder       │  Chunk-Conformer with limited lookahead
└────────┬────────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌──────────────────┐
│  CTC  │  │ Attention decoder│
│  head │──▶  (triggered)     │
└───┬───┘  └────────┬─────────┘
    │               │
    └──────┬────────┘
           ▼
   α·CTC + (1−α)·Attention
           │
           ▼
   ┌───────────────┐
   │ Subtitle engine│  Timestamps + SRT/WebVTT output
   └───────────────┘
```

**The trigger condition** (the key design decision) fires attention rescoring
when the CTC beam detects a confident boundary — a blank spike (silence),
a word boundary token, or a score peak. This keeps attention rescoring rare
and cheap, preserving low latency.

**Joint decoding score:**
```
score = α · log P_ctc(y|x) + (1 − α) · log P_attn(y|x)
```
Typical α: 0.5 for streaming (balanced), 0.7 for very low latency (CTC-heavy).

---

## Streaming decode loop (per chunk)

```
Initialize beam B = {∅}
─────────────────────────────────────────────────
For each audio chunk:
  1. Encode chunk → append H_t to encoder cache
  2. Update CTC prefix beam with frames in H_t
  3. Check trigger condition:
     ├── No  → wait for next chunk
     └── Yes → rescore top-K hyps with attention
                combine scores: α·CTC + (1-α)·Attn
                emit best hypothesis as subtitle segment
─────────────────────────────────────────────────
```

Three trigger strategies (ablation dimension):
| Strategy | Fires when… |
|---|---|
| `blank_spike` | Long blank run detected (silence/pause) |
| `word_boundary` | Top hypothesis ends with a word-boundary BPE token |
| `score_peak` | Top CTC log-prob crosses threshold θ |

---

## Files in this project

```
espnet2_baseline/
├── INSTALL.md                ← installation steps (CPU, no GPU)
├── PROJECT_SUMMARY.md        ← this file
├── download_model.py         ← step 1: fetch pretrained model
├── inference.py              ← step 2: verify offline ASR works
├── streaming_decode.py       ← step 3: file-based streaming + SRT output
├── live_subtitle.py          ← step 4: live system audio → overlay subtitles
└── conf/
    └── decode_asr.yaml       ← decoding parameters reference
```

### What each file does

**`download_model.py`**
Downloads the pretrained LibriSpeech Conformer (~200MB, cached after first run).
Model: `jkang/espnet2_librispeech_100_conformer`
Architecture: Conformer encoder + Transformer decoder, joint CTC/Attention,
5000 BPE tokens. WER ~6% on LibriSpeech test-clean.

**`inference.py`**
Offline transcription of any audio file. Also prints word-level timestamps
from CTC alignment — useful to understand the model before looking at streaming.

**`streaming_decode.py`**
Simulates streaming by processing an audio file in fixed-size chunks.
Implements the full pipeline: CTC prefix beam search → trigger detection →
attention rescoring → SRT output. Use this for the paper ablation table.

**`live_subtitle.py`**
Captures system audio via WASAPI loopback (whatever is playing through
your speakers — browser, media player, video call) and shows subtitles
in a transparent always-on-top overlay window in real time.

---

## Setup (complete, in order)

### 1. Install Python packages

```bash
python3 -m venv espnet_env
source espnet_env/bin/activate        # Windows: espnet_env\Scripts\activate

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install espnet espnet_model_zoo
pip install soundfile librosa soundcard
```

Python 3.9 or 3.10 recommended. If on 3.11/3.12, install 3.10 via pyenv first.

### 2. Download the pretrained model (one time)

```bash
python download_model.py
```

### 3. Verify offline inference works

```bash
python inference.py --audio your_audio.wav
```

### 4. Run streaming decode on a file

```bash
python streaming_decode.py --audio your_audio.wav --output subtitles.srt
```

### 5. Run live subtitles from system audio

```bash
# See available devices
python live_subtitle.py --list-devices

# Run (uses default speakers/headphones)
python live_subtitle.py
```

---

## Parameters and what to tune

| Parameter | Flag | Default | Effect |
|---|---|---|---|
| Chunk size | `--chunk_ms` | 320 | Latency knob. Smaller = faster but less accurate. |
| CTC weight | `--ctc_weight` | 0.5 | 0.0 = pure attention, 1.0 = pure CTC |
| Beam size | `--beam_size` | 10 | Accuracy vs speed. Use 5 for live, 60 for best WER. |
| Trigger | `--trigger` | blank_spike | What fires attention rescoring |

---

## Ablation experiments for the paper

Run these on `streaming_decode.py` with a reference audio file:

**Table 1 — CTC weight**
```bash
for w in 0.0 0.1 0.3 0.5 0.7 1.0; do
    python streaming_decode.py --audio test.wav --ctc_weight $w --chunk_ms 320
done
```

**Table 2 — Chunk size (latency vs accuracy)**
```bash
for ms in 160 320 640 1280; do
    python streaming_decode.py --audio test.wav --chunk_ms $ms --ctc_weight 0.5
done
```

**Table 3 — Trigger condition**
```bash
for t in blank_spike word_boundary score_peak; do
    python streaming_decode.py --audio test.wav --trigger $t --chunk_ms 320
done
```

Each run prints a summary line:
```
Chunk size: 320ms | CTC weight: 0.5 | Trigger: blank_spike
Triggers fired: 14/48 chunks | Segments: 7 | RTF: 3.2x
```
Collect these into your paper's Table 2/3.

---

## Live subtitle controls

| Action | Effect |
|---|---|
| Drag window | Reposition on screen |
| Scroll wheel | Resize subtitle text |
| Right-click | Menu (quit, toggle border) |
| Q or Escape | Quit |

---

## Performance on CPU (realistic expectations)

| Audio length | Approximate decode time |
|---|---|
| 5 seconds | 3–8 seconds |
| 30 seconds | 20–50 seconds |
| Live stream | 2–5 second lag |

This is a hardware constraint. The decoding logic is identical to a GPU
deployment — only speed differs. For a paper, WER measurements on pre-recorded
files are sufficient; live latency is reported separately as "emission delay."

---

## Paper structure this maps to

| Paper section | Covered by |
|---|---|
| 3.1 Model architecture | Streaming encoder + dual-branch design |
| 3.2 Streaming decoding algorithm | CTCPrefixBeam + TriggerDetector classes |
| 3.3 Joint scoring | α·CTC + (1-α)·Attn formula |
| 3.4 Subtitle engine | SRT output with CTC timestamps |
| 4.1 Experimental setup | `jkang/espnet2_librispeech_100_conformer`, LibriSpeech |
| 4.2 Ablation: CTC weight | Table 1 commands above |
| 4.3 Ablation: chunk size | Table 2 commands above |
| 4.4 Ablation: trigger | Table 3 commands above |
| 5. Application | `live_subtitle.py`, WASAPI loopback, Windows 11 |

---

## Key references

| Topic | Reference |
|---|---|
| Joint CTC/Attention | Watanabe et al., 2017 — *Hybrid CTC/Attention for E2E ASR* |
| Conformer encoder | Gulati et al., 2020 — *Conformer: Convolution-augmented Transformer* |
| CTC prefix beam search | Graves et al., 2006 — *Connectionist Temporal Classification* |
| Streaming attention | Moritz et al. — *Triggered Attention for E2E ASR* |
| ESPnet2 toolkit | Watanabe et al., 2018 — *ESPnet: End-to-End Speech Processing Toolkit* |
