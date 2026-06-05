# ESPnet2 Baseline Setup
## Combined CTC/Attention Streaming Decoding for Subtitles

---

## Directory Structure

```
espnet2_baseline/
├── conf/
│   ├── train_asr_conformer.yaml       ← Stage 1: offline baseline
│   ├── train_asr_chunk_conformer.yaml ← Stage 2: streaming encoder
│   └── decode_asr.yaml                ← inference config (shared)
├── run.sh                             ← recipe driver
└── README.md                          ← this file
```

---

## Prerequisites

### 1. Install ESPnet2

```bash
git clone https://github.com/espnet/espnet
cd espnet/tools
make -j4        # builds Kaldi, Python env, all dependencies (~30–60 min)
```

Verify install:
```bash
cd espnet/egs2/TEMPLATE/asr1
./run.sh --help
```

### 2. Get LibriSpeech

```bash
# Full 960h (for final system)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/train-other-500.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
wget https://www.openslr.org/resources/12/test-other.tar.gz

# Quick dev run: just train-clean-100 (~5h, good for config debugging)
# Set train_set=train_clean_100 in run.sh
```

### 3. Link this config into your ESPnet2 recipe

```bash
cd espnet/egs2/librispeech/asr1
ln -s /path/to/this/conf/train_asr_conformer.yaml conf/
ln -s /path/to/this/conf/decode_asr.yaml conf/
cp /path/to/this/run.sh .
chmod +x run.sh
```

---

## Running the Baseline

### Quick sanity check (train-clean-100, 1 GPU, ~2 days)

```bash
./run.sh \
  --stage 1 --stop_stage 6 \
  --train_set train_clean_100 \
  --ngpu 1
```

### Full 960h run (4 GPUs, ~3–4 days)

```bash
./run.sh \
  --stage 1 --stop_stage 6 \
  --train_set train_960 \
  --ngpu 4
```

### Decode only (after training)

```bash
./asr.sh \
  --stage 6 --stop_stage 6 \
  --asr_config conf/train_asr_conformer.yaml \
  --decode_config conf/decode_asr.yaml \
  --test_sets "test_clean test_other"
```

WER results appear in:
```
exp/asr_train_conformer_raw_en_bpe5000/decode_asr_beam60/
```

---

## Expected Results (Offline Baseline)

| Model | test-clean WER | test-other WER |
|---|---|---|
| Conformer + joint CTC/Attn (960h) | ~2.5% | ~5.8% |
| Conformer + joint CTC/Attn (100h) | ~5.0% | ~13.0% |

These are your Table 1 offline baselines.

---

## Paper Ablation Roadmap

### Ablation 1 — CTC weight (offline)

Run `decode_asr.yaml` with different `ctc_weight` values, same trained model:

```bash
for ctc_w in 0.0 0.1 0.3 0.5 0.7 1.0; do
  ./asr.sh --stage 6 \
    --decode_config <(echo "beam_size: 60
ctc_weight: ${ctc_w}
penalty: 0.0
maxlenratio: 0.0
minlenratio: 0.0
lm_weight: 0.0") \
    --test_sets "test_clean test_other"
done
```

Expected finding: ctc_weight ≈ 0.3 minimises WER.

### Ablation 2 — Chunk size vs. latency (streaming)

Train with `conf/train_asr_chunk_conformer.yaml`, varying `chunk_size` and `right_context`:

```bash
for chunk in 8 16 32; do
  for ctx in 0 2 4; do
    # Edit chunk_size and right_context in the config, then train
    # Record: WER on test-clean, average emission delay (ms)
  done
done
```

This produces your main latency-accuracy trade-off curve.

### Ablation 3 — Trigger condition (streaming)

Implement three trigger strategies in `local/streaming_decode.py`:
- `blank_spike`: trigger when blank probability > 0.9 for 3+ frames
- `word_boundary`: trigger at end-of-word BPE tokens
- `score_peak`: trigger when top-1 CTC score crosses threshold θ

Measure: WER + subtitle segment quality (NER, reading speed compliance).

---

## Measuring Latency

Use ESPnet2's built-in latency measurement:

```python
from espnet2.asr.streaming import StreamingASR

model = StreamingASR.from_pretrained("exp/asr_train_chunk_conformer_...")
latency = model.measure_emission_delay(test_wavs, chunk_size=8)
print(f"Average word emission delay: {latency['mean_ms']:.1f} ms")
print(f"90th percentile delay:       {latency['p90_ms']:.1f} ms")
```

Target for real-time subtitling: mean delay < 1000ms, p90 < 1500ms.

---

## Subtitle Output

After decoding, convert ESPnet2 output to SRT:

```bash
python local/to_srt.py \
  --hyp_file exp/.../text \
  --ctc_timestamps exp/.../ctc_align.json \
  --output subtitles.srt \
  --max_chars_per_line 42 \     # EBU standard
  --max_display_duration 7.0    # seconds
```

---

## Next Steps After Baseline

1. **Baseline solid** → swap `train_asr_conformer.yaml` → `train_asr_chunk_conformer.yaml`
2. **Streaming encoder trained** → implement `local/streaming_decode.py` (CTC beam + trigger)
3. **Trigger implemented** → add attention rescoring over cached encoder states
4. **Full system working** → run subtitle quality evaluation (NER, reading speed)
5. **Evaluation complete** → write Section 4 (Experiments) of paper
