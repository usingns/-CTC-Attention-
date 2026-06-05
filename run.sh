#!/usr/bin/env bash
# =============================================================================
# run.sh — ESPnet2 LibriSpeech Baseline Recipe
# Project: Combined CTC/Attention Streaming Decoding for Subtitles
#
# Stages:
#   1  Download & prepare LibriSpeech data
#   2  Speed perturbation (3-way: 0.9, 1.0, 1.1)
#   3  Compute global MVN stats
#   4  Train BPE tokenizer (5000 units)
#   5  Train model
#   6  Decode & score (WER on test-clean, test-other)
#
# Usage:
#   ./run.sh                        # run all stages
#   ./run.sh --stage 5 --stop_stage 5   # train only
#   ./run.sh --stage 6              # decode only (needs trained model)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit these to match your environment
# ---------------------------------------------------------------------------
ESPNET_ROOT=/path/to/espnet           # ESPnet root directory
LIBRISPEECH_ROOT=/path/to/LibriSpeech # Raw LibriSpeech corpus root

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
asr_config=conf/train_asr_conformer.yaml
decode_config=conf/decode_asr.yaml
train_set=train_960
valid_set=dev
test_sets="test_clean test_other dev_clean dev_other"

# For quick iteration on a single GPU: swap to train-clean-100
# train_set=train_clean_100

nbpe=5000
bpe_nlsyms=               # non-linguistic symbols (empty for LibriSpeech)
lm_config=conf/train_lm.yaml  # optional; set lm_train=false to skip LM
use_lm=false               # set true once baseline is solid

# ---------------------------------------------------------------------------
# Stage 1 — Data preparation
# ---------------------------------------------------------------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Data preparation"
    local/data.sh --stage 1 --stop_stage 1 \
        --train_set ${train_set} \
        --train_dev ${valid_set} \
        --test_sets ${test_sets} \
        "${LIBRISPEECH_ROOT}"
fi

# ---------------------------------------------------------------------------
# Stage 2 — Speed perturbation (optional but recommended for LibriSpeech)
# ---------------------------------------------------------------------------
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Speed perturbation"
    # ESPnet2 handles this internally via --speed_perturb_factors
    # Handled in asr.sh below
    echo "Speed perturbation handled by asr.sh"
fi

# ---------------------------------------------------------------------------
# Stages 3-6 — Run ESPnet2 asr.sh (the main recipe driver)
# ---------------------------------------------------------------------------
./asr.sh \
    --stage 3 \
    --stop_stage 6 \
    --ngpu 1 \
    --use_lm ${use_lm} \
    --token_type bpe \
    --nbpe ${nbpe} \
    --feats_type raw \
    --audio_format flac \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --lm_config "${lm_config}" \
    --local_data_opts "--train_set ${train_set}"
