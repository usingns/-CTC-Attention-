"""
streaming_decode.py
-------------------
Simulated streaming decoding with joint CTC/Attention and subtitle output.

Processes audio in fixed-size chunks (simulating a live audio stream), runs
CTC prefix beam search incrementally, fires attention rescoring at trigger
points, combines scores, and writes SRT subtitle output.

This is the core research code for the project. The pretrained model encoder
is non-causal, so "streaming" is simulated by chunking encoder outputs — the
decoding logic is identical to what a truly causal chunk-Conformer would use.

Usage:
    python streaming_decode.py --audio speech.wav --output subtitles.srt
    python streaming_decode.py --audio speech.wav --chunk_ms 320 --ctc_weight 0.5 \\
                               --trigger blank_spike --beam_size 10

Key parameters (ablation table):
    --chunk_ms      160 / 320 / 640 / 1280   (latency knob)
    --ctc_weight    0.0 / 0.3 / 0.5 / 0.7 / 1.0
    --trigger       blank_spike / word_boundary / score_peak
    --beam_size     5 (fast dev) / 10 (balanced) / 60 (paper results)
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
MODEL_TAG = "jkang/espnet2_librispeech_100_conformer"
SAMPLE_RATE = 16000
FRAME_SHIFT_MS = 10      # frontend frame shift
SUBSAMPLE_FACTOR = 4     # conv2d subsampling in Conformer
ENC_FRAME_MS = FRAME_SHIFT_MS * SUBSAMPLE_FACTOR   # 40ms per encoder frame
NEG_INF = -1e30
BLANK_ID = 0
# ---------------------------------------------------------------------------


# ===========================================================================
# CTC Prefix Beam Search
# ===========================================================================

@dataclass
class BeamEntry:
    """One hypothesis in the CTC prefix beam."""
    prefix: Tuple[int, ...]              # token IDs so far (no blanks)
    log_prob_blank: float = NEG_INF      # log P(prefix ends with blank)
    log_prob_nonblank: float = NEG_INF   # log P(prefix ends with non-blank)
    last_token_frame: int = 0            # frame where last non-blank was emitted

    @property
    def total_log_prob(self) -> float:
        return math.log(
            math.exp(self.log_prob_blank) + math.exp(self.log_prob_nonblank)
        )


def log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class CTCPrefixBeam:
    """
    Streaming CTC prefix beam search.

    Processes encoder frames one at a time. Call update() for each new
    chunk of frames; call get_beam() to read the current hypotheses.

    Reference: Graves et al. 2006 + Maas et al. streaming extension.
    """

    def __init__(self, beam_size: int = 10, blank_id: int = BLANK_ID):
        self.beam_size = beam_size
        self.blank_id = blank_id
        self.frame_idx = 0

        # Initial state: empty prefix, log P = 0.0 (log(1))
        init = BeamEntry(prefix=(), log_prob_blank=0.0, log_prob_nonblank=NEG_INF)
        self.beam: Dict[Tuple[int, ...], BeamEntry] = {(): init}

    def update(self, log_probs: np.ndarray):
        """
        Update beam with one frame of CTC log probabilities.

        Args:
            log_probs: (vocab_size,) log probabilities from CTC head
        """
        vocab_size = len(log_probs)
        new_beam: Dict[Tuple[int, ...], BeamEntry] = {}

        for prefix, entry in self.beam.items():
            p_total = entry.total_log_prob

            # --- Extend with blank: prefix stays the same ---
            p_blank = p_total + log_probs[self.blank_id]
            if prefix not in new_beam:
                new_beam[prefix] = BeamEntry(
                    prefix=prefix,
                    log_prob_blank=NEG_INF,
                    log_prob_nonblank=NEG_INF,
                    last_token_frame=entry.last_token_frame,
                )
            new_beam[prefix].log_prob_blank = log_add(
                new_beam[prefix].log_prob_blank, p_blank
            )

            # --- Extend with each non-blank token ---
            for token_id in range(vocab_size):
                if token_id == self.blank_id:
                    continue

                new_prefix = prefix + (token_id,)
                log_p = log_probs[token_id]

                # CTC merging rule: if new token == last token of prefix,
                # it can only extend if the previous frame was a blank
                if prefix and prefix[-1] == token_id:
                    p_extend = entry.log_prob_blank + log_p
                else:
                    p_extend = p_total + log_p

                if new_prefix not in new_beam:
                    new_beam[new_prefix] = BeamEntry(
                        prefix=new_prefix,
                        log_prob_blank=NEG_INF,
                        log_prob_nonblank=NEG_INF,
                        last_token_frame=self.frame_idx,
                    )
                new_beam[new_prefix].log_prob_nonblank = log_add(
                    new_beam[new_prefix].log_prob_nonblank, p_extend
                )
                new_beam[new_prefix].last_token_frame = self.frame_idx

        # Prune to top-K by total log prob
        sorted_entries = sorted(
            new_beam.values(),
            key=lambda e: e.total_log_prob,
            reverse=True,
        )
        self.beam = {
            e.prefix: e for e in sorted_entries[: self.beam_size]
        }
        self.frame_idx += 1

    def get_beam(self) -> List[BeamEntry]:
        """Return beam sorted by total log prob (best first)."""
        return sorted(
            self.beam.values(),
            key=lambda e: e.total_log_prob,
            reverse=True,
        )

    def recent_blank_ratio(self, window: int = 10) -> float:
        """
        Fraction of recent frames dominated by blank in the top hypothesis.
        Used by the blank_spike trigger.
        """
        # Proxy: if top hypothesis hasn't changed token in `window` frames
        top = self.get_beam()[0]
        frames_since_last = self.frame_idx - top.last_token_frame
        return min(frames_since_last / window, 1.0)


# ===========================================================================
# Trigger conditions
# ===========================================================================

class TriggerDetector:
    """
    Decides when to fire attention rescoring on the current beam.

    Three strategies (ablation dimension):
      blank_spike     — blank ratio in recent frames crosses a threshold
      word_boundary   — top hypothesis ends with a word-boundary BPE token
      score_peak      — top CTC log prob crosses an absolute threshold
    """

    def __init__(
        self,
        strategy: str = "blank_spike",
        blank_threshold: float = 0.7,
        score_threshold: float = -5.0,
        token_list: Optional[List[str]] = None,
    ):
        assert strategy in ("blank_spike", "word_boundary", "score_peak"), \
            f"Unknown trigger strategy: {strategy}"
        self.strategy = strategy
        self.blank_threshold = blank_threshold
        self.score_threshold = score_threshold
        self.token_list = token_list or []

    def should_trigger(
        self,
        beam: CTCPrefixBeam,
        chunk_log_probs: np.ndarray,       # (frames, vocab) for this chunk
    ) -> bool:
        top = beam.get_beam()[0]

        if self.strategy == "blank_spike":
            ratio = beam.recent_blank_ratio(window=8)
            return ratio >= self.blank_threshold

        elif self.strategy == "word_boundary":
            if not top.prefix:
                return False
            last_token_id = top.prefix[-1]
            if last_token_id < len(self.token_list):
                tok = self.token_list[last_token_id]
                # BPE word boundary: next token would start with ▁
                # Trigger when top hyp ends with a complete word token
                return tok.endswith("</s>") or (
                    len(top.prefix) > 1 and
                    self.token_list[top.prefix[-1]].startswith("▁")
                )
            return False

        elif self.strategy == "score_peak":
            score = top.total_log_prob
            # Normalise by prefix length to avoid length bias
            norm_score = score / max(len(top.prefix), 1)
            return norm_score > self.score_threshold

        return False


# ===========================================================================
# Attention rescoring
# ===========================================================================

def attention_rescore(
    top_k_hypotheses: List[BeamEntry],
    encoder_cache: torch.Tensor,    # (1, T_enc, D) all encoder states so far
    model,                          # ESPnet2 ASR model
    token_list: List[str],
    ctc_weight: float = 0.5,
) -> Tuple[str, float, List[int]]:
    """
    Rescore top-K CTC hypotheses using the attention decoder.

    Returns (best_text, best_score, best_token_ids).
    """
    if ctc_weight == 1.0:
        # Pure CTC — skip attention entirely
        best = top_k_hypotheses[0]
        text = "".join(token_list[t] for t in best.prefix).replace("▁", " ").strip()
        return text, best.total_log_prob, list(best.prefix)

    best_score = NEG_INF
    best_text = ""
    best_ids = []

    with torch.no_grad():
        enc_len = torch.tensor([encoder_cache.shape[1]])

        for entry in top_k_hypotheses:
            if not entry.prefix:
                continue

            # Prepare decoder input: [sos] + prefix tokens
            sos_id = len(token_list) - 1   # ESPnet convention: sos = last token
            hyp_ids = torch.tensor([[sos_id] + list(entry.prefix)])

            try:
                # Run attention decoder over cached encoder states
                decoder_out, _ = model.decoder(
                    encoder_cache,
                    enc_len,
                    hyp_ids,
                    torch.tensor([hyp_ids.shape[1]]),
                )
                # decoder_out: (1, L, vocab) log probs at each position

                # Sum log probs of each hypothesis token
                attn_score = 0.0
                log_probs = torch.log_softmax(decoder_out[0], dim=-1)  # (L, vocab)
                for pos, token_id in enumerate(entry.prefix):
                    if pos < log_probs.shape[0]:
                        attn_score += log_probs[pos, token_id].item()

                # Normalise by length
                norm_attn = attn_score / max(len(entry.prefix), 1)
                norm_ctc = entry.total_log_prob / max(len(entry.prefix), 1)

                # Joint score
                joint_score = (
                    ctc_weight * norm_ctc + (1.0 - ctc_weight) * norm_attn
                )

                if joint_score > best_score:
                    best_score = joint_score
                    best_ids = list(entry.prefix)
                    best_text = (
                        "".join(token_list[t] for t in entry.prefix)
                        .replace("▁", " ")
                        .strip()
                    )

            except Exception:
                # Fallback to CTC score if attention decoder fails
                score = entry.total_log_prob / max(len(entry.prefix), 1)
                if score > best_score:
                    best_score = score
                    best_ids = list(entry.prefix)
                    best_text = (
                        "".join(token_list[t] for t in entry.prefix)
                        .replace("▁", " ")
                        .strip()
                    )

    return best_text, best_score, best_ids


# ===========================================================================
# SRT output
# ===========================================================================

def ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format HH:MM:SS,mmm."""
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1_000
    ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass
class SubtitleSegment:
    text: str
    start_ms: int
    end_ms: int


def write_srt(segments: List[SubtitleSegment], output_path: str):
    """Write subtitle segments to an SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{ms_to_srt_time(seg.start_ms)} --> {ms_to_srt_time(seg.end_ms)}\n")
            f.write(f"{seg.text}\n\n")
    print(f"Subtitles written to: {output_path} ({len(segments)} segments)")


# ===========================================================================
# Main streaming decode loop
# ===========================================================================

def load_audio(path: str) -> np.ndarray:
    import soundfile as sf
    speech, sr = sf.read(path, dtype="float32")
    if speech.ndim == 2:
        speech = speech.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            import librosa
            speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
        except ImportError:
            print(f"Warning: audio is {sr}Hz not 16000Hz. Install librosa to resample.")
    return speech


def streaming_decode(
    audio: np.ndarray,
    model,
    frontend,
    normalize,
    token_list: List[str],
    chunk_ms: int = 320,
    ctc_weight: float = 0.5,
    beam_size: int = 10,
    trigger_strategy: str = "blank_spike",
    max_segment_tokens: int = 15,      # max words per subtitle segment
    min_segment_tokens: int = 2,       # don't emit very short segments
) -> Tuple[List[SubtitleSegment], dict]:

    # Convert chunk_ms to audio samples
    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
    total_chunks = math.ceil(len(audio) / chunk_samples)

    # Full encode upfront (simulating streaming with the non-causal model)
    # In a true streaming system, this would run per-chunk with a causal encoder.
    print(f"Encoding audio ({len(audio)/SAMPLE_RATE:.1f}s)...")
    with torch.no_grad():
        speech_tensor = torch.tensor(audio).unsqueeze(0)
        lengths = torch.tensor([len(audio)])
        feats, feat_lens = frontend(speech_tensor, lengths)
        if normalize is not None:
            feats, feat_lens = normalize(feats, feat_lens)
        encoder_out, enc_lens, _ = model.encode(feats, feat_lens)
        ctc_log_probs = model.ctc.log_softmax(encoder_out).squeeze(0).numpy()  # (T, V)

    total_enc_frames = ctc_log_probs.shape[0]
    enc_frames_per_chunk = max(1, int(chunk_ms / ENC_FRAME_MS))

    print(f"Decoding: {total_chunks} chunks × {chunk_ms}ms "
          f"| beam={beam_size} | ctc_weight={ctc_weight} | trigger={trigger_strategy}\n")

    # Initialise beam and trigger
    beam = CTCPrefixBeam(beam_size=beam_size, blank_id=BLANK_ID)
    trigger = TriggerDetector(
        strategy=trigger_strategy,
        token_list=token_list,
    )

    segments: List[SubtitleSegment] = []
    encoder_cache = torch.zeros(1, 0, encoder_out.shape[-1])
    triggers_fired = 0
    segment_start_ms = 0
    emitted_prefix_len = 0    # how many tokens of beam have already been emitted

    for chunk_idx in range(total_chunks):
        frame_start = chunk_idx * enc_frames_per_chunk
        frame_end = min(frame_start + enc_frames_per_chunk, total_enc_frames)
        if frame_start >= total_enc_frames:
            break

        chunk_log_probs = ctc_log_probs[frame_start:frame_end]  # (F, V)
        chunk_enc = encoder_out[:, frame_start:frame_end, :]     # (1, F, D)

        # Feed chunk frames into CTC beam
        for frame in chunk_log_probs:
            beam.update(frame)

        # Accumulate encoder cache (for attention rescoring)
        encoder_cache = torch.cat([encoder_cache, chunk_enc], dim=1)

        chunk_time_ms = (chunk_idx + 1) * chunk_ms

        # Check trigger
        if trigger.should_trigger(beam, chunk_log_probs):
            triggers_fired += 1
            top_k = beam.get_beam()[:min(5, beam_size)]

            # Only rescore if we have new tokens to emit
            top_prefix_len = len(top_k[0].prefix) if top_k else 0
            new_tokens = top_prefix_len - emitted_prefix_len

            if new_tokens >= min_segment_tokens:
                text, score, token_ids = attention_rescore(
                    top_k,
                    encoder_cache,
                    model,
                    token_list,
                    ctc_weight=ctc_weight,
                )

                if text.strip():
                    seg = SubtitleSegment(
                        text=text.strip(),
                        start_ms=segment_start_ms,
                        end_ms=chunk_time_ms,
                    )
                    segments.append(seg)
                    print(f"  [{ms_to_srt_time(seg.start_ms)} → {ms_to_srt_time(seg.end_ms)}]  {seg.text}")
                    segment_start_ms = chunk_time_ms
                    emitted_prefix_len = top_prefix_len

    # Flush anything remaining in the beam after all chunks
    top_k = beam.get_beam()[:min(5, beam_size)]
    if top_k and len(top_k[0].prefix) > emitted_prefix_len:
        text, _, _ = attention_rescore(
            top_k, encoder_cache, model, token_list, ctc_weight=ctc_weight
        )
        if text.strip():
            end_ms = int(len(audio) / SAMPLE_RATE * 1000)
            seg = SubtitleSegment(
                text=text.strip(),
                start_ms=segment_start_ms,
                end_ms=end_ms,
            )
            segments.append(seg)
            print(f"  [{ms_to_srt_time(seg.start_ms)} → {ms_to_srt_time(seg.end_ms)}]  {seg.text}")

    stats = {
        "total_chunks": total_chunks,
        "triggers_fired": triggers_fired,
        "segments": len(segments),
    }
    return segments, stats


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Streaming ASR with subtitle output")
    parser.add_argument("--audio",       required=True,              help="Input audio file")
    parser.add_argument("--output",      default="subtitles.srt",    help="Output SRT file")
    parser.add_argument("--chunk_ms",    type=int,   default=320,    help="Chunk size in ms (latency knob)")
    parser.add_argument("--ctc_weight",  type=float, default=0.5,    help="CTC weight in joint score [0.0-1.0]")
    parser.add_argument("--beam_size",   type=int,   default=10,     help="Beam size")
    parser.add_argument("--trigger",     default="blank_spike",
                        choices=["blank_spike", "word_boundary", "score_peak"],
                        help="Trigger condition for attention rescoring")
    args = parser.parse_args()

    # Load model
    try:
        from espnet2.bin.asr_inference import Speech2Text
    except ImportError:
        print("ESPnet not found. Run: pip install espnet espnet_model_zoo")
        print("Then: python download_model.py")
        sys.exit(1)

    print("Loading model...")
    t0 = time.time()
    s2t = Speech2Text.from_pretrained(
        MODEL_TAG, device="cpu", beam_size=args.beam_size, ctc_weight=args.ctc_weight
    )
    model     = s2t.asr_model
    frontend  = s2t.frontend
    normalize = s2t.normalize
    token_list = s2t.converter.token_list
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # Load audio
    audio = load_audio(args.audio)
    duration_s = len(audio) / SAMPLE_RATE
    print(f"Audio: {duration_s:.1f}s\n")

    # Run streaming decode
    t1 = time.time()
    segments, stats = streaming_decode(
        audio=audio,
        model=model,
        frontend=frontend,
        normalize=normalize,
        token_list=token_list,
        chunk_ms=args.chunk_ms,
        ctc_weight=args.ctc_weight,
        beam_size=args.beam_size,
        trigger_strategy=args.trigger,
    )
    elapsed = time.time() - t1

    # Write SRT
    write_srt(segments, args.output)

    # Summary (ablation table row)
    print(f"\n{'='*50}")
    print(f"Decoding summary")
    print(f"{'='*50}")
    print(f"  Audio duration:   {duration_s:.1f}s")
    print(f"  Chunk size:       {args.chunk_ms}ms")
    print(f"  CTC weight:       {args.ctc_weight}")
    print(f"  Beam size:        {args.beam_size}")
    print(f"  Trigger:          {args.trigger}")
    print(f"  Triggers fired:   {stats['triggers_fired']} / {stats['total_chunks']} chunks")
    print(f"  Segments output:  {stats['segments']}")
    print(f"  Decode time:      {elapsed:.1f}s  (RTF={elapsed/duration_s:.2f}x)")
    print(f"  Output:           {args.output}")


if __name__ == "__main__":
    main()
