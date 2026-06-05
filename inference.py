"""
inference.py
------------
Offline inference with the pretrained joint CTC/Attention Conformer.
Use this to verify your setup works before running the streaming decoder.

Usage:
    python inference.py --audio speech.wav
    python inference.py --audio speech.wav --beam_size 5 --ctc_weight 0.3
"""

import argparse
import sys
import time

MODEL_TAG = "jkang/espnet2_librispeech_100_conformer"
SAMPLE_RATE = 16000


def load_audio(path: str):
    """Load and resample audio to 16kHz mono."""
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        print("Run: pip install soundfile")
        sys.exit(1)

    speech, sr = sf.read(path, dtype="float32")

    # Convert stereo to mono
    if speech.ndim == 2:
        speech = speech.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        try:
            import librosa
            speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
            print(f"Resampled from {sr}Hz to {SAMPLE_RATE}Hz")
        except ImportError:
            print(f"Warning: audio is {sr}Hz, expected 16000Hz.")
            print("Install librosa for automatic resampling: pip install librosa")

    duration = len(speech) / SAMPLE_RATE
    print(f"Audio loaded: {duration:.1f}s")
    return speech, duration


def main():
    parser = argparse.ArgumentParser(description="Offline ASR inference")
    parser.add_argument("--audio",      required=True, help="Path to audio file (WAV recommended)")
    parser.add_argument("--beam_size",  type=int,   default=10,  help="Beam size (5=fast, 60=accurate)")
    parser.add_argument("--ctc_weight", type=float, default=0.3, help="CTC weight in joint decoding [0.0-1.0]")
    parser.add_argument("--nbest",      type=int,   default=1,   help="Number of hypotheses to print")
    args = parser.parse_args()

    # Load model
    try:
        from espnet2.bin.asr_inference import Speech2Text
    except ImportError:
        print("ESPnet not found. Run: pip install espnet espnet_model_zoo")
        print("Then run: python download_model.py")
        sys.exit(1)

    print(f"Loading model (beam_size={args.beam_size}, ctc_weight={args.ctc_weight})...")
    t0 = time.time()
    speech2text = Speech2Text.from_pretrained(
        MODEL_TAG,
        device="cpu",
        beam_size=args.beam_size,
        ctc_weight=args.ctc_weight,
        nbest=args.nbest,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # Load audio
    speech, duration = load_audio(args.audio)

    # Run inference
    print("Running inference...")
    t1 = time.time()
    results = speech2text(speech)
    elapsed = time.time() - t1
    rtf = elapsed / duration  # real-time factor (lower = faster)

    # Print results
    print(f"\n{'='*50}")
    print(f"Transcription ({elapsed:.1f}s decode, RTF={rtf:.2f}x):")
    print(f"{'='*50}")
    for rank, (text, token, token_int, hyp) in enumerate(results, 1):
        score = hyp.score.item() if hasattr(hyp.score, 'item') else hyp.score
        print(f"[{rank}] (score={score:.2f}) {text}")

    # Print CTC token-level alignment from the best hypothesis
    print(f"\nCTC alignment (word-level timestamps):")
    try:
        import torch
        import numpy as np

        model    = speech2text.asr_model
        frontend = speech2text.frontend
        normalize = speech2text.normalize

        # Re-extract features to get CTC alignment
        with torch.no_grad():
            import torch
            speech_tensor = torch.tensor(speech).unsqueeze(0)
            lengths       = torch.tensor([len(speech)])

            # Frontend + normalization
            feats, feat_lens = frontend(speech_tensor, lengths)
            if normalize is not None:
                feats, feat_lens = normalize(feats, feat_lens)

            # Encode
            encoder_out, enc_lens, _ = model.encode(feats, feat_lens)

            # CTC greedy decode for timestamps
            ctc_log_probs = model.ctc.log_softmax(encoder_out)       # (1, T, V)
            ctc_ids = ctc_log_probs.argmax(dim=-1).squeeze(0).numpy()  # (T,)

        # Convert frame IDs to word-level timestamps
        # Frame shift: 10ms * 4 (subsampling) = 40ms per encoder frame
        frame_ms  = 40
        blank_id  = 0
        tokens    = speech2text.converter.token_list

        prev_id    = blank_id
        word_start = None
        current_word_chars = []

        for frame_idx, token_id in enumerate(ctc_ids):
            t_ms = frame_idx * frame_ms
            if token_id != blank_id and token_id != prev_id:
                tok = tokens[token_id]
                # BPE word boundary marker (▁ prefix = new word)
                if tok.startswith("▁") and current_word_chars:
                    word = "".join(current_word_chars).replace("▁", "")
                    print(f"  {word_start/1000:.2f}s – {t_ms/1000:.2f}s  {word}")
                    current_word_chars = [tok]
                    word_start = t_ms
                else:
                    if word_start is None:
                        word_start = t_ms
                    current_word_chars.append(tok)
            prev_id = token_id

        # Flush last word
        if current_word_chars:
            word = "".join(current_word_chars).replace("▁", "")
            end_ms = len(ctc_ids) * frame_ms
            print(f"  {word_start/1000:.2f}s – {end_ms/1000:.2f}s  {word}")

    except Exception as e:
        print(f"  (CTC alignment failed: {e})")


if __name__ == "__main__":
    main()
