"""
download_model.py
-----------------
Downloads the pretrained LibriSpeech joint CTC/Attention Conformer from the
ESPnet model zoo and verifies it loads correctly on CPU.

Run once before using inference.py or streaming_decode.py:
    python download_model.py
"""

import sys

# ---------------------------------------------------------------------------
# Pretrained model tag
# Conformer + joint CTC/Attention, trained on LibriSpeech 960h
# WER: ~2.5% test-clean, ~5.8% test-other
# ---------------------------------------------------------------------------
# Valid Hugging Face repo ID — joint CTC/Attention Conformer, LibriSpeech 100h
# Same architecture (Conformer encoder + Transformer decoder, ctc_weight=0.3,
# 5000 BPE units) as described in the project. Smaller training set than 960h
# but identical model structure — perfect for learning and experimentation.
MODEL_TAG = "jkang/espnet2_librispeech_100_conformer"


def main():
    print("Checking dependencies...")
    try:
        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader
    except ImportError as e:
        print(f"\nMissing package: {e}")
        print("Run: pip install espnet espnet_model_zoo")
        sys.exit(1)

    print("Downloading pretrained model (one-time, ~200MB)...")
    print(f"Model: {MODEL_TAG}\n")

    try:
        # ModelDownloader caches in ~/.cache/espnet — safe to run multiple times
        d = ModelDownloader()
        model_path = d.download_and_unpack(MODEL_TAG)
        print(f"Files cached at: {model_path['asr_train_config']}\n")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("Check your internet connection and try again.")
        print("If the issue persists, see the troubleshooting section in INSTALL.md")
        sys.exit(1)

    print("Loading model on CPU (this takes ~30 seconds)...")
    try:
        speech2text = Speech2Text.from_pretrained(
            MODEL_TAG,
            device="cpu",
            beam_size=5,       # small beam just for this verification
            ctc_weight=0.3,
        )
    except Exception as e:
        print(f"\nModel load failed: {e}")
        sys.exit(1)

    vocab_size = len(speech2text.converter.token_list)
    print(f"Model ready.")
    print(f"Vocabulary size: {vocab_size} BPE tokens")
    print(f"Encoder type:    {type(speech2text.asr_model.encoder).__name__}")
    print(f"Decoder type:    {type(speech2text.asr_model.decoder).__name__}")
    print("\nSetup complete. You can now run:")
    print("  python inference.py --audio your_audio.wav")
    print("  python streaming_decode.py --audio your_audio.wav --output subtitles.srt")


if __name__ == "__main__":
    main()
