"""
live_subtitle.py
----------------
Real-time subtitles from system audio on Windows 11.

Captures whatever is playing through your speakers (browser, media player,
video call) via WASAPI loopback — no virtual cable needed. Feeds audio into
the streaming CTC/Attention ASR pipeline and shows subtitles in a small
always-on-top overlay window.

Install extra dependency first:
    pip install soundcard

Usage:
    python live_subtitle.py                    # use default speakers
    python live_subtitle.py --list-devices     # show available loopback sources
    python live_subtitle.py --device "Speakers (Realtek)"
    python live_subtitle.py --chunk_ms 640 --ctc_weight 0.5

Controls (overlay window):
    Drag        move the window
    Scroll      resize font
    Q / Esc     quit
"""

import argparse
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import torch

MODEL_TAG = "jkang/espnet2_librispeech_100_conformer"
TARGET_SR = 16_000        # model expects 16kHz
CHUNK_MS = 320            # default chunk size
CHANNELS = 1              # mono


# ===========================================================================
# 1. WASAPI loopback capture thread
# ===========================================================================

def list_loopback_devices():
    """Print all available loopback (output) devices and exit."""
    try:
        import soundcard as sc
    except ImportError:
        print("Run: pip install soundcard")
        sys.exit(1)
    print("Available loopback devices (copy the name for --device):\n")
    for spk in sc.all_speakers():
        print(f"  {spk.name}")
    sys.exit(0)


def capture_thread(
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    device_name: str | None,
    chunk_ms: int,
):
    """
    Capture system audio via WASAPI loopback and push chunks to audio_queue.
    Runs in a dedicated thread so it never blocks the ASR or UI.
    """
    try:
        import soundcard as sc
    except ImportError:
        print("\nMissing dependency: pip install soundcard")
        stop_event.set()
        return

    # Find the loopback device
    try:
        if device_name:
            speakers = [s for s in sc.all_speakers() if device_name.lower() in s.name.lower()]
            if not speakers:
                print(f"Device '{device_name}' not found. Run --list-devices to see options.")
                stop_event.set()
                return
            speaker = speakers[0]
        else:
            speaker = sc.default_speaker()
        print(f"Capturing loopback from: {speaker.name}")
    except Exception as e:
        print(f"Could not open loopback device: {e}")
        stop_event.set()
        return

    chunk_samples = int(TARGET_SR * chunk_ms / 1000)

    try:
        with sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True,
        ).recorder(samplerate=TARGET_SR, channels=CHANNELS) as mic:

            print("Listening... (speak or play audio)\n")
            while not stop_event.is_set():
                # record() blocks for exactly chunk_samples frames
                data = mic.record(numframes=chunk_samples)
                # data shape: (chunk_samples, channels) → flatten to 1D float32
                mono = data[:, 0].astype(np.float32)
                audio_queue.put(mono)

    except Exception as e:
        print(f"Capture error: {e}")
        stop_event.set()


# ===========================================================================
# 2. Streaming ASR thread
# ===========================================================================

def load_model(beam_size: int, ctc_weight: float):
    """Load the pretrained ESPnet2 model onto CPU."""
    try:
        from espnet2.bin.asr_inference import Speech2Text
    except ImportError:
        print("ESPnet not found. Run: pip install espnet espnet_model_zoo")
        print("Then: python download_model.py")
        sys.exit(1)

    print("Loading ASR model...")
    t0 = time.time()
    s2t = Speech2Text.from_pretrained(
        MODEL_TAG,
        device="cpu",
        beam_size=beam_size,
        ctc_weight=ctc_weight,
    )
    print(f"Model ready in {time.time()-t0:.1f}s\n")
    return s2t


# Reuse CTCPrefixBeam and attention_rescore from streaming_decode.py
# (copied inline here so live_subtitle.py is self-contained)

NEG_INF = -1e30
import math

def log_add(a, b):
    if a == NEG_INF: return b
    if b == NEG_INF: return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class BeamEntry:
    def __init__(self, prefix, log_prob_blank=NEG_INF, log_prob_nonblank=NEG_INF, last_frame=0):
        self.prefix = prefix
        self.log_prob_blank = log_prob_blank
        self.log_prob_nonblank = log_prob_nonblank
        self.last_frame = last_frame

    @property
    def total(self):
        return log_add(self.log_prob_blank, self.log_prob_nonblank)


class CTCPrefixBeam:
    def __init__(self, beam_size=10, blank_id=0):
        self.beam_size = beam_size
        self.blank_id = blank_id
        self.frame_idx = 0
        init = BeamEntry((), log_prob_blank=0.0)
        self.beam = {(): init}

    def update(self, log_probs):
        new_beam = {}
        for prefix, entry in self.beam.items():
            p_total = entry.total
            # blank extension
            p_b = p_total + log_probs[self.blank_id]
            if prefix not in new_beam:
                new_beam[prefix] = BeamEntry(prefix, last_frame=entry.last_frame)
            new_beam[prefix].log_prob_blank = log_add(new_beam[prefix].log_prob_blank, p_b)
            # token extensions
            for tid in range(len(log_probs)):
                if tid == self.blank_id:
                    continue
                np_ = prefix + (tid,)
                lp = log_probs[tid]
                p_ext = (entry.log_prob_blank + lp) if (prefix and prefix[-1] == tid) else (p_total + lp)
                if np_ not in new_beam:
                    new_beam[np_] = BeamEntry(np_, last_frame=self.frame_idx)
                new_beam[np_].log_prob_nonblank = log_add(new_beam[np_].log_prob_nonblank, p_ext)
                new_beam[np_].last_frame = self.frame_idx
        sorted_e = sorted(new_beam.values(), key=lambda e: e.total, reverse=True)
        self.beam = {e.prefix: e for e in sorted_e[:self.beam_size]}
        self.frame_idx += 1

    def top(self):
        return sorted(self.beam.values(), key=lambda e: e.total, reverse=True)

    def blank_run(self, window=8):
        top = self.top()[0]
        return min((self.frame_idx - top.last_frame) / window, 1.0)


def decode_prefix(prefix, token_list):
    return "".join(token_list[t] for t in prefix).replace("▁", " ").strip()


def asr_thread(
    audio_queue: queue.Queue,
    text_queue: queue.Queue,
    stop_event: threading.Event,
    s2t,
    chunk_ms: int,
    ctc_weight: float,
    beam_size: int,
    trigger_strategy: str,
):
    """
    Pull audio chunks from audio_queue, run streaming ASR,
    push transcribed text segments to text_queue.
    """
    model      = s2t.asr_model
    frontend   = s2t.frontend
    normalize  = s2t.normalize
    token_list = s2t.converter.token_list

    BLANK_ID = 0
    ENC_FRAME_MS = 40        # 10ms frame shift × 4x subsampling
    enc_per_chunk = max(1, chunk_ms // ENC_FRAME_MS)

    beam = CTCPrefixBeam(beam_size=beam_size, blank_id=BLANK_ID)
    enc_cache = torch.zeros(1, 0, 256)   # encoder state cache; 256 = model dim
    audio_buf = np.zeros(0, dtype=np.float32)

    MIN_FRAMES_BETWEEN_TRIGGERS = 4   # debounce: don't trigger every chunk
    last_trigger_frame = -MIN_FRAMES_BETWEEN_TRIGGERS
    last_emitted_len = 0
    chunk_count = 0

    while not stop_event.is_set():
        # Collect one chunk from the queue (block up to 0.5s)
        try:
            chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        audio_buf = np.concatenate([audio_buf, chunk])

        # Need at least 1 second of audio before first encode (warm-up)
        if len(audio_buf) < TARGET_SR:
            continue

        # Encode buffered audio
        try:
            with torch.no_grad():
                t = torch.tensor(audio_buf).unsqueeze(0)
                l = torch.tensor([len(audio_buf)])
                feats, fl = frontend(t, l)
                if normalize:
                    feats, fl = normalize(feats, fl)
                enc_out, _, _ = model.encode(feats, fl)
                ctc_lp = model.ctc.log_softmax(enc_out).squeeze(0).numpy()

            # Feed new frames into beam (only the frames not yet processed)
            new_frames = ctc_lp[max(0, enc_cache.shape[1] - enc_per_chunk):]
            for frame in new_frames:
                beam.update(frame)

            enc_cache = enc_out   # update full cache

        except Exception as e:
            continue

        chunk_count += 1

        # Check trigger
        triggered = False
        if chunk_count - last_trigger_frame >= MIN_FRAMES_BETWEEN_TRIGGERS:
            top = beam.top()
            if not top:
                continue
            best = top[0]

            if trigger_strategy == "blank_spike":
                triggered = beam.blank_run(window=8) >= 0.7
            elif trigger_strategy == "word_boundary":
                if best.prefix and best.prefix[-1] < len(token_list):
                    tok = token_list[best.prefix[-1]]
                    triggered = tok.startswith("▁") and len(best.prefix) > 1
            elif trigger_strategy == "score_peak":
                norm = best.total / max(len(best.prefix), 1)
                triggered = norm > -5.0

        if triggered and beam.top():
            top_k = beam.top()[:5]
            best = top_k[0]
            new_tokens = len(best.prefix) - last_emitted_len

            if new_tokens >= 2:
                # Simple joint score: ctc only (attention rescore is expensive live)
                # For live use, CTC-only output is snappy; rescore offline if needed
                text = decode_prefix(best.prefix, token_list)
                if text.strip():
                    text_queue.put(text.strip())
                    last_emitted_len = len(best.prefix)
                    last_trigger_frame = chunk_count

                # Keep only recent audio (last 10 seconds) to bound memory
                keep_samples = TARGET_SR * 10
                if len(audio_buf) > keep_samples:
                    audio_buf = audio_buf[-keep_samples:]
                    # Reset beam on buffer trim to avoid stale state
                    beam = CTCPrefixBeam(beam_size=beam_size, blank_id=BLANK_ID)
                    enc_cache = torch.zeros(1, 0, 256)
                    last_emitted_len = 0


# ===========================================================================
# 3. Subtitle overlay (tkinter, always on top, draggable)
# ===========================================================================

def subtitle_overlay(text_queue: queue.Queue, stop_event: threading.Event):
    """
    A small always-on-top window that displays the latest subtitle text.
    Drag to reposition. Scroll to resize font. Q or Escape to quit.
    """
    try:
        import tkinter as tk
    except ImportError:
        print("tkinter not available — using console output only")
        while not stop_event.is_set():
            try:
                text = text_queue.get(timeout=0.5)
                print(f"SUBTITLE: {text}")
            except queue.Empty:
                pass
        return

    root = tk.Tk()
    root.title("Live Subtitles")
    root.attributes("-topmost", True)        # always on top
    root.attributes("-alpha", 0.88)          # slight transparency
    root.overrideredirect(True)              # no window chrome (borderless)
    root.configure(bg="black")

    # Position: bottom-centre of screen
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    w, h = 900, 90
    x = (sw - w) // 2
    y = sh - h - 80
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.minsize(300, 60)

    font_size = [22]   # mutable so inner functions can modify

    label = tk.Label(
        root,
        text="",
        font=("Segoe UI", font_size[0], "bold"),
        fg="white",
        bg="black",
        wraplength=860,
        justify="center",
        padx=16,
        pady=10,
    )
    label.pack(expand=True, fill="both")

    # Keep last 2 segments for context
    history = deque(maxlen=2)

    def poll_text():
        changed = False
        while not text_queue.empty():
            try:
                text = text_queue.get_nowait()
                history.append(text)
                changed = True
            except queue.Empty:
                break
        if changed:
            label.config(text=" / ".join(history))
        if stop_event.is_set():
            root.destroy()
            return
        root.after(80, poll_text)

    # Drag support
    drag = {"x": 0, "y": 0}

    def on_press(e):
        drag["x"] = e.x
        drag["y"] = e.y

    def on_drag(e):
        dx = e.x - drag["x"]
        dy = e.y - drag["y"]
        nx = root.winfo_x() + dx
        ny = root.winfo_y() + dy
        root.geometry(f"+{nx}+{ny}")

    root.bind("<ButtonPress-1>", on_press)
    root.bind("<B1-Motion>", on_drag)

    # Scroll to resize font
    def on_scroll(e):
        delta = 1 if e.delta > 0 else -1
        font_size[0] = max(12, min(48, font_size[0] + delta))
        label.config(font=("Segoe UI", font_size[0], "bold"))

    root.bind("<MouseWheel>", on_scroll)

    # Quit
    def quit_app(e=None):
        stop_event.set()
        root.destroy()

    root.bind("<q>", quit_app)
    root.bind("<Escape>", quit_app)

    # Right-click menu
    menu = tk.Menu(root, tearoff=0)
    menu.add_command(label="Quit", command=quit_app)
    menu.add_command(label="Toggle border", command=lambda: root.overrideredirect(not root.overrideredirect()))

    def show_menu(e):
        menu.tk_popup(e.x_root, e.y_root)

    root.bind("<ButtonPress-3>", show_menu)

    poll_text()
    root.mainloop()


# ===========================================================================
# 4. Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Live subtitles from system audio (Windows 11)")
    parser.add_argument("--list-devices",  action="store_true",    help="List loopback devices and exit")
    parser.add_argument("--device",        default=None,           help="Loopback device name (partial match OK)")
    parser.add_argument("--chunk_ms",      type=int,   default=320, help="Audio chunk size in ms")
    parser.add_argument("--ctc_weight",    type=float, default=0.5, help="CTC weight [0.0-1.0]")
    parser.add_argument("--beam_size",     type=int,   default=5,   help="Beam size (5=fast for live use)")
    parser.add_argument("--trigger",       default="blank_spike",
                        choices=["blank_spike", "word_boundary", "score_peak"],
                        help="Trigger condition for subtitle emission")
    args = parser.parse_args()

    if args.list_devices:
        list_loopback_devices()

    # Load model before starting threads
    s2t = load_model(beam_size=args.beam_size, ctc_weight=args.ctc_weight)

    # Shared primitives
    audio_queue = queue.Queue(maxsize=50)   # drops old audio if ASR falls behind
    text_queue  = queue.Queue()
    stop_event  = threading.Event()

    # Start capture thread
    cap = threading.Thread(
        target=capture_thread,
        args=(audio_queue, stop_event, args.device, args.chunk_ms),
        daemon=True,
    )
    cap.start()

    # Start ASR thread
    asr = threading.Thread(
        target=asr_thread,
        args=(audio_queue, text_queue, stop_event, s2t,
              args.chunk_ms, args.ctc_weight, args.beam_size, args.trigger),
        daemon=True,
    )
    asr.start()

    print("Overlay window opening — drag to reposition, scroll to resize, Q to quit.\n")

    # UI runs on the main thread (tkinter requirement)
    try:
        subtitle_overlay(text_queue, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("\nStopped.")


if __name__ == "__main__":
    main()
