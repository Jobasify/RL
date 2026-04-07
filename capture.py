"""
Factorio screen + audio capture pipeline.
Captures frames via mss, displays them with OpenCV, and records audio via PyAudio.
Press 'q' to stop. Press 's' to save a 3-second snapshot (frames + audio).
"""

import ctypes
import ctypes.wintypes
import time
import wave
import os
import threading
from pathlib import Path

import cv2
import mss
import numpy as np
import pyaudio


# ---------------------------------------------------------------------------
# Window detection
# ---------------------------------------------------------------------------

def find_factorio_window():
    """Find the Factorio window and return its bounding box (left, top, width, height)."""
    user32 = ctypes.windll.user32

    # Try common Factorio window titles
    titles_to_try = ["Factorio", "factorio"]
    hwnd = None
    for title in titles_to_try:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            break

    # Fallback: enumerate all windows looking for one containing "factorio"
    if not hwnd:
        found = []

        def enum_callback(h, _):
            length = user32.GetWindowTextLengthW(h)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(h, buf, length + 1)
                if "factorio" in buf.value.lower():
                    found.append((h, buf.value))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        user32.EnumWindows(WNDENUMPROC(enum_callback), 0)

        if found:
            hwnd, title = found[0]
            print(f"Found window: '{title}'")
        else:
            return None

    # Get window rect
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


# ---------------------------------------------------------------------------
# Audio capture (runs in background thread)
# ---------------------------------------------------------------------------

class AudioCapture:
    """Captures system audio via WASAPI loopback (Windows stereo mix)."""

    def __init__(self, rate=44100, channels=2, chunk=1024):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.frames = []
        self.recording = False
        self.saving = False
        self._pa = None
        self._stream = None
        self._thread = None
        self._lock = threading.Lock()

    def _find_loopback_device(self, pa):
        """Find a WASAPI loopback or stereo mix device."""
        device_count = pa.get_device_count()
        candidates = []
        for i in range(device_count):
            info = pa.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            max_input = info.get("maxInputChannels", 0)
            if max_input > 0 and any(kw in name for kw in ["stereo mix", "loopback", "what u hear", "wave out"]):
                candidates.append((i, info))
        if candidates:
            return candidates[0]
        # Fallback: just use default input
        default_idx = pa.get_default_input_device_info()["index"]
        return default_idx, pa.get_device_info_by_index(default_idx)

    def start(self):
        self._pa = pyaudio.PyAudio()
        idx, info = self._find_loopback_device(self._pa)
        device_index = idx if isinstance(idx, int) else idx
        print(f"Audio device: {info['name']} (index={device_index})")

        actual_rate = int(info.get("defaultSampleRate", self.rate))
        actual_channels = min(self.channels, int(info.get("maxInputChannels", self.channels)))
        self.rate = actual_rate
        self.channels = actual_channels

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk,
            )
        except Exception as e:
            print(f"WARNING: Could not open audio device: {e}")
            print("Audio capture disabled. Screen capture will still work.")
            self._pa.terminate()
            self._pa = None
            return False

        self.recording = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        return True

    def _record_loop(self):
        while self.recording:
            try:
                data = self._stream.read(self.chunk, exception_on_overflow=False)
                with self._lock:
                    self.frames.append(data)
                    # Keep only last 10 seconds in memory
                    max_frames = (self.rate // self.chunk) * 10
                    if len(self.frames) > max_frames:
                        self.frames = self.frames[-max_frames:]
            except Exception:
                break

    def save_snapshot(self, filepath, seconds=3):
        """Save the last N seconds of audio to a WAV file."""
        if not self._pa:
            print("Audio not available, skipping save.")
            return
        with self._lock:
            frames_needed = (self.rate // self.chunk) * seconds
            snapshot = list(self.frames[-frames_needed:])

        if not snapshot:
            print("No audio data captured yet.")
            return

        wf = wave.open(str(filepath), "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.rate)
        wf.writeframes(b"".join(snapshot))
        wf.close()
        print(f"Saved {len(snapshot)} audio chunks ({seconds}s) to {filepath}")

    def stop(self):
        self.recording = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def main():
    output_dir = Path("captures")
    output_dir.mkdir(exist_ok=True)

    # Find Factorio window
    print("Looking for Factorio window...")
    bbox = find_factorio_window()
    if bbox is None:
        print("Factorio window not found! Make sure the game is running.")
        print("Falling back to full primary monitor capture for testing.")
        bbox = None  # mss will capture the full screen

    if bbox:
        print(f"Factorio window: {bbox['width']}x{bbox['height']} at ({bbox['left']}, {bbox['top']})")

    # Start audio capture
    audio = AudioCapture()
    audio_ok = audio.start()

    # Screen capture loop
    print("\nCapturing... Press 'q' to quit, 's' to save a 3-second snapshot.")
    frame_count = 0
    fps_start = time.time()
    fps_display = 0.0

    with mss.mss() as sct:
        monitor = bbox if bbox else sct.monitors[1]  # primary monitor fallback

        while True:
            # Grab frame
            img = sct.grab(monitor)
            frame = np.array(img)  # BGRA
            frame = frame[:, :, :3]  # Drop alpha -> BGR (OpenCV native)

            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # Draw FPS on frame
            display = frame.copy()
            cv2.putText(display, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Resize for display if too large
            h, w = display.shape[:2]
            max_display = 1280
            if w > max_display:
                scale = max_display / w
                display = cv2.resize(display, (int(w * scale), int(h * scale)))

            cv2.imshow("Factorio Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Save snapshot: 3 seconds of frames + audio
                ts = int(time.time())
                snap_dir = output_dir / f"snapshot_{ts}"
                snap_dir.mkdir(exist_ok=True)

                # Save current frame as reference
                cv2.imwrite(str(snap_dir / "frame.png"), frame)
                print(f"Saved frame to {snap_dir / 'frame.png'}")

                # Save last 3 seconds of frames (capture burst)
                print("Capturing 3-second frame burst...")
                burst_frames = []
                burst_start = time.time()
                while time.time() - burst_start < 3.0:
                    img = sct.grab(monitor)
                    f = np.array(img)[:, :, :3]
                    burst_frames.append(f)

                for i, bf in enumerate(burst_frames):
                    cv2.imwrite(str(snap_dir / f"frame_{i:04d}.png"), bf)
                print(f"Saved {len(burst_frames)} frames to {snap_dir}/")

                # Save audio
                audio.save_snapshot(snap_dir / "audio.wav", seconds=3)
                print(f"Snapshot saved to {snap_dir}/\n")

    audio.stop()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
