"""
Observation preprocessing pipeline for the RL agent.
Takes raw Factorio frames and produces stacked, normalised observations
ready for a neural network: (4, 128, 128) float32 in [0, 1].
"""

import time

import cv2
import mss
import numpy as np

from capture import find_factorio_window


class ObservationProcessor:
    """Preprocesses raw frames into neural network observations."""

    def __init__(self, size=128, stack_size=4):
        self.size = size
        self.stack_size = stack_size
        self.frame_stack = []

    def preprocess(self, frame_bgr):
        """Raw BGR frame -> (128, 128) float32 in [0, 1]."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def push(self, frame_bgr):
        """Add a frame and return the stacked observation once we have enough."""
        processed = self.preprocess(frame_bgr)
        self.frame_stack.append(processed)
        if len(self.frame_stack) > self.stack_size:
            self.frame_stack.pop(0)

    def get(self):
        """Return the current stacked observation: shape (stack_size, 128, 128).
        Pads with zeros if we don't have enough frames yet."""
        if not self.frame_stack:
            return np.zeros((self.stack_size, self.size, self.size), dtype=np.float32)
        while len(self.frame_stack) < self.stack_size:
            self.frame_stack.insert(0, self.frame_stack[0])
        return np.stack(self.frame_stack, axis=0)

    def reset(self):
        self.frame_stack.clear()


def main():
    print("=== Observation Pipeline Test ===\n")

    bbox = find_factorio_window()
    if bbox is None:
        print("Factorio not found, falling back to primary monitor.")
        bbox = None

    if bbox:
        print(f"Factorio window: {bbox['width']}x{bbox['height']} at ({bbox['left']}, {bbox['top']})")

    proc = ObservationProcessor(size=128, stack_size=4)

    print(f"\nPreprocessing: resize to {proc.size}x{proc.size}, grayscale, normalise [0,1], stack {proc.stack_size} frames")
    print("Press 'q' to quit.\n")

    fps_count = 0
    fps_start = time.time()
    fps_display = 0.0

    with mss.mss() as sct:
        monitor = bbox if bbox else sct.monitors[1]

        while True:
            img = sct.grab(monitor)
            frame = np.array(img)[:, :, :3]

            proc.push(frame)
            obs = proc.get()

            # FPS
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # Build display: show all 4 stacked frames side by side
            display_frames = []
            for i in range(proc.stack_size):
                # Convert back to uint8 for display
                vis = (obs[i] * 255).astype(np.uint8)
                # Add frame label
                cv2.putText(vis, f"t-{proc.stack_size - 1 - i}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
                display_frames.append(vis)

            # Horizontal stack of all 4 frames
            strip = np.hstack(display_frames)

            # Scale up for visibility (4 * 128 = 512 wide, scale to ~1024)
            scale = 2
            strip_big = cv2.resize(strip, (strip.shape[1] * scale, strip.shape[0] * scale),
                                   interpolation=cv2.INTER_NEAREST)

            # Add info bar
            info = np.zeros((40, strip_big.shape[1]), dtype=np.uint8)
            cv2.putText(info, f"Observation shape: {obs.shape}  dtype: {obs.dtype}  "
                        f"range: [{obs.min():.2f}, {obs.max():.2f}]  FPS: {fps_display:.0f}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)

            final = np.vstack([strip_big, info])
            cv2.imshow("RL Observation", final)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    # Print final summary
    print(f"\nFinal observation shape: {obs.shape}")
    print(f"  dtype: {obs.dtype}")
    print(f"  range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  meaning: {obs.shape[0]} stacked frames, each {obs.shape[1]}x{obs.shape[2]} grayscale, normalised")
    print("\nDone.")


if __name__ == "__main__":
    main()
