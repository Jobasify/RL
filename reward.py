"""
Visual reward signal for the Factorio RL agent.
Infers reward purely from pixel changes — no game API.

Monitors specific screen regions for change:
  - Resource HUD (top-right): item counts changing = gathering/production
  - Minimap (top-left): exploration and expansion
  - Hotbar (bottom): inventory changes = crafting/picking up items
  - Game world (center): structural changes = building/destruction

Reward is a weighted sum of detected changes in each region.
"""

import time

import cv2
import mss
import numpy as np

from capture import find_factorio_window


class RegionMonitor:
    """Tracks pixel changes in a named screen region."""

    def __init__(self, name, x, y, w, h, weight=1.0):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.weight = weight
        self.prev_gray = None
        self.change_score = 0.0
        self.stagnant_frames = 0

    def update(self, frame_bgr):
        """Compute change score for this region from a full BGR frame."""
        roi = frame_bgr[self.y:self.y + self.h, self.x:self.x + self.w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.change_score = 0.0
            return 0.0

        # Absolute pixel difference, normalised to [0, 1]
        diff = np.abs(gray - self.prev_gray) / 255.0

        # Mean change across the region
        raw_change = float(diff.mean())

        # Threshold out noise (small jitter, compression artifacts)
        noise_floor = 0.008
        self.change_score = max(0.0, raw_change - noise_floor)

        # Track stagnation
        if self.change_score < 0.001:
            self.stagnant_frames += 1
        else:
            self.stagnant_frames = 0

        self.prev_gray = gray
        return self.change_score


class RewardSignal:
    """Computes a composite reward from visual changes across screen regions."""

    def __init__(self, window_w, window_h):
        self.window_w = window_w
        self.window_h = window_h

        # Define monitoring regions scaled to window size
        # These target Factorio's default UI layout
        self.regions = [
            # Resource counters (top-right HUD area)
            RegionMonitor(
                "Resources",
                x=int(window_w * 0.85), y=int(window_h * 0.01),
                w=int(window_w * 0.14), h=int(window_h * 0.08),
                weight=3.0,  # Resource gain is high-value
            ),
            # Minimap (top-left)
            RegionMonitor(
                "Minimap",
                x=int(window_w * 0.01), y=int(window_h * 0.01),
                w=int(window_w * 0.12), h=int(window_h * 0.15),
                weight=1.0,  # Exploration
            ),
            # Hotbar / quickbar (bottom-center)
            RegionMonitor(
                "Hotbar",
                x=int(window_w * 0.25), y=int(window_h * 0.92),
                w=int(window_w * 0.50), h=int(window_h * 0.07),
                weight=2.0,  # Inventory changes = crafting/collecting
            ),
            # Game world center (where buildings appear)
            RegionMonitor(
                "World",
                x=int(window_w * 0.20), y=int(window_h * 0.20),
                w=int(window_w * 0.60), h=int(window_h * 0.55),
                weight=1.5,  # Structural changes
            ),
        ]

        self.total_reward = 0.0
        self.frame_count = 0

    def compute(self, frame_bgr):
        """Compute reward for the current frame. Returns (reward, details_dict)."""
        self.frame_count += 1
        details = {}
        reward = 0.0

        for region in self.regions:
            change = region.update(frame_bgr)
            weighted = change * region.weight
            details[region.name] = {
                "raw_change": change,
                "weighted": weighted,
                "stagnant": region.stagnant_frames,
            }
            reward += weighted

        # Bonus for significant change (new structure, big event)
        if reward > 0.05:
            reward *= 1.5  # Amplify big changes

        # Penalty for total stagnation (all regions idle)
        all_stagnant = all(r.stagnant_frames > 30 for r in self.regions)
        if all_stagnant:
            reward -= 0.01  # Gentle nudge to do something

        self.total_reward += reward
        return reward, details

    def draw_overlay(self, frame_bgr):
        """Draw monitoring regions and reward info on the frame."""
        display = frame_bgr.copy()

        for region in self.regions:
            # Color based on change: green = active, red = stagnant
            if region.stagnant_frames > 30:
                color = (0, 0, 180)  # Red
            elif region.change_score > 0.01:
                color = (0, 200, 0)  # Green
            else:
                color = (0, 140, 140)  # Yellow

            # Draw rectangle
            cv2.rectangle(display,
                          (region.x, region.y),
                          (region.x + region.w, region.y + region.h),
                          color, 2)

            # Label with name and score
            label = f"{region.name}: {region.change_score:.4f} (x{region.weight:.0f})"
            cv2.putText(display, label,
                        (region.x, region.y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return display


def main():
    print("=== Visual Reward Signal Test ===\n")

    bbox = find_factorio_window()
    if bbox is None:
        print("Factorio not found, falling back to primary monitor.")
        bbox = None

    if bbox:
        print(f"Factorio window: {bbox['width']}x{bbox['height']} at ({bbox['left']}, {bbox['top']})")
        win_w, win_h = bbox["width"], bbox["height"]
    else:
        win_w, win_h = 2560, 1440

    reward_signal = RewardSignal(win_w, win_h)

    print(f"\nMonitoring {len(reward_signal.regions)} regions:")
    for r in reward_signal.regions:
        print(f"  {r.name:12s} ({r.x}, {r.y}) {r.w}x{r.h}  weight={r.weight}")

    print("\nCapturing... Press 'q' to quit.\n")

    fps_count = 0
    fps_start = time.time()
    fps_display = 0.0

    with mss.mss() as sct:
        monitor = bbox if bbox else sct.monitors[1]

        while True:
            img = sct.grab(monitor)
            frame = np.array(img)[:, :, :3]

            reward, details = reward_signal.compute(frame)

            # FPS
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # Draw overlay
            display = reward_signal.draw_overlay(frame)

            # Info bar at top
            info_text = (f"Reward: {reward:+.5f}  "
                         f"Cumulative: {reward_signal.total_reward:+.3f}  "
                         f"Frame: {reward_signal.frame_count}  "
                         f"FPS: {fps_display:.0f}")
            cv2.putText(display, info_text, (10, win_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Resize for display
            max_display = 1280
            if display.shape[1] > max_display:
                scale = max_display / display.shape[1]
                display = cv2.resize(display, (int(display.shape[1] * scale),
                                               int(display.shape[0] * scale)))

            cv2.imshow("Reward Monitor", display)

            # Print every 60 frames
            if reward_signal.frame_count % 60 == 0:
                parts = "  ".join(f"{k}: {v['raw_change']:.4f}" for k, v in details.items())
                print(f"[{reward_signal.frame_count:5d}] reward={reward:+.5f}  cumul={reward_signal.total_reward:+.3f}  {parts}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"\nFinal cumulative reward: {reward_signal.total_reward:+.4f} over {reward_signal.frame_count} frames")
    print("Done.")


if __name__ == "__main__":
    main()
