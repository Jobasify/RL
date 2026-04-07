"""
Claude as the primary Factorio player.
CNN as the student observer learning from expert demonstrations.

Every 5 seconds Claude sees the screen, reasons about the game state,
and decides exactly one action. The CNN watches at 5x reward weight
and gradually learns to replicate Claude's decisions.

Over time Claude fades out and the CNN takes over.
"""

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np
from pynput.mouse import Button
from pynput.keyboard import Key

from control import FactorioController, mouse, keyboard

PLAYER_LOG = Path("logs/player.log")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
EXPERT_REWARD_MULTIPLIER = 5.0
DECISION_INTERVAL = 5.0  # Seconds between Claude decisions

# Fade schedule: (step_threshold, claude_pct)
FADE_SCHEDULE = [
    (0,     1.00),   # 0-10k: Claude 100%
    (10000, 0.70),   # 10k-30k: Claude 70%
    (30000, 0.40),   # 30k-60k: Claude 40%
    (60000, 0.10),   # 60k+: Claude 10%
]

SYSTEM_PROMPT = (
    "You are playing Factorio with the goal of launching a rocket. "
    "You have full knowledge of the game. Look at this screenshot and decide "
    "exactly one action to take right now. Consider your current situation, "
    "what you have, what you need, and what the most logical next step toward "
    "the rocket is.\n\n"
    "Return JSON only, no markdown:\n"
    '{"reason": "why you chose this", '
    '"action_type": "move/mine/build/craft/open_menu/close_menu/research", '
    '"direction": "north/south/east/west/none", '
    '"target": "what to interact with or none", '
    '"key": "specific key to press or none", '
    '"mouse_action": "left_click/right_click_hold/none", '
    '"duration": 3}'
)

# Map action types to execution
DIRECTION_KEYS = {
    "north": "w", "south": "s", "east": "d", "west": "a",
}

KEY_MAP = {
    "e": "e", "q": "q", "m": "m", "space": " ", "escape": Key.esc,
    "tab": Key.tab, "t": "t", "r": "r", "c": "c", "z": "z",
}


def _log(message):
    PLAYER_LOG.parent.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PLAYER_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def get_claude_ratio(step):
    """Return the probability that Claude plays this step."""
    ratio = FADE_SCHEDULE[0][1]
    for threshold, pct in FADE_SCHEDULE:
        if step >= threshold:
            ratio = pct
    return ratio


class ClaudePlayer:
    """Claude as the expert Factorio player."""

    def __init__(self, ctrl, monitor):
        self.ctrl = ctrl
        self.monitor = monitor
        self._client = None
        self.total_decisions = 0
        self.total_reward = 0.0

    def _init_client(self):
        if self._client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
            else:
                self._client = anthropic.Anthropic()

    def _capture_b64(self):
        """Capture screen as base64 JPEG."""
        with mss.mss() as sct:
            img = sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def decide(self):
        """Capture screen, ask Claude what to do. Returns action dict or None."""
        self._init_client()
        image_b64 = self._capture_b64()

        try:
            response = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": "What's your next move?"},
                    ],
                }],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            action = json.loads(text)
            self.total_decisions += 1
            return action
        except Exception as e:
            _log(f"DECIDE ERROR: {e}")
            return None

    def execute(self, action):
        """Execute Claude's decided action via control.py.
        Returns list of (action_id, duration) for CNN observation."""
        action_type = action.get("action_type", "move")
        direction = action.get("direction", "none")
        target = action.get("target", "none")
        key = action.get("key", "none")
        mouse_action = action.get("mouse_action", "none")
        duration = min(action.get("duration", 3), 10)
        reason = action.get("reason", "")

        print(f"  [CLAUDE] {action_type}: {reason[:80]}")
        _log(f"ACTION: {json.dumps(action)}")

        executed_actions = []

        # --- Movement ---
        if action_type == "move" and direction in DIRECTION_KEYS:
            k = DIRECTION_KEYS[direction]
            self.ctrl.hold_key(k, duration=duration)
            action_id = {"north": 0, "south": 2, "east": 3, "west": 1}[direction]
            executed_actions.append((action_id, duration))

        # --- Mining (right-click hold) ---
        elif action_type == "mine":
            # Move toward target direction first if specified
            if direction in DIRECTION_KEYS:
                k = DIRECTION_KEYS[direction]
                self.ctrl.hold_key(k, duration=min(duration / 2, 2))
                time.sleep(0.2)

            cx, cy = self.ctrl.width // 2, self.ctrl.height // 2
            self.ctrl.move(cx, cy)
            time.sleep(0.1)
            mouse.press(Button.right)
            time.sleep(duration)
            mouse.release(Button.right)
            executed_actions.append((9, duration))  # right click

        # --- Building ---
        elif action_type == "build":
            cx, cy = self.ctrl.width // 2, self.ctrl.height // 2
            self.ctrl.click(cx, cy, "left")
            time.sleep(0.3)
            executed_actions.append((8, 0.3))

        # --- Crafting ---
        elif action_type == "craft":
            self.ctrl.press_key("e", duration=0.1)
            time.sleep(0.8)
            # Click in crafting area (Claude should have specified where)
            cx, cy = self.ctrl.width // 2, self.ctrl.height // 2
            self.ctrl.click(cx, cy, "left")
            time.sleep(0.5)
            self.ctrl.press_key("e", duration=0.1)
            executed_actions.append((15, 1.5))  # E key

        # --- Open/close menu ---
        elif action_type in ("open_menu", "close_menu"):
            if key and key != "none" and key.lower() in KEY_MAP:
                actual_key = KEY_MAP[key.lower()]
                self.ctrl.press_key(actual_key, duration=0.1)
                executed_actions.append((15, 0.1))
            else:
                self.ctrl.press_key("e", duration=0.1)
                executed_actions.append((15, 0.1))
            time.sleep(0.5)

        # --- Research ---
        elif action_type == "research":
            self.ctrl.press_key("t", duration=0.1)
            time.sleep(1.0)
            executed_actions.append((15, 1.0))

        # --- Key press fallback ---
        elif key and key != "none":
            if key.lower() in KEY_MAP:
                actual_key = KEY_MAP[key.lower()]
                self.ctrl.press_key(actual_key, duration=0.1)
            time.sleep(0.3)
            executed_actions.append((17, 0.3))

        # --- Mouse action fallback ---
        elif mouse_action == "left_click":
            cx, cy = self.ctrl.width // 2, self.ctrl.height // 2
            self.ctrl.click(cx, cy, "left")
            executed_actions.append((8, 0.1))

        elif mouse_action == "right_click_hold":
            cx, cy = self.ctrl.width // 2, self.ctrl.height // 2
            self.ctrl.move(cx, cy)
            mouse.press(Button.right)
            time.sleep(duration)
            mouse.release(Button.right)
            executed_actions.append((9, duration))

        else:
            # No-op
            time.sleep(0.5)
            executed_actions.append((17, 0.5))

        return executed_actions


class RewardTracker:
    """Tracks Claude vs CNN reward separately."""

    def __init__(self):
        self.claude_rewards = []
        self.cnn_rewards = []

    def add_claude(self, reward):
        self.claude_rewards.append(reward)
        if len(self.claude_rewards) > 1000:
            self.claude_rewards.pop(0)

    def add_cnn(self, reward):
        self.cnn_rewards.append(reward)
        if len(self.cnn_rewards) > 1000:
            self.cnn_rewards.pop(0)

    def claude_avg(self, n=100):
        recent = self.claude_rewards[-n:]
        return sum(recent) / len(recent) if recent else 0.0

    def cnn_avg(self, n=100):
        recent = self.cnn_rewards[-n:]
        return sum(recent) / len(recent) if recent else 0.0

    def convergence(self):
        """How close CNN is to Claude's performance (0.0 to 1.0+)."""
        c = self.claude_avg()
        n = self.cnn_avg()
        if c <= 0:
            return 1.0 if n >= 0 else 0.0
        return n / c


def main():
    """Standalone test — Claude plays Factorio."""
    from capture import find_factorio_window

    print("=== Claude Player Test ===\n")

    bbox = find_factorio_window()
    if not bbox:
        print("Factorio not found!")
        return

    ctrl = FactorioController()
    player = ClaudePlayer(ctrl, bbox)

    print(f"Starting in 3 seconds — switch to Factorio!")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("\nClaude is playing. Press Ctrl+C to stop.\n")

    try:
        for step in range(20):
            print(f"\n--- Decision {step + 1} ---")
            action = player.decide()
            if action:
                print(f"  Reason: {action.get('reason', '?')}")
                print(f"  Action: {action.get('action_type')} "
                      f"dir={action.get('direction')} "
                      f"target={action.get('target')}")
                player.execute(action)
            else:
                print("  (no decision)")
            time.sleep(2)
    except KeyboardInterrupt:
        pass

    print(f"\nClaude made {player.total_decisions} decisions.")
    print("Done.")


if __name__ == "__main__":
    main()
