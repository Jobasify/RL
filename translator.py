"""
Action translator: bridges Claude's natural language advice to game execution.
Parses advice into structured intents, locates targets visually,
executes demonstrations, and records them as high-value experiences.
"""

import base64
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np

DEMO_REWARD_MULTIPLIER = 3.0
DEMO_DURATION_DEFAULT = 10
DEMO_LOG = Path("logs/translator.log")

PARSE_PROMPT = """Convert this Factorio advice into a structured action intent. Return JSON only, no markdown:
{
  "intent": "mine/build/move/craft/explore",
  "direction": "north/south/east/west/none",
  "target": "iron_ore/copper_ore/stone/tree/none",
  "duration_seconds": 5,
  "primary_action": "left_click/right_click/key_e/key_w/key_a/key_s/key_d",
  "repeat": true
}

Advice: """

# HSV colour ranges for target detection in Factorio
TARGET_COLOURS = {
    "iron_ore": {
        "lower": np.array([0, 20, 60]),
        "upper": np.array([25, 120, 160]),
        "name": "iron ore (brownish-grey)",
    },
    "copper_ore": {
        "lower": np.array([8, 80, 100]),
        "upper": np.array([25, 200, 220]),
        "name": "copper ore (orange-brown)",
    },
    "stone": {
        "lower": np.array([0, 0, 150]),
        "upper": np.array([30, 40, 220]),
        "name": "stone (light grey)",
    },
    "tree": {
        "lower": np.array([35, 40, 30]),
        "upper": np.array([85, 255, 150]),
        "name": "trees (dark green)",
    },
}

# Map direction strings to screen quadrant bias (x_bias, y_bias) relative to center
DIRECTION_BIAS = {
    "north": (0.0, -0.3),
    "south": (0.0, 0.3),
    "east": (0.3, 0.0),
    "west": (-0.3, 0.0),
    "none": (0.0, 0.0),
}

# Map primary_action strings to network action IDs
ACTION_MAP = {
    "left_click": 8,
    "right_click": 9,
    "key_e": 15,
    "key_w": 0,
    "key_a": 1,
    "key_s": 2,
    "key_d": 3,
}

# Direction to movement action
DIRECTION_MOVE = {
    "north": 0,   # W
    "south": 2,   # S
    "east": 3,    # D
    "west": 1,    # A
}


def _log(message):
    DEMO_LOG.parent.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DEMO_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


class ActionTranslator:
    """Translates natural language advice into executed game actions."""

    def __init__(self, ctrl, obs_proc, reward_signal, replay, rollout,
                 monitor, strategy_vec, sct, audio_proc=None, audio_capture=None):
        self.ctrl = ctrl
        self.obs_proc = obs_proc
        self.reward_signal = reward_signal
        self.replay = replay
        self.rollout = rollout
        self.monitor = monitor
        self.strategy_vec = strategy_vec
        self.sct = sct
        self.audio_proc = audio_proc
        self.audio_capture = audio_capture
        self._client = None

    def _init_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except Exception as e:
                _log(f"Client init failed: {e}")
                return False
        return True

    # ------------------------------------------------------------------
    # Step 1: Parse advice into intent
    # ------------------------------------------------------------------

    def parse_intent(self, advice_text):
        """Send advice to Claude to get structured action intent."""
        if not self._init_client():
            return None
        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": PARSE_PROMPT + advice_text,
                }],
            )
            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            intent = json.loads(text)
            return intent
        except Exception as e:
            _log(f"Parse failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Step 2: Locate target visually
    # ------------------------------------------------------------------

    def locate_target(self, frame_bgr, target, direction="none"):
        """Find target on screen using colour detection + direction bias.
        Returns (x, y) in window coordinates or None."""
        if target == "none" or target not in TARGET_COLOURS:
            return None

        colours = TARGET_COLOURS[target]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, colours["lower"], colours["upper"])

        # Apply direction bias — weight pixels in the hinted quadrant
        h, w = mask.shape
        cx, cy = w // 2, h // 2
        bx, by = DIRECTION_BIAS.get(direction, (0, 0))

        # Create weight map biased toward direction
        ys, xs = np.mgrid[0:h, 0:w]
        bias_x = cx + int(bx * w)
        bias_y = cy + int(by * h)
        dist = np.sqrt((xs - bias_x) ** 2 + (ys - bias_y) ** 2).astype(np.float32)
        max_dist = np.sqrt(w ** 2 + h ** 2)
        weight = 1.0 - (dist / max_dist) * 0.5  # Closer to bias = higher weight

        weighted_mask = mask.astype(np.float32) * weight

        # Find the region with highest weighted density
        # Use a coarse grid search
        best_score = 0
        best_pos = None
        cell = 64
        for y in range(0, h - cell, cell // 2):
            for x in range(0, w - cell, cell // 2):
                score = weighted_mask[y:y + cell, x:x + cell].sum()
                if score > best_score:
                    best_score = score
                    best_pos = (x + cell // 2, y + cell // 2)

        if best_score < 100:  # Minimum threshold
            return None

        return best_pos

    # ------------------------------------------------------------------
    # Step 2.5: Hover and read tooltip
    # ------------------------------------------------------------------

    def hover_and_read(self, target_pos, sct):
        """Hover over target, wait for tooltip, ask Claude to read it.

        Returns:
            tooltip_info: dict with entity_type, quantity, etc. or None on failure
            should_proceed: True if the tooltip confirms a valid target
        """
        if not self._init_client():
            return None, True  # Can't read, proceed anyway

        tx, ty = target_pos

        # Move mouse slowly to target (looks natural, triggers tooltip)
        self.ctrl.move(tx, ty)
        print(f"  [HOVER] Hovering at ({tx}, {ty}), waiting for tooltip...")
        time.sleep(0.5)  # Wait for Factorio tooltip to appear

        # Capture screenshot with tooltip visible
        import mss as mss_mod
        with mss_mod.mss() as hover_sct:
            img = hover_sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]

        # Crop region around cursor for focused tooltip read (±200px)
        h, w = frame.shape[:2]
        x1 = max(0, tx - 200)
        y1 = max(0, ty - 200)
        x2 = min(w, tx + 200)
        y2 = min(h, ty + 200)
        crop = frame[y1:y2, x1:x2]

        # Encode cropped region as base64 JPEG
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
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
                        {
                            "type": "text",
                            "text": (
                                "Read any tooltip or text visible in this Factorio screenshot. "
                                "What exactly is the player hovering over? What does the tooltip "
                                "say about quantity, type, or available actions? "
                                'Return JSON only, no markdown: '
                                '{"entity_type": "...", "quantity": "...", '
                                '"available_action": "...", "description": "..."}'
                            ),
                        },
                    ],
                }],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            tooltip_info = json.loads(text)

            entity = tooltip_info.get("entity_type", "unknown")
            qty = tooltip_info.get("quantity", "unknown")
            desc = tooltip_info.get("description", "")

            print(f"  [HOVER] Read: {entity} (qty: {qty})")
            if desc:
                print(f"  [HOVER] Detail: {desc}")
            _log(f"TOOLTIP at ({tx},{ty}): {json.dumps(tooltip_info)}")

            # Decide if we should proceed
            entity_lower = entity.lower()
            # Skip if it's clearly not what we expected (e.g. hovering over water, UI elements)
            skip_keywords = ["water", "void", "nothing", "no tooltip", "ui", "button", "menu"]
            if any(kw in entity_lower for kw in skip_keywords):
                print(f"  [HOVER] Unexpected target: '{entity}' — skipping demo")
                _log(f"SKIP: unexpected entity '{entity}'")
                return tooltip_info, False

            return tooltip_info, True

        except Exception as e:
            _log(f"Tooltip read failed: {e}")
            print(f"  [HOVER] Tooltip read failed, proceeding anyway")
            return None, True  # Proceed on failure

    # ------------------------------------------------------------------
    # Step 3 & 4: Execute demonstration and record experiences
    # ------------------------------------------------------------------

    def execute_demo(self, intent, sct):
        """Execute a demonstration based on parsed intent.
        Returns (steps_taken, total_reward)."""
        from train import execute_action, FRAME_SKIP

        target = intent.get("target", "none")
        direction = intent.get("direction", "none")
        duration = min(intent.get("duration_seconds", DEMO_DURATION_DEFAULT), 15)
        primary = intent.get("primary_action", "left_click")
        repeat = intent.get("repeat", True)
        intent_name = intent.get("intent", "unknown")

        primary_action_id = ACTION_MAP.get(primary, 8)
        move_action_id = DIRECTION_MOVE.get(direction)

        print(f"  [TRANSLATOR] Intent: {intent_name}, target: {target}, "
              f"direction: {direction}, duration: {duration}s")
        _log(f"DEMO START: {intent}")

        # Capture initial frame to try to locate target
        img = sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]
        target_pos = self.locate_target(frame, target, direction)

        if target_pos:
            tx, ty = target_pos
            print(f"  [DEMO] Target '{target}' found at ({tx}, {ty})")
            _log(f"Target located: ({tx}, {ty})")

            # Hover and read tooltip before committing
            tooltip_info, should_proceed = self.hover_and_read(target_pos, sct)
            if not should_proceed:
                print(f"  [DEMO] Aborted — tooltip didn't match expected target")
                return 0, 0.0

        elif direction != "none" and move_action_id is not None:
            print(f"  [DEMO] No target found, moving {direction}")

        steps = 0
        total_reward = 0.0
        demo_start = time.time()

        while time.time() - demo_start < duration:
            # Capture observation before action
            img = sct.grab(self.monitor)
            frame = np.array(img)[:, :, :3]
            self.obs_proc.push(frame)
            obs = self.obs_proc.get()

            # Decide action for this step
            if target_pos and primary_action_id in (8, 9):
                # Click on target
                action_id = primary_action_id
            elif move_action_id is not None and (not target_pos or steps % 3 == 0):
                # Move in the hinted direction periodically
                action_id = move_action_id
            else:
                action_id = primary_action_id

            # Execute with frame skip
            step_reward = 0.0
            for _ in range(FRAME_SKIP):
                execute_action(self.ctrl, action_id)
                img2 = sct.grab(self.monitor)
                skip_frame = np.array(img2)[:, :, :3]
                r, _ = self.reward_signal.compute(skip_frame)
                step_reward += r

            # Apply demo reward multiplier
            boosted_reward = step_reward * DEMO_REWARD_MULTIPLIER

            # Capture next observation
            self.obs_proc.push(skip_frame)
            next_obs = self.obs_proc.get()

            # Store as high-value experience
            self.replay.push(obs, action_id, boosted_reward, next_obs, False)
            if self.rollout is not None:
                # Use 0 log_prob and 0 value for demo steps —
                # PPO won't use these for ratio computation properly,
                # but the reward signal still shapes the value function
                # Get audio features for this step (zero fallback for shape consistency)
                from audio import AUDIO_FEATURE_DIM
                demo_audio = None
                if self.audio_proc and self.audio_capture:
                    try:
                        demo_audio, _ = self.audio_proc.process(self.audio_capture)
                    except Exception:
                        demo_audio = np.zeros(AUDIO_FEATURE_DIM, dtype=np.float32)
                elif self.audio_proc is not None:
                    # Audio proc exists but capture unavailable — use zeros
                    demo_audio = np.zeros(AUDIO_FEATURE_DIM, dtype=np.float32)
                self.rollout.push(obs, action_id, 0.0, boosted_reward, 0.0, False,
                                  strategy=self.strategy_vec, audio=demo_audio)

            total_reward += step_reward
            steps += 1

            if steps % 5 == 0:
                print(f"  [DEMO] step {steps}, reward: {step_reward:+.3f} "
                      f"(x{DEMO_REWARD_MULTIPLIER:.0f} = {boosted_reward:+.3f})")

            # Re-locate target periodically
            if target_pos and steps % 10 == 0:
                new_pos = self.locate_target(skip_frame, target, direction)
                if new_pos:
                    target_pos = new_pos
                    self.ctrl.move(target_pos[0], target_pos[1])

        _log(f"DEMO END: {steps} steps, reward={total_reward:.3f}")
        print(f"  [DEMO] Complete: {steps} steps, total reward: {total_reward:+.3f}")
        return steps, total_reward

    # ------------------------------------------------------------------
    # Full pipeline: advice -> parse -> locate -> execute -> return
    # ------------------------------------------------------------------

    def translate_and_execute(self, advice_text, sct):
        """Full pipeline: parse advice, find target, execute demo."""
        print(f"\n  [ADVISOR] Claude says: {advice_text.split(chr(10))[0]}")

        # Parse first suggestion only
        first_line = ""
        for line in advice_text.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                first_line = line.lstrip("0123456789.) ").strip()
                break
        if not first_line:
            first_line = advice_text.split("\n")[0].strip()

        intent = self.parse_intent(first_line)
        if intent is None:
            print("  [TRANSLATOR] Failed to parse intent, skipping demo")
            return 0, 0.0

        steps, reward = self.execute_demo(intent, sct)
        print(f"  [CNN] Resuming control\n")
        return steps, reward
