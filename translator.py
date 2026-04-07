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
                import os
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    self._client = anthropic.Anthropic(api_key=api_key)
                else:
                    self._client = anthropic.Anthropic()
            except Exception as e:
                _log(f"Client init failed: {e}")
                return False
        return True

    def verify_api(self):
        """Test API auth on startup. Returns True if working."""
        if not self._init_client():
            print("  [TRANSLATOR] API client init FAILED")
            _log("API verify: client init failed")
            return False
        try:
            # Minimal API call to verify auth
            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": "Reply OK"}],
            )
            print("  [TRANSLATOR] API auth verified OK")
            _log("API verify: OK")
            return True
        except Exception as e:
            print(f"  [TRANSLATOR] API auth FAILED: {e}")
            _log(f"API verify: FAILED — {e}")
            return False

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
        """Find the densest ore pixel within a direction-biased search.
        Returns (x, y) in window coordinates or None."""
        if target == "none" or target not in TARGET_COLOURS:
            return None

        colours = TARGET_COLOURS[target]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, colours["lower"], colours["upper"])

        # Apply direction bias
        h, w = mask.shape
        cx, cy = w // 2, h // 2
        bx, by = DIRECTION_BIAS.get(direction, (0, 0))
        bias_x = cx + int(bx * w)
        bias_y = cy + int(by * h)
        ys, xs = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xs - bias_x) ** 2 + (ys - bias_y) ** 2).astype(np.float32)
        max_dist = np.sqrt(w ** 2 + h ** 2)
        weight = 1.0 - (dist / max_dist) * 0.5

        weighted_mask = mask.astype(np.float32) * weight

        # Coarse search: find best 64x64 cell
        best_score = 0
        best_cx, best_cy = cx, cy
        cell = 64
        for y in range(0, h - cell, cell // 2):
            for x in range(0, w - cell, cell // 2):
                score = weighted_mask[y:y + cell, x:x + cell].sum()
                if score > best_score:
                    best_score = score
                    best_cx = x + cell // 2
                    best_cy = y + cell // 2

        if best_score < 100:
            return None

        # Fine search: find densest point within 50px radius of coarse match
        r = 50
        x1 = max(0, best_cx - r)
        y1 = max(0, best_cy - r)
        x2 = min(w, best_cx + r)
        y2 = min(h, best_cy + r)
        patch = mask[y1:y2, x1:x2]

        if patch.sum() == 0:
            return (best_cx, best_cy)

        # Gaussian blur to find density peak
        blurred = cv2.GaussianBlur(patch.astype(np.float32), (21, 21), 0)
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        fine_x = x1 + max_loc[0]
        fine_y = y1 + max_loc[1]

        _log(f"Target located: coarse=({best_cx},{best_cy}) -> fine=({fine_x},{fine_y})")
        return (fine_x, fine_y)

    def walk_toward(self, target_pos, frame_bgr, target, sct):
        """Walk the character to ore using fixed-duration walk based on pixel offset.
        Character is at screen center. Walk time = pixel distance / speed constant.
        Then re-scan once to find ore at the new position."""
        cx = self.ctrl.width // 2
        cy = self.ctrl.height // 2
        tx, ty = target_pos
        proximity = 300  # Mining range is generous at 1440p
        px_per_second = 400

        dx = tx - cx
        dy = ty - cy
        dist = (dx ** 2 + dy ** 2) ** 0.5

        if dist <= proximity:
            print(f"  [WALK] Ore within {dist:.0f}px of center — in mining range, no walk")
            return (tx, ty)

        print(f"  [WALK] Ore at ({tx},{ty}), {dist:.0f}px from center")
        print(f"  [WALK] Offset: {dx:+.0f}px horizontal, {dy:+.0f}px vertical")

        # Walk horizontal component
        if abs(dx) > 100:
            key = "d" if dx > 0 else "a"
            walk_time = abs(dx) / px_per_second
            print(f"  [WALK] Walking {'east' if dx > 0 else 'west'} for {walk_time:.1f}s")
            self.ctrl.hold_key(key, duration=walk_time)
            time.sleep(0.1)

        # Walk vertical component
        if abs(dy) > 100:
            key = "s" if dy > 0 else "w"
            walk_time = abs(dy) / px_per_second
            print(f"  [WALK] Walking {'south' if dy > 0 else 'north'} for {walk_time:.1f}s")
            self.ctrl.hold_key(key, duration=walk_time)
            time.sleep(0.1)

        # Re-scan for ore at new position
        print(f"  [WALK] Arrived — re-scanning for {target}...")
        import mss as mss_mod
        with mss_mod.mss() as cap_sct:
            img = cap_sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]
        new_pos = self.locate_target(frame, target)

        if new_pos:
            ntx, nty = new_pos
            new_dist = ((ntx - cx) ** 2 + (nty - cy) ** 2) ** 0.5
            # If walk made things worse, use original — character is probably already in range
            if new_dist > dist:
                print(f"  [WALK] Ore moved further ({new_dist:.0f}px > {dist:.0f}px) — "
                      f"using original position, likely in range")
                return (tx, ty)
            print(f"  [WALK] Ore now at ({ntx},{nty}), {new_dist:.0f}px from center")
            return (ntx, nty)
        else:
            print(f"  [WALK] Ore not found in re-scan — using screen center")
            return (cx, cy)

    def verify_on_ore(self, tx, ty):
        """Ask Claude if cursor is on an ore tile. Returns True/False."""
        if not self._init_client():
            return True  # Can't verify, assume yes

        try:
            import mss as mss_mod
            with mss_mod.mss() as cap_sct:
                img = cap_sct.grab(self.monitor)
            frame = np.array(img)[:, :, :3]

            # Crop 100x100 around cursor
            h, w = frame.shape[:2]
            x1 = max(0, tx - 50)
            y1 = max(0, ty - 50)
            x2 = min(w, tx + 50)
            y2 = min(h, ty + 50)
            crop = frame[y1:y2, x1:x2]

            _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=50,
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
                            "text": ("Is the cursor sitting directly on a minable ore tile "
                                     "or on empty ground? Return JSON only: "
                                     '{"on_ore": true} or {"on_ore": false}'),
                        },
                    ],
                }],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            on_ore = result.get("on_ore", False)
            print(f"  [TARGET] Ore check at ({tx},{ty}): {'ON ORE' if on_ore else 'MISS'}")
            _log(f"ORE CHECK ({tx},{ty}): on_ore={on_ore}")
            return on_ore
        except Exception as e:
            _log(f"Ore verify failed: {e}")
            return True  # Assume yes on failure

    def nudge_to_ore(self, tx, ty):
        """Try nudging cursor ±10px in each direction until on ore.
        Returns corrected (x, y) or original if all fail."""
        nudges = [(0, 0), (10, 0), (-10, 0), (0, 10), (0, -10),
                  (10, 10), (-10, -10), (10, -10), (-10, 10)]
        for dx, dy in nudges:
            nx, ny = tx + dx, ty + dy
            self.ctrl.move(nx, ny)
            time.sleep(0.15)
            if self.verify_on_ore(nx, ny):
                if dx != 0 or dy != 0:
                    print(f"  [TARGET] Nudged to ({nx},{ny}) — on ore")
                    _log(f"NUDGE SUCCESS: ({tx},{ty}) -> ({nx},{ny})")
                return nx, ny
        print(f"  [TARGET] All nudges failed, using original ({tx},{ty})")
        return tx, ty

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
        time.sleep(1.2)  # Factorio needs time to render tooltips

        # Capture screenshot with tooltip visible
        import mss as mss_mod
        with mss_mod.mss() as hover_sct:
            img = hover_sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]

        # Crop around cursor, biased rightward (Factorio renders tooltips to the right)
        h, w = frame.shape[:2]
        x1 = max(0, tx - 150)
        y1 = max(0, ty - 300)
        x2 = min(w, tx + 450)
        y2 = min(h, ty + 300)
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

        except json.JSONDecodeError:
            _log(f"TOOLTIP MISS: Claude returned non-JSON response")
            print(f"  [HOVER] TOOLTIP MISS — no structured data, proceeding with demo")
            return None, True
        except Exception as e:
            _log(f"TOOLTIP MISS: {e}")
            print(f"  [HOVER] TOOLTIP MISS — {e}, proceeding with demo")
            return None, True

    # ------------------------------------------------------------------
    # Step 2.7: Craft intent — Claude-guided UI navigation
    # ------------------------------------------------------------------

    def _capture_b64(self):
        """Capture full screen as base64 JPEG for Claude."""
        import mss as mss_mod
        with mss_mod.mss() as cap_sct:
            img = cap_sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8"), frame

    def _ask_claude_with_image(self, image_b64, prompt):
        """Send image + prompt to Claude, return text response."""
        response = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
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
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text.strip()

    def execute_craft(self, intent, sct):
        """Handle craft intent with Claude-guided UI navigation.
        Opens inventory, asks Claude what to click, executes, closes inventory."""
        if not self._init_client():
            print("  [CRAFT] No API client, skipping craft demo")
            return 0, 0.0

        item = intent.get("target", "burner mining drill")
        if item == "none":
            item = "burner mining drill"

        print(f"  [CRAFT] Starting craft sequence for: {item}")
        _log(f"CRAFT START: {item}")

        steps = 0
        total_reward = 0.0

        # Step 1: Open inventory
        print(f"  [CRAFT] Opening inventory (E)...")
        self.ctrl.press_key("e", duration=0.1)
        time.sleep(0.8)  # Wait for inventory animation

        # Step 2: Capture and ask Claude what to click
        image_b64, _ = self._capture_b64()
        try:
            craft_prompt = (
                f"The agent wants to craft '{item}' in Factorio. "
                f"Look at this screenshot. Is the crafting/inventory menu open? "
                f"Can you see '{item}' or its components in the crafting panel? "
                f"If yes, describe exactly where to click to craft it — give approximate "
                f"screen coordinates as percentage from left/top (e.g. 60% from left, 40% from top). "
                f"If the item isn't visible, say what's needed first. "
                f'Return JSON only: {{"visible": true/false, "click_x_pct": 0.0-1.0, '
                f'"click_y_pct": 0.0-1.0, "instructions": "..."}}'
            )
            response_text = self._ask_claude_with_image(image_b64, craft_prompt)

            # Parse response
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            craft_info = json.loads(response_text)

            visible = craft_info.get("visible", False)
            instructions = craft_info.get("instructions", "")

            print(f"  [CRAFT] Claude says: {instructions}")
            _log(f"CRAFT VISION: {json.dumps(craft_info)}")

            if visible:
                # Step 3: Click where Claude says
                click_x = int(craft_info.get("click_x_pct", 0.5) * self.ctrl.width)
                click_y = int(craft_info.get("click_y_pct", 0.5) * self.ctrl.height)
                print(f"  [CRAFT] Clicking at ({click_x}, {click_y})")

                self.ctrl.click(click_x, click_y, "left")
                time.sleep(0.3)

                # Click again for quantity (some items need multiple clicks)
                self.ctrl.click(click_x, click_y, "left")
                time.sleep(0.3)

                # Capture reward from the crafting action
                import mss as mss_mod
                with mss_mod.mss() as cap_sct:
                    img = cap_sct.grab(self.monitor)
                frame = np.array(img)[:, :, :3]
                r, _ = self.reward_signal.compute(frame)
                total_reward += r
                steps += 1

                # Step 3.5: Verify — did crafting work?
                verify_b64, _ = self._capture_b64()
                verify_text = self._ask_claude_with_image(
                    verify_b64,
                    f"Did the crafting succeed? Can you see '{item}' being crafted or "
                    f"appearing in the inventory? Just say yes or no and what you see."
                )
                print(f"  [CRAFT] Verify: {verify_text[:100]}")
                _log(f"CRAFT VERIFY: {verify_text}")
            else:
                print(f"  [CRAFT] Item not visible in crafting menu")
                _log(f"CRAFT: item not visible")

        except Exception as e:
            print(f"  [CRAFT] Claude guidance failed: {e}")
            _log(f"CRAFT ERROR: {e}")

        # Step 4: Close inventory
        print(f"  [CRAFT] Closing inventory (E)...")
        self.ctrl.press_key("e", duration=0.1)
        time.sleep(0.5)

        # Capture final reward
        import mss as mss_mod
        with mss_mod.mss() as cap_sct:
            img = cap_sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]
        r, _ = self.reward_signal.compute(frame)
        total_reward += r
        steps += 1

        print(f"  [CRAFT] Complete: {steps} steps, reward: {total_reward:+.3f}")
        _log(f"CRAFT END: {steps} steps, reward={total_reward:.3f}")
        return steps, total_reward

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

        # Mining uses right-click hold in Factorio (not left-click)
        if intent_name == "mine":
            primary_action_id = 9  # Right click
            print(f"  [TRANSLATOR] Intent: {intent_name} (right-click hold), "
                  f"target: {target}, direction: {direction}, duration: {duration}s")
        else:
            print(f"  [TRANSLATOR] Intent: {intent_name}, target: {target}, "
                  f"direction: {direction}, duration: {duration}s")
        _log(f"DEMO START: {intent}")

        # Route craft intents to specialized handler
        if intent_name == "craft":
            return self.execute_craft(intent, sct)

        # Capture initial frame to try to locate target
        img = sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]
        target_pos = self.locate_target(frame, target, direction)

        if target_pos:
            tx, ty = target_pos
            print(f"  [DEMO] Target '{target}' found at ({tx}, {ty})")
            _log(f"Target located: ({tx}, {ty})")

            # Walk to ore if too far from character (screen center)
            if intent_name == "mine":
                target_pos = self.walk_toward(target_pos, frame, target, sct)
                tx, ty = target_pos

                # Precision targeting: verify cursor is on ore, nudge if not
                self.ctrl.move(tx, ty)
                time.sleep(0.15)
                tx, ty = self.nudge_to_ore(tx, ty)
                target_pos = (tx, ty)

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

        # For mining: hold right-click on target for the full duration
        mining_hold = (intent_name == "mine" and target_pos)
        if mining_hold:
            from pynput.mouse import Button
            from control import mouse
            print(f"  [DEMO] Holding right-click on target for {duration}s...")
            mouse.press(Button.right)

        while time.time() - demo_start < duration:
            # Capture observation before action
            img = sct.grab(self.monitor)
            frame = np.array(img)[:, :, :3]
            self.obs_proc.push(frame)
            obs = self.obs_proc.get()

            # Decide action for this step
            if mining_hold:
                # Mouse is already held — just keep it pressed, no extra action
                action_id = 17  # No-op (hold is already active)
            elif target_pos and primary_action_id in (8, 9):
                action_id = primary_action_id
            elif move_action_id is not None and (not target_pos or steps % 3 == 0):
                action_id = move_action_id
            else:
                action_id = primary_action_id

            # Execute with frame skip (for mining hold, just capture frames)
            step_reward = 0.0
            for _ in range(FRAME_SKIP):
                if not mining_hold:
                    execute_action(self.ctrl, action_id)
                img2 = sct.grab(self.monitor)
                skip_frame = np.array(img2)[:, :, :3]
                r, details = self.reward_signal.compute(skip_frame)
                step_reward += r
                if details.get("inventory_gain"):
                    print(f"  [DEMO] MINING SUCCESS — inventory gained!")
                time.sleep(0.05)  # Pace for mining hold

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

        # Release right-click if mining hold was active
        if mining_hold:
            from pynput.mouse import Button
            from control import mouse
            mouse.release(Button.right)
            print(f"  [DEMO] Released right-click hold")

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
