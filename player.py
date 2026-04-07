"""
Claude as the direct Factorio player.
No interpretation layer — Claude outputs exact key/mouse sequences,
control.py executes them literally.

CNN observes at 5x reward weight and learns to replicate.
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
from craft import CraftingSystem, PlacementSystem, RECIPES, DISPLAY_NAMES

PLAYER_LOG = Path("logs/player.log")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
EXPERT_REWARD_MULTIPLIER = 5.0
DECISION_INTERVAL = 30.0

# --- API cost controls ---
MAX_CALLS_PER_HOUR = 60
HARD_SPEND_LIMIT = 5.00
EST_INPUT_TOKENS_PER_CALL = 1500
EST_OUTPUT_TOKENS_PER_CALL = 200
COST_PER_INPUT_TOKEN = 3.00 / 1_000_000
COST_PER_OUTPUT_TOKEN = 15.00 / 1_000_000
EST_COST_PER_CALL = (EST_INPUT_TOKENS_PER_CALL * COST_PER_INPUT_TOKEN +
                     EST_OUTPUT_TOKENS_PER_CALL * COST_PER_OUTPUT_TOKEN)

# Fade schedule: (step_threshold, claude_pct)
FADE_SCHEDULE = [
    (0,     1.00),
    (10000, 0.70),
    (30000, 0.40),
    (60000, 0.10),
]

SYSTEM_PROMPT = (
    "You are directly controlling Factorio through keyboard and mouse. "
    "The character is always at screen center (1280, 720 on a 2560x1440 screen). "
    "Your goal is to launch a rocket.\n\n"
    "You are called every 30 seconds. Each call should include a COMPLETE action "
    "sequence — move AND mine/build/craft in one response. Every response must "
    "include at least one right_click_hold or left_click, not just key presses.\n\n"
    "IMPORTANT: You have persistent memory. Your game state is provided with each "
    "call. Do NOT repeat actions you have done enough of. Progress through the "
    "tech tree: mine ore → craft stone furnace → smelt plates → craft drills → "
    "automate mining → craft assemblers → research → expand → launch rocket.\n\n"
    "Return JSON only, no markdown:\n"
    '{\n'
    '  "reasoning": "why — reference your inventory and current goal",\n'
    '  "inputs": [\n'
    '    {"type": "key", "key": "w", "duration": 0.5},\n'
    '    {"type": "right_click_hold", "x": 1280, "y": 720, "duration": 3.0}\n'
    '  ],\n'
    '  "state_update": {\n'
    '    "inventory_changes": {"iron_ore": 5},\n'
    '    "milestone_reached": "mined first iron" or null,\n'
    '    "current_goal": "what to focus on now",\n'
    '    "next_milestone": "next thing to achieve"\n'
    '  }\n'
    '}\n\n'
    "IMPORTANT FACTORIO CONTROLS:\n"
    "- E opens AND closes inventory and entity UIs. Do NOT use escape to close UIs.\n"
    "- Escape opens the game menu (pause). Only use escape if you want to pause.\n"
    "- Right-click on entities opens their UI. E closes it.\n"
    "- To mine: right_click_hold on a resource tile for 3+ seconds.\n"
    "- To craft: open inventory (E), click the item in crafting panel, close (E).\n\n"
    "CRAFTING: To craft items, include a 'craft' field instead of trying to navigate the UI:\n"
    '  "craft": ["stone_furnace", "iron_gear_wheel"]\n'
    "Available items: stone_furnace, burner_mining_drill, iron_gear_wheel, "
    "transport_belt, burner_inserter, wooden_chest, pipe\n"
    "The crafting system will handle the UI automatically.\n\n"
    "PLACING: To place a building, include a 'place' field:\n"
    '  "place": {"item": "stone_furnace", "x": 1300, "y": 700}\n\n'
    "Available input types:\n"
    '- {"type": "key", "key": "w/a/s/d/e/q/space/tab/escape/1-9", "duration": seconds}\n'
    '- {"type": "left_click", "x": pixel_x, "y": pixel_y}\n'
    '- {"type": "right_click", "x": pixel_x, "y": pixel_y}\n'
    '- {"type": "right_click_hold", "x": pixel_x, "y": pixel_y, "duration": seconds}\n'
    '- {"type": "mouse_move", "x": pixel_x, "y": pixel_y}\n'
    '- {"type": "wait", "duration": seconds}'
)

# Map key names to action IDs for CNN observation
KEY_TO_ACTION_ID = {
    "w": 0, "a": 1, "s": 2, "d": 3,
    "e": 15, "q": 16, "space": 14,
}

SPECIAL_KEYS = {
    "escape": Key.esc, "tab": Key.tab, "shift": Key.shift,
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


class GameState:
    """Persistent game state that carries between Claude decisions."""

    def __init__(self):
        self.session_start = datetime.now().isoformat()
        self.decisions_made = 0
        self.inventory = {}
        self.milestones_reached = []
        self.current_goal = "mine iron ore and stone"
        self.next_milestone = "craft stone furnace"
        self.history = []  # Last 10 decisions

    def update(self, state_update, reasoning):
        """Apply Claude's state update."""
        self.decisions_made += 1

        # Update inventory
        changes = state_update.get("inventory_changes", {})
        for item, qty in changes.items():
            self.inventory[item] = self.inventory.get(item, 0) + qty

        # Milestone
        milestone = state_update.get("milestone_reached")
        if milestone and milestone != "null":
            self.milestones_reached.append(milestone)
            print(f"  [STATE] MILESTONE: {milestone}")
            _log(f"MILESTONE: {milestone}")

        # Goals
        goal = state_update.get("current_goal")
        if goal:
            self.current_goal = goal
        nxt = state_update.get("next_milestone")
        if nxt:
            self.next_milestone = nxt

        # History (keep last 10)
        self.history.append({
            "decision": self.decisions_made,
            "reasoning": reasoning[:100],
            "goal": self.current_goal,
        })
        if len(self.history) > 10:
            self.history.pop(0)

    def apply_ground_truth(self, real_inventory):
        """Overwrite estimated inventory with verified values.
        Recheck milestones against what's actually there."""
        old_inv = dict(self.inventory)
        self.inventory = real_inventory

        # Log corrections
        for item in set(list(old_inv.keys()) + list(real_inventory.keys())):
            old_val = old_inv.get(item, 0)
            new_val = real_inventory.get(item, 0)
            if old_val != new_val:
                print(f"  [VERIFY] {item}: {old_val} -> {new_val} (corrected)")

        # Recheck milestones — remove any that don't match inventory reality
        # Keep milestones that are action-based (placed, built) since inventory
        # won't show those. Only remove item-based hallucinations.
        verified = []
        for m in self.milestones_reached:
            m_lower = m.lower()
            # Keep action milestones (placed, built, started, opened)
            if any(kw in m_lower for kw in ["placed", "built", "started", "opened", "researched"]):
                verified.append(m)
            # Keep crafting milestones only if the item is/was in inventory
            elif "crafted" in m_lower:
                # Check if any inventory item matches
                if any(k in m_lower for k in real_inventory.keys()):
                    verified.append(m)
                else:
                    print(f"  [VERIFY] Removing unconfirmed milestone: {m}")
                    _log(f"MILESTONE REMOVED (unconfirmed): {m}")
            else:
                verified.append(m)
        self.milestones_reached = verified

    def to_prompt(self):
        """Format state for Claude's prompt."""
        inv_str = ", ".join(f"{k}: {v}" for k, v in self.inventory.items()) or "empty"
        milestones_str = ", ".join(self.milestones_reached[-5:]) or "none yet"
        history_str = "\n".join(
            f"  #{h['decision']}: {h['reasoning']}" for h in self.history[-5:]
        ) or "  (first action)"

        return (
            f"SESSION STATE (decision #{self.decisions_made + 1}):\n"
            f"  Inventory (estimated): {inv_str}\n"
            f"  Milestones: {milestones_str}\n"
            f"  Current goal: {self.current_goal}\n"
            f"  Next milestone: {self.next_milestone}\n"
            f"  Recent history:\n{history_str}"
        )


class ClaudePlayer:
    """Claude as the direct Factorio controller with persistent game state."""

    def __init__(self, ctrl, monitor):
        self.ctrl = ctrl
        self.monitor = monitor
        self._client = None
        self.total_decisions = 0
        self.total_reward = 0.0
        self._call_timestamps = []
        self._total_spend = 0.0
        self._budget_exceeded = False
        self._last_reasoning = ""
        self.state = GameState()
        self.crafter = CraftingSystem(ctrl, monitor)
        self.placer = PlacementSystem(ctrl, monitor)

    def print_cost_estimate(self):
        calls_per_hour = min(3600 / DECISION_INTERVAL, MAX_CALLS_PER_HOUR)
        cost_per_hour = calls_per_hour * EST_COST_PER_CALL
        hours_until_limit = HARD_SPEND_LIMIT / cost_per_hour if cost_per_hour > 0 else float("inf")
        print(f"  API cost estimate:")
        print(f"    Rate: 1 call / {DECISION_INTERVAL:.0f}s = {calls_per_hour:.0f} calls/hr "
              f"(max {MAX_CALLS_PER_HOUR}/hr)")
        print(f"    Cost: ~${EST_COST_PER_CALL:.4f}/call = ~${cost_per_hour:.2f}/hr")
        print(f"    Limit: ${HARD_SPEND_LIMIT:.2f} = ~{hours_until_limit:.1f} hours of play")

    def _check_rate_limit(self):
        if self._budget_exceeded:
            return False
        now = time.time()
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 3600]
        return len(self._call_timestamps) < MAX_CALLS_PER_HOUR

    def _record_call(self):
        self._call_timestamps.append(time.time())
        self._total_spend += EST_COST_PER_CALL
        if self._total_spend >= HARD_SPEND_LIMIT:
            self._budget_exceeded = True
            print(f"\n  *** BUDGET LIMIT: ${self._total_spend:.2f} — Claude disabled ***\n")
            _log(f"BUDGET EXCEEDED: ${self._total_spend:.2f}")

    @property
    def spend(self):
        return self._total_spend

    @property
    def is_available(self):
        return not self._budget_exceeded and self._check_rate_limit()

    def _init_client(self):
        if self._client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def _capture_b64(self):
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
        """Capture screen, ask Claude for exact input sequence."""
        if not self.is_available:
            return None

        self._init_client()
        image_b64 = self._capture_b64()

        try:
            response = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
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
                        {"type": "text", "text": (
                            f"{self.state.to_prompt()}\n\n"
                            f"What inputs should you execute right now?"
                        )},
                    ],
                }],
            )
            text = response.content[0].text.strip()
            _log(f"RAW RESPONSE: {text[:500]}")
            # Strip markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            # Try to find JSON in the response if it's not pure JSON
            if not text.startswith("{"):
                # Look for first { and last }
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    text = text[start:end + 1]
                else:
                    _log(f"NO JSON FOUND in response: {text[:200]}")
                    print(f"  [CLAUDE] No JSON in response, skipping")
                    self._record_call()
                    return None
            result = json.loads(text)
            self._record_call()
            self.total_decisions += 1
            self._last_reasoning = result.get("reasoning", "")

            # Apply state update
            state_update = result.get("state_update", {})
            self.state.update(state_update, self._last_reasoning)

            return result
        except json.JSONDecodeError as e:
            _log(f"JSON PARSE ERROR: {e} | raw: {text[:200] if 'text' in dir() else 'no response'}")
            print(f"  [CLAUDE] Bad JSON response, skipping")
            self._record_call()
            return None
        except Exception as e:
            _log(f"DECIDE ERROR: {type(e).__name__}: {e}")
            return None

    def verify_inventory(self):
        """Open inventory, screenshot, ask Claude to read it, update ground truth."""
        if not self.is_available:
            return

        self._init_client()
        print(f"\n  [VERIFY] Opening inventory for ground truth check...")
        _log("VERIFY: opening inventory")

        # Open inventory
        self.ctrl.press_key("e", duration=0.1)
        time.sleep(1.0)

        # Capture
        image_b64 = self._capture_b64()

        try:
            response = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
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
                        {"type": "text", "text": (
                            "Read the Factorio inventory screen. List every item "
                            "you can see and its quantity. Include items in the "
                            "character inventory, crafting queue, and hotbar. "
                            "Return JSON only, no markdown: "
                            '{"item_name": quantity, "item_name": quantity}'
                        )},
                    ],
                }],
            )
            text = response.content[0].text.strip()
            self._record_call()

            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end > start:
                    text = text[start:end + 1]

            real_inventory = json.loads(text)
            # Ensure all values are integers
            real_inventory = {k: int(v) for k, v in real_inventory.items() if isinstance(v, (int, float))}

            inv_str = ", ".join(f"{k}: {v}" for k, v in real_inventory.items())
            print(f"  [VERIFY] Real inventory: {inv_str}")
            _log(f"VERIFY RESULT: {json.dumps(real_inventory)}")

            # Apply ground truth
            self.state.apply_ground_truth(real_inventory)

        except Exception as e:
            print(f"  [VERIFY] Failed to read inventory: {e}")
            _log(f"VERIFY ERROR: {e}")

        # Close inventory
        self.ctrl.press_key("e", duration=0.1)
        time.sleep(0.5)
        print(f"  [VERIFY] Inventory closed, resuming play\n")

    def execute(self, decision):
        """Execute Claude's raw input sequence. Returns list of (action_id, duration)."""
        reasoning = decision.get("reasoning", "")
        inputs = decision.get("inputs", [])

        print(f"  [CLAUDE] {reasoning[:100]}")
        _log(f"DECISION: {json.dumps(decision)}")

        executed = []

        # Handle craft commands — routed through CraftingSystem
        craft_items = decision.get("craft", [])
        if craft_items:
            for item in craft_items:
                item_key = item.replace(" ", "_").lower()
                if item_key in RECIPES:
                    success = self.crafter.craft(item_key)
                    if success:
                        # Update state with real crafting result
                        self.state.inventory[item_key] = self.state.inventory.get(item_key, 0) + 1
                        # Deduct ingredients
                        for ingredient, qty in RECIPES[item_key].items():
                            self.state.inventory[ingredient] = max(0,
                                self.state.inventory.get(ingredient, 0) - qty)
                    executed.append((15, 2.0))  # E key approximate
                else:
                    print(f"  [CRAFT] Unknown recipe: {item}")

        # Handle place commands — routed through PlacementSystem
        place_cmd = decision.get("place")
        if place_cmd:
            item_key = place_cmd.get("item", "").replace(" ", "_").lower()
            px = int(place_cmd.get("x", 1280))
            py = int(place_cmd.get("y", 720))
            if self.placer.place(item_key, px, py):
                self.state.inventory[item_key] = max(0,
                    self.state.inventory.get(item_key, 0) - 1)
            executed.append((8, 1.0))

        for inp in inputs:
            inp_type = inp.get("type", "")
            duration = min(inp.get("duration", 0.1), 10)

            if inp_type == "key":
                key = inp.get("key", "")
                if key in SPECIAL_KEYS:
                    self.ctrl.press_key(SPECIAL_KEYS[key], duration=duration)
                elif len(key) == 1:
                    self.ctrl.hold_key(key, duration=duration)
                action_id = KEY_TO_ACTION_ID.get(key, 17)
                executed.append((action_id, duration))

            elif inp_type == "left_click":
                x = int(inp.get("x", 1280))
                y = int(inp.get("y", 720))
                self.ctrl.click(x, y, "left")
                executed.append((8, 0.1))

            elif inp_type == "right_click":
                x = int(inp.get("x", 1280))
                y = int(inp.get("y", 720))
                self.ctrl.click(x, y, "right")
                executed.append((9, 0.1))

            elif inp_type == "right_click_hold":
                x = int(inp.get("x", 1280))
                y = int(inp.get("y", 720))
                self.ctrl.move(x, y)
                time.sleep(0.05)
                mouse.press(Button.right)
                time.sleep(duration)
                mouse.release(Button.right)
                print(f"  [CLAUDE] right_click_hold ({x},{y}) for {duration:.1f}s")
                executed.append((9, duration))

            elif inp_type == "mouse_move":
                x = int(inp.get("x", 1280))
                y = int(inp.get("y", 720))
                self.ctrl.move(x, y)
                executed.append((17, 0.05))

            elif inp_type == "wait":
                time.sleep(duration)
                executed.append((17, duration))

            time.sleep(0.05)  # Small gap between inputs

        return executed


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
        c = self.claude_avg()
        n = self.cnn_avg()
        if c <= 0:
            return 1.0 if n >= 0 else 0.0
        return n / c


def main():
    """Standalone test — Claude plays Factorio directly."""
    from capture import find_factorio_window

    print("=== Claude Direct Control Test ===\n")

    bbox = find_factorio_window()
    if not bbox:
        print("Factorio not found!")
        return

    ctrl = FactorioController()
    player = ClaudePlayer(ctrl, bbox)
    player.print_cost_estimate()

    print(f"\nStarting in 3 seconds — switch to Factorio!")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("\nClaude is playing. Press Ctrl+C to stop.\n")

    try:
        for step in range(20):
            # Ground truth check every 5 decisions
            if step > 0 and step % 5 == 0:
                player.verify_inventory()

            print(f"\n--- Decision {step + 1} ---")
            decision = player.decide()
            if decision:
                player.execute(decision)
            else:
                print("  (no decision)")
            time.sleep(5)
    except KeyboardInterrupt:
        pass

    print(f"\nClaude made {player.total_decisions} decisions. Spent: ${player.spend:.2f}")
    print(f"\nFinal game state:")
    print(f"  Inventory: {player.state.inventory}")
    print(f"  Milestones: {player.state.milestones_reached}")
    print(f"  Current goal: {player.state.current_goal}")
    print(f"  Next milestone: {player.state.next_milestone}")
    print("Done.")


if __name__ == "__main__":
    main()
