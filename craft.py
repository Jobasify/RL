"""
Crafting library — reliable UI navigation for Factorio crafting.
Claude decides WHAT to craft. This module handles HOW.

Each recipe: open inventory → find craft button via Claude → click →
verify success via Claude → close inventory.
"""

import base64
import json
import os
import time
from pathlib import Path

import cv2
import mss
import numpy as np

CRAFT_LOG = Path("logs/craft.log")

# Early game recipes: item -> ingredients needed
RECIPES = {
    "iron_gear_wheel":    {"iron_plate": 2},
    "stone_furnace":      {"stone": 5},
    "burner_mining_drill": {"iron_gear_wheel": 3, "iron_plate": 3, "stone_furnace": 1},
    "transport_belt":     {"iron_gear_wheel": 1, "iron_plate": 1},
    "burner_inserter":    {"iron_gear_wheel": 1, "iron_plate": 1},
    "wooden_chest":       {"wood": 2},
    "offshore_pump":      {"iron_gear_wheel": 2, "electronic_circuit": 2, "pipe": 1},
    "boiler":             {"stone_furnace": 1, "pipe": 4},
    "steam_engine":       {"iron_gear_wheel": 8, "iron_plate": 10, "pipe": 5},
    "pipe":               {"iron_plate": 1},
    "electric_mining_drill": {"iron_gear_wheel": 5, "iron_plate": 10, "electronic_circuit": 3},
    "electronic_circuit": {"iron_plate": 1, "copper_cable": 3},
    "copper_cable":       {"copper_plate": 1},
}

# Factorio display names (for Claude to find in crafting panel)
DISPLAY_NAMES = {
    "iron_gear_wheel": "Iron gear wheel",
    "stone_furnace": "Stone furnace",
    "burner_mining_drill": "Burner mining drill",
    "transport_belt": "Transport belt",
    "burner_inserter": "Burner inserter",
    "wooden_chest": "Wooden chest",
    "offshore_pump": "Offshore pump",
    "boiler": "Boiler",
    "steam_engine": "Steam engine",
    "pipe": "Pipe",
    "electric_mining_drill": "Electric mining drill",
    "electronic_circuit": "Electronic circuit",
    "copper_cable": "Copper cable",
}


def _log(message):
    CRAFT_LOG.parent.mkdir(exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CRAFT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def _capture_b64(monitor):
    """Capture screen as base64 JPEG."""
    with mss.mss() as sct:
        img = sct.grab(monitor)
    frame = np.array(img)[:, :, :3]
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


class CraftingSystem:
    """Handles crafting UI interactions reliably."""

    def __init__(self, ctrl, monitor):
        self.ctrl = ctrl
        self.monitor = monitor
        self._client = None

    def _init_client(self):
        if self._client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def _ask_claude(self, image_b64, prompt):
        """Send image + prompt, return text."""
        self._init_client()
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
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text.strip()

    def _parse_coords(self, text):
        """Extract x,y percentages from Claude's response."""
        # Strip markdown
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        # Find JSON
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                text = text[start:end + 1]
            else:
                return None
        try:
            data = json.loads(text)
            x_pct = float(data.get("x", data.get("x_pct", 0.5)))
            y_pct = float(data.get("y", data.get("y_pct", 0.5)))
            # Convert percentages (0-100) to (0-1) if needed
            if x_pct > 1.0:
                x_pct /= 100.0
            if y_pct > 1.0:
                y_pct /= 100.0
            return x_pct, y_pct
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def craft(self, item, count=1):
        """Craft an item using the Factorio crafting UI.

        Returns True if crafting succeeded, False otherwise.
        """
        display_name = DISPLAY_NAMES.get(item, item.replace("_", " ").title())
        print(f"  [CRAFT] Crafting {count}x {display_name}...")
        _log(f"CRAFT START: {count}x {item}")

        # Step 1: Open inventory
        self.ctrl.press_key("e", duration=0.1)
        time.sleep(1.0)

        success = False
        try:
            for attempt in range(count):
                # Step 2: Screenshot and find the craft button
                image_b64 = _capture_b64(self.monitor)
                find_prompt = (
                    f"Look at this Factorio crafting/inventory screen. "
                    f"Find the craft button or icon for '{display_name}'. "
                    f"Return the location as JSON with x and y as percentages "
                    f"of screen dimensions (0.0 to 1.0): "
                    f'{{"x": 0.5, "y": 0.5}}'
                )

                try:
                    response = self._ask_claude(image_b64, find_prompt)
                    coords = self._parse_coords(response)
                except Exception as e:
                    print(f"  [CRAFT] Failed to find {display_name}: {e}")
                    _log(f"CRAFT FIND ERROR: {e}")
                    break

                if coords is None:
                    print(f"  [CRAFT] Could not locate {display_name} in crafting panel")
                    _log(f"CRAFT: item not found")
                    break

                x_pct, y_pct = coords
                click_x = int(x_pct * self.ctrl.width)
                click_y = int(y_pct * self.ctrl.height)
                print(f"  [CRAFT] Found at ({click_x}, {click_y}) — clicking")
                _log(f"CRAFT CLICK: ({click_x}, {click_y})")

                # Step 3: Click to craft
                self.ctrl.click(click_x, click_y, "left")
                time.sleep(0.5)

                # Click again for items that need multiple clicks or confirmation
                self.ctrl.click(click_x, click_y, "left")
                time.sleep(1.0)

            # Step 4: Verify crafting succeeded
            verify_b64 = _capture_b64(self.monitor)
            verify_prompt = (
                f"Look at this Factorio inventory screen. "
                f"Can you see '{display_name}' in the player's inventory "
                f"(not the crafting panel, but the actual inventory grid)? "
                f"Also check if there's a crafting progress bar. "
                f'Return JSON: {{"crafted": true}} or {{"crafted": false}}'
            )

            try:
                verify_text = self._ask_claude(verify_b64, verify_prompt)
                if verify_text.startswith("```"):
                    verify_text = verify_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                if not verify_text.startswith("{"):
                    start = verify_text.find("{")
                    end = verify_text.rfind("}")
                    if start != -1 and end > start:
                        verify_text = verify_text[start:end + 1]
                result = json.loads(verify_text)
                success = result.get("crafted", False)
            except Exception:
                success = False

            if success:
                print(f"  [CRAFT] SUCCESS: {display_name} crafted!")
                _log(f"CRAFT SUCCESS: {item}")
            else:
                print(f"  [CRAFT] UNCONFIRMED: could not verify {display_name}")
                _log(f"CRAFT UNCONFIRMED: {item}")

        finally:
            # Step 5: Close inventory
            self.ctrl.press_key("e", duration=0.1)
            time.sleep(0.5)
            print(f"  [CRAFT] Inventory closed")

        return success

    def can_craft(self, item, inventory):
        """Check if we have ingredients to craft an item."""
        recipe = RECIPES.get(item)
        if not recipe:
            return False
        for ingredient, qty in recipe.items():
            if inventory.get(ingredient, 0) < qty:
                return False
        return True

    def suggest_next(self, inventory, milestones):
        """Suggest what to craft next based on inventory and progress."""
        # Priority order for early game
        priority = [
            "stone_furnace",
            "iron_gear_wheel",
            "burner_mining_drill",
            "transport_belt",
            "burner_inserter",
        ]
        for item in priority:
            if self.can_craft(item, inventory):
                return item
        return None


class PlacementSystem:
    """Handles building placement UI interactions."""

    def __init__(self, ctrl, monitor):
        self.ctrl = ctrl
        self.monitor = monitor
        self._client = None

    def _init_client(self):
        if self._client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def place(self, item, x, y):
        """Place an item from inventory at screen coordinates (x, y).

        In Factorio: select item from hotbar/inventory, then left-click to place.
        """
        display_name = DISPLAY_NAMES.get(item, item.replace("_", " ").title())
        print(f"  [PLACE] Placing {display_name} at ({x}, {y})")
        _log(f"PLACE: {item} at ({x}, {y})")

        # Open inventory, find the item, pick it up
        self.ctrl.press_key("e", duration=0.1)
        time.sleep(0.8)

        # Ask Claude where the item is in inventory
        image_b64 = _capture_b64(self.monitor)
        try:
            self._init_client()
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
                        {"type": "text", "text": (
                            f"Find '{display_name}' in the inventory grid. "
                            f"Return its location as percentage of screen: "
                            f'{{"x": 0.5, "y": 0.5}}'
                        )},
                    ],
                }],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end > start:
                    text = text[start:end + 1]
            data = json.loads(text)
            ix = float(data.get("x", 0.5))
            iy = float(data.get("y", 0.5))
            if ix > 1.0:
                ix /= 100.0
            if iy > 1.0:
                iy /= 100.0
            inv_x = int(ix * self.ctrl.width)
            inv_y = int(iy * self.ctrl.height)

            # Click item in inventory to pick it up
            self.ctrl.click(inv_x, inv_y, "left")
            time.sleep(0.3)
        except Exception as e:
            print(f"  [PLACE] Failed to find item in inventory: {e}")
            self.ctrl.press_key("e", duration=0.1)
            return False

        # Close inventory
        self.ctrl.press_key("e", duration=0.1)
        time.sleep(0.5)

        # Place at target location
        self.ctrl.click(x, y, "left")
        time.sleep(0.3)

        print(f"  [PLACE] {display_name} placed at ({x}, {y})")
        _log(f"PLACE SUCCESS: {item} at ({x}, {y})")
        return True


def main():
    """Test crafting system."""
    from capture import find_factorio_window
    from control import FactorioController

    print("=== Crafting System Test ===\n")

    bbox = find_factorio_window()
    if not bbox:
        print("Factorio not found!")
        return

    ctrl = FactorioController()
    crafter = CraftingSystem(ctrl, bbox)

    # Test inventory check
    test_inv = {"stone": 10, "iron_plate": 5, "iron_gear_wheel": 3}
    print(f"Test inventory: {test_inv}")
    for item in RECIPES:
        can = crafter.can_craft(item, test_inv)
        if can:
            print(f"  Can craft: {DISPLAY_NAMES.get(item, item)}")

    suggestion = crafter.suggest_next(test_inv, [])
    print(f"\nSuggested next craft: {suggestion}")

    # Live test
    print(f"\nStarting live craft test in 3 seconds — switch to Factorio!")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    success = crafter.craft("stone_furnace")
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")


if __name__ == "__main__":
    main()
