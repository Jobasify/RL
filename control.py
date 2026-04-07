"""
Factorio mouse + keyboard control layer.
All coordinates are relative to the Factorio window.
Uses pynput for input simulation.
"""

import time
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController

from capture import find_factorio_window


mouse = MouseController()
keyboard = KeyboardController()


class FactorioController:
    """Send mouse and keyboard input to the Factorio window."""

    def __init__(self):
        bbox = find_factorio_window()
        if bbox is None:
            raise RuntimeError("Factorio window not found! Make sure the game is running.")
        self.left = bbox["left"]
        self.top = bbox["top"]
        self.width = bbox["width"]
        self.height = bbox["height"]
        print(f"Factorio window: {self.width}x{self.height} at ({self.left}, {self.top})")

    def _to_screen(self, x, y):
        """Convert window-relative coordinates to absolute screen coordinates."""
        return self.left + x, self.top + y

    def move(self, x, y):
        """Move mouse to (x, y) relative to the Factorio window."""
        sx, sy = self._to_screen(x, y)
        mouse.position = (sx, sy)
        print(f"  mouse move -> ({x}, {y})  [screen: ({sx}, {sy})]")

    def click(self, x, y, button="left"):
        """Click at (x, y) relative to the Factorio window."""
        self.move(x, y)
        time.sleep(0.05)
        btn = Button.left if button == "left" else Button.right
        mouse.click(btn)
        print(f"  {button} click at ({x}, {y})")

    def double_click(self, x, y):
        """Double-click at (x, y) relative to the Factorio window."""
        self.move(x, y)
        time.sleep(0.05)
        mouse.click(Button.left, 2)
        print(f"  double click at ({x}, {y})")

    def drag(self, x1, y1, x2, y2, button="left", duration=0.3):
        """Click and drag from (x1, y1) to (x2, y2)."""
        btn = Button.left if button == "left" else Button.right
        sx1, sy1 = self._to_screen(x1, y1)
        sx2, sy2 = self._to_screen(x2, y2)

        mouse.position = (sx1, sy1)
        time.sleep(0.05)
        mouse.press(btn)
        print(f"  drag start ({x1}, {y1})")

        # Smooth drag with interpolation
        steps = max(int(duration / 0.016), 5)
        for i in range(1, steps + 1):
            t = i / steps
            cx = int(sx1 + (sx2 - sx1) * t)
            cy = int(sy1 + (sy2 - sy1) * t)
            mouse.position = (cx, cy)
            time.sleep(duration / steps)

        mouse.release(btn)
        print(f"  drag end   ({x2}, {y2})")

    def press_key(self, key, duration=0.1):
        """Press and release a key. Accepts characters or pynput Key objects."""
        if isinstance(key, str) and len(key) == 1:
            keyboard.press(key)
            time.sleep(duration)
            keyboard.release(key)
            print(f"  key press: '{key}' ({duration:.2f}s)")
        else:
            keyboard.press(key)
            time.sleep(duration)
            keyboard.release(key)
            print(f"  key press: {key} ({duration:.2f}s)")

    def key_combo(self, *keys):
        """Press a key combination (e.g. ctrl+c). All keys held then released."""
        for k in keys:
            keyboard.press(k)
            time.sleep(0.02)
        time.sleep(0.05)
        for k in reversed(keys):
            keyboard.release(k)
            time.sleep(0.02)
        names = [str(k) for k in keys]
        print(f"  key combo: {' + '.join(names)}")

    def hold_key(self, key, duration=0.5):
        """Hold a key for a duration (useful for movement keys)."""
        keyboard.press(key)
        print(f"  key hold: '{key}' for {duration:.2f}s")
        time.sleep(duration)
        keyboard.release(key)
        print(f"  key release: '{key}'")


# ---------------------------------------------------------------------------
# Test sequence
# ---------------------------------------------------------------------------

def run_test(ctrl):
    """Run a test sequence to verify mouse and keyboard control."""
    cx, cy = ctrl.width // 2, ctrl.height // 2
    print(f"\nWindow center: ({cx}, {cy})")

    # --- Mouse movement ---
    print("\n[1] Mouse movement - corners and center")
    margin = 50
    points = [
        (margin, margin, "top-left"),
        (ctrl.width - margin, margin, "top-right"),
        (ctrl.width - margin, ctrl.height - margin, "bottom-right"),
        (margin, ctrl.height - margin, "bottom-left"),
        (cx, cy, "center"),
    ]
    for x, y, label in points:
        print(f"  Moving to {label}...")
        ctrl.move(x, y)
        time.sleep(0.4)

    # --- Clicks ---
    print("\n[2] Click tests")
    ctrl.click(cx, cy, "left")
    time.sleep(0.3)
    ctrl.click(cx + 100, cy, "right")
    time.sleep(0.3)

    # --- Drag ---
    print("\n[3] Drag test")
    ctrl.drag(cx - 100, cy, cx + 100, cy, duration=0.5)
    time.sleep(0.3)

    # --- WASD movement ---
    print("\n[4] WASD movement keys (0.3s each)")
    for key in ["w", "a", "s", "d"]:
        ctrl.hold_key(key, duration=0.3)
        time.sleep(0.2)

    # --- Key combo ---
    print("\n[5] Key combo test (Escape to open/close menu)")
    ctrl.press_key(Key.esc, duration=0.1)
    time.sleep(0.5)
    ctrl.press_key(Key.esc, duration=0.1)

    # --- Modifier combo ---
    print("\n[6] Modifier combo test (Shift+W = run)")
    keyboard.press(Key.shift)
    time.sleep(0.05)
    ctrl.hold_key("w", duration=0.5)
    keyboard.release(Key.shift)
    print("  released shift")

    print("\nTest sequence complete.")


def main():
    print("=== Factorio Controller Test ===\n")
    ctrl = FactorioController()

    print("\nStarting test in 3 seconds — switch to Factorio!")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    run_test(ctrl)


if __name__ == "__main__":
    main()
