"""
Vision-language advisor for the Factorio RL agent.
Periodically captures the game screen, sends it to Claude for analysis,
and converts the advice into a strategy embedding that conditions the CNN.

Runs in a background thread alongside training.
"""

import base64
import io
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ADVISOR_INTERVAL = 60  # Seconds between advice updates
ADVISOR_LOG = Path("logs/advisor.log")
ADVISOR_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (
    "You are advising an AI agent learning to play Factorio from scratch. "
    "Look at this screenshot and give 3 specific, simple actionable suggestions "
    "for what the agent should prioritise in the next 60 seconds. Be concrete - "
    "say what to do, not general strategy. Format as a numbered list, nothing else."
)


class Advisor:
    """Background thread that asks Claude for contextual game advice."""

    def __init__(self, knowledge_base, sct, monitor, device):
        """
        Args:
            knowledge_base: KnowledgeBase instance for encoding advice into embeddings
            sct: mss screen capture instance
            monitor: monitor/window bounding box for capture
            device: torch device
        """
        self.kb = knowledge_base
        self.sct = sct
        self.monitor = monitor
        self.device = device

        self._strategy_vec = None  # Latest advisor strategy vector
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._client = None
        self._last_advice = ""

        ADVISOR_LOG.parent.mkdir(exist_ok=True)

    def _init_client(self):
        """Lazy-init Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except Exception as e:
                self._log(f"Failed to init Anthropic client: {e}")
                return False
        return True

    def _capture_screenshot_b64(self):
        """Capture current screen and return as base64 JPEG."""
        import mss
        with mss.mss() as sct:
            img = sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]  # BGR
        # Resize to reasonable size for API (720p max)
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        # Encode as JPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def _ask_claude(self, image_b64):
        """Send screenshot to Claude and get advice."""
        response = self._client.messages.create(
            model=ADVISOR_MODEL,
            max_tokens=300,
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
                    {
                        "type": "text",
                        "text": "What should the agent do right now?",
                    },
                ],
            }],
        )
        return response.content[0].text.strip()

    def _advice_to_embedding(self, advice_text):
        """Convert advice text to a strategy embedding via sentence transformer."""
        model = self.kb._load_model()
        embedding = model.encode(advice_text, convert_to_numpy=True,
                                  normalize_embeddings=True)
        return embedding.astype(np.float32)

    def _log(self, message):
        """Append to advisor log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        with open(ADVISOR_LOG, "a", encoding="utf-8") as f:
            f.write(line)

    def _advisor_loop(self):
        """Background loop: capture -> ask Claude -> update embedding."""
        # Wait for initial interval before first advice
        time.sleep(ADVISOR_INTERVAL)

        while self._running:
            try:
                if not self._init_client():
                    time.sleep(ADVISOR_INTERVAL)
                    continue

                # Capture
                image_b64 = self._capture_screenshot_b64()

                # Ask Claude
                advice = self._ask_claude(image_b64)
                self._last_advice = advice

                # Convert to embedding
                embedding = self._advice_to_embedding(advice)

                with self._lock:
                    self._strategy_vec = embedding

                # Log
                self._log(f"ADVICE:\n{advice}")
                print(f"\n  [ADVISOR] New advice received:")
                for line in advice.split("\n"):
                    if line.strip():
                        print(f"    {line.strip()}")
                print()

            except Exception as e:
                self._log(f"ERROR: {e}")
                # Fall back silently — strategy_vec stays as previous

            time.sleep(ADVISOR_INTERVAL)

    def start(self):
        """Start the advisor background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._advisor_loop, daemon=True)
        self._thread.start()
        self._log("Advisor started")
        print(f"  Advisor running (every {ADVISOR_INTERVAL}s, model: {ADVISOR_MODEL})")

    def stop(self):
        """Stop the advisor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._log("Advisor stopped")

    def get_strategy(self):
        """Get the latest advisor strategy vector, or None if not yet available."""
        with self._lock:
            return self._strategy_vec


def main():
    """Test the advisor standalone."""
    import mss
    from capture import find_factorio_window
    from knowledge import KnowledgeBase
    import torch

    print("=== Advisor Test ===\n")

    bbox = find_factorio_window()
    if bbox is None:
        print("Factorio not found!")
        return

    print(f"Factorio: {bbox['width']}x{bbox['height']}")

    kb = KnowledgeBase()
    kb.build()

    device = torch.device("cpu")
    sct = mss.mss()

    advisor = Advisor(kb, sct, bbox, device)

    # Single advice cycle (don't need to start background thread)
    if not advisor._init_client():
        print("No API key. Set ANTHROPIC_API_KEY env var.")
        return

    print("\nCapturing screenshot and asking Claude...")
    image_b64 = advisor._capture_screenshot_b64()
    advice = advisor._ask_claude(image_b64)

    print(f"\nAdvice:\n{advice}")

    embedding = advisor._advice_to_embedding(advice)
    print(f"\nStrategy vector: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")

    sct.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
