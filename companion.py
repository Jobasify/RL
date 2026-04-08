"""
Conversational screen-aware companion.
Talk to it, it sees your screen, talks back, and can control the mouse/keyboard.

Usage:
    python companion.py              # Anthropic API (default)
    python companion.py --local      # Ollama local (llava)
    python companion.py --local --model moondream  # Ollama with moondream

Requirements:
    pip install openai-whisper pyttsx3 anthropic sounddevice numpy mss Pillow requests
"""

import argparse
import io
import os
import sys
import base64
import json
import threading
import queue
import time

import numpy as np
import sounddevice as sd
import whisper
import pyttsx3
import mss
from PIL import Image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WHISPER_MODEL = "base.en"           # base.en for accuracy, tiny.en for speed
SAMPLE_RATE = 16000                # Whisper expects 16 kHz mono
SILENCE_THRESHOLD = 0.003          # RMS below this = silence (Yeti at low gain)
SILENCE_DURATION = 0.8             # seconds of silence to end utterance
MAX_RECORD_SECONDS = 30            # safety cap per utterance
MIC_GAIN = 15.0                    # software amplification for quiet mics
SCREENSHOT_MAX_WIDTH = 2560        # send full native resolution for UI text accuracy
CLAUDE_MODEL = "claude-sonnet-4-20250514"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llava"

SYSTEM_PROMPT = (
    "You are a helpful companion who can see the user's screen in real time. "
    "Everything you write is spoken aloud via text-to-speech automatically. "
    "Give full, detailed responses. Describe what you see thoroughly when asked. "
    "No markdown or formatting — just plain conversational speech.\n\n"
    "If you need to perform actions on screen, include a JSON block fenced with "
    "```actions\n[...]\n``` containing a list of action objects. Each action is one of:\n"
    '  {"type":"click", "x":<int>, "y":<int>, "button":"left"|"right"}\n'
    '  {"type":"hold_click", "x":<int>, "y":<int>, "button":"right", "duration":5.0} — hold mouse button down (for mining in Factorio, use right button for 3-5 seconds)\n'
    '  {"type":"double_click", "x":<int>, "y":<int>}\n'
    '  {"type":"move", "x":<int>, "y":<int>}\n'
    '  {"type":"key", "key":"<char or special key name>", "duration":0.1}\n'
    '  {"type":"key_combo", "keys":["ctrl","c"]}\n'
    '  {"type":"hold_key", "key":"w", "duration":0.5}\n'
    '  {"type":"drag", "x1":<int>, "y1":<int>, "x2":<int>, "y2":<int>}\n'
    '  {"type":"type_text", "text":"hello"}\n'
    '  {"type":"wait", "seconds":0.5}\n'
    "Coordinates are absolute screen pixels matching the screenshot dimensions.\n"
    "When performing timed actions like mining for a duration, always use the hold_click "
    "JSON format with explicit x, y, button, and duration fields. Do not use bbox_2d format.\n"
    "When the user specifies a time duration like one minute or 30 seconds, always use that "
    "exact duration in seconds in the action. One minute = 60, 30 seconds = 30. "
    "Never substitute a shorter duration.\n"
    "IMPORTANT: In Factorio, the player character is ALWAYS at the center of the screen "
    "(approximately 640, 360 on a 1280x720 display). To mine resources the character is "
    "standing on, right click hold at screen center coordinates (640, 360), NOT on UI panels.\n"
    "After executing any action, always take a new screenshot to observe the actual result. "
    "Never predict or estimate outcomes — only report what you can actually see in the "
    "screenshot after the action completes. If you cannot see the result clearly, say so.\n"
    "Only include the actions block when the user asks you to DO something. "
    "For pure conversation, just respond normally."
)


# ---------------------------------------------------------------------------
# Screenshot helper
# ---------------------------------------------------------------------------

def grab_screenshot() -> tuple[str, int, int, float]:
    """Capture the primary monitor, return (base64_jpeg, width, height, scale).
    scale = actual_screen_width / screenshot_width — multiply coordinates by this."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        actual_w = monitor["width"]

    frame = Image.frombytes("RGB", (img.width, img.height), img.rgb)

    scale = 1.0
    if frame.width > SCREENSHOT_MAX_WIDTH:
        scale = frame.width / SCREENSHOT_MAX_WIDTH
        frame = frame.resize(
            (SCREENSHOT_MAX_WIDTH, int(frame.height / scale)),
            Image.LANCZOS,
        )

    buf = io.BytesIO()
    frame.save(buf, format="JPEG", quality=75)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return b64, frame.width, frame.height, scale


# ---------------------------------------------------------------------------
# Speech-to-text: Whisper
# ---------------------------------------------------------------------------

class Listener:
    """Continuously listens on the default mic using a persistent stream.
    Detects speech via energy ratio above ambient noise floor."""

    def __init__(self):
        print("Loading Whisper model...")
        self.model = whisper.load_model(WHISPER_MODEL)
        print(f"Whisper '{WHISPER_MODEL}' ready.")

        # Calibrate noise floor
        print("Calibrating mic (stay quiet for 1 second)...")
        cal = sd.rec(SAMPLE_RATE, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        self.noise_floor = np.sqrt(np.mean(cal ** 2)) * MIC_GAIN
        self.speech_threshold = max(self.noise_floor * 3.0, 0.005)
        print(f"  Noise floor: {self.noise_floor:.6f}, speech threshold: {self.speech_threshold:.6f}")

    def listen(self) -> str:
        """Block until the user speaks and stops. Returns transcription."""
        audio_chunks = []
        silent_chunks = 0
        speaking = False
        chunk_duration = 0.1  # 100ms
        chunk_samples = int(SAMPLE_RATE * chunk_duration)
        max_chunks = int(MAX_RECORD_SECONDS / chunk_duration)
        silence_chunks_needed = int(SILENCE_DURATION / chunk_duration)

        # Use a continuous input stream for gapless recording
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=chunk_samples)
        stream.start()

        try:
            while True:
                chunk, overflowed = stream.read(chunk_samples)
                chunk = chunk * MIC_GAIN
                rms = np.sqrt(np.mean(chunk ** 2))

                if rms > self.speech_threshold:
                    if not speaking:
                        print("  [listening...]")
                    speaking = True
                    silent_chunks = 0
                    audio_chunks.append(chunk.copy())
                elif speaking:
                    silent_chunks += 1
                    audio_chunks.append(chunk.copy())
                    if silent_chunks >= silence_chunks_needed:
                        break

                if len(audio_chunks) >= max_chunks:
                    break
        finally:
            stream.stop()
            stream.close()

        if not audio_chunks:
            return ""

        audio = np.concatenate(audio_chunks, axis=0).flatten()
        audio = np.clip(audio, -1.0, 1.0)
        # Trim trailing silence
        trim_samples = int(silence_chunks_needed * chunk_samples * 0.6)
        if len(audio) > trim_samples:
            audio = audio[:-trim_samples]

        result = self.model.transcribe(
            audio, language="en", fp16=False,
            no_speech_threshold=0.6,
            logprob_threshold=-0.5,       # reject low-confidence transcriptions
            compression_ratio_threshold=1.8,  # reject repetitive hallucinations
        )
        text = result["text"].strip()

        # Filter out common Whisper hallucinations
        if not text:
            return ""
        avg_logprob = result.get("segments", [{}])[0].get("avg_logprob", 0) if result.get("segments") else 0
        if avg_logprob < -1.0:
            return ""  # very low confidence, likely hallucination

        return text


# ---------------------------------------------------------------------------
# Text-to-speech: pyttsx3
# ---------------------------------------------------------------------------

class Speaker:
    """Speaks text aloud using a persistent PowerShell process with SAPI.
    Supports cancel() to interrupt speech immediately."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._start_process()

    def _start_process(self):
        import subprocess
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command",
             "Add-Type -AssemblyName System.Speech; "
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             "$s.Rate = 3; "
             "while ($true) { $line = [Console]::In.ReadLine(); "
             "if ($line -eq $null) { break }; "
             "$s.Speak($line) }"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def say(self, text: str):
        t = threading.Thread(target=self._speak, args=(text,), daemon=True)
        t.start()

    @staticmethod
    def _sanitize(text: str) -> str:
        """Strip emoji and non-ASCII characters that crash Windows TTS."""
        return text.encode("ascii", "ignore").decode("ascii")

    def _speak(self, text: str):
        with self._lock:
            try:
                for line in self._sanitize(text).replace(". ", ".\n").split("\n"):
                    line = line.strip()
                    if line:
                        self._proc.stdin.write(line + "\n")
                        self._proc.stdin.flush()
            except Exception as e:
                print(f"  [TTS error: {e}]")

    def cancel(self):
        """Kill current speech immediately and restart the TTS process."""
        with self._lock:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._start_process()

    def stop(self):
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

def _convert_bbox_action(obj: dict) -> dict | None:
    """Convert a bbox_2d + label object into a standard action dict."""
    import re
    bbox = obj.get("bbox_2d")
    label = obj.get("label", "").lower()
    if not bbox or len(bbox) != 4:
        return None

    x = int((bbox[0] + bbox[2]) / 2)
    y = int((bbox[1] + bbox[3]) / 2)

    # Parse duration from label (e.g. "for 60 seconds")
    dur_match = re.search(r"(\d+(?:\.\d+)?)\s*seconds?", label)
    duration = float(dur_match.group(1)) if dur_match else 3.0

    # Determine button from label
    if "right" in label:
        button = "right"
    else:
        button = "left"

    if "hold" in label or "mining" in label or duration > 1.0:
        return {"type": "hold_click", "x": x, "y": y,
                "button": button, "duration": duration}
    else:
        return {"type": "click", "x": x, "y": y, "button": button}


def parse_actions(response_text: str) -> list[dict]:
    """Extract action list from any fenced JSON code block in the response.
    Handles ```actions, ```json, or bare ``` blocks containing a JSON array.
    Also handles bbox_2d format from vision-language models."""
    import re
    # Match ```actions, ```json, or bare ``` blocks with a JSON array or object
    for pattern in [
        r"```(?:actions|json)?\s*\n(\[.*?\])\s*\n```",
        r"```(?:actions|json)?\s*\n(\{.*?\})\s*\n```",
    ]:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            raw = match.group(1)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    # Unwrap nested {"actions": [...]} format
                    if "actions" in parsed and isinstance(parsed["actions"], list):
                        parsed = parsed["actions"]
                    else:
                        parsed = [parsed]
                # Convert any bbox_2d items to standard actions
                results = []
                for item in parsed:
                    if "bbox_2d" in item:
                        converted = _convert_bbox_action(item)
                        if converted:
                            results.append(converted)
                    else:
                        results.append(item)
                return results
            except json.JSONDecodeError:
                continue

    # Fallback: detect inline bbox_2d JSON (no code fence)
    bbox_pattern = r"\{[^}]*'bbox_2d'[^}]*\}|\{[^}]*\"bbox_2d\"[^}]*\}"
    matches = re.findall(bbox_pattern, response_text)
    if matches:
        results = []
        for m in matches:
            # Normalize single quotes to double quotes for JSON parsing
            normalized = m.replace("'", '"')
            try:
                obj = json.loads(normalized)
                converted = _convert_bbox_action(obj)
                if converted:
                    results.append(converted)
            except json.JSONDecodeError:
                continue
        if results:
            return results

    return []


def strip_actions_block(text: str) -> str:
    """Remove any fenced code block and inline bbox_2d JSON from spoken text."""
    import re
    text = re.sub(r"```(?:actions|json)?\s*\n.*?\n```", "", text, flags=re.DOTALL)
    text = re.sub(r"\{[^}]*['\"]bbox_2d['\"][^}]*\}", "", text)
    return text.strip()


def execute_actions(actions: list[dict], scale: float = 1.0, interrupted=None):
    """Execute a list of action dicts using pynput directly.
    scale: multiply coordinates by this to map screenshot coords to real screen.
    interrupted: callable that returns True if we should abort."""
    from pynput.mouse import Button, Controller as MouseController
    from pynput.keyboard import Key, Controller as KeyboardController

    mouse = MouseController()
    kb = KeyboardController()

    def sx(v):
        return int(v * scale)
    def sy(v):
        return int(v * scale)

    KEY_MAP = {
        "enter": Key.enter, "return": Key.enter,
        "tab": Key.tab, "escape": Key.esc, "esc": Key.esc,
        "space": Key.space, "backspace": Key.backspace,
        "delete": Key.delete, "up": Key.up, "down": Key.down,
        "left": Key.left, "right": Key.right,
        "shift": Key.shift, "ctrl": Key.ctrl, "alt": Key.alt,
        "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4,
        "f5": Key.f5, "f6": Key.f6, "f7": Key.f7, "f8": Key.f8,
        "f9": Key.f9, "f10": Key.f10, "f11": Key.f11, "f12": Key.f12,
    }

    def resolve_key(name: str):
        lower = name.lower()
        if lower in KEY_MAP:
            return KEY_MAP[lower]
        if len(name) == 1:
            return name
        return name

    for action in actions:
        if interrupted and interrupted():
            print("  [interrupted]")
            break
        t = action.get("type", "")
        try:
            if t == "click":
                bname = action.get("button", "left")
                ax, ay = sx(action["x"]), sy(action["y"])
                mouse.position = (ax, ay)
                time.sleep(0.05)
                btn = Button.right if bname == "right" else Button.left
                mouse.click(btn)
                print(f"  [ACTION EXECUTED] {bname}_click at ({ax}, {ay})")
            elif t == "hold_click":
                bname = action.get("button", "left")
                ax, ay = sx(action["x"]), sy(action["y"])
                dur = action.get("duration", 3.0)
                mouse.position = (ax, ay)
                time.sleep(0.05)
                btn = Button.right if bname == "right" else Button.left
                mouse.press(btn)
                print(f"  [ACTION EXECUTED] {bname}_click_hold at ({ax}, {ay}) for {dur}s")
                time.sleep(dur)
                mouse.release(btn)
            elif t == "double_click":
                ax, ay = sx(action["x"]), sy(action["y"])
                mouse.position = (ax, ay)
                time.sleep(0.05)
                mouse.click(Button.left, 2)
                print(f"  [ACTION EXECUTED] double_click at ({ax}, {ay})")
            elif t == "move":
                ax, ay = sx(action["x"]), sy(action["y"])
                mouse.position = (ax, ay)
                print(f"  [ACTION EXECUTED] move to ({ax}, {ay})")
            elif t == "key":
                k = resolve_key(action["key"])
                dur = action.get("duration", 0.1)
                kb.press(k)
                time.sleep(dur)
                kb.release(k)
                print(f"  [ACTION EXECUTED] key '{action['key']}' for {dur}s")
            elif t == "key_combo":
                keys = [resolve_key(k) for k in action["keys"]]
                for k in keys:
                    kb.press(k)
                    time.sleep(0.02)
                time.sleep(0.05)
                for k in reversed(keys):
                    kb.release(k)
                print(f"  [ACTION EXECUTED] key_combo {action['keys']}")
            elif t == "hold_key":
                k = resolve_key(action["key"])
                dur = action.get("duration", 0.5)
                kb.press(k)
                time.sleep(dur)
                kb.release(k)
                print(f"  [ACTION EXECUTED] hold_key '{action['key']}' for {dur}s")
            elif t == "drag":
                x1, y1 = sx(action["x1"]), sy(action["y1"])
                x2, y2 = sx(action["x2"]), sy(action["y2"])
                mouse.position = (x1, y1)
                time.sleep(0.05)
                mouse.press(Button.left)
                steps = 20
                for i in range(1, steps + 1):
                    frac = i / steps
                    x = int(x1 + (x2 - x1) * frac)
                    y = int(y1 + (y2 - y1) * frac)
                    mouse.position = (x, y)
                    time.sleep(0.015)
                mouse.release(Button.left)
                print(f"  [ACTION EXECUTED] drag ({x1}, {y1}) -> ({x2}, {y2})")
            elif t == "type_text":
                for ch in action["text"]:
                    kb.press(ch)
                    kb.release(ch)
                    time.sleep(0.03)
                print(f"  [ACTION EXECUTED] type_text '{action['text']}'")
            elif t == "wait":
                dur = action.get("seconds", 0.5)
                time.sleep(dur)
                print(f"  [ACTION EXECUTED] wait {dur}s")
            else:
                print(f"  [ACTION UNKNOWN] type '{t}': {action}")
        except Exception as e:
            print(f"  [ACTION ERROR] {t}: {e}")

        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Brain: Claude with vision
# ---------------------------------------------------------------------------

class Brain:
    """Sends screenshot + user text to Claude API, returns response."""

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=API_KEY)
        self.history: list[dict] = []
        self.max_history = 20
        self.interrupted = False  # set True to abort current stream

    def think(self, user_text: str, screenshot_b64: str,
              speaker: "Speaker | None" = None) -> str:
        """Send screenshot + text to Claude. If speaker is provided, streams
        sentences to TTS as they arrive for minimal latency."""
        self.history.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": screenshot_b64,
                    },
                },
                {"type": "text", "text": user_text},
            ],
        })

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Stream response — speak sentences as they arrive, skip code blocks
        full_reply = ""
        sentence_buf = ""
        in_code_block = False

        self.interrupted = False
        with self.client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=self.history,
        ) as stream:
            for text in stream.text_stream:
                if self.interrupted:
                    break
                full_reply += text
                if not speaker:
                    continue

                sentence_buf += text

                # Check for code fence toggles
                while "```" in sentence_buf:
                    idx = sentence_buf.index("```")
                    if not in_code_block:
                        # Speak everything before the fence
                        before = sentence_buf[:idx].strip()
                        if before:
                            speaker._speak(before)
                        in_code_block = True
                        sentence_buf = sentence_buf[idx + 3:]
                    else:
                        # Discard everything inside the fence
                        in_code_block = False
                        sentence_buf = sentence_buf[idx + 3:]

                # If inside code block, don't speak anything
                if in_code_block:
                    continue

                # Flush complete sentences to TTS
                while any(sep in sentence_buf for sep in ".!?\n"):
                    for i, ch in enumerate(sentence_buf):
                        if ch in ".!?\n":
                            sentence = sentence_buf[:i + 1].strip()
                            sentence_buf = sentence_buf[i + 1:]
                            if sentence:
                                speaker._speak(sentence)
                            break

        # Flush remaining buffer (only if not in a code block)
        if speaker and sentence_buf.strip() and not in_code_block:
            speaker._speak(sentence_buf.strip())

        self.history.append({"role": "assistant", "content": full_reply})
        return full_reply


class OllamaBrain:
    """Sends screenshot + user text to a local Ollama vision model."""

    def __init__(self, model: str = OLLAMA_DEFAULT_MODEL):
        import requests
        self._requests = requests
        self.model = model
        self.history: list[dict] = []
        self.max_history = 20
        self.interrupted = False

        # Verify Ollama is running and model is available
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            r.raise_for_status()
            available = [m["name"] for m in r.json().get("models", [])]
            # Match full name (llava:13b) or base name (llava matches llava:latest)
            models = available
            match = any(model == m or model == m.split(":")[0] for m in available)
            if not match:
                print(f"  Model '{model}' not found. Available: {models}")
                print(f"  Run: ollama pull {model}")
                sys.exit(1)
            print(f"  Ollama connected — model: {model}")
        except requests.ConnectionError:
            print(f"  ERROR: Cannot connect to Ollama at {OLLAMA_URL}")
            print("  Make sure Ollama is running: ollama serve")
            sys.exit(1)

    def think(self, user_text: str, screenshot_b64: str,
              speaker: "Speaker | None" = None) -> str:
        self.history.append({
            "role": "user",
            "content": user_text,
            "images": [screenshot_b64],
        })

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        self.interrupted = False
        full_reply = ""
        sentence_buf = ""
        in_code_block = False

        # Stream from Ollama
        resp = self._requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}]
                            + self.history,
                "stream": True,
            },
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if self.interrupted:
                break
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = chunk.get("message", {}).get("content", "")
            if not text:
                continue

            full_reply += text

            if not speaker:
                continue

            sentence_buf += text

            # Code fence tracking
            while "```" in sentence_buf:
                idx = sentence_buf.index("```")
                if not in_code_block:
                    before = sentence_buf[:idx].strip()
                    if before:
                        speaker._speak(before)
                    in_code_block = True
                    sentence_buf = sentence_buf[idx + 3:]
                else:
                    in_code_block = False
                    sentence_buf = sentence_buf[idx + 3:]

            if in_code_block:
                continue

            while any(sep in sentence_buf for sep in ".!?\n"):
                for i, ch in enumerate(sentence_buf):
                    if ch in ".!?\n":
                        sentence = sentence_buf[:i + 1].strip()
                        sentence_buf = sentence_buf[i + 1:]
                        if sentence:
                            speaker._speak(sentence)
                        break

        if speaker and sentence_buf.strip() and not in_code_block:
            speaker._speak(sentence_buf.strip())

        self.history.append({"role": "assistant", "content": full_reply})
        return full_reply


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Conversational screen-aware companion")
    parser.add_argument("--local", action="store_true",
                        help="Use local Ollama instead of Anthropic API")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model name (default: llava). e.g. --model moondream")
    args = parser.parse_args()

    print("=" * 50)
    print("  Companion — talk to me, I can see your screen")
    print("=" * 50)

    if args.local:
        model = args.model or OLLAMA_DEFAULT_MODEL
        print(f"  RUNNING LOCAL — no API costs (model: {model})")
        brain = OllamaBrain(model=model)
    else:
        print("  RUNNING API — using Anthropic")
        brain = Brain()

    print()

    listener = Listener()
    speaker = Speaker()

    # Background listener — always recording, puts transcriptions in a queue
    speech_queue: queue.Queue[str] = queue.Queue()
    HALLUCINATIONS = {
        "", "you", "you.", "thank you.", "thanks for watching!",
        "thanks for watching.", "thank you for watching.",
        "...", "the end", "the end.", "bye.", "bye",
        "so", "so.", "hmm", "hmm.", "oh.", "oh",
        "thanks.", "thanks", "i'm sorry.", "subtitles by",
        "subscribe", "like and subscribe",
    }

    def listen_loop():
        while True:
            try:
                text = listener.listen()
                if not text:
                    continue
                if text.lower().strip() in HALLUCINATIONS:
                    continue
                speech_queue.put(text)
            except Exception as e:
                print(f"  [Listener error: {e}]")

    listen_thread = threading.Thread(target=listen_loop, daemon=True)
    listen_thread.start()

    print("\nReady. Start speaking...\n")
    speaker.say("I'm ready. What's up?")

    def check_interrupt() -> str | None:
        """Check if user said something new. Returns text or None."""
        try:
            return speech_queue.get_nowait()
        except queue.Empty:
            return None

    try:
        while True:
            # Wait for speech
            text = speech_queue.get()
            # Drain any queued up speech, keep the latest
            while not speech_queue.empty():
                text = speech_queue.get_nowait()

            print(f"\n[You] {text}")

            # Interrupt anything in progress
            speaker.cancel()
            brain.interrupted = True
            time.sleep(0.05)

            MAX_CONTINUATION = 10
            user_msg = text
            for step in range(MAX_CONTINUATION):
                # Check for new speech before each step
                new_text = check_interrupt()
                if new_text:
                    print(f"\n[You] {new_text} (interrupted)")
                    speaker.cancel()
                    user_msg = new_text
                    brain.interrupted = True
                    time.sleep(0.05)

                screenshot_b64, sw, sh, scale = grab_screenshot()
                if step == 0:
                    print(f"  (screen: {sw}x{sh}, scale: {scale:.1f}x)")

                reply = brain.think(user_msg, screenshot_b64, speaker=speaker)

                if brain.interrupted:
                    break

                print(f"\n[Companion] {strip_actions_block(reply)}\n")

                actions = parse_actions(reply)

                if actions:
                    print(f"  Executing {len(actions)} action(s) (scale {scale:.1f}x)...")
                    execute_actions(actions, scale,
                                    interrupted=lambda: not speech_queue.empty())
                    if not speech_queue.empty():
                        print("  [interrupted by user]")
                        break
                    print("  Done.")
                    time.sleep(0.3)
                    user_msg = "Here's what the screen looks like now. Continue with the task if there's more to do, or just say 'done' if finished."
                else:
                    break

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        speaker.stop()


if __name__ == "__main__":
    main()
