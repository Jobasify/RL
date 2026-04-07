# Factorio RL Agent

A reinforcement learning agent that learns to play [Factorio](https://factorio.com) the way a human does — by looking at the screen and controlling the mouse and keyboard.

**No game API. No mods. Pure pixel input, human interface output.**

## Approach

The agent operates through the same interface a human player would:

- **Perception**: Real-time screen capture and audio capture from the running game
- **Action**: Simulated mouse movements, clicks, and keyboard input
- **Learning**: Reinforcement learning from raw pixel and audio observations

This is intentionally the hardest way to build a game-playing AI. The goal is to explore what's possible when an agent has no privileged access to game internals — just eyes and hands.

## Current Status

**Capture pipeline** — the perception layer is working:

- Detects the Factorio window automatically on Windows 11
- Captures frames in real time via `mss` with FPS overlay
- Captures game audio via PyAudio (WASAPI loopback)
- Saves frame bursts + audio snapshots for verification

## Usage

```bash
pip install -r requirements.txt
python capture.py
```

**Controls:**
- `s` — Save a 3-second snapshot (frames + audio) to `captures/`
- `q` — Quit

**Audio setup (Windows):** Enable "Stereo Mix" in Sound settings → Recording tab → right-click → Show Disabled Devices.

## Requirements

- Windows 11
- Python 3.10+
- Factorio running in windowed mode

## Roadmap

- [x] Screen capture pipeline
- [x] Audio capture pipeline
- [ ] Mouse and keyboard control layer
- [ ] Observation preprocessing (downscale, grayscale, frame stacking)
- [ ] Reward signal design
- [ ] RL training loop
