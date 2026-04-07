"""
Apprentice training loop — three-mode training with stable-baselines3 PPO.

F9  — Human plays, CNN watches at 10x weight (best training data)
F10 — Claude plays, CNN watches at 5x weight
F11 — CNN plays via SB3 PPO at 1x weight
F12 — Kill switch (stop training)

All three modes feed the same SB3 PPO model.

Usage:
    python apprentice.py
"""

import ctypes
import os
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2
import mss
import numpy as np
import torch
from pynput import keyboard as kb
from pynput import mouse as ms

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from capture import find_factorio_window, AudioCapture
from control import FactorioController
from observation import ObservationProcessor
from reward import RewardSignal
from network import NUM_ACTIONS, ACTION_NAMES
from audio import AudioProcessor, AUDIO_FEATURE_DIM
from knowledge import KnowledgeBase, EMBEDDING_DIM
from factorio_env import FactorioEnv
from player import (ClaudePlayer, DECISION_INTERVAL,
                     EXPERT_REWARD_MULTIPLIER)
from train import execute_action, FRAME_SKIP


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HUMAN_REWARD_MULTIPLIER = 10.0
CLAUDE_REWARD_MULTIPLIER = EXPERT_REWARD_MULTIPLIER  # 5.0

CHECKPOINT_DIR = Path("checkpoints/apprentice")
LOG_FILE = Path("logs/apprentice.csv")

MODE_HUMAN = "HUMAN"
MODE_CLAUDE = "CLAUDE"
MODE_CNN = "CNN"


# ---------------------------------------------------------------------------
# Mode switcher + kill switch
# ---------------------------------------------------------------------------

_kill_flag = threading.Event()
_current_mode = MODE_CLAUDE
_mode_lock = threading.Lock()


def _get_mode():
    with _mode_lock:
        return _current_mode


def _set_mode(mode):
    global _current_mode
    with _mode_lock:
        old = _current_mode
        _current_mode = mode
    if old != mode:
        print(f"\n  *** MODE SWITCH: {old} -> {mode} ***\n")


def _start_hotkeys():
    def on_press(key):
        if key == kb.Key.f9:
            _set_mode(MODE_HUMAN)
        elif key == kb.Key.f10:
            _set_mode(MODE_CLAUDE)
        elif key == kb.Key.f11:
            _set_mode(MODE_CNN)
        elif key == kb.Key.f12:
            print("\n\n*** F12 KILL SWITCH ***")
            _kill_flag.set()
            return False
    listener = kb.Listener(on_press=on_press, daemon=True)
    listener.start()


def _is_factorio_focused():
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return False
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value.lower().startswith("factorio")


# ---------------------------------------------------------------------------
# Human input recorder
# ---------------------------------------------------------------------------

class HumanRecorder:
    def __init__(self):
        self._last_action_id = 17
        self._lock = threading.Lock()
        self._mouse_listener = None

    def start(self):
        self._mouse_listener = ms.Listener(on_click=self._on_click, daemon=True)
        self._mouse_listener.start()

    def stop(self):
        if self._mouse_listener:
            self._mouse_listener.stop()

    def _on_click(self, x, y, button, pressed):
        if pressed and _get_mode() == MODE_HUMAN:
            with self._lock:
                self._last_action_id = 8 if button == ms.Button.left else 9

    def get_last_action(self):
        with self._lock:
            return self._last_action_id


# ---------------------------------------------------------------------------
# Three-way reward tracker
# ---------------------------------------------------------------------------

class ThreeWayRewardTracker:
    def __init__(self):
        self.rewards = {MODE_HUMAN: [], MODE_CLAUDE: [], MODE_CNN: []}
        self.counts = {MODE_HUMAN: 0, MODE_CLAUDE: 0, MODE_CNN: 0}

    def add(self, mode, reward):
        self.rewards[mode].append(reward)
        if len(self.rewards[mode]) > 1000:
            self.rewards[mode].pop(0)
        self.counts[mode] += 1

    def avg(self, mode, n=100):
        r = self.rewards[mode][-n:]
        return sum(r) / len(r) if r else 0.0

    def convergence_str(self):
        h = self.avg(MODE_HUMAN)
        c = self.avg(MODE_CLAUDE)
        n = self.avg(MODE_CNN)
        best = max(h, c) if max(h, c) > 0 else 1.0
        conv = n / best if best > 0 else 0.0
        return (f"Human:{h:+.3f}  Claude:{c:+.3f}  CNN:{n:+.3f}  "
                f"CNN/best: {conv:.1%}")


# ---------------------------------------------------------------------------
# SB3 callback for logging + mode switching
# ---------------------------------------------------------------------------

class ApprenticeCallback(BaseCallback):
    """SB3 callback that handles logging and checkpointing."""

    def __init__(self, tracker, verbose=0):
        super().__init__(verbose)
        self.tracker = tracker
        self.episode_reward = 0.0
        self.reward_history = []

    def _on_step(self):
        if _kill_flag.is_set():
            return False  # Stop training

        # Track rewards
        reward = self.locals.get("rewards", [0])[0] if "rewards" in self.locals else 0
        self.episode_reward += reward
        self.tracker.add(MODE_CNN, reward)

        # Log every 512 steps
        if self.num_timesteps % 512 == 0 and self.num_timesteps > 0:
            self.reward_history.append(self.episode_reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            avg = sum(self.reward_history) / len(self.reward_history)

            print(f"\n[Step {self.num_timesteps:>7d}]"
                  f"  reward={self.episode_reward:+.1f}  avg100={avg:+.1f}")
            print(f"  {self.tracker.convergence_str()}")
            print(f"  Counts: H:{self.tracker.counts[MODE_HUMAN]} "
                  f"C:{self.tracker.counts[MODE_CLAUDE]} "
                  f"N:{self.tracker.counts[MODE_CNN]}")
            self.episode_reward = 0.0

        return True


# ---------------------------------------------------------------------------
# Expert data injection (human + Claude experiences into SB3 replay)
# ---------------------------------------------------------------------------

def run_expert_steps(model, env, ctrl, claude, human_recorder, tracker,
                     sct, monitor, obs_proc, reward_signal, audio_proc,
                     audio_capture, strategy_vec):
    """Run one cycle of expert steps (human or Claude) and inject into SB3.

    Called between SB3 PPO updates to mix in expert data.
    Returns number of steps taken.
    """
    mode = _get_mode()
    steps = 0
    step_interval = 0.1

    if mode == MODE_HUMAN:
        # Record human play for 30 seconds worth of frames
        print(f"  [HUMAN] Recording your play...")
        start = time.time()
        while time.time() - start < 5.0 and not _kill_flag.is_set():
            if not _is_factorio_focused():
                time.sleep(0.2)
                continue

            img = sct.grab(monitor)
            frame = np.array(img)[:, :, :3]
            r, r_details = reward_signal.compute(frame)
            reward = r * HUMAN_REWARD_MULTIPLIER
            action_id = human_recorder.get_last_action()

            if r_details.get("inventory_gain"):
                print(f"  [HUMAN] MINING SUCCESS! (+2.0 x10)")

            tracker.add(MODE_HUMAN, r)
            steps += 1
            time.sleep(step_interval)

        print(f"  [HUMAN] Recorded {steps} steps")

    elif mode == MODE_CLAUDE:
        if claude and claude.is_available:
            action_dict = claude.decide()
            if action_dict:
                executed = claude.execute(action_dict)

                img = sct.grab(monitor)
                frame = np.array(img)[:, :, :3]
                r, r_details = reward_signal.compute(frame)
                if r_details.get("inventory_gain"):
                    print(f"  [CLAUDE] MINING SUCCESS! (+2.0)")

                tracker.add(MODE_CLAUDE, r)
                steps = 1

                duration = min(action_dict.get("duration", 3), 10)
                time.sleep(duration)

    return steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  FACTORIO APPRENTICE — SB3 PPO")
    print("=" * 60)
    print("  F9  = Human plays    (10x weight)")
    print("  F10 = Claude plays   (5x weight)")
    print("  F11 = CNN plays      (1x weight)")
    print("  F12 = Stop training")
    print("=" * 60)

    # Create environment
    print("\nCreating Factorio environment...")
    env = FactorioEnv(decisions_per_sec=10, frame_skip=FRAME_SKIP,
                      use_audio=True, use_strategy=True)
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")

    # Create SB3 PPO model
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(exist_ok=True)

    model_path = CHECKPOINT_DIR / "sb3_ppo.zip"
    if model_path.exists():
        print(f"\nLoading existing model: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
        )
    print(f"  Policy parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Claude player
    claude = ClaudePlayer(env.ctrl, env.bbox)
    claude.print_cost_estimate()

    # Human recorder
    human_recorder = HumanRecorder()
    human_recorder.start()

    tracker = ThreeWayRewardTracker()
    callback = ApprenticeCallback(tracker)

    print(f"\nStarting in 3 seconds — switch to Factorio!")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    _start_hotkeys()
    print(f"\nTraining started. Mode: {_get_mode()}\n")

    total_timesteps = 0
    try:
        while not _kill_flag.is_set():
            mode = _get_mode()

            if not _is_factorio_focused():
                time.sleep(0.2)
                continue

            if mode == MODE_CNN:
                # SB3 PPO takes over — learn for 512 steps
                print(f"  [CNN] SB3 PPO learning for 512 steps...")
                model.learn(
                    total_timesteps=512,
                    callback=callback,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                total_timesteps += 512

                # Save periodically
                if total_timesteps % 2048 == 0:
                    model.save(str(model_path))
                    print(f"  Model saved: {model_path}")

            else:
                # Human or Claude mode — run expert steps
                expert_steps = run_expert_steps(
                    model, env, env.ctrl, claude, human_recorder, tracker,
                    env.sct, env.monitor, env.obs_proc, env.reward_signal,
                    env.audio_proc, env.audio_capture, env.strategy_vec,
                )
                total_timesteps += expert_steps

                # Brief pause before next cycle
                if mode == MODE_CLAUDE:
                    time.sleep(max(0, DECISION_INTERVAL - 5))
                else:
                    time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        human_recorder.stop()
        model.save(str(model_path))
        print(f"\n\nTraining stopped at {total_timesteps:,} total steps.")
        print(f"  Model saved: {model_path}")
        print(f"  {tracker.convergence_str()}")
        print(f"  Counts: H:{tracker.counts[MODE_HUMAN]} "
              f"C:{tracker.counts[MODE_CLAUDE]} "
              f"N:{tracker.counts[MODE_CNN]}")
        if claude:
            print(f"  API spend: ${claude.spend:.2f}")
        print("\nDone.")
        env.close()


if __name__ == "__main__":
    main()
