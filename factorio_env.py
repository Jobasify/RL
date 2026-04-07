"""
Gymnasium environment wrapping the Factorio capture/control/reward pipeline.
Used by stable-baselines3 PPO as the training environment.

Handles: screen capture, observation preprocessing, action execution,
reward computation, audio processing. All our existing components
wrapped in the Gymnasium interface.
"""

import time

import cv2
import gymnasium as gym
import mss
import numpy as np
from gymnasium import spaces

from capture import find_factorio_window, AudioCapture
from control import FactorioController
from observation import ObservationProcessor
from reward import RewardSignal
from network import NUM_ACTIONS
from audio import AudioProcessor, AUDIO_FEATURE_DIM
from knowledge import KnowledgeBase, EMBEDDING_DIM
from train import execute_action, FRAME_SKIP


class FactorioEnv(gym.Env):
    """Gymnasium environment for Factorio via screen capture + HID control."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, decisions_per_sec=10, frame_skip=FRAME_SKIP,
                 use_audio=True, use_strategy=True):
        super().__init__()

        # Find Factorio
        self.bbox = find_factorio_window()
        if not self.bbox:
            raise RuntimeError("Factorio window not found!")
        self.win_w = self.bbox["width"]
        self.win_h = self.bbox["height"]

        # Components
        self.ctrl = FactorioController()
        self.obs_proc = ObservationProcessor(size=128, stack_size=4)
        self.reward_signal = RewardSignal(self.win_w, self.win_h)
        self.sct = mss.mss()
        self.monitor = self.bbox
        self.frame_skip = frame_skip
        self.step_interval = 1.0 / decisions_per_sec

        # Audio
        self.use_audio = use_audio
        self.audio_capture = None
        self.audio_proc = None
        if use_audio:
            self.audio_capture = AudioCapture()
            if self.audio_capture.start():
                self.audio_proc = AudioProcessor()
            else:
                self.use_audio = False

        # Strategy
        self.use_strategy = use_strategy
        self.strategy_vec = None
        if use_strategy:
            kb = KnowledgeBase()
            kb.build()
            self.strategy_vec = kb.get_strategy_vector("Exploration")

        # Spaces
        # Observation: flattened (4, 128, 128) + optional strategy + optional audio
        obs_size = 4 * 128 * 128
        if use_strategy:
            obs_size += EMBEDDING_DIM
        if use_audio:
            obs_size += AUDIO_FEATURE_DIM

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._step_count = 0

    def _get_obs(self):
        """Capture screen and build observation vector."""
        img = self.sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]
        self.obs_proc.push(frame)
        obs = self.obs_proc.get()  # (4, 128, 128)

        parts = [obs.flatten()]

        if self.use_strategy and self.strategy_vec is not None:
            parts.append(self.strategy_vec)

        if self.use_audio and self.audio_proc:
            audio_feats, self._last_audio_events = self.audio_proc.process(self.audio_capture)
            parts.append(audio_feats)
        else:
            self._last_audio_events = {"reward_adjustment": 0.0}

        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obs_proc.reset()
        self._step_count = 0
        self._last_audio_events = {"reward_adjustment": 0.0}
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        step_start = time.time()
        action_id = int(action)

        # Execute with frame skip
        total_reward = 0.0
        for _ in range(self.frame_skip):
            execute_action(self.ctrl, action_id)
            img = self.sct.grab(self.monitor)
            frame = np.array(img)[:, :, :3]
            r, r_details = self.reward_signal.compute(frame)
            total_reward += r

        # Audio reward
        total_reward += self._last_audio_events.get("reward_adjustment", 0.0)

        # Get next observation
        obs = self._get_obs()

        self._step_count += 1
        terminated = False
        truncated = False

        # Pace control
        elapsed = time.time() - step_start
        sleep_time = self.step_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        return obs, total_reward, terminated, truncated, {}

    def close(self):
        if self.audio_capture:
            self.audio_capture.stop()
        self.sct.close()
