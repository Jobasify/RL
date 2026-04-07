"""
Gymnasium environment wrapping the Factorio capture/control/reward pipeline.
Used by stable-baselines3 PPO as the training environment.

Observation is compact: CNN features (256) + strategy (384) + audio (128) = 768 dims.
Our existing CNN extracts spatial features and pools them, so SB3's MlpPolicy
gets meaningful visual summaries rather than raw pixels.
"""

import time

import cv2
import gymnasium as gym
import mss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from capture import find_factorio_window, AudioCapture
from control import FactorioController
from observation import ObservationProcessor
from reward import RewardSignal
from network import NUM_ACTIONS
from audio import AudioProcessor, AUDIO_FEATURE_DIM
from knowledge import KnowledgeBase, EMBEDDING_DIM
from train import execute_action, FRAME_SKIP


# Visual feature dimension after pooling
VISUAL_FEATURE_DIM = 256


class VisualFeatureExtractor(nn.Module):
    """Frozen CNN that extracts compact visual features from stacked frames.
    Same conv architecture as network.py, but with global average pool -> 256 dims."""

    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # 64 * 2 * 2 = 256
        # Random init — features will still be useful as texture/edge detectors
        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)

    @torch.no_grad()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.reshape(x.size(0), -1)  # (B, 256)


class FactorioEnv(gym.Env):
    """Gymnasium environment for Factorio via screen capture + HID control.

    Observation: CNN features (256) + strategy (384) + audio (128) = 768 dims.
    """

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

        # Visual feature extractor (frozen CNN -> 256 dims)
        self.device = torch.device("cpu")
        self.feature_extractor = VisualFeatureExtractor().to(self.device)
        self.feature_extractor.eval()

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

        # Observation space: visual (256) + strategy (384) + audio (128) = 768
        obs_size = VISUAL_FEATURE_DIM
        if use_strategy:
            obs_size += EMBEDDING_DIM
        if self.use_audio:
            obs_size += AUDIO_FEATURE_DIM
        self._obs_size = obs_size

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        print(f"  Env obs: visual({VISUAL_FEATURE_DIM})"
              f" + strategy({EMBEDDING_DIM if use_strategy else 0})"
              f" + audio({AUDIO_FEATURE_DIM if self.use_audio else 0})"
              f" = {obs_size}")

        self._step_count = 0
        self._last_audio_events = {"reward_adjustment": 0.0}

    def _get_obs(self):
        """Capture screen and build compact observation vector."""
        img = self.sct.grab(self.monitor)
        frame = np.array(img)[:, :, :3]
        self.obs_proc.push(frame)
        obs_raw = self.obs_proc.get()  # (4, 128, 128) float32

        # Extract visual features via CNN
        obs_t = torch.from_numpy(obs_raw).unsqueeze(0).to(self.device)
        visual_feats = self.feature_extractor(obs_t).squeeze(0).numpy()  # (256,)

        parts = [visual_feats]

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
