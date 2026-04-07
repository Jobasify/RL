"""
Actor-Critic convolutional neural network for Factorio RL agent.
Takes (4, 128, 128) stacked grayscale observations.
Outputs a policy (action probabilities) and a value estimate.

Action space (discrete, 18 actions):
  0-3:   Move (W, A, S, D)
  4-7:   Move + Shift (sprint W, A, S, D)
  8:     Left click (at current mouse pos)
  9:     Right click
  10-13: Mouse move (up, down, left, right by a step)
  14:    Space (shoot / use)
  15:    E (open inventory / interact)
  16:    Q (drop / pipette)
  17:    No-op (do nothing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


NUM_ACTIONS = 18


# ---------------------------------------------------------------------------
# Spatial attention module
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """Learns which spatial regions of the feature map to focus on.

    Takes (B, C, H, W) conv features, produces attention weights per
    spatial position, returns attended features + the attention map.
    """

    def __init__(self, channels):
        super().__init__()
        # 1x1 conv to produce single-channel attention map
        self.query = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.attn_conv = nn.Conv2d(channels // 4, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable blend

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature maps

        Returns:
            attended: (B, C, H, W) attention-weighted features
            attn_map: (B, 1, H, W) attention weights (for visualization)
        """
        q = F.relu(self.query(x))
        k = F.relu(self.key(x))
        attn_logits = self.attn_conv(q * k)  # (B, 1, H, W)
        attn_map = torch.sigmoid(attn_logits)

        # Blend: attended = x + gamma * (attn * x)
        attended = x + self.gamma * (attn_map * x)
        return attended, attn_map


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """CNN actor-critic with spatial attention, strategy vector, and audio features.

    Input pipeline:
      Visual: (B, 4, 128, 128) -> 3 conv layers -> spatial attention -> flatten
      Strategy: (B, 384) knowledge/advisor embedding
      Audio: (B, 128) mel spectrogram features
      Combined: visual + strategy + audio -> 512 FC -> policy + value heads
    """

    def __init__(self, in_channels=4, num_actions=NUM_ACTIONS,
                 strategy_dim=0, audio_dim=0):
        super().__init__()
        self.strategy_dim = strategy_dim
        self.audio_dim = audio_dim

        # --- Convolutional feature extractor ---
        # Input: (B, 4, 128, 128)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0)  # -> (B, 32, 31, 31)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)           # -> (B, 64, 14, 14)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)           # -> (B, 64, 12, 12)

        # --- Spatial attention ---
        self.attention = SpatialAttention(64)
        self._last_attn_map = None  # Cached for visualization

        # Calculate flattened size
        self._flat_size = 64 * 12 * 12  # 9216

        # --- Shared fully connected layer ---
        # Attended CNN features + strategy + audio -> 512
        self.fc = nn.Linear(self._flat_size + strategy_dim + audio_dim, 512)

        # --- Policy head (actor) ---
        self.policy = nn.Linear(512, num_actions)

        # --- Value head (critic) ---
        self.value = nn.Linear(512, 1)

        # Orthogonal init (standard for PPO)
        for layer in [self.conv1, self.conv2, self.conv3, self.fc]:
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def _extract_features(self, x, strategy=None, audio=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Spatial attention
        x, attn_map = self.attention(x)
        self._last_attn_map = attn_map.detach()

        x = x.reshape(x.size(0), -1)
        # Concatenate strategy and audio vectors if provided
        extras = []
        if strategy is not None:
            extras.append(strategy)
        if audio is not None:
            extras.append(audio)
        if extras:
            x = torch.cat([x] + extras, dim=-1)
        x = F.relu(self.fc(x))
        return x

    def forward(self, x, strategy=None, audio=None):
        """Full forward pass. Returns action logits and value estimate."""
        features = self._extract_features(x, strategy, audio)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value

    def get_action(self, x, strategy=None, audio=None):
        """Sample an action from the policy and return action, log_prob, value."""
        logits, value = self.forward(x, strategy, audio)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, x, actions, strategy=None, audio=None):
        """Evaluate given actions. Returns log_probs, entropy, values (for PPO update)."""
        logits, value = self.forward(x, strategy, audio)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)

    def get_attention_map(self):
        """Return the last attention map as numpy (H, W). For visualization."""
        if self._last_attn_map is None:
            return None
        return self._last_attn_map[0, 0].cpu().numpy()


ACTION_NAMES = [
    "Move W", "Move A", "Move S", "Move D",
    "Sprint W", "Sprint A", "Sprint S", "Sprint D",
    "Left Click", "Right Click",
    "Mouse Up", "Mouse Down", "Mouse Left", "Mouse Right",
    "Space", "E (interact)", "Q (pipette)", "No-op",
]


def main():
    print("=== Actor-Critic Network ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    batch_size = 8
    dummy_obs = torch.randn(batch_size, 4, 128, 128, device=device)

    # --- Full model: CNN + attention + strategy + audio ---
    print("--- Full model: CNN + attention + strategy (384) + audio (128) ---")
    net = ActorCritic(strategy_dim=384, audio_dim=128).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {total_params:,}")

    dummy_strategy = torch.randn(batch_size, 384, device=device)
    dummy_audio = torch.randn(batch_size, 128, device=device)

    logits, value = net(dummy_obs, strategy=dummy_strategy, audio=dummy_audio)
    print(f"Input: obs {tuple(dummy_obs.shape)} + strategy (8,384) + audio (8,128)")
    print(f"  -> logits: {tuple(logits.shape)}, value: {tuple(value.shape)}")

    actions, lp, vals = net.get_action(dummy_obs, strategy=dummy_strategy, audio=dummy_audio)
    print(f"Actions: {[ACTION_NAMES[a] for a in actions.tolist()[:4]]}...")

    lp_eval, ent, v_eval = net.evaluate(dummy_obs, actions, strategy=dummy_strategy, audio=dummy_audio)
    print(f"Evaluate: log_probs {tuple(lp_eval.shape)}, entropy={ent.mean().item():.3f}")

    # --- Attention map ---
    attn = net.get_attention_map()
    print(f"\nAttention map: shape={attn.shape}, range=[{attn.min():.3f}, {attn.max():.3f}]")

    # --- Breakdown ---
    attn_params = sum(p.numel() for p in net.attention.parameters())
    print(f"\nFC input: {net._flat_size} (visual) + {net.strategy_dim} (strategy) + {net.audio_dim} (audio) = {net._flat_size + net.strategy_dim + net.audio_dim}")
    print(f"Attention module: {attn_params:,} params")
    print(f"Total: {total_params:,} params")

    # --- Backward compatible: no extras ---
    print("\n--- Baseline mode: CNN + attention only ---")
    net0 = ActorCritic().to(device)
    p0 = sum(p.numel() for p in net0.parameters())
    logits0, _ = net0(dummy_obs)
    print(f"Parameters: {p0:,}, logits: {tuple(logits0.shape)}")


if __name__ == "__main__":
    main()
