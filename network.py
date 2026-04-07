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


class ActorCritic(nn.Module):
    """CNN actor-critic for PPO on Factorio screen observations.

    Optionally accepts a strategy vector from the knowledge system.
    When provided, the strategy vector is concatenated with CNN features
    before the policy and value heads — pixels + knowledge = decisions.
    """

    def __init__(self, in_channels=4, num_actions=NUM_ACTIONS, strategy_dim=0):
        super().__init__()
        self.strategy_dim = strategy_dim

        # --- Convolutional feature extractor ---
        # Input: (B, 4, 128, 128)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0)  # -> (B, 32, 31, 31)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)           # -> (B, 64, 14, 14)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)           # -> (B, 64, 12, 12)

        # Calculate flattened size
        self._flat_size = 64 * 12 * 12  # 9216

        # --- Shared fully connected layer ---
        # CNN features + optional strategy vector -> 512
        self.fc = nn.Linear(self._flat_size + strategy_dim, 512)

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

    def _extract_features(self, x, strategy=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        # Concatenate strategy vector if provided
        if strategy is not None:
            x = torch.cat([x, strategy], dim=-1)
        x = F.relu(self.fc(x))
        return x

    def forward(self, x, strategy=None):
        """Full forward pass. Returns action logits and value estimate."""
        features = self._extract_features(x, strategy)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value

    def get_action(self, x, strategy=None):
        """Sample an action from the policy and return action, log_prob, value."""
        logits, value = self.forward(x, strategy)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, x, actions, strategy=None):
        """Evaluate given actions. Returns log_probs, entropy, values (for PPO update)."""
        logits, value = self.forward(x, strategy)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)


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

    # --- Test without strategy (backward compatible) ---
    print("--- Mode 1: CNN only (no strategy vector) ---")
    net = ActorCritic().to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {total_params:,}")

    batch_size = 8
    dummy_obs = torch.randn(batch_size, 4, 128, 128, device=device)
    logits, value = net(dummy_obs)
    print(f"Input: {tuple(dummy_obs.shape)} -> logits: {tuple(logits.shape)}, value: {tuple(value.shape)}")

    actions, log_probs, values = net.get_action(dummy_obs)
    print(f"Actions: {[ACTION_NAMES[a] for a in actions.tolist()[:4]]}...")

    # --- Test with strategy vector ---
    print("\n--- Mode 2: CNN + strategy vector (knowledge-augmented) ---")
    strategy_dim = 384
    net2 = ActorCritic(strategy_dim=strategy_dim).to(device)
    total_params2 = sum(p.numel() for p in net2.parameters())
    print(f"Parameters: {total_params2:,} (+{total_params2 - total_params:,} from strategy)")

    dummy_strategy = torch.randn(batch_size, strategy_dim, device=device)
    logits2, value2 = net2(dummy_obs, strategy=dummy_strategy)
    print(f"Input: obs {tuple(dummy_obs.shape)} + strategy {tuple(dummy_strategy.shape)}")
    print(f"  -> logits: {tuple(logits2.shape)}, value: {tuple(value2.shape)}")

    actions2, lp2, v2 = net2.get_action(dummy_obs, strategy=dummy_strategy)
    print(f"Actions: {[ACTION_NAMES[a] for a in actions2.tolist()[:4]]}...")

    lp_eval, ent, v_eval = net2.evaluate(dummy_obs, actions2, strategy=dummy_strategy)
    print(f"Evaluate: log_probs {tuple(lp_eval.shape)}, entropy mean={ent.mean().item():.3f}")

    print(f"\nBoth modes work. FC layer: {net._flat_size} (CNN only) vs "
          f"{net2._flat_size}+{strategy_dim} (CNN+strategy) -> 512")


if __name__ == "__main__":
    main()
