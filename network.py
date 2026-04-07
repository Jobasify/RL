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
    """CNN actor-critic for PPO on Factorio screen observations."""

    def __init__(self, in_channels=4, num_actions=NUM_ACTIONS):
        super().__init__()

        # --- Convolutional feature extractor ---
        # Input: (B, 4, 128, 128)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0)  # -> (B, 32, 31, 31)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)           # -> (B, 64, 14, 14)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)           # -> (B, 64, 12, 12)

        # Calculate flattened size
        self._flat_size = 64 * 12 * 12  # 9216

        # --- Shared fully connected layer ---
        self.fc = nn.Linear(self._flat_size, 512)

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

    def _extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

    def forward(self, x):
        """Full forward pass. Returns action logits and value estimate."""
        features = self._extract_features(x)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value

    def get_action(self, x):
        """Sample an action from the policy and return action, log_prob, value."""
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, x, actions):
        """Evaluate given actions. Returns log_probs, entropy, values (for PPO update)."""
        logits, value = self.forward(x)
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

    net = ActorCritic().to(device)
    print(net)

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Forward pass test ---
    print("\n--- Forward pass test ---")
    batch_size = 8
    dummy_obs = torch.randn(batch_size, 4, 128, 128, device=device)
    print(f"Input shape:  {tuple(dummy_obs.shape)}")

    logits, value = net(dummy_obs)
    print(f"Logits shape: {tuple(logits.shape)}  (batch, {NUM_ACTIONS} actions)")
    print(f"Value shape:  {tuple(value.shape)}  (batch, 1)")

    # --- Action sampling test ---
    print("\n--- Action sampling test ---")
    actions, log_probs, values = net.get_action(dummy_obs)
    print(f"Sampled actions: {actions.tolist()}")
    print(f"Action names:    {[ACTION_NAMES[a] for a in actions.tolist()]}")
    print(f"Log probs shape: {tuple(log_probs.shape)}")
    print(f"Values shape:    {tuple(values.shape)}")

    # --- Evaluate test (for PPO) ---
    print("\n--- Evaluate test (PPO update) ---")
    log_probs_eval, entropy, values_eval = net.evaluate(dummy_obs, actions)
    print(f"Log probs:  {tuple(log_probs_eval.shape)}")
    print(f"Entropy:    {tuple(entropy.shape)}  mean={entropy.mean().item():.3f}")
    print(f"Values:     {tuple(values_eval.shape)}")

    # --- Policy distribution for single observation ---
    print("\n--- Single observation policy ---")
    single = torch.randn(1, 4, 128, 128, device=device)
    logits, value = net(single)
    probs = F.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()
    print(f"Value estimate: {value.item():.4f}")
    print("Action probabilities:")
    for i, (name, p) in enumerate(zip(ACTION_NAMES, probs)):
        bar = "#" * int(p * 50)
        print(f"  {i:2d} {name:14s} {p:.3f} {bar}")

    print("\nAll shapes correct. Network ready for training.")


if __name__ == "__main__":
    main()
