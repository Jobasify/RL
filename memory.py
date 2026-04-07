"""
Experience replay buffer for the Factorio RL agent.
Stores (observation, action, reward, next_observation, done) tuples
in a fixed-size circular buffer. Supports random batch sampling,
statistics tracking, and save/load persistence.
"""

import os
import random
from pathlib import Path

import numpy as np


class ReplayBuffer:
    """Circular experience buffer with batch sampling and persistence."""

    def __init__(self, capacity=10_000, obs_shape=(4, 128, 128)):
        self.capacity = capacity
        self.obs_shape = obs_shape

        # Pre-allocate arrays for efficiency
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.position = 0  # Next write index
        self.size = 0       # Current number of stored experiences
        self.total_added = 0

    def push(self, obs, action, reward, next_obs, done):
        """Store a single experience. Overwrites oldest when full."""
        self.observations[self.position] = obs
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_obs
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def sample(self, batch_size=32):
        """Sample a random batch. Returns dict of numpy arrays."""
        indices = random.sample(range(self.size), min(batch_size, self.size))
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def stats(self):
        """Return buffer statistics."""
        if self.size == 0:
            return {"size": 0, "capacity": self.capacity, "total_added": 0,
                    "mean_reward": 0.0, "recent_100_reward": 0.0,
                    "min_reward": 0.0, "max_reward": 0.0}

        rewards = self.rewards[:self.size]
        recent_n = min(100, self.size)
        # Recent 100 = the last 100 experiences added
        if self.size < self.capacity:
            recent_rewards = rewards[max(0, self.size - recent_n):self.size]
        else:
            # Buffer has wrapped — recent entries are behind the write position
            indices = [(self.position - 1 - i) % self.capacity for i in range(recent_n)]
            recent_rewards = self.rewards[indices]

        return {
            "size": self.size,
            "capacity": self.capacity,
            "total_added": self.total_added,
            "fill_pct": self.size / self.capacity * 100,
            "mean_reward": float(rewards.mean()),
            "recent_100_reward": float(recent_rewards.mean()),
            "min_reward": float(rewards.min()),
            "max_reward": float(rewards.max()),
        }

    def save(self, path="buffer.npz"):
        """Save buffer to disk."""
        path = Path(path)
        np.savez_compressed(
            path,
            observations=self.observations[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_observations=self.next_observations[:self.size],
            dones=self.dones[:self.size],
            position=np.array([self.position]),
            total_added=np.array([self.total_added]),
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Buffer saved to {path} ({size_mb:.1f} MB, {self.size} experiences)")

    def load(self, path="buffer.npz"):
        """Load buffer from disk."""
        path = Path(path)
        if not path.exists():
            print(f"No saved buffer at {path}")
            return False

        data = np.load(path)
        n = len(data["actions"])
        self.size = min(n, self.capacity)
        self.observations[:self.size] = data["observations"][:self.size]
        self.actions[:self.size] = data["actions"][:self.size]
        self.rewards[:self.size] = data["rewards"][:self.size]
        self.next_observations[:self.size] = data["next_observations"][:self.size]
        self.dones[:self.size] = data["dones"][:self.size]
        self.position = int(data["position"][0]) if self.size < self.capacity else 0
        self.total_added = int(data["total_added"][0])

        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Buffer loaded from {path} ({size_mb:.1f} MB, {self.size} experiences)")
        return True


def print_stats(buf):
    s = buf.stats()
    print(f"  Size:             {s['size']:,} / {s['capacity']:,} ({s.get('fill_pct', 0):.1f}%)")
    print(f"  Total added:      {s['total_added']:,}")
    print(f"  Mean reward:      {s['mean_reward']:+.4f}")
    print(f"  Recent 100 avg:   {s['recent_100_reward']:+.4f}")
    print(f"  Reward range:     [{s['min_reward']:+.4f}, {s['max_reward']:+.4f}]")


def main():
    print("=== Experience Replay Buffer Test ===\n")

    obs_shape = (4, 128, 128)
    buf = ReplayBuffer(capacity=10_000, obs_shape=obs_shape)

    # Fill with synthetic experiences
    print("Filling buffer with 500 synthetic experiences...")
    for i in range(500):
        obs = np.random.rand(*obs_shape).astype(np.float32)
        action = random.randint(0, 17)
        reward = random.gauss(0.05, 0.2)
        next_obs = np.random.rand(*obs_shape).astype(np.float32)
        done = random.random() < 0.01
        buf.push(obs, action, reward, next_obs, done)

    print("\nBuffer stats after 500 experiences:")
    print_stats(buf)

    # Sample a batch
    print("\n--- Batch sample (32) ---")
    batch = buf.sample(batch_size=32)
    print(f"  observations:      {batch['observations'].shape}  {batch['observations'].dtype}")
    print(f"  actions:           {batch['actions'].shape}  {batch['actions'].dtype}")
    print(f"  rewards:           {batch['rewards'].shape}  {batch['rewards'].dtype}")
    print(f"  next_observations: {batch['next_observations'].shape}  {batch['next_observations'].dtype}")
    print(f"  dones:             {batch['dones'].shape}  {batch['dones'].dtype}")
    print(f"  reward range:      [{batch['rewards'].min():+.4f}, {batch['rewards'].max():+.4f}]")

    # Test save/load
    print("\n--- Save/Load test ---")
    save_path = "test_buffer.npz"
    buf.save(save_path)

    buf2 = ReplayBuffer(capacity=10_000, obs_shape=obs_shape)
    buf2.load(save_path)
    print("\nLoaded buffer stats:")
    print_stats(buf2)

    # Verify data integrity
    assert buf2.size == buf.size
    assert np.array_equal(buf2.actions[:buf.size], buf.actions[:buf.size])
    assert np.allclose(buf2.rewards[:buf.size], buf.rewards[:buf.size])
    print("\nData integrity check: PASSED")

    # Test overflow (circular behavior)
    print("\n--- Overflow test ---")
    print("Adding 10,500 more experiences to a capacity-10,000 buffer...")
    for i in range(10_500):
        obs = np.random.rand(*obs_shape).astype(np.float32)
        action = random.randint(0, 17)
        reward = 1.0 if i >= 10_000 else -1.0  # Last 500 are positive
        next_obs = np.random.rand(*obs_shape).astype(np.float32)
        buf.push(obs, action, reward, next_obs, False)

    print("\nBuffer stats after overflow:")
    print_stats(buf)
    print(f"  (recent 100 should be ~+1.0 since last 500 entries have reward=+1.0)")

    # Cleanup
    os.remove(save_path)
    print(f"\nCleaned up {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
