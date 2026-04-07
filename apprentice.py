"""
Apprentice training loop — Claude plays, CNN watches and learns.

Claude is the expert player making intelligent decisions every 5 seconds.
The CNN observes every action at 5x reward weight and learns to replicate
Claude's play through PPO updates. Over time Claude fades out and the CNN
takes over, playing independently.

Usage:
    python apprentice.py              # Full system
    python apprentice.py --baseline   # CNN only (no Claude, for comparison)
"""

import ctypes
import os
import time
import threading
import random
from pathlib import Path

import cv2
import mss
import numpy as np
import torch
import torch.nn.functional as F
from pynput import keyboard as kb

from capture import find_factorio_window, AudioCapture
from control import FactorioController
from observation import ObservationProcessor
from reward import RewardSignal
from memory import ReplayBuffer
from network import ActorCritic, NUM_ACTIONS, ACTION_NAMES
from knowledge import KnowledgeBase, EMBEDDING_DIM
from audio import AudioProcessor, AUDIO_FEATURE_DIM
from player import (ClaudePlayer, RewardTracker, get_claude_ratio,
                     EXPERT_REWARD_MULTIPLIER, DECISION_INTERVAL, _log)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STEPS_PER_UPDATE = 512
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
LR = 3e-4
DECISIONS_PER_SEC = 10
CHECKPOINT_INTERVAL = 1000
FRAME_SKIP = 4

CHECKPOINT_DIR = Path("checkpoints/apprentice")
LOG_FILE = Path("logs/apprentice.csv")


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

_kill_flag = threading.Event()


def _start_kill_switch():
    def on_press(key):
        if key == kb.Key.f12:
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
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.strategies = []
        self.audios = []

    def push(self, obs, action, log_prob, reward, value, done,
             strategy=None, audio=None):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.strategies.append(strategy)
        self.audios.append(audio)

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value):
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            next_done = 0.0 if t == n - 1 else dones[t + 1]
            delta = rewards[t] + GAMMA * next_value * (1 - next_done) - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1 - next_done) * last_gae
            advantages[t] = last_gae
        return advantages, advantages + values

    def get_batches(self, advantages, returns):
        n = len(self.rewards)
        observations = np.stack(self.observations)
        actions = np.array(self.actions, dtype=np.int64)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)
        has_strategy = self.strategies[0] is not None
        has_audio = self.audios[0] is not None
        if has_strategy:
            strategies = np.stack(self.strategies)
        if has_audio:
            audios = np.stack(self.audios)
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, MINIBATCH_SIZE):
            idx = indices[start:start + MINIBATCH_SIZE]
            batch = {
                "obs": observations[idx],
                "actions": actions[idx],
                "old_log_probs": old_log_probs[idx],
                "advantages": advantages[idx],
                "returns": returns[idx],
            }
            if has_strategy:
                batch["strategies"] = strategies[idx]
            if has_audio:
                batch["audios"] = audios[idx]
            yield batch

    def clear(self):
        self.__init__()


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(net, optimizer, rollout, device):
    has_strategy = rollout.strategies[0] is not None
    has_audio = rollout.audios[0] is not None
    with torch.no_grad():
        last_obs = torch.from_numpy(rollout.observations[-1]).unsqueeze(0).to(device)
        last_strat = torch.from_numpy(rollout.strategies[-1]).unsqueeze(0).to(device) if has_strategy else None
        last_aud = torch.from_numpy(rollout.audios[-1]).unsqueeze(0).to(device) if has_audio else None
        _, last_value = net(last_obs, strategy=last_strat, audio=last_aud)
        last_value = last_value.item()

    advantages, returns = rollout.compute_gae(last_value)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_p, total_v, total_e, n_batch = 0, 0, 0, 0
    for _ in range(PPO_EPOCHS):
        for batch in rollout.get_batches(advantages, returns):
            obs_t = torch.from_numpy(batch["obs"]).to(device)
            act_t = torch.from_numpy(batch["actions"]).to(device)
            old_lp = torch.from_numpy(batch["old_log_probs"]).to(device)
            adv_t = torch.from_numpy(batch["advantages"]).to(device)
            ret_t = torch.from_numpy(batch["returns"]).to(device)
            strat_t = torch.from_numpy(batch["strategies"]).to(device) if "strategies" in batch else None
            aud_t = torch.from_numpy(batch["audios"]).to(device) if "audios" in batch else None

            log_probs, entropy, values = net.evaluate(obs_t, act_t, strategy=strat_t, audio=aud_t)
            ratio = torch.exp(log_probs - old_lp)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(values, ret_t)
            e_mean = entropy.mean()
            loss = p_loss + VF_COEF * v_loss - ENT_COEF * e_mean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_p += p_loss.item()
            total_v += v_loss.item()
            total_e += e_mean.item()
            n_batch += 1

    return {"policy_loss": total_p / n_batch, "value_loss": total_v / n_batch,
            "entropy": total_e / n_batch}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  FACTORIO APPRENTICE — Claude plays, CNN learns")
    print("=" * 60)

    baseline_mode = "--baseline" in os.sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Components
    bbox = find_factorio_window()
    if not bbox:
        print("ERROR: Factorio not found!")
        return
    win_w, win_h = bbox["width"], bbox["height"]
    print(f"Factorio: {win_w}x{win_h}")

    ctrl = FactorioController()
    obs_proc = ObservationProcessor(size=128, stack_size=4)
    reward_signal = RewardSignal(win_w, win_h)

    # Audio
    audio_capture = AudioCapture()
    audio_ok = audio_capture.start()
    audio_proc = AudioProcessor(device=str(device)) if audio_ok else None
    audio_dim = AUDIO_FEATURE_DIM if audio_ok else 0

    # Knowledge
    knowledge = KnowledgeBase()
    knowledge.build()
    strategy_vec = knowledge.get_strategy_vector("Exploration")
    strategy_t = torch.from_numpy(strategy_vec).unsqueeze(0).to(device)

    # Network
    net = ActorCritic(strategy_dim=EMBEDDING_DIM, audio_dim=audio_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"CNN parameters: {total_params:,}")

    # Claude player
    claude = None if baseline_mode else ClaudePlayer(ctrl, bbox)
    if claude:
        print("Claude player: ACTIVE")
    else:
        print("Claude player: DISABLED (baseline mode)")

    reward_tracker = RewardTracker()
    rollout = RolloutBuffer()
    replay = ReplayBuffer(capacity=10_000)

    # Checkpoints
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(exist_ok=True)

    # Screen capture
    sct = mss.mss()
    monitor = bbox

    global_step = 0
    update_count = 0
    episode_reward = 0.0
    reward_history = []
    last_claude_time = 0
    claude_decisions = 0
    cnn_decisions = 0

    # Countdown
    print(f"\nStarting in 3 seconds — switch to Factorio! F12 to stop.")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    _start_kill_switch()
    print("\nTraining started.\n")

    paused = False
    step_interval = 1.0 / DECISIONS_PER_SEC

    try:
        while not _kill_flag.is_set():
            step_start = time.time()

            # Focus guard
            if not _is_factorio_focused():
                if not paused:
                    print("  [PAUSED] Factorio lost focus")
                    paused = True
                time.sleep(0.2)
                continue
            if paused:
                print("  [RESUMED]")
                paused = False

            # Capture observation
            img = sct.grab(monitor)
            frame = np.array(img)[:, :, :3]
            obs_proc.push(frame)
            obs = obs_proc.get()

            # Audio
            audio_vec = None
            audio_t = None
            audio_reward = 0.0
            if audio_proc:
                audio_vec, audio_events = audio_proc.process(audio_capture)
                audio_t = torch.from_numpy(audio_vec).unsqueeze(0).to(device)
                audio_reward = audio_events["reward_adjustment"]

            # Decide: Claude or CNN?
            now = time.time()
            claude_ratio = get_claude_ratio(global_step) if claude else 0.0
            use_claude = (claude
                          and random.random() < claude_ratio
                          and now - last_claude_time >= DECISION_INTERVAL)

            if use_claude:
                # --- CLAUDE PLAYS ---
                action_dict = claude.decide()
                if action_dict:
                    reason = action_dict.get("reason", "")[:60]
                    action_type = action_dict.get("action_type", "?")
                    print(f"  [CLAUDE] {action_type}: {reason}")

                    # Execute Claude's action
                    executed = claude.execute(action_dict)

                    # Capture result
                    img2 = sct.grab(monitor)
                    result_frame = np.array(img2)[:, :, :3]
                    r, r_details = reward_signal.compute(result_frame)
                    total_reward = r + audio_reward
                    if r_details.get("inventory_gain"):
                        print(f"  [CLAUDE] MINING SUCCESS! (+2.0)")

                    # Store at expert weight
                    expert_reward = total_reward * EXPERT_REWARD_MULTIPLIER
                    action_id = executed[0][0] if executed else 17

                    # Get CNN's opinion for the log_prob/value
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                        logits, value = net(obs_t, strategy=strategy_t, audio=audio_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        log_prob = dist.log_prob(torch.tensor([action_id], device=device)).item()
                        value_val = value.item()

                    obs_proc.push(result_frame)
                    next_obs = obs_proc.get()

                    rollout.push(obs, action_id, log_prob, expert_reward, value_val, False,
                                 strategy=strategy_vec, audio=audio_vec)
                    replay.push(obs, action_id, expert_reward, next_obs, False)

                    reward_tracker.add_claude(total_reward)
                    episode_reward += total_reward
                    last_claude_time = now
                    claude_decisions += 1
                    global_step += 1

                    # Sleep for the action duration
                    duration = min(action_dict.get("duration", 3), 10)
                    time.sleep(max(0, duration - (time.time() - step_start)))
                else:
                    # Claude failed to decide, fall through to CNN
                    use_claude = False

            if not use_claude:
                # --- CNN PLAYS ---
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                    action, log_prob, value = net.get_action(obs_t, strategy=strategy_t, audio=audio_t)
                    action_id = action.item()
                    log_prob_val = log_prob.item()
                    value_val = value.item()

                # Execute with frame skip
                from train import execute_action
                total_reward = 0.0
                for _ in range(FRAME_SKIP):
                    if _kill_flag.is_set():
                        break
                    execute_action(ctrl, action_id)
                    img2 = sct.grab(monitor)
                    skip_frame = np.array(img2)[:, :, :3]
                    r, r_details = reward_signal.compute(skip_frame)
                    total_reward += r
                    if r_details.get("inventory_gain"):
                        print(f"  [CNN] MINING SUCCESS! (+2.0)")
                total_reward += audio_reward

                obs_proc.push(skip_frame)
                next_obs = obs_proc.get()

                rollout.push(obs, action_id, log_prob_val, total_reward, value_val, False,
                             strategy=strategy_vec, audio=audio_vec)
                replay.push(obs, action_id, total_reward, next_obs, False)

                reward_tracker.add_cnn(total_reward)
                episode_reward += total_reward
                cnn_decisions += 1
                global_step += 1

            # PPO update
            if len(rollout) >= STEPS_PER_UPDATE:
                loss_info = ppo_update(net, optimizer, rollout, device)
                update_count += 1
                rollout.clear()

                reward_history.append(episode_reward)
                if len(reward_history) > 100:
                    reward_history.pop(0)
                rolling_avg = sum(reward_history) / len(reward_history)

                c_avg = reward_tracker.claude_avg()
                n_avg = reward_tracker.cnn_avg()
                conv = reward_tracker.convergence()
                ratio = get_claude_ratio(global_step)

                print(f"\n[Step {global_step:>7d} | Update {update_count:>3d}]"
                      f"  reward={episode_reward:+.1f}  avg100={rolling_avg:+.1f}"
                      f"  p_loss={loss_info['policy_loss']:.4f}"
                      f"  entropy={loss_info['entropy']:.3f}")
                print(f"  Claude avg: {c_avg:+.3f}  CNN avg: {n_avg:+.3f}"
                      f"  convergence: {conv:.1%}"
                      f"  ratio: {ratio:.0%} Claude / {1-ratio:.0%} CNN"
                      f"  (C:{claude_decisions} N:{cnn_decisions})")

                # CSV log
                if update_count == 1:
                    LOG_FILE.write_text(
                        "step,update,reward,avg100,claude_avg,cnn_avg,convergence,"
                        "claude_ratio,p_loss,entropy\n")
                with open(LOG_FILE, "a") as f:
                    f.write(f"{global_step},{update_count},{episode_reward:.4f},"
                            f"{rolling_avg:.4f},{c_avg:.4f},{n_avg:.4f},{conv:.4f},"
                            f"{ratio:.2f},{loss_info['policy_loss']:.4f},"
                            f"{loss_info['entropy']:.3f}\n")

                episode_reward = 0.0

            # Checkpoint
            if global_step % CHECKPOINT_INTERVAL == 0 and global_step > 0:
                path = CHECKPOINT_DIR / f"step_{global_step:07d}.pt"
                torch.save({
                    "step": global_step,
                    "model_state": net.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, path)
                print(f"  Checkpoint: {path}")

                # Attention map
                attn = net.get_attention_map()
                if attn is not None:
                    attn_dir = Path("attention_maps")
                    attn_dir.mkdir(exist_ok=True)
                    vis = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
                    vis = (vis * 255).astype(np.uint8)
                    vis = cv2.resize(vis, (256, 256), interpolation=cv2.INTER_NEAREST)
                    colored = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    cv2.imwrite(str(attn_dir / f"attn_{global_step:07d}.png"), colored)

            # Pace
            if not use_claude:
                elapsed = time.time() - step_start
                sleep_time = step_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n\nTraining stopped at step {global_step}.")
        if global_step > 0:
            path = CHECKPOINT_DIR / f"step_{global_step:07d}.pt"
            torch.save({
                "step": global_step,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, path)
            print(f"  Checkpoint saved: {path}")

        print(f"\nFinal stats:")
        print(f"  Steps:            {global_step:,}")
        print(f"  Updates:          {update_count}")
        print(f"  Claude decisions: {claude_decisions:,}")
        print(f"  CNN decisions:    {cnn_decisions:,}")
        print(f"  Claude avg100:    {reward_tracker.claude_avg():+.3f}")
        print(f"  CNN avg100:       {reward_tracker.cnn_avg():+.3f}")
        print(f"  Convergence:      {reward_tracker.convergence():.1%}")
        print(f"  Current ratio:    {get_claude_ratio(global_step):.0%} Claude")
        print("\nDone.")
        sct.close()


if __name__ == "__main__":
    main()
