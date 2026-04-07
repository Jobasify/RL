"""
Apprentice training loop — three-mode training with seamless switching.

F9  — Human plays, CNN watches at 10x weight (best training data)
F10 — Claude plays, CNN watches at 5x weight
F11 — CNN plays independently at 1x weight
F12 — Kill switch (stop training)

All three modes feed the same PPO buffer and training loop.

Usage:
    python apprentice.py
"""

import ctypes
import os
import time
import threading
import random
from pathlib import Path
from datetime import datetime

import cv2
import mss
import numpy as np
import torch
import torch.nn.functional as F
from pynput import keyboard as kb
from pynput import mouse as ms

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

HUMAN_REWARD_MULTIPLIER = 10.0
CLAUDE_REWARD_MULTIPLIER = EXPERT_REWARD_MULTIPLIER  # 5.0

CHECKPOINT_DIR = Path("checkpoints/apprentice")
LOG_FILE = Path("logs/apprentice.csv")

# Modes
MODE_HUMAN = "HUMAN"
MODE_CLAUDE = "CLAUDE"
MODE_CNN = "CNN"


# ---------------------------------------------------------------------------
# Mode switcher + kill switch
# ---------------------------------------------------------------------------

_kill_flag = threading.Event()
_current_mode = MODE_CLAUDE  # Default start mode
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
        _log(f"MODE SWITCH: {old} -> {mode}")


def _start_hotkeys():
    """Listen for F9/F10/F11 mode switches and F12 kill."""
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
    return listener


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
    """Records human mouse and keyboard inputs for training data."""

    def __init__(self):
        self._last_action_id = 17  # no-op
        self._last_action_time = time.time()
        self._lock = threading.Lock()
        self._mouse_listener = None
        self._active = False

    def start(self):
        """Start recording human inputs."""
        self._active = True
        # Mouse listener runs in background
        self._mouse_listener = ms.Listener(
            on_click=self._on_click,
            daemon=True,
        )
        self._mouse_listener.start()

    def stop(self):
        self._active = False
        if self._mouse_listener:
            self._mouse_listener.stop()

    def _on_click(self, x, y, button, pressed):
        if not self._active or _get_mode() != MODE_HUMAN:
            return
        if pressed:
            with self._lock:
                if button == ms.Button.left:
                    self._last_action_id = 8
                elif button == ms.Button.right:
                    self._last_action_id = 9
                self._last_action_time = time.time()

    def record_key(self, key_char):
        """Record a key press (called from the main hotkey listener)."""
        key_map = {"w": 0, "a": 1, "s": 2, "d": 3, "e": 15, "q": 16, " ": 14}
        with self._lock:
            self._last_action_id = key_map.get(key_char, 17)
            self._last_action_time = time.time()

    def get_last_action(self):
        """Get the most recent human action ID."""
        with self._lock:
            return self._last_action_id


# ---------------------------------------------------------------------------
# Extended reward tracker (3-way)
# ---------------------------------------------------------------------------

class ThreeWayRewardTracker:
    """Tracks Human vs Claude vs CNN reward separately."""

    def __init__(self):
        self.human_rewards = []
        self.claude_rewards = []
        self.cnn_rewards = []

    def add(self, mode, reward):
        if mode == MODE_HUMAN:
            self.human_rewards.append(reward)
            if len(self.human_rewards) > 1000:
                self.human_rewards.pop(0)
        elif mode == MODE_CLAUDE:
            self.claude_rewards.append(reward)
            if len(self.claude_rewards) > 1000:
                self.claude_rewards.pop(0)
        else:
            self.cnn_rewards.append(reward)
            if len(self.cnn_rewards) > 1000:
                self.cnn_rewards.pop(0)

    def avg(self, mode, n=100):
        if mode == MODE_HUMAN:
            r = self.human_rewards[-n:]
        elif mode == MODE_CLAUDE:
            r = self.claude_rewards[-n:]
        else:
            r = self.cnn_rewards[-n:]
        return sum(r) / len(r) if r else 0.0

    def counts(self):
        return {
            MODE_HUMAN: len(self.human_rewards),
            MODE_CLAUDE: len(self.claude_rewards),
            MODE_CNN: len(self.cnn_rewards),
        }

    def convergence_str(self):
        h = self.avg(MODE_HUMAN)
        c = self.avg(MODE_CLAUDE)
        n = self.avg(MODE_CNN)
        best = max(h, c) if max(h, c) > 0 else 1.0
        conv = n / best if best > 0 else 0.0
        return (f"Human:{h:+.3f}  Claude:{c:+.3f}  CNN:{n:+.3f}  "
                f"CNN/best: {conv:.1%}")


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
    print("  FACTORIO APPRENTICE — Human / Claude / CNN")
    print("=" * 60)
    print("  F9  = Human plays    (10x weight)")
    print("  F10 = Claude plays   (5x weight)")
    print("  F11 = CNN plays      (1x weight)")
    print("  F12 = Stop training")
    print("=" * 60)

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
    print(f"CNN parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Claude player
    claude = ClaudePlayer(ctrl, bbox)
    claude.print_cost_estimate()

    # Human recorder
    human_recorder = HumanRecorder()
    human_recorder.start()

    tracker = ThreeWayRewardTracker()
    rollout = RolloutBuffer()
    replay = ReplayBuffer(capacity=10_000)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(exist_ok=True)

    sct = mss.mss()
    monitor = bbox

    global_step = 0
    update_count = 0
    episode_reward = 0.0
    reward_history = []
    last_claude_time = 0
    mode_counts = {MODE_HUMAN: 0, MODE_CLAUDE: 0, MODE_CNN: 0}

    print(f"\nStarting in 3 seconds — switch to Factorio!")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    _start_hotkeys()
    print(f"\nTraining started. Mode: {_get_mode()}\n")

    paused = False
    step_interval = 1.0 / DECISIONS_PER_SEC

    try:
        while not _kill_flag.is_set():
            step_start = time.time()
            mode = _get_mode()

            # Focus guard
            if not _is_factorio_focused():
                if not paused:
                    print(f"  [PAUSED] Factorio lost focus")
                    paused = True
                time.sleep(0.2)
                continue
            if paused:
                print(f"  [RESUMED] Mode: {mode}")
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

            # ============================================================
            # HUMAN MODE — record what the human does
            # ============================================================
            if mode == MODE_HUMAN:
                action_id = human_recorder.get_last_action()

                # Don't execute anything — human is controlling
                # Just capture the result after a frame skip delay
                time.sleep(step_interval * FRAME_SKIP)

                img2 = sct.grab(monitor)
                result_frame = np.array(img2)[:, :, :3]
                r, r_details = reward_signal.compute(result_frame)
                total_reward = (r + audio_reward) * HUMAN_REWARD_MULTIPLIER

                if r_details.get("inventory_gain"):
                    print(f"  [HUMAN] MINING SUCCESS! (+2.0 x10)")

                # Get CNN's evaluation of the human's action
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                    logits, value = net(obs_t, strategy=strategy_t, audio=audio_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    lp = dist.log_prob(torch.tensor([action_id], device=device)).item()
                    vv = value.item()

                obs_proc.push(result_frame)
                rollout.push(obs, action_id, lp, total_reward, vv, False,
                             strategy=strategy_vec, audio=audio_vec)
                replay.push(obs, action_id, total_reward, obs_proc.get(), False)

                tracker.add(MODE_HUMAN, r + audio_reward)
                episode_reward += r + audio_reward
                mode_counts[MODE_HUMAN] += 1
                global_step += 1

            # ============================================================
            # CLAUDE MODE — Claude decides every 30s, CNN fills gaps
            # ============================================================
            elif mode == MODE_CLAUDE:
                now = time.time()
                use_claude = (claude.is_available
                              and now - last_claude_time >= DECISION_INTERVAL)

                if use_claude:
                    action_dict = claude.decide()
                    if action_dict:
                        executed = claude.execute(action_dict)

                        img2 = sct.grab(monitor)
                        result_frame = np.array(img2)[:, :, :3]
                        r, r_details = reward_signal.compute(result_frame)
                        total_reward = r + audio_reward
                        if r_details.get("inventory_gain"):
                            print(f"  [CLAUDE] MINING SUCCESS! (+2.0)")

                        expert_reward = total_reward * CLAUDE_REWARD_MULTIPLIER
                        action_id = executed[0][0] if executed else 17

                        with torch.no_grad():
                            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                            logits, value = net(obs_t, strategy=strategy_t, audio=audio_t)
                            dist = torch.distributions.Categorical(logits=logits)
                            lp = dist.log_prob(torch.tensor([action_id], device=device)).item()
                            vv = value.item()

                        obs_proc.push(result_frame)
                        rollout.push(obs, action_id, lp, expert_reward, vv, False,
                                     strategy=strategy_vec, audio=audio_vec)
                        replay.push(obs, action_id, expert_reward, obs_proc.get(), False)

                        tracker.add(MODE_CLAUDE, total_reward)
                        episode_reward += total_reward
                        last_claude_time = now
                        mode_counts[MODE_CLAUDE] += 1
                        global_step += 1

                        duration = min(action_dict.get("duration", 3), 10)
                        time.sleep(max(0, duration - (time.time() - step_start)))
                    else:
                        use_claude = False

                if not use_claude:
                    # CNN fills gaps between Claude decisions
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                        action, log_prob, value = net.get_action(
                            obs_t, strategy=strategy_t, audio=audio_t)
                        action_id = action.item()
                        lp_val = log_prob.item()
                        vv = value.item()

                    from train import execute_action
                    total_reward = 0.0
                    for _ in range(FRAME_SKIP):
                        if _kill_flag.is_set():
                            break
                        execute_action(ctrl, action_id)
                        img2 = sct.grab(monitor)
                        sf = np.array(img2)[:, :, :3]
                        r, rd = reward_signal.compute(sf)
                        total_reward += r
                    total_reward += audio_reward

                    obs_proc.push(sf)
                    rollout.push(obs, action_id, lp_val, total_reward, vv, False,
                                 strategy=strategy_vec, audio=audio_vec)
                    replay.push(obs, action_id, total_reward, obs_proc.get(), False)

                    tracker.add(MODE_CNN, total_reward)
                    episode_reward += total_reward
                    mode_counts[MODE_CNN] += 1
                    global_step += 1

            # ============================================================
            # CNN MODE — CNN plays independently
            # ============================================================
            elif mode == MODE_CNN:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                    action, log_prob, value = net.get_action(
                        obs_t, strategy=strategy_t, audio=audio_t)
                    action_id = action.item()
                    lp_val = log_prob.item()
                    vv = value.item()

                from train import execute_action
                total_reward = 0.0
                for _ in range(FRAME_SKIP):
                    if _kill_flag.is_set():
                        break
                    execute_action(ctrl, action_id)
                    img2 = sct.grab(monitor)
                    sf = np.array(img2)[:, :, :3]
                    r, rd = reward_signal.compute(sf)
                    total_reward += r
                total_reward += audio_reward

                obs_proc.push(sf)
                rollout.push(obs, action_id, lp_val, total_reward, vv, False,
                             strategy=strategy_vec, audio=audio_vec)
                replay.push(obs, action_id, total_reward, obs_proc.get(), False)

                tracker.add(MODE_CNN, total_reward)
                episode_reward += total_reward
                mode_counts[MODE_CNN] += 1
                global_step += 1

            # ============================================================
            # PPO update
            # ============================================================
            if len(rollout) >= STEPS_PER_UPDATE:
                loss_info = ppo_update(net, optimizer, rollout, device)
                update_count += 1
                rollout.clear()

                reward_history.append(episode_reward)
                if len(reward_history) > 100:
                    reward_history.pop(0)
                rolling_avg = sum(reward_history) / len(reward_history)

                print(f"\n[Step {global_step:>7d} | Update {update_count:>3d} | {mode}]"
                      f"  reward={episode_reward:+.1f}  avg100={rolling_avg:+.1f}"
                      f"  p_loss={loss_info['policy_loss']:.4f}"
                      f"  entropy={loss_info['entropy']:.3f}")
                print(f"  {tracker.convergence_str()}")
                print(f"  Counts: H:{mode_counts[MODE_HUMAN]} "
                      f"C:{mode_counts[MODE_CLAUDE]} "
                      f"N:{mode_counts[MODE_CNN]}"
                      f"  spend: ${claude.spend:.2f}")

                # CSV
                if update_count == 1:
                    LOG_FILE.write_text(
                        "step,update,mode,reward,avg100,human_avg,claude_avg,"
                        "cnn_avg,p_loss,entropy\n")
                with open(LOG_FILE, "a") as f:
                    f.write(f"{global_step},{update_count},{mode},"
                            f"{episode_reward:.4f},{rolling_avg:.4f},"
                            f"{tracker.avg(MODE_HUMAN):.4f},"
                            f"{tracker.avg(MODE_CLAUDE):.4f},"
                            f"{tracker.avg(MODE_CNN):.4f},"
                            f"{loss_info['policy_loss']:.4f},"
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

                attn = net.get_attention_map()
                if attn is not None:
                    attn_dir = Path("attention_maps")
                    attn_dir.mkdir(exist_ok=True)
                    vis = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
                    vis = (vis * 255).astype(np.uint8)
                    vis = cv2.resize(vis, (256, 256), interpolation=cv2.INTER_NEAREST)
                    colored = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    cv2.imwrite(str(attn_dir / f"attn_{global_step:07d}.png"), colored)

            # Pace (human mode doesn't pace — human controls timing)
            if mode != MODE_HUMAN:
                elapsed = time.time() - step_start
                sleep_time = step_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        human_recorder.stop()
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
        print(f"  Steps:          {global_step:,}")
        print(f"  Updates:        {update_count}")
        print(f"  Human steps:    {mode_counts[MODE_HUMAN]:,}")
        print(f"  Claude steps:   {mode_counts[MODE_CLAUDE]:,}")
        print(f"  CNN steps:      {mode_counts[MODE_CNN]:,}")
        print(f"  {tracker.convergence_str()}")
        print(f"  API spend:      ${claude.spend:.2f}")
        print("\nDone.")
        sct.close()


if __name__ == "__main__":
    main()
