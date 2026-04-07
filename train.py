"""
PPO training loop for the Factorio RL agent.
Connects all components: capture, control, observation, reward, memory, network.
Runs at 10 decisions/sec, PPO update every 512 steps.

Safety: F12 = instant kill switch. Actions only execute when Factorio is focused.
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

from capture import find_factorio_window
from control import FactorioController
from observation import ObservationProcessor
from reward import RewardSignal
from memory import ReplayBuffer
from network import ActorCritic, NUM_ACTIONS, ACTION_NAMES
from knowledge import KnowledgeBase, EMBEDDING_DIM
from advisor import Advisor
from translator import ActionTranslator
from audio import AudioProcessor, AUDIO_FEATURE_DIM
from capture import AudioCapture


# ---------------------------------------------------------------------------
# Safety: kill switch + focus guard
# ---------------------------------------------------------------------------

_kill_flag = threading.Event()


def _start_kill_switch():
    """Listen for F12 globally to stop training instantly."""
    def on_press(key):
        if key == kb.Key.f12:
            print("\n\n*** F12 KILL SWITCH — stopping training ***")
            _kill_flag.set()
            return False  # Stop listener
    listener = kb.Listener(on_press=on_press, daemon=True)
    listener.start()
    return listener


def _is_factorio_focused():
    """Check if the Factorio window is the current foreground window."""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return False
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value.lower().startswith("factorio")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STEPS_PER_UPDATE = 512      # Collect this many steps before PPO update
PPO_EPOCHS = 4              # Passes over rollout data per update
MINIBATCH_SIZE = 64         # Minibatch size for PPO
GAMMA = 0.99                # Discount factor
GAE_LAMBDA = 0.95           # GAE lambda
CLIP_EPS = 0.2              # PPO clipping
VF_COEF = 0.5               # Value loss coefficient
ENT_COEF = 0.01             # Entropy bonus coefficient
MAX_GRAD_NORM = 0.5         # Gradient clipping
LR = 3e-4                   # Learning rate
DECISIONS_PER_SEC = 10      # Pace limiter
CHECKPOINT_INTERVAL = 1000  # Save every N steps
MOUSE_STEP = 80             # Pixels per mouse move action
FRAME_SKIP = 4              # Repeat each action for N frames, observe the last
STUCK_WINDOW = 500          # Check avg100 trend over this many steps
STUCK_THRESHOLD = 0.01      # If avg100 improves less than this, agent is stuck
LOOP_REPEAT_MIN = 5         # Consecutive same-action threshold for loop detection
LOOP_PENALTY = 0.1          # Penalty for unproductive repetition

CHECKPOINT_DIR = Path("checkpoints")

# ---------------------------------------------------------------------------
# Curriculum stages — reward weights change as the agent progresses
# Each stage: (name, step_threshold, region_multipliers, noop_penalty, click_bonus)
#   - region_multipliers: applied on top of base weights in reward.py
#   - noop_penalty: subtracted from reward when action 17 (no-op) is chosen
#   - click_bonus: added to reward when action 8 (left click) is chosen
# ---------------------------------------------------------------------------

STAGE_THRESHOLDS = [5_000, 20_000]  # Boundaries between stages

CURRICULUM = [
    {   # Stage 1: Exploration (steps 0 - 5,000)
        "name": "Exploration",
        "multipliers": {"Resources": 0.5, "Minimap": 2.0, "Hotbar": 1.0, "World": 2.0},
        "noop_penalty": 0.02,   # Discourage standing still
        "click_bonus": 0.0,     # No click bonus yet
    },
    {   # Stage 2: Gathering (steps 5,000 - 20,000)
        "name": "Gathering",
        "multipliers": {"Resources": 1.0, "Minimap": 0.5, "Hotbar": 3.0, "World": 1.0},
        "noop_penalty": 0.01,
        "click_bonus": 0.005,   # Mining = sustained clicking
    },
    {   # Stage 3: Automation (steps 20,000+)
        "name": "Automation",
        "multipliers": {"Resources": 1.0, "Minimap": 0.5, "Hotbar": 1.5, "World": 3.0},
        "noop_penalty": 0.005,
        "click_bonus": 0.002,
    },
]


def get_stage(step):
    """Return the current curriculum stage index and config."""
    for i, threshold in enumerate(STAGE_THRESHOLDS):
        if step < threshold:
            return i, CURRICULUM[i]
    return len(CURRICULUM) - 1, CURRICULUM[-1]


class RepetitionTracker:
    """Detects unproductive action loops.

    Tracks consecutive repetitions of the same action and whether
    reward per repetition is increasing, flat, or declining.
    Penalises only unproductive loops (flat/declining reward).
    """

    def __init__(self, repeat_min=LOOP_REPEAT_MIN, penalty=LOOP_PENALTY):
        self.repeat_min = repeat_min
        self.penalty = penalty
        self.last_action = -1
        self.repeat_count = 0
        self.repeat_rewards = []  # Reward for each repetition in current streak

    def update(self, action_id, reward):
        """Track action and return penalty (0.0 or negative)."""
        if action_id == self.last_action:
            self.repeat_count += 1
            self.repeat_rewards.append(reward)
        else:
            self.last_action = action_id
            self.repeat_count = 1
            self.repeat_rewards = [reward]

        if self.repeat_count < self.repeat_min:
            return 0.0

        # We have 5+ consecutive same actions — check reward trend
        recent = self.repeat_rewards[-self.repeat_min:]
        first_half = sum(recent[:len(recent) // 2])
        second_half = sum(recent[len(recent) // 2:])

        if second_half >= first_half:
            # Reward is stable or increasing — productive repetition (mining etc)
            return 0.0
        else:
            # Reward is declining — unproductive loop
            return -self.penalty


def check_stuck(reward_history, window=STUCK_WINDOW, threshold=STUCK_THRESHOLD):
    """Detect if the agent is stuck by checking avg100 trend.
    Returns True if avg100 has been flat/declining over the window."""
    if len(reward_history) < window:
        return False
    recent = reward_history[-window:]
    # Compare first half avg to second half avg
    half = window // 2
    first_half = sum(recent[:half]) / half
    second_half = sum(recent[half:]) / half
    improvement = second_half - first_half
    return improvement < threshold


def apply_stage(reward_signal, stage_config, base_weights):
    """Apply a curriculum stage's multipliers to the reward signal."""
    for region in reward_signal.regions:
        base = base_weights[region.name]
        mult = stage_config["multipliers"].get(region.name, 1.0)
        region.weight = base * mult


# ---------------------------------------------------------------------------
# Curriculum: start a fresh Factorio game via UI automation
# ---------------------------------------------------------------------------

def start_fresh_game(ctrl):
    """Automate Factorio menus to start a new freeplay game.

    Navigates: Main Menu -> Play -> New Game -> Generate -> Play.
    Call this when Factorio is at the main menu. If already in-game,
    this is a no-op (set CURRICULUM_NEW_GAME = False to skip).
    """
    from pynput.keyboard import Key

    print("\n--- Curriculum: Starting fresh Factorio game ---")
    cx, cy = ctrl.width // 2, ctrl.height // 2

    # Press Escape first in case we're in-game — brings up menu
    ctrl.press_key(Key.esc, duration=0.1)
    time.sleep(1.0)

    # Click "Quit to title" if in-game menu (bottom area)
    ctrl.click(cx, int(ctrl.height * 0.72), "left")
    time.sleep(0.5)
    # Confirm quit if prompted
    ctrl.click(cx, int(ctrl.height * 0.52), "left")
    time.sleep(2.0)

    # Now at main menu — click "Play" (center-ish)
    ctrl.click(cx, int(ctrl.height * 0.35), "left")
    time.sleep(1.0)

    # Click "New Game"
    ctrl.click(cx, int(ctrl.height * 0.30), "left")
    time.sleep(1.0)

    # Click "Play" to start with default settings
    # The play button is typically bottom-right area
    ctrl.click(int(ctrl.width * 0.85), int(ctrl.height * 0.92), "left")
    time.sleep(1.0)

    # If there's a generate/confirm step, click through it
    ctrl.click(int(ctrl.width * 0.85), int(ctrl.height * 0.92), "left")
    time.sleep(5.0)  # Wait for world generation

    print("--- Fresh game started (or attempted) ---\n")


# ---------------------------------------------------------------------------
# Rollout buffer (on-policy, for PPO)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one rollout of experience for PPO updates."""

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
        """Compute GAE advantages and discounted returns."""
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

        returns = advantages + values
        return advantages, returns

    def get_batches(self, advantages, returns):
        """Yield minibatches for PPO update."""
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
            end = start + MINIBATCH_SIZE
            idx = indices[start:end]
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
# Action execution
# ---------------------------------------------------------------------------

def execute_action(ctrl, action_id):
    """Map network action ID to Factorio input via FactorioController."""
    cx, cy = ctrl.width // 2, ctrl.height // 2

    if action_id == 0:
        ctrl.hold_key("w", duration=0.08)
    elif action_id == 1:
        ctrl.hold_key("a", duration=0.08)
    elif action_id == 2:
        ctrl.hold_key("s", duration=0.08)
    elif action_id == 3:
        ctrl.hold_key("d", duration=0.08)
    elif action_id == 4:
        from pynput.keyboard import Key
        from control import keyboard
        keyboard.press(Key.shift)
        ctrl.hold_key("w", duration=0.08)
        keyboard.release(Key.shift)
    elif action_id == 5:
        from pynput.keyboard import Key
        from control import keyboard
        keyboard.press(Key.shift)
        ctrl.hold_key("a", duration=0.08)
        keyboard.release(Key.shift)
    elif action_id == 6:
        from pynput.keyboard import Key
        from control import keyboard
        keyboard.press(Key.shift)
        ctrl.hold_key("s", duration=0.08)
        keyboard.release(Key.shift)
    elif action_id == 7:
        from pynput.keyboard import Key
        from control import keyboard
        keyboard.press(Key.shift)
        ctrl.hold_key("d", duration=0.08)
        keyboard.release(Key.shift)
    elif action_id == 8:
        ctrl.click(cx, cy, "left")
    elif action_id == 9:
        ctrl.click(cx, cy, "right")
    elif action_id == 10:
        from control import mouse
        x, y = mouse.position
        mouse.position = (x, max(0, y - MOUSE_STEP))
    elif action_id == 11:
        from control import mouse
        x, y = mouse.position
        mouse.position = (x, y + MOUSE_STEP)
    elif action_id == 12:
        from control import mouse
        x, y = mouse.position
        mouse.position = (max(0, x - MOUSE_STEP), y)
    elif action_id == 13:
        from control import mouse
        x, y = mouse.position
        mouse.position = (x + MOUSE_STEP, y)
    elif action_id == 14:
        ctrl.press_key(" ", duration=0.05)
    elif action_id == 15:
        ctrl.press_key("e", duration=0.05)
    elif action_id == 16:
        ctrl.press_key("q", duration=0.05)
    elif action_id == 17:
        pass  # No-op


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(net, optimizer, rollout, device):
    """Run PPO update on collected rollout. Returns mean loss, entropy."""
    # Bootstrap value for last state
    has_strategy = rollout.strategies[0] is not None
    has_audio = rollout.audios[0] is not None
    with torch.no_grad():
        last_obs = torch.from_numpy(rollout.observations[-1]).unsqueeze(0).to(device)
        last_strat = None
        last_aud = None
        if has_strategy:
            last_strat = torch.from_numpy(rollout.strategies[-1]).unsqueeze(0).to(device)
        if has_audio:
            last_aud = torch.from_numpy(rollout.audios[-1]).unsqueeze(0).to(device)
        _, last_value = net(last_obs, strategy=last_strat, audio=last_aud)
        last_value = last_value.item()

    advantages, returns = rollout.compute_gae(last_value)

    # Normalise advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_batches = 0

    for epoch in range(PPO_EPOCHS):
        for batch in rollout.get_batches(advantages, returns):
            obs_t = torch.from_numpy(batch["obs"]).to(device)
            act_t = torch.from_numpy(batch["actions"]).to(device)
            old_lp_t = torch.from_numpy(batch["old_log_probs"]).to(device)
            adv_t = torch.from_numpy(batch["advantages"]).to(device)
            ret_t = torch.from_numpy(batch["returns"]).to(device)
            strat_t = None
            aud_t = None
            if "strategies" in batch:
                strat_t = torch.from_numpy(batch["strategies"]).to(device)
            if "audios" in batch:
                aud_t = torch.from_numpy(batch["audios"]).to(device)

            log_probs, entropy, values = net.evaluate(obs_t, act_t, strategy=strat_t, audio=aud_t)

            # Policy loss (clipped surrogate)
            ratio = torch.exp(log_probs - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, ret_t)

            # Entropy bonus
            entropy_mean = entropy.mean()

            # Combined loss
            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()
            num_batches += 1

    return {
        "policy_loss": total_policy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
        "entropy": total_entropy / num_batches,
        "total_loss": (total_policy_loss + total_value_loss) / num_batches,
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(net, optimizer, step, reward_total, path):
    torch.save({
        "step": step,
        "reward_total": reward_total,
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"  Checkpoint saved: {path} (step {step})")


def load_latest_checkpoint(net, optimizer, device, checkpoint_dir=CHECKPOINT_DIR):
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        print("No checkpoint found, starting fresh.")
        return 0, 0.0
    latest = checkpoints[-1]
    data = torch.load(latest, map_location=device, weights_only=True)
    net.load_state_dict(data["model_state"])
    optimizer.load_state_dict(data["optimizer_state"])
    step = data["step"]
    reward_total = data.get("reward_total", 0.0)
    print(f"Resumed from {latest} (step {step}, reward {reward_total:.2f})")
    return step, reward_total


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  FACTORIO RL AGENT — PPO TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Init components ---
    print("\nInitialising components...")
    bbox = find_factorio_window()
    if bbox is None:
        print("ERROR: Factorio window not found! Start the game first.")
        return

    win_w, win_h = bbox["width"], bbox["height"]
    print(f"Factorio: {win_w}x{win_h}")

    ctrl = FactorioController()
    obs_proc = ObservationProcessor(size=128, stack_size=4)
    reward_signal = RewardSignal(win_w, win_h)

    # Capture base weights before curriculum modifies them
    base_weights = {r.name: r.weight for r in reward_signal.regions}

    # --- Experiment mode ---
    baseline_mode = "--baseline" in os.sys.argv
    experiment_tag = "baseline" if baseline_mode else "knowledge"
    log_file = Path(f"logs/{experiment_tag}.csv")
    log_file.parent.mkdir(exist_ok=True)

    # --- Audio system ---
    print("\nStarting audio capture...")
    audio_capture = AudioCapture()
    audio_ok = audio_capture.start()
    audio_proc = AudioProcessor(device=str(device)) if audio_ok else None
    audio_dim = AUDIO_FEATURE_DIM if audio_ok else 0
    if not audio_ok:
        print("  Audio disabled — running without audio features")

    if baseline_mode:
        print("\n*** BASELINE MODE — pure CNN, no knowledge vector ***")
        knowledge = None
        strategy_vec = None
        strategy_t = None
        net = ActorCritic(strategy_dim=0, audio_dim=audio_dim).to(device)
    else:
        print("\n*** KNOWLEDGE MODE — CNN + strategy + audio ***")
        knowledge = KnowledgeBase()
        knowledge.build()
        net = ActorCritic(strategy_dim=EMBEDDING_DIM, audio_dim=audio_dim).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    replay = ReplayBuffer(capacity=10_000)
    rollout = RolloutBuffer()

    # Experiment-specific checkpoints
    exp_checkpoint_dir = CHECKPOINT_DIR / experiment_tag
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    global_step, cumulative_reward = load_latest_checkpoint(
        net, optimizer, device, exp_checkpoint_dir)

    # Screen capture
    sct = mss.mss()
    monitor = bbox

    # --- Vision-language advisor + translator ---
    advisor = None
    translator = None
    if not baseline_mode:
        advisor = Advisor(knowledge, sct, monitor, device)
        advisor.start()
        translator = ActionTranslator(
            ctrl=ctrl, obs_proc=obs_proc, reward_signal=reward_signal,
            replay=replay, rollout=rollout, monitor=monitor,
            strategy_vec=None, sct=sct,
            audio_proc=audio_proc, audio_capture=audio_capture,
        )
        translator.verify_api()

    # --- Timing ---
    step_interval = 1.0 / DECISIONS_PER_SEC

    print(f"\nConfig: {STEPS_PER_UPDATE} steps/update, {PPO_EPOCHS} epochs, "
          f"lr={LR}, clip={CLIP_EPS}, frame_skip={FRAME_SKIP}, "
          f"{DECISIONS_PER_SEC} decisions/sec")

    # --- Curriculum: optionally start a fresh game ---
    if "--fresh" in os.sys.argv:
        print("\nStarting in 3 seconds — will create fresh game...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        start_fresh_game(ctrl)
    else:
        print(f"\nStarting in 3 seconds — switch to Factorio! (use --fresh for new game)")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

    # Start safety systems
    _start_kill_switch()
    print("\nTraining started. F12 = kill switch. Pauses when Factorio loses focus.\n")

    paused = False
    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0
    update_count = 0
    last_loss_info = {}
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    reward_history = []  # Rolling window of per-update rewards
    current_stage_idx = -1  # Force initial stage print
    last_refresh_step = 0   # Track last knowledge refresh
    rep_tracker = RepetitionTracker()

    try:
        while not _kill_flag.is_set():
            step_start = time.time()

            # 0. Focus guard — pause when Factorio isn't focused
            if not _is_factorio_focused():
                if not paused:
                    print("  [PAUSED] Factorio lost focus — waiting...")
                    paused = True
                time.sleep(0.2)
                continue
            if paused:
                print("  [RESUMED] Factorio focused")
                paused = False

            # 0.5 Curriculum stage check + strategy update
            stage_idx, stage = get_stage(global_step)
            if stage_idx != current_stage_idx:
                current_stage_idx = stage_idx
                apply_stage(reward_signal, stage, base_weights)
                if knowledge is not None:
                    strategy_vec = knowledge.get_strategy_vector(stage["name"])
                    strategy_t = torch.from_numpy(strategy_vec).unsqueeze(0).to(device)
                print(f"\n{'='*60}")
                print(f"  STAGE {stage_idx + 1}: {stage['name'].upper()} (step {global_step:,}+)")
                print(f"  Weights: {stage['multipliers']}")
                print(f"  No-op penalty: {stage['noop_penalty']}  Click bonus: {stage['click_bonus']}")
                if knowledge is not None:
                    print(f"  Strategy vector updated (knowledge-conditioned)")
                print(f"{'='*60}\n")

            # 1. Capture and preprocess
            img = sct.grab(monitor)
            frame = np.array(img)[:, :, :3]
            obs_proc.push(frame)
            obs = obs_proc.get()  # (4, 128, 128)

            # 1.5 Check for advisor update + run demonstration
            if advisor is not None:
                advisor_vec = advisor.get_strategy()
                if advisor_vec is not None:
                    strategy_vec = advisor_vec
                    strategy_t = torch.from_numpy(strategy_vec).unsqueeze(0).to(device)

                # Check if new advice arrived — trigger demonstration
                new_advice = advisor.consume_advice()
                if new_advice is not None and translator is not None:
                    translator.strategy_vec = strategy_vec
                    demo_steps, demo_reward = translator.translate_and_execute(
                        new_advice, sct)
                    global_step += demo_steps
                    episode_reward += demo_reward

            # 1.7 Process audio
            audio_vec = None
            audio_t = None
            audio_reward = 0.0
            if audio_proc is not None:
                audio_features, audio_events = audio_proc.process(audio_capture)
                audio_vec = audio_features
                audio_t = torch.from_numpy(audio_features).unsqueeze(0).to(device)
                audio_reward = audio_events["reward_adjustment"]
                if audio_events["mining"]:
                    pass  # Reward added via adjustment, no log spam
                if audio_events["attack_warning"]:
                    print(f"  [AUDIO] Attack warning detected! ({audio_reward:+.2f})")
                if audio_events["entity_placed"]:
                    print(f"  [AUDIO] Entity placement sound (+0.1)")

            # 2. Get action from network (CNN + strategy + audio)
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                action, log_prob, value = net.get_action(obs_t, strategy=strategy_t, audio=audio_t)
                action_id = action.item()
                log_prob_val = log_prob.item()
                value_val = value.item()

            # 3. Execute action with frame skip — repeat for N frames
            total_reward = 0.0
            for _skip in range(FRAME_SKIP):
                if _kill_flag.is_set():
                    break
                execute_action(ctrl, action_id)
                # Capture after each repeated action
                img2 = sct.grab(monitor)
                skip_frame = np.array(img2)[:, :, :3]
                r, r_details = reward_signal.compute(skip_frame)
                total_reward += r
                if r_details.get("inventory_gain"):
                    print(f"  [REWARD] MINING SUCCESS — inventory gained! (+2.0)")
            action_counts[action_id] += 1

            # 3.4 Audio reward
            total_reward += audio_reward

            # 3.5 Curriculum action shaping
            if action_id == 17:  # No-op penalty
                total_reward -= stage["noop_penalty"]
            if action_id == 8:   # Left click bonus (mining)
                total_reward += stage["click_bonus"]

            # 3.6 Unproductive loop detection
            loop_penalty = rep_tracker.update(action_id, total_reward)
            total_reward += loop_penalty

            # 4. Use last frame from skip sequence as next observation
            obs_proc.push(skip_frame)
            next_obs = obs_proc.get()

            done = False  # Factorio doesn't have episodes; continuous

            # 5. Store experience (accumulated reward across skipped frames)
            rollout.push(obs, action_id, log_prob_val, total_reward, value_val, done,
                         strategy=strategy_vec if knowledge else None,
                         audio=audio_vec)
            replay.push(obs, action_id, total_reward, next_obs, done)

            episode_reward += total_reward
            episode_steps += 1
            global_step += 1

            # 6. PPO update
            if len(rollout) >= STEPS_PER_UPDATE:
                loss_info = ppo_update(net, optimizer, rollout, device)
                update_count += 1
                last_loss_info = loss_info
                rollout.clear()

                # Track rolling reward
                reward_history.append(episode_reward)
                if len(reward_history) > 100:
                    reward_history.pop(0)
                rolling_avg = sum(reward_history) / len(reward_history)

                # Log
                replay_stats = replay.stats()
                top_actions = np.argsort(action_counts)[::-1][:5]
                action_dist = "  ".join(f"{ACTION_NAMES[a]}:{action_counts[a]}" for a in top_actions)

                print(f"[Step {global_step:>7d} | Update {update_count:>3d} | {stage['name']}] "
                      f"reward={episode_reward:+.3f}  "
                      f"avg100={rolling_avg:+.3f}  "
                      f"p_loss={loss_info['policy_loss']:.4f}  "
                      f"v_loss={loss_info['value_loss']:.4f}  "
                      f"entropy={loss_info['entropy']:.3f}  "
                      f"buf={replay_stats['size']:,}")
                print(f"  Top actions: {action_dist}")

                # CSV log for experiment comparison
                if update_count == 1:
                    log_file.write_text("step,update,reward,avg100,p_loss,v_loss,entropy,stage\n")
                with open(log_file, "a") as f:
                    f.write(f"{global_step},{update_count},{episode_reward:.4f},"
                            f"{rolling_avg:.4f},{loss_info['policy_loss']:.4f},"
                            f"{loss_info['value_loss']:.4f},{loss_info['entropy']:.3f},"
                            f"{stage['name']}\n")

                episode_reward = 0.0
                episode_steps = 0

                # 6.5 Stagnation detection — refresh knowledge if stuck
                if (knowledge is not None
                        and check_stuck(reward_history)
                        and global_step - last_refresh_step > STUCK_WINDOW):
                    print(f"\n{'!'*60}")
                    print(f"  STUCK DETECTED at step {global_step:,}")
                    print(f"  avg100 flat/declining over last {STUCK_WINDOW} updates")
                    print(f"  Refreshing knowledge base...")
                    print(f"{'!'*60}")
                    knowledge.refresh(stage["name"])
                    strategy_vec = knowledge.get_strategy_vector(stage["name"])
                    strategy_t = torch.from_numpy(strategy_vec).unsqueeze(0).to(device)
                    last_refresh_step = global_step
                    print(f"  Knowledge refreshed. Resuming training.\n")

            # 7. Checkpoint + attention map
            if global_step % CHECKPOINT_INTERVAL == 0 and global_step > 0:
                save_checkpoint(
                    net, optimizer, global_step, cumulative_reward + episode_reward,
                    exp_checkpoint_dir / f"step_{global_step:07d}.pt",
                )
                # Save attention heatmap
                attn_map = net.get_attention_map()
                if attn_map is not None:
                    attn_dir = Path("attention_maps")
                    attn_dir.mkdir(exist_ok=True)
                    # Normalize and save as heatmap
                    attn_vis = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                    attn_vis = (attn_vis * 255).astype(np.uint8)
                    attn_vis = cv2.resize(attn_vis, (256, 256), interpolation=cv2.INTER_NEAREST)
                    attn_colored = cv2.applyColorMap(attn_vis, cv2.COLORMAP_JET)
                    cv2.imwrite(str(attn_dir / f"attn_step_{global_step:07d}.png"), attn_colored)
                    print(f"  Attention map saved: {attn_dir / f'attn_step_{global_step:07d}.png'}")

            # 8. Pace control
            elapsed = time.time() - step_start
            sleep_time = step_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n\nTraining stopped at step {global_step}.")
        save_checkpoint(
            net, optimizer, global_step, cumulative_reward + episode_reward,
            CHECKPOINT_DIR / f"step_{global_step:07d}.pt",
        )
        stats = replay.stats()
        print(f"\nFinal stats:")
        print(f"  Steps:          {global_step:,}")
        print(f"  Updates:        {update_count}")
        print(f"  Buffer:         {stats['size']:,} / {stats['capacity']:,}")
        print(f"  Mean reward:    {stats['mean_reward']:+.4f}")
        print(f"  Recent 100 avg: {stats['recent_100_reward']:+.4f}")
        if last_loss_info:
            print(f"  Last p_loss:    {last_loss_info['policy_loss']:.4f}")
            print(f"  Last entropy:   {last_loss_info['entropy']:.3f}")
        print("\nDone.")
        if advisor is not None:
            advisor.stop()
        sct.close()


if __name__ == "__main__":
    main()
