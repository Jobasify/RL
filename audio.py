"""
Audio processing pipeline for the Factorio RL agent.
Converts game audio into mel spectrograms, extracts features via a small CNN,
and detects specific audio events for bonus rewards.

The agent hears the game — mining sounds confirm productive action,
attack warnings signal threat before it's visible on screen.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AUDIO_FEATURE_DIM = 128
SAMPLE_RATE = 48000
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024


# ---------------------------------------------------------------------------
# Mel spectrogram (numpy, no librosa dependency at import time)
# ---------------------------------------------------------------------------

def audio_to_mel(audio_np, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
                 hop_length=HOP_LENGTH):
    """Convert audio numpy array to a mel spectrogram.

    Args:
        audio_np: (samples,) or (samples, channels) float32 array
        sr: sample rate

    Returns:
        mel_db: (n_mels, time_frames) float32 in dB scale
    """
    import librosa

    # Convert to mono if stereo
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)

    # Ensure float32
    audio_np = audio_np.astype(np.float32)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio_np, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, power=2.0,
    )

    # Convert to dB scale, normalise to roughly [0, 1]
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db + 80.0) / 80.0  # Shift from [-80, 0] to [0, 1]
    mel_db = np.clip(mel_db, 0.0, 1.0)

    return mel_db.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio feature CNN
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """Small 2-layer CNN that processes mel spectrograms into a feature vector."""

    def __init__(self, n_mels=N_MELS, output_dim=AUDIO_FEATURE_DIM):
        super().__init__()
        # Input: (B, 1, n_mels, time_frames)
        # We'll fix time_frames to 128 by padding/cropping
        self.target_frames = 128

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # -> (B, 16, 32, 64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # -> (B, 32, 16, 32)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # -> (B, 32, 4, 4)
        self.fc = nn.Linear(32 * 4 * 4, output_dim)

        # Init
        for layer in [self.conv1, self.conv2, self.fc]:
            if hasattr(layer, 'weight'):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(layer.bias)

    def forward(self, mel):
        """Process mel spectrogram to feature vector.

        Args:
            mel: (B, 1, n_mels, time_frames) tensor

        Returns:
            features: (B, output_dim) tensor
        """
        x = F.relu(self.conv1(mel))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


# ---------------------------------------------------------------------------
# Audio event detection
# ---------------------------------------------------------------------------

class AudioEventDetector:
    """Detects specific Factorio audio events from mel spectrograms.

    Uses simple energy-band analysis — no training required.
    Factorio audio cues have distinctive frequency signatures:
      - Mining: sustained mid-frequency rhythmic pattern
      - Attack warning: sharp high-frequency burst
      - Entity placement: brief mid-high click/thud
    """

    def __init__(self):
        self.prev_energy = None
        self.mining_streak = 0

    def detect(self, mel_db):
        """Analyse mel spectrogram for audio events.

        Args:
            mel_db: (n_mels, time_frames) normalised mel spectrogram

        Returns:
            dict with event flags and reward adjustments
        """
        events = {
            "mining": False,
            "attack_warning": False,
            "entity_placed": False,
            "reward_adjustment": 0.0,
        }

        if mel_db.shape[1] < 4:
            return events

        # Energy in frequency bands
        n_mels = mel_db.shape[0]
        low = mel_db[:n_mels // 3, :].mean()       # Low frequencies
        mid = mel_db[n_mels // 3:2 * n_mels // 3, :].mean()  # Mid
        high = mel_db[2 * n_mels // 3:, :].mean()   # High

        total_energy = mel_db.mean()

        # Temporal variation (how much the signal changes over time)
        temporal_var = mel_db.std(axis=1).mean()

        # --- Mining detection ---
        # Sustained mid-frequency energy with rhythmic variation
        if mid > 0.3 and temporal_var > 0.05 and temporal_var < 0.3:
            self.mining_streak += 1
            if self.mining_streak >= 3:  # ~0.6 seconds sustained
                events["mining"] = True
                events["reward_adjustment"] += 0.05
        else:
            self.mining_streak = 0

        # --- Attack warning detection ---
        # Sharp high-frequency spike with high overall energy
        if self.prev_energy is not None:
            energy_delta = total_energy - self.prev_energy
            if high > 0.4 and energy_delta > 0.15:
                events["attack_warning"] = True
                events["reward_adjustment"] -= 0.2

        # --- Entity placement detection ---
        # Brief mid-high energy burst followed by quiet
        if mid > 0.25 and high > 0.2 and temporal_var > 0.2:
            if self.prev_energy is not None and total_energy - self.prev_energy > 0.08:
                events["entity_placed"] = True
                events["reward_adjustment"] += 0.1

        self.prev_energy = total_energy
        return events


# ---------------------------------------------------------------------------
# Combined audio processor
# ---------------------------------------------------------------------------

class AudioProcessor:
    """Full audio pipeline: capture buffer -> mel -> CNN features + event detection."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.encoder = AudioEncoder().to(self.device)
        self.detector = AudioEventDetector()
        self._last_features = torch.zeros(1, AUDIO_FEATURE_DIM)

    def process(self, audio_capture):
        """Process recent audio from AudioCapture instance.

        Args:
            audio_capture: AudioCapture instance from capture.py

        Returns:
            features: (AUDIO_FEATURE_DIM,) numpy array
            events: dict from AudioEventDetector
        """
        audio_np = audio_capture.get_recent_audio(seconds=2)

        if audio_np is None:
            return (np.zeros(AUDIO_FEATURE_DIM, dtype=np.float32),
                    {"mining": False, "attack_warning": False,
                     "entity_placed": False, "reward_adjustment": 0.0})

        # Convert to mel spectrogram
        mel = audio_to_mel(audio_np)

        # Event detection (on numpy)
        events = self.detector.detect(mel)

        # Pad/crop time dimension to fixed size
        target = self.encoder.target_frames
        if mel.shape[1] < target:
            mel = np.pad(mel, ((0, 0), (0, target - mel.shape[1])))
        else:
            mel = mel[:, :target]

        # CNN feature extraction
        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.encoder(mel_t).squeeze(0).cpu().numpy()

        return features, events

    @property
    def feature_dim(self):
        return AUDIO_FEATURE_DIM


def main():
    print("=== Audio Pipeline Test ===\n")

    # Test with synthetic audio
    print("Testing mel spectrogram conversion...")
    fake_audio = np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.1
    mel = audio_to_mel(fake_audio)
    print(f"  Audio: {fake_audio.shape} -> Mel: {mel.shape}")
    print(f"  Mel range: [{mel.min():.3f}, {mel.max():.3f}]")

    print("\nTesting AudioEncoder CNN...")
    encoder = AudioEncoder()
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {total_params:,}")

    # Pad to target frames
    target = encoder.target_frames
    if mel.shape[1] < target:
        mel_padded = np.pad(mel, ((0, 0), (0, target - mel.shape[1])))
    else:
        mel_padded = mel[:, :target]
    mel_t = torch.from_numpy(mel_padded).unsqueeze(0).unsqueeze(0)
    features = encoder(mel_t)
    print(f"  Input: {tuple(mel_t.shape)} -> Features: {tuple(features.shape)}")

    print("\nTesting AudioEventDetector...")
    detector = AudioEventDetector()
    events = detector.detect(mel)
    print(f"  Events: {events}")

    print("\nTesting full AudioProcessor...")
    proc = AudioProcessor()
    # Simulate with fake capture
    class FakeCapture:
        def get_recent_audio(self, seconds=2):
            return np.random.randn(SAMPLE_RATE * seconds, 2).astype(np.float32) * 0.1
    feats, evts = proc.process(FakeCapture())
    print(f"  Features: shape={feats.shape}, range=[{feats.min():.3f}, {feats.max():.3f}]")
    print(f"  Events: {evts}")

    print(f"\nAudio feature dim: {AUDIO_FEATURE_DIM}")
    print("Done.")


if __name__ == "__main__":
    main()
