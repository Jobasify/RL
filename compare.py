"""
Compare baseline vs knowledge experiment runs.
Reads CSV logs from logs/baseline.csv and logs/knowledge.csv,
plots avg100 reward curves side by side.
"""

import sys
from pathlib import Path

import numpy as np

LOG_DIR = Path("logs")


def load_log(path):
    """Load a CSV log file into a dict of numpy arrays."""
    if not path.exists():
        return None
    lines = path.read_text().strip().split("\n")
    header = lines[0].split(",")
    data = {h: [] for h in header}
    for line in lines[1:]:
        values = line.split(",")
        for h, v in zip(header, values):
            try:
                data[h].append(float(v))
            except ValueError:
                data[h].append(v)
    # Convert numeric columns
    for h in ["step", "update", "reward", "avg100", "p_loss", "v_loss", "entropy"]:
        if h in data:
            data[h] = np.array(data[h], dtype=np.float64)
    return data


def print_comparison(baseline, knowledge):
    """Print a text comparison of two runs."""
    print("=" * 70)
    print("  EXPERIMENT COMPARISON: Baseline (CNN) vs Knowledge (CNN + Strategy)")
    print("=" * 70)

    for name, data in [("Baseline", baseline), ("Knowledge", knowledge)]:
        if data is None:
            print(f"\n  {name}: No data (run with {'--baseline' if name == 'Baseline' else ''})")
            continue

        n = len(data["update"])
        print(f"\n  {name}:")
        print(f"    Updates:        {int(n)}")
        print(f"    Steps:          {int(data['step'][-1]):,}")
        print(f"    Final avg100:   {data['avg100'][-1]:+.3f}")
        print(f"    Peak avg100:    {data['avg100'].max():+.3f} (update {int(data['update'][data['avg100'].argmax()])})")
        print(f"    Final entropy:  {data['entropy'][-1]:.3f}")
        print(f"    Mean reward:    {data['reward'].mean():+.3f}")

    if baseline is not None and knowledge is not None:
        # Compare at matching update counts
        min_updates = min(len(baseline["update"]), len(knowledge["update"]))
        if min_updates >= 10:
            b_avg = baseline["avg100"][:min_updates]
            k_avg = knowledge["avg100"][:min_updates]
            diff = k_avg - b_avg

            print(f"\n  Head-to-head ({min_updates} updates compared):")
            print(f"    Knowledge avg100 - Baseline avg100:")
            print(f"      Mean difference:  {diff.mean():+.4f}")
            print(f"      Final difference: {diff[-1]:+.4f}")
            wins = (diff > 0).sum()
            print(f"      Knowledge ahead:  {wins}/{min_updates} updates ({100*wins/min_updates:.0f}%)")

            if diff[-1] > 0.01:
                print(f"\n    >> Knowledge vector appears to HELP (+{diff[-1]:.3f} at final update)")
            elif diff[-1] < -0.01:
                print(f"\n    >> Knowledge vector appears to HURT ({diff[-1]:.3f} at final update)")
            else:
                print(f"\n    >> No clear difference yet. Need more training data.")

    # ASCII sparkline of avg100 trend
    for name, data in [("Baseline", baseline), ("Knowledge", knowledge)]:
        if data is None or len(data["avg100"]) < 5:
            continue
        avg = data["avg100"]
        lo, hi = avg.min(), avg.max()
        rng = hi - lo if hi > lo else 1.0
        bars = "▁▂▃▄▅▆▇█"
        spark = ""
        # Sample ~60 points
        step = max(1, len(avg) // 60)
        for i in range(0, len(avg), step):
            idx = int((avg[i] - lo) / rng * (len(bars) - 1))
            spark += bars[min(idx, len(bars) - 1)]
        print(f"\n  {name} avg100 trend: {spark}")

    print()


def main():
    baseline = load_log(LOG_DIR / "baseline.csv")
    knowledge = load_log(LOG_DIR / "knowledge.csv")

    if baseline is None and knowledge is None:
        print("No experiment logs found. Run training first:")
        print("  python train.py --baseline    # Run 1: pure CNN")
        print("  python train.py               # Run 2: CNN + knowledge")
        return

    print_comparison(baseline, knowledge)


if __name__ == "__main__":
    main()
