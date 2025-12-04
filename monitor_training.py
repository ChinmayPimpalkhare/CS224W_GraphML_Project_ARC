#!/usr/bin/env python3
"""
Monitor GraphFlix training progress in real-time.
Shows loss curves, beta evolution, and estimates completion time.
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def monitor_training(run_dir):
    """Monitor training progress from history file."""
    history_file = Path(run_dir) / "history.json"

    if not history_file.exists():
        print(f"❌ History file not found: {history_file}")
        print("Training may not have started yet or run_dir is incorrect.")
        return

    print("=" * 80)
    print("📊 GraphFlix Training Monitor")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Press Ctrl+C to stop monitoring")
    print("=" * 80)
    print()

    last_size = 0
    last_epoch = 0

    try:
        while True:
            # Check if file has been updated
            current_size = history_file.stat().st_size

            if current_size != last_size:
                with open(history_file) as f:
                    data = json.load(f)

                train_history = data.get("train", [])
                val_history = data.get("val", [])

                if not train_history:
                    print("⏳ Waiting for training data...")
                    time.sleep(5)
                    continue

                latest_epoch = train_history[-1]
                epoch_num = latest_epoch["epoch"]

                # Only print update if we have a new epoch
                if epoch_num > last_epoch:
                    last_epoch = epoch_num

                    # Clear screen (optional)
                    # print("\033[2J\033[H")

                    print(f"\n{'='*80}")
                    print(
                        f"📈 Epoch {epoch_num}/{data.get('config', {}).get('epochs', 50)}"
                    )
                    print(f"{'='*80}")

                    # Training metrics
                    print(f"🔥 Training:")
                    print(f"   Loss: {latest_epoch['loss']:.4f}")
                    print(f"   Beta: {latest_epoch.get('beta', 1.0):.4f}")
                    print(f"   Time: {format_time(latest_epoch.get('epoch_time', 0))}")

                    # Validation metrics (if available)
                    if val_history:
                        latest_val = val_history[-1]
                        if latest_val["epoch"] == epoch_num:
                            print(f"\n✅ Validation:")
                            print(f"   Loss: {latest_val['loss']:.4f}")
                            if "recall@10" in latest_val:
                                print(f"   Recall@10: {latest_val['recall@10']:.4f}")
                            if "ndcg@10" in latest_val:
                                print(f"   NDCG@10: {latest_val['ndcg@10']:.4f}")
                            if "mrr" in latest_val:
                                print(f"   MRR: {latest_val['mrr']:.4f}")

                    # Progress and time estimates
                    total_epochs = data.get("config", {}).get("epochs", 50)
                    progress = (epoch_num / total_epochs) * 100

                    if len(train_history) > 1:
                        avg_epoch_time = sum(
                            e.get("epoch_time", 0) for e in train_history
                        ) / len(train_history)
                        remaining_epochs = total_epochs - epoch_num
                        eta_seconds = avg_epoch_time * remaining_epochs

                        print(f"\n⏱️  Progress:")
                        print(f"   {progress:.1f}% complete")
                        print(f"   Avg epoch time: {format_time(avg_epoch_time)}")
                        print(f"   ETA: {format_time(eta_seconds)}")

                    # Loss trend
                    if len(train_history) >= 5:
                        recent_losses = [e["loss"] for e in train_history[-5:]]
                        trend = (
                            "📉 Decreasing"
                            if recent_losses[-1] < recent_losses[0]
                            else "📈 Increasing"
                        )
                        print(f"\n📊 Loss Trend (last 5 epochs): {trend}")
                        print(f"   {recent_losses[0]:.4f} → {recent_losses[-1]:.4f}")

                    print("=" * 80)

                last_size = current_size

            # Wait before checking again
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped.")
        print("Training is still running in the background.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def find_latest_run(runs_dir="runs"):
    """Find the most recent training run."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None

    run_dirs = [
        d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith("graphflix_")
    ]
    if not run_dirs:
        return None

    # Sort by modification time
    latest = max(run_dirs, key=lambda d: d.stat().st_mtime)
    return latest


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # Auto-detect latest run
        run_dir = find_latest_run()
        if run_dir is None:
            print("❌ No training runs found in 'runs/' directory")
            print("\nUsage: python monitor_training.py [run_directory]")
            print("Example: python monitor_training.py runs/graphflix_20251128_123456")
            sys.exit(1)
        print(f"🔍 Auto-detected latest run: {run_dir}")
        print()

    monitor_training(run_dir)
