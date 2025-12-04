#!/usr/bin/env python3
"""
Evaluate trained GraphFlix model on test set.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.graphflix.data.graphflix_dataloader import load_graph_and_create_dataloader
from src.graphflix.evaluation.metrics import evaluate_model, print_metrics
from src.graphflix.models.graphflix import GraphFlix


def evaluate_graphflix(
    checkpoint_path,
    data_dir="data/processed/ml1m",
    batch_size=32,
    k_hops=2,
    split="test",
):
    """
    Evaluate GraphFlix model on test set.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to processed data
        batch_size: Batch size for evaluation
        k_hops: K-hop subgraph sampling
        split: 'val' or 'test'
    """

    # Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("=" * 80)
    print(f"GraphFlix Evaluation on {split.upper()} set")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get hyperparameters
    if "hyperparams" in checkpoint:
        hyperparams = checkpoint["hyperparams"]
        print("Hyperparameters from checkpoint:")
        for k, v in hyperparams.items():
            print(f"  {k}: {v}")
    else:
        # Default hyperparams
        hyperparams = {"dim": 64, "heads": 4, "num_layers": 3, "beta_init": 1.0}
        print("Using default hyperparameters")

    # Get the data_dir used during training from checkpoint
    training_data_dir = None
    if "config" in checkpoint:
        config = checkpoint["config"]
        training_data_dir = config.get("data_dir", None)

    # Try to infer training dataset from model state dict
    if training_data_dir is None and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if "P" in state_dict:
            num_users_in_checkpoint = state_dict["P"].shape[0]
            print(
                f"\nCheckpoint contains precomputed data for {num_users_in_checkpoint} users"
            )

            # Infer dataset based on number of users
            if num_users_in_checkpoint == 6040:
                training_data_dir = "data/processed/ml1m"
                print(f"  -> Detected training dataset: ml1m (full dataset)")
            elif num_users_in_checkpoint == 604:
                training_data_dir = "data/processed/ml1m_10pct"
                print(f"  -> Detected training dataset: ml1m_10pct (10% subsample)")
            else:
                print(f"  -> Unknown dataset with {num_users_in_checkpoint} users")

    if training_data_dir:
        print(f"\nCheckpoint was trained on: {training_data_dir}")
        if training_data_dir != data_dir:
            print(f"\n⚠️  WARNING: Dataset mismatch detected!")
            print(f"   Training dataset: {training_data_dir}")
            print(f"   Evaluation dataset: {data_dir}")
            print(f"\n   The model will use precomputed data from the checkpoint.")
            print(f"   Evaluation will use graph/interactions from: {data_dir}")
            print(f"   This may cause errors if user/movie IDs don't overlap!\n")
    else:
        print(f"\nWARNING: Checkpoint doesn't contain data_dir info.")
        print(f"Using evaluation data from: {data_dir}")

    # Create dataloader (use specified data_dir for evaluation data)
    print(f"\nCreating {split} dataloader from: {data_dir}")
    test_loader, graph_data, node_type_map = load_graph_and_create_dataloader(
        data_dir=data_dir, batch_size=batch_size, k_hops=k_hops, split=split, seed=42
    )
    print(f"  {split.capitalize()} batches: {len(test_loader)}")

    # Create model WITHOUT loading precomputed data (will come from checkpoint)
    print(f"\nCreating model (precomputed data will be loaded from checkpoint)")
    model = GraphFlix(
        dim=hyperparams["dim"],
        heads=hyperparams["heads"],
        num_layers=hyperparams["num_layers"],
        beta_init=hyperparams.get("beta_init", 1.0),
        d_phi=192,
        precomputed_data_path=None,  # Don't load from disk - use checkpoint data
    ).to(device)

    # Load model weights (includes P and Phi matrices from training)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Verify loaded data dimensions
    if hasattr(model, "P") and model.P is not None:
        print(f"  Loaded user profiles from checkpoint: {model.P.shape}")
    if hasattr(model, "Phi") and model.Phi is not None:
        print(f"  Loaded movie metadata from checkpoint: {model.Phi.shape}")

    print(f"  Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Beta: {model.beta.item():.4f}")

    # Evaluate
    print(f"\nEvaluating on {split} set...")
    print("-" * 80)

    metrics = evaluate_model(
        model=model, dataloader=test_loader, k_list=[10, 20, 50], device=device
    )

    # Print results
    print()
    print_metrics(metrics, prefix=f"{split.upper()} Set Results")

    # Save results
    results_path = Path(checkpoint_path).parent / f"{split}_results.txt"
    with open(results_path, "w") as f:
        f.write(f"GraphFlix {split.upper()} Set Evaluation\n")
        f.write("=" * 80 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', 'unknown')}\n")
        f.write("\nMetrics:\n")
        for k, v in sorted(metrics.items()):
            f.write(f"  {k}: {v:.4f}\n")

    print(f"\nResults saved to: {results_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GraphFlix model")

    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/ml1m_10pct",
        help="Path to processed data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument("--k_hops", type=int, default=2, help="K-hop subgraph sampling")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on",
    )

    args = parser.parse_args()

    evaluate_graphflix(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        k_hops=args.k_hops,
        split=args.split,
    )
