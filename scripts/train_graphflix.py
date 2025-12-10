#!/usr/bin/env python3
"""
GraphFlix Training Script

Implements the full training loop with:
- BPR loss optimization
- Learning rate scheduling
- Gradient clipping
- Checkpointing
- Logging (loss, metrics, beta trajectory)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from src.graphflix.data.graphflix_dataloader import load_graph_and_create_dataloader
from src.graphflix.evaluation.metrics import evaluate_model
from src.graphflix.models.graphflix import GraphFlix
from src.graphflix.training.bpr_loss import bpr_loss


class Trainer:
    """GraphFlix model trainer."""

    def __init__(self, config, save_dir="runs"):
        """
        Args:
            config: Configuration dict
            save_dir: Directory to save checkpoints and logs
        """
        self.config = config
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Create save directory with timestamp
        # If save_dir already contains a timestamp pattern (from shell script), use it directly
        save_dir_path = Path(save_dir)
        if save_dir_path.name.startswith("graphflix_") and any(c.isdigit() for c in save_dir_path.name):
            # Already has timestamp from shell script, use it directly
            self.save_dir = save_dir_path
        else:
            # No timestamp, add one
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = save_dir_path / f"graphflix_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging FIRST before any output
        self.log_file = self.save_dir / "training.log"
        self._setup_logging()

        self.log(f"Device: {self.device}")
        self.log(f"Save directory: {self.save_dir}")
        self.log(f"Training logs: {self.log_file}")

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))
        self.log(f"TensorBoard logs: {self.save_dir / 'tensorboard'}")

        # Save config
        with open(self.save_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        # Initialize model
        self.model = self._build_model()

        # Initialize optimizer
        self.optimizer = self._build_optimizer()

        # Initialize scheduler
        self.scheduler = self._build_scheduler()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = -float("inf")

        # Logging
        self.train_history = []
        self.val_history = []

    def _setup_logging(self):
        """Setup file and console logging."""
        import logging
        
        # Create logger
        self.logger = logging.getLogger('GraphFlixTrainer')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler - logs everything to file
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler - also print to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False

    def log(self, message, level='info'):
        """Log a message to both file and console."""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'debug':
            self.logger.debug(message)

    def _build_model(self):
        """Build GraphFlix model."""
        model_config = self.config["params"]
        data_config = self.config.get("data", {})

        # Convert numeric parameters to proper types
        model = GraphFlix(
            dim=int(model_config["dim"]),
            heads=int(model_config["heads"]),
            num_layers=int(model_config["num_layers"]),
            num_node_types=int(model_config.get("num_node_types", 5)),
            beta_init=float(model_config["beta_init"]),
            d_phi=int(self.config["profile"]["d_phi"]),
            max_path_distance=int(model_config.get("max_path_distance", 5)),
            dropout=float(model_config.get("dropout", 0.1)),
            user_row_only=bool(model_config.get("user_row_only", True)),
            precomputed_data_path=data_config.get(
                "precomputed_path", "data/processed/ml1m"
            ),
        )

        model = model.to(self.device)

        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.log("\nModel: GraphFlix")
        self.log(f"  Total parameters: {num_params:,}")
        self.log(f"  Trainable parameters: {num_trainable:,}")
        self.log(f"  Beta init: {model.beta.item():.4f}")

        return model

    def _build_optimizer(self):
        """Build optimizer."""
        training_config = self.config["training"]

        # Convert to float in case config has string values
        lr = float(training_config["lr"])
        weight_decay = float(training_config.get("weight_decay", 1e-5))

        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.log("\nOptimizer: AdamW")
        self.log(f"  Learning rate: {lr}")
        self.log(f"  Weight decay: {weight_decay}")

        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        training_config = self.config["training"]
        scheduler_type = training_config.get("scheduler", "cosine")

        if scheduler_type == "cosine":
            # Convert lr_min to float (YAML may read scientific notation as string)
            lr_min = float(training_config.get("lr_min", 1e-6))
            scheduler = CosineAnnealingLR(
                self.optimizer, T_max=training_config["epochs"], eta_min=lr_min
            )
            self.log("Scheduler: CosineAnnealingLR")
            self.log(f"  Min LR: {lr_min}")
        elif scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5, verbose=True
            )
            self.log("Scheduler: ReduceLROnPlateau")
        else:
            scheduler = None
            self.log("Scheduler: None")

        return scheduler

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = batch.to(self.device)

            # Forward pass
            scores_pos, scores_neg, aux = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                node_types=batch.node_types,
                batch=batch.batch,
                user_ids=batch.user_ids,
                movie_ids_pos=batch.movie_ids_pos,
                movie_ids_neg=batch.movie_ids_neg,
                batch_info=batch.batch_info,
            )

            # Compute BPR loss
            loss = bpr_loss(scores_pos, scores_neg)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_clip = float(self.config["training"].get("grad_clip", 0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log to TensorBoard (batch level) - happens EVERY batch
            self.writer.add_scalar('Loss/train_batch', loss.item(), self.global_step)
            self.writer.add_scalar('Beta/train_batch', aux['beta'], self.global_step)

            # Print progress and flush TensorBoard every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / num_batches
                elapsed = time.time() - epoch_start
                self.log(
                    f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Beta: {aux['beta']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
                # Flush TensorBoard to ensure data is written to disk
                self.writer.flush()

        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start

        # Flush TensorBoard at end of epoch to ensure all data is written
        self.writer.flush()

        return {
            "loss": avg_loss,
            "epoch_time": epoch_time,
            "beta": self.model.beta.item(),
        }

    @torch.no_grad()
    def validate(self, dataloader, eval_dataloader=None):
        """Validate the model."""
        self.model.eval()

        # Compute validation loss
        val_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)

            scores_pos, scores_neg, aux = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                node_types=batch.node_types,
                batch=batch.batch,
                user_ids=batch.user_ids,
                movie_ids_pos=batch.movie_ids_pos,
                movie_ids_neg=batch.movie_ids_neg,
                batch_info=batch.batch_info,
            )

            loss = bpr_loss(scores_pos, scores_neg)
            val_loss += loss.item()
            num_batches += 1

        avg_val_loss = val_loss / num_batches

        # Compute ranking metrics if eval_dataloader provided
        metrics = {"loss": avg_val_loss}

        if eval_dataloader is not None and self.config["eval"].get(
            "compute_metrics", True
        ):
            self.log("  Computing ranking metrics...")
            ranking_metrics = evaluate_model(
                model=self.model,
                dataloader=eval_dataloader,
                k_list=self.config["eval"].get("k_list", [10, 20]),
                device=self.device,
            )
            metrics.update(ranking_metrics)

        return metrics

    def train(self, train_loader, val_loader=None, val_eval_loader=None):
        """Full training loop."""
        num_epochs = self.config["training"]["epochs"]
        eval_every = self.config["eval"].get("eval_every", 5)

        self.log("Starting Training")
        self.log("=" * 80)
        self.log(f"Epochs: {num_epochs}")
        self.log(f"Batch size: {self.config['training']['batch_size']}")
        self.log(f"Eval every: {eval_every} epochs")
        self.log("=" * 80 + "\n")

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch

            self.log(f"Epoch {epoch}/{num_epochs}")
            self.log("-" * 80)

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)

            # Log epoch-level training metrics to TensorBoard
            self.writer.add_scalar('Loss/train_epoch', train_metrics['loss'], epoch)
            self.writer.add_scalar('Beta/train_epoch', train_metrics['beta'], epoch)
            self.writer.add_scalar('Time/train_epoch', train_metrics['epoch_time'], epoch)

            self.log(
                f"  Train Loss: {train_metrics['loss']:.4f} | "
                f"Beta: {train_metrics['beta']:.4f} | "
                f"Time: {train_metrics['epoch_time']:.1f}s"
            )

            # Validate
            val_metrics = None
            if val_loader is not None and epoch % eval_every == 0:
                self.log("  Validating...")
                val_metrics = self.validate(val_loader, val_eval_loader)
                self.val_history.append({"epoch": epoch, **val_metrics})

                # Log validation metrics to TensorBoard
                self.writer.add_scalar('Loss/val_epoch', val_metrics['loss'], epoch)
                if "recall@10" in val_metrics:
                    self.writer.add_scalar('Metrics/recall@10', val_metrics['recall@10'], epoch)
                    self.writer.add_scalar('Metrics/ndcg@10', val_metrics['ndcg@10'], epoch)
                if "recall@20" in val_metrics:
                    self.writer.add_scalar('Metrics/recall@20', val_metrics['recall@20'], epoch)
                    self.writer.add_scalar('Metrics/ndcg@20', val_metrics['ndcg@20'], epoch)

                self.log(f"  Val Loss: {val_metrics['loss']:.4f}")
                if "recall@10" in val_metrics:
                    self.log(
                        f"  Recall@10: {val_metrics['recall@10']:.4f} | "
                        f"NDCG@10: {val_metrics['ndcg@10']:.4f}"
                    )

                # Check for best model
                primary_metric = val_metrics.get("recall@10", -val_metrics["loss"])
                if primary_metric > self.best_metric:
                    self.best_metric = primary_metric
                    self.save_checkpoint("best.pt")
                    self.log(" New best model saved!")
                    

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Only step if we have validation metrics
                    if val_metrics is not None:
                        self.scheduler.step(
                            val_metrics.get("recall@10", -val_metrics["loss"])
                        )
                else:
                    self.scheduler.step()

            # Log learning rate to TensorBoard
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('LearningRate/lr', current_lr, epoch)
            
            # Flush TensorBoard after all epoch metrics are logged
            self.writer.flush()

            # Save checkpoint every N epochs
            if epoch % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

            self.log("")

        # Save final model
        self.save_checkpoint("final.pt")

        # Save training history
        self.save_history()

        # Close TensorBoard writer
        self.writer.close()

        self.log("=" * 80)
        self.log("Training Complete!")
        self.log("=" * 80)
        self.log(f"\nTraining log saved to: {self.log_file}")
        self.log(f"To view training metrics in TensorBoard, run:")
        self.log(f"  tensorboard --logdir={self.save_dir / 'tensorboard'}")
        self.log(f"  Then open http://localhost:6006 in your browser\n")

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "config": self.config,
            "best_metric": self.best_metric,
        }

        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]

        self.log(f"Loaded checkpoint from epoch {self.epoch}")

    def save_history(self):
        """Save training history to JSON."""
        history = {
            "train": self.train_history,
            "val": self.val_history,
            "config": self.config,
        }

        history_path = self.save_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        self.log(f"Training history saved to: {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train GraphFlix model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/graphflix_full.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/ml1m",
        help="Path to processed data",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--save_dir", type=str, default="runs", help="Directory to save checkpoints"
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, graph_data, node_type_map = load_graph_and_create_dataloader(
        data_dir=args.data_dir,
        batch_size=config["training"]["batch_size"],
        k_hops=config["training"].get("k_hops", 2),
        split="train",
        seed=config.get("seed", 42),
    )

    val_loader, _, _ = load_graph_and_create_dataloader(
        data_dir=args.data_dir,
        batch_size=config["training"]["batch_size"],
        k_hops=config["training"].get("k_hops", 2),
        split="val",
        seed=config.get("seed", 42),
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create trainer (this will setup logging)
    trainer = Trainer(config, save_dir=args.save_dir)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.log(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(train_loader, val_loader)

    trainer.log(f"\nResults saved to: {trainer.save_dir}")
    print(f"\nTraining complete! Logs saved to: {trainer.log_file}")
    print(f"To monitor training progress, run in another terminal:")
    print(f"  tail -f {trainer.log_file}")


if __name__ == "__main__":
    main()
