"""Compatibility model training with triplet loss.

CompatibilityTrainer: Orchestrates training loop, validation, checkpointing.
Supports joint multi-task training with Street2Shop for street-to-shop robustness.
"""

import itertools
import pathlib

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from src.config import config as cfg, load_config
from src.dataloader.data_loaders import FashionCompleteTheLookDataloader, Street2ShopDataloader
from src.models.compatibility_model import CompatibilityModel
from src.utils import init_weights, plot_learning_curves

# Disable anomaly detection and profilers for speed
torch.autograd.set_detect_anomaly(False)
if hasattr(torch.autograd.profiler, "profile"):
    torch.autograd.profiler.profile(False)
if hasattr(torch.autograd.profiler, "emit_nvtx"):
    torch.autograd.profiler.emit_nvtx(False)


class CompatibilityTrainer:
    """Train CompatibilityModel on triplet (anchor, positive, negative) data."""

    def __init__(
        self,
        batch_size: int = cfg.BATCH_SIZE,
        num_epochs: int = cfg.NUM_EPOCHS,
        starting_epoch: int = 0,
        learning_rate: float = cfg.LEARNING_RATE,
        margin: float = cfg.MARGIN,
        save_dir: pathlib.Path | str | None = None,
        device: torch.device | None = None,
        use_street2shop: bool = False,
        street2shop_weight: float = 0.5,
        street2shop_batch_size: int = 32,
        use_polyvore_in_compatibility: bool = False,
    ):
        """Initialize trainer.

        Args:
            batch_size: Training batch size for CTL.
            num_epochs: Number of epochs to train.
            starting_epoch: Epoch to resume from (loads checkpoint if exists).
            learning_rate: Adam learning rate.
            margin: Triplet margin loss margin.
            save_dir: Directory for checkpoints. Default: cfg.TRAINED_MODEL_DIR.
            device: Device for training. Default: cfg.device.
            use_street2shop: If True, jointly train on Street2Shop.
            street2shop_weight: Loss weight for Street2Shop relative to CTL.
            street2shop_batch_size: Batch size for Street2Shop.
            use_polyvore_in_compatibility: If True, merge Polyvore triplets into CTL.
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.starting_epoch = starting_epoch
        self.learning_rate = learning_rate
        self.margin = margin
        self.save_dir = pathlib.Path(save_dir or cfg.TRAINED_MODEL_DIR)
        self.device = device or cfg.device
        self.use_street2shop = use_street2shop
        self.street2shop_weight = street2shop_weight
        self.street2shop_batch_size = street2shop_batch_size
        self.use_polyvore_in_compatibility = use_polyvore_in_compatibility
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        """Run full training loop: load data, train, validate, save checkpoints."""
        model = CompatibilityModel()
        train_loader = FashionCompleteTheLookDataloader(
            batch_size=self.batch_size,
            use_polyvore_in_compatibility=self.use_polyvore_in_compatibility,
        ).triplet_data_loader()
        val_loader = FashionCompleteTheLookDataloader(
            batch_size=max(self.batch_size // 9, 1),
            image_type="validation",
        ).triplet_data_loader()

        s2s_train_loader = None
        if self.use_street2shop:
            try:
                s2s_train_loader = Street2ShopDataloader(
                    split="train",
                    batch_size=self.street2shop_batch_size,
                ).triplet_data_loader()
            except Exception as e:
                raise RuntimeError(
                    f"Street2Shop enabled but dataloader failed: {e}. " "Run prepare street2shop to prepare pairs.csv."
                ) from e

        # Freeze backbone; train only embedding head
        for name, param in model.named_parameters():
            if name.split(".")[0] == "base_model":
                param.requires_grad = False

        model.apply(init_weights)
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        criterion = torch.nn.TripletMarginLoss(margin=self.margin).to(self.device)

        # Resume from checkpoint if available
        start_epoch = self._maybe_load_checkpoint(model, optimizer)

        training_losses: list[float] = []
        validation_losses: list[float] = []
        avg_training_losses: list[float] = []
        avg_validation_losses: list[float] = []
        loss = torch.tensor(0.0)

        s2s_iter = itertools.cycle(iter(s2s_train_loader)) if s2s_train_loader is not None else None

        for e in tqdm(range(start_epoch, start_epoch + self.num_epochs), desc="Epochs"):
            for i, (anchor, positive, negative) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
                optimizer.zero_grad(set_to_none=True)
                ctl_loss = self._triplet_loss(anchor, positive, negative, criterion, model)
                loss = ctl_loss

                if s2s_iter is not None:
                    try:
                        s2s_anchor, s2s_pos, s2s_neg = next(s2s_iter)
                        s2s_loss = self._triplet_loss(s2s_anchor, s2s_pos, s2s_neg, criterion, model)
                        loss = ctl_loss + self.street2shop_weight * s2s_loss
                    except StopIteration:
                        s2s_iter = itertools.cycle(iter(s2s_train_loader))
                        s2s_anchor, s2s_pos, s2s_neg = next(s2s_iter)
                        s2s_loss = self._triplet_loss(s2s_anchor, s2s_pos, s2s_neg, criterion, model)
                        loss = ctl_loss + self.street2shop_weight * s2s_loss

                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    training_losses.append(loss.cpu().detach().numpy())
                    avg_training_losses.append(float(np.mean(training_losses)))
                    scheduler.step(training_losses[-1])
                    model.eval()
                    with torch.no_grad():
                        val_loss = self._validate_batch(val_loader, criterion, model)
                        validation_losses.append(val_loss)
                        avg_validation_losses.append(float(np.mean(validation_losses)))
                    model.train()
                    log_msg = (
                        f"\nAvg Train Loss: {np.mean(training_losses):.4f}, "
                        f"Step Train: {training_losses[-1]:.4f}, "
                        f"Avg Val Loss: {np.mean(validation_losses):.4f}, "
                        f"Step Val: {validation_losses[-1]:.4f}\n"
                    )
                    print(log_msg)

            self._save_checkpoint(model, optimizer, e, loss)

        plot_learning_curves(avg_training_losses, avg_validation_losses)

    def _triplet_loss(self, anchor, positive, negative, criterion, model) -> torch.Tensor:
        """Compute triplet margin loss for one batch."""
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        a_feat = model(anchor)
        p_feat = model(positive)
        n_feat = model(negative)
        return criterion(a_feat, p_feat, n_feat)

    def _validate_batch(self, val_loader, criterion, model) -> float:
        """Get validation loss for one batch."""
        iterator = iter(val_loader)
        try:
            anchor, positive, negative = next(iterator)
        except StopIteration:
            iterator = iter(val_loader)
            anchor, positive, negative = next(iterator)
        loss = self._triplet_loss(anchor, positive, negative, criterion, model)
        return loss.cpu().detach().numpy()

    def _maybe_load_checkpoint(self, model, optimizer) -> int:
        """Load checkpoint if exists; return starting epoch."""
        prev_path = self.save_dir / f"trained_compatibility_model_epoch{self.starting_epoch - 1}.pth"
        if prev_path.exists():
            ckpt = torch.load(prev_path)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            return ckpt.get("epoch", self.starting_epoch) + 1
        return self.starting_epoch

    def _save_checkpoint(self, model, optimizer, epoch: int, loss: torch.Tensor) -> None:
        """Save checkpoint to save_dir."""
        path = self.save_dir / f"trained_compatibility_model_epoch{epoch}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "epoch": epoch,
            },
            path,
        )


def train_compatibility_model(
    starting_epoch: int = 0,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    config_path: str | pathlib.Path | None = None,
) -> None:
    """Train compatibility model (backward-compatible entry point).

    Args:
        starting_epoch: Epoch to resume from.
        num_epochs: Override config.
        batch_size: Override config.
        config_path: Path to train.yaml. Uses defaults if None.
    """
    defaults = {
        "batch_size": cfg.BATCH_SIZE,
        "num_epochs": cfg.NUM_EPOCHS,
        "starting_epoch": 0,
        "learning_rate": cfg.LEARNING_RATE,
        "margin": cfg.MARGIN,
        "save_dir": str(cfg.TRAINED_MODEL_DIR),
        "use_street2shop": False,
        "street2shop_weight": 0.5,
        "street2shop_batch_size": 32,
        "use_polyvore_in_compatibility": False,
    }
    config = load_config(config_path, defaults)
    trainer = CompatibilityTrainer(
        batch_size=config.get("batch_size", batch_size or cfg.BATCH_SIZE),
        num_epochs=config.get("num_epochs", num_epochs or cfg.NUM_EPOCHS),
        starting_epoch=config.get("starting_epoch", starting_epoch),
        learning_rate=config.get("learning_rate", cfg.LEARNING_RATE),
        margin=config.get("margin", cfg.MARGIN),
        save_dir=config.get("save_dir", cfg.TRAINED_MODEL_DIR),
        use_street2shop=config.get("use_street2shop", False),
        street2shop_weight=config.get("street2shop_weight", 0.5),
        street2shop_batch_size=config.get("street2shop_batch_size", 32),
        use_polyvore_in_compatibility=config.get("use_polyvore_in_compatibility", False),
    )
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--starting-epoch", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    train_compatibility_model(
        config_path=args.config,
        starting_epoch=args.starting_epoch or 0,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )
