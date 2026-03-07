"""Legacy training entry point. Delegates to CompatibilityTrainer."""

from src.config import config as cfg
from src.models.trainer import train_compatibility_model

if __name__ == "__main__":
    train_compatibility_model(
        starting_epoch=0,
        num_epochs=cfg.NUM_EPOCHS,
        batch_size=cfg.BATCH_SIZE,
    )
