"""Evaluation for compatibility model.

CompatibilityEvaluator: Sample triplets from test set, compute accuracy
(anchor-positive closer than anchor-negative).
"""

import pickle
from pathlib import Path

import pandas as pd
import torch

from src.config import config as cfg
from src.config import get_simple_logger
from src.dataloader.data_loaders import FashionCompleteTheLookDataloader
from src.features.embeddings import CompatibleProductEmbedder
from src.utils import calculate_similarity

logger = get_simple_logger(__name__)


class CompatibilityEvaluator:
    """Evaluate compatibility model on test triplets.

    For each outfit: sample anchor, positive (same outfit, different category),
    negative (same category, different outfit). Correct if sim(anchor, pos) > sim(anchor, neg).
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        embedding_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
    ):
        """Initialize evaluator.

        Args:
            model_path: Path to CompatibilityModel checkpoint. Used to compute embeddings if not cached.
            embedding_path: Path to cached test embeddings. If exists, skips model inference.
            metadata_path: Path to dataset_metadata_ctl_single.csv.
        """
        self.model_path = Path(model_path) if model_path else None
        self.embedding_path = Path(
            embedding_path
            or cfg.PACKAGE_ROOT
            / "features/cached_embeddings"
            / "compatible_product_test_embedding.pickle"
        )
        self.metadata_path = Path(
            metadata_path
            or cfg.DATASET_DIR / "metadata" / "dataset_metadata_ctl_single.csv"
        )
        self._test_features: torch.Tensor | None = None
        self._metadata_test: pd.DataFrame | None = None

    def _load_embeddings(self) -> None:
        """Load or compute test embeddings."""
        if self._test_features is not None:
            return
        if self.embedding_path.exists():
            with open(self.embedding_path, "rb") as f:
                self._test_features = pickle.load(f)
            return
        # Compute embeddings
        data_loader = FashionCompleteTheLookDataloader(
            image_type="test"
        ).single_data_loader()
        embedder = CompatibleProductEmbedder(
            model_path=self.model_path or cfg.TRAINED_MODEL_DIR
        )
        self._test_features = embedder.extract(
            data_loader,
            task_name="compatible_product_test",
            save_path=self.embedding_path,
        )

    def _load_metadata(self) -> None:
        """Load and prepare test metadata."""
        if self._metadata_test is not None:
            return
        metadata = pd.read_csv(self.metadata_path)
        self._metadata_test = metadata[metadata["image_type"] == "test"].copy()
        self._metadata_test["product_id"] = self._metadata_test.reset_index().index
        self._metadata_test["original_image_signature"] = (
            self._metadata_test["image_single_signature"].str.split("_").str[0]
        )

    def evaluate(self) -> float:
        """Compute accuracy: fraction of triplets where anchor-positive > anchor-negative.

        Returns:
            Accuracy in [0, 1].
        """
        self._load_embeddings()
        self._load_metadata()
        triplets = self._sample_triplets()
        correct = 0
        for anchor, positive, negative in triplets:
            a_feat = self._test_features[anchor["product_id"], :].unsqueeze(0)
            p_feat = self._test_features[positive["product_id"], :].unsqueeze(0)
            n_feat = self._test_features[negative["product_id"], :].unsqueeze(0)
            sim_ap = calculate_similarity(a_feat, p_feat, sim_function="cosine")
            sim_an = calculate_similarity(a_feat, n_feat, sim_function="cosine")
            if sim_ap > sim_an:
                correct += 1
        return correct / len(triplets) if triplets else 0.0

    def _sample_triplets(self) -> list[tuple[dict, dict, dict]]:
        """Sample (anchor, positive, negative) triplets from test metadata."""
        triplets = []
        for sig in self._metadata_test["original_image_signature"].unique():
            image_src = self._metadata_test[
                self._metadata_test["original_image_signature"] == sig
            ]
            anchor = image_src.sample(1).to_dict(orient="records")[0]
            pos_candidates = image_src[
                image_src["product_type"] != anchor["product_type"]
            ]
            if pos_candidates.empty:
                continue
            positive = pos_candidates.sample(1).to_dict(orient="records")[0]
            neg_candidates = self._metadata_test[
                (self._metadata_test["product_type"] == positive["product_type"])
                & (self._metadata_test["original_image_signature"] != sig)
            ]
            if neg_candidates.empty:
                continue
            negative = neg_candidates.sample(1).to_dict(orient="records")[0]
            triplets.append((anchor, positive, negative))
        return triplets


def evaluation() -> float:
    """Legacy entry point: run CompatibilityEvaluator and return accuracy."""
    evaluator = CompatibilityEvaluator()
    return evaluator.evaluate()


if __name__ == "__main__":
    accuracy = evaluation()
    logger.info(f"The correct percentage of the compatibility test is {accuracy}")
