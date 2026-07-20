import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)


_classifier = joblib.load(Path("models") / "classification_best_model.joblib")


def classify_embedding(embedding: np.ndarray) -> dict:
    """Classify an embedding and return cat_id, confidence, and cat name"""
    embedding = np.asarray(embedding, dtype=np.float32)

    # reshape to 2D if needed
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    # get probabilities (single model call)
    proba = _classifier.predict_proba(embedding)[0]
    cat_id = int(np.argmax(proba))
    confidence = float(proba[cat_id])
    cat_name = _classifier.cat_names_[cat_id]

    return {
        "cat_id": cat_id,
        "cat_name": cat_name,
        "confidence": confidence,
    }
