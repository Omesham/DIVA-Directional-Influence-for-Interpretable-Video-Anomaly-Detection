# codes/xai/base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseExplainer(ABC):
    """
    Parent class for all explainers (LIME, SHAP, Grad‑CAM, …).
    Each child must implement .explain(X) and return a list of
    {feature_idx: weight} dicts – one per input vector.
    """
    def __init__(self, grader, train_feats, **kwargs):
        self.grader = grader            # KNNGrader (black‑box scorer)
        self.train_feats = train_feats  # (N_train, D)

    # Helper every explainer can reuse future proof
    def _score_batch(self, X):
        """Return CKNN anomaly score for each row in X (shape [M, D])."""
        return np.array([self.grader.grade_flat(x[None, :])[0] for x in X])

    @abstractmethod
    def explain(self, X):
        """
        Parameters
        ----------
        X : np.ndarray  shape (M, D)
        Returns
        -------
        list[dict] : length M, each dict maps feature_index -> importance
        """
        ...
