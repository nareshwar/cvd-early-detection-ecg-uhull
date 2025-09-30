"""
Faithfulness metrics for time-series explanations.
- Insertion/Deletion curves
- Randomization sanity checks (skeleton)
Framework-agnostic: provide a predict_fn(x_batch)->probabilities.
"""

from typing import Callable, Tuple
import numpy as np


def _mask_with_baseline(x: np.ndarray, mask: np.ndarray, baseline: float = 0.0) -> np.ndarray:
    """
    x: (L, T) or (T,) input
    mask: same shape as x with 1=keep, 0=replace
    baseline: value to replace masked positions
    """
    return x * mask + baseline * (1 - mask)


def _score(predict_fn: Callable, x: np.ndarray, class_index: int) -> float:
    """
    predict_fn expects shape (N, L, T) or (N, T). We add batch dim if needed.
    Returns probability for class_index.
    """
    xb = x[None, ...]
    probs = predict_fn(xb)  # (N, C)
    return float(probs[0, class_index])


def insertion_deletion_auc(
    predict_fn: Callable,
    x: np.ndarray,
    attribution: np.ndarray,
    class_index: int,
    steps: int = 20,
    mode: str = "insertion",
    baseline: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute insertion/deletion curve AUC.
    - Sort positions by |attribution| descending.
    - Progressively insert (or delete) those positions and record model score.
    Returns (fractions, scores, auc).
    """
    a = np.abs(attribution).reshape(-1)
    order = np.argsort(-a)  # descending
    x_flat = x.reshape(-1)
    N = x_flat.size

    # Start from baseline or full input
    if mode == "insertion":
        cur = np.full_like(x_flat, baseline)
    elif mode == "deletion":
        cur = x_flat.copy()
    else:
        raise ValueError("mode must be 'insertion' or 'deletion'")

    fractions = []
    scores = []
    step = max(1, N // steps)

    for k in range(0, N+1, step):
        if k > 0:
            idx = order[k-step:k]
            if mode == "insertion":
                cur[idx] = x_flat[idx]
            else:
                cur[idx] = baseline

        cur_reshaped = cur.reshape(x.shape)
        s = _score(predict_fn, cur_reshaped, class_index)
        fractions.append(k / N)
        scores.append(s)

    fractions = np.array(fractions)
    scores = np.array(scores)
    # Trapezoidal AUC
    auc = float(np.trapz(scores, fractions))
    return fractions, scores, auc


def weight_randomization_sanity(attribution_fn: Callable, model_getter: Callable, x: np.ndarray) -> None:
    """
    Skeleton for Adebayo-style sanity check:
    - progressively randomize layers of a model (model_getter with randomize=True, layer_idx=i)
    - recompute attributions, compare similarity vs original (e.g., Spearman/Pearson)
    This is left as a placeholder to adapt to your chosen framework (TF/PyTorch).
    """
    pass
