"""
bridge.py — helpers to connect your existing notebook variables to the XAI refactor.
"""

from typing import Tuple, Optional, Callable, List
import numpy as np

# ---- Batch discovery ----

COMMON_ECG_NAMES = [
    "X_test", "X_val", "X_valid", "X", "ecg_data", "ecg_array", "signals", "data", "test_X", "val_X", "X_eval"
]

def _get_from_globals(names: List[str], g: dict):
    for n in names:
        if n in g:
            return g[n], n
    return None, None

def find_ecg_batch(g: dict) -> Tuple[np.ndarray, str]:
    """
    Look through the caller's globals for an ECG batch-like array.
    Returns the array and the variable name.
    Raises a RuntimeError if not found.
    """
    arr, name = _get_from_globals(COMMON_ECG_NAMES, g)
    if arr is None:
        # try nested dicts common in loaders
        for n in COMMON_ECG_NAMES:
            if n in g and isinstance(g[n], dict):
                for k, v in g[n].items():
                    if isinstance(v, np.ndarray):
                        arr, name = v, f"{n}['{k}']"
                        break
            if arr is not None:
                break
    if arr is None:
        raise RuntimeError("Could not locate an ECG batch. Please set one of: " + ", ".join(COMMON_ECG_NAMES))
    return arr, name

def to_batch_L_T(x: np.ndarray) -> np.ndarray:
    """
    Normalize array to shape (B, L, T).
    Accepts (B, T, L), (L, T), (T, L). Adds batch dim if needed.
    """
    if x.ndim == 3:
        # guess which dim is leads
        B, A, B_or_T = x.shape
        # if middle dim < last, probably (B, L, T)
        if x.shape[1] < x.shape[2]:
            return x  # (B, L, T)
        else:
            return np.transpose(x, (0, 2, 1))  # (B, T, L) -> (B, L, T)
    elif x.ndim == 2:
        L, T = x.shape if x.shape[0] < x.shape[1] else (x.shape[1], x.shape[0])
        if x.shape[0] == L:
            x2 = x
        else:
            x2 = x.T
        return x2[None, ...]
    else:
        raise ValueError("Unsupported ECG array shape. Expect 2D or 3D array.")

# ---- Sampling rate discovery ----

COMMON_FS_NAMES = ["fs", "sampling_rate", "sr", "FS", "Fs"]

def find_fs(g: dict, default: Optional[int] = 500) -> int:
    val, name = _get_from_globals(COMMON_FS_NAMES, g)
    if val is None:
        if default is None:
            raise RuntimeError("Sampling rate not found. Please define 'fs' (e.g., 500).")
        return default
    return int(val)

# ---- Model predict_fn (Keras) ----

def make_keras_predict_fn(model, expects_time_lead: bool = True) -> Callable:
    """
    Returns predict_fn(xb)->probs for a Keras model.
    expects_time_lead: if True, converts (B, L, T) -> (B, T, L) before predict.
    """
    import numpy as np
    def predict_fn(xb):
        x = xb
        if expects_time_lead:
            if x.ndim == 3 and x.shape[1] < x.shape[2]:
                x = np.transpose(x, (0, 2, 1))
        probs = model.predict(x, verbose=0)
        return probs
    return predict_fn

# ---- Integrated Gradients (Keras fallback) ----

def integrated_gradients_keras(model, x: np.ndarray, target_index: int, steps: int = 64) -> np.ndarray:
    """
    Compute IG attribution for a single sample (L, T) using a Keras model.
    Assumes the model input is (B, T, L). Transposes if needed.
    Returns attribution array (L, T).
    """
    import numpy as np
    import tensorflow as tf

    x1 = x[None, ...]  # (1, L, T)
    x_in = np.transpose(x1, (0, 2, 1))  # (1, T, L)
    baseline = np.zeros_like(x_in)
    alphas = tf.linspace(0.0, 1.0, steps+1)

    with tf.GradientTape() as tape:
        tape.watch(alphas)
        diffs = x_in - baseline
        path = baseline + tf.reshape(alphas, (-1,1,1,1)) * diffs  # (steps+1, 1, T, L)
        path = tf.reshape(path, (-1, x_in.shape[1], x_in.shape[2]))  # (steps+1, T, L)
        logits = model(path, training=False)
        if logits.shape[-1] == 1:
            probs = tf.nn.sigmoid(logits)
            target = probs[:, 0]
        else:
            probs = tf.nn.softmax(logits, axis=-1)
            target = probs[:, target_index]
    grads = tape.gradient(target, path)  # (steps+1, T, L)
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    ig = tf.reduce_mean(avg_grads, axis=0) * (x_in[0] - baseline[0])  # (T, L)
    ig_np = ig.numpy().T  # (L, T)
    return ig_np

