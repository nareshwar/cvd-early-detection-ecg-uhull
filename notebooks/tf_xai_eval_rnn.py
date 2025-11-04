
import numpy as np
from typing import Callable, Optional, Tuple, Dict, List, Union

try:
    import tensorflow as tf
except Exception as e:
    raise ImportError("This module requires TensorFlow. Please `pip install tensorflow` in your environment.") from e


Tensor = "tf.Tensor"


def _ensure_batch(x: tf.Tensor) -> tf.Tensor:
    return x if tf.rank(x) >= 3 else tf.expand_dims(x, 0)


def _to_format_btc(x: tf.Tensor, data_format: str) -> tf.Tensor:
    """
    Return tensor in BTC format (batch, time, channels).
    Accepts data_format in {'BTC','BCT'}.
    """
    if data_format == "BTC":
        return x
    elif data_format == "BCT":
        return tf.transpose(x, perm=[0, 2, 1])
    else:
        raise ValueError("data_format must be 'BTC' or 'BCT'")


def _from_format_btc(x_btc: tf.Tensor, data_format: str) -> tf.Tensor:
    if data_format == "BTC":
        return x_btc
    elif data_format == "BCT":
        return tf.transpose(x_btc, perm=[0, 2, 1])
    else:
        raise ValueError("data_format must be 'BTC' or 'BCT'")


def _aggregate_time_importance(attr_btc: tf.Tensor, p: int = 1) -> tf.Tensor:
    """
    Aggregate channel-wise importance to a single importance over time (B, T).
    Uses Lp over channels (default L1).
    """
    if p == 1:
        imp = tf.reduce_sum(tf.abs(attr_btc), axis=-1)  # (B,T)
    elif p == 2:
        imp = tf.norm(attr_btc, ord="euclidean", axis=-1)  # (B,T)
    else:
        imp = tf.pow(tf.reduce_sum(tf.pow(tf.abs(attr_btc), p), axis=-1), 1.0 / p)
    return imp


def _normalize_per_sample(a_bt: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    a_min = tf.reduce_min(a_bt, axis=-1, keepdims=True)
    a_max = tf.reduce_max(a_bt, axis=-1, keepdims=True)
    return (a_bt - a_min) / (a_max - a_min + eps)


def _topk_mask(imp_bt: tf.Tensor, frac: float) -> tf.Tensor:
    """
    Returns a boolean mask (B,T) with top-frac positions set True.
    """
    shape = tf.shape(imp_bt)
    B = shape[0]
    T = shape[1]
    k = tf.maximum(1, tf.cast(tf.round(frac * tf.cast(T, tf.float32)), tf.int32))
    values, indices = tf.math.top_k(imp_bt, k=k, sorted=False)
    # Build mask
    batch_range = tf.reshape(tf.range(B), (-1, 1))
    scatter_idx = tf.concat([tf.tile(batch_range, [1, k])[..., tf.newaxis], tf.reshape(indices, (B, k, 1))], axis=-1)
    mask = tf.scatter_nd(scatter_idx, tf.ones((B, k), dtype=tf.bool), tf.shape(imp_bt))
    return mask


def _select_target(outputs: tf.Tensor, target: Optional[Union[int, tf.Tensor]]) -> tf.Tensor:
    """
    Given outputs (B,K) or (B,) select per-sample target.
    If target is None and outputs is (B,K), uses argmax.
    If outputs is (B,), returns as is.
    """
    if len(outputs.shape) == 1:
        return outputs
    B = tf.shape(outputs)[0]
    K = outputs.shape[-1]
    if target is None:
        idx = tf.argmax(outputs, axis=-1, output_type=tf.int32)  # (B,)
    elif isinstance(target, int):
        idx = tf.fill([B], tf.cast(target, tf.int32))
    else:
        idx = tf.cast(target, tf.int32)
    batch_idx = tf.range(B, dtype=tf.int32)
    gather_idx = tf.stack([batch_idx, idx], axis=1)
    return tf.gather_nd(outputs, gather_idx)  # (B,)


def _logit_from_prob(p: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    p = tf.clip_by_value(p, eps, 1.0 - eps)
    return tf.math.log(p) - tf.math.log1p(-p)


class TFAttribution:
    """
    TensorFlow/Keras attribution methods for 1D time-series models.
    Works with models expecting (B,T,C) by default (Keras channels_last). Set data_format='BCT' if needed.
    """

    def __init__(self, model: tf.keras.Model, data_format: str = "BTC", use_logit: bool = True):
        self.model = model
        self.data_format = data_format
        self.use_logit = use_logit

    def _forward(self, x_btc: tf.Tensor) -> tf.Tensor:
        return self.model(_from_format_btc(x_btc, self.data_format), training=False)

    def _scalar_for_grad(self, outputs: tf.Tensor, target: Optional[Union[int, tf.Tensor]]) -> tf.Tensor:
        sel = _select_target(outputs, target)  # (B,)
        if self.use_logit and len(outputs.shape) == 2:
            # Treat final activation as sigmoid and take logit to emulate pre-activation gradient
            sel = _logit_from_prob(sel)
        return tf.reduce_sum(sel)  # scalar

    def saliency(self, x: tf.Tensor, target: Optional[Union[int, tf.Tensor]] = None) -> tf.Tensor:
        x_btc = _to_format_btc(_ensure_batch(tf.convert_to_tensor(x)), self.data_format)
        x_btc = tf.cast(x_btc, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_btc)
            outputs = self._forward(x_btc)
            scalar = self._scalar_for_grad(outputs, target)
        grad = tape.gradient(scalar, x_btc)
        return _from_format_btc(grad, self.data_format)

    def grad_x_input(self, x: tf.Tensor, target: Optional[Union[int, tf.Tensor]] = None) -> tf.Tensor:
        x_btc = _to_format_btc(_ensure_batch(tf.convert_to_tensor(x)), self.data_format)
        x_btc = tf.cast(x_btc, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_btc)
            outputs = self._forward(x_btc)
            scalar = self._scalar_for_grad(outputs, target)
        grad = tape.gradient(scalar, x_btc)
        attr = grad * x_btc
        return _from_format_btc(attr, self.data_format)

    def integrated_gradients(
        self,
        x: tf.Tensor,
        target: Optional[Union[int, tf.Tensor]] = None,
        baseline: Optional[tf.Tensor] = None,
        steps: int = 50,
        random_alpha: bool = False,
    ) -> tf.Tensor:
        x_btc = _to_format_btc(_ensure_batch(tf.convert_to_tensor(x)), self.data_format)
        x_btc = tf.cast(x_btc, tf.float32)
        if baseline is None:
            baseline = tf.zeros_like(x_btc)
        else:
            baseline = _to_format_btc(_ensure_batch(tf.convert_to_tensor(baseline)), self.data_format)
            baseline = tf.cast(baseline, tf.float32)

        if random_alpha:
            alphas = tf.random.uniform((steps,), 0.0, 1.0, dtype=x_btc.dtype)
        else:
            alphas = (tf.range(steps, dtype=x_btc.dtype) + 0.5) / tf.cast(steps, x_btc.dtype)

        grads = tf.zeros_like(x_btc)
        for i in range(steps):
            a = alphas[i]
            x_i = baseline + a * (x_btc - baseline)
            with tf.GradientTape() as tape:
                tape.watch(x_i)
                outputs = self._forward(x_i)
                scalar = self._scalar_for_grad(outputs, target)
            g = tape.gradient(scalar, x_i)
            grads = grads + g

        avg_grad = grads / float(steps)
        attr = (x_btc - baseline) * avg_grad
        return _from_format_btc(attr, self.data_format)

    def smoothgrad(self, x: tf.Tensor, target: Optional[Union[int, tf.Tensor]] = None, stdev: float = 0.1, samples: int = 25) -> tf.Tensor:
        x_btc = _to_format_btc(_ensure_batch(tf.convert_to_tensor(x)), self.data_format)
        x_btc = tf.cast(x_btc, tf.float32)
        # per-sample std scaling
        flat = tf.reshape(x_btc, [tf.shape(x_btc)[0], -1])
        x_std = tf.maximum(tf.math.reduce_std(flat, axis=1, keepdims=True), 1e-8)
        x_std = tf.reshape(x_std, [tf.shape(x_btc)[0]] + [1] * (len(x_btc.shape) - 1))
        acc = tf.zeros_like(x_btc)
        for _ in range(samples):
            noise = tf.random.normal(tf.shape(x_btc), dtype=x_btc.dtype) * (stdev * x_std)
            x_noisy = x_btc + noise
            with tf.GradientTape() as tape:
                tape.watch(x_noisy)
                outputs = self._forward(x_noisy)
                scalar = self._scalar_for_grad(outputs, target)
            g = tape.gradient(scalar, x_noisy)
            acc = acc + g
        return _from_format_btc(acc / float(samples), self.data_format)


class TFMetrics:
    """
    Faithfulness and consistency metrics for time-series attributions (TensorFlow/Keras).
    """

    def __init__(self, model: tf.keras.Model, data_format: str = "BTC", use_logit: bool = False):
        self.model = model
        self.data_format = data_format
        self.use_logit = use_logit

    def _forward(self, x_btc: tf.Tensor) -> tf.Tensor:
        return self.model(_from_format_btc(x_btc, self.data_format), training=False)

    def _target_scores(self, outputs: tf.Tensor, target: Optional[Union[int, tf.Tensor]]) -> tf.Tensor:
        sel = _select_target(outputs, target)
        if self.use_logit and len(outputs.shape) == 2:
            sel = _logit_from_prob(sel)
        return sel  # (B,)

    def _btc(self, x: tf.Tensor) -> tf.Tensor:
        return _to_format_btc(_ensure_batch(tf.convert_to_tensor(x)), self.data_format)

    def deletion_insertion_curves(
        self,
        x: tf.Tensor,
        attr: tf.Tensor,
        target: Optional[Union[int, tf.Tensor]] = None,
        baseline_value: float = 0.0,
        steps: int = 20,
        aggregate_channels_p: int = 1,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        x_btc = self._btc(x)
        attr_btc = self._btc(attr)
        B = tf.shape(x_btc)[0]
        T = tf.shape(x_btc)[1]
        C = tf.shape(x_btc)[2]

        imp = _aggregate_time_importance(attr_btc, p=aggregate_channels_p)  # (B,T)
        imp_norm = _normalize_per_sample(imp)

        fracs = tf.linspace(0.0, 1.0, steps)
        baseline = tf.fill(tf.shape(x_btc), tf.cast(baseline_value, x_btc.dtype))

        def scores(tensor_btc):
            outputs = self._forward(tensor_btc)
            sel = self._target_scores(outputs, target)
            return tf.reduce_mean(sel).numpy()

        # initial
        x_del = tf.identity(x_btc)
        x_ins = tf.identity(baseline)

        scores_del = [scores(x_del)]
        scores_ins = [scores(x_ins)]

        # Precompute order
        order = tf.argsort(imp_norm, axis=1, direction="DESCENDING")  # (B,T)

        for i in range(1, steps):
            frac = fracs[i].numpy().item()
            k = max(1, int(round(frac * int(T.numpy()))))
            # Build mask from precomputed order
            idx = order[:, :k]  # (B,k)
            batch_range = tf.reshape(tf.range(tf.shape(x_btc)[0]), (-1, 1))
            scatter_idx = tf.concat([tf.tile(batch_range, [1, k])[..., tf.newaxis], tf.reshape(idx, (tf.shape(x_btc)[0], k, 1))], axis=-1)
            mask_bt = tf.scatter_nd(scatter_idx, tf.ones((tf.shape(x_btc)[0], k), dtype=tf.bool), tf.shape(imp_norm))

            # Expand to channels
            mask_btc = tf.tile(mask_bt[..., tf.newaxis], [1, 1, tf.shape(x_btc)[2]])

            # Deletion
            x_del = tf.where(mask_btc, baseline, x_btc)
            scores_del.append(scores(x_del))

            # Insertion
            x_ins = tf.where(mask_btc, x_btc, baseline)
            scores_ins.append(scores(x_ins))

        fractions = fracs.numpy()
        return {
            "deletion": (fractions, np.array(scores_del)),
            "insertion": (fractions, np.array(scores_ins)),
        }

    @staticmethod
    def _auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.trapz(y, x))

    def deletion_insertion_auc(
        self,
        x: tf.Tensor,
        attr: tf.Tensor,
        target: Optional[Union[int, tf.Tensor]] = None,
        baseline_value: float = 0.0,
        steps: int = 20,
        aggregate_channels_p: int = 1,
    ) -> Dict[str, float]:
        curves = self.deletion_insertion_curves(x, attr, target, baseline_value, steps, aggregate_channels_p)
        del_auc = self._auc_trapezoid(curves["deletion"][0], curves["deletion"][1])
        ins_auc = self._auc_trapezoid(curves["insertion"][0], curves["insertion"][1])
        return {"deletion_auc": del_auc, "insertion_auc": ins_auc}

    def consistency_noise_robustness(
        self,
        x: tf.Tensor,
        attr_fn: Callable[[tf.Tensor], tf.Tensor],
        noise_levels: List[float] = [0.01, 0.05, 0.1],
        samples_per_level: int = 5,
        aggregate_channels_p: int = 1,
        topk_fracs: List[float] = [0.05, 0.1, 0.2],
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Measures attribution consistency under additive Gaussian noise.
        Returns average rank correlation and top-k Jaccard across noise levels.
        """
        x_btc = self._btc(x)
        B = int(x_btc.shape[0])
        T = int(x_btc.shape[1])

        base_attr = self._btc(attr_fn(_from_format_btc(x_btc, self.data_format)))
        base_imp = _normalize_per_sample(_aggregate_time_importance(base_attr, p=aggregate_channels_p)).numpy()  # (B,T)

        def rankdata(a: np.ndarray) -> np.ndarray:
            temp = np.argsort(a, kind="mergesort")
            ranks = np.empty_like(temp, dtype=float)
            ranks[temp] = np.arange(len(a), dtype=float) + 1.0
            _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
            sums = np.bincount(inv, ranks)
            avg_ranks = sums[inv] / counts[inv]
            return avg_ranks

        # per-sample std
        flat = tf.reshape(x_btc, [tf.shape(x_btc)[0], -1])
        x_std = tf.maximum(tf.math.reduce_std(flat, axis=1, keepdims=True), 1e-8)
        x_std = tf.reshape(x_std, [tf.shape(x_btc)[0]] + [1] * (len(x_btc.shape) - 1))

        spearman_list: List[float] = []
        jaccard: Dict[float, List[float]] = {f: [] for f in topk_fracs}

        for sigma in noise_levels:
            for _ in range(samples_per_level):
                noise = tf.random.normal(tf.shape(x_btc), dtype=x_btc.dtype) * (sigma * x_std)
                x_noisy = x_btc + noise
                noisy_attr = self._btc(attr_fn(_from_format_btc(x_noisy, self.data_format)))
                noisy_imp = _normalize_per_sample(_aggregate_time_importance(noisy_attr, p=aggregate_channels_p)).numpy()

                # Spearman per sample
                corr_batch = []
                for b in range(B):
                    a = base_imp[b]
                    c = noisy_imp[b]
                    ra = rankdata(a) - np.mean(rankdata(a))
                    rc = rankdata(c) - np.mean(rankdata(c))
                    denom = (np.linalg.norm(ra) * np.linalg.norm(rc) + 1e-12)
                    corr = float(np.dot(ra, rc) / denom)
                    corr_batch.append(corr)
                spearman_list.append(float(np.mean(corr_batch)))

                # top-k Jaccard
                for f in topk_fracs:
                    k = max(1, int(round(f * T)))
                    base_idx = np.argpartition(-base_imp, kth=k-1, axis=1)[:, :k]
                    noisy_idx = np.argpartition(-noisy_imp, kth=k-1, axis=1)[:, :k]
                    inter = 0.0
                    for b in range(B):
                        s1 = set(base_idx[b].tolist())
                        s2 = set(noisy_idx[b].tolist())
                        j = len(s1 & s2) / float(len(s1 | s2))
                        inter += j
                    jaccard[f].append(inter / B)

        out: Dict[str, Union[float, Dict[str, float]]] = {}
        if spearman_list:
            out["spearman_robustness"] = float(np.mean(spearman_list))
        out["topk_jaccard"] = {str(f): float(np.mean(vals)) if len(vals) > 0 else float("nan") for f, vals in jaccard.items()}
        return out

    def infidelity(
        self,
        x: tf.Tensor,
        attr: tf.Tensor,
        target: Optional[Union[int, tf.Tensor]] = None,
        sigma: float = 0.05,
        samples: int = 50,
    ) -> float:
        """
        Infidelity (Yeh et al. 2019): E[(f(x+e)-f(x) - <attr, e>)^2],
        with e ~ N(0, sigma^2 I). Lower is better.
        """
        x_btc = self._btc(x)
        attr_btc = self._btc(attr)
        B = int(x_btc.shape[0])

        outputs_x = self._forward(x_btc)
        sel_x = self._target_scores(outputs_x, target)  # (B,)

        attr_flat = tf.reshape(attr_btc, [B, -1])
        flat = tf.reshape(x_btc, [B, -1])
        x_std = tf.maximum(tf.math.reduce_std(flat, axis=1, keepdims=True), 1e-8)

        total = 0.0
        for _ in range(samples):
            noise = tf.random.normal(tf.shape(x_btc), dtype=x_btc.dtype) * (sigma * tf.reshape(x_std, [B] + [1] * (len(x_btc.shape) - 1)))
            outputs_xe = self._forward(x_btc + noise)
            sel_xe = self._target_scores(outputs_xe, target)  # (B,)
            diff = tf.reshape(sel_xe - sel_x, [B, 1])  # (B,1)
            inner = tf.reduce_sum(attr_flat * tf.reshape(noise, [B, -1]), axis=1, keepdims=True)
            total += float(tf.reduce_mean(tf.square(diff - inner)).numpy())
        return total / float(samples)

    def sensitivity_n(
        self,
        x: tf.Tensor,
        attr: tf.Tensor,
        target: Optional[Union[int, tf.Tensor]] = None,
        n: int = 50,
        trials: int = 50,
    ) -> float:
        """
        Sensitivity-n (Ancona et al. 2017): correlation between attribution sum over a random subset of n features
        and the output change when those features are set to baseline (0).
        """
        x_btc = self._btc(x)
        attr_btc = self._btc(attr)

        B = int(x_btc.shape[0])
        T = int(x_btc.shape[1])
        C = int(x_btc.shape[2])
        idx_space = C * T

        outputs_x = self._forward(x_btc)
        sel_x = self._target_scores(outputs_x, target)  # (B,)

        attr_flat = tf.reshape(attr_btc, [B, -1])  # (B, C*T)

        corrs = []
        for _ in range(trials):
            idx = tf.random.shuffle(tf.range(idx_space))[:n]  # (n,)
            # Build mask (C,T)
            c_idx = idx // T
            t_idx = idx % T
            mask_ct = tf.scatter_nd(
                tf.stack([c_idx, t_idx], axis=1),
                tf.ones((n,), dtype=tf.bool),
                shape=(C, T),
            )
            mask_btc = tf.tile(tf.expand_dims(tf.transpose(mask_ct, [1, 0]), axis=0), [B, 1, 1])  # (B,T,C)

            x_masked = tf.where(mask_btc, tf.zeros_like(x_btc), x_btc)
            outputs_xm = self._forward(x_masked)
            sel_xm = self._target_scores(outputs_xm, target)  # (B,)
            delta = tf.reshape(sel_x - sel_xm, [B, 1])  # (B,1)

            # Flatten mask and sum attr
            mask_flat = tf.reshape(tf.transpose(mask_ct, [1, 0]), [-1])  # (T*C,)
            mask_indices = tf.where(mask_flat)[:, 0]
            attr_sum = tf.reduce_sum(tf.gather(attr_flat, mask_indices, axis=1), axis=1, keepdims=True)  # (B,1)

            # Pearson correlation across batch
            a = attr_sum - tf.reduce_mean(attr_sum)
            b = delta - tf.reduce_mean(delta)
            denom = tf.maximum(tf.math.reduce_std(a) * tf.math.reduce_std(b), 1e-12)
            corrs.append(float(tf.reduce_mean(a * b) / denom))
        return float(np.mean(corrs))


def quick_example_usage():
    """
    Example (Keras):
    from tf_xai_eval_timeseries import TFAttribution, TFMetrics
    model = residual_network_1d()  # your model
    data_format = "BTC"  # Keras Conv1D default
    x = tf.random.normal((8, 5000, 12))  # (B,T,C)
    target_class = 3  # choose your class index

    attr = TFAttribution(model, data_format=data_format).integrated_gradients(x, target=target_class, steps=64)

    metrics = TFMetrics(model, data_format=data_format)
    print(metrics.deletion_insertion_auc(x, attr, target=target_class, steps=21))
    print(metrics.consistency_noise_robustness(x, lambda xb: TFAttribution(model, data_format=data_format).saliency(xb, target=target_class)))
    print("infidelity:", metrics.infidelity(x, attr, target=target_class))
    print("sensitivity_n:", metrics.sensitivity_n(x, attr, target=target_class, n=128, trials=20))
    """
    pass
