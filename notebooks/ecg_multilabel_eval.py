
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt

@dataclass
class ClassMetrics:
    class_name: str
    prevalence: float
    accuracy: float
    sensitivity: float  # recall for positive
    specificity: float
    auroc: float
    auprc: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

def _youden_threshold_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Guard for all-0 or all-1
    if y_true.max() == y_true.min():
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thr[best_idx])

def _specificity_from_cm(cm: np.ndarray) -> float:
    tn, fp, fn, tp = cm.ravel()
    denom = (tn + fp)
    return float(tn / denom) if denom > 0 else 0.0

def _class_metrics(y_true_bin: np.ndarray, y_prob: np.ndarray, threshold: Optional[float] = None, class_name: str = "") -> ClassMetrics:
    if threshold is None:
        threshold = _youden_threshold_binary(y_true_bin, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true_bin, y_pred)
    sens = recall_score(y_true_bin, y_pred, pos_label=1, zero_division=0)
    spec = _specificity_from_cm(cm)
    # AUROC / AUPRC: handle degenerate cases
    try:
        auroc = roc_auc_score(y_true_bin, y_prob)
    except Exception:
        auroc = np.nan
    try:
        auprc = average_precision_score(y_true_bin, y_prob)
    except Exception:
        auprc = np.nan
    prev = float(np.mean(y_true_bin))
    return ClassMetrics(class_name, prev, acc, sens, spec, auroc, auprc, float(threshold), int(tp), int(fp), int(tn), int(fn))

def compute_per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    y_true: (N, C) binary matrix
    y_prob: (N, C) prob matrix
    class_names: length C
    thresholds: optional dict class_name -> threshold
    """
    rows = []
    for j, cname in enumerate(class_names):
        thr = None
        if thresholds is not None and cname in thresholds:
            thr = float(thresholds[cname])
        cm = _class_metrics(y_true[:, j].astype(int), y_prob[:, j].astype(float), threshold=thr, class_name=cname)
        rows.append(cm.__dict__)
    df = pd.DataFrame(rows)
    # micro/macro AUROC
    try:
        micro_auroc = roc_auc_score(y_true, y_prob, average="micro")
    except Exception:
        micro_auroc = np.nan
    try:
        macro_auroc = roc_auc_score(y_true, y_prob, average="macro")
    except Exception:
        macro_auroc = np.nan
    df.attrs["micro_auroc"] = micro_auroc
    df.attrs["macro_auroc"] = macro_auroc
    return df.sort_values("prevalence", ascending=False)

def plot_roc_for_top_k(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], k: int = 6) -> None:
    # pick top-k by prevalence
    prev = np.mean(y_true, axis=0)
    idx = np.argsort(-prev)[:k]
    for j in idx:
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(y_true[:, j], y_prob[:, j])
        auc = roc_auc_score(y_true[:, j], y_prob[:, j]) if np.unique(y_true[:, j]).size > 1 else np.nan
        plt.figure()
        plt.plot(fpr, tpr, label=f"{class_names[j]} (AUC={auc:.3f})")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC — {class_names[j]}")
        plt.legend(loc="lower right")
        plt.show()

def fold_eval_table(folds: List[Tuple[np.ndarray, np.ndarray]], ecg_filenames: np.ndarray, y_true: np.ndarray, y_prob_by_fold: Dict[int, np.ndarray], class_names: List[str]) -> pd.DataFrame:
    """
    folds: list of (train_idx, val_idx)
    y_prob_by_fold: dict fold_idx -> (N_val, C) probs aligned to folds[fold_idx][1]
    Returns a long table with per-fold, per-class AUROC/AUPRC (good for aggregation).
    """
    rows = []
    for fidx, (tr, va) in enumerate(folds):
        yp = y_prob_by_fold[fidx]
        yt = y_true[va]
        for j, cname in enumerate(class_names):
            try:
                auroc = roc_auc_score(yt[:, j], yp[:, j]) if np.unique(yt[:, j]).size > 1 else np.nan
            except Exception:
                auroc = np.nan
            try:
                auprc = average_precision_score(yt[:, j], yp[:, j])
            except Exception:
                auprc = np.nan
            rows.append({"fold": fidx, "class": cname, "prevalence": float(yt[:, j].mean()), "auroc": auroc, "auprc": auprc, "n_val": int(len(yt))})
    return pd.DataFrame(rows)

def make_eval_dataframe_for_fold(val_idx: np.ndarray, ecg_filenames: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], meta_df: Optional[pd.DataFrame] = None, file_col: str = "filename") -> pd.DataFrame:
    """
    Returns a tidy dataframe with one row per recording in the validation fold, containing:
    - filename, per-class labels and probs (wide), and optionally merged metadata (sex, age, etc.).
    """
    fn = np.asarray(ecg_filenames[val_idx], dtype=object)
    df = pd.DataFrame({"filename": fn})
    # attach labels and probs
    for j, cname in enumerate(class_names):
        df[f"y_{cname}"] = y_true[val_idx, j].astype(int)
        df[f"p_{cname}"] = y_prob[:, j].astype(float)
    if meta_df is not None:
        df = df.merge(meta_df, left_on="filename", right_on=file_col, how="left")
    return df

def subgroup_table_binary(df: pd.DataFrame, label_col: str, prob_col: str, group_col: str, threshold: Optional[float] = None) -> pd.DataFrame:
    """
    For a chosen binary target (one class vs rest), compute metrics by subgroup.
    """
    y_true = df[label_col].values.astype(int)
    y_prob = df[prob_col].values.astype(float)
    if threshold is None:
        threshold = _youden_threshold_binary(y_true, y_prob)
    out = []
    for g, gdf in df.groupby(group_col):
        if len(gdf) < 20:
            continue
        yt = gdf[label_col].values.astype(int)
        yp = (gdf[prob_col].values.astype(float) >= threshold).astype(int)
        cm = confusion_matrix(yt, yp, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(yt, yp)
        sens = recall_score(yt, yp, pos_label=1, zero_division=0)
        spec = _specificity_from_cm(cm)
        try:
            auroc = roc_auc_score(yt, gdf[prob_col].values.astype(float))
        except Exception:
            auroc = np.nan
        try:
            auprc = average_precision_score(yt, gdf[prob_col].values.astype(float))
        except Exception:
            auprc = np.nan
        out.append({group_col: g, "n": len(gdf), "prevalence": float(yt.mean()), "Accuracy": acc, "Sensitivity": sens, "Specificity": spec, "AUROC": auroc, "AUPRC": auprc, "TP": tp, "FP": fp, "TN": tn, "FN": fn})
    return pd.DataFrame(out).sort_values("n", ascending=False)
