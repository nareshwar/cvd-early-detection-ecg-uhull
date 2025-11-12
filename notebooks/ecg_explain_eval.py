
"""
ECG explanation evaluator (class‑aware windows, HR‑adaptive)
-----------------------------------------------------------
Adds:
 - 'qrs_term' (terminal QRS) and 'qrs_on' (QRS onset) windows
 - HR‑aware P/QRS timings
 - Class‑specific window selection wired to your 27 labels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks

LEADS12 = ("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")

# ---------- I/O ----------
def load_mat_TF(mat_path: str) -> np.ndarray:
    d = loadmat(mat_path, squeeze_me=True)
    for k in ("val","ECG","ecg","data","signal"):
        if k in d:
            arr = np.asarray(d[k])
            break
    else:
        raise ValueError(f"No ECG array found in {mat_path}")
    if arr.ndim == 1:
        arr = arr[None, :]
    x = arr.T if arr.shape[0] in (8,12) else arr
    if x.shape[0] < x.shape[1]:
        x = x.T
    return x.astype(np.float32, copy=False)

# ---------- R‑peaks ----------
def _bandpass(x: np.ndarray, fs: float, lo: float = 5.0, hi: float = 18.0, order: int = 3) -> np.ndarray:
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x, axis=0)

def _moving_integral(sig: np.ndarray, fs: float, w_sec: float = 0.150) -> np.ndarray:
    w = max(1, int(round(w_sec*fs)))
    ker = np.ones(w, dtype=np.float32) / w
    out = np.empty_like(sig)
    for j in range(sig.shape[1]):
        out[:, j] = np.convolve(sig[:, j], ker, mode="same")
    return out

def detect_rpeaks(x: np.ndarray, fs: float, prefer: Sequence[str] = ("II","V2","V3"),
                  lead_names: Optional[Sequence[str]] = None, refractory_ms: int = 250,
                  min_prom: Optional[float] = None) -> np.ndarray:
    T, F = x.shape
    if lead_names is not None:
        chosen = [lead_names.index(L) for L in prefer if L in lead_names]
        if not chosen:
            chosen = [int(np.argmax(np.std(x, axis=0)))]
    else:
        chosen = [0]
    xb = _bandpass(x[:, chosen], fs)
    diff = np.diff(xb, axis=0, prepend=xb[:1])
    sq = diff**2
    feat = _moving_integral(sq, fs, w_sec=0.15).sum(axis=1)
    thr = np.median(feat) + 0.5*np.median(np.abs(feat - np.median(feat)))
    distance = int(round(refractory_ms/1000 * fs))
    if min_prom is None:
        min_prom = 0.5*np.std(feat)
    peaks, _ = find_peaks(feat, height=thr, distance=distance, prominence=min_prom)
    # refine
    ref, win = [], int(0.08*fs)
    xb_sum = xb.sum(axis=1)
    for p in peaks:
        s = max(0, p-win); e = min(T, p+win+1)
        ref.append(s + int(np.argmax(xb_sum[s:e])))
    return np.array(sorted(set(ref)), dtype=int)

# ---------- Windows ----------
@dataclass
class BeatWindows:
    pre: List[Tuple[float,float]]       # P / pre‑QRS
    qrs: List[Tuple[float,float]]       # early/mid QRS
    qrs_term: List[Tuple[float,float]]  # terminal QRS (R’ / terminal S)
    qrs_on: List[Tuple[float,float]]    # QRS onset (for Q‑wave)
    stt: List[Tuple[float,float]]       # early ST/T
    tlate: List[Tuple[float,float]]     # late T tail
    pace: List[Tuple[float,float]]      # pacer spike region
    beat: List[Tuple[float,float]]      # coarse whole‑beat

def _median_hr_bpm(r_sec: Sequence[float]) -> float:
    if len(r_sec) < 2: return 70.0
    rr = np.diff(np.asarray(r_sec, float))
    rr = rr[(rr > 0.2) & (rr < 2.5)]
    if len(rr) == 0: return 70.0
    return float(60.0 / np.median(rr))

def build_windows_from_rpeaks(r_sec: Sequence[float], *, class_name: Optional[str] = None) -> BeatWindows:
    """HR‑aware windows. Tighten pre‑QRS at high HR; widen terminal QRS for BBB."""
    hr = _median_hr_bpm(r_sec)
    # defaults
    pre_lo  = 0.22 if hr < 100 else 0.18
    pre_hi  = 0.05 if hr < 100 else 0.04
    q_on_lo = 0.06  # look back to capture Q onset (R-60ms)
    q_on_hi = 0.01  # until 10ms before R

    qrs_early_post = 0.08
    qrs_term_post  = 0.16
    if class_name and ("bundle branch block" in class_name or "intraventricular" in class_name):
        qrs_term_post = 0.20  # wider terminal window for conduction disease

    pre, qrs, qrs_term, qrs_on, stt, tlate, pace, beat = [], [], [], [], [], [], [], []
    for r in r_sec:
        pre.append((r - pre_lo,   r - pre_hi))
        qrs_on.append((r - q_on_lo, r - q_on_hi))
        qrs.append((r - 0.05,     r + qrs_early_post))
        qrs_term.append((r + 0.04, r + qrs_term_post))
        stt.append((r + 0.06,     r + 0.20))
        tlate.append((r + 0.20,   r + 0.40))
        pace.append((r - 0.01,    r + 0.02))
        beat.append((r - 0.30,    r + 0.40))
    return BeatWindows(pre, qrs, qrs_term, qrs_on, stt, tlate, pace, beat)

# ---------- Registry (27 classes) ----------
@dataclass(frozen=True)
class ClassConfig:
    strict_leads: Tuple[str, ...]
    lenient_leads: Tuple[str, ...]
    window_keys: Tuple[str, ...]

REGISTRY: Dict[str, ClassConfig] = {
    # Atrial / sinus
    "sinus rhythm": ClassConfig(("II","V1"), ("II","V1","I","aVF","V2"), ("pre",)),
    "sinus tachycardia": ClassConfig(("II","V1"), ("II","V1","I","aVF","V2"), ("pre",)),
    "sinus bradycardia": ClassConfig(("II","V1"), ("II","V1","I","aVF","V2"), ("pre",)),
    "sinus arrhythmia": ClassConfig(("II","V1"), ("II","V1","I","aVF","V2"), ("pre",)),
    "bradycardia": ClassConfig(("II","V1"), ("II","V1","I","aVF"), ("pre",)),

    "atrial fibrillation": ClassConfig(("II","V1"), ("II","V1","V2","V3","aVF"), ("pre",)),
    "atrial flutter":      ClassConfig(("II","V1"), ("II","III","aVF","V1","V2"), ("pre",)),

    # AV conduction
    "1st degree av block":       ClassConfig(("II","V5","V6","I","aVL"), ("II","I","aVL","V5","V6","V2"), ("pre",)),
    "prolonged pr interval":     ClassConfig(("II","V5","V6","I","aVL"), ("II","I","aVL","V5","V6","V2"), ("pre",)),

    # Ventricular conduction / fascicles / axis
    "right bundle branch block":        ClassConfig(("V1","V2","I","V6"), ("V1","V2","aVR","I","V6","V3"), ("qrs","qrs_term")),
    "incomplete right bundle branch block": ClassConfig(("V1","V2","I","V6"), ("V1","V2","aVR","I","V6","V3"), ("qrs","qrs_term")),
    "complete right bundle branch block":   ClassConfig(("V1","V2","I","V6"), ("V1","V2","aVR","I","V6","V3"), ("qrs","qrs_term")),
    "left bundle branch block":        ClassConfig(("V5","V6","I","aVL"), ("V4","V5","V6","I","aVL"), ("qrs","qrs_term")),
    "nonspecific intraventricular conduction disorder": ClassConfig(("V1","V2","V5","V6","I","aVL"), ("V1","V2","V3","V4","V5","V6"), ("qrs","qrs_term")),
    "left anterior fascicular block":  ClassConfig(("I","aVL"), ("I","aVL","II"), ("qrs",)),
    "left axis deviation":             ClassConfig(("I","aVL"), ("I","aVL","II"), ("qrs",)),
    "right axis deviation":            ClassConfig(("aVF","III"), ("aVF","III","II"), ("qrs",)),

    # Repolarization / QT
    "prolonged qt interval": ClassConfig(("II","V5"), ("II","V5","V6","I"), ("tlate",)),
    "t wave abnormal":       ClassConfig(("V2","V3","V4","V5","V6"), ("V2","V3","V4","V5","V6","II","I"), ("stt","tlate")),
    "t wave inversion":      ClassConfig(("V2","V3","V4","V5","V6"), ("V1","V2","V3","V4","V5","V6","II"), ("stt","tlate")),

    # Q / voltage
    "qwave abnormal":        ClassConfig(("V1","V2","V3","V4","II","III","aVF"), ("V1","V2","V3","V4","V5","V6","II","III","aVF","I","aVL"), ("qrs_on",)),
    "low qrs voltages":      ClassConfig(("I","II","III","aVR","aVL","aVF"), ("I","II","III","aVR","aVL","aVF"), ("beat",)),

    # Ectopy / SVPB / VPC
    "premature atrial contraction":            ClassConfig(("II","V1"), ("II","V1","I","aVF","V2"), ("pre",)),
    "supraventricular premature beats":        ClassConfig(("II","V1"), ("II","V1","I","aVF","V2"), ("pre",)),
    "ventricular premature beats":             ClassConfig(("V1","V2","V3"), ("V1","V2","V3","II","V4"), ("qrs","qrs_term")),
    "premature ventricular contractions":      ClassConfig(("V1","V2","V3"), ("V1","V2","V3","II","V4"), ("qrs","qrs_term")),

    # Pacing
    "pacing rhythm":           ClassConfig(("V1","V2","II"), ("V1","V2","II","V3","V5","V6"), ("pace","qrs")),
}

# ---------- AUC helpers ----------
def _overlap(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    s = max(a[0], b[0]); e = min(a[1], b[1])
    return max(0.0, e - s)

def integrate_attribution(perlead_spans: Dict[str, List[Tuple[float,float,float]]],
                          tokens: List[Tuple[str, Tuple[float,float]]]) -> np.ndarray:
    scores = np.zeros(len(tokens), dtype=np.float64)
    for i, (lead, (s, e)) in enumerate(tokens):
        spans = perlead_spans.get(lead, ())
        if not spans: continue
        acc = 0.0
        for (a,b,w) in spans:
            ov = _overlap((s,e), (a,b))
            if ov > 0:
                acc += ov * float(w)
        scores[i] = acc
    return scores

def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(int)
    P = labels.sum(); N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores)+1)
    for s in np.unique(scores):
        idx = np.where(scores == s)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - P*(P+1)/2.0) / (P * N)
    return float(auc)

# ---------- Tokens / labels ----------
def build_tokens(lead_names: Sequence[str],
                 windows: BeatWindows,
                 which: Iterable[str]) -> List[Tuple[str, Tuple[float,float]]]:
    key_to_list = {
        "pre": windows.pre, "qrs": windows.qrs, "qrs_term": windows.qrs_term, "qrs_on": windows.qrs_on,
        "stt": windows.stt, "tlate": windows.tlate, "pace": windows.pace, "beat": windows.beat
    }
    time_windows: List[Tuple[float,float]] = []
    for k in which:
        time_windows.extend(key_to_list[k])
    tokens = []
    for L in lead_names:
        for w in time_windows:
            tokens.append((L, w))
    return tokens

def make_labels_for_tokens(tokens: List[Tuple[str, Tuple[float,float]]],
                           positives_leads: Sequence[str],
                           positive_windows: Iterable[str],
                           windows: BeatWindows) -> np.ndarray:
    win_map = {}
    for k, lst in (("pre", windows.pre), ("qrs", windows.qrs), ("qrs_term", windows.qrs_term),
                   ("qrs_on", windows.qrs_on), ("stt", windows.stt),
                   ("tlate", windows.tlate), ("pace", windows.pace), ("beat", windows.beat)):
        for w in lst:
            win_map[w] = k
    labels = np.zeros(len(tokens), dtype=int)
    positives_leads = set(positives_leads)
    positive_windows = set(positive_windows)
    for i, (lead, win) in enumerate(tokens):
        if (lead in positives_leads) and (win_map.get(win) in positive_windows):
            labels[i] = 1
    return labels

@dataclass
class AttAUCResult:
    strict_auc: float
    lenient_auc: float
    n_tokens: int

def token_level_attauc(perlead_spans: Dict[str, List[Tuple[float,float,float]]],
                       class_name: str,
                       r_sec: Sequence[float],
                       lead_names: Sequence[str] = LEADS12) -> AttAUCResult:
    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)
    tokens = build_tokens(lead_names, windows, which=("pre","qrs","qrs_term","qrs_on","stt","tlate","pace","beat"))
    scores = integrate_attribution(perlead_spans, tokens)
    y_strict = make_labels_for_tokens(tokens, cfg.strict_leads, cfg.window_keys, windows)
    y_lenient = make_labels_for_tokens(tokens, cfg.lenient_leads, cfg.window_keys, windows)
    auc_s = rank_auc(scores, y_strict)
    auc_l = rank_auc(scores, y_lenient)
    return AttAUCResult(strict_auc=auc_s, lenient_auc=auc_l, n_tokens=len(tokens))

# ---------- Deletion ----------
def _apply_deletions(x: np.ndarray, fs: float,
                     deletions: List[Tuple[str, Tuple[float,float]]],
                     lead_names: Sequence[str],
                     baseline: str = "zero") -> np.ndarray:
    y = x.copy()
    if baseline == "mean":
        base = y.mean(axis=0)
    elif baseline == "zero":
        base = np.zeros(y.shape[1], dtype=y.dtype)
    else:
        base = y.mean(axis=0)
    name_to_idx = {L:i for i,L in enumerate(lead_names)}
    for L, (s,e) in deletions:
        i = name_to_idx.get(L, None)
        if i is None: continue
        s_i = max(0, int(round(s * fs)))
        e_i = min(y.shape[0], int(round(e * fs)))
        if s_i < e_i:
            y[s_i:e_i, i] = base[i]
    return y

@dataclass
class DeletionCurve:
    fractions: List[float]
    delta_prob_positive: List[float]
    delta_prob_control: List[float]

def targeted_deletion_curve(x: np.ndarray, fs: float,
                            perlead_spans: Dict[str, List[Tuple[float,float,float]]],
                            class_name: str,
                            r_sec: Sequence[float],
                            predict_proba: Callable[[np.ndarray], float],
                            fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3),
                            lead_names: Sequence[str] = LEADS12,
                            baseline: str = "zero",
                            rng: int = 0) -> DeletionCurve:
    rng = np.random.default_rng(rng)
    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)
    tokens = build_tokens(lead_names, windows, which=("pre","qrs","qrs_term","qrs_on","stt","tlate","pace","beat"))
    scores = integrate_attribution(perlead_spans, tokens)
    y_pos = make_labels_for_tokens(tokens, cfg.lenient_leads, cfg.window_keys, windows)
    pos_idx = np.where(y_pos == 1)[0]
    if len(pos_idx) == 0:
        return DeletionCurve(list(fractions), [0.0]*len(fractions), [0.0]*len(fractions))
    order = pos_idx[np.argsort(scores[pos_idx])[::-1]]
    durations = np.array([t[1][1] - t[1][0] for t in tokens], dtype=float)
    p0 = float(predict_proba(x))
    deltas_pos, deltas_ctl = [], []
    for frac in fractions:
        total_pos_dur = durations[pos_idx].sum()
        target_dur = frac * total_pos_dur
        cum = 0.0; chosen = []
        for i in order:
            chosen.append(i); cum += durations[i]
            if cum >= target_dur: break
        deletions_pos = [tokens[i] for i in chosen]
        neg_idx = np.where(y_pos == 0)[0]
        rng.shuffle(neg_idx)
        cum = 0.0; chosen_ctl = []
        for i in neg_idx:
            chosen_ctl.append(i); cum += durations[i]
            if cum >= target_dur: break
        deletions_ctl = [tokens[i] for i in chosen_ctl]
        x_pos = _apply_deletions(x, fs, deletions_pos, lead_names, baseline=baseline)
        x_ctl = _apply_deletions(x, fs, deletions_ctl, lead_names, baseline=baseline)
        dp_pos = float(p0 - float(predict_proba(x_pos)))
        dp_ctl = float(p0 - float(predict_proba(x_ctl)))
        deltas_pos.append(dp_pos); deltas_ctl.append(dp_ctl)
    return DeletionCurve(list(map(float, fractions)),
                         list(map(float, deltas_pos)),
                         list(map(float, deltas_ctl)))

# ---------- Main ----------
@dataclass
class EvaluationOutput:
    strict_attauc: float
    lenient_attauc: float
    n_tokens: int
    deletion_curve: Optional[DeletionCurve]

def evaluate_explanation(mat_path: str,
                         fs: float,
                         payload: Dict,
                         class_name: str,
                         rpeaks_sec: Optional[Sequence[float]] = None,
                         lead_names: Sequence[str] = LEADS12,
                         model_predict_proba: Optional[Callable[[np.ndarray], float]] = None,
                         deletion_fractions: Sequence[float] = (0.05,0.1,0.2,0.3),
                         baseline: str = "zero",
                         rng: int = 0) -> EvaluationOutput:
    x = load_mat_TF(mat_path)
    raw_spans = payload.get("perlead_spans", {})
    perlead_spans = {str(L): [(float(s), float(e), float(w)) for (s,e,w) in spans]
                     for L,spans in raw_spans.items()}
    if rpeaks_sec is None:
        r_idx = detect_rpeaks(x, fs, lead_names=lead_names)
        r_sec = (r_idx / float(fs)).tolist()
    else:
        r_sec = list(map(float, rpeaks_sec))
    att = token_level_attauc(perlead_spans, class_name, r_sec, lead_names=lead_names)
    if model_predict_proba is not None:
        curve = targeted_deletion_curve(
            x, fs, perlead_spans, class_name, r_sec,
            predict_proba=model_predict_proba,
            fractions=deletion_fractions,
            lead_names=lead_names,
            baseline=baseline,
            rng=rng
        )
    else:
        curve = None
    return EvaluationOutput(
        strict_attauc=att.strict_auc,
        lenient_attauc=att.lenient_auc,
        n_tokens=att.n_tokens,
        deletion_curve=curve
    )
