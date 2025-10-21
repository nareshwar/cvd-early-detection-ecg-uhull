
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

__all__ = [
    "overlay_signal_with_importance",
    "overlay_intervals",
    "overlay_spectrum_with_importance",
    "importance_colorbar"
]

def _extent_for_background(x, y):
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    return (float(np.nanmin(x)), float(np.nanmax(x)), ymin - pad, ymax + pad)

def overlay_signal_with_importance(t, y, importance, *, cmap="RdBu_r", alpha=0.35, signed=True, ax=None, title="ECG with importance overlay"):
    """
    Paint a smooth heat-strip behind a 1D signal and draw the signal on top.
    t, y, importance: (N,) arrays. importance can be signed (positive/negative) or nonnegative.
    """
    t = np.asarray(t).ravel()
    y = np.asarray(y).ravel()
    imp = np.asarray(importance).ravel()
    assert t.shape == y.shape == imp.shape, "t, y, importance must have the same shape"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    extent = _extent_for_background(t, y)
    if signed:
        vmax = float(np.nanmax(np.abs(imp))) or 1.0
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = float(np.nanmax(imp)) or 1.0

    ax.imshow(imp[None, :], extent=extent, aspect="auto",
              cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, origin="lower")
    ax.plot(t, y, linewidth=1.2)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ECG [mV]")
    ax.set_title(title)
    return ax

def overlay_intervals(t, y, intervals, *, cmap="RdBu_r", alpha=0.35, ax=None, title="ECG with highlighted intervals"):
    """
    intervals: list of (start_t, end_t, weight). Positive -> red, negative -> blue.
    """
    t = np.asarray(t).ravel()
    y = np.asarray(y).ravel()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(t, y, linewidth=1.2)
    if intervals:
        vmax = max(abs(w) for _, _, w in intervals) or 1.0
    else:
        vmax = 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap_obj = get_cmap(cmap)
    for s, e, w in intervals:
        ax.axvspan(s, e, color=cmap_obj(norm(w)), alpha=alpha)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ECG [mV]")
    ax.set_title(title)
    return ax

def overlay_spectrum_with_importance(f, amplitude, importance, *, cmap="RdBu_r", alpha=0.35, signed=True, ax=None, title="Spectrum + importance"):
    f = np.asarray(f).ravel()
    amp = np.asarray(amplitude).ravel()
    imp = np.asarray(importance).ravel()
    assert f.shape == amp.shape == imp.shape, "f, amplitude, importance must have same shape"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    extent = _extent_for_background(f, amp)
    if signed:
        vmax = float(np.nanmax(np.abs(imp))) or 1.0
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = float(np.nanmax(imp)) or 1.0

    ax.imshow(imp[None, :], extent=extent, aspect="auto",
              cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, origin="lower")
    ax.plot(f, amp, linewidth=1.0)
    ax.set_xlabel("f [kHz]")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    return ax

def importance_colorbar(ax, vmin, vmax, *, cmap="RdBu_r", label="importance"):
    import matplotlib as mpl
    mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    plt.colorbar(mappable, ax=ax, label=label)


def overlay_lime_timeshap(
    t, y, shap_imp, lime_intervals=None, *,
    shap_cmap="RdBu_r", shap_alpha=0.35, shap_signed=True,
    lime_alpha=0.35, lime_cmap="Reds",
    title="ECG + TimeSHAP (background) + LIME intervals",
    ax=None
):
    """
    Combined overlay:
      - TimeSHAP per-timestep importance is shown as a smooth background strip.
      - LIME intervals are drawn as translucent bands on top.

    Parameters
    ----------
    t, y : (N,) arrays
    shap_imp : (N,) array of per-step importances (signed or nonnegative)
    lime_intervals : list of (start_t, end_t, weight) where weight>=0 (or signed if you wish)
    shap_cmap, lime_cmap : colormaps for SHAP and LIME overlays
    shap_alpha, lime_alpha : transparencies
    shap_signed : if True, center SHAP color scale around zero
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap

    t = np.asarray(t).ravel()
    y = np.asarray(y).ravel()
    shap_imp = np.asarray(shap_imp).ravel()
    assert t.shape == y.shape == shap_imp.shape, "t, y, shap_imp must have the same shape"

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3))

    # Background SHAP
    extent = _extent_for_background(t, y)
    if shap_signed:
        vmax = float(np.nanmax(np.abs(shap_imp))) or 1.0
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = float(np.nanmax(shap_imp)) or 1.0

    ax.imshow(shap_imp[None, :], extent=extent, aspect="auto",
              cmap=shap_cmap, vmin=vmin, vmax=vmax, alpha=shap_alpha, origin="lower")

    # ECG trace
    ax.plot(t, y, linewidth=1.2)

    # LIME intervals
    if lime_intervals:
        # If any interval weights are provided, normalize by max(abs(w))
        try:
            wmax = max(abs(w) for _, _, w in lime_intervals) or 1.0
        except Exception:
            wmax = 1.0
        norm = Normalize(vmin=0.0, vmax=wmax)  # assume nonnegative for visibility; if signed this still works with abs
        cmap_obj = get_cmap(lime_cmap)
        for s, e, w in lime_intervals:
            ax.axvspan(s, e, color=cmap_obj(norm(abs(w))), alpha=lime_alpha, linewidth=0)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("ECG [mV]")
    ax.set_title(title)
    return ax, (vmin, vmax)
