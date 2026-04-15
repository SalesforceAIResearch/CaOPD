"""
Calibration metrics: Brier Score, ECE, SPR (Strict Pairwise Ranking), and bin statistics.

SPR is introduced in our paper to replace AUROC for measuring confidence discrimination.
Unlike AUROC which gives 0.5 credit to ties (P(c+ > c-) + 0.5 * P(c+ = c-)),
SPR strictly measures P(c+ > c-), assigning zero credit when confidences are tied.
This heavily penalizes confidence saturation (e.g., always outputting c=1.0).

Only numpy is required. No scikit-learn dependency.
"""
import numpy as np


def get_brier(correctness: np.ndarray, confidence: np.ndarray) -> float:
    """Brier Score: mean squared error between confidence and correctness."""
    return float(np.mean((confidence - correctness) ** 2))


def get_ece(correctness: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) with equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_indices = np.digitize(confidence, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_conf = np.mean(confidence[mask])
            bin_acc = np.mean(correctness[mask])
            bin_weight = np.sum(mask) / len(confidence)
            ece += bin_weight * np.abs(bin_conf - bin_acc)
    return float(ece)


def get_spr(correctness: np.ndarray, confidence: np.ndarray) -> float:
    """Strict Pairwise Ranking (SPR): P(c+ > c-).

    For every (correct, incorrect) pair, SPR counts the fraction where the correct
    response has *strictly* higher confidence.  Ties get zero credit.

    A saturated model (always c=1.0) scores SPR = 0, clearly exposing the collapse
    that AUROC (= 0.5 for the same model) obscures.

    Reference: CaOPD paper, Appendix D.2.
    """
    if np.nanmin(correctness) == np.nanmax(correctness):
        return 0.0
    pos = confidence[correctness >= 0.5]
    neg = confidence[correctness < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(np.mean(pos[:, None] > neg[None, :]))


def get_overconfidence_gap(correctness: np.ndarray, confidence: np.ndarray) -> float:
    """Overconfidence Gap (OCG): mean_confidence - accuracy.  Positive = overconfident."""
    return float(np.mean(confidence) - np.mean(correctness))


def bin_stats(
    correctness: np.ndarray, confidence: np.ndarray, n_bins: int = 10
) -> list:
    """Per-bin statistics: count, mean_confidence, accuracy."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_indices = np.digitize(confidence, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    out = []
    for i in range(n_bins):
        mask = bin_indices == i
        n = int(np.sum(mask))
        if n == 0:
            out.append({
                "bin": i, "low": i / n_bins, "high": (i + 1) / n_bins,
                "count": 0, "mean_conf": None, "accuracy": None,
            })
            continue
        out.append({
            "bin": i,
            "low": i / n_bins,
            "high": (i + 1) / n_bins,
            "count": n,
            "mean_conf": float(np.mean(confidence[mask])),
            "accuracy": float(np.mean(correctness[mask])),
        })
    return out
