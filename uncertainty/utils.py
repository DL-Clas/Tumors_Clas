import numpy as np


DEFAULT_EPSILON = 1e-12
DEFAULT_N_BINS = 15


def compute_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Compute Expected Calibration Error.

    Partitions predictions into `n_bins` equally spaced confidence bins
    and computes the expected absolute difference between accuracy and
    mean confidence in each bin.

    Args:
        probabilities: Predicted softmax probabilities of the true class.
            Shape [N].
        targets: One-hot ground-truth labels.  Shape [N, C] or [N].
            If 1D, treated as class indices.
        n_bins: Number of equally spaced probability bins (default 15).

    Returns:
        ECE score in [0, 1].  Lower is better.
    """
    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must be 2D [N, C], got shape {probabilities.shape}"
        )

    if targets.ndim == 1:
        targets_onehot = np.zeros((len(probabilities), int(probabilities.max()) + 1))
        targets_onehot[np.arange(len(targets)), targets] = 1
        targets = targets_onehot

    pred_class = probabilities.argmax(axis=1)
    confidences = probabilities[np.arange(len(probabilities)), pred_class]

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = np.logical_and(
            confidences > bin_lower,
            confidences <= bin_upper,
        )
        bin_size = in_bin.sum()

        if bin_size > 0:
            bin_accuracy = (pred_class[in_bin] == targets[in_bin].argmax(axis=1)).mean()
            bin_confidence = confidences[in_bin].mean()
            ece += (bin_size / len(confidences)) * np.abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_nll(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute Negative Log-Likelihood.

    NLL = -1/N * sum_i log(p_i(y_true))

    Args:
        probabilities: Predicted softmax probabilities.  Shape [N, C].
        targets: Ground-truth labels as class indices.  Shape [N].

    Returns:
        NLL score.  Lower is better.
    """
    if targets.ndim == 2:
        targets = targets.argmax(axis=1)

    n_samples = len(probabilities)
    log_probs = np.log(probabilities[np.arange(n_samples), targets] + DEFAULT_EPSILON)
    nll = -log_probs.mean()
    return float(nll)


def compute_brier_score(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute Brier Score.

    BS = 1/N * sum_i sum_c (p_i_c - y_i_c)^2

    where y_i_c is the one-hot encoded ground truth.

    Args:
        probabilities: Predicted softmax probabilities.  Shape [N, C].
        targets: Ground-truth labels.  Shape [N] (class indices) or [N, C]
            (one-hot).

    Returns:
        Brier score in [0, 2].  Lower is better.
    """
    if targets.ndim == 1:
        n_classes = probabilities.shape[1]
        targets_onehot = np.zeros((len(targets), n_classes))
        targets_onehot[np.arange(len(targets)), targets] = 1.0
        targets = targets_onehot

    squared_errors = (probabilities - targets) ** 2
    brier = squared_errors.sum(axis=1).mean()
    return float(brier)


if __name__ == '__main__':
    print("utils.py — standalone test")

    rng = np.random.RandomState(42)
    probs = rng.dirichlet(alpha=np.ones(5), size=100)
    targets = rng.randint(0, 5, size=100)

    ece = compute_ece(probs, targets)
    nll = compute_nll(probs, targets)
    brier = compute_brier_score(probs, targets)

    print(f"ECE:  {ece:.4f}")
    print(f"NLL:  {nll:.4f}")
    print(f"Brier: {brier:.4f}")
    print("Standalone test passed.")
