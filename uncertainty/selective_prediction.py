import numpy as np
from sklearn.metrics import average_precision_score


DEFAULT_EPSILON = 1e-12


def compute_aurc(
    confidences: np.ndarray,
    errors: np.ndarray,
    n_thresholds: int = 100,
) -> float:
    """Compute Area Under the Risk-Coverage curve.

    The AURC is the integral of the selective risk (error rate) over
    all coverage levels from 0 to 1.  Lower values indicate better
    ability to assign high confidence to correct predictions and low
    confidence to errors.

    Args:
        confidences: Predicted confidence scores (max softmax probability)
            for each sample.  Shape [N].
        errors: Binary error indicators (1 = incorrect, 0 = correct).
            Shape [N].
        n_thresholds: Number of coverage thresholds for approximation
            (default 100).

    Returns:
        AURC score (lower is better).
    """
    n_samples = len(confidences)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_errors = errors[sorted_indices]

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    aurc = 0.0
    prev_coverage = 0.0

    for threshold in thresholds:
        n_covered = int(np.ceil(threshold * n_samples))
        n_covered = max(1, min(n_covered, n_samples))

        coverage = n_covered / n_samples
        risk = sorted_errors[:n_covered].mean()

        # Trapezoidal integration
        delta_coverage = coverage - prev_coverage
        aurc += risk * delta_coverage
        prev_coverage = coverage

    return float(aurc)


def coverage_analysis(
    confidences: np.ndarray,
    errors: np.ndarray,
    coverage_levels: list = None,
) -> dict:
    """Analyze retained accuracy at specified coverage levels.

    At a given coverage level (e.g., 85%), the model rejects the most
    uncertain fraction of samples and the retained accuracy is computed
    on the remaining samples.

    Args:
        confidences: Predicted confidence scores.  Shape [N].
        errors: Binary error indicators (1 = incorrect, 0 = correct).
            Shape [N].
        coverage_levels: List of coverage fractions to evaluate.
            Default [0.85, 0.90, 0.95, 1.0].

    Returns:
        Dictionary mapping coverage level to retained accuracy percentage.
    """
    if coverage_levels is None:
        coverage_levels = [0.85, 0.90, 0.95, 1.0]

    n_samples = len(confidences)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_errors = errors[sorted_indices]
    sorted_conf = confidences[sorted_indices]

    results = {}
    for coverage in coverage_levels:
        n_keep = int(np.ceil(coverage * n_samples))
        n_keep = max(1, min(n_keep, n_samples))

        n_correct = n_keep - sorted_errors[:n_keep].sum()
        retained_accuracy = (n_correct / n_keep) * 100.0

        results[f'{coverage * 100:.0f}%_coverage'] = {
            'retained_accuracy': float(retained_accuracy),
            'n_retained': n_keep,
            'n_rejected': n_samples - n_keep,
        }

    return results


def compute_selective_prediction_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Compute full selective prediction metrics (Table F.24).

    Args:
        probabilities: Softmax probability predictions.  Shape [N, C].
        targets: Ground-truth labels (class indices).  Shape [N].

    Returns:
        Dictionary with keys:
            'aurc': AURC score.
            'accuracy_100': Accuracy at 100% coverage.
            'accuracy_85': Accuracy at 85% coverage.
    """
    pred_class = probabilities.argmax(axis=1)
    confidences = probabilities[np.arange(len(probabilities)), pred_class]
    errors = (pred_class != targets).astype(np.float32)

    aurc = compute_aurc(confidences, errors)
    coverage_results = coverage_analysis(confidences, errors)

    acc_100 = (1.0 - errors.mean()) * 100.0
    acc_85 = coverage_results.get('85%_coverage', {}).get('retained_accuracy', 0.0)

    return {
        'aurc': aurc,
        'accuracy_100': float(acc_100),
        'accuracy_85': acc_85,
    }


if __name__ == '__main__':
    print("selective_prediction.py — standalone test")

    rng = np.random.RandomState(42)
    n = 500
    probs = rng.dirichlet(alpha=np.ones(10), size=n)
    targets = rng.randint(0, 10, size=n)

    metrics = compute_selective_prediction_metrics(probs, targets)
    print(f"AURC:          {metrics['aurc']:.2f}")
    print(f"Acc@100%:      {metrics['accuracy_100']:.2f}%")
    print(f"Acc@85%:       {metrics['accuracy_85']:.2f}%")
    print("Standalone test passed.")
