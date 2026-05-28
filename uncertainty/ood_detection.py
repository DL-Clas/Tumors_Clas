import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


DEFAULT_EPSILON = 1e-12


def compute_ood_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> dict:
    """Compute OOD detection metrics.

    Higher scores indicate in-distribution (ID) confidence; lower scores
    suggest OOD.  The binary classification is: ID=1, OOD=0.

    Args:
        id_scores: Confidence or uncertainty scores for in-distribution
            samples (trained on BTD-4).  Shape [N_id].
        ood_scores: Scores for out-of-distribution samples (BTD-7).
            Shape [N_ood].

    Returns:
        Dictionary with keys:
            'auroc': AUROC score (higher = better separation).
            'aupr': AUPR score (higher = better ID precision).
            'fpr_at_95_tpr': FPR when TPR = 95% (lower = better).
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([
        np.ones_like(id_scores),
        np.zeros_like(ood_scores),
    ])

    auroc = roc_auc_score(labels, scores)

    aupr = average_precision_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    target_tpr = 0.95

    if np.any(tpr >= target_tpr):
        idx = np.where(tpr >= target_tpr)[0][0]
        fpr_at_95 = fpr[idx]
    else:
        fpr_at_95 = 1.0

    return {
        'auroc': float(auroc),
        'aupr': float(aupr),
        'fpr_at_95_tpr': float(fpr_at_95),
    }


def evaluate_ood_detection(
    model,
    id_loader,
    ood_loader,
    device,
    score_type: str = 'confidence',
    n_forward: int = 1,
) -> dict:
    """Evaluate OOD detection performance using a trained model.

    Args:
        model: BTNet-TS model or uncertainty wrapper (e.g., MCDropoutModel,
            EvidentialModel, EnsembleModel).
        id_loader: DataLoader for in-distribution data (BTD-4).
        ood_loader: DataLoader for out-of-distribution data (BTD-7).
        device: torch.device.
        score_type: Type of score to use for OOD detection.
            'confidence' — max softmax probability.
            'entropy' — predictive entropy.
            'epistemic' — epistemic uncertainty (from MC dropout, EDL, etc.)
        n_forward: Number of forward passes for stochastic methods.

    Returns:
        Dictionary from compute_ood_metrics().
    """
    id_scores = []
    ood_scores = []

    model.eval()

    with torch.no_grad():
        for loader, scores_list in [(id_loader, id_scores), (ood_loader, ood_scores)]:
            for batch in loader:
                if len(batch) == 3:
                    images, _, _ = batch
                else:
                    images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(device)

                if hasattr(model, 'predict_with_uncertainty'):
                    result = model.predict_with_uncertainty(images)
                    if score_type == 'confidence':
                        scores = result['mean_probs'].max(dim=1).values
                    elif score_type == 'entropy':
                        probs = result['mean_probs']
                        entropy = -(probs * torch.log(probs + DEFAULT_EPSILON)).sum(dim=1)
                        scores = -entropy  # negate so higher = more ID-like
                    elif score_type == 'epistemic':
                        scores = -result.get('epistemic_uncertainty',
                                              torch.zeros(images.size(0)))
                    else:
                        scores = result['mean_probs'].max(dim=1).values
                else:
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    if score_type == 'confidence':
                        scores = probs.max(dim=1).values
                    elif score_type == 'entropy':
                        entropy = -(probs * torch.log(probs + DEFAULT_EPSILON)).sum(dim=1)
                        scores = -entropy
                    else:
                        scores = probs.max(dim=1).values

                scores_list.extend(scores.cpu().numpy())

    return compute_ood_metrics(
        np.array(id_scores),
        np.array(ood_scores),
    )


if __name__ == '__main__':
    print("ood_detection.py — standalone test")

    rng = np.random.RandomState(42)
    id_scores = rng.beta(10, 1, size=500)
    ood_scores = rng.beta(1, 10, size=200)

    metrics = compute_ood_metrics(id_scores, ood_scores)
    print(f"AUROC:         {metrics['auroc']:.3f}")
    print(f"AUPR:          {metrics['aupr']:.3f}")
    print(f"FPR@95%TPR:    {metrics['fpr_at_95_tpr']:.3f}")
    print("Standalone test passed.")
