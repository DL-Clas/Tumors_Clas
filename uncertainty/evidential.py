import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


DEFAULT_EPSILON = 1e-10
DEFAULT_ANNEALING_STEP = 10


def relu_evidence(logits: torch.Tensor) -> torch.Tensor:
    """Apply ReLU activation to produce non-negative evidence."""
    return F.relu(logits)


def softplus_evidence(logits: torch.Tensor) -> torch.Tensor:
    """Apply Softplus activation to produce non-negative evidence.

    Used in the implementation as specified in Appendix F.1:
    'the final Softmax activation was replaced with a non-negative
    activation function (Softplus) to output evidence values'.
    """
    return F.softplus(logits)


class EvidentialLoss(nn.Module):
    """Dirichlet-based evidential loss function.

    Computes the loss as: L = L_ace + lambda_t * L_KL

    where L_ace is the sum of squared error (accuracy term) and L_KL
    is the KL divergence between the Dirichlet distribution and the
    uniform Dirichlet prior (to penalize incorrect evidence).

    The annealing coefficient lambda_t increases over epochs to prevent
    the KL term from dominating early training.

    Args:
        annealing_step: Epoch at which lambda_t reaches 1.0 (default 10).
        device: torch.device.
    """

    def __init__(self, annealing_step: int = DEFAULT_ANNEALING_STEP, device=None):
        super().__init__()
        self.annealing_step = annealing_step
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(
        self,
        evidence: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        """Compute evidential loss.

        Args:
            evidence: Non-negative evidence tensor. Shape [B, C].
            target: Ground-truth labels (class indices). Shape [B].
            epoch: Current training epoch for annealing schedule.

        Returns:
            Scalar loss value.
        """
        alpha = evidence + 1.0
        alpha_0 = alpha.sum(dim=1, keepdim=True)
        S = alpha_0

        # Accuracy term (sum of squares)
        one_hot = F.one_hot(target, num_classes=evidence.size(1)).float()
        L_ace = (one_hot - alpha / S).pow(2).sum(dim=1).mean()

        # KL divergence term (annealed)
        annealing_coef = min(1.0, epoch / self.annealing_step)

        K = evidence.size(1)
        alpha_tilde = one_hot + (1.0 - one_hot) * alpha
        alpha_tilde_0 = alpha_tilde.sum(dim=1, keepdim=True)

        # KL( Dir(alpha_tilde) || Dir(1, 1, ..., 1) )
        digamma_alpha_tilde = torch.digamma(alpha_tilde)
        digamma_alpha_tilde_0 = torch.digamma(alpha_tilde_0)

        term1 = torch.lgamma(alpha_tilde_0) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        term2 = (alpha_tilde - 1.0) * (digamma_alpha_tilde - digamma_alpha_tilde_0)

        L_kl = (term1 + term2.sum(dim=1, keepdim=True)).squeeze().mean()

        total_loss = L_ace + annealing_coef * L_kl
        return total_loss


class EvidentialModel(nn.Module):
    """Wrapper for BTNet-TS with evidential output layer.

    Replaces the final fc layer's Softmax with a Softplus activation
    to produce evidence values.  Provides methods to extract epistemic
    and aleatoric uncertainty from the Dirichlet distribution.

    Args:
        net: BTNet-TS model instance.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.evidence_activation = softplus_evidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting evidence (not probabilities).

        Args:
            x: Input tensor of shape [B, 3, 224, 224].

        Returns:
            Evidence tensor of shape [B, C] (non-negative).
        """
        logits = self.net(x)
        evidence = self.evidence_activation(logits)
        return evidence

    def predict(self, x: torch.Tensor) -> dict:
        """Predict with full uncertainty decomposition.

        Args:
            x: Input tensor of shape [B, 3, 224, 224].

        Returns:
            Dictionary with keys:
                'probs': Predicted probabilities (Dirichlet mean) [B, C].
                'evidence': Evidence values [B, C].
                'aleatoric_uncertainty': Expected data uncertainty [B].
                'epistemic_uncertainty': Distributional uncertainty [B].
                'total_uncertainty': Predictive uncertainty [B].
                'prediction': Most likely class [B].
        """
        with torch.no_grad():
            evidence = self.forward(x)
            alpha = evidence + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            probs = alpha / S

            # Predictive entropy (total uncertainty): H(E[p])
            # Aleatoric: expected entropy of the categorical under Dirichlet E[H(p)]
            # Epistemic: mutual information = H(E[p]) - E[H(p)]
            alpha_0 = S.squeeze()
            K = evidence.size(1)

            expected_p = probs
            log_expected_p = torch.log(expected_p + DEFAULT_EPSILON)
            total_uncertainty = -(expected_p * log_expected_p).sum(dim=1)

            # E[H(p)] — expected categorical entropy under Dirichlet (aleatoric)
            digamma_alpha = torch.digamma(alpha)
            digamma_alpha_0 = torch.digamma(alpha_0)
            aleatoric = -(alpha / alpha_0.unsqueeze(1)) * (
                digamma_alpha - digamma_alpha_0.unsqueeze(1)
            )
            aleatoric = aleatoric.sum(dim=1)

            # Mutual information (epistemic)
            epistemic = total_uncertainty - aleatoric

        return {
            'probs': probs.cpu(),
            'evidence': evidence.cpu(),
            'aleatoric_uncertainty': aleatoric.cpu(),
            'epistemic_uncertainty': epistemic.cpu(),
            'total_uncertainty': total_uncertainty.cpu(),
            'prediction': probs.argmax(dim=1).cpu(),
        }


if __name__ == '__main__':
    print("evidential.py — standalone test")
    from net.MyDiagX import MyDiag21_tiny

    device = torch.device('cpu')
    model = MyDiag21_tiny(num_classes=4)
    ev_model = EvidentialModel(model)
    ev_model.eval()

    dummy = torch.randn(4, 3, 224, 224)
    result = ev_model.predict(dummy)
    print(f"Probs shape:          {result['probs'].shape}")
    print(f"Evidence shape:       {result['evidence'].shape}")
    print(f"Aleatoric uncertainty: {result['aleatoric_uncertainty'].tolist()}")
    print(f"Epistemic uncertainty: {result['epistemic_uncertainty'].tolist()}")

    # Test loss
    criterion = EvidentialLoss(annealing_step=5)
    evidence = ev_model(dummy)
    loss = criterion(evidence, torch.randint(0, 4, (4,)), epoch=5)
    print(f"Evidential loss: {loss.item():.4f}")
    print("Standalone test passed.")
