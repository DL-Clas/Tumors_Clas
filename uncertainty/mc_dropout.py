import torch
import torch.nn as nn
import numpy as np


DEFAULT_P_RETAIN = 0.8
DEFAULT_N_FORWARD = 50


class MCDropoutModel(nn.Module):
    """Wrapper that enables MC Dropout estimation for BTNet-TS.

    The wrapper inserts dropout layers into the MFS-GD module and runs
    multiple stochastic forward passes at inference time to estimate
    predictive uncertainty.

    Args:
        net: BTNet-TS model instance (MyDiag_Model or subclass).
        p_retain: Retention probability (1 - dropout rate).  Default 0.8
            as specified in Appendix F.1.
        n_forward: Number of stochastic forward passes.  Default 50.
        positions: List of module names within MFS-GD where dropout
            should be inserted.  Default inserts after the tail_conv
            in each GFF (MFS_GD) module.
    """

    def __init__(
        self,
        net: nn.Module,
        p_retain: float = DEFAULT_P_RETAIN,
        n_forward: int = DEFAULT_N_FORWARD,
        positions: list = None,
    ):
        super().__init__()
        self.net = net
        self.p_retain = p_retain
        self.n_forward = n_forward
        self.dropout = nn.Dropout(p=1.0 - p_retain)

        if positions is None:
            self._inject_dropout_into_mfs_gd()

    def _inject_dropout_into_mfs_gd(self):
        """Add dropout after each MFS_GD (GFF) module's tail convolution."""
        for module_name in ['GFF2', 'GFF3', 'GFF4']:
            module = getattr(self.net, module_name, None)
            if module is not None and hasattr(module, 'tail_conv'):
                original_forward = module.tail_conv.forward

                def make_dropout_forward(orig_fn, drop_layer):
                    def new_forward(x):
                        x = orig_fn(x)
                        return drop_layer(x)
                    return new_forward

                module.tail_conv.forward = make_dropout_forward(
                    original_forward, self.dropout,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass through the network.

        At inference time, this enables dropout and performs one
        stochastic pass.  For uncertainty estimation, call
        `predict_with_uncertainty()` instead.

        Args:
            x: Input tensor of shape [B, 3, 224, 224].

        Returns:
            Logits tensor of shape [B, num_classes].
        """
        return self.net(x)

    def predict_with_uncertainty(self, x: torch.Tensor) -> dict:
        """Run multiple stochastic forward passes and aggregate results.

        Args:
            x: Input tensor of shape [B, 3, 224, 224].

        Returns:
            Dictionary with keys:
                'mean_probs': Mean softmax probabilities [B, C].
                'std_probs': Standard deviation of probabilities [B, C].
                'epistemic_uncertainty': Predictive std averaged over
                    classes [B].
                'logits': Mean logits [B, C].
        """
        self.net.train()
        all_probs = []

        with torch.no_grad():
            for _ in range(self.n_forward):
                logits = self.net(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

        self.net.eval()

        all_probs_tensor = torch.stack(all_probs, dim=0)  # [T, B, C]
        mean_probs = all_probs_tensor.mean(dim=0)
        std_probs = all_probs_tensor.std(dim=0)
        epistemic_uncertainty = std_probs.mean(dim=1)

        mean_logits = torch.log(mean_probs + 1e-12)

        return {
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'epistemic_uncertainty': epistemic_uncertainty,
            'logits': mean_logits,
        }


if __name__ == '__main__':
    print("mc_dropout.py — standalone test")
    from net.MyDiagX import MyDiag21_tiny

    device = torch.device('cpu')
    model = MyDiag21_tiny(num_classes=4)
    model.eval()

    mc_model = MCDropoutModel(model, p_retain=0.8, n_forward=10)
    dummy = torch.randn(4, 3, 224, 224)

    result = mc_model.predict_with_uncertainty(dummy)
    print(f"Mean probs shape:  {result['mean_probs'].shape}")
    print(f"Epistemic uncertainty: {result['epistemic_uncertainty'].tolist()}")
    print("Standalone test passed.")
