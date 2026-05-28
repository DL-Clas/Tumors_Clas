import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


DEFAULT_EPSILON = 1e-8


class BayesianAttentionLayer(nn.Module):
    """Variational Bayesian wrapper for a 3D tensor attention parameter.

    Replaces a deterministic attention tensor P with a distribution
    q(w | mu, log_var).  At each forward pass, a sample is drawn via
    the reparameterization trick: w = mu + sigma * epsilon.

    Args:
        shape: Shape of the attention tensor (e.g. [1, C, 7, 7]).
        prior_log_var: Prior log-variance for KL regularization (default -5.0).
    """

    def __init__(
        self,
        shape: tuple,
        prior_log_var: float = -5.0,
    ):
        super().__init__()
        self.shape = shape

        self.mu = nn.Parameter(torch.ones(shape))
        self.log_var = nn.Parameter(torch.full(shape, -5.0))

        self.prior_mu = 1.0
        self.prior_log_var = prior_log_var

    def forward(self) -> torch.Tensor:
        """Sample attention weights using reparameterization trick.

        Returns:
            Sampled attention tensor of shape `self.shape`.
        """
        sigma = torch.exp(0.5 * self.log_var)
        epsilon = torch.randn_like(sigma)
        w = self.mu + sigma * epsilon
        return w

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL(q || prior) under Gaussian prior assumption.

        KL[N(mu, sigma^2) || N(mu_prior, sigma_prior^2)]
        = log(sigma_prior/sigma) + (sigma^2 + (mu-mu_prior)^2)/(2*sigma_prior^2) - 0.5

        Returns:
            Scalar KL divergence summed over all parameters.
        """
        sigma_sq = torch.exp(self.log_var)
        prior_sigma_sq = torch.exp(torch.tensor(self.prior_log_var))

        kl = (
            torch.log(torch.sqrt(prior_sigma_sq / (sigma_sq + DEFAULT_EPSILON)))
            + (sigma_sq + (self.mu - self.prior_mu) ** 2) / (2.0 * prior_sigma_sq)
            - 0.5
        )
        return kl.sum()


class BayesianAttentionModel(nn.Module):
    """Wrapper that replaces ER-3DA attention tensors with Bayesian layers.

    The ER-3DA module uses three attention tensors:
      - P_xy: shape [1, C, 7, 7]  (channel-wise spatial attention)
      - P_zx: shape [1, 1, C, 7]  (channel-height plane)
      - P_zy: shape [1, 1, C, 7]  (channel-width plane)

    This wrapper replaces them with `BayesianAttentionLayer` instances,
    learns variational parameters, and provides a KL loss term.

    Args:
        net: BTNet-TS model instance (MyDiag_Model or subclass).
        in_channels: Number of input channels at the deepest ER-3DA block
            (default 512 for MyDiag21_base).
    """

    def __init__(self, net: nn.Module, in_channels: int = 512):
        super().__init__()
        self.net = net

        # Locate the ER-3DA blocks and replace their attention tensors
        self._replace_attention_layers(net, in_channels)

        self.kl_loss = 0.0

    def _replace_attention_layers(self, net: nn.Module, in_channels: int):
        """Find all ER-3DA blocks and replace P_xy, P_zx, P_zy."""
        self.bayesian_params = nn.ModuleList()

        for module in net.modules():
            module_class_name = module.__class__.__name__
            if module_class_name in ('ER_3DA_ResBlock', 'ER_3DA_Bottleneck'):
                if hasattr(module, 'params_xy'):
                    bayes_xy = BayesianAttentionLayer(module.params_xy.shape)
                    bayes_zx = BayesianAttentionLayer(module.params_zx.shape)
                    bayes_zy = BayesianAttentionLayer(module.params_zy.shape)

                    module.bayes_xy = bayes_xy
                    module.bayes_zx = bayes_zx
                    module.bayes_zy = bayes_zy

                    del module.params_xy
                    del module.params_zx
                    del module.params_zy

                    original_forward = module.forward

                    def make_bayesian_forward(orig_fn, mod):
                        def bayesian_forward(x):
                            mod.params_xy = mod.bayes_xy()
                            mod.params_zx = mod.bayes_zx()
                            mod.params_zy = mod.bayes_zy()
                            return orig_fn(x)
                        return bayesian_forward

                    module.forward = make_bayesian_forward(original_forward, module)
                    self.bayesian_params.extend([bayes_xy, bayes_zx, bayes_zy])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sampled attention weights.

        Args:
            x: Input tensor of shape [B, 3, 224, 224].

        Returns:
            Logits tensor of shape [B, num_classes].
        """
        return self.net(x)

    def compute_kl_loss(self) -> torch.Tensor:
        """Compute total KL divergence across all Bayesian attention layers.

        Returns:
            Scalar KL loss tensor.
        """
        total_kl = torch.tensor(0.0, device=self.net.fc.weight.device)
        for layer in self.bayesian_params:
            total_kl = total_kl + layer.kl_divergence()
        return total_kl


if __name__ == '__main__':
    print("bayesian_attention.py — standalone test")
    from net.MyDiagX import MyDiag21_tiny

    device = torch.device('cpu')
    model = MyDiag21_tiny(num_classes=4)
    model.eval()

    bayes_model = BayesianAttentionModel(model, in_channels=256)
    dummy = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out = bayes_model(dummy)
        kl = bayes_model.compute_kl_loss()

    print(f"Output shape: {out.shape}")
    print(f"KL loss: {kl.item():.4f}")
    print("Standalone test passed.")
