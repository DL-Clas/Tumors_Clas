import torch
import torch.nn as nn
import numpy as np


DEFAULT_EPSILON = 1e-12


def compute_gradient_norms_per_branch(
    model: nn.Module,
    batch: tuple,
    loss_function: nn.Module,
    device: torch.device,
) -> dict:
    """Compute L2 gradient norms for each ER-3DA branch after one backward pass.

    Registers forward hooks on the four ER-3DA layer outputs (layer1..layer4),
    performs one forward + backward pass, and extracts the L2 norm of
    gradients flowing into each layer's parameters.

    Args:
        model: BTNet-TS model instance.
        batch: Tuple (images, labels) from DataLoader.
        loss_function: Loss criterion (e.g. nn.CrossEntropyLoss).
        device: torch.device.

    Returns:
        Dictionary mapping branch name (e.g. 'er3da_1') to L2 gradient norm.
    """
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    gradient_norms = {}
    grad_handles = []

    def make_hook(name: str):
        def hook(grad):
            param_norm = grad.norm(p=2).item()
            if name not in gradient_norms:
                gradient_norms[name] = 0.0
            gradient_norms[name] += param_norm
        return hook

    for layer_name, layer in [('er3da_1', model.layer1),
                               ('er3da_2', model.layer2),
                               ('er3da_3', model.layer3),
                               ('er3da_4', model.layer4)]:
        for param in layer.parameters():
            if param.requires_grad:
                handle = param.register_hook(make_hook(layer_name))
                grad_handles.append(handle)

    model.zero_grad()
    outputs = model(images)
    loss = loss_function(outputs, labels)
    loss.backward()

    for handle in grad_handles:
        handle.remove()

    return gradient_norms


def compute_forward_activation_variance(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute channel-wise activation variance for each branch and fused output.

    Registers forward hooks on layer1..layer4 outputs and on the MFS-GD
    fused output (after GFF4 / the final fusion step).  Returns variance
    averaged over channels then over spatial dimensions.

    Args:
        model: BTNet-TS model instance.
        images: Input tensor of shape [B, 3, 224, 224].
        device: torch.device.

    Returns:
        Dictionary with keys 'x1', 'x2', 'x3', 'x4', 'x_fused' containing
        mean channel-wise variance values.
    """
    activations = {}

    def make_hook(name, container):
        def hook(module, inp, out):
            # out shape: [B, C, H, W]
            act = out.detach()
            # Channel-wise variance: var over spatial dims, mean over channels
            var_per_channel = act.var(dim=[2, 3])  # [B, C]
            container[name] = var_per_channel.mean(dim=1).mean(dim=0).item()
        return hook

    handles = []
    handles.append(model.layer1.register_forward_hook(make_hook('x1', activations)))
    handles.append(model.layer2.register_forward_hook(make_hook('x2', activations)))
    handles.append(model.layer3.register_forward_hook(make_hook('x3', activations)))
    handles.append(model.layer4.register_forward_hook(make_hook('x4', activations)))

    with torch.no_grad():
        model(images.to(device))

    for h in handles:
        h.remove()

    return activations


def compute_spatial_entropy(
    feature_map: torch.Tensor,
) -> torch.Tensor:
    """Compute spatial Shannon entropy of a feature map.

    Treats the normalized spatial activation of each channel as a
    probability distribution over spatial locations.

    H(X) = - sum_i p_i * log(p_i + epsilon)

    Args:
        feature_map: Tensor of shape [B, C, H, W].

    Returns:
        Scalar entropy value averaged over batch and channels.
    """
    B, C, H, W = feature_map.shape
    feature_flat = feature_map.view(B, C, -1)  # [B, C, H*W]

    # Softmax over spatial locations per channel
    probs = torch.softmax(feature_flat, dim=-1)  # [B, C, H*W]

    log_probs = torch.log(probs + DEFAULT_EPSILON)
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, C]
    return entropy.mean().item()


def compute_branch_cka_similarity(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute CKA between fused MFS-GD output and each individual branch.

    Registers hooks on layer1..layer4 and on the post-fusion feature map
    (after the adaptive average pool).  Returns per-branch CKA scores.

    Args:
        model: BTNet-TS model instance.
        images: Input tensor of shape [B, 3, 224, 224].
        device: torch.device.

    Returns:
        Dictionary with keys 'cka_er3da_1' through 'cka_er3da_4' and
        'cka_er3da_1_vs_4' (for the low CKA verification).
    """
    activations = {}

    def make_hook(name, container):
        def hook(module, inp, out):
            act = out.detach()
            # Global average pool to get sample-wise vectors
            pooled = act.mean(dim=[2, 3])  # [B, C]
            container[name] = pooled
        return hook

    handles = []
    handles.append(model.layer1.register_forward_hook(make_hook('x1', activations)))
    handles.append(model.layer2.register_forward_hook(make_hook('x2', activations)))
    handles.append(model.layer3.register_forward_hook(make_hook('x3', activations)))
    handles.append(model.layer4.register_forward_hook(make_hook('x4', activations)))

    def hook_fc_input(module, inp, out):
        activations['x_fused'] = inp[0].detach()

    handle_fc = model.fc.register_forward_hook(hook_fc_input)

    with torch.no_grad():
        model(images.to(device))

    handle_fc.remove()
    for h in handles:
        h.remove()

    from .representational_similarity import compute_cka

    results = {}
    branches = ['x1', 'x2', 'x3', 'x4']

    for branch_name in branches:
        if branch_name in activations and 'x_fused' in activations:
            results[f'cka_er3da_{branch_name[1:]}'] = compute_cka(
                activations[branch_name], activations['x_fused'],
            )

    if 'x1' in activations and 'x4' in activations:
        results['cka_er3da_1_vs_4'] = compute_cka(
            activations['x1'], activations['x4'],
        )

    return results


def branch_wise_masking_analysis(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_function: nn.Module = None,
) -> dict:
    """Evaluate accuracy and confidence drop when masking each ER-3DA branch.

    For each of the 4 ER-3DA branches, zeros out the feature map before
    the next block and measures the resulting accuracy and mean target-class
    confidence.

    Args:
        model: BTNet-TS model instance.
        loader: DataLoader yielding (images, labels) or (images, labels, paths).
        device: torch.device.
        loss_function: Optional loss function.  If None, uses CrossEntropyLoss.

    Returns:
        Dictionary with keys:
            'baseline': {'accuracy', 'mean_confidence'}
            'masked_1' ... 'masked_4': {'accuracy', 'confidence_drop',
                                        'accuracy_drop'}
    """
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()

    def evaluate_with_mask(masked_layer=None):
        """Run evaluation, optionally zeroing a layer's output."""
        model.eval()
        correct = 0
        total = 0
        confidences = []

        hook_handle = None

        if masked_layer is not None:

            def masking_hook(module, inp, out):
                return torch.zeros_like(out)

            layer_map = {
                1: model.layer1,
                2: model.layer2,
                3: model.layer3,
                4: model.layer4,
            }
            hook_handle = layer_map[masked_layer].register_forward_hook(masking_hook)

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                batch_conf = probs.gather(1, labels.view(-1, 1)).squeeze()
                confidences.extend(batch_conf.cpu().tolist())

        if hook_handle is not None:
            hook_handle.remove()

        accuracy = (correct / total) * 100 if total > 0 else 0.0
        mean_conf = float(np.mean(confidences)) * 100 if confidences else 0.0
        return accuracy, mean_conf

    baseline_acc, baseline_conf = evaluate_with_mask(masked_layer=None)

    results = {'baseline': {'accuracy': baseline_acc, 'mean_confidence': baseline_conf}}

    for layer_idx in range(1, 5):
        masked_acc, masked_conf = evaluate_with_mask(masked_layer=layer_idx)
        conf_drop = masked_conf - baseline_conf
        acc_drop = baseline_acc - masked_acc
        results[f'masked_{layer_idx}'] = {
            'accuracy': masked_acc,
            'confidence_drop': conf_drop,
            'accuracy_drop': acc_drop,
        }

    return results


if __name__ == '__main__':
    print("fusion_analysis.py — standalone test")

    dummy_feat = torch.randn(4, 64, 14, 14)
    entropy_val = compute_spatial_entropy(dummy_feat)
    print(f"Spatial entropy (random): {entropy_val:.4f}")
    print("Standalone test passed.")
