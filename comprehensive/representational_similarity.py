import torch
import numpy as np


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute Centered Kernel Alignment between two representation matrices.

    Uses the linear kernel k(x, y) = x^T y.  CKA = HSIC(X, X, Y, Y) /
    sqrt(HSIC(X, X, X, X) * HSIC(Y, Y, Y, Y)).

    Args:
        X: Feature tensor of shape [N, d1].
        Y: Feature tensor of shape [N, d2].

    Returns:
        CKA score in [0, 1] where lower values indicate higher orthogonality.

    Raises:
        ValueError: If X and Y have different number of samples.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples, "
            f"got X.shape[0]={X.shape[0]}, Y.shape[0]={Y.shape[0]}"
        )
    N = X.shape[0]

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    def hsic(K_mat: torch.Tensor, L_mat: torch.Tensor) -> torch.Tensor:
        """Compute unbiased HSIC."""
        ones = torch.ones(N, 1, device=K_mat.device)
        H = torch.eye(N, device=K_mat.device) - (1.0 / N) * (ones @ ones.T)
        K_centered = H @ K_mat @ H
        L_centered = H @ L_mat @ H
        return (K_centered * L_centered).sum() / ((N - 1) ** 2)

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    denominator = torch.sqrt(hsic_kk * hsic_ll)
    if denominator < 1e-12:
        return 0.0
    cka_val = hsic_kl / denominator
    return cka_val.item()


def compute_svcca(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute Singular Value CCA between two representation matrices.

    Both matrices are truncated to retain 99% of the cumulative variance
    before computing canonical correlations.  Returns the mean correlation.

    Args:
        X: Feature tensor of shape [N, d1].
        Y: Feature tensor of shape [N, d2].

    Returns:
        Mean SVCCA correlation score in [0, 1].
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples, "
            f"got X.shape[0]={X.shape[0]}, Y.shape[0]={Y.shape[0]}"
        )
    N = X.shape[0]

    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    U_X, S_X, _ = torch.svd(X_centered)
    U_Y, S_Y, _ = torch.svd(Y_centered)

    cum_var_X = torch.cumsum(S_X ** 2, dim=0) / (S_X ** 2).sum()
    cum_var_Y = torch.cumsum(S_Y ** 2, dim=0) / (S_Y ** 2).sum()

    rank_X = int((cum_var_X <= 0.99).sum().item()) + 1
    rank_Y = int((cum_var_Y <= 0.99).sum().item()) + 1

    rank_X = max(1, min(rank_X, X.shape[1]))
    rank_Y = max(1, min(rank_Y, Y.shape[1]))

    X_proj = X_centered @ U_X[:, :rank_X]
    Y_proj = Y_centered @ U_Y[:, :rank_Y]

    Q_X, _ = torch.qr(X_proj)
    Q_Y, _ = torch.qr(Y_proj)

    C = Q_X.T @ Q_Y
    _, S_cca, _ = torch.svd(C)

    mean_cca = S_cca.mean().item()
    return mean_cca


def extract_representations(model, loader, device, num_classes=4):
    """Extract intermediate representations from each core module.

    Collects spatial features from ER-3DA 4, graph embeddings from GCF-2S,
    grouped features from MFS-GD, and the final classifier logits.

    Args:
        model: BTNet-TS model instance (MyDiag_Model or subclass).
        loader: DataLoader yielding (images, labels, paths).
        device: torch.device for computation.
        num_classes: Number of output classes (default 4 for BTD-4).

    Returns:
        Dictionary containing tensors:
            'er3da_4': Features after ER-3DA block 4 [N, C].
            'gcf_2s': Graph embeddings pooled to vector [N, C'].
            'mfs_gd': Post-MFS-GD pooled features [N, C''].
            'final_logits': Classifier logits [N, num_classes].
    """
    model.eval()
    all_er3da = []
    all_mfs = []
    all_gcn = []
    all_logits = []

    def hook_er3da(module, inp, out):
        all_er3da.append(out.detach())

    def hook_mfs(module, inp, out):
        all_mfs.append(out.detach())

    gcn_hook_buffer = []

    def hook_gcn(module, inp, out):
        gcn_hook_buffer.append(out.detach())

    hook_handle_er3da = model.layer4.register_forward_hook(hook_er3da)
    hook_handle_mfs = model.GFF4.register_forward_hook(hook_mfs)
    hook_handle_gcn = model.gcn3.register_forward_hook(hook_gcn)

    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            gcn_hook_buffer.clear()
            logits = model(images)
            all_logits.append(logits.detach())
            if gcn_hook_buffer:
                gcn_outputs = torch.cat(gcn_hook_buffer, dim=1)
                all_gcn.append(gcn_outputs.t())

    hook_handle_er3da.remove()
    hook_handle_mfs.remove()
    hook_handle_gcn.remove()

    er3da_tensor = torch.cat(all_er3da, dim=0)
    er3da_pooled = er3da_tensor.mean(dim=[2, 3])

    mfs_tensor = torch.cat(all_mfs, dim=0)
    mfs_pooled = mfs_tensor.mean(dim=[2, 3])

    gcf_tensor = torch.cat(all_gcn, dim=0)

    logits_tensor = torch.cat(all_logits, dim=0)

    return {
        'er3da_4': er3da_pooled,
        'gcf_2s': gcf_tensor,
        'mfs_gd': mfs_pooled,
        'final_logits': logits_tensor,
    }


def compute_module_similarity_table(model, loader, device, num_classes=4):
    """Compute the CKA and SVCCA similarity table (Table E.19).

    Args:
        model: BTNet-TS model instance.
        loader: DataLoader with `__getitem__` returning (img, label, path).
        device: torch.device.

    Returns:
        Dictionary with keys like 'cka_er3da_gcf', 'svcca_er3da_gcf', etc.
    """
    reps = extract_representations(model, loader, device, num_classes)

    pairs = [
        ('er3da_gcf', reps['er3da_4'], reps['gcf_2s']),
        ('er3da_mfs', reps['er3da_4'], reps['mfs_gd']),
        ('final_gcf', reps['final_logits'], reps['gcf_2s']),
        ('final_mfs', reps['final_logits'], reps['mfs_gd']),
    ]

    results = {}
    for name, rep_a, rep_b in pairs:
        results[f'cka_{name}'] = compute_cka(rep_a, rep_b)
        results[f'svcca_{name}'] = compute_svcca(rep_a, rep_b)

    return results


if __name__ == '__main__':
    print("representational_similarity.py — standalone test")

    X = torch.randn(100, 32)
    Y = torch.randn(100, 16)

    cka_score = compute_cka(X, Y)
    svcca_score = compute_svcca(X, Y)
    print(f"CKA score (random data): {cka_score:.4f}")
    print(f"SVCCA score (random data): {svcca_score:.4f}")

    Z = X.clone()
    cka_identical = compute_cka(X, Z)
    print(f"CKA score (identical): {cka_identical:.4f}")

    print("Standalone test passed.")
