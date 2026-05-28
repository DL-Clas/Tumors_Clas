import os
import torch
import torch.nn as nn
import numpy as np


DEFAULT_N_MEMBERS = 5
DEFAULT_SEEDS = [42, 123, 456, 789, 1111]


class EnsembleModel(nn.Module):
    """Deep Ensemble of BTNet-TS models.

    Stores multiple model instances and aggregates their predictions
    via mean softmax probabilities.  Ensemble variance provides a
    measure of epistemic uncertainty.

    Args:
        models: List of BTNet-TS model instances.
    """

    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all ensemble members.

        Args:
            x: Input tensor of shape [B, 3, 224, 224].

        Returns:
            Mean logits across ensemble members.  Shape [B, C].
        """
        all_logits = []
        for model in self.models:
            logits = model(x)
            all_logits.append(logits)

        mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        return mean_logits

    def predict_with_uncertainty(self, x: torch.Tensor) -> dict:
        """Aggregate ensemble predictions and compute uncertainty.

        Args:
            x: Input tensor of shape [B, 3, 224, 224].

        Returns:
            Dictionary with keys:
                'mean_probs': Mean softmax probabilities [B, C].
                'std_probs': Std of softmax probabilities [B, C].
                'epistemic_uncertainty': Entropy of mean probs minus
                    mean of entropies [B].
                'all_probs': Full ensemble probability tensor [M, B, C].
        """
        all_probs = []
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

        all_probs_tensor = torch.stack(all_probs, dim=0)  # [M, B, C]
        mean_probs = all_probs_tensor.mean(dim=0)
        std_probs = all_probs_tensor.std(dim=0)

        # Epistemic: mutual information = H(mean) - mean(H(member))
        entropy_mean = -(mean_probs * torch.log(mean_probs + 1e-12)).sum(dim=1)
        member_entropy = -(all_probs_tensor * torch.log(all_probs_tensor + 1e-12)).sum(dim=2)
        mean_entropy = member_entropy.mean(dim=0)
        epistemic_uncertainty = entropy_mean - mean_entropy

        return {
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'epistemic_uncertainty': epistemic_uncertainty,
            'all_probs': all_probs_tensor,
        }


def train_ensemble(
    create_model_fn: callable,
    data_root: str,
    weights_dir: str,
    n_members: int = DEFAULT_N_MEMBERS,
    seeds: list = None,
    batch_size: int = 16,
    epochs: int = 15,
    lr: float = 1e-3,
) -> EnsembleModel:
    """Train an ensemble of BTNet-TS models with different random seeds.

    Args:
        create_model_fn: Function that instantiates the model
            (e.g. `lambda: MyDiag21(num_classes=4)`).
        data_root: Path to dataset root with fold files.
        weights_dir: Directory to save individual model weights.
        n_members: Number of ensemble members (default 5).
        seeds: List of random seeds for each member.  If None, uses
            DEFAULT_SEEDS.
        batch_size: Training batch size.
        epochs: Training epochs per member.
        lr: Learning rate.

    Returns:
        EnsembleModel containing all trained members.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from train import set_seed, KFoldDataset, EarlyStopping, get_num_classes, init_weights
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from torch.optim import lr_scheduler
    import torch.optim as optim

    if seeds is None:
        seeds = DEFAULT_SEEDS[:n_members]

    num_classes = get_num_classes(data_root)
    models = []

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    k_folds = 5
    for member_idx, seed in enumerate(seeds):
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*50}")
        print(f"Training ensemble member {member_idx + 1}/{n_members} (seed={seed})")
        print(f"{'='*50}")

        all_fold_weights = []
        for fold in range(1, k_folds + 1):
            val_txt = [f"fold_{fold}.txt"]
            train_txts = [f"fold_{i}.txt" for i in range(1, k_folds + 1) if i != fold]

            train_dataset = KFoldDataset(data_root, train_txts, data_transform["train"])
            val_dataset = KFoldDataset(data_root, val_txt, data_transform["val"])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            net = create_model_fn()
            net.fc = nn.Linear(net.fc.in_features, num_classes)
            net.apply(init_weights)
            net.to(device)

            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

            save_path = os.path.join(weights_dir, f'Ensemble_{member_idx}_Fold{fold}_Best.pth')
            early_stopping = EarlyStopping(patience=10, path=save_path)

            for epoch in range(epochs):
                net.train()
                for images, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = net(images.to(device))
                    loss = loss_function(outputs, labels.to(device))
                    loss.backward()
                    optimizer.step()

                net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_images, val_labels in val_loader:
                        v_loss = loss_function(net(val_images.to(device)), val_labels.to(device))
                        val_loss += v_loss.item()

                avg_val_loss = val_loss / len(val_loader)
                scheduler.step()
                early_stopping(avg_val_loss, net)

                if early_stopping.early_stop:
                    break

            fold_model = create_model_fn()
            fold_model.fc = nn.Linear(fold_model.fc.in_features, num_classes)
            fold_model.load_state_dict(torch.load(save_path, map_location=device))
            fold_model.to(device)
            fold_model.eval()
            all_fold_weights.append(fold_model)

        models.append(all_fold_weights[0])

    ensemble = EnsembleModel(models)
    return ensemble


if __name__ == '__main__':
    print("ensemble.py — standalone test")
    from net.MyDiagX import MyDiag21_tiny

    models_list = []
    for i in range(3):
        m = MyDiag21_tiny(num_classes=4)
        m.eval()
        models_list.append(m)

    ensemble = EnsembleModel(models_list)
    dummy = torch.randn(4, 3, 224, 224)
    result = ensemble.predict_with_uncertainty(dummy)

    print(f"Mean probs shape: {result['mean_probs'].shape}")
    print(f"Epistemic uncertainty: {result['epistemic_uncertainty'].tolist()}")
    print("Standalone test passed.")
