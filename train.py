import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import lr_scheduler

# ==================================================
# 模型选择区（取消对应注释即可切换模型）
# ==================================================
# 1. ResNet 系列 BTD-3
# from torchvision.models import resnet18 as create_model

# 2. MobileNet 系列 BTD-4
# from torchvision.models import mobilenet_v2 as create_model

# 3. EfficientNet 系列 BTD-44
from torchvision.models import efficientnet_b0 as create_model

# 4. 自定义模型
# from net.MyDiagX import MyDiag21 as create_model
# ==================================================

def set_seed(seed=42):
    """固定随机种子以保障实验可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m):
    """
    统一使用 Xavier Uniform 初始化
    对于卷积层和全连接层应用 Xavier 初始化，对于 Batch Normalization 层保持默认或设为恒等
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class KFoldDataset(Dataset):
    """自定义数据集类，支持带空格的路径解析"""
    def __init__(self, data_root, txt_files, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        for txt_file in txt_files:
            txt_path = os.path.join(data_root, txt_file)
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    # 使用 rsplit 解决路径中包含空格的问题
                    parts = line.rsplit(maxsplit=1) 
                    if len(parts) == 2:
                        self.samples.append((os.path.join(data_root, parts[0]), int(parts[1])))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, path='best_model.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_num_classes(data_root):
    """动态获取类别数"""
    mapping_path = os.path.join(data_root, "class_mapping.txt")
    if not os.path.exists(mapping_path): return 3
    return sum(1 for line in open(mapping_path) if line.strip())

def main(args):
    set_seed(42)
    # 优先使用 GPU 加速
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 自动识别当前选择的模型名称
    model_name = create_model.__name__
    model_weights_dir = os.path.join(args.weights_dir, model_name)
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)
        
    print(f"{'='*50}")
    print(f"🚀 [Experiment] Target Model: {model_name}")
    print(f"📦 [Storage] Weights: {model_weights_dir}")
    print(f"⚙️ [Initialization] Unified Xavier Uniform")
    print(f"💻 [Device] Using device: {device}")
    print(f"{'='*50}")

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
        ])
    }

    assert os.path.exists(args.data_root), f"Path '{args.data_root}' does not exist."
    num_classes = get_num_classes(args.data_root)

    # 启动 5-Fold 训练
    for fold in range(1, args.k_folds + 1):
        print(f"\n▶️ Starting Fold {fold}/{args.k_folds}")
        
        val_txt = [f"fold_{fold}.txt"]
        train_txts = [f"fold_{i}.txt" for i in range(1, args.k_folds + 1) if i != fold]
        
        train_dataset = KFoldDataset(args.data_root, train_txts, data_transform["train"])
        val_dataset = KFoldDataset(args.data_root, val_txt, data_transform["val"])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 1. 实例化模型（不使用预训练权重，为了进行公平对比实验）
        net = create_model(weights=None)
        
        # 2. 动态修改输出层
        if hasattr(net, 'fc'): # ResNet, ShuffleNet
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif hasattr(net, 'classifier'): # MobileNet, VGG, EfficientNet
            if isinstance(net.classifier, nn.Sequential):
                in_features = net.classifier[-1].in_features
                net.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = net.classifier.in_features
                net.classifier = nn.Linear(in_features, num_classes)
        
        # 3. 【核心修改】应用 Xavier 统一初始化
        net.apply(init_weights)
        net.to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

        save_path = os.path.join(model_weights_dir, f'Fold{fold}_Best.pth')
        early_stopping = EarlyStopping(patience=10, path=save_path)

        for epoch in range(args.epochs):
            net.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    v_loss = loss_function(net(val_images.to(device)), val_labels.to(device))
                    val_loss += v_loss.item()

            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']

            print(f'[Fold {fold} Epoch {epoch+1:03d}] LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

            scheduler.step()
            early_stopping(avg_val_loss, net)
            
            if early_stopping.early_stop:
                print(f"🛑 Early stopping triggered for Fold {fold}.")
                break
        print(f"✅ Fold {fold} Complete. Best Weight: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--data_root', type=str, default=os.path.join(BASE_DIR, "data", "BTD-44"))
    parser.add_argument('--weights_dir', type=str, default=os.path.join(BASE_DIR, "weights"))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--k_folds', type=int, default=5)
    
    main(parser.parse_args())