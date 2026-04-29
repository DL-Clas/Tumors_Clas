import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import lr_scheduler

# Import the core model (make sure net/MyDiagX.py is in your path)
from net.MyDiagX import MyDiag21 as create_model

def set_seed(seed=42):
    """
    Set a fixed random seed to ensure the reproducibility of the experiment
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """
    Early Stopping Mechanism (Patience=10, Monitor Val Loss)
    """
    def __init__(self, patience=10, verbose=False, delta=0, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def main():
    # 1. Basic Configuration
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameter settings (adjust as needed)
    batch_size = 16
    epochs = 100
    lr = 0.001
    save_path = './weights/BTNet_TS_Best.pth'
    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    # 2. Data Augmentation and Loading (24x224)
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

    # Data Path Settings
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./data"))
    image_path = os.path.join(data_root, "BTD-4")
    assert os.path.exists(image_path), f"Path '{image_path}' does not exist."

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # Retrieve the category index mapping and save it
    class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())
    with open('class_indices.json', 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"])
    val_num = len(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Using {train_num} images for training, {val_num} images for validation.")

    # 3. Models, Loss Functions, and Optimizers
    net = create_model(num_classes=len(class_list))
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # lr setting
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Instantiate the EarlyStop class
    early_stopping = EarlyStopping(patience=10, verbose=True, path=save_path)

    # 4. Training 
    for epoch in range(epochs):
        # Train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.desc = f"Train Epoch [{epoch+1}/{epochs}] loss:{loss:.3f}"

        # Verification
        net.eval()
        val_acc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                
                # Calculate validation loss for early stopping
                v_loss = loss_function(outputs, val_labels.to(device))
                val_loss += v_loss.item()
                
                predict_y = torch.max(outputs, dim=1)[1]
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = f"Valid Epoch [{epoch+1}/{epochs}]"

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        final_val_acc = val_acc / val_num

        print(f'[Epoch {epoch+1}] train_loss: {avg_train_loss:.4f}  val_loss: {avg_val_loss:.4f}  val_accuracy: {final_val_acc:.4f}')

        # Update learning rate
        scheduler.step()

        # Early-stage screening
        early_stopping(avg_val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training finished.")
            break

    print('Finished Training.')

if __name__ == '__main__':
    main()
