import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import random, time, warnings
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_name = ["nasi_goreng", "rendang", "soto_ayam", "bakso", "gado_gado"]
num_class = len(class_name)
map_label = {c: i for i, c in enumerate(class_name)}
map_index = {i: c for c, i in map_label.items()}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
lr = 1e-3
batch_size = 32
epochs = 10
num_workers = 0

class ResidualBlock(nn.Module):
    """BasicBlock ResNet-v1: Conv-BN-ReLU -> Conv-BN -> +skip -> ReLU"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.proj = None
        if stride != 1 or in_c != out_c:
            self.proj = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = F.relu(out + identity)  # residual sum + ReLU
        return out


class ResNet34(nn.Module):
    """Konfigurasi ResNet-34: [3,4,6,3] dengan channel [64,128,256,512]"""
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,  64,  3, stride=1)
        self.layer2 = self._make_layer(64,  128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d((1,1))
        self.fc     = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, in_c, out_c, n_blocks, stride):
        blocks = [ResidualBlock(in_c, out_c, stride=stride)]
        for _ in range(n_blocks-1):
            blocks.append(ResidualBlock(out_c, out_c, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x); x = torch.flatten(x, 1)
        return self.fc(x)

def create_resnet(num_classes=5):
    return ResNet34(num_classes)

class Makanan(Dataset):
    def __init__(self, dataframe, root_dir: Path, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["label_idx"])
        if self.transform: image = self.transform(image)
        return image, label, str(img_path)

def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct, targets.size(0)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        c, t = accuracy_from_logits(outputs, labels)
        correct += c; total += t
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

def plot_curves(history, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy Curve
    plt.figure(figsize=(6,4))
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(save_dir / "acc_curve.png"); plt.close()

    # Loss Curve
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Loss Curve")
    plt.savefig(save_dir / "loss_curve.png"); plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    DATA_DIR = Path("/content/drive/MyDrive/Competitive Deep Learning/train")
    CSV_PATH = Path("/content/drive/MyDrive/Competitive Deep Learning/train.csv")
    SAVE_DIR = Path("/content/drive/MyDrive/Competitive Deep Learning/resnet_result")

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    df["label_idx"] = df["label"].apply(lambda l: map_label[l])
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label_idx"], random_state=SEED
    )

    # Augmentation
    IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_loader = DataLoader(Makanan(train_df, DATA_DIR, train_tf),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(Makanan(val_df, DATA_DIR, val_tf),
                              batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model
    model = create_resnet(num_classes=num_class).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_acc, y_true, y_pred = 0, [], []

    for epoch in range(1, epochs+1):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()

            run_loss += loss.item() * images.size(0)
            c, t = accuracy_from_logits(outputs, labels)
            run_correct += c; run_total += t

        train_loss = run_loss / run_total
        train_acc = run_correct / run_total
        val_loss, val_acc, y_pred, y_true = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step()
        print(f"[{epoch}/{epochs}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pth")


    plot_curves(history, SAVE_DIR)
    plot_confusion_matrix(y_true, y_pred, class_name, SAVE_DIR)
    print(f"Training selesai. Hasil disimpan di: {SAVE_DIR}")
