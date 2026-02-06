import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

DATA_DIR = Path("/-minipj_0210/juju9590/resNet18_v1/dataset")
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
IMG_SIZE = 224

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ✅ 증강: 좌우반전은 보류(역주행 정의 꼬일 수 있음)
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(DATA_DIR/"train", transform=train_tf)
val_ds   = datasets.ImageFolder(DATA_DIR/"val",   transform=eval_tf)
test_ds  = datasets.ImageFolder(DATA_DIR/"test",  transform=eval_tf)

print("classes:", train_ds.classes)  # ['normal', 'wrongway'] 기대
num_classes = len(train_ds.classes)

# ✅ 불균형 처리: WeightedRandomSampler (가장 간단/강력)
targets = [y for _, y in train_ds.samples]
class_count = torch.bincount(torch.tensor(targets))
class_weight = 1.0 / class_count.float()
sample_weight = [class_weight[t].item() for t in targets]
sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ✅ 모델: ResNet18 (빠르고 충분히 좋음)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# 백본 freeze (초고속 MVP)
for p in model.parameters():
    p.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

def run_epoch(loader, train=False):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss/total, correct/total

best_val = 0.0
for epoch in range(1, EPOCHS+1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)
    print(f"[{epoch}/{EPOCHS}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), "best.pt")

print("✅ best val acc:", best_val)

# ---- Test 평가 (wrongway recall/precision 확인) ----
model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in tqdm(test_loader, leave=False):
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().tolist()
        y_pred += pred
        y_true += y.tolist()

print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=test_ds.classes))
