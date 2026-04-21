
import os, sys, time, argparse, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

# --- Параметри
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="C:\\coach_data\\dataset_reid")
    ap.add_argument("--out", type=str, required=True, help="C:\\coach_data\\reid_runs\\coach_binary")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--img_h", type=int, default=256)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--freeze_until", type=int, default=0, help="Заморозити шари до idx (ResNet blocks)")
    return ap.parse_args()

# --- Набір даних (ImageFolder): expect train/coach, train/other, val/...
def get_dataloaders(root, img_h, img_w, batch):
    train_tf = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")
    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds   = datasets.ImageFolder(val_dir, val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch*2, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

class Backbone(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.feat_dim = 2048
        self.embed = nn.Linear(self.feat_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, return_embed=False):
        f = self.feature_extractor(x)
        f = f.view(f.size(0), -1)
        e = self.embed(f)
        e = self.bn(e)
        e = nn.functional.normalize(e, p=2, dim=1)  # 1) ембед L2
        if return_embed:
            return e
        # Класифікаційна голова для бінарного coach vs other
        return e

class ClassifierHead(nn.Module):
    def __init__(self, in_dim=512, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

def train_one_epoch(model, head, loader, opt, ce, device):
    model.train(); head.train()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feats = model(imgs)                # (B,512) L2-normalized
        logits = head(feats)               # (B,2)
        loss = ce(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += float(loss.item()) * imgs.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(imgs.size(0))
    return loss_sum/total, correct/total

@torch.inference_mode()
def evaluate(model, head, loader, ce, device):
    model.eval(); head.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feats = model(imgs)
        logits = head(feats)
        loss = ce(logits, labels)
        loss_sum += float(loss.item()) * imgs.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(imgs.size(0))
    return loss_sum/total, correct/total

def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, classes = get_dataloaders(args.data_root, args.img_h, args.img_w, args.batch)
    print("[INFO] classes:", classes, " (expect ['coach','other'])")

    model = Backbone(out_dim=512).to(device)
    head  = ClassifierHead(in_dim=512, num_classes=2).to(device)

    # Оптимізатор
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    ce  = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, head, train_loader, opt, ce, device)
        va_loss, va_acc = evaluate(model, head, val_loader, ce, device)
        print(f"[E{epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} || val loss {va_loss:.4f} acc {va_acc:.3f}")

        # Збереження кращої
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "epoch": epoch,
                "val_acc": va_acc,
                "classes": classes,
            }, os.path.join(args.out, "model_best.pt"))
            print(f"[SAVE] best acc {best_acc:.3f} -> model_best.pt")

    print("[DONE] best val acc:", best_acc)

if __name__ == "__main__":
    main()
