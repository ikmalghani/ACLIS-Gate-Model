#!/usr/bin/env python3
"""ACLIS Leaf Gate — alternative training (Distill + Field-aug + EMA + QAT)

Local conversion of `Leaf Gate Model/Alt Leaf Gate Model/aclis_leaf_gate_alt_distill_qat.ipynb`.
No Colab. Run from this folder after the dataset exists:

  python3 build_leaf_nonleaf_dataset.py
  python3 -u aclis_leaf_gate_alt_distill_qat.py
  python3 -u aclis_leaf_gate_alt_distill_qat.py --export-only   # reuse existing EMA weights

Student: TinyLeafGate @ 96×96  (same MCU topology as the baseline gate)
Teacher: MobileNetV3-Small (train-only KD)
Classes: leaf / non-leaf

Dataset (next to this script):
  leaf_nonleaf_dataset/{train,val,test}/{leaf,non-leaf}/

Deploy artifact (always written unless --skip-export):
  leaf_gate_output_alt/aclis_leaf_gate_96x_alt_full_int8.tflite

A training report is always written (skipped with --export-only):
  leaf_gate_output_alt/training_report.md
  leaf_gate_output_alt/training_report.pdf
  leaf_gate_output_alt/confusion_matrix.png
  leaf_gate_output_alt/metrics.json
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(
        description="TinyLeafGate alt recipe (KD + MixUp + EMA + QAT), local")
    p.add_argument("--dataset", type=str, default=None,
                   help="Dataset root with train/val/test/{leaf,non-leaf}")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--skip-teacher", action="store_true",
                   help="Load existing teacher weights; error if missing")
    p.add_argument("--skip-export", action="store_true",
                   help="Skip INT8 TFLite export (not recommended)")
    p.add_argument("--export-only", action="store_true",
                   help="Skip training; export TFLite from existing EMA weights")
    p.add_argument("--skip-qat", action="store_true",
                   help="Skip the QAT-proxy fine-tune")
    return p.parse_args()


ARGS = parse_args()
# TensorFlow 2.20 cannot use this NVIDIA driver; keep TFLite conversion on CPU.
# Hide GPUs before TF is imported (export-only never needs CUDA).
if ARGS.export_only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Not on Colab — using local paths", flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

print("PyTorch     :", torch.__version__, flush=True)
print("CUDA        :", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0),
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)",
          flush=True)

# ---------------------------------------------------------------------------
# 1 — Config
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATASET_NAME = "leaf_nonleaf_dataset"
DATASET_CANDIDATES = [
    Path(ARGS.dataset) if ARGS.dataset else None,
    HERE / DATASET_NAME,
]
DATASET_CANDIDATES = [p for p in DATASET_CANDIDATES if p is not None]

OUTPUT_DIR = str(HERE / "leaf_gate_output_alt")
SAVE_PATH = os.path.join(OUTPUT_DIR, "aclis_leaf_gate_96x_alt.pth")
EMA_PATH = os.path.join(OUTPUT_DIR, "aclis_leaf_gate_96x_alt_ema.pth")
TEACHER_PATH = os.path.join(OUTPUT_DIR, "leaf_gate_teacher_mnv3.pth")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "aclis_leaf_gate_96x_alt_checkpoint.pth")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "aclis_leaf_gate_96x_alt_full_int8.tflite")
REPORT_MD = os.path.join(OUTPUT_DIR, "training_report.md")
REPORT_PDF = os.path.join(OUTPUT_DIR, "training_report.pdf")
REPORT_CM_PNG = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
REPORT_JSON = os.path.join(OUTPUT_DIR, "metrics.json")

IMAGE_SIZE = 96
NUM_CLASSES = 2
CLASSES = ["leaf", "non-leaf"]  # ImageFolder alphabetical order
BATCH_SIZE = 64
TARGET_ACC = 0.92

TEACHER_EPOCHS = 8
STUDENT_EPOCHS = 40
LR_TEACHER = 3e-4
LR_STUDENT = 1e-3
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 3
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.7
LABEL_SMOOTH = 0.05
MIXUP_ALPHA = 0.2
EMA_DECAY = 0.999
QAT_EPOCHS = 5
PATIENCE = 12

if ARGS.batch_size is not None:
    BATCH_SIZE = ARGS.batch_size
elif torch.cuda.is_available():
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if mem_gb < 5.5 and BATCH_SIZE > 32:
        print(f"GPU has {mem_gb:.1f} GB — dropping BATCH_SIZE {BATCH_SIZE} → 32",
              flush=True)
        BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Alt leaf-gate config ready", flush=True)
print(f"  HERE            : {HERE}", flush=True)
print(f"  OUTPUT_DIR      : {OUTPUT_DIR}", flush=True)
print(f"  Classes         : {CLASSES}  (NUM_CLASSES={NUM_CLASSES})", flush=True)
print(f"  Image size      : {IMAGE_SIZE}x{IMAGE_SIZE}  batch={BATCH_SIZE}", flush=True)
print(f"  Student         : TinyLeafGate  epochs={STUDENT_EPOCHS}  "
      f"KD a={KD_ALPHA}  T={KD_TEMPERATURE}", flush=True)
print(f"  MixUp a={MIXUP_ALPHA}  EMA={EMA_DECAY}  QAT epochs={QAT_EPOCHS}", flush=True)
print(f"  resume={ARGS.resume}  export_only={ARGS.export_only}  "
      f"skip_export={ARGS.skip_export}", flush=True)
if ARGS.export_only and ARGS.skip_export:
    raise SystemExit("Cannot combine --export-only with --skip-export")

# ---------------------------------------------------------------------------
# 2 — Dataset
# ---------------------------------------------------------------------------
DATASET_COUNTS = {}


def dataset_ready(root):
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            d = os.path.join(root, split, cls)
            if not os.path.isdir(d) or not any(Path(d).iterdir()):
                return False
    return True


def summarize(root):
    for split in ("train", "val", "test"):
        counts = {
            cls: len([
                p for p in os.listdir(os.path.join(root, split, cls))
                if os.path.isfile(os.path.join(root, split, cls, p))
            ])
            for cls in CLASSES
        }
        DATASET_COUNTS[split] = counts
        print(f"  [{split}] " + "  ".join(f"{k}={v}" for k, v in counts.items()),
              flush=True)


DATASET_DIR = None
for cand in DATASET_CANDIDATES:
    if dataset_ready(str(cand)):
        DATASET_DIR = str(cand)
        break

if DATASET_DIR is None:
    raise FileNotFoundError(
        "leaf/non-leaf dataset not found.\n"
        f"  Expected: {DATASET_NAME}/{{train,val,test}}/{{leaf,non-leaf}}/\n"
        f"  Looked in: {DATASET_CANDIDATES}\n"
        "  Build it first:\n"
        "    python build_leaf_nonleaf_dataset.py"
    )

print(f"Using dataset: {DATASET_DIR}", flush=True)
summarize(DATASET_DIR)

_probe = ImageFolder(os.path.join(DATASET_DIR, "train"))
print("ImageFolder classes:", _probe.classes, flush=True)
assert _probe.classes == CLASSES, f"Expected {CLASSES}, got {_probe.classes}"
print("Class index OK: leaf=0, non-leaf=1", flush=True)

# ---------------------------------------------------------------------------
# 3 — Models
# ---------------------------------------------------------------------------
print("Model cell started", flush=True)


class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu6(self.bn1(self.dw(x)))
        x = F.relu6(self.bn2(self.pw(x)))
        return x


class TinyLeafGate(nn.Module):
    """Identical topology to the baseline gate — isolates the training recipe."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparable(32, 48, stride=2),
            DepthwiseSeparable(48, 64, stride=2),
            DepthwiseSeparable(64, 96, stride=2),
            DepthwiseSeparable(96, 128, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def build_teacher(num_classes=2):
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        m = models.mobilenet_v3_small(weights=weights)
    except Exception:
        m = models.mobilenet_v3_small(pretrained=True)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student = TinyLeafGate(NUM_CLASSES).to(DEVICE)
teacher = build_teacher(NUM_CLASSES).to(DEVICE)

n_params = sum(p.numel() for p in student.parameters())
print("=" * 65, flush=True)
print("ALT recipe — TinyLeafGate student + MobileNetV3-Small teacher", flush=True)
print("=" * 65, flush=True)
print(f"Device            : {DEVICE}", flush=True)
print(f"Student params    : {n_params:,}", flush=True)
print(f"Teacher params    : {sum(p.numel() for p in teacher.parameters()):,}", flush=True)
print(f"Input             : {IMAGE_SIZE}×{IMAGE_SIZE}", flush=True)
print("=" * 65, flush=True)

# ---------------------------------------------------------------------------
# 4 — Dataloaders
# ---------------------------------------------------------------------------
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        for offset in range(5):
            idx = (index + offset) % len(self.samples)
            path, target = self.samples[idx]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, target
            except Exception:
                continue
        return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), 0


class RandomGaussianBlur:
    def __init__(self, p=0.35, radius=(0.3, 1.8)):
        self.p = p
        self.radius = radius

    def __call__(self, img):
        if random.random() > self.p:
            return img
        r = random.uniform(*self.radius)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class RandomJPEG:
    def __init__(self, p=0.4, quality=(35, 90)):
        self.p = p
        self.quality = quality

    def __call__(self, img):
        if random.random() > self.p:
            return img
        q = random.randint(*self.quality)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0), ratio=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=35),
    transforms.ColorJitter(brightness=0.55, contrast=0.55, saturation=0.4, hue=0.12),
    transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.85, 1.15)),
    RandomGaussianBlur(p=0.35),
    RandomJPEG(p=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])


class FixedCameraStress:
    def __init__(self):
        self.resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img):
        img = self.resize(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.9))
        arr = np.asarray(img).astype(np.float32)
        arr = (arr - 127.5) * 0.82 + 127.5 - 12.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=55)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return self.norm(self.to_tensor(img))


stress_transform = FixedCameraStress()
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_dataset = SafeImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transform)
val_dataset = SafeImageFolder(os.path.join(DATASET_DIR, "val"), transform=val_transform)
test_dataset = SafeImageFolder(os.path.join(DATASET_DIR, "test"), transform=val_transform)
stress_dataset = SafeImageFolder(os.path.join(DATASET_DIR, "test"), transform=stress_transform)
assert train_dataset.classes == CLASSES

counts = [0] * NUM_CLASSES
for _, y in train_dataset.samples:
    counts[y] += 1
print("Train counts:", dict(zip(CLASSES, counts)), flush=True)

weights = [1.0 / counts[y] for _, y in train_dataset.samples]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

nw = 2 if torch.cuda.is_available() else 0
pin = torch.cuda.is_available()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=nw, pin_memory=pin)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=nw, pin_memory=pin)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=nw, pin_memory=pin)
stress_loader = DataLoader(stress_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=nw, pin_memory=pin)

class_weights = torch.tensor(
    [sum(counts) / (NUM_CLASSES * c) for c in counts],
    dtype=torch.float32, device=DEVICE,
)
print("Class weights:", {c: float(w) for c, w in zip(CLASSES, class_weights)}, flush=True)
print("Loaders ready (train + clean val/test + camera-stress test)", flush=True)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += len(images)
            for p, l in zip(preds, labels):
                class_total[l.item()] += 1
                class_correct[l.item()] += int(p.item() == l.item())
    per_class = {
        CLASSES[i]: (class_correct[i] / class_total[i] if class_total[i] else 0.0)
        for i in range(NUM_CLASSES)
    }
    return total_loss / max(total, 1), correct / max(total, 1), per_class


def collect_preds(model, loader):
    """Return y_true, y_pred arrays for the confusion matrix."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def confusion_matrix(y_true, y_pred, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_str(per_class):
    return " ".join(f"{c}={per_class[c]:.3f}" for c in CLASSES)


if ARGS.export_only:
    if not os.path.isfile(EMA_PATH):
        raise FileNotFoundError(
            f"--export-only needs existing EMA weights:\n  {EMA_PATH}\n"
            "Train once without this flag first.")
    print("Export-only: loading existing EMA, skipping teacher / KD / QAT", flush=True)
    student.load_state_dict(torch.load(EMA_PATH, map_location=DEVICE))
    student.to(DEVICE).eval()
else:
    # ---------------------------------------------------------------------------
    # 5a — Teacher
    # ---------------------------------------------------------------------------
    print("Teacher training started", flush=True)

    teacher_crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=LR_TEACHER, weight_decay=WEIGHT_DECAY)
    sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=TEACHER_EPOCHS, eta_min=1e-6)

    load_existing_teacher = os.path.isfile(TEACHER_PATH) and (ARGS.skip_teacher or ARGS.resume)
    if ARGS.skip_teacher and not os.path.isfile(TEACHER_PATH):
        raise FileNotFoundError(f"--skip-teacher but missing {TEACHER_PATH}")

    if load_existing_teacher and os.path.isfile(TEACHER_PATH):
        teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        _, va_acc, per_class = evaluate(teacher, val_loader, teacher_crit)
        print(f"Loaded existing teacher  val={va_acc:.4f}  → {TEACHER_PATH}", flush=True)
        print(f"   {per_class_str(per_class)}", flush=True)
        best_t_acc = va_acc
    else:
        best_t_acc = 0.0
        print("=" * 65, flush=True)
        print(f"Teacher fine-tune  epochs={TEACHER_EPOCHS}  lr={LR_TEACHER}", flush=True)
        print("=" * 65, flush=True)
        for epoch in range(TEACHER_EPOCHS):
            teacher.train()
            total_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                opt_t.zero_grad()
                outputs = teacher(images)
                loss = teacher_crit(outputs, labels)
                loss.backward()
                opt_t.step()
                total_loss += loss.item() * len(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(images)
            sched_t.step()
            tr_acc = correct / max(total, 1)
            _, va_acc, per_class = evaluate(teacher, val_loader, teacher_crit)
            marker = ""
            if va_acc > best_t_acc:
                best_t_acc = va_acc
                torch.save(teacher.state_dict(), TEACHER_PATH)
                marker = " ← saved"
            print(f"  Epoch {epoch+1:3d}/{TEACHER_EPOCHS} | train {tr_acc:.3f} | val {va_acc:.3f} | "
                  f"{per_class_str(per_class)}{marker}", flush=True)
        teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"Teacher ready  best val={best_t_acc:.4f}  → {TEACHER_PATH}", flush=True)

    # ---------------------------------------------------------------------------
    # 5b — Student KD + MixUp + EMA
    # ---------------------------------------------------------------------------
    print("Student distillation started", flush=True)


    class ModelEMA:
        def __init__(self, model, decay=0.999):
            self.ema = copy.deepcopy(model).eval()
            for p in self.ema.parameters():
                p.requires_grad_(False)
            self.decay = decay

        @torch.no_grad()
        def update(self, model):
            d = self.decay
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
                else:
                    v.copy_(msd[k])


    def mixup_batch(x, y, alpha):
        if alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


    def kd_loss(student_logits, teacher_logits, y_a, y_b, lam, class_w):
        log_p = F.log_softmax(student_logits, dim=1)
        ce_a = F.nll_loss(log_p, y_a, weight=class_w, reduction="none")
        ce_b = F.nll_loss(log_p, y_b, weight=class_w, reduction="none")
        hard = (lam * ce_a + (1.0 - lam) * ce_b).mean()
        T = KD_TEMPERATURE
        soft = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)
        return KD_ALPHA * soft + (1.0 - KD_ALPHA) * hard


    def lr_at_epoch(epoch):
        if epoch < WARMUP_EPOCHS:
            return LR_STUDENT * float(epoch + 1) / float(WARMUP_EPOCHS)
        progress = (epoch - WARMUP_EPOCHS) / max(1, STUDENT_EPOCHS - WARMUP_EPOCHS)
        return 1e-6 + 0.5 * (LR_STUDENT - 1e-6) * (1.0 + math.cos(math.pi * progress))


    opt_s = torch.optim.AdamW(student.parameters(), lr=LR_STUDENT, weight_decay=WEIGHT_DECAY)
    ema = ModelEMA(student, decay=EMA_DECAY)
    ce_eval = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    patience_counter = 0
    start_epoch = 0

    if ARGS.resume and os.path.isfile(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        student.load_state_dict(ckpt["student"])
        ema.ema.load_state_dict(ckpt["ema"])
        best_val_acc = float(ckpt.get("best_val_acc", 0.0))
        patience_counter = int(ckpt.get("patience_counter", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        if "opt_s" in ckpt:
            opt_s.load_state_dict(ckpt["opt_s"])
        print(f"Resumed student from epoch {start_epoch}/{STUDENT_EPOCHS}  "
              f"best_val={best_val_acc:.4f}", flush=True)
        if start_epoch >= STUDENT_EPOCHS:
            print("Student already finished all epochs — skipping KD loop", flush=True)

    print("=" * 65, flush=True)
    print(f"Student KD  epochs={STUDENT_EPOCHS}  lr={LR_STUDENT}  α_KD={KD_ALPHA}  MixUp={MIXUP_ALPHA}",
          flush=True)
    print("=" * 65, flush=True)

    for epoch in range(start_epoch, STUDENT_EPOCHS):
        lr = lr_at_epoch(epoch)
        for g in opt_s.param_groups:
            g["lr"] = lr

        student.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            mixed, y_a, y_b, lam = mixup_batch(images, labels, MIXUP_ALPHA)
            opt_s.zero_grad()
            s_logits = student(mixed)
            with torch.no_grad():
                t_logits = teacher(mixed)
            loss = kd_loss(s_logits, t_logits, y_a, y_b, lam, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt_s.step()
            ema.update(student)
            total_loss += loss.item() * len(images)
            correct += (s_logits.argmax(1) == y_a).sum().item()
            total += len(images)

        tr_acc = correct / max(total, 1)
        _, va_acc, per_class = evaluate(ema.ema, val_loader, ce_eval)
        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_counter = 0
            torch.save(student.state_dict(), SAVE_PATH)
            torch.save(ema.ema.state_dict(), EMA_PATH)
            marker = " ← saved EMA"
        else:
            patience_counter += 1
            marker = f" (patience {patience_counter}/{PATIENCE})"

        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "ema": ema.ema.state_dict(),
            "opt_s": opt_s.state_dict(),
            "best_val_acc": best_val_acc,
            "patience_counter": patience_counter,
        }, CHECKPOINT_PATH)

        print(f"  Epoch {epoch+1:3d}/{STUDENT_EPOCHS} | lr {lr:.2e} | train~{tr_acc:.3f} | "
              f"val(EMA) {va_acc:.3f} | {per_class_str(per_class)}{marker}", flush=True)

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    if not os.path.isfile(EMA_PATH):
        torch.save(ema.ema.state_dict(), EMA_PATH)
        torch.save(student.state_dict(), SAVE_PATH)

    student.load_state_dict(torch.load(EMA_PATH, map_location=DEVICE))
    print(f"Best EMA val accuracy: {best_val_acc:.4f}", flush=True)
    print(f"Loaded EMA weights from {EMA_PATH}", flush=True)

    # ---------------------------------------------------------------------------
    # 5c — QAT-proxy
    # ---------------------------------------------------------------------------
    if not ARGS.skip_qat:
        print("QAT fine-tune started", flush=True)

        def int8_noise_forward(module, x):
            if x.dim() == 4:
                xmin = x.amin(dim=(0, 2, 3), keepdim=True)
                xmax = x.amax(dim=(0, 2, 3), keepdim=True)
            else:
                xmin = x.amin(dim=0, keepdim=True)
                xmax = x.amax(dim=0, keepdim=True)
            scale = (xmax - xmin).clamp_min(1e-6) / 255.0
            x_q = torch.round((x - xmin) / scale) * scale + xmin
            return x + (x_q - x).detach()

        class QATProxy(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, x):
                x = self.base.stem(x)
                if self.training:
                    x = int8_noise_forward(self, x)
                x = self.base.blocks(x)
                if self.training:
                    x = int8_noise_forward(self, x)
                return self.base.head(x)

        student = student.to(DEVICE)
        qat_model = QATProxy(student).to(DEVICE)
        opt_q = torch.optim.AdamW(qat_model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
        ce_q = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

        print("=" * 65, flush=True)
        print(f"QAT-proxy fine-tune  epochs={QAT_EPOCHS}  lr=1e-4", flush=True)
        print("=" * 65, flush=True)

        for epoch in range(QAT_EPOCHS):
            qat_model.train()
            total_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                opt_q.zero_grad()
                logits = qat_model(images)
                with torch.no_grad():
                    t_logits = teacher(images)
                hard = ce_q(logits, labels)
                soft = F.kl_div(
                    F.log_softmax(logits / KD_TEMPERATURE, dim=1),
                    F.softmax(t_logits / KD_TEMPERATURE, dim=1),
                    reduction="batchmean",
                ) * (KD_TEMPERATURE ** 2)
                loss = 0.5 * soft + 0.5 * hard
                loss.backward()
                opt_q.step()
                total_loss += loss.item() * len(images)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(images)
            tr_acc = correct / max(total, 1)
            student.eval()
            _, va_acc, per_class = evaluate(student, val_loader, ce_eval)
            print(f"  QAT {epoch+1:3d}/{QAT_EPOCHS} | train {tr_acc:.3f} | val {va_acc:.3f} | "
                  f"{per_class_str(per_class)}", flush=True)

        torch.save(student.state_dict(), EMA_PATH)
        torch.save(student.state_dict(), SAVE_PATH)
        print(f"QAT-proxy done — weights saved to {EMA_PATH}", flush=True)

    # ---------------------------------------------------------------------------
    # 6 — PyTorch FP32 test + confusion matrix
    # ---------------------------------------------------------------------------
    student.load_state_dict(torch.load(EMA_PATH, map_location=DEVICE))
    student.to(DEVICE).eval()

    te_loss, te_acc, per_class = evaluate(student, test_loader, ce_eval)
    st_loss, st_acc, st_per = evaluate(student, stress_loader, ce_eval)
    y_true, y_pred = collect_preds(student, test_loader)
    cm = confusion_matrix(y_true, y_pred)
    y_true_s, y_pred_s = collect_preds(student, stress_loader)
    cm_stress = confusion_matrix(y_true_s, y_pred_s)

    print("=" * 65, flush=True)
    print("ALT model — PyTorch FP32 TEST", flush=True)
    print("=" * 65, flush=True)
    print(f"  Overall accuracy     : {te_acc:.4f} ({te_acc*100:.1f}%)", flush=True)
    print(f"  Camera-stress acc    : {st_acc:.4f} ({st_acc*100:.1f}%)", flush=True)
    print(f"  Target               : ≥{TARGET_ACC*100:.0f}%  "
          f"{'PASS' if te_acc >= TARGET_ACC else 'below target'}", flush=True)
    print("  Per-class (clean):", flush=True)
    for cls, acc in per_class.items():
        print(f"    {cls:<10}  {acc:.3f}  ({acc*100:.1f}%)", flush=True)
    print("  Per-class (stress):", flush=True)
    for cls, acc in st_per.items():
        print(f"    {cls:<10}  {acc:.3f}  ({acc*100:.1f}%)", flush=True)
    print("  Confusion matrix (clean, rows=true, cols=pred):", flush=True)
    print(f"               pred {CLASSES[0]:>10} {CLASSES[1]:>10}", flush=True)
    for i, cls in enumerate(CLASSES):
        print(f"    true {cls:<10} {cm[i, 0]:10d} {cm[i, 1]:10d}", flush=True)
    print(f"  Weights              : {EMA_PATH}", flush=True)
    print("=" * 65, flush=True)

    # Gate-specific error counts
    fn_leaf = int(cm[0, 1])   # true leaf → predicted non-leaf
    fp_leaf = int(cm[1, 0])   # true non-leaf → predicted leaf


    def write_training_report():
        """Markdown + PDF + PNG + JSON: overall acc, per-class acc, confusion matrix."""
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        support = cm.sum(axis=1)
        col_sum = cm.sum(axis=0)
        prec = [
            (float(cm[i, i] / col_sum[i]) if col_sum[i] else 0.0) for i in range(NUM_CLASSES)
        ]
        rec = [float(per_class[c]) for c in CLASSES]
        f1 = [
            (2 * prec[i] * rec[i] / (prec[i] + rec[i]) if (prec[i] + rec[i]) else 0.0)
            for i in range(NUM_CLASSES)
        ]

        metrics = {
            "created": stamp,
            "dataset": DATASET_DIR,
            "classes": CLASSES,
            "image_size": IMAGE_SIZE,
            "student_params": int(n_params),
            "best_ema_val": float(best_val_acc),
            "overall_accuracy": float(te_acc),
            "camera_stress_accuracy": float(st_acc),
            "per_class_accuracy": {c: float(per_class[c]) for c in CLASSES},
            "per_class_accuracy_stress": {c: float(st_per[c]) for c in CLASSES},
            "precision": {c: prec[i] for i, c in enumerate(CLASSES)},
            "f1": {c: f1[i] for i, c in enumerate(CLASSES)},
            "confusion_matrix_clean": cm.tolist(),
            "confusion_matrix_stress": cm_stress.tolist(),
            "fn_leaf": fn_leaf,
            "fp_leaf": fp_leaf,
            "dataset_counts": DATASET_COUNTS,
            "target_accuracy": TARGET_ACC,
        }
        with open(REPORT_JSON, "w") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")

        md = []
        md.append("# Better Gate Model — Alt Training Report")
        md.append("")
        md.append(f"Generated: {stamp}  ")
        md.append("Recipe: TinyLeafGate student + MobileNetV3-Small teacher, "
                  "KD + MixUp + EMA + field augs + QAT-proxy  ")
        md.append(f"Dataset: `{DATASET_DIR}`")
        md.append("")
        md.append("## Overall accuracy")
        md.append("")
        md.append("| Split | Accuracy | Correct / N |")
        md.append("|---|---:|---:|")
        n_test = int(cm.sum())
        n_ok = int(np.trace(cm))
        n_stress = int(cm_stress.sum())
        n_ok_s = int(np.trace(cm_stress))
        md.append(f"| Clean test (FP32) | **{te_acc*100:.2f}%** | {n_ok} / {n_test} |")
        md.append(f"| Camera-stress test (FP32) | {st_acc*100:.2f}% | {n_ok_s} / {n_stress} |")
        md.append(f"| Best EMA val | {best_val_acc*100:.2f}% | — |")
        md.append(f"| Script target | {TARGET_ACC*100:.0f}% | "
                  f"{'PASS' if te_acc >= TARGET_ACC else 'below target'} |")
        md.append("")
        md.append("## Per-class accuracy")
        md.append("")
        md.append("| Class | Support | Precision | Recall (clean) | F1 | Recall (stress) |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for i, c in enumerate(CLASSES):
            md.append(
                f"| {c} | {int(support[i])} | {prec[i]:.3f} | "
                f"{rec[i]:.3f} ({rec[i]*100:.1f}%) | {f1[i]:.3f} | {st_per[c]:.3f} |"
            )
        md.append("")
        md.append("FN_leaf (true leaf → non-leaf, gate rejects a real leaf): "
                  f"**{fn_leaf}**  ")
        md.append("FP_leaf (true non-leaf → leaf, disease CNN would run on junk): "
                  f"**{fp_leaf}**")
        md.append("")
        md.append("## Confusion matrix — clean test")
        md.append("")
        md.append("Rows = true label, columns = predicted label.")
        md.append("")
        md.append("| true \\ pred | " + " | ".join(CLASSES) + " | n |")
        md.append("|---|" + "|".join(["---:"] * NUM_CLASSES) + "|---:|")
        for i, c in enumerate(CLASSES):
            cells = " | ".join(str(int(cm[i, j])) for j in range(NUM_CLASSES))
            md.append(f"| **{c}** | {cells} | {int(support[i])} |")
        md.append("")
        md.append(f"![Confusion matrix]({os.path.basename(REPORT_CM_PNG)})")
        md.append("")
        md.append("## Dataset")
        md.append("")
        md.append("| Split | " + " | ".join(CLASSES) + " | total |")
        md.append("|---|" + "|".join(["---:"] * NUM_CLASSES) + "|---:|")
        for split in ("train", "val", "test"):
            vals = [DATASET_COUNTS.get(split, {}).get(c, 0) for c in CLASSES]
            md.append(f"| {split} | " + " | ".join(f"{v:,}" for v in vals)
                      + f" | {sum(vals):,} |")
        md.append("")
        md.append("## Training setup")
        md.append("")
        md.append(f"- Student: TinyLeafGate ({n_params:,} params), {IMAGE_SIZE}×{IMAGE_SIZE}")
        md.append("- Teacher: MobileNetV3-Small (train only)")
        md.append(f"- KD α={KD_ALPHA}, T={KD_TEMPERATURE}, MixUp α={MIXUP_ALPHA}, "
                  f"EMA={EMA_DECAY}, label smooth={LABEL_SMOOTH}")
        md.append(f"- AdamW lr={LR_STUDENT:g}, warmup {WARMUP_EPOCHS} + cosine, "
                  f"{STUDENT_EPOCHS} student epochs, QAT-proxy {QAT_EPOCHS} epochs")
        md.append("- Field augs: blur p=0.35, JPEG q∈[35,90] p=0.4, harsh colour/lighting")
        md.append("")
        Path(REPORT_MD).write_text("\n".join(md) + "\n")
        print(f"Training report (markdown) → {REPORT_MD}", flush=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError:
            print("matplotlib not installed — skip PDF/PNG. pip install matplotlib",
                  flush=True)
            return

        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASSES)
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix — clean test  (acc {te_acc*100:.2f}%)")
        vmax = max(int(cm.max()), 1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v = int(cm[i, j])
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > 0.55 * vmax else "black", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(REPORT_CM_PNG, dpi=140)
        print(f"Confusion matrix PNG → {REPORT_CM_PNG}", flush=True)

        with PdfPages(REPORT_PDF) as pdf:
            fig1, axes = plt.subplots(2, 1, figsize=(8.27, 11.69),
                                      gridspec_kw={"height_ratios": [0.9, 1.1]})
            fig1.suptitle("Better Gate Model — Alt Training Report", fontsize=14,
                          fontweight="bold", y=0.98)
            ax0 = axes[0]
            ax0.axis("off")
            lines = [
                f"Generated  {stamp}",
                f"Dataset    {DATASET_DIR}",
                "",
                f"Overall accuracy (clean test)     {te_acc*100:6.2f}%    {n_ok} / {n_test}",
                f"Camera-stress accuracy            {st_acc*100:6.2f}%    {n_ok_s} / {n_stress}",
                f"Best EMA validation               {best_val_acc*100:6.2f}%",
                f"Target                            {TARGET_ACC*100:6.0f}%    "
                f"{'PASS' if te_acc >= TARGET_ACC else 'below target'}",
                "",
                "Per-class accuracy (recall)",
            ]
            for i, c in enumerate(CLASSES):
                lines.append(
                    f"  {c:<10}  clean {rec[i]*100:5.1f}%   stress {st_per[c]*100:5.1f}%   "
                    f"P={prec[i]:.3f}  F1={f1[i]:.3f}  n={int(support[i])}"
                )
            lines += [
                "",
                f"FN_leaf (true leaf → non-leaf)    {fn_leaf}",
                f"FP_leaf (true non-leaf → leaf)    {fp_leaf}",
                "",
                f"Student TinyLeafGate  {n_params:,} params   {IMAGE_SIZE}×{IMAGE_SIZE}",
                f"Teacher MobileNetV3-Small   KD α={KD_ALPHA} T={KD_TEMPERATURE}  "
                f"MixUp={MIXUP_ALPHA}  EMA={EMA_DECAY}",
            ]
            ax0.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                     family="monospace", fontsize=9, transform=ax0.transAxes)

            ax1 = axes[1]
            im = ax1.imshow(cm, cmap="Greens")
            ax1.set_xticks(range(NUM_CLASSES))
            ax1.set_yticks(range(NUM_CLASSES))
            ax1.set_xticklabels(CLASSES, fontsize=11)
            ax1.set_yticklabels(CLASSES, fontsize=11)
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("True")
            ax1.set_title("Confusion matrix — clean PyTorch FP32 test")
            for i in range(NUM_CLASSES):
                for j in range(NUM_CLASSES):
                    v = int(cm[i, j])
                    ax1.text(j, i, str(v), ha="center", va="center", fontsize=14,
                             color="white" if v > 0.55 * vmax else "black",
                             fontweight="bold")
            fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            fig1.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig1)
            plt.close(fig1)

            fig2, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            ax.set_title("Dataset splits (balanced leaf / non-leaf)", loc="left",
                         fontsize=13, fontweight="bold")
            table_data = [["Split"] + CLASSES + ["total"]]
            for split in ("train", "val", "test"):
                vals = [DATASET_COUNTS.get(split, {}).get(c, 0) for c in CLASSES]
                table_data.append([split] + [f"{v:,}" for v in vals] + [f"{sum(vals):,}"])
            table = ax.table(cellText=table_data, loc="upper center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#1b6b45")
                    cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#eef4f0")
            ax.text(
                0.5, 0.55,
                "Pools were collected from PlantVillage first (leaf vs non-leaf),\n"
                "balanced to the same count, then split into train/val/test.\n\n"
                "Confusion matrix rows are true labels; columns are predictions.\n"
                "Diagonal cells are correct counts.",
                ha="center", va="top", fontsize=10, transform=ax.transAxes,
            )
            pdf.savefig(fig2)
            plt.close(fig2)

        plt.close(fig)
        print(f"Training report PDF → {REPORT_PDF}", flush=True)


    write_training_report()

# ---------------------------------------------------------------------------
# 7 — INT8 TFLite export (required)
# ---------------------------------------------------------------------------
if ARGS.skip_export:
    print("Skipping INT8 TFLite export (--skip-export).", flush=True)
    print("Alt leaf-gate training complete (PyTorch FP32 only).", flush=True)
    print(f"  Dataset : {DATASET_DIR}", flush=True)
    print(f"  EMA     : {EMA_PATH}", flush=True)
    print(f"  Report  : {REPORT_MD}", flush=True)
    sys.exit(0)

if not ARGS.export_only:
    # Fresh CPU process: TF 2.20 cannot use this NVIDIA driver.
    print("Launching CPU TFLite export from saved EMA weights…", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    rc = subprocess.call(
        [sys.executable, str(HERE / "aclis_leaf_gate_alt_distill_qat.py"), "--export-only"],
        env=env,
    )
    sys.exit(rc)

print("TinyEngine-friendly export started", flush=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    import tensorflow as tf
except ImportError as e:
    print(f"TensorFlow not installed — cannot export TFLite ({e})", flush=True)
    print(
        "Install tensorflow in bettergatemodelenv:\n"
        "  python3 -m pip install tensorflow==2.20.0 tf-keras==2.20.1",
        flush=True,
    )
    raise
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

student.load_state_dict(torch.load(EMA_PATH, map_location="cpu"))
student.cpu().eval()
sd = student.state_dict()
model = student


def pt_conv_to_keras(w):
    return w.detach().cpu().numpy().transpose(2, 3, 1, 0)


def pt_dw_to_keras(w):
    return w.detach().cpu().numpy().transpose(2, 3, 0, 1)


def pt_linear_to_conv1x1(w):
    arr = w.detach().cpu().numpy().T
    return arr.reshape(1, 1, arr.shape[0], arr.shape[1])


def keras_bn_weights(prefix):
    return [
        sd[f"{prefix}.weight"].detach().cpu().numpy(),
        sd[f"{prefix}.bias"].detach().cpu().numpy(),
        sd[f"{prefix}.running_mean"].detach().cpu().numpy(),
        sd[f"{prefix}.running_var"].detach().cpu().numpy(),
    ]


def pt_matched_bn(name):
    return tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=name)


def conv_pt_pad(x, layer, name):
    x = tf.keras.layers.ZeroPadding2D(1, name=f"{name}_pad")(x)
    return layer(x)


def build_keras_twin():
    def ds_block(x, cout, stride, name):
        x = conv_pt_pad(
            x,
            tf.keras.layers.DepthwiseConv2D(
                3, strides=stride, padding="valid", use_bias=False, name=f"{name}_dw"),
            f"{name}_dw",
        )
        x = pt_matched_bn(f"{name}_bn1")(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu1")(x)
        x = tf.keras.layers.Conv2D(cout, 1, use_bias=False, name=f"{name}_pw")(x)
        x = pt_matched_bn(f"{name}_bn2")(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu2")(x)
        return x

    inp = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input")
    x = conv_pt_pad(
        inp,
        tf.keras.layers.Conv2D(
            32, 3, strides=2, padding="valid", use_bias=False, name="stem_conv"),
        "stem",
    )
    x = pt_matched_bn("stem_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="stem_relu")(x)
    x = ds_block(x, 48, 2, "b0")
    x = ds_block(x, 64, 2, "b1")
    x = ds_block(x, 96, 2, "b2")
    x = ds_block(x, 128, 1, "b3")
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="gap")(x)
    x = tf.keras.layers.Conv2D(NUM_CLASSES, 1, use_bias=True, name="cls_conv")(x)
    out = tf.keras.layers.Reshape((NUM_CLASSES,), name="output")(x)
    return tf.keras.Model(inp, out, name="TinyLeafGate_ALT_TE")


with tf.device("/CPU:0"):
    kmodel = build_keras_twin()
    kmodel.get_layer("stem_conv").set_weights([pt_conv_to_keras(sd["stem.0.weight"])])
    kmodel.get_layer("stem_bn").set_weights(keras_bn_weights("stem.1"))
    for kname, ptname in [("b0", "blocks.0"), ("b1", "blocks.1"),
                          ("b2", "blocks.2"), ("b3", "blocks.3")]:
        kmodel.get_layer(f"{kname}_dw").set_weights([pt_dw_to_keras(sd[f"{ptname}.dw.weight"])])
        kmodel.get_layer(f"{kname}_bn1").set_weights(keras_bn_weights(f"{ptname}.bn1"))
        kmodel.get_layer(f"{kname}_pw").set_weights([pt_conv_to_keras(sd[f"{ptname}.pw.weight"])])
        kmodel.get_layer(f"{kname}_bn2").set_weights(keras_bn_weights(f"{ptname}.bn2"))
    kmodel.get_layer("cls_conv").set_weights([
        pt_linear_to_conv1x1(sd["head.3.weight"]),
        sd["head.3.bias"].detach().cpu().numpy(),
    ])
print("Keras twin built and weights copied", flush=True)

kmodel.trainable = False
n_check = min(16, len(val_dataset))
diffs = []
for i in range(n_check):
    img, _ = val_dataset[i]
    with torch.no_grad():
        pt_out = model(img.unsqueeze(0)).numpy().reshape(-1)
    x = img.numpy().transpose(1, 2, 0)[None, ...].astype(np.float32)
    k_out = kmodel.predict(x, verbose=0).reshape(-1)
    diffs.append(np.max(np.abs(pt_out - k_out)))
print(f"Float max|PT-Keras| over {n_check} images: "
      f"mean={np.mean(diffs):.5f}  max={np.max(diffs):.5f}", flush=True)
if np.max(diffs) > 0.01:
    raise RuntimeError(
        f"PT↔Keras logit mismatch too large (max={np.max(diffs):.5f}). Do not export."
    )

print("INT8 TFLite conversion", flush=True)
calib_ds = val_dataset


def make_rep_data():
    n = min(200, len(calib_ds))
    for i in range(n):
        img, _ = calib_ds[i]
        x = img.numpy().transpose(1, 2, 0)[None, ...].astype(np.float32)
        yield [x]


converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = make_rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
tflite_kb = os.path.getsize(TFLITE_PATH) / 1024.0
print(f"INT8 TFLite saved: {TFLITE_PATH}  ({tflite_kb:.1f} KB)", flush=True)
print("Alt leaf-gate training + export complete.", flush=True)
