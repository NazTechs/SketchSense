"""
train_quickdraw.py

Optional training script for SketchSense.

Why this exists:
- SketchSense's default mode uses a pretrained doodle classifier (345 categories).
- If you want an *offline* model, faster startup, or a *custom reduced label set*,
  you can train a small CNN directly on the official Quick, Draw! dataset.

Dataset:
- Homepage: https://quickdraw.withgoogle.com/data
- We'll use the official 28x28 NumPy bitmap files hosted on Google Cloud Storage:
  https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy

Important note:
- Each category .npy contains MANY samples (often 50-100+ MB per class).
  This script downloads ONLY the categories needed for SketchSense.

After training:
- The script saves weights to: SketchSense/models/quickdraw_9cls_cnn.pth
- The app can use this file in fixed-label mode (9 classes).

Run:
  cd SketchSense
  python train_quickdraw.py
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return

    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        total = int(response.headers.get("Content-Length") or "0")
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = (downloaded / total) * 100
                sys.stdout.write(f"\rDownloading {dest.name}: {pct:5.1f}%")
                sys.stdout.flush()
        if total:
            sys.stdout.write("\n")


def download_quickdraw_npy(category: str, out_dir: Path) -> Path:
    """
    Download one Quick, Draw! numpy bitmap file.
    Category names may include spaces (e.g. "smiley face").
    """
    from ai_model import QUICKDRAW_NPY_BASE_URL

    filename = f"{category}.npy"
    url = QUICKDRAW_NPY_BASE_URL + urllib.parse.quote(filename)
    dest = out_dir / filename
    _download(url, dest)
    return dest


def load_quickdraw_samples(npy_path: Path, max_items: int) -> np.ndarray:
    """
    Load up to `max_items` samples from a QuickDraw .npy file.

    Returns uint8 array shaped (N, 28, 28).
    The exact intensity convention depends on how you interpret the bitmaps;
    this script can auto-invert them later to match the app's preprocessing.
    """
    data = np.load(npy_path, mmap_mode="r")  # shape (N, 784)
    n = int(min(max_items, data.shape[0]))
    # Read the first chunk contiguously for speed (memmap friendly).
    return np.asarray(data[:n], dtype=np.uint8).reshape(n, 28, 28)


def build_model(num_classes: int):
    import torch
    from torch import nn

    class QuickDraw9CNN(nn.Module):
        def __init__(self, num_classes_: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(128, num_classes_),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    return QuickDraw9CNN(num_classes)


def main() -> None:
    from ai_model import DEFAULT_LABELS, LOCAL_QUICKDRAW_MODEL_PATH

    parser = argparse.ArgumentParser(description="Train a 9-class QuickDraw model for SketchSense.")
    parser.add_argument("--data-dir", default="quickdraw_data", help="Where to cache downloaded .npy files.")
    parser.add_argument("--samples-per-class", type=int, default=20000, help="Max samples per label.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-auto-invert",
        action="store_true",
        help=(
            "Disable auto-inversion of QuickDraw bitmaps. "
            "By default we detect if the downloaded .npy bitmaps appear inverted and fix them "
            "so training matches the app's preprocessing."
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Map SketchSense labels -> QuickDraw categories.
    # Note: QuickDraw doesn't have a literal "person" class, so we approximate it with faces.
    categories_by_label: dict[str, list[str]] = {
        "cat": ["cat"],
        "dog": ["dog"],
        "car": ["car"],
        "house": ["house"],
        "tree": ["tree"],
        "bird": ["bird"],
        "fish": ["fish"],
        "flower": ["flower"],
        "person": ["face", "smiley face"],
    }

    data_dir = Path(args.data_dir)

    # -------- Download + load --------
    print("Preparing Quick, Draw! data (this may take a while on first run)...")
    images_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []

    for label_idx, label in enumerate(DEFAULT_LABELS):
        src_categories = categories_by_label.get(label)
        if not src_categories:
            raise RuntimeError(f"No QuickDraw category mapping for label '{label}'.")

        per_source = max(1, args.samples_per_class // len(src_categories))
        for cat in src_categories:
            npy_path = download_quickdraw_npy(cat, data_dir)
            imgs = load_quickdraw_samples(npy_path, max_items=per_source)

            # Quick, Draw! numpy bitmaps are sometimes described with an "ink-as-high" convention
            # (background=0, stroke=255), but different tools may interpret or export them with
            # the opposite grayscale convention. We make training robust by auto-detecting the
            # convention and inverting when needed.
            if not args.no_auto_invert:
                mean_val = float(imgs.mean())
                # If the mean is very high, the images are mostly "white" (255) background,
                # suggesting stroke pixels are low. Invert so background becomes 0 and ink is 255.
                if mean_val > 127.0:
                    imgs = 255 - imgs
                    print(f"Auto-inverted '{cat}' bitmaps (mean={mean_val:.1f}).")

            images_parts.append(imgs)
            labels_parts.append(np.full((imgs.shape[0],), label_idx, dtype=np.int64))
            print(f"Loaded {imgs.shape[0]:6d} samples for '{label}' from '{cat}'.")

    images = np.concatenate(images_parts, axis=0)  # (N, 28, 28) uint8
    labels = np.concatenate(labels_parts, axis=0)  # (N,) int64

    # Shuffle
    idx = np.arange(images.shape[0])
    np.random.shuffle(idx)
    images = images[idx]
    labels = labels[idx]

    # Train/val split
    val_count = max(1000, int(0.1 * images.shape[0]))
    x_val = images[:val_count]
    y_val = labels[:val_count]
    x_train = images[val_count:]
    y_train = labels[val_count:]

    print(f"Train: {x_train.shape[0]}  Val: {x_val.shape[0]}  Classes: {len(DEFAULT_LABELS)}")

    # -------- Torch training --------
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Minimal augmentation to better match real user drawings (small shifts/rotations).
    train_tfms = T.Compose(
        [
            T.ToPILImage(),
            T.RandomAffine(degrees=10, translate=(0.10, 0.10), scale=(0.9, 1.1), fill=0),
            T.ToTensor(),  # -> float in [0, 1], shape (1, 28, 28)
        ]
    )
    val_tfms = T.Compose([T.ToTensor()])

    class NumpyQuickDraw(Dataset):
        def __init__(self, x: np.ndarray, y: np.ndarray, tfms):
            self.x = x
            self.y = y
            self.tfms = tfms

        def __len__(self) -> int:
            return int(self.y.shape[0])

        def __getitem__(self, i: int):
            img = self.x[i]  # (28, 28) uint8
            label = int(self.y[i])
            img_t = self.tfms(img)  # (1, 28, 28) float
            return img_t, label

    train_ds = NumpyQuickDraw(x_train, y_train, train_tfms)
    val_ds = NumpyQuickDraw(x_val, y_val, val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(DEFAULT_LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    def eval_accuracy() -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        return (correct / total) if total else 0.0

    best_acc = -math.inf
    print(f"Training on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            running_loss += float(loss.item()) * int(yb.numel())
            seen += int(yb.numel())

        acc = eval_accuracy()
        avg_loss = running_loss / max(1, seen)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs}  loss={avg_loss:.4f}  val_acc={acc*100:5.1f}%  ({dt:.1f}s)")

        if acc > best_acc:
            best_acc = acc
            LOCAL_QUICKDRAW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), LOCAL_QUICKDRAW_MODEL_PATH)
            print(f"Saved: {LOCAL_QUICKDRAW_MODEL_PATH}")

    print("\nDone.")
    print("Next: run the app and it will auto-use the local model:")
    print("  python SketchSense.py")


if __name__ == "__main__":
    main()
