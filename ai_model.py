"""
ai_model.py

AI inference pipeline used by SketchSense.

Goal:
- Take a PIL image of the user's canvas drawing.
- Preprocess it to 224x224 RGB and normalize it.
- Run inference using a pretrained model if available.

Default backend (if available):
- A sketch-trained image classifier (Quick, Draw! / doodle recognition).
  This is usually much better for simple black-and-white sketches than models
  trained on natural photos.

Secondary backend:
- OpenAI CLIP via Hugging Face transformers for "zero-shot" classification.
  This can work surprisingly well, but it isn't trained specifically for
  doodles/sketches.

Fallback backend:
- A tiny deterministic heuristic that produces *some* output even if torch /
  transformers / model weights are unavailable. This keeps the app runnable.

Note for beginners:
CLIP and ImageNet models are not trained specifically for simple sketches.
If you want better sketch recognition, replace the backend with a sketch-specific
model (e.g., QuickDraw-trained classifier) or use a stronger open-vocabulary model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image, ImageFilter, ImageOps


DEFAULT_LABELS: list[str] = [
    "cat",
    "dog",
    "car",
    "house",
    "tree",
    "bird",
    "fish",
    "flower",
    "person",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float  # 0..1


@dataclass(frozen=True)
class PredictionResult:
    top: Prediction
    top5: List[Prediction]
    backend: str
    model_id: str | None = None


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Quick, Draw! dataset reference:
# - Homepage: https://quickdraw.withgoogle.com/data
# - Preprocessed 28x28 grayscale bitmaps per class:
#   https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy
#   Each .npy row is a flattened 28*28 image. Pixel values are 0=white background, 255=black stroke.
QUICKDRAW_NPY_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

# Optional: If you train a small 9-class model from the Quick, Draw! dataset yourself,
# save its state_dict here and SketchSense will use it automatically.
LOCAL_QUICKDRAW_MODEL_PATH = Path(__file__).resolve().parent / "models" / "quickdraw_9cls_cnn.pth"

# Sketch-specific model candidates (preferred over ImageNet-style models).
#
# 1) Xenova/quickdraw-mobilevit-small
#    - Trained for doodle/sketch recognition (Quick, Draw! 345 classes)
#    - Very fast (small input size / lightweight backbone), good for "predict on mouse release"
#
# 2) kmewhort/beit-sketch-classifier
#    - Fine-tuned on Quick, Draw! sketches, but tends to be slower/heavier.
QUICKDRAW_SKETCH_MODEL_CANDIDATES = [
    "Xenova/quickdraw-mobilevit-small",
    "kmewhort/beit-sketch-classifier",
]

# Optional: local copy of the Hugging Face model repositories.
# If you are offline / behind a firewall, you can download the model files once
# and place them under `SketchSense/models/hf/<repo_id>/` and the app will load
# from disk (no network required).
LOCAL_HF_MODELS_DIR = Path(__file__).resolve().parent / "models" / "hf"


def _canon(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


# Some sketch datasets/models don't include a literal "person" class (Quick, Draw!
# tends to have classes like "face" / "smiley face"). We still must output the
# required labels, so we map "person" to close sketch categories.
TARGET_LABEL_SYNONYMS: dict[str, list[str]] = {
    # Vehicles: users often draw "truck-ish" cars, and QuickDraw has multiple vehicle classes.
    "car": ["car", "police car", "pickup truck", "truck", "bus", "van"],
    # Trees: QuickDraw has both "tree" and "palm tree".
    "tree": ["tree", "palm tree"],
    # Birds: the model might predict a specific bird type instead of the generic "bird".
    "bird": ["bird", "duck", "flamingo", "owl", "parrot", "penguin", "swan"],
    # Fish: the model might predict a specific sea creature.
    "fish": ["fish", "shark", "whale", "dolphin", "sea turtle", "octopus"],
    # People: many sketch datasets use "face" / "smiley face" rather than a full person.
    "person": ["person", "stick figure", "face", "smiley face", "smiley_face", "skull"],
}

# When doing token/substring-style matching, avoid false friends.
TOKEN_MATCH_EXCLUSIONS: dict[str, set[str]] = {
    "dog": {"hot dog", "hot-dog"},
    "house": {"house plant"},
}


def preprocess_canvas_pil(pil_image: Image.Image) -> Image.Image:
    """
    Preprocess the canvas image for model input:
    - convert grayscale to RGB if needed
    - resize to 224x224

    We keep a PIL image output here so the GUI can show a "processed preview"
    (this is the same 224x224 input size used for inference).
    """
    # Convert to grayscale first so we can do sketch-style preprocessing.
    gray = pil_image.convert("L")

    # Autocrop empty borders (keeps the sketch centered better).
    # getbbox() finds the bounding box of non-zero pixels, so we invert first.
    bbox = ImageOps.invert(gray).getbbox()
    if bbox:
        gray = gray.crop(bbox)

    # Resize to the model-friendly square size without squashing the aspect ratio.
    # This keeps sketches centered and avoids distortions that hurt sketch classifiers.
    gray = ImageOps.pad(
        gray,
        (224, 224),
        method=Image.Resampling.BICUBIC,
        color=255,
        centering=(0.5, 0.5),
    )

    # Make strokes more "dataset-like":
    # - autocontrast to strengthen faint strokes
    # - threshold to remove anti-aliasing grays (QuickDraw-style sketches are crisp)
    gray = ImageOps.autocontrast(gray)
    threshold = 200
    bw = gray.point(lambda p: 255 if p > threshold else 0)

    # Slightly thicken strokes: invert -> dilate white -> invert back.
    ink = ImageOps.invert(bw).filter(ImageFilter.MaxFilter(3))
    bw = ImageOps.invert(ink)

    return bw.convert("RGB")


def preprocess_canvas_quickdraw_bitmap(pil_image: Image.Image) -> Image.Image:
    """
    Preprocess for Quick, Draw! 28x28 bitmap models / training data.

    Output:
    - mode "L", size 28x28
    - pixel convention matches Quick, Draw! numpy bitmaps:
        0 = white background
        255 = black stroke

    Why this exists:
    - The GUI canvas is "black ink on white background".
    - Quick, Draw! bitmap files are stored as 0=white, 255=black.
    - Some sketch-trained models (e.g. QuickDraw MobileViT) use this convention.
    """
    preview_224 = preprocess_canvas_pil(pil_image)  # RGB 224x224, binarized + centered

    # Convert back to grayscale and downsample to 28x28.
    # NOTE: Downsampling can introduce gray anti-aliasing values around strokes,
    # so we re-binarize to keep the bitmap "dataset-like".
    gray = preview_224.convert("L").resize((28, 28), Image.Resampling.BICUBIC)

    # Re-binarize after resize to avoid faint gray halos.
    # Our preview has black strokes (0) on white background (255).
    threshold = 200
    bw = gray.point(lambda p: 255 if p > threshold else 0)

    # QuickDraw bitmaps are commonly used with an "ink-as-high" convention:
    # background = 0, stroke = 255. Invert to match that convention.
    return ImageOps.invert(bw)


class SketchClassifier:
    """
    Zero-shot sketch classifier with a robust fallback.

    The model is loaded lazily to keep the UI responsive at startup.
    """

    def __init__(self, labels: list[str] | None = None) -> None:
        # If `labels` is provided, we restrict predictions to that fixed vocabulary.
        #
        # If `labels` is None, we prefer using the full Quick, Draw! vocabulary from a
        # QuickDraw-trained classifier (345 categories). This matches the user's request:
        # "use a lot of labels so everything I draw can be detected".
        self._fixed_label_set = labels is not None
        # In "full vocabulary" mode (labels=None), we still keep a small default
        # label list so the app can show *something* if the QuickDraw model can't
        # be downloaded/loaded (e.g. no internet). Once the QuickDraw model loads,
        # we replace this with the full 345-category label set from the model.
        self.labels = labels[:] if labels is not None else DEFAULT_LABELS[:]
        self._backend: str | None = None
        self._load_error: str | None = None

        # Optional local QuickDraw-trained 9-class model (loaded lazily)
        self._qd_torch = None
        self._qd_model = None
        self._qd_device = None

        # QuickDraw/Sketch model members (created lazily)
        self._sk_torch = None
        self._sk_processor = None
        self._sk_model = None
        self._sk_device = None
        self._sk_model_id: str | None = None
        self._sk_target_label_indices: list[list[int]] | None = None
        self._sk_id2label: dict[int, str] | None = None
        self._sk_preprocess_kind: str | None = None

        # CLIP members (created lazily)
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._device = None
        self._text_features = None

        logger.info(
            "SketchClassifier init (fixed_labels=%s, labels_count=%s)",
            self._fixed_label_set,
            len(self.labels),
        )

    @property
    def backend(self) -> str:
        return self._backend or "uninitialized"

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def pop_load_error(self) -> str | None:
        """
        Return the last model load error (if any) and clear it.
        Useful to show a one-time warning in the UI without spamming.
        """
        err = self._load_error
        self._load_error = None
        return err

    def predict(self, pil_image: Image.Image) -> PredictionResult:
        """
        Returns the top prediction + confidence and the top-5 list.
        """
        if self._backend is None:
            # Prefer:
            # 1) A local 9-class model trained from the official Quick, Draw! dataset (if you trained it)
            # 2) A general QuickDraw sketch classifier (345 classes) projected down to our 9 labels
            # 3) CLIP (zero-shot)
            # 4) Heuristic fallback
            # If we're using the full QuickDraw vocabulary (labels=None), skip the
            # local 9-class model (it can't cover 345 categories).
            if self._fixed_label_set:
                self._try_init_local_quickdraw_model()
            if self._backend is None:
                self._try_init_quickdraw_sketch_model()
                if self._backend is None and self._fixed_label_set:
                    self._try_init_clip()

            logger.info("Selected backend: %s", self._backend or "none")

        processed = preprocess_canvas_pil(pil_image)

        if self._backend == "quickdraw-local":
            return self._predict_local_quickdraw_model(pil_image)

        if self._backend == "quickdraw-hf":
            if self._fixed_label_set:
                return self._predict_quickdraw_sketch_model(pil_image, processed)
            return self._predict_quickdraw_sketch_model_full(pil_image, processed)

        if self._backend == "clip":
            return self._predict_clip(processed)

        # Always return something in fallback mode.
        return self._predict_fallback(processed)

    # ----------------------------------------
    # Local QuickDraw 9-class backend (trained by user)
    # ----------------------------------------
    def _try_init_local_quickdraw_model(self) -> None:
        """
        If a local state_dict exists at `SketchSense/models/quickdraw_9cls_cnn.pth`,
        load it and use it as the primary backend.

        This lets you train on the official Quick, Draw! dataset (from:
        https://quickdraw.withgoogle.com/data) specifically for the 9 required labels.

        See `SketchSense/train_quickdraw.py` for a starter training script.
        """
        try:
            if not self._fixed_label_set:
                # The local model is specifically a 9-class model; only use it when
                # the app is running in fixed-label mode.
                return

            if not LOCAL_QUICKDRAW_MODEL_PATH.exists():
                return

            import torch
            from torch import nn

            device = "cuda" if torch.cuda.is_available() else "cpu"

            class QuickDraw9CNN(nn.Module):
                def __init__(self, num_classes: int) -> None:
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
                        nn.Linear(128, num_classes),
                    )

                def forward(self, x):
                    x = self.features(x)
                    return self.classifier(x)

            model = QuickDraw9CNN(num_classes=len(self.labels))

            try:
                state = torch.load(LOCAL_QUICKDRAW_MODEL_PATH, map_location=device, weights_only=True)
            except TypeError:
                # Older PyTorch versions don't support weights_only.
                state = torch.load(LOCAL_QUICKDRAW_MODEL_PATH, map_location=device)

            model.load_state_dict(state)
            model.eval()
            model.to(device)

            self._qd_torch = torch
            self._qd_model = model
            self._qd_device = device

            self._backend = "quickdraw-local"
            self._load_error = None
            logger.info("Loaded local QuickDraw model: %s", str(LOCAL_QUICKDRAW_MODEL_PATH))
        except Exception as exc:
            # Not fatal: fall back to other backends.
            self._backend = None
            self._load_error = (
                "Local QuickDraw model found but failed to load. Will try other backends.\n"
                f"Details: {exc}"
            )
            logger.exception("Failed to load local QuickDraw model: %s", exc)

    def _predict_local_quickdraw_model(self, pil_image: Image.Image) -> PredictionResult:
        """
        Predict using a locally trained 9-class CNN trained on Quick, Draw! bitmaps.
        """
        torch = self._qd_torch

        # Convert canvas -> QuickDraw-style 28x28 bitmap.
        bmp28 = preprocess_canvas_quickdraw_bitmap(pil_image)  # "L" 28x28

        def infer_probs(img28: Image.Image):
            # To tensor: [1, 1, 28, 28] in 0..1 range (same as dataset rescale_factor=1/255).
            import numpy as np

            arr = np.asarray(img28, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self._qd_device)
            with torch.no_grad():
                logits = self._qd_model(x)[0]
            return logits.softmax(dim=-1)

        probs = infer_probs(bmp28)

        # Some QuickDraw pipelines/models use the opposite intensity convention.
        # To be robust to a mismatch between training and inference conventions,
        # we optionally try an inverted bitmap and keep the more confident result.
        if float(probs.max().item()) < 0.65:
            probs_inv = infer_probs(ImageOps.invert(bmp28))
            if float(probs_inv.max().item()) > float(probs.max().item()):
                probs = probs_inv

        topk = min(5, probs.numel())
        values, indices = probs.topk(topk)

        top5: list[Prediction] = []
        for score, idx in zip(values.tolist(), indices.tolist(), strict=True):
            top5.append(Prediction(label=self.labels[idx], confidence=float(score)))

        top = top5[0] if top5 else Prediction(label=self.labels[0], confidence=0.0)
        return PredictionResult(top=top, top5=top5, backend="quickdraw-local", model_id=str(LOCAL_QUICKDRAW_MODEL_PATH))

    # ----------------------------------------
    # QuickDraw / Sketch-trained (preferred) backend
    # ----------------------------------------
    def _try_init_quickdraw_sketch_model(self) -> None:
        """
        Try to initialize a sketch-trained classifier fine-tuned on Quick, Draw!.

        Why this helps:
        Models trained on natural photos (ImageNet) often fail on doodles.
        A QuickDraw-trained model usually works much better for line drawings.
        """
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            device = "cuda" if torch.cuda.is_available() else "cpu"

            last_exc: Exception | None = None
            for model_id in QUICKDRAW_SKETCH_MODEL_CANDIDATES:
                try:
                    # Prefer local on-disk copies when present (offline-friendly).
                    local_dir = LOCAL_HF_MODELS_DIR / model_id
                    load_id = str(local_dir) if local_dir.exists() else model_id

                    logger.info("Loading sketch model candidate: %s", load_id)
                    processor = AutoImageProcessor.from_pretrained(load_id)
                    # Force using the original model weights as-is. This avoids any optional
                    # `.bin` -> `.safetensors` conversion logic that can run in the background.
                    model = AutoModelForImageClassification.from_pretrained(load_id, use_safetensors=False)
                    model.eval()
                    model.to(device)

                    id2label = {int(k): v for k, v in model.config.id2label.items()}
                    self._sk_id2label = id2label

                    # Some processors route through torchvision's PIL->tensor conversion,
                    # which has caused native crashes (heap corruption) on some Windows
                    # setups when used from a QThread. To avoid this, we use a minimal,
                    # pure-numpy preprocessing path for BEiT-like models.
                    model_type = getattr(model.config, "model_type", "") or ""
                    preprocess_kind = "auto"
                    if model_type.lower().startswith("beit"):
                        preprocess_kind = "beit-numpy"
                    self._sk_preprocess_kind = preprocess_kind

                    # If the caller did not provide a label vocabulary, use the model's
                    # native label set (Quick, Draw! 345 categories).
                    if not self._fixed_label_set:
                        ordered = [id2label[i] for i in sorted(id2label)]
                        if not ordered:
                            raise RuntimeError(f"Sketch model '{model_id}' has empty id2label.")
                        self.labels = ordered

                        self._sk_torch = torch
                        self._sk_processor = processor
                        self._sk_model = model
                        self._sk_device = device
                        self._sk_model_id = load_id
                        self._sk_target_label_indices = None

                        self._backend = "quickdraw-hf"
                        self._load_error = None
                        logger.info("Loaded QuickDraw sketch model (full vocab): %s", load_id)
                        return

                    # Otherwise, build a mapping from our required labels -> indices in the model output.
                    # The model may have hundreds of classes; we "project" them down to our fixed label set.
                    canon_id2label = {idx: _canon(name) for idx, name in id2label.items()}

                    target_label_indices: list[list[int]] = []
                    for target in self.labels:
                        wanted = {_canon(target), *(_canon(s) for s in TARGET_LABEL_SYNONYMS.get(target, []))}
                        indices = [idx for idx, name in canon_id2label.items() if name in wanted]

                        # If direct synonym matching failed, fall back to a token match.
                        # Example: "police car" should count as "car".
                        if not indices:
                            token = _canon(target)
                            excluded = TOKEN_MATCH_EXCLUSIONS.get(target, set())
                            for idx, name in canon_id2label.items():
                                if name in excluded:
                                    continue
                                name_words = set(name.split())
                                token_words = set(token.split())
                                if token_words and token_words.issubset(name_words):
                                    indices.append(idx)

                        target_label_indices.append(indices)

                    # If we can't map *any* labels, this backend is useless.
                    if not any(target_label_indices):
                        raise RuntimeError(
                            f"Sketch model '{model_id}' loaded, but could not map any target labels."
                        )

                    self._sk_torch = torch
                    self._sk_processor = processor
                    self._sk_model = model
                    self._sk_device = device
                    self._sk_model_id = load_id
                    self._sk_target_label_indices = target_label_indices

                    self._backend = "quickdraw-hf"
                    self._load_error = None
                    logger.info("Loaded QuickDraw sketch model (projected labels): %s", load_id)
                    return
                except Exception as exc:
                    last_exc = exc
                    logger.exception("Failed sketch model candidate '%s': %s", model_id, exc)

            raise RuntimeError(f"All sketch model candidates failed. Last error: {last_exc}")
        except Exception as exc:
            # Not fatal: in fixed-label mode we can still try CLIP; otherwise we fall back.
            self._backend = None

            hint = ""
            if isinstance(exc, ModuleNotFoundError):
                if (exc.name or "").startswith("torch"):
                    hint = (
                        "Missing dependency: PyTorch.\n"
                        "Fix: install torch + torchvision (see requirements.txt / pytorch.org).\n"
                    )
                elif (exc.name or "").startswith("transformers"):
                    hint = (
                        "Missing dependency: transformers.\n"
                        "Fix: `pip install -r requirements.txt`\n"
                    )

            # Common case: first run needs to download weights from Hugging Face.
            # If the machine has no internet / blocked by firewall, the load will fail
            # and the app will drop to heuristic fallback.
            self._load_error = (
                "Quick, Draw! doodle model unavailable.\n"
                + hint
                + "Common causes:\n"
                "- Missing packages (torch/transformers)\n"
                "- No internet access to download model weights on first run\n"
                "- Corporate proxy/firewall blocking huggingface.co\n"
                "\nOffline option:\n"
                f"- Download the model once and place it under: {LOCAL_HF_MODELS_DIR / 'Xenova/quickdraw-mobilevit-small'}\n"
                f"\nDetails: {exc}"
            )
            logger.exception("QuickDraw sketch model init failed: %s", exc)

    def _predict_quickdraw_sketch_model(self, pil_image: Image.Image, processed: Image.Image) -> PredictionResult:
        """
        Run the sketch-trained classifier and convert its many-class output to our 9 labels.

        We do this by aggregating the logits of all source classes mapped to each target label.
        (We use log-sum-exp pooling, which is a smooth max that also rewards multiple supporting classes.)
        """
        torch = self._sk_torch

        def infer_logits(img: Image.Image):
            # Avoid torchvision conversion on some Windows setups (can crash).
            if self._sk_preprocess_kind == "beit-numpy":
                import numpy as np

                rgb = img.convert("RGB")
                arr = np.asarray(rgb, dtype=np.float32) / 255.0  # HWC
                mean = getattr(self._sk_model.config, "image_mean", None) or [0.5, 0.5, 0.5]
                std = getattr(self._sk_model.config, "image_std", None) or [0.5, 0.5, 0.5]
                arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
                pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._sk_device)
                with torch.no_grad():
                    return self._sk_model(pixel_values=pixel_values).logits[0]

            inputs = self._sk_processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._sk_device) for k, v in inputs.items()}
            with torch.no_grad():
                return self._sk_model(**inputs).logits[0]  # (num_classes,)

        def project_logits_to_probs(logits_all):
            # Convert many-class logits -> our fixed label set logits.
            target_logits = []
            for indices in self._sk_target_label_indices or []:
                if not indices:
                    # If a label couldn't be mapped, force it to a very low logit.
                    target_logits.append(torch.tensor(-1e9, device=logits_all.device))
                else:
                    src = logits_all[indices]
                    # logsumexp is a smooth max and corresponds to summing probabilities in log-space.
                    target_logits.append(src.logsumexp(dim=0))

            target_logits = torch.stack(target_logits, dim=0)  # (num_targets,)
            return target_logits.softmax(dim=-1)

        # Some sketch models (e.g., QuickDraw MobileViT) expect a 1-channel input,
        # while others (e.g., BEiT) expect 3-channel RGB. We select the right
        # input image based on `config.num_channels` when available.
        img_for_model = processed
        try:
            num_channels = getattr(self._sk_model.config, "num_channels", None)
            if num_channels == 1:
                img_for_model = preprocess_canvas_quickdraw_bitmap(pil_image)  # "L" 28x28, QuickDraw convention
        except Exception:
            pass

        logits_all = infer_logits(img_for_model)
        probs = project_logits_to_probs(logits_all)

        # Some QuickDraw bitmap models may have an inverted intensity convention.
        # To be robust, we optionally try an inverted version and keep the more confident result.
        # (This is kept cheap by only running the second pass if confidence is low.)
        try_second_pass = float(probs.max().item()) < 0.65
        if try_second_pass and (self._sk_model_id or "").startswith("Xenova/quickdraw") and img_for_model.mode == "L":
            logits_inv = infer_logits(ImageOps.invert(img_for_model))
            probs_inv = project_logits_to_probs(logits_inv)
            if float(probs_inv.max().item()) > float(probs.max().item()):
                probs = probs_inv

        topk = min(5, probs.numel())
        values, indices = probs.topk(topk)

        top5: list[Prediction] = []
        for score, idx in zip(values.tolist(), indices.tolist(), strict=True):
            top5.append(Prediction(label=self.labels[idx], confidence=float(score)))

        top = top5[0] if top5 else Prediction(label=self.labels[0], confidence=0.0)
        return PredictionResult(top=top, top5=top5, backend="quickdraw-hf", model_id=self._sk_model_id)

    def _predict_quickdraw_sketch_model_full(self, pil_image: Image.Image, processed: Image.Image) -> PredictionResult:
        """
        Predict using the model's *native* label space (e.g. Quick, Draw! 345 categories).

        This is the preferred mode when you want "lots of labels" for doodle recognition.
        """
        torch = self._sk_torch
        if torch is None or self._sk_model is None or self._sk_processor is None:
            raise RuntimeError("Sketch model not initialized.")

        raw_id2label = self._sk_id2label or getattr(self._sk_model.config, "id2label", None)
        if not raw_id2label:
            raise RuntimeError("Sketch model id2label not available.")
        id2label = {int(k): v for k, v in raw_id2label.items()}

        def infer_logits(img: Image.Image):
            # Avoid torchvision conversion on some Windows setups (can crash).
            if self._sk_preprocess_kind == "beit-numpy":
                import numpy as np

                rgb = img.convert("RGB")
                arr = np.asarray(rgb, dtype=np.float32) / 255.0  # HWC
                mean = getattr(self._sk_model.config, "image_mean", None) or [0.5, 0.5, 0.5]
                std = getattr(self._sk_model.config, "image_std", None) or [0.5, 0.5, 0.5]
                arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
                pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._sk_device)
                with torch.no_grad():
                    return self._sk_model(pixel_values=pixel_values).logits[0]

            inputs = self._sk_processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._sk_device) for k, v in inputs.items()}
            with torch.no_grad():
                return self._sk_model(**inputs).logits[0]  # (num_classes,)

        # Select the right input representation based on the model config.
        img_for_model = processed
        try:
            num_channels = getattr(self._sk_model.config, "num_channels", None)
            if num_channels == 1:
                img_for_model = preprocess_canvas_quickdraw_bitmap(pil_image)  # "L" 28x28, ink-as-high
        except Exception:
            pass

        logits = infer_logits(img_for_model)
        probs = logits.softmax(dim=-1)

        # Some QuickDraw bitmap models may have an inverted intensity convention.
        # To be robust, optionally try an inverted image and keep the more confident result.
        try_second_pass = float(probs.max().item()) < 0.65
        if try_second_pass and (self._sk_model_id or "").startswith("Xenova/quickdraw") and img_for_model.mode == "L":
            logits_inv = infer_logits(ImageOps.invert(img_for_model))
            probs_inv = logits_inv.softmax(dim=-1)
            if float(probs_inv.max().item()) > float(probs.max().item()):
                probs = probs_inv

        topk = min(5, probs.numel())
        values, indices = probs.topk(topk)

        top5: list[Prediction] = []
        for score, idx in zip(values.tolist(), indices.tolist(), strict=True):
            label = id2label.get(int(idx), str(int(idx)))
            top5.append(Prediction(label=label, confidence=float(score)))

        top = top5[0] if top5 else Prediction(label="unknown", confidence=0.0)
        return PredictionResult(top=top, top5=top5, backend="quickdraw-hf", model_id=self._sk_model_id)

    # -----------------------
    # CLIP (secondary) backend
    # -----------------------
    def _try_init_clip(self) -> None:
        """
        Try to initialize the CLIP backend.
        If anything fails (missing packages, missing weights, no internet),
        we switch to fallback mode.
        """
        try:
            import torch
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode
            from transformers import CLIPModel, CLIPTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"

            model_id = "openai/clip-vit-base-patch32"
            model = CLIPModel.from_pretrained(model_id)
            tokenizer = CLIPTokenizer.from_pretrained(model_id)

            model.eval()
            model.to(device)

            # Our required preprocessing pipeline:
            # - 224x224 RGB
            # - normalize with CLIP mean/std
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (224, 224),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(CLIP_MEAN, CLIP_STD),
                ]
            )

            # Precompute label text embeddings once for fast inference.
            prompts = [f"a sketch of a {label}" for label in self.labels]
            tokens = tokenizer(prompts, padding=True, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                text_features = model.get_text_features(**tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            self._torch = torch
            self._model = model
            self._tokenizer = tokenizer
            self._transform = transform
            self._device = device
            self._text_features = text_features

            self._backend = "clip"
            # If we already have a warning (e.g., sketch model couldn't load),
            # keep it so the UI can show it once. Otherwise clear errors.
            if not self._load_error:
                self._load_error = None
            logger.info("Loaded CLIP backend: %s", model_id)
        except Exception as exc:
            self._backend = "fallback"
            hint = ""
            if isinstance(exc, ModuleNotFoundError) and (exc.name or "").startswith("transformers"):
                hint = "Missing dependency: transformers. Fix: `pip install -r requirements.txt`\n"
            self._load_error = (
                "CLIP backend unavailable. Falling back to a simple heuristic.\n"
                + hint
                + f"Details: {exc}"
            )
            logger.exception("CLIP init failed: %s", exc)

    def _predict_clip(self, processed: Image.Image) -> PredictionResult:
        torch = self._torch

        pixel_values = self._transform(processed).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_features = self._model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Similarity scores -> probabilities over the allowed labels.
        # The 100.0 scale is a common CLIP convention.
        logits = 100.0 * (image_features @ self._text_features.T)
        probs = logits.softmax(dim=-1)[0]

        topk = min(5, probs.numel())
        values, indices = probs.topk(topk)

        top5: list[Prediction] = []
        for score, idx in zip(values.tolist(), indices.tolist(), strict=True):
            top5.append(Prediction(label=self.labels[idx], confidence=float(score)))

        top = top5[0] if top5 else Prediction(label=self.labels[0], confidence=0.0)
        return PredictionResult(top=top, top5=top5, backend="clip", model_id="openai/clip-vit-base-patch32")

    # -----------------------
    # Fallback backend
    # -----------------------
    def _predict_fallback(self, processed: Image.Image) -> PredictionResult:
        """
        A deterministic heuristic:
        - compute "ink coverage" (how many dark pixels exist)
        - map coverage to a label bucket

        This is NOT a real sketch recognizer; it just keeps the app functional.
        Replace this with a real sketch model (QuickDraw, CLIP variants, etc.)
        when you want accuracy.
        """
        gray = processed.convert("L")
        # Invert so "ink" becomes high values.
        inv = ImageOps.invert(gray)
        # Normalize to 0..1 and average.
        ink = sum(inv.getdata()) / (255.0 * inv.size[0] * inv.size[1])

        # Pick a label based on rough thresholds.
        buckets = [
            (0.02, "fish"),
            (0.04, "bird"),
            (0.07, "flower"),
            (0.10, "tree"),
            (0.14, "house"),
            (0.18, "car"),
            (0.24, "cat"),
            (0.30, "dog"),
            (1.01, "person"),
        ]
        label = "person"
        for threshold, candidate in buckets:
            if ink <= threshold:
                label = candidate
                break

        # Fake a confidence that "looks" like a probability but stays modest.
        confidence = float(max(0.10, min(0.55, 0.10 + ink)))

        ordered = [label] + [l for l in self.labels if l != label]
        top5 = [
            Prediction(label=ordered[i], confidence=max(0.01, confidence * (0.8**i)))
            for i in range(min(5, len(ordered)))
        ]
        return PredictionResult(top=top5[0], top5=top5, backend="fallback", model_id=None)
