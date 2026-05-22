# SketchSense

SketchSense is a small PySide6 desktop app with a drawing canvas and live predictions (updated on mouse release).

Model inference runs in an isolated subprocess (`worker.py`) so native crashes from ML libraries do not take down the GUI.

## Features

- Freehand drawing canvas with adjustable brush size
- Load an image into the canvas (Open image...)
- Live top-1 and top-5 predictions with confidence
- Built-in previews of the model input transforms
- Subprocess-based inference for robustness

## Requirements

- Python 3.10+
- Windows/macOS/Linux should work (instructions below assume Windows PowerShell)

## Quick start (Windows)

```powershell
cd SketchSense
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python SketchSense.py
```

## Usage

- Draw on the canvas; predictions update on mouse release.
- Use the **Open image...** button to load an image into the canvas.
- Use **Clear canvas** to reset.

## Feeding raw PNG bytes (API)

If you already have encoded PNG bytes and want to push them into the canvas:

```python
from SketchSense import MainWindow

win = MainWindow()
win.feed_png_bytes(png_bytes)
win.show()
```

## Models and offline use

On first run, the default sketch model may need to download weights (Hugging Face). If you are offline or behind a firewall, SketchSense will still run but may fall back to a lightweight heuristic backend.

Options for offline/controlled environments:

- Train a small local model (fixed 9 labels) and save it to `SketchSense/models/quickdraw_9cls_cnn.pth`:

  ```powershell
  cd SketchSense
  .\.venv\Scripts\python train_quickdraw.py
  ```

- Pre-download Hugging Face model files and place them under `SketchSense/models/hf/<repo_id>/` (see `ai_model.py` for the expected directory layout).

## Project layout

- `SketchSense.py`: main window, UI wiring, previews, and worker lifecycle
- `drawing_canvas.py`: paint-like canvas widget
- `worker.py`: inference subprocess wrapper (IPC + restart logic)
- `ai_model.py`: preprocessing + model selection/fallbacks
- `train_quickdraw.py`: optional training script for an offline 9-class model

## License

No license is included yet. If you plan to publish this as open source, add a `LICENSE` file before sharing the repository publicly.
