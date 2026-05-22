# SketchSense

PySide6 drawing canvas + simple AI-powered predictions.

The GUI runs inference in an isolated subprocess (`worker.py`) so native crashes
from ML libraries won't take down the UI.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python SketchSense.py
```

## Use

- Draw on the canvas; predictions update on mouse release.
- Use **Open image…** to load an image into the canvas.
- If you want to feed raw encoded image bytes from code, call `feed_png_bytes()`.
