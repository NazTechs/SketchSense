"""
worker.py

This module runs AI inference in a *separate process* to keep the GUI stable.

Why a subprocess instead of QThread?
- Some combinations of PyTorch / torchvision / transformers can hard-crash on Windows
  (native heap corruption / access violations). When that happens in-process, the
  whole GUI closes with no Python exception.
- Running inference out-of-process isolates those native crashes. If the worker
  process dies, the GUI stays open and we can restart the worker automatically.

Architecture:
- MainWindow starts a QProcess running: `python worker.py --worker`
- MainWindow sends requests via the worker's stdin (JSON lines).
- Worker returns results via stdout (JSON lines).

Message protocol (one JSON object per line):
Request:
  {"type":"predict","id":123,"png_b64":"..."}
Response:
  {"type":"result","id":123,"ok":true,"backend":"...","model_id":"...","top":{"label":"...","confidence":0.12},"top5":[...]}
Error:
  {"type":"result","id":123,"ok":false,"error":"..."}
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import traceback
from dataclasses import asdict

from PIL import Image

from ai_model import PredictionResult, SketchClassifier


def _safe_print_json(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _decode_png_b64(png_b64: str) -> Image.Image:
    raw = base64.b64decode(png_b64.encode("ascii"))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def run_worker() -> int:
    # Quiet HF noise in worker process.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

    clf = SketchClassifier(labels=None)  # full-vocab QuickDraw when available

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue

        if msg.get("type") == "ping":
            _safe_print_json({"type": "pong"})
            continue

        if msg.get("type") != "predict":
            continue

        req_id = msg.get("id")
        try:
            pil_img = _decode_png_b64(msg["png_b64"])
            result: PredictionResult = clf.predict(pil_img)
            payload = {
                "type": "result",
                "id": req_id,
                "ok": True,
                "backend": result.backend,
                "model_id": result.model_id,
                "top": asdict(result.top),
                "top5": [asdict(p) for p in result.top5],
            }
            _safe_print_json(payload)
        except Exception as exc:
            payload = {
                "type": "result",
                "id": req_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            }
            _safe_print_json(payload)

    return 0


class SubprocessPredictor:
    """
    GUI-side controller for the worker QProcess.
    """

    def __init__(self, qtcore, qtgui, qtwidgets, parent=None):
        self.QtCore = qtcore
        self.process = qtcore.QProcess(parent)
        self.process.setProcessChannelMode(qtcore.QProcess.ProcessChannelMode.MergedChannels)

        self._buf = bytearray()
        self._next_id = 1
        self._pending: dict[int, object] = {}

        self.process.readyReadStandardOutput.connect(self._on_ready_read)
        self.process.errorOccurred.connect(self._on_proc_error)
        self.process.finished.connect(self._on_proc_finished)

        # Callbacks (set by GUI)
        self.on_result = None  # (req_id:int, payload:dict) -> None
        self.on_worker_log = None  # (text:str) -> None

    def start(self) -> None:
        if self.process.state() != self.QtCore.QProcess.ProcessState.NotRunning:
            return

        py = sys.executable
        script = os.path.join(os.path.dirname(__file__), "worker.py")
        self.process.start(py, [script, "--worker"])

    def stop(self) -> None:
        if self.process.state() == self.QtCore.QProcess.ProcessState.NotRunning:
            return
        try:
            self.process.readyReadStandardOutput.disconnect(self._on_ready_read)
        except Exception:
            pass
        try:
            self.process.errorOccurred.disconnect(self._on_proc_error)
        except Exception:
            pass
        try:
            self.process.finished.disconnect(self._on_proc_finished)
        except Exception:
            pass
        self.process.terminate()
        self.process.waitForFinished(1500)
        if self.process.state() != self.QtCore.QProcess.ProcessState.NotRunning:
            self.process.kill()

    def is_running(self) -> bool:
        return self.process.state() != self.QtCore.QProcess.ProcessState.NotRunning

    def predict_png_bytes(self, png_bytes: bytes) -> int:
        if not self.is_running():
            self.start()
        req_id = self._next_id
        self._next_id += 1

        msg = {
            "type": "predict",
            "id": req_id,
            "png_b64": base64.b64encode(png_bytes).decode("ascii"),
        }
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        self.process.write(data)
        return req_id

    def _emit_log(self, text: str) -> None:
        if self.on_worker_log:
            self.on_worker_log(text)

    def _on_proc_error(self, err) -> None:
        self._emit_log(f"Worker process error: {err}")

    def _on_proc_finished(self, code, status) -> None:
        self._emit_log(f"Worker process finished (code={code}, status={status})")

    def _on_ready_read(self) -> None:
        chunk = bytes(self.process.readAllStandardOutput())
        if not chunk:
            return
        self._buf.extend(chunk)
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            line = self._buf[:nl].decode("utf-8", errors="replace").strip()
            del self._buf[: nl + 1]
            if not line:
                continue
            # Worker logs (non-JSON) are forwarded as plain text.
            if not (line.startswith("{") and line.endswith("}")):
                self._emit_log(line)
                continue
            try:
                payload = json.loads(line)
            except Exception:
                self._emit_log(line)
                continue
            if payload.get("type") == "result" and self.on_result:
                self.on_result(int(payload.get("id") or 0), payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help="Run in worker (stdin/stdout) mode.")
    args = ap.parse_args()
    if args.worker:
        return run_worker()
    print("This module is meant to be started by SketchSense.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
