"""
SketchSense.py (main entry point)

SketchSense:
- PySide6 drawing canvas
- Auto-predict on mouse release
- Modern dark UI
- AI inference runs in an isolated subprocess (worker.py) so native crashes
  from ML libraries won't close the GUI.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PIL import Image

from ai_model import preprocess_canvas_pil, preprocess_canvas_quickdraw_bitmap
from drawing_canvas import DrawingCanvas
from worker import SubprocessPredictor


def qimage_to_png_bytes(qimage: QtGui.QImage) -> bytes:
    ba = QtCore.QByteArray()
    buf = QtCore.QBuffer(ba)
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    qimage.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def pil_to_qpixmap(pil_image: Image.Image) -> QtGui.QPixmap:
    bio = io.BytesIO()
    pil_image.save(bio, format="PNG")
    qimg = QtGui.QImage.fromData(bio.getvalue(), "PNG")
    return QtGui.QPixmap.fromImage(qimg)


def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(20, 22, 26))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(230, 230, 230))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(28, 31, 36))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(35, 39, 45))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(230, 230, 230))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(35, 39, 45))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(230, 230, 230))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(80, 140, 255))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(10, 10, 10))
    app.setPalette(palette)

    app.setStyleSheet(
        """
        QMainWindow { background: #14161A; }
        QGroupBox {
            border: 1px solid #2B2F36;
            border-radius: 10px;
            margin-top: 12px;
            padding: 10px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: #C9D1D9;
        }
        QPushButton {
            background: #2A2F38;
            border: 1px solid #3A404B;
            padding: 10px 12px;
            border-radius: 10px;
        }
        QPushButton:hover { background: #323846; }
        QPushButton:pressed { background: #262B33; }
        QSlider::groove:horizontal {
            height: 8px;
            background: #2B2F36;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            width: 18px;
            margin: -6px 0;
            background: #4F8CFF;
            border-radius: 9px;
        }
        QTableWidget {
            background: #1B1F26;
            border: 1px solid #2B2F36;
            border-radius: 10px;
            gridline-color: #2B2F36;
        }
        QLabel#TitleLabel { font-size: 20px; font-weight: 800; color: #FFFFFF; }
        QLabel#BigValue { font-size: 18px; font-weight: 800; color: #FFFFFF; }
        QLabel#Subtle { color: #9AA4B2; }
        """
    )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SketchSense")
        self.resize(1180, 720)

        self._prediction_in_progress = False
        self._pending_prediction = False
        self._last_request_id: Optional[int] = None

        self._build_ui()
        self._predictor = SubprocessPredictor(QtCore, QtGui, QtWidgets, parent=self)
        self._predictor.on_result = self._on_worker_result
        self._predictor.on_worker_log = self._append_worker_log
        self._predictor.start()

        self.canvas.drawingFinished.connect(self.auto_predict)
        QtCore.QTimer.singleShot(0, self.auto_predict)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            # Prevent late QProcess signals from calling back into deleted widgets.
            self._predictor.on_result = None
            self._predictor.on_worker_log = None
            self._predictor.stop()
        except Exception:
            pass
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(12)

        title = QtWidgets.QLabel("SketchSense")
        title.setObjectName("TitleLabel")
        subtitle = QtWidgets.QLabel("Draw something — predictions update on mouse release.")
        subtitle.setObjectName("Subtle")
        left.addWidget(title)
        left.addWidget(subtitle)

        canvas_frame = QtWidgets.QFrame()
        canvas_frame.setStyleSheet("QFrame { background: #FFFFFF; border-radius: 12px; }")
        cfl = QtWidgets.QVBoxLayout(canvas_frame)
        cfl.setContentsMargins(10, 10, 10, 10)
        self.canvas = DrawingCanvas()
        cfl.addWidget(self.canvas)
        left.addWidget(canvas_frame, 1)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(12)
        brush_label = QtWidgets.QLabel("Brush size")
        brush_label.setObjectName("Subtle")
        self.brush_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brush_slider.setRange(2, 48)
        self.brush_slider.setValue(18)
        self.brush_slider.valueChanged.connect(self.canvas.set_brush_size)
        self.open_btn = QtWidgets.QPushButton("Open image…")
        self.open_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton))
        self.open_btn.clicked.connect(self.open_image_dialog)
        self.clear_btn = QtWidgets.QPushButton("Clear canvas")
        self.clear_btn.clicked.connect(self.canvas.clear)
        controls.addWidget(brush_label)
        controls.addWidget(self.brush_slider, 1)
        controls.addWidget(self.open_btn)
        controls.addWidget(self.clear_btn)
        left.addLayout(controls)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(12)

        diagnosis_box = QtWidgets.QGroupBox("Diagnosis")
        dl = QtWidgets.QVBoxLayout(diagnosis_box)
        dl.setSpacing(10)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(16)
        top_pred_label = QtWidgets.QLabel("Top prediction")
        top_pred_label.setObjectName("Subtle")
        self.top_pred_value = QtWidgets.QLabel("—")
        self.top_pred_value.setObjectName("BigValue")
        conf_label = QtWidgets.QLabel("Confidence")
        conf_label.setObjectName("Subtle")
        self.conf_value = QtWidgets.QLabel("—")
        self.conf_value.setObjectName("BigValue")
        col1 = QtWidgets.QVBoxLayout()
        col1.addWidget(top_pred_label)
        col1.addWidget(self.top_pred_value)
        col2 = QtWidgets.QVBoxLayout()
        col2.addWidget(conf_label)
        col2.addWidget(self.conf_value)
        top_row.addLayout(col1, 1)
        top_row.addLayout(col2, 1)
        dl.addLayout(top_row)

        self.model_badge = QtWidgets.QLabel("Model: —")
        self.model_badge.setObjectName("Subtle")
        dl.addWidget(self.model_badge)

        self.top5_table = QtWidgets.QTableWidget(5, 2)
        self.top5_table.setHorizontalHeaderLabels(["Label", "Confidence"])
        self.top5_table.verticalHeader().setVisible(False)
        self.top5_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.top5_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.top5_table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.top5_table.horizontalHeader().setStretchLastSection(True)
        self.top5_table.setShowGrid(False)
        self.top5_table.setAlternatingRowColors(True)
        self.top5_table.setFixedHeight(220)
        dl.addWidget(self.top5_table)

        previews_box = QtWidgets.QGroupBox("Previews")
        pl = QtWidgets.QVBoxLayout(previews_box)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        self.preview224 = QtWidgets.QLabel()
        self.preview224.setMinimumSize(224, 224)
        self.preview224.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview224.setStyleSheet("QLabel { background: #0F1115; border: 1px dashed #2B2F36; border-radius: 10px; }")
        row.addWidget(self.preview224, 1)

        qd_col = QtWidgets.QVBoxLayout()
        qd_title = QtWidgets.QLabel("QuickDraw input (28×28)")
        qd_title.setObjectName("Subtle")
        qd_col.addWidget(qd_title)
        self.preview28 = QtWidgets.QLabel()
        self.preview28.setMinimumSize(120, 120)
        self.preview28.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview28.setStyleSheet("QLabel { background: #0F1115; border: 1px solid #2B2F36; border-radius: 10px; }")
        qd_col.addWidget(self.preview28)
        row.addLayout(qd_col)
        pl.addLayout(row)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setObjectName("Subtle")
        pl.addWidget(self.status_label)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        self.log_box.setStyleSheet("QPlainTextEdit { background: #0F1115; border: 1px solid #2B2F36; border-radius: 10px; padding: 8px; }")
        pl.addWidget(self.log_box)

        right.addWidget(diagnosis_box)
        right.addWidget(previews_box)
        right.addStretch(1)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

    def feed_png_bytes(self, png_bytes: bytes) -> None:
        if not self.canvas.set_png_bytes(png_bytes):
            raise ValueError("Invalid image data (expected a PNG or other supported encoded image format).")

    @QtCore.Slot()
    def open_image_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;All files (*)",
        )
        if not path:
            return

        try:
            with open(path, "rb") as f:
                data = f.read()
            self.feed_png_bytes(data)
        except Exception as exc:
            self.status_label.setText("Open failed")
            self.log_box.appendPlainText(f"Open image failed: {exc}")
            QtWidgets.QMessageBox.warning(self, "Open image", f"Could not open image:\n{exc}")

    @QtCore.Slot()
    def auto_predict(self) -> None:
        qimg = self.canvas.grab_canvas_image()

        # UI previews (cheap, in-GUI)
        pil_img = Image.open(io.BytesIO(qimage_to_png_bytes(qimg))).convert("RGB")
        p224 = preprocess_canvas_pil(pil_img)
        bmp28 = preprocess_canvas_quickdraw_bitmap(pil_img)
        self._set_preview(self.preview224, p224, 224, smooth=True)
        self._set_preview(self.preview28, bmp28, 120, smooth=False)

        if self._prediction_in_progress:
            self._pending_prediction = True
            self.status_label.setText("Queued…")
            return

        self._pending_prediction = False
        self._prediction_in_progress = True
        self.status_label.setText("Analyzing…")
        self.top_pred_value.setText("Analyzing…")
        self.conf_value.setText("—")
        self._clear_table()

        self._last_request_id = self._predictor.predict_png_bytes(qimage_to_png_bytes(qimg))

    def _set_preview(self, label: QtWidgets.QLabel, img: Image.Image, size: int, smooth: bool) -> None:
        pix = pil_to_qpixmap(img)
        label.setPixmap(
            pix.scaled(
                size,
                size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation if smooth else QtCore.Qt.TransformationMode.FastTransformation,
            )
        )

    def _clear_table(self) -> None:
        for r in range(self.top5_table.rowCount()):
            for c in range(self.top5_table.columnCount()):
                self.top5_table.setItem(r, c, QtWidgets.QTableWidgetItem("—"))

    def _append_worker_log(self, text: str) -> None:
        try:
            self.log_box.appendPlainText(text)
        except RuntimeError:
            # Widget already deleted during shutdown.
            pass

    def _on_worker_result(self, req_id: int, payload: dict) -> None:
        # During shutdown, callbacks may be detached.
        if not hasattr(self, "top_pred_value"):
            return
        # Ignore stale responses if user drew again quickly.
        if self._last_request_id is not None and req_id != self._last_request_id:
            return

        self._prediction_in_progress = False

        if not payload.get("ok"):
            self.status_label.setText("Worker error (restarting)…")
            err = payload.get("error", "unknown error")
            tb = payload.get("traceback", "")
            self.log_box.appendPlainText(err)
            if tb:
                self.log_box.appendPlainText(tb)
            # If worker died, restart it.
            if not self._predictor.is_running():
                self._predictor.start()
            if self._pending_prediction:
                self._pending_prediction = False
                self.auto_predict()
            return

        self.status_label.setText("Ready")
        self.model_badge.setText(f"Model: {payload.get('backend','?').upper()} ({payload.get('model_id') or 'n/a'})")

        top = payload["top"]
        self.top_pred_value.setText(top["label"])
        self.conf_value.setText(f"{float(top['confidence']) * 100:.1f}%")

        top5 = payload.get("top5") or []
        for row in range(5):
            if row < len(top5):
                pred = top5[row]
                self.top5_table.setItem(row, 0, QtWidgets.QTableWidgetItem(pred["label"]))
                self.top5_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{float(pred['confidence']) * 100:.1f}%"))
            else:
                self.top5_table.setItem(row, 0, QtWidgets.QTableWidgetItem("—"))
                self.top5_table.setItem(row, 1, QtWidgets.QTableWidgetItem("—"))

        if self._pending_prediction:
            self._pending_prediction = False
            self.auto_predict()


def main() -> None:
    # Minimal logging: show errors but keep console readable.
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "SketchSense_gui.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )

    def _excepthook(exc_type, exc, tb):
        logging.error("Uncaught exception:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = _excepthook

    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    globals()["_SKETCHSENSE_WINDOW"] = win
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
