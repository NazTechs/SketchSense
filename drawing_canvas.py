from PySide6 import QtCore, QtGui, QtWidgets


class DrawingCanvas(QtWidgets.QWidget):
    """
    A simple "paint-like" canvas implemented as a QWidget.

    Drawing system notes:
    - We keep an internal QImage (self._image) as the persistent drawing surface.
    - Mouse events draw directly onto that QImage using QPainter + a round pen for
      smooth freehand lines.
    - paintEvent() simply blits the QImage onto the widget.

    Signal/slot flow:
    - On mouse release we emit drawingFinished(), which the main window connects
      to auto prediction.
    """

    drawingFinished = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StaticContents, True)

        self._pen_color = QtGui.QColor(20, 20, 20)  # near-black for nicer look
        self._pen_width = 18

        self._drawing = False
        self._last_pos = QtCore.QPoint()

        # Start with a reasonable size; it will be expanded on resizeEvent().
        self._image = QtGui.QImage(
            900, 600, QtGui.QImage.Format.Format_ARGB32_Premultiplied
        )
        self._image.fill(QtCore.Qt.GlobalColor.white)

        self.setMinimumSize(520, 360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    def set_brush_size(self, size: int) -> None:
        self._pen_width = max(1, int(size))

    def clear(self) -> None:
        self._image.fill(QtCore.Qt.GlobalColor.white)
        self.update()
        # Clearing is conceptually a "finished" action, so the rest of the app
        # can react (e.g., re-run prediction).
        self.drawingFinished.emit()

    def grab_canvas_image(self) -> QtGui.QImage:
        """
        Return a copy of the current canvas QImage.
        This is used by the main window to capture the drawing for preprocessing.
        """
        return self._image.copy()

    def set_png_bytes(self, png_bytes: bytes) -> bool:
        qimg = QtGui.QImage.fromData(png_bytes, "PNG")
        if qimg.isNull():
            qimg = QtGui.QImage.fromData(png_bytes)
        if qimg.isNull():
            return False
        self.set_qimage(qimg)
        return True

    def set_qimage(self, qimage: QtGui.QImage) -> None:
        if qimage.isNull():
            return
        self._ensure_backing_store()

        if qimage.format() != QtGui.QImage.Format.Format_ARGB32_Premultiplied:
            qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_ARGB32_Premultiplied)

        target_size = self._image.size()
        scaled = qimage.scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        new_image = QtGui.QImage(target_size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        new_image.fill(QtCore.Qt.GlobalColor.white)
        painter = QtGui.QPainter(new_image)
        x = int((target_size.width() - scaled.width()) / 2)
        y = int((target_size.height() - scaled.height()) / 2)
        painter.drawImage(QtCore.QPoint(x, y), scaled)
        painter.end()

        self._image = new_image
        self.update()
        self.drawingFinished.emit()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        # Draw only the region requested by Qt for efficiency.
        dirty = event.rect()
        painter.drawImage(dirty, self._image, dirty)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drawing = True
            self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (event.buttons() & QtCore.Qt.MouseButton.LeftButton) and self._drawing:
            self._draw_line_to(event.position().toPoint())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._drawing:
            self._draw_line_to(event.position().toPoint())
            self._drawing = False
            # IMPORTANT: this is the "trigger point" for real-time prediction.
            self.drawingFinished.emit()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """
        Ensure our backing QImage always covers the widget.
        This mimics the approach used by Qt's classic "Scribble" example.
        """
        if self.width() > self._image.width() or self.height() > self._image.height():
            new_width = max(self.width(), self._image.width())
            new_height = max(self.height(), self._image.height())

            new_image = QtGui.QImage(
                QtCore.QSize(new_width, new_height),
                QtGui.QImage.Format.Format_ARGB32_Premultiplied,
            )
            new_image.fill(QtCore.Qt.GlobalColor.white)

            painter = QtGui.QPainter(new_image)
            painter.drawImage(QtCore.QPoint(0, 0), self._image)
            painter.end()

            self._image = new_image

        super().resizeEvent(event)

    def _ensure_backing_store(self) -> None:
        if self.width() <= self._image.width() and self.height() <= self._image.height():
            return

        new_width = max(self.width(), self._image.width())
        new_height = max(self.height(), self._image.height())

        new_image = QtGui.QImage(
            QtCore.QSize(new_width, new_height),
            QtGui.QImage.Format.Format_ARGB32_Premultiplied,
        )
        new_image.fill(QtCore.Qt.GlobalColor.white)

        painter = QtGui.QPainter(new_image)
        painter.drawImage(QtCore.QPoint(0, 0), self._image)
        painter.end()

        self._image = new_image

    def _draw_line_to(self, end_pos: QtCore.QPoint) -> None:
        painter = QtGui.QPainter(self._image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        pen = QtGui.QPen(self._pen_color, self._pen_width, QtCore.Qt.PenStyle.SolidLine)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        painter.drawLine(self._last_pos, end_pos)
        painter.end()

        # Update only the changed area (+ some padding for pen width).
        rad = self._pen_width // 2 + 2
        self.update(QtCore.QRect(self._last_pos, end_pos).normalized().adjusted(-rad, -rad, rad, rad))
        self._last_pos = end_pos
