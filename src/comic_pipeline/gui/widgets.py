from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from comic_pipeline.gui.state import STATUS_LABELS


_STATUS_COLORS = {
    "pending": "#6b7280",
    "mask_ready": "#0f766e",
    "nano_pending": "#2563eb",
    "restored": "#7c3aed",
    "done": "#15803d",
    "check": "#b91c1c",
}


class StatusBadge(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        font = QFont(self.font())
        font.setBold(True)
        self.setFont(font)
        self.set_status("pending")

    def set_status(self, status: str) -> None:
        label = STATUS_LABELS.get(status, status)
        color = _STATUS_COLORS.get(status, "#374151")
        self.setText(label)
        self.setStyleSheet(
            f"""
            QLabel {{
                color: white;
                background-color: {color};
                border: 1px solid rgba(0, 0, 0, 0.25);
                border-radius: 4px;
                padding: 2px 8px;
            }}
            """
        )


class PreviewCanvas(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self._placeholder = "아직 생성되지 않음"
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self._source_pixmap = None
        self.update()

    def set_image_path(self, path: Path | None) -> None:
        if path is None or not path.exists():
            self._source_pixmap = None
            self.update()
            return
        pixmap = QPixmap(str(path))
        self._source_pixmap = pixmap if not pixmap.isNull() else None
        self.update()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if self._source_pixmap is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#f3f4f6"))
            pen = QPen(QColor("#9ca3af"))
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.rect().adjusted(8, 8, -8, -8))
            painter.setPen(QColor("#4b5563"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._placeholder,
            )
            painter.end()
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#e5e7eb"))
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()


class PreviewWidget(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PreviewCanvas(self)
        self.canvas.set_placeholder(f"{title}\n아직 생성되지 않음")
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll)

    def set_image_path(self, path: Path | None) -> None:
        if path is None:
            self.canvas.set_placeholder(f"{self.title}\n아직 생성되지 않음")
        else:
            self.canvas.set_image_path(path)
