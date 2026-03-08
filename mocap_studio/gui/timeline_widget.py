"""
Timeline Widget — Custom QPainter timeline with multi-track bars and scrubber.

Shows horizontal bars for each loaded track (color-coded), a vertical
scrubber line, and frame number display.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QRect, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QMouseEvent


# Same colors as the GL viewer
TRACK_COLORS_QT = [
    QColor(64, 224, 108),   # green
    QColor(77, 140, 242),   # blue
    QColor(242, 217, 64),   # yellow
    QColor(242, 89, 89),    # red
    QColor(184, 102, 230),  # purple
]


class TimelineWidget(QWidget):
    """Custom-painted multi-track timeline with scrubber."""

    frame_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMaximumHeight(180)

        self._current_frame: int = 0
        self._max_frame: int = 100

        # Per-track: (frame_count, offset, trim_in, trim_out, visible)
        self._tracks: List[Optional[dict]] = [None] * 5

        self._dragging = False

        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_max_frame(self, mf: int):
        self._max_frame = max(1, mf)
        self.update()

    def set_current_frame(self, frame: int):
        self._current_frame = frame
        self.update()

    def set_track_info(self, slot: int, frame_count: int, offset: int,
                       trim_in: int, trim_out: int, visible: bool):
        self._tracks[slot] = {
            "frame_count": frame_count,
            "offset": offset,
            "trim_in": trim_in,
            "trim_out": trim_out,
            "visible": visible,
        }
        self.update()

    def clear_track(self, slot: int):
        self._tracks[slot] = None
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(30, 30, 34))

        # Track bars
        track_height = 14
        track_gap = 4
        margin_top = 10
        margin_left = 40
        margin_right = 10
        bar_width = w - margin_left - margin_right

        for slot in range(5):
            y = margin_top + slot * (track_height + track_gap)
            td = self._tracks[slot]

            # Track label
            painter.setPen(QColor(140, 140, 150))
            font = QFont("Segoe UI", 8)
            painter.setFont(font)
            painter.drawText(2, y + track_height - 2, f"T{slot+1}")

            if td is None:
                # Empty slot — show dim background
                painter.fillRect(margin_left, y, bar_width, track_height,
                                 QColor(45, 45, 50))
                continue

            # Full track range background
            painter.fillRect(margin_left, y, bar_width, track_height,
                             QColor(40, 40, 48))

            if self._max_frame <= 0:
                continue

            scale = bar_width / self._max_frame
            offset = td["offset"]
            fc = td["frame_count"]
            ti = td["trim_in"]
            to = td["trim_out"]

            # Full range (dim)
            x0 = margin_left + offset * scale
            w0 = fc * scale
            dim_color = QColor(TRACK_COLORS_QT[slot])  # copy to avoid mutating shared object
            dim_color.setAlpha(60)
            painter.fillRect(int(x0), y, int(w0), track_height, dim_color)

            # Trimmed active range (bright)
            xa = margin_left + (offset + ti) * scale
            wa = (to - ti) * scale
            bright_color = QColor(TRACK_COLORS_QT[slot])  # copy to avoid mutating shared object
            if not td["visible"]:
                bright_color.setAlpha(100)
            else:
                bright_color.setAlpha(220)
            painter.fillRect(int(xa), y, max(int(wa), 1), track_height, bright_color)

        # Scrubber line
        scrubber_y_start = margin_top - 4
        scrubber_y_end = margin_top + 5 * (track_height + track_gap) + 4
        if self._max_frame > 0:
            scrub_x = margin_left + (self._current_frame / self._max_frame) * bar_width
            pen = QPen(QColor(255, 255, 255, 200), 2)
            painter.setPen(pen)
            painter.drawLine(int(scrub_x), scrubber_y_start,
                             int(scrub_x), scrubber_y_end)

            # Frame indicator triangle
            painter.setBrush(QColor(255, 255, 255, 200))
            painter.setPen(Qt.NoPen)
            from PySide6.QtGui import QPolygonF
            from PySide6.QtCore import QPointF
            tri_size = 5
            triangle = QPolygonF([
                QPointF(scrub_x - tri_size, scrubber_y_start),
                QPointF(scrub_x + tri_size, scrubber_y_start),
                QPointF(scrub_x, scrubber_y_start + tri_size),
            ])
            painter.drawPolygon(triangle)

        # Frame number
        painter.setPen(QColor(200, 200, 210))
        font = QFont("Consolas", 9)
        painter.setFont(font)
        frame_text = f"Frame: {self._current_frame} / {self._max_frame}"
        painter.drawText(margin_left, h - 6, frame_text)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._update_frame_from_mouse(event)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            self._update_frame_from_mouse(event)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False
        event.accept()

    def _update_frame_from_mouse(self, event: QMouseEvent):
        margin_left = 40
        margin_right = 10
        bar_width = self.width() - margin_left - margin_right
        if bar_width <= 0:
            return

        x = event.position().x() - margin_left
        frac = max(0.0, min(1.0, x / bar_width))
        frame = int(frac * self._max_frame)
        self._current_frame = frame
        self.frame_changed.emit(frame)
        self.update()
