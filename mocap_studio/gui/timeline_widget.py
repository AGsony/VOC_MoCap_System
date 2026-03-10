"""
Timeline Widget — Custom QPainter timeline with multi-track bars, scrubber,
and full mouse interaction (scrub, drag offset, trim in/out, pan, zoom).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import math

from PySide6.QtWidgets import QWidget, QScrollBar
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QMouseEvent, QWheelEvent, QCursor


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
    track_offset_changed = Signal(int, float)  # slot, new_offset
    track_trim_changed = Signal(int, int, int) # slot, trim_in, trim_out

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMaximumHeight(200)

        self._current_frame: int = 0
        self._max_frame: int = 100

        # Per-track: (frame_count, offset, track_scale, trim_in, trim_out, visible)
        self._tracks: List[Optional[dict]] = [None] * 5

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # View parameters
        self._zoom: float = 1.0      # pixels per frame
        self._pan_x: float = 0.0     # frame offset at left edge

        # Interaction state
        self._drag_mode: str = ""    # "scrub", "pan", "offset", "trim_in", "trim_out"
        self._drag_slot: int = -1
        self._last_mouse_x: int = 0
        self._drag_start_val: float = 0.0

        # Layout constants
        self.TRACK_HEIGHT = 24
        self.TRACK_GAP = 6
        self.MARGIN_TOP = 24
        self.MARGIN_LEFT = 40
        self.MARGIN_RIGHT = 10
        self.HANDLE_WIDTH = 12

        self.h_scrollbar = QScrollBar(Qt.Horizontal, self)
        self.h_scrollbar.valueChanged.connect(self._on_scrollbar_scrolled)
        self.h_scrollbar.setCursor(Qt.ArrowCursor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _on_scrollbar_scrolled(self, value):
        self._pan_x = value
        self.update()

    def _update_scrollbar_range(self):
        w = self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT
        if w <= 0 or self._zoom <= 0:
            return
            
        visible_frames = w / self._zoom
        max_pan = max(0.0, self._max_frame - visible_frames)
        
        # Scrollbar works with integers
        self.h_scrollbar.setRange(0, int(math.ceil(max_pan)))
        self.h_scrollbar.setPageStep(int(visible_frames))
        
        # Keep pan_x within bounds
        self._pan_x = max(0.0, min(self._pan_x, max_pan))
        
        self.h_scrollbar.blockSignals(True)
        self.h_scrollbar.setValue(int(self._pan_x))
        self.h_scrollbar.blockSignals(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        self.h_scrollbar.setGeometry(self.MARGIN_LEFT, h - 14, w - self.MARGIN_LEFT, 14)
        self._update_scrollbar_range()

    def set_max_frame(self, mf: int):
        self._max_frame = max(1, mf)
        # Auto-fit zoom on first load if we haven't manipulated it
        w = self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT
        if w > 0:
            self._zoom = w / self._max_frame
        self._update_scrollbar_range()
        self.update()

    def set_current_frame(self, frame: int):
        self._current_frame = frame
        w = self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT
        if w > 0 and self._zoom > 0:
            visible_frames = w / self._zoom
            if frame < self._pan_x or frame > self._pan_x + visible_frames:
                self._pan_x = max(0.0, frame - visible_frames / 2.0)
                self._update_scrollbar_range()
        self.update()

    def set_track_info(self, slot: int, frame_count: int, offset: float, scale: float,
                       trim_in: int, trim_out: int, visible: bool):
        self._tracks[slot] = {
            "frame_count": frame_count,
            "offset": offset,
            "scale": scale,
            "trim_in": trim_in,
            "trim_out": trim_out,
            "visible": visible,
        }
        self.update()

    def clear_track(self, slot: int):
        self._tracks[slot] = None
        self.update()

    # ------------------------------------------------------------------
    # Coordinate Math
    # ------------------------------------------------------------------
    def _frame_to_x(self, frame: float) -> float:
        """Convert a frame number to a widget X pixel coordinate."""
        return self.MARGIN_LEFT + (frame - self._pan_x) * self._zoom

    def _x_to_frame(self, x: float) -> float:
        """Convert a widget X pixel coordinate to a frame number."""
        return ((x - self.MARGIN_LEFT) / self._zoom) + self._pan_x

    def _get_track_rects(self, slot: int) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return (y, dim_x, bright_x, bright_w) for a track slot."""
        y = self.MARGIN_TOP + slot * (self.TRACK_HEIGHT + self.TRACK_GAP)
        td = self._tracks[slot]
        if not td:
            return y, None, None, None

        offset = td["offset"]
        fc = td["frame_count"]
        ti = td["trim_in"]
        to = td["trim_out"]

        dim_x = self._frame_to_x(offset)
        bright_x = self._frame_to_x(offset + ti)
        bright_w = (to - ti) * self._zoom

        return y, dim_x, bright_x, bright_w

    def _hit_test(self, x: int, y: int) -> Tuple[str, int]:
        """Return (mode, slot) under the given mouse coordinates."""
        # Check if inside a track's Y bounds
        for slot in range(5):
            ty, dim_x, bright_x, bright_w = self._get_track_rects(slot)
            if dim_x is None: continue
            
            if ty <= y <= ty + self.TRACK_HEIGHT:
                # Within this track's Y. Now check X.
                if bright_x is not None and bright_w is not None:
                    # Check handles (left, right)
                    if bright_x - self.HANDLE_WIDTH <= x <= bright_x + self.HANDLE_WIDTH:
                        return "trim_in", slot
                    if bright_x + bright_w - self.HANDLE_WIDTH <= x <= bright_x + bright_w + self.HANDLE_WIDTH:
                        return "trim_out", slot
                    # Check body
                    if bright_x < x < bright_x + bright_w:
                        return "offset", slot

        # If not on a specific track part, click on timeline area scrubs
        if x >= self.MARGIN_LEFT:
            return "scrub", -1

        return "", -1

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Background timeline track area
        painter.fillRect(self.MARGIN_LEFT, 0, w - self.MARGIN_LEFT, h, QColor(30, 30, 34))
        
        # Title/Track Name break section
        painter.fillRect(0, 0, self.MARGIN_LEFT, h, QColor(22, 22, 26))
        painter.setPen(QColor(100, 100, 110))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(4, 16, "Tracks")
        
        # Vertical divider line
        painter.setPen(QColor(50, 50, 60))
        painter.drawLine(self.MARGIN_LEFT, 0, self.MARGIN_LEFT, h)

        # Ruler
        painter.setPen(QColor(140, 140, 150))
        font = QFont("Consolas", 8)
        painter.setFont(font)
        
        min_pixels_between_labels = 45.0
        min_frames = min_pixels_between_labels / max(0.0001, self._zoom)
        
        magnitude = 10 ** math.floor(math.log10(max(1, min_frames))) if min_frames > 0 else 1
        val = min_frames / magnitude if magnitude > 0 else 1
        
        if val <= 1:
            label_step = 1 * magnitude
        elif val <= 2:
            label_step = 2 * magnitude
        elif val <= 5:
            label_step = 5 * magnitude
        else:
            label_step = 10 * magnitude
            
        label_step = max(1, int(label_step))

        start_f = max(0, int(self._x_to_frame(self.MARGIN_LEFT)))
        end_f = int(self._x_to_frame(w)) + 2

        for f in range(start_f, end_f):
            x = self._frame_to_x(f)
            if self.MARGIN_LEFT <= x <= w:
                painter.drawLine(int(x), self.MARGIN_TOP - 4, int(x), self.MARGIN_TOP - 1)
                if f % label_step == 0:
                    painter.drawLine(int(x), self.MARGIN_TOP - 12, int(x), self.MARGIN_TOP - 1)
                    painter.drawText(int(x) + 2, self.MARGIN_TOP - 4, str(f))

        # Track bars
        for slot in range(5):
            y, dim_x, bright_x, bright_w = self._get_track_rects(slot)
            
            # Track label
            painter.setPen(QColor(140, 140, 150))
            font = QFont("Segoe UI", 8)
            painter.setFont(font)
            painter.drawText(2, int(y + self.TRACK_HEIGHT - 2), f"T{slot+1}")

            td = self._tracks[slot]
            if td is None or dim_x is None:
                # Empty slot
                painter.fillRect(self.MARGIN_LEFT, int(y), w - self.MARGIN_LEFT - self.MARGIN_RIGHT, self.TRACK_HEIGHT, QColor(45, 45, 50))
                continue

            # Full line background
            painter.fillRect(self.MARGIN_LEFT, int(y), w - self.MARGIN_LEFT - self.MARGIN_RIGHT, self.TRACK_HEIGHT, QColor(40, 40, 48))

            if self._max_frame <= 0:
                continue

            # Dim range
            fc = td["frame_count"]
            dim_w = fc * self._zoom
            dim_color = QColor(TRACK_COLORS_QT[slot])
            dim_color.setAlpha(60)
            painter.fillRect(int(dim_x), int(y), max(int(dim_w), 1), self.TRACK_HEIGHT, dim_color)

            # Bright range
            bright_color = QColor(TRACK_COLORS_QT[slot])
            if not td["visible"]:
                bright_color.setAlpha(100)
            else:
                bright_color.setAlpha(220)
            
            bx = int(bright_x)
            bw = max(int(bright_w), 1)
            painter.fillRect(bx, int(y), bw, self.TRACK_HEIGHT, bright_color)
            
            # Handles
            painter.fillRect(bx, int(y), 2, self.TRACK_HEIGHT, QColor(255, 255, 255, 150))
            painter.fillRect(bx + bw - 2, int(y), 2, self.TRACK_HEIGHT, QColor(255, 255, 255, 150))
            
            # Interpolation indicators (Red Bars)
            # Determine if track requires sub-frame interpolation
            # Determine if track requires sub-frame interpolation
            if abs(td["scale"] - 1.0) > 1e-4 or abs(td["offset"] - round(td["offset"])) > 1e-4:
                g_start = int(math.ceil(td["offset"] + td["trim_in"] * td["scale"]))
                g_end = int(math.floor(td["offset"] + td["trim_out"] * td["scale"]))
                
                bar_thick = max(1, int(self._zoom * 0.2))
                
                for g_f in range(g_start, min(g_end + 1, self._max_frame)):
                    local_f = (g_f - td["offset"]) / td["scale"]
                    if abs(local_f - round(local_f)) > 1e-4:
                        ix = self._frame_to_x(g_f)
                        if self.MARGIN_LEFT <= ix <= w:
                            painter.fillRect(int(ix) - bar_thick // 2, int(y), bar_thick, self.TRACK_HEIGHT, QColor(255, 50, 50, 255))

        # Scrubber line
        scrubber_y_start = self.MARGIN_TOP - 4
        scrubber_y_end = self.MARGIN_TOP + 5 * (self.TRACK_HEIGHT + self.TRACK_GAP) + 4
        
        scrub_x = self._frame_to_x(self._current_frame)
        if self.MARGIN_LEFT - 1 <= scrub_x <= w + 1:
            pen = QPen(QColor(255, 255, 255, 200), 2)
            painter.setPen(pen)
            painter.drawLine(int(scrub_x), int(scrubber_y_start), int(scrub_x), int(scrubber_y_end))

            # Triangle
            painter.setBrush(QColor(255, 255, 255, 200))
            painter.setPen(Qt.NoPen)
            from PySide6.QtGui import QPolygonF
            from PySide6.QtCore import QPointF
            tri_size = 9
            triangle = QPolygonF([
                QPointF(scrub_x - tri_size, scrubber_y_start),
                QPointF(scrub_x + tri_size, scrubber_y_start),
                QPointF(scrub_x, scrubber_y_start + tri_size + 2),
            ])
            painter.drawPolygon(triangle)

        # Frame number
        painter.setPen(QColor(200, 200, 210))
        font = QFont("Consolas", 9)
        painter.setFont(font)
        frame_text = f"Frame: {self._current_frame} / {self._max_frame}"
        painter.drawText(self.MARGIN_LEFT, h - 18, frame_text)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent):
        x, y = event.x(), event.y()
        
        # 1. Did we click the scrubber?
        sx = self._frame_to_x(self._current_frame)
        # Even if empty, allow grabbing
        if abs(x - sx) < 12 and self.MARGIN_TOP - 10 <= y <= self.MARGIN_TOP + 5 * (24 + self.TRACK_GAP) + 20: # TRACK_HEIGHT changed from 18 to 24
            self._drag_mode = "scrub"
            return

        # 2. Check track interaction
        if event.button() == Qt.MiddleButton:
            self._drag_mode = "pan"
            self._last_mouse_x = x
            self.setCursor(Qt.ClosedHandCursor)
            
        elif event.button() == Qt.LeftButton:
            mode, slot = self._hit_test(x, y)
            self._drag_mode = mode
            self._drag_slot = slot
            self._last_mouse_x = x
            
            if mode == "scrub":
                # This case is handled above, but if _hit_test returns scrub for other reasons
                self._update_scrub(x)
            elif mode == "offset" and slot >= 0:
                self._drag_start_val = self._tracks[slot]["offset"]
                self.setCursor(Qt.SizeAllCursor)
            elif mode == "trim_in" and slot >= 0:
                self._drag_start_val = self._tracks[slot]["trim_in"]
                self.setCursor(Qt.SizeHorCursor)
            elif mode == "trim_out" and slot >= 0:
                self._drag_start_val = self._tracks[slot]["trim_out"]
                self.setCursor(Qt.SizeHorCursor)

        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        x, y = event.x(), event.y()
        
        # Hover cursors if not dragging
        if not self._drag_mode:
            mode, _ = self._hit_test(x, y)
            if mode in ("trim_in", "trim_out"):
                self.setCursor(Qt.SizeHorCursor)
            elif mode == "offset":
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        dx = x - self._last_mouse_x
        d_frame = dx / self._zoom

        if self._drag_mode == "pan":
            self._pan_x -= d_frame
            self._update_scrollbar_range()
            self._last_mouse_x = x
            self.update()
            
        elif self._drag_mode == "scrub":
            frame = int(round(self._x_to_frame(x)))
            # Do NOT clamp to max_frame here so the user can drag arbitrarily far
            frame = max(0, frame)
            if frame != self._current_frame:
                self._current_frame = frame
                self.frame_changed.emit(frame)
                self.update()
            
        elif self._drag_slot >= 0:
            td = self._tracks[self._drag_slot]
            if not td: return
            
            if self._drag_mode == "offset":
                new_offset = self._drag_start_val + d_frame
                # Snap to integer if close
                if abs(new_offset - round(new_offset)) < 0.2:
                    new_offset = round(new_offset)
                td["offset"] = new_offset
                self.track_offset_changed.emit(self._drag_slot, new_offset)
                self.update()
                
            elif self._drag_mode == "trim_in":
                # dx moves trim_in
                new_ti = int(round(self._drag_start_val + d_frame))
                new_ti = max(0, min(new_ti, td["trim_out"] - 1))
                td["trim_in"] = new_ti
                self.track_trim_changed.emit(self._drag_slot, new_ti, td["trim_out"])
                self.update()
                
            elif self._drag_mode == "trim_out":
                new_to = int(round(self._drag_start_val + d_frame))
                new_to = max(td["trim_in"] + 1, min(new_to, td["frame_count"] - 1))
                td["trim_out"] = new_to
                self.track_trim_changed.emit(self._drag_slot, td["trim_in"], new_to)
                self.update()

        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_mode = ""
        self._drag_slot = -1
        self.setCursor(Qt.ArrowCursor)
        self.update()
        event.accept()

    def _update_scrub(self, x: float):
        frame = int(round(self._x_to_frame(x)))
        frame = max(0, frame)
        if frame != self._current_frame:
            self._current_frame = frame
            self.frame_changed.emit(frame)
            self.update()

    def wheelEvent(self, event: QWheelEvent):
        # Zoom in/out based on mouse position
        delta = event.angleDelta().y()
        if delta == 0: return
        
        factor = 1.1 if delta > 0 else 0.9
        
        # Keep mouse X stationary during zoom
        mouse_x = event.position().toPoint().x()
        if mouse_x < self.MARGIN_LEFT: mouse_x = self.MARGIN_LEFT
        
        frame_at_mouse = self._x_to_frame(mouse_x)
        self._zoom *= factor
        # Max zoom: 50 pixels per frame. Min zoom: fit 100000 frames in window
        self._zoom = max(0.005, min(50.0, self._zoom))
        
        # Adjust pan so the same frame is under the mouse
        self._pan_x = frame_at_mouse - ((mouse_x - self.MARGIN_LEFT) / self._zoom)
        self._update_scrollbar_range()
        self.update()
        event.accept()

    def _update_scrub(self, x: int):
        frame = int(round(self._x_to_frame(x)))
        frame = max(0, min(frame, self._max_frame))
        if frame != self._current_frame:
            self._current_frame = frame
            self.frame_changed.emit(frame)
            self.update()
