"""
OpenGL 3D Skeleton Viewer — QOpenGLWidget subclass.

Renders multiple skeleton tracks as colored lines with an arcball camera,
grid floor, and joint points. Designed for 60fps with up to 5 skeletons.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPoint, QTimer, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent

from OpenGL.GL import *  # noqa
from OpenGL.GLU import *  # noqa

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except ImportError:
    from PySide6.QtWidgets import QOpenGLWidget  # type: ignore


# Track colors: green, blue, yellow, red, purple
TRACK_COLORS = [
    (0.25, 0.88, 0.42),   # green
    (0.30, 0.55, 0.95),   # blue
    (0.95, 0.85, 0.25),   # yellow
    (0.95, 0.35, 0.35),   # red
    (0.72, 0.40, 0.90),   # purple
]

# Dimmer versions for joint points
TRACK_POINT_COLORS = [
    (0.15, 0.65, 0.30),
    (0.20, 0.40, 0.75),
    (0.75, 0.65, 0.18),
    (0.75, 0.25, 0.25),
    (0.55, 0.28, 0.70),
]


class GLViewer(QOpenGLWidget):
    """OpenGL skeleton viewer with arcball camera."""

    frame_requested = Signal(int)
    joint_selected = Signal(int, int, str)  # slot, joint_index, joint_name

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Camera state
        self._cam_yaw = 30.0      # degrees
        self._cam_pitch = 25.0    # degrees
        self._cam_dist = 250.0    # distance from target
        self._cam_target = np.array([0.0, 80.0, 0.0])  # look-at point
        self._cam_pan_offset = np.array([0.0, 0.0, 0.0])

        # Mouse interaction
        self._last_mouse: Optional[QPoint] = None
        self._mouse_button: int = 0

        # Data
        self._tracks_data: List[Optional[dict]] = [None] * 5
        # Each entry: {"positions": (F,J,3), "bone_pairs": [(p,c),...],
        #              "visible": bool, "hidden_joints": set, "translate": (x,y,z)}

        self._current_frame: int = 0

        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)

        # Drag state
        self._last_mouse_pos = QPoint()
        self._is_dragging = False
        self._drag_threshold = 4  # pixels — click vs drag
        
        self.setFocusPolicy(Qt.StrongFocus)

        # Selection state
        self._selected_slot: int = -1
        self._selected_joint: int = -1
        # The original _drag_threshold was here, but it's now moved to "Drag state"
        # self._drag_threshold = 4  # pixels — click vs drag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_track_data(self, slot: int, positions: Optional[np.ndarray],
                       bone_pairs: Optional[list], visible: bool = True,
                       hidden_joints: Optional[set] = None,
                       translate: tuple = (0.0, 0.0, 0.0),
                       rotate: tuple = (0.0, 0.0, 0.0),
                       joint_names: Optional[list] = None):
        """
        Update data for a track slot.
        positions: (F, J, 3) aligned positions
        bone_pairs: list of (parent_idx, child_idx)
        hidden_joints: set of joint indices to skip
        translate: (tx, ty, tz) world-space offset
        rotate: (rx, ry, rz) degrees offset
        joint_names: list of joint names for picking
        """
        if positions is not None and bone_pairs is not None:
            self._tracks_data[slot] = {
                "positions": positions,
                "bone_pairs": bone_pairs,
                "visible": visible,
                "hidden_joints": hidden_joints or set(),
                "translate": translate,
                "rotate": rotate,
                "joint_names": joint_names or [],
            }
        else:
            self._tracks_data[slot] = None
        self.update()

    def set_track_visibility(self, slot: int, visible: bool):
        if self._tracks_data[slot]:
            self._tracks_data[slot]["visible"] = visible
            self.update()

    def set_current_frame(self, frame: int):
        self._current_frame = frame
        self.update()

    def reset_camera(self):
        self._cam_yaw = 30.0
        self._cam_pitch = 25.0
        self._cam_dist = 250.0
        self._cam_target = np.array([0.0, 80.0, 0.0])
        self._cam_pan_offset = np.array([0.0, 0.0, 0.0])
        self.update()

    # ------------------------------------------------------------------
    # OpenGL setup
    # ------------------------------------------------------------------
    def initializeGL(self):
        glClearColor(0.12, 0.12, 0.14, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glPointSize(5.0)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        self._update_projection(w, h)

    def _update_projection(self, w: int, h: int):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / max(h, 1)
        gluPerspective(45.0, aspect, 1.0, 5000.0)
        glMatrixMode(GL_MODELVIEW)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Camera
        target = self._cam_target + self._cam_pan_offset
        yaw_r = math.radians(self._cam_yaw)
        pitch_r = math.radians(self._cam_pitch)

        eye_x = target[0] + self._cam_dist * math.cos(pitch_r) * math.sin(yaw_r)
        eye_y = target[1] + self._cam_dist * math.sin(pitch_r)
        eye_z = target[2] + self._cam_dist * math.cos(pitch_r) * math.cos(yaw_r)

        gluLookAt(eye_x, eye_y, eye_z,
                  target[0], target[1], target[2],
                  0, 1, 0)

        self._draw_grid()
        self._draw_axes()

        for slot in range(5):
            td = self._tracks_data[slot]
            if td is None or not td["visible"]:
                continue
            self._draw_skeleton(td, slot)

    def _draw_grid(self):
        """Draw a floor grid on the XZ plane."""
        glBegin(GL_LINES)
        grid_size = 500
        step = 25
        glColor4f(0.28, 0.28, 0.30, 0.6)
        for i in range(-grid_size, grid_size + step, step):
            # Lines along X
            glVertex3f(-grid_size, 0, i)
            glVertex3f(grid_size, 0, i)
            # Lines along Z
            glVertex3f(i, 0, -grid_size)
            glVertex3f(i, 0, grid_size)
        glEnd()

    def _draw_axes(self):
        """Draw small RGB axes at origin."""
        glLineWidth(3.0)
        glBegin(GL_LINES)
        axis_len = 30
        # X — red
        glColor3f(0.9, 0.2, 0.2)
        glVertex3f(0, 0.1, 0)
        glVertex3f(axis_len, 0.1, 0)
        # Y — green
        glColor3f(0.2, 0.9, 0.2)
        glVertex3f(0, 0.1, 0)
        glVertex3f(0, axis_len + 0.1, 0)
        # Z — blue
        glColor3f(0.2, 0.2, 0.9)
        glVertex3f(0, 0.1, 0)
        glVertex3f(0, 0.1, axis_len)
        glEnd()
        glLineWidth(2.0)

    def _draw_skeleton(self, track_data: dict, slot: int):
        """Draw one skeleton as lines and joint points."""
        positions = track_data["positions"]
        bone_pairs = track_data["bone_pairs"]
        hidden = track_data.get("hidden_joints", set())
        translate = track_data.get("translate", (0.0, 0.0, 0.0))
        rotate = track_data.get("rotate", (0.0, 0.0, 0.0))
        frame = self._current_frame

        if frame >= len(positions):
            frame = len(positions) - 1
        if frame < 0:
            return

        joint_pos = positions[frame]  # (J, 3)

        # Apply 3D position and rotation offset
        glPushMatrix()
        if translate != (0.0, 0.0, 0.0):
            glTranslatef(translate[0], translate[1], translate[2])
        if rotate != (0.0, 0.0, 0.0):
            glRotatef(rotate[2], 0, 0, 1) # Z
            glRotatef(rotate[1], 0, 1, 0) # Y
            glRotatef(rotate[0], 1, 0, 0) # X

        # Draw bones as lines (skip hidden joints)
        color = TRACK_COLORS[slot % len(TRACK_COLORS)]
        glColor3f(*color)
        glLineWidth(2.5)
        glBegin(GL_LINES)
        for parent_idx, child_idx in bone_pairs:
            if parent_idx in hidden or child_idx in hidden:
                continue
            if parent_idx < len(joint_pos) and child_idx < len(joint_pos):
                p = joint_pos[parent_idx]
                c = joint_pos[child_idx]
                glVertex3f(p[0], p[1], p[2])
                glVertex3f(c[0], c[1], c[2])
        glEnd()

        # Draw joints as points (skip hidden)
        pt_color = TRACK_POINT_COLORS[slot % len(TRACK_POINT_COLORS)]
        glColor3f(*pt_color)
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for j in range(len(joint_pos)):
            if j in hidden:
                continue
            p = joint_pos[j]
            glVertex3f(p[0], p[1], p[2])
        glEnd()

        # Draw selected joint highlight
        if slot == self._selected_slot and self._selected_joint >= 0:
            sel_j = self._selected_joint
            if sel_j < len(joint_pos) and sel_j not in hidden:
                p = joint_pos[sel_j]
                glColor3f(1.0, 1.0, 1.0)
                glPointSize(10.0)
                glBegin(GL_POINTS)
                glVertex3f(p[0], p[1], p[2])
                glEnd()
                # Draw a bright ring
                glColor4f(1.0, 1.0, 0.5, 0.7)
                glPointSize(16.0)
                glBegin(GL_POINTS)
                glVertex3f(p[0], p[1], p[2])
                glEnd()

        glPopMatrix()

    # ------------------------------------------------------------------
    # Mouse interaction — Arcball camera
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse = event.position().toPoint()
        self._press_pos = event.position().toPoint()  # for click detection
        self._mouse_button = event.button()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse is None:
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._last_mouse.x()
        dy = pos.y() - self._last_mouse.y()
        self._last_mouse = pos

        if self._mouse_button == Qt.LeftButton:
            # Orbit
            self._cam_yaw -= dx * 0.4
            self._cam_pitch += dy * 0.3
            self._cam_pitch = max(-89, min(89, self._cam_pitch))

        elif self._mouse_button == Qt.MiddleButton:
            # Pan
            speed = self._cam_dist * 0.002
            yaw_r = math.radians(self._cam_yaw)
            right = np.array([math.cos(yaw_r), 0, -math.sin(yaw_r)])
            up = np.array([0, 1, 0])
            self._cam_pan_offset -= right * dx * speed
            self._cam_pan_offset += up * dy * speed

        elif self._mouse_button == Qt.RightButton:
            # Zoom (drag)
            self._cam_dist += dy * 1.0
            self._cam_dist = max(10, min(3000, self._cam_dist))

        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        release_pos = event.position().toPoint()

        # Detect click (not drag) on left button
        if (self._mouse_button == Qt.LeftButton and
                self._press_pos is not None):
            dx = abs(release_pos.x() - self._press_pos.x())
            dy = abs(release_pos.y() - self._press_pos.y())
            if dx <= self._drag_threshold and dy <= self._drag_threshold:
                self._pick_joint(release_pos.x(), release_pos.y())

        self._last_mouse = None
        self._mouse_button = 0
        self._press_pos = None
        event.accept()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._cam_dist *= factor
        self._cam_dist = max(10, min(3000, self._cam_dist))
        self.update()
        event.accept()

    # ------------------------------------------------------------------
    # Ray picking
    # ------------------------------------------------------------------
    def _pick_joint(self, screen_x: int, screen_y: int):
        """
        Find the nearest joint to the clicked screen position.
        Uses gluProject to project each joint to screen space and
        finds the closest one within a pixel threshold.
        """
        self.makeCurrent()

        # Get current matrices
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        # Flip Y for OpenGL (bottom-up)
        gl_y = viewport[3] - screen_y

        best_dist = 20.0  # max pixel distance to select
        best_slot = -1
        best_joint = -1
        best_name = ""

        frame = self._current_frame

        for slot in range(5):
            td = self._tracks_data[slot]
            if td is None or not td["visible"]:
                continue

            positions = td["positions"]
            hidden = td.get("hidden_joints", set())
            translate = td.get("translate", (0.0, 0.0, 0.0))
            joint_names = td.get("joint_names", [])

            f = min(frame, len(positions) - 1)
            if f < 0:
                continue
            joint_pos = positions[f]

            for j in range(len(joint_pos)):
                if j in hidden:
                    continue

                # World position with translation
                wx = joint_pos[j][0] + translate[0]
                wy = joint_pos[j][1] + translate[1]
                wz = joint_pos[j][2] + translate[2]

                # Project to screen
                try:
                    sx, sy, sz = gluProject(
                        wx, wy, wz, modelview, projection, viewport
                    )
                except Exception:
                    continue

                # Distance in screen pixels
                dist = math.sqrt((sx - screen_x) ** 2 + (sy - gl_y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_slot = slot
                    best_joint = j
                    best_name = joint_names[j] if j < len(joint_names) else f"joint_{j}"

        if best_slot >= 0:
            self._selected_slot = best_slot
            self._selected_joint = best_joint
            self.joint_selected.emit(best_slot, best_joint, best_name)
        else:
            # Click on empty space — deselect
            self._selected_slot = -1
            self._selected_joint = -1

        self.update()
