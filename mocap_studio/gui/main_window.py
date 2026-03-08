"""
Main Window — Top-level layout wiring all components together.

Layout:
  ┌─────────────────────────────┬───────────────┐
  │      OpenGL 3D Viewer       │ Track Controls │
  │                             │   (panel)      │
  ├─────────────────────────────┴───────────────┤
  │  [Play] [Pause] [Stop]  Frame: N / M        │
  ├─────────────────────────────────────────────┤
  │           Timeline                           │
  ├─────────────────────────────────────────────┤
  │        Script Editor + Output                │
  └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QFileDialog, QComboBox, QStatusBar,
    QMenuBar, QMenu, QSpinBox, QMessageBox, QApplication,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon

import numpy as np

from ..core.session import Session
from ..core.track import Track
from ..core.skeleton import Skeleton

from .gl_viewer import GLViewer
from .timeline_widget import TimelineWidget
from .track_panel import TrackPanel
from .script_editor import ScriptEditor
from .styles import DARK_STYLESHEET

log = logging.getLogger("mocap_studio.gui.main_window")


def _load_file(filepath: str) -> Track:
    """Load a track from FBX or BVH file."""
    ext = os.path.splitext(filepath)[1].lower()
    log.info(f"Loading file: {filepath} (ext={ext})")
    if ext == ".bvh":
        from ..core.bvh_extract import load_bvh
        return load_bvh(filepath)
    elif ext == ".fbx":
        try:
            from ..core.fbx_extract import load_fbx
            return load_fbx(filepath)
        except ImportError as e:
            log.error(f"FBX SDK import error: {e}")
            raise ImportError(str(e))
    else:
        log.error(f"Unsupported file format: {ext}")
        raise ValueError(f"Unsupported file format: {ext}\nSupported: .fbx, .bvh")


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoCap Align & Compare")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        self.setStyleSheet(DARK_STYLESHEET)

        self._session = Session()
        self._playing = False
        self._play_timer = QTimer()
        self._play_timer.setInterval(16)  # ~60fps
        self._play_timer.timeout.connect(self._on_play_tick)

        self._playback_speed = 1.0

        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------
    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        save_session_act = QAction("Save Session", self)
        save_session_act.setShortcut("Ctrl+S")
        save_session_act.triggered.connect(self._on_save_session)
        file_menu.addAction(save_session_act)

        load_session_act = QAction("Load Session", self)
        load_session_act.setShortcut("Ctrl+O")
        load_session_act.triggered.connect(self._on_load_session)
        file_menu.addAction(load_session_act)

        file_menu.addSeparator()

        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # View menu
        view_menu = menubar.addMenu("View")
        reset_cam = QAction("Reset Camera", self)
        reset_cam.setShortcut("Ctrl+R")
        reset_cam.triggered.connect(lambda: self._viewer.reset_camera())
        view_menu.addAction(reset_cam)

        # Scripts menu
        scripts_menu = menubar.addMenu("Scripts")
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
        if os.path.isdir(scripts_dir):
            for fname in sorted(os.listdir(scripts_dir)):
                if fname.endswith(".py"):
                    act = QAction(fname, self)
                    path = os.path.join(scripts_dir, fname)
                    act.triggered.connect(
                        lambda checked, p=path: self._script_editor.load_script_file(p)
                    )
                    scripts_menu.addAction(act)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Top area: viewer + track panel
        top_splitter = QSplitter(Qt.Horizontal)

        self._viewer = GLViewer()
        self._viewer.joint_selected.connect(self._on_joint_selected)
        top_splitter.addWidget(self._viewer)

        self._track_panel = TrackPanel()
        self._track_panel.load_requested.connect(self._on_load_track)
        self._track_panel.unload_requested.connect(self._on_unload_track)
        self._track_panel.settings_changed.connect(self._on_track_settings_changed)
        self._track_panel.joints_changed.connect(self._on_joints_changed)
        top_splitter.addWidget(self._track_panel)

        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(top_splitter, 5)

        # Playback controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(6)

        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setFixedWidth(80)
        self._play_btn.clicked.connect(self._on_play)
        controls_layout.addWidget(self._play_btn)

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.setFixedWidth(80)
        self._pause_btn.clicked.connect(self._on_pause)
        controls_layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.setFixedWidth(80)
        self._stop_btn.clicked.connect(self._on_stop)
        controls_layout.addWidget(self._stop_btn)

        controls_layout.addSpacing(20)

        controls_layout.addWidget(QLabel("Frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 0)
        self._frame_spin.setFixedWidth(100)
        self._frame_spin.valueChanged.connect(self._on_frame_spinbox_changed)
        controls_layout.addWidget(self._frame_spin)

        self._frame_label = QLabel("/ 0")
        controls_layout.addWidget(self._frame_label)

        controls_layout.addSpacing(20)

        controls_layout.addWidget(QLabel("Speed:"))
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self._speed_combo.setCurrentIndex(2)
        self._speed_combo.currentTextChanged.connect(self._on_speed_changed)
        self._speed_combo.setFixedWidth(70)
        controls_layout.addWidget(self._speed_combo)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Bottom area: timeline + script editor
        bottom_splitter = QSplitter(Qt.Vertical)

        self._timeline = TimelineWidget()
        self._timeline.frame_changed.connect(self._on_timeline_frame_changed)
        bottom_splitter.addWidget(self._timeline)

        self._script_editor = ScriptEditor()
        self._script_editor.set_session(self._session)
        bottom_splitter.addWidget(self._script_editor)

        bottom_splitter.setSizes([120, 250])
        main_layout.addWidget(bottom_splitter, 3)

        central.setLayout(main_layout)

    def _setup_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._status_label = QLabel("Ready — Load FBX or BVH files to begin")
        self._statusbar.addPermanentWidget(self._status_label)

    # ------------------------------------------------------------------
    # Track loading
    # ------------------------------------------------------------------
    def _on_load_track(self, slot: int):
        log.info(f"User requested load for slot {slot}")
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load Track {slot + 1}", "",
            "Motion Files (*.fbx *.bvh);;FBX Files (*.fbx);;BVH Files (*.bvh);;All Files (*)"
        )
        if not path:
            log.debug("File dialog cancelled.")
            return

        try:
            track = _load_file(path)
        except Exception as e:
            log.error(f"Failed to load track into slot {slot}: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self._session.load_track(slot, track)
        self._update_track_ui(slot)
        self._update_viewer()
        self._update_timeline()
        self._update_frame_range()
        self._update_status()

    def _on_unload_track(self, slot: int):
        log.info(f"Unloading track from slot {slot}")
        self._session.remove_track(slot)
        self._track_panel.track_controls[slot].set_unloaded()
        self._viewer.set_track_data(slot, None, None)
        self._timeline.clear_track(slot)
        self._update_frame_range()
        self._update_status()

    def _update_track_ui(self, slot: int):
        track = self._session.tracks[slot]
        if track is None:
            return
        tc = self._track_panel.track_controls[slot]
        tc.set_loaded(
            name=track.name,
            joint_names=track.skeleton.joint_names,
            frame_count=track.frame_count,
            align_joint=track.align_joint,
        )

    # ------------------------------------------------------------------
    # Settings changed
    # ------------------------------------------------------------------
    def _on_track_settings_changed(self, slot: int):
        track = self._session.tracks[slot]
        if track is None:
            return

        tc = self._track_panel.track_controls[slot]
        track.offset = tc.offset_spin.value()
        track.trim_in = tc.trim_in_spin.value()
        track.trim_out = tc.trim_out_spin.value()

        old_align = track.align_joint
        track.align_joint = tc.align_combo.currentText()
        if track.align_joint != old_align:
            track.invalidate_cache()

        track.visible = tc.visible_cb.isChecked()
        track.translate_x = tc.pos_x_spin.value()
        track.translate_y = tc.pos_y_spin.value()
        track.translate_z = tc.pos_z_spin.value()

        # Check reference
        if tc.ref_radio.isChecked():
            self._session.reference_index = slot

        log.debug(
            f"Track {slot} settings: offset={track.offset}, "
            f"trim={track.trim_in}-{track.trim_out}, "
            f"align={track.align_joint}, visible={track.visible}, "
            f"pos=({track.translate_x}, {track.translate_y}, {track.translate_z})"
        )

        self._update_viewer()
        self._update_timeline()
        self._update_frame_range()

    def _on_joints_changed(self, slot: int, hidden_joints: set):
        track = self._session.tracks[slot]
        if track is None:
            return
        track.hidden_joints = hidden_joints
        log.info(f"Track {slot}: {len(hidden_joints)} joints hidden")
        self._update_viewer()

    def _on_joint_selected(self, slot: int, joint_index: int, joint_name: str):
        """Handle bone picking from the 3D viewer."""
        if 0 <= slot < 5:
            tc = self._track_panel.track_controls[slot]
            tc.set_selected_joint(joint_name)
            log.info(f"Picked track {slot+1} joint: {joint_name} ({joint_index})")

    # ------------------------------------------------------------------
    # Viewer update
    # ------------------------------------------------------------------
    def _update_viewer(self):
        for slot in range(5):
            track = self._session.tracks[slot]
            if track is None:
                self._viewer.set_track_data(slot, None, None)
                continue

            # Use aligned positions and apply offset
            aligned = track.aligned_positions
            if aligned is None:
                self._viewer.set_track_data(slot, None, None)
                continue

            # Apply frame offset by prepending/trimming
            offset = track.offset
            if offset > 0:
                # Pad beginning with first frame
                pad = np.tile(aligned[0:1], (offset, 1, 1))
                aligned = np.concatenate([pad, aligned], axis=0)
            elif offset < 0:
                # Trim beginning
                aligned = aligned[abs(offset):]

            self._viewer.set_track_data(
                slot, aligned,
                track.skeleton.get_bone_pairs(),
                track.visible,
                hidden_joints=track.hidden_joint_indices,
                translate=track.translate,
                joint_names=track.skeleton.joint_names,
            )

        self._viewer.set_current_frame(self._session.current_frame)

    def _update_timeline(self):
        mx = self._session.max_frame
        self._timeline.set_max_frame(mx)

        for slot in range(5):
            track = self._session.tracks[slot]
            if track is None:
                self._timeline.clear_track(slot)
                continue
            self._timeline.set_track_info(
                slot,
                track.frame_count,
                track.offset,
                track.trim_in,
                track.trim_out,
                track.visible,
            )

    def _update_frame_range(self):
        mx = self._session.max_frame
        self._frame_spin.setRange(0, max(0, mx - 1))
        self._frame_label.setText(f"/ {mx}")

    def _update_status(self):
        loaded = len(self._session.loaded_tracks)
        mx = self._session.max_frame
        self._status_label.setText(
            f"{loaded} track{'s' if loaded != 1 else ''} loaded  |  "
            f"Frame {self._session.current_frame}/{mx}"
        )

    # ------------------------------------------------------------------
    # Frame navigation
    # ------------------------------------------------------------------
    def _set_frame(self, frame: int):
        mx = self._session.max_frame
        frame = max(0, min(frame, mx - 1))
        self._session.current_frame = frame
        self._viewer.set_current_frame(frame)
        self._timeline.set_current_frame(frame)
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(frame)
        self._frame_spin.blockSignals(False)
        self._update_status()

    def _on_timeline_frame_changed(self, frame: int):
        self._set_frame(frame)

    def _on_frame_spinbox_changed(self, value: int):
        self._set_frame(value)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def _on_play(self):
        log.info(f"Playback started (speed={self._playback_speed}x)")
        self._playing = True
        self._play_timer.start()
        self._play_btn.setEnabled(False)

    def _on_pause(self):
        log.info(f"Playback paused at frame {self._session.current_frame}")
        self._playing = False
        self._play_timer.stop()
        self._play_btn.setEnabled(True)

    def _on_stop(self):
        log.info("Playback stopped, reset to frame 0")
        self._playing = False
        self._play_timer.stop()
        self._play_btn.setEnabled(True)
        self._set_frame(0)

    def _on_play_tick(self):
        mx = self._session.max_frame
        if mx <= 0:
            return
        new_frame = self._session.current_frame + int(self._playback_speed)
        if new_frame >= mx:
            new_frame = 0  # loop
        self._set_frame(new_frame)

    def _on_speed_changed(self, text: str):
        try:
            self._playback_speed = float(text.replace("x", ""))
            log.info(f"Playback speed changed to {self._playback_speed}x")
        except ValueError:
            self._playback_speed = 1.0
            log.warning(f"Invalid speed value '{text}', defaulting to 1.0x")

    # ------------------------------------------------------------------
    # Session I/O
    # ------------------------------------------------------------------
    def _on_save_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "",
            "MoCap Session (*.json);;All Files (*)"
        )
        if path:
            try:
                self._session.save_session(path)
                self._status_label.setText(f"Session saved: {os.path.basename(path)}")
            except Exception as e:
                log.error(f"Failed to save session: {e}", exc_info=True)
                QMessageBox.critical(self, "Save Error", str(e))

    def _on_load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "MoCap Session (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            self._session.load_session(path, loader_fn=_load_file)
        except Exception as e:
            log.error(f"Failed to load session: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", str(e))
            return

        # Update all UI
        for slot in range(5):
            track = self._session.tracks[slot]
            if track is not None:
                self._update_track_ui(slot)
            else:
                self._track_panel.track_controls[slot].set_unloaded()

        self._update_viewer()
        self._update_timeline()
        self._update_frame_range()
        self._set_frame(self._session.current_frame)
        self._update_status()
        self._status_label.setText(f"Session loaded: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            if self._playing:
                self._on_pause()
            else:
                self._on_play()
        elif key == Qt.Key_Right:
            step = 10 if event.modifiers() & Qt.ShiftModifier else 1
            self._set_frame(self._session.current_frame + step)
        elif key == Qt.Key_Left:
            step = 10 if event.modifiers() & Qt.ShiftModifier else 1
            self._set_frame(self._session.current_frame - step)
        elif key == Qt.Key_Home:
            self._set_frame(0)
        elif key == Qt.Key_End:
            self._set_frame(self._session.max_frame - 1)
        else:
            super().keyPressEvent(event)
