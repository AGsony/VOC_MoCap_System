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
    QTabWidget, QScrollArea, QProgressDialog, QCheckBox,
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
from .console_widget import ConsoleWidget
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

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(60000)  # 60s
        self._autosave_timer.timeout.connect(self._on_autosave_tick)
        self._autosave_timer.start()

        self._playback_speed = 1.0

        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()

        QTimer.singleShot(100, self._check_autosave_recovery)

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
        self._track_panel.auto_align_requested.connect(self._on_auto_align_requested)
        self._track_panel.file_dropped.connect(self._on_file_dropped)
        self._track_panel.rest_pose_requested.connect(self._on_rest_pose_requested)
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

        self._cut_btn = QPushButton("✂ Cut")
        self._cut_btn.setToolTip("Trim all loaded tracks to the current playhead position")
        self._cut_btn.clicked.connect(self._on_cut_requested)
        controls_layout.addWidget(self._cut_btn)

        self._export_btn = QPushButton("💾 Export Timeline")
        self._export_btn.setToolTip("Export the entire aligned timeline to BVH/FBX")
        self._export_btn.clicked.connect(self._on_export_requested)
        controls_layout.addWidget(self._export_btn)

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

        controls_layout.addSpacing(20)

        self._snap_cb = QCheckBox("Snap to Frame")
        self._snap_cb.setChecked(False)
        self._snap_cb.toggled.connect(self._on_snap_toggled)
        controls_layout.addWidget(self._snap_cb)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Bottom area: timeline + script editor
        bottom_splitter = QSplitter(Qt.Vertical)

        self._timeline = TimelineWidget()
        self._timeline.setMinimumHeight(200)
        self._timeline.frame_changed.connect(self._on_timeline_frame_changed)
        self._timeline.track_offset_changed.connect(self._on_track_offset_changed)
        self._timeline.track_trim_changed.connect(self._on_track_trim_changed)
        
        timeline_scroll = QScrollArea()
        timeline_scroll.setWidgetResizable(True)
        timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        timeline_scroll.setWidget(self._timeline)
        bottom_splitter.addWidget(timeline_scroll)

        bottom_tabs = QTabWidget()
        bottom_tabs.setTabPosition(QTabWidget.South)

        self._console = ConsoleWidget()
        bottom_tabs.addTab(self._console, "System Console")

        self._script_editor = ScriptEditor()
        self._script_editor.set_session(self._session)
        bottom_tabs.addTab(self._script_editor, "Script Editor")

        bottom_splitter.addWidget(bottom_tabs)

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

        progress = QProgressDialog(f"Parsing '{os.path.basename(path)}'...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Importing Track")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()

        try:
            track = _load_file(path)
        except Exception as e:
            log.error(f"Failed to load track into slot {slot}: {e}", exc_info=True)
            progress.close()
            QMessageBox.critical(self, "Load Error", str(e))
            return

        progress.close()
        self._session.load_track(slot, track)
        self._update_track_ui(slot)
        self._update_viewer()
        self._update_timeline()
        self._update_frame_range()
        self._update_status()

    def _on_file_dropped(self, slot: int, path: str):
        log.info(f"File dropped into slot {slot}: {path}")
        
        progress = QProgressDialog(f"Parsing '{os.path.basename(path)}'...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Importing Track")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        try:
            track = _load_file(path)
        except Exception as e:
            log.error(f"Failed to load dropped track into slot {slot}: {e}", exc_info=True)
            progress.close()
            QMessageBox.critical(self, "Load Error", str(e))
            return

        progress.close()
        self._session.load_track(slot, track)
        self._update_track_ui(slot)
        self._update_viewer()
        self._update_timeline()
        self._update_frame_range()
        self._update_status()

    def _on_rest_pose_requested(self, slot: int):
        track = self._session.tracks[slot]
        if track is None: return
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Assign Rest Pose File", "",
            "MoCap Files (*.fbx *.bvh)"
        )
        if not path:
            return

        try:
            filename = os.path.basename(path)
            log.info(f"Loading rest pose from {filename} for slot {slot}")
            
            progress = QProgressDialog(f"Parsing '{filename}'...", "Cancel", 0, 0, self)
            progress.setWindowTitle("Importing Rest Pose")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            from core.fbx_extract import load_fbx
            from core.bvh_extract import load_bvh
            
            ext = path.lower().split('.')[-1]
            if ext == 'fbx':
                rest_track = load_fbx(path)
            elif ext == 'bvh':
                rest_track = load_bvh(path)
            else:
                progress.close()
                raise ValueError("Unsupported file format for rest pose.")
                
            track.rest_pose_positions = rest_track.positions[0]
            track.rest_pose_quaternions = rest_track.quaternions[0] if rest_track.quaternions is not None else None
            track.rest_pose_name = filename
            
            progress.close()
            self._update_track_ui(slot)
            
        except Exception as e:
            log.error(f"Failed to load rest pose: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", f"Could not load rest pose:\n{e}")

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
            offset=track.offset,
            scale=track.scale,
            trim_in=track.trim_in,
            trim_out=track.trim_out,
            visible=track.visible,
            translate=(track.translate_x, track.translate_y, track.translate_z),
            rotate=(track.rotate_x, track.rotate_y, track.rotate_z),
            rest_pose_name=track.rest_pose_name,
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
        track.scale = tc.scale_spin.value()
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
        track.rotate_x = tc.rot_x_spin.value()
        track.rotate_y = tc.rot_y_spin.value()
        track.rotate_z = tc.rot_z_spin.value()

        # Check reference
        if tc.ref_radio.isChecked():
            self._session.reference_index = slot

        log.debug(
            f"Track {slot} settings: offset={track.offset}, scale={track.scale}, "
            f"trim={track.trim_in}-{track.trim_out}, "
            f"align={track.align_joint}, visible={track.visible}, "
            f"translate=({track.translate_x}, {track.translate_y}, {track.translate_z}), "
            f"rotate=({track.rotate_x}, {track.rotate_y}, {track.rotate_z})"
        )

        self._update_viewer()
        self._update_timeline()
        self._update_frame_range()

    def _on_track_offset_changed(self, slot: int, offset: float):
        track = self._session.tracks[slot]
        if track is not None:
            track.offset = offset
            tc = self._track_panel.track_controls[slot]
            tc.offset_spin.blockSignals(True)
            tc.offset_spin.setValue(offset)
            tc.offset_spin.blockSignals(False)
            self._update_viewer()
            self._update_frame_range()
            
    def _on_track_trim_changed(self, slot: int, trim_in: int, trim_out: int):
        track = self._session.tracks[slot]
        if track is not None:
            track.trim_in = trim_in
            track.trim_out = trim_out
            tc = self._track_panel.track_controls[slot]
            tc.trim_in_spin.blockSignals(True)
            tc.trim_out_spin.blockSignals(True)
            tc.trim_in_spin.setValue(trim_in)
            tc.trim_out_spin.setValue(trim_out)
            tc.trim_in_spin.blockSignals(False)
            tc.trim_out_spin.blockSignals(False)
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

    def _on_auto_align_requested(self, slot: int):
        ref_idx = self._session.reference_index
        if slot == ref_idx:
            QMessageBox.information(self, "Auto-Align", "Cannot auto-align the reference track to itself.")
            return
            
        ref_track = self._session.tracks[ref_idx]
        test_track = self._session.tracks[slot]
        
        if not ref_track or not test_track:
            return
            
        from core.align import auto_align_tracks
        optimal_offset = auto_align_tracks(ref_track, test_track)
        
        log.info(f"Auto-align track {slot} -> {optimal_offset} offset")
        self._on_track_offset_changed(slot, optimal_offset)

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

            # Apply frame offset and scale via sub-frame interpolation
            offset = track.offset
            scale = track.scale
            
            F, J, C = aligned.shape
            
            # The Viewer expects `aligned[f]` to correspond to global frame `f`.
            # So we must generate an array starting at global frame 0.
            if offset > 0:
                total_frames = int(np.ceil(offset + F * scale))
            else:
                total_frames = int(np.ceil(F * scale + offset))
            total_frames = max(0, total_frames)
            
            if total_frames > 0:
                global_frames = np.arange(total_frames)
                
                # local_time = (global_frame - offset) / scale
                local_times = (global_frames - offset) / scale
                
                # Clamp local_times to the valid range of the original track [0, F-1]
                # This naturally handles padding: 
                # global frames < offset will clamp to local 0.
                # global frames > offset + (F-1) * scale will clamp to local F-1.
                local_times = np.clip(local_times, 0, F - 1)
                
                # Perform linear interpolation
                idx0 = np.floor(local_times).astype(int)
                idx1 = np.minimum(idx0 + 1, F - 1)
                frac = (local_times - idx0)[:, np.newaxis, np.newaxis]
                
                aligned = aligned[idx0] * (1.0 - frac) + aligned[idx1] * frac
            else:
                aligned = np.zeros((0, J, C))

            self._viewer.set_track_data(
                slot, aligned,
                track.skeleton.get_bone_pairs(),
                track.visible,
                hidden_joints=track.hidden_joint_indices,
                translate=track.translate,
                rotate=(track.rotate_x, track.rotate_y, track.rotate_z),
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
                track.scale,
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
        self._update_timeline()

    def _on_export_requested(self):
        log.info(f"Timeline Export requested.")
        path, filter_str = QFileDialog.getSaveFileName(
            self, "Export Timeline", "",
            "FBX Files (*.fbx);;BVH Files (*.bvh);;All Files (*)"
        )
        if not path:
            return
            
        progress = QProgressDialog("Exporting Timeline Data...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Export Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        
        def update_progress(val: int) -> bool:
            progress.setValue(val)
            QApplication.processEvents()
            return progress.wasCanceled()

        try:
            if path.lower().endswith('.fbx'):
                from mocap_studio.core.exporter import export_timeline_to_fbx
                export_timeline_to_fbx(self._session, path, progress_callback=update_progress)
            else:
                from mocap_studio.core.exporter import export_timeline_to_bvh
                export_timeline_to_bvh(self._session, path, progress_callback=update_progress)
                
            if progress.wasCanceled():
                log.info("Timeline Export was canceled by user.")
                # We do not delete partial files here, but log it
            else:
                progress.setValue(100)
                QMessageBox.information(self, "Export Timeline", f"Successfully exported timeline to:\n{path}")
                
        except Exception as e:
            log.error(f"Failed to export timeline: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Error", str(e))
        finally:
            progress.close()

    def _on_cut_requested(self):
        gframe = self._session.current_frame
        cut_count = 0
        
        for slot, track in enumerate(self._session.tracks):
            if track is None:
                continue
                
            local_f = int(round((gframe - track.offset) / track.scale))
            
            if local_f < 0 or local_f >= track.frame_count:
                continue # Playhead is outside this track

            midpoint = (track.trim_in + track.trim_out) / 2.0
            if local_f <= midpoint:
                track.trim_in = local_f
            else:
                track.trim_out = local_f
            
            cut_count += 1
            self._update_track_ui(slot)

        if cut_count > 0:
            log.info(f"Timeline Cut: Trimmed {cut_count} tracks at frame {gframe}")
            self._update_timeline()
            self._update_frame_range()
            self._update_viewer()
        else:
            QMessageBox.warning(self, "Cut Timeline", "Playhead is outside the bounds of all loaded tracks.")

    # ------------------------------------------------------------------
    # Playback Control
    # ------------------------------------------------------------------
    def _on_play(self):
        log.info(f"Playback started (speed={self._playback_speed}x)")
        self._playing = True
        self._play_btn.setText("⏸ Pause")
        self._play_timer.start()

    def _on_pause(self):
        log.info(f"Playback paused at frame {self._session.current_frame}")
        self._playing = False
        self._play_btn.setText("▶ Play")
        self._play_timer.stop()
        self._play_btn.setEnabled(True)

    def _on_stop(self):
        log.info("Playback stopped, reset to frame 0")
        self._playing = False
        self._play_btn.setText("▶ Play")
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

    def keyPressEvent(self, event):
        """Keyboard shortcuts for playback."""
        if event.key() == Qt.Key_Space:
            if self._playing:
                self._on_pause()
            else:
                self._on_play()
            event.accept()
        elif event.key() == Qt.Key_Left:
            step = 10 if event.modifiers() == Qt.ShiftModifier else 1
            self._set_frame(self._session.current_frame - step)
            event.accept()
        elif event.key() == Qt.Key_Right:
            step = 10 if event.modifiers() == Qt.ShiftModifier else 1
            self._set_frame(self._session.current_frame + step)
            event.accept()
        elif event.key() == Qt.Key_Home:
            self._set_frame(0)
            event.accept()
        elif event.key() == Qt.Key_End:
            self._set_frame(self._session.max_frame - 1)
            event.accept()
        else:
            super().keyPressEvent(event)

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
    # Auto-Save & Settings
    # ------------------------------------------------------------------
    def _on_snap_toggled(self, checked: bool):
        self._timeline.snap_to_frame = checked

    def _get_autosave_path(self) -> str:
        return os.path.expanduser("~/.mocap_studio_autosave.json")

    def _on_autosave_tick(self):
        if len(self._session.loaded_tracks) == 0:
            return
        path = self._get_autosave_path()
        try:
            self._session.save_session(path)
            log.debug(f"Auto-saved session to {path}")
        except Exception as e:
            log.error(f"Auto-save failed: {e}")

    def _check_autosave_recovery(self):
        path = self._get_autosave_path()
        if os.path.exists(path):
            reply = QMessageBox.question(
                self, "Recover Session",
                "An auto-saved session was found. Would you like to recover it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    self._session.load_session(path, loader_fn=_load_file)
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
                    self._status_label.setText("Recovered auto-saved session.")
                    log.info("Auto-saved session recovered.")
                except Exception as e:
                    log.error(f"Failed to recover auto-save: {e}", exc_info=True)
                    QMessageBox.warning(self, "Recovery Failed", f"Could not recover session:\n{e}")
            else:
                try:
                    os.remove(path)
                    log.info("Auto-saved session discarded by user.")
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Keyboard shortcuts handled in top-level keyPressEvent above
    # ------------------------------------------------------------------
