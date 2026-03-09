"""
Track Panel — Per-track controls for loading, alignment, offset, trim,
              3D position, and joint visibility.

Shows 5 collapsible track control groups, each with:
  - Load button
  - Alignment joint dropdown
  - Frame offset spinbox
  - In/Out frame spinboxes
  - XYZ position spinboxes
  - Joint visibility dropdown
  - Visibility checkbox
  - Reference radio button
"""

from __future__ import annotations

from typing import Optional, List, Set

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QRadioButton,
    QScrollArea, QSizePolicy, QButtonGroup, QLineEdit, QListWidget,
    QListWidgetItem, QFrame,
)
from PySide6.QtCore import Signal, Qt, QPoint


# Track colors for label styling
TRACK_COLOR_STYLES = [
    "color: #40E06C;",  # green
    "color: #4D8CF2;",  # blue
    "color: #F2D940;",  # yellow
    "color: #F25959;",  # red
    "color: #B866E6;",  # purple
]


# ---------------------------------------------------------------------------
# Joint dropdown popup
# ---------------------------------------------------------------------------
class JointDropdownPopup(QFrame):
    """Dropdown popup for joint visibility, appears below the trigger button."""

    joints_changed = Signal(set)  # emitted with set of hidden joint names

    def __init__(self, joint_names: List[str], hidden_joints: Set[str],
                 parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.setStyleSheet("""
            JointDropdownPopup {
                background: #1e1e24;
                border: 1px solid #4a4a55;
                border-radius: 4px;
            }
        """)

        self._joint_names = joint_names
        self.setFixedWidth(280)
        self.setMinimumHeight(200)
        self.setMaximumHeight(400)

        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # Search filter
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter joints...")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        self._filter.setFixedHeight(24)
        layout.addWidget(self._filter)

        # Show All / Hide All
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        show_btn = QPushButton("Show All")
        show_btn.setFixedHeight(22)
        show_btn.clicked.connect(self._show_all)
        btn_row.addWidget(show_btn)
        hide_btn = QPushButton("Hide All")
        hide_btn.setFixedHeight(22)
        hide_btn.clicked.connect(self._hide_all)
        btn_row.addWidget(hide_btn)
        layout.addLayout(btn_row)

        # Joint list with checkboxes
        self._list = QListWidget()
        self._list.setStyleSheet("""
            QListWidget {
                background: #16161a;
                border: 1px solid #3a3a42;
                border-radius: 3px;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 2px 4px;
            }
            QListWidget::item:hover {
                background: #2a2a35;
            }
        """)

        for name in joint_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if name in hidden_joints else Qt.Checked)
            self._list.addItem(item)

        layout.addWidget(self._list, 1)

        # Info label
        self._info = QLabel()
        self._info.setStyleSheet("color: #888; font-size: 10px;")
        self._update_info()
        layout.addWidget(self._info)

        self.setLayout(layout)

    def _apply_filter(self, text: str):
        text = text.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(text not in item.text().lower())

    def _show_all(self):
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked)
        self._list.blockSignals(False)
        self._update_info()

    def _hide_all(self):
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Unchecked)
        self._list.blockSignals(False)
        self._update_info()

    def _update_info(self):
        hidden = self.get_hidden_joints()
        total = len(self._joint_names)
        self._info.setText(f"{total - len(hidden)}/{total} joints visible")

    def get_hidden_joints(self) -> Set[str]:
        hidden = set()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.Unchecked:
                hidden.add(item.text())
        return hidden

    def hideEvent(self, event):
        """Emit changes when the dropdown closes."""
        self.joints_changed.emit(self.get_hidden_joints())
        super().hideEvent(event)


# ---------------------------------------------------------------------------
# Draggable SpinBox
# ---------------------------------------------------------------------------
class DraggableSpinBox(QDoubleSpinBox):
    """A spinbox that allows changing its value by clicking and dragging horizontally."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_start_pos = None
        self._drag_start_value = 0.0
        
        # We must intercept mouse events on the internal line edit
        self.lineEdit().setCursor(Qt.SizeHorCursor)
        self.lineEdit().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.lineEdit():
            # Mouse Press
            if event.type() == event.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                self._drag_start_pos = event.globalPosition()
                self._drag_start_value = self.value()
                # Clear focus to allow dragging instead of text selection
                self.lineEdit().clearFocus()
                return True
            # Mouse Move
            elif event.type() == event.Type.MouseMove and self._drag_start_pos is not None:
                delta = event.globalPosition().x() - self._drag_start_pos.x()
                sensitivity = self.singleStep()
                modifiers = event.modifiers()
                if modifiers & Qt.ShiftModifier:
                    sensitivity *= 10.0
                elif modifiers & Qt.ControlModifier:
                    sensitivity *= 0.1
                
                self.setValue(self._drag_start_value + (delta * sensitivity))
                return True
            # Mouse Release
            elif event.type() == event.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._drag_start_pos = None
                return True
                
        return super().eventFilter(obj, event)

# ---------------------------------------------------------------------------
# Single track controls
# ---------------------------------------------------------------------------
class SingleTrackControls(QGroupBox):
    """Controls for a single track slot."""

    load_requested = Signal(int)    # slot
    unload_requested = Signal(int)  # slot
    settings_changed = Signal(int)  # slot
    joints_changed = Signal(int, set)  # slot, hidden_joints
    auto_align_requested = Signal(int) # slot
    file_dropped = Signal(int, str) # slot, path
    rest_pose_requested = Signal(int) # slot

    def __init__(self, slot: int, parent=None):
        super().__init__(parent)
        self.slot = slot
        self._loaded = False
        self._joint_names: List[str] = []
        self._hidden_joints: Set[str] = set()

        self.setAcceptDrops(True)

        self.setTitle(f"  Track {slot + 1}")
        self.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #3a3a42;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 14px;
                font-weight: bold;
                {TRACK_COLOR_STYLES[slot]}
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # Row 1: Load, Visibility
        row1 = QHBoxLayout()
        self.load_btn = QPushButton("Load FBX/BVH")
        self.load_btn.clicked.connect(lambda: self.load_requested.emit(self.slot))
        self.load_btn.setFixedHeight(26)
        row1.addWidget(self.load_btn)

        self.rest_btn = QPushButton("Assign Rest Pose")
        self.rest_btn.clicked.connect(lambda: self.rest_pose_requested.emit(self.slot))
        self.rest_btn.setFixedHeight(26)
        row1.addWidget(self.rest_btn)

        self.unload_btn = QPushButton("✕")
        self.unload_btn.setFixedWidth(26)
        self.unload_btn.setFixedHeight(26)
        self.unload_btn.setToolTip("Unload track")
        self.unload_btn.clicked.connect(lambda: self.unload_requested.emit(self.slot))
        self.unload_btn.setEnabled(False)
        row1.addWidget(self.unload_btn)

        self.visible_cb = QCheckBox("Vis")
        self.visible_cb.setChecked(True)
        self.visible_cb.toggled.connect(lambda: self.settings_changed.emit(self.slot))
        row1.addWidget(self.visible_cb)
        layout.addLayout(row1)

        # Row 2: Filename label
        self.filename_label = QLabel("No file loaded")
        self.filename_label.setStyleSheet("color: #888; font-size: 10px; font-weight: normal;")
        self.filename_label.setWordWrap(True)
        layout.addWidget(self.filename_label)

        # Row 3: Alignment joint
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Align:"))
        self.align_combo = QComboBox()
        self.align_combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        self.align_combo.setMinimumWidth(60)
        self.align_combo.currentTextChanged.connect(
            lambda: self.settings_changed.emit(self.slot)
        )
        row3.addWidget(self.align_combo, 1)

        self.auto_btn = QPushButton("Auto-Sync")
        self.auto_btn.setFixedHeight(22)
        self.auto_btn.setToolTip("Auto-sync this track's offset against the Reference track")
        self.auto_btn.clicked.connect(lambda: self.auto_align_requested.emit(self.slot))
        row3.addWidget(self.auto_btn)

        self.ref_radio = QRadioButton("Ref")
        self.ref_radio.setToolTip("Set as reference track")
        row3.addWidget(self.ref_radio)
        layout.addLayout(row3)

        # Row 4: Offset & Scale
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Offset:"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-10000.0, 10000.0)
        self.offset_spin.setDecimals(2)
        self.offset_spin.setSingleStep(1.0)
        self.offset_spin.setValue(0.0)
        self.offset_spin.setMinimumWidth(50)
        self.offset_spin.valueChanged.connect(
            lambda: self.settings_changed.emit(self.slot)
        )
        row4.addWidget(self.offset_spin)
        
        row4.addWidget(QLabel("Scale:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 10.0)
        self.scale_spin.setDecimals(3)
        self.scale_spin.setSingleStep(0.01)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setMinimumWidth(50)
        self.scale_spin.valueChanged.connect(
            lambda: self.settings_changed.emit(self.slot)
        )
        row4.addWidget(self.scale_spin, 1)
        layout.addLayout(row4)

        # Row 5: Trim In/Out
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("In:"))
        self.trim_in_spin = QSpinBox()
        self.trim_in_spin.setRange(0, 99999)
        self.trim_in_spin.setMinimumWidth(50)
        self.trim_in_spin.valueChanged.connect(
            lambda: self.settings_changed.emit(self.slot)
        )
        row5.addWidget(self.trim_in_spin)
        row5.addWidget(QLabel("Out:"))
        self.trim_out_spin = QSpinBox()
        self.trim_out_spin.setRange(0, 99999)
        self.trim_out_spin.setMinimumWidth(50)
        self.trim_out_spin.valueChanged.connect(
            lambda: self.settings_changed.emit(self.slot)
        )
        row5.addWidget(self.trim_out_spin)
        layout.addLayout(row5)

        # Row 6: Position XYZ
        row6 = QHBoxLayout()
        row6.setSpacing(2)

        self.pos_x_spin = self._make_pos_spin("X")
        self.pos_y_spin = self._make_pos_spin("Y")
        self.pos_z_spin = self._make_pos_spin("Z")

        lx = QLabel("X:")
        lx.setFixedWidth(14)
        ly = QLabel("Y:")
        ly.setFixedWidth(14)
        lz = QLabel("Z:")
        lz.setFixedWidth(14)

        row6.addWidget(lx)
        row6.addWidget(self.pos_x_spin, 1)
        row6.addWidget(ly)
        row6.addWidget(self.pos_y_spin, 1)
        row6.addWidget(lz)
        row6.addWidget(self.pos_z_spin, 1)
        layout.addLayout(row6)

        # Row 7: Rotation XYZ
        row7 = QHBoxLayout()
        row7.setSpacing(2)

        self.rot_x_spin = self._make_pos_spin("RX", 360.0)
        self.rot_y_spin = self._make_pos_spin("RY", 360.0)
        self.rot_z_spin = self._make_pos_spin("RZ", 360.0)

        lrx = QLabel("RX:")
        lrx.setFixedWidth(20)
        lry = QLabel("RY:")
        lry.setFixedWidth(20)
        lrz = QLabel("RZ:")
        lrz.setFixedWidth(20)

        row7.addWidget(lrx)
        row7.addWidget(self.rot_x_spin, 1)
        row7.addWidget(lry)
        row7.addWidget(self.rot_y_spin, 1)
        row7.addWidget(lrz)
        row7.addWidget(self.rot_z_spin, 1)
        layout.addLayout(row7)

        # Row 8: Joints dropdown button
        row8 = QHBoxLayout()
        self.joints_btn = QPushButton("🦴 Joints ▾")
        self.joints_btn.setFixedHeight(24)
        self.joints_btn.setToolTip("Show/hide individual joints")
        self.joints_btn.clicked.connect(self._on_joints_clicked)
        row8.addWidget(self.joints_btn)

        self.joints_count_label = QLabel("")
        self.joints_count_label.setStyleSheet(
            "color: #888; font-size: 10px; font-weight: normal;"
        )
        row8.addWidget(self.joints_count_label)
        row8.addStretch()
        layout.addLayout(row8)

        self.setLayout(layout)
        self._set_controls_enabled(False)

    # ------------------------------------------------------------------
    def _make_pos_spin(self, axis: str, spin_range: float=10000.0) -> DraggableSpinBox:
        spin = DraggableSpinBox()
        spin.setRange(-spin_range, spin_range)
        spin.setSingleStep(1.0)
        spin.setDecimals(1)
        spin.setValue(0.0)
        spin.setFixedHeight(22)
        spin.setMinimumWidth(40)  # Stop the layout from forcing itself too wide
        spin.setToolTip(f"{axis} position offset")
        spin.valueChanged.connect(
            lambda: self.settings_changed.emit(self.slot)
        )
        return spin

    # ------------------------------------------------------------------
    def _on_joints_clicked(self):
        if not self._loaded or not self._joint_names:
            return
        popup = JointDropdownPopup(
            self._joint_names, self._hidden_joints, parent=self,
        )
        popup.joints_changed.connect(self._on_popup_closed)

        # Position below the button
        btn_pos = self.joints_btn.mapToGlobal(QPoint(0, self.joints_btn.height()))
        popup.move(btn_pos)
        popup.show()

    def _on_popup_closed(self, hidden: set):
        self._hidden_joints = hidden
        self._update_joints_label()
        self.joints_changed.emit(self.slot, self._hidden_joints)

    # ------------------------------------------------------------------
    def _update_joints_label(self):
        total = len(self._joint_names)
        hidden = len(self._hidden_joints)
        if hidden > 0:
            self.joints_count_label.setText(f"({total - hidden}/{total})")
            self.joints_count_label.setStyleSheet(
                "color: #e8a735; font-size: 10px; font-weight: normal;"
            )
        else:
            self.joints_count_label.setText(f"({total})")
            self.joints_count_label.setStyleSheet(
                "color: #888; font-size: 10px; font-weight: normal;"
            )

    # ------------------------------------------------------------------
    def _set_controls_enabled(self, enabled: bool):
        self.align_combo.setEnabled(enabled)
        self.offset_spin.setEnabled(enabled)
        self.trim_in_spin.setEnabled(enabled)
        self.trim_out_spin.setEnabled(enabled)
        self.visible_cb.setEnabled(enabled)
        self.ref_radio.setEnabled(enabled)
        self.unload_btn.setEnabled(enabled)
        self.pos_x_spin.setEnabled(enabled)
        self.pos_y_spin.setEnabled(enabled)
        self.pos_z_spin.setEnabled(enabled)
        self.rot_x_spin.setEnabled(enabled)
        self.rot_y_spin.setEnabled(enabled)
        self.rot_z_spin.setEnabled(enabled)
        self.joints_btn.setEnabled(enabled)
        self.rest_btn.setEnabled(enabled)

    def set_loaded(self, name: str, joint_names: List[str],
                   frame_count: int, align_joint: str,
                   offset: float = 0.0, scale: float = 1.0,
                   trim_in: int = 0, trim_out: int = 0,
                   visible: bool = True, translate: tuple = (0.0, 0.0, 0.0),
                   rotate: tuple = (0.0, 0.0, 0.0),
                   rest_pose_name: str = ""):
        """Called after a track is loaded successfully."""
        self._loaded = True
        self._joint_names = joint_names
        self._hidden_joints = set()
        self.filename_label.setText(name)
        self.filename_label.setStyleSheet("color: #ccc; font-size: 10px; font-weight: normal;")

        # Block all signals during setup
        self.align_combo.blockSignals(True)
        self.offset_spin.blockSignals(True)
        self.scale_spin.blockSignals(True)
        self.trim_in_spin.blockSignals(True)
        self.trim_out_spin.blockSignals(True)
        self.visible_cb.blockSignals(True)
        self.pos_x_spin.blockSignals(True)
        self.pos_y_spin.blockSignals(True)
        self.pos_z_spin.blockSignals(True)
        self.rot_x_spin.blockSignals(True)
        self.rot_y_spin.blockSignals(True)
        self.rot_z_spin.blockSignals(True)

        self.align_combo.clear()
        self.align_combo.addItems(joint_names)
        idx = self.align_combo.findText(align_joint)
        if idx >= 0:
            self.align_combo.setCurrentIndex(idx)

        self.offset_spin.setValue(offset)
        self.scale_spin.setValue(scale)
        self.trim_in_spin.setRange(0, frame_count - 1)
        self.trim_out_spin.setRange(0, frame_count - 1)
        self.trim_in_spin.setValue(trim_in)
        self.trim_out_spin.setValue(trim_out)
        self.visible_cb.setChecked(visible)
        self.pos_x_spin.setValue(translate[0])
        self.pos_y_spin.setValue(translate[1])
        self.pos_z_spin.setValue(translate[2])
        self.rot_x_spin.setValue(rotate[0])
        self.rot_y_spin.setValue(rotate[1])
        self.rot_z_spin.setValue(rotate[2])

        if rest_pose_name:
            self.rest_btn.setText(f"Rest: {rest_pose_name}")
            self.rest_btn.setToolTip(f"Rest Pose: {rest_pose_name}")
        else:
            self.rest_btn.setText("Assign Rest Pose")
            self.rest_btn.setToolTip("Assign a reference frame 0 rest pose for exports")

        # Unblock all signals
        self.align_combo.blockSignals(False)
        self.offset_spin.blockSignals(False)
        self.trim_in_spin.blockSignals(False)
        self.trim_out_spin.blockSignals(False)
        self.visible_cb.blockSignals(False)
        self.pos_x_spin.blockSignals(False)
        self.pos_y_spin.blockSignals(False)
        self.pos_z_spin.blockSignals(False)
        self.rot_x_spin.blockSignals(False)
        self.rot_y_spin.blockSignals(False)
        self.rot_z_spin.blockSignals(False)

        self._update_joints_label()
        self.load_btn.setText("Reload")
        self._set_controls_enabled(True)

    def set_unloaded(self):
        self._loaded = False
        self._joint_names = []
        self._hidden_joints = set()
        self.filename_label.setText("No file loaded")
        self.filename_label.setStyleSheet("color: #888; font-size: 10px; font-weight: normal;")

        # Block signals during teardown
        self.align_combo.blockSignals(True)
        self.offset_spin.blockSignals(True)
        self.scale_spin.blockSignals(True) # Added scale_spin
        self.trim_in_spin.blockSignals(True)
        self.trim_out_spin.blockSignals(True)
        self.visible_cb.blockSignals(True)
        self.pos_x_spin.blockSignals(True)
        self.pos_y_spin.blockSignals(True)
        self.pos_z_spin.blockSignals(True)
        self.rot_x_spin.blockSignals(True)
        self.rot_y_spin.blockSignals(True)
        self.rot_z_spin.blockSignals(True)

        self.align_combo.clear()
        self.offset_spin.setValue(0)
        self.scale_spin.setValue(1.0) # Added scale_spin
        self.trim_in_spin.setValue(0)
        self.trim_out_spin.setValue(0)
        self.pos_x_spin.setValue(0.0)
        self.pos_y_spin.setValue(0.0)
        self.pos_z_spin.setValue(0.0)
        self.rot_x_spin.setValue(0.0)
        self.rot_y_spin.setValue(0.0)
        self.rot_z_spin.setValue(0.0)

        self.align_combo.blockSignals(False)
        self.offset_spin.blockSignals(False)
        self.scale_spin.blockSignals(False) # Added scale_spin
        self.trim_in_spin.blockSignals(False)
        self.trim_out_spin.blockSignals(False)
        self.visible_cb.blockSignals(False)
        self.pos_x_spin.blockSignals(False)
        self.pos_y_spin.blockSignals(False)
        self.pos_z_spin.blockSignals(False)
        self.rot_x_spin.blockSignals(False)
        self.rot_y_spin.blockSignals(False)
        self.rot_z_spin.blockSignals(False)

        self.joints_count_label.setText("")
        self.load_btn.setText("Load FBX/BVH")
        self._set_controls_enabled(False)

    def set_selected_joint(self, joint_name: str):
        """Highlight a joint as selected (from 3D viewer picking)."""
        if not self._loaded:
            return
        idx = self.align_combo.findText(joint_name)
        if idx >= 0:
            # Flash the align combo to indicate the selected joint
            self.align_combo.blockSignals(True)
            self.align_combo.setCurrentIndex(idx)
            self.align_combo.blockSignals(False)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if path.lower().endswith(('.fbx', '.bvh')):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.fbx', '.bvh')):
                self.file_dropped.emit(self.slot, path)
                event.acceptProposedAction()

# ---------------------------------------------------------------------------
# Track Panel (scroll area)
# ---------------------------------------------------------------------------
class TrackPanel(QScrollArea):
    """Scrollable panel containing controls for all 5 tracks."""

    load_requested = Signal(int)
    unload_requested = Signal(int)
    settings_changed = Signal(int)
    joints_changed = Signal(int, set)
    auto_align_requested = Signal(int)
    file_dropped = Signal(int, str)
    rest_pose_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(360)
        self.setMaximumWidth(450)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 16, 6)  # Sensible right margin for scrollbar + gap

        self.track_controls: List[SingleTrackControls] = []
        for i in range(5):
            tc = SingleTrackControls(i)
            tc.load_requested.connect(self.load_requested.emit)
            tc.unload_requested.connect(self.unload_requested.emit)
            tc.settings_changed.connect(self.settings_changed.emit)
            tc.joints_changed.connect(self.joints_changed.emit)
            tc.auto_align_requested.connect(self.auto_align_requested.emit)
            tc.file_dropped.connect(self.file_dropped.emit)
            tc.rest_pose_requested.connect(self.rest_pose_requested.emit)
            self.track_controls.append(tc)
            layout.addWidget(tc)

        layout.addStretch()
        container.setLayout(layout)
        self.setWidget(container)

        # Mutual exclusivity for reference radio buttons
        self._ref_group = QButtonGroup(self)
        for tc in self.track_controls:
            self._ref_group.addButton(tc.ref_radio, tc.slot)

        # Default: Track 1 is reference
        self.track_controls[0].ref_radio.setChecked(True)
