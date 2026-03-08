"""
Joint Visibility Dialog — Per-bone show/hide with search filter.

Opens from the track panel's "Joints..." button.
Returns the set of hidden joint names on accept.
"""

from __future__ import annotations

from typing import List, Set

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLineEdit, QLabel, QDialogButtonBox,
)
from PySide6.QtCore import Qt


class JointVisibilityDialog(QDialog):
    """Dialog for toggling visibility of individual joints."""

    def __init__(self, joint_names: List[str], hidden_joints: Set[str],
                 track_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Joint Visibility — {track_name}" if track_name else "Joint Visibility")
        self.setMinimumSize(300, 420)
        self.resize(320, 500)

        self._joint_names = joint_names
        self._result_hidden: Set[str] = set(hidden_joints)

        layout = QVBoxLayout()
        layout.setSpacing(6)

        # Header info
        info_label = QLabel(f"{len(joint_names)} joints  •  "
                           f"{len(joint_names) - len(hidden_joints)} visible")
        info_label.setStyleSheet("color: #aab0bc; font-size: 11px;")
        layout.addWidget(info_label)
        self._info_label = info_label

        # Search filter
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter joints...")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        # Show All / Hide All buttons
        btn_row = QHBoxLayout()
        show_all_btn = QPushButton("Show All")
        show_all_btn.clicked.connect(self._show_all)
        show_all_btn.setFixedHeight(26)
        btn_row.addWidget(show_all_btn)

        hide_all_btn = QPushButton("Hide All")
        hide_all_btn.clicked.connect(self._hide_all)
        hide_all_btn.setFixedHeight(26)
        btn_row.addWidget(hide_all_btn)
        layout.addLayout(btn_row)

        # Joint list
        self._list = QListWidget()
        self._list.setStyleSheet("""
            QListWidget {
                background: #16161a;
                border: 1px solid #3a3a42;
                border-radius: 3px;
            }
            QListWidget::item {
                padding: 3px 4px;
                border-bottom: 1px solid #2a2a32;
            }
            QListWidget::item:hover {
                background: #2a2a35;
            }
        """)

        for name in joint_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if name in hidden_joints:
                item.setCheckState(Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)
            self._list.addItem(item)

        self._list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list, 1)

        # OK / Cancel
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    def get_hidden_joints(self) -> Set[str]:
        """Return the set of joint names that are unchecked (hidden)."""
        hidden = set()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.Unchecked:
                hidden.add(item.text())
        return hidden

    # ------------------------------------------------------------------
    def _update_info(self):
        hidden = self.get_hidden_joints()
        total = len(self._joint_names)
        visible = total - len(hidden)
        self._info_label.setText(f"{total} joints  •  {visible} visible")

    # ------------------------------------------------------------------
    def _on_item_changed(self, item):
        self._update_info()

    # ------------------------------------------------------------------
    def _apply_filter(self, text: str):
        text = text.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            matches = text in item.text().lower()
            item.setHidden(not matches)

    # ------------------------------------------------------------------
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
