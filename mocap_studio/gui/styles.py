"""
Dark theme stylesheet for MoCap Align & Compare.

VFX-standard dark color scheme.
"""

DARK_STYLESHEET = """
/* === Global === */
QWidget {
    background-color: #1e1e22;
    color: #d4d4dc;
    font-family: "Segoe UI", "Roboto", sans-serif;
    font-size: 12px;
}

QMainWindow {
    background-color: #1e1e22;
}

/* === Buttons === */
QPushButton {
    background-color: #35353d;
    color: #d4d4dc;
    border: 1px solid #4a4a55;
    border-radius: 4px;
    padding: 4px 12px;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #45455a;
    border-color: #6a6a7a;
}
QPushButton:pressed {
    background-color: #2a2a35;
}
QPushButton:disabled {
    background-color: #28282e;
    color: #555;
    border-color: #333;
}

/* === ComboBox === */
QComboBox {
    background-color: #2a2a32;
    color: #d4d4dc;
    border: 1px solid #4a4a55;
    border-radius: 3px;
    padding: 3px 8px;
    min-height: 20px;
}
QComboBox:hover {
    border-color: #6a6a7a;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #2a2a32;
    color: #d4d4dc;
    border: 1px solid #4a4a55;
    selection-background-color: #3a5a8a;
}

/* === SpinBox === */
QSpinBox {
    background-color: #2a2a32;
    color: #d4d4dc;
    border: 1px solid #4a4a55;
    border-radius: 3px;
    padding: 2px 6px;
    min-height: 20px;
}
QSpinBox:hover {
    border-color: #6a6a7a;
}
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #35353d;
    border: none;
    width: 16px;
}

/* === CheckBox / RadioButton === */
QCheckBox, QRadioButton {
    spacing: 5px;
    color: #d4d4dc;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #555;
    border-radius: 3px;
    background: #2a2a32;
}
QRadioButton::indicator {
    border-radius: 7px;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: #4a8af5;
    border-color: #4a8af5;
}

/* === Labels === */
QLabel {
    color: #aab0bc;
    background: transparent;
}

/* === ScrollArea === */
QScrollArea {
    border: none;
    background: #1e1e22;
}
QScrollBar:vertical {
    background: #1e1e22;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #45455a;
    min-height: 30px;
    border-radius: 4px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* === GroupBox === */
QGroupBox {
    border: 1px solid #3a3a42;
    border-radius: 5px;
    margin-top: 8px;
    padding-top: 16px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #aab0bc;
}

/* === Splitter === */
QSplitter::handle {
    background: #3a3a42;
    height: 3px;
}
QSplitter::handle:hover {
    background: #5a5a6a;
}

/* === StatusBar === */
QStatusBar {
    background: #16161a;
    color: #888;
    border-top: 1px solid #2a2a32;
    font-size: 11px;
}

/* === MenuBar === */
QMenuBar {
    background: #1e1e22;
    color: #d4d4dc;
    border-bottom: 1px solid #2a2a32;
}
QMenuBar::item:selected {
    background: #35353d;
}
QMenu {
    background: #2a2a32;
    color: #d4d4dc;
    border: 1px solid #4a4a55;
}
QMenu::item:selected {
    background: #3a5a8a;
}
"""
