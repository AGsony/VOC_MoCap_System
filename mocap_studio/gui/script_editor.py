"""
Script Editor — Embedded Python scripting with syntax highlighting.

Provides a code editor (QPlainTextEdit), Run button, and an output log.
Scripts are exec()'d with the session object in scope.
"""

from __future__ import annotations

import io
import logging
import sys
import time
import traceback
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QTextEdit,
    QPushButton, QLabel, QFileDialog, QComboBox, QSplitter,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QFont, QColor, QSyntaxHighlighter, QTextCharFormat,
    QFontMetrics,
)

import re

log = logging.getLogger("mocap_studio.gui.script_editor")


class PythonHighlighter(QSyntaxHighlighter):
    """Basic Python syntax highlighter."""

    KEYWORDS = [
        "and", "as", "assert", "async", "await", "break", "class",
        "continue", "def", "del", "elif", "else", "except", "finally",
        "for", "from", "global", "if", "import", "in", "is", "lambda",
        "nonlocal", "not", "or", "pass", "raise", "return", "try",
        "while", "with", "yield", "True", "False", "None",
    ]

    BUILTINS = [
        "print", "range", "len", "int", "float", "str", "list", "dict",
        "set", "tuple", "type", "isinstance", "enumerate", "zip", "map",
        "sorted", "reversed", "abs", "max", "min", "sum", "open",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules = []

        # Keyword format
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor(198, 120, 221))
        kw_fmt.setFontWeight(QFont.Bold)
        for kw in self.KEYWORDS:
            self._rules.append((re.compile(rf"\b{kw}\b"), kw_fmt))

        # Builtin format
        bi_fmt = QTextCharFormat()
        bi_fmt.setForeground(QColor(97, 175, 239))
        for bi in self.BUILTINS:
            self._rules.append((re.compile(rf"\b{bi}\b"), bi_fmt))

        # String format
        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor(152, 195, 121))
        self._rules.append((re.compile(r'\".*?\"'), str_fmt))
        self._rules.append((re.compile(r"\'.*?\'"), str_fmt))

        # Number format
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor(209, 154, 102))
        self._rules.append((re.compile(r"\b\d+\.?\d*\b"), num_fmt))

        # Comment format
        cmt_fmt = QTextCharFormat()
        cmt_fmt.setForeground(QColor(92, 99, 112))
        cmt_fmt.setFontItalic(True)
        self._rules.append((re.compile(r"#.*$"), cmt_fmt))

        # Function/method call
        fn_fmt = QTextCharFormat()
        fn_fmt.setForeground(QColor(97, 175, 239))
        self._rules.append((re.compile(r"\b[A-Za-z_]\w*(?=\s*\()"), fn_fmt))

    def highlightBlock(self, text: str):
        for pattern, fmt in self._rules:
            for match in pattern.finditer(text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, fmt)


class ScriptEditor(QWidget):
    """Embedded Python script editor with execution and output."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._session = None  # set by main window

        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Script Editor"))

        self.script_combo = QComboBox()
        self.script_combo.setMinimumWidth(140)
        self.script_combo.addItem("(untitled)")
        toolbar.addWidget(self.script_combo)

        self.new_btn = QPushButton("New")
        self.new_btn.clicked.connect(self._on_new)
        self.new_btn.setFixedHeight(24)
        toolbar.addWidget(self.new_btn)

        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self._on_open)
        self.open_btn.setFixedHeight(24)
        toolbar.addWidget(self.open_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setFixedHeight(24)
        toolbar.addWidget(self.save_btn)

        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: #2d6a3f;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 2px 12px;
                font-weight: bold;
            }
            QPushButton:hover { background: #3a8a52; }
            QPushButton:pressed { background: #235530; }
        """)
        self.run_btn.clicked.connect(self._on_run)
        self.run_btn.setFixedHeight(24)
        toolbar.addWidget(self.run_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Splitter: editor + output
        splitter = QSplitter(Qt.Vertical)

        # Code editor
        self.editor = QPlainTextEdit()
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.Monospace)
        self.editor.setFont(font)
        self.editor.setTabStopDistance(
            QFontMetrics(font).horizontalAdvance(" ") * 4
        )
        self.editor.setPlaceholderText(
            "# Access track data via 'session'\n"
            "# ref = session.tracks[0]\n"
            "# print(ref.positions.shape)"
        )
        self.editor.setStyleSheet("""
            QPlainTextEdit {
                background: #1e1e22;
                color: #abb2bf;
                border: 1px solid #3a3a42;
                border-radius: 3px;
                selection-background-color: #3e4451;
            }
        """)
        self._highlighter = PythonHighlighter(self.editor.document())
        splitter.addWidget(self.editor)

        # Output log
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 9))
        self.output.setMaximumHeight(120)
        self.output.setStyleSheet("""
            QTextEdit {
                background: #16161a;
                color: #98c379;
                border: 1px solid #3a3a42;
                border-radius: 3px;
            }
        """)
        self.output.setPlaceholderText("Output will appear here...")
        splitter.addWidget(self.output)

        splitter.setSizes([200, 80])
        layout.addWidget(splitter)
        self.setLayout(layout)

        self._current_path: Optional[str] = None

    # ------------------------------------------------------------------
    def set_session(self, session):
        self._session = session

    def load_script_file(self, path: str):
        """Load a script file into the editor."""
        log.info(f"Loading script: {path}")
        with open(path, "r") as f:
            self.editor.setPlainText(f.read())
        self._current_path = path
        import os
        name = os.path.basename(path)
        # Update combo
        idx = self.script_combo.findText(name)
        if idx < 0:
            self.script_combo.addItem(name)
            idx = self.script_combo.count() - 1
        self.script_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    def _on_new(self):
        log.debug("New script")
        self.editor.clear()
        self._current_path = None
        self.script_combo.setCurrentIndex(0)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Script", "",
            "Python Files (*.py);;All Files (*)"
        )
        if path:
            self.load_script_file(path)

    def _on_save(self):
        if self._current_path:
            path = self._current_path
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Script", "",
                "Python Files (*.py);;All Files (*)"
            )
        if path:
            with open(path, "w") as f:
                f.write(self.editor.toPlainText())
            self._current_path = path
            log.info(f"Script saved: {path}")

    def _on_run(self):
        """Execute the script with session in scope."""
        code = self.editor.toPlainText()
        code_lines = code.count('\n') + 1
        script_name = self._current_path or "(untitled)"
        log.info(f"Running script: {script_name} ({code_lines} lines)")
        self.output.clear()

        # Capture stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        buffer = io.StringIO()

        t_start = time.perf_counter()
        success = True

        try:
            sys.stdout = buffer
            sys.stderr = buffer

            import numpy as np_mod
            import scipy as scipy_mod
            from scipy.spatial.transform import Rotation, Slerp

            scope = {
                "session": self._session,
                "np": np_mod,
                "numpy": np_mod,
                "scipy": scipy_mod,
                "Rotation": Rotation,
                "Slerp": Slerp,
                "print": lambda *a, **kw: print(*a, **kw, file=buffer),
            }
            exec(code, scope)

        except Exception:
            success = False
            traceback.print_exc(file=buffer)

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        elapsed = time.perf_counter() - t_start
        output_text = buffer.getvalue()

        if success:
            log.info(f"Script completed successfully in {elapsed:.3f}s")
        else:
            log.error(f"Script failed after {elapsed:.3f}s")
            log.debug(f"Script error output:\n{output_text}")

        if output_text:
            self.output.setPlainText(output_text)
        else:
            self.output.setPlainText("(script completed with no output)")

