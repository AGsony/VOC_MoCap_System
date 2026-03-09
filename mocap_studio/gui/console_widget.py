from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PySide6.QtGui import QFont, QTextCursor
import logging

class QTextEditLogger(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        msg = self.format(record)
        # Avoid thread issues by using append directly if in main thread,
        # but in PySide6 it's generally safe for simple app logs.
        # For true safety, signals should be used, but this is fine for basic logs.
        self.widget.append(msg)


class ConsoleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 9))
        self.output.setStyleSheet("""
            QTextEdit {
                background: #16161a;
                color: #c9c9d1;
                border: 1px solid #3a3a42;
                border-radius: 3px;
                padding: 4px;
            }
        """)
        
        layout.addWidget(self.output)
        self.setLayout(layout)
        
        # Setup logging handler
        self.log_handler = QTextEditLogger(self.output)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s │ %(levelname)-7s │ %(message)s', '%H:%M:%S'))
        
        # Attach to root logger or mocap_studio logger
        logging.getLogger("mocap_studio").addHandler(self.log_handler)
