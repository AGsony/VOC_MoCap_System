"""
MoCap Align & Compare — Entry Point

Launch the application:
    python -m mocap_studio.main
"""

import sys
import os
import logging
import logging.handlers
from datetime import datetime

# Guarantee safe PyOpenGL platform bindings at runtime when frozen using PyInstaller
os.environ['PYOPENGL_PLATFORM'] = 'win32'
import OpenGL
import OpenGL.platform.win32
import OpenGL.arrays.numpymodule

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QSharedMemory
from PySide6.QtGui import QFont

from mocap_studio.gui.main_window import MainWindow


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
MAX_LOG_FILES = 20
MAX_LOG_AGE_DAYS = 7


def _cleanup_old_logs():
    """Remove log files older than MAX_LOG_AGE_DAYS, keeping at most MAX_LOG_FILES."""
    if not os.path.isdir(LOG_DIR):
        return
    import glob
    import time as _time

    log_files = sorted(
        glob.glob(os.path.join(LOG_DIR, "mocap_studio_*.log")),
        key=os.path.getmtime,
        reverse=True,
    )

    now = _time.time()
    cutoff = now - MAX_LOG_AGE_DAYS * 86400
    removed = 0

    for i, path in enumerate(log_files):
        # Keep the newest MAX_LOG_FILES regardless of age
        if i >= MAX_LOG_FILES or os.path.getmtime(path) < cutoff:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass

    return removed


def setup_logging():
    """Configure application-wide logging to both file and console."""
    os.makedirs(LOG_DIR, exist_ok=True)

    # Clean up old log files before creating a new one
    removed = _cleanup_old_logs()

    log_filename = os.path.join(
        LOG_DIR, f"mocap_studio_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    root_logger = logging.getLogger("mocap_studio")
    root_logger.setLevel(logging.DEBUG)

    # File handler — detailed (DEBUG level)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d │ %(levelname)-7s │ %(name)-30s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    # Console handler — summary (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    root_logger.info("=" * 60)
    root_logger.info("MoCap Align & Compare v0.1.0 starting")
    root_logger.info(f"Log file: {log_filename}")
    root_logger.info(f"Python: {sys.version}")
    root_logger.info(f"Platform: {sys.platform}")
    if removed:
        root_logger.info(f"Cleaned up {removed} old log file(s)")
    root_logger.info("=" * 60)

    return root_logger



# ---------------------------------------------------------------------------
# Single-instance guard
# ---------------------------------------------------------------------------
SHARED_MEMORY_KEY = "MoCapAlignCompare_SingleInstance_v1"


def check_single_instance(app: QApplication) -> QSharedMemory:
    """
    Ensure only one instance of the app is running.

    Uses Qt QSharedMemory as a cross-platform named lock.
    Returns the QSharedMemory object (must stay alive for the app lifetime).
    Returns None and shows a message if another instance is already running.
    """
    logger = logging.getLogger("mocap_studio.main")

    shared_mem = QSharedMemory(SHARED_MEMORY_KEY)

    # On Unix, stale shared memory can persist after a crash.
    # Attempt to attach and immediately detach to clean up.
    if shared_mem.attach():
        shared_mem.detach()

    if not shared_mem.create(1):
        logger.warning("Another instance is already running — exiting.")
        QMessageBox.warning(
            None,
            "MoCap Align & Compare",
            "Another instance of MoCap Align && Compare is already running.\n\n"
            "Only one instance can run at a time.",
        )
        return None

    logger.info("Single-instance lock acquired.")
    return shared_mem


def check_system_requirements(app: QApplication, log: logging.Logger) -> bool:
    """
    Verify Python 3.10 and the FBX SDK exist. Show a popup and return False if they do not.
    """
    if sys.version_info.major != 3 or sys.version_info.minor != 10:
        log.warning(f"Unsupported Python version: {sys.version}")
        QMessageBox.warning(
            None,
            "Unsupported Python Version",
            f"MoCap Studio requires Python 3.10.x to interface with the Autodesk FBX SDK natively.\n\n"
            f"You are currently running: Python {sys.version.split()[0]}.\n"
            f"Please run this software in a Python 3.10 Conda environment.",
        )
        return False
        
    try:
        import fbx
    except ImportError:
        log.warning("FBX SDK not found during startup check.")
        QMessageBox.critical(
            None,
            "Missing Dependencies: Autodesk FBX SDK",
            "The Autodesk FBX Python SDK was not found on your system. "
            "MoCap Studio cannot launch without it.\n\n"
            "1. Download the Python 3.10 Windows Wheel (.whl) from the Autodesk FBX Developer site.\n"
            "2. Install it in your current environment using:\n\n"
            "     pip install <filename>.whl\n\n"
            "Application will now exit.",
        )
        return False
        
    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log = setup_logging()

    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("MoCap Align & Compare")
    app.setOrganizationName("VOC MoCap")
    app.setApplicationVersion("0.1.0")

    # --- System Requirements check ---
    if not check_system_requirements(app, log):
        sys.exit(1)

    # --- Single-instance check ---
    shared_mem = check_single_instance(app)
    if shared_mem is None:
        log.info("Exiting (duplicate instance).")
        sys.exit(1)

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    log.info("Creating main window...")
    window = MainWindow()
    window.show()
    log.info("Main window shown — entering event loop.")

    exit_code = app.exec()
    log.info(f"Application exiting with code {exit_code}.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
