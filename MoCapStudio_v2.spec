# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['mocap_studio\\main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'numpy', 'OpenGL', 'OpenGL.GL', 'OpenGL.GLU', 'OpenGL.platform.win32', 'OpenGL.arrays.numpymodule'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MoCapStudio_v2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
