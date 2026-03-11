# -*- mode: python ; coding: utf-8 -*-
import os

version_file = 'version.txt'
if not os.path.exists(version_file):
    current_version = '1.0.0'
else:
    with open(version_file, 'r') as f:
        current_version = f.read().strip()
    
    parts = current_version.split('.')
    if len(parts) == 3:
        major, minor, patch = parts
        patch = str(int(patch) + 1)
        current_version = f"{major}.{minor}.{patch}"
    
with open(version_file, 'w') as f:
    f.write(current_version)

exe_name = f'MoCapStudio_v2_{current_version}'

a = Analysis(
    ['mocap_studio\\main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    # CRITICAL: 'fbx' and 'FbxCommon' MUST be explicitly included in hiddenimports.
    # Additionally, ensure the build environment uses Python 3.10 (for FBX SDK support).
    hiddenimports=['PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'numpy', 'OpenGL', 'OpenGL.GL', 'OpenGL.GLU', 'OpenGL.platform.win32', 'OpenGL.arrays.numpymodule', 'fbx', 'FbxCommon'],
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
    name=exe_name,
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
