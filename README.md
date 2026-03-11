# MoCap Studio v2

MoCap Studio is an enterprise-grade Motion Capture processing and validation application. It supports loading pure 3-dimensional joint positions via JSON, auto-syncing timelines across differing multi-camera capture environments, and re-exporting native skeletal tracking data back into professional BVH or FBX configurations for software like Blender and MotionBuilder.

## System Requirements

This software strictly targets modern Windows OS environments and relies heavily upon the Autodesk FBX SDK.

### Hardware Prerequisites
- **OS**: Windows 10 or Windows 11 (64-bit)
- **RAM**: 8GB+ System RAM (16GB+ recommended for long timeline sequences)
- **CPU**: Multi-core processor (Threading architecture is heavily utilized for background I/O sequences)
- **Graphics**: OpenGL 3.3 compatible graphics processing unit.

### Software Prerequisites
1. **Python 3.10**
   * The application is compiled against Python 3.10.x. Due to strict Autodesk FBX bindings, Python 3.11 or higher is **not supported** natively by the core FBX libraries.

2. **Conda Environment (Recommended)**
   * We heavily recommend isolating the application via Miniconda or Anaconda:
   ```bash
   conda create -n mocap_studio python=3.10
   conda activate mocap_studio
   ```

3. **Core Dependencies**
   * Review the `requirements.txt` file (or automatically handled on launch).
   * Key packages: `pyside6`, `numpy`, `scipy`, `pyopengl`.

4. **Autodesk FBX SDK for Python (Required for FBX I/O)**
   * To natively decode and encode FBX scene graphs, the proprietary Autodesk FBX Python SDK is explicitly required.
   * **Download**: [Autodesk FBX SDK Developer Page](https://aps.autodesk.com/developer/overview/fbx-sdk)
   * **Installation**: Download the `Python 3.10` native `.whl` (wheel) file from the Autodesk site and run `pip install [filename].whl`.

## Running the Application

### From Source
Ensure all prerequisites are met, then launch the program entry point:
```bash
python -m mocap_studio.main
```

### From Standalone Executable
If you compiled the application software using the `build.bat` scripts, you can run the standalone binary `MoCapStudio_v2.exe` from the `dist` folder. 

**Note for Standalone FBX Users**: Due to how `PyInstaller` unpacks compiled environments, if you are missing the Autodesk SDK at compile-time, the standalone executable will not be capable of rendering or saving FBX pipelines. Ensure `fbx` is importable locally *before* invoking `build.bat`.

## Core Features
- **JSON, FBX, and BVH Extractions**: Load multi-frame motion trackers seamlessly.
- **Cross-Correlation Auto-Sync**: Mathematically align offset skeletons using velocity differential padding.
- **Threaded Viewport & I/O Managers**: Native UI spinners never lock while background hardware spins up massive data processing pools.
- **Python Scripting Engine**: Access live PySide session node-graphs natively inside the live GUI editor window.
