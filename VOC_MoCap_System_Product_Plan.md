# MoCap Align & Compare — Development Plan v2

**Date:** March 8, 2026  
**Project:** Tooling for MoCap PreVis Error Metric Study (v0.1)  
**Constraint:** ~3 hours initial dev time (Phase 1), expandable (Phase 2)

---

## 0. What Changed From v1

Phase 1 priority has shifted from "prove the metrics math" to "build a usable alignment tool." This is the right call — alignment quality is the gating problem. If the data isn't properly aligned, no metric has meaning.

| Removed from Phase 1 | Added to Phase 1 | Moved to Phase 2 |
|---|---|---|
| MPJPE / MPJRE computation | OpenGL 3D viewer at 60fps | All 4 metrics (MPJPE, MPJVE, MPJRE, MPJSE) |
| CSV export | Python scripting engine (create, load, save, run scripts) | Per-region rollups |
| Metrics panel in GUI | Session save/load (JSON project file) | CSV/JSON metrics export |
| — | User-configurable alignment joint (not hardcoded to hips) | — |

**Removed from Phase 2 entirely** (can be added later as Phase 3 if needed):
Weighted whole-body aggregation, p90 thresholds, Accept/Reject label ingestion, requirement matrix output matching study §8. These are study-pipeline features, not tool features. The tool's job is alignment + visualization + scripting. The study math can live in scripts run through the tool's scripting engine.

---

## 1. Research Summary: Useful Libraries & Repos

### 1.1 Critical Dependencies

| Library | Purpose | Why this one |
|---|---|---|
| **Autodesk FBX Python SDK** | FBX parsing — raw joint data extraction | Only lossless per-joint transforms from FBX without a DCC tool |
| **NumPy** | All array math | Standard, fast, vectorized |
| **SciPy** (`scipy.spatial.transform`) | `Slerp` for quaternion interpolation, `Rotation` for conversions | Production-grade; handles piecewise interpolation |
| **PySide6** | GUI framework (timeline, controls, layout) | Qt is the VFX/games standard; PySide6 is LGPL |
| **PyOpenGL** | 3D skeleton rendering in Qt OpenGLWidget | Lightweight, embeds in Qt, 60fps capable |
| **PyYAML** | Config files (joint maps, regions) | Simple, human-readable |

### 1.2 Useful GitHub Repos

| Repo | What it offers | How we use it |
|---|---|---|
| **[eherr/vis_utils](https://github.com/eherr/vis_utils)** | Simple OpenGL renderer + scene graph for skeleton animations. PySide2, MIT licensed | Direct architecture reference for our OpenGL viewer. Proven approach: Qt + OpenGL + skeleton lines |
| **[eherr/anim_utils](https://github.com/eherr/anim_utils)** | BVH/FBX reader → SkeletonBuilder → `get_global_position` per frame. Forward kinematics from local rotations | Reference for skeleton data structures; potential direct use for BVH path |
| **[eherr/motion_preprocessing_tool](https://github.com/eherr/motion_preprocessing_tool)** | Full GUI: skeleton viewer, drag-drop FBX/BVH, animation controls, scene list | Closest existing tool. Architecture reference for how to wire Qt + OpenGL + animation state |
| **[matRandall/FBXMotionToolkit](https://github.com/matRandall/FBXMotionToolkit)** | Python FBX SDK wrapper: joint extraction, resample, CSV | Code reference for FBX extraction layer |
| **[TemugeB/Python_BVH_viewer](https://github.com/TemugeB/Python_BVH_viewer)** | Pure Python BVH viewer, frame-by-frame world-space coordinate computation | Simple reference for skeleton rendering math without engine dependencies |
| **[tomochi222/motion-data-analyzer](https://github.com/tomochi222/motion-data-analyzer)** | Multi-sensor comparison of simultaneously recorded data (OptiTrack, Perception Neuron) | Directly relevant — proves multi-system comparison is viable in Python |

### 1.3 Key Technical Insights

**OpenGL skeleton viewer in Qt:**
The DFKI `vis_utils` repo proves this is a solved problem. The approach: `QOpenGLWidget` subclass, simple line-drawing shader, bone pairs as GL_LINES, camera as an arcball model-view matrix. No mesh, no skinning, no textures. Just colored lines between joints. This runs at hundreds of FPS for 5 skeletons (~500 line segments total).

**Alignment joint flexibility:**
Not all mocap skeletons use "Hips" as root. Some use "Root", some have an explicit world-space root node above the skeleton, some have the pelvis as root with hips as children. The tool needs a **user-configurable alignment joint** — default to the first joint named "Hips", "Pelvis", "Root", or "hip" (case-insensitive search), but let the user pick any joint from a dropdown.

**Scripting engine:**
Python is the scripting language (we're already in Python). The simplest approach: an embedded code editor (QPlainTextEdit with syntax highlighting) that `exec()`s user code with the session object in scope. Scripts can access `session.tracks[0].positions`, manipulate data, and call `session.export()`. This gives full flexibility without building every possible feature into the GUI.

**Frame interpolation via SciPy:**
`scipy.spatial.transform.Slerp` takes a set of known rotations at known times and returns a callable that evaluates at any arbitrary time — exactly what we need for resampling. For positions, `numpy.interp` handles per-component linear interpolation. Both are one-liners once the data is in the right shape.

---

## 2. Architecture

```
mocap_studio/
│
├── core/                           # No GUI dependency — pure data
│   ├── fbx_extract.py              # FBX SDK → raw NumPy arrays
│   ├── bvh_extract.py              # BVH parser → raw NumPy arrays (fallback)
│   ├── skeleton.py                 # Joint hierarchy, parent-child pairs, name lookup
│   ├── track.py                    # Single animation track (positions, rotations, metadata)
│   ├── session.py                  # Multi-track session: 5 tracks, alignment state, project I/O
│   └── align.py                    # Frame offset, alignment-joint subtraction, interpolation/resample
│
├── gui/
│   ├── main_window.py              # Top-level layout: viewer + timeline + panels
│   ├── gl_viewer.py                # QOpenGLWidget: skeleton line renderer, arcball camera
│   ├── timeline_widget.py          # QPainter multi-track timeline, scrubber, trim handles
│   ├── track_panel.py              # Per-track: load button, color, visibility, alignment joint selector
│   ├── script_editor.py            # QPlainTextEdit + Run button, syntax highlighting, output log
│   └── styles.py                   # Dark theme stylesheet
│
├── config/
│   ├── joint_regions.yaml          # Joint name → region (core/hands/feet/head)
│   └── joint_aliases.yaml          # Per-system joint name → canonical name
│
├── scripts/                        # Bundled example scripts (user can add more)
│   ├── export_aligned_csv.py       # Export aligned data to CSV
│   ├── compute_mpjpe.py            # Compute MPJPE between reference and test tracks
│   └── compute_mpjre.py            # Compute MPJRE between reference and test tracks
│
└── main.py                         # Entry point
```

### 2.1 Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                     SESSION (5 tracks)                    │
│                                                           │
│  Track 1 ████████████████████████████  (Ref) [Align: Hips]│
│  Track 2     ██████████████████████        [Align: Pelvis]│
│  Track 3   ████████████████████████████    [Align: Root]  │
│  Track 4       ████████████████████        [Align: Hips]  │
│  Track 5    ██████████████████████████     [Align: hip]   │
│                                                           │
│  Per track, user configures:                              │
│    • Alignment joint (dropdown from skeleton joints)      │
│    • Frame offset (drag or numeric)                       │
│    • In/Out trim points                                   │
│    • Visibility toggle                                    │
│                                                           │
│  System provides:                                         │
│    • 3D overlaid view at any frame (OpenGL, 60fps)        │
│    • Resampled data when frame counts differ (SLERP+lerp) │
│    • Full data exposed to Python scripting engine          │
│                                                           │
│  Session saves/loads as JSON:                             │
│    { tracks: [{path, offset, in, out, align_joint}...] }  │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Alignment Joint System

The old plan hardcoded "hips" as the root alignment joint. Here's the new approach:

```
On FBX load:
  1. Parse full skeleton hierarchy
  2. Auto-detect alignment joint candidate:
     - Search joint names (case-insensitive) for: "Hips", "Pelvis", "hip", 
       "Root", "root", "pelvis", "CENTER", "center"
     - If found → set as default alignment joint for this track
     - If not found → set to the skeleton's actual root node, warn user
  3. User can change alignment joint at any time via dropdown
     (populated with all joint names from this track's skeleton)
  
Alignment operation (per frame):
  alignment_pos = track.positions[frame, alignment_joint_index, :]
  aligned_positions = track.positions[frame, :, :] - alignment_pos
  
  This translates the entire skeleton so the chosen joint is at the origin.
  Each track can use a DIFFERENT alignment joint if their skeletons differ.
```

---

## 3. Phase 1 — MVP (3-Hour Build)

### 3.1 Goal

A working desktop app where you can load 5 FBX files, see their skeletons overlaid in a real-time OpenGL viewer, scrub through frames, manually offset tracks for alignment, pick the alignment joint per track, save/load sessions, and run Python scripts against the loaded data.

No metrics computation in the GUI. Metrics are available as example scripts.

### 3.2 Hour-by-Hour Build Plan

#### Hour 1: Core Data Pipeline + OpenGL Viewer Foundation (0:00 – 1:00)

| Time | Task | Deliverable |
|---|---|---|
| 0:00 | Project structure, `conda create`, install deps | Working environment |
| 0:10 | `fbx_extract.py` — Load FBX via SDK, traverse scene graph, discover all skeleton joints and hierarchy | Joint name list + parent map |
| 0:25 | `fbx_extract.py` — Frame extraction loop: for each frame at 1/60s, call `EvaluateGlobalTransform` on every joint → store positions `(F,J,3)` and quaternions `(F,J,4)` in NumPy arrays | Raw data arrays |
| 0:35 | `skeleton.py` — Skeleton class: joint names, parent indices, `get_bone_pairs()` returns list of `(parent_idx, child_idx)` for line rendering | Skeleton object |
| 0:40 | `track.py` — Track dataclass: positions, quaternions, skeleton, metadata (filename, fps, frame count) | Loadable Track |
| 0:45 | `gl_viewer.py` — `QOpenGLWidget` subclass: init shaders (simple vertex color), set up projection/view matrices, draw grid floor | Empty OpenGL viewport in Qt |
| 0:55 | `gl_viewer.py` — Render one skeleton: bone pairs as `GL_LINES`, joint positions as `GL_POINTS`. Hardcode frame 0 of loaded track | Single skeleton visible |

#### Hour 2: Multi-Track + Timeline + Playback (1:00 – 2:00)

| Time | Task | Deliverable |
|---|---|---|
| 1:00 | `gl_viewer.py` — Arcball camera: mouse drag = orbit, scroll = zoom, middle-drag = pan | Navigable 3D view |
| 1:10 | `gl_viewer.py` — Multi-track rendering: 5 color slots (green, blue, yellow, red, purple), draw each loaded track's skeleton at current frame | 5 overlaid skeletons |
| 1:20 | `session.py` — Session class: list of 5 Track slots, current frame, per-track offset/trim/visibility/alignment_joint | Session state management |
| 1:30 | `timeline_widget.py` — QPainter custom widget: horizontal bars per track (colored), global scrubber (vertical line), frame number display. Mouse drag scrubber → emits `frame_changed` signal | Visual timeline |
| 1:40 | `main_window.py` — Layout: OpenGL viewer (top, 70% height), timeline (bottom, 20%), control bar (middle, 10%). Load button per track. Play/Pause/Stop. Frame counter | Running app shell |
| 1:50 | Wire it all: Load button → `fbx_extract` → Track → Session → viewer redraws on `frame_changed`. Playback via QTimer at 60fps triggering frame advance | End-to-end: load, view, play, scrub |

#### Hour 3: Alignment + Scripting + Session I/O (2:00 – 3:00)

| Time | Task | Deliverable |
|---|---|---|
| 2:00 | `track_panel.py` — Per-track controls: alignment joint dropdown (populated from skeleton), frame offset spinbox, in/out frame spinboxes, visibility checkbox | Per-track control panel |
| 2:05 | `align.py` — Alignment joint subtraction: given joint index, subtract that joint's position from all joints per frame. Applied in viewer render path (non-destructive) | Skeletons align at chosen joint |
| 2:15 | `align.py` — Resample function: given track at N frames, produce resampled track at M frames. Positions via `numpy.interp`, rotations via `scipy.spatial.transform.Slerp` | Frame interpolation working |
| 2:25 | `session.py` — `save_session(path)`: serialize to JSON (track file paths, offsets, trim points, alignment joints, reference track index). `load_session(path)`: deserialize, re-extract FBX data, restore state | Session persistence |
| 2:35 | `script_editor.py` — QPlainTextEdit with monospace font, basic Python syntax highlighting (QSyntaxHighlighter), "Run" button, output log (QTextEdit below). `exec()` with `{"session": session, "np": numpy, "scipy": scipy}` in scope | Working script editor |
| 2:45 | Bundled scripts: `scripts/export_aligned_csv.py`, `scripts/compute_mpjpe.py` — dropdown to select + load into editor, or "New Script" / "Open Script" / "Save Script" | Example scripts bundled |
| 2:55 | Test full flow: load 5 FBX, align, save session, reload, run MPJPE script | Integration verified |

### 3.3 Phase 1 Feature → Requirement Mapping

| Study Requirement | Phase 1 Feature | Notes |
|---|---|---|
| §4 Frame alignment 60 FPS | `fbx_extract.py` evaluates at 1/60s intervals | Exact frame sampling via FBX SDK |
| §4 Reference trajectory | Track 1 designated as reference (configurable) | Toggle in track panel |
| §1.3 "Aligned at root" | `align.py` with user-chosen alignment joint | Not hardcoded to hips — user picks per track |
| "5 tracks simultaneously" | `session.py` + `gl_viewer.py` multi-track | Color-coded, toggleable visibility |
| "Manual alignment" | Timeline offset + trim per track | Integer frame offset in Phase 1 |
| "Frame interpolation" | `align.py` resample (SLERP + lerp) | For non-matching frame counts |
| "Visualizer for verification" | OpenGL 60fps skeleton viewer | Arcball camera, grid floor, 5 colors |
| "No data distortion" | FBX SDK `EvaluateGlobalTransform`, no DCC import | Core design principle |
| "Scripting" | Embedded Python editor with session API | Metrics, export, custom analysis |
| "Session management" | JSON project save/load | Reproducible alignment state |

### 3.4 What Phase 1 Does NOT Include

- Any metrics in the GUI (available via scripting)
- Sub-frame offset (integer only)
- Draggable trim handles in timeline (numeric input only)
- Auto-alignment / cross-correlation
- BVH fallback path
- Joint labels / hover tooltips
- Error heatmap visualization
- Undo/redo
- Drag-and-drop file loading
- Polished dark theme (basic styling only)

### 3.5 Phase 1 GUI Wireframe

```
┌────────────────────────────────────────────────────────────────┐
│  MoCap Align & Compare                          [─]  [□]  [×] │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────┐  ┌────────────────┐  │
│  │                                      │  │ Track Controls │  │
│  │         OPENGL 3D VIEWER             │  │                │  │
│  │     (skeleton lines, grid floor,     │  │ T1 [Load] ✓    │  │
│  │      arcball camera)                 │  │  Align: [Hips▾]│  │
│  │                                      │  │  Offset: [0  ] │  │
│  │  🟢 T1  🔵 T2  🟡 T3               │  │  In/Out: 0-1800│  │
│  │  🔴 T4  🟣 T5                       │  │                │  │
│  │                                      │  │ T2 [Load] ✓    │  │
│  │                                      │  │  Align:[Pelvis▾│  │
│  │                                      │  │  Offset: [12 ] │  │
│  │                                      │  │  In/Out: 0-1790│  │
│  │                                      │  │                │  │
│  │                                      │  │ T3 [Load] ☐    │  │
│  │                                      │  │ T4 [Load] ☐    │  │
│  │                                      │  │ T5 [Load] ☐    │  │
│  └──────────────────────────────────────┘  └────────────────┘  │
│                                                                │
│  Frame: [  247  ] / 1800    [▶] [⏸] [⏹]    Speed: [1x ▾]     │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ T1 ████████████████████████████████████                  │  │
│  │ T2 ·····████████████████████████████·····                │  │
│  │ T3 ···██████████████████████████████████··               │  │
│  │ T4 ·········██████████████████████·······                │  │
│  │ T5 ····████████████████████████████████···               │  │
│  │    ▼ scrubber                                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Script Editor                    [New][Open][Save][▶ Run]│  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ # Access all track data via 'session'                │ │  │
│  │ │ ref = session.tracks[0]                              │ │  │
│  │ │ test = session.tracks[1]                             │ │  │
│  │ │ # Positions shape: (frames, joints, 3)               │ │  │
│  │ │ diff = ref.aligned_positions - test.aligned_positions│ │  │
│  │ │ mpjpe = np.mean(np.linalg.norm(diff, axis=-1))      │ │  │
│  │ │ print(f"MPJPE: {mpjpe:.2f} cm")                     │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  │ Output:                                                  │  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ MPJPE: 2.47 cm                                      │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  [Status: 5 tracks loaded | Frame 247/1800 | 60.0 fps]        │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 2 — Full Tool (~1-2 Weeks)

Phase 2 focuses on making the alignment workflow polished and production-ready. Metrics are available through scripting — the GUI doesn't need a dedicated metrics panel.

### 4.1 Timeline Polish
- Draggable in/out trim handles (not just numeric input)
- Draggable track offset (horizontal drag to shift in time)
- Sub-frame offset support (float, not just integer)
- Snap-to-frame option
- Timeline zoom in/out (mouse wheel)
- Per-track frame number display at scrubber position
- Keyboard: left/right = ±1 frame, shift+left/right = ±10, space = play/pause, home/end = first/last frame

### 4.2 Viewer Polish
- Joint labels on hover (tooltip showing joint name + position)
- Per-joint error heatmap mode (color joints by distance from reference — driven by script output)
- Grid floor scale labels
- Screenshot export (current frame → PNG)
- Optional: skinned mesh preview if FBX contains mesh data

### 4.3 Alignment Refinement
- Auto-alignment hint: cross-correlation of alignment-joint velocity curves → suggests frame offset
- Per-track time stretch/compress with interpolation (for clock drift > 1 frame over a long take)
- Multi-joint alignment: align on centroid of N joints, not just one
- Visual feedback: ghost skeleton at ±1 frame (onion skinning)

### 4.4 Scripting Enhancements
- Autocomplete / intellisense for `session.*` API
- Script output can push data back to viewer (e.g., color joints by error)
- Script library browser (bundled + user scripts)
- Script can register custom timeline overlays (e.g., per-frame error graph)
- `session.export_fbx(track_index, path)` — export aligned/trimmed data back to FBX

### 4.5 Session & Project Management
- Recent projects list
- Auto-save / recovery
- Session diff: compare two sessions (different alignment settings for the same data)
- Batch session creation from folder structure

### 4.6 UI Polish
- Dark theme (standard for VFX tools)
- Track mute/solo (M/S buttons)
- Right-click context menus throughout
- Resizable panels (splitters)
- Status bar: FPS, frame count, joint count, memory usage
- Error/warning log panel (collapsible)
- Drag-and-drop file loading onto track slots

---

## 5. Technical Deep Dive: Frame Alignment & Interpolation

### 5.1 The Problem

Five mocap systems capture the same performance simultaneously, but:
- No timecode sync between systems
- Different internal clocks → clips start/end at different times
- Some systems may capture at different effective rates or drop frames
- Clips are different lengths even for the "same" motion
- Root/alignment joint may be named differently or be at a different position in the hierarchy

### 5.2 The Solution: Manual Align + Interpolated Resample

**Step 1 — Load all 5 tracks, display in timeline.**
Each track shows its full length. User can see they're different lengths.

**Step 2 — User picks alignment joint per track.**
Each skeleton may name joints differently. User picks from dropdown per track. The viewer immediately shows skeletons aligned at those joints.

**Step 3 — User aligns visually.**
User scrubs to a distinctive motion moment (foot strike, hand clap, T-pose start) and adjusts each track's frame offset so that moment lines up across all tracks. The overlaid skeletons in the viewer confirm alignment.

**Step 4 — User sets trim (in/out points).**
User marks the common segment where ALL 5 tracks have valid data.

**Step 5 — System resamples to common frame count.**
After trimming, each track may have slightly different frame counts. The system resamples non-reference tracks to match reference:

```
Reference: 1800 frames (Track 1)
Track 2:   1803 frames → resample to 1800

For each target frame t in [0, 1799]:
  source_t = t * (1803 / 1800)      # fractional source frame index
  lo = floor(source_t)               # flanking frame below
  hi = ceil(source_t)                # flanking frame above  
  alpha = source_t - lo              # interpolation weight
  
  Positions:  lerp(pos[lo], pos[hi], alpha)           # numpy.interp per component
  Rotations:  slerp(quat[lo], quat[hi], alpha)        # scipy Slerp
```

### 5.3 Interpolation Quality Note

At 60fps with sub-frame interpolation (typically < 0.5 frame drift over a 30-second take), the introduced error is on the order of microseconds of motion — far below any mocap system's measurement precision.

---

## 6. Scripting API Design

The scripting engine exposes the session data to user Python code. This is how metrics and export happen without bloating the GUI.

### 6.1 Objects Available in Script Scope

```python
# Automatically available in every script:
session         # Session object — all tracks, alignment state
np              # numpy
scipy           # scipy
Rotation        # scipy.spatial.transform.Rotation
Slerp           # scipy.spatial.transform.Slerp
print           # redirected to script output panel

# Session API:
session.tracks              # list of Track objects (up to 5)
session.reference_index     # int, which track is reference
session.current_frame       # int, current scrubber position

# Track API:
track = session.tracks[0]
track.name                  # str, filename stem
track.fps                   # float, frame rate
track.frame_count           # int, total frames
track.positions             # np.ndarray (F, J, 3) — raw world positions
track.quaternions           # np.ndarray (F, J, 4) — raw world rotations (w,x,y,z)
track.aligned_positions     # np.ndarray (F, J, 3) — after alignment joint subtraction
track.skeleton              # Skeleton object
track.skeleton.joint_names  # list[str]
track.skeleton.parent_map   # dict[int, int] — child_idx → parent_idx
track.offset                # int, frame offset
track.trim_in               # int, in-point frame
track.trim_out              # int, out-point frame
track.align_joint           # str, name of alignment joint
track.align_joint_index     # int, index of alignment joint

# Resample utility:
from core.align import resample_track
resampled = resample_track(track, target_frame_count=1800)
```

### 6.2 Example Bundled Scripts

**`scripts/compute_mpjpe.py`**
```python
# Compute Mean Per Joint Position Error vs reference
ref = session.tracks[session.reference_index]
for i, track in enumerate(session.tracks):
    if i == session.reference_index or track is None:
        continue
    diff = ref.aligned_positions[:track.frame_count] - track.aligned_positions
    per_joint = np.linalg.norm(diff, axis=-1)           # (F, J)
    mpjpe = np.mean(per_joint)
    print(f"Track {i} ({track.name}): MPJPE = {mpjpe:.3f} cm")
```

**`scripts/export_aligned_csv.py`**
```python
# Export aligned position data for all tracks to CSV
import csv, os
out_dir = os.path.expanduser("~/mocap_export")
os.makedirs(out_dir, exist_ok=True)
for i, track in enumerate(session.tracks):
    if track is None:
        continue
    path = os.path.join(out_dir, f"track_{i}_{track.name}_aligned.csv")
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["frame"] + [f"{j}_x,{j}_y,{j}_z" for j in track.skeleton.joint_names])
        for frame in range(track.frame_count):
            row = [frame]
            for j in range(len(track.skeleton.joint_names)):
                p = track.aligned_positions[frame, j]
                row.extend([f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
            w.writerow(row)
    print(f"Exported: {path}")
```

---

## 7. Dependencies & Setup

### 7.1 Required (Phase 1)

| Package | Version | Install | Purpose |
|---|---|---|---|
| Python | 3.10 (strict) | conda | Required by FBX SDK |
| Autodesk FBX Python SDK | 2020.3+ | Manual from Autodesk | FBX parsing |
| NumPy | ≥1.24 | pip | Array math |
| SciPy | ≥1.10 | pip | Slerp, Rotation |
| PySide6 | ≥6.5 | pip | GUI framework |
| PyOpenGL | ≥3.1 | pip | 3D rendering |
| PyYAML | ≥6.0 | pip | Config files |

### 7.2 Environment Setup

```bash
conda create -n mocap_studio python=3.10
conda activate mocap_studio
pip install numpy scipy pyside6 pyopengl pyyaml

# FBX SDK: download from https://aps.autodesk.com/developer/overview/fbx-sdk
# Install the FBX Python SDK .whl for Python 3.10
pip install <path-to-fbx-sdk-wheel>
```

---

## 8. Risk Register

| Risk | Impact | Probability | Mitigation |
|---|---|---|---|
| FBX SDK + PySide6 + PyOpenGL version conflicts | Blocks app | Low | All support Python 3.10; conda isolates env |
| FBX files have different axis systems (Y-up vs Z-up) | Skeletons visually wrong | Medium | `FbxAxisSystem.ConvertScene()` normalizes on load |
| Joint naming differs across 5 systems | Can't align | High | Auto-detect + user dropdown per track |
| Frame rate not exactly 60fps | Drift over long takes | Medium | Resample on load; warn if source fps ≠ 60 |
| OpenGL complexity in 3-hour scope | Viewer incomplete | Medium | Fallback: matplotlib for Phase 1, OpenGL as first Phase 2 task. But attempting OpenGL first since skeleton rendering is simple geometry |
| `exec()` scripting is a security risk | Not a concern for internal studio tool | N/A | Acceptable for single-user studio use |
| 3 hours produces rough tool | Needs polish | Certain | Architecture is clean; Phase 2 is polish, not redesign |

---

## 9. Definition of Done

### Phase 1 (3 hours)

- [ ] Can load up to 5 FBX files simultaneously
- [ ] OpenGL viewer shows all loaded skeletons as colored lines, 60fps
- [ ] Arcball camera (orbit, zoom, pan)
- [ ] Grid floor visible
- [ ] Scrubber moves through frames, viewer updates in real-time
- [ ] Play/Pause at 60fps
- [ ] Per-track alignment joint selectable from dropdown
- [ ] Per-track frame offset adjustable (numeric)
- [ ] Per-track trim in/out adjustable (numeric)
- [ ] Alignment subtraction applied in viewer (non-destructive)
- [ ] Session save to JSON
- [ ] Session load from JSON (re-extracts FBX data, restores state)
- [ ] Script editor: create, open, save, run Python scripts
- [ ] Script has access to `session` object with all track data
- [ ] Bundled example scripts: MPJPE computation, CSV export

### Phase 2 (1-2 weeks)

- [ ] Draggable timeline trim handles + offset drag
- [ ] Sub-frame offset
- [ ] Keyboard shortcuts (arrows, space, home/end)
- [ ] Joint labels on hover
- [ ] Screenshot export
- [ ] Auto-alignment hint (cross-correlation)
- [ ] Script autocomplete for session API
- [ ] Script-driven joint coloring in viewer
- [ ] Dark theme
- [ ] Drag-and-drop file loading
- [ ] Track mute/solo
- [ ] Resizable panel layout
- [ ] FBX export of aligned/trimmed data (via scripting)

---

## 10. Decision Points

After the 3-hour Phase 1 build:

1. **Does the FBX SDK extract data that matches the source software?** Load one file, check joint positions against native app output at the same frame.

2. **Can you visually align the 5 clips?** Load all 5, overlay, scrub to a known sync point. If they line up visually, the tool works.

3. **Is OpenGL rendering fast enough for 5 skeletons at 60fps?** Should be trivially fast (~500 line segments), but verify on target hardware.

4. **Does the scripting engine give enough flexibility?** Run the MPJPE script. If it produces sensible numbers, the "metrics via scripting" approach is validated — and you never need to build metrics into the GUI.

If all four are yes, Phase 2 is pure polish and feature expansion on a proven foundation.
