"""
FBX Extraction — Load FBX via Autodesk FBX Python SDK.

Traverses the scene graph, discovers all skeleton joints and hierarchy,
extracts per-frame global transforms into NumPy arrays.

If the FBX SDK is not installed, provides a graceful fallback with a
clear error message.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Tuple

import numpy as np

from .skeleton import Skeleton
from .track import Track

log = logging.getLogger("mocap_studio.core.fbx_extract")


def _try_import_fbx():
    """Try importing the FBX SDK; return (fbx_module, FbxManager) or (None, None)."""
    try:
        import fbx  # type: ignore
        manager = fbx.FbxManager.Create()
        log.debug("FBX SDK imported successfully.")
        return fbx, manager
    except ImportError:
        log.warning("FBX SDK not available — import failed.")
        return None, None


def load_fbx(filepath: str) -> Track:
    """
    Load an FBX file and return a Track with positions and quaternions.

    Uses the Autodesk FBX Python SDK (must be installed separately).
    """
    t_start = time.perf_counter()
    log.info(f"Loading FBX: {filepath}")

    fbx, manager = _try_import_fbx()
    if fbx is None:
        raise ImportError(
            "Autodesk FBX Python SDK is not installed.\n"
            "Download from: https://aps.autodesk.com/developer/overview/fbx-sdk\n"
            "Install the Python 3.10 wheel, then re-run."
        )

    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    try:
        return _do_load_fbx(fbx, manager, filepath, t_start)
    except Exception:
        manager.Destroy()
        raise


def _do_load_fbx(fbx, manager, filepath: str, t_start: float) -> Track:
    """Internal FBX loading — separated so manager cleanup is guaranteed."""
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(filepath, -1, manager.GetIOSettings()):
        err_msg = importer.GetStatus().GetErrorString()
        log.error(f"FBX Importer initialization failed: {err_msg}")
        raise RuntimeError(f"FBX Importer failed: {err_msg}")

    scene = fbx.FbxScene.Create(manager, "scene")
    importer.Import(scene)
    importer.Destroy()
    log.debug("FBX scene imported successfully.")

    # Normalize to Y-up, right-handed
    target_axis = fbx.FbxAxisSystem.MayaYUp
    target_axis.ConvertScene(scene)
    log.debug("Axis system converted to MayaYUp.")

    # ---- Discover skeleton joints ----
    joint_names = []
    joint_nodes = []
    parent_indices = []
    node_to_index = {}

    def _traverse(node, parent_idx=-1):
        attr = node.GetNodeAttribute()
        if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton:
            idx = len(joint_names)
            joint_names.append(node.GetName())
            joint_nodes.append(node)
            parent_indices.append(parent_idx)
            node_to_index[node.GetUniqueID()] = idx
            parent_idx = idx

        for i in range(node.GetChildCount()):
            _traverse(node.GetChild(i), parent_idx)

    root = scene.GetRootNode()
    _traverse(root)

    if not joint_names:
        log.warning("No skeleton joints found — falling back to all scene nodes.")
        # Fallback: treat all nodes as "joints"
        def _traverse_all(node, parent_idx=-1):
            idx = len(joint_names)
            joint_names.append(node.GetName())
            joint_nodes.append(node)
            parent_indices.append(parent_idx)
            for i in range(node.GetChildCount()):
                _traverse_all(node.GetChild(i), idx)
        _traverse_all(root)

    log.info(f"Discovered {len(joint_names)} joints: {joint_names[:10]}{'...' if len(joint_names) > 10 else ''}")

    skeleton = Skeleton(joint_names=joint_names, parent_indices=parent_indices)
    num_joints = len(joint_names)

    # ---- Get animation time span ----
    anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
    if anim_stack:
        time_span = anim_stack.GetLocalTimeSpan()
        log.debug(f"Animation stack found: {anim_stack.GetName()}")
    else:
        time_span = scene.GetGlobalSettings().GetTimelineDefaultTimeSpan()
        log.warning("No animation stack — using default timeline time span.")

    start_time = time_span.GetStart()
    end_time = time_span.GetStop()
    log.debug(f"Time span: {start_time.GetSecondDouble():.3f}s → {end_time.GetSecondDouble():.3f}s")

    fps = 60.0
    fbx_time = fbx.FbxTime()
    fbx_time.SetSecondDouble(1.0 / fps)
    frame_duration = fbx_time

    # Count frames
    current = fbx.FbxTime()
    current.Set(start_time.Get())
    frame_count = 0
    while current <= end_time:
        frame_count += 1
        current += frame_duration

    log.info(f"Frame count: {frame_count} @ {fps} fps ({frame_count / fps:.2f}s)")

    # ---- Extract per-frame transforms ----
    positions = np.zeros((frame_count, num_joints, 3), dtype=np.float64)
    quaternions = np.zeros((frame_count, num_joints, 4), dtype=np.float64)

    log.info(f"Extracting transforms for {frame_count} frames × {num_joints} joints...")
    extract_start = time.perf_counter()

    current = fbx.FbxTime()
    current.Set(start_time.Get())
    for f in range(frame_count):
        for j, node in enumerate(joint_nodes):
            global_xform = node.EvaluateGlobalTransform(current)
            t = global_xform.GetT()
            q = global_xform.GetQ()
            positions[f, j, 0] = t[0]
            positions[f, j, 1] = t[1]
            positions[f, j, 2] = t[2]
            # FBX quaternion order: (x, y, z, w) → we store (w, x, y, z)
            quaternions[f, j, 0] = q[3]
            quaternions[f, j, 1] = q[0]
            quaternions[f, j, 2] = q[1]
            quaternions[f, j, 3] = q[2]
        current += frame_duration

        # Progress logging every 500 frames
        if (f + 1) % 500 == 0 or f == frame_count - 1:
            elapsed = time.perf_counter() - extract_start
            pct = (f + 1) / frame_count * 100
            log.debug(f"  Frame {f+1}/{frame_count} ({pct:.0f}%) — {elapsed:.1f}s elapsed")

    extract_elapsed = time.perf_counter() - extract_start
    log.info(f"Transform extraction complete in {extract_elapsed:.2f}s")

    # Log position range for sanity check
    pos_min = positions.min(axis=(0, 1))
    pos_max = positions.max(axis=(0, 1))
    log.debug(f"Position range: min={pos_min}, max={pos_max}")

    track = Track(
        name=os.path.splitext(os.path.basename(filepath))[0],
        source_path=filepath,
        fps=fps,
        frame_count=frame_count,
        skeleton=skeleton,
        positions=positions,
        quaternions=quaternions,
    )
    track.auto_setup()

    total_elapsed = time.perf_counter() - t_start
    log.info(
        f"FBX load complete: \"{track.name}\" — "
        f"{frame_count} frames, {num_joints} joints, "
        f"align_joint=\"{track.align_joint}\", "
        f"total time {total_elapsed:.2f}s"
    )

    manager.Destroy()
    return track
