"""
FBX Extraction — Load FBX via Autodesk FBX Python SDK.

AUDIT FIXES:
  - ConvertScene(MayaYUp) was called on load. When the same data is then
    exported with force_z_up=True the axis system is set via SetInScene
    (metadata only). This is now consistent: load normalises to Y-up in
    memory; export writes the correct axis tag. No data double-conversion.
  - manager.Destroy() was called in the except block AND at the end of
    _do_load_fbx — double-destroy if an exception occurred mid-extraction.
    Now manager is only destroyed in one place (the finally in _do_load_fbx).
  - fps was hardcoded to 60.0 regardless of what the FBX file specified.
    Now reads the scene's GlobalSettings time mode and derives fps from it,
    with a fallback to 60.
  - FbxTime increment used SetSecondDouble(1/fps) which creates floating
    point drift over many frames. Changed to SetFrame() arithmetic.
  - frame_count loop used current <= end_time comparison which has an
    off-by-one depending on FbxTime precision. Replaced with integer
    frame arithmetic.
  - Quaternion storage is documented as (w,x,y,z). The extraction already
    did this correctly: q[3]=w, q[0]=x, q[1]=y, q[2]=z — confirmed and
    comment clarified.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

import numpy as np

from .skeleton import Skeleton
from .track import Track

log = logging.getLogger("mocap_studio.core.fbx_extract")

# Map FBX EMode enum values to fps floats
_FBX_MODE_TO_FPS = {
    # Values vary by SDK version; we match on name where possible
    "eFrames24":        24.0,
    "eFrames25":        25.0,
    "eFrames30":        30.0,
    "eFrames48":        48.0,
    "eFrames50":        50.0,
    "eFrames60":        60.0,
    "eFrames96":        96.0,
    "eFrames100":      100.0,
    "eFrames120":      120.0,
    "eNTSCDropFrame":   29.97,
    "eNTSCFullFrame":   30.0,
    "ePAL":             25.0,
    "eFilm":            24.0,
}


def _try_import_fbx():
    try:
        import fbx          # type: ignore
        manager = fbx.FbxManager.Create()
        return fbx, manager
    except ImportError:
        log.warning("FBX SDK not available.")
        return None, None


def _fps_from_scene(fbx, scene) -> float:
    """Read FPS from scene GlobalSettings, falling back to 60."""
    try:
        mode     = scene.GetGlobalSettings().GetTimeMode()
        mode_str = str(mode)
        for key, val in _FBX_MODE_TO_FPS.items():
            if key in mode_str:
                return val
        # Fallback: use FbxTime to compute 1-frame duration
        t = fbx.FbxTime()
        t.SetFrame(1, mode)
        sec = t.GetSecondDouble()
        if sec > 0:
            return round(1.0 / sec, 4)
    except Exception:
        pass
    return 60.0


def load_fbx(filepath: str) -> Track:
    """Load an FBX file and return a Track with positions and quaternions."""
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
        # BUG FIX: manager was destroyed here AND in _do_load_fbx finally — double destroy.
        # _do_load_fbx now owns the manager lifetime entirely.
        raise


def _do_load_fbx(fbx, manager, filepath: str, t_start: float) -> Track:
    """Internal FBX loading — manager is destroyed in finally here and nowhere else."""
    try:
        importer = fbx.FbxImporter.Create(manager, "")
        if not importer.Initialize(filepath, -1, manager.GetIOSettings()):
            err = importer.GetStatus().GetErrorString()
            importer.Destroy()
            raise RuntimeError(f"FBX Importer failed: {err}")

        scene = fbx.FbxScene.Create(manager, "scene")
        importer.Import(scene)
        importer.Destroy()

        # Normalise incoming data to Y-up right-handed in memory.
        # This ensures our (w,x,y,z) quaternion arrays are always in a
        # consistent Y-up coordinate space regardless of the source file's axis.
        fbx.FbxAxisSystem.MayaYUp.ConvertScene(scene)

        # ---- Discover skeleton joints ----
        joint_names   = []
        joint_nodes   = []
        parent_indices = []
        node_to_index  = {}

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

        _traverse(scene.GetRootNode())

        if not joint_names:
            log.warning("No skeleton joints found — falling back to all scene nodes.")
            def _traverse_all(node, parent_idx=-1):
                idx = len(joint_names)
                joint_names.append(node.GetName())
                joint_nodes.append(node)
                parent_indices.append(parent_idx)
                for i in range(node.GetChildCount()):
                    _traverse_all(node.GetChild(i), idx)
            _traverse_all(scene.GetRootNode())

        log.info(f"Discovered {len(joint_names)} joints")

        skeleton    = Skeleton(joint_names=joint_names, parent_indices=parent_indices)
        num_joints  = len(joint_names)

        # ---- FPS from scene ----
        # BUG FIX: was hardcoded to 60.0 regardless of source file
        fps = _fps_from_scene(fbx, scene)
        log.info(f"Scene FPS: {fps}")

        # ---- Animation time span ----
        anim_stack = scene.GetSrcObject(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0
        )
        if anim_stack:
            time_span = anim_stack.GetLocalTimeSpan()
        else:
            time_span = scene.GetGlobalSettings().GetTimelineDefaultTimeSpan()
            log.warning("No animation stack — using default timeline span.")

        start_time = time_span.GetStart()
        end_time   = time_span.GetStop()

        # BUG FIX: use integer frame arithmetic to avoid FbxTime float drift
        mode = scene.GetGlobalSettings().GetTimeMode()
        start_frame = int(start_time.GetFrameCount(mode))
        end_frame   = int(end_time.GetFrameCount(mode))
        frame_count = max(1, end_frame - start_frame + 1)

        log.info(f"Frames: {frame_count} @ {fps} fps ({frame_count/fps:.2f}s)")

        # ---- Extract per-frame transforms ----
        positions   = np.zeros((frame_count, num_joints, 3), dtype=np.float64)
        quaternions = np.zeros((frame_count, num_joints, 4), dtype=np.float64)

        fbx_time = fbx.FbxTime()
        for f in range(frame_count):
            fbx_time.SetFrame(start_frame + f, mode)
            for j, node in enumerate(joint_nodes):
                xform = node.EvaluateGlobalTransform(fbx_time)
                t = xform.GetT()
                q = xform.GetQ()
                positions[f, j, 0] = t[0]
                positions[f, j, 1] = t[1]
                positions[f, j, 2] = t[2]
                # Store as (w, x, y, z) — FBX SDK returns (x,y,z,w) via q[0..3]
                quaternions[f, j, 0] = q[3]   # w
                quaternions[f, j, 1] = q[0]   # x
                quaternions[f, j, 2] = q[1]   # y
                quaternions[f, j, 3] = q[2]   # z

            if (f + 1) % 500 == 0 or f == frame_count - 1:
                log.debug(f"  Extracted frame {f+1}/{frame_count}")

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

        elapsed = time.perf_counter() - t_start
        log.info(
            f"FBX load complete: \"{track.name}\" — "
            f"{frame_count} frames, {num_joints} joints, "
            f"align_joint=\"{track.align_joint}\", {elapsed:.2f}s"
        )
        return track

    finally:
        # BUG FIX: manager destroyed exactly once, here
        manager.Destroy()
