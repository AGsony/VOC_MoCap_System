"""
BVH Extraction — Fallback parser for BVH motion-capture files.

Pure-Python BVH parser that builds a Skeleton + per-frame joint positions
stored as NumPy arrays, providing the same Track interface as fbx_extract.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

from .skeleton import Skeleton
from .track import Track

log = logging.getLogger("mocap_studio.core.bvh_extract")


def _parse_hierarchy(lines: List[str]):
    """
    Parse HIERARCHY section of a BVH file.
    Returns (joint_names, parent_indices, channels_per_joint, channel_order_per_joint).
    """
    joint_names: List[str] = []
    parent_indices: List[int] = []
    channels_per_joint: List[int] = []
    channel_order: List[List[str]] = []
    offsets: List[Tuple[float, float, float]] = []

    stack: List[int] = []  # stack of parent indices

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("ROOT") or line.startswith("JOINT"):
            name = line.split()[-1]
            parent_idx = stack[-1] if stack else -1
            idx = len(joint_names)
            joint_names.append(name)
            parent_indices.append(parent_idx)

        elif line.startswith("End Site"):
            # End effectors — add as a joint for rendering
            parent_idx = stack[-1] if stack else -1
            name = joint_names[parent_idx] + "_End" if parent_idx >= 0 else "End"
            idx = len(joint_names)
            joint_names.append(name)
            parent_indices.append(parent_idx)
            channels_per_joint.append(0)
            channel_order.append([])

            # Read offset and closing brace
            i += 1  # {
            i += 1  # OFFSET line
            off_line = lines[i].strip()
            parts = off_line.split()
            offsets.append((float(parts[1]), float(parts[2]), float(parts[3])))
            i += 1  # }
            i += 1
            continue

        elif line == "{":
            stack.append(len(joint_names) - 1)
            i += 1
            continue

        elif line == "}":
            if stack:
                stack.pop()
            i += 1
            continue

        elif line.startswith("OFFSET"):
            parts = line.split()
            offsets.append((float(parts[1]), float(parts[2]), float(parts[3])))
            i += 1
            continue

        elif line.startswith("CHANNELS"):
            parts = line.split()
            n_ch = int(parts[1])
            ch_names = parts[2:]
            channels_per_joint.append(n_ch)
            channel_order.append(ch_names)
            i += 1
            continue

        else:
            i += 1
            continue

        i += 1

    return joint_names, parent_indices, offsets, channels_per_joint, channel_order


def _euler_to_matrix(rx: float, ry: float, rz: float, order: str) -> np.ndarray:
    """Convert Euler angles (degrees) to a 3x3 rotation matrix."""
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    matrices = {"X": Rx, "Y": Ry, "Z": Rz}
    # order is like "ZXY" — apply right to left
    result = np.eye(3)
    for axis in order:
        result = matrices[axis] @ result
    return result


def load_bvh(filepath: str) -> Track:
    """Load a BVH file and return a Track."""
    t_start = time.perf_counter()
    log.info(f"Loading BVH: {filepath}")

    with open(filepath, "r") as f:
        content = f.read()
    log.debug(f"File read: {len(content)} bytes")

    # Split into hierarchy and motion sections
    parts = re.split(r"MOTION\s*\n", content, maxsplit=1)
    hierarchy_text = parts[0]
    motion_text = parts[1] if len(parts) > 1 else ""

    lines = hierarchy_text.split("\n")
    joint_names, parent_indices, offsets, channels_per_joint, channel_order = (
        _parse_hierarchy(lines)
    )

    skeleton = Skeleton(joint_names=joint_names, parent_indices=parent_indices)
    num_joints = len(joint_names)
    log.info(f"Hierarchy parsed: {num_joints} joints")
    log.debug(f"Joint names: {joint_names[:10]}{'...' if num_joints > 10 else ''}")

    # Parse motion section
    motion_lines = motion_text.strip().split("\n")
    frame_count = 0
    frame_time = 1.0 / 60.0

    data_start = 0
    for i, line in enumerate(motion_lines):
        line = line.strip()
        if line.startswith("Frames:"):
            frame_count = int(line.split(":")[1].strip())
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split(":")[1].strip())
        elif line and not line.startswith("Frames") and not line.startswith("Frame"):
            data_start = i
            break

    fps = 1.0 / frame_time if frame_time > 0 else 60.0
    log.info(f"Motion: {frame_count} frames @ {fps:.1f} fps (frame_time={frame_time:.6f}s)")

    # Pre-compute offsets as ndarray
    offset_arr = np.zeros((num_joints, 3), dtype=np.float64)
    for j in range(min(len(offsets), num_joints)):
        offset_arr[j] = offsets[j]

    # Determine rotation order for each joint
    rot_orders = []
    for ch_list in channel_order:
        # Extract rotation channel order (X/Y/Zrotation)
        rot_axes = ""
        for ch in ch_list:
            ch_upper = ch.upper()
            if "ROTATION" in ch_upper:
                rot_axes += ch_upper[0]  # first char: X, Y, or Z
        rot_orders.append(rot_axes if rot_axes else "ZYX")

    # Parse frame data
    positions = np.zeros((frame_count, num_joints, 3), dtype=np.float64)
    quaternions = np.zeros((frame_count, num_joints, 4), dtype=np.float64)
    quaternions[:, :, 0] = 1.0  # w=1 identity

    frame_data_lines = motion_lines[data_start : data_start + frame_count]

    for f, line in enumerate(frame_data_lines):
        values = [float(v) for v in line.strip().split()]
        vi = 0  # value index
        # Compute global transforms via forward kinematics
        global_transforms = [np.eye(4) for _ in range(num_joints)]

        ch_joint_idx = 0  # index into joints that have channels
        for j in range(num_joints):
            n_ch = channels_per_joint[j] if j < len(channels_per_joint) else 0

            # Local transform
            local = np.eye(4)
            local[:3, 3] = offset_arr[j]

            if n_ch > 0 and vi + n_ch <= len(values):
                ch_list = channel_order[j] if j < len(channel_order) else []

                tx = ty = tz = 0.0
                rx_val = ry_val = rz_val = 0.0

                for ci, ch_name in enumerate(ch_list):
                    ch_upper = ch_name.upper()
                    val = values[vi + ci]
                    if ch_upper == "XPOSITION":
                        tx = val
                    elif ch_upper == "YPOSITION":
                        ty = val
                    elif ch_upper == "ZPOSITION":
                        tz = val
                    elif ch_upper == "XROTATION":
                        rx_val = val
                    elif ch_upper == "YROTATION":
                        ry_val = val
                    elif ch_upper == "ZROTATION":
                        rz_val = val

                # Position channels (usually only root has them)
                if any("POSITION" in ch.upper() for ch in ch_list):
                    local[:3, 3] = [tx, ty, tz]

                # Rotation
                rot_order = rot_orders[j] if j < len(rot_orders) else "ZYX"
                rot_mat = _euler_to_matrix(rx_val, ry_val, rz_val, rot_order)
                local[:3, :3] = rot_mat

                vi += n_ch

            # Global = parent_global @ local
            parent = parent_indices[j]
            if parent >= 0:
                global_transforms[j] = global_transforms[parent] @ local
            else:
                global_transforms[j] = local

            # Extract position
            positions[f, j, :] = global_transforms[j][:3, 3]

            # Extract quaternion from rotation matrix
            rot = _Rotation.from_matrix(global_transforms[j][:3, :3])
            q_xyzw = rot.as_quat()  # scipy: (x,y,z,w)
            quaternions[f, j, :] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    # Normalise positions to metres.
    # BVH has no unit metadata — infer from Y extent across all joints and frames.
    # A human skeleton spans ~1.7m / 170cm / 1700mm root-to-head.
    _y_extent = float(positions[:, :, 1].max() - positions[:, :, 1].min())
    if _y_extent > 500.0:
        positions *= 0.001
        log.info(f"BVH unit normalisation: Y extent={_y_extent:.1f} assumed millimetres, scaled x0.001 to metres.")
    elif _y_extent > 10.0:
        positions *= 0.01
        log.info(f"BVH unit normalisation: Y extent={_y_extent:.1f} assumed centimetres, scaled x0.01 to metres.")
    else:
        log.debug(f"BVH positions appear to be in metres already (Y extent={_y_extent:.3f}).")

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
        f"BVH load complete: \"{track.name}\" — "
        f"{frame_count} frames, {num_joints} joints, "
        f"align_joint=\"{track.align_joint}\", "
        f"total time {elapsed:.2f}s"
    )
    return track
