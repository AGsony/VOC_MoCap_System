"""
BVH Extraction — Pure-Python BVH motion-capture file parser.

AUDIT FIXES:
  - _euler_to_matrix(): rotation order was applied left-to-right but BVH
    convention requires right-to-left (innermost first). The original comment
    said "apply right to left" but the loop applied left to right.
    Fixed by reversing the axis iteration order.
  - End Site joints were added to joint_names/parent_indices but channels
    and offsets lists could fall out of sync with joint_names because the
    continue statement skipped the normal offset append at the bottom.
    Refactored so End Site always appends to all four lists consistently.
  - frame_count was read from "Frames:" line but the actual data line count
    was used separately — if the header lied, extraction silently under/over-ran.
    Now uses min(frame_count, actual_data_lines).
  - fps was derived as 1/frame_time; if frame_time was 0 this would produce
    a ZeroDivisionError (already guarded) but the resulting fps=60 wasn't
    logged as a fallback — now it is.
  - Global transform computation used a flat list of np.eye(4) — list is
    O(n) for index lookup. Changed to a pre-allocated numpy array for clarity;
    logic is unchanged.
  - track.fps was set after Track construction; auto_setup() was called
    before fps was finalised. Reordered.
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
    joint_names:        List[str]             = []
    parent_indices:     List[int]             = []
    channels_per_joint: List[int]             = []
    channel_order:      List[List[str]]       = []
    offsets:            List[Tuple[float, float, float]] = []

    stack: List[int] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("ROOT") or line.startswith("JOINT"):
            name       = line.split()[-1]
            parent_idx = stack[-1] if stack else -1
            joint_names.append(name)
            parent_indices.append(parent_idx)
            i += 1
            continue

        elif line.startswith("End Site"):
            parent_idx = stack[-1] if stack else -1
            name = (joint_names[parent_idx] + "_End") if parent_idx >= 0 else "End"
            joint_names.append(name)
            parent_indices.append(parent_idx)
            channels_per_joint.append(0)
            channel_order.append([])

            # Consume { OFFSET } block
            i += 1   # {
            i += 1   # OFFSET line
            off_line = lines[i].strip().split()
            offsets.append((float(off_line[1]), float(off_line[2]), float(off_line[3])))
            i += 1   # }
            i += 1
            continue

        elif line == "{":
            stack.append(len(joint_names) - 1)

        elif line == "}":
            if stack:
                stack.pop()

        elif line.startswith("OFFSET"):
            parts = line.split()
            offsets.append((float(parts[1]), float(parts[2]), float(parts[3])))

        elif line.startswith("CHANNELS"):
            parts  = line.split()
            n_ch   = int(parts[1])
            ch_names = parts[2:2 + n_ch]
            channels_per_joint.append(n_ch)
            channel_order.append(ch_names)

        i += 1

    return joint_names, parent_indices, offsets, channels_per_joint, channel_order


def _euler_to_matrix(rx: float, ry: float, rz: float, order: str) -> np.ndarray:
    """
    Convert Euler angles (degrees) to 3×3 rotation matrix.

    BUG FIX: original applied matrices left-to-right (R[order[0]] @ result ...)
    but BVH convention is the LAST axis in the channel string is the innermost
    rotation, meaning it should be applied first (rightmost in matrix product).
    Corrected to iterate in reverse so the result is R[0] @ R[1] @ R[2].
    """
    rx_r, ry_r, rz_r = np.radians(rx), np.radians(ry), np.radians(rz)

    cx, sx = np.cos(rx_r), np.sin(rx_r)
    cy, sy = np.cos(ry_r), np.sin(ry_r)
    cz, sz = np.cos(rz_r), np.sin(rz_r)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    matrices = {"X": Rx, "Y": Ry, "Z": Rz}

    # BUG FIX: reversed order so first axis is outermost (applied last)
    result = np.eye(3)
    for axis in reversed(order):
        result = matrices[axis] @ result
    return result


def load_bvh(filepath: str) -> Track:
    """Load a BVH file and return a Track."""
    t_start = time.perf_counter()
    log.info(f"Loading BVH: {filepath}")

    with open(filepath, "r") as f:
        content = f.read()

    parts          = re.split(r"MOTION\s*\n", content, maxsplit=1)
    hierarchy_text = parts[0]
    motion_text    = parts[1] if len(parts) > 1 else ""

    lines = hierarchy_text.split("\n")
    joint_names, parent_indices, offsets, channels_per_joint, channel_order = (
        _parse_hierarchy(lines)
    )

    skeleton   = Skeleton(joint_names=joint_names, parent_indices=parent_indices)
    num_joints = len(joint_names)
    log.info(f"Hierarchy: {num_joints} joints")

    # Parse motion header
    motion_lines = motion_text.strip().split("\n")
    frame_count  = 0
    frame_time   = 1.0 / 60.0
    data_start   = 0

    for idx, line in enumerate(motion_lines):
        line = line.strip()
        if line.startswith("Frames:"):
            frame_count = int(line.split(":")[1].strip())
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split(":")[1].strip())
        elif line and not line.startswith("Frames") and not line.startswith("Frame"):
            data_start = idx
            break

    # BUG FIX: guard divide-by-zero and log fallback explicitly
    if frame_time <= 0:
        log.warning("frame_time <= 0 in BVH — defaulting to 60 fps")
        fps = 60.0
    else:
        fps = 1.0 / frame_time

    log.info(f"Motion: {frame_count} frames @ {fps:.3f} fps")

    offset_arr = np.zeros((num_joints, 3), dtype=np.float64)
    for j in range(min(len(offsets), num_joints)):
        offset_arr[j] = offsets[j]

    rot_orders = []
    for ch_list in channel_order:
        rot_axes = "".join(
            ch[0].upper()
            for ch in ch_list
            if "ROTATION" in ch.upper()
        )
        rot_orders.append(rot_axes if rot_axes else "ZYX")

    positions   = np.zeros((frame_count, num_joints, 3), dtype=np.float64)
    quaternions = np.zeros((frame_count, num_joints, 4), dtype=np.float64)
    quaternions[:, :, 0] = 1.0   # w=1 identity

    frame_data_lines = motion_lines[data_start:data_start + frame_count]
    # BUG FIX: use actual line count in case header frame_count was wrong
    actual_frames = min(frame_count, len(frame_data_lines))
    if actual_frames != frame_count:
        log.warning(
            f"BVH header says {frame_count} frames but only {actual_frames} data lines found."
        )

    # Pre-allocate global transform storage
    global_mats = np.tile(np.eye(4), (num_joints, 1, 1))  # (J, 4, 4)

    for f, line in enumerate(frame_data_lines[:actual_frames]):
        values = [float(v) for v in line.strip().split()]
        vi     = 0
        global_mats[:] = np.eye(4)

        for j in range(num_joints):
            n_ch   = channels_per_joint[j] if j < len(channels_per_joint) else 0
            local  = np.eye(4)
            local[:3, 3] = offset_arr[j]

            if n_ch > 0 and vi + n_ch <= len(values):
                ch_list = channel_order[j] if j < len(channel_order) else []
                tx = ty = tz = 0.0
                rx_val = ry_val = rz_val = 0.0

                for ci, ch_name in enumerate(ch_list):
                    ch_upper = ch_name.upper()
                    val = values[vi + ci]
                    if   ch_upper == "XPOSITION": tx     = val
                    elif ch_upper == "YPOSITION": ty     = val
                    elif ch_upper == "ZPOSITION": tz     = val
                    elif ch_upper == "XROTATION": rx_val = val
                    elif ch_upper == "YROTATION": ry_val = val
                    elif ch_upper == "ZROTATION": rz_val = val

                if any("POSITION" in ch.upper() for ch in ch_list):
                    local[:3, 3] = [tx, ty, tz]

                rot_order = rot_orders[j] if j < len(rot_orders) else "ZYX"
                local[:3, :3] = _euler_to_matrix(rx_val, ry_val, rz_val, rot_order)
                vi += n_ch

            parent = parent_indices[j]
            if parent >= 0:
                global_mats[j] = global_mats[parent] @ local
            else:
                global_mats[j] = local

            positions[f, j, :] = global_mats[j, :3, 3]

            rot    = _Rotation.from_matrix(global_mats[j, :3, :3])
            q_xyzw = rot.as_quat()   # scipy (x,y,z,w)
            # Store as (w,x,y,z)
            quaternions[f, j, :] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    track = Track(
        name=os.path.splitext(os.path.basename(filepath))[0],
        source_path=filepath,
        fps=fps,
        frame_count=actual_frames,
        skeleton=skeleton,
        positions=positions[:actual_frames],
        quaternions=quaternions[:actual_frames],
    )
    track.auto_setup()

    elapsed = time.perf_counter() - t_start
    log.info(
        f"BVH load complete: \"{track.name}\" — "
        f"{actual_frames} frames, {num_joints} joints, "
        f"align_joint=\"{track.align_joint}\", {elapsed:.2f}s"
    )
    return track
