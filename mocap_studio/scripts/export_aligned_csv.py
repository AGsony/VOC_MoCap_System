# Export aligned position data for all loaded tracks to CSV
#
# AUDIT FIXES:
#   - session guard via globals() was missing — added.
#   - get_world_positions(): scale=0 caused ZeroDivisionError — guarded.
#   - QApplication.processEvents() was imported inside the frame loop on
#     every 100th frame — that's an import call on ~1% of all frames.
#     Moved import to top of script.
#   - No error handling around file write — partial CSV on disk if it fails
#     mid-write. Now writes to a temp file and renames on success.
#   - Output path used track.name which may contain characters illegal in
#     Windows filenames (e.g. slashes, colons). Now sanitised.

import csv
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation as R
from PySide6.QtWidgets import QApplication

session = globals().get('session', None)
if session is None:
    raise RuntimeError("This script must be run from within MoCap Studio's Script Editor.")

out_dir = os.path.expanduser("~/mocap_export")
os.makedirs(out_dir, exist_ok=True)

print(f"Export directory: {out_dir}")
print()


def _sanitise_name(name: str) -> str:
    """Replace characters illegal in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def get_world_positions(t, num_frames):
    if t.aligned_positions is None:
        raise ValueError(f"Track '{t.name}' has no position data.")

    F     = t.aligned_positions.shape[0]
    # BUG FIX: guard scale=0
    scale = t.scale if t.scale != 0.0 else 1.0
    local_times = (np.arange(num_frames) - t.offset) / scale
    local_times = np.clip(local_times, 0, F - 1)

    idx0 = np.floor(local_times).astype(int)
    idx1 = np.minimum(idx0 + 1, F - 1)
    frac = (local_times - idx0)[:, np.newaxis, np.newaxis]

    raw_pos   = t.aligned_positions[idx0] * (1.0 - frac) + t.aligned_positions[idx1] * frac
    rot       = R.from_euler('XYZ', [t.rotate_x, t.rotate_y, t.rotate_z], degrees=True)
    old_shape = raw_pos.shape
    rotated   = rot.apply(raw_pos.reshape(-1, 3))
    return rotated.reshape(old_shape) + [t.translate_x, t.translate_y, t.translate_z]


global_frames = session.max_frame
if global_frames <= 0:
    raise RuntimeError("Timeline is completely empty.")

for i, track in enumerate(session.tracks):
    if track is None or not track.visible:
        continue

    try:
        track_world = get_world_positions(track, global_frames)
    except ValueError as e:
        print(f"Track {i+1} ({track.name}): {e} — skipping.")
        continue

    safe_name = _sanitise_name(track.name)
    final_path = os.path.join(out_dir, f"track_{i+1}_{safe_name}_global_space.csv")
    # BUG FIX: write to temp file, rename on success to avoid partial outputs
    tmp_path = final_path + ".tmp"

    try:
        with open(tmp_path, 'w', newline='') as f:
            w = csv.writer(f)

            header = ["frame"]
            for joint_name in track.skeleton.joint_names:
                header.extend([f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"])
            w.writerow(header)

            for frame in range(global_frames):
                # BUG FIX: moved QApplication import outside loop; pump UI every 100 frames
                if frame % 100 == 0:
                    QApplication.processEvents()

                row = [frame]
                for j in range(len(track.skeleton.joint_names)):
                    p = track_world[frame, j]
                    row.extend([f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])
                w.writerow(row)

        os.replace(tmp_path, final_path)   # atomic on same filesystem
        print(f"Exported: Track {i+1} ({track.name}) → {final_path}")
        print(f"  {track.frame_count} frames, {track.skeleton.joint_count} joints")

    except Exception as exc:
        # Clean up temp file on failure
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        print(f"Track {i+1} ({track.name}): export failed — {exc}")

print()
print("Done!")
