# Export aligned position data for all loaded tracks to CSV
#
# Usage: Load and align tracks, then run this script.
# Output: One CSV file per track in ~/mocap_export/

import csv
import os

out_dir = os.path.expanduser("~/mocap_export")
os.makedirs(out_dir, exist_ok=True)

print(f"Export directory: {out_dir}")
print()

session = globals().get('session', None)
if session is None:
    raise RuntimeError("This script must be run from within MoCap Studio's Script Editor.")

def get_world_positions(t, num_frames):
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    # Isolate coordinates within dynamic timeline boundaries
    F = t.aligned_positions.shape[0]
    local_times = (np.arange(num_frames) - t.offset) / t.scale
    local_times = np.clip(local_times, 0, F - 1)
    
    idx0 = np.floor(local_times).astype(int)
    idx1 = np.minimum(idx0 + 1, F - 1)
    frac = (local_times - idx0)[:, np.newaxis, np.newaxis]
    
    raw_pos = t.aligned_positions[idx0] * (1.0 - frac) + t.aligned_positions[idx1] * frac
    
    # Process explicitly physical Euclidean coordinates natively manipulated inside the 3D Viewer layer
    rot = R.from_euler('XYZ', [t.rotate_x, t.rotate_y, t.rotate_z], degrees=True)
    old_shape = raw_pos.shape
    rotated = rot.apply(raw_pos.reshape(-1, 3))
    return rotated.reshape(old_shape) + [t.translate_x, t.translate_y, t.translate_z]


global_frames = session.max_frame
if global_frames <= 0:
    raise RuntimeError("Timeline is completely empty.")

for i, track in enumerate(session.tracks):
    if track is None or not track.visible:
        continue

    if track.aligned_positions is None:
        print(f"Track {i + 1} ({track.name}): No position data, skipping.")
        continue

    # Flatten coordinates down using timeline matrix rules
    track_world = get_world_positions(track, global_frames)
    path = os.path.join(out_dir, f"track_{i + 1}_{track.name}_global_space.csv")

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)

        # Header row
        header = ["frame"]
        for joint_name in track.skeleton.joint_names:
            header.extend([f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"])
        w.writerow(header)

        # Data rows
        for frame in range(global_frames):
            if frame % 100 == 0:
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()
                
            row = [frame]
            for j in range(len(track.skeleton.joint_names)):
                p = track_world[frame, j]
                row.extend([f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])
            w.writerow(row)

    print(f"Exported: Track {i + 1} ({track.name}) → {path}")
    print(f"  {track.frame_count} frames, {track.skeleton.joint_count} joints")

print()
print("Done!")
