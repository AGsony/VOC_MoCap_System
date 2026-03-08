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

for i, track in enumerate(session.tracks):
    if track is None:
        continue

    aligned = track.aligned_positions
    if aligned is None:
        print(f"Track {i + 1} ({track.name}): No position data, skipping.")
        continue

    path = os.path.join(out_dir, f"track_{i + 1}_{track.name}_aligned.csv")

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)

        # Header row
        header = ["frame"]
        for joint_name in track.skeleton.joint_names:
            header.extend([f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"])
        w.writerow(header)

        # Data rows
        for frame in range(track.frame_count):
            row = [frame]
            for j in range(len(track.skeleton.joint_names)):
                p = aligned[frame, j]
                row.extend([f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])
            w.writerow(row)

    print(f"Exported: Track {i + 1} ({track.name}) → {path}")
    print(f"  {track.frame_count} frames, {track.skeleton.joint_count} joints")

print()
print("Done!")
