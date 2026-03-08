# Compute Mean Per Joint Rotation Error (MPJRE) vs reference track
#
# Usage: Load tracks, set reference, then run this script.
# Output: MPJRE (in degrees) for each non-reference track.

ref_idx = session.reference_index
ref = session.tracks[ref_idx]

if ref is None:
    print("ERROR: No reference track loaded.")
else:
    print(f"Reference: Track {ref_idx + 1} ({ref.name})")
    print()

    ref_quat = ref.quaternions
    if ref_quat is None:
        print("ERROR: Reference track has no rotation data.")
    else:
        for i, track in enumerate(session.tracks):
            if i == ref_idx or track is None:
                continue

            test_quat = track.quaternions
            if test_quat is None:
                print(f"Track {i + 1} ({track.name}): No rotation data")
                continue

            min_frames = min(ref_quat.shape[0], test_quat.shape[0])
            min_joints = min(ref_quat.shape[1], test_quat.shape[1])

            # Compute geodesic distance between quaternions
            # angle = 2 * arccos(|q1 · q2|)
            q1 = ref_quat[:min_frames, :min_joints, :]
            q2 = test_quat[:min_frames, :min_joints, :]

            # Dot product (w1*w2 + x1*x2 + y1*y2 + z1*z2)
            dot = np.sum(q1 * q2, axis=-1)
            dot = np.clip(np.abs(dot), 0.0, 1.0)

            angle_rad = 2.0 * np.arccos(dot)  # (F, J)
            angle_deg = np.degrees(angle_rad)

            mpjre = np.mean(angle_deg)
            mpjre_std = np.std(angle_deg)
            max_err = np.max(angle_deg)

            print(f"Track {i + 1} ({track.name}):")
            print(f"  MPJRE:  {mpjre:.3f} degrees  (std: {mpjre_std:.3f})")
            print(f"  Max:    {max_err:.3f} degrees")
            print(f"  Frames compared: {min_frames}")
            print(f"  Joints compared: {min_joints}")
            print()
