# Compute Mean Per Joint Position Error (MPJPE) vs reference track
#
# Usage: Load tracks, set reference, then run this script.
# Output: MPJPE for each non-reference track.

ref_idx = session.reference_index
ref = session.tracks[ref_idx]

if ref is None:
    print("ERROR: No reference track loaded.")
else:
    print(f"Reference: Track {ref_idx + 1} ({ref.name})")
    print(f"  Joints: {ref.skeleton.joint_count}")
    print(f"  Frames: {ref.frame_count}")
    print(f"  Align joint: {ref.align_joint}")
    print()

    ref_pos = ref.aligned_positions
    if ref_pos is None:
        print("ERROR: Reference track has no position data.")
    else:
        for i, track in enumerate(session.tracks):
            if i == ref_idx or track is None:
                continue

            test_pos = track.aligned_positions
            if test_pos is None:
                print(f"Track {i + 1} ({track.name}): No position data")
                continue

            # Use the minimum frame count between ref and test
            min_frames = min(ref_pos.shape[0], test_pos.shape[0])
            min_joints = min(ref_pos.shape[1], test_pos.shape[1])

            diff = ref_pos[:min_frames, :min_joints, :] - test_pos[:min_frames, :min_joints, :]
            per_joint = np.linalg.norm(diff, axis=-1)  # (F, J)

            mpjpe = np.mean(per_joint)
            mpjpe_std = np.std(per_joint)
            max_err = np.max(per_joint)

            print(f"Track {i + 1} ({track.name}):")
            print(f"  MPJPE:  {mpjpe:.3f} units  (std: {mpjpe_std:.3f})")
            print(f"  Max:    {max_err:.3f} units")
            print(f"  Frames compared: {min_frames}")
            print(f"  Joints compared: {min_joints}")
            print()
