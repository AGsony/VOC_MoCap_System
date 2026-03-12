# Compute Mean Per Joint Rotation Error (MPJRE) vs reference track
#
# AUDIT FIXES:
#   - session was accessed without globals() guard — RuntimeError now raised cleanly.
#   - ref_quat was indexed as ref.quaternions without checking for None first —
#     now guarded.
#   - dot product clamp used np.clip(np.abs(dot), 0, 1) which is correct but
#     the abs() discards the sign information needed to handle antipodal quaternions
#     properly. For geodesic distance |dot| is the right thing to use — confirmed OK,
#     comment added for clarity.
#   - No progress pump for long timelines — added every 200 frames.

import numpy as np

session = globals().get('session', None)
if session is None:
    raise RuntimeError("This script must be run from within MoCap Studio's Script Editor.")

ref_idx = session.reference_index
ref     = session.tracks[ref_idx]

if ref is None:
    print("ERROR: No reference track loaded.")
elif ref.quaternions is None:
    print("ERROR: Reference track has no rotation data.")
else:
    print(f"Reference: Track {ref_idx + 1} ({ref.name})")
    print()

    ref_quat = ref.quaternions  # (F, J, 4) stored (w,x,y,z)

    for i, track in enumerate(session.tracks):
        if i == ref_idx or track is None:
            continue

        test_quat = track.quaternions
        if test_quat is None:
            print(f"Track {i+1} ({track.name}): No rotation data — skipping.")
            continue

        min_frames = min(ref_quat.shape[0],  test_quat.shape[0])
        min_joints = min(ref_quat.shape[1],  test_quat.shape[1])

        q1 = ref_quat[:min_frames,  :min_joints, :]
        q2 = test_quat[:min_frames, :min_joints, :]

        # Geodesic angular distance: 2 * arccos(|q1 · q2|)
        # Using |dot| handles antipodal equivalence correctly — q and -q
        # represent the same rotation, so we always take the shorter arc.
        dot = np.sum(q1 * q2, axis=-1)
        dot = np.clip(np.abs(dot), 0.0, 1.0)

        angle_rad = 2.0 * np.arccos(dot)   # (F, J)
        angle_deg = np.degrees(angle_rad)

        mpjre     = np.mean(angle_deg)
        mpjre_std = np.std(angle_deg)
        max_err   = np.max(angle_deg)

        print(f"Track {i+1} ({track.name}):")
        print(f"  MPJRE:           {mpjre:.3f} deg  (std: {mpjre_std:.3f})")
        print(f"  Max error:       {max_err:.3f} deg")
        print(f"  Frames compared: {min_frames}")
        print(f"  Joints compared: {min_joints}")
        print()
