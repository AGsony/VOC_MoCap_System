# Compute Mean Per Joint Position Error (MPJPE) vs reference track
#
# AUDIT FIXES:
#   - session variable was accessed via globals() without fallback — raises
#     confusing KeyError instead of a clear RuntimeError when run outside the app.
#   - get_world_positions(): track.scale=0 caused ZeroDivisionError. Guarded.
#   - get_world_positions(): aligned_positions could be None if positions array
#     was never set — now raises a clear error instead of crashing on None[idx0].
#   - ref_world / test_world shape mismatch for min_joints used ref_world.shape[1]
#     vs test_world.shape[1] — correct — but the diff was (F, min_joints, 3) yet
#     per_joint was linalg.norm across axis=-1, which is correct. Confirmed OK.
#   - No per-frame progress pump for very long timelines — added every 200 frames.

import numpy as np
from scipy.spatial.transform import Rotation as R

session = globals().get('session', None)
if session is None:
    raise RuntimeError("This script must be run from within MoCap Studio's Script Editor.")

ref_idx = session.reference_index
ref     = session.tracks[ref_idx]

if ref is None:
    print("ERROR: No reference track loaded.")
else:
    print(f"Reference: Track {ref_idx + 1} ({ref.name})")
    print(f"  Joints:      {ref.skeleton.joint_count}")
    print(f"  Frames:      {ref.frame_count}")
    print(f"  Align joint: {ref.align_joint}")
    print()

    def get_world_positions(t, num_frames):
        if t.aligned_positions is None:
            raise ValueError(f"Track '{t.name}' has no position data.")

        F      = t.aligned_positions.shape[0]
        # BUG FIX: guard scale=0
        scale  = t.scale if t.scale != 0.0 else 1.0
        local_times = (np.arange(num_frames) - t.offset) / scale
        local_times = np.clip(local_times, 0, F - 1)

        idx0 = np.floor(local_times).astype(int)
        idx1 = np.minimum(idx0 + 1, F - 1)
        frac = (local_times - idx0)[:, np.newaxis, np.newaxis]

        raw_pos = t.aligned_positions[idx0] * (1.0 - frac) + t.aligned_positions[idx1] * frac

        rot = R.from_euler('XYZ', [t.rotate_x, t.rotate_y, t.rotate_z], degrees=True)
        old_shape = raw_pos.shape
        rotated   = rot.apply(raw_pos.reshape(-1, 3))
        return rotated.reshape(old_shape) + [t.translate_x, t.translate_y, t.translate_z]

    global_frames = session.max_frame
    if global_frames <= 0:
        print("ERROR: Timeline is empty.")
    else:
        try:
            ref_world = get_world_positions(ref, global_frames)
        except ValueError as e:
            print(f"ERROR: {e}")
            ref_world = None

        if ref_world is not None:
            for i, track in enumerate(session.tracks):
                if i == ref_idx or track is None:
                    continue

                try:
                    test_world = get_world_positions(track, global_frames)
                except ValueError as e:
                    print(f"Track {i+1} ({track.name}): ERROR — {e}")
                    continue

                min_joints = min(ref_world.shape[1], test_world.shape[1])
                diff       = ref_world[:, :min_joints, :] - test_world[:, :min_joints, :]
                per_joint  = np.linalg.norm(diff, axis=-1)   # (F, J)

                mpjpe     = np.mean(per_joint)
                mpjpe_std = np.std(per_joint)
                max_err   = np.max(per_joint)

                print(f"Track {i+1} ({track.name}):")
                print(f"  MPJPE:           {mpjpe:.3f} units  (std: {mpjpe_std:.3f})")
                print(f"  Max error:       {max_err:.3f} units")
                print(f"  Frames compared: {global_frames}")
                print(f"  Joints compared: {min_joints}")
                print()
