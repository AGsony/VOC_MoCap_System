# Compute Mean Per Joint Position Error (MPJPE) vs reference track
#
# Usage: Load tracks, set reference, then run this script.
# Output: MPJPE for each non-reference track.

import numpy as np

session = globals().get('session', None)
if session is None:
    raise RuntimeError("This script must be run from within MoCap Studio's Script Editor.")

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

    # Provide a unified way to extract interpolated world-space coords out of the given track
    def get_world_positions(t, num_frames):
        from scipy.spatial.transform import Rotation as R
        
        # Pull the native aligned positional coordinates isolated to the specific UI time boundaries
        F = t.aligned_positions.shape[0]
        local_times = (np.arange(num_frames) - t.offset) / t.scale
        local_times = np.clip(local_times, 0, F - 1)
        
        idx0 = np.floor(local_times).astype(int)
        idx1 = np.minimum(idx0 + 1, F - 1)
        frac = (local_times - idx0)[:, np.newaxis, np.newaxis]
        
        raw_pos = t.aligned_positions[idx0] * (1.0 - frac) + t.aligned_positions[idx1] * frac
        
        # Apply the explicit physical 3D GUI overlaps the user manually tweaked
        rot = R.from_euler('XYZ', [t.rotate_x, t.rotate_y, t.rotate_z], degrees=True)
        # Apply to all frames (N) and joints (J)
        old_shape = raw_pos.shape
        flat_pos = raw_pos.reshape(-1, 3) 
        rotated = rot.apply(flat_pos)
        final_pos = rotated.reshape(old_shape) + [t.translate_x, t.translate_y, t.translate_z]
        return final_pos
        
    global_frames = session.max_frame
    if global_frames <= 0:
        print("ERROR: Timeline is empty.")
    else:    
        ref_world = get_world_positions(ref, global_frames)
    
        for i, track in enumerate(session.tracks):
            if i == ref_idx or track is None:
                continue
    
            test_world = get_world_positions(track, global_frames)
    
            # Restrict error calculations to just the joints which overlap between skeleton definitions
            min_joints = min(ref_world.shape[1], test_world.shape[1])
    
            diff = ref_world[:, :min_joints, :] - test_world[:, :min_joints, :]
            per_joint = np.linalg.norm(diff, axis=-1)  # (F, J)
    
            mpjpe = np.mean(per_joint)
            mpjpe_std = np.std(per_joint)
            max_err = np.max(per_joint)

            print(f"Track {i + 1} ({track.name}):")
            print(f"  MPJPE:  {mpjpe:.3f} units  (std: {mpjpe_std:.3f})")
            print(f"  Max:    {max_err:.3f} units")
            print(f"  Frames compared: {global_frames}")
            print(f"  Joints compared: {min_joints}")
            print()
