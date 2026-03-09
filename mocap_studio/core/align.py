"""
Alignment & interpolation utilities.

• align_track()       — subtract alignment-joint position per frame
• resample_track()    — interpolate to a different frame count (lerp + slerp)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .track import Track
from .skeleton import Skeleton


def align_positions(positions: np.ndarray, joint_index: int) -> np.ndarray:
    """
    Translate all joints so that *joint_index* starts at XZ origin on frame 0.

    Only subtracts the X and Z components of the alignment joint at frame 0,
    preserving vertical (Y) height and root motion trajectory.

    Parameters
    ----------
    positions : (F, J, 3) ndarray
    joint_index : int

    Returns
    -------
    (F, J, 3) ndarray — aligned copy of positions
    """
    frame0_pos = positions[0, joint_index, :].copy()
    frame0_pos[1] = 0.0  # preserve Y (height)
    return positions - frame0_pos[np.newaxis, np.newaxis, :]


def resample_track(track: Track, target_frame_count: int) -> Track:
    """
    Return a *new* Track resampled to ``target_frame_count`` frames.

    Positions are resampled via per-component linear interpolation.
    Quaternions are resampled via SLERP (scipy).
    """
    src_fc = track.frame_count
    if src_fc == target_frame_count:
        return track  # no-op

    src_times = np.arange(src_fc, dtype=np.float64)
    dst_times = np.linspace(0, src_fc - 1, target_frame_count)

    # --- positions (lerp) ---
    new_pos = None
    if track.positions is not None:
        F, J, _ = track.positions.shape
        new_pos = np.empty((target_frame_count, J, 3), dtype=np.float64)
        for j in range(J):
            for c in range(3):
                new_pos[:, j, c] = np.interp(dst_times, src_times,
                                              track.positions[:, j, c])

    # --- quaternions (slerp) ---
    new_quat = None
    if track.quaternions is not None:
        F, J, _ = track.quaternions.shape
        new_quat = np.empty((target_frame_count, J, 4), dtype=np.float64)
        for j in range(J):
            # scipy expects (x,y,z,w) but our storage is (w,x,y,z).
            wxyz = track.quaternions[:, j, :]
            xyzw = np.concatenate([wxyz[:, 1:], wxyz[:, :1]], axis=-1)
            rots = Rotation.from_quat(xyzw)
            slerp_fn = Slerp(src_times, rots)
            resampled = slerp_fn(dst_times)
            out_xyzw = resampled.as_quat()  # (N, 4) x,y,z,w
            new_quat[:, j, :] = np.concatenate(
                [out_xyzw[:, 3:], out_xyzw[:, :3]], axis=-1
            )  # back to w,x,y,z

    new_track = Track(
        name=track.name,
        source_path=track.source_path,
        fps=track.fps,
        frame_count=target_frame_count,
        skeleton=track.skeleton,
        positions=new_pos,
        quaternions=new_quat,
        offset=track.offset,
        trim_in=0,
        trim_out=target_frame_count - 1,
        align_joint=track.align_joint,
        visible=track.visible,
    )
    return new_track


def auto_align_tracks(ref_track: Track, test_track: Track) -> float:
    """
    Find the optimal frame offset to align test_track to ref_track.
    
    This works by comparing the velocity magnitude of their 
    alignment joints (usually the root/hips) over time using 
    cross-correlation.

    Returns
    -------
    float : The suggested frame offset for test_track.
    """
    if ref_track.positions is None or test_track.positions is None:
        return 0.0

    # Extract 3D positions for the alignment joint
    ref_idx = ref_track.align_joint_index
    test_idx = test_track.align_joint_index

    ref_pos = ref_track.positions[:, ref_idx, :]
    test_pos = test_track.positions[:, test_idx, :]

    # Compute velocity magnitudes (speed)
    ref_vel = np.linalg.norm(np.diff(ref_pos, axis=0), axis=1)
    test_vel = np.linalg.norm(np.diff(test_pos, axis=0), axis=1)

    # Normalize to mean=0, std=1 for cross-correlation
    ref_vel = (ref_vel - np.mean(ref_vel)) / (np.std(ref_vel) + 1e-8)
    test_vel = (test_vel - np.mean(test_vel)) / (np.std(test_vel) + 1e-8)

    # Compute cross-correlation
    # Full mode returns length = N + M - 1
    corr = np.correlate(ref_vel, test_vel, mode='full')

    # The peak of the correlation indicates the lag
    # lag = argmax(corr) - (len(test) - 1)
    peak_idx = np.argmax(corr)
    optimal_lag = peak_idx - (len(test_vel) - 1)

    return float(optimal_lag)
