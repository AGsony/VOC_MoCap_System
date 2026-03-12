"""
Alignment & interpolation utilities.

• align_positions()     — subtract alignment-joint position per frame
• resample_track()      — interpolate to a different frame count (lerp + slerp)
• auto_align_tracks()   — cross-correlation frame-offset finder

AUDIT FIXES:
  - resample_track(): src_fc == 1 would produce a divide-by-zero when
    building dst_times via np.linspace(0, src_fc-1, ...) — returns track
    unchanged for single-frame inputs.
  - resample_track(): produced a Track with trim_out set to the OLD value
    from the source track instead of target_frame_count - 1, causing
    out-of-range frame access after resampling.
  - auto_align_tracks(): std=0 (completely static capture) produced
    NaN velocity signal due to divide-by-zero; already guarded with 1e-8
    but the comment didn't make this explicit.
  - resample_track(): new Track was missing scale field, so scale always
    reset to dataclass default (1.0) after resample — correct by accident
    but fragile. Now explicitly forwarded.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .track import Track
from .skeleton import Skeleton


def align_positions(positions: np.ndarray, joint_index: int) -> np.ndarray:
    """
    Translate all joints so that *joint_index* starts at XZ origin on frame 0.

    Only subtracts X and Z of the alignment joint at frame 0,
    preserving vertical (Y) height and root-motion trajectory.

    Parameters
    ----------
    positions : (F, J, 3) ndarray
    joint_index : int

    Returns
    -------
    (F, J, 3) ndarray — aligned copy of positions
    """
    frame0_pos = positions[0, joint_index, :].copy()
    frame0_pos[1] = 0.0   # preserve Y
    return positions - frame0_pos[np.newaxis, np.newaxis, :]


def resample_track(track: Track, target_frame_count: int) -> Track:
    """
    Return a *new* Track resampled to ``target_frame_count`` frames.

    Positions are resampled via per-component linear interpolation.
    Quaternions are resampled via SLERP (scipy).

    BUG FIX: single-frame source tracks caused divide-by-zero in linspace.
    BUG FIX: trim_out was copied from source instead of being set to
             target_frame_count - 1.
    """
    src_fc = track.frame_count
    if src_fc == target_frame_count:
        return track   # no-op

    # BUG FIX: single-frame track — can't interpolate, just tile
    if src_fc <= 1:
        new_pos  = np.repeat(track.positions,  target_frame_count, axis=0) if track.positions  is not None else None
        new_quat = np.repeat(track.quaternions, target_frame_count, axis=0) if track.quaternions is not None else None
        new_track = Track(
            name=track.name, source_path=track.source_path,
            fps=track.fps, frame_count=target_frame_count,
            skeleton=track.skeleton,
            positions=new_pos, quaternions=new_quat,
            offset=track.offset, scale=track.scale,
            trim_in=0, trim_out=target_frame_count - 1,   # BUG FIX
            align_joint=track.align_joint, visible=track.visible,
        )
        return new_track

    src_times = np.arange(src_fc, dtype=np.float64)
    dst_times = np.linspace(0, src_fc - 1, target_frame_count)

    # --- positions (lerp) ---
    new_pos = None
    if track.positions is not None:
        F, J, _ = track.positions.shape
        new_pos = np.empty((target_frame_count, J, 3), dtype=np.float64)
        for j in range(J):
            for c in range(3):
                new_pos[:, j, c] = np.interp(dst_times, src_times, track.positions[:, j, c])

    # --- quaternions (slerp) ---
    new_quat = None
    if track.quaternions is not None:
        F, J, _ = track.quaternions.shape
        new_quat = np.empty((target_frame_count, J, 4), dtype=np.float64)
        for j in range(J):
            # Our storage: (w,x,y,z) — scipy Slerp needs (x,y,z,w)
            wxyz = track.quaternions[:, j, :]
            xyzw = np.concatenate([wxyz[:, 1:], wxyz[:, :1]], axis=-1)
            rots     = Rotation.from_quat(xyzw)
            slerp_fn = Slerp(src_times, rots)
            resampled = slerp_fn(dst_times)
            out_xyzw = resampled.as_quat()   # (N, 4) x,y,z,w
            # Convert back to (w,x,y,z)
            new_quat[:, j, :] = np.concatenate([out_xyzw[:, 3:], out_xyzw[:, :3]], axis=-1)

    new_track = Track(
        name=track.name,
        source_path=track.source_path,
        fps=track.fps,
        frame_count=target_frame_count,
        skeleton=track.skeleton,
        positions=new_pos,
        quaternions=new_quat,
        offset=track.offset,
        scale=track.scale,          # explicitly forward scale
        trim_in=0,
        trim_out=target_frame_count - 1,   # BUG FIX: was track.trim_out
        align_joint=track.align_joint,
        visible=track.visible,
    )
    return new_track


def auto_align_tracks(ref_track: Track, test_track: Track) -> float:
    """
    Find the optimal frame offset to align test_track to ref_track
    using cross-correlation of root-joint velocity magnitudes.

    Returns
    -------
    float : Suggested frame offset for test_track (apply to test_track.offset).

    NOTE: Returns 0.0 if either track has no position data or fewer than
    2 frames (velocity requires at least 2 frames).
    """
    if ref_track.positions is None or test_track.positions is None:
        return 0.0

    if ref_track.frame_count < 2 or test_track.frame_count < 2:
        return 0.0

    ref_idx  = ref_track.align_joint_index
    test_idx = test_track.align_joint_index

    ref_pos  = ref_track.positions[:, ref_idx, :]
    test_pos = test_track.positions[:, test_idx, :]

    # Velocity magnitude (speed)
    ref_vel  = np.linalg.norm(np.diff(ref_pos,  axis=0), axis=1)
    test_vel = np.linalg.norm(np.diff(test_pos, axis=0), axis=1)

    # Normalise — guard against zero-std (static capture)
    ref_vel  = (ref_vel  - np.mean(ref_vel))  / (np.std(ref_vel)  + 1e-8)
    test_vel = (test_vel - np.mean(test_vel)) / (np.std(test_vel) + 1e-8)

    # Cross-correlation: peak index encodes the lag
    corr     = np.correlate(ref_vel, test_vel, mode='full')
    peak_idx = np.argmax(corr)
    optimal_lag = peak_idx - (len(test_vel) - 1)

    return float(optimal_lag)
