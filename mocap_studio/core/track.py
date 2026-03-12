"""
Track — Single animation track (positions, rotations, metadata).

Wraps the raw data extracted from an FBX (or BVH) file along with
user-configurable alignment state.

AUDIT FIXES:
  - scale=0.0 default would cause ZeroDivisionError in interpolation; default is now 1.0
  - aligned_positions property silently ignored translate/rotate offsets applied in
    the viewer — callers expecting "world" positions got raw aligned data instead.
    Property now clearly documented as XZ-origin-aligned only (no viewer transforms).
  - state_dict() did not serialise rotate_x/y/z — loading a saved session lost
    all user rotation adjustments silently.
  - restore_state() did not restore rotate_x/y/z for the same reason.
  - hidden_joints was a mutable set default shared across instances in some
    Python versions — now always initialised fresh in auto_setup().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np

from .skeleton import Skeleton


@dataclass
class Track:
    """One loaded mocap animation."""

    # --- identity ---
    name: str = ""
    source_path: str = ""

    # --- timing ---
    fps: float = 60.0
    frame_count: int = 0

    # --- skeleton ---
    skeleton: Skeleton = field(default_factory=Skeleton)

    # --- raw data ---
    positions: Optional[np.ndarray] = field(default=None, repr=False)
    # shape (F, J, 3) — world-space positions per frame per joint
    quaternions: Optional[np.ndarray] = field(default=None, repr=False)
    # shape (F, J, 4) — world-space quaternions (w, x, y, z)

    # --- rest pose injection ---
    rest_pose_positions: Optional[np.ndarray]   = field(default=None, repr=False)
    rest_pose_quaternions: Optional[np.ndarray] = field(default=None, repr=False)
    rest_pose_name: str = ""

    # --- user alignment state ---
    offset: float = 0.0
    # BUG FIX: was defaulting to 1.0 but field order put it after mutable defaults;
    # kept explicit default here for clarity.
    scale: float = 1.0
    trim_in: int = 0
    trim_out: int = 0
    align_joint: str = ""
    visible: bool = True

    # --- 3D position offset ---
    translate_x: float = 0.0
    translate_y: float = 0.0
    translate_z: float = 0.0

    # --- 3D rotation offset (degrees) ---
    rotate_x: float = 0.0
    rotate_y: float = 0.0
    rotate_z: float = 0.0

    # --- joint visibility ---
    hidden_joints: Set[str] = field(default_factory=set)

    # --- cache ---
    _aligned_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _aligned_cache_key: Optional[str]    = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    @property
    def align_joint_index(self) -> int:
        idx = self.skeleton.name_to_index(self.align_joint)
        return idx if idx >= 0 else 0

    # ------------------------------------------------------------------
    @property
    def aligned_positions(self) -> Optional[np.ndarray]:
        """
        Return positions with the frame-0 XZ position of the alignment joint
        subtracted from all frames.

        NOTE: This does NOT apply translate_x/y/z or rotate_x/y/z.
        Those viewer-space transforms must be applied separately (see
        export scripts and the 3D viewer render path).
        """
        if self.positions is None:
            return None

        cache_key = self.align_joint
        if self._aligned_cache is not None and self._aligned_cache_key == cache_key:
            return self._aligned_cache

        aji = self.align_joint_index
        frame0_pos = self.positions[0, aji, :].copy()
        frame0_pos[1] = 0.0   # preserve vertical height

        result = self.positions - frame0_pos[np.newaxis, np.newaxis, :]

        self._aligned_cache     = result
        self._aligned_cache_key = cache_key
        return result

    # ------------------------------------------------------------------
    @property
    def translate(self) -> tuple:
        return (self.translate_x, self.translate_y, self.translate_z)

    # ------------------------------------------------------------------
    def invalidate_cache(self):
        self._aligned_cache     = None
        self._aligned_cache_key = None

    # ------------------------------------------------------------------
    @property
    def trimmed_range(self) -> range:
        return range(self.trim_in, min(self.trim_out + 1, self.frame_count))

    # ------------------------------------------------------------------
    @property
    def hidden_joint_indices(self) -> Set[int]:
        return {
            self.skeleton.name_to_index(name)
            for name in self.hidden_joints
            if self.skeleton.name_to_index(name) >= 0
        }

    # ------------------------------------------------------------------
    def auto_setup(self) -> None:
        """Called right after extraction — sets sane defaults."""
        self.trim_in  = 0
        self.trim_out = max(0, self.frame_count - 1)
        self.align_joint = self.skeleton.auto_detect_alignment_joint()
        self.translate_x = self.translate_y = self.translate_z = 0.0
        self.rotate_x    = self.rotate_y    = self.rotate_z    = 0.0
        # BUG FIX: always create a fresh set, never share across instances
        self.hidden_joints = set()
        self.invalidate_cache()

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """
        Serialise user-editable state.

        BUG FIX: rotate_x/y/z were not included — user rotation adjustments
        were silently lost on session save/load.
        """
        return {
            "source_path": self.source_path,
            "name":        self.name,
            "offset":      self.offset,
            "scale":       self.scale,
            "trim_in":     self.trim_in,
            "trim_out":    self.trim_out,
            "align_joint": self.align_joint,
            "visible":     self.visible,
            "translate":   [self.translate_x, self.translate_y, self.translate_z],
            "rotate":      [self.rotate_x,    self.rotate_y,    self.rotate_z],
            "hidden_joints": sorted(self.hidden_joints),
        }

    def restore_state(self, d: dict) -> None:
        """
        Restore user-editable state from dict.

        BUG FIX: rotate_x/y/z were not restored — now reads 'rotate' key
        with backward-compatible fallback to [0,0,0] for older session files.
        """
        self.offset      = d.get("offset",  0.0)
        self.scale       = d.get("scale",   1.0)
        self.trim_in     = d.get("trim_in", 0)
        self.trim_out    = d.get("trim_out", self.frame_count - 1)
        self.align_joint = d.get("align_joint", self.align_joint)
        self.visible     = d.get("visible",  True)

        translate = d.get("translate", [0.0, 0.0, 0.0])
        self.translate_x = translate[0] if len(translate) > 0 else 0.0
        self.translate_y = translate[1] if len(translate) > 1 else 0.0
        self.translate_z = translate[2] if len(translate) > 2 else 0.0

        # BUG FIX: rotate was never saved or restored
        rotate = d.get("rotate", [0.0, 0.0, 0.0])
        self.rotate_x = rotate[0] if len(rotate) > 0 else 0.0
        self.rotate_y = rotate[1] if len(rotate) > 1 else 0.0
        self.rotate_z = rotate[2] if len(rotate) > 2 else 0.0

        self.hidden_joints = set(d.get("hidden_joints", []))
        self.invalidate_cache()
