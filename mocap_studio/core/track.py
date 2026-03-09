"""
Track — Single animation track (positions, rotations, metadata).

Wraps the raw data extracted from an FBX (or BVH) file along with
user-configurable alignment state.
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
    name: str = ""                  # filename stem
    source_path: str = ""           # full path to source file

    # --- timing ---
    fps: float = 60.0
    frame_count: int = 0

    # --- skeleton ---
    skeleton: Skeleton = field(default_factory=Skeleton)

    # --- raw data  (set after extraction) ---
    positions: Optional[np.ndarray] = field(default=None, repr=False)
    # shape (F, J, 3) — world-space positions per frame per joint
    quaternions: Optional[np.ndarray] = field(default=None, repr=False)
    # shape (F, J, 4) — world-space quaternions (w, x, y, z)

    # --- user alignment state ---
    offset: float = 0.0             # frame offset (float for sub-frame accuracy)
    scale: float = 1.0              # time scale / stretch
    trim_in: int = 0                # in-point frame
    trim_out: int = 0               # out-point frame (inclusive)
    align_joint: str = ""           # name of alignment joint
    visible: bool = True            # visibility toggle

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
    # set of joint NAMES that should not be rendered

    # --- cache ---
    _aligned_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _aligned_cache_key: Optional[str] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    @property
    def align_joint_index(self) -> int:
        idx = self.skeleton.name_to_index(self.align_joint)
        return idx if idx >= 0 else 0

    # ------------------------------------------------------------------
    @property
    def aligned_positions(self) -> Optional[np.ndarray]:
        """
        Returns positions aligned for comparison.

        Alignment subtracts only the FIRST FRAME's XZ position of the
        alignment joint from all frames.  This:
          - Aligns the starting position so multiple tracks overlap
          - Preserves vertical (Y) height so skeletons stand on the ground
          - Preserves root motion (walking/running trajectory)
        """
        if self.positions is None:
            return None

        # Cache key: align_joint name
        cache_key = self.align_joint
        if self._aligned_cache is not None and self._aligned_cache_key == cache_key:
            return self._aligned_cache

        aji = self.align_joint_index

        # Get frame-0 XZ position of the alignment joint
        # shape (3,) — only subtract X and Z, keep Y=0
        frame0_pos = self.positions[0, aji, :].copy()
        frame0_pos[1] = 0.0  # don't touch Y (height)

        # Broadcast subtract: (F, J, 3) - (3,) → aligns XZ starting position
        result = self.positions - frame0_pos[np.newaxis, np.newaxis, :]

        self._aligned_cache = result
        self._aligned_cache_key = cache_key
        return result

    # ------------------------------------------------------------------
    @property
    def translate(self) -> tuple:
        """Return (tx, ty, tz) tuple for viewer."""
        return (self.translate_x, self.translate_y, self.translate_z)

    # ------------------------------------------------------------------
    def invalidate_cache(self):
        """Force re-computation of aligned positions on next access."""
        self._aligned_cache = None
        self._aligned_cache_key = None

    # ------------------------------------------------------------------
    @property
    def trimmed_range(self) -> range:
        """Frame indices for the trimmed segment."""
        return range(self.trim_in, min(self.trim_out + 1, self.frame_count))

    # ------------------------------------------------------------------
    @property
    def hidden_joint_indices(self) -> Set[int]:
        """Return set of joint indices that are hidden."""
        return {
            self.skeleton.name_to_index(name)
            for name in self.hidden_joints
            if self.skeleton.name_to_index(name) >= 0
        }

    # ------------------------------------------------------------------
    def auto_setup(self) -> None:
        """
        Called right after extraction — sets sane defaults for
        trim, alignment joint, etc.
        """
        self.trim_in = 0
        self.trim_out = max(0, self.frame_count - 1)
        self.align_joint = self.skeleton.auto_detect_alignment_joint()
        self.translate_x = 0.0
        self.translate_y = 0.0
        self.translate_z = 0.0
        self.hidden_joints = set()
        self.invalidate_cache()

    # ------------------------------------------------------------------
    # Serialization (for session save/load)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Serialize user-editable state (not the heavy arrays)."""
        return {
            "source_path": self.source_path,
            "name": self.name,
            "offset": self.offset,
            "trim_in": self.trim_in,
            "trim_out": self.trim_out,
            "align_joint": self.align_joint,
            "visible": self.visible,
            "translate": [self.translate_x, self.translate_y, self.translate_z],
            "hidden_joints": sorted(self.hidden_joints),
        }

    def restore_state(self, d: dict) -> None:
        """Restore user-editable state from dict."""
        self.offset = d.get("offset", 0)
        self.trim_in = d.get("trim_in", 0)
        self.trim_out = d.get("trim_out", self.frame_count - 1)
        self.align_joint = d.get("align_joint", self.align_joint)
        self.visible = d.get("visible", True)
        translate = d.get("translate", [0.0, 0.0, 0.0])
        self.translate_x = translate[0] if len(translate) > 0 else 0.0
        self.translate_y = translate[1] if len(translate) > 1 else 0.0
        self.translate_z = translate[2] if len(translate) > 2 else 0.0
        self.hidden_joints = set(d.get("hidden_joints", []))
        self.invalidate_cache()
