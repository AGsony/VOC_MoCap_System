"""
Skeleton — Joint hierarchy, parent-child pairs, name lookup.

Represents the skeleton topology for a single mocap track.
No GUI dependency — pure data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Candidate joint names for auto-detecting the alignment/root joint
ALIGNMENT_JOINT_CANDIDATES = [
    "hips", "pelvis", "hip", "root", "center",
    "hips_jnt", "pelvis_jnt", "root_jnt",
]


@dataclass
class Skeleton:
    """Immutable skeleton topology."""

    joint_names: List[str] = field(default_factory=list)
    parent_indices: List[int] = field(default_factory=list)  # -1 = root

    # ----- derived helpers (computed lazily) -----
    _bone_pairs: Optional[List[Tuple[int, int]]] = field(
        default=None, init=False, repr=False
    )
    _name_to_index: Optional[Dict[str, int]] = field(
        default=None, init=False, repr=False
    )

    # ------------------------------------------------------------------
    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    # ------------------------------------------------------------------
    def name_to_index(self, name: str) -> int:
        """Return joint index by name (case-sensitive)."""
        if self._name_to_index is None:
            self._name_to_index = {n: i for i, n in enumerate(self.joint_names)}
        return self._name_to_index.get(name, -1)

    # ------------------------------------------------------------------
    def get_bone_pairs(self) -> List[Tuple[int, int]]:
        """Return list of (parent_idx, child_idx) for line-drawing."""
        if self._bone_pairs is None:
            self._bone_pairs = [
                (self.parent_indices[i], i)
                for i in range(len(self.parent_indices))
                if self.parent_indices[i] >= 0
            ]
        return self._bone_pairs

    # ------------------------------------------------------------------
    @property
    def parent_map(self) -> Dict[int, int]:
        """child_idx → parent_idx dict (excludes roots)."""
        return {i: p for i, p in enumerate(self.parent_indices) if p >= 0}

    # ------------------------------------------------------------------
    def auto_detect_alignment_joint(self) -> str:
        """
        Return the name of the best-guess alignment joint.
        Searches joint names (case-insensitive) against a list of common
        root-joint names.  Falls back to the first root node if none match.
        """
        lower_names = [n.lower() for n in self.joint_names]
        for candidate in ALIGNMENT_JOINT_CANDIDATES:
            for idx, ln in enumerate(lower_names):
                if ln == candidate:
                    return self.joint_names[idx]
        # Fallback: first root joint
        for i, p in enumerate(self.parent_indices):
            if p < 0:
                return self.joint_names[i]
        return self.joint_names[0] if self.joint_names else ""

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize for JSON session file."""
        return {
            "joint_names": self.joint_names,
            "parent_indices": self.parent_indices,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skeleton":
        return cls(
            joint_names=d["joint_names"],
            parent_indices=d["parent_indices"],
        )
