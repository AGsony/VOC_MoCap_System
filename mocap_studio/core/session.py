"""
Session — Multi-track session: up to 5 tracks, alignment state, project I/O.

The Session is the top-level data object that the GUI and the scripting
engine both operate on.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from .track import Track

log = logging.getLogger("mocap_studio.core.session")


MAX_TRACKS = 5


class Session:
    """Manages up to 5 mocap tracks and session persistence."""

    def __init__(self) -> None:
        self.tracks: List[Optional[Track]] = [None] * MAX_TRACKS
        self.reference_index: int = 0
        self.current_frame: int = 0
        self._project_path: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def loaded_tracks(self) -> List[Track]:
        """Return only non-None tracks."""
        return [t for t in self.tracks if t is not None]

    # ------------------------------------------------------------------
    @property
    def max_frame(self) -> int:
        """Longest (offset-adjusted) frame count across loaded tracks."""
        mx = 0
        for t in self.tracks:
            if t is not None:
                end = t.offset + t.frame_count
                mx = max(mx, end)
        return mx

    # ------------------------------------------------------------------
    def load_track(self, slot: int, track: Track) -> None:
        """Load a track into one of the 5 slots (0..4)."""
        if 0 <= slot < MAX_TRACKS:
            self.tracks[slot] = track
            log.info(f"Track loaded into slot {slot}: \"{track.name}\" ({track.frame_count} frames, {track.skeleton.joint_count} joints)")

    # ------------------------------------------------------------------
    def remove_track(self, slot: int) -> None:
        if 0 <= slot < MAX_TRACKS:
            old = self.tracks[slot]
            self.tracks[slot] = None
            log.info(f"Track removed from slot {slot}{': ' + old.name if old else ''}")

    # ------------------------------------------------------------------
    # Project persistence (JSON)
    # ------------------------------------------------------------------
    def save_session(self, path: str) -> None:
        """Save session state (not data) to JSON."""
        log.info(f"Saving session to: {path}")
        data = {
            "version": 1,
            "reference_index": self.reference_index,
            "current_frame": self.current_frame,
            "tracks": [],
        }
        for t in self.tracks:
            if t is not None:
                data["tracks"].append(t.state_dict())
            else:
                data["tracks"].append(None)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._project_path = path
        loaded = len(self.loaded_tracks)
        log.info(f"Session saved: {loaded} track(s), ref_idx={self.reference_index}, frame={self.current_frame}")

    # ------------------------------------------------------------------
    def load_session(self, path: str, loader_fn=None) -> None:
        """
        Load session from JSON.

        ``loader_fn(source_path) -> Track`` is called for each track
        that has a source file.  If not provided, tracks will be
        un-populated (state-only restore).
        """
        log.info(f"Loading session from: {path}")
        with open(path, "r") as f:
            data = json.load(f)

        self.reference_index = data.get("reference_index", 0)
        self.current_frame = data.get("current_frame", 0)
        log.debug(f"Session version={data.get('version')}, ref_idx={self.reference_index}, frame={self.current_frame}")

        for i, td in enumerate(data.get("tracks", [])):
            if i >= MAX_TRACKS:
                log.warning(f"Session file has more than {MAX_TRACKS} tracks, ignoring extras.")
                break
            if td is None:
                self.tracks[i] = None
                log.debug(f"  Slot {i}: empty")
                continue
            src = td.get("source_path", "")
            if loader_fn and src and os.path.isfile(src):
                log.info(f"  Slot {i}: loading {src}")
                track = loader_fn(src)
                track.restore_state(td)
                self.tracks[i] = track
            else:
                self.tracks[i] = None
                if src:
                    log.warning(f"  Slot {i}: source file not found: {src}")
                else:
                    log.debug(f"  Slot {i}: no source path")

        self._project_path = path
        loaded = len(self.loaded_tracks)
        log.info(f"Session loaded: {loaded} track(s) restored")
