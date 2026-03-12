"""
Session — Multi-track session: up to 5 tracks, alignment state, project I/O.

AUDIT FIXES:
  - save_session(): did not call track.state_dict() for rotate_x/y/z
    (that fix lives in track.py) but the session version number was never
    incremented — added version bump to 2 so old files can be detected.
  - load_session(): loader_fn exceptions were not caught — a corrupt or
    moved source file would raise and abort loading ALL remaining tracks
    instead of just skipping the bad slot.
  - load_session(): no validation that data["tracks"] is actually a list —
    a malformed JSON file would raise a confusing TypeError.
  - max_frame: returned 0 when all tracks were None, which is correct, but
    callers compared <= 0 to detect empty — added explicit property note.
  - remove_track(): silently accepted out-of-range slot indices with no
    feedback; now logs a warning.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, List, Optional

from .track import Track

log = logging.getLogger("mocap_studio.core.session")

MAX_TRACKS = 5
SESSION_VERSION = 2   # bumped from 1 when rotate state was added to Track


class Session:
    """Manages up to 5 mocap tracks and session persistence."""

    def __init__(self) -> None:
        self.tracks: List[Optional[Track]] = [None] * MAX_TRACKS
        self.reference_index: int = 0
        self.current_frame:   int = 0
        self._project_path: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def loaded_tracks(self) -> List[Track]:
        """Return only non-None tracks."""
        return [t for t in self.tracks if t is not None]

    # ------------------------------------------------------------------
    @property
    def max_frame(self) -> int:
        """
        Longest (offset-adjusted) frame count across loaded tracks.
        Returns 0 when no tracks are loaded.
        """
        mx = 0
        for t in self.tracks:
            if t is not None:
                end = t.offset + t.frame_count
                mx  = max(mx, int(end))
        return mx

    # ------------------------------------------------------------------
    def load_track(self, slot: int, track: Track) -> None:
        if 0 <= slot < MAX_TRACKS:
            self.tracks[slot] = track
            log.info(
                f"Track loaded into slot {slot}: \"{track.name}\" "
                f"({track.frame_count} frames, {track.skeleton.joint_count} joints)"
            )
        else:
            log.warning(f"load_track: slot {slot} is out of range (0–{MAX_TRACKS-1})")

    # ------------------------------------------------------------------
    def remove_track(self, slot: int) -> None:
        # BUG FIX: silently ignored invalid slots; now warns
        if not (0 <= slot < MAX_TRACKS):
            log.warning(f"remove_track: slot {slot} is out of range (0–{MAX_TRACKS-1})")
            return
        old = self.tracks[slot]
        self.tracks[slot] = None
        log.info(f"Track removed from slot {slot}{': ' + old.name if old else ''}")

    # ------------------------------------------------------------------
    def save_session(self, path: str) -> None:
        """Save session state (not raw data arrays) to JSON."""
        log.info(f"Saving session to: {path}")
        data = {
            "version":         SESSION_VERSION,
            "reference_index": self.reference_index,
            "current_frame":   self.current_frame,
            "tracks": [
                t.state_dict() if t is not None else None
                for t in self.tracks
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._project_path = path
        log.info(
            f"Session saved: {len(self.loaded_tracks)} track(s), "
            f"ref_idx={self.reference_index}, frame={self.current_frame}"
        )

    # ------------------------------------------------------------------
    def load_session(self, path: str, loader_fn: Optional[Callable] = None) -> None:
        """
        Load session from JSON.

        ``loader_fn(source_path) -> Track`` is called for each track that
        has a source file.  If not provided, tracks will be un-populated.

        BUG FIX: individual track load failures no longer abort the entire
        session load — the bad slot is set to None and a warning is logged.
        BUG FIX: validates that data["tracks"] is a list before iterating.
        """
        log.info(f"Loading session from: {path}")
        with open(path, "r") as f:
            data = json.load(f)

        file_version = data.get("version", 1)
        if file_version < SESSION_VERSION:
            log.info(
                f"Session file version {file_version} is older than current "
                f"{SESSION_VERSION} — some fields may default."
            )

        self.reference_index = data.get("reference_index", 0)
        self.current_frame   = data.get("current_frame",   0)

        raw_tracks = data.get("tracks", [])
        # BUG FIX: guard against malformed JSON where "tracks" is not a list
        if not isinstance(raw_tracks, list):
            log.error("Session file has malformed 'tracks' field — expected a list.")
            return

        for i, td in enumerate(raw_tracks):
            if i >= MAX_TRACKS:
                log.warning(f"Session file has more than {MAX_TRACKS} tracks — ignoring extras.")
                break

            if td is None:
                self.tracks[i] = None
                continue

            src = td.get("source_path", "")
            if loader_fn and src and os.path.isfile(src):
                # BUG FIX: catch per-slot load errors so one bad file doesn't abort all
                try:
                    track = loader_fn(src)
                    track.restore_state(td)
                    self.tracks[i] = track
                    log.info(f"  Slot {i}: loaded \"{src}\"")
                except Exception as exc:
                    self.tracks[i] = None
                    log.error(f"  Slot {i}: failed to load \"{src}\" — {exc}")
            else:
                self.tracks[i] = None
                if src and not os.path.isfile(src):
                    log.warning(f"  Slot {i}: source file not found: \"{src}\"")

        self._project_path = path
        log.info(f"Session loaded: {len(self.loaded_tracks)} track(s) restored")
