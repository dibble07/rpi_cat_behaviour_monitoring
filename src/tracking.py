from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import utils
from config import settings


def bbox_iou(box_a: utils.Bbox, box_b: utils.Bbox) -> float:
    """Calculate intersection-over-union for two xyxy boxes."""

    # unpack box coords
    ax1, ay1, ax2, ay2 = box_a.xyxy
    bx1, by1, bx2, by2 = box_b.xyxy

    # determine intersection coord
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # calculate areas
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area


@dataclass(slots=True)
class TrackFrame:
    """Detection snapshot used by the tracker."""

    frame_hash: str
    bbox: utils.Bbox
    class_id: int
    confidence: float


class Track:
    """Ordered collection of per-frame matches for a single target."""

    def __init__(self, track_id: int, frame_index: int) -> None:
        self.track_id = track_id
        self._frames: list[Optional[TrackFrame]] = [None] * frame_index

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def frames(self) -> list[Optional[TrackFrame]]:
        return self._frames

    @property
    def latest_frame(self) -> TrackFrame:
        for frame in reversed(self.frames):
            if frame is not None:
                return frame
        raise ValueError("Track has no detections")

    @property
    def latest_bbox(self) -> utils.Bbox:
        return self.latest_frame.bbox

    @property
    def latest_detection_index(self) -> int:
        return len(self.frames) - self.trailing_miss_count() - 1

    def score(self, candidate: TrackFrame) -> float:
        if candidate.class_id == self.latest_frame.class_id:
            return bbox_iou(self.latest_bbox, candidate.bbox)
        else:
            return 0.0

    def append(self, frame: Optional[TrackFrame]) -> None:
        self.frames.append(frame)

    def trailing_miss_count(self) -> int:
        for i, f in enumerate(reversed(self.frames)):
            if f is not None:
                return i
        return len(self.frames)

    def is_expired(self) -> bool:
        return self.trailing_miss_count() >= settings.BUFFER_DUR * settings.FPS


class TrackManager:
    """Greedy multi-object track assignment."""

    def __init__(self) -> None:
        self.tracks: list[Track] = []

    def __len__(self) -> int:
        lengths = {len(track) for track in self.tracks}
        match len(lengths):
            case 0:
                return 0
            case 1:
                return lengths.pop()
            case _:
                raise ValueError(
                    f"All tracks must be same length. Found: {sorted(lengths)}"
                )

    @property
    def active_tracks(self) -> list[Track]:
        return [track for track in self.tracks if not track.is_expired()]

    def _new_track(self, track_frame: TrackFrame, frame_index: int) -> Track:
        track = Track(track_id=len(self.tracks) + 1, frame_index=frame_index)
        track.append(track_frame)
        return track

    def update(self, candidates: List[TrackFrame]) -> None:

        # store length of tracks before update
        frame_index = len(self)

        # score all track/candidate combinations
        scores = []
        for track_index, track in enumerate(self.tracks):
            for candidate_index, candidate in enumerate(candidates):
                score = track.score(candidate)
                if score >= settings.TRACK_IOU_THRESHOLD:
                    scores.append(
                        (
                            score,
                            track.latest_detection_index,
                            candidate.confidence,
                            track_index,
                            candidate_index,
                        )
                    )
        scores.sort(key=lambda item: (-item[0], -item[1], -item[2]))

        # greedily assign unmatched candidates to unmatched tracks
        matched_tracks, matched_candidates = set(), set()
        for _, _, _, track_index, candidate_index in scores:
            if (
                track_index not in matched_tracks
                and candidate_index not in matched_candidates
            ):
                self.tracks[track_index].append(candidates[candidate_index])
                matched_tracks.add(track_index)
                matched_candidates.add(candidate_index)

        # assign blank to unmatched tracks
        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_tracks:
                track.append(None)

        # create new tracks for unmatched candidates
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index not in matched_candidates:
                self.tracks.append(self._new_track(candidate, frame_index))

    def all_tracks_mask(self, frame_width: int, frame_height: int) -> np.ndarray:
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        for track in self.active_tracks:
            x1, y1, x2, y2 = track.latest_bbox.xyxy
            if x2 >= x1 and y2 >= y1:
                mask[y1 : y2 + 1, x1 : x2 + 1] = 255

        return mask
