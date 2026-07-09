from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np

import utils
from config import settings

logger = logging.getLogger(__name__)


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


class TrackState(Enum):
    ACTIVE = auto()
    EXPIRED = auto()


@dataclass(slots=True)
class TrackSummary:
    """Cached metadata derived from all frames in a track."""

    frame_count: int
    latest_frame: TrackFrame
    latest_bbox: utils.Bbox
    latest_detection_index: int
    trailing_miss_count: int
    state: TrackState


class Track:
    """Ordered collection of per-frame matches for a single target."""

    def __init__(self, track_id: int, frame_index: int) -> None:
        self.track_id = track_id
        self._frames: list[Optional[TrackFrame]] = [None] * frame_index
        self._update_summary()

    def __len__(self) -> int:
        return self.summary.frame_count

    @property
    def frames(self) -> list[Optional[TrackFrame]]:
        return self._frames

    @property
    def summary(self) -> TrackSummary:
        return self._summary

    @property
    def latest_frame(self) -> TrackFrame:
        if self.summary.latest_frame is None:
            raise ValueError("Track has no detections")
        return self.summary.latest_frame

    @property
    def latest_bbox(self) -> utils.Bbox:
        if self.summary.latest_bbox is None:
            raise ValueError("Track has no detections")
        return self.summary.latest_bbox

    @property
    def latest_detection_index(self) -> int:
        if self.summary.latest_detection_index is None:
            raise ValueError("Track has no detections")
        return self.summary.latest_detection_index

    @property
    def trailing_miss_count(self) -> int:
        if self.summary.trailing_miss_count is not None:
            return self.summary.trailing_miss_count
        raise ValueError("Track has no detections")

    @property
    def state(self) -> TrackState:
        return self.summary.state

    @property
    def is_expired(self) -> bool:
        return self.state is TrackState.EXPIRED

    def score(self, candidate: TrackFrame) -> float:
        if candidate.class_id == self.latest_frame.class_id:
            return bbox_iou(self.latest_bbox, candidate.bbox)
        else:
            return 0.0

    def append(self, frame: Optional[TrackFrame]) -> None:
        self.frames.append(frame)
        self._update_summary()

    def _update_summary(self) -> None:

        # identify info about end of track
        latest_frame = None
        trailing_miss_count = 0
        for frame in reversed(self.frames):
            if frame is None:
                trailing_miss_count += 1
                continue
            latest_frame = frame
            break

        # calculate state
        state = (
            TrackState.EXPIRED
            if trailing_miss_count >= settings.BUFFER_DUR * settings.FPS
            else TrackState.ACTIVE
        )

        self._summary = TrackSummary(
            frame_count=len(self.frames),
            latest_frame=latest_frame,
            latest_bbox=latest_frame.bbox if latest_frame is not None else None,
            latest_detection_index=len(self.frames) - trailing_miss_count - 1,
            trailing_miss_count=trailing_miss_count,
            state=state,
        )


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
        return [track for track in self.tracks if not track.is_expired]

    def _new_track(self, track_frame: TrackFrame, frame_index: int) -> Track:
        track = Track(track_id=len(self.tracks) + 1, frame_index=frame_index)
        logger.info(
            f"({track_frame.frame_hash}) New Track: id = {track.track_id} , class = {track_frame.class_id}"
        )
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
