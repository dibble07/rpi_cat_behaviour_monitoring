from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, auto
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from filterpy.kalman import KalmanFilter
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import models

import classification
import utils
from config import settings

logger = logging.getLogger(__name__)


_EMBEDDING_PREPROCESS = models.ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms(
    crop_size=settings.EMBEDDING_IMGSZ,
    resize_size=settings.EMBEDDING_IMGSZ,
)
_embedding_session: ort.InferenceSession = ort.InferenceSession(
    Path("models") / f"{settings.MODEL_EMBEDDING_PATH}.onnx",
    providers=["CPUExecutionProvider"],
)
_embedding_input_name = _embedding_session.get_inputs()[0].name


def embed_image(image_np: np.ndarray) -> np.ndarray:
    """Generate an L2-normalized embedding for an RGB image array using cached ONNX session."""

    image_np = np.asarray(image_np, dtype=np.uint8)

    tensor = (
        _EMBEDDING_PREPROCESS(Image.fromarray(image_np)).unsqueeze(0).to("cpu").half()
    )
    embedding = _embedding_session.run(None, {_embedding_input_name: tensor.numpy()})[0]
    embedding = embedding.astype(np.float32)
    denom = np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)
    embedding = embedding / np.clip(denom, 1e-12, None)
    return embedding[0]


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
    image: np.ndarray
    bbox: utils.Bbox
    object_name: str
    confidence: float
    roi: np.ndarray = field(init=False, repr=False)
    _roi_embedding: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _cat_name_proba: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        h, w = self.image.shape[:2]
        x1, y1, x2, y2 = self.bbox.xyxy
        x1, y1, x2, y2 = utils.expand_bbox_from_bounds(
            x_min=x1,
            x_max=x2,
            y_min=y1,
            y_max=y2,
            image_width=w,
            image_height=h,
            pad=0,
            target_aspect_ratio=1.0,
        )
        self.roi = self.image[y1 : y2 + 1, x1 : x2 + 1].copy()

    @property
    def roi_embedding(self) -> np.ndarray:
        if self._roi_embedding is None:
            start = datetime.now()
            self._roi_embedding = embed_image(self.roi)
            utils.log_timing(logger, "Embedding", start, self.frame_hash)
        return self._roi_embedding

    @property
    def cat_name_proba(self) -> np.ndarray:
        if self._cat_name_proba is None:
            embedding = self.roi_embedding
            start = datetime.now()
            result = classification.classify_embedding(embedding)
            self._cat_name_proba = result["proba"]
            utils.log_timing(logger, "Identification", start, self.frame_hash)
        return self._cat_name_proba


class TrackState(IntEnum):
    NEW = auto()
    ACTIVE = auto()
    STALE = auto()
    EXPIRED = auto()


@dataclass(slots=True)
class TrackSummary:
    """Cached metadata derived from all frames in a track."""

    track_id: int
    frame_count: int
    first_detection_index: int
    latest_detection_index: int
    last_frame: Optional[TrackFrame]
    last_valid_frame: TrackFrame
    state: TrackState
    estimated_bbox: utils.Bbox
    history: list[Optional[tuple[int, int]]]
    cat_name: Optional[str] = None
    cat_conf: Optional[float] = None
    cat_name_entropy: Optional[str] = None
    cat_conf_entropy: Optional[float] = None


class Track:
    """Ordered collection of per-frame matches for a single target."""

    def __init__(
        self, track_id: int, frame_index: int, frame: TrackFrame, frame_hash: str
    ) -> None:
        self.track_id = track_id
        self._first_detection_index = frame_index
        self._frames: list[Optional[TrackFrame]] = [None] * frame_index
        self._frame_hash = frame_hash
        self.append(frame)

    def __len__(self) -> int:
        return self.summary.frame_count

    @property
    def summary(self) -> TrackSummary:
        return self._summary

    def score(self, candidate: TrackFrame) -> float:
        if candidate.object_name == self.summary.last_valid_frame.object_name:
            # bounding box
            iou = bbox_iou(self.summary.last_valid_frame.bbox, candidate.bbox)
            a, b = self.summary.last_valid_frame.bbox.cxcywhn, candidate.bbox.cxcywhn
            centroid_sim = (1 - np.hypot(a[0] - b[0], a[1] - b[1]) / np.sqrt(2.0)) ** 2
            track_area, cand_area = a[2] * a[3], b[2] * b[3]
            size_sim = min(track_area, cand_area) / max(track_area, cand_area)

            # confidence
            conf = np.sqrt(
                self.summary.last_valid_frame.confidence * candidate.confidence
            )

            # visual similarity
            if self.summary.last_valid_frame.object_name == "cat":
                valid_frames = [f for f in self._frames if f is not None]
                ent_wgt = utils.entropy_weights(
                    np.stack([f.cat_name_proba for f in valid_frames])
                )
                embeddings = np.stack([f.roi_embedding for f in valid_frames])
                ref_emb = np.average(embeddings, axis=0, weights=ent_wgt)
                denom = np.linalg.norm(ref_emb, ord=2)
                ref_emb = ref_emb / np.clip(denom, 1e-12, None)
            else:
                ref_emb = self.summary.last_valid_frame.roi_embedding
            visual = (ref_emb @ candidate.roi_embedding) ** 2

            # current track age
            latest_frame_age = (
                self.summary.frame_count - self.summary.latest_detection_index - 1
            )
            recency_score = np.exp(-latest_frame_age / settings.FPS)
            track_age = (
                self.summary.latest_detection_index - self.summary.first_detection_index
            )
            age_score = 1 - np.exp(-track_age / settings.FPS)

            # aggregate component scores
            score = np.average(
                [iou, centroid_sim, conf, visual, recency_score, age_score, size_sim],
                weights=[2, 2, 2, 3, 1, 0.5, 1],
            )
            logger.debug(
                f"({self._frame_hash}) Track {self.track_id} match score (candidate_conf={candidate.confidence:.3f})"
                f": iou={iou:.3f} centroid_sim={centroid_sim:.3f} conf={conf:.3f} visual={visual:.3f} "
                f"recency_score={recency_score:.3f} age_score={age_score:.3f} size_sim={size_sim:.3f} score={score:.3f}"
            )
            return score
        else:
            return -1.0

    def append(self, frame: Optional[TrackFrame]) -> None:
        self._frames.append(frame)
        self._update_summary()

    def _init_kf(self, meas: np.ndarray) -> None:
        kf = KalmanFilter(
            dim_z=4,  # [cx, cy, w, h]
            dim_x=8,  # [cx, cy, w, h, vcx, vcy, vw, vh]
        )
        kf.F = np.eye(8)
        kf.F[:4, 4:] = np.eye(4)  # position += velocity each frame
        kf.H = np.eye(4, 8)  # observe cx, cy, w, h directly
        kf.R[2:, 2:] *= 10.0  # higher measurement noise on size vs position
        kf.P[4:, 4:] *= 1000.0  # high initial uncertainty on velocity
        kf.P *= 10.0
        kf.Q[4:, 4:] *= 0.01  # slow velocity evolution
        kf.x[:4] = meas  # current state
        self._kf = kf

    def _next_bbox_from_kf(self) -> utils.Bbox:
        next_state = self._kf.F @ self._kf.x
        cx, cy, w, h = next_state[:4].flatten()
        return utils.Bbox(
            xyxy=(
                int(round(cx - w / 2)),
                int(round(cy - h / 2)),
                int(round(cx + w / 2)),
                int(round(cy + h / 2)),
            )
        )

    def _update_summary(self) -> None:
        prev_summary = getattr(self, "_summary", None)

        # update kalman filter
        last_frame = self._frames[-1]
        if last_frame is not None:
            meas = np.array(last_frame.bbox.cxcywh).reshape(4, 1)
            if not hasattr(self, "_kf"):
                self._init_kf(meas)
            else:
                self._kf.predict()
                self._kf.update(meas)
        elif hasattr(self, "_kf"):
            self._kf.predict()

        # identify simple info about track
        frame_count = len(self._frames)
        history_frames = self._frames[
            -int(np.ceil(settings.TRACK_HISTORY_DUR * settings.FPS)) :
        ]
        history = [f.bbox.cxcywh[:2] if f is not None else None for f in history_frames]

        # identify info about end of track
        for i, frame in enumerate(reversed(self._frames)):
            if frame is not None:
                last_valid_frame = frame
                latest_detection_index = frame_count - i - 1
                break

        # calculate state transitions
        match getattr(prev_summary, "state", None):
            case None:
                state = TrackState.NEW
            case TrackState.NEW:
                new_frame_count = int(np.ceil(settings.FPS) * settings.TRACK_NEW_DUR)
                frames_init = self._frames[
                    self._first_detection_index : self._first_detection_index
                    + new_frame_count
                ]
                frames_init_valid = [f for f in frames_init if f is not None]
                if (
                    len(frames_init_valid) > new_frame_count / 2
                    and sum(
                        [
                            f.confidence >= settings.TRACK_NEW_CONF_THRESHOLD
                            for f in frames_init_valid
                        ]
                    )
                    > new_frame_count / 4
                ):
                    state = TrackState.ACTIVE
                elif self._first_detection_index + new_frame_count >= frame_count:
                    state = TrackState.NEW
                else:
                    state = TrackState.EXPIRED
            case TrackState.ACTIVE:
                state = (
                    TrackState.ACTIVE
                    if self._frames[-1] is not None
                    else TrackState.STALE
                )
            case TrackState.STALE:
                if self._frames[-1] is not None:
                    state = TrackState.ACTIVE
                elif (
                    frame_count - latest_detection_index - 1
                    >= settings.TRACK_STALE_DUR * settings.FPS
                ):
                    state = TrackState.EXPIRED
                else:
                    state = TrackState.STALE
            case TrackState.EXPIRED:
                state = TrackState.EXPIRED

        # aggregate cat name based weighted by entropy
        if last_frame is not None:
            if last_frame.object_name == "cat":
                probs = np.stack([f.cat_name_proba for f in self._frames if f])
                ent_wgt = utils.entropy_weights(probs)
                probs_avg = np.average(probs, axis=0, weights=ent_wgt)
                cat_id = int(np.argmax(probs_avg))
                cat_name = classification._classifier.classes_[cat_id]
                cat_conf = probs_avg[cat_id]
            else:
                cat_name = None
                cat_conf = None
        else:
            cat_name = prev_summary.cat_name
            cat_conf = prev_summary.cat_conf

        self._summary = TrackSummary(
            track_id=self.track_id,
            frame_count=frame_count,
            first_detection_index=self._first_detection_index,
            last_frame=last_frame,
            last_valid_frame=last_valid_frame,
            latest_detection_index=latest_detection_index,
            history=history,
            state=state,
            estimated_bbox=self._next_bbox_from_kf(),
            cat_name=cat_name,
            cat_conf=cat_conf,
        )

        # log updates
        if prev_summary is None:
            logger.info(
                f"({self._frame_hash}) Track {self.track_id} created: cat={cat_name} state={state.name[0]} object_confidence={last_valid_frame.confidence:.2f}"
            )
        else:
            unchanged_state = prev_summary.state == state
            unchanged_cat = prev_summary.cat_name == cat_name
            log_str = (
                f"({self._frame_hash}) Track {self.track_id} summary update: "
                f"cat={prev_summary.cat_name}->{cat_name} "
                f"state={prev_summary.state.name[0]}->{state.name[0]} "
                f"object_confidence={last_valid_frame.confidence:.2f} "
                f"missed_frames={frame_count - latest_detection_index - 1}"
            )
            if unchanged_state and unchanged_cat:
                logger.debug(log_str)
            else:
                logger.info(log_str)


class TrackManager:
    """Hungarian multi-object track assignment."""

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
    def non_expired_tracks(self) -> list[Track]:
        return [
            track for track in self.tracks if track.summary.state < TrackState.EXPIRED
        ]

    def get_track(self, track_id: int) -> Track:
        matches = [track for track in self.tracks if track.track_id == track_id]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one track with id {track_id}, found {len(matches)}"
            )
        return matches[0]

    def _new_track(
        self, track_frame: TrackFrame, frame_index: int, frame_hash: str
    ) -> Track:
        track = Track(
            track_id=len(self.tracks) + 1,
            frame_index=frame_index,
            frame=track_frame,
            frame_hash=frame_hash,
        )
        return track

    def update(self, candidates: List[TrackFrame], frame_hash: str) -> None:

        # store frame hash on all tracks for logging
        for track in self.tracks:
            track._frame_hash = frame_hash

        # store length of tracks before update
        frame_index = len(self)

        # score assignable track/candidate combinations directly into padded matrix
        assignable_tracks = self.non_expired_tracks
        track_count = len(assignable_tracks)
        candidate_count = len(candidates)
        padded_size = track_count + candidate_count
        padded_scores = (
            np.zeros((padded_size, padded_size)) + settings.TRACK_MATCH_THRESHOLD
        )
        for track_index, track in enumerate(assignable_tracks):
            for candidate_index, candidate in enumerate(candidates):
                padded_scores[track_index, candidate_index] = track.score(candidate)

        # assign tracks to candidates/dummies
        row_ind, col_ind = linear_sum_assignment(padded_scores, maximize=True)

        # assign matched candidates to tracks
        matched_tracks, matched_candidates = set(), set()
        for track_index, candidate_index in zip(row_ind, col_ind):
            if track_index < track_count and candidate_index < candidate_count:
                track = assignable_tracks[track_index]
                track.append(candidates[candidate_index])
                matched_tracks.add(track)
                matched_candidates.add(candidate_index)
                logger.debug(
                    f"({frame_hash}) Track {track.track_id} match: candidate={candidate_index} "
                    f"(conf={candidates[candidate_index].confidence:.3f}) score={padded_scores[track_index, candidate_index]:.3f}"
                )

        # assign blank to unmatched tracks
        for track in self.tracks:
            if track not in matched_tracks:
                track.append(None)
                logger.debug(f"Track {track.track_id} no matching candidate")

        # create new tracks for unmatched candidates
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index not in matched_candidates:
                self.tracks.append(self._new_track(candidate, frame_index, frame_hash))

    def all_tracks_mask(self, frame_width: int, frame_height: int) -> np.ndarray:
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        for track in self.non_expired_tracks:
            x1, y1, x2, y2 = track.summary.estimated_bbox.xyxy
            if x2 >= x1 and y2 >= y1:
                mask[y1 : y2 + 1, x1 : x2 + 1] = 255

        return mask
