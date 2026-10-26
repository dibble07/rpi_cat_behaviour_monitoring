from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, auto
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
import onnxruntime as ort
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

import utils
from config import (
    METADATA_DIR,
    SYSTEM,
    TIMESTAMP_FORMAT,
    TRACK_SUMMARIES_PATH,
    load_track_summaries,
    settings,
)
from shared import cam
from video_io import Cv2Camera, FfmpegWriter

logger = logging.getLogger(__name__)


CAT_CLASSIFICATION_MODEL = joblib.load(Path("models") / "identification.joblib")


def classify_embedding(embedding: np.ndarray) -> dict:
    """Identify which cat an embedding most likely belongs to"""
    embedding = np.asarray(embedding, dtype=np.float32)

    # reshape to 2D if needed
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    # get probabilities and index
    proba = CAT_CLASSIFICATION_MODEL.predict_proba(embedding)[0]
    cat_id = int(np.argmax(proba))

    return {
        "cat_name": CAT_CLASSIFICATION_MODEL.classes_[cat_id],
        "confidence": float(proba[cat_id]),
        "proba": proba,
    }


class VideoHashMap:
    def __init__(self) -> None:
        self._videos: dict[str, dict] = {}

    def __contains__(self, frame_hash: str) -> bool:
        return any(frame_hash in v["hashes"] for v in self._videos.values())

    def append(
        self,
        writer: FfmpegWriter,
        frame_hash: str,
    ) -> None:
        """Record that a frame has been written to a writer's video"""

        # reject frames already in a video
        if frame_hash in self:
            raise ValueError(f"Duplicate frame hash in video map: {frame_hash}")

        # add frame hash, registering the video on first use
        video_name = os.path.basename(writer.output_path.replace(".tmp.", "."))
        if video_name not in self._videos:
            self._videos[video_name] = {
                "initial_dt_tm": datetime.strptime(
                    writer.init_timestamp, TIMESTAMP_FORMAT
                ),
                "hashes": [],
            }
        self._videos[video_name]["hashes"].append(frame_hash)

    def get(self, frame_hash: str) -> dict:
        """Locate the video and position of a written frame"""
        for video_name, video in self._videos.items():
            if frame_hash in video["hashes"]:
                return {
                    "video_name": video_name,
                    "video_start_dt_tm": video["initial_dt_tm"],
                    "video_hash_index": video["hashes"].index(frame_hash),
                }
        raise KeyError(f"Frame hash not found in video map: {frame_hash}")


_EMBEDDING_MODEL_PATH = Path("models") / f"{settings.MODEL_EMBEDDING_PATH}.onnx"
_EMBEDDING_SESSION = ort.InferenceSession(
    _EMBEDDING_MODEL_PATH,
    providers=["CPUExecutionProvider"],
)
_EMBEDDING_INPUT_NAME = _EMBEDDING_SESSION.get_inputs()[0].name


@lru_cache(maxsize=1)
def embedding_model_sha256() -> str:
    """Fingerprint the embedding model so behaviour models can check compatibility"""
    return hashlib.sha256(_EMBEDDING_MODEL_PATH.read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def behaviour_session(model_path: str) -> tuple[ort.InferenceSession, dict]:
    """Load a behaviour model and its config, checking embedding compatibility"""

    # load single-threaded inference session
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        model_path, sess_options=options, providers=["CPUExecutionProvider"]
    )

    # read config and check it matches the embedding model
    metadata = session.get_modelmeta().custom_metadata_map
    config = json.loads(metadata["behaviour_config"])
    if config["embedding_model_sha256"] != embedding_model_sha256():
        raise ValueError("Behaviour model requires a different embedding model")
    return session, config


def classify_behaviour(
    embedding: np.ndarray,
    hidden_state: np.ndarray | None = None,
    model_path: str | Path = Path("models") / f"{settings.MODEL_BEHAVIOUR_PATH}.onnx",
) -> dict:
    """Predict a cat's behaviour for one frame, carrying recurrent state between frames"""

    # start from a reset state when none is given
    session, config = behaviour_session(str(Path(model_path).resolve()))
    if hidden_state is None:
        hidden_state = np.zeros(config["hidden_channels"], dtype=np.float32)

    # run model and pick the most probable behaviour
    logits, next_state = session.run(
        ["logits", "next_hidden_state"],
        {
            "embedding": embedding[None],
            "hidden_state": hidden_state[None],
        },
    )
    probabilities = np.exp(logits[0] - logits[0].max())
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    return {
        "behaviour": config["behaviours"][index],
        "confidence": float(probabilities[index]),
        "next_hidden_state": next_state[0],
    }


def embed_image(image: np.ndarray) -> np.ndarray:
    """Generate an L2-normalised embedding for an RGB image array using cached ONNX session"""

    # pad to square with grey filler
    h, w = image.shape[:2]
    if h != w:
        side = max(h, w)
        padded = np.full((side, side, image.shape[2]), 128, dtype=image.dtype)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        padded[y_off : y_off + h, x_off : x_off + w] = image
        image = padded

    # resize image
    imgsz = settings.EMBEDDING_IMGSZ
    cropped = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    # imagenet normalisation
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img = (cropped / 255.0 - mean) / std
    tensor = img.transpose(2, 0, 1)[np.newaxis].astype(np.float16)

    # create and normalise embedding
    embedding = _EMBEDDING_SESSION.run(None, {_EMBEDDING_INPUT_NAME: tensor})[0].astype(
        np.float32
    )
    denom = np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)
    embedding = embedding / np.clip(denom, 1e-12, None)
    return embedding[0]


@dataclass(slots=True)
class TrackFrame:
    """Detection snapshot used by the tracker"""

    frame_hash: str
    image: np.ndarray | None
    bbox: utils.Bbox
    object_name: str
    confidence: float
    frame_wh: tuple[int, int] = field(init=False)
    roi: np.ndarray | None = field(init=False, repr=False)
    _roi_embedding: np.ndarray | None = field(init=False, default=None, repr=False)
    _cat_name_proba: np.ndarray | None = field(init=False, default=None, repr=False)
    _cat_name_proba_adjusted: np.ndarray | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        if self.image is None:
            raise ValueError("TrackFrame.image is required at initialisation")

        # crop a square region of interest around the bbox
        self.frame_wh = tuple(reversed(self.image.shape[:2]))
        x1, y1, x2, y2 = self.bbox.xyxy
        x1, y1, x2, y2 = utils.expand_bbox_from_bounds(
            x_min=x1,
            x_max=x2,
            y_min=y1,
            y_max=y2,
            image_width=self.frame_wh[0],
            image_height=self.frame_wh[1],
            pad=0,
            target_aspect_ratio=1.0,
        )
        self.roi = self.image[y1 : y2 + 1, x1 : x2 + 1].copy()

    @property
    def roi_embedding(self) -> np.ndarray:
        """Embedding of the region of interest, freeing images on the Pi once computed"""
        if self._roi_embedding is None:
            if self.roi is None:
                raise RuntimeError("TrackFrame ROI is not available for embedding")
            start = datetime.now()
            self._roi_embedding = embed_image(self.roi)
            utils.log_timing(logger, "Embedding", start, self.frame_hash)
            if SYSTEM == "Linux":
                self.roi = None
                self.image = None
        return self._roi_embedding

    @property
    def cat_name_proba(self) -> np.ndarray:
        """Cat identity probabilities for the region of interest"""
        if self._cat_name_proba is None:
            embedding = self.roi_embedding
            start = datetime.now()
            result = classify_embedding(embedding)
            self._cat_name_proba = result["proba"]
            utils.log_timing(logger, "Identification", start, self.frame_hash)
        return self._cat_name_proba

    @property
    def cat_name_proba_adjusted(self) -> np.ndarray:
        """Cat identity probabilities adjusted for other cats in the frame"""
        if self._cat_name_proba_adjusted is None:
            raise RuntimeError("Adjusted cat probabilities are not available")
        return self._cat_name_proba_adjusted

    @cat_name_proba_adjusted.setter
    def cat_name_proba_adjusted(self, probabilities: np.ndarray) -> None:
        self._cat_name_proba_adjusted = probabilities


class TrackState(IntEnum):
    NEW = auto()
    ACTIVE = auto()
    STALE = auto()
    EXPIRED = auto()

    def __bool__(self) -> bool:
        return self is not TrackState.EXPIRED


@dataclass(slots=True)
class TrackSummary:
    """Cached metadata derived from all frames in a track"""

    track_id: int
    frame_count: int
    first_detection_index: int
    first_detection_hash: str
    latest_detection_index: int
    last_detection_hash: str
    last_frame: TrackFrame | None
    last_valid_frame: TrackFrame
    object_name: str
    object_valid: bool
    state: TrackState
    estimated_bbox: utils.Bbox
    history: list[tuple | None]
    confirmed: bool = False
    cat_name: str | None = None
    cat_conf: float | None = None

    def __bool__(self) -> bool:
        return self.confirmed and bool(self.state)


class Track:
    """Ordered collection of per-frame matches for a single target"""

    def __init__(
        self,
        track_id: int,
        frame_index: int,
        frame: TrackFrame,
        frame_hash: str,
        frame_image: np.ndarray,
    ) -> None:
        self.track_id = track_id
        self._first_detection_index = frame_index
        self._first_detection_hash = frame_hash
        self.frames: list[TrackFrame | None] = []
        self.frame_hash = frame_hash
        self.expired_at_frame_count: int | None = None
        self.behaviour_history: dict[str, str] = {}
        self._behaviour_hidden_state: np.ndarray | None = None
        self.excluded_object_sample: tuple[float, str, np.ndarray] | None = None
        self.append(frame, frame_image)

    def __len__(self) -> int:
        return self.summary.frame_count

    def __bool__(self) -> bool:
        return bool(self.summary)

    def score(self, candidate: TrackFrame) -> float:
        """Score how well a candidate detection continues this track"""
        if candidate.object_name == self.summary.object_name:
            # bounding box
            iou = centroid_sim = size_sim = 0.0
            for ref_bbox in [
                self.summary.last_valid_frame.bbox,
                self.summary.estimated_bbox,
            ]:
                iou = max(iou, utils.bbox_overlap(ref_bbox, candidate.bbox)[0])
                a, b = ref_bbox.cxcywhn, candidate.bbox.cxcywhn
                centroid_sim = max(
                    centroid_sim,
                    (1 - np.hypot(a[0] - b[0], a[1] - b[1]) / np.sqrt(2.0)) ** 2,
                )
                track_area, cand_area = a[2] * a[3], b[2] * b[3]
                size_sim = max(
                    size_sim, min(track_area, cand_area) / max(track_area, cand_area)
                )

            # confidence
            conf = np.sqrt(
                self.summary.last_valid_frame.confidence * candidate.confidence
            )

            # visual similarity
            if self.summary.object_name == "cat":
                valid_frames = [f for f in self.frames if f is not None]
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
            track_age = self.summary.latest_detection_index
            age_score = 1 - np.exp(-track_age / settings.FPS)

            # aggregate component scores
            score = np.average(
                [iou, centroid_sim, conf, visual, recency_score, age_score, size_sim],
                weights=[2, 2, 2, 3, 1, 0.5, 1],
            )
            logger.debug(
                f"({self.frame_hash}) Track {self.track_id} match score (candidate_conf={candidate.confidence:.3f})"
                f": iou={iou:.3f} centroid_sim={centroid_sim:.3f} conf={conf:.3f} visual={visual:.3f} "
                f"recency_score={recency_score:.3f} age_score={age_score:.3f} size_sim={size_sim:.3f} score={score:.3f}"
            )
            return score
        else:
            return -1.0

    def append(
        self, frame: TrackFrame | None, frame_image: np.ndarray | None = None
    ) -> None:
        """Add a frame's match, or a gap, and refresh the track summary"""

        # keep the lowest-confidence excluded-object frame as a training sample
        if (
            frame is not None
            and frame.object_name in settings.EXCLUDED_OBJECTS
            and frame.confidence < settings.TRACK_NEW_CONF_THRESHOLD + 0.01
            and frame_image is not None
        ):
            if (
                self.excluded_object_sample is None
                or frame.confidence < self.excluded_object_sample[0]
            ):
                self.excluded_object_sample = (
                    frame.confidence,
                    frame.frame_hash,
                    frame_image.copy(),
                )

        # update behaviour prediction for cat frames
        if frame is not None and frame.object_name == "cat":
            embedding = frame.roi_embedding
            start = datetime.now()
            result = classify_behaviour(embedding, self._behaviour_hidden_state)
            self._behaviour_hidden_state = result["next_hidden_state"]
            self.behaviour_history[frame.frame_hash] = result["behaviour"]
            utils.log_timing(logger, "Behaviour", start, frame.frame_hash)

        # store frame and refresh summary
        self.frames.append(frame)
        self._update_summary()

    def _init_kf(self, meas: np.ndarray) -> None:
        """Create a constant-velocity Kalman filter for the bbox"""
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

    def _next_bbox_from_kf(self, frame_wh: tuple[int, int]) -> utils.Bbox:
        """Predict the next frame's bbox from the Kalman filter"""
        next_state = self._kf.F @ self._kf.x
        next_state[2:4, 0] = np.maximum(next_state[2:4, 0], 2)
        cx, cy, w, h = next_state[:4].flatten()
        return utils.Bbox(
            xyxy=(
                int(round(cx - w / 2)),
                int(round(cy - h / 2)),
                int(round(cx + w / 2)),
                int(round(cy + h / 2)),
            ),
            frame_wh=frame_wh,
        )

    def _update_summary(self) -> None:
        """Recalculate the track summary after a frame is added"""
        prev_summary = getattr(self, "summary", None)

        # update kalman filter
        last_frame = self.frames[-1]
        if last_frame is not None:
            meas = np.array(last_frame.bbox.cxcywh).reshape(4, 1)
            if not hasattr(self, "_kf"):
                self._init_kf(meas)
            else:
                self._kf.predict()
                self._kf.update(meas)
        elif hasattr(self, "_kf"):
            self._kf.predict()

        # keep predicted sizes at a minimum and stop them shrinking further
        if hasattr(self, "_kf"):
            sizes = self._kf.x[2:4, 0]
            at_minimum = sizes < 2
            sizes[at_minimum] = 2
            size_velocities = self._kf.x[6:8, 0]
            size_velocities[at_minimum & (size_velocities < 0)] = 0

        # identify simple info about track
        frame_count = len(self.frames)
        history_duration = int(np.ceil(settings.TRACK_HISTORY_DUR * settings.FPS))
        history_frames = self.frames[-history_duration:]
        history = [f.bbox.cxcywh[:2] if f is not None else None for f in history_frames]

        # identify info about end of track
        latest_detection_index = max(i for i, f in enumerate(self.frames) if f)
        last_valid_frame = self.frames[latest_detection_index]

        # calculate state transitions
        match getattr(prev_summary, "state", None):
            case None:
                state = TrackState.NEW
            case TrackState.NEW:
                new_frame_count = settings.TRACK_NEW_FRAMES
                frames_init_valid = [f for f in self.frames[:new_frame_count] if f]
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
                elif new_frame_count >= frame_count:
                    state = TrackState.NEW
                else:
                    state = TrackState.EXPIRED
            case TrackState.ACTIVE:
                state = (
                    TrackState.ACTIVE
                    if self.frames[-1] is not None
                    else TrackState.STALE
                )
            case TrackState.STALE:
                if self.frames[-1] is not None:
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
        confirmed = getattr(prev_summary, "confirmed", False) or state in {
            TrackState.ACTIVE,
            TrackState.STALE,
        }

        # aggregate cat name based weighted by entropy
        if last_frame:
            if last_frame.object_name == "cat":
                probs = np.stack([f.cat_name_proba_adjusted for f in self.frames if f])
                ent_wgt = utils.entropy_weights(probs)
                probs_avg = np.average(probs, axis=0, weights=ent_wgt)
                cat_id = int(np.argmax(probs_avg))
                cat_name = CAT_CLASSIFICATION_MODEL.classes_[cat_id]
                cat_conf = probs_avg[cat_id]
            else:
                cat_name = None
                cat_conf = None
        else:
            cat_name = prev_summary.cat_name
            cat_conf = prev_summary.cat_conf

        # record when the track expired
        if not state and prev_summary.state:
            self.expired_at_frame_count = frame_count

        # store new summary
        frame_wh = last_valid_frame.frame_wh
        self.summary = TrackSummary(
            track_id=self.track_id,
            frame_count=frame_count,
            first_detection_index=self._first_detection_index,
            first_detection_hash=self._first_detection_hash,
            last_frame=last_frame,
            last_valid_frame=last_valid_frame,
            object_name=last_valid_frame.object_name,
            object_valid=(
                last_valid_frame.object_name not in settings.EXCLUDED_OBJECTS
            ),
            latest_detection_index=latest_detection_index,
            last_detection_hash=last_valid_frame.frame_hash,
            history=history,
            state=state,
            estimated_bbox=self._next_bbox_from_kf(frame_wh),
            confirmed=confirmed,
            cat_name=cat_name,
            cat_conf=cat_conf,
        )

        # log updates
        if prev_summary is None:
            logger.info(
                f"({self.frame_hash}) Track {self.track_id} created: cat={cat_name} state={state.name[0]} object_confidence={last_valid_frame.confidence:.2f}"
            )
        else:
            unchanged_state = prev_summary.state == state
            unchanged_cat = prev_summary.cat_name == cat_name
            log_str = (
                f"({self.frame_hash}) Track {self.track_id} summary update: "
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
    """Hungarian multi-object track assignment"""

    def __init__(self) -> None:
        self.manager_id = datetime.now().strftime(TIMESTAMP_FORMAT)
        self.tracks: list[Track] = []
        self._next_track_id = 1

    def _export_track_summary(
        self,
        track: Track,
        video_hash_map: VideoHashMap,
    ) -> None:
        """Save a finished track's summary and per-frame annotations"""

        # find track frames that were written to video
        video_track_hashes = [
            f.frame_hash for f in track.frames if f and f.frame_hash in video_hash_map
        ]
        if track.summary.first_detection_hash in video_hash_map and video_track_hashes:

            # append track summary to the summaries file
            start_match = video_hash_map.get(track.summary.first_detection_hash)
            end_match = video_hash_map.get(video_track_hashes[-1])
            start_offset_s = start_match["video_hash_index"] / settings.FPS
            end_offset_s = end_match["video_hash_index"] / settings.FPS
            row = {
                "manager_id": self.manager_id,
                "track_id": track.track_id,
                "video_name": start_match["video_name"],
                "track_elapsed_start_s": start_offset_s,
                "track_elapsed_end_s": end_offset_s,
                "cat_id": track.summary.cat_name,
                "object_name": track.summary.object_name,
                "behaviours": list(set(track.behaviour_history.values())),
                "track_start_dt_tm": (
                    start_match["video_start_dt_tm"] + timedelta(seconds=start_offset_s)
                ).isoformat(timespec="microseconds"),
            }
            summaries = load_track_summaries()
            summaries.append(row)
            TRACK_SUMMARIES_PATH.write_text(
                json.dumps(summaries, default=str, indent=4)
            )

            # export per-frame annotation info to a track-specific file
            annotations = {
                str(video_hash_map.get(f.frame_hash)["video_hash_index"]): {
                    "bbox": f.bbox.cxcywhn,
                    "behaviour": track.behaviour_history.get(f.frame_hash),
                }
                for f in track.frames
                if f and f.frame_hash in video_hash_map
            }
            track_filename = f"track-{self.manager_id}-{track.track_id}.json"
            with open(os.path.join(METADATA_DIR, track_filename), "w") as f:
                json.dump(annotations, f, indent=4)

            # export full track data for behaviour model training
            if SYSTEM != "Linux" and isinstance(cam, Cv2Camera):
                latest_detection_index = track.summary.latest_detection_index
                frames = track.frames[: latest_detection_index + 1]
                frame_hashes = [f.frame_hash if f else None for f in frames]
                images = {f.frame_hash: f.roi for f in frames if f}
                embeddings = {f.frame_hash: f.roi_embedding for f in frames if f}
                full_track = {
                    "video_path": cam.video_path,
                    "cat_id": track.summary.cat_name,
                    "frame_hashes": frame_hashes,
                    "images": images,
                    "embeddings": embeddings,
                    "embedding_model_sha256": embedding_model_sha256(),
                }
                joblib.dump(
                    full_track,
                    os.path.join(
                        METADATA_DIR, track_filename.replace(".json", ".joblib")
                    ),
                )
        else:
            logger.warning(
                f"Track {track.track_id} has frames that are not in the video hash map."
            )

    def __len__(self) -> int:
        lengths = {len(t) + t.summary.first_detection_index for t in self.tracks}
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
        """Tracks that have not yet expired"""
        return [t for t in self.tracks if t.summary.state]

    def get_track(self, track_id: int) -> Track:
        """Find the single track with the given id"""
        matches = [t for t in self.tracks if t.track_id == track_id]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one track with id {track_id}, found {len(matches)}"
            )
        return matches[0]

    def update(
        self,
        candidates: list[TrackFrame],
        frame_hash: str,
        video_hash_map: VideoHashMap,
        frame_image: np.ndarray,
    ) -> None:
        """Match a frame's detections to tracks, then start and prune tracks"""

        # adjust identity probabilities based on other ROIs in this frame
        cat_candidates = [f for f in candidates if f.object_name == "cat"]
        if cat_candidates:
            probs = np.stack([f.cat_name_proba for f in cat_candidates])
            adjusted = utils.adjust_probabilities_by_context(probs)
            for frame, probabilities in zip(cat_candidates, adjusted):
                frame.cat_name_proba_adjusted = probabilities

        # store frame hash on all tracks for logging
        for track in self.tracks:
            track.frame_hash = frame_hash

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
                track.append(candidates[candidate_index], frame_image)
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
                self.tracks.append(
                    Track(
                        track_id=self._next_track_id,
                        frame_index=frame_index,
                        frame=candidate,
                        frame_hash=frame_hash,
                        frame_image=frame_image,
                    )
                )
                self._next_track_id += 1

        # prune expired tracks that have been processed by the recording buffer
        n_tracks = len(self.tracks)
        self.remove_tracks(
            "expired",
            video_hash_map=video_hash_map,
        )
        if n_pruned := n_tracks - len(self.tracks):
            logger.debug(f"({frame_hash}) Pruned {n_pruned} expired track(s)")

    def remove_tracks(
        self,
        selection: str,
        video_hash_map: VideoHashMap,
    ) -> None:
        """Remove tracks based on the selection criteria"""

        # select tracks to remove
        match selection:
            case "expired":
                tracks_to_delete = [
                    t
                    for t in self.tracks
                    if not t.summary.state
                    and t.summary.frame_count - t.expired_at_frame_count
                    > settings.TRACK_NEW_FRAMES
                ]
            case "all":
                tracks_to_delete = self.tracks.copy()
            case _:
                raise ValueError(f"Invalid selection for pruning tracks: {selection}")

        # remove selected tracks
        for track in tracks_to_delete:
            if track.summary.confirmed and track.excluded_object_sample is not None:
                conf, frame_hash, image = track.excluded_object_sample
                sample_path = (
                    METADATA_DIR
                    / f"excluded-{self.manager_id}-track-{track.track_id}-{frame_hash}.jpg"
                )
                cv2.imwrite(str(sample_path), image)
                logger.info(
                    f"Saved excluded-object training sample "
                    f"(conf={conf:.3f}): {sample_path}"
                )
            if track.summary.confirmed and track.summary.object_valid:
                self._export_track_summary(track, video_hash_map)
            self.tracks.remove(track)

    def all_tracks_mask(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Mask covering the current and predicted boxes of live tracks"""
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        for track in self.non_expired_tracks:
            for ref_bbox in [
                track.summary.last_valid_frame.bbox,
                track.summary.estimated_bbox,
            ]:
                x1, y1, x2, y2 = ref_bbox.xyxy
                if x2 >= x1 and y2 >= y1:
                    mask[y1 : y2 + 1, x1 : x2 + 1] = 255

        return mask
