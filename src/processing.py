from __future__ import annotations

import json
import logging
import os
import queue
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np

import utils
from config import METADATA_DIR, TIMESTAMP_FORMAT, settings
from detection import Frame
from shared import frame_queue, set_recording_queue_size, shutdown_event
from tracking import TrackManager, TrackState, VideoHashMap
from video_io import FfmpegWriter

logger = logging.getLogger(__name__)


def _release_writers(
    writer: FfmpegWriter,
    video_hash_map: VideoHashMap = None,
    frame_hash: Optional[str] = None,
    log_msg: str = "",
) -> Optional[FfmpegWriter]:
    """Release the video writer, with optional logging."""
    hash_msg = f"({frame_hash}) " if frame_hash else ""
    log_msg = f" ({log_msg})" if log_msg else ""

    writer.release()
    final_path = writer.output_path.replace(".tmp.", ".")
    os.rename(writer.output_path, final_path)
    logger.warning(f"{hash_msg}Saving recording{log_msg}: {final_path}")

    # video_name matches the one recorded against track_summaries.jsonl rows for this recording
    if video_hash_map:
        video_name = os.path.basename(writer.output_path.replace(".tmp.", "."))
        hashes = video_hash_map.get_hashes(video_name)
        with open(os.path.join(METADATA_DIR, f"video-{video_name}.json"), "w") as f:
            json.dump(hashes, f, indent=4)
    else:
        logger.warning(f"No video hash map available for saving video metadata.")

    return None


def processing_thread():
    """Process frames to detect objects and record videos"""
    logger.info("Processing thread started")

    # initialise buffers
    pre_buffer: deque[Frame] = deque(
        maxlen=int(np.ceil(settings.FPS * settings.BUFFER_DUR))
    )
    processing_buffer: deque[Frame] = deque()
    replay_buffer: deque[tuple[datetime, np.ndarray]] = deque()

    # initialise state and previous frame
    recording = False
    writer = None
    prev_frame = None
    frames_since_detection = 0
    track_manager = TrackManager()
    video_hash_map = VideoHashMap()

    while not shutdown_event.is_set() or not frame_queue.empty() or replay_buffer:

        # get frame from capture queue
        try:
            if replay_buffer:
                timestamp, image = replay_buffer.popleft()
                logger.info("Processing replay frame")
            else:
                timestamp, image = frame_queue.get(timeout=0.1)
            start_capture = datetime.now()
            frame_captured = Frame(
                timestamp=timestamp,
                image=image,
                prev_frame=prev_frame,
                prev_track_mask=track_manager.all_tracks_mask(
                    settings.FRAME_WIDTH, settings.FRAME_HEIGHT
                ),
                forced_detection_run=frames_since_detection + 1 >= settings.FPS,
            )
        except queue.Empty:
            continue

        logger.debug(f"({frame_captured.hash}) Running processing")

        # detect objects and update tracking state
        track_frames, did_run_detection = frame_captured.detect_objects()
        track_manager.update(
            track_frames,
            frame_captured.hash,
            video_hash_map=video_hash_map,
        )
        frame_captured.processing_track_summaries = [
            t.summary for t in track_manager.non_expired_tracks
        ]

        # add captured frame to processing buffer
        processing_buffer.append(frame_captured)

        # update non-detection counter
        if did_run_detection:
            frames_since_detection = 0
        else:
            frames_since_detection += 1

        # update current frame to be previous frame
        prev_frame = frame_captured

        # log processing rate
        elapsed_capture = utils.log_timing(
            logger, "Processing", start_capture, frame_captured.hash
        )

        # process frames in the buffer if enough frames have been captured
        start_recording = datetime.now()
        while len(processing_buffer) > int(
            np.ceil(settings.FPS * settings.TRACK_NEW_DUR)
        ):

            # extract frame and check for excluded objects
            frame_recording = processing_buffer.popleft()
            frame_recording.recording_track_summaries = [
                track_manager.get_track(t.track_id).summary
                for t in frame_recording.processing_track_summaries
            ]
            assert all(
                [
                    s.state != TrackState.NEW
                    for s in frame_recording.recording_track_summaries
                ]
            )
            current_confirmed_summaries_recording = [
                s
                for s in frame_recording.recording_track_summaries
                if s.state in [TrackState.ACTIVE, TrackState.STALE]
            ]
            has_excluded_object = any(
                s.last_valid_frame.object_name in settings.EXCLUDED_OBJECTS
                for s in current_confirmed_summaries_recording
            )

            if any(
                s.state in [TrackState.ACTIVE, TrackState.STALE]
                for s in current_confirmed_summaries_recording
            ):

                if has_excluded_object:

                    # clear buffers
                    processing_buffer.clear()
                    pre_buffer.clear()
                    track_manager.remove_tracks("all", video_hash_map=video_hash_map)
                    logger.info(
                        f"({frame_recording.hash}) Clearing buffer and Tracks due to detection of excluded object"
                    )

                else:

                    # initialise recording and write pre buffer to video file
                    if not recording:

                        writer = FfmpegWriter(
                            frame_recording.timestamp.strftime(TIMESTAMP_FORMAT),
                            settings.FPS,
                            settings.FRAME_WIDTH,
                            settings.FRAME_HEIGHT,
                            settings.VIDEO_QUALITY,
                            queue_size_callback=set_recording_queue_size,
                        )
                        logger.warning(
                            f"({frame_recording.hash}) Starting recording: {writer.output_path}"
                        )
                        # flush buffer
                        pre_buffer_len = len(pre_buffer)
                        while pre_buffer:
                            bf = pre_buffer.popleft()
                            video_hash_map.append(writer, bf.hash)
                            writer.write(bf.image, bf.hash)
                        logger.info(
                            f"({frame_recording.hash}) Written {pre_buffer_len} frames from pre detection buffer"
                        )

                        recording = True

            if recording:
                assert writer is not None

                # write current frame and assess post buffer termination
                if not has_excluded_object:
                    video_hash_map.append(writer, frame_recording.hash)
                    writer.write(frame_recording.image, frame_recording.hash)

                # stop recording close video file
                recording_tracks_valid = any(
                    s.state < TrackState.EXPIRED
                    for s in frame_recording.processing_track_summaries
                )
                if not recording_tracks_valid or has_excluded_object:
                    if has_excluded_object:
                        log_msg = "excluded object detected"
                    elif not recording_tracks_valid:
                        log_msg = "all tracks expired"
                    writer = _release_writers(
                        writer,
                        video_hash_map,
                        frame_recording.hash,
                        log_msg,
                    )

                    if not track_manager.non_expired_tracks:
                        replayed = len(processing_buffer)
                        if replayed:
                            while processing_buffer:
                                frame_replay = processing_buffer.popleft()
                                replay_buffer.append(
                                    (frame_replay.timestamp, frame_replay.image)
                                )
                            logger.info(
                                f"({frame_recording.hash}) Queued {replayed} delayed frame(s) for replay"
                            )
                            pre_buffer.clear()
                            track_manager.remove_tracks(
                                "all", video_hash_map=video_hash_map
                            )
                            prev_frame = None
                            frames_since_detection = 0
                            logger.info(
                                f"({frame_recording.hash}) Reset processing state before replaying delayed frames"
                            )
                    recording = False

            else:

                # store current frame image and timestamp to rolling buffer
                if not has_excluded_object:
                    pre_buffer.append(frame_recording)

        # log recording rate
        elapsed_recording = utils.log_timing(
            logger, "Recording", start_recording, frame_captured.hash
        )

        # log overall FPS
        if (overall_fps := 1 / (elapsed_capture + elapsed_recording)) < settings.FPS:
            logger.warning(f"Processing/recording thread slow: {overall_fps:.1f} FPS")

    # cleanup
    if processing_buffer:
        logger.info(f"Discarding {len(processing_buffer)} delayed frame(s)")
        processing_buffer.clear()
    track_manager.remove_tracks("all", video_hash_map=video_hash_map)
    if writer:
        writer = _release_writers(writer, video_hash_map)

    logger.info("Processing thread stopped")
