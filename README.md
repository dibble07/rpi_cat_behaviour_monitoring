# RPi Cat Behaviour Monitoring

## Purpose

A real-time video monitoring system running on Raspberry Pi that detects cats and other objects using object detection. When specific behaviours or detections occur, the system automatically records annotated video clips with tracking information (IDs, confidence scores, bounding boxes). The system uses multi-threaded processing to efficiently handle continuous video capture, motion-aware detection, and recording workflows on resource-constrained hardware.

## Demo

![Cat monitoring system in action](readme.gif)

## Logic and Architecture

### Thread-Based Approach

The system uses three threads that synchronise via thread-safe queues and event signals. This approach is taken to avoid delays and buffer in one process blocking other periodic behaviours that require specific timing.

### Capture Thread

Continuously acquires frames from the camera and enqueues them:
1. Acquires raw frame from camera
1. Enqueues the image and timestamp to `frame_queue` for processing

### Processing Thread

The core pipeline. For each frame dequeued, it:
1. Identifies search regions based on observed motion, image foreground and existing tracks
2. Crops image to area of interest and runs object detection
3. Updates Tracks
    - Assigns detectiosn to existing Tracks
    - Updates state of each Track
4. Buffers and records video clips based on track state

#### Object Detection

Areas that have motion, are not background or contain existing tracks are searched using a detection model. If no detections have been run for a period of time, detection will be run on the entire frame.

```mermaid
flowchart TD
    A["Dequeued Frame"] --> B["Motion detection"]
    A --> C["Background subtraction"]
    A --> D["Existing Track locations"]
    B --> E{"Search area<br/>present?"}
    C -->E
    D -->E
    E -->|YES| F["Run Detection"]
    E -->|NO| G["Skip Detection"]
    F --> H["Extract Detections"]
    G --> H
```

#### Tracking updates

The system maintains persistent tracks across frames by assigning detections to existing Tracks based on:
- Spatial overlap between detection and forecast Track location
- Detection confidence
- Visual similarity
- Age of most recent detection in Track

```mermaid
flowchart TD
    A["Frame detections<br/>and existing Tracks"] --> B["Compute match scores for<br/>all possible pairings"]
    B --> C["Determine best global detection-Track pairing"]
    C --> D{"Approve match<br/>found?"}
    D -->|YES| E["Update Track"]
    D -->|NO| F["New Track"]
```

#### Tracking State Machine

Tracks move between various states based on the age and confidence of recent detections. These states determine the initilisation and termination of recording as well as annotation of the video.

| State | Desciption |
|-------|-------|
| **NEW** | Holding state until enough detections of high enough confidence are matched |
| **ACTIVE** | Confirmed Track with detection in current Frame |
| **STALE** | Previously active Track without current detection |
| **EXPIRED** | Tracks that were not confirmed or were stale for too long |

```mermaid
stateDiagram-v2    
    NEW --> ACTIVE: Sustained<br/>presence
    NEW --> EXPIRED: Too few<br/>detections<br/>of required<br/>confidence
    ACTIVE --> STALE: Detection<br/>lost
    STALE --> ACTIVE: Detection<br/>reacquired
    STALE --> EXPIRED: Timeout
```

#### Recording

Recording is deliberately decoupled from processing via a buffer to introduce a delay in the recording. The delay enables New Tracks to reach maturity (Active or Expired) before impacting the recording. This avoids small videos initiated by false positives or splitting of large files due to brief false detections of an excluded class.

The exported video relies on a few different components and conditions:
- Rolling pre-detection buffer
    - Maintains a rolling buffer of recent frames
- When a track transitions to Active state
    1. Pre-buffer frames are flushed to disk (provides context before detection)
    1. Video recording begins
    1. Frames are annotated with: track ID, state, frame count, confidence
- While Active tracks exist
    - Frames written continuously if active tracks or are of permitted class
    - Active tracks of excluded class will terminate recording and clear buffers
- When no ACTIVE tracks remain
    - Video file closed

## File Structure

### Root Level & Configuration

| File/Directory | Purpose |
|---|---|
| [pyproject.toml](pyproject.toml) | Python project dependencies and metadata |
| [settings.toml](settings.toml) | Application configuration |
| [RPi_setup.md](RPi_setup.md) | Raspberry Pi hardware setup and initialisation guide |
| [src/](src/) | Core application source code |
| [datasets/](datasets/) | Model training and app development datasets |
| [models/](models/) | Trained model files for deployment |
| [models_staging/](models_staging/) | Model experimentation and evaluation |
| [notebooks/](notebooks/) | Jupyter notebooks for analysis, model training and app development |

### Source Files

| File | Purpose |
|---|---|
| [src/app.py](src/app.py) | Main entry point: spawns threads, coordinates graceful shutdown |
| [src/shared.py](src/shared.py) | Shared state and synchronisation primitives: `frame_queue`, `shutdown_event`, camera instance |
| [src/capture.py](src/capture.py) | Camera frame acquisition: reads frames and enqueues to `frame_queue` |
| [src/processing.py](src/processing.py) | Core processing pipeline: object detection, tracking, recording |
| [src/monitoring.py](src/monitoring.py) | Resource monitoring: logs system metrics periodically |
| [src/camera.py](src/camera.py) | Camera abstraction layer: handles different camera backends |
| [src/config.py](src/config.py) | Configuration loader: reads settings from `settings.toml` |
| [src/tracking.py](src/tracking.py) | Tracking system: Hungarian algorithm, Kalman filtering, state machine |
| [src/utils.py](src/utils.py) | Utility functions: bounding box operations, logging helpers |

### Datasets

| Directory | Purpose |
|---|---|
| **classification_data/** | Training data for cat identification and behavious classification: contains labels and train/validation splits |
| **coco_metadata/** | COCO object detection training data configuration files: base and variant configurations |
| **finetune_data/** | Fine-tuning dataset with images and label annotations |
| **finetune_data_cropped/** | Pre-processed version of fine-tuning data |
| **mock_inputs/** | Raw video for development camera mocking |
| **raw_video/** | Raw video files for offline processing and testing |
