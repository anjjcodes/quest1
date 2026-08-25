# Module: `vision/`

The visual validation stages. Both run *after* the frame exists, gate the verdict
(`ResultStatus.NOT_ONSCREEN`) and **fail open** — a check that cannot run adds a warning and
never loses the localisation. Both use MediaPipe task models that are downloaded once on first
use (`model_files.ensure_model_file`) and cached under `data/models/`; `mediapipe` itself is
imported lazily so importing the package stays cheap and tests can fake the backend.

## `face_detector.py` — V2: is a face visible in the frame?

```python
class FaceDetector:
    def __init__(self, config: FaceDetectionConfig)
    def detect(self, image: np.ndarray) -> FaceDetectionResult        # BGR array
    def detect_file(self, path: Path) -> FaceDetectionResult          # the saved output frame
    def close(self)                                                   # also a context manager
```

* Backend: BlazeFace short-range via the MediaPipe Tasks API (`.tflite`, ~230 KB).
* `min_detection_confidence` default **0.3**, calibrated on real frames: faces in cinematic
  mid/wide shots score as low as ~0.34 (close-ups ~0.97) while non-face junk stays ≤ ~0.2;
  MediaPipe's usual 0.5 default misses distant faces.
* Boxes are clamped to the image; results are sorted best-confidence first.
* The detector instance is cached after the first call but is **not thread-safe** — one pipeline
  job at a time (the server default) is fine.

## `mouth_movement.py` — V3: do the lips move during the line?

```python
class MouthMovementAnalyzer:
    def __init__(self, config: MouthMovementConfig)
    def analyze(self, video: VideoInfo, start, end) -> MouthMovementResult   # absolute seconds
```

* Backend: MediaPipe Face Landmarker (`.task`, ~3.7 MB) in VIDEO running mode (temporal
  smoothing), one face per frame (the most prominent).
* Per-frame signal: **mouth openness** = inner-lip gap (landmarks 13–14) / mouth width
  (61–291) — invariant to face size, camera distance and resolution.
* Movement score = standard deviation of that series over the window. Measured ~0.09 while
  speaking vs ~0.001 landmark jitter on a still face; threshold default **0.02** sits between.
* Verdict is **indeterminate** (`moving=None`, pipeline fails open) when fewer than
  `min_face_frames` frames contain a face — absence of a face is not evidence of a still mouth.
* Frames are read sequentially after one seek (the clip is only a few seconds long). A fresh
  landmarker is created per `analyze()` call: VIDEO mode requires monotonically increasing
  timestamps for the lifetime of the landmarker, so a reused instance would crash on the next
  job and would carry tracking state across videos.

## `model_files.py`

`ensure_model_file(path, url, timeout, error_cls)` — shared download-and-cache for the task
models: atomic write via a `.part` file, rejects tiny responses (an error page is not a model),
raises the caller's own stage error class.
