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
* Movement score = the **detrended** standard deviation over the most active sliding window
  of `score_window_seconds` (0.5 s), not one number over the whole line. Detrended so a head
  that merely turns during a reaction shot (openness slides to 0.042 raw, 0.016 detrended) is
  not read as speech; sliding so a line that is only on camera for its last second still
  counts. Measured on the cached corpus: present but silent **0.003–0.021**, speaking on
  camera **0.046–0.123**; threshold default **0.03** sits in the gap.
* Frames are **not** landmarked whole. The landmarker's own detector only finds large faces,
  so each frame goes through BlazeFace first and the landmarker sees a padded crop upscaled to
  `min_crop_size`. The last known box is carried forward for `box_carry_seconds` over frames
  detection misses, because BlazeFace flickers on small faces (a 290 px face in 1920x1080 was
  found in 4 of 26 frames; the carry lifted that to 22). Bounded so a cut cannot attribute one
  shot's face to the previous run.
* Windows never span a shot change (`shot_change_shift` / `shot_change_scale`) or a gap longer
  than `max_gap_seconds`: the step between two still faces' resting mouths is not a mouth
  opening.
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
