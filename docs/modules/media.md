# Module: `media/`

Everything that touches video/audio files: probing, downloading, audio extraction, frame
extraction. All four wrap external tools (yt-dlp, FFmpeg, OpenCV) behind small classes that
raise `DialogueLocatorError` subclasses and log at their boundaries.

---

## `probe.py`

```python
@dataclass(frozen=True)
class MediaProbe: duration, has_video, has_audio, fps, width, height, frame_count, video_codec, audio_codec

def ensure_binary(name: str) -> str                       # shutil.which or ConfigurationError with install hint
def probe_media(path, ffprobe_binary="ffprobe", timeout=60) -> MediaProbe
```

Runs `ffprobe -print_format json -show_format -show_streams`. Notes:

* `fps` prefers `avg_frame_rate` (true average) over `r_frame_rate` (container tick rate,
  inflated for variable-frame-rate files); fractions like `30000/1001` are parsed exactly.
* `frame_count` uses `nb_frames` when the container has it (MP4 does), else `round(duration × fps)`
  (WebM/MKV often omit it).
* Raises `UnsupportedVideoError` for missing/unreadable/stream-less files.

Used by the downloader (to fill `VideoInfo`), the audio extractor (to check an audio stream
exists *before* running ffmpeg, and to validate the output), and indirectly by frames.

---

## `downloader.py`

```python
def is_url(source) -> bool            # http(s) + host
def validate_url(url) -> str          # InvalidURLError on empty / bad scheme / no host (logged WARNING)

class VideoDownloader:
    def __init__(self, config: DownloadConfig, ffprobe_binary="ffprobe")
    def fetch_search_media(self, source, dest_dir, progress=None, reuse_existing=True) -> MediaInfo
    def fetch_video_clip(self, source, start, end, dest_dir, progress=None, reuse_existing=True) -> VideoInfo
```

### Behaviour

1. Local path → verify it exists, probe, return (both methods). This is the offline path used by
   tests and the documented fallback when a host is unreachable.
2. `fetch_search_media` (URL) → yt-dlp with the audio-first ladder
   `bestaudio[m4a] / bestaudio / worst[height<=search_H] / worst / worst*[acodec!=none]` —
   the cheap download that feeds transcription/matching/verification. May have no video stream;
   raises `UnsupportedVideoError` if it has no *audio* stream (nothing to search). Cached as
   `dest_dir/media.*`.
3. `fetch_video_clip` (URL) → yt-dlp `download_ranges` for `[start−pad, end+pad]` with
   `force_keyframes_at_cuts` (re-encoded cut, so `VideoInfo.clip_start` maps source timestamps
   exactly), format ladder `best[height<=H][ext=mp4] / bestvideo[≤H][mp4]+bestaudio[m4a] / … / best`.
   Runs only after a verified match, so a full-quality download is only ever a few seconds long.
   Cached as `dest_dir/clip_<start_ms>_<end_ms>.*`.
4. Both use `noplaylist`, retries/socket timeout from config, messages routed to our logger, and a
   **time-throttled** progress hook (one event per `progress_interval_seconds` with size, speed
   and ETA — percent buckets read as a hang on slow hosts).
5. `reuse_existing`: if the method's own cached file exists (ignoring `.part`/`.ytdl` leftovers)
   it is used without touching the network. The pipeline keys `dest_dir` by `sha1(source)`.
6. The resulting file is probed; `fps/duration/frame_count` come from ffprobe, never from the
   site's metadata (often rounded or absent, e.g. on ok.ru).

### Error mapping

| yt-dlp raises | we raise |
|---|---|
| `UnsupportedError` | `UnsupportedVideoError` |
| `DownloadError`, `ExtractorError` | `DownloadError` (with the `ERROR:` prefix stripped from the message) |
| anything else | `DownloadError("Unexpected error …")` + traceback in log |
| "success" but no file | `DownloadError` |

Known site behaviours are documented in [Operations](../07-operations-troubleshooting.md):
ISP-level resets for ok.ru, YouTube "confirm you're not a bot"/429.

---

## `audio.py`

```python
class AudioExtractor:
    def __init__(self, config: AudioConfig)
    def extract(self, media: MediaInfo | VideoInfo, dest_dir, progress=None, reuse_existing=True) -> AudioInfo
    def extract_clip(self, source: Path, start, end, dest_path) -> AudioInfo
```

Command: `ffmpeg -hide_banner -nostdin -v error -y [-ss S -t D] -i SRC -vn -sn -dn -ac 1 -ar 16000 -c:a pcm_s16le -progress pipe:1 -nostats OUT.wav`

* Output is exactly Whisper's native input (16 kHz mono 16-bit PCM), so no resampling happens
  later and the whole track can be loaded into memory once (`transcription.base.load_pcm`).
* `-progress pipe:1` lines (`out_time_us=`) are parsed against the probed duration to emit
  progress every 5 %.
* Checks `ensure_binary(ffmpeg)` first (fail fast, before announcing work), then `has_audio` via
  probe (clear "no audio stream" error instead of an ffmpeg exit code).
* Output is validated by **probed duration**, not file size: a clip requested past the end of the
  media produces a 78-byte WAV with a header and no samples — that must be an error.
* Timeout (`AudioConfig.timeout_seconds`) kills ffmpeg and raises `AudioExtractionError`.
* `extract_clip` uses `-ss` *before* `-i` (input seeking: instant on WAV). Clip timestamps are
  relative to `start`; callers add it back. (The verifier ended up slicing the in-memory array
  instead, but the method remains for CLI/debug use.)

---

## `frames.py` {#frames}

```python
def timestamp_to_frame(t, fps) -> int          # floor(t·fps + 1e-6), clamped ≥ 0
def frame_to_timestamp(n, fps) -> float        # n / fps

class FrameExtractor:
    def __init__(self, config: FrameConfig, ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")
    def extract(self, video: VideoInfo, timestamp, dest_path) -> FrameInfo
    def read_frame(self, video: VideoInfo, frame_number, fps=None) -> np.ndarray   # BGR
```

### Frame numbering

Frame *k* is displayed during `[k/fps, (k+1)/fps)`, so the frame visible at time `t` is
`floor(t × fps)` — **not** `round`, which would pick the next frame for the second half of every
interval. The epsilon keeps `t` exactly on a boundary in the frame that starts there.
`FrameInfo.timestamp` is the extracted frame's own presentation time and `frame_number` is
`floor(timestamp × fps)`, so the two are always consistent with each other (requested 1.500 s
@ 25 fps → frame 37 → `00:00:01.480`). Note the timestamp equals `k/fps` only for a whole
video: when the frame comes from a clip it is `clip_start + local_frame/fps`, and since the cut
point need not land on a source frame boundary the result can sit a few milliseconds off the
`k/fps` grid (canonical run: frame 7782 @ 23.976 fps, timestamp 324.603 s, while
`7782/23.976` = 324.578 s).

### Reading

1. OpenCV: `cap.set(CAP_PROP_POS_FRAMES, n)`; verify `cap.get(CAP_PROP_POS_FRAMES)` landed on `n`;
   `cap.read()`.
2. If the seek landed elsewhere or decoding failed → **ffmpeg fallback**:
   `ffmpeg -ss <n/fps> -i video -frames:v 1 -f rawvideo -pix_fmt bgr24 pipe:1`, reshaped with the
   probed width/height. Frame-accurate, no temp file, no re-encode.
3. Both paths are proven equivalent by `test_opencv_and_ffmpeg_paths_agree` (same frame: mean abs
   pixel diff ≈ 0.5; a frame 10 later: ≈ 5).

Requests past the end clamp to the last frame with a WARNING; negative → frame 0. All failures →
`FrameExtractionError` (including an unwritable output directory).

`extract` accepts a *clip* (`VideoInfo.clip_start > 0`): seeking is clip-relative while the
reported frame number and timestamp stay absolute to the source. `read_frame` is public so other
visual stages can reuse the same seek/fallback logic on single frames (the V3 mouth check reads
its window sequentially instead — one seek, then `cap.read()` per frame — which is the right
pattern for dense scanning).
