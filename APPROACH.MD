# Approach

My working assumption is that onscreen dialogue means the person saying the line is visible
in the frame while saying it. Everything below follows from that, and the design went
through three versions before it got there.

The first idea was the obvious one: download the video, extract the audio, transcribe all of
it, then fuzzy match the transcript against the input dialogue. That works, but it is brute
force. On a one hour video you transcribe the full hour before you can answer, even when the
line sits in the first minute. So I split the audio into chunks and searched each chunk as
soon as it was transcribed, returning the moment one matched, with the worst case still being
the whole video. Chunking introduced its own bug: the dialogue can be spliced across a chunk
boundary, half in one chunk and half in the next, so neither half scores a match on its own.
The fix was overlap, each new chunk starting a few words before the previous one ended. At
that point the chunks are really just a sliding window over the words, so I dropped fixed
chunks entirely: every new word the model emits extends a window, the window is scored
against the dialogue, and the search stops at the first window that matches.

The next bottleneck was the download itself. Fetching the full video before searching took
longer than the search, and the frames are only needed after a match is found. So I split the
download in two: fetch only the audio stream first, stream the transcription over it with the
sliding window matcher, and only after a verified match fetch a few seconds of high quality
video around the timestamp to extract the frame.

That plus the model choice was version 1. A fast Whisper model does the streaming search,
because it is quick enough to run over everything. When a window crosses the 80 score
threshold, a larger model retranscribes just that region to confirm the wording and tighten
the timestamp, because running the large model over the whole track was far too slow.

Version 1 only proves the line is in the audio, not that it is onscreen. The edge cases made
that concrete: someone speaks two words on camera and then the camera turns away, or the
first two words land off camera before the shot cuts to the speaker. As far as my research
went these still count as onscreen dialogue, but a pure voice over does not. Version 2 added
face detection, and version 3 added a mouth movement check on top of it, because person A can
be speaking while the camera holds on person B, which is a reaction shot rather than onscreen
dialogue.

So each version narrows the question one step further:

- V1 asks: is the dialogue spoken in the audio?
- V2 asks: is a face visible in the matched frame?
- V3 asks: is that visible face moving its mouth?
- The future step asks: is that mouth movement in sync with this audio?

V3 is a heuristic, not a full active speaker detector. It filters out reaction shots and
voice overs, but only the sync step would be real evidence that the visible person is the one
delivering the line.

The whole pipeline in one picture:

```
Video URL
    │
    ├── audio only download (yt-dlp)
    │        │
    │        ▼
    │   fast Whisper (base), streaming
    │        │
    │        ▼
    │   sliding window fuzzy match
    │        │
    │        ▼
    │    candidate timestamp
    │        │
    │        ▼
    │   larger Whisper (small) on ±12 s
    │        │
    │        ▼
    │    verified timestamp
    │        │
    │        ▼
    └── video clip download (±5 s)
             │
             ▼
        frame extraction
             │
             ▼
         face check (V2)
             │
             ▼
     mouth movement check (V3) ──► frame moves to where
             │                      the speaker is on camera
             ▼
      result JSON + frame image
```

Here is the same flow, step by step.

### Step 1. Validate before doing any work

Before touching the network I check that the dialogue is usable (at least two words, not empty
after normalisation) and that the source is either a valid http(s) URL or an existing local
file. A bad input fails in milliseconds with a clear message instead of after a long download.

### Step 2. Download only the audio

I ask `yt-dlp` for the best audio only stream. Some hosts like ok.ru do not offer separate
audio streams, so the fallback is the smallest video rendition that still carries audio. Either
way this is the cheap download, tens of megabytes instead of gigabytes, and it is all the
search needs. Downloads are cached per source URL, so running a second dialogue against the
same video skips this step entirely.

### Step 3. Extract clean audio for the model

FFmpeg converts whatever was downloaded into 16 kHz mono PCM, which is exactly what Whisper
expects, so no resampling happens later. I then load the whole track into memory as a float
array once. That matters later, because the verifier can slice its window straight out of this
array without touching the disk or running FFmpeg again.

### Step 4. Transcribe and match at the same time

A small Whisper model (`base`, about 60x realtime on my CPU, greedy decoding for speed) streams
the transcription word by word, with a timestamp on every word. Each word goes straight into a
sliding window matcher:

- Both the dialogue and the transcript words are normalised the same way first: lowercase,
  punctuation and quotes stripped, unicode cleaned up, so "Don't!" and "dont" compare equal.
- For a dialogue of n words I score windows of n minus 2 up to n plus 2 words ending at the
  newest word, because the speech model sometimes splits or merges words. Each window is
  scored 0 to 100 with RapidFuzz, and 80 counts as a match.
- When a window crosses 80 I do not stop immediately. I wait up to 2 more words and keep the
  best overlapping window, because the true match often scores higher one word later. This
  settling step is what pins the start timestamp to the right word.
- The moment the match settles, I close the transcription generator. Whisper stops decoding
  right there, which is the whole point of streaming: a line 5 minutes in costs 5 minutes of
  transcription, not the full video.
- If the whole track finishes with no match, I keep the 3 best scoring non overlapping windows
  as near misses, so the user can see what the video actually says at the closest spots.
- One more safety net: Whisper runs with a voice activity filter that skips silence, which is
  good for speed, but loud music can trick it into deleting real speech. So if the first scan
  finds nothing, I run one more scan with the filter off before reporting not found. This
  costs an extra pass only when the answer would otherwise have been a miss.

### Step 5. Verify the candidate with a bigger model

The fast model is good enough to find the spot but not good enough to be trusted on the exact
words and timing. So a larger model (`small`, full beam search, voice filter off) transcribes
just the 12 seconds either side of the candidate, sliced from the audio already in memory. I
score every window in that clip and take the best one:

- If the fast pass already scored 90 or above, I skip this step. Verification exists to check
  uncertain matches, not to re prove near perfect ones.
- If the big model's best score is within 5 points of the first pass, it is confirmed, and I
  use the big model's words and timestamps because they are tighter.
- If it drops more than 5 points, I reject the verification, keep the first pass timestamp,
  and attach a warning. The big model has misheard things before, so it does not get to
  silently overwrite a good match.

### Step 6. Download a few seconds of video

Only now, with a verified timestamp, do I download real video: a full quality clip from 5
seconds before the match to 5 seconds after it. The cut is re encoded so the clip starts at an
exact time rather than at the nearest keyframe, and I record that offset, so a timestamp in
the source maps precisely to a position in the clip. A not found result never reaches this
step, so it never pays for a video download at all.

### Step 7. Turn the time into a frame and save it

Frame k is on screen from k/fps until (k+1)/fps, so the frame showing at time t is
floor(t × fps). The fps comes from ffprobe on the actual downloaded file, never from the
website's metadata, which is often rounded or missing. I seek to that frame with OpenCV,
verify the seek actually landed where I asked, and fall back to FFmpeg if it did not. The
reported timestamp is the frame's own presentation time, so the printed timestamp and frame
number always agree with each other.

### Step 8. Check the line is actually onscreen

Two checks run on the result, both cheap and both calibrated on real frames from the test
corpus rather than on guessed defaults.

**Face check (V2)** — MediaPipe BlazeFace looks for a face in the extracted frame. I lowered
the confidence threshold to 0.3 because faces in cinematic mid and wide shots score as low as
0.34 while non face junk stays at or below 0.2, and the model's usual 0.5 default misses
distant faces. I set it by manually scoring detections on frames spanning closeups, medium
shots and wide cinematic shots. No face here is a provisional `not_onscreen`, which V3 can
still overturn.

**Mouth check (V3)** — MediaPipe's Face Landmarker tracks the lips across every frame of the
matched window. Per frame the signal is mouth openness, the inner lip gap divided by the
mouth width, which makes it independent of face size and camera distance. Three things sit
between that per frame number and a verdict, and each of them exists because the naive
version failed on a real clip:

- The landmarker never sees the whole frame. It runs its own face detector, and that detector
  only finds faces filling a good part of the image, so a 290 px face in a 1920x1080 wide shot
  is simply not found and a speaker plainly on camera contributes no frames at all. Each frame
  goes through BlazeFace first and the landmarker gets a padded crop around the best box,
  upscaled to at least 256 px. Frames where BlazeFace finds nothing fall back to the whole
  frame, which is never worse than not cropping.
- The last known box is carried forward, for up to 0.4 s, onto frames where detection finds
  nothing. See decision 6 for why.
- The score is the *detrended* standard deviation over the most active 0.5 s sliding window,
  not one number over the whole line. Detrended because a face that is merely present does not
  hold a fixed mouth opening: it turns, tilts and settles, and openness slides smoothly with
  it. On a listener in a reaction shot that slide alone reached 0.042, while the same window's
  spread around a fitted line was 0.016. A speaking mouth oscillates instead of sliding and
  keeps almost all of its spread (0.052 raw, 0.047 detrended). Sliding, because speech is
  bursts: a line that is off camera for most of its length and on camera for the last second
  still reads as moving, which is the answer the brief wants.

Windows are only scored inside runs of consecutive same shot frames, and only when a run has
at least 5 samples. A cut is spotted by losing the face or by the face box jumping, so a step
between two still faces' resting mouths can never be read as a mouth opening. Measured over
the cached corpus, faces present but silent score 0.003 to 0.021 and faces speaking on camera
score 0.046 to 0.123, so the threshold sits at 0.03, in the gap.

The landmarker runs at the stock 0.5 confidence, deliberately stricter than the 0.3 that feeds
it. BlazeFace fires on blurred rubble at 0.45 to 0.62, higher than it scores some real faces,
so it cannot police itself; a box the landmarker can fit no mesh to is not a face whatever its
detection score.

**V3, not V2, settles the verdict**, because it is the only check that sees the whole window
instead of a single frame:

- A mouth moving anywhere during the line is `found`, and the answer frame moves to that
  moment. This holds even when the line's first frame had no face, which is how a line that
  opens on a title card and cuts to the speaker gets the right answer.
- Faces on camera but none of them moving is `not_onscreen`: narration, dubbing or a reaction
  shot. If the line itself opened off camera the frame moves to the face the verdict is about,
  so the image explains the result.
- Barely any frame of the line landmarked is `not_onscreen`, per the rubble case above.
- Both checks fail open. If a check crashes, or a face is present throughout but its frames
  never line up into a scorable run, the localisation stands with a warning attached, because
  a broken check should never delete a correct answer.

### Step 9. What comes out

The result is one JSON record plus the frame image: status (`found`, `not_found` or
`not_onscreen`), the timestamp as HH:MM:SS.mmm, the frame number, the matched text as heard,
the match score, what the first pass found versus what the verifier decided, the face result
(count, per face boxes and scores), the mouth result (score against threshold, frames analysed
versus frames with a face, the window measured, and `movement_start`, the moment the speaker
comes on camera), any warnings, near misses when nothing matched, and how long each stage
took. When V3 moved the answer frame, the frame and timestamp in the record are the moved
ones, and the face check has been re run on that frame so the two never disagree.
The command line prints the short form, the API returns the full record, and the web UI shows
it with live progress. All three interfaces run the exact same pipeline, and every stage of
that pipeline can be swapped out in the constructor, which is also how the whole thing is
tested offline with fakes.

## Key Decisions

### 1. Search the audio instead of the video frames

- **What I chose:** speech to text with Faster Whisper, then fuzzy text matching with
  RapidFuzz over the words it produces.
- **Alternative considered:** running OCR or a vision model over the video frames, in case
  "dialogue appears" meant subtitle text.
- **Why I chose X over Y:** the reference video has no subtitles, the dialogue is spoken. The
  audio of a one hour video is around 50 MB while the full video is around 1.5 GB, and the
  speech model gives me a timestamp for every word, which is exactly what I need to pick a
  frame. With OCR I would be decoding and reading thousands of frames to answer a question the
  audio answers directly.

### 2. Stream the transcription and stop at the first match

- **What I chose:** the sliding window matcher of Step 4, fed word by word, with the decoder
  closed the moment a match settles.
- **Alternative considered:** transcribe the whole video first, then search the text.
- **Why I chose X over Y:** transcription is the slowest stage and the line can be anywhere.
  On my JFK test video the scan stopped at 51% of the audio, so it cost half of a full
  transcription. The worst case, where the line is at the end or not there at all, costs the
  same as transcribing everything, so there is no downside.

### 3. Use two models, a fast one to search and a bigger one to verify

- **What I chose:** the two model split of Steps 4 and 5 — `base` searches, `small` re reads
  the candidate window, and the 5 point rule decides which one's timestamp survives.
- **Alternative considered:** one accurate model for everything, which runs at about 3x
  realtime against `base`'s 60x, or trusting the fast model alone.
- **Why I chose X over Y:** finding a candidate and confirming it need different levels of
  accuracy. The fast model does mishear words, and on one repeated phrase it was 6.4 seconds
  off until the bigger model fixed it. But I do not blindly trust the big model either,
  because it once heard "fronts" where the audio clearly says "trunks", and the 5 point rule
  caught that. So the cheap model finds, the careful model checks, and neither one gets the
  final say alone.

### 4. Download the audio first and the video last

- **What I chose:** the two part download of Steps 2 and 6 — audio only for the search, and a
  ±5 s high quality clip afterwards purely to extract the frame.
- **Alternative considered:** downloading the whole video up front, which is what my first
  version did.
- **Why I chose X over Y:** if the dialogue is not in the video, I now pay for no video
  download at all. If it is, I pay for a few seconds instead of the whole file. On slow hosts
  like ok.ru that is the difference between minutes and an hour.

### 5. Define "onscreen" as a visible face whose mouth is moving

- **What I chose:** two small MediaPipe checks that run after the frame is extracted, as
  described in Step 8. BlazeFace answers whether there is a face in the frame at all, and the
  Face Landmarker tracks the lips across the matched window to see whether the mouth moves.
- **Alternative considered:** heavier active speaker detection models like SyncNet that match
  lip motion to the audio.
- **Why I chose X over Y:** the two checks answer the real question, is someone visibly
  saying this line, in about a second of CPU each, and every threshold could be set from
  measurement on real clips instead of guessing. A sync model would be more precise about who
  exactly is speaking, but it is much harder to run, tune and explain, and it can be added
  later as one more stage. What this did cost is that the obvious scoring rule was wrong
  twice: a plain standard deviation over the whole line reads a listener's head turn as
  speech, and reads a one second on camera burst at the end of a line as silence. Detrending
  fixed the first and the sliding window the second, and both changes came out of a test case
  that failed, not out of the design.

### 6. Carry the last face box forward instead of trusting the detector every frame

- **What I chose:** BlazeFace runs on every frame of the window, and on frames where it finds
  nothing the last known box is reused for up to 0.4 s, with the landmarker deciding whether
  that box really holds a face. The carry is bounded so a cut cannot attribute a new shot's
  face to the previous run, and a cut is caught on the frame detection resumes.
- **Alternative considered:** upscaling the whole frame before detection, tiling it, or
  lowering the 0.3 detection threshold further.
- **Why I chose X over Y:** this was the bug that cost me the most time. BlazeFace does not
  flicker because the threshold is too high, it flickers because it letterboxes the whole
  frame down to 192x192, so a face filling 15% of the width arrives about 29 px across however
  large the source is. On a mid shot of a man talking straight to camera, a 290 to 416 px face
  scored 0.30 to 0.39 against a 0.30 threshold and was found in only 4 of 26 frames, which is
  nowhere near enough consecutive samples to score, so a speaker plainly on camera read as
  "no face". Upscaling the frame cannot help, because the letterbox throws the extra pixels
  away again. Lowering the threshold only lets more rubble in, since BlazeFace scores blurred
  debris at 0.45 to 0.62. Carrying the box forward attacks the actual problem, an unstable
  detector in front of a stable one: on that clip it lifted 4 landmarked frames to 22, turning
  a scattered dust of samples into a clean speech oscillation, and it costs no extra model
  calls. The risk it introduces, a stale box surviving a cut, is what bounds it and what the
  shot change check exists for.

### 7. Size the thread pool to the performance cores, not to every core

- **What I chose:** `cpu_threads=0` resolves to the machine's performance core count, read
  from `hw.perflevel0.logicalcpu` on macOS and falling back to `os.cpu_count()` elsewhere.
- **Alternative considered:** `os.cpu_count()`, which is what I had, on the reasoning that
  ctranslate2's default of 4 threads leaves a modern CPU idle.
- **Why I chose X over Y:** that reasoning was wrong on this machine and it cost more than
  every other optimisation in this document combined. An M2 has 4 performance and 4 efficiency
  cores, and ctranslate2 runs each parallel region across whatever it is given, so the
  performance cores finish and then wait at the barrier for the efficiency cores. Scanning the
  same 331 s of audio with `base`/int8, producing byte identical output every time: 5.4 s at 4
  threads, 90.7 s at 6, 90.9 s at 8. A 17x cliff the moment work lands on an efficiency core.
  The verify pass behaves the same way, 9.7 s against 17.3 s. I found this only because a run
  that should have taken 25 seconds took two and a half minutes, and the raw library on the
  same audio with the same settings was 5x faster than my own wrapper around it — the gap
  between those two numbers was the whole bug. The general lesson I took: "use all the cores"
  stopped being sound advice when CPUs stopped having one kind of core.

## Assumptions

- "Dialogue" means spoken audio, not subtitle text, for the reasons in decision 1. If
  subtitles were the target, I would swap the speech recognition stage for OCR on the frames
  and keep the rest of the pipeline.
- The first occurrence of the line is the one that is wanted.
- The audio is English. The text normaliser only handles ASCII for now.
- The URL is something `yt-dlp` can fetch, and ffmpeg is installed. Everything runs on CPU,
  no GPU is assumed.
- The dialogue has at least two words, because single words match far too easily to give a
  meaningful timestamp.

## Extensibility (not implemented, just noted)

- If another visual check came up, say checking whether the detected face is turned towards
  the camera, this design supports it via the same pattern V2 and V3 used (an analyser class
  in `vision/`, a config block, one stage method in the pipeline) without needing a rewrite.
- If the speech engine had to change, only the `Transcriber` interface needs implementing.
  The matching and verification code does not know Whisper exists.
- If a second opinion on the match were needed, for example another speech engine acting as
  an independent judge, it plugs into the existing `Verifier` chain.
- If jobs had to survive a server restart, only `api/jobs.py` changes. The job store sits
  deliberately behind that one module.

## Results

Validated against the required input (`https://ok.ru/video/248244667877`, dialogue "My mind
rebels at stagnation"): status `found`, timestamp 00:05:25.312, frame 7799 at 23.976 fps,
match score 100. This run exercises the whole design end to end: the fast pass heard "My mind
rebelled. It's stagnation." and scored 86.7, the verification model retranscribed the window,
scored 100 on the correct wording and pulled the start back by 0.54 seconds, the frame came
back with one face at 0.953 confidence, and V3 scored the mouth at 0.074 over 74 frames, all
of which had a face, then moved the answer frame to the start of the most active window. That
run took 114 seconds after scanning 5 minutes 30 seconds of audio, 79.9 s of it transcription
and 27.7 s verification, with the audio already cached. The categorised cases below come from
eleven more dialogue cases across ten other videos, listed in `test_cases.json`.

The kinds of cases I actually ran, and what happened:

- Dialogue early or midway in the video: the streaming search stops as soon as the match
  settles, so it costs only as much transcription as the line is deep into the track. The JFK
  video in decision 2 is this case.
- Dialogue near the end, or not present at all: both cost a full transcription pass, which is
  the designed worst case. A miss triggers one retry with the voice filter off, then reports
  `not_found` with the three closest scoring windows as near misses.
- Dialogue spanning transcription segments: the sliding window is scored at every new word,
  so a line that straddles segment boundaries still matches as one window. This is the case
  that killed the fixed chunk design.
- Repeated phrase: the fast model locked onto the wrong occurrence and the verification model
  corrected it, as described in decision 3.
- Face present and speaking: the normal `found` case, covering both Sherlock clips and the
  zoo video. These are the upper cluster in Step 8's calibration, 0.046 to 0.123.
- Dialogue present but no face in the frame: a title card and an off camera line both came
  back `not_onscreen`, with the full match details still reported.
- Face present but mouth not moving: the face scored 0.343, which is exactly the wide shot
  case the 0.3 threshold exists for, while the mouth landed under the movement threshold, so
  the verdict was `not_onscreen`. The Spider-Man trailer's "my name is Peter Parker" is the
  same shape, a voice over read over a visible face.
- Onscreen at the start, camera cuts away for the rest ("im just a neighbour across the
  hall"): the sliding window scores the seconds the speaker is actually on camera and the
  cutaway afterwards cannot dilute them, so this is `found`. A single score over the whole
  line would have averaged this into silence.
- Line starts off camera, cuts to the speaker mid line ("you found a way to suppress mutating
  dna right"): V2 sees no face in the first frame and provisionally says `not_onscreen`, then
  V3 finds the movement later in the window, overturns it to `found`, and moves the answer
  frame to where the speaker appears. Without the frame move the record would have claimed
  onscreen over a frame showing nobody.
- Borderline canary (FRIENDS, "Not really"): matches "You really" at exactly the 80.0 match
  threshold and scores 0.032 mouth against the 0.030 threshold, with two faces in frame over
  18 frames. It currently reports onscreen. I keep it as a tripwire for threshold changes, not
  as ground truth, and it is the case that makes me want a sync check rather than a bare
  movement number.

Known and not handled, stated plainly rather than left to be discovered:

- Multiple faces in frame. The landmarker judges the highest scoring box, so the tracked face
  can change identity mid window. Short scoring windows contain the damage, half a second of
  one shot, but do not remove it. The FRIENDS case above is a two face frame.
- An occluded or turned away mouth. A speaker with a hand over their mouth or in profile
  contributes low or no openness, and the result reads as a still mouth.
- Minimum duration. A single frame flash of a speaking face can win a window if it lands in a
  scorable run; there is no floor on how briefly the speaker may appear.

### Test coverage

309 automated tests across 14 files. 302 of them run offline against fakes and small
generated fixtures in about 15 seconds, needing no network and no downloads. The seven that
touch real model weights or hosts are opt in: five behind a `network` marker, two behind a
`DL_RUN_MODEL_TESTS=1` env var.

| Module | Covered | Not covered |
| --- | --- | --- |
| `matching` (43) | normalisation, window tolerance, settling, dialogue split across segment boundaries, near miss ranking | non Latin scripts |
| `verification` (16) | confirm, reject on a >5 point drop, skip above 90, window clamping, one real tiny→base run | other ASR engines as verifier |
| `transcription` (23) | streaming word emission, early close, VAD retry, thread sizing and its fallbacks | real Whisper accuracy |
| `media` — audio, downloader, frames, probe (78) | PCM conversion, yt-dlp fallback and caching, clip offset arithmetic, frame index from fps, OpenCV seek failure → FFmpeg | live hosts |
| `vision/face_detector` (19) | threshold behaviour, box sorting, model download failure, fail open | detector accuracy on new footage |
| `vision/mouth_movement` (35) | detrending vs a drifting mouth, best window vs whole window, speech at the end of a line, cuts between still faces, gap bridging, crop upscaling, box carry expiry and cut on resume, threshold sits between the measured clusters | the corpus itself is 12 cases |
| `pipeline` (43) | every verdict path, frame moves and the re run face check, indeterminate fails open, stages disabled | end to end timing |
| `api`, `cli`, `config`, `models` (52) | job lifecycle, progress events, exit codes, serialisation | concurrency under load |

The gap I would close first is the corpus size. Every threshold in Step 8 is calibrated on 12
cases, and `test_default_threshold_sits_between_the_measured_clusters` pins the numbers those
clips produced, so the tests protect the calibration from drifting but cannot tell me it
generalises.

## What I'd improve with more time

- Download speed on hosts that throttle each connection. ok.ru serves about 200 KB/s on a
  single connection and that dominates the total runtime there. An external downloader like
  aria2c would parallelise it. With the thread fix in decision 7 this is now comfortably the
  slowest thing left.
- A throughput check in the test suite. Decision 7's bug was invisible to 300 passing tests
  because every one of them asserts on output, not on time. A test that scans a fixed clip and
  fails if it drops below some realtime factor would have caught it the day it appeared.
- Support for languages other than English. The blocker is one regex in the normaliser, plus
  testing with scripts other than Latin.
- Crowd scenes. Landmarking every detected face instead of only the highest scoring box, and
  taking the best mouth among them, would close the multiple faces gap listed above.
- A real lip sync check. The mouth check proves a mouth is moving, not that it is moving in
  time with the matched audio, so someone chewing or laughing in the shot still passes. A
  sync model like SyncNet compares the lip motion against the audio directly, and it would
  slot in as one more stage after the movement check.
