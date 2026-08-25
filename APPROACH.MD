# Approach

My working assumption is that an onscreen dialogue means the person saying the line is
visible in the frame while saying it. Everything below follows from that, and the design went
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
went these still count as onscreen dialogue, but a pure voice over does not. So version 2
added face detection: extract the matched frame and require a face in it. That is still not
enough, because person A can be speaking while the camera holds on person B, which is a
reaction shot, not an onscreen dialogue. So version 3 adds a mouth movement check on top of
the face check: track the mouth across the matched window and require it to actually move.

So each version narrows the question one step further:

- V1 asks: is the dialogue spoken in the audio?
- V2 asks: is a face visible in the matched frame?
- V3 asks: is that visible face moving its mouth?
- The future step asks: is that mouth movement in sync with this audio?

V3 is a heuristic, not a full active speaker detector. It filters out reaction shots and
voice overs, but it cannot prove the visible person is the one speaking. Only the sync step would provide much stronger evidence that the visible person is the one delivering the dialogue.

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
     mouth movement check (V3)
             │
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

A small Whisper model (`base`, about 12x realtime on CPU, greedy decoding for speed) streams
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

Two checks run on the result, both cheap and both calibrated from real frames:

- Face check (V2): MediaPipe BlazeFace looks for a face in the extracted frame. I lowered the
  confidence threshold to 0.3 because faces in wide cinematic shots score around 0.34 while
  junk stays under 0.2, and the usual 0.5 default misses distant faces. I set it by manually
  scoring detections on frames from the test videos, spanning closeups, medium shots and wide
  cinematic shots. No face means the
  verdict becomes `not_onscreen`, though all the match details are still reported.
- Mouth check (V3): only runs when a face was found. MediaPipe's Face Landmarker tracks the
  lips across the clip's frames for the matched window. Per frame I compute mouth openness as
  the inner lip gap divided by the mouth width, which makes the number independent of face
  size and camera distance. The movement score is the standard deviation of that signal:
  around 0.085 to 0.095 across my speaking test cases, and about 0.001 of landmark jitter on
  a genuinely static face, so the threshold sits at 0.02. Both numbers came from measuring
  that signal on real clips rather than guessing. The honest caveat is that a real actor
  holding still is not a static image: my one non speaking test case measured 0.0195, which
  clears the threshold by 0.0005, so the margin on that side is thin. A still mouth means a
  reaction shot or a voice over, so the verdict is
  `not_onscreen`. If fewer than 5 frames contain a face I call it indeterminate rather than
  guessing.
- Both checks fail open. If a check itself crashes, the localisation stands and a warning is
  attached, because a broken check should never delete a correct answer.

### Step 9. What comes out

The result is one JSON record plus the frame image: status (`found`, `not_found` or
`not_onscreen`), the timestamp as HH:MM:SS.mmm, the frame number, the matched text as heard,
the match score, what the first pass found versus what the verifier decided, the face and
mouth results, any warnings, near misses when nothing matched, and how long each stage took.
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

- **What I chose:** every word goes into a sliding window matcher as soon as the model
  produces it, and I close the decoder once a match is confirmed.
- **Alternative considered:** transcribe the whole video first, then search the text.
- **Why I chose X over Y:** transcription is the slowest stage and the line can be anywhere.
  On my JFK test video the scan stopped at 51% of the audio, so it cost half of a full
  transcription. The worst case, where the line is at the end or not there at all, costs the
  same as transcribing everything, so there is no downside.

### 3. Use two models, a fast one to search and a bigger one to verify

- **What I chose:** Whisper `base` (about 12x realtime on my CPU) does the streaming search.
  When it finds a candidate, `small` transcribes just the 12 seconds either side of it again
  to confirm the wording and tighten the timestamp. If the verifier scores more than 5 points
  below the first pass, I keep the first pass timestamp and attach a warning instead of
  trusting it.
- **Alternative considered:** one accurate model for everything, which runs at about 3x
  realtime (18 minutes instead of 5 on the reference video), or trusting the fast model alone.
- **Why I chose X over Y:** finding a candidate and confirming it need different levels of
  accuracy. The fast model does mishear words, and on one repeated phrase it was 6.4 seconds
  off until the bigger model fixed it. But I do not blindly trust the big model either,
  because it once heard "fronts" where the audio clearly says "trunks", and the 5 point rule
  caught that. So the cheap model finds, the careful model checks, and neither one gets the
  final say alone.

### 4. Download the audio first and the video last

- **What I chose:** for the search I fetch only the audio stream. After the match is verified
  I fetch a high quality clip of just the 5 seconds before and after it, purely to extract
  the frame.
- **Alternative considered:** downloading the whole video up front, which is what my first
  version did.
- **Why I chose X over Y:** if the dialogue is not in the video, I now pay for no video
  download at all. If it is, I pay for a few seconds instead of the whole file. On slow hosts
  like ok.ru that is the difference between minutes and an hour.

### 5. Define "onscreen" as a visible face whose mouth is moving

- **What I chose:** two small MediaPipe checks that run after the frame is extracted.
  BlazeFace answers whether there is a face in the frame at all, and the Face Landmarker
  tracks the lips across the matched window to see whether the mouth actually moves. Both
  fail open, meaning if a check crashes the result stands with a warning, because a broken
  check should never delete a correct answer.
- **Alternative considered:** heavier active speaker detection models like SyncNet that match
  lip motion to the audio.
- **Why I chose X over Y:** the two checks answer the real question, is someone visibly
  saying this line, in about a second of CPU each, and I could set every threshold from
  measurement instead of guessing. Mouth openness varies by about 0.085 to 0.095 while
  someone speaks and about 0.001 on a static face, so 0.02 separates them, though my one non
  speaking test case landed at 0.0195 and only just fell on the right side. A sync model
  would be more precise about who exactly is speaking, but it is much harder to run, tune and
  explain, and it can be added later as one more stage.

## Assumptions

- "Dialogue" means spoken audio, not subtitle text. The reference video has no subtitles. If
  subtitles were the target, I would swap the speech recognition stage for OCR on the frames
  and keep the rest of the pipeline.
- The first occurrence of the line is the one that is wanted.
- The audio is English. The text normaliser only handles ASCII for now.
- The URL is something `yt-dlp` can fetch, and ffmpeg is installed. Everything runs on CPU,
  no GPU is assumed.
- The dialogue has at least two words, because single words match far too easily to give a
  meaningful timestamp.

## Extensibility (not implemented, just noted)

- If another visual check came up, say checking for subtitle text in the frame, this design
  supports it via the same pattern V2 and V3 used (an analyser class in `vision/`, a config
  block, one stage method in the pipeline) without needing a rewrite.
- If the speech engine had to change, only the `Transcriber` interface needs implementing.
  The matching and verification code does not know Whisper exists.
- If a second opinion on the match were needed, for example another speech engine acting as
  an independent judge, it plugs into the existing `Verifier` chain.
- If jobs had to survive a server restart, only `api/jobs.py` changes. The job store sits
  deliberately behind that one module.

## Results

Validated against the required input (`https://ok.ru/video/248244667877`, dialogue "My mind
rebels at stagnation"): status `found`, timestamp 00:05:24.603, frame 7782 at 23.976 fps,
match score 100. This run exercises the whole design end to end: the fast pass heard "My mind
rebelled. It's stagnation." and scored 86.7, the verification model retranscribed the window,
scored 100 on the correct wording and pulled the timestamp back by 0.54 seconds, then the
frame came back with one face at 0.949 confidence and a mouth movement score of 0.087 over 74
frames, all of which had a face. That run took 59 seconds in total with the audio already
cached, 38.7 s of it transcription and 15.6 s verification, after scanning 5 minutes 30
seconds of audio. The categorised cases below are drawn from testing across nine more videos
covering the scenarios this design was built to handle.

The kinds of cases I actually ran, and what happened:

- Dialogue early or midway in the video: the streaming search stops as soon as the match
  settles. On my JFK test video the scan stopped at 51% of the audio, so it cost about half
  of a full transcription.
- Dialogue near the end, or not present at all: both cost a full transcription pass, which is
  the designed worst case. A miss triggers one retry with the voice filter off, then reports
  `not_found` with the three closest scoring windows as near misses.
- Dialogue spanning transcription segments: the sliding window is scored at every new word,
  so a line that straddles segment boundaries still matches as one window. This is the case
  that killed the fixed chunk design.
- Repeated phrase: the fast model locked onto the wrong occurrence, 6.4 seconds off, and the
  verification model corrected it.
- Face present and speaking: the normal `found` case. On the Sherlock clip the face scored
  0.846 and the mouth 0.085, on the ok.ru required input 0.949 and 0.087.
- Dialogue present but no face in the frame: a title card and an off camera line both came
  back `not_onscreen` with no face detected, and the full match details still reported.
- Face present but mouth not moving: the face scored 0.343, which is exactly the wide shot
  case the 0.3 threshold exists for, while the mouth scored 0.0195 against the 0.02
  threshold, so the verdict was `not_onscreen`. That is the narrowest margin I have seen and
  the reason I want a sync check rather than a bare movement number.
- Known limitation, reaction shot with a background speaker: the mouth check judges the most
  prominent face, so a moving mouth in the background behind a still face in the foreground
  can be judged on the wrong person. This stays open until a sync check exists.

The automated tests cover the sliding window case where a dialogue is split across segment
boundaries, the path where the verifier rejects a bad second opinion and keeps the first pass
timestamp, and the near miss reporting when nothing matches.

## What I'd improve with more time

- Download speed on hosts that throttle each connection. ok.ru serves about 200 KB/s on a
  single connection and that dominates the total runtime there. An external downloader like
  aria2c would parallelise it.
- Support for languages other than English. The blocker is one regex in the normaliser, plus
  testing with scripts other than Latin.
- Crowd scenes. The mouth check judges the most prominent face, so a speaker in the
  background behind a silent face in the foreground could be judged on the wrong person.
  Checking every detected face would fix it.
- A real lip sync check. The mouth check proves a mouth is moving, not that it is moving in
  time with the matched audio, so someone chewing or laughing in the shot still passes. A
  sync model like SyncNet compares the lip motion against the audio directly, and it would
  slot in as one more stage after the movement check.
