# Module: `web/`

A single static page served by the FastAPI app. No build step, no framework, no external assets.

```
web/templates/index.html    structure (three panels + history)
web/static/app.css          light theme, design tokens in :root, responsive at ≤ 960 px
web/static/app.js           vanilla JS client — talks only to /api/*
```

## Layout

```
┌ header: brand · config chips (fast / verify / threshold) · API docs · settings ┐
├───────────────────────────┬──────────────────────────────────────────────────┤
│ 1 Input                   │ 3 Result                                          │
│   source, dialogue, reuse │   empty ▸ found ▸ not found ▸ error               │
│   [Find dialogue][Cancel] │   frame image · OUTPUT block (PS format, Copy)    │
│ 2 Pipeline                │   facts: score, first pass, verification, face    │
│   stepper (9 stages)      │          check, mouth, audio scanned, video,      │
│                           │          processing time                          │
│   progress bar + message  │   timing bar · warnings · raw JSON                │
│   event log               │ Recent jobs (status, dialogue, source, result…)   │
└───────────────────────────┴──────────────────────────────────────────────────┘
```

## Behaviour (`app.js`)

* **Submit** → `POST /api/jobs`; a 422 is shown inline under the form as `[input] message`.
* **Watch** a job: `#job=<id>` is written to the URL (refresh/share re-attaches), the stepper and
  bar are reset, polling starts (1 s).
* **Progress rendering**: stage → stepper state (`done` ✓ / `active` pulsing / `failed` ✕ /
  `skipped` struck-through / `off` for stages the job's settings disabled). On `not_found` five
  stages are struck through: `download_video`, `verification`, `frame`, `face_detection` and
  `mouth_movement`; on any finished job an optional stage with no recorded timing is shown as
  skipped too. `fraction` → determinate bar
  (download %, audio %, transcription position) or indeterminate; stage timings appear next to each
  step when the result arrives; the event log appends only new entries from the ring buffer.
* **Result rendering**: `found` → image (+ Download link), the PS-format output block with a Copy
  button, fact cards (first pass shows "→ refined by verifier" when the timestamp moved),
  verification badges (green confirmed / red rejected-failed / grey), audio scanned vs video
  duration, a proportional timing bar with legend, amber warnings. `not_found` → the near-miss
  table. `failed`/`cancelled` → stage, message, a plain-language hint (network reset → "host
  unreachable from this network…", 429/bot check → "site is rate-limiting…").
* **Resilience**: a 404 while polling (server restarted, job expired) drops the job, clears the
  hash, re-enables the form with an explanation; transient errors retry 5× at 2 s before giving up.
* **History**: `GET /api/jobs` on load, after each finished job and via Refresh; "view" re-watches.
* **Settings modal**: the gear button (top right) opens a modal seeded from `GET /api/settings`
  with the server defaults. Edits live in a page-local `pageConfig` that is sent as
  `JobCreate.settings` with every submit and forgotten on refresh — nothing on the server is
  mutated. The three stage toggles are **order dependent**: `cascadeStages()` unchecks and
  disables each downstream box when its upstream is off, in the order verification → face
  detection → mouth movement, and `apply_setting_overrides` enforces the same cascade server side.
  Seven numeric fields are validated with `checkValidity()` before Apply (`max_occurrences`
  additionally rejects `0`, which `checkValidity()` cannot express alongside `-1`); there is a
  reset to defaults, and Escape or a backdrop click closes.
* **Occurrence chip**: while a job runs, progress `details.attempt` is shown as
  "Occurrence N of M" (just "Occurrence N" when `max_occurrences` is `-1`, which has no M).
  Without it the stepper appears to silently rewind to transcription when an occurrence is
  rejected as not onscreen.

## Styling notes (`app.css`)

Tokens in `:root` (`--primary`, `--ok`, `--warn`, `--err`, surfaces, lines). Light theme only by
request; base font 16.5 px. Everything is class-based, so re-theming means editing the token
block. Screens narrower than 960 px stack the two columns.

## Verifying UI changes

`node --check src/dialogue_locator/web/static/app.js` for syntax; `test_web_ui_served` checks the
page and assets are served; for visual checks, run the server and use headless Chrome:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --window-size=1380,1500 --virtual-time-budget=5000 --screenshot=ui.png "http://127.0.0.1:8000/#job=<id>"
```
