/* Dialogue Locator UI — vanilla JS; talks only to /api/*. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const STAGES = ["input", "download", "audio", "transcription", "verification", "download_video", "frame", "face_detection", "mouth_movement"];
  const POLL_MS = 1000;
  const MAX_POLL_FAILURES = 5;
  const TIMING_COLORS = { input: "#94a3b8", download: "#60a5fa", audio: "#34d399", transcription: "#f59e0b", matching: "#f59e0b", download_video: "#3b82f6", verification: "#a78bfa", frame: "#f472b6", face_detection: "#22d3ee", mouth_movement: "#fb7185" };

  const els = {
    form: $("job-form"), source: $("source"), dialogue: $("dialogue"), reuse: $("reuse"),
    submit: $("submit-btn"), cancel: $("cancel-btn"), formError: $("form-error"),
    jobId: $("job-id"), stages: $("stages"),
    bar: $("progress-bar"), message: $("progress-message"), pct: $("progress-pct"), attempt: $("progress-attempt"), log: $("event-log"), logCount: $("log-count"),
    resultCard: $("result-card"), resultBadge: $("result-badge"), empty: $("result-empty"),
    found: $("result-found"), notFound: $("result-notfound"), errorBox: $("result-error"), json: $("result-json"),
    jobRows: $("job-rows"), health: $("health"), refresh: $("refresh-jobs"), copy: $("copy-btn"),
    settingsBtn: $("settings-btn"), settingsOverlay: $("settings-overlay"), settingsClose: $("settings-close"),
    settingsSave: $("settings-save"), settingsStatus: $("settings-status"),
  };

  // Settings live in this page only: serverDefaults comes from /api/settings,
  // pageConfig is what the user tweaked here. Jobs carry pageConfig with them;
  // refreshing the page discards it and starts from the defaults again.
  let serverDefaults = null;
  let pageConfig = null;

  let currentJob = null;
  let pollTimer = null;
  let pollFailures = 0;
  let renderedEvents = 0;

  // ------------------------------------------------------------ helpers
  const api = async (path, opts = {}) => {
    const res = await fetch(path, { headers: { "content-type": "application/json" }, ...opts });
    let body = null;
    try { body = await res.json(); } catch (_) { /* no body */ }
    if (!res.ok) {
      const msg = (body && (body.message || (body.detail && JSON.stringify(body.detail)))) || res.statusText;
      const err = new Error(msg); err.body = body; err.status = res.status; throw err;
    }
    return body;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtClock = (iso) => new Date(iso).toLocaleTimeString([], { hour12: false });
  const fmtSecs = (s) => (s == null ? "" : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${s.toFixed(1)}s`);
  const fmtAgo = (iso) => {
    const d = (Date.now() - new Date(iso).getTime()) / 1000;
    if (d < 60) return "just now";
    if (d < 3600) return `${Math.floor(d / 60)} min ago`;
    if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
    return new Date(iso).toLocaleString();
  };
  const pctText = (f) => (f == null ? "" : `${Math.round(f * 100)}%`);
  const stripPct = (msg) => msg.replace(/\s\d{1,3}%$/, "");  // server messages may end with "NN%"

  // ------------------------------------------------------------ submit / cancel
  els.form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    els.formError.hidden = true;
    const source = els.source.value.trim(), dialogue = els.dialogue.value.trim();
    if (!source || !dialogue) { showFormError("Both the video source and the dialogue are required."); return; }
    els.submit.disabled = true;
    try {
      const job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ source, dialogue, reuse_cached_media: els.reuse.checked, settings: pageConfig }) });
      watch(job.job_id);
    } catch (err) {
      showFormError((err.body && err.body.stage ? `[${err.body.stage}] ` : "") + err.message);
      els.submit.disabled = false;
    }
  });
  function showFormError(text) { els.formError.textContent = text; els.formError.hidden = false; }

  els.cancel.addEventListener("click", async () => {
    if (!currentJob) return;
    els.cancel.disabled = true;
    try { await api(`/api/jobs/${currentJob}`, { method: "DELETE" }); }
    catch (_) { els.cancel.disabled = false; /* request failed; allow another attempt */ }
  });

  els.copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("output-block").textContent); els.copy.textContent = "Copied"; }
    catch (_) { els.copy.textContent = "Copy failed"; }
    setTimeout(() => { els.copy.textContent = "Copy"; }, 1500);
  });
  els.refresh.addEventListener("click", loadJobs);

  // ------------------------------------------------------------ watching a job
  function resetPipelineIdle() {
    els.jobId.textContent = "";
    els.log.innerHTML = ""; els.logCount.textContent = "";
    els.message.textContent = "Waiting for a job…"; els.pct.textContent = "";
    els.bar.className = "bar"; els.bar.style.width = "0";
    for (const li of els.stages.children) { li.className = ""; li.querySelector(".meta").textContent = ""; }
  }

  function watch(jobId) {
    stopPolling();
    pollFailures = 0; renderedEvents = 0;
    currentJob = jobId;
    history.replaceState(null, "", `#job=${jobId}`);
    els.formError.hidden = true;
    resetPipelineIdle();
    els.jobId.textContent = `#${jobId}`;
    els.message.textContent = "Starting…";
    els.cancel.hidden = true; els.cancel.disabled = false;  // shown by poll() once the job is live
    els.submit.disabled = true;
    showResultSection("empty");
    els.empty.querySelector("p").textContent = "Processing… the extracted frame, timestamp and matched text will appear here.";
    document.querySelector(".raw").hidden = true;
    els.resultBadge.hidden = true;
    poll();
  }
  function stopPolling() { if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; } }

  function dropJob(reason) {
    stopPolling();
    currentJob = null;
    history.replaceState(null, "", location.pathname);
    resetPipelineIdle();
    els.cancel.hidden = true;
    els.submit.disabled = false;
    els.empty.querySelector("p").textContent = "The extracted frame, timestamp and matched text will appear here.";
    showFormError(reason);
    loadJobs();
  }

  async function poll() {
    if (!currentJob) return;
    let job;
    try { job = await api(`/api/jobs/${currentJob}`); pollFailures = 0; }
    catch (err) {
      if (err.status === 404) { dropJob(`Job ${currentJob} no longer exists on the server (it restarted or the job expired). Start a new job.`); return; }
      if (++pollFailures >= MAX_POLL_FAILURES) { dropJob(`Lost contact with the server while watching job ${currentJob}: ${err.message}`); return; }
      els.message.textContent = `Server not responding — retrying ${pollFailures}/${MAX_POLL_FAILURES}…`;
      pollTimer = setTimeout(poll, POLL_MS * 2);
      return;
    }
    renderProgress(job);
    const finished = ["done", "failed", "cancelled"].includes(job.status);
    els.cancel.hidden = finished;
    if (finished) {
      renderResult(job);
      els.submit.disabled = false;
      currentJob = null;
      loadJobs();
    } else {
      pollTimer = setTimeout(poll, POLL_MS);
    }
  }

  // ------------------------------------------------------------ progress
  function renderProgress(job) {
    const p = job.progress;
    const stage = p ? p.stage : "input";
    // "matching" is folded into the transcription stage in the stepper.
    const idx = STAGES.indexOf(stage === "matching" ? "transcription" : stage);
    const finished = ["done", "failed", "cancelled"].includes(job.status);
    const notFound = job.status === "done" && job.result && job.result.status === "not_found";
    const notOnscreen = job.status === "done" && job.result && job.result.status === "not_onscreen";
    const timings = (job.result && job.result.stage_timings) || {};

    for (const li of els.stages.children) {
      const s = li.dataset.stage, i = STAGES.indexOf(s);
      li.className = "";
      const meta = li.querySelector(".meta");
      if (pageConfig && pageConfig.stages && pageConfig.stages[s] === false) { li.classList.add("skipped"); meta.textContent = "off"; continue; }
      if (timings[s] != null) meta.textContent = fmtSecs(timings[s]);
      if (notFound && (s === "download_video" || s === "verification" || s === "frame" || s === "face_detection" || s === "mouth_movement")) { li.classList.add("skipped"); meta.textContent = "skipped"; continue; }
      // Only a *successful* job can mark an optional stage as skipped; on a
      // failed/cancelled job a missing timing means the job died earlier.
      if (job.status === "done" && s !== "input" && s !== "download" && s !== "audio" && s !== "transcription" && timings[s] == null) { li.classList.add("skipped"); meta.textContent = "skipped"; continue; }
      if (job.status === "done" || (stage === "done" && job.status === "running")) li.classList.add("done");
      else if (i < idx) li.classList.add("done");
      else if (i === idx) li.classList.add(finished ? "failed" : "active");
    }

    // Which occurrence of the dialogue is being evaluated (max_occurrences > 1):
    // without this chip the stepper silently rewinds to transcription when an
    // occurrence is rejected as not onscreen.
    // max_attempts is absent when every occurrence is being evaluated (-1),
    // so there is no "of N" to show - just which one we are on.
    const att = p && p.details && p.details.attempt ? p.details : null;
    els.attempt.hidden = !(att && job.status === "running");
    if (att) els.attempt.textContent = att.max_attempts
      ? `Occurrence ${att.attempt} of ${att.max_attempts}`
      : `Occurrence ${att.attempt}`;

    if (job.status === "queued") {
      els.message.textContent = "Queued — waiting for a free worker…"; els.pct.textContent = "";
      els.bar.className = "bar indeterminate";
    } else if (p) {
      els.message.textContent = stripPct(p.message);
      els.pct.textContent = pctText(p.fraction);
      if (p.fraction == null) els.bar.className = "bar indeterminate";
      else { els.bar.className = "bar"; els.bar.style.width = `${Math.round(p.fraction * 100)}%`; }
    }
    if (job.status === "done") { els.bar.className = "bar done"; els.bar.style.width = "100%"; els.message.textContent = notFound ? "Finished — dialogue not found" : notOnscreen ? "Finished — not an onscreen dialogue" : "Finished"; els.pct.textContent = fmtSecs(timings.total); }
    if (job.status === "cancelled") { els.bar.className = "bar failed"; els.message.textContent = "Cancelled"; els.pct.textContent = ""; }
    if (job.status === "failed") { els.bar.className = "bar failed"; els.message.textContent = `Failed during ${job.error ? job.error.stage : "?"}`; els.pct.textContent = ""; }

    const log = job.progress_log || [];
    if (log.length < renderedEvents) { els.log.innerHTML = ""; renderedEvents = 0; }  // ring buffer rolled
    for (let i = renderedEvents; i < log.length; i++) {
      const e = log[i];
      const li = document.createElement("li");
      const occ = e.details && e.details.attempt ? `#${e.details.attempt} ` : "";
      li.innerHTML = `<span class="t">${esc(fmtClock(e.at))}</span><span class="s">${esc(occ + e.stage)}</span><span>${esc(stripPct(e.message))}${e.fraction != null ? " · " + pctText(e.fraction) : ""}</span>`;
      els.log.appendChild(li);
    }
    renderedEvents = log.length;
    els.logCount.textContent = log.length ? `${log.length} events` : "";
    els.log.scrollTop = els.log.scrollHeight;
  }

  // ------------------------------------------------------------ result
  function showResultSection(which) {
    for (const [name, el] of Object.entries({ empty: els.empty, found: els.found, notfound: els.notFound, error: els.errorBox })) el.hidden = name !== which;
  }
  function badge(text, cls) { els.resultBadge.textContent = text; els.resultBadge.className = `badge ${cls}`; els.resultBadge.hidden = false; }

  function renderResult(job) {
    els.json.textContent = JSON.stringify(job.result || job.error || job, null, 2);
    document.querySelector(".raw").hidden = false;

    if (job.status !== "done") {
      showResultSection("error");
      badge(job.status, job.status);
      const e = job.error || { stage: job.status, message: job.status, details: {} };
      $("err-stage").textContent = e.stage;
      $("err-message").textContent = e.message;
      $("err-hint").textContent = errorHint(e);
      const det = e.details && Object.keys(e.details).length ? JSON.stringify(e.details, null, 2) : "";
      $("err-details").textContent = det; $("err-details").hidden = !det;
      return;
    }

    const r = job.result;
    if (r.status === "not_found") {
      showResultSection("notfound");
      badge("not found", "not_found");
      $("nf-dialogue").textContent = r.dialogue;
      $("nf-rows").innerHTML = (r.near_misses || []).map((c) =>
        `<tr><td class="num">${c.score.toFixed(1)}</td><td class="mono">${esc(c.timestamp)}</td><td>“${esc(c.matched_text)}”</td></tr>`).join("")
        || `<tr><td colspan="3" class="muted">No speech was transcribed.</td></tr>`;
      $("nf-scanned").textContent = r.transcribed_seconds != null ? `Scanned ${fmtSecs(r.transcribed_seconds)} of audio${r.video && r.video.duration ? ` (video is ${fmtSecs(r.video.duration)})` : ""}.` : "";
      return;
    }

    showResultSection("found");
    const notOnscreen = r.status === "not_onscreen";
    const banner = $("r-onscreen");
    banner.hidden = !notOnscreen;
    if (notOnscreen) {
      // The verdict comes from the window scan, so the reason must too: the
      // frame's own face says nothing about a line that opens off camera.
      const scanned = r.mouth_movement;  // null when the check did not run
      // A null verdict here always means "nothing held up as a face": when the
      // scan genuinely cannot tell, the pipeline keeps the result as found.
      banner.textContent = scanned && scanned.moving === false
        ? "Not an onscreen dialogue — a face is on camera during the line, but its mouth is not moving (narration, dubbing or a reaction shot)."
        : scanned && scanned.frames_with_face === 0
          ? "Not an onscreen dialogue — the line is heard here, but no face appears at any point during it."
          : scanned
            ? `Not an onscreen dialogue — nothing that holds up as a face during the line: facial landmarks in only ${scanned.frames_with_face} of ${scanned.frames_analyzed} frames.`
            : "Not an onscreen dialogue — the line is heard at this timestamp, but no human face is visible in the frame.";
    }
    // The mouth check moves the answer frame to where the speaker comes on
    // camera, which can be later than where the line starts in the audio.
    const mm = r.mouth_movement;
    // Only worth explaining when the gap is visible to a viewer: a two-frame
    // move off a title card needs no banner, a cut away from the speaker does.
    const MOVED_NOTE_SECONDS = 0.2;
    const movedFrame = Boolean(
      mm && r.match && r.frame && r.frame.timestamp - r.match.start > MOVED_NOTE_SECONDS);
    const movedToSpeaker = Boolean(movedFrame && mm.movement_start != null);
    const movedBanner = $("r-moved");
    movedBanner.hidden = !movedFrame;
    if (movedFrame) movedBanner.textContent =
      `The line starts at ${r.match.timestamp} while the camera is elsewhere. `
      + (movedToSpeaker
        ? `Showing ${r.frame.timestamp_str}, where the speaker is on camera saying it.`
        : `Showing ${r.frame.timestamp_str}, where the face judged for this line first appears.`);
    const hasFrame = Boolean(job.frame_url && r.frame);
    $("frame-figure").hidden = !hasFrame;
    if (hasFrame) {
      const url = `${job.frame_url}?t=${Date.now()}`;
      $("frame-img").src = url; $("frame-link").href = url; $("frame-download").href = url;
      $("frame-download").setAttribute("download", `frame_${job.job_id}.${String(r.frame.image_path || "").endsWith(".png") ? "png" : "jpg"}`);
      $("frame-caption").textContent = `frame ${r.frame.frame_number} · ${r.frame.timestamp_str} · ${r.frame.width}×${r.frame.height} · ${r.frame.fps.toFixed(3)} fps`;
    }
    $("output-block").textContent = [
      `Timestamp : ${r.timestamp}`,
      `Frame     : ${r.frame_number ?? "—"}`,
      ...(movedFrame ? [`Line audio: ${r.match.timestamp} (starts off camera)`] : []),
      `Text      : "${r.matched_text}"`,
    ].join("\n");

    $("r-score").textContent = `${r.match_score.toFixed(1)} / 100`;
    const moved = r.first_pass && r.match && Math.abs(r.first_pass.start - r.match.start) > 0.001;
    $("r-firstpass").textContent = r.first_pass ? `${r.first_pass.score.toFixed(1)} at ${r.first_pass.timestamp}${moved ? " → refined by verifier" : ""}` : "—";
    $("r-verify").innerHTML = (r.verifications || []).map((v) =>
      `<span class="badge ${esc(v.status)}" title="${esc(v.message || "")}">${esc(v.verifier)} · ${esc(v.status)}${v.score != null ? " " + v.score.toFixed(1) : ""}</span>`).join(" ")
      || `<span class="badge none">disabled</span>`;
    // Presence, not a head count: the detector runs at a deliberately loose
    // threshold (real faces in wide shots score ~0.30), so it also fires on
    // non-faces. Only the best box is ever judged, so that is what is reported.
    const fd = r.face_detection;
    $("r-face").innerHTML = r.face_present === true
      ? `<span class="badge confirmed">face detected · ${(fd.faces[0].confidence * 100).toFixed(0)}%</span>`
      : r.face_present === false
        ? `<span class="badge not_onscreen">no face in frame</span>`
        : `<span class="badge none">not run</span>`;
    $("r-mouth").innerHTML = !mm
      ? `<span class="badge none">not run</span>`
      : mm.moving === true
        ? `<span class="badge confirmed">moving · score ${mm.movement_score.toFixed(3)}${movedToSpeaker ? ` · from ${esc(r.frame.timestamp_str)}` : ""}</span>`
        : mm.moving === false
          ? `<span class="badge not_onscreen">not moving · score ${(mm.movement_score ?? 0).toFixed(3)}</span>`
          : `<span class="badge none" title="facial landmarks in ${mm.frames_with_face} of ${mm.frames_analyzed} frames">no face to judge</span>`;
    $("r-scanned").textContent = r.transcribed_seconds != null ? `${fmtSecs(r.transcribed_seconds)}${r.video && r.video.duration ? ` of ${fmtSecs(r.video.duration)} (${Math.round(100 * r.transcribed_seconds / r.video.duration)}%)` : ""}` : "—";
    $("r-video").textContent = r.video ? `${r.video.title || "—"} · ${r.video.width}×${r.video.height} @ ${(r.video.fps || 0).toFixed(3)} fps` : "—";
    $("r-total").textContent = fmtSecs((r.stage_timings || {}).total);

    const t = r.stage_timings || {}, total = t.total || 1;
    const parts = STAGES.filter((s) => t[s]).map((s) => ({ s, v: t[s] }));
    $("r-timings").innerHTML = parts.map(({ s, v }) => `<span style="width:${(100 * v / total).toFixed(2)}%;background:${TIMING_COLORS[s]}" title="${s} ${fmtSecs(v)}"></span>`).join("");
    let legend = document.querySelector(".timings-legend");
    if (!legend) { legend = document.createElement("div"); legend.className = "timings-legend"; $("r-timings").after(legend); }
    legend.innerHTML = parts.map(({ s, v }) => `<span><i style="background:${TIMING_COLORS[s]}"></i>${esc(s)} ${fmtSecs(v)}</span>`).join("");

    $("r-warnings").innerHTML = (r.warnings || []).map((w) => `<li>${esc(w)}</li>`).join("");
    if (notOnscreen) badge("not an onscreen dialogue", "not_onscreen");
    else if ((r.warnings || []).length) badge("found · with warnings", "warn");
    else badge("found", "found");
  }

  function errorHint(e) {
    const m = (e.message || "").toLowerCase();
    if (m.includes("connection reset") || m.includes("connection refused") || m.includes("name or service") || m.includes("timed out"))
      return "The video host looks unreachable from this network (blocked or offline). Try another network/VPN, or download the file elsewhere and enter its local path.";
    if (m.includes("confirm you") || m.includes("429"))
      return "The site is rate-limiting this machine. Wait a while, switch network, or run yt-dlp with browser cookies and use the local file.";
    if (e.stage === "input") return "Fix the input above and try again.";
    if (e.stage === "audio") return "The video seems to have no usable audio track, so there is nothing to transcribe.";
    return "";
  }

  // ------------------------------------------------------------ history / health
  async function loadJobs() {
    let jobs;
    try { jobs = (await api("/api/jobs")).jobs; } catch (_) { return; }
    if (!jobs.length) { els.jobRows.innerHTML = `<tr><td colspan="6" class="muted">No jobs yet.</td></tr>`; return; }
    els.jobRows.innerHTML = jobs.map((j) => {
      const res = j.result
        ? (j.result.status === "found" ? `${j.result.timestamp} · frame ${j.result.frame_number}`
          : j.result.status === "not_onscreen" ? `not onscreen · ${j.result.timestamp}`
          : "not found")
        : (j.error ? `${j.error.stage}: ${j.error.message}` : "");
      return `<tr>
        <td><span class="status ${esc(j.status)}">${esc(j.status)}</span></td>
        <td>“${esc(j.dialogue)}”</td>
        <td class="src" title="${esc(j.source)}">${esc(j.source)}</td>
        <td class="mono small">${esc(res.slice(0, 90))}</td>
        <td class="when small">${esc(fmtAgo(j.created_at))}</td>
        <td><a class="link" href="#job=${esc(j.job_id)}" data-job="${esc(j.job_id)}">view</a></td></tr>`;
    }).join("");
  }
  els.jobRows.addEventListener("click", (ev) => {
    const a = ev.target.closest("a[data-job]");
    if (a) { ev.preventDefault(); watch(a.dataset.job); }
  });

  async function loadHealth() {
    try {
      const h = await api("/api/health");
      els.health.innerHTML = [
        `<span class="chip">fast <b>${esc(h.fast_model)}</b></span>`,
        `<span class="chip">verify <b>${esc(h.verification_enabled ? h.verify_model : "off")}</b></span>`,
        `<span class="chip">threshold <b>${esc(h.match_threshold)}</b></span>`,
      ].join("");
    } catch (_) { els.health.innerHTML = `<span class="chip">server unreachable</span>`; }
  }

  // ------------------------------------------------------------ settings
  const STAGE_BOXES = [$("st-verification"), $("st-face"), $("st-mouth")];  // dependency order

  async function loadSettings() {
    try {
      serverDefaults = await api("/api/settings");
      pageConfig = JSON.parse(JSON.stringify(serverDefaults));
    } catch (_) { /* older server */ }
    return pageConfig;
  }

  function cascadeStages() {
    // whisper -> face -> lip: turning a stage off turns off everything after
    // it; a stage can only be (re-)enabled while the one before it is on.
    for (let i = 0; i < STAGE_BOXES.length; i++) {
      const upstreamOn = i === 0 || (STAGE_BOXES[i - 1].checked && !STAGE_BOXES[i - 1].disabled);
      STAGE_BOXES[i].disabled = !upstreamOn;
      if (!upstreamOn) STAGE_BOXES[i].checked = false;
      STAGE_BOXES[i].closest("label").classList.toggle("disabled", STAGE_BOXES[i].disabled);
    }
  }
  STAGE_BOXES.forEach((box) => box.addEventListener("change", cascadeStages));

  function fillSettingsForm(s) {
    STAGE_BOXES[0].checked = s.stages.verification;
    STAGE_BOXES[1].checked = s.stages.face_detection;
    STAGE_BOXES[2].checked = s.stages.mouth_movement;
    cascadeStages(null);
    $("cfg-threshold").value = s.match_threshold;
    $("cfg-occurrences").value = s.max_occurrences;
    $("cfg-face-conf").value = s.face_min_confidence;
    $("cfg-mouth-thr").value = s.mouth_movement_threshold;
    $("cfg-mouth-frames").value = s.mouth_min_face_frames;
    $("cfg-mouth-window").value = s.mouth_max_window_seconds;
    $("cfg-max-height").value = s.max_video_height;
  }

  async function openSettings() {
    els.settingsStatus.textContent = "";
    const s = pageConfig || await loadSettings();
    if (!s) { els.settingsStatus.textContent = "Could not load settings from the server."; }
    else fillSettingsForm(s);
    els.settingsOverlay.hidden = false;
    const first = els.settingsOverlay.querySelector("input:not(:disabled), button");
    if (first) first.focus();
  }
  function closeSettings() { els.settingsOverlay.hidden = true; els.settingsBtn.focus(); }

  els.settingsBtn.addEventListener("click", openSettings);
  els.settingsClose.addEventListener("click", closeSettings);
  els.settingsOverlay.addEventListener("click", (ev) => { if (ev.target === els.settingsOverlay) closeSettings(); });
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !els.settingsOverlay.hidden) closeSettings(); });

  $("settings-reset").addEventListener("click", () => {
    if (!serverDefaults) return;
    fillSettingsForm(serverDefaults);
    els.settingsStatus.textContent = "Defaults restored — press Apply to use them.";
  });

  els.settingsSave.addEventListener("click", () => {
    // A blank or out-of-range number would only surface as a server 422 at
    // submit time; catch it here, next to the field.
    const fields = ["cfg-threshold", "cfg-occurrences", "cfg-face-conf", "cfg-mouth-thr", "cfg-mouth-frames", "cfg-mouth-window", "cfg-max-height"].map($);
    for (const input of fields) {
      if (input.id === "cfg-occurrences" && parseInt(input.value, 10) === 0) {
        input.setCustomValidity("Use 1 or more, or -1 for every occurrence.");
        input.reportValidity();
        input.setCustomValidity("");
        input.focus();
        els.settingsStatus.textContent = "Fix the highlighted value first.";
        return;
      }
      if (input.value.trim() === "" || !input.checkValidity()) {
        input.reportValidity();
        input.focus();
        els.settingsStatus.textContent = "Fix the highlighted value first.";
        return;
      }
    }
    // Applied in this page only: sent along with each job, gone on refresh.
    pageConfig = {
      stages: {
        verification: STAGE_BOXES[0].checked,
        face_detection: STAGE_BOXES[1].checked,
        mouth_movement: STAGE_BOXES[2].checked,
      },
      match_threshold: parseFloat($("cfg-threshold").value),
      max_occurrences: parseInt($("cfg-occurrences").value, 10),
      face_min_confidence: parseFloat($("cfg-face-conf").value),
      mouth_movement_threshold: parseFloat($("cfg-mouth-thr").value),
      mouth_min_face_frames: parseInt($("cfg-mouth-frames").value, 10),
      mouth_max_window_seconds: parseFloat($("cfg-mouth-window").value),
      max_video_height: parseInt($("cfg-max-height").value, 10),
    };
    els.settingsStatus.textContent = "Applied to jobs from this page — refresh resets to defaults.";
  });

  loadHealth();
  loadJobs();
  loadSettings();
  const m = location.hash.match(/^#job=([\w-]+)$/);
  if (m) watch(m[1]);
})();
