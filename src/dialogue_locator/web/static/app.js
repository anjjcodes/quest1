/* Dialogue Locator UI — vanilla JS; talks only to /api/*. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const STAGES = ["input", "download", "audio", "transcription", "verification", "download_video", "frame", "face_detection"];
  const POLL_MS = 1000;
  const MAX_POLL_FAILURES = 5;
  const TIMING_COLORS = { input: "#94a3b8", download: "#60a5fa", audio: "#34d399", transcription: "#f59e0b", matching: "#f59e0b", download_video: "#3b82f6", verification: "#a78bfa", frame: "#f472b6", face_detection: "#22d3ee" };

  const els = {
    form: $("job-form"), source: $("source"), dialogue: $("dialogue"), reuse: $("reuse"),
    submit: $("submit-btn"), cancel: $("cancel-btn"), formError: $("form-error"),
    progressCard: $("progress-card"), jobId: $("job-id"), stages: $("stages"),
    bar: $("progress-bar"), message: $("progress-message"), pct: $("progress-pct"), log: $("event-log"), logCount: $("log-count"),
    resultCard: $("result-card"), resultBadge: $("result-badge"), empty: $("result-empty"),
    found: $("result-found"), notFound: $("result-notfound"), errorBox: $("result-error"), json: $("result-json"),
    jobRows: $("job-rows"), health: $("health"), refresh: $("refresh-jobs"), copy: $("copy-btn"),
  };

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
      const job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ source, dialogue, reuse_cached_media: els.reuse.checked }) });
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
    try { await api(`/api/jobs/${currentJob}`, { method: "DELETE" }); } catch (_) { /* may have finished */ }
  });

  els.copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("output-block").textContent); els.copy.textContent = "Copied"; }
    catch (_) { els.copy.textContent = "Copy failed"; }
    setTimeout(() => { els.copy.textContent = "Copy"; }, 1500);
  });
  els.refresh.addEventListener("click", loadJobs);

  // ------------------------------------------------------------ watching a job
  function watch(jobId) {
    stopPolling();
    pollFailures = 0; renderedEvents = 0;
    currentJob = jobId;
    history.replaceState(null, "", `#job=${jobId}`);
    els.formError.hidden = true;
    els.log.innerHTML = ""; els.logCount.textContent = "";
    els.jobId.textContent = `#${jobId}`;
    els.progressCard.hidden = false;
    els.cancel.hidden = false; els.cancel.disabled = false;
    els.submit.disabled = true;
    showResultSection("empty");
    els.empty.querySelector("p").textContent = "Processing… the extracted frame, timestamp and matched text will appear here.";
    document.querySelector(".raw").hidden = true;
    els.resultBadge.hidden = true;
    for (const li of els.stages.children) { li.className = ""; li.querySelector(".meta").textContent = ""; }
    poll();
  }
  function stopPolling() { if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; } }

  function dropJob(reason) {
    stopPolling();
    currentJob = null;
    history.replaceState(null, "", location.pathname);
    els.progressCard.hidden = true;
    els.cancel.hidden = true;
    els.submit.disabled = false;
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
    if (["done", "failed", "cancelled"].includes(job.status)) {
      renderResult(job);
      els.cancel.hidden = true;
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
    const idx = STAGES.indexOf(stage);
    const finished = ["done", "failed", "cancelled"].includes(job.status);
    const notFound = job.status === "done" && job.result && job.result.status === "not_found";
    const notOnscreen = job.status === "done" && job.result && job.result.status === "not_onscreen";
    const timings = (job.result && job.result.stage_timings) || {};

    for (const li of els.stages.children) {
      const s = li.dataset.stage, i = STAGES.indexOf(s);
      li.className = "";
      const meta = li.querySelector(".meta");
      if (timings[s] != null) meta.textContent = fmtSecs(timings[s]);
      if (notFound && (s === "download_video" || s === "verification" || s === "frame" || s === "face_detection")) { li.classList.add("skipped"); meta.textContent = "skipped"; continue; }
      if (job.status === "done" || (stage === "done" && job.status === "running")) li.classList.add("done");
      else if (i < idx) li.classList.add("done");
      else if (i === idx) li.classList.add(finished ? "failed" : "active");
    }

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
      li.innerHTML = `<span class="t">${esc(fmtClock(e.at))}</span><span class="s">${esc(e.stage)}</span><span>${esc(stripPct(e.message))}${e.fraction != null ? " · " + pctText(e.fraction) : ""}</span>`;
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
    if (notOnscreen) banner.textContent = "Not an onscreen dialogue — the line is heard at this timestamp, but no human face is visible in the frame.";
    const url = `${job.frame_url}?t=${Date.now()}`;
    $("frame-img").src = url; $("frame-link").href = url; $("frame-download").href = url;
    $("frame-caption").textContent = r.frame ? `frame ${r.frame.frame_number} · ${r.frame.timestamp_str} · ${r.frame.width}×${r.frame.height} · ${r.frame.fps.toFixed(3)} fps` : "";
    $("output-block").textContent = [
      `Timestamp : ${r.timestamp}`,
      `Frame     : ${r.frame_number}`,
      `Text      : "${r.matched_text}"`,
    ].join("\n");

    $("r-score").textContent = `${r.match_score.toFixed(1)} / 100`;
    const moved = r.first_pass && r.match && Math.abs(r.first_pass.start - r.match.start) > 0.001;
    $("r-firstpass").textContent = r.first_pass ? `${r.first_pass.score.toFixed(1)} at ${r.first_pass.timestamp}${moved ? " → refined by verifier" : ""}` : "—";
    $("r-verify").innerHTML = (r.verifications || []).map((v) =>
      `<span class="badge ${esc(v.status)}" title="${esc(v.message || "")}">${esc(v.verifier)} · ${esc(v.status)}${v.score != null ? " " + v.score.toFixed(1) : ""}</span>`).join(" ")
      || `<span class="badge none">disabled</span>`;
    const fd = r.face_detection;
    $("r-face").innerHTML = r.face_present === true
      ? `<span class="badge confirmed">${fd.face_count} face${fd.face_count > 1 ? "s" : ""} · ${(fd.faces[0].confidence * 100).toFixed(0)}%</span>`
      : r.face_present === false
        ? `<span class="badge not_onscreen">no face in frame</span>`
        : `<span class="badge none">not run</span>`;
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
        `<span class="chip">v${esc(h.version)}</span>`,
      ].join("");
    } catch (_) { els.health.innerHTML = `<span class="chip">server unreachable</span>`; }
  }

  loadHealth();
  loadJobs();
  const m = location.hash.match(/^#job=([\w-]+)$/);
  if (m) watch(m[1]);
})();
