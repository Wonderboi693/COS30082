// Face Attendance web client. Talks to the FastAPI backend (src/api) on the same origin.
const $ = (id) => document.getElementById(id);
const FRAME_WIDTH = 640;      // frames are downscaled before upload
const FRAME_GAP_MS = 250;     // pause between recognition requests

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}
const putJSON = (path, data) => api(path, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab));
  ({ people: loadPeople, log: loadLog, models: loadModels })[btn.dataset.tab]?.();
}));

// ------------------------------------------------------------------ status + settings
function applyStatus(s) {
  const el = $("status");
  el.className = "status ok";
  el.textContent = `${s.model_label} · ${s.device} · ${s.enrolled} enrolled · liveness ${s.liveness.enabled ? "on" : "off"} · emotion ${s.emotion.enabled ? "on" : "off"}`;
  $("threshold").value = s.threshold;
  $("thresholdVal").textContent = s.threshold.toFixed(2);
  $("toggleLiveness").checked = s.liveness.enabled; $("toggleLiveness").disabled = !s.liveness.available;
  $("toggleEmotion").checked = s.emotion.enabled; $("toggleEmotion").disabled = !s.emotion.available;
}
async function loadStatus() {
  try { applyStatus(await api("/api/health")); }
  catch (e) { $("status").className = "status bad"; $("status").textContent = "Backend not reachable"; }
}
async function loadModelSelect() {
  const models = await api("/api/models");
  $("modelSelect").innerHTML = models.map((m) =>
    `<option value="${esc(m.key)}" ${m.active ? "selected" : ""} ${m.available ? "" : "disabled"}>` +
    `${esc(m.label)} — ${esc(m.owner)}${m.val_auc ? ` (AUC ${m.val_auc.toFixed(3)})` : ""}${m.available ? "" : " — no checkpoint"}</option>`).join("");
}
$("modelSelect").addEventListener("change", async (e) => {
  const sel = e.target; sel.disabled = true;
  try {
    const r = await api("/api/models/active", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key: sel.value }) });
    if (r.reembedded_people) $("liveLatency").textContent = `Re-embedded ${r.reembedded_people} enrolled people for ${r.model}.`;
  } catch (err) { alert(err.message); }
  sel.disabled = false;
  await loadStatus(); await loadModelSelect();
});
$("threshold").addEventListener("input", (e) => ($("thresholdVal").textContent = Number(e.target.value).toFixed(2)));
$("threshold").addEventListener("change", async (e) => applyStatus(await putJSON("/api/settings", { threshold: Number(e.target.value) })));
$("toggleLiveness").addEventListener("change", async (e) => applyStatus(await putJSON("/api/settings", { liveness_enabled: e.target.checked })));
$("toggleEmotion").addEventListener("change", async (e) => applyStatus(await putJSON("/api/settings", { emotion_enabled: e.target.checked })));

// ------------------------------------------------------------------ camera helpers
async function openCamera(video, placeholder) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
  video.srcObject = stream;
  await video.play();
  placeholder.style.display = "none";
  return stream;
}
function closeCamera(video, stream, placeholder) {
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  placeholder.style.display = "";
}
function grabFrame(video) {
  const scale = Math.min(1, FRAME_WIDTH / video.videoWidth);
  const c = document.createElement("canvas");
  c.width = Math.round(video.videoWidth * scale);
  c.height = Math.round(video.videoHeight * scale);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return new Promise((resolve) => c.toBlob(resolve, "image/jpeg", 0.85));
}

// ------------------------------------------------------------------ live check-in
let liveStream = null, liveRunning = false;
$("liveStart").addEventListener("click", async () => {
  try { liveStream = await openCamera($("liveVideo"), $("livePlaceholder")); }
  catch (e) { alert("Cannot open camera: " + e.message); return; }
  liveRunning = true;
  $("liveStart").disabled = true; $("liveStop").disabled = false;
  liveLoop();
});
$("liveStop").addEventListener("click", () => {
  liveRunning = false;
  closeCamera($("liveVideo"), liveStream, $("livePlaceholder"));
  const cv = $("liveOverlay"); cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
  $("liveStart").disabled = false; $("liveStop").disabled = true;
});

async function liveLoop() {
  while (liveRunning) {
    const video = $("liveVideo");
    if (video.videoWidth) {
      try {
        const fd = new FormData();
        fd.append("image", await grabFrame(video), "frame.jpg");
        fd.append("log_attendance", $("liveLog").checked);
        fd.append("stream", true);   // smooth liveness/emotion over frames (AdaptiveSecurityFusion)
        const r = await api("/api/recognize", { method: "POST", body: fd });
        if (liveRunning) { drawOverlay(r); listResults(r); }
      } catch (e) { $("liveLatency").textContent = "Error: " + e.message; }
    }
    await new Promise((res) => setTimeout(res, FRAME_GAP_MS));
  }
}

const EMOTION_ICONS = {
  happy: "😊",
  sad: "😢",
  angry: "😠",
  surprise: "😲",
  fear: "😨",
  disgust: "🤢",
  neutral: "😐",
  confused: "🤔",
  shy: "😳",
  uncertain: "😶"
};
const emotionIcon = (name) => EMOTION_ICONS[String(name || "").toLowerCase()] || "🙂";

const isSpoof = (f) => f.liveness && !f.liveness.passed;
const faceColor = (f) => (isSpoof(f) ? "#f59e0b" : f.access_granted ? "#22c55e" : "#ef4444");
function faceLabel(f) {
  if (isSpoof(f)) return "SPOOF?";
  const id = f.identity || {};
  const sim = id.similarity == null ? "" : ` ${id.similarity.toFixed(2)}`;
  const emo = f.emotion ? ` · ${emotionIcon(f.emotion.dominant)} ${f.emotion.dominant}` : "";
  return `${id.name || "Unknown"}${sim}${emo}`;
}
function drawOverlay(r) {
  const cv = $("liveOverlay");
  const [W, H] = r.image_size;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  ctx.lineWidth = Math.max(2, W / 240);
  ctx.font = `${Math.max(14, W / 40)}px system-ui, sans-serif`;
  for (const f of r.faces) {
    const [x, y, w, h] = f.box;
    const mx = W - x - w;                // the video is mirrored on screen
    const color = faceColor(f), label = faceLabel(f);
    ctx.strokeStyle = color; ctx.strokeRect(mx, y, w, h);
    const tw = ctx.measureText(label).width + 10, th = parseInt(ctx.font) + 8;
    ctx.fillStyle = color; ctx.fillRect(mx, Math.max(0, y - th), tw, th);
    ctx.fillStyle = "#000"; ctx.fillText(label, mx + 5, Math.max(th - 6, y - 6));
  }
}
function listResults(r) {
  $("liveLatency").textContent = `${r.faces.length} face(s) · ${r.latency_ms} ms · ${r.model} · threshold ${r.threshold.toFixed(2)}`;
  const ul = $("liveResults");
  if (!r.faces.length) { ul.innerHTML = '<li class="muted">No face detected</li>'; return; }
  ul.innerHTML = r.faces.map((f) => {
    const id = f.identity || {};
    const tags = [];
    if (f.liveness) tags.push(`liveness ${f.liveness.score.toFixed(2)}${f.liveness.smoothed ? ` (avg ${f.liveness.frames}f)` : ""}`);
    if (f.emotion) {
      const icon = emotionIcon(f.emotion.dominant);
      const smoothedTag = f.emotion.smoothed ? " (smoothed)" : "";
      tags.push(`${icon} ${f.emotion.dominant} ${(f.emotion.confidence * 100).toFixed(0)}%${smoothedTag}`);
    }
    if (id.closest && !id.matched) tags.push(`closest: ${id.closest}`);
    if (f.checked_in) tags.push("✔ checked in");
    const cls = isSpoof(f) ? "warn" : f.access_granted ? "ok" : "bad";
    return `<li class="${cls}"><b>${esc(faceLabel(f))}</b><br>${tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</li>`;
  }).join("");
}

// ------------------------------------------------------------------ enroll
let enrollStream = null;
const enrollBlobs = [];
function renderThumbs() {
  $("enrollThumbs").innerHTML = "";
  enrollBlobs.forEach((b) => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(b);
    $("enrollThumbs").appendChild(img);
  });
  $("enrollSnap").textContent = `Capture (${enrollBlobs.length})`;
}
$("enrollCam").addEventListener("click", async () => {
  if (enrollStream) {
    closeCamera($("enrollVideo"), enrollStream, $("enrollPlaceholder"));
    enrollStream = null; $("enrollCam").textContent = "Start camera"; $("enrollSnap").disabled = true;
    return;
  }
  try {
    enrollStream = await openCamera($("enrollVideo"), $("enrollPlaceholder"));
    $("enrollCam").textContent = "Stop camera"; $("enrollSnap").disabled = false;
  } catch (e) { alert("Cannot open camera: " + e.message); }
});
$("enrollSnap").addEventListener("click", async () => { enrollBlobs.push(await grabFrame($("enrollVideo"))); renderThumbs(); });
$("enrollFiles").addEventListener("change", (e) => { enrollBlobs.push(...e.target.files); e.target.value = ""; renderThumbs(); });
$("enrollClear").addEventListener("click", () => { enrollBlobs.length = 0; renderThumbs(); $("enrollMsg").textContent = ""; });
$("enrollSubmit").addEventListener("click", async () => {
  const msg = $("enrollMsg");
  const name = $("enrollName").value.trim();
  if (!name) { msg.className = "msg bad"; msg.textContent = "Enter a name first."; return; }
  if (!enrollBlobs.length) { msg.className = "msg bad"; msg.textContent = "Capture or upload at least one photo."; return; }
  const fd = new FormData();
  fd.append("name", name);
  if ($("enrollId").value.trim()) fd.append("person_id", $("enrollId").value.trim());
  enrollBlobs.forEach((b, i) => fd.append("images", b, b.name || `capture_${i}.jpg`));
  msg.className = "msg"; msg.textContent = "Enrolling…";
  try {
    const r = await api("/api/enroll", { method: "POST", body: fd });
    msg.className = "msg ok";
    msg.textContent = `Enrolled ${r.name} (${r.id}) with ${r.samples} photo(s)` + (r.skipped_images ? `; ${r.skipped_images} photo(s) had no face.` : ".");
    enrollBlobs.length = 0; renderThumbs(); loadStatus();
  } catch (e) { msg.className = "msg bad"; msg.textContent = e.message; }
});

// ------------------------------------------------------------------ verify
for (const k of ["A", "B"]) {
  $("verify" + k).addEventListener("change", (e) => {
    const f = e.target.files[0];
    $("verify" + k + "Img").src = f ? URL.createObjectURL(f) : "";
  });
}
$("verifyBtn").addEventListener("click", async () => {
  const a = $("verifyA").files[0], b = $("verifyB").files[0], msg = $("verifyMsg");
  if (!a || !b) { msg.className = "msg center bad"; msg.textContent = "Choose two images."; return; }
  const fd = new FormData();
  fd.append("image_a", a); fd.append("image_b", b);
  msg.className = "msg center"; msg.textContent = "Comparing…";
  try {
    const r = await api("/api/verify", { method: "POST", body: fd });
    msg.className = "msg center " + (r.match ? "ok" : "bad");
    msg.textContent = `${r.match ? "Same person" : "Different people"} — cosine ${r.similarity.toFixed(3)} (threshold ${r.threshold.toFixed(3)}, ${r.model})`;
  } catch (e) { msg.className = "msg center bad"; msg.textContent = e.message; }
});

// ------------------------------------------------------------------ people
async function loadPeople() {
  const box = $("peopleList");
  try {
    const people = await api("/api/people");
    if (!people.length) { box.innerHTML = '<p class="muted">No one enrolled yet.</p>'; return; }
    box.innerHTML = people.map((p) => `
      <div class="card">
        ${p.photo ? `<img src="/api/people/photo/${encodeURIComponent(p.photo)}" alt="">` : ""}
        <b>${esc(p.name)}</b>
        <small class="muted">${esc(p.id)} · ${p.samples} photo(s)</small>
        <button class="danger" data-del="${esc(p.id)}">Delete</button>
      </div>`).join("");
    box.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm(`Delete ${b.dataset.del}?`)) return;
      await api("/api/people/" + encodeURIComponent(b.dataset.del), { method: "DELETE" });
      loadPeople(); loadStatus();
    }));
  } catch (e) { box.innerHTML = `<p class="msg bad">${esc(e.message)}</p>`; }
}
$("peopleRefresh").addEventListener("click", loadPeople);

// ------------------------------------------------------------------ attendance log
function today() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
$("logDate").value = today();
async function loadLog() {
  const body = $("logBody");
  try {
    const rows = await api("/api/attendance?day=" + encodeURIComponent($("logDate").value || today()));
    body.innerHTML = rows.length ? rows.map((r) => `<tr><td>${esc(r.timestamp.replace("T", " "))}</td><td>${esc(r.name)}</td><td>${esc(r.person_id)}</td><td>${esc(r.model)}</td><td>${esc(r.similarity)}</td><td>${esc(r.liveness_score || "–")}</td><td>${r.emotion ? `${emotionIcon(r.emotion)} ${esc(r.emotion)}` : "–"}</td></tr>`).join("")
      : '<tr><td colspan="7" class="muted">No check-ins for this day.</td></tr>';
  } catch (e) { body.innerHTML = `<tr><td colspan="7" class="msg bad">${esc(e.message)}</td></tr>`; }
}
$("logRefresh").addEventListener("click", loadLog);

// ------------------------------------------------------------------ models
async function loadModels() {
  const models = await api("/api/models");
  $("modelsBody").innerHTML = models.map((m) => `<tr>
    <td>${esc(m.label)}</td><td>${esc(m.owner)}</td><td><code>checkpoints/${esc(m.checkpoint)}</code></td>
    <td>${m.val_auc ? m.val_auc.toFixed(4) : "–"}</td>
    <td>${m.threshold != null ? m.threshold.toFixed(4) : "0.30 (not calibrated)"}</td>
    <td>${m.active ? "<b>active</b>" : m.available ? "available" : "no checkpoint"}</td></tr>`).join("");
}

loadStatus();
loadModelSelect().catch(() => {});
