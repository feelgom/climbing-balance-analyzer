import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const LM = {
  NOSE: 0,
  L_EAR: 7, R_EAR: 8,
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13, R_ELBOW: 14,
  L_WRIST: 15, R_WRIST: 16,
  L_PINKY: 17, R_PINKY: 18,
  L_INDEX: 19, R_INDEX: 20,
  L_THUMB: 21, R_THUMB: 22,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
  L_HEEL: 29, R_HEEL: 30,
  L_FOOT_INDEX: 31, R_FOOT_INDEX: 32,
};

const VIS_THRESH = 0.35;

const mid = (lm, i, j) => ({
  x: (lm[i].x + lm[j].x) / 2,
  y: (lm[i].y + lm[j].y) / 2,
  visibility: Math.min(lm[i].visibility ?? 1, lm[j].visibility ?? 1),
});

const along = (lm, pIdx, dIdx, r) => {
  const p = lm[pIdx], d = lm[dIdx];
  return {
    x: p.x + (d.x - p.x) * r,
    y: p.y + (d.y - p.y) * r,
    visibility: Math.min(p.visibility ?? 1, d.visibility ?? 1),
  };
};

// Body-segment mass fractions and COM position ratios (proximal->distal),
// from Winter/Dempster anthropometric tables. Sums to ~1.0 across all segments.
const SEGMENTS = [
  { key: "head", mass: 0.081, compute: (lm) => {
      const p = mid(lm, LM.L_EAR, LM.R_EAR);
      return { x: p.x, y: p.y, visibility: Math.min(p.visibility, lm[LM.NOSE].visibility ?? 1) };
  }},
  { key: "trunk", mass: 0.497, compute: (lm) => {
      const sh = mid(lm, LM.L_SHOULDER, LM.R_SHOULDER);
      const hip = mid(lm, LM.L_HIP, LM.R_HIP);
      return { x: (sh.x + hip.x) / 2, y: (sh.y + hip.y) / 2, visibility: Math.min(sh.visibility, hip.visibility) };
  }},
  { key: "upperArmL", mass: 0.027, compute: (lm) => along(lm, LM.L_SHOULDER, LM.L_ELBOW, 0.436) },
  { key: "upperArmR", mass: 0.027, compute: (lm) => along(lm, LM.R_SHOULDER, LM.R_ELBOW, 0.436) },
  { key: "forearmL", mass: 0.016, compute: (lm) => along(lm, LM.L_ELBOW, LM.L_WRIST, 0.430) },
  { key: "forearmR", mass: 0.016, compute: (lm) => along(lm, LM.R_ELBOW, LM.R_WRIST, 0.430) },
  { key: "handL", mass: 0.006, compute: (lm) => along(lm, LM.L_WRIST, LM.L_INDEX, 0.5) },
  { key: "handR", mass: 0.006, compute: (lm) => along(lm, LM.R_WRIST, LM.R_INDEX, 0.5) },
  { key: "thighL", mass: 0.100, compute: (lm) => along(lm, LM.L_HIP, LM.L_KNEE, 0.433) },
  { key: "thighR", mass: 0.100, compute: (lm) => along(lm, LM.R_HIP, LM.R_KNEE, 0.433) },
  { key: "shankL", mass: 0.0465, compute: (lm) => along(lm, LM.L_KNEE, LM.L_ANKLE, 0.433) },
  { key: "shankR", mass: 0.0465, compute: (lm) => along(lm, LM.R_KNEE, LM.R_ANKLE, 0.433) },
  { key: "footL", mass: 0.0145, compute: (lm) => along(lm, LM.L_ANKLE, LM.L_FOOT_INDEX, 0.5) },
  { key: "footR", mass: 0.0145, compute: (lm) => along(lm, LM.R_ANKLE, LM.R_FOOT_INDEX, 0.5) },
];

const SKELETON_GROUPS = [
  { color: "#5b9bff", pairs: [[LM.L_SHOULDER, LM.R_SHOULDER], [LM.L_SHOULDER, LM.L_HIP], [LM.R_SHOULDER, LM.R_HIP], [LM.L_HIP, LM.R_HIP]] },
  { color: "#ffb454", pairs: [[LM.L_SHOULDER, LM.L_ELBOW], [LM.L_ELBOW, LM.L_WRIST], [LM.R_SHOULDER, LM.R_ELBOW], [LM.R_ELBOW, LM.R_WRIST]] },
  { color: "#7ee787", pairs: [[LM.L_HIP, LM.L_KNEE], [LM.L_KNEE, LM.L_ANKLE], [LM.R_HIP, LM.R_KNEE], [LM.R_KNEE, LM.R_ANKLE]] },
  { color: "#d2a8ff", pairs: [[LM.L_EAR, LM.NOSE], [LM.R_EAR, LM.NOSE]] },
];

const MODEL_URLS = {
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
  heavy: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
};

// Hysteresis thresholds for "static contact" detection (normalized to shoulder width per frame).
// A wide gap between ENTER/EXIT prevents flicker when a limb is briefly still mid-reach.
const ENTER_STATIC_THRESH = 0.015;
const EXIT_STATIC_THRESH = 0.045;
const ENTER_STATIC_FRAMES = 6;
const HIST_LEN = 5;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const fileInput = document.getElementById("fileInput");
const btnWebcam = document.getElementById("btnWebcam");
const btnStop = document.getElementById("btnStop");
const modelSelect = document.getElementById("modelSelect");
const staticOnlyToggle = document.getElementById("staticOnlyToggle");
const beepToggle = document.getElementById("beepToggle");
const banner = document.getElementById("banner");
const loadingOverlay = document.getElementById("loadingOverlay");
const statusText = document.getElementById("statusText");
const supportModeText = document.getElementById("supportModeText");
const stabilityText = document.getElementById("stabilityText");
const fpsText = document.getElementById("fpsText");

let poseLandmarker = null;
let currentModelKey = null;
let mediaStream = null;
let rafId = null;
let running = false;
let lastFrameTime = performance.now();
let fpsSmoothed = 0;
let lastBeepTime = 0;
let audioCtx = null;

const extremityHistory = { LH: [], RH: [], LF: [], RF: [] };
const contactState = { LH: "unknown", RH: "unknown", LF: "unknown", RF: "unknown" };
const staticStreak = { LH: 0, RH: 0, LF: 0, RF: 0 };

async function ensureLandmarker(modelKey) {
  if (poseLandmarker && currentModelKey === modelKey) return poseLandmarker;
  loadingOverlay.classList.remove("hidden");
  if (poseLandmarker) {
    poseLandmarker.close();
    poseLandmarker = null;
  }
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URLS[modelKey], delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch (err) {
    // GPU delegate unavailable (no WebGL2, disabled GPU, etc.) — fall back to CPU.
    console.warn("GPU delegate init failed, falling back to CPU:", err);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URLS[modelKey], delegate: "CPU" },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  }
  currentModelKey = modelKey;
  loadingOverlay.classList.add("hidden");
  return poseLandmarker;
}

function resetTrackingState() {
  for (const k of Object.keys(extremityHistory)) {
    extremityHistory[k] = [];
    contactState[k] = "unknown";
    staticStreak[k] = 0;
  }
}

async function startWebcam() {
  await stopAll();
  await ensureLandmarker(modelSelect.value);
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = mediaStream;
  await video.play();
  onVideoReady();
}

async function startVideoFile(file) {
  await stopAll();
  await ensureLandmarker(modelSelect.value);
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.loop = true;
  await video.play();
  onVideoReady();
}

function onVideoReady() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  resetTrackingState();
  running = true;
  btnStop.disabled = false;
  statusText.textContent = "분석 중...";
  lastFrameTime = performance.now();
  loop();
}

async function stopAll() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  video.pause();
  video.removeAttribute("src");
  video.srcObject = null;
  btnStop.disabled = true;
  banner.classList.add("hidden");
  statusText.textContent = "대기 중";
  supportModeText.textContent = "-";
  stabilityText.textContent = "-";
  fpsText.textContent = "-";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && !video.paused && !video.ended) {
    const now = performance.now();
    const result = poseLandmarker.detectForVideo(video, now);
    processFrame(result);
    const dt = now - lastFrameTime;
    lastFrameTime = now;
    const fps = 1000 / Math.max(dt, 1);
    fpsSmoothed = fpsSmoothed ? fpsSmoothed * 0.9 + fps * 0.1 : fps;
    fpsText.textContent = fpsSmoothed.toFixed(0);
  }
  rafId = requestAnimationFrame(loop);
}

function toPixel(lm, w, h) {
  return lm.map((p) => ({ x: p.x * w, y: p.y * h, z: p.z, visibility: p.visibility }));
}

function computeCOM(lmPix) {
  let sumW = 0, sx = 0, sy = 0;
  for (const seg of SEGMENTS) {
    const p = seg.compute(lmPix);
    if ((p.visibility ?? 1) < VIS_THRESH) continue;
    sumW += seg.mass;
    sx += p.x * seg.mass;
    sy += p.y * seg.mass;
  }
  if (sumW === 0) return null;
  return { x: sx / sumW, y: sy / sumW };
}

function avgVisiblePoints(lmPix, idxs, visThresh) {
  const pts = idxs.map((i) => lmPix[i]).filter((p) => (p.visibility ?? 1) >= visThresh);
  if (pts.length === 0) {
    const fallback = lmPix[idxs[0]];
    if ((fallback.visibility ?? 1) < VIS_THRESH) return null;
    return { x: fallback.x, y: fallback.y };
  }
  return {
    x: pts.reduce((s, p) => s + p.x, 0) / pts.length,
    y: pts.reduce((s, p) => s + p.y, 0) / pts.length,
  };
}

function getExtremities(lmPix) {
  return {
    LH: avgVisiblePoints(lmPix, [LM.L_WRIST, LM.L_INDEX, LM.L_THUMB], VIS_THRESH),
    RH: avgVisiblePoints(lmPix, [LM.R_WRIST, LM.R_INDEX, LM.R_THUMB], VIS_THRESH),
    LF: avgVisiblePoints(lmPix, [LM.L_ANKLE, LM.L_HEEL, LM.L_FOOT_INDEX], VIS_THRESH),
    RF: avgVisiblePoints(lmPix, [LM.R_ANKLE, LM.R_HEEL, LM.R_FOOT_INDEX], VIS_THRESH),
  };
}

function updateContactStates(extremities, bodyScale) {
  for (const key of Object.keys(extremityHistory)) {
    const pt = extremities[key];
    if (!pt || !bodyScale) continue;
    const hist = extremityHistory[key];
    hist.push(pt);
    if (hist.length > HIST_LEN) hist.shift();
    if (hist.length < 2) continue;
    const prev = hist[hist.length - 2];
    const dist = Math.hypot(pt.x - prev.x, pt.y - prev.y);
    const speed = dist / bodyScale;
    if (contactState[key] === "static") {
      if (speed > EXIT_STATIC_THRESH) {
        contactState[key] = "moving";
        staticStreak[key] = 0;
      }
    } else {
      if (speed < ENTER_STATIC_THRESH) {
        staticStreak[key]++;
        if (staticStreak[key] >= ENTER_STATIC_FRAMES) contactState[key] = "static";
      } else {
        staticStreak[key] = 0;
        contactState[key] = "moving";
      }
    }
  }
}

function convexHull(points) {
  if (points.length <= 2) return points;
  const pts = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function getSupportBase(extremities) {
  const useStaticOnly = staticOnlyToggle.checked;
  if (useStaticOnly) {
    const staticPts = Object.entries(contactState)
      .filter(([k, v]) => v === "static" && extremities[k])
      .map(([k]) => ({ key: k, ...extremities[k] }));
    if (staticPts.length >= 2) return { points: staticPts, mode: "confident" };
  }
  const allPts = Object.entries(extremities)
    .filter(([, v]) => v)
    .map(([k, v]) => ({ key: k, ...v }));
  if (allPts.length >= 2) return { points: allPts, mode: "fallback" };
  return { points: [], mode: "none" };
}

function evaluateBalance(comPix, supportBase, bodyScale) {
  const hull = convexHull(supportBase.points);
  const xs = hull.map((p) => p.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const balanced = comPix.x >= minX && comPix.x <= maxX;
  let marginRatio;
  let side = null;
  if (balanced) {
    marginRatio = Math.min(comPix.x - minX, maxX - comPix.x) / bodyScale;
  } else {
    if (comPix.x < minX) { marginRatio = (comPix.x - minX) / bodyScale; side = "left"; }
    else { marginRatio = (comPix.x - maxX) / bodyScale; side = "right"; }
  }
  return { hull, balanced, marginRatio, side };
}

function processFrame(result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (!result.landmarks || result.landmarks.length === 0) {
    statusText.textContent = "사람이 감지되지 않았습니다";
    supportModeText.textContent = "-";
    stabilityText.textContent = "-";
    banner.classList.add("hidden");
    return;
  }

  const lmPix = toPixel(result.landmarks[0], canvas.width, canvas.height);
  const shoulderVisible = (lmPix[LM.L_SHOULDER].visibility ?? 1) >= VIS_THRESH && (lmPix[LM.R_SHOULDER].visibility ?? 1) >= VIS_THRESH;
  const hipVisible = (lmPix[LM.L_HIP].visibility ?? 1) >= VIS_THRESH && (lmPix[LM.R_HIP].visibility ?? 1) >= VIS_THRESH;
  const bodyScale = shoulderVisible
    ? Math.hypot(lmPix[LM.L_SHOULDER].x - lmPix[LM.R_SHOULDER].x, lmPix[LM.L_SHOULDER].y - lmPix[LM.R_SHOULDER].y)
    : hipVisible
    ? Math.hypot(lmPix[LM.L_HIP].x - lmPix[LM.R_HIP].x, lmPix[LM.L_HIP].y - lmPix[LM.R_HIP].y)
    : canvas.width * 0.15;

  const extremities = getExtremities(lmPix);
  updateContactStates(extremities, bodyScale);
  const com = computeCOM(lmPix);
  const supportBase = getSupportBase(extremities);

  drawSkeleton(lmPix);
  drawExtremities(extremities);

  if (!com) {
    statusText.textContent = "무게중심 계산 불가 (신체 노출 부족)";
    supportModeText.textContent = "-";
    stabilityText.textContent = "-";
    banner.classList.add("hidden");
    return;
  }

  if (supportBase.points.length < 2) {
    statusText.textContent = "지지점 부족 (손/발 인식 안 됨)";
    supportModeText.textContent = "불가";
    stabilityText.textContent = "-";
    banner.classList.add("hidden");
    drawCOM(com, null);
    return;
  }

  const evalResult = evaluateBalance(com, supportBase, bodyScale);
  drawSupportPolygon(evalResult.hull, supportBase.mode);
  drawCOM(com, evalResult.balanced);

  supportModeText.textContent = supportBase.mode === "confident"
    ? `확실 (정지 사지 ${supportBase.points.length}개)`
    : `추정 (전체 사지 ${supportBase.points.length}개)`;

  if (evalResult.balanced) {
    statusText.textContent = "균형 유지";
    const stabilityPct = Math.max(0, Math.min(100, evalResult.marginRatio * 150));
    stabilityText.textContent = `안정 ${stabilityPct.toFixed(0)}%`;
    banner.classList.add("hidden");
  } else {
    const sideLabel = evalResult.side === "left" ? "왼쪽" : "오른쪽";
    statusText.textContent = `균형 이탈 (${sideLabel}으로 치우침)`;
    stabilityText.textContent = `어깨너비의 ${Math.abs(evalResult.marginRatio * 100).toFixed(0)}%만큼 이탈`;
    banner.textContent = `⚠ 무게중심 이탈: ${sideLabel}으로 치우침`;
    banner.classList.remove("hidden");
    banner.classList.remove("balanced");
    maybeBeep();
  }
}

function drawSkeleton(lmPix) {
  for (const group of SKELETON_GROUPS) {
    ctx.strokeStyle = group.color;
    ctx.lineWidth = 4;
    for (const [a, b] of group.pairs) {
      const pa = lmPix[a], pb = lmPix[b];
      if ((pa.visibility ?? 1) < VIS_THRESH || (pb.visibility ?? 1) < VIS_THRESH) continue;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }
  }
}

const EXTREMITY_LABELS = { LH: "L손", RH: "R손", LF: "L발", RF: "R발" };

function drawExtremities(extremities) {
  for (const key of Object.keys(extremities)) {
    const pt = extremities[key];
    if (!pt) continue;
    const isStatic = contactState[key] === "static";
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = isStatic ? "#2f6feb" : "rgba(200,200,200,0.35)";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "12px sans-serif";
    ctx.fillText(EXTREMITY_LABELS[key], pt.x + 10, pt.y - 6);
  }
}

function drawSupportPolygon(hull, mode) {
  if (hull.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(hull[0].x, hull[0].y);
  for (let i = 1; i < hull.length; i++) ctx.lineTo(hull[i].x, hull[i].y);
  if (hull.length > 2) ctx.closePath();
  ctx.strokeStyle = mode === "confident" ? "#2f6feb" : "#c98a2b";
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.stroke();
  if (hull.length > 2) {
    ctx.fillStyle = mode === "confident" ? "rgba(47,111,235,0.15)" : "rgba(201,138,43,0.15)";
    ctx.fill();
  }
}

function drawCOM(com, balanced) {
  const color = balanced === null ? "#c98a2b" : balanced ? "#3fb950" : "#e5534b";
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(com.x, 0);
  ctx.lineTo(com.x, canvas.height);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  ctx.arc(com.x, com.y, 10, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(com.x - 14, com.y);
  ctx.lineTo(com.x + 14, com.y);
  ctx.moveTo(com.x, com.y - 14);
  ctx.lineTo(com.x, com.y + 14);
  ctx.stroke();
}

function maybeBeep() {
  if (!beepToggle.checked) return;
  const now = performance.now();
  if (now - lastBeepTime < 1500) return;
  lastBeepTime = now;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.frequency.value = 880;
  gain.gain.value = 0.15;
  osc.connect(gain).connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + 0.18);
}

btnWebcam.addEventListener("click", () => {
  startWebcam().catch((err) => {
    console.error(err);
    alert("웹캠을 시작할 수 없습니다: " + err.message);
  });
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  startVideoFile(file).catch((err) => {
    console.error(err);
    alert("영상 파일을 열 수 없습니다: " + err.message);
  });
});

btnStop.addEventListener("click", () => {
  stopAll();
});

modelSelect.addEventListener("change", async () => {
  if (!running) return;
  const wasVideo = !!video.src;
  const wasWebcam = !!mediaStream;
  await ensureLandmarker(modelSelect.value);
  if (!wasVideo && !wasWebcam) return;
});
