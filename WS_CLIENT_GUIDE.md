# WebSocket Client Quickstart (EchoPath)

1. Connect to `/ws`
2. Capture webcam frames
3. Send each frame as base64 JPEG JSON
4. Receive detections + distance points
5. Draw boxes and labeled distance points on a canvas overlay

---

## 1) Minimal HTML structure

```html
<div class="stage">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="overlay"></canvas>
</div>

<button id="startBtn">Start</button>
<button id="stopBtn" disabled>Stop</button>
<input id="fpsInput" type="number" min="1" max="30" value="5" />
<div id="stats">Detections: 0</div>
<div id="status">Idle</div>
```

The `video` element previews the camera, and `overlay` is used for drawing boxes + points.

---

## 2) Core JavaScript state and setup

```javascript
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const fpsInput = document.getElementById("fpsInput");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let ws = null;
let stream = null;
let timer = null;
let latestDetections = [];
let latestDistPoints = [];
```

---

## 3) Connect to the WebSocket

```javascript
function setStatus(text) {
  statusEl.textContent = text;
}

async function start() {
  // Camera first
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  // Overlay size = rendered video box size
  const rect = video.getBoundingClientRect();
  overlay.width = Math.max(1, Math.floor(rect.width));
  overlay.height = Math.max(1, Math.floor(rect.height));

  // Connect /ws
  const wsProto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${wsProto}://${location.host}/ws`);

  ws.onopen = () => {
    setStatus("Connected");
    startBtn.disabled = true;
    stopBtn.disabled = false;
    startLoop();
  };

  ws.onerror = () => setStatus("WebSocket error");
  ws.onclose = () => stop();

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "detections" && Array.isArray(msg.data)) {
      latestDetections = msg.data;
      latestDistPoints = Array.isArray(msg.dist_points) ? msg.dist_points : [];
      const pointCount = getDistPointsToRender().length;
      statsEl.textContent = `Detections: ${latestDetections.length} | Dist points: ${pointCount}`;
      drawResults();
    } else if (msg.type === "error") {
      setStatus(`Server error: ${msg.error}`);
    }
  };
}
```

---

## 4) Send images as base64 JPEG payloads

Server expects:

```json
{ "type": "image", "data": "<base64-jpeg-without-data-url-prefix>" }
```

Client loop:

```javascript
function startLoop() {
  const fps = Math.max(1, Math.min(30, Number(fpsInput.value) || 5));
  const interval = Math.floor(1000 / fps);

  if (timer) clearInterval(timer);
  timer = setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth || !video.videoHeight) return;

    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);
    const base64 = dataUrl.split(",")[1];

    ws.send(JSON.stringify({ type: "image", data: base64 }));
  }, interval);
}
```

---

## 5) Render detections and distance points

Expected detection object includes normalized bbox fields and optional `dist_point`:

```json
{
  "xmin": 0.12,
  "ymin": 0.21,
  "xmax": 0.32,
  "ymax": 0.54,
  "label": "person",
  "confidence": 0.93,
  "dist_point": {
    "label": "person",
    "confidence": 0.93,
    "distance": 0.41,
    "x": 0.63,
    "y": 0.54
  }
}
```

Drawing code:

```javascript
function getDistPointsToRender() {
  const pointsFromDetections = latestDetections
    .map((d) => d?.dist_point)
    .filter(
      (p) =>
        p &&
        Number.isFinite(p.x) &&
        Number.isFinite(p.y) &&
        Number.isFinite(p.distance),
    );

  if (pointsFromDetections.length > 0) return pointsFromDetections;

  return latestDistPoints.filter(
    (p) =>
      p &&
      Number.isFinite(p.x) &&
      Number.isFinite(p.y) &&
      Number.isFinite(p.distance),
  );
}

function drawResults() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.lineWidth = 2;
  ctx.font = "14px system-ui, sans-serif";

  // 1) Draw normalized bboxes
  for (const d of latestDetections) {
    const x = d.xmin * overlay.width;
    const y = d.ymin * overlay.height;
    const w = (d.xmax - d.xmin) * overlay.width;
    const h = (d.ymax - d.ymin) * overlay.height;

    ctx.strokeStyle = "#22c55e";
    ctx.strokeRect(x, y, w, h);

    const label = d.label ?? "obj";
    const conf = typeof d.confidence === "number" ? d.confidence.toFixed(2) : "";
    const text = conf ? `${label} ${conf}` : String(label);

    const pad = 4;
    const tw = ctx.measureText(text).width;
    const th = 16;
    ctx.fillStyle = "#22c55e";
    ctx.fillRect(x, Math.max(0, y - th), tw + pad * 2, th);
    ctx.fillStyle = "#111";
    ctx.fillText(text, x + pad, Math.max(12, y - 4));
  }

  // 2) Draw distance points + (distance, class)
  for (const p of getDistPointsToRender()) {
    const nx = Math.max(0, Math.min(1, Number(p.x)));
    const ny = Math.max(0, Math.min(1, Number(p.y)));
    const px = nx * overlay.width;
    const py = ny * overlay.height;

    ctx.fillStyle = "#f97316";
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();

    const distanceText = Number(p.distance).toFixed(2);
    const cls = p.label ?? "obj";
    const pointText = `(${distanceText}, ${cls})`;

    const pad = 4;
    const tw = ctx.measureText(pointText).width;
    const th = 16;
    const tx = Math.max(0, Math.min(px + 8, overlay.width - (tw + pad * 2)));
    const ty = Math.max(th, py - 8);

    ctx.fillStyle = "#f97316";
    ctx.fillRect(tx, ty - th, tw + pad * 2, th);
    ctx.fillStyle = "#111";
    ctx.fillText(pointText, tx + pad, ty - 4);
  }
}
```

---

## 6) Stop / cleanup

```javascript
function stop() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }

  if (ws) {
    const s = ws;
    ws = null;
    if (s.readyState === WebSocket.OPEN || s.readyState === WebSocket.CONNECTING) {
      s.close();
    }
  }

  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }

  latestDetections = [];
  latestDistPoints = [];
  statsEl.textContent = "Detections: 0 | Dist points: 0";
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Idle");
}
```

---
