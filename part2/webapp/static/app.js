/* app.js — 页面交互：上传/粘贴图片、画板绘制、调用 /api/predict */

const $ = (id) => document.getElementById(id);

const els = {
  device: $("device"),
  tabs: document.querySelectorAll(".tab"),
  panels: { upload: $("panel-upload"), draw: $("panel-draw") },
  drop: $("drop"),
  file: $("file"),
  tryBtn: $("try"),
  source: $("source"),
  sourceImg: $("source-img"),
  sourceMeta: $("source-meta"),
  canvas: $("canvas"),
  hint: $("hint"),
  width: $("width"),
  clear: $("clear"),
  recognize: $("recognize"),
  digit: $("digit"),
  confidence: $("confidence"),
  meter: $("meter"),
  preview: $("preview"),
  previewHint: $("preview-hint"),
  bars: $("bars"),
  status: $("status"),
};

let busy = false;
let barsBuilt = false;

function setStatus(text, kind) {
  els.status.textContent = text;
  els.status.className = "status" + (kind ? " is-" + kind : "");
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const where = data.gpu ? data.gpu : "CPU";
    els.device.innerHTML = `推理设备 <b>${data.device === "cuda" ? "GPU" : "CPU"}</b> · ${where}`;
  } catch (err) {
    els.device.textContent = "服务未连接";
  }
}

/* 顶部标签切换 */
els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    els.tabs.forEach((t) => {
      const on = t === tab;
      t.classList.toggle("is-on", on);
      t.setAttribute("aria-selected", String(on));
    });
    Object.entries(els.panels).forEach(([key, panel]) => {
      panel.classList.toggle("is-on", key === tab.dataset.panel);
    });
    if (tab.dataset.panel === "draw") setupCanvas();
  });
});

function buildBars() {
  if (barsBuilt) return;
  els.bars.innerHTML = "";
  for (let d = 0; d < 10; d += 1) {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="k">${d}</span>` +
      `<div class="track"><i></i></div>` +
      `<span class="v">—</span>`;
    els.bars.appendChild(li);
  }
  barsBuilt = true;
}

/* 结果渲染 */
function renderResult(data) {
  els.digit.textContent = String(data.digit);
  els.digit.classList.remove("is-empty");

  const pct = data.confidence * 100;
  els.confidence.textContent = pct.toFixed(2) + "%";
  els.meter.style.width = Math.max(2, pct).toFixed(1) + "%";

  els.preview.src = data.preview;
  els.preview.hidden = false;
  els.previewHint.hidden = true;

  buildBars();
  [...els.bars.children].forEach((li, d) => {
    const p = data.probs[d];
    li.classList.toggle("is-top", d === data.digit);
    li.querySelector("i").style.width = (p * 100).toFixed(2) + "%";
    li.querySelector(".v").textContent = (p * 100).toFixed(1) + "%";
  });

  const notes = [`识别用时 ${data.elapsed_ms} ms`];
  if (data.inverted) notes.push("已自动反色");
  setStatus("识别完成 · " + notes.join(" · "));
}

function clearResult(message) {
  els.digit.textContent = "?";
  els.digit.classList.add("is-empty");
  els.confidence.textContent = "–";
  els.meter.style.width = "0%";
  if (barsBuilt) {
    [...els.bars.children].forEach((li) => {
      li.classList.remove("is-top");
      li.querySelector("i").style.width = "0%";
      li.querySelector(".v").textContent = "—";
    });
  }
  if (message) setStatus(message);
}

/* 调用后端 */
async function predict(formData) {
  if (busy) return;
  busy = true;
  els.recognize.disabled = true;
  setStatus("识别中…", "busy");

  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || !data.ok) {
      clearResult();
      setStatus(data.error || `请求失败（HTTP ${res.status}）`, "error");
      return;
    }
    renderResult(data);
  } catch (err) {
    clearResult();
    setStatus("请求出错：" + err.message, "error");
  } finally {
    busy = false;
    els.recognize.disabled = false;
  }
}

function predictFile(file) {
  const form = new FormData();
  form.append("file", file, file.name || "upload.png");
  predict(form);
}

function predictCanvas() {
  const form = new FormData();
  form.append("image", els.canvas.toDataURL("image/png"));
  predict(form);
}

/* 上传：点击、拖拽、粘贴 */
els.file.addEventListener("change", () => {
  const file = els.file.files && els.file.files[0];
  if (!file) return;
  showSource(file);
  predictFile(file);
  els.file.value = "";
});

function showSource(file) {
  const url = URL.createObjectURL(file);
  els.sourceImg.src = url;
  els.sourceImg.onload = () => URL.revokeObjectURL(url);
  els.source.hidden = false;
  els.sourceMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
}

["dragenter", "dragover"].forEach((type) => {
  els.drop.addEventListener(type, (e) => {
    e.preventDefault();
    els.drop.classList.add("is-over");
  });
});

["dragleave", "drop"].forEach((type) => {
  els.drop.addEventListener(type, (e) => {
    e.preventDefault();
    els.drop.classList.remove("is-over");
  });
});

els.drop.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!file || !file.type.startsWith("image/")) {
    setStatus("请拖入一张图片文件", "error");
    return;
  }
  showSource(file);
  predictFile(file);
});

els.tryBtn.addEventListener("click", async () => {
  try {
    const res = await fetch("/api/example");
    const data = await res.json();
    if (!data.ok) {
      setStatus(data.error || "取样本失败", "error");
      return;
    }
    els.sourceImg.src = data.image;
    els.source.hidden = false;
    els.sourceMeta.textContent = `MNIST 测试集样本 #${data.index} · 真实标签 ${data.label}`;

    const form = new FormData();
    form.append("image", data.image);
    predict(form);
  } catch (err) {
    setStatus("取样本失败：" + err.message, "error");
  }
});

window.addEventListener("paste", (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) {
        e.preventDefault();
        showSource(file);
        predictFile(file);
      }
      return;
    }
  }
});

/* 画板 */
const ctx = els.canvas.getContext("2d");
const dpr = window.devicePixelRatio || 1;
let SIZE = 0;

/* 画板宽度跟着布局走：只有真正可见、且尺寸变了才重建位图，
   否则切到"上传图片"时量不到宽度，会留下一张被拉伸的旧位图 */
function setupCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  if (rect.width < 50) return;
  const size = Math.round(rect.width);
  if (size === SIZE) return;
  SIZE = size;
  els.canvas.width = SIZE * dpr;
  els.canvas.height = SIZE * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#111827";
  paintBackground();
  dirty = false;
  els.hint.classList.remove("is-hidden");
}

function paintBackground() {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.restore();
}

let drawing = false;
let dirty = false;
let last = null;
let idleTimer = null;

function pointOf(e) {
  const rect = els.canvas.getBoundingClientRect();
  const scale = SIZE / rect.width;
  return {
    x: (e.clientX - rect.left) * scale,
    y: (e.clientY - rect.top) * scale,
  };
}

function strokeWidth() {
  return Number(els.width.value);
}

function beginStroke(e) {
  if (e.button !== undefined && e.button !== 0 && e.pointerType === "mouse") return;
  e.preventDefault();
  clearTimeout(idleTimer);
  try {
    els.canvas.setPointerCapture(e.pointerId);
  } catch (err) {
    // 少数浏览器/合成事件没有可捕获的指针，忽略即可，绘制不受影响
  }
  drawing = true;
  dirty = true;
  last = pointOf(e);
  els.hint.classList.add("is-hidden");

  const p = last;
  ctx.beginPath();
  ctx.fillStyle = ctx.strokeStyle;
  ctx.arc(p.x, p.y, strokeWidth() / 2, 0, Math.PI * 2);
  ctx.fill();
}

function moveStroke(e) {
  if (!drawing) return;
  e.preventDefault();
  const p = pointOf(e);
  if (last) {
    ctx.lineWidth = strokeWidth();
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }
  last = p;
}

function endStroke(e) {
  if (!drawing) return;
  drawing = false;
  last = null;
  if (e.pointerId !== undefined && els.canvas.hasPointerCapture(e.pointerId)) {
    els.canvas.releasePointerCapture(e.pointerId);
  }
  // 抬笔后停顿一下自动识别，不必每次都点按钮
  clearTimeout(idleTimer);
  if (dirty) {
    idleTimer = setTimeout(() => {
      if (!busy) predictCanvas();
    }, 260);
  }
}

els.canvas.addEventListener("pointerdown", beginStroke);
els.canvas.addEventListener("pointermove", moveStroke);
els.canvas.addEventListener("pointerup", endStroke);
els.canvas.addEventListener("pointercancel", endStroke);
els.canvas.addEventListener("pointerleave", (e) => {
  if (drawing && e.pointerType === "mouse") endStroke(e);
});

els.clear.addEventListener("click", () => {
  paintBackground();
  dirty = false;
  clearTimeout(idleTimer);
  els.hint.classList.remove("is-hidden");
  clearResult("画板已清空，写一个数字试试");
});

els.recognize.addEventListener("click", () => {
  if (!dirty) {
    setStatus("画板还是空的，先写一个数字", "error");
    return;
  }
  predictCanvas();
});

window.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    if (els.panels.draw.classList.contains("is-on") && dirty) predictCanvas();
  }
});

new ResizeObserver(() => setupCanvas()).observe(els.canvas);

buildBars();
setupCanvas();
loadHealth();
