"use strict";

const $ = (sel) => document.querySelector(sel);
const els = {
  status: $("#status"),
  auth: $("#auth"),
  apiKey: $("#api-key"),
  saveKey: $("#save-key"),
  conversation: $("#conversation"),
  question: $("#question"),
  ask: $("#ask"),
  mic: $("#mic"),
  recStatus: $("#rec-status"),
  speak: $("#speak"),
  newChat: $("#new-chat"),
  template: $("#turn-template"),
};

const MAX_ROWS_RENDERED = 200;
const MAX_RECORDING_MS = 60_000;
const state = { meta: null, history: [], busy: false, recorder: null, chunks: [], recTimer: null };

// ---------- storage (per-viewer conveniences only) ----------
const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } },
};

function headers(extra = {}) {
  const h = { ...extra };
  const key = store.get("s2s_api_key");
  if (key) h["X-API-Key"] = key;
  return h;
}

async function errorMessage(res) {
  try {
    const body = await res.json();
    return body?.error?.message || body?.detail || `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

// ---------- startup ----------
async function init() {
  try {
    const res = await fetch("v1/meta");
    state.meta = await res.json();
  } catch {
    state.meta = { auth_required: false, voice_enabled: false, max_history_turns: 5 };
  }
  els.auth.hidden = !state.meta.auth_required || Boolean(store.get("s2s_api_key"));
  if (!state.meta.voice_enabled) {
    els.mic.disabled = true;
    els.mic.title = "Voice input is disabled on the server (no speech-to-text key configured)";
    els.recStatus.textContent = "Voice input disabled on the server; type your question.";
  }
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    els.mic.disabled = true;
    els.recStatus.textContent = "This browser cannot record audio; type your question.";
  }
  els.speak.checked = store.get("s2s_speak") === "1";
  checkReady();
}

async function checkReady() {
  try {
    const res = await fetch("readyz");
    const body = await res.json();
    els.status.textContent = body.ready ? "ready" : "not ready";
    els.status.className = `pill ${body.ready ? "ok" : "bad"}`;
    els.status.title = Object.entries(body.checks || {}).map(([k, v]) => `${k}: ${v}`).join("\n");
  } catch {
    els.status.textContent = "offline";
    els.status.className = "pill bad";
  }
}

// ---------- recording ----------
function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return candidates.find((t) => MediaRecorder.isTypeSupported?.(t)) || "";
}

async function startRecording() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  } catch (err) {
    els.recStatus.textContent = `Microphone unavailable: ${err.message}`;
    return;
  }
  const mimeType = pickMimeType();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  state.recorder = recorder;
  state.chunks = [];
  recorder.ondataavailable = (e) => { if (e.data.size) state.chunks.push(e.data); };
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    clearInterval(state.recTimer);
    els.mic.classList.remove("recording");
    els.mic.setAttribute("aria-label", "Start recording");
    const blob = new Blob(state.chunks, { type: recorder.mimeType || "audio/webm" });
    state.recorder = null;
    if (blob.size < 1000) {
      els.recStatus.textContent = "Recording was too short.";
      return;
    }
    await transcribeAndAsk(blob);
  };
  recorder.start();
  els.mic.classList.add("recording");
  els.mic.setAttribute("aria-label", "Stop recording");
  const started = Date.now();
  els.recStatus.textContent = "Listening… click the mic again to stop.";
  state.recTimer = setInterval(() => {
    const elapsed = Date.now() - started;
    els.recStatus.textContent = `Listening… ${Math.floor(elapsed / 1000)}s (click to stop)`;
    if (elapsed > MAX_RECORDING_MS) recorder.stop();
  }, 250);
}

async function transcribeAndAsk(blob) {
  setBusy(true);
  els.recStatus.textContent = "Transcribing…";
  const ext = blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm";
  const form = new FormData();
  form.append("audio", blob, `speech.${ext}`);
  try {
    const res = await fetch("v1/transcribe", { method: "POST", headers: headers(), body: form });
    if (!res.ok) throw new Error(await errorMessage(res));
    const transcript = await res.json();
    els.question.value = transcript.text;
    const conf = transcript.confidence != null ? ` (confidence ${(transcript.confidence * 100).toFixed(0)}%)` : "";
    els.recStatus.textContent = `Heard: “${transcript.text}”${conf}. Edit and press Ask to retry.`;
    setBusy(false);
    await ask(transcript.text, true);
  } catch (err) {
    els.recStatus.textContent = `Transcription failed: ${err.message}`;
    setBusy(false);
  }
}

// ---------- asking ----------
function setBusy(busy) {
  state.busy = busy;
  els.ask.disabled = busy;
  if (state.meta?.voice_enabled) els.mic.disabled = busy && !state.recorder;
}

async function ask(question, fromSpeech) {
  question = question.trim();
  if (!question || state.busy) return;
  setBusy(true);
  const turn = renderTurn(question, fromSpeech);
  const body = JSON.stringify({ question, history: state.history, from_speech: fromSpeech });
  try {
    const res = await fetch("v1/query/stream", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body,
    });
    if (!res.ok) throw new Error(await errorMessage(res));
    await readEvents(res.body, (event, data) => handleEvent(turn, event, data));
  } catch (err) {
    showError(turn, err.message);
  } finally {
    turn.finishTimeline();
    setBusy(false);
    els.question.focus();
  }
}

async function readEvents(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message";
      const data = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (data.length) onEvent(event, JSON.parse(data.join("\n")));
    }
  }
}

function handleEvent(turn, event, data) {
  switch (event) {
    case "thinking":
      turn.step(firstSentence(data.text));
      break;
    case "tool_call":
      turn.step(describeToolCall(data));
      break;
    case "tool_result":
      if (data.tool === "run_sql") {
        turn.step(data.ok ? `Query returned ${data.row_count} row(s) in ${data.elapsed_ms} ms` : `Query failed: ${data.error}`, !data.ok);
      } else if (data.summary) {
        turn.step(data.summary);
      }
      break;
    case "fallback":
      turn.step(`Request rerouted to ${data.to_model || "a fallback model"}`);
      break;
    case "result":
      showResult(turn, data);
      break;
    case "error":
      showError(turn, data.message);
      break;
  }
}

function describeToolCall({ tool, input }) {
  if (tool === "run_sql") return "Running SQL";
  if (tool === "describe_tables") return `Inspecting ${(input.table_names || []).join(", ")}`;
  if (tool === "search_column_values") return `Looking up “${input.search_term}” in ${input.table}.${input.column}`;
  return `Calling ${tool}`;
}

function firstSentence(text) {
  const clean = (text || "").replace(/\s+/g, " ").trim();
  const m = clean.match(/^(.{20,200}?[.!?])\s/);
  return m ? m[1] : clean.slice(0, 200);
}

// ---------- rendering ----------
function renderTurn(question, fromSpeech) {
  const node = els.template.content.firstElementChild.cloneNode(true);
  node.querySelector(".question-text").textContent = question;
  node.querySelector(".source").textContent = fromSpeech ? "voice" : "typed";
  const timeline = node.querySelector(".timeline");
  els.conversation.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  let live = null;
  return {
    node,
    step(text, isError = false) {
      if (!text) return;
      live?.classList.remove("live");
      live = document.createElement("li");
      live.textContent = text;
      live.className = isError ? "err" : "live";
      timeline.appendChild(live);
    },
    finishTimeline() { live?.classList.remove("live"); },
  };
}

function showResult(turn, r) {
  const node = turn.node;
  if (els.question.value.trim() === r.question.trim()) els.question.value = "";
  node.querySelector(".answer").hidden = false;
  node.querySelector(".answer-text").textContent = r.answer;

  const clarify = node.querySelector(".clarify");
  clarify.hidden = !r.clarification_question;
  clarify.textContent = r.clarification_question || "";

  const assumptions = node.querySelector(".assumptions");
  assumptions.replaceChildren(...(r.assumptions || []).map((a) => Object.assign(document.createElement("li"), { textContent: a })));
  assumptions.hidden = !r.assumptions?.length;

  if (r.sql) {
    const details = node.querySelector(".sql");
    details.hidden = false;
    details.querySelector("code").textContent = r.sql;
    details.querySelector(".copy").onclick = (e) => {
      e.preventDefault();
      navigator.clipboard?.writeText(r.sql);
      e.target.textContent = "Copied";
      setTimeout(() => (e.target.textContent = "Copy"), 1200);
    };
  }
  if (r.columns?.length) renderTable(node.querySelector(".table-wrap"), r);

  const tokens = r.usage.input_tokens + r.usage.cache_read_tokens + r.usage.cache_write_tokens;
  node.querySelector(".meta").textContent =
    `${r.row_count}${r.truncated ? "+" : ""} row(s) · ${(r.timings_ms.total / 1000).toFixed(1)}s · ` +
    `${r.tool_calls} tool call(s) · ${tokens.toLocaleString()} in / ${r.usage.output_tokens.toLocaleString()} out tokens ` +
    `(${r.usage.cache_read_tokens.toLocaleString()} cached) · ${r.model}`;

  state.history.push({ question: r.question, sql: r.sql, answer: r.clarification_question || r.answer });
  state.history = state.history.slice(-(state.meta?.max_history_turns || 5));
  if (els.speak.checked && "speechSynthesis" in window) {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(r.clarification_question || r.answer));
  }
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

function renderTable(wrap, r) {
  wrap.hidden = false;
  const thead = wrap.querySelector("thead");
  const tbody = wrap.querySelector("tbody");
  const head = document.createElement("tr");
  for (const c of r.columns) head.appendChild(Object.assign(document.createElement("th"), { textContent: c }));
  thead.replaceChildren(head);
  const rows = r.rows.slice(0, MAX_ROWS_RENDERED).map((row) => {
    const tr = document.createElement("tr");
    for (const v of row) {
      const td = document.createElement("td");
      if (typeof v === "number") {
        td.className = "num";
        td.textContent = Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 2 });
      } else {
        td.textContent = v === null ? "NULL" : typeof v === "object" ? JSON.stringify(v) : String(v);
      }
      tr.appendChild(td);
    }
    return tr;
  });
  tbody.replaceChildren(...rows);
  if (r.rows.length > MAX_ROWS_RENDERED) {
    const tr = document.createElement("tr");
    const td = Object.assign(document.createElement("td"), { colSpan: r.columns.length, textContent: `Showing ${MAX_ROWS_RENDERED} of ${r.rows.length} rows` });
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
}

function showError(turn, message) {
  const el = turn.node.querySelector(".error");
  el.hidden = false;
  el.textContent = message;
}

// ---------- wiring ----------
els.ask.addEventListener("click", () => ask(els.question.value, false));
els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask(els.question.value, false);
  }
});
els.mic.addEventListener("click", () => {
  if (state.recorder) state.recorder.stop();
  else if (!state.busy) startRecording();
});
els.saveKey.addEventListener("click", () => {
  store.set("s2s_api_key", els.apiKey.value.trim());
  els.auth.hidden = true;
});
els.speak.addEventListener("change", () => store.set("s2s_speak", els.speak.checked ? "1" : "0"));
els.newChat.addEventListener("click", () => {
  state.history = [];
  els.conversation.replaceChildren();
  els.question.value = "";
  els.recStatus.textContent = "";
});

init();
