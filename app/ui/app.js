"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("tts-form");
const generateButton = byId("generate");
const statusBox = byId("status");
const result = byId("result");
const player = byId("audio-player");
const download = byId("download");
let resultUrl = null;
let playbackUrl = null;
let voicePreviewUrl = null;
const defaultVoiceByLanguage = {english: "Marcus"};

function baseUrl() {
  const entered = byId("server-url").value.trim().replace(/\/+$/, "");
  if (entered) return entered;
  return "http://127.0.0.1:8080";
}

function syncApiDocsLink() {
  byId("api-docs").href = `${baseUrl()}/docs`;
}

function apiKeyHeaders() {
  const headers = {};
  const key = byId("api-key").value;
  if (key) headers["x-api-key"] = key;
  return headers;
}

function requestHeaders() {
  return {"content-type": "application/json", ...apiKeyHeaders()};
}

let serverModel = "";   // GET /controls `model`: the name of the server's model, shown next to "Server ready"

function setConnection(kind, label) {
  byId("connection-dot").className = `status-dot ${kind || ""}`;
  byId("connection-label").textContent = kind === "connected" && serverModel ? `${label} · ${serverModel}` : label;
}

function setStatus(kind, message) {
  statusBox.replaceChildren();
  const dot = document.createElement("span");
  dot.className = `status-dot ${kind || ""}`;
  dot.setAttribute("aria-hidden", "true");
  const text = document.createElement("span");
  text.textContent = message;
  statusBox.append(dot, text);
}

function setLoading(loading) {
  generateButton.disabled = loading;
  generateButton.classList.toggle("is-loading", loading);
  generateButton.querySelector(".button-label").textContent = loading ? "Generating…" : "Generate speech";
}

function showVoicePlaceholder(message) {
  const select = byId("voice-select");
  select.replaceChildren();
  const option = new Option(message, "");
  option.disabled = true;
  option.selected = true;
  select.add(option);
  byId("voice-style-badge").hidden = true;
}

function displayStyle(style) {
  return style ? style.charAt(0).toUpperCase() + style.slice(1) : "";
}

function syncVoiceStyle() {
  const option = byId("voice-select").selectedOptions[0];
  const style = option?.dataset.style || "";
  const badge = byId("voice-style-badge");
  badge.hidden = !style;
  badge.querySelector("strong").textContent = displayStyle(style);
}

function optionalNumber(id) {
  const value = byId(id).value.trim();
  return value === "" ? undefined : Number(value);
}

function optionalJson(id, label) {
  const value = byId(id).value.trim();
  if (!value) return undefined;
  try { return JSON.parse(value); }
  catch (_) { throw new Error(`${label} must be valid JSON.`); }
}

function bytesToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

async function audioBlobBase64(blob) {
  if (Math.ceil(blob.size / 3) * 4 > 8 * 1024 * 1024) {
    throw new Error("The encoded voice reference exceeds the server's 8 MiB request limit.");
  }
  return bytesToBase64(await blob.arrayBuffer());
}

async function voiceFileBase64() {
  const file = byId("voice-file").files[0];
  return file ? audioBlobBase64(file) : undefined;
}

async function selectedVoiceBase64() {
  const url = byId("voice-select").value;
  if (!url) return undefined;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load the selected sample voice (${response.status}).`);
  return audioBlobBase64(await response.blob());
}

// --- Request fields the server reads -------------------------------------------
// GET /controls of a server whose model reads only some request fields (TontaubeV1.1 Flash) lists them as `fields`,
// with the model's name. The inputs of the other fields are disabled and not sent. null: every field (TontaubeV1.1).

let serverFields = null;
let serverDevice = "";   // GET /controls `device`: where the server's model runs ("GPU (<name>)", "CPU (<n> threads)")
const fieldInputs = {
  language: ["language"], tag: ["tag"], prompt_max_tokens: ["prompt-max-tokens"],
  voice_audio_b64: ["voice-select", "refresh-voices", "voice-file"], voice_path: ["voice-path"],
  voice_tokens: ["voice-tokens"], prefix_text: ["prefix-text"], prefix_tokens: ["prefix-tokens"],
  top_p: ["top-p"], top_k: ["top-k"], frequency_penalty: ["frequency-penalty"],
  silence_logit_bias: ["silence-logit-bias"], acoustic_temperature: ["acoustic-temperature"],
  acoustic_top_k: ["acoustic-top-k"], max_new_tokens: ["max-new-tokens"],
  streaming_initial_seconds: ["streaming-initial-seconds"], vllm_priority: ["vllm-priority"],
  vibevoice_postprocess: ["vibevoice-postprocess"],
};

function supported(field) {
  return !serverFields || serverFields.includes(field);
}

function setServerFields(fields, model) {
  const limited = Array.isArray(fields);
  if (limited !== Boolean(serverFields)) {
    // Flash reads written text with its built-in verbalizer unless told not to; TontaubeV1.1's is optional.
    byId("use-verbalization").checked = limited;
    byId("verbalization-hint").textContent = limited ? "Built into this model" : "Requires the optional server model";
    // Flash is a streaming model: start in stream mode there (the complete file stays selectable).
    const mode = document.querySelector(`input[name="transport"][value="${limited ? "stream" : "batch"}"]`);
    if (mode) mode.checked = true;
  }
  serverFields = limited ? fields : null;
  for (const [field, ids] of Object.entries(fieldInputs)) {
    for (const id of ids) byId(id).disabled = !supported(field);
  }
  if (!supported("language")) byId("language").value = "english";
  byId("voice-upload").hidden = !supported("voice_audio_b64");
  byId("voice-hint").textContent = supported("voice_audio_b64")
    ? "Sample voices are read from the UI's local voice folder and uploaded with each request. Uploading a reference below overrides the selected sample."
    : `${model || "This server's model"} has one voice, Marcus, and speaks English.`;
  byId("sliders-card").hidden = !supported("controls");
  syncTransportState();
}

// --- Style sliders -----------------------------------------------------------
// The catalog comes from GET /controls. A slider the user has not touched is "Model
// default" and is not sent. A range input has no empty value, so an untouched slider
// sits dimmed at its middle step; the first interaction sets it, and its Reset button
// (or "Reset all") returns it to the model default.

const CONTROLS_RETRY_MS = [3000, 5000, 8000, 12000, 15000];  // model still loading: slow, capped
const CONTROLS_DEBOUNCE_MS = 600;
const CONTROLS_TIMEOUT_MS = 8000;
const sliderValues = new Map();  // name -> {bins, value}; absent = model default
let sliderCatalog = [];
let catalogSignature = null;
let controlsRequest = 0;
let controlsAbort = null;
let controlsTimer = null;
let controlsRetries = 0;
let controlsRetryWanted = false;

function validCatalog(list) {
  return list.filter((c) => c && typeof c.name === "string" && c.name
    && Number.isInteger(c.bins) && c.bins >= 1)
    .map((c) => ({
      name: c.name, bins: c.bins, description: String(c.description || ""),
      low: String(c.low || ""), high: String(c.high || ""),
      defaultStep: Number.isInteger(c.default) && c.default >= 0 && c.default < c.bins ? c.default : null,
    }));
}

function sliderTitle(control) {
  return control.description || control.name.replaceAll("_", " ");
}

function sliderRestingStep(control) {
  // Where an unset slider sits: the step the catalog names as the model's default, else the middle.
  if (control.defaultStep !== null) return control.defaultStep;
  return Math.floor((control.bins - 1) / 2);
}

function sliderStepMeaning(control, step) {
  if (control.bins < 2) return "";
  if (step === 0 && control.low) return control.low;
  if (step === control.bins - 1 && control.high) return control.high;
  if (control.low && control.high) return `between ${control.low} and ${control.high}`;
  return "";
}

function sliderValueText(control, step, isSet) {
  if (!isSet) return "Model default, not set";
  const meaning = sliderStepMeaning(control, step);
  return `Step ${step} of 0 to ${control.bins - 1}${meaning ? `, ${meaning}` : ""}`;
}

function setSliderStatus(kind, message) {
  const box = byId("slider-status");
  box.replaceChildren();
  box.hidden = !message;
  if (!message) return;
  const dot = document.createElement("span");
  dot.className = `status-dot ${kind || ""}`;
  dot.setAttribute("aria-hidden", "true");
  const text = document.createElement("span");
  text.textContent = message;
  box.append(dot, text);
}

function setSliderCount() {
  const set = sliderCatalog.filter((c) => sliderValues.has(c.name)).length;
  byId("slider-count").textContent = sliderCatalog.length
    ? (set ? `${set} of ${sliderCatalog.length} set` : "None set, the model decides")
    : "";
  byId("reset-sliders").setAttribute("aria-disabled", set ? "false" : "true");
}

function paintSlider(row, control) {
  const entry = sliderValues.get(control.name);
  const isSet = entry !== undefined;
  const step = isSet ? entry.value : sliderRestingStep(control);
  const input = row.querySelector("input");
  input.value = String(step);
  input.setAttribute("aria-valuetext", sliderValueText(control, step, isSet));
  row.classList.toggle("is-set", isSet);
  row.querySelector(".slider-value").textContent = isSet ? `Step ${step}` : "Model default";
  row.querySelector(".slider-reset").hidden = !isSet;
  setSliderCount();
}

function setSliderValue(row, control, value) {
  if (value === null) sliderValues.delete(control.name);
  else sliderValues.set(control.name, {bins: control.bins, value});
  paintSlider(row, control);
}

const sliderNavigationKeys = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]);

function buildSliderRow(control, index) {
  const row = document.createElement("div");
  row.className = "slider-row";
  row.dataset.control = control.name;
  const inputId = `slider-${index}`;

  const head = document.createElement("div");
  head.className = "slider-head";
  const label = document.createElement("label");
  label.htmlFor = inputId;
  label.textContent = sliderTitle(control);
  const value = document.createElement("span");
  value.className = "slider-value";
  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "text-button slider-reset";
  reset.textContent = "Reset";
  reset.setAttribute("aria-label", `Reset ${sliderTitle(control)} to model default`);
  head.append(label, value, reset);

  const input = document.createElement("input");
  input.type = "range";
  input.id = inputId;
  input.min = "0";
  input.max = String(control.bins - 1);
  input.step = "1";
  input.dataset.control = control.name;
  const ends = document.createElement("div");
  ends.className = "slider-ends";
  const low = document.createElement("span");
  low.textContent = control.low ? `0 · ${control.low}` : "0";
  const high = document.createElement("span");
  high.textContent = control.high ? `${control.bins - 1} · ${control.high}` : String(control.bins - 1);
  ends.append(low, high);
  ends.id = `${inputId}-ends`;
  input.setAttribute("aria-describedby", ends.id);
  row.append(head, input, ends);

  // An untouched slider has no value to "change", so a click on the thumb or a
  // navigation key also counts as setting it (input events alone would miss both).
  const activate = () => { if (!sliderValues.has(control.name)) setSliderValue(row, control, Number(input.value)); };
  input.addEventListener("input", () => setSliderValue(row, control, Number(input.value)));
  input.addEventListener("click", activate);
  input.addEventListener("keydown", (event) => { if (sliderNavigationKeys.has(event.key)) activate(); });
  reset.addEventListener("click", () => { setSliderValue(row, control, null); input.focus(); });
  paintSlider(row, control);
  return row;
}

function clearSliders() {
  sliderCatalog = [];
  catalogSignature = null;
  byId("slider-fields").replaceChildren();
  setSliderCount();
}

function showCatalog(controls) {
  const catalog = validCatalog(controls);
  for (const [name, entry] of [...sliderValues]) {
    const control = catalog.find((c) => c.name === name);
    if (!control || control.bins !== entry.bins) sliderValues.delete(name);
  }
  const signature = JSON.stringify(catalog);
  if (signature !== catalogSignature) {
    catalogSignature = signature;
    sliderCatalog = catalog;
    byId("slider-fields").replaceChildren(...catalog.map(buildSliderRow));
  }
  setSliderCount();
  setSliderStatus("", catalog.length ? "" : "This model has no style sliders.");
}

function resetAllSliders() {
  if (byId("reset-sliders").getAttribute("aria-disabled") === "true") return;
  for (const control of sliderCatalog) sliderValues.delete(control.name);
  for (const row of byId("slider-fields").children) {
    paintSlider(row, sliderCatalog.find((c) => c.name === row.dataset.control));
  }
  byId("slider-fields").querySelector("input")?.focus();
}

function selectedControls() {
  const controls = {};
  for (const control of sliderCatalog) {
    const entry = sliderValues.get(control.name);
    if (entry) controls[control.name] = entry.value;
  }
  return Object.keys(controls).length ? controls : undefined;
}

function cancelControlsWork() {
  controlsRequest += 1;
  clearTimeout(controlsTimer);
  controlsTimer = null;
  if (controlsAbort) controlsAbort.abort();
  controlsAbort = null;
}

function controlsUnavailable(message) {
  clearSliders();
  controlsRetryWanted = false;
  byId("sliders-card").hidden = false;
  setSliderStatus("error", message);
}

async function loadControls() {
  cancelControlsWork();
  if (document.hidden) { controlsRetryWanted = true; return; }
  const request = controlsRequest;
  const controller = new AbortController();
  controlsAbort = controller;
  const timeout = setTimeout(() => controller.abort(), CONTROLS_TIMEOUT_MS);
  const url = baseUrl();
  if (!sliderCatalog.length) setSliderStatus("", "Looking for the server's style sliders…");
  try {
    const response = await fetch(`${url}/controls`, {headers: apiKeyHeaders(), signal: controller.signal});
    if (request !== controlsRequest) return;
    if (response.status === 503) {
      clearSliders();
      setConnection("", "Server loading");
      const delay = CONTROLS_RETRY_MS[Math.min(controlsRetries, CONTROLS_RETRY_MS.length - 1)];
      controlsRetries += 1;
      controlsRetryWanted = true;
      setSliderStatus("", "The model is still loading. The sliders will appear here as soon as it is ready.");
      controlsTimer = setTimeout(loadControls, delay);
      return;
    }
    if (response.status === 401 || response.status === 403) {
      controlsUnavailable("The server wants an API key, or rejected the one entered. Enter it under Connection.");
      return;
    }
    if (!response.ok) {
      controlsUnavailable(`The server could not list its sliders (${response.status}).`);
      return;
    }
    const body = await response.json();
    if (request !== controlsRequest) return;
    controlsRetryWanted = false;
    controlsRetries = 0;
    showCatalog(Array.isArray(body.controls) ? body.controls : []);
    serverDevice = typeof body.device === "string" ? body.device : "";
    serverModel = typeof body.model === "string" ? body.model : "";
    setServerFields(body.fields, body.model);
    setConnection("connected", "Server ready");
  } catch (error) {
    if (request !== controlsRequest) return;  // superseded, or cancelled while hidden
    controlsUnavailable(error.name === "AbortError"
      ? "The server did not answer in time, so its sliders could not be listed. Use Check connection to try again."
      : "Server not reachable, so its sliders cannot be listed. Check the Server URL, then use Check connection.");
  } finally {
    clearTimeout(timeout);
    if (controlsAbort === controller) controlsAbort = null;
  }
}

function refreshControls() {
  controlsRetries = 0;
  return loadControls();
}

let controlsDebounce = null;
function scheduleControlsRefresh() {
  // The server address or key changed: drop any retry loop aimed at the old one.
  cancelControlsWork();
  controlsRetryWanted = false;
  clearTimeout(controlsDebounce);
  if (!sliderCatalog.length) setSliderStatus("", "Waiting for the new server address…");
  controlsDebounce = setTimeout(refreshControls, CONTROLS_DEBOUNCE_MS);
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    // Stop polling in a hidden tab; resume on return if still waiting for the model.
    if (controlsTimer !== null || controlsAbort !== null) controlsRetryWanted = true;
    cancelControlsWork();
  } else if (controlsRetryWanted) {
    refreshControls();
  }
});

async function buildPayload(streaming) {
  const text = byId("text").value.trim();
  if (!text) throw new Error("Enter text to generate.");
  const temperature = optionalNumber("temperature");
  if (temperature === undefined) throw new Error("Enter a temperature.");
  const payload = {
    text,
    language: byId("language").value,
    tag: byId("tag").value,
    temperature,
    priority: byId("priority").value,
    bitrate: byId("bitrate").value,
    use_verbalization: byId("use-verbalization").checked,
  };

  const numberFields = {
    top_p: "top-p", top_k: "top-k", frequency_penalty: "frequency-penalty",
    silence_logit_bias: "silence-logit-bias", seed: "seed",
    max_new_tokens: "max-new-tokens", acoustic_temperature: "acoustic-temperature",
    acoustic_top_k: "acoustic-top-k",
  };
  for (const [key, id] of Object.entries(numberFields)) {
    const value = optionalNumber(id);
    if (value !== undefined) payload[key] = value;
  }

  const controls = selectedControls();
  if (controls) payload.controls = controls;
  const prefixText = byId("prefix-text").value;
  if (prefixText) payload.prefix_text = prefixText;
  const vibevoice = byId("vibevoice-postprocess").value;
  if (vibevoice) payload.vibevoice_postprocess = vibevoice === "true";

  const promptCap = optionalNumber("prompt-max-tokens");
  if (promptCap !== undefined) payload.prompt_max_tokens = promptCap;
  if (supported("voice_audio_b64")) {
    const uploaded = await voiceFileBase64();
    const manualVoicePath = byId("voice-path").value.trim();
    const voiceTokens = optionalJson("voice-tokens", "Voice tokens");
    const explicitVoiceSources = [Boolean(uploaded), Boolean(manualVoicePath), voiceTokens !== undefined]
      .filter(Boolean).length;
    if (explicitVoiceSources > 1) {
      throw new Error("Provide only one uploaded voice, server-side voice path, or set of voice-token rows.");
    }
    if (uploaded) payload.voice_audio_b64 = uploaded;
    else if (manualVoicePath) payload.voice_path = manualVoicePath;
    else if (voiceTokens !== undefined) payload.voice_tokens = voiceTokens;
    else {
      const selected = await selectedVoiceBase64();
      if (selected) payload.voice_audio_b64 = selected;
    }
    if (!payload.voice_audio_b64 && !payload.voice_path && payload.voice_tokens === undefined) {
      throw new Error("Select a sample voice, upload a voice reference, or provide a server path or token rows.");
    }
  }
  const prefixTokens = optionalJson("prefix-tokens", "Prefix tokens");
  if (prefixTokens !== undefined) payload.prefix_tokens = prefixTokens;

  if (streaming) {
    payload.format = byId("format").value;   // pcm16 (default) or mp3
    payload.streaming_initial_seconds = optionalNumber("streaming-initial-seconds");
    if (payload.vibevoice_postprocess === false) throw new Error("Streaming requires VibeVoice processing.");
  } else {
    payload.format = byId("format").value;
    const vllmPriority = optionalNumber("vllm-priority");
    if (vllmPriority !== undefined) payload.vllm_priority = vllmPriority;
  }
  for (const field of Object.keys(payload)) {
    if (!supported(field)) delete payload[field];
  }
  return payload;
}

async function parseError(response) {
  const fallback = `${response.status} ${response.statusText}`;
  try {
    const data = await response.json();
    const detail = data.error || data.detail || fallback;
    return Array.isArray(detail) ? detail.map((item) => item.msg).join("; ") : detail;
  } catch (_) { return fallback; }
}

function decodeBase64(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function generateComplete(payload) {
  const started = performance.now();
  const response = await fetch(`${baseUrl()}/predict`, {method: "POST", headers: requestHeaders(), body: JSON.stringify(payload)});
  if (!response.ok) throw new Error(await parseError(response));
  const data = await response.json();
  return {blob: new Blob([decodeBase64(data.audio_b64)], {type: data.mime_type}), mime: data.mime_type, elapsed: (performance.now() - started) / 1000};
}

function waitForEvent(target, success, failure) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      target.removeEventListener(success, onSuccess);
      if (failure) target.removeEventListener(failure, onFailure);
    };
    const onSuccess = () => { cleanup(); resolve(); };
    const onFailure = () => { cleanup(); reject(new Error("Browser audio streaming failed.")); };
    target.addEventListener(success, onSuccess, {once: true});
    if (failure) target.addEventListener(failure, onFailure, {once: true});
  });
}

async function openMp3Playback() {
  if (!("MediaSource" in window) || !MediaSource.isTypeSupported("audio/mpeg")) return null;

  const mediaSource = new MediaSource();
  if (playbackUrl) URL.revokeObjectURL(playbackUrl);
  playbackUrl = URL.createObjectURL(mediaSource);
  player.src = playbackUrl;
  result.hidden = false;
  download.hidden = true;
  byId("result-meta").textContent = "Waiting for the first MP3 chunk…";
  let sourceBuffer;
  try {
    await waitForEvent(mediaSource, "sourceopen", "error");
    sourceBuffer = mediaSource.addSourceBuffer("audio/mpeg");
  } catch (_) {
    URL.revokeObjectURL(playbackUrl);
    playbackUrl = null;
    player.removeAttribute("src");
    player.load();
    result.hidden = true;
    return null;
  }

  return {
    async append(bytes) {
      if (sourceBuffer.updating) await waitForEvent(sourceBuffer, "updateend", "error");
      const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      sourceBuffer.appendBuffer(copy);
      await waitForEvent(sourceBuffer, "updateend", "error");
      player.play().catch(() => {});
    },
    close() {
      if (mediaSource.readyState === "open" && !sourceBuffer.updating) mediaSource.endOfStream();
    },
  };
}

// Raw PCM stream (format pcm16: int16 little-endian mono at the rate each event names in `sample_rate`): played
// through Web Audio as it arrives.
let stopLivePcm = null;   // stops the PCM stream that is still playing through Web Audio, if any
let streamAbort = null;   // aborts the stream request in flight, if any

function syncStop() {
  byId("stop").hidden = !streamAbort && !stopLivePcm;
}

function openPcmPlayback() {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  let ctx = null;   // made with the first chunk, at the stream's rate
  let next = 0;
  let last = null;
  let complete = false;   // every chunk of the stream is queued
  const stop = () => {    // also the natural end: the context is closed, so contexts do not pile up
    if (stopLivePcm !== stop) return;
    stopLivePcm = null;
    if (ctx) ctx.close();
    syncStop();
  };
  if (stopLivePcm) stopLivePcm();
  stopLivePcm = stop;
  result.hidden = false;
  download.hidden = true;
  byId("result-meta").textContent = "Waiting for the first audio chunk…";
  return {
    async append(bytes, rate) {
      const n = Math.floor(bytes.byteLength / 2);
      if (!n || stopLivePcm !== stop) return;   // stopped: the chunks still go into the saved file
      if (!ctx) ctx = new Ctx({sampleRate: rate});
      const view = new DataView(bytes.buffer, bytes.byteOffset, n * 2);
      const buf = ctx.createBuffer(1, n, rate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < n; i++) ch[i] = view.getInt16(i * 2, true) / 32768;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      next = Math.max(next, ctx.currentTime + 0.02);
      src.start(next);
      next += buf.duration;
      last = src;
      src.onended = () => { if (complete && src === last) stop(); };
    },
    close() {
      complete = true;
      if (!ctx || ctx.currentTime >= next) stop();
    },
  };
}

function pcmToWav(parts, rate) {
  const size = parts.reduce((total, part) => total + part.byteLength, 0);
  const header = new DataView(new ArrayBuffer(44));
  const text = (offset, value) => { for (let i = 0; i < value.length; i++) header.setUint8(offset + i, value.charCodeAt(i)); };
  text(0, "RIFF"); header.setUint32(4, 36 + size, true); text(8, "WAVE"); text(12, "fmt ");
  header.setUint32(16, 16, true); header.setUint16(20, 1, true); header.setUint16(22, 1, true);
  header.setUint32(24, rate, true); header.setUint32(28, rate * 2, true); header.setUint16(32, 2, true);
  header.setUint16(34, 16, true); text(36, "data"); header.setUint32(40, size, true);
  return new Blob([header.buffer, ...parts], {type: "audio/wav"});
}

async function generateStream(payload) {
  const started = performance.now();
  const abort = streamAbort;
  const response = await fetch(`${baseUrl()}/stream`, {method: "POST", headers: requestHeaders(), body: JSON.stringify(payload), signal: abort.signal});
  if (!response.ok) throw new Error(await parseError(response));
  if (!response.body) throw new Error("This browser cannot read streaming responses.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parts = [];
  const pcm = payload.format === "pcm16";
  const livePlayback = pcm ? openPcmPlayback() : await openMp3Playback();
  let rate = 24000;   // a PCM stream's sample rate (the events' `sample_rate`; 24 kHz from servers that do not name it)
  let buffer = "";
  let chunks = 0;
  let ttfa = null;   // seconds from sending the request to the first audio chunk (time to first audio)
  while (true) {
    let read;
    try { read = await reader.read(); }
    catch (error) { if (abort.signal.aborted) break; throw error; }   // Stop: keep what has arrived
    buffer += decoder.decode(read.value || new Uint8Array(), {stream: !read.done});
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const event of events) {
      const line = event.split("\n").find((entry) => entry.startsWith("data:"));
      if (!line) continue;
      const data = JSON.parse(line.slice(5).trim());
      if (data.error) throw new Error(data.error);
      if (data.audio_b64) {
        const bytes = decodeBase64(data.audio_b64);
        if (ttfa === null) ttfa = (performance.now() - started) / 1000;
        parts.push(bytes);
        if (data.sample_rate) rate = data.sample_rate;
        if (livePlayback) await livePlayback.append(bytes, rate);
        chunks += 1;
        setStatus("", `Receiving audio chunk ${chunks}…`);
        byId("result-meta").textContent = `Playing ${pcm ? "PCM" : "MP3"} stream · received ${chunks} chunk${chunks === 1 ? "" : "s"}`;
      }
    }
    if (read.done) break;
  }
  const stopped = abort.signal.aborted;
  if (!parts.length) throw stopped ? new DOMException("Stopped", "AbortError") : new Error("The stream ended without audio.");
  if (livePlayback) livePlayback.close();
  if (pcm) return {blob: pcmToWav(parts, rate), mime: `audio/pcm (${rate / 1000} kHz, saved as WAV)`, elapsed: (performance.now() - started) / 1000, chunks, ttfa, livePlayback: Boolean(livePlayback), pcm: true, stopped};
  return {blob: new Blob(parts, {type: "audio/mpeg"}), mime: "audio/mpeg", elapsed: (performance.now() - started) / 1000, chunks, ttfa, livePlayback: Boolean(livePlayback), stopped};
}

function showResult(output, extension) {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(output.blob);
  if (!output.livePlayback || output.pcm) {
    if (playbackUrl) URL.revokeObjectURL(playbackUrl);
    playbackUrl = null;
    player.src = resultUrl;
  }
  download.href = resultUrl;
  download.download = `tontaube-${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`;
  download.hidden = false;
  byId("result-meta").textContent = `${output.mime} · ${(output.blob.size / 1024).toFixed(1)} KiB · generated in ${output.elapsed.toFixed(1)} s${output.ttfa != null ? ` · latency ${Math.round(output.ttfa * 1000)} ms` : ""}${output.chunks ? ` · ${output.chunks} streamed chunks` : ""}${serverDevice ? ` · ${serverDevice}` : ""}`;
  result.hidden = false;
  result.scrollIntoView({behavior: "smooth", block: "nearest"});
  if (!output.livePlayback && !output.stopped) player.play().catch(() => {});
}

async function checkServer({refresh = true} = {}) {
  if (refresh) serverModel = "";   // named again by the controls loaded below
  setConnection("", "Checking…");
  try {
    const response = await fetch(`${baseUrl()}/readyz`, {headers: apiKeyHeaders()});
    if (response.status === 503) {
      setConnection("", "Server loading");
      setStatus("", "Inference server is reachable and still loading models.");
      if (refresh) refreshControls();
      return;
    }
    if (!response.ok) throw new Error(await parseError(response));
    setConnection("connected", "Server ready");
    if (refresh) await refreshControls();
  } catch (error) {
    setConnection("error", "Unavailable");
    setStatus("error", `Could not reach the inference API: ${error.message}`);
  }
}

async function refreshVoices(quiet = false) {
  const select = byId("voice-select");
  const refresh = byId("refresh-voices");
  const previous = select.value;
  refresh.disabled = true;
  try {
    const language = encodeURIComponent(byId("language").value);
    const response = await fetch(`/ui/voices?language=${language}`);
    if (!response.ok) throw new Error(await parseError(response));
    const data = await response.json();
    const voices = Array.isArray(data.voices) ? data.voices : [];
    select.replaceChildren();
    if (!voices.length) {
      showVoicePlaceholder("No sample voices found for this language");
      if (!quiet) setStatus("error", "No audio files were found in the local UI voice folder.");
      return;
    }
    const styleGroups = new Map();
    for (const voice of voices) {
      const option = new Option(voice.name, voice.url);
      option.dataset.style = voice.style || "";
      if (!voice.style) {
        select.add(option);
        continue;
      }
      let group = styleGroups.get(voice.style);
      if (!group) {
        group = document.createElement("optgroup");
        group.label = displayStyle(voice.style);
        styleGroups.set(voice.style, group);
        select.append(group);
      }
      group.append(option);
    }
    if (voices.some((voice) => voice.url === previous)) {
      select.value = previous;
    } else {
      const defaultName = defaultVoiceByLanguage[byId("language").value];
      const preferred = voices.find((voice) => voice.name === defaultName);
      if (preferred) select.value = preferred.url;
    }
    syncVoiceStyle();
    if (!quiet) setStatus("ready", `Loaded ${voices.length} sample voice${voices.length === 1 ? "" : "s"}.`);
  } catch (error) {
    showVoicePlaceholder("Could not load voices from the local UI folder");
    if (!quiet) setStatus("error", `Could not refresh voices: ${error.message}`);
  } finally {
    refresh.disabled = false;
  }
}

function syncTransportState() {
  const streaming = document.querySelector('input[name="transport"]:checked').value === "stream";
  const format = byId("format");
  for (const option of format.options) {
    option.disabled = streaming ? !["pcm16", "mp3"].includes(option.value) : option.value === "pcm16";
  }
  if (streaming && !["pcm16", "mp3"].includes(format.value)) format.value = "pcm16";
  if (!streaming && format.value === "pcm16") format.value = "wav";
  byId("vllm-priority").disabled = streaming || !supported("vllm_priority");
  byId("streaming-initial-seconds").disabled = !streaming || !supported("streaming_initial_seconds");
}

function showVoiceFile() {
  const file = byId("voice-file").files[0];
  byId("voice-file-meta").hidden = !file;
  byId("voice-preview").pause();
  byId("voice-preview").removeAttribute("src");
  byId("voice-preview").hidden = !file;
  if (voicePreviewUrl) URL.revokeObjectURL(voicePreviewUrl);
  voicePreviewUrl = undefined;
  if (!file) return;
  byId("voice-file-name").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MiB`;
  voicePreviewUrl = URL.createObjectURL(file);
  byId("voice-preview").src = voicePreviewUrl;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const streaming = document.querySelector('input[name="transport"]:checked').value === "stream";
  setLoading(true);
  setStatus("", streaming ? "Starting stream…" : "Generating speech…");
  if (stopLivePcm) stopLivePcm();   // one stream at a time
  player.pause();
  streamAbort = streaming ? new AbortController() : null;
  syncStop();
  try {
    const payload = await buildPayload(streaming);
    const output = streaming ? await generateStream(payload) : await generateComplete(payload);
    showResult(output, streaming ? (payload.format === "pcm16" ? "wav" : "mp3") : payload.format);
    setStatus("ready", output.stopped ? "Stopped" : "Generation complete");
    setConnection("connected", "Server ready");
  } catch (error) { setStatus(...(error.name === "AbortError" ? ["", "Stopped"] : ["error", error.message || "Generation failed"])); }
  finally { streamAbort = null; syncStop(); setLoading(false); }
});
// Stop: the request in flight, the PCM stream still playing, and the player bar (the MP3 stream plays through it).
byId("stop").addEventListener("click", () => {
  if (streamAbort) streamAbort.abort();
  if (stopLivePcm) stopLivePcm();
  player.pause();
});
// Playing the saved file takes over from a PCM stream that is still sounding.
player.addEventListener("play", () => { if (stopLivePcm) stopLivePcm(); });

byId("check-server").addEventListener("click", () => checkServer());
byId("server-url").addEventListener("input", () => { syncApiDocsLink(); scheduleControlsRefresh(); });
byId("api-key").addEventListener("input", scheduleControlsRefresh);
byId("reset-sliders").addEventListener("click", resetAllSliders);
byId("refresh-voices").addEventListener("click", () => refreshVoices());
byId("language").addEventListener("change", () => refreshVoices());
byId("text").addEventListener("input", () => { byId("char-count").textContent = `${byId("text").value.length.toLocaleString()} characters`; });
byId("voice-file").addEventListener("change", showVoiceFile);
byId("clear-voice").addEventListener("click", () => { byId("voice-file").value = ""; showVoiceFile(); });
byId("random-seed").addEventListener("click", () => { byId("seed").value = String(Math.floor(Math.random() * 2147483647)); });
byId("voice-select").addEventListener("change", syncVoiceStyle);
document.querySelectorAll('input[name="transport"]').forEach((input) => input.addEventListener("change", syncTransportState));
syncApiDocsLink();
byId("text").dispatchEvent(new Event("input"));
syncTransportState();
refreshVoices(true);
refreshControls();
checkServer({refresh: false});
