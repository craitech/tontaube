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
const defaultVoiceByLanguage = {english: "Miles"};

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

function setConnection(kind, label) {
  byId("connection-dot").className = `status-dot ${kind || ""}`;
  byId("connection-label").textContent = label;
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

  const prefixText = byId("prefix-text").value;
  if (prefixText) payload.prefix_text = prefixText;
  const vibevoice = byId("vibevoice-postprocess").value;
  if (vibevoice) payload.vibevoice_postprocess = vibevoice === "true";

  const promptCap = optionalNumber("prompt-max-tokens");
  if (promptCap !== undefined) payload.prompt_max_tokens = promptCap;
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
  const prefixTokens = optionalJson("prefix-tokens", "Prefix tokens");
  if (prefixTokens !== undefined) payload.prefix_tokens = prefixTokens;

  if (streaming) {
    payload.format = "mp3";
    payload.streaming_initial_seconds = optionalNumber("streaming-initial-seconds");
    if (payload.vibevoice_postprocess === false) throw new Error("MP3 streaming requires VibeVoice processing.");
  } else {
    payload.format = byId("format").value;
    const vllmPriority = optionalNumber("vllm-priority");
    if (vllmPriority !== undefined) payload.vllm_priority = vllmPriority;
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

async function generateStream(payload) {
  const started = performance.now();
  const response = await fetch(`${baseUrl()}/stream`, {method: "POST", headers: requestHeaders(), body: JSON.stringify(payload)});
  if (!response.ok) throw new Error(await parseError(response));
  if (!response.body) throw new Error("This browser cannot read streaming responses.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parts = [];
  const livePlayback = await openMp3Playback();
  let buffer = "";
  let chunks = 0;
  while (true) {
    const read = await reader.read();
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
        parts.push(bytes);
        if (livePlayback) await livePlayback.append(bytes);
        chunks += 1;
        setStatus("", `Receiving audio chunk ${chunks}…`);
        byId("result-meta").textContent = `Playing MP3 stream · received ${chunks} chunk${chunks === 1 ? "" : "s"}`;
      }
    }
    if (read.done) break;
  }
  if (!parts.length) throw new Error("The stream ended without audio.");
  if (livePlayback) livePlayback.close();
  return {blob: new Blob(parts, {type: "audio/mpeg"}), mime: "audio/mpeg", elapsed: (performance.now() - started) / 1000, chunks, livePlayback: Boolean(livePlayback)};
}

function showResult(output, extension) {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(output.blob);
  if (!output.livePlayback) {
    if (playbackUrl) URL.revokeObjectURL(playbackUrl);
    playbackUrl = null;
    player.src = resultUrl;
  }
  download.href = resultUrl;
  download.download = `tontaube-${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`;
  download.hidden = false;
  byId("result-meta").textContent = `${output.mime} · ${(output.blob.size / 1024).toFixed(1)} KiB · generated in ${output.elapsed.toFixed(1)} s${output.chunks ? ` · ${output.chunks} streamed chunks` : ""}`;
  result.hidden = false;
  result.scrollIntoView({behavior: "smooth", block: "nearest"});
  player.play().catch(() => {});
}

async function checkServer() {
  setConnection("", "Checking…");
  try {
    const response = await fetch(`${baseUrl()}/readyz`, {headers: apiKeyHeaders()});
    if (response.status === 503) {
      setConnection("", "Server loading");
      setStatus("", "Inference server is reachable and still loading models.");
      return;
    }
    if (!response.ok) throw new Error(await parseError(response));
    setConnection("connected", "Server ready");
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
      if (!quiet) setStatus("error", "No audio files were found in the server voice folder.");
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
  byId("format").disabled = streaming;
  byId("vllm-priority").disabled = streaming;
  byId("streaming-initial-seconds").disabled = !streaming;
  if (streaming) byId("format").value = "mp3";
}

function showVoiceFile() {
  const file = byId("voice-file").files[0];
  byId("voice-file-meta").hidden = !file;
  byId("voice-preview").hidden = !file;
  if (!file) return;
  byId("voice-file-name").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MiB`;
  if (voicePreviewUrl) URL.revokeObjectURL(voicePreviewUrl);
  voicePreviewUrl = URL.createObjectURL(file);
  byId("voice-preview").src = voicePreviewUrl;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const streaming = document.querySelector('input[name="transport"]:checked').value === "stream";
  setLoading(true);
  setStatus("", streaming ? "Starting MP3 stream…" : "Generating speech…");
  try {
    const payload = await buildPayload(streaming);
    const output = streaming ? await generateStream(payload) : await generateComplete(payload);
    showResult(output, streaming ? "mp3" : payload.format);
    setStatus("ready", "Generation complete");
    setConnection("connected", "Server ready");
  } catch (error) { setStatus("error", error.message || "Generation failed"); }
  finally { setLoading(false); }
});

byId("check-server").addEventListener("click", checkServer);
byId("server-url").addEventListener("input", syncApiDocsLink);
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
checkServer();
