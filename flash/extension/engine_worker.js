// Engine worker: puts the model files into the WebAssembly heap once (from the OPFS cache, or downloaded with
// progress while being written to the cache, SHA-256 checked), loads the engine (tts.js: WebAssembly SIMD + threads)
// and speaks into the shared audio ring. A request runs to its end inside tts_speak; stop goes through the ring's
// CANCEL word, since this worker is busy while it speaks.
import { C, RingWriter } from './ring.js';

let M = null, ring = null, ctrl = null, kept = null, keptLen = 0, text = '';
// option gpu: the GPU worker (decoder, text reading), its flags (Int32 [0] decoder calls, [1] text reading: 1 done,
// 2 failed), the snapshot slot of each decoder state id; lazy: files kept out of the heap {name: {h, size}}
let gpu = null;
const lazy = {};
const GPU_SNAPS = 32;  // = gpu_worker.js
const post = (m, transfer) => self.postMessage(m, transfer || []);
const now = () => performance.now();
const abs = () => performance.timeOrigin + performance.now();  // comparable across the page and its workers

function heapU8(p, n) { return new Uint8Array(M.wasmMemory.buffer, p, n); }

// [begin, end) byte offsets into s as UTF-8 (as ccall encodes it: a lone surrogate as U+FFFD) -> UTF-16 indices
function u16Ranges(s, ranges) {
  const at = [];  // per byte: index of the character it belongs to
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i), pair = c >= 0xd800 && c < 0xdc00 && (s.charCodeAt(i + 1) & 0xfc00) === 0xdc00;
    for (let k = pair ? 4 : c < 0x80 ? 1 : c < 0x800 ? 2 : 3; k > 0; k--) at.push(i);
    if (pair) i++;
  }
  at.push(s.length);
  const u = (b) => at[Math.max(0, Math.min(b, at.length - 1))];
  return ranges.map(([a, z]) => [u(a), u(z)]);
}

// Audio and player state for the engine (called from C++ through EM_JS).
// kept (option keep): every sample of the request as it stands, replacements applied
function keep(at, src) {
  while (keptLen > at) {
    const last = kept[kept.length - 1];
    if (keptLen - last.length >= at) { kept.pop(); keptLen -= last.length; }
    else { kept[kept.length - 1] = last.subarray(0, at - (keptLen - last.length)); keptLen = at; }
  }
  kept.push(src.slice());
  keptLen += src.length;
}
const hooks = {
  push(p, n) {
    const src = new Float32Array(M.wasmMemory.buffer, p, n);
    if (kept) keep(keptLen, src);
    ring.push(src);
  },
  // the player's read position (samples of this request it has consumed) and the replacement of audio it has not
  // reached yet (redraw of a flagged spot; see RingWriter.replace)
  played() { return ring.played(); },
  replace(at, p, n, guard) {
    const src = new Float32Array(M.wasmMemory.buffer, p, n);
    const r = ring.replace(at, src, guard);
    if (r && kept) keep(at, src);
    return r;
  },
  buffered() { return ring.buffered(); },
  cancelled() { return ring.cancelled(); },
  status(json) { post({ type: 'status', status: JSON.parse(json), t: now() }); },
  sources(json) { post({ type: 'sources', ranges: u16Ranges(text, JSON.parse(json)) }); },  // each chunk's characters of the text
  chunk(c, at) { post({ type: 'chunk', c, at }); },  // chunk c starts at this sample of the request's audio
  // the GPU worker's calls (engine/gpu_web.h): post, then wait where an answer is needed (this thread is inside
  // the engine's WebAssembly call and cannot await)
  gpuBegin(lat, n) { return gpuCall(0, { op: 'begin', lat: new Float32Array(M.wasmMemory.buffer, lat, 64 * n).slice() }); },
  gpuDecode(lat, audio) { return gpuCall(0, { op: 'decode', lat: new Float32Array(M.wasmMemory.buffer, lat, 64).slice(), audio }); },
  gpuSubmit(lat, audio) {  // the frame to the GPU; this thread goes on (the next backbone step) until gpuWait
    Atomics.store(gpu.f, 0, 0);
    gpu.w.postMessage({ op: 'decode', lat: new Float32Array(M.wasmMemory.buffer, lat, 64).slice(), audio });
  },
  gpuWait() { Atomics.wait(gpu.f, 0, 0); return Atomics.load(gpu.f, 0) === 1 ? 1 : 0; },
  gpuSave(id) {  // only queued: the state goes into the next snapshot slot
    const slot = gpu.next++;
    gpu.slots.set(id, slot);
    if (gpu.slots.size > GPU_SNAPS) gpu.slots.delete(gpu.slots.keys().next().value);
    gpu.w.postMessage({ op: 'save', slot });
  },
  gpuLoad(id) {  // 0: that state's slot was reused since (the engine then rebuilds it on the CPU)
    const slot = gpu.slots.get(id);
    if (slot === undefined) return 0;
    gpu.w.postMessage({ op: 'load', slot });
    return 1;
  },
  gpuPrefill(emb, pos, n, base, hid, kv, S) {
    const e = new Float32Array(M.wasmMemory.buffer, emb, n * 1024).slice(), p = new Float32Array(M.wasmMemory.buffer, pos, n).slice();
    return gpuCall(1, { op: 'prefill', emb: e, pos: p, n, base, hid, kv, S }, [e.buffer, p.buffer]);
  },
  gpuPrefillStart(emb, pos, n, base, kv, S) {  // the next chunk's prefix, while this one goes on
    const e = new Float32Array(M.wasmMemory.buffer, emb, n * 1024).slice(), p = new Float32Array(M.wasmMemory.buffer, pos, n).slice();
    Atomics.store(gpu.f, 1, 0);
    gpu.w.postMessage({ op: 'prefill', emb: e, pos: p, n, base, hid: 0, kv, S }, [e.buffer, p.buffer]);
    return 1;
  },
  gpuPrefillWait() { Atomics.wait(gpu.f, 1, 0); return Atomics.load(gpu.f, 1) === 1 ? 1 : 0; },
  gpuPrefillReady() { return Atomics.load(gpu.f, 1) !== 0 ? 1 : 0; },
  // a file kept out of the heap, read when the engine needs it (the CPU decoder after a GPU failure)
  fileSize(name) { return lazy[name] ? lazy[name].size : -1; },
  fileRead(name, p, n) { return lazy[name] && readInto(lazy[name].h, p, n) === n ? 1 : 0; },
  result(p, n) {  // realtime off: the whole audio at the end
    const audio = new Float32Array(M.wasmMemory.buffer, p, n).slice();
    post({ type: 'result', audio }, [audio.buffer]);
  },
};

async function opfsDir(version) {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle('models-' + version, { create: true });
}

// One model file into the heap: cached copy if it is complete and was checked, else download (and cache).
function gpuCall(i, msg, transfer) {
  Atomics.store(gpu.f, i, 0);
  gpu.w.postMessage(msg, transfer || []);
  Atomics.wait(gpu.f, i, 0);
  return Atomics.load(gpu.f, i) === 1 ? 1 : 0;
}
// a cached file (sync access handle h) into the heap at p
function readInto(h, p, size) {
  let off = 0;
  try {
    off = h.read(heapU8(p, size), { at: 0 });  // straight into the heap
  } catch (e) {                                 // no shared views here: in pieces through a bounce buffer
    const tmp = new Uint8Array(8 << 20);
    while (off < size) {
      const n = h.read(tmp.subarray(0, Math.min(tmp.length, size - off)), { at: off });
      heapU8(p + off, n).set(tmp.subarray(0, n));
      off += n;
    }
  }
  return off;
}
// The GPU worker for option gpu: decoder from vv.bin (in the heap, or else from the cache), text reading from lm.bin
// in the heap. Resolves with its 'gpu-ready' message, or null (no usable WebGPU: everything stays on the CPU).
// What this device's GPU gave at its first use (per adapter: decoder variant and its timing, Auto's check), in the
// origin's private storage next to the model cache: { [adapter key]: { variant, variant_ms, check, ok } }
const DEVICE_FILE = 'gpu-device.json';
async function readDevices() {
  try { return JSON.parse(await (await (await (await navigator.storage.getDirectory()).getFileHandle(DEVICE_FILE)).getFile()).text()); }
  catch (e) { return {}; }
}
async function writeDevices(d) {
  try {
    const w = await (await (await navigator.storage.getDirectory()).getFileHandle(DEVICE_FILE, { create: true })).createWritable();
    await w.write(JSON.stringify(d));
    await w.close();
  } catch (e) { /* not fatal: measured again next time */ }
}
// the text reading on the GPU needs the language model a second time (~0.4 GiB of GPU memory): not on phones
function gpuTextDefault() {
  return !/Android|Mobile|iPhone|iPad/i.test(navigator.userAgent) && (navigator.deviceMemory || 8) >= 8;
}

async function startGpu(m, ptrs, version, known) {
  const w = new Worker(new URL('./gpu_worker.js', import.meta.url), { type: 'module' });
  const flags = new SharedArrayBuffer(8);
  const r = await new Promise((res) => {
    w.onmessage = (e) => { if (e.data.type === 'gpu-ready' || e.data.type === 'gpu-failed') res(e.data); };
    w.onerror = (e) => res({ type: 'gpu-failed', reason: e.message });
    w.postMessage({ cmd: 'init', memory: M.wasmMemory, flags, text: m.gpuText ?? gpuTextDefault(), lm: ptrs['lm.bin'], loseAt: m.gpuLoseAt,
                    vv: ptrs['vv.bin'] || { dir: 'models-' + version }, mode: m.gpu, known });
  });
  if (r.type !== 'gpu-ready') { w.terminate(); post({ type: 'log', s: 'GPU off: ' + r.reason }); return { failed: r.reason }; }
  w.onmessage = (e) => { if (e.data.type === 'gpu-lost') post({ type: 'log', s: 'GPU device lost: ' + e.data.message }); };
  gpu = { w, f: new Int32Array(flags), next: 0, slots: new Map(), info: r };
  return r;
}

async function loadFile(base, f, dir, progress, keepOut = false) {
  if (keepOut) {  // a cached, checked file stays out of the heap (opened for reading when needed); else loaded
    let ok = null;
    try { ok = await (await (await dir.getFileHandle(f.name + '.ok')).getFile()).text(); } catch (e) { /* none */ }
    const file = ok === f.sha256 ? await (await dir.getFileHandle(f.name)).getFile() : null;
    if (file && file.size === f.size) { progress(f, f.size, 'cache'); return { p: 0, ms: 0, from: 'cache, not loaded' }; }
  }
  const p = M._tts_alloc(f.size);
  if (!p) throw new Error('out of memory for ' + f.name);
  const fh = await dir.getFileHandle(f.name, { create: true });
  let okMarker = null;
  try { okMarker = await (await (await dir.getFileHandle(f.name + '.ok')).getFile()).text(); } catch (e) { /* none */ }
  const t0 = now();
  if (okMarker === f.sha256) {
    const h = await fh.createSyncAccessHandle();
    try {
      if (h.getSize() === f.size) {
        let off = 0;
        try {
          off = h.read(heapU8(p, f.size), { at: 0 });  // straight into the heap
        } catch (e) {                                     // no shared views here: in pieces through a bounce buffer
          const tmp = new Uint8Array(8 << 20);
          while (off < f.size) {
            const n = h.read(tmp.subarray(0, Math.min(tmp.length, f.size - off)), { at: off });
            heapU8(p + off, n).set(tmp.subarray(0, n));
            off += n;
          }
        }
        if (off === f.size) { progress(f, f.size, 'cache'); return { p, ms: now() - t0, from: 'cache' }; }
      }
    } finally { h.close(); }
  }
  // download, written to the heap and the cache as it arrives
  const res = await fetch(base + f.name, { cache: 'no-store' });
  if (!res.ok) throw new Error(`download of ${f.name} failed: HTTP ${res.status}`);
  const h = await fh.createSyncAccessHandle();
  let off = 0;
  try {
    h.truncate(0);
    const rd = res.body.getReader();
    for (;;) {
      const { done, value } = await rd.read();
      if (done) break;
      if (off + value.length > f.size) throw new Error(f.name + ' is larger than the manifest says');
      heapU8(p + off, value.length).set(value);
      h.write(value, { at: off });
      off += value.length;
      progress(f, off, 'download');
    }
    h.flush();
  } finally { h.close(); }
  if (off !== f.size) throw new Error(`${f.name}: ${off} of ${f.size} bytes`);
  const t1 = now();
  const sha = M.UTF8ToString(M._tts_sha256(p, f.size));
  if (sha !== f.sha256) {
    await dir.removeEntry(f.name);
    throw new Error(`${f.name}: checksum mismatch`);
  }
  const ok = await (await dir.getFileHandle(f.name + '.ok', { create: true })).createWritable();
  await ok.write(f.sha256);
  await ok.close();
  return { p, ms: now() - t0, from: 'download', sha_ms: now() - t1 };
}

async function init(m) {
  const t0 = now();
  // speed options for the CPU decoder (pipelining, int8 activations in the decoder) keep the engine on the CPU: the
  // GPU option is off with them (the reason in the loaded message)
  if (m.gpu === true) m.gpu = 'on';
  if (m.gpu !== 'on' && m.gpu !== 'auto') m.gpu = false;  // 'off', false, unset
  const cpuOnly = m.gpu && (m.cfg || '').match(/(^|\s)(pipe=|act8=\S*vv)/);
  if (cpuOnly) m.gpu = false;
  ring = new RingWriter(m.ring);
  ctrl = ring.c;
  const createTTS = (await import(`./${m.variant || 'tts'}.js`)).default;
  // engine options (tts_config): pipelined (pipe=A,B) the engine starts pools of A, B and A + B threads and the
  // decoder's thread, all while tts_load blocks this worker (no thread can be added then)
  const pipe = /pipe=(\d+),(\d+)/.exec(m.cfg || '');
  const pool = pipe ? 2 * (Number(pipe[1]) + Number(pipe[2])) : Math.max(1, m.threads - 1);
  M = await createTTS({ ttsPool: pool, ttsHooks: hooks, print: (s) => post({ type: 'log', s }), printErr: (s) => post({ type: 'log', s }) });
  M.ttsHooks = hooks;
  const t1 = now();
  // the model list (names, sizes, SHA-256): the one packaged with the extension, else the model source's
  let man;
  try { man = await (await fetch(new URL('./models.json', import.meta.url), { cache: 'no-store' })).json(); }
  catch (e) { man = await (await fetch(m.base + 'models.json', { cache: 'no-store' })).json(); }
  const files = man.files.filter((f) => f.name !== 'bwe.bin' || m.bwe);
  const total = files.reduce((a, f) => a + f.size, 0), got = {};
  let last = 0;
  const progress = (f, n, how) => {
    got[f.name] = n;
    const t = now();
    if (t - last > 200 || n === f.size) {
      last = t;
      const sum = Object.values(got).reduce((a, b) => a + b, 0);
      post({ type: 'progress', file: f.name, how, done: sum, total });
    }
  };
  const dir = await opfsDir(man.version);
  const info = {};
  const ptrs = {};
  for (const f of files) {
    const r = await loadFile(m.base, f, dir, progress, m.gpu && f.name === 'vv.bin');
    info[f.name] = { from: r.from, ms: Math.round(r.ms), sha_ms: r.sha_ms ? Math.round(r.sha_ms) : undefined };
    if (!r.p) continue;
    ptrs[f.name] = { p: r.p, n: f.size };
    M.ccall('tts_register', null, ['string', 'number', 'number'], [f.name, r.p, f.size]);
  }
  // a download (first install or new model files): one request for the model repository's root config.json, which is
  // what the hub counts as a download; not awaited, any answer or error ignored
  if (Object.values(info).some((x) => x.from === 'download')) {
    try { fetch(new URL('../config.json', m.base), { mode: 'cors', cache: 'no-store' }).catch(() => {}); } catch (e) { /* bad base */ }
  }
  const t2 = now();
  try {  // cached files of older model versions
    const root = await navigator.storage.getDirectory();
    for await (const [name] of root.entries())
      if (name.startsWith('models-') && name !== 'models-' + man.version) await root.removeEntry(name, { recursive: true });
  } catch (e) { /* not fatal */ }
  if (m.cfg) {
    const err = M.UTF8ToString(M.ccall('tts_config', 'number', ['string'], [m.cfg]));
    if (err) throw new Error(err);
  }
  let gpuInfo = cpuOnly ? { refused: 'speed option ' + cpuOnly[2] + ' (CPU decoder): the engine runs on the CPU' } : null;
  const devices = m.gpu ? await readDevices() : {};
  if (m.gpu) {
    gpuInfo = await startGpu(m, ptrs, man.version, devices);
    const f = files.find((x) => x.name === 'vv.bin');
    if (!ptrs['vv.bin'] && !gpu) {  // no GPU after all: the CPU decoder's file into the heap now
      const r = await loadFile(m.base, f, dir, progress);
      M.ccall('tts_register', null, ['string', 'number', 'number'], [f.name, r.p, f.size]);
    } else if (!ptrs['vv.bin']) {  // kept out: read from the cache only if the engine ever needs the CPU decoder
      lazy['vv.bin'] = { h: await (await dir.getFileHandle('vv.bin')).createSyncAccessHandle(), size: f.size };
      M._tts_lazy_vv(1);
    }
  }
  const msg = M.UTF8ToString(M._tts_load(m.threads, m.bwe ? 1 : 0));
  if (gpu) {
    M._tts_gpu(1, gpu.info.text ? 1 : 0);
    const key = gpu.info.key, d = devices[key] || {};
    d.variant = gpu.info.variant;
    if (gpu.info.variant_ms) d.variant_ms = gpu.info.variant_ms;
    if (m.gpu === 'auto' && !d.check) {  // Auto, first use on this device: the GPU decoder against the CPU decoder
      d.check = JSON.parse(M.UTF8ToString(M._tts_gpu_check(8)));
      d.ok = d.check.snr_db > 50 && d.check.gpu_ms < d.check.cpu_ms;
    }
    devices[key] = d;
    await writeDevices(devices);
    gpuInfo = { ...gpuInfo, check: d.check, ok: d.ok };
    if (m.gpu === 'auto' && !d.ok) {  // the CPU decoder (its file was loaded for the check) and the CPU text reading
      M._tts_gpu(0, 0);
      gpu.w.terminate();
      gpu = null;
      gpuInfo = { ...gpuInfo, failed: `Auto: the CPU (GPU check ${d.check.snr_db} dB, ${d.check.gpu_ms} vs ${d.check.cpu_ms} ms per frame on the CPU)` };
    }
  }
  if (gpuInfo) gpuInfo.using = gpu ? `GPU (decoder ${gpu.info.decoder}${gpu.info.variant === 'wt' ? ' wt' : ''}${gpu.info.text ? ', text reading' : ''})` : 'CPU';
  const t3 = now();
  post({ type: 'loaded', msg, threads: m.threads, variant: m.variant || 'tts', files: info, ms: { module: Math.round(t1 - t0), files: Math.round(t2 - t1), engine: Math.round(t3 - t2), total: Math.round(t3 - t0) },
         heapMiB: Math.round(M.wasmMemory.buffer.byteLength / 1048576), bwe: !!M._tts_rate(1) && M._tts_rate(1) === 48000, gpu: gpuInfo });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.cmd === 'init') await init(m);
    else if (m.cmd === 'speak') {
      const rate = M._tts_rate(m.bwe ? 1 : 0);
      kept = m.keep ? [] : null;
      keptLen = 0;
      text = m.text;
      post({ type: 'begin', rate, t: abs() });
      M.ccall('tts_speak', null, ['string', 'number', 'number', 'number', 'number', 'number', 'number', 'number'],
              [m.text, m.verbalize === false ? 0 : 1, m.bwe ? 1 : 0, m.seed ?? Math.floor(Math.random() * 2 ** 31), m.pace ?? 4,
               m.redraw === false ? 0 : 1, m.leadCut === false ? 0 : 1, m.realtime === false ? 0 : 1]);
      if (kept) {
        const all = new Float32Array(kept.reduce((a, x) => a + x.length, 0));
        let o = 0;
        for (const x of kept) { all.set(x, o); o += x.length; }
        post({ type: 'kept', audio: all }, [all.buffer]);
        kept = null;
      }
      Atomics.store(ctrl, C.DONE, 1);
      post({ type: 'end', t: now() });
    } else if (m.cmd === 'gpustatus') {  // the GPU decoder's state and fallback counters
      post({ type: 'gpustatus', status: gpu ? JSON.parse(M.UTF8ToString(M._tts_gpu_status())) : null });
    } else if (m.cmd === 'prepare') {
      const r = JSON.parse(M.UTF8ToString(M.ccall('tts_prepare', 'number', ['string', 'number'], [m.text, 1])));
      if (r.sources) r.sources = u16Ranges(m.text, r.sources);
      post({ type: 'prepared', result: r });
    }
  } catch (err) {
    post({ type: 'error', error: String(err && err.stack || err) });
  }
};
