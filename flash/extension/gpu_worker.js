// GPU worker (option gpu): the VibeVoice decoder and the text reading of chunk prefixes on WebGPU, for the engine
// worker, which runs a request inside one WebAssembly call and so cannot await anything itself. It posts each GPU
// call here and waits on a flag of a SharedArrayBuffer (flags[0]: decoder calls, flags[1]: text reading; 1 done,
// 2 failed); audio, cache rows and hidden states are written straight into the engine's shared heap. The device is
// made with the default limits and only optional shader-f16 (phones); a failed call (device lost) answers 2 and the
// engine goes on on the CPU.
import { VVGpu } from './vv_gpu.js';
import { LMGpu } from './lm_gpu.js';

const SNAPS = 32;  // decoder state snapshots for the redraw (it goes back a few states; an older one is rebuilt on the CPU)
let dev = null, dec = null, lm = null, mem = null, flags = null, snaps = [], lost = false, decodes = 0, loseAt = 0;

async function init(m) {
  if (!navigator.gpu) throw new Error('no WebGPU');
  let ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!ad) {  // none once in a while (the GPU process starting up): one more try after a short wait
    await new Promise((r) => setTimeout(r, 500));
    ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  }
  if (!ad) throw new Error('no WebGPU adapter');
  const info = ad.info || {};
  const key = [info.vendor, info.architecture, info.device, info.description].join('|');
  // a software GPU (SwiftShader, llvmpipe, lavapipe, WARP) runs on the processor, slower than the CPU engine: never used
  if (info.isFallbackAdapter || ad.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|softpipe|basic render/i.test(key))
    throw new Error(`only a software GPU (${[info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'fallback adapter'})`);
  const known = (m.known || {})[key];
  if (m.mode === 'auto' && known && known.ok === false) throw new Error(`Auto: the CPU (this GPU's check: ${known.check && known.check.snr_db} dB, ${known.check && known.check.gpu_ms} vs ${known.check && known.check.cpu_ms} ms per frame)`);
  dev = await ad.requestDevice({ requiredFeatures: ad.features.has('shader-f16') ? ['shader-f16'] : [] });
  dev.lost.then((i) => { lost = true; self.postMessage({ type: 'gpu-lost', reason: i.reason, message: i.message }); });
  mem = m.memory;
  flags = new Int32Array(m.flags);
  loseAt = m.loseAt || 0;  // tests: the device is destroyed during this decode (the CPU fallback mid-request)
  const mode = dev.features.has('shader-f16') ? 'f16' : 'p16';  // p16: float16 storage without shader-f16
  dev.pushErrorScope('validation');
  // vv.bin from the heap when it is there, else from the model cache (it is then kept out of the heap)
  const vv = m.vv.p ? new Uint8Array(mem.buffer, m.vv.p, m.vv.n)
    : await (await (await (await navigator.storage.getDirectory()).getDirectoryHandle(m.vv.dir)).getFileHandle('vv.bin')).getFile().then((f) => f.arrayBuffer());
  // the decoder variant of this device: as chosen before, else the faster of plain and transposed weights (wt: the
  // gemm layers' weights [unit][N], faster on Mali) by a short timing; both give the same numbers
  let variant = known && known.variant, variant_ms = null;
  if (variant !== 'plain' && variant !== 'wt') {
    variant_ms = {};
    let ref = null;
    for (const v of ['plain', 'wt']) {
      const d = new VVGpu(dev, vv, { mode, wt: v === 'wt' });
      const r = await timeDecoder(d);
      variant_ms[v] = r.ms;
      if (ref) variant_ms.max_abs_diff = Math.max(...r.audio.map((x, i) => Math.abs(x - ref[i])));
      else ref = r.audio;
      d.destroy();
    }
    variant = variant_ms.wt < variant_ms.plain ? 'wt' : 'plain';
  }
  dec = new VVGpu(dev, vv, { mode, wt: variant === 'wt' });
  snaps = Array.from({ length: SNAPS }, () => dev.createBuffer({ size: 4 * dec.stateSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }));
  if (m.text) lm = new LMGpu(dev, new Uint8Array(mem.buffer, m.lm.p, m.lm.n), { S: 1024, f16: mode === 'f16' });
  await dev.queue.onSubmittedWorkDone();
  const err = await dev.popErrorScope();
  if (err) throw new Error(err.message);
  const a = await dec.decode(new Float32Array(64));  // one frame from a zero state: does it compute at all
  if (!a.every(Number.isFinite)) throw new Error('decoder check failed');
  dec.reset();
  return { key, variant, variant_ms, decoder: mode, text: lm ? mode : null, adapter: `${info.vendor || ''} ${info.architecture || ''}`.trim(),
           gpu_MiB: Math.round((dec.bytes + SNAPS * 4 * dec.stateSize + (lm ? lm.bytes : 0)) / 2 ** 20) };
}

// ms per frame of a decoder one frame at a time (with read back), after two frames to warm up; the last frame's audio
async function timeDecoder(d) {
  const lat = new Float32Array(64).map((_, i) => Math.sin(i * 0.7));
  let audio = null;
  for (let k = 0; k < 2; k++) await d.decode(lat);
  const t0 = performance.now();
  for (let k = 0; k < 6; k++) audio = await d.decode(lat);
  return { ms: +((performance.now() - t0) / 6).toFixed(2), audio };
}

async function call(m) {
  const i = m.op === 'prefill' ? 1 : 0;
  try {
    if (lost) throw new Error('GPU device lost');
    if (m.op === 'decode') {
      const p = dec.decode(m.lat);
      if (++decodes === loseAt) dev.destroy();
      const a = await p;
      new Float32Array(mem.buffer, m.audio, 3200).set(a);
    } else if (m.op === 'save') {
      dec.saveState(snaps[m.slot % SNAPS]);
      return;  // queued only: nobody waits for it
    } else if (m.op === 'load') {
      dec.loadState(snaps[m.slot % SNAPS]);
      return;
    } else if (m.op === 'begin') {  // the start state: the priming latents decoded, audio dropped
      dec.reset();
      for (let k = 0; k < m.lat.length / 64; k++) dec.decode(m.lat.subarray(64 * k, 64 * k + 64), { read: false });
    } else if (m.op === 'prefill') {
      // a chunk prefix (it attends only to its own rows): on the GPU at rows 0..n-1, its cache rows land at base..
      // of the engine's cache, the last hidden state in hid (if asked)
      // (read back in layer groups while the later layers still compute)
      if (!lm || m.n > lm.o.S || m.base + m.n > m.S) throw new Error('does not fit');
      const H16 = new Uint16Array(mem.buffer), per = m.n * 128, kvl = 2 * lm.nkv;
      await lm.feedRead(m.emb, m.pos, !!m.hid, (kv, l0, l1, hid) => {
        for (let k = 0; k < (l1 - l0) * kvl; k++) H16.set(kv.subarray(k * per, (k + 1) * per), (m.kv >> 1) + ((l0 * kvl + k) * m.S + m.base) * 128);
        if (hid) new Float32Array(mem.buffer, m.hid, 1024).set(hid);
      });
    }
    Atomics.store(flags, i, 1);
  } catch (e) {
    if (m.op === 'save' || m.op === 'load') return;
    Atomics.store(flags, i, 2);
  }
  Atomics.notify(flags, i);
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.cmd === 'init') {
    try { self.postMessage({ type: 'gpu-ready', ...(await init(m)) }); }
    catch (err) { self.postMessage({ type: 'gpu-failed', reason: String(err && err.message || err) }); }
  } else call(m);
};
