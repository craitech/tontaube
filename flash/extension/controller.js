// Controller of one engine: the engine worker, the shared audio ring and the AudioWorklet player; builds the status
// line. Used by the extension's offscreen document.
import { C, makeRing } from './ring.js';

// 48 kHz (bwe): true or false is used as asked. 'auto' (first use, before the user has chosen): one silent offline
// request of about 2 s at 48 kHz measures this device's speed with the upsampler; 48 kHz when it is at least
// BWE_MIN_SPEED times real time, else 24 kHz. The host gets the choice through onBweDecided(on) and keeps it as the
// setting; this controller keeps it too, so 'auto' measures once.
const BWE_MIN_SPEED = 1.5;
const PROBE_TEXT = 'Hello, this is a quick speed check.';

export class Controller {
  constructor(onUpdate) {
    this.onUpdate = onUpdate || (() => {});
    this.ring = makeRing(30, 48000);
    this.ctrl = new Int32Array(this.ring.ctrl);
    this.worker = null;
    this.ctx = null;
    this.node = null;
    this.loaded = null;      // the worker's 'loaded' message
    this.st = { phase: 'idle' };
    this.req = null;         // the running request
    this.onChunk = null;     // see speak(): highlight
    this.onBweDecided = null;  // see BWE_MIN_SPEED
    this.bweAuto = null;       // the choice for 'auto', once made
  }

  update(patch) { Object.assign(this.st, patch); this.onUpdate(this.st); }

  // threads: engine threads (pool size); variant: tts (SIMD), tts_relaxed (+ fused multiply-add), tts_scalar;
  // cfg: engine options (tts_config, e.g. 'pipe=3,3 share=1'), none by default
  // gpu: 'auto' (the GPU where WebGPU is there and its first-use check on this device passed: the same output as the
  // CPU decoder and faster), 'on', 'off' (default); the decoder on the GPU, and the text reading too unless gpuText is
  // false (default: not on phones) (gpuLoseAt: tests, the GPU device is destroyed during that decode)
  async init({ base, threads, bwe = false, variant, cfg = '', gpu = 'off', gpuText = null, gpuLoseAt = 0 } = {}) {
    if (this.loaded) return this.loaded;
    if (this.initP) return this.initP;
    threads = threads || Controller.defaultThreads();
    bwe = !!bwe;  // 'auto': with the upsampler
    this.args = { base, threads, bwe, variant, cfg, gpu, gpuText };
    this.worker = new Worker(new URL('./engine_worker.js', import.meta.url), { type: 'module' });
    this.initP = new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => this.onWorker(e.data, resolve, reject);
      this.worker.onerror = (e) => reject(new Error(e.message));
    });
    this.update({ phase: 'loading', progress: null, error: null, threads });
    this.worker.postMessage({ cmd: 'init', base, threads, bwe, variant: variant || Controller.defaultVariant(), cfg, ring: this.ring, gpu,
                              gpuText, gpuLoseAt });
    return this.initP;
  }

  // The fastest build this browser can run: relaxed SIMD (fused multiply-add) where available, else plain SIMD.
  static defaultVariant() {
    // modules of one function: f32x4.relaxed_madd (relaxed SIMD), f32x4.add (SIMD)
    const relaxed = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 19,
      1, 17, 0, 65, 0, 253, 17, 65, 0, 253, 17, 65, 0, 253, 17, 253, 133, 2, 11]));
    const simd = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1,
      13, 0, 65, 0, 253, 17, 65, 0, 253, 17, 253, 228, 1, 11]));
    return relaxed ? 'tts_relaxed' : simd ? 'tts' : 'tts_scalar';
  }

  // Engine threads from the logical core count (which counts both threads of a core): about one per physical core,
  // 4 to 8 (the backbone and the decoder's wide layers are memory-bound beyond that); small machines keep one free.
  static defaultThreads() {
    const n = navigator.hardwareConcurrency || 4;
    return n >= 8 ? Math.max(4, Math.min(8, Math.floor(n / 2))) : Math.max(1, n - 1);
  }

  onWorker(m, resolve, reject) {
    if (m.type === 'progress') this.update({ progress: m });
    else if (m.type === 'loaded') { this.loaded = m; this.update({ phase: 'ready', load: m, progress: null }); resolve(m); }
    else if (m.type === 'error') { this.update({ phase: 'error', error: m.error }); if (!this.loaded) reject(new Error(m.error)); this.finish('error'); }
    else if (m.type === 'begin') { if (this.req) this.req.tBegin = m.t - performance.timeOrigin; }  // page clock
    else if (m.type === 'status') { if (this.req) { this.req.engine = m.status; if (this.req.tFirst == null && m.status.first_audio_ms >= 0) this.req.tFirst = m.status.first_audio_ms; this.refresh(); } }
    else if (m.type === 'end') { if (this.req) { this.req.tEnd = m.t; this.req.genDone = true; this.refresh(); this.maybeDone(); } }
    else if (m.type === 'sources') { if (this.req) this.req.sources = m.ranges; }
    else if (m.type === 'chunk') { if (this.req) this.req.starts[m.c] = m.at; }
    else if (m.type === 'prepared') { if (this.prepRes) this.prepRes(m.result); }
    else if (m.type === 'result') { if (this.req) this.req.audio = m.audio; }
    else if (m.type === 'kept') { this.kept = m.audio; }
    else if (m.type === 'gpustatus') { if (this.gpuRes) this.gpuRes(m.status); }
    else if (m.type === 'log') console.log('[engine]', m.s);
  }

  async audio(rate) {
    if (this.ctx && this.ctx.sampleRate === rate) { if (this.ctx.state !== 'running') await this.ctx.resume(); return; }
    if (this.ctx) await this.ctx.close();
    this.ctx = new AudioContext({ sampleRate: rate, latencyHint: 'playback' });
    await this.ctx.audioWorklet.addModule(new URL('./player.js', import.meta.url));
    this.node = new AudioWorkletNode(this.ctx, 'ring-player', { outputChannelCount: [1], processorOptions: this.ring });
    this.node.port.onmessage = (e) => {
      if (!this.req) return;
      if (e.data.started) { this.req.tPlay = performance.now(); this.refresh(); }
      if (e.data.ended) { this.req.played = true; this.maybeDone(); }
      if (e.data.gap != null) this.req.gapAt.push(e.data.gap / this.req.rate);
    };
    this.node.connect(this.ctx.destination);
    if (this.ctx.state !== 'running') await this.ctx.resume();
  }

  // Speak text: returns a promise that resolves with the final status when playback has ended (or was stopped).
  // keep: also collect every sample that is played, redraws applied (controller.kept after the end); redraw: false turns
  // the redraw of flagged spots off (on by default)
  // highlight: a tag; while it plays, onChunk gets { tag, index, start, end, len } whenever the chunk being heard
  // changes (start, end: its characters in text; len: text.length) and { tag, index: -1 } when the request ends
  // leadCut: start at the request's first sound (default; false: from the first sample). realtime: play as it is made
  // (default); false: nothing is played, every flagged spot gets its redraws (no playback budget), and the promise
  // resolves with the whole audio (status.audio, Float32Array at status.rate)
  async speak(text, { delayMs = 0, bwe = false, verbalize = true, seed, pace = null, keep = false, highlight = null, redraw = true,
                      leadCut = true, realtime = true, quiet = false } = {}) {
    await this.init({ bwe });
    await this.stop();
    // the upsampler is loaded with the engine: when 48 kHz is first asked for, load the engine again with it
    if (bwe && !this.loaded.bwe && !this.args.bwe) {
      this.worker.terminate();
      this.loaded = this.initP = null;
      await this.init({ ...this.args, bwe: true });
    }
    if (bwe === 'auto') bwe = await this.bweChoose();
    const rate = bwe && this.loaded.bwe ? 48000 : 24000;
    // the upsampler looks ~52 ms ahead: the first frame gives only ~82 ms of audio, so start a little later
    if (rate === 48000) delayMs = Math.max(delayMs, 150);
    if (realtime) await this.audio(rate);
    const c = this.ctrl;
    for (const k of [C.W, C.R, C.GAPS, C.GAP_SAMPLES, C.CANCEL, C.DONE, C.PLAYED, C.REPLACED]) Atomics.store(c, k, 0);
    Atomics.store(c, C.RATE, rate);
    Atomics.store(c, C.DELAY, Math.round(delayMs * rate / 1000));
    Atomics.store(c, C.STATE, realtime ? 1 : 0);
    const t0 = performance.now();
    this.req = { t0, rate, text, delayMs, engine: null, tPlay: null, tFirst: null, genDone: false, played: false,
                 tag: highlight, sources: null, starts: [], cur: -1, realtime, audio: null, gapAt: [], quiet };
    this.req.done = new Promise((res) => { this.req.resolve = res; });
    if (realtime) this.req.timer = setInterval(() => this.follow(), 100);
    if (!quiet) this.update({ phase: 'speaking' });
    // pacing: generate until ~4 s are queued (more with a long start delay), then rest until ~2 s are left
    const hi = pace ?? Math.max(4, delayMs / 1000 + 1);
    this.worker.postMessage({ cmd: 'speak', text, bwe: rate === 48000, verbalize, seed, pace: hi, keep, redraw, leadCut, realtime });
    return this.req.done;
  }

  // 'auto' (see BWE_MIN_SPEED): measured once; without a measure (stopped, error) 24 kHz for this read, measured again
  // at the next
  async bweChoose() {
    if (this.bweAuto != null) return this.bweAuto;
    const t0 = performance.now();
    const s = await this.speak(PROBE_TEXT, { bwe: true, realtime: false, redraw: false, leadCut: false, quiet: true });
    this.probe = { speed: s.speed, ms: Math.round(performance.now() - t0) };
    if (s.state !== 'done' || !(s.speed > 0)) return false;
    this.bweAuto = s.speed >= BWE_MIN_SPEED;
    if (this.onBweDecided) try { this.onBweDecided(this.bweAuto); } catch (e) { /* the host's */ }
    return this.bweAuto;
  }

  // the GPU decoder's state and fallback counters (null: GPU off)
  gpuStatus() { return new Promise((res) => { this.gpuRes = res; this.worker.postMessage({ cmd: 'gpustatus' }); }); }

  maybeDone() {
    const r = this.req;
    if (!r || !r.genDone) return;
    if (r.played || Atomics.load(this.ctrl, C.STATE) === 0) this.finish(r.engine && r.engine.state === 'stopped' ? 'stopped' : 'done');
  }

  // The chunk being heard: samples played in this request (counted from the start of playback, so after the start
  // delay; at the ring's rate) less the audio output's latency, against the samples where the chunks start.
  follow() {
    const r = this.req;
    if (!r || !r.sources) return;
    const played = Atomics.load(this.ctrl, C.PLAYED);
    if (!played) return;
    const lat = this.ctx ? Math.min(0.5, (this.ctx.baseLatency || 0) + (this.ctx.outputLatency || 0)) : 0;
    const pos = Math.max(0, played - lat * r.rate);
    let c = Math.max(0, r.cur);
    while (r.starts[c + 1] != null && r.starts[c + 1] <= pos) c++;
    if (c === r.cur || !r.sources[c]) return;
    r.cur = c;
    this.refresh();
    if (r.tag && this.onChunk) this.onChunk({ tag: r.tag, index: c, start: r.sources[c][0], end: r.sources[c][1], len: r.text.length });
  }

  finish(state) {
    const r = this.req;
    if (!r) return;
    this.req = null;
    clearInterval(r.timer);
    if (r.tag && this.onChunk) this.onChunk({ tag: r.tag, index: -1 });
    const s = this.line(r, state);
    if (!r.quiet) this.update({ phase: this.loaded ? 'ready' : 'error', last: s });
    r.resolve(r.realtime ? s : { ...s, audio: r.audio });
    // the audio thread renders silence while the context runs: suspend it when nothing follows soon
    setTimeout(() => { if (!this.req && this.ctx && this.ctx.state === 'running') this.ctx.suspend(); }, 3000);
  }

  async stop() {
    if (!this.req) return;
    const r = this.req;
    Atomics.store(this.ctrl, C.CANCEL, 1);
    Atomics.store(this.ctrl, C.STATE, 0);
    Atomics.store(this.ctrl, C.R, Atomics.load(this.ctrl, C.W));
    await new Promise((res) => {  // the worker ends the request at its next frame
      const t = setInterval(() => { if (r.genDone) { clearInterval(t); res(); } }, 10);
    });
    if (this.req === r) this.finish('stopped');
  }

  refresh() { if (this.req && !this.req.quiet) this.update({ now: this.line(this.req, 'speaking') }); }

  // The status line: what was read, time to first audio, speed vs real time, gaps, per-frame costs.
  line(r, state) {
    const e = r.engine || {}, c = this.ctrl;
    const gaps = Atomics.load(c, C.GAPS), gapMs = 1000 * Atomics.load(c, C.GAP_SAMPLES) / r.rate;
    const s = {
      state, chunks: e.chunks, chunk: r.cur + 1, chars: e.chars, unknown: e.unknown || '',
      // from the request: engine start (page clock) + the engine's own time to the first audio
      first_audio_ms: e.first_audio_ms >= 0 && r.tBegin != null ? Math.round(r.tBegin - r.t0 + e.first_audio_ms) : e.first_audio_ms,
      playback_ms: r.tPlay ? Math.round(r.tPlay - r.t0) : null,
      audio_s: e.audio_s, speed: e.speed, gaps, gap_ms: Math.round(gapMs), gap_at: r.gapAt.map((t) => +t.toFixed(2)),
      verbalize_ms: e.verbalize_ms, prefill_ms: e.prefill_ms, lm_ms: e.lm_ms, head_ms: e.head_ms, ae_ms: e.ae_ms,
      vv_ms: e.vv_ms, bwe_ms: e.bwe_ms, threads: e.threads, rate: r.rate, paused_s: e.paused_s, pauses: e.pauses,
      det_ms: e.det_ms, redraw: e.redraw, spots: e.spots, attempts: e.attempts, clean: e.clean, kept_best: e.kept_best,
      budget: e.budget, previous_chunk: e.previous_chunk, extra_frames: e.extra_frames, early_stop: e.early_stop,
      played_stop: e.played_stop, refused: e.refused, replaced: e.replaced, player_refused: e.player_refused, late: e.late,
      replaced_ring: Atomics.load(c, C.REPLACED), cut_ms: e.cut_ms, first_sound_ms: e.first_sound_ms,
    };
    s.text = `${state}: ${s.chunks ?? '?'} chunk${s.chunks === 1 ? '' : 's'}, ${s.chars ?? '?'} characters` +
      (s.unknown ? `, not known (read as spaces): ${s.unknown}` : '') +
      `\nfirst audio after ${s.first_audio_ms ?? '-'} ms, playback started after ${s.playback_ms ?? '-'} ms` +
      `\n${(s.audio_s ?? 0).toFixed(1)} s of audio, ${(s.speed ?? 0).toFixed(2)}x real time, gaps ${gaps} (${s.gap_ms} ms missing` +
      `${s.gap_at.length ? `; at ${s.gap_at.map((t) => t.toFixed(1)).join(', ')} s` : ''})` +
      `\ntext read in ${s.prefill_ms ?? '-'} ms; per frame: model ${s.lm_ms ?? '-'}, head ${s.head_ms ?? '-'}, ` +
      `autoencoder ${s.ae_ms ?? '-'}, decoder ${s.vv_ms ?? '-'}${r.rate === 48000 ? ` + upsampler ${s.bwe_ms}` : ''}, detector ${s.det_ms ?? '-'} ms; ${s.threads} threads` +
      (s.redraw ? `\nflagged spots ${s.spots ?? 0}, redrawn ${s.replaced ?? 0}` : '');
    return s;
  }

  prepare(text) {
    return new Promise((res) => { this.prepRes = res; this.worker.postMessage({ cmd: 'prepare', text }); });
  }
}
