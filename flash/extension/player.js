// AudioWorklet: plays the engine's samples from a SharedArrayBuffer ring as they arrive. Playback starts `delay`
// samples after the first sample came in; once playing, every render quantum short of samples (before the engine
// said it is done) is a gap, and its start (samples played) is posted. Layout of the control words: see ring.js.
import { C } from './ring.js';

class RingPlayer extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { ctrl, data } = options.processorOptions;
    this.c = new Int32Array(ctrl);
    this.d = new Float32Array(data);
    this.cap = this.d.length;
    this.firstFrame = -1;
    this.inGap = false;
  }
  process(inputs, outputs) {
    const out = outputs[0][0], n = out.length, c = this.c;
    out.fill(0);
    const state = Atomics.load(c, C.STATE);
    if (state === 0) { this.firstFrame = -1; this.inGap = false; return true; }
    const w = Atomics.load(c, C.W);
    let r = Atomics.load(c, C.R);
    if (state === 1) {  // armed: wait for the first sample, then the delay
      if (w === r) { this.firstFrame = -1; this.inGap = false; return true; }  // nothing yet (also: a new request right after the last)
      if (this.firstFrame < 0) this.firstFrame = currentFrame;
      if (currentFrame - this.firstFrame < Atomics.load(c, C.DELAY)) return true;
      Atomics.store(c, C.STATE, 2);
      this.port.postMessage({ started: true, frame: currentFrame });
    }
    const avail = w - r, k = Math.min(n, avail);
    for (let i = 0; i < k; i++) out[i] = this.d[(r + i) % this.cap];
    r += k;
    Atomics.store(c, C.R, r);
    Atomics.add(c, C.PLAYED, k);
    if (k < n) {
      if (Atomics.load(c, C.DONE)) {  // the engine is done and everything is played
        Atomics.store(c, C.STATE, 0);
        this.port.postMessage({ ended: true });
      } else {
        Atomics.add(c, C.GAP_SAMPLES, n - k);
        if (!this.inGap) { Atomics.add(c, C.GAPS, 1); this.port.postMessage({ gap: r }); }
        this.inGap = true;
      }
    } else {
      this.inGap = false;
    }
    Atomics.notify(c, C.R);
    return true;
  }
}
registerProcessor('ring-player', RingPlayer);
