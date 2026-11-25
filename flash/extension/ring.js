// The shared audio ring between the engine worker (writer) and the AudioWorklet (reader).
// ctrl: Int32Array words; data: Float32Array samples (positions are sample counts modulo its length).
export const C = {
  W: 0,            // samples written
  R: 1,            // samples played: the player's read position, counted from the start of the request
  STATE: 2,        // 0 idle, 1 armed (waiting for the first sample + delay), 2 playing
  DELAY: 3,        // start delay in samples
  GAPS: 4,         // gaps after playback started
  GAP_SAMPLES: 5,  // samples missing in those gaps
  CANCEL: 6,       // 1: stop (the engine checks it once per frame)
  DONE: 7,         // 1: the engine has written its last sample
  RATE: 8,         // sample rate of the samples in the ring
  PLAYED: 9,       // samples played in this request
  REPLACED: 10,    // replacements of written, not yet played audio
  WORDS: 16,
};

export function makeRing(seconds = 30, rate = 48000) {
  return {
    ctrl: new SharedArrayBuffer(C.WORDS * 4),
    data: new SharedArrayBuffer(seconds * rate * 4),
  };
}

// The writer's side: append samples, or replace the written samples from a position on that the player has not
// read yet (the redraw of a flagged spot). The player only ever reads below W and advances R by one render quantum
// (128 samples) at a time, so audio at least `guard` samples past R can be rewritten safely.
export class RingWriter {
  constructor(ring) {
    this.c = new Int32Array(ring.ctrl);
    this.d = new Float32Array(ring.data);
    this.cap = this.d.length;
  }
  played() { return Atomics.load(this.c, C.R); }
  written() { return Atomics.load(this.c, C.W); }
  buffered() { return (Atomics.load(this.c, C.W) - Atomics.load(this.c, C.R)) / Atomics.load(this.c, C.RATE); }
  cancelled() { return Atomics.load(this.c, C.CANCEL) !== 0; }
  // src at ring position `at` (the ring full: wait for the player); false if stopped meanwhile
  put(at, src) {
    const c = this.c, n = src.length;
    while (at + n - Atomics.load(c, C.R) > this.cap) {
      if (Atomics.load(c, C.CANCEL)) return false;
      Atomics.wait(c, C.R, Atomics.load(c, C.R), 50);
    }
    const o = at % this.cap, k = Math.min(n, this.cap - o);
    this.d.set(src.subarray(0, k), o);
    if (k < n) this.d.set(src.subarray(k), 0);
    return true;
  }
  push(src) {
    const w = Atomics.load(this.c, C.W);
    if (this.put(w, src)) Atomics.store(this.c, C.W, w + src.length);
  }
  // The audio from sample `at` on becomes src (also shorter or longer than what it replaces). Returns 1 if done,
  // 0 if refused (`at` is not `guard` samples ahead of the read position, or past the written end), 2 if done but
  // the player passed `at` while it was written (must not happen with a sane guard; counted by the caller).
  replace(at, src, guard) {
    const c = this.c;
    if (at > Atomics.load(c, C.W) || at < Atomics.load(c, C.R) + guard) return 0;
    if (!this.put(at, src)) return 0;
    Atomics.store(c, C.W, at + src.length);
    Atomics.add(c, C.REPLACED, 1);
    return Atomics.load(c, C.R) > at ? 2 : 1;
  }
}
