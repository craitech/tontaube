// Offscreen document: holds the engine (worker + threads) and the audio player while the extension runs; answers
// the popup's status requests and broadcasts every status change.
import { Controller } from './controller.js';

// Idle for this long: ask the service worker to close this document (frees the engine's memory; the next request
// loads it again from the browser's cache).
const IDLE_MS = 5 * 60 * 1000;
let idle = null;
const ctl = new Controller((st) => {
  chrome.runtime.sendMessage({ to: 'popup', status: st }).catch(() => {});
  clearTimeout(idle);
  if (st.phase === 'ready' || st.phase === 'error')
    idle = setTimeout(() => chrome.runtime.sendMessage({ to: 'sw', cmd: 'close-offscreen' }).catch(() => {}), IDLE_MS);
});

// "48 kHz" chosen at first use (setting 'auto'): kept as the setting in chrome.storage by the service worker (this
// document cannot reach it)
ctl.onBweDecided = (on) => chrome.runtime.sendMessage({ to: 'sw', cmd: 'bwe-decided', on }).catch(() => {});

// the chunk being heard, for the highlight in the reading tab (this document cannot reach tabs: the service worker can)
ctl.onChunk = (h) => chrome.runtime.sendMessage({ ...h, to: 'sw', cmd: 'highlight' }).catch(() => {});

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.to !== 'offscreen') return false;
  (async () => {
    if (msg.cmd === 'status') return { status: ctl.st, isolated: self.crossOriginIsolated };
    if (msg.cmd === 'init') { ctl.init(msg.init).catch(() => {}); return { ok: true }; }
    if (msg.cmd === 'stop') { await ctl.stop(); return { ok: true }; }
    if (msg.cmd === 'speak') {
      ctl.init(msg.init).then(() => ctl.speak(msg.text, msg.opts)).catch((e) => ctl.update({ phase: 'error', error: String(e) }));
      return { ok: true };
    }
    return { error: 'unknown command' };
  })().then(reply, (e) => reply({ error: String(e) }));
  return true;
});
self.ctl = ctl;
