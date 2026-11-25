// Service worker (short-lived): routes the popup's and the keyboard commands' requests. It reads the tab with the
// extractor (activeTab: only after the user clicked the action or pressed a command) and hands the text to the
// offscreen document, which holds the engine and plays the audio. With "highlight" on, it marks in the tab the part
// of the page whose chunk is being heard (the offscreen document reports it; it cannot reach tabs itself).
import { extractText } from './extract.js';

const DEFAULTS = { delayMs: 0, bwe: 'auto', redraw: true, threads: 0, gpu: 'auto', highlight: false, modelBase: 'https://huggingface.co/TontaubeAI/TontaubeV1.1-Flash/resolve/v1.1.0/web/' };

async function settings() { return { ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) }; }

async function ensureOffscreen() {
  const have = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
  if (have.length) return;
  await chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: ['AUDIO_PLAYBACK', 'WORKERS'],
    justification: 'Runs the voice model (WebAssembly workers) and plays its audio.',
  });
}

const toOffscreen = (msg) => chrome.runtime.sendMessage({ ...msg, to: 'offscreen' });

const NOTES = {
  pdf: 'This tab shows a PDF in Chrome\'s viewer, which extensions cannot read. Copy the text into a web page or open the PDF in another viewer.',
  canvas: 'This page draws most of its text on a canvas; only the little text in the page itself was found.',
  iframes: 'Text inside embedded frames is not read.',
};

// ---- highlight: a CSS custom highlight (the page's DOM is not changed), readable on light and dark pages
const HL_CSS = '::highlight(tontaube-reading) { background-color: rgba(255, 186, 0, 0.42); }';

// Runs in the tab, in the extension's isolated world next to extract.js's __rvExtract (self-contained): highlights
// where the characters [start, end) of the extracted text came from, or clears (start < 0). Scrolls (smoothly) only
// when the new part is out of view and the previous one was in view.
function paint(start, end, len) {
  const NAME = 'tontaube-reading', H = globalThis.CSS && CSS.highlights, X = globalThis.__rvExtract;
  if (!H) return -1;
  if (start < 0) { H.delete(NAME); return 0; }
  if (!X || X.text.length !== len) return -1;  // the tab was read again since
  const prev = H.get(NAME), rs = X.rangesFor(start, end);
  const boxes = (list) => list.map((r) => r.getBoundingClientRect()).filter((b) => b.width || b.height);
  const seen = prev && boxes([...prev]).some((b) => b.bottom > 0 && b.top < innerHeight);
  H.set(NAME, new Highlight(...rs));
  const now = boxes(rs);
  if (!seen || !now.length) return rs.length;
  const top = Math.min(...now.map((b) => b.top)), bottom = Math.max(...now.map((b) => b.bottom));
  if (top >= 0 && bottom <= innerHeight) return rs.length;
  const behavior = matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
  let el = rs[0].startContainer;
  if (el.nodeType !== 1) el = el.parentElement;
  for (; el && el !== document.body && el !== document.documentElement; el = el.parentElement) {  // an inner scroller?
    const o = getComputedStyle(el).overflowY;
    if ((o === 'auto' || o === 'scroll') && el.scrollHeight > el.clientHeight) break;
  }
  if (el && el !== document.body && el !== document.documentElement)
    el.scrollBy({ top: top - el.getBoundingClientRect().top - el.clientHeight / 4, behavior });
  else scrollBy({ top: top - innerHeight / 4, behavior });
  return rs.length;
}

// One at a time, in the order the offscreen document sent them. A failed injection (tab closed or navigated: the
// activeTab grant ends there) stops the highlighting of that reading quietly; the reading goes on.
let hlQueue = Promise.resolve(), hlDead = null;
function highlight(m) {
  hlQueue = hlQueue.then(async () => {
    const clear = m.index < 0;
    if (!m.tag || hlDead === m.tag.rid) return;
    if (!clear) {
      const { reading } = await chrome.storage.session.get('reading');
      if (!reading || reading.rid !== m.tag.rid || !(await settings()).highlight) return;
    }
    try {
      await chrome.scripting.executeScript({ target: { tabId: m.tag.tabId }, func: paint, args: clear ? [-1, -1, 0] : [m.start, m.end, m.len] });
    } catch (e) { hlDead = m.tag.rid; }
  }).catch(() => {});
}

// the option switched off during a reading: clear now
chrome.storage.onChanged.addListener(async (ch, area) => {
  if (area !== 'local' || !ch.highlight || ch.highlight.newValue) return;
  const { reading } = await chrome.storage.session.get('reading');
  if (reading) highlight({ tag: reading, index: -1 });
});

async function readTab(mode) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return { error: 'no active tab' };
  let res;
  try {
    [{ result: res }] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: extractText });
  } catch (e) {
    return { error: 'This page cannot be read by extensions (browser pages, the Chrome Web Store, other extensions\' pages).' };
  }
  const notes = (res.notes || []).map((n) => NOTES[n]).filter(Boolean);
  if (mode === 'selection' && res.source !== 'selection') return { error: 'Nothing is selected.', notes };
  if (!res.text) return { error: 'No readable text found on this page.', notes };
  const s = await settings();
  await ensureOffscreen();
  let hl = null;
  if (s.highlight) {
    hl = { tabId: tab.id, rid: crypto.randomUUID() };
    const css = { target: { tabId: tab.id }, css: HL_CSS };
    await chrome.scripting.removeCSS(css).catch(() => {});
    await chrome.scripting.insertCSS(css).catch(() => {});
  }
  await chrome.storage.session.set({ readingTab: tab.id, reading: hl });
  toOffscreen({ cmd: 'speak', text: res.text, opts: { delayMs: s.delayMs, bwe: s.bwe, redraw: s.redraw, highlight: hl }, init: { base: s.modelBase, threads: s.threads || undefined, bwe: s.bwe, gpu: s.gpu } });
  return { ok: true, source: res.source, chars: res.text.length, title: res.title, notes };
}

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.to !== 'sw') return false;
  (async () => {
    if (msg.cmd === 'read') return readTab(msg.mode);
    if (msg.cmd === 'stop') { try { await toOffscreen({ cmd: 'stop' }); } catch (e) { /* not running */ } return { ok: true }; }
    if (msg.cmd === 'highlight') { highlight(msg); return { ok: true }; }
    if (msg.cmd === 'close-offscreen') { await chrome.offscreen.closeDocument().catch(() => {}); return { ok: true }; }
    if (msg.cmd === 'bwe-decided') { await chrome.storage.local.set({ bwe: !!msg.on }); return { ok: true }; }  // controller.js
    if (msg.cmd === 'load') { const s = await settings(); await ensureOffscreen(); toOffscreen({ cmd: 'init', init: { base: s.modelBase, threads: s.threads || undefined, bwe: s.bwe, gpu: s.gpu } }); return { ok: true }; }
    return { error: 'unknown command' };
  })().then(reply, (e) => reply({ error: String(e) }));
  return true;
});

// closing the tab that is being read stops the reading
chrome.tabs.onRemoved.addListener(async (tabId) => {
  const { readingTab } = await chrome.storage.session.get('readingTab');
  if (tabId === readingTab) toOffscreen({ cmd: 'stop' }).catch(() => {});
});

chrome.commands.onCommand.addListener((cmd) => {
  if (cmd === 'read-page') readTab('auto');
  else if (cmd === 'stop') toOffscreen({ cmd: 'stop' }).catch(() => {});
});
