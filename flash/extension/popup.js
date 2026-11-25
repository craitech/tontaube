// Popup: read page / read selection / stop, the settings (highlight, 48 kHz, redraw, start delay, GPU; kept in chrome.storage), the
// model download and the engine's status. The popup closes on any click outside it; the engine keeps running in the
// offscreen document.
const $ = (id) => document.getElementById(id);
const sw = (msg) => chrome.runtime.sendMessage({ ...msg, to: 'sw' });
const num = (v, d = 0) => Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const ms = (v, d = 0) => (v == null || v < 0 ? '–' : num(v, d) + ' ms');

// the answer to a button: an error or the extractor's notes
function message(text, kind) {
  $('msg').hidden = !text;
  $('msg').textContent = text || '';
  $('msg').className = 'msg ' + (kind || '');
}

function headline(state, cls, what, where = '') {
  $('state').textContent = state;
  $('state').className = 'pill ' + (cls || '');
  $('status').hidden = !what;
  $('what').textContent = what || '';
  $('where').textContent = where;
}

// one request's figures (controller.js line())
function figures(s, live) {
  const n = s.chunks ?? 0;
  headline(live ? 'Reading' : 'Ready', live ? 'live' : '',
    live ? (s.chunk ? 'Reading' : 'Starting…') : s.state === 'stopped' ? 'Stopped' : 'Read',
    live && s.chunk ? `chunk ${s.chunk} of ${n}` : `${num(n)} chunk${n === 1 ? '' : 's'}, ${num(s.chars ?? 0)} characters`);
  $('figures').hidden = false;
  $('first').textContent = ms(s.first_audio_ms);
  $('speed').textContent = s.speed ? num(s.speed, 2) + '×' : '–';
  $('gaps').textContent = s.gaps ? `${s.gaps} (${num(s.gap_ms)} ms)` : '0';
  $('unknown').hidden = !s.unknown;
  $('unknown').textContent = s.unknown ? 'Not known to the voice (read as spaces): ' + s.unknown : '';
  const rows = [
    ['playback started after', ms(s.playback_ms)],
    ['audio', s.audio_s != null ? num(s.audio_s, 1) + ' s' : '–'],
    ['text read in', ms(s.prefill_ms)],
    ['per frame: model', ms(s.lm_ms, 1)],
    ['flow head', ms(s.head_ms, 1)],
    ['autoencoder', ms(s.ae_ms, 1)],
    ['decoder', ms(s.vv_ms, 1)],
    ['detector', ms(s.det_ms, 1)],
  ];
  if (s.redraw) rows.push(['flagged spots', `${num(s.spots ?? 0)}, redrawn ${num(s.replaced ?? 0)}`]);
  if (s.rate === 48000) rows.push(['upsampler', ms(s.bwe_ms, 1)]);
  rows.push(['output', `${s.rate / 1000} kHz, ${s.threads ?? '–'} threads`]);
  if (s.pauses) rows.push(['paced', `${s.pauses} pauses, ${num(s.paused_s, 1)} s`]);
  $('details').replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = k;
    dd.textContent = v;
    return [dt, dd];
  }));
}

// The graphics card in use, and why not (the engine's loaded info, st.load.gpu); nothing before the engine is loaded
const linuxDesktop = /Linux/.test((navigator.userAgentData && navigator.userAgentData.platform) || navigator.userAgent) && !/Android/i.test(navigator.userAgent);
function gpuReason(g) {
  const code = (t) => { const c = document.createElement('code'); c.textContent = t; return c; };
  if (!g || g.refused || /^GPU/.test(g.using || '')) return null;
  const f = g.failed || '';
  if (/^Auto: the CPU/.test(f)) {
    const db = g.check ? g.check.snr_db : Number((f.match(/(-?[\d.]+) dB/) || [])[1]);
    return db > 50
      ? { why: 'Your graphics card was checked on first use and was not faster here, so the processor is used.', steps: [['To use it anyway, set GPU to On.']] }
      : { why: 'Your graphics card was checked on first use and did not give the same audio, so the processor is used.' };
  }
  if (linuxDesktop && /no WebGPU|software GPU/.test(f)) {
    return { why: 'Chrome on Linux does not offer the graphics card to web pages by default.', steps: [
      ['Type ', code('chrome://flags'), ' into the address bar.'],
      ['Set "Vulkan" and "Default ANGLE Vulkan" to Enabled.'],
      ['Relaunch Chrome and read again.'],
      ['Still "processor"? Close Chrome completely and start it from a terminal with: ', code('google-chrome-stable --enable-features=Vulkan --use-angle=vulkan')],
      ['Without the graphics card everything still works on the processor, only slower.'],
    ] };
  }
  return { why: 'This browser offers no usable graphics card here.', steps: [
    ['In Chrome\'s settings, under System, switch on "Use graphics acceleration when available", then relaunch.'],
    ['Update Chrome (Settings, About Chrome).'],
    [code('chrome://gpu'), ' should list WebGPU as "Hardware accelerated".'],
  ] };
}
// the numbered steps of the guide box
function guideSteps(steps) {
  return steps.map((s) => { const li = document.createElement('li'); li.append(...s); return li; });
}
function gpuUse(st) {
  const g = st && st.load ? st.load.gpu : null, on = !!(g && /^GPU/.test(g.using || ''));
  const r = st && st.load ? gpuReason(g) : null;
  $('gpuStatus').hidden = !(st && st.load);
  $('gpuStatus').classList.toggle('on', on);
  $('gpuUse').textContent = on ? 'In use: graphics card' : 'In use: processor (CPU)';
  $('gpuWhy').hidden = !r;
  $('gpuWhyText').textContent = r ? r.why : '';
  $('gpuGuide').hidden = !(r && r.steps);
  $('gpuSteps').replaceChildren(...guideSteps((r && r.steps) || []));
}

// "48 kHz": true, false, or 'auto' until the first read chooses it (controller.js onBweDecided, kept by sw.js)
function bweShow(v) {
  $('bwe').checked = v !== false;
  $('bweUse').textContent = v === true || v === false ? '' : '(set at first use)';
}

// a changed setting: when it takes effect, shown until the next read starts
let cur = null, curPhase = null;  // the engine's last status and phase
const reading = () => !!cur && (cur.phase === 'speaking' || cur.phase === 'loading');
function note(text) {
  $('setNote').textContent = text || '';
  $('setNote').hidden = !text;
}
function nextRead(reload = false) {
  note('Applies from the next read.' + (reload ? ' The model reloads then, which takes a moment.' : '') +
    (reading() ? ' The current read continues as it is.' : ''));
}

// the info buttons: the text on hover and keyboard focus (CSS), and on a click or tap
function infoButtons() {
  const close = () => { for (const b of document.querySelectorAll('.info[aria-expanded="true"]')) b.setAttribute('aria-expanded', 'false'); };
  for (const b of document.querySelectorAll('.info')) {
    b.onclick = (e) => {
      e.stopPropagation();
      const open = b.getAttribute('aria-expanded') !== 'true';
      close();
      b.removeAttribute('data-dismissed');
      b.setAttribute('aria-expanded', String(open));
    };
    b.onblur = () => b.removeAttribute('data-dismissed');
  }
  document.addEventListener('click', close);
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    close();
    if (document.activeElement && document.activeElement.classList.contains('info')) document.activeElement.setAttribute('data-dismissed', '');
  });
}

function show(st) {
  const phase = st ? st.phase : null;
  if (phase !== curPhase && (phase === 'loading' || phase === 'speaking')) note('');  // a read starts
  cur = st;
  curPhase = phase;
  gpuUse(st);
  $('load').hidden = true;
  $('figures').hidden = true;
  if (!st) return headline('Not loaded', '', 'The voice model loads on the first request.');
  if (st.phase === 'loading') {
    const p = st.progress;
    headline('Loading', 'live');
    $('load').hidden = false;
    $('loadWhat').textContent = p ? (p.how === 'cache' ? 'Loading the voice model' : 'Downloading the voice model') : 'Starting the engine…';
    $('loadNum').textContent = p ? `${num(p.done / 1048576)} of ${num(p.total / 1048576)} MiB` : '';
    if (p) { $('prog').max = p.total; $('prog').value = p.done; } else $('prog').removeAttribute('value');
  } else if (st.phase === 'error') {
    headline('Error', 'bad');
    message('Error: ' + st.error, 'err');
  } else if (st.phase === 'speaking' && st.now) figures(st.now, true);
  else if (st.last) figures(st.last, false);
  else headline('Ready', '', st.load ? st.load.msg : st.phase);
}

async function init() {
  const s = await chrome.storage.local.get(['delayMs', 'bwe', 'redraw', 'highlight', 'gpu']);
  $('gpu').value = ['auto', 'on', 'off'].includes(s.gpu) ? s.gpu : 'auto';
  // a new GPU setting takes effect with the next load of the engine: close the loaded one now
  $('gpu').onchange = async () => {
    const was = reading(), loaded = !!(cur && cur.load);
    await chrome.storage.local.set({ gpu: $('gpu').value });
    sw({ cmd: 'close-offscreen' }).catch(() => {});
    cur = null;
    note(was ? 'Applies from the next read. The reading stops now.'
      : loaded ? 'Applies from the next read. The model reloads then, which takes a moment.' : 'Applies from the next read.');
  };
  $('delay').value = s.delayMs ?? 0;
  bweShow(s.bwe ?? 'auto');  // sw.js DEFAULTS
  chrome.storage.onChanged.addListener((ch, area) => { if (area === 'local' && ch.bwe) bweShow(ch.bwe.newValue ?? 'auto'); });
  $('redraw').checked = s.redraw ?? true;
  $('hl').checked = !!s.highlight;
  $('delay').onchange = () => { chrome.storage.local.set({ delayMs: Math.max(0, Number($('delay').value) || 0) }); nextRead(); };
  $('redraw').onchange = () => { chrome.storage.local.set({ redraw: $('redraw').checked }); nextRead(); };
  $('bwe').onchange = () => {
    chrome.storage.local.set({ bwe: $('bwe').checked });
    $('bweUse').textContent = '';
    // switched on while the engine runs without the upsampler: the next read loads it again with it (controller.js)
    nextRead($('bwe').checked && !!(cur && cur.load) && !cur.load.bwe);
  };
  // switched off during a reading, the highlight is cleared at once (sw.js)
  $('hl').onchange = () => {
    chrome.storage.local.set({ highlight: $('hl').checked });
    if (!$('hl').checked && reading()) note('Applies now.'); else nextRead();
  };
  infoButtons();
  const read = async (mode) => {
    message('Reading the page…');
    const r = await sw({ cmd: 'read', mode });
    const notes = r && r.notes && r.notes.length ? r.notes.join('\n') : '';
    if (r && r.error) message(r.error + (notes ? '\n' + notes : ''), 'err');
    else message(notes, 'warn');
  };
  $('page').onclick = () => read('auto');
  $('sel').onclick = () => read('selection');
  $('stop').onclick = () => sw({ cmd: 'stop' });
  chrome.runtime.onMessage.addListener((msg) => { if (msg.to === 'popup') show(msg.status); });
  const keys = Object.fromEntries((await chrome.commands.getAll()).map((c) => [c.name, c.shortcut]));
  if (keys['read-page']) $('hint').textContent = `${keys['read-page']} reads the selection or the page` + (keys.stop ? `, ${keys.stop} stops.` : '.');
  try {
    const r = await chrome.runtime.sendMessage({ to: 'offscreen', cmd: 'status' });
    show(r && r.status);
  } catch (e) { show(null); }
}
init();
