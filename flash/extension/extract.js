// Runs inside the tab (chrome.scripting.executeScript): the selection if there is one, else the page's main text.
// Main text: <article> / <main> / [role=main] if present, else the block with the most paragraph text; hidden
// elements, navigation, forms, code-like and script content are skipped; block elements become line breaks.
// Formulas (MathJax 2 / 3 / 4, KaTeX, MathML, Wikipedia) are read once, from their TeX source where the page has it,
// else from their MathML converted to TeX, as U+FDD0 TeX U+FDD1; display formulas stand on their own line.
// After each call globalThis.__rvExtract = { text, rangesFor(start, end) }: the DOM Ranges that the characters
// [start, end) of text came from (text pieces; a formula as one range around its rendered element).
// Self-contained (no imports): it is serialized into the page.
export function extractText() {
  const notes = [];
  const NC = /[\uFDD0-\uFDEF]/, NCG = /[\uFDD0-\uFDEF]+/g;

  // Source map: every piece of page text gets consecutive raw indices; per piece its first raw index (pS), its node
  // (pN: a Text, or the element of a formula or image alt) and its offset in that Text (pO). Inserted characters: -1.
  const pS = [], pN = [], pO = [];
  let raw = 0;
  const buf = () => ({ parts: [], ix: [] });
  const put = (b, s, node = null, off = 0) => {
    if (!s) return;
    b.parts.push(s);
    if (node) { pS.push(raw); pN.push(node); pO.push(off); }
    for (let i = 0; i < s.length; i++) b.ix.push(node ? raw + i : -1);
    if (node) raw += s.length;
  };
  const putText = (b, node, a, z) => {
    const d = node.data.slice(a, z);
    if (!NC.test(d)) { put(b, d, node, a); return; }
    let i = 0;
    for (const m of d.matchAll(NCG)) { put(b, d.slice(i, m.index), node, a + i); i = m.index + m[0].length; }
    put(b, d.slice(i), node, a + i);
  };
  // a regex replace that carries the map: f(m) gives [char, position in the match it comes from]
  const rep = (t, re, f) => {
    const parts = [], ix = [];
    let last = 0;
    for (const m of t.s.matchAll(re)) {
      parts.push(t.s.slice(last, m.index));
      for (let i = last; i < m.index; i++) ix.push(t.ix[i]);
      for (const [c, k] of f(m, t.ix)) { parts.push(c); ix.push(t.ix[m.index + k]); }
      last = m.index + m[0].length;
    }
    parts.push(t.s.slice(last));
    for (let i = last; i < t.s.length; i++) ix.push(t.ix[i]);
    return { s: parts.join(''), ix };
  };
  const trim = (t) => {
    const a = t.s.length - t.s.trimStart().length, z = Math.max(a, t.s.trimEnd().length);
    return { s: t.s.slice(a, z), ix: t.ix.slice(a, z) };
  };
  const finish = (b) => {
    let t = { s: b.parts.join(''), ix: b.ix };
    t = rep(t, /[ \t\u00a0]+/g, () => [[' ', 0]]);
    t = rep(t, / *\n */g, (m) => [['\n', m[0].indexOf('\n')]]);
    t = rep(t, /\.\.\n/g, (m, ix) => [['.', ix[m.index] >= 0 ? 0 : 1], ['\n', 2]]);
    t = rep(t, /([.!?:;,])\.\n/g, (m) => [[m[1], 0], ['\n', 2]]);
    t = rep(t, /\n{3,}/g, () => [['\n', 0], ['\n', 1]]);
    return trim(t);
  };
  const finalize = (source, t) => {
    const text = t.s, ix = t.ix;
    let p = 0;
    const pieceOf = (r) => {
      if (pS[p] <= r && (p + 1 === pS.length || r < pS[p + 1])) return p;
      let lo = 0, hi = pS.length - 1;
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (pS[mid] <= r) lo = mid; else hi = mid - 1; }
      return (p = lo);
    };
    const rangesFor = (start, end) => {
      const out = [];
      let cur = null;
      const flush = () => {
        if (!cur) return;
        const r = document.createRange();
        if (cur.el) r.selectNode(cur.el);
        else { const n = cur.node.length; r.setStart(cur.node, Math.min(cur.a, n)); r.setEnd(cur.node, Math.min(cur.z, n)); }
        out.push(r);
        cur = null;
      };
      for (let i = Math.max(0, start | 0), e = Math.min(ix.length, end); i < e; i++) {
        const r = ix[i];
        if (r < 0) continue;
        const q = pieceOf(r), node = pN[q];
        if (!node.isConnected) continue;
        if (node.nodeType !== 3) { if (!cur || cur.el !== node) { flush(); cur = { el: node }; } continue; }
        const o = pO[q] + r - pS[q];
        if (cur && cur.node === node && o >= cur.z) { cur.z = o + 1; continue; }
        flush();
        cur = { node, a: o, z: o + 1 };
      }
      flush();
      return out;
    };
    globalThis.__rvExtract = { text, rangesFor };
    return { source, text, title: document.title, notes };
  };

  // ---- formulas ----
  const texClean = (s) => (s || '').replace(NCG, '').replace(/(^|[^\\])((?:\\\\)*)%[^\n]*/g, '$1$2').replace(/\s+/g, ' ').trim();
  const soup = (el) => (el ? el.textContent.replace(NCG, '').replace(/[\u2061-\u2064\u200b]/g, '').replace(/\s+/g, ' ').trim() : '');
  const mwTex = (e) => {  // Wikipedia (Parsoid): data-mw='{"name":"math", ..., "body":{"extsrc":"TeX"}}'
    try { const o = JSON.parse(e.getAttribute('data-mw') || 'null'); return o && o.body && typeof o.body.extsrc === 'string' ? o.body.extsrc : ''; } catch (x) { return ''; }
  };
  const attrTex = (e) => (e ? texClean(e.getAttribute('data-latex') || mwTex(e)) : '');
  const blockMath = (m) => m.getAttribute('display') === 'block' || m.getAttribute('mode') === 'display';
  const parseMath = (s) => {
    if (!s || !/<math[\s>]/i.test(s)) return null;
    try { return new DOMParser().parseFromString(s, 'text/html').querySelector('math'); } catch (x) { return null; }
  };
  // TeX of a <math>: x-tex annotation, data-latex / data-mw, alttext that looks like TeX, else converted
  const mathTex = (m, noConvert) => {
    for (const a of m.querySelectorAll(':scope > semantics > annotation')) {
      if (/^(application\/x-tex|application\/x-latex|text\/x-tex|tex|latex)$/i.test(a.getAttribute('encoding') || '')) {
        const t = texClean(a.textContent);
        if (t) return t;
      }
    }
    const t = attrTex(m);
    if (t) return t;
    const alt = m.getAttribute('alttext') || '';
    if (/[\\^_{}]/.test(alt)) return texClean(alt);
    return noConvert ? '' : mml2tex(m);
  };

  // MathML -> TeX (presentation MathML as MathJax, KaTeX, LaTeXML and authors write it)
  const BIG = { '∑': 'sum', '∏': 'prod', '∐': 'coprod', '∫': 'int', '∬': 'iint', '∭': 'iiint', '∮': 'oint', '⋃': 'bigcup',
    '⋂': 'bigcap', '⨁': 'bigoplus', '⨂': 'bigotimes', '⨀': 'bigodot', '⋁': 'bigvee', '⋀': 'bigwedge', '⨄': 'biguplus' };
  const FUNC = /^(arccos|arcsin|arctan|arg|cos|cosh|cot|coth|csc|deg|det|dim|exp|gcd|hom|inf|ker|lg|lim|liminf|limsup|ln|log|max|min|Pr|sec|sin|sinh|sup|tan|tanh)$/;
  const ACC = { '^': 'hat', 'ˆ': 'hat', '\u0302': 'hat', '¯': 'bar', 'ˉ': 'bar', '‾': 'overline', '―': 'overline', '→': 'vec',
    '\u20d7': 'vec', '~': 'tilde', '˜': 'tilde', '\u0303': 'tilde', '˙': 'dot', '¨': 'ddot', 'ˇ': 'check', '˘': 'breve', '´': 'acute',
    '`': 'grave', '˚': 'mathring', '⏞': 'overbrace', '←': 'overleftarrow', '↔': 'overleftrightarrow' };
  const UACC = { '_': 'underline', '¯': 'underline', '‾': 'underline', '―': 'underline', '⏟': 'underbrace' };
  const WIDE = { bar: 'overline', vec: 'overrightarrow', hat: 'widehat', tilde: 'widetilde' };
  const VAR = { bold: 'mathbf', 'double-struck': 'mathbb', script: 'mathcal', fraktur: 'mathfrak', 'bold-italic': 'boldsymbol',
    'sans-serif': 'mathsf', monospace: 'mathtt' };
  const ENV = { '()': 'pmatrix', '[]': 'bmatrix', '||': 'vmatrix', '‖‖': 'Vmatrix', '{}': 'Bmatrix', '{': 'cases' };
  const SPACED = /^[=<>≤≥≠≈≡≅∼≃∝∈∉⊂⊃⊆⊇→←↔⇒⇐⇔↦+−±∓×÷⋅∗∘∪∩∧∨⊕⊗-]$/;
  const OPEN = /^[([{|‖⟨⌊⌈]$/, CLOSE = /^[)\]}|‖⟩⌋⌉]$/;
  const esc = (s) => s.replace(/[{}#$%&_\\~^]/g, (c) => ({ '\\': '\\backslash ', '~': '\\sim ', '^': '\\^{}' }[c] || '\\' + c));
  const escText = (s) => s.replace(/[{}#$%&_\\~^]/g, (c) => ({ '\\': '\\textbackslash ', '~': '\\~{}', '^': '\\^{}' }[c] || '\\' + c));
  const tok = (n) => n.textContent.replace(/[\u2061-\u2064\u200b]/g, '').trim();
  const kids = (n) => [...n.children];
  const unwrap = (n) => { while (n && /^(mrow|mstyle|mpadded)$/.test(n.localName) && n.children.length === 1) n = n.firstElementChild; return n; };
  const moText = (n) => (n && n.localName === 'mo' ? tok(n) : null);
  const bracketed = (n) => {
    if (!n || n.localName !== 'mrow' || n.children.length < 2) return false;
    const a = moText(unwrap(n.firstElementChild)), z = moText(unwrap(n.lastElementChild));
    return OPEN.test(a || '') && CLOSE.test(z || '');
  };
  const atom = (n) => { const u = unwrap(n); return !u || /^(mi|mn|mo|mtext|msub|msup|msubsup|mover|munder|munderover|mfrac|msqrt|mroot|mfenced|mtable)$/.test(u.localName) || bracketed(u); };
  const grp = (n) => (atom(n) ? tx(n) : '{' + tx(n) + '}');
  const sc = (sub, sup) => (sub ? '_{' + sub + '}' : '') + (sup ? '^{' + sup + '}' : '');
  const single = (n) => { const u = unwrap(n); return !!u && /^(mi|mn)$/.test(u.localName) && [...tok(u)].length <= 1; };
  const opName = (n) => { const u = unwrap(n); return u && /^m[io]$/.test(u.localName) ? tok(u) : ''; };
  const fence = (c) => (c === '{' || c === '}' ? '\\' + c : c);
  const scripts = (base, sub, sup) => {
    const b = opName(base), pr = sup && moText(unwrap(sup));
    const s = sub && tx(sub), p = sup && tx(sup);
    if (pr && /^[′″‴']+$/.test(pr) && !sub) return grp(base) + pr.replace(/″/g, "''").replace(/‴/g, "'''").replace(/′/g, "'");
    if (BIG[b]) return '\\' + BIG[b] + sc(s, p);
    if (FUNC.test(b)) return '\\' + b + sc(s, p);
    return grp(base) + sc(s, p);
  };
  const underOver = (base, under, over) => {
    const b = opName(base), u = under && tx(under), o = over && tx(over);
    if (BIG[b]) return '\\' + BIG[b] + sc(u, o);
    if (FUNC.test(b)) return '\\' + b + sc(u, o);
    if (/^arg\s?(max|min)$/.test(b)) return '\\operatorname*{' + b.replace(/\s/, '') + '}' + sc(u, o);
    const ub = unwrap(base);
    const oa = over && ACC[moText(unwrap(over))], ua = under && UACC[moText(unwrap(under))];
    // \underbrace{x}_{n}, \overbrace{x}^{n}: the brace is an inner munder / mover
    if (under && !ua && ub && ub.localName === 'munder' && UACC[moText(unwrap(ub.children[1]))] === 'underbrace')
      return '\\underbrace{' + tx(ub.children[0]) + '}_{' + u + '}';
    if (over && !oa && ub && ub.localName === 'mover' && ACC[moText(unwrap(ub.children[1]))] === 'overbrace')
      return '\\overbrace{' + tx(ub.children[0]) + '}^{' + o + '}';
    let r = tx(base);
    if (ua) r = '\\' + ua + '{' + r + '}';
    if (oa) r = '\\' + (single(base) || !WIDE[oa] ? oa : WIDE[oa]) + '{' + r + '}';
    if (under && !ua) r = '\\underset{' + u + '}{' + r + '}';
    if (over && !oa) r = '\\overset{' + o + '}{' + r + '}';
    return r;
  };
  const table = (t, env) => {
    const rows = kids(t).filter((r) => r.localName === 'mtr' || r.localName === 'mlabeledtr').map((r) => {
      const cells = kids(r).filter((c) => c.localName === 'mtd');
      if (r.localName === 'mlabeledtr') cells.shift();
      return cells.map((c) => tx(c).trim()).join(' & ');
    });
    env = env || (/^\s*right\s+left/.test(t.getAttribute('columnalign') || '') ? 'aligned' : 'matrix');
    return '\\begin{' + env + '} ' + rows.join(' \\\\ ') + ' \\end{' + env + '}';
  };
  const row = (list) => {
    const xs = [];
    for (let i = 0; i < list.length; i++) {
      const a = unwrap(list[i]), o = moText(a), next = unwrap(list[i + 1]);
      if (o !== null && next && next.localName === 'mtable') {  // ( table ), [ table ], { table: matrices and cases
        const c = moText(unwrap(list[i + 2]));
        const env = ENV[o + (c || '')];
        if (env) { xs.push({ s: table(next, env) }); i += c === null ? 1 : 2; continue; }
      }
      if (o === '(' && next && next.localName === 'mfrac' && /^\s*0/.test(next.getAttribute('linethickness') || '') &&
          moText(unwrap(list[i + 2])) === ')') {
        xs.push({ s: '\\binom{' + tx(next.children[0]) + '}{' + tx(next.children[1]) + '}' });
        i += 2;
        continue;
      }
      xs.push({ s: tx(list[i]), op: o !== null && SPACED.test(o) });
    }
    let out = '', prevOp = true;
    for (const x of xs) {
      let s = x.s;
      if (!s) continue;
      if (x.op && !prevOp) s = ' ' + s + ' ';
      if (/\\[a-zA-Z]+$/.test(out) && /^[a-zA-Z]/.test(s)) out += ' ';
      out += s;
      prevOp = !!x.op || /^[([{⟨⌊⌈]$/.test(s);
    }
    return out;
  };
  const tx = (n) => {
    if (!n) return '';
    const k = kids(n);
    switch (n.localName) {
      case 'math': case 'mrow': case 'mstyle': case 'mpadded': case 'merror': case 'mtd': return row(k);
      case 'semantics': return tx(k[0]);
      case 'annotation': case 'annotation-xml': case 'mphantom': case 'none': case 'mprescripts': return '';
      case 'mi': {
        const t = tok(n);
        if ([...t].length > 1) return FUNC.test(t) ? '\\' + t : '\\operatorname{' + escText(t) + '}';
        const v = VAR[n.getAttribute('mathvariant')];
        return v && t ? '\\' + v + '{' + esc(t) + '}' : esc(t);
      }
      case 'mn': return esc(tok(n));
      case 'mo': { const t = tok(n); return BIG[t] ? '\\' + BIG[t] : FUNC.test(t) ? '\\' + t : esc(t); }
      case 'mtext': {
        const t = n.textContent.replace(/\s+/g, ' ');
        return t.trim() ? (t[0] === ' ' ? ' ' : '') + '\\text{' + escText(t.trim()) + '}' + (t.endsWith(' ') ? ' ' : '') : ' ';
      }
      case 'ms': return '\\text{"' + escText(n.textContent) + '"}';
      case 'mspace': {  // \quad and wider separate parts of a formula
        const w = parseFloat(/^\s*[\d.]+em\s*$/.test(n.getAttribute('width') || '') ? n.getAttribute('width') : '0');
        return w >= 1.9 ? ' \\qquad ' : w >= 0.9 ? ' \\quad ' : ' ';
      }
      case 'msup': return scripts(k[0], null, k[1]);
      case 'msub': return scripts(k[0], k[1], null);
      case 'msubsup': return scripts(k[0], k[1], k[2]);
      case 'munder': return underOver(k[0], k[1], null);
      case 'mover': return underOver(k[0], null, k[1]);
      case 'munderover': return underOver(k[0], k[1], k[2]);
      case 'mfrac':
        if (n.getAttribute('bevelled') === 'true') return grp(k[0]) + '/' + grp(k[1]);
        if (/^\s*0+(\.0*)?\s*([a-z]+|%)?\s*$/i.test(n.getAttribute('linethickness') || 'x')) return '{' + tx(k[0]) + ' \\atop ' + tx(k[1]) + '}';
        return '\\frac{' + tx(k[0]) + '}{' + tx(k[1]) + '}';
      case 'msqrt': return '\\sqrt{' + row(k) + '}';
      case 'mroot': return '\\sqrt[' + tx(k[1]) + ']{' + tx(k[0]) + '}';
      case 'mfenced': {
        const o = n.getAttribute('open') ?? '(', c = n.getAttribute('close') ?? ')';
        const sep = (n.getAttribute('separators') ?? ',').replace(/\s/g, '');
        if (k.length === 1 && unwrap(k[0]).localName === 'mtable' && ENV[o + c]) return table(unwrap(k[0]), ENV[o + c]);
        return fence(o) + k.map((x, i) => (i ? sep[Math.min(i - 1, sep.length - 1)] || '' : '') + tx(x)).join('') + fence(c);
      }
      case 'menclose': {
        const no = n.getAttribute('notation') || '', s = row(k);
        return /box/.test(no) ? '\\boxed{' + s + '}' : /strike/.test(no) ? '\\cancel{' + s + '}' : /radical/.test(no) ? '\\sqrt{' + s + '}' : s;
      }
      case 'mtable': return table(n, '');
      case 'mmultiscripts': {
        let pre = '', post = '', isPre = false;
        const nz = (x) => (x && x.localName !== 'none' ? tx(x) : '');
        for (let i = 1; i < k.length;) {
          if (k[i].localName === 'mprescripts') { isPre = true; i++; continue; }
          const s = sc(nz(k[i]), nz(k[i + 1]));
          if (isPre) pre += s; else post += s;
          i += 2;
        }
        return (pre ? '{}' + pre : '') + grp(k[0]) + post;
      }
      case 'maction': return tx(k[(Number(n.getAttribute('selection')) || 1) - 1]);
      default: return k.length ? row(k) : esc(tok(n));
    }
  };
  const mml2tex = (m) => texClean(tx(m)).replace(/\{ /g, '{').replace(/ \}/g, '}');

  // what an element is: 1 a formula (read once, as a whole), 2 a copy or helper of one (skipped), 0 neither
  const FRAME = /(^|\s)(MathJax(_CHTML|_SVG|_PHTML|_MathML)?|mjx-chtml)(\s|$)/;
  const kind = (el) => {
    const t = el.localName;
    if (t === 'script') return /^math\//i.test(el.type) ? 1 : 0;
    if (t === 'math' || t === 'mjx-container') return 1;
    if (t.startsWith('mjx-')) return 2;
    const c = el.classList;
    if (c.contains('katex') || c.contains('katex-display') || c.contains('katex-error') || c.contains('mwe-math-element')) return 1;
    if (el.id && el.id.endsWith('-Frame') && FRAME.test(el.getAttribute('class') || '')) return 1;
    if (c.contains('MathJax_Preview') || c.contains('MJX_Assistive_MathML') || c.contains('katex-html') || c.contains('katex-mathml') ||
        /^MathJax_(Message|Hidden|Font_Test|SVG_Hidden)$/.test(el.id)) return 2;
    return 0;
  };
  // MathJax 2: the source is the <script type="math/..."> after the rendered frame (id: the script's id + "-Frame")
  const mj2 = (s, frame) => {
    if (s && !frame && s.id) frame = document.getElementById(s.id + '-Frame');
    let host = frame || s;
    if (frame) {
      const p = frame.parentElement;
      if (p && /(^|\s)(MathJax(_SVG|_PHTML|_MathML)?_Display|MJXc-display)(\s|$)/.test(p.getAttribute('class') || '')) host = p;
    } else if (s.previousElementSibling && s.previousElementSibling.classList.contains('MathJax_Preview')) host = s.previousElementSibling;
    const type = s ? s.type.toLowerCase() : '';
    let display = /mode\s*=\s*display/.test(type) || (!!frame && host !== frame), tex = '';
    if (type.startsWith('math/tex')) tex = texClean(s.textContent);
    else {
      const m = (type.startsWith('math/mml') && parseMath(s.textContent)) ||
        (frame && (frame.querySelector('math') || parseMath(frame.getAttribute('data-mathml'))));
      if (m) { tex = mathTex(m); display = display || blockMath(m); } else if (s) tex = texClean(s.textContent);
    }
    return { key: s || frame, el: host, tex, display };
  };
  // MathJax 3 / 4: TeX in data-latex (MathJax 4) or data-mw (Wikipedia); else the assistive MathML; else speech.
  // Wikipedia keeps the source <math> hidden right before the container (all that a lazy placeholder has).
  const mjx = (el) => {
    const r = el.querySelector('mjx-math, [data-mml-node="math"]'), am = el.querySelector('mjx-assistive-mml math');
    const d = el.getAttribute('display');
    let display = d === 'true' || d === 'block', tex = attrTex(r) || (am && mathTex(am));
    if (!tex) {
      const p = el.previousElementSibling, pm = p && invisible(p) && (p.localName === 'math' ? p : p.querySelector(':scope > math'));
      if (pm) { tex = mathTex(pm); display = display || blockMath(pm); }
    }
    if (tex) return { key: el, el, tex, display };
    const sp = el.getAttribute('data-semantic-speech-none') || el.getAttribute('aria-label') ||
      (el.querySelector('[aria-label]') || el).getAttribute('aria-label');
    if (sp) return { key: el, el, text: sp.replace(NCG, ''), display };
    return { key: el, el, tex: soup(el.querySelector('mjx-math')), display };
  };
  const formula = (el) => {
    const t = el.localName, c = el.classList;
    if (t === 'script') return mj2(el, null);
    if (t === 'mjx-container') return mjx(el);
    if (t === 'math') return { key: el, el, tex: mathTex(el), display: blockMath(el) };
    if (c.contains('katex-error')) return { key: el, el, tex: texClean(el.textContent), display: false };
    if (c.contains('katex') || c.contains('katex-display')) {
      const k = c.contains('katex') ? el : el.querySelector('.katex') || el, m = k.querySelector('math');
      return { key: el, el, tex: (m && mathTex(m)) || soup(k.querySelector('.katex-html')), display: c.contains('katex-display') || (!!m && blockMath(m)) };
    }
    if (c.contains('mwe-math-element')) {  // Wikipedia: MathML (annotation / alttext) and / or a fallback image with the TeX as alt
      const m = el.querySelector('math'), img = el.querySelector('img');
      const tex = (m && mathTex(m, true)) || (img && texClean(img.alt)) || (m && mml2tex(m)) || '';
      const display = c.contains('mwe-math-element-block') ||
        !!el.querySelector('.mwe-math-mathml-display, .mwe-math-fallback-image-display, math[display="block"]');
      return { key: el, el, tex, display };
    }
    const s = document.getElementById(el.id.slice(0, -6));
    return mj2(s && s.localName === 'script' ? s : null, el);
  };
  const emit = (b, f, done) => {
    if (done.has(f.key)) return;
    done.add(f.key);
    if (!f.tex && !f.text) return;
    put(b, f.display ? '\n' : f.text ? ' ' : '');
    put(b, f.text || '\uFDD0' + f.tex + '\uFDD1', f.el);
    put(b, f.display ? '\n' : f.text ? ' ' : '');
  };

  // ---- walking ----
  const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG', 'CANVAS', 'IFRAME', 'OBJECT', 'EMBED', 'VIDEO',
    'AUDIO', 'NAV', 'FOOTER', 'ASIDE', 'FORM', 'BUTTON', 'SELECT', 'TEXTAREA', 'INPUT', 'LABEL', 'MENU', 'DIALOG',
    'PRE', 'CODE', 'SUP']);
  // an explicit selection: only content that is not shown
  const SEL_SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'TITLE', 'DESC', 'DEFS', 'METADATA', 'TEXTAREA', 'SELECT',
    'IFRAME', 'OBJECT', 'EMBED', 'VIDEO', 'AUDIO', 'CANVAS']);
  const BLOCK = new Set(['P', 'DIV', 'SECTION', 'ARTICLE', 'MAIN', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LI', 'UL', 'OL',
    'BLOCKQUOTE', 'TR', 'TABLE', 'DL', 'DT', 'DD', 'FIGCAPTION', 'BR', 'HR', 'TD', 'TH']);
  const ROLE_SKIP = /^(navigation|banner|contentinfo|complementary|search|menu|menubar|toolbar|dialog|alert|status)$/;
  const invisible = (el) => {
    if (el.hidden) return true;
    const cs = getComputedStyle(el);
    return cs.display === 'none' || cs.visibility === 'hidden' || cs.visibility === 'collapse' ||
      (cs.position === 'absolute' && el.offsetWidth <= 1 && el.offsetHeight <= 1);
  };
  const hidden = (el) => el.getAttribute('aria-hidden') === 'true' || ROLE_SKIP.test(el.getAttribute('role') || '') || invisible(el);
  const cls = (el) => ((el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className) || '') + ' ' + (el.id || '');
  const NOISE = /(^|[\s_-])(comment|share|social|related|promo|advert|ads?|cookie|consent|newsletter|subscribe|breadcrumb|sidebar|footer|menu|popup|modal)([\s_-]|$)/i;

  // root's text into b; with a range (a selection) only what it covers, under the relaxed rules
  function collect(root, b, range, done) {
    const walk = (n) => {
      if (range && n !== root && !range.intersectsNode(n)) return;
      if (n.nodeType === 3) {
        let a = 0, z = n.data.length;
        if (range && n === range.startContainer) a = range.startOffset;
        if (range && n === range.endContainer) z = range.endOffset;
        if (z > a) putText(b, n, a, z);
        return;
      }
      if (n.nodeType !== 1) return;
      const el = n, k = kind(el);
      if (k === 2) return;
      if (k === 1) { if (el.localName === 'script' || !invisible(el)) emit(b, formula(el), done); return; }
      const tag = el.tagName;
      if (range) {
        if (SEL_SKIP.has(tag.toUpperCase()) || invisible(el)) return;
      } else {
        if (SKIP.has(tag) || hidden(el)) return;
        if (tag === 'HEADER' && !el.closest('article, main, [role="main"]')) return;  // the site's header
        if (el !== root && NOISE.test(cls(el)) && el.textContent.length < 3000) return;
      }
      if (el.shadowRoot) for (const c of el.shadowRoot.childNodes) walk(c);
      const block = BLOCK.has(tag) && !(range && el === root);
      if (block) put(b, '\n');
      if (tag === 'IMG' && el.alt && el.alt.length > 20) {
        const alt = el.alt.trim().replace(NCG, '');
        put(b, '\n');
        put(b, alt, el);
        if (alt && !/[.!?]$/.test(alt)) put(b, '.');
        put(b, '\n');
      }
      for (const c of el.childNodes) walk(c);
      if (block) put(b, /^H[1-6]$/.test(tag) ? '.\n' : '\n');
    };
    walk(root);
  }
  const text = (root) => { const b = buf(); collect(root, b, null, new Set()); return finish(b); };

  const sel = window.getSelection ? window.getSelection() : null;
  if (sel && sel.rangeCount) {
    const b = buf(), done = new Set();
    for (let i = 0; i < sel.rangeCount; i++) {
      const r = sel.getRangeAt(i);
      if (r.collapsed) continue;
      let root = r.commonAncestorContainer;
      for (let e = root.nodeType === 1 ? root : root.parentElement; e; e = e.parentElement) if (kind(e) === 1) root = e;  // inside a formula: all of it
      if (i) put(b, '\n');
      collect(root, b, r, done);
    }
    const t = finish(b);
    if (t.s.length > 1) return finalize('selection', t);
    const plain = String(sel).trim().replace(NCG, '');  // e.g. a selection inside a text field
    if (plain.length > 1) return finalize('selection', { s: plain, ix: plain.split('').map(() => -1) });
  }
  if (document.contentType === 'application/pdf' || document.querySelector('embed[type="application/pdf"]')) {
    notes.push('pdf');
    return finalize('none', { s: '', ix: [] });
  }

  let root = document.querySelector('article') || document.querySelector('main, [role="main"]');
  if (!root) {  // the element holding the most text in <p> children
    let best = null, bestLen = 0;
    for (const p of document.querySelectorAll('p')) {
      const parent = p.parentElement;
      if (!parent || parent._rv) continue;
      parent._rv = true;
      let n = 0;
      for (const c of parent.children) if (c.tagName === 'P') n += c.textContent.length;
      if (n > bestLen) { bestLen = n; best = parent; }
    }
    for (const p of document.querySelectorAll('p')) if (p.parentElement) delete p.parentElement._rv;
    root = bestLen > 400 ? best : document.body;
  }
  let t = root ? text(root) : { s: '', ix: [] };
  if (root !== document.body && t.s.length < 200) t = text(document.body);
  if (document.querySelectorAll('iframe').length) notes.push('iframes');
  const canvas = [...document.querySelectorAll('canvas')].some((c) => c.width * c.height > 300000);
  if (canvas && t.s.length < 500) notes.push('canvas');
  const h1 = document.querySelector('h1');
  if (h1 && root && !root.contains(h1)) {
    const hb = buf(), w = document.createTreeWalker(h1, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) putText(hb, n, 0, n.data.length);
    const h = trim({ s: hb.parts.join(''), ix: hb.ix });
    if (h.s && !t.s.startsWith(h.s)) t = { s: h.s + '.\n' + t.s, ix: h.ix.concat([-1, -1], t.ix) };
  }
  return finalize('page', t);
}
