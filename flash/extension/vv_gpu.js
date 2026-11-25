// WebGPU VibeVoice decoder: the same function as the engine's vv_dec.h (streaming, exact causal state), one 64-value
// latent frame -> 3,200 samples at 24 kHz per call. Weights from vv.bin (TTLM) uploaded once: int8 linears stay
// int8 (dequantized in the shader: code * scale + min per 128 inputs), the rest float32 (float16 in the f16 / p16
// modes).
//
// Per frame one compute pass of 93 dispatches, all shapes fixed, so every bind group and parameter
// block is made once. As in vv_dec.h every causal convolution is a linear layer over a window of consecutive rows of
// [state rows; new rows]. The streaming state is one buffer of 139,200 floats in the CPU's save_state order, kept
// twice (ping-pong: a frame reads the last frame's, writes its own), so a redraw's save / load is one buffer copy.
//
// Modes: see VVGpu below. In every mode the residual stream, norms, states, int8 scales and all sums
// stay float32.

// src: the file as an ArrayBuffer, or a Uint8Array view of it (e.g. inside a WebAssembly heap: no copy)
export function parseTTLM(src) {
  const u8 = src instanceof Uint8Array ? src : new Uint8Array(src), buf = u8.buffer, base = u8.byteOffset;
  const dv = new DataView(buf, base, u8.byteLength);
  if (String.fromCharCode(...u8.subarray(0, 8)) !== 'TTLM0001') throw new Error('not a TTLM file');  // (shared views too)
  const n = dv.getUint32(8, true), t = new Map();
  const DT = { 0: Uint8Array, 1: Int8Array, 2: Uint16Array, 3: Float32Array, 4: Int32Array };
  for (let i = 0; i < n; i++) {
    const e = 16 + i * 104;
    let name = '';
    for (let k = 0; k < 48 && u8[e + k]; k++) name += String.fromCharCode(u8[e + k]);
    const dt = dv.getUint32(e + 48, true), nd = dv.getUint32(e + 52, true);
    const dims = [];
    for (let k = 0; k < nd; k++) dims.push(Number(dv.getBigUint64(e + 56 + 8 * k, true)));
    const off = Number(dv.getBigUint64(e + 88, true)), nb = Number(dv.getBigUint64(e + 96, true));
    const A = DT[dt];
    t.set(name, { dt, dims, data: new A(buf, base + off, nb / A.BYTES_PER_ELEMENT) });
  }
  return t;
}

function f16tof32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}
const F16 = (() => { const t = new Float32Array(65536); for (let i = 0; i < 65536; i++) t[i] = f16tof32(i); return t; })();

// float32 -> float16 bits (round to nearest even)
function f32tof16(arr) {
  const out = new Uint16Array(arr.length), f = new Float32Array(1), u = new Uint32Array(f.buffer);
  for (let i = 0; i < arr.length; i++) {
    f[0] = arr[i];
    const x = u[0], sign = (x >>> 16) & 0x8000;
    let e = ((x >>> 23) & 255) - 127 + 15, m = x & 0x7fffff;
    if (e >= 31) { out[i] = sign | 0x7c00; continue; }
    if (e <= 0) {
      if (e < -10) { out[i] = sign; continue; }
      m |= 0x800000;
      const sh = 14 - e, half = 1 << (sh - 1), rest = m & ((1 << sh) - 1);
      let r = m >>> sh;
      if (rest > half || (rest === half && (r & 1))) r++;
      out[i] = sign | r;
      continue;
    }
    let r = (e << 10) | (m >>> 13);
    const rest = m & 0x1fff;
    if (rest > 0x1000 || (rest === 0x1000 && (r & 1))) r++;
    out[i] = sign | r;
  }
  return out;
}

// ---- WGSL helpers ------------------------------------------------------------------------------------------------

const COMMON = `
fn gelu(x: f32) -> f32 {
  let z = abs(x) * 0.70710678118654752;
  let t = 1.0 / (1.0 + 0.5 * z);
  let q = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 +
          t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
  let erfc = t * exp(-z * z + q);
  return select(x * 0.5 * erfc, x * (1.0 - 0.5 * erfc), x >= 0.0);
}
fn gelu4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w)); }
fn i8x4(w: u32) -> vec4<f32> {
  return vec4<f32>(vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4<u32>(24u));
}
fn ld16(w: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn st16(v: vec4<f32>) -> vec2<u32> { return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw)); }`;

// Types of one mode. Activations that feed linears (window P, norm rows P2, hidden rows H) and float weights are
// stored as S: f32 | p16 (two halves per u32: pack2x16float, core WGSL, no shader-f16) | f16 (shader-f16).
function types(mode) {
  const S = mode === 'f32' ? 'f32' : mode === 'f16' ? 'f16' : 'p16';
  const t = {
    mode, S, f16: mode === 'f16',
    arr: S === 'f32' ? 'vec4<f32>' : S === 'f16' ? 'vec4<f16>' : 'vec2<u32>',
    bytes4: S === 'f32' ? 16 : 8,
    ld: (e) => (S === 'f32' ? `(${e})` : S === 'f16' ? `vec4<f32>(${e})` : `ld16(${e})`),
    st: (v) => (S === 'f32' ? `(${v})` : S === 'f16' ? `vec4<f16>(${v})` : `st16(${v})`),
  };
  // inner products of the linears: operand load / dequant / dot
  t.ldx = (e) => (t.f16 ? `(${e})` : t.ld(e));
  t.ldw = (e) => (t.f16 ? `(${e})` : t.ld(e));
  t.deq = (e) => (t.f16 ? `vec4<f16>(${e})` : `(${e})`);
  t.dot = (a, b) => (t.f16 ? `f32(dot(${a}, ${b}))` : `dot(${a}, ${b})`);
  t.sv = t.f16 ? 'vec4<f16>' : 'vec4<f32>';  // operand type in registers / workgroup memory
  return t;
}

function preamble(t) {
  return (t.f16 ? 'enable f16;\n' : '') + COMMON;
}

function rowReduce(tpr) {
  return `const TPR = ${tpr}u; const RPW = ${64 / tpr}u;
var<workgroup> red: array<f32, 64>;
fn reduce_row(li: u32, v: f32) -> f32 {
  workgroupBarrier();
  red[li] = v;
  workgroupBarrier();
  for (var s = TPR / 2u; s > 0u; s = s >> 1u) {
    if ((li % TPR) < s) { red[li] = red[li] + red[li + s]; }
    workgroupBarrier();
  }
  return red[(li / TPR) * TPR];
}`;
}

// ---- row kernels -----------------------------------------------------------------------------------------------

// P <- [STR rows (R); SRC rows (T), rms-normalised if rms]
function winShader(t, tpr) {
  return `${preamble(t)}
struct Prm { R: u32, C: u32, T: u32, soff: u32, rms: u32, eps: f32, s0: u32, s1: u32 }
@group(0) @binding(0) var<uniform> p: Prm;
@group(0) @binding(1) var<storage, read> STR: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> SRC: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> P: array<${t.arr}>;
${rowReduce(tpr)}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = li % TPR;
  let row = wg.x * RPW + li / TPR;
  let nv = p.C / 4u;
  let isrc = row >= p.R && row < p.R + p.T;
  var ss = 0.0;
  if (row < p.R) {
    for (var v = lane; v < nv; v += TPR) { P[row * nv + v] = ${t.st('STR[p.soff / 4u + row * nv + v]')}; }
  } else if (isrc) {
    let s = (row - p.R) * nv;
    for (var v = lane; v < nv; v += TPR) { let x = SRC[s + v]; ss += dot(x, x); }
  }
  let tot = reduce_row(li, ss);
  if (isrc) {
    let s = (row - p.R) * nv;
    let inv = select(1.0, inverseSqrt(tot / f32(p.C) + p.eps), p.rms != 0u);
    for (var v = lane; v < nv; v += TPR) { P[row * nv + v] = ${t.st('SRC[s + v] * inv')}; }
  }
}`;
}

// fused window + depthwise conv: XO = XI + dwconv([STR rows; rms(XI) rows]); P2 <- rms(XO). Workgroups 0..G-1 own
// RPW output rows each; workgroup G writes the new state (the window's last R rows) into STW (ping-pong: STR is
// last frame's state, so nothing read here is written here).
function wdwShader(t, tpr) {
  const rpw = 64 / tpr, nwin = rpw + 6;
  return `${preamble(t)}
struct Prm { C: u32, T: u32, R: u32, soff: u32, eps: f32, G: u32, s1: u32, s2: u32 }
@group(0) @binding(0) var<uniform> p: Prm;
@group(0) @binding(1) var<storage, read> XI: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> STR: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> DW: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> DWB: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> XO: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> P2: array<${t.arr}>;
@group(0) @binding(7) var<storage, read_write> STW: array<vec4<f32>>;
${rowReduce(tpr)}
var<workgroup> inv: array<f32, ${nwin}>;
// virtual window row v (0..R+T): state row v, else XI row v - R normalised
fn vrow(v: u32, c4: u32, nv: u32, v0: u32) -> vec4<f32> {
  if (v < p.R) { return STR[p.soff / 4u + v * nv + c4]; }
  return XI[(v - p.R) * nv + c4] * inv[v - v0];
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = li % TPR;
  let r = li / TPR;
  let nv = p.C / 4u;
  let state = wg.x == p.G;
  // window rows this workgroup reads: [v0, v0 + n)
  let v0 = select(wg.x * RPW, p.T, state);
  let n = select(RPW + 6u, p.R, state);
  for (var b = 0u; b < ${Math.ceil(nwin / rpw)}u; b++) {
    let k = b * RPW + r;
    let v = v0 + k;
    var ss = 0.0;
    if (k < n && v >= p.R && v < p.R + p.T) {
      for (var c4 = lane; c4 < nv; c4 += TPR) { let x = XI[(v - p.R) * nv + c4]; ss += dot(x, x); }
    }
    let tot = reduce_row(li, ss);
    if (lane == 0u && k < n) { inv[k] = select(1.0, inverseSqrt(tot / f32(p.C) + p.eps), v >= p.R); }
  }
  workgroupBarrier();
  if (state) {
    for (var k = r; k < p.R; k += RPW) {
      for (var c4 = lane; c4 < nv; c4 += TPR) { STW[p.soff / 4u + k * nv + c4] = vrow(v0 + k, c4, nv, v0); }
    }
    return;
  }
  let row = wg.x * RPW + r;
  var ss = 0.0;
  if (row < p.T) {
    for (var c4 = lane; c4 < nv; c4 += TPR) {
      var acc = DWB[c4];
      for (var j = 0u; j < 7u; j++) { acc += DW[j * nv + c4] * vrow(row + j, c4, nv, v0); }
      let x = XI[row * nv + c4] + acc;
      XO[row * nv + c4] = x;
      ss += dot(x, x);
    }
  }
  let tot = reduce_row(li, ss);
  if (row < p.T) {
    let iv = inverseSqrt(tot / f32(p.C) + p.eps);
    for (var c4 = lane; c4 < nv; c4 += TPR) { P2[row * nv + c4] = ${t.st('XO[row * nv + c4] * iv')}; }
  }
}`;
}

// audio[t] = b + w . window rows t..t+6 (C = 32), rows from the state STR / X directly
function headShader(t) {
  return `${preamble(t)}
struct Prm { T: u32, soff: u32, C: u32, R: u32 }
@group(0) @binding(0) var<uniform> p: Prm;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> STR: array<vec4<f32>>;
fn vr(v: u32, c4: u32, nv: u32) -> vec4<f32> { if (v < p.R) { return STR[p.soff / 4u + v * nv + c4]; } return X[(v - p.R) * nv + c4]; }
@group(0) @binding(2) var<storage, read> HW: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> HB: array<f32>;
@group(0) @binding(4) var<storage, read_write> AUD: array<f32>;
@group(0) @binding(5) var<storage, read_write> STW: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  let nv = p.C / 4u;
  if (t < p.R * nv) { STW[p.soff / 4u + t] = vr(p.T + t / nv, t % nv, nv); }
  if (t < p.T) {
    var acc = 0.0;
    for (var j = 0u; j < 7u; j++) {
      for (var c4 = 0u; c4 < nv; c4++) { acc += dot(HW[j * nv + c4], vr(t + j, c4, nv)); }
    }
    AUD[t] = acc + HB[0];
  }
}`;
}

// ---- linears -----------------------------------------------------------------------------------------------------
// Y[m * ys + n] = epi(b[n] + sum_k X[m * xs + k] W[n][k]); rows of X may overlap (windows). Every output goes out
// as vec4 (4 consecutive n of one row). Side copy: workgroup 0 also copies cn vec4 X[csrc..] -> STW[cdst..] (a
// resampling window's new state; X is not written here). epi: none (Y f32), gelu (Y activation type), add (Y += , f32).

function linHead(t, kind, ep) {
  const q8 = kind !== 'f32';
  const YA = ep === 'gelu' ? t.arr : 'vec4<f32>';
  return `${preamble(t)}
struct Prm { M: u32, K: u32, N: u32, xs: u32, ys: u32, gx: u32, cn: u32, csrc: u32, cdst: u32, s0: u32, s1: u32, s2: u32 }
@group(0) @binding(0) var<uniform> p: Prm;
@group(0) @binding(1) var<storage, read> X: array<${t.arr}>;
@group(0) @binding(2) var<storage, read> W: array<${q8 ? 'XX' : t.arr}>;
${q8 ? '@group(0) @binding(3) var<storage, read> SM: array<vec2<f32>>;' : ''}
@group(0) @binding(4) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> Y: array<${YA}>;
@group(0) @binding(6) var<storage, read_write> STW: array<vec4<f32>>;`;
}
function epiStore(t, ep, idx, v) {
  if (ep === 'gelu') return `Y[${idx}] = ${t.st(`gelu4(${v})`)};`;
  if (ep === 'add') return `Y[${idx}] = Y[${idx}] + ${v};`;
  return `Y[${idx}] = ${v};`;
}
const sideCopy = (t, wg) => `if (p.cn > 0u && wg.x == 0u && wg.y == 0u) {
    for (var i = li; i < p.cn; i += ${wg}u) { STW[p.cdst + i] = ${t.ld('X[p.csrc + i]')}; }
  }`;

// M <= 8 rows: WG threads share K, NB = 4 outputs per workgroup (memory-bound: every weight read once, 16 B per
// thread and step, coalesced); sums over threads through workgroup memory.
function gemvShader(t, kind, MR, ep, WG) {
  const NV = MR * 4, q8 = kind !== 'f32';
  let body = '', decl = '';
  for (let m = 0; m < MR; m++) for (let b = 0; b < 4; b++) decl += `var a${m}_${b} = 0.0; `;
  if (q8) {
    for (let b = 0; b < 4; b++)
      body += `let w${b} = W[(n0 + ${b}u) * nw + i]; let s${b} = SM[(n0 + ${b}u) * gk + i / 8u];
      let c${b}0 = i8x4(w${b}.x); let c${b}1 = i8x4(w${b}.y); let c${b}2 = i8x4(w${b}.z); let c${b}3 = i8x4(w${b}.w);\n`;
    for (let m = 0; m < MR; m++) {
      body += `{ let xb = ${m}u * p.xs / 4u + i * 4u;
        let x0 = ${t.ld('X[xb]')}; let x1 = ${t.ld('X[xb + 1u]')}; let x2 = ${t.ld('X[xb + 2u]')}; let x3 = ${t.ld('X[xb + 3u]')};
        let xs = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));\n`;
      for (let b = 0; b < 4; b++) body += `        a${m}_${b} += s${b}.x * (dot(c${b}0, x0) + dot(c${b}1, x1) + dot(c${b}2, x2) + dot(c${b}3, x3)) + s${b}.y * xs;\n`;
      body += '      }\n';
    }
  } else {
    for (let b = 0; b < 4; b++) body += `let w${b} = ${t.ld(`W[(n0 + ${b}u) * nw + i]`)};\n`;
    for (let m = 0; m < MR; m++) {
      body += `{ let x = ${t.ld(`X[${m}u * p.xs / 4u + i]`)};`;
      for (let b = 0; b < 4; b++) body += ` a${m}_${b} += dot(x, w${b});`;
      body += ' }\n';
    }
  }
  let red = '';
  for (let m = 0; m < MR; m++) for (let b = 0; b < 4; b++) red += `red[li * ${NV}u + ${m * 4 + b}u] = a${m}_${b};\n`;
  red += `workgroupBarrier();
  if (li < ${NV}u) { var s = 0.0; for (var g = 0u; g < ${WG}u; g++) { s += red[g * ${NV}u + li]; } tot[li] = s; }`;
  return `${linHead(t, kind, ep).replace('XX', 'vec4<u32>')}
var<workgroup> red: array<f32, ${WG * NV}>;
var<workgroup> tot: array<f32, ${NV}>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  ${sideCopy(t, WG)}
  let n0 = (wg.y * p.gx + wg.x) * 4u;
  let nw = p.K / ${q8 ? 16 : 4}u;
  let gk = p.K / 128u;
  ${decl}
  if (n0 < p.N) {
    for (var i = li; i < nw; i += ${WG}u) {
      ${body}
    }
  }
  ${red}
  workgroupBarrier();
  if (li < ${MR}u && n0 < p.N) {
    let v = vec4<f32>(tot[li * 4u], tot[li * 4u + 1u], tot[li * 4u + 2u], tot[li * 4u + 3u]) + B[n0 / 4u];
    ${epiStore(t, ep, '(li * p.ys + n0) / 4u', 'v')}
  }
}`;
}

// M > 8 rows: 8 x 8 threads, each TM rows (strided by 8) x TN consecutive columns, operands straight from the
// buffers (L1 serves the reuse; Mali's workgroup memory is no faster); wt: weights stored transposed ([unit][N]), so
// neighbouring threads read neighbouring words (faster on Mali)
function gemmShader(t, kind, ep, TM, TN, wt = false) {
  const WI = (n, u) => (wt ? `${u} * p.N + ${n}` : `${n} * nw + ${u}`);   // weight unit u of row n
  const SI = (n, g) => (wt ? `(${g}) * p.N + ${n}` : `${n} * gk + ${g}`);  // scale / min of group g of row n
  const BM = 8 * TM, BN = 8 * TN, q8 = kind !== 'f32';
  let idx = '', decl = '', fma = '', out = '', lx = '', lw = '';
  for (let i = 0; i < TM; i++) idx += `let m${i} = mb + ty + ${8 * i}u; let xr${i} = min(m${i}, p.M - 1u) * p.xs / 4u;\n  `;
  for (let j = 0; j < TN; j++) idx += `let n${j} = nb + ${TN}u * tx + ${j}u; let nr${j} = min(n${j}, p.N - 1u);\n  `;
  for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) decl += `var a${i}_${j} = 0.0; `;
  for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) fma += `a${i}_${j} += ${t.dot(`x${i}`, `w${j}`)};\n      `;
  for (let i = 0; i < TM; i++)
    for (let q = 0; q < TN / 4; q++)
      out += `if (m${i} < p.M && n${4 * q} < p.N) {
    let v = vec4<f32>(a${i}_${4 * q}, a${i}_${4 * q + 1}, a${i}_${4 * q + 2}, a${i}_${4 * q + 3}) + B[n${4 * q} / 4u];
    ${epiStore(t, ep, `(m${i} * p.ys + n${4 * q}) / 4u`, 'v')}
  }\n  `;
  const wload = (n, k) => (q8 ? t.deq(`i8x4(W[${WI(n, k)}]) * SM[${SI(n, `${k} / 32u`)}].x + SM[${SI(n, `${k} / 32u`)}].y`)
    : t.ldw(`W[${WI(n, k)}]`));
  for (let i = 0; i < TM; i++) lx += `let x${i} = ${t.ldx(`X[xr${i} + k4]`)};\n      `;
  for (let j = 0; j < TN; j++) lw += `let w${j} = ${wload(`nr${j}`, 'k4')};\n      `;
  return `${linHead(t, kind, ep).replace('XX', 'u32')}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  ${sideCopy(t, 64)}
  let tx = li % 8u;
  let ty = li / 8u;
  let mb = wg.y * ${BM}u;
  let nb = wg.x * ${BN}u;
  let nw = p.K / 4u;
  let gk = p.K / 128u;
  ${idx}
  ${decl}
  for (var k4 = 0u; k4 < nw; k4++) {
      ${lx}${lw}${fma}
  }
  ${out}
}`;
}

// ---- the decoder ------------------------------------------------------------------------------------------------
// opts: mode f32 | p16 | f16; wt (the gemm layers' weights transposed); TM, TN (gemm register block); gemvMax (largest
// M through gemv); gemvWG (threads of a gemv workgroup). Window + depthwise conv in one dispatch, head without window:
// 93 dispatches per frame.

export class VVGpu {
  constructor(device, file, opts = {}) {
    this.dev = device;
    this.o = { mode: 'f32', wt: false, TM: 4, TN: 4, gemvMax: 8, gemvWG: 64, ...opts };
    this.t = types(this.o.mode);
    this.f = parseTTLM(file);
    this.pipes = new Map();
    this.bytes = 0;
    this.build();
    this.f = null;  // the weights are on the GPU: no reference to the file left (the page can drop its copy)
  }

  buf(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    const b = this.dev.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
    this.bytes += b.size;
    (this.all = this.all || []).push(b);
    return b;
  }
  upload(data) {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const b = this.buf(data.byteLength);
    if (bytes.byteLength % 4) { const c = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4); c.set(bytes); this.dev.queue.writeBuffer(b, 0, c); }
    else this.dev.queue.writeBuffer(b, 0, bytes);
    return b;
  }
  f32buf(name) { return this.upload(this.f.get(name).data); }
  wbuf(name) {  // float weights in the storage type
    const d = this.f.get(name).data;
    return this.upload(this.t.S === 'f32' ? d : f32tof16(d));
  }
  pipe(key, code) {
    if (!this.pipes.has(key)) {
      const module = this.dev.createShaderModule({ code });
      this.pipes.set(key, this.dev.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } }));
    }
    return this.pipes.get(key);
  }
  // a linear layer: float [N][K], or int8 codes [N][K] with a float16 scale and min per 128 inputs; M: its rows per
  // frame (with wt, the layers that go through the gemm (M > gemvMax) are stored transposed: units of 4 inputs of all
  // rows side by side, [unit][N], scales [group][N])
  lin(name, M = 1) {
    const f = this.f, wt = this.o.wt && M > this.o.gemvMax;
    const tr = (src, N, U, w) => {  // [N][U] units of w elements -> [U][N]
      const o = new src.constructor(src.length);
      for (let n = 0; n < N; n++) for (let u = 0; u < U; u++) for (let e = 0; e < w; e++) o[(u * N + n) * w + e] = src[(n * U + u) * w + e];
      return o;
    };
    if (f.has(name + '.w')) {
      const w = f.get(name + '.w'), [N, K] = w.dims;
      let d = wt ? tr(w.data, N, K / 4, 4) : w.data;
      if (this.t.S !== 'f32') d = f32tof16(d);
      return { kind: 'f32', q8: false, wt, N, K, W: this.upload(d), B: this.f32buf(name + '.bias') };
    }
    const c = f.get(name + '.codes'), s = f.get(name + '.scales').data, m = f.get(name + '.mins').data;
    const [N, G] = c.dims;
    let sm = new Float32Array(2 * N * G);
    for (let i = 0; i < N * G; i++) { sm[2 * i] = F16[s[i]]; sm[2 * i + 1] = F16[m[i]]; }
    const K = G * 128;
    let codes = c.data;
    if (wt) { codes = tr(codes, N, K / 4, 4); sm = tr(sm, N, G, 2); }
    return { kind: 'q8', q8: true, wt, N, K, W: this.upload(codes), SM: this.upload(sm), B: this.f32buf(name + '.bias') };
  }

  build() {
    const f = this.f;
    const c = f.get('cfg_vv').data;
    this.NS = c[0]; this.HOP = c[1]; this.LAT = c[2]; this.DWK = c[3]; this.HK = c[4];
    this.eps = f.get('cfg_vvf').data[0];
    const S = [];
    let rows = 1, maxx = 0, maxh = 0, stoff = 0;
    for (let i = 0; i < this.NS; i++) {
      const s = { C: c[5 + 4 * i], stride: c[6 + 4 * i], depth: c[7 + 4 * i], cin: c[8 + 4 * i], k: i === 0 ? this.DWK : 2 };
      s.Tin = rows; rows *= s.stride; s.T = rows;
      s.res = this.lin(`s${i}.res`, s.Tin);
      s.stoff = stoff; stoff += (s.k - 1) * s.cin;
      s.b = [];
      for (let j = 0; j < s.depth; j++) {
        const p = `s${i}.b${j}.`;
        const b = { dw: this.f32buf(p + 'dw'), dwb: this.f32buf(p + 'dwb'), l1: this.lin(p + 'l1', rows), l2: this.lin(p + 'l2', rows) };
        b.stoff = stoff; stoff += (this.DWK - 1) * s.C;
        s.b.push(b);
        maxh = Math.max(maxh, rows * b.l1.N);
      }
      maxx = Math.max(maxx, rows * s.C, (s.Tin + s.k) * s.cin, (rows + this.DWK) * s.C);
      S.push(s);
    }
    this.S = S;
    this.T = rows;
    this.head = { W: this.f32buf('head.w'), B: this.f32buf('head.bias'), stoff };
    stoff += (this.HK - 1) * S[this.NS - 1].C;
    this.stateSize = stoff;
    this.weightBytes = this.bytes;
    const ab = this.t.bytes4 / 4;
    this.X = [this.buf(4 * maxx), this.buf(4 * maxx)];
    this.P = this.buf(ab * maxx);
    this.P2 = this.buf(ab * maxx);
    this.H = this.buf(ab * maxh);
    this.ST = [this.buf(4 * stoff), this.buf(4 * stoff)];  // ping-pong: read last frame's, write this frame's
    this.LATB = this.buf(4 * 64);
    this.AUD = this.buf(4 * this.T);
    this.PRM = this.dev.createBuffer({ size: 256 * 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.prm = new ArrayBuffer(256 * 512);
    this.slot = 0;
    this.D = [this.plan(0), this.plan(1)];
    this.dev.queue.writeBuffer(this.PRM, 0, this.prm, 0, 256 * this.slot);
    this.par = 0;
    this.rb = [];
  }

  // the fixed list of dispatches of one frame reading state ST[par], writing ST[1 - par]: {pl, bg, x, y, stage}
  plan(par) {
    const dev = this.dev, t = this.t, o = this.o, D = [];
    const STR = this.ST[par], STW = this.ST[1 - par];
    const pdv = new DataView(this.prm);
    const param = (vals) => {  // [u32 | {f: float}]
      const off = 256 * this.slot++;
      vals.forEach((v, i) => (typeof v === 'object' ? pdv.setFloat32(off + 4 * i, v.f, true) : pdv.setUint32(off + 4 * i, v, true)));
      return { buffer: this.PRM, offset: off, size: 48 };
    };
    const bg = (pl, bufs) => dev.createBindGroup({
      layout: pl.getBindGroupLayout(0),
      entries: Object.entries(bufs).map(([b, r]) => ({ binding: Number(b), resource: r.buffer ? r : { buffer: r } })),
    });
    const tprOf = (C) => Math.min(64, Math.max(1, C / 4));
    const win = (stage, R, C, T, soff, src, rms) => {
      const tpr = tprOf(C), pl = this.pipe(`win${tpr}`, winShader(t, tpr));
      D.push({ stage, pl, bg: bg(pl, { 0: param([R, C, T, soff, rms ? 1 : 0, { f: this.eps }, 0, 0]), 1: STR, 2: src, 3: this.P }),
               x: Math.ceil((R + T) / (64 / tpr)) });
    };
    const lin = (stage, L, X, M, xs, Y, ys, ep, copy = [0, 0, 0]) => {
      const q8 = L.q8;
      let pl, gx, gy;
      if (M <= o.gemvMax) {
        // the sums of a gemv workgroup must fit the device's workgroup memory (default 16 KiB): else 64 threads
        let WG = o.gemvWG;
        if (4 * (WG * M * 4 + M * 4) > dev.limits.maxComputeWorkgroupStorageSize) WG = 64;
        pl = this.pipe(`gemv${L.kind}${M}${ep}${WG}`, gemvShader(t, L.kind, M, ep, WG));
        const nwg = L.N / 4;
        gx = Math.min(nwg, 32768); gy = Math.ceil(nwg / gx);
      } else {
        const TM = o.TM, TN = Math.max(4, Math.min(o.TN, L.N / 8));
        pl = this.pipe(`gemm${L.kind}${ep}${TM}x${TN}${L.wt}`, gemmShader(t, L.kind, ep, TM, TN, L.wt));
        gx = Math.ceil(L.N / (8 * TN)); gy = Math.ceil(M / (8 * TM));
      }
      const b = { 0: param([M, L.K, L.N, xs, ys, gx, ...copy, 0, 0, 0]), 1: X, 2: L.W, 4: L.B, 5: Y, 6: STW };
      if (q8) b[3] = L.SM;
      D.push({ stage, pl, bg: bg(pl, b), x: gx, y: gy });
    };
    let cur = 0;  // X[cur]: the current rows
    for (let i = 0; i < this.NS; i++) {
      const s = this.S[i], R = s.k - 1, src = i === 0 ? this.LATB : this.X[cur];
      // resampling: window, linear (+ its new state as a side copy)
      win(i, R, s.cin, s.Tin, s.stoff, src, false);
      cur = 1 - cur;
      lin(i, s.res, this.P, s.Tin, s.cin, this.X[cur], s.res.N, 'none', [R * s.cin / 4, s.Tin * s.cin / 4, s.stoff / 4]);
      const C = s.C, T = s.T, Rd = this.DWK - 1;
      for (const b of s.b) {
        const tpr = tprOf(C), G = Math.ceil(T / (64 / tpr)), pl = this.pipe(`wdw${tpr}`, wdwShader(t, tpr));
        D.push({ stage: i, pl, bg: bg(pl, { 0: param([C, T, Rd, b.stoff, { f: this.eps }, G, 0, 0]), 1: this.X[cur], 2: STR, 3: b.dw,
                                            4: b.dwb, 5: this.X[1 - cur], 6: this.P2, 7: STW }), x: G + 1 });
        cur = 1 - cur;
        lin(i, b.l1, this.P2, T, C, this.H, b.l1.N, 'gelu');
        lin(i, b.l2, this.H, T, b.l1.N, this.X[cur], C, 'add');
      }
    }
    const Ch = this.S[this.NS - 1].C, Rh = this.HK - 1;
    const pl = this.pipe('head', headShader(t));
    const hb = { 0: param([this.T, this.head.stoff, Ch, Rh]), 1: this.X[cur], 2: this.head.W, 3: this.head.B, 4: this.AUD, 5: STW, 6: STR };
    D.push({ stage: this.NS, pl, bg: bg(pl, hb), x: Math.ceil(this.T / 64) });
    return D;
  }

  destroy() { for (const b of [...this.all, ...this.rb, this.PRM]) b.destroy(); this.all = []; this.rb = []; }
  reset() { this.dev.queue.writeBuffer(this.ST[this.par], 0, new Float32Array(this.stateSize)); }
  // the streaming state (before the next frame) to a snapshot buffer / back: GPU-side copies in queue order
  saveState(dst) { const e = this.dev.createCommandEncoder(); e.copyBufferToBuffer(this.ST[this.par], 0, dst, 0, 4 * this.stateSize); this.dev.queue.submit([e.finish()]); }
  loadState(src) { const e = this.dev.createCommandEncoder(); e.copyBufferToBuffer(src, 0, this.ST[this.par], 0, 4 * this.stateSize); this.dev.queue.submit([e.finish()]); }
  // one frame whose latent is already on the GPU (src buffer, 64 floats at byte offset off): encodes into enc
  encodeFrom(enc, src, off = 0) {
    enc.copyBufferToBuffer(src, off, this.LATB, 0, 256);
    const pass = enc.beginComputePass();
    for (const d of this.D[this.par]) { pass.setPipeline(d.pl); pass.setBindGroup(0, d.bg); pass.dispatchWorkgroups(d.x, d.y || 1); }
    pass.end();
    this.par = 1 - this.par;
  }
  // one frame: encodes, submits; resolves with the audio (Float32Array HOP), or null if !read. Several may be in
  // flight (the queue keeps their order).
  decode(lat, { read = true } = {}) {
    const dev = this.dev;
    dev.queue.writeBuffer(this.LATB, 0, lat);
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const d of this.D[this.par]) {
      pass.setPipeline(d.pl);
      pass.setBindGroup(0, d.bg);
      pass.dispatchWorkgroups(d.x, d.y || 1);
    }
    pass.end();
    this.par = 1 - this.par;
    let rb = null;
    if (read) {
      rb = this.rb.find((b) => !b.busy);
      if (!rb) { rb = dev.createBuffer({ size: 4 * this.T, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); this.rb.push(rb); }
      rb.busy = true;
      enc.copyBufferToBuffer(this.AUD, 0, rb, 0, 4 * this.T);
    }
    dev.queue.submit([enc.finish()]);
    if (!read) return null;
    return rb.mapAsync(GPUMapMode.READ).then(() => {
      const a = new Float32Array(rb.getMappedRange().slice(0));
      rb.unmap();
      rb.busy = false;
      return a;
    });
  }
}
