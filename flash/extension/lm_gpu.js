// WebGPU language model backbone of the frame model (Qwen3-style, lm.bin): the text reading (prefill: n tokens at
// once, causal), with the KV cache on the GPU (float16, the CPU's layout
// [layer][k|v][kv head][S][128]). Same function as the engine's Model::step (tt_engine.cpp): RMS norms, fused qkv,
// per-head q / k norm, RoPE (cos / sin per token from the caller, as the CPU computes them), attention over cache rows
// r0..r1-1 of each token, o, gate_up, SiLU * up, down; weights stay int4 (per 64 inputs) / int8 (per 128) and are
// dequantized in the shaders; activations float32 (option f16: the linears' input rows float16, products of 4 in
// float16, sums float32; residual stream, q / k / v and gate / up rows stay float32).
import { parseTTLM } from './vv_gpu.js';

export const F16 = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    t[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export const COMMON = `
fn q4lo(w: u32) -> vec4<f32> { return vec4<f32>(f32(w & 15u), f32((w >> 8u) & 15u), f32((w >> 16u) & 15u), f32((w >> 24u) & 15u)); }
fn q4hi(w: u32) -> vec4<f32> { return vec4<f32>(f32((w >> 4u) & 15u), f32((w >> 12u) & 15u), f32((w >> 20u) & 15u), f32(w >> 28u)); }
fn i8x4(w: u32) -> vec4<f32> {
  return vec4<f32>(vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4<u32>(24u));
}
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
fn mish(v: f32) -> f32 { return v * tanh(select(log(1.0 + exp(v)), v, v > 20.0)); }`;

// Linear layers: Y[m * ys + n] (+)= b[n] + sum_k X[m * xs + k] W[n][k], M rows, every output as vec4 of 4
// consecutive n. kinds: q4 (u32 unit = 4 + 4 inputs of a 64-group: low nibbles k..k+3, high nibbles k+32..k+35),
// q8 (u32 unit = 4 inputs), f32 (vec4 unit). act: '' | 'mish' applied to the output.
function linHead(kind, h = false) {
  return `${h ? 'enable f16;' : ''}${COMMON}
struct Prm { M: u32, K: u32, N: u32, xs: u32, ys: u32, gx: u32, hasb: u32, add: u32, act: u32, gate: u32, xoff: u32, yoff: u32 }
@group(0) @binding(0) var<uniform> p: Prm;
@group(0) @binding(1) var<storage, read> X0: array<vec4<${h ? 'f16' : 'f32'}>>;
@group(0) @binding(2) var<storage, read> W: array<${kind === 'f32' ? 'vec4<f32>' : 'u32'}>;
${kind === 'f32' ? '' : '@group(0) @binding(3) var<storage, read> SM: array<vec2<f32>>;'}
@group(0) @binding(4) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> Y: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> GT: array<vec4<f32>>;
fn X(i: u32) -> vec4<f32> { return vec4<f32>(X0[p.xoff + i]); }
${h ? 'fn XH(i: u32) -> vec4<f16> { return X0[p.xoff + i]; }' : ''}`;
}
// unit u of row n -> (lo weights, x index of lo, [hi weights, x index of hi]) as WGSL statements; xb: row base (vec4 index)
function unitDot(kind, n, u, xrow, acc) {
  if (kind === 'q4') {
    return `{ let w = W[${n} * nw + ${u}]; let g = ${u} / 8u; let s = SM[${n} * gk + g];
      let xi = ${xrow} + g * 16u + ${u} % 8u;
      ${acc} += s.x * (dot(q4lo(w), X(xi)) + dot(q4hi(w), X(xi + 8u))) + s.y * dot(X(xi) + X(xi + 8u), vec4<f32>(1.0)); }`;
  }
  if (kind === 'q8') {
    return `{ let s = SM[${n} * gk + ${u} / 32u]; let x = X(${xrow} + ${u});
      ${acc} += s.x * dot(i8x4(W[${n} * nw + ${u}]), x) + s.y * dot(x, vec4<f32>(1.0)); }`;
  }
  return `${acc} += dot(W[${n} * nw + ${u}], X(${xrow} + ${u}));`;
}
const unitsPerRow = (kind) => (kind === 'q4' ? 'p.K / 8u' : 'p.K / 4u');
const groupsPerRow = (kind) => (kind === 'q4' ? 'p.K / 64u' : 'p.K / 128u');

// y = W x + b; mish(y) if act; y * GT[n] if gate; Y += y if add, else Y = y
function epilogue(idx, v) {
  return `{ let bb = select(vec4<f32>(0.0), B[n4], p.hasb != 0u); var vv = ${v} + bb;
      if (p.act == 1u) { vv = vec4<f32>(mish(vv.x), mish(vv.y), mish(vv.z), mish(vv.w)); }
      if (p.gate != 0u) { vv = vv * GT[p.gate - 1u + n4]; }
      let yi = p.yoff + ${idx};
      if (p.add != 0u) { Y[yi] = Y[yi] + vv; } else { Y[yi] = vv; } }`;
}

// M <= 8: 64 threads split K; 4 outputs per workgroup
export function gemvShader(kind, MR, h = false) {
  const NV = MR * 4;
  let decl = '', body = '', red = '';
  for (let m = 0; m < MR; m++) for (let b = 0; b < 4; b++) decl += `var a${m}_${b} = 0.0; `;
  for (let b = 0; b < 4; b++) for (let m = 0; m < MR; m++) body += unitDot(kind, `(n0 + ${b}u)`, 'i', `${m}u * p.xs / 4u`, `a${m}_${b}`) + '\n      ';
  for (let m = 0; m < MR; m++) for (let b = 0; b < 4; b++) red += `red[li * ${NV}u + ${m * 4 + b}u] = a${m}_${b};\n  `;
  return `${linHead(kind, h)}
var<workgroup> red: array<f32, ${64 * NV}>;
var<workgroup> tot: array<f32, ${NV}>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let n0 = (wg.y * p.gx + wg.x) * 4u;
  let nw = ${unitsPerRow(kind)};
  let gk = ${groupsPerRow(kind)};
  ${decl}
  if (n0 < p.N) {
    for (var i = li; i < nw; i += 64u) {
      ${body}
    }
  }
  ${red}
  workgroupBarrier();
  if (li < ${NV}u) { var s = 0.0; for (var g = 0u; g < 64u; g++) { s += red[g * ${NV}u + li]; } tot[li] = s; }
  workgroupBarrier();
  if (li < ${MR}u && n0 < p.N) {
    let n4 = n0 / 4u;
    let v = vec4<f32>(tot[li * 4u], tot[li * 4u + 1u], tot[li * 4u + 2u], tot[li * 4u + 3u]);
    ${epilogue('(li * p.ys + n0) / 4u', 'v')}
  }
}`;
}

// M > 8: 8 x 8 threads, TM rows (strided by 8) x 4 consecutive columns each
// h: input rows stored as float16, weights dequantized to float16, products of 4 in float16, sums in float32
export function gemmShader(kind, TM, h = false) {
  let idx = '', decl = '', body = '', out = '';
  for (let i = 0; i < TM; i++) idx += `let m${i} = mb + ty + ${8 * i}u; let xr${i} = min(m${i}, p.M - 1u) * p.xs / 4u;\n  `;
  for (let j = 0; j < 4; j++) idx += `let n${j} = min(nb + 4u * tx + ${j}u, p.N - 1u);\n  `;
  for (let i = 0; i < TM; i++) for (let j = 0; j < 4; j++) decl += `var a${i}_${j} = 0.0; `;
  if (h && kind === 'q4') {
    body = `let g = u / 8u; let o = g * 16u + u % 8u;\n`;
    for (let i = 0; i < TM; i++) body += `    let xl${i} = XH(xr${i} + o); let xh${i} = XH(xr${i} + o + 8u);\n`;
    for (let j = 0; j < 4; j++) {
      body += `    { let w = W[n${j} * nw + u]; let s = SM[n${j} * gk + g]; let lo = vec4<f16>(q4lo(w) * s.x + s.y); let hi = vec4<f16>(q4hi(w) * s.x + s.y);\n`;
      for (let i = 0; i < TM; i++) body += `      a${i}_${j} += f32(dot(lo, xl${i})) + f32(dot(hi, xh${i}));\n`;
      body += '    }\n';
    }
  } else if (h && kind === 'q8') {
    for (let i = 0; i < TM; i++) body += `    let x${i} = XH(xr${i} + u);\n`;
    for (let j = 0; j < 4; j++) {
      body += `    { let s = SM[n${j} * gk + u / 32u]; let c = vec4<f16>(i8x4(W[n${j} * nw + u]) * s.x + s.y);\n`;
      for (let i = 0; i < TM; i++) body += `      a${i}_${j} += f32(dot(c, x${i}));\n`;
      body += '    }\n';
    }
  } else if (kind === 'q4') {
    body = `let g = u / 8u; let o = g * 16u + u % 8u;\n`;
    for (let i = 0; i < TM; i++) body += `    let xl${i} = X(xr${i} + o); let xh${i} = X(xr${i} + o + 8u); let xs${i} = dot(xl${i} + xh${i}, vec4<f32>(1.0));\n`;
    for (let j = 0; j < 4; j++) {
      body += `    { let w = W[n${j} * nw + u]; let s = SM[n${j} * gk + g]; let lo = q4lo(w); let hi = q4hi(w);\n`;
      for (let i = 0; i < TM; i++) body += `      a${i}_${j} += s.x * (dot(lo, xl${i}) + dot(hi, xh${i})) + s.y * xs${i};\n`;
      body += '    }\n';
    }
  } else if (kind === 'q8') {
    for (let i = 0; i < TM; i++) body += `    let x${i} = X(xr${i} + u); let xs${i} = dot(x${i}, vec4<f32>(1.0));\n`;
    for (let j = 0; j < 4; j++) {
      body += `    { let s = SM[n${j} * gk + u / 32u]; let c = i8x4(W[n${j} * nw + u]);\n`;
      for (let i = 0; i < TM; i++) body += `      a${i}_${j} += s.x * dot(c, x${i}) + s.y * xs${i};\n`;
      body += '    }\n';
    }
  } else {
    for (let i = 0; i < TM; i++) body += `    let x${i} = X(xr${i} + u);\n`;
    for (let j = 0; j < 4; j++) {
      body += `    { let w = W[n${j} * nw + u];\n`;
      for (let i = 0; i < TM; i++) body += `      a${i}_${j} += dot(w, x${i});\n`;
      body += '    }\n';
    }
  }
  for (let i = 0; i < TM; i++)
    out += `if (m${i} < p.M && nb + 4u * tx < p.N) {
    let n4 = (nb + 4u * tx) / 4u;
    let v = vec4<f32>(a${i}_0, a${i}_1, a${i}_2, a${i}_3);
    ${epilogue(`(m${i} * p.ys + nb + 4u * tx) / 4u`, 'v')}
  }\n  `;
  return `${linHead(kind, h)}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let tx = li % 8u;
  let ty = li / 8u;
  let mb = wg.y * ${8 * TM}u;
  let nb = wg.x * 32u;
  let nw = ${unitsPerRow(kind)};
  let gk = ${groupsPerRow(kind)};
  ${idx}
  ${decl}
  for (var u = 0u; u < nw; u++) {
${body}
  }
  ${out}
}`;
}

// rows of C normalised: RMS with weight (rms) or layer norm (ln: eps 1e-5, biased variance) with optional weight /
// bias; then optional modulation y * (1 + MOD[sc + k]) + MOD[sh + k] and mish. Offsets in vec4 units.
export function normShader(kind, h = false) {
  return `${h ? 'enable f16;' : ''}${COMMON}
struct Prm { C: u32, eps: f32, hasw: u32, hasb: u32, xs: u32, ys: u32, act: u32, modu: u32, sh: u32, sc: u32, xoff: u32, yoff: u32 }
@group(0) @binding(0) var<uniform> p: Prm;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> NW: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> NB: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Y: array<vec4<${h ? 'f16' : 'f32'}>>;
@group(0) @binding(5) var<storage, read> MOD: array<vec4<f32>>;
var<workgroup> red: array<f32, 64>;
fn total(li: u32, v: f32) -> f32 {
  workgroupBarrier();
  red[li] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) { if (li < s) { red[li] = red[li] + red[li + s]; } workgroupBarrier(); }
  return red[0];
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let nv = p.C / 4u;
  let xb = p.xoff + wg.x * p.xs / 4u;
  let yb = p.yoff + wg.x * p.ys / 4u;
  ${kind === 'ln' ? `var s1 = 0.0;
  for (var v = li; v < nv; v += 64u) { s1 += dot(X[xb + v], vec4<f32>(1.0)); }
  let mean = total(li, s1) / f32(p.C);
  var s2 = 0.0;
  for (var v = li; v < nv; v += 64u) { let d = X[xb + v] - vec4<f32>(mean); s2 += dot(d, d); }
  let r = 1.0 / sqrt(total(li, s2) / f32(p.C) + 1e-5);` : `var s2 = 0.0;
  for (var v = li; v < nv; v += 64u) { let x = X[xb + v]; s2 += dot(x, x); }
  let mean = 0.0;
  let r = 1.0 / sqrt(total(li, s2) / f32(p.C) + p.eps);`}
  for (var v = li; v < nv; v += 64u) {
    var y = (X[xb + v] - vec4<f32>(mean)) * r;
    if (p.hasw != 0u) { y = y * NW[v]; }
    if (p.hasb != 0u) { y = y + NB[v]; }
    if (p.modu != 0u) { y = y * (vec4<f32>(1.0) + MOD[p.sc + v]) + MOD[p.sh + v]; }
    if (p.act == 1u) { y = vec4<f32>(mish(y.x), mish(y.y), mish(y.z), mish(y.w)); }
    Y[yb + v] = vec4<${h ? 'f16' : 'f32'}>(y);
  }
}`;
}

// per token m and head i (0..nq+2nkv-1): q / k: RMS norm with q_norm / k_norm, RoPE (pairs d, d + 64; cos / sin of the
// token), q stays in QKV, k -> KV cache; v -> KV cache. 64 threads: dims t and t + 64.
export const ROPE = `${COMMON}
struct Prm { nq: u32, nkv: u32, S: u32, layer: u32, eps: f32, W: u32, s0: u32, s1: u32 }
@group(0) @binding(0) var<uniform> p: Prm;
@group(0) @binding(1) var<storage, read_write> QKV: array<f32>;
@group(0) @binding(2) var<storage, read> QN: array<f32>;
@group(0) @binding(3) var<storage, read> KN: array<f32>;
@group(0) @binding(4) var<storage, read> CS: array<f32>;     // [m][128]: cos 0..63, sin 64..127
@group(0) @binding(5) var<storage, read> TOK: array<vec4<u32>>; // [m]: write row, r0, r1
@group(0) @binding(6) var<storage, read_write> KV: array<u32>;   // float16 pairs
var<workgroup> red: array<f32, 64>;
fn put(at: u32, a: f32, b: f32) { KV[at / 2u] = pack2x16float(vec2<f32>(a, b)); }
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let i = wg.x;
  let m = wg.y;
  let base = m * p.W + i * 128u;
  var a = QKV[base + t];
  var b = QKV[base + t + 64u];
  let isq = i < p.nq;
  let isk = i >= p.nq && i < p.nq + p.nkv;
  red[t] = a * a + b * b;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  if (isq || isk) {
    let r = 1.0 / sqrt(red[0] / 128.0 + p.eps);
    let wa = select(KN[t], QN[t], isq);
    let wb = select(KN[t + 64u], QN[t + 64u], isq);
    a = a * r * wa;
    b = b * r * wb;
    let c = CS[m * 128u + t];
    let s = CS[m * 128u + 64u + t];
    let ra = a * c - b * s;
    let rb = b * c + a * s;
    a = ra;
    b = rb;
  }
  if (isq) { QKV[base + t] = a; QKV[base + t + 64u] = b; return; }
  // cache row: [layer][k|v][kv head][S][128]; dims t and t+64 go out as halves of two pairs: neighbours swap via red
  let kvi = select(1u, 0u, isk);
  let h = select(i - p.nq - p.nkv, i - p.nq, isk);
  let row = TOK[m].x;
  let at = ((kvi * p.nkv + h) * p.S + row) * 128u;  // this layer's cache buffer: [k|v][kv head][S][128]
  workgroupBarrier();
  red[t] = a;
  workgroupBarrier();
  if (t % 2u == 0u) { put(at + t, a, red[t + 1u]); }
  workgroupBarrier();
  red[t] = b;
  workgroupBarrier();
  if (t % 2u == 0u) { put(at + 64u + t, b, red[t + 1u]); }
}`;

// one token m, one query head (64 threads): online softmax over cache rows r0..r1-1 in blocks of 64; out [m][qh][128]
export const ATTN = (h = false) => `${h ? 'enable f16;' : ''}
struct Prm { nq: u32, nkv: u32, S: u32, layer: u32, W: u32, AW: u32, scale: f32, s0: u32 }
@group(0) @binding(0) var<uniform> p: Prm;
@group(0) @binding(1) var<storage, read> QKV: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> KV: array<vec2<u32>>;   // 4 halves
@group(0) @binding(3) var<storage, read> TOK: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> A: array<${h ? 'f16' : 'f32'}>;
var<workgroup> q: array<vec4<f32>, 32>;
var<workgroup> pr: array<f32, 64>;
var<workgroup> red: array<f32, 64>;
fn ld(i: u32) -> vec4<f32> { let w = KV[i]; return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let qh = wg.x;
  let m = wg.y;
  let h = qh / (p.nq / p.nkv);
  if (t < 32u) { q[t] = QKV[(m * p.W + qh * 128u) / 4u + t]; }
  let r0 = TOK[m].y;
  let r1 = TOK[m].z;
  let kb = h * p.S * 32u;                        // vec4 units of K rows of this head (this layer's buffer)
  let vb = (p.nkv + h) * p.S * 128u;             // float index of V rows
  var mx = -3.0e38;
  var sum = 0.0;
  var o0 = 0.0;
  var o1 = 0.0;
  workgroupBarrier();
  for (var j0 = r0; j0 < r1; j0 += 64u) {
    let j = j0 + t;
    var sc = -3.0e38;
    if (j < r1) {
      var acc = 0.0;
      for (var d = 0u; d < 32u; d++) { acc += dot(q[d], ld(kb + j * 32u + d)); }
      sc = acc * p.scale;
    }
    red[t] = sc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) { if (t < s) { red[t] = max(red[t], red[t + s]); } workgroupBarrier(); }
    let nmx = max(mx, red[0]);
    let e = select(0.0, exp(sc - nmx), j < r1);
    pr[t] = e;
    workgroupBarrier();
    red[t] = e;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
    let f = exp(mx - nmx);
    sum = sum * f + red[0];
    o0 = o0 * f;
    o1 = o1 * f;
    let n = min(64u, r1 - j0);
    for (var k = 0u; k < n; k++) {
      let w = pr[k];
      let vr = vb + (j0 + k) * 128u;
      let v2 = KV[(vr + 2u * t) / 4u];
      let pair = select(unpack2x16float(v2.x), unpack2x16float(v2.y), (t % 2u) == 1u);
      o0 += w * pair.x;
      o1 += w * pair.y;
    }
    mx = nmx;
    workgroupBarrier();
  }
  let ob = m * p.AW + qh * 128u + 2u * t;
  A[ob] = ${h ? 'f16' : ''}(o0 / sum);
  A[ob + 1u] = ${h ? 'f16' : ''}(o1 / sum);
}`;

// a[m][k] = silu(g[m][k]) * u[m][k] (g = G[m][k], u = G[m][I + k])
export const SILU = (h = false) => `${h ? 'enable f16;' : ''}${COMMON}
struct Prm { M: u32, I: u32, gs: u32, s0: u32 }
@group(0) @binding(0) var<uniform> p: Prm;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<${h ? 'f16' : 'f32'}>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let k = g.x;
  let m = g.y;
  if (k < p.I) { let x = G[m * p.gs + k]; Y[m * p.I + k] = ${h ? 'f16' : ''}(silu(x) * G[m * p.gs + p.I + k]); }
}`;

export class LMGpu {
  // opts: S (cache rows), MAXM (largest batch of tokens per pass), TM (gemm register rows)
  constructor(device, file, opts = {}) {
    this.dev = device;
    // S: cache rows; MAXM: tokens per pass; f16: linear inputs (norm, attention, SiLU rows) float16
    this.o = { S: 2048, MAXM: 512, TM: 4, f16: false, ...opts };
    this.f = parseTTLM(file);
    this.pipes = new Map();
    this.bytes = 0;
    this.all = [];
    this.build();
    this.f = null;  // the weights are on the GPU: no reference to the file left (the page can drop its copy)
  }
  buf(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    const b = this.dev.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
    this.bytes += b.size;
    this.all.push(b);
    return b;
  }
  upload(data) {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const b = this.buf(data.byteLength);
    if (bytes.byteLength % 4) { const c = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4); c.set(bytes); this.dev.queue.writeBuffer(b, 0, c); }
    else this.dev.queue.writeBuffer(b, 0, bytes);
    return b;
  }
  pipe(key, code) {
    if (!this.pipes.has(key)) {
      const module = this.dev.createShaderModule({ code });
      this.pipes.set(key, this.dev.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } }));
    }
    return this.pipes.get(key);
  }
  // a linear layer of a TTLM file: q4 (u8 codes [N][G][32]), q8 (i8 [N][G][128]) or f32 (.w [N][K])
  static lin(self, f, name) {
    if (f.has(name + '.w')) {
      const w = f.get(name + '.w');
      return { kind: 'f32', N: w.dims[0], K: w.dims[1], W: self.upload(w.data), B: f.has(name + '.bias') ? self.upload(f.get(name + '.bias').data) : null };
    }
    const c = f.get(name + '.codes'), s = f.get(name + '.scales').data, m = f.get(name + '.mins').data;
    const [N, G] = c.dims, kind = c.dt === 0 ? 'q4' : 'q8';
    const sm = new Float32Array(2 * N * G);
    for (let i = 0; i < N * G; i++) { sm[2 * i] = F16[s[i]]; sm[2 * i + 1] = F16[m[i]]; }
    return { kind, N, K: G * (kind === 'q4' ? 64 : 128), W: self.upload(c.data), SM: self.upload(sm),
             B: f.has(name + '.bias') ? self.upload(f.get(name + '.bias').data) : null };
  }
  build() {
    const f = this.f, c = f.get('cfg').data, cf = f.get('cfg_f').data;
    this.H = c[0]; this.NL = c[1]; this.nq = c[2]; this.nkv = c[3]; this.I = c[5];
    this.eps = cf[0]; this.theta = cf[1];
    this.WQ = (this.nq + 2 * this.nkv) * 128;
    this.L = [];
    for (let l = 0; l < this.NL; l++) {
      const p = `L${l}.`, g = (n) => this.upload(f.get(p + n).data);
      this.L.push({ in_norm: g('in_norm'), post_norm: g('post_norm'), q_norm: g('q_norm'), k_norm: g('k_norm'),
                    qkv: LMGpu.lin(this, f, p + 'qkv'), o: LMGpu.lin(this, f, p + 'o'),
                    gate_up: LMGpu.lin(this, f, p + 'gate_up'), down: LMGpu.lin(this, f, p + 'down') });
    }
    this.norm = this.upload(f.get('norm').data);
    this.weightBytes = this.bytes;
    const { S, MAXM } = this.o, H = this.H;
    // one cache buffer per layer ([k|v][kv head][S][128] float16): S = 4096 is 16 MiB each, inside the default
    // 128 MiB binding limit of phone GPUs
    this.KV = Array.from({ length: this.NL }, () => this.buf(2 * this.nkv * S * 128 * 2));
    this.X = this.buf(4 * MAXM * H);
    this.HN = this.buf(4 * MAXM * H);
    this.QKV = this.buf(4 * MAXM * this.WQ);
    this.A = this.buf(4 * MAXM * this.nq * 128);
    this.G = this.buf(4 * MAXM * 2 * this.I);
    this.ACT = this.buf(4 * MAXM * this.I);
    this.CS = this.buf(4 * MAXM * 128);
    this.TOK = this.buf(16 * MAXM);
    this.ZERO = this.buf(4 * Math.max(2 * this.I, this.WQ));
    this.PRM = this.dev.createBuffer({ size: 256 * 4096, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.plans = new Map();
  }
  bg(pl, bufs) {
    return this.dev.createBindGroup({
      layout: pl.getBindGroupLayout(0),
      entries: Object.entries(bufs).map(([b, r]) => ({ binding: Number(b), resource: r.buffer ? r : { buffer: r } })),
    });
  }
  // dispatch list of one pass of M tokens through every layer (made once per M; parameters in PRM slots)
  plan(M) {
    if (this.plans.has(M)) { const r = this.plans.get(M); this.plans.delete(M); this.plans.set(M, r); return r; }  // most recent last
    const D = [], prm = new ArrayBuffer(256 * 512), pdv = new DataView(prm);
    let slot = 0;
    // 8 parameter regions; when all are used the least recently used plan goes (another M: another plan)
    let base;
    if (this.plans.size < 8) base = 256 * 512 * this.plans.size;
    else { const [k, old] = this.plans.entries().next().value; this.plans.delete(k); base = old.base; }
    const param = (vals) => {
      const off = 256 * slot++;
      vals.forEach((v, i) => (typeof v === 'object' ? pdv.setFloat32(off + 4 * i, v.f, true) : pdv.setUint32(off + 4 * i, v, true)));
      return { buffer: this.PRM, offset: base + off, size: 48 };
    };
    const lin = (L, X, xs, Y, ys, add) => {
      const TM = this.o.TM;
      let pl, gx, gy;
      const h = this.o.f16;
      if (M <= 8) {
        pl = this.pipe(`gemv${L.kind}${M}${h}`, gemvShader(L.kind, M, h));
        gx = L.N / 4; gy = 1;
      } else {
        pl = this.pipe(`gemm${L.kind}${TM}${h}`, gemmShader(L.kind, TM, h));
        gx = Math.ceil(L.N / 32); gy = Math.ceil(M / (8 * TM));
      }
      const b = { 0: param([M, L.K, L.N, xs, ys, gx, L.B ? 1 : 0, add ? 1 : 0, 0, 0, 0, 0]), 1: X, 2: L.W, 4: L.B || this.ZERO, 5: Y, 6: this.ZERO };
      if (L.kind !== 'f32') b[3] = L.SM;
      D.push({ pl, bg: this.bg(pl, b), x: gx, y: gy });
    };
    const rms = (X, w, Y) => {
      const pl = this.pipe(`rms${this.o.f16}`, normShader('rms', this.o.f16));
      D.push({ pl, bg: this.bg(pl, { 0: param([this.H, { f: this.eps }, 1, 0, this.H, this.H, 0, 0, 0, 0, 0, 0]), 1: X, 2: w, 3: this.ZERO, 4: Y, 5: this.ZERO }), x: M });
    };
    const H = this.H;
    const L0 = [];  // index in D where each layer starts
    for (let l = 0; l < this.NL; l++) {
      const L = this.L[l];
      L0.push(D.length);
      rms(this.X, L.in_norm, this.HN);
      lin(L.qkv, this.HN, H, this.QKV, this.WQ, false);
      let pl = this.pipe('rope', ROPE);
      D.push({ pl, bg: this.bg(pl, { 0: param([this.nq, this.nkv, this.o.S, l, { f: this.eps }, this.WQ, 0, 0]), 1: this.QKV, 2: L.q_norm,
                                     3: L.k_norm, 4: this.CS, 5: this.TOK, 6: this.KV[l] }), x: this.nq + 2 * this.nkv, y: M });
      pl = this.pipe(`attn${this.o.f16}`, ATTN(this.o.f16));
      D.push({ pl, bg: this.bg(pl, { 0: param([this.nq, this.nkv, this.o.S, l, this.WQ, this.nq * 128, { f: 1 / Math.sqrt(128) }, 0]),
                                     1: this.QKV, 2: this.KV[l], 3: this.TOK, 4: this.A }), x: this.nq, y: M });
      lin(L.o, this.A, this.nq * 128, this.X, H, true);
      rms(this.X, L.post_norm, this.HN);
      lin(L.gate_up, this.HN, H, this.G, 2 * this.I, false);
      pl = this.pipe(`silu${this.o.f16}`, SILU(this.o.f16));
      D.push({ pl, bg: this.bg(pl, { 0: param([M, this.I, 2 * this.I, 0]), 1: this.G, 2: this.ACT }), x: Math.ceil(this.I / 256), y: M });
      lin(L.down, this.ACT, this.I, this.X, H, true);
    }
    this.dev.queue.writeBuffer(this.PRM, base, prm, 0, 256 * slot);
    L0.push(D.length);
    const r = { D, M, base, L0 };
    this.plans.set(M, r);
    return r;
  }
  // RoPE tables of the tokens as the CPU computes them (float angles, float cos / sin)
  cosSin(pos) {
    const M = pos.length, cs = new Float32Array(M * 128), fr = Math.fround;
    for (let m = 0; m < M; m++)
      for (let i = 0; i < 64; i++) {
        const invf = fr(1 / fr(Math.pow(this.theta, fr(fr(2 * i) / 128))));
        const ang = fr(fr(pos[m]) * invf);
        cs[m * 128 + i] = Math.cos(ang);
        cs[m * 128 + 64 + i] = Math.sin(ang);
      }
    return cs;
  }
  // n tokens (emb [n][H] float32, pos [n]) at cache rows row0.., each attending rows a0 .. its own row; passes of at
  // most MAXM tokens (same function as the CPU's batches of 8). Leaves X with the last pass's hidden rows.
  feed(emb, pos, row0, a0, enc = null) {
    const n = pos.length, MAXM = this.o.MAXM, q = this.dev.queue;
    for (let i = 0; i < n; i += MAXM) {
      const M = Math.min(MAXM, n - i), pl = this.plan(M);
      const tok = new Uint32Array(4 * M);
      for (let m = 0; m < M; m++) { tok[4 * m] = row0 + i + m; tok[4 * m + 1] = a0; tok[4 * m + 2] = row0 + i + m + 1; }
      q.writeBuffer(this.X, 0, emb, i * this.H, M * this.H);
      q.writeBuffer(this.CS, 0, this.cosSin(pos.subarray(i, i + M)));
      q.writeBuffer(this.TOK, 0, tok);
      const e = this.dev.createCommandEncoder(), pass = e.beginComputePass();
      for (const d of pl.D) { pass.setPipeline(d.pl); pass.setBindGroup(0, d.bg); pass.dispatchWorkgroups(d.x, d.y || 1, d.z || 1); }
      pass.end();
      q.submit([e.finish()]);
      this.lastM = M;
    }
  }
  // A prefix read (rows 0..n-1, each attending from row 0) with its cache rows read back while it still computes:
  // the last pass in `groups` submits of a few layers, each followed by the copy of those layers' rows into its own
  // buffer; the read backs of the first groups overlap the compute of the later ones. put(kv, l0, l1, hid) gets the
  // rows of layers l0..l1-1 ([layer][kv][head][row][128] float16 bits) in layer order, hid with the last group.
  async feedRead(emb, pos, hid, put, groups = 4) {
    const n = pos.length, MAXM = this.o.MAXM, q = this.dev.queue, per = n * 256, kvl = 2 * this.nkv;
    const last = Math.floor((n - 1) / MAXM) * MAXM;
    if (last > 0) this.feed(emb.subarray(0, last * this.H), pos.subarray(0, last), 0, 0);
    const M = n - last, pl = this.plan(M);
    const tok = new Uint32Array(4 * M);
    for (let m = 0; m < M; m++) { tok[4 * m] = last + m; tok[4 * m + 2] = last + m + 1; }
    q.writeBuffer(this.X, 0, emb, last * this.H, M * this.H);
    q.writeBuffer(this.CS, 0, this.cosSin(pos.subarray(last, n)));
    q.writeBuffer(this.TOK, 0, tok);
    this.lastM = M;
    const parts = [];
    for (let g = 0; g < groups; g++) {
      const l0 = Math.floor((g * this.NL) / groups), l1 = Math.floor(((g + 1) * this.NL) / groups), hb = hid && g === groups - 1 ? 4 * this.H : 0;
      const size = hb + (l1 - l0) * kvl * per;
      const rb = this.dev.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = this.dev.createCommandEncoder(), pass = e.beginComputePass();
      for (const d of pl.D.slice(pl.L0[l0], pl.L0[l1])) { pass.setPipeline(d.pl); pass.setBindGroup(0, d.bg); pass.dispatchWorkgroups(d.x, d.y || 1, d.z || 1); }
      pass.end();
      for (let l = l0; l < l1; l++)
        for (let k = 0; k < kvl; k++) e.copyBufferToBuffer(this.KV[l], k * this.o.S * 256, rb, hb + ((l - l0) * kvl + k) * per, per);
      if (hb) e.copyBufferToBuffer(this.X, 4 * (M - 1) * this.H, rb, 0, hb);
      q.submit([e.finish()]);
      parts.push({ rb, l0, l1, hb, size, p: rb.mapAsync(GPUMapMode.READ) });
    }
    let err = null;
    for (const t of parts) {
      try {
        await t.p;
        if (!err) {
          const m = t.rb.getMappedRange();
          put(new Uint16Array(m, t.hb, (t.size - t.hb) / 2), t.l0, t.l1, t.hb ? new Float32Array(m, 0, this.H) : null);
        }
      } catch (x) { err = err || x; }
      t.rb.destroy();
    }
    if (err) throw err;
  }
  destroy() { for (const b of [...this.all, this.PRM]) b.destroy(); this.all = []; }
}
