# Browser engine (WebAssembly)

The CPU engine of the Chrome extension in `../extension/`: text -> rule verbalizer and chunker (`../text/`) -> frame
model -> VibeVoice decoder -> detector and redraw of flagged spots -> start at the first sound -> optional 48 kHz
upsampler -> audio, one 133 ms frame at a time, in WebAssembly with SIMD and threads.

## Build

```
./build.sh [EMSDK_DIR]
```

Builds `tts.{js,wasm}` (SIMD), `tts_relaxed.{js,wasm}` (+ relaxed SIMD) and `tts_scalar.{js,wasm}` into
`../extension/`. `EMSDK_DIR` is an installed and activated emsdk (tested with Emscripten 6.0.13); without it, `em++`
must be on `PATH`.

## Sources (`../engine/`)

| file | what |
|---|---|
| `tts_web.cpp` | the engine and its C API for the page (`tts_alloc`, `tts_register`, `tts_config`, `tts_load`, `tts_rate`, `tts_speak`, `tts_prepare`, `tts_sha256`); `tts_config(spec)` before `tts_load` sets speed options, all off by default (the `Cfg` comment in `tts_web.cpp` lists them) |
| `tt_engine.cpp`, `web_kernels.h`, `web_linear.h` | transformer steps on the CPU; WebAssembly SIMD kernels for the q4 / q8 linears and attention |
| `fm.cpp` | the frame model: chunk prefixes, flow head, autoencoder decoder, frame by frame with rewind support |
| `vv_dec.h` | streaming VibeVoice decoder with an exact causal state (saved and restored per frame for the redraw) |
| `pitch_detect.h`, `redraw.h` | streaming detector of pitch-jump spots and the redraw of flagged frames |
| `lead_cut.h` | starts a request 5 ms before its first sample above -78 dBFS |
| `bwe.h` | AP-BWE 24 -> 48 kHz, streaming |
| `sha256.h` | checksums of downloaded model files |

## Requests

`tts_speak(text, verbalize, use_bwe, seed, pace_hi, redraw, lead_cut, realtime)`; in the extension the engine worker
message `{ cmd: 'speak', text, seed, bwe, redraw, leadCut, realtime, ... }` or `Controller.speak(text, opts)`:

| option | default | meaning |
|---|---|---|
| `leadCut` | on | start at the request's first sound; off: from the first sample |
| `realtime` | on | the audio streams to the player as it is made (pacing; a redraw only replaces audio not yet played); off: nothing is played, every flagged spot gets all its redraws, and the whole audio is returned at the end (worker message `{ type: 'result', audio }`, `Controller.speak` resolves with `audio`) |
| `redraw` | on | off: no detector, no redraws |
| `bwe` | off | 48 kHz output through the upsampler (`bwe.bin`); the extension's *48 kHz* setting is switched on at first use when the device is fast enough |

Model files (`models.json` lists names, sizes and SHA-256): `lm.bin`, `misc.bin`, `vv.bin`, `detector.bin`, and
`bwe.bin` for 48 kHz.

## GPU (WebGPU)

The extension's *GPU* setting (`gpu` of `Controller.init`: `auto`, `on` or `off`; the extension's default is `auto`)
moves the VibeVoice decoder, and the text reading of the chunk prefixes except on phones, to the GPU through WebGPU
(`gpu_worker.js`, `vv_gpu.js`, `lm_gpu.js`); everything else stays in WebAssembly. `auto` uses the GPU only where a
short check on first use gives the same output as the CPU decoder and is faster; the result is kept per GPU. If the
GPU device is lost during a request, the engine continues on the CPU.
