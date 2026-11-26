# TontaubeV1.1 Flash

TontaubeV1.1 Flash is a small, fast English text-to-speech model by Tontaube
with one voice, Marcus. It streams speech at about 25-30x real time on an
NVIDIA GPU, with the first audio after about 10 ms, and at about 2-4x real
time on a CPU. It also runs inside Chrome on your own computer.

[Try it in your browser](https://huggingface.co/spaces/TontaubeAI/TontaubeV1.1-Flash) ·
[Model weights](https://huggingface.co/TontaubeAI/TontaubeV1.1-Flash) ·
[TontaubeV1.1](../README.md)

| | Tested on | Speed |
|---|---|---|
| GPU engine (command line, server) | Linux with an NVIDIA GPU | about 25-30x real time per stream, over 100x across many streams at once (RTX 3090); first audio after about 10 ms |
| CPU engine (command line, server) | Linux, Windows, macOS (Apple silicon) | about 2-4x real time, depending on the CPU |
| Chrome extension | Linux, Windows | about 2x real time on a desktop, 3-6x with its graphics card |
| [Demo page](https://huggingface.co/spaces/TontaubeAI/TontaubeV1.1-Flash) on a phone | Chrome on Android | about real time (Pixel 8) |
| Phone engine (native) | in development | TBD |

The weights are a download of about 700 MB.

All of them read written English aloud, with a built-in rule verbalizer that
turns numbers, dates, times, money, units and abbreviations into words. All
start playback at the first sound and redraw the rare moments where the pitch
jumps (see [How it works](#how-it-works)).

## Quick start

### Linux

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/); for the GPU,
an NVIDIA GPU (the repository's environment brings PyTorch).

```bash
git clone --branch v1.2.0 --depth 1 https://github.com/craitech/tontaube.git
cd tontaube
uv run python flash/speak.py "Dr. Weber's invoice from 5/9/2025 lists 3 items." --wav out.wav
uv run python flash/speak.py --device cpu "Dr. Weber's invoice from 5/9/2025 lists 3 items." --wav out.wav
```

The first run downloads the weights. On the CPU it also downloads the engine,
which comes prebuilt with each release; a modified checkout, or
`FLASH_BUILD_FROM_SOURCE=1`, builds it with a C++17 compiler (`g++`) instead.

On a machine without an NVIDIA GPU, `--extra cpu` gives the environment
PyTorch's much smaller CPU-only build. Pass it to every `uv run` command:

```bash
uv run --extra cpu python flash/speak.py --device cpu "Dr. Weber's invoice from 5/9/2025 lists 3 items." --wav out.wav
uv run --extra cpu tontaube serve --model flash --device cpu
```

### Windows

Flash runs natively on Windows (x86-64) on the CPU, with the same commands in
PowerShell. Install [uv](https://docs.astral.sh/uv/) and Git, then:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone --branch v1.2.0 --depth 1 https://github.com/craitech/tontaube.git
cd tontaube
uv run python flash/speak.py --device cpu "Dr. Weber's invoice from 5/9/2025 lists 3 items." --wav out.wav
uv run tontaube serve --model flash --device cpu
```

On Windows the environment holds only what Flash needs on the CPU, and the
engine is downloaded with the weights on first use. To build it from source
instead (a modified checkout, or `FLASH_BUILD_FROM_SOURCE=1`), install LLVM
MinGW (`winget install MartinStorsjo.LLVM-MinGW.UCRT`) and open a new terminal.
WAV files and the server's MP3 stream need nothing more. MP3 and Opus files
from `/predict` need FFmpeg (`winget install Gyan.FFmpeg`), and `/ws/stream`
needs the Opus library (`opus.dll`).

### macOS

Flash runs natively on Macs with Apple silicon (M1 or newer, macOS 11 or
newer) on the CPU, with the same commands in the Terminal. Install Apple's
command line tools (`xcode-select --install`, for Git) and
[uv](https://docs.astral.sh/uv/), then:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --branch v1.2.0 --depth 1 https://github.com/craitech/tontaube.git
cd tontaube
uv run python flash/speak.py --device cpu "Dr. Weber's invoice from 5/9/2025 lists 3 items." --wav out.wav
uv run tontaube serve --model flash --device cpu
```

As on Windows, the environment holds only what Flash needs on the CPU, and the
engine is downloaded with the weights on first use; a modified checkout, or
`FLASH_BUILD_FROM_SOURCE=1`, builds it with the command line tools instead. On
an M1 it runs at about 2-3x real time. WAV files and the server's MP3 stream
need nothing more. MP3 and Opus files from `/predict` need FFmpeg
(`brew install ffmpeg`). `/ws/stream` is not available on macOS.

### Options and Python

`flash/speak.py` options:

| Option | Meaning |
|---|---|
| `--device gpu\|cpu` | where it runs; the GPU when there is one |
| `--variant int4\|int8\|fp16` | weight format on the GPU; `int4` is the default and the smallest |
| `--rate 48000\|24000` | the output sample rate; 48 kHz is the default, 24 kHz is faster |
| `--lines FILE` | one text per line, the model loaded once; `--wav` is then a folder |
| `--seed N` | another seed gives another take |
| `--no-redraw` | keep flagged pitch jumps as drawn |
| `--cutoff C`, `--max-takes K` | how sensitive the redraw is (default 0.85) and how many takes a spot may get (default 4) |
| `--no-cut` | play the short silence before the first sound |
| `--weights DIR` | use a local copy of the weights instead of downloading them |
| `--threads N` | how many CPU cores to use (default: chosen automatically) |

From Python, in the repository folder:

```python
from flash.gpu.engine import Engine           # or flash.cpu.engine

eng = Engine("int4")                          # loads the weights (a few seconds)
for start, audio in eng.stream("Some text."):  # float32, 48 kHz
    player.put(start, audio)                  # audio for the player's samples from `start` on
wave = eng.speak("Some text.")                # everything at once
```

Both engines deliver 48 kHz audio; `Engine("int4", sample_rate=24000)` gives
24 kHz, which is faster. `stream` yields each 133 ms frame as soon as it
exists. After a redraw, `start` lies before the end of what came so far, and
the player replaces its audio from `start` on. A replacement never reaches
audio that has already been played, for a player that plays in real time from
the first piece; pass `played=` (a function returning the samples the player
has consumed) for an exact check. One request at a time per engine.

## Server

The repository's inference server hosts Flash with the same HTTP API as
TontaubeV1.1, on an NVIDIA GPU or on the CPU:

```bash
uv run tontaube serve --model flash --device gpu
uv run tontaube serve --model flash --device cpu
```

Without `--device` it uses the GPU when there is one. The first start
downloads the weights. When `/readyz` answers, the API is ready on
`http://127.0.0.1:8080`:

```bash
curl --silent http://127.0.0.1:8080/predict \
  -H 'content-type: application/json' \
  -d '{"text": "Dr. Weber'\''s invoice from 5/9/2025 lists 3 items.", "format": "wav"}' \
| python3 -c 'import base64,json,sys; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)["audio_b64"]))' \
> flash.wav
```

| Route | Returns |
|---|---|
| `POST /predict` | the whole utterance as WAV, MP3 or Opus |
| `POST /stream` | audio chunks as server-sent events: raw PCM16 by default, `"format": "mp3"` or `"opus"` selectable |
| `WS /ws/stream` | Opus frames of 20 ms |
| `POST /v1/audio/speech` | OpenAI's text-to-speech API (see [OpenAI-compatible API](../README.md#openai-compatible-api)) |

The server delivers 48 kHz audio; `--rate 24000` gives 24 kHz, which is
faster.

A request takes `text` (one string or a list of strings) and, optionally,
`seed`, `temperature` (default 0.7), `format`, `bitrate`, `priority`,
`trim_silence_padding_ms` and `use_verbalization`. The built-in verbalizer
reads numbers, dates and so on unless `use_verbalization` is `false`.
TontaubeV1.1's voice, style and slider fields are refused with status 422. The
browser interface (`uv run tontaube ui`) works with a Flash server too and
switches those controls off.

Streams only ever append. Audio goes out once a redraw can no longer change
it, so a client plays it like any other stream. Without a `seed`, every
request is a new take.

On the GPU the server answers up to 64 requests at once in batches, over 100x
real time combined on an RTX 3090. On the CPU it answers one request at a
time, and further requests wait. `FLASH_ENGINES` sets the number of requests
answered at once, and `FLASH_MODEL_PATH` points to a local copy of the
weights. API keys and CORS work as for TontaubeV1.1 (see the
[configuration](../README.md#configuration)).

For CPU hosting, `Dockerfile.cpu` builds a small image without CUDA, vLLM or
the TontaubeV1.1 models:

```bash
docker build -f Dockerfile.cpu -t tontaube-flash-cpu:1.2.0 .
docker run --rm -p 127.0.0.1:8080:8080 -v tontaube-cache:/root/.cache tontaube-flash-cpu:1.2.0
```

The GPU image of the main [Docker](../README.md#docker) section serves Flash
with `-e TTS_MODEL=flash`.

## Browser extension

The Chrome extension reads the current page or the selected text aloud.
Everything runs on your computer; no text leaves the browser.

1. Open `chrome://extensions`, switch on *Developer mode*, choose *Load
   unpacked* and select the `flash/extension` folder of this repository.
2. Click the toolbar icon and choose *Read page* or *Read selection*, or press
   Alt+Shift+R (read) and Alt+Shift+S (stop).

The first use downloads the model (about 700 MB) from Hugging Face and keeps it in
the browser's storage; later starts load it in about a second.

The popup's *GPU* setting decides whether the extension also uses the
computer's graphics card (through WebGPU) next to the CPU:

- *Auto* (default) uses it where a short check on first use shows the same
  audio as on the CPU and a faster result;
- *On* always uses it;
- *Off* runs on the CPU alone.

The popup shows whether the graphics card is in use, and if not, why, with a
short guide. On Linux, Chrome offers the graphics card to web pages only after
enabling *Vulkan* and *Default ANGLE Vulkan* in `chrome://flags` and relaunching
Chrome. Where that is not enough, starting Chrome with
`google-chrome-stable --enable-features=Vulkan --use-angle=vulkan` works.
Without the graphics card the extension runs on the CPU.

The *48 kHz* setting raises the audio from the model's 24 kHz to 48 kHz,
which takes extra time. On first use the extension switches it on when the
computer is fast enough, and it can be changed at any time. The *Redraw pitch
jumps* setting switches the redraw on or off (see
[How it works](#how-it-works)).

To rebuild the WebAssembly engine from source, see
[`flash/web`](web/README.md).

## How it works

TontaubeV1.1 Flash speaks through its own codec, an autoencoder built for its
single voice that encodes the voice's VibeVoice latents into 16 values per
frame at 7.5 Hz. A Qwen3-0.6B backbone reads the text character by character
and predicts these 16 values one frame at a time with a small sampling head.
The codec's decoder expands them back into VibeVoice latents, and the
VibeVoice decoder turns those into 24 kHz audio, which an upsampler (AP-BWE)
raises to 48 kHz. Long texts are read in chunks
of up to a few sentences, each with the previous chunk as context.

- **Redraw.** A small detector listens to the audio while it streams. Where
  the pitch jumps to another note, the engine goes back one frame and draws
  those frames again before they are played, up to three times, and keeps the
  best take. This adds no latency. On slow devices a jump in the first second
  of a sentence may play before a redraw could replace it.
- **Start at the first sound.** The model begins each request with a short
  silence, which playback skips.

## License

The code in this folder is part of this repository and licensed under the
Apache License 2.0. The TontaubeV1.1 Flash weights are released under the
Apache License 2.0 as well; see the
[model card](https://huggingface.co/TontaubeAI/TontaubeV1.1-Flash) for their
notices.
