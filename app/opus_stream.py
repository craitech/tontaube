"""Opus streaming encoder for WebSocket audio delivery.

Encodes audio chunks into Opus frames (20ms each) with a simple binary
framing protocol for WebSocket transport:

  [frame_count: uint16_le] [len1: uint16_le] [opus_data_1] [len2: uint16_le] [opus_data_2] ...

The client uses WebCodecs AudioDecoder to decode each frame individually.
"""
import struct

import numpy as np
import opuslib

from app.stream_writers import pcm16

# Opus frame duration in seconds — 20ms is the standard for real-time audio
FRAME_DURATION = 0.02


class OpusStreamEncoder:
    """Stateful Opus encoder for streaming audio chunks."""

    def __init__(self, sample_rate: int = 24000, bitrate: int = 32000):
        self._sr = sample_rate
        self._frame_samples = int(sample_rate * FRAME_DURATION)
        self._encoder = opuslib.Encoder(sample_rate, 1, opuslib.APPLICATION_AUDIO)
        self._encoder.bitrate = bitrate
        self._rest = np.zeros(0, np.int16)

    def encode(self, audio) -> bytes:
        """Encode a waveform chunk (torch tensor or numpy array) to a framed Opus message.

        Samples that do not fill a 20 ms frame wait for the next call; flush()
        sends them at the end. Returns a binary blob ready to send over WebSocket.
        """
        pcm = np.concatenate([self._rest, pcm16(audio)])
        fs = self._frame_samples
        n_frames = len(pcm) // fs
        self._rest = pcm[n_frames * fs:]
        return self._pack([self._encoder.encode(pcm[i * fs:(i + 1) * fs].tobytes(), fs) for i in range(n_frames)])

    def flush(self) -> bytes:
        """The samples left over from encode(), zero-padded to one frame; b"" if there are none."""
        if not len(self._rest):
            return b""
        padded = np.zeros(self._frame_samples, dtype=np.int16)
        padded[:len(self._rest)] = self._rest
        self._rest = self._rest[:0]
        return self._pack([self._encoder.encode(padded.tobytes(), self._frame_samples)])

    @staticmethod
    def _pack(frames: list[bytes]) -> bytes:
        """[frame_count:u16] [len:u16 data]..."""
        parts = [struct.pack('<H', len(frames))]
        for frame in frames:
            parts.append(struct.pack('<H', len(frame)))
            parts.append(frame)
        return b''.join(parts)
