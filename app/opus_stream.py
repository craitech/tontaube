"""Opus streaming encoder for WebSocket audio delivery.

Encodes audio chunks into Opus frames (20ms each) with a simple binary
framing protocol for WebSocket transport:

  [frame_count: uint16_le] [len1: uint16_le] [opus_data_1] [len2: uint16_le] [opus_data_2] ...

The client uses WebCodecs AudioDecoder to decode each frame individually.
"""
import struct

import numpy as np
import opuslib

# Opus frame duration in seconds — 20ms is the standard for real-time audio
FRAME_DURATION = 0.02


class OpusStreamEncoder:
    """Stateful Opus encoder for streaming audio chunks."""

    def __init__(self, sample_rate: int = 24000, bitrate: int = 32000):
        self._sr = sample_rate
        self._frame_samples = int(sample_rate * FRAME_DURATION)
        self._encoder = opuslib.Encoder(sample_rate, 1, opuslib.APPLICATION_AUDIO)
        self._encoder.bitrate = bitrate

    def encode(self, audio_tensor) -> bytes:
        """Encode a torch audio tensor to a framed Opus message.

        Returns a binary blob ready to send over WebSocket.
        """
        pcm_f32 = np.clip(audio_tensor.squeeze().cpu().float().numpy(), -1.0, 1.0)
        pcm_s16 = (pcm_f32 * 32767).astype(np.int16)

        n = len(pcm_s16)
        fs = self._frame_samples
        n_frames = n // fs

        # Encode each 20ms frame
        frames: list[bytes] = []
        for i in range(n_frames):
            frame_pcm = pcm_s16[i * fs:(i + 1) * fs]
            opus_data = self._encoder.encode(frame_pcm.tobytes(), fs)
            frames.append(opus_data)

        # Handle leftover samples: pad to frame size
        remainder = n % fs
        if remainder > 0:
            padded = np.zeros(fs, dtype=np.int16)
            padded[:remainder] = pcm_s16[n_frames * fs:]
            opus_data = self._encoder.encode(padded.tobytes(), fs)
            frames.append(opus_data)

        # Pack: [frame_count:u16] [len:u16 data]...
        parts = [struct.pack('<H', len(frames))]
        for frame in frames:
            parts.append(struct.pack('<H', len(frame)))
            parts.append(frame)
        return b''.join(parts)
