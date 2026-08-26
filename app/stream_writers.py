"""Incremental MP3 encoding for SSE responses."""
import numpy as np


class Mp3StreamWriter:
    """Incrementally encodes audio chunks to a continuous MP3 stream.

    Unlike OGG, partial MP3 streams are decodable by the browser — each
    fragment contains self-contained frames, so the accumulate+redecode
    pattern works from the very first chunk.
    """

    def __init__(self, sample_rate: int, bitrate: int = 64):
        import lameenc
        self._encoder = lameenc.Encoder()
        self._encoder.set_bit_rate(bitrate)
        self._encoder.set_in_sample_rate(sample_rate)
        self._encoder.set_channels(1)
        self._encoder.set_quality(2)
        self._started = False

    def write_chunk(self, audio: 'torch.Tensor') -> bytes:
        """Encode audio chunk and return MP3 frame bytes."""
        self._started = True
        samples = audio.squeeze().cpu().float().numpy()
        pcm_s16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return self._encoder.encode(pcm_s16)

    def close(self) -> bytes:
        """Flush the encoder and return remaining MP3 bytes."""
        if not self._started:
            return b''
        return self._encoder.flush()
