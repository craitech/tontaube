import base64
from pathlib import Path

from app import utils


def test_uploaded_voice_decode_is_bounded_at_prompt_duration(monkeypatch):
    captured = {}

    class Audio:
        def __getitem__(self, _slice):
            return self

        def export(self, path, format):
            Path(path).write_bytes(b"wav")

    def fake_from_file(_source, **kwargs):
        captured.update(kwargs)
        return Audio()

    monkeypatch.setattr(utils.AudioSegment, "from_file", fake_from_file)
    output = utils.decode_voice_audio_b64(
        base64.b64encode(b"audio").decode("ascii"),
        prompt_seconds=60,
    )
    try:
        assert captured["duration"] == 60
    finally:
        Path(output).unlink()
