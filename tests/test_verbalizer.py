"""Tests for Verbalizer tag-aware verbalization."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.verbalizer import _TAG_RE


# ── _TAG_RE splitting tests ──────────────────────────────────────────────────

class TestTagRegex:
    def test_single_tag(self):
        parts = _TAG_RE.split("Hello <<moan>> world")
        assert parts == ["Hello ", "<<moan>>", " world"]

    def test_single_angle_brackets_not_matched(self):
        parts = _TAG_RE.split("5 < 10 and 10 > 5")
        assert parts == ["5 < 10 and 10 > 5"]

    def test_fullmatch(self):
        assert _TAG_RE.fullmatch("<<moan>>")
        assert not _TAG_RE.fullmatch("hello <<moan>>")
        assert not _TAG_RE.fullmatch("hello")


# ── Verbalizer.verbalize() integration tests ─────────────────────────────────

@pytest.fixture
def mock_verbalizer():
    """Create a Verbalizer with mocked engine, only testing verbalize() logic."""
    with patch("app.verbalizer.AsyncLLMEngine"), \
         patch("app.verbalizer.AutoTokenizer"):
        from app.verbalizer import Verbalizer
        v = MagicMock(spec=Verbalizer)
        v.verbalize = Verbalizer.verbalize.__get__(v, Verbalizer)
        v._verbalize_plain = AsyncMock(side_effect=lambda text: f"VERB({text})")
        return v


@pytest.mark.asyncio
async def test_verbalize_no_tags(mock_verbalizer):
    result = await mock_verbalizer.verbalize("Hello $5 world")
    assert result == "VERB(Hello $5 world)"


@pytest.mark.asyncio
async def test_verbalize_tag_preserved(mock_verbalizer):
    result = await mock_verbalizer.verbalize("He said <<moan>> and $5")
    assert result == "VERB(He said )<<moan>>VERB( and $5)"


@pytest.mark.asyncio
async def test_verbalize_only_tag(mock_verbalizer):
    result = await mock_verbalizer.verbalize("<<moan>>")
    assert result == "<<moan>>"
    mock_verbalizer._verbalize_plain.assert_not_called()


@pytest.mark.asyncio
async def test_verbalize_multiple_tags(mock_verbalizer):
    result = await mock_verbalizer.verbalize("Hi <<sigh>> then <<moan>> bye")
    assert result == "VERB(Hi )<<sigh>>VERB( then )<<moan>>VERB( bye)"


@pytest.mark.asyncio
async def test_verbalize_whitespace_between_tags_not_verbalized(mock_verbalizer):
    result = await mock_verbalizer.verbalize("<<moan>> <<gasp>>")
    # " " is whitespace-only, so not verbalized, just preserved
    assert result == "<<moan>> <<gasp>>"
    mock_verbalizer._verbalize_plain.assert_not_called()
