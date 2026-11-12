"""Public view of the style sliders (app/controls.py PUBLIC_OVERRIDES).

The shipped TontaubeV1.1 bundle registers seven sliders. This release offers five: hesitation and varied_pacing
are not offered and legato has 3 steps, so the API lists five sliders and refuses the rest, while the registered
token set (tokenizer check, boundary ids) stays complete.
"""
import glob
import json
import os
from types import SimpleNamespace

import pytest

from app.config import UTT_CONT, UTT_END, ServerConfig
from app.controls import PUBLIC_OVERRIDES, ControlCatalog
from app.inference import TTSEngine
from app.utils import check_boundary_tokens
from tests.test_boundary_tokens import write_bundle

# Same shape as the cb0/controls.json of TontaubeV1.1.
SHIPPED_CONTROLS = {"specs": [
    {"name": "rate", "n_bins": 5, "description": "speaking rate",
     "low_description": "much slower", "high_description": "much faster"},
    {"name": "pause", "n_bins": 5, "description": "pauses",
     "low_description": "fewer shorter pauses", "high_description": "more longer pauses"},
    {"name": "pitch_range", "n_bins": 6, "description": "intonation",
     "low_description": "flat monotone", "high_description": "very expressive"},
    {"name": "pitch_level", "n_bins": 5, "description": "voice pitch",
     "low_description": "low deep voice", "high_description": "high voice"},
    {"name": "hesitation", "n_bins": 3, "description": "hesitation",
     "low_description": "fluent", "high_description": "hesitant"},
    {"name": "legato", "n_bins": 5, "description": "legato versus staccato articulation",
     "low_description": "crisp staccato", "high_description": "smooth legato"},
    {"name": "varied_pacing", "n_bins": 5, "description": "variation of speaking pace",
     "low_description": "steady even pace", "high_description": "lively varied pace"},
]}
ALL_BINS = {"rate": 5, "pause": 5, "pitch_range": 6, "pitch_level": 5, "hesitation": 3, "legato": 5,
            "varied_pacing": 5}
PUBLIC_BINS = {"rate": 5, "pause": 5, "pitch_range": 6, "pitch_level": 5, "legato": 3}
HIDDEN = ("hesitation", "varied_pacing")

SHIPPED_DIRS = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--TontaubeAI--TontaubeV1.1-int4/snapshots/*/cb0")))


@pytest.fixture
def catalog(tmp_path):
    (tmp_path / "controls.json").write_text(json.dumps(SHIPPED_CONTROLS))
    return ControlCatalog.load(str(tmp_path))


@pytest.fixture
def engine(catalog):
    engine = TTSEngine(ServerConfig())
    engine.controls = catalog
    return engine


def _tags(engine, controls=None, segment_controls=None, n_segments=1):
    return engine._cb0_system_tags(
        "", n_segments, "english", "audiobook", True, controls=controls, segment_controls=segment_controls)


# ---------------------------------------------------------------------------
# What GET /controls lists
# ---------------------------------------------------------------------------

def test_catalog_lists_the_five_public_sliders_and_the_legato_texts(catalog):
    entries = catalog.catalog()
    assert [e["name"] for e in entries] == ["rate", "pause", "pitch_range", "pitch_level", "legato"]
    assert {e["name"]: e["bins"] for e in entries} == PUBLIC_BINS
    legato = entries[-1]
    assert legato == {"name": "legato", "bins": 3, "description": "crisp articulation",
                      "low": "crisp staccato", "high": "normal articulation", "default": 2}
    for text in (legato["description"], legato["low"], legato["high"]):
        assert "smooth" not in text and "versus" not in text
    assert "legato" not in legato["high"]
    # The other four are the bundle's own entries, untouched.
    assert entries[0] == {"name": "rate", "bins": 5, "description": "speaking rate",
                          "low": "much slower", "high": "much faster"}
    assert catalog.describe() == PUBLIC_BINS


def test_override_table_names_exactly_three_sliders():
    assert set(PUBLIC_OVERRIDES) == {"hesitation", "varied_pacing", "legato"}
    assert PUBLIC_OVERRIDES["hesitation"] is None and PUBLIC_OVERRIDES["varied_pacing"] is None
    assert PUBLIC_OVERRIDES["legato"]["n_bins"] == 3


def test_override_never_raises_a_bin_count_and_ignores_sliders_a_bundle_lacks(tmp_path):
    (tmp_path / "controls.json").write_text(json.dumps({"specs": [
        {"name": "legato", "n_bins": 2}, {"name": "rate", "n_bins": 5}]}))
    assert ControlCatalog.load(str(tmp_path)).describe() == {"legato": 2, "rate": 5}


# ---------------------------------------------------------------------------
# What a request may send
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", HIDDEN)
def test_hidden_slider_is_an_unknown_name_like_any_other(engine, name):
    with pytest.raises(ValueError) as hidden:
        _tags(engine, controls={name: 1})
    with pytest.raises(ValueError) as unknown:
        _tags(engine, controls={"nonsense": 1})
    assert str(hidden.value) == str(unknown.value).replace("nonsense", name)
    assert f"unknown controls ['{name}']" in str(hidden.value)
    assert "available: " in str(hidden.value) and "hesitation" not in str(hidden.value).split("available:")[1]
    assert "varied_pacing" not in str(hidden.value).split("available:")[1]


@pytest.mark.parametrize("name", HIDDEN)
def test_hidden_slider_is_refused_in_segment_controls_and_token_strings(engine, name):
    with pytest.raises(ValueError, match="unknown controls"):
        _tags(engine, segment_controls=[{name: 0}], n_segments=1)
    with pytest.raises(ValueError, match="unknown controls"):
        _tags(engine, controls={"rate": 1}, segment_controls=[None, {name: 0}], n_segments=2)
    for token in (f"<|ctrl_{name}_00|>", f"<|ctrl_rate_01|><|ctrl_{name}_01|>"):
        with pytest.raises(ValueError, match="unknown control tokens"):
            _tags(engine, controls=token)
        with pytest.raises(ValueError, match="unknown control tokens"):
            _tags(engine, controls={"rate": 1}, segment_controls=[token], n_segments=1)


@pytest.mark.parametrize("step", [3, 4, 5, -1])
def test_legato_steps_above_2_are_out_of_range(engine, step):
    with pytest.raises(ValueError, match=r"control 'legato' must be an integer bin in \[0, 3\)"):
        _tags(engine, controls={"legato": step})
    with pytest.raises(ValueError, match=r"control 'legato' must be an integer bin in \[0, 3\)"):
        _tags(engine, controls={"rate": 2}, segment_controls=[{"legato": step}], n_segments=1)
    if step >= 0:
        with pytest.raises(ValueError, match="unknown control tokens"):
            _tags(engine, controls=f"<|ctrl_legato_{step:02d}|>")


def test_other_sliders_keep_their_full_range(engine):
    for name, bins in PUBLIC_BINS.items():
        for step in range(bins):
            tag = _tags(engine, controls={name: step})[0]
            assert f"<|ctrl_{name}_{step:02d}|>" in tag
        with pytest.raises(ValueError, match=f"control '{name}' must be an integer bin in"):
            _tags(engine, controls={name: bins})


def test_legato_steps_0_to_2_render_their_tokens(engine):
    assert _tags(engine, controls={"legato": 0})[0] == "<|im_start|>english : audiobook<|ctrl_legato_00|><|im_end|>\n"
    for step in (0, 1, 2):
        assert f"<|ctrl_legato_{step:02d}|><|im_end|>" in _tags(engine, controls={"legato": step})[0]
        assert _tags(engine, controls=f"<|ctrl_legato_{step:02d}|>")[0].count(f"<|ctrl_legato_{step:02d}|>") == 1
    # Spec (training) order is kept: legato after the pitch sliders.
    assert _tags(engine, controls={"legato": 1, "rate": 3})[0] == (
        "<|im_start|>english : audiobook<|ctrl_rate_03|><|ctrl_legato_01|><|im_end|>\n")
    request_tag, segment_tags = _tags(
        engine, controls={"rate": 3}, segment_controls=[{"legato": 0}, None], n_segments=2)
    assert segment_tags[0].endswith("<|ctrl_rate_03|><|ctrl_legato_00|><|im_end|>\n")
    assert segment_tags[1] == request_tag


def test_request_surface_reports_a_hidden_slider_as_an_option_error(engine):
    from app.main import TTSRequestOptionError, _require_controls

    def req(controls):
        return SimpleNamespace(language="english", tag="audiobook", use_system_tag=True, controls=controls)

    _require_controls(req({"legato": 2}), engine, "Hi.", None)
    for bad in ({"hesitation": 1}, {"legato": 3}, "<|ctrl_varied_pacing_02|>"):
        with pytest.raises(TTSRequestOptionError):
            _require_controls(req(bad), engine, "Hi.", None)
    with pytest.raises(TTSRequestOptionError):
        _require_controls(req(None), engine, ["a", "b"], [None, {"varied_pacing": 0}])


# ---------------------------------------------------------------------------
# The registered token set stays complete
# ---------------------------------------------------------------------------

def test_registered_token_set_keeps_all_seven_sliders_and_all_bins(catalog):
    assert [s.name for s in catalog.specs] == list(ALL_BINS)
    assert {s.name: s.n_bins for s in catalog.specs} == ALL_BINS
    assert len(catalog.all_tokens) == sum(ALL_BINS.values()) == 34
    for name in HIDDEN:
        assert f"<|ctrl_{name}_00|>" in catalog.all_tokens
    for step in range(5):
        assert f"<|ctrl_legato_{step:02d}|>" in catalog.all_tokens
    assert catalog.public_tokens < catalog.all_tokens
    assert len(catalog.public_tokens) == sum(PUBLIC_BINS.values())


class _VocabTokenizer:
    """Ids as in the shipped bundle: control tokens in spec order from `base`, boundary tokens after them."""

    unk_token_id = None

    def __init__(self, vocab):
        self.vocab = dict(vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def __len__(self):
        return max(self.vocab.values()) + 1


def _fixture_vocab(catalog, base=1000):
    ordered = [s.token(i) for s in catalog.specs for i in range(s.n_bins)]
    vocab = {token: base + i for i, token in enumerate(ordered)}
    return vocab, base + len(ordered)


def test_tokenizer_and_boundary_checks_pass_against_the_full_set_only(catalog, tmp_path):
    vocab, end_id = _fixture_vocab(catalog)
    vocab.update({UTT_END: end_id, UTT_CONT: end_id + 1})
    tokenizer = _VocabTokenizer(vocab)
    catalog.check_tokenizer(tokenizer)
    path = write_bundle(tmp_path / "bundle", ids=(end_id, end_id + 1), vocab_size=end_id + 2, table_rows=end_id + 2)
    assert check_boundary_tokens(path, tokenizer, 0, sorted(catalog.all_tokens)) == (end_id, end_id + 1)
    # varied_pacing holds the last control ids: a filtered token set would break the boundary-id check.
    with pytest.raises(ValueError, match="do not follow the last slider id"):
        check_boundary_tokens(path, tokenizer, 0, sorted(catalog.public_tokens))


def test_tokenizer_check_still_demands_a_hidden_slider_token(catalog):
    vocab, _ = _fixture_vocab(catalog)
    del vocab["<|ctrl_hesitation_01|>"]
    with pytest.raises(ValueError, match="lacks control tokens"):
        catalog.check_tokenizer(_VocabTokenizer(vocab))


@pytest.mark.skipif(not SHIPPED_DIRS, reason="TontaubeV1.1-int4 is not in the Hugging Face cache")
def test_shipped_bundle_catalog_and_token_ids():
    cb0 = SHIPPED_DIRS[-1]
    catalog = ControlCatalog.load(cb0)
    assert {s.name: s.n_bins for s in catalog.specs} == ALL_BINS
    assert catalog.describe() == PUBLIC_BINS
    # transformers is mocked in the suite: read the token ids from tokenizer.json itself.
    with open(os.path.join(cb0, "tokenizer.json")) as f:
        added = json.load(f)["added_tokens"]
    tokenizer = _VocabTokenizer({a["content"]: a["id"] for a in added})
    catalog.check_tokenizer(tokenizer)
    ids = sorted(tokenizer.vocab[t] for t in catalog.all_tokens)
    assert ids == list(range(ids[0], ids[0] + 34))
    assert tokenizer.vocab["<|ctrl_varied_pacing_04|>"] == ids[-1]
