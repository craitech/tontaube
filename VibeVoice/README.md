# VibeVoice acoustic tokenizer

This directory contains selected tokenizer modules from
[Microsoft VibeVoice](https://github.com/microsoft/VibeVoice). Tontaube uses
their acoustic-tokenizer runtime for audio re-encoding and low-latency
streaming decode. Some unused configuration and class definitions remain in
the selected upstream modules; the VibeVoice language model, processors,
demos, and conversion scripts are not included.

The retained implementation provides `VibeVoiceAcousticTokenizerModel` and
`VibeVoiceTokenizerStreamingCache`. Tontaube loads the corresponding weights
from the `acoustic_tokenizer/` directory in the model bundle.

The vendored source remains subject to the
[Microsoft VibeVoice source-code license](LICENSE). VibeVoice model weights are
obtained separately from Microsoft and remain subject to their upstream terms.
