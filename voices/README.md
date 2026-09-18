# Sample voices

Voice references live in the single [`samples/`](samples/) directory. Optional
language and speaking-style metadata is stored in
[`manifest.json`](manifest.json). The browser
interface uses declared languages to filter its list and displays each filename
without its extension. Files without a declared language appear for every
language, and files without a declared style remain ungrouped. Add or replace a
supported audio file, optionally add its language and style metadata, and use
**Refresh voices** to update the list. The UI groups catalogued voices by style
and displays the selected style as a badge. This does not change or restrict the
independently selected synthesis style.

The manifest uses ISO language codes and the public Tontaube speaking styles:

```json
{
  "version": 1,
  "voices": {
    "Frederick.mp3": {"language": "en", "style": "audiobook"}
  }
}
```

The 39 standard voices are supplied as MP3 references reconstructed from their
hosted voice prompts using DualCodec, VibeVoice and hybrid MossFormer2.
The UI uploads the selected audio for voice cloning, just like a user-supplied
recording. Seven legacy audio references remain available, including Miles.
Marcus is the default voice. Marcus and Elias have no declared style.

Database styles `Narration` and `Conversation` correspond to manifest styles
`audiobook` and `conversational`; `Agentic` corresponds to `agentic`. Omit a
style or language when it is unspecified. The manifest does not change the
generation request's selected style.

The bundled voice references are released under the [MIT License](LICENSE).

Set `TTS_UI_VOICE_PATH` when the standalone UI should serve a different voice
catalogue root containing `manifest.json` and `samples/`. The selected audio
is sent to the inference API with the request;
it does not need to exist on the inference server. Reference files should
contain clean speech from one speaker and should not exceed 60 seconds.
