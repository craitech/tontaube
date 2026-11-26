# Tontaube Reader privacy policy

Last updated: 26 November 2026

Tontaube Reader is a Chrome extension by Tontaube that reads web pages
aloud with a voice model that runs on your own computer. This policy explains
what the extension does with your data. In short, it collects nothing.

## What the extension reads

The extension reads text only when you ask it to, by clicking its toolbar
button and choosing *Read page* or *Read selection*, or by pressing one of its
keyboard shortcuts. It then reads the text of the current tab (the selected
text, or else the page's main text). It does not read other tabs, and it does
not read pages in the background.

## Where your text goes

Nowhere. The text is turned into speech on your computer, inside Chrome. It is
not sent to us or to anyone else, and it is not kept after the reading ends.

## What is stored on your computer

- Your settings (highlight, 48 kHz, redraw pitch jumps, start delay, GPU), in
  Chrome's storage for the extension.
- The voice model files (about 700 MB) and the result of a short check of your
  graphics card, in the extension's private browser storage, so that the model
  is downloaded only once.

This data stays on your computer. Removing the extension deletes it. While a
reading runs, the extension also remembers which tab it is reading. That note
is kept in memory only and is gone when the browser closes.

## Network access

The extension connects to the internet only to download the voice model files
from Hugging Face (huggingface.co and its download servers). This happens on
first use, and again only when an update of the extension brings new model
files or the stored files were removed. The download sends no page text and
no personal information. Like any web server, Hugging Face receives your IP
address and basic browser information with the request, and its own
[privacy policy](https://huggingface.co/privacy) applies to that. One of the
requests fetches the model's small configuration file, which Hugging Face
counts as a download of the model.

## No data collection

We do not collect, store, sell or share any personal data, browsing history or
page content. The extension has no analytics, no advertising and no tracking,
and it needs no account.

The extension's use of information complies with the Chrome Web Store User
Data Policy, including its Limited Use requirements.

## Changes

If this policy changes, the new version will be posted at this address with a
new date.

## Contact

data-protection@craitech.io
