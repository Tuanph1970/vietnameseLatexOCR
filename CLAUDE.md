# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-purpose batch converter: scanned PDFs containing **Vietnamese text and
mathematical formulas** → Markdown with LaTeX math. `pdf2md.py` is the whole
program; everything else is data or docs.

## Commands

```bash
python pdf2md.py                 # data_source/ -> output/, recursive, resumable
python pdf2md.py --list          # dry run: which PDFs were found
python pdf2md.py --limit 1       # single-file trial
python pdf2md.py --device cpu    # no GPU
```

There is no test suite, linter config, or package manifest beyond
`requirements.txt`. To verify a change end to end, generate an image-only PDF
(matplotlib + PIL, using `static/fonts/GoNotoCurrent-Regular.ttf` from the
marker install for Vietnamese glyphs), drop it in `data_source/`, and run the
script — a PDF with zero text-layer characters is what exercises the OCR path.

## Architecture

`pdf2md.py` wraps **marker-pdf** (which runs **surya-ocr** models underneath).
Marker supplies the document reconstruction — layout, reading order, headings,
tables, and the equation processor that turns formula regions into LaTeX. The
script adds the batch/recursion layer around it:

1. `find_pdfs()` walks the input tree; `destination_for()` mirrors each relative
   path into the output root with a `.md` suffix.
2. Work list and up-to-date skipping are decided **before** the models load,
   since loading costs seconds and a few GB of VRAM.
3. `create_model_dict()` is called **once** and the model dict is reused; a fresh
   `PdfConverter` is built per file (what marker's own batch CLI does).
4. Figures/pictures are cropped by marker (`image_blocks` = Picture + Figure;
   tables and equations are reconstructed as text instead, which is what we
   want). They are saved into a folder named exactly like the `.md` stem, and
   `postprocess()` NFC-normalises the text and rewrites each bare image
   reference to `<stem>/<file>`.

### Constraints that are easy to break

- **Import order.** `marker.settings` is a pydantic `BaseSettings` instantiated
  at module import time, so `TORCH_DEVICE` must be exported before the first
  `marker.*`/`surya.*` import. Every marker import in this file is therefore
  deferred inside a function — keep it that way, and do not add a top-level
  `import marker` or `from marker.util import ...`.
- **`page_range` stays a string** in the config dict. Marker's `ConfigParser`
  calls `parse_range_str()` on that key itself and fails on a pre-parsed list.
- **`output_format` must be present** in the config dict: `ConfigParser.get_renderer()`
  indexes `cli_options["output_format"]` directly rather than using `.get()`.
- **Falsy config values are dropped.** `generate_config_dict()` skips any option
  whose value is falsy, so `False`/`0`/`""` cannot be passed through it — express
  such settings as the non-default truthy case instead.
- **Batch sizes.** Marker's CUDA defaults (recognition batch 256) assume a
  datacenter GPU and OOM the target 12 GB RTX 3060. `VRAM_PROFILES` overrides
  them; `12gb` is the default profile.
- Surya recognition is multilingual with **no language parameter** — do not
  reintroduce a `languages` config key, it was removed upstream.
- **Image format comes from `OUTPUT_IMAGE_FORMAT`**, an env var read by
  `marker.settings`. It sets both the encoder and the extension in the generated
  link, so it must be exported before the first marker import — not applied at
  `image.save()` time, or the link and the file will disagree.
- Marker emits `<img src='name'>` with **single** quotes; the link rewrite has to
  match either quote style as well as the `![](name)` markdown form.

## Licensing

Repo is Apache 2.0, but marker/surya are GPL-3.0 with a commercial-use revenue
condition. Keep that distinction accurate in docs; do not describe the pipeline
as unconditionally free for commercial use.
