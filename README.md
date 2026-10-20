# vietnameseLatexOCR

Convert **scanned PDFs containing Vietnamese text and mathematical formulas** into
Markdown, with the formulas written as LaTeX.

- Input: a folder (searched **recursively**) of PDFs, in any nesting of subfolders.
- Output: one `*.md` per `*.pdf`, same filename, mirrored folder structure.
- Math → `$inline$` and `$$display$$` LaTeX.
- Vietnamese diacritics preserved and normalised to Unicode NFC.

## Why Marker / Surya

[Marker](https://github.com/datalab-to/marker) is the pipeline built on top of
[Surya](https://github.com/datalab-to/surya) by the same authors. Surya gives the
layout, text-detection and recognition models; Marker adds the document
reconstruction that produces Markdown — reading order, headings, lists, tables,
and an equation processor that OCRs formula regions into LaTeX. Using Surya
directly would mean rebuilding that layer by hand, so Marker is the better
starting point for this task while still being "SuryaOCR" underneath.

Its recognition model is multilingual and handles Vietnamese without any
language flag (newer Surya versions dropped language hints entirely), and the
whole stack fits comfortably on a 12 GB RTX 3060.

**Alternatives considered**

| Tool | Verdict |
| --- | --- |
| [MinerU](https://github.com/opendatalab/MinerU) | The closest competitor and genuinely good at formulas (UniMERNet → LaTeX). Worth trying if Marker's math output disappoints on your scans. Heavier install; AGPL-3.0. |
| [Docling](https://github.com/docling-project/docling) (IBM) | MIT-licensed and pleasant to use, but Vietnamese depends on the OCR engine you plug in (Tesseract/RapidOCR) and formula handling is weaker. |
| [Nougat](https://github.com/facebookresearch/nougat) (Meta) | Excellent LaTeX for academic papers, but trained on English arXiv — not suitable for Vietnamese. |
| [olmOCR](https://github.com/allenai/olmocr) (AI2) | Strong quality, but it's a 7B VLM; tight on a 12 GB card and much slower per page. |

A licensing note, since it is easy to miss: Marker and Surya are **GPL-3.0 with a
commercial-use condition** — free for research and personal use, and free for
organisations under roughly \$2 M in revenue/funding, but a paid licence above
that. Docling (MIT) is the permissive option if that matters for you.

## Setup

GPU inference needs the CUDA build of PyTorch installed *before* `marker-pdf`,
otherwise pip may pick the CPU-only wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The first conversion downloads roughly 2–3 GB of model weights into the Hugging
Face cache (`~/.cache/huggingface`). Later runs start in a few seconds.

## Usage

Drop your PDFs — individual files or whole folder trees — into `data_source/`,
then:

```bash
python pdf2md.py
```

`data_source/toan/chuong_1/dao_ham.pdf` → `output/toan/chuong_1/dao_ham.md`.

### Images

Regions the layout model classifies as a figure or picture are **cropped out of
the page** and saved into a folder named exactly like the Markdown file, and the
Markdown links to them with a relative path:

```
output/toan/chuong_2/do_thi.md
output/toan/chuong_2/do_thi/_page_0_Figure_2.png
output/toan/chuong_2/do_thi/_page_0_Figure_5.png
```

```markdown
Xét đồ thị của hàm số bậc hai dưới đây:

![](do_thi/_page_0_Figure_2.png)
```

Because the `.md` and its folder sit side by side, the links keep working if you
move or copy the pair elsewhere.

Crops are taken from a 192 DPI render of the page (not the low-res layout
thumbnail) and written as PNG. Tune with `--image-dpi 300` for small or dense
diagrams, `--image-format jpeg` for photo-heavy scans, `--image-dir-suffix
_images` if you prefer `do_thi_images/`, or `--no-images` to skip cropping.

Note that a figure **caption** is usually treated as part of the figure, so it
ends up inside the cropped image rather than as separate text in the Markdown.

Useful flags:

```bash
python pdf2md.py --list                  # just show which PDFs were found
python pdf2md.py --limit 1               # trial run on one file first
python pdf2md.py --page-range 0-4        # only the first 5 pages of each PDF
python pdf2md.py --in-place              # write .md beside each .pdf
python pdf2md.py --overwrite             # redo files that already have a .md
python pdf2md.py -i /other/dir -o /md    # different folders
python pdf2md.py --no-images             # text only, no figure crops
python pdf2md.py --image-dpi 300         # sharper figure crops
python pdf2md.py --device cpu            # no GPU
```

Re-runs are resumable: a PDF whose `.md` is already newer than the PDF is
skipped, so an interrupted batch can simply be restarted. Per-file failures are
logged and do not stop the batch; the summary lists them at the end.

### GPU memory

`--vram-profile` sets the model batch sizes. Marker's own CUDA defaults assume a
datacenter GPU (recognition batch 256) and will exhaust a 3060, so this script
defaults to the `12gb` profile. If you still hit `CUDA out of memory`, drop to
`--vram-profile 6gb`; on a larger card use `24gb` or `default`.

Expect very roughly 2–6 s/page for a text-heavy scan on the 3060, depending on
how many formula and table regions each page contains.

## Checking the output

OCR is never perfect, and formula-heavy scans are the hardest case. Worth
spot-checking:

- formulas that silently became plain text instead of `$...$`
- Vietnamese diacritics on low-contrast or skewed scans
- reading order in multi-column pages

Rendering the Markdown somewhere with math support (VS Code preview, Obsidian,
Pandoc, MkDocs with KaTeX) makes problems much easier to see than reading the
raw file.

If the scans themselves are poor, deskewing and raising the contrast before
conversion helps more than any OCR setting.
