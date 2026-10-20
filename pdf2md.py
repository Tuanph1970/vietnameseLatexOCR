#!/usr/bin/env python3
"""Convert scanned PDFs (Vietnamese text + math formulas) to Markdown with LaTeX math.

Walks an input folder recursively, converts every ``*.pdf`` to ``*.md`` using
Marker (which runs the Surya OCR models underneath), and mirrors the input
directory structure into the output folder.

    python pdf2md.py                      # data_source/ -> output/
    python pdf2md.py --in-place           # write the .md next to each .pdf
    python pdf2md.py -i some/dir -o md/   # explicit folders

Math is emitted as ``$...$`` / ``$$...$$``, Vietnamese diacritics are
normalised to Unicode NFC so the text greps and renders consistently.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

# Marker reads some of these at import time, so they must be set before any
# `marker.*` / `surya.*` import happens (all of those are deferred to main()).
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

LOG = logging.getLogger("pdf2md")

# Batch sizes per VRAM budget. Marker's CUDA defaults (e.g. recognition=256)
# assume a datacenter GPU and will OOM a 12 GB card on dense scans.
VRAM_PROFILES: dict[str, dict[str, int]] = {
    "6gb": {
        "detection_batch_size": 4,
        "layout_batch_size": 4,
        "recognition_batch_size": 16,
        "equation_batch_size": 8,
        "table_rec_batch_size": 4,
        "ocr_error_batch_size": 4,
    },
    "12gb": {
        "detection_batch_size": 12,
        "layout_batch_size": 8,
        "recognition_batch_size": 32,
        "equation_batch_size": 16,
        "table_rec_batch_size": 8,
        "ocr_error_batch_size": 8,
    },
    "24gb": {
        "detection_batch_size": 24,
        "layout_batch_size": 16,
        "recognition_batch_size": 96,
        "equation_batch_size": 32,
        "table_rec_batch_size": 16,
        "ocr_error_batch_size": 16,
    },
    # Leave every batch size unset and let marker/surya pick its own defaults.
    "default": {},
}


@dataclass
class Totals:
    converted: int = 0
    skipped: int = 0
    failed: list[tuple[Path, str]] = field(default_factory=list)
    seconds: float = 0.0
    pages: int = 0
    images: int = 0


def fold(text: str) -> str:
    """Normalise a name for matching: lowercase, no diacritics, no separators.

    Lets a single ``*viet tay*`` match 'VIẾT TAY', 'viet tay' and 'viet-tay',
    which all appear in real filenames.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    stripped = stripped.replace("\u0111", "d")  # đ has no combining form
    return re.sub(r"[-_\s]+", " ", stripped)


def is_excluded(pdf: Path, root: Path, patterns: list[str]) -> bool:
    """True if the path matches any --exclude glob (accent- and case-insensitive)."""
    rel = fold(str(pdf.relative_to(root)))
    return any(fnmatch(rel, fold(pat)) for pat in patterns)


def find_pdfs(root: Path) -> list[Path]:
    """Every PDF under `root`, recursively, in stable order.

    Hidden directories and macOS resource-fork files (``._foo.pdf``) are skipped.
    """
    pdfs: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or not name.lower().endswith(".pdf"):
                continue
            pdfs.append(Path(dirpath) / name)
    return pdfs


def destination_for(pdf: Path, input_root: Path, output_root: Path, in_place: bool) -> Path:
    """Where the .md for `pdf` goes — same stem, mirrored subdirectory."""
    if in_place:
        return pdf.with_suffix(".md")
    return (output_root / pdf.relative_to(input_root)).with_suffix(".md")


def is_up_to_date(pdf: Path, md: Path) -> bool:
    return md.exists() and md.stat().st_mtime >= pdf.stat().st_mtime


def md_link(path: str) -> str:
    """Escape a path for a markdown link destination.

    Only the characters that actually break the ``](dest)`` syntax are encoded:
    a ')' in the folder name would close the destination early, and a space
    ends it in stricter parsers. Vietnamese letters are left alone so the link
    stays readable. '%' goes first, or it would corrupt the escapes added after it.
    """
    for char, code in (("%", "%25"), ("(", "%28"), (")", "%29"), (" ", "%20")):
        path = path.replace(char, code)
    return path


def postprocess(markdown: str, image_dir_name: str | None, image_names: list[str]) -> str:
    """Normalise Vietnamese text and repoint image links at the images folder."""
    # Surya can emit decomposed diacritics (e.g. "e" + U+0302); NFC composes
    # them into the single codepoints Vietnamese text is normally stored as.
    markdown = unicodedata.normalize("NFC", markdown)

    for name in image_names:
        # Marker references each crop by bare filename. Point it at the folder
        # (or just keep the bare name when images sit beside the .md).
        target = md_link(f"{image_dir_name}/{name}" if image_dir_name else name)
        escaped = re.escape(name)
        # ![alt](name) and HTML <img src='name'> / src="name" both occur.
        markdown = re.sub(rf"\]\(\s*{escaped}\s*\)", f"]({target})", markdown)
        markdown = re.sub(rf"""src=(['"]){escaped}\1""", f'src="{target}"', markdown)

    # markdownify occasionally leaves a bare block tag behind. Drop those, but
    # keep <a>/<img>, which marker emits deliberately.
    markdown = re.sub(r"</?(?:p|br|div|span)\s*/?>", "", markdown)

    # Collapse the runs of blank lines marker leaves between blocks.
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
    return markdown.strip() + "\n"


def build_config(args: argparse.Namespace) -> dict:
    config: dict = {
        "output_format": "markdown",
        # Scans have no usable text layer; forcing OCR also avoids trusting a
        # bad pre-existing OCR layer that some scanners embed.
        "force_ocr": not args.no_force_ocr,
        "strip_existing_ocr": not args.no_force_ocr,
        # Inline math recognition is what turns formulas into LaTeX.
        "disable_ocr_math": False,
        "inline_math_delimiters": ("$", "$"),
        "block_math_delimiters": ("$$", "$$"),
        "extract_images": not args.no_images,
        # Crop from the high-resolution page render, not the layout thumbnail.
        "image_extraction_mode": "highres",
        "highres_image_dpi": args.image_dpi,
        "disable_tqdm": args.quiet,
        "pdftext_workers": 1,
    }
    config.update(VRAM_PROFILES[args.vram_profile])
    if args.page_range:
        # Left as a string on purpose: marker's ConfigParser calls
        # parse_range_str() on this key and chokes on a pre-parsed list.
        config["page_range"] = args.page_range
    if args.paginate:
        config["paginate_output"] = True
    return config


def convert_one(
    pdf: Path,
    md_path: Path,
    models: dict,
    config: dict,
    save_images: bool,
    image_dir_suffix: str = "",
) -> tuple[int, int]:
    """Convert one PDF, write the Markdown and the cropped figures.

    Returns ``(page_count, image_count)``.
    """
    from marker.config.parser import ConfigParser
    from marker.output import text_from_rendered

    config_parser = ConfigParser(config)
    converter = config_parser.get_converter_cls()(
        artifact_dict=models,
        config=config_parser.generate_config_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    rendered = converter(str(pdf))
    text, _ext, images = text_from_rendered(rendered)
    images = images or {}

    # Figures/pictures are cropped out of the page by marker; we drop them in a
    # folder named after the .md file so the two travel together.
    image_dir_name = None
    if save_images and images:
        image_dir_name = f"{md_path.stem}{image_dir_suffix}"
        image_dir = md_path.parent / image_dir_name
        image_dir.mkdir(parents=True, exist_ok=True)
        for name, image in images.items():
            try:
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                image.save(image_dir / name)
            except Exception as exc:  # a single bad crop must not lose the page
                LOG.warning("  could not save image %s: %s", name, exc)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        postprocess(text, image_dir_name, list(images)),
        encoding="utf-8",
    )

    page_stats = (getattr(rendered, "metadata", None) or {}).get("page_stats") or []
    return len(page_stats), len(images)


def free_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Recursively convert scanned PDFs (Vietnamese + math) to Markdown with LaTeX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", type=Path, default=here / "data_source",
                   help="Folder to scan recursively for PDFs.")
    p.add_argument("-o", "--output", type=Path, default=here / "output",
                   help="Folder to write Markdown into, mirroring the input tree.")
    p.add_argument("--in-place", action="store_true",
                   help="Write each .md next to its .pdf instead of into --output.")
    p.add_argument("--overwrite", action="store_true",
                   help="Reconvert even when an up-to-date .md already exists.")
    p.add_argument("--list", action="store_true",
                   help="List the PDFs that would be converted, then exit.")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                   help="Skip PDFs whose relative path matches this glob. Repeatable, "
                        "case- and accent-insensitive, so '*viet tay*' also matches "
                        "'VIẾT TAY'. Example: --exclude '*viet tay*' --exclude '*vt*'.")
    p.add_argument("--limit", type=int, default=0,
                   help="Convert at most N files (0 = no limit). Useful for a trial run.")
    p.add_argument("--page-range", type=str, default=None,
                   help="Only these pages, 0-based, e.g. '0,2,5-10'.")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device, e.g. cuda, cuda:0, cpu. Default: cuda when available.")
    p.add_argument("--vram-profile", choices=sorted(VRAM_PROFILES), default="12gb",
                   help="Batch-size preset. '12gb' suits an RTX 3060 12GB.")
    p.add_argument("--no-force-ocr", action="store_true",
                   help="Trust an existing text layer when the PDF has one (faster on digital PDFs).")
    p.add_argument("--no-images", action="store_true",
                   help="Do not crop figures/pictures out of the page.")
    p.add_argument("--image-dir-suffix", type=str, default="",
                   help="Suffix for the image folder. Default '' = folder named exactly "
                        "like the .md file; e.g. '_images' gives 'name_images/'.")
    p.add_argument("--image-format", choices=["png", "jpeg"], default="png",
                   help="Format for the cropped figures. PNG is lossless, better for "
                        "diagrams and line art; JPEG is smaller for photos.")
    p.add_argument("--image-dpi", type=int, default=192,
                   help="Resolution the page is rendered at before cropping figures. "
                        "Raise to ~300 for small or dense diagrams.")
    p.add_argument("--paginate", action="store_true",
                   help="Insert page separators in the Markdown.")
    p.add_argument("--quiet", action="store_true", help="Hide per-model progress bars.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    input_root: Path = args.input.resolve()
    if not input_root.is_dir():
        LOG.error("Input folder does not exist: %s", input_root)
        return 2
    output_root: Path = args.output.resolve()

    # Must happen before any marker/surya import: marker.settings is a pydantic
    # BaseSettings instantiated at module import time.
    if args.device:
        os.environ["TORCH_DEVICE"] = args.device
    # Drives both the encoder and the extension marker puts in the <img> link,
    # so it has to be set here rather than at save time.
    os.environ["OUTPUT_IMAGE_FORMAT"] = args.image_format.upper()

    pdfs = find_pdfs(input_root)
    if not pdfs:
        LOG.error("No PDF files found under %s", input_root)
        return 1

    excluded = [p for p in pdfs if is_excluded(p, input_root, args.exclude)]
    dropped = set(excluded)
    pdfs = [p for p in pdfs if p not in dropped]
    if excluded:
        LOG.info("Excluded %d file(s) by --exclude.", len(excluded))

    if args.list:
        for pdf in pdfs:
            LOG.info("%s", pdf.relative_to(input_root))
        LOG.info("\n%d PDF file(s).", len(pdfs))
        if excluded:
            LOG.info("\nExcluded:")
            for pdf in excluded:
                LOG.info("  %s", pdf.relative_to(input_root))
        return 0

    # Decide the work list before loading the (slow) models.
    jobs: list[tuple[Path, Path]] = []
    totals = Totals()
    for pdf in pdfs:
        md_path = destination_for(pdf, input_root, output_root, args.in_place)
        if not args.overwrite and is_up_to_date(pdf, md_path):
            totals.skipped += 1
            continue
        jobs.append((pdf, md_path))
    if args.limit:
        jobs = jobs[: args.limit]

    LOG.info("Found %d PDF(s) under %s", len(pdfs), input_root)
    LOG.info("To convert: %d   up-to-date (skipped): %d", len(jobs), totals.skipped)
    if not jobs:
        return 0

    LOG.info("Loading OCR models (first run downloads ~2-3 GB of weights)...")
    from marker.logger import configure_logging
    from marker.models import create_model_dict
    from marker.settings import settings as marker_settings

    configure_logging()
    models = create_model_dict(device=args.device)
    LOG.info("Models ready on device=%s dtype=%s   vram-profile=%s",
             marker_settings.TORCH_DEVICE_MODEL, marker_settings.MODEL_DTYPE, args.vram_profile)

    config = build_config(args)
    run_start = time.time()

    for index, (pdf, md_path) in enumerate(jobs, start=1):
        rel = pdf.relative_to(input_root)
        size_mb = pdf.stat().st_size / 1e6
        LOG.info("[%d/%d] %s (%.1f MB)", index, len(jobs), rel, size_mb)
        started = time.time()
        try:
            pages, n_images = convert_one(
                pdf, md_path, models, config,
                save_images=not args.no_images,
                image_dir_suffix=args.image_dir_suffix,
            )
        except KeyboardInterrupt:
            LOG.warning("Interrupted. %d file(s) converted so far.", totals.converted)
            return 130
        except Exception as exc:
            LOG.exception("  FAILED: %s", exc)
            totals.failed.append((rel, f"{type(exc).__name__}: {exc}"))
            free_vram()
            continue

        elapsed = time.time() - started
        totals.converted += 1
        totals.pages += pages
        totals.images += n_images
        totals.seconds += elapsed
        LOG.info("  -> %s  (%d page(s), %d image(s), %.1fs)",
                 md_path, pages, n_images, elapsed)
        free_vram()

    LOG.info("\n%s", "-" * 60)
    LOG.info("Converted %d   skipped %d   failed %d   cropped %d image(s)   in %.1fs",
             totals.converted, totals.skipped, len(totals.failed), totals.images,
             time.time() - run_start)
    if totals.pages and totals.seconds:
        LOG.info("Throughput: %.2f s/page over %d page(s)",
                 totals.seconds / totals.pages, totals.pages)
    if totals.failed:
        LOG.info("\nFailures:")
        for rel, reason in totals.failed:
            LOG.info("  %s\n    %s", rel, reason)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
