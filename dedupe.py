#!/usr/bin/env python3
"""Find handwriting and duplicate PDFs before running the OCR batch.

OCR time is the scarce resource, so it is worth pruning the input first. This
reports three kinds of waste and never deletes anything unless you ask:

  handwriting   filenames saying "viết tay" (any spelling) — weakest OCR case
  identical     byte-for-byte copies
  subset        every page of A also appears in B, so A is redundant
  spacing       "cách dòng" variants: same document re-flowed with blank
                writing space, so it has more pages and less content per page

The first three are detected from content. The spacing variant is detected from
the filename on purpose: re-flowing changes every page's layout, so no
image-level comparison can match the two (verified on real files).

    python dedupe.py                      # report only
    python dedupe.py --emit-exclude       # print --exclude args for pdf2md.py
    python dedupe.py --move-to ../_dupes  # relocate (reversible), never delete
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path

HANDWRITING = ["*viet tay*", "*vt*"]
# "cách dòng" / "không cách dòng" — line spacing added for written answers.
SPACING_MARK = re.compile(r"\(?\b(?:k(?:hong)?\s+)?cach\s*dong\)?", re.I)


def fold(text: str) -> str:
    """Lowercase, strip Vietnamese diacritics, collapse -/_ to spaces."""
    d = unicodedata.normalize("NFD", text.lower())
    s = "".join(c for c in d if not unicodedata.combining(c)).replace("đ", "d")
    return re.sub(r"[-_\s]+", " ", s).strip()


def find_pdfs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def is_handwriting(pdf: Path, root: Path) -> bool:
    rel = fold(str(pdf.relative_to(root)))
    return any(fnmatch(rel, fold(p)) for p in HANDWRITING)


def page_hashes(pdf: Path, scale: float = 0.4) -> list[int]:
    """One perceptual hash per page, for detecting re-scans and subsets."""
    import pypdfium2 as pdfium
    from PIL import Image

    out: list[int] = []
    doc = pdfium.PdfDocument(str(pdf))
    try:
        for i in range(len(doc)):
            im = doc[i].render(scale=scale).to_pil().convert("L").resize((16, 16), Image.LANCZOS)
            px = list(im.getdata())
            avg = sum(px) / len(px)
            out.append(int("".join("1" if p > avg else "0" for p in px), 2))
    finally:
        doc.close()
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def contains_all(small: list[int], big: list[int], tol: int) -> int:
    """How many pages of `small` have a near-match in `big`."""
    return sum(1 for x in small if any(hamming(x, y) <= tol for y in big))


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=here / "data_source")
    ap.add_argument("--tolerance", type=int, default=12,
                    help="Max per-page hash distance to call two pages the same.")
    ap.add_argument("--emit-exclude", action="store_true",
                    help="Print ready-to-paste --exclude arguments for pdf2md.py.")
    ap.add_argument("--move-to", type=Path, default=None,
                    help="Relocate redundant files here instead of reporting only. "
                         "Reversible: nothing is deleted, the tree shape is kept.")
    ap.add_argument("--keep-handwriting", action="store_true",
                    help="Do not treat handwritten files as prunable.")
    args = ap.parse_args(argv)

    root = args.input.resolve()
    if not root.is_dir():
        print(f"No such folder: {root}", file=sys.stderr)
        return 2

    pdfs = find_pdfs(root)
    if not pdfs:
        print(f"No PDFs under {root}", file=sys.stderr)
        return 1

    hand = [] if args.keep_handwriting else [p for p in pdfs if is_handwriting(p, root)]
    rest = [p for p in pdfs if p not in set(hand)]

    print(f"{len(pdfs)} PDF(s) under {root}\n")
    if hand:
        print(f"── handwriting ({len(hand)}) " + "─" * 30)
        for p in hand:
            print(f"   {p.relative_to(root)}")
        print()

    print(f"hashing pages of {len(rest)} file(s)...", file=sys.stderr)
    sigs: dict[Path, list[int]] = {}
    md5: dict[str, list[Path]] = defaultdict(list)
    for p in rest:
        md5[hashlib.md5(p.read_bytes()).hexdigest()].append(p)
        try:
            sigs[p] = page_hashes(p)
        except Exception as exc:
            print(f"   ! could not read {p.name}: {exc}", file=sys.stderr)
            sigs[p] = []

    prunable: list[tuple[Path, str]] = [(p, "handwriting") for p in hand]

    identical = [v for v in md5.values() if len(v) > 1]
    if identical:
        print(f"── byte-identical ({len(identical)} group(s)) " + "─" * 18)
        for group in identical:
            keep, *drop = sorted(group)
            print(f"   keep {keep.relative_to(root)}")
            for d in drop:
                print(f"   drop {d.relative_to(root)}")
                prunable.append((d, "identical copy"))
        print()

    # Subset: every page of A also appears in B, and B has more.
    subsets: list[tuple[Path, Path]] = []
    done = {p for group in identical for p in sorted(group)[1:]}
    for a in rest:
        if a in done or not sigs[a]:
            continue
        for b in rest:
            if a is b or b in done or not sigs[b] or len(sigs[b]) <= len(sigs[a]):
                continue
            if contains_all(sigs[a], sigs[b], args.tolerance) == len(sigs[a]):
                subsets.append((a, b))
                done.add(a)
                break
    if subsets:
        print(f"── redundant: every page already in another file ({len(subsets)}) " + "─" * 5)
        for a, b in subsets:
            print(f"   drop {a.relative_to(root)}  ({len(sigs[a])}p)")
            print(f"     ⊂  {b.relative_to(root)}  ({len(sigs[b])}p)")
            prunable.append((a, f"subset of {b.name}"))
        print()

    # Spacing variants, grouped by name within a folder.
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in rest:
        key = str(p.parent) + "||" + SPACING_MARK.sub("", fold(p.stem)).strip(" -_()")
        groups[key].append(p)
    spacing = [v for v in groups.values() if len(v) > 1]
    if spacing:
        print(f"── 'cách dòng' spacing variants ({len(spacing)} group(s)) " + "─" * 12)
        for group in spacing:
            ranked = sorted(group, key=lambda p: len(sigs.get(p, [])))
            keep, *drop = ranked  # fewest pages = most content per page
            print(f"   keep {keep.relative_to(root)}  ({len(sigs.get(keep, []))}p, compact)")
            for d in drop:
                print(f"   drop {d.relative_to(root)}  ({len(sigs.get(d, []))}p, spaced)")
                if d not in done:
                    prunable.append((d, "spacing variant"))
        print()

    pages_saved = sum(len(sigs.get(p, [])) for p, _ in prunable if p in sigs)
    print("─" * 60)
    print(f"{len(prunable)} file(s) prunable "
          f"({pages_saved} page(s) of hashed files, plus {len(hand)} handwritten file(s))")

    if args.emit_exclude:
        print("\n--exclude arguments:")
        pats = sorted({f"*{fold(p.stem)}*" for p, _ in prunable})
        print("  " + " ".join(f"--exclude '{p}'" for p in pats))

    if args.move_to:
        dest_root = args.move_to.resolve()
        print(f"\nmoving {len(prunable)} file(s) to {dest_root}")
        for p, why in prunable:
            dest = dest_root / p.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
            print(f"   {p.relative_to(root)}   [{why}]")
        print("Nothing was deleted - move the folder back to undo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
