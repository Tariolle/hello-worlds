#!/usr/bin/env python
"""Repaint plot backgrounds to match the Metropolis slide background.

Called by the presentation Makefile before the LaTeX passes. Every figure the deck
embeds (the ``\\includegraphics`` names in ``main.tex``) gets its white / near-white
pixels replaced with ``SLIDE_BG``, so the plots blend into the slide instead of
sitting on a bright white card.

Non-destructive: it writes only into this folder's ``figures/`` directory, which is
listed *first* on ``main.tex``'s ``\\graphicspath`` and therefore shadows the copies
under ``../results/*``. The canonical experiment artifacts in ``results/`` are left
untouched. For a figure that has no local copy yet, the source is pulled fresh from
``results/`` once, then recolored here.

Idempotent: re-running finds no near-white pixels left, so it is a no-op on the pixels.

  python recolor_figures.py            # recolor every figure the deck includes
  python recolor_figures.py --refresh  # discard figures/ copies, repull from results/, recolor
  python recolor_figures.py a.png ...  # recolor specific files in place
"""
import os
import re
import shutil
import sys

try:
    from PIL import Image
except ImportError:  # keep `make` working even if Pillow is absent — just skip.
    print("[recolor_figures] Pillow not installed; skipping background recolor "
          "(pip install pillow to enable).", file=sys.stderr)
    sys.exit(0)

# Metropolis palette — mirrors the \definecolor block in main.tex.
TRAIN_C = "#2E86AB"
VAL_C = "#E51F13"
LR_C = "#7FB069"
SLIDE_BG = "#FAFAFA"
WHITE_THRESHOLD = 235

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
MAIN_TEX = os.path.join(HERE, "main.tex")

# Where the deck's figures live, mirroring main.tex's \graphicspath (minus figures/).
SOURCE_DIRS = [os.path.join(HERE, p) for p in (
    "../results/label_eff", "../results/benchmark", "../results/robustness",
    "../results/calibration", "../results/latent", "../results/loss",
    "../results/riemann", "..",
)]


def _replace_near_white(image):
    """Replace opaque white/near-white pixels with SLIDE_BG, in place. Returns image."""
    pixels = image.load()
    replacement = tuple(int(SLIDE_BG[i:i + 2], 16) for i in (1, 3, 5))
    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = pixels[x, y]
            if a > 0 and r >= WHITE_THRESHOLD and g >= WHITE_THRESHOLD and b >= WHITE_THRESHOLD:
                pixels[x, y] = (*replacement, a)
    return image


def savefig_slide_bg(fig, path):
    """Save a Matplotlib figure, then repaint white/near-white pixels to SLIDE_BG.

    Drop-in replacement for ``fig.savefig(path)`` inside the plotting scripts.
    """
    fig.savefig(path, bbox_inches="tight")
    image = Image.open(path).convert("RGBA")
    _replace_near_white(image).save(path)


def recolor_png(path):
    """Repaint an already-saved PNG in place (no live figure needed)."""
    image = Image.open(path).convert("RGBA")
    _replace_near_white(image).save(path)


def included_figures():
    """Figure basenames the deck embeds, parsed from main.tex \\includegraphics."""
    tex = open(MAIN_TEX, encoding="utf-8").read()
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    return [n if n.lower().endswith(".png") else n + ".png" for n in names]


def _resolve(name, refresh=False):
    """Locate the figure to recolor and return (path, origin-label).

    Normal mode reuses an existing ``figures/`` copy; ``refresh`` discards it and
    repulls a fresh copy from ``results/*`` (falling back to the local copy only
    when no ``results/`` source exists, e.g. a deck-only hand-made figure).
    """
    local = os.path.join(FIG_DIR, name)
    if not refresh and os.path.exists(local):
        return local, "figures/"
    for d in SOURCE_DIRS:
        src = os.path.join(d, name)
        if os.path.exists(src):
            os.makedirs(FIG_DIR, exist_ok=True)
            shutil.copyfile(src, local)
            return local, os.path.relpath(d, HERE)
    if os.path.exists(local):  # refresh requested but no results/ source to repull from
        return local, "figures/ (no source)"
    return None, None


def main(argv):
    refresh = "--refresh" in argv
    files = [a for a in argv[1:] if not a.startswith("-")]
    if files:  # explicit file list -> recolor those in place
        for p in files:
            recolor_png(p)
            print(f"[recolor_figures] recolored {p}")
        return
    done = 0
    for name in included_figures():
        path, origin = _resolve(name, refresh=refresh)
        if path is None:
            print(f"[recolor_figures] WARNING: {name} not found on the figure path", file=sys.stderr)
            continue
        recolor_png(path)
        print(f"[recolor_figures] {name:<28} <- {origin}")
        done += 1
    verb = "repulled + repainted" if refresh else "repainted"
    print(f"[recolor_figures] {done} figure(s) {verb} to {SLIDE_BG} in figures/")


if __name__ == "__main__":
    main(sys.argv)
