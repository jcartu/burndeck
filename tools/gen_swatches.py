#!/usr/bin/env python3
"""Generate deterministic 120x24 SVG swatch strips for every burndeck theme.

Reads each static/themes/<id>.js file, extracts the accent / accent2 / good /
warn / crit / void hex colors plus the theme name and tag from the `theme:` and
top-level `name:` / `tag:` fields, and writes docs/swatches/<id>.svg — a
rounded-rect strip of six equal color bands (void, accent, accent2, good, warn,
crit) with no text. Output is stable: fixed ordering, no timestamps.

Usage: python tools/gen_swatches.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THEMES_DIR = ROOT / "static" / "themes"
OUT_DIR = ROOT / "docs" / "swatches"

THEME_ORDER = [
    "oblivion", "ares", "k2049", "muthur", "jarvis", "nightcity",
    "arrakis", "lumon", "apex", "construct", "gargantua", "thegrid", "tokyo",
]

BAND_KEYS = ["void", "accent", "accent2", "good", "warn", "crit"]

W, H = 120, 24
RADIUS = 4


def extract(src):
    """Return (name, tag, {key: hex}) from a theme file's JS source."""
    name_m = re.search(r"name:\s*'([^']*)'", src)
    tag_m = re.search(r"tag:\s*'([^']*)'", src)
    name = name_m.group(1) if name_m else ""
    tag = tag_m.group(1) if tag_m else ""

    # Isolate the theme: { ... } block (first nested brace group after "theme:").
    ti = src.find("theme:")
    if ti == -1:
        raise ValueError("no theme: block found")
    depth = 0
    start = src.find("{", ti)
    end = start
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    block = src[start:end + 1]

    colors = {}
    for key in BAND_KEYS:
        m = re.search(
            rf"{key}:\s*'#([0-9a-fA-F]{{6}})'", block
        )
        if not m:
            raise ValueError(f"color {key!r} not found in theme block")
        colors[key] = "#" + m.group(1).lower()

    return name, tag, colors


def svg(colors):
    bw = W / len(BAND_KEYS)
    parts = []
    for i, key in enumerate(BAND_KEYS):
        x = round(i * bw, 3)
        w = round(bw, 3)
        # First and last bands get rounded outer corners; middle bands are plain rects.
        rx = RADIUS if i == 0 else 0
        # We draw left-rounded for the first band and right-rounded for the last
        # by using a per-band rect with selective corner radii.
        parts.append(f'  <rect x="{x}" y="0" width="{w}" height="{H}" fill="{colors[key]}" rx="{rx}"/>')
    inner = "\n".join(parts)

    # Build a clip path so only the outer corners are rounded.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">\n'
        f'  <defs>\n'
        f'    <clipPath id="c">\n'
        f'      <rect x="0" y="0" width="{W}" height="{H}" rx="{RADIUS}"/>\n'
        f'    </clipPath>\n'
        f'  </defs>\n'
        f'  <g clip-path="url(#c)">\n'
        f'{inner}\n'
        f'  </g>\n'
        f'</svg>\n'
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tid in THEME_ORDER:
        path = THEMES_DIR / f"{tid}.js"
        if not path.exists():
            print(f"SKIP {tid}: file not found", file=sys.stderr)
            continue
        src = path.read_text(encoding="utf-8")
        try:
            name, tag, colors = extract(src)
        except ValueError as e:
            print(f"FAIL {tid}: {e}", file=sys.stderr)
            sys.exit(1)
        out = OUT_DIR / f"{tid}.svg"
        out.write_text(svg(colors), encoding="utf-8")
        print(f"  {tid:12s} {name:20s} {tag}")
    print(f"\nWrote {len(THEME_ORDER)} swatches to {OUT_DIR}/")


if __name__ == "__main__":
    main()
