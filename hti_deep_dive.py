"""Haiti — the spring 2027 (March–June) rains after the 2026/27 El Niño (builder = "hti").

The analysis was run on 6 November 2026 by the scripts in deep_dives/data/hti/analysis/ (order and inputs in the
README there). Their outputs are committed next to them: page_data.json (department and national figures plus the
simplified department outlines), national_springs.json (national CHIRPS/ERA5 spring totals and winter Niño3.4 by
year) and results/ (the full tables behind the page). This builder fills deep_dives/data/hti/template.html with those
numbers. The charts, maps, tooltips and the department table are drawn from the data, so they always follow it. The
page text is hand-written: _check() stops the build when its key figures drift from the data (national typical total
and range, combined outlook, every row of the national evidence table, the 105–125% span, the department change span,
the wettest/driest department, the agreement classes, and the figures in the /enso/ card's one-line summary). Other
figures in the prose (onset, dry-spell, impact and forecast-provider notes) are not guarded: reread them on refresh.

    uv run python hti_deep_dive.py              # build out/hti/ (gitignored) to review
    uv run python hti_deep_dive.py --publish    # + docs/enso/hti/ and the /enso/ card index

Refresh: rerun the analysis scripts with the new forecasts, copy their outputs over the committed ones, reword the
template where the build complains, then build with --publish.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLUG = "hti"
SPEC = ROOT / "deep_dives" / f"{SLUG}.toml"
DATA = ROOT / "deep_dives" / "data" / SLUG
OUT = ROOT / "out" / SLUG
DOCS = ROOT / "docs" / "enso" / SLUG


# Department label positions nudged off neighbours and the coast (projected map units, checked inside each outline).
LABEL_AT = {"HT07": (240, 690), "HT10": (430, 612), "HT09": (470, 165)}


def _r10(x: float) -> int:
    """Round half up to the nearest 10 (Python's round() rounds halves to even)."""
    return int(x / 10 + 0.5) * 10


def _median(xs: list) -> int:
    xs = sorted(xs)
    n = len(xs)
    return round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def _agreement(v: dict) -> str:
    """The rule the agreement map states: CHIRPS/ERA5 correlation of spring rain with winter Niño3.4."""
    if v["r_ch"] >= 0.15 and v["r_er"] >= 0.15:
        return "both"
    if v["r_ch"] >= 0.20 and v["r_er"] < 0.10:
        return "chirps"
    if v["r_ch"] < 0.20 and v["r_er"] < 0.10 and v["ch_an"][0] < 100 and v["ch_an"][2] < 100:  # 1983, 1998
        return "unc"
    return "?"


def _check(page: str, d: dict, one_line: str) -> None:
    """Stop the build when a figure written into the template text no longer matches the data."""
    m = d["m"]
    nat = m["HT"]
    depts = {k: v for k, v in m.items() if k != "HT"}
    row = lambda c, lo, hi: f'<td class="n">{c}%</td><td class="n">{lo}–{hi}%</td>'  # noqa: E731
    centres = [_median(nat["ch_an"]), _median(nat["er_an"]), _median(nat["im_an"]), nat["ch_reg"][0], nat["er_reg"][0],
               nat["nmme"][0], nat["s5"][0]]
    expect = {
        row(_median(nat["ch_an"]), min(nat["ch_an"]), max(nat["ch_an"])): "evidence row: CHIRPS past springs",
        row(_median(nat["er_an"]), min(nat["er_an"]), max(nat["er_an"])): "evidence row: ERA5 past springs",
        row(_median(nat["im_an"]), min(nat["im_an"]), max(nat["im_an"])): "evidence row: IMERG past springs",
        row(*nat["ch_reg"]): "evidence row: CHIRPS regression",
        row(*nat["er_reg"]): "evidence row: ERA5 regression",
        row(*nat["nmme"]): "evidence row: NMME",
        row(*nat["s5"]): "evidence row: SEAS5",
        f'{nat["s5_pct"]}th percentile of its own history': "SEAS5 percentile",
        f"centres between {min(centres)}% and {max(centres)}%": "span of the evidence centres",
        f'{nat["central"]}% of typical': "tile: combined central %",
        f'{nat["typ_mm"]} mm': "typical national total",
        f'between {nat["typ_p10"]} and {nat["typ_p90"]} mm': "typical national range",
        f'{nat["central"]}% ({nat["central_mm"]} mm)': "combined national central estimate",
        f'{nat["lo"]}–{nat["hi"]}% ({nat["lo_mm"]}–{nat["hi_mm"]} mm)': "combined national likely range",
        f'{min(nat["ch_an"])}–{max(nat["ch_an"])}%': "CHIRPS past strong El Niño springs",
        f'{min(nat["er_an"])}–{max(nat["er_an"])}%': "ERA5 past strong El Niño springs",
    }
    problems = [why for s, why in expect.items() if s not in page]
    # tiles round the 2027 range to the nearest 10 mm and the central value to "about 500 mm"
    if f'Likely range {_r10(nat["lo_mm"])}–{_r10(nat["hi_mm"])} mm' not in page:
        problems.append("rounded national likely range in the tile")
    if int(nat["central_mm"] / 100 + 0.5) * 100 != 500:
        problems.append('"About 500 mm" tile')
    for s_, why in {f'typical {nat["typ_mm"]} mm': "card: typical total",
                    f'likely {_r10(nat["lo_mm"])}–{_r10(nat["hi_mm"])} mm': "card: likely range",
                    "about 500 mm": "card: central"}.items():
        if s_ not in one_line:
            problems.append(why)
    change = sorted(v["central"] - 100 for v in depts.values())
    if f"+{change[0]}% to +{change[-1]}% everywhere" not in page:
        problems.append("department change range (+x% to +y% everywhere)")
    wettest = max(depts.values(), key=lambda v: v["typ_mm"])
    driest = min(depts.values(), key=lambda v: v["typ_mm"])
    if (f'about {_r10(driest["typ_mm"])} mm in {driest["name"]}, {_r10(wettest["typ_mm"])} mm in {wettest["name"]}'
            not in page):
        problems.append("wettest/driest department in the notes")
    for v in depts.values():  # the agreement classes are hand-set in map_data.py; they must follow the stated rule
        if v.get("cls") != _agreement(v):
            problems.append(f'agreement class for {v["name"]} ({v.get("cls")} vs rule {_agreement(v)})')
    if problems:
        raise SystemExit("hti: page text no longer matches the data: " + "; ".join(problems))


def render(spec: dict) -> str:
    tpl = (DATA / "template.html").read_text(encoding="utf-8")
    data = json.loads((DATA / "page_data.json").read_text(encoding="utf-8"))
    for pc, (x, y) in LABEL_AT.items():
        data["geo"][pc]["lx"], data["geo"][pc]["ly"] = x, y
    springs = json.loads((DATA / "national_springs.json").read_text(encoding="utf-8"))
    for token in ("__DATA__", "__PTS__"):
        if tpl.count(token) != 1:
            raise SystemExit(f"hti: template must hold {token} exactly once")
    page = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__PTS__", json.dumps(springs))
    _check(page, data, spec["assessment"]["one_line"])
    return page


def write_index() -> None:
    import enso_deep_dive as edd
    specs = [tomllib.loads(p.read_text(encoding="utf-8")) | {"slug": p.stem} for p in sorted((ROOT / "deep_dives").glob("*.toml"))]
    (ROOT / "docs" / "enso" / "index.html").write_text(edd.render_index(specs, 0), encoding="utf-8")


def build(spec: dict | None = None, *_args, publish: bool = False) -> None:
    spec = spec or tomllib.loads(SPEC.read_text(encoding="utf-8"))
    page = render(spec)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"  wrote {OUT}/index.html")
    if publish:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "index.html").write_text(page, encoding="utf-8")
        write_index()
        print(f"  published -> {DOCS}/index.html and docs/enso/index.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="also write docs/enso/hti/ and the /enso/ card index")
    build(publish=ap.parse_args().publish)


if __name__ == "__main__":
    main()
