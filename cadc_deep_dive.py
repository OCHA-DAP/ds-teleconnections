"""Dry Corridor (Guatemala, Honduras, El Salvador): Primera and Postrera 2027 after the 2026/27 El Niño (builder = "cadc").

The analysis was run on 9 November 2026 by the scripts in deep_dives/data/dry-corridor/analysis/ (order and inputs in the
README there). Their outputs are committed next to them: page_data.json (every unit's figures, the NMME rows, the national
series for the charts and the simplified department outlines) and results/ (the full tables behind the page). This builder
fills deep_dives/data/dry-corridor/template.html with those numbers. Charts, maps, tooltips and tables are drawn from the
data, so they always follow it. The page text is hand-written: _check() stops the build when a figure written into it no
longer matches the data (the twelve bottom-line tiles, the spans quoted in the bullets and notes, the regression and NMME
figures, the onset and end-of-rains figures, the department ranges, the 2023/2025/2026 figures and the /enso/ card).
Figures quoted from outside sources (CPC, IRI, FEWS NET, EM-DAT) are not guarded: reread them on refresh.

    uv run python cadc_deep_dive.py              # build out/dry-corridor/ (gitignored) to review
    uv run python cadc_deep_dive.py --publish    # + docs/enso/dry-corridor/ and the /enso/ card index

Refresh: rerun the analysis scripts with the new forecasts, copy their outputs over the committed ones, reword the
template where the build complains, then build with --publish.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLUG = "dry-corridor"
SPEC = ROOT / "deep_dives" / f"{SLUG}.toml"
DATA = ROOT / "deep_dives" / "data" / SLUG
OUT = ROOT / "out" / SLUG
DOCS = ROOT / "docs" / "enso" / SLUG
AOIS = {"GT_AOI": "Guatemala", "HN_AOI": "Honduras", "SV_AOI": "El Salvador"}
NATS = ["GT", "HN", "SV"]
SIX = list(AOIS) + NATS


def _r10(x: float) -> int:
    """Round half up to the nearest 10 (Python's round() rounds halves to even)."""
    return int(x / 10 + 0.5) * 10


def _r(x: float) -> int:
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def _n(x: int) -> str:
    return f"{x:,}"


def _span(xs) -> str:
    return f"{min(xs)}–{max(xs)}"


def _pm(x: float) -> str:
    return ("+" if x > 0 else "−" if x < 0 else "") + str(abs(x))


def _check(page: str, d: dict, one_line: str) -> None:
    """Stop the build when a figure written into the template text no longer matches the data."""
    U = d["u"]
    dept = {k: v for k, v in U.items() if v["level"] == 1}
    expect: dict[str, str] = {}
    # the twelve tiles, in their season block
    tiles = page.split('<div class="tiles"', 1)[1].split("</div>\n  <div class=\"bottomline\">", 1)[0]
    blocks = {"pri": tiles.split("Postrera<small>", 1)[0], "pos": tiles.split("Postrera<small>", 1)[1]}
    problems = []
    for k, blk in blocks.items():
        for pc, cn in AOIS.items():
            v = U[pc][k]
            t = (f'<span class="hd">{cn}</span><span class="num">About {_n(_r10(v["central_mm"]))} mm</span>'
                 f'<span class="lbl">expected in 2027 ({v["central"]}% of typical), likely {_n(_r10(v["lo_mm"]))}–{_n(_r10(v["hi_mm"]))} mm. '
                 f'Typical {_n(v["typ_mm"])} mm; 8 seasons in 10 bring {_n(v["typ_p10"])}–{_n(v["typ_p90"])} mm.</span>')
            if t not in blk:
                problems.append(f"tile {k} {cn}")
    pri_aoi = [U[p]["pri"]["central"] for p in AOIS]
    pri_nat = [U[p]["pri"]["central"] for p in NATS]
    pos_aoi = [U[p]["pos"]["central"] for p in AOIS]
    expect[f"({_span(pri_aoi)}% in the framework departments, {_span(pri_nat)}% nationally)"] = "Primera central span"
    expect[f"Postrera leans slightly below typical, within the noise</b> ({_span(pos_aoi)}% in the framework departments)"] = "Postrera central span"
    # 2023, 2025, 2026
    six = lambda key, k="pri": [U[p][k][key] for p in SIX]  # noqa: E731
    expect[f"Primera came in at {_span(six('y2026'))}% of typical."] = "Primera 2026 (bullet)"
    expect[f"Primera 2026 then came in at {_span(six('y2026'))}% of typical in CHIRPS ({_span(six('y2026_er'))}% in ERA5)"] = "Primera 2026"
    expect[f"below typical in most areas ({_span(six('y2025'))}%)"] = "Primera 2025"
    late = d["y2026_late"]
    expect[f"September 2026 at {_span(late['CHIRPS_Sep'].values())}% in CHIRPS, and September–October at {_span(late['ERA5_SO'].values())}% in ERA5"] = "Sep/Sep–Oct 2026"
    y23 = six("y2023") + six("y2023_er")
    expect[f"came in at {_span(y23)}%."] = "2023 observed"
    # regressions (CHIRPS and ERA5, six areas)
    R = json.loads((DATA / "results" / "regression_summary.json").read_text(encoding="utf-8"))
    expect[f"which is why the regressions centre at {_span(R['pri_reg'])}%"] = "Primera regression centres"
    expect[f"Without the Atlantic term they centre at {_span(R['pri_n34only'])}%"] = "Niño3.4-only centres"
    expect[f"(correlations of {min(R['pri_loo']):.2f}–{max(R['pri_loo']):.2f} with observed rainfall, against {min(R['pri_loo_n34only']):.2f}–{max(R['pri_loo_n34only']):.2f})"] = "Primera skill"
    if not all(a > b for a, b in zip(R["pri_loo"], R["pri_loo_n34only"])):
        problems.append("gradient regression no longer beats Niño3.4 alone everywhere")
    expect[f"the regressions fit it at {_span(R['fit2023'])}%"] = "2023 fit"
    expect[f"would pull Primera toward {_span([U[p]['pri']['raw_nmme_reg'] for p in SIX])}%"] = "lingering El Niño sensitivity"
    expect[f"(cross-validated correlations of up to {max(R['pos_loo']):.2f}"] = "Postrera skill"
    sc = {s: [U[p]["pos"]["scen"][s][0] for p in SIX] for s in ("nino", "nina")}
    expect[f"Postrera would centre at about {_span(sc['nino'])}% of typical; if La Niña formed (−1 °C), at about {_span(sc['nina'])}%"] = "Postrera scenarios"
    inp = d["inputs"]
    expect[f"Niño3.4 at +{inp['n34_mjj'][0]:.2f} ± {inp['n34_mjj'][1]:.2f} °C in May–July"] = "MJJ input"
    expect[f"the tropical Atlantic at +{inp['tna_mjj'][0]:.2f} ± {inp['tna_mjj'][1]:.2f} °C"] = "TNA input"
    son = inp["n34_son"]
    expect[f"the 2027 input is wide: {'+' if son[0] >= 0 else '−'}{abs(son[0]):.2f} ± {son[1]:.2f} °C"] = "SON input"
    expect[f"Corrected, May–July sits at +{inp['n34_mjj'][0]:.2f} °C and September–October at {('+' if inp['n34_son'][0] >= 0 else '−')}{abs(inp['n34_son'][0]):.2f} °C"] = "corrected inputs"
    # NMME
    nm = d["nmme"]
    expect[f"puts Primera at {nm['GT']['pri']['c']}% in Guatemala, {nm['HN']['pri']['c']}% in Honduras and {nm['SV']['pri']['c']}% in El Salvador"] = "NMME Primera"
    expect[f"{nm['GT']['so']['c']}% in Guatemala, {nm['HN']['so']['c']}% in Honduras, {nm['SV']['so']['c']}% in El Salvador"] = "NMME Sep–Oct"
    if not all(nm[c]["pri"]["rank"] == "36/37" for c in NATS):
        problems.append("NMME 'second-wettest of 37'")
    # sub-seasons in past strong El Niño years (CHIRPS, six areas)
    mj = [U[p]["pri"]["MJ_ch"][i] for p in SIX for i in (0, 2, 4)]  # 1983, 1998, 2016
    expect[f"May–June rain came in at {_span(mj)}% of typical"] = "May–June 1983/1998/2016"
    on_below = [sum(x < 100 for x in U[p]["pos"]["ON_ch"]) for p in SIX]
    if sorted(on_below)[1] < 4:
        problems.append("October–November below typical in four of six years in most areas")
    # departments
    for k, lbl in (("pri", "pri"), ("pos", "pos")):
        wet = max(dept.values(), key=lambda v: v[k]["typ_mm"]); dry = min(dept.values(), key=lambda v: v[k]["typ_mm"])
        expect[f"about {_r10(dry[k]['typ_mm'])} mm in {dry['name']} to {_n(_r10(wet[k]['typ_mm']))} mm in {wet['name']}"] = f"{lbl} wettest/driest department"
        ch = [v[k]["central"] - 100 for v in dept.values()]
        expect[f"{_pm(min(ch))}% to {_pm(max(ch))}%"] = f"{lbl} department change span"
    lowpos = sorted(dept.values(), key=lambda v: v["pos"]["central"])[:3]
    expect[f"lowest in {lowpos[0]['name']}, {lowpos[1]['name']} and {lowpos[2]['name']}"] = "lowest Postrera departments"
    hipri = sorted(dept.values(), key=lambda v: -v["pri"]["central"])[:3]
    if not all(v["name"] in "Valle, La Unión and Islas de la Bahía" for v in hipri):
        problems.append("largest Primera rises (Valle, La Unión, Islas de la Bahía)")
    mjd = {pc: st.median(v["pri"]["MJ_ch"]) - 100 for pc, v in dept.items()}
    expect[f"below typical in {sum(x < 0 for x in mjd.values())} of {len(mjd)} departments"] = "May–June departments below typical"
    if sorted(dept[pc]["name"] for pc, x in mjd.items() if x >= 0) != ["Chiquimula", "Choluteca", "Jalapa"]:
        problems.append("May–June: only Chiquimula, Jalapa and Choluteca at or above typical")
    ond = {pc: st.median(v["pos"]["ON_ch"]) - 100 for pc, v in dept.items()}
    expect[f"fell short in the median of past strong El Niño seasons in {sum(x < 0 for x in ond.values())} of {len(ond)} departments"] = "Oct–Nov departments"
    # timing
    on = {pc: st.median(v["tim"]["onset_an"]) for pc, v in dept.items()}
    later = [x for x in on.values() if x > 0]
    expect[f"Onset came later than typical in {len(later)} of {len(on)} departments (median of the six seasons, by {_r(min(later))} to {_r(max(later))} days)"] = "onset by department"
    en = [st.median(v["tim"]["end_an"]) for v in dept.values()]
    expect[f"(median shifts of {_pm(_r(min(en)))} to {_pm(_r(max(en)))} days by department)"] = "end of rains by department"
    main = [U[p]["tim"]["main_after"] for p in SIX]; alt = [U[p]["tim"]["alt_after"] for p in SIX]
    expect[f"average delay at {_r(min(main))}–{_r(max(main))} days by country and framework area under the main rule ({_r(min(alt))}–{_r(max(alt))} days under the simpler one)"] = "average onset delay"
    shares = [U[p]["tim"]["main_late_after"] for p in SIX] + [U[p]["tim"]["alt_late_after"] for p in SIX]
    expect[f"averaged {_span(shares)}% in those years"] = "late-onset share"
    dur = [U[p]["tim"]["main_during"] for p in SIX]
    expect[f"were also late, by {_r(min(dur))}–{_r(max(dur))} days on average"] = "onset in El Niño seasons"
    typ_on = {AOIS.get(p, p): U[p]["tim"]["onset_typ"] for p in NATS}
    for pc, day in (("SV", "1 May"), ("GT", "7 May"), ("HN", "15 May")):
        if U[pc]["tim"]["onset_typ"] != {"1 May": 30, "7 May": 36, "15 May": 44}[day]:
            problems.append(f"typical onset {pc} ({typ_on})")
    ja = [sum(x >= 100 for x in U[p]["pri"]["JA_ch"]) for p in SIX]
    words = ["no", "one", "two", "three", "four", "five", "six"]
    expect[f"at or above typical in {words[min(ja)]} to {words[max(ja)]} of the six years"] = "July–August"
    # card
    for s_, why in {f"({_span(pri_aoi)}%)": "card: Primera span", f"({_span(pos_aoi)}%)": "card: Postrera span",
                    f"{_r(min(main))}–{_r(max(main))} days late": "card: onset delay"}.items():
        if s_ not in one_line:
            problems.append(why)
    problems += [why for s, why in expect.items() if s not in page]
    if problems:
        raise SystemExit("dry-corridor: page text no longer matches the data: " + "; ".join(problems))


def render(spec: dict) -> str:
    tpl = (DATA / "template.html").read_text(encoding="utf-8")
    data = json.loads((DATA / "page_data.json").read_text(encoding="utf-8"))
    if tpl.count("__DATA__") != 1:
        raise SystemExit("dry-corridor: template must hold __DATA__ exactly once")
    page = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
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
    ap.add_argument("--publish", action="store_true", help="also write docs/enso/dry-corridor/ and the /enso/ card index")
    build(publish=ap.parse_args().publish)


if __name__ == "__main__":
    main()
