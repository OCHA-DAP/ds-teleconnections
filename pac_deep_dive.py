"""Fiji, Solomon Islands, Tonga and Vanuatu: El Niño and the 2026/27 wet season (builder = "pac").

The analysis was run on 17 November 2026 by the scripts in deep_dives/data/pacific/analysis/ (order and inputs in the
README there); their output is committed as deep_dives/data/pacific/page_data.json (plus results/ for the tables behind
it). This builder fills deep_dives/data/pacific/template.html: every number in the text is a {{token}} computed here from
page_data.json, the charts are drawn from the same data, and _check() stops the build when a sentence's wording (driest,
within the range, and so on) no longer holds. The table of official outlooks comes from deep_dives/data/pacific/outlooks.html
and is not checked: reread it on refresh.

    uv run python pac_deep_dive.py              # build out/pacific/ (gitignored) to review
    uv run python pac_deep_dive.py --publish    # + docs/enso/pacific/ and the /enso/ card index
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLUG = "pacific"
SPEC = ROOT / "deep_dives" / f"{SLUG}.toml"
DATA = ROOT / "deep_dives" / "data" / SLUG
OUT = ROOT / "out" / SLUG
DOCS = ROOT / "docs" / "enso" / SLUG
BUILT = "17 November 2026"


def _r(x: float) -> int:
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def _ord(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _dry(rank: int) -> str:
    return "driest" if rank == 1 else f"{_ord(rank)} driest"


def _wet(rank: int, n: int) -> str:
    w = n - rank + 1
    return "wettest" if w == 1 else f"{_ord(w)} wettest"


def _span(xs) -> str:
    lo, hi = _r(min(xs)), _r(max(xs))
    return f"{lo}" if lo == hi else f"{lo}–{hi}"


def _to(xs) -> str:
    """Signed range with 'to' (en dashes between negative numbers read badly)."""
    f = lambda v: ("−" if v < 0 else "") + format(abs(v), ".2f")  # noqa: E731
    return f(min(xs)) if min(xs) == max(xs) else f"{f(min(xs))} to {f(max(xs))}"


def _rng(xs, fmt: str = ".2f") -> str:
    lo, hi = format(min(xs), fmt), format(max(xs), fmt)
    return lo if lo == hi else f"{lo}–{hi}"


# Drought responses in 2015/16, none funded by CERF (sources opened 17 Nov 2026; not checked by the build: reread on refresh)
RESP_2015 = {
    "FJ": ('Drought warning; water deliveries for 67,000 people '
           '(<a href="https://reliefweb.int/report/myanmar/asia-and-pacific-weekly-regional-humanitarian-snapshot-24-30-november-2015">OCHA</a>).'),
    "TO": ('Drought warning; water shipped to \'Eua '
           '(<a href="https://www.rnz.co.nz/international/pacific-news/288787/tonga-hoping-for-more-rain-as-it-considers-drought-options">RNZ</a>).'),
    "VU": ('After Cyclone Pam: government food for about 15,000 households '
           '(<a href="https://www.rnz.co.nz/international/programmes/datelinepacific/audio/201771681/drought-aid-distribution-begins-in-vanuatu">RNZ</a>); '
           '90,000 people targeted with food '
           '(<a href="https://reliefweb.int/report/myanmar/asia-and-pacific-weekly-regional-humanitarian-snapshot-24-30-november-2015">OCHA</a>).'),
    "SB": ('Drought watch; five areas of concern, no relief funding yet in November '
           '(<a href="https://www.rnz.co.nz/international/programmes/datelinepacific/audio/201777766/drought-awareness-needed-in-solomons-ndmo">RNZ</a>).'),
}


def U_prov_skill(d: dict, pc: str) -> float:
    return next(p["skill"] for p in d["provinces"] if p["pc"] == pc)


def tokens(d: dict) -> tuple[dict, list[str]]:
    U = d["units"]
    T: dict[str, str] = {"built": BUILT}
    problems: list[str] = []
    ix = U["FJ"]["enso_est"]
    T["n34_oct"] = f"{ix['oisst_oct']['v2026']:.2f}"
    T["n34_oct_max"] = f"{ix['oisst_oct']['max_hind']:.2f}"
    T["n34_oct_max_year"] = str(ix["oisst_oct"]["max_year"])
    T["psl_oct"], T["psl_oct_max"] = f"{ix['psl_oct']['v2026']:.2f}", f"{ix['psl_oct']['max_hind']:.2f}"
    T["oni_aso"], T["oni_aso_max"] = f"{ix['oni_aso']['v2026']:.2f}", f"{ix['oni_aso']['max_hind']:.2f}"
    if not all(ix[k]["v2026"] > ix[k]["max_hind"] for k in ix):
        problems.append("2026 Niño3.4 no longer above the 1982–2025 range in every series")
    T["n_fc"] = str(U["FJ"]["tri"]["NA"]["n"])
    T["n_obs"] = str(U["FJ"]["JASO"]["n"])
    VAR = ("fc", "fc_trend")  # El Niño-only forecast: three series, without and with a year term
    for u in ["FJ", "TO", "VU", "SB_SE", "SB_NW", "SB_C"]:
        na = U[u]["tri"]["NA"]
        T[f"{u}_NA"] = str(_r(na["fc"]))
        T[f"{u}_NA_rankw"] = _dry(na["rank"])
        T[f"{u}_NDJ"] = str(_r(U[u]["tri"]["NDJ"]["fc"]))
        T[f"{u}_JASO"] = str(_r(U[u]["JASO"]["pct"]))
        T[f"{u}_JASO_rankw"] = _dry(U[u]["JASO"]["rank"])
        oe = U[u]["obs_equiv"]
        T[f"{u}_oexp"], T[f"{u}_olo"], T[f"{u}_ohi"] = str(_r(oe["exp"])), str(_r(oe["lo"])), str(_r(oe["hi"]))
        o = dict(zip(U[u]["years"], U[u]["obs"]))
        T[f"{u}_o15"], T[f"{u}_o23"] = str(_r(o[2015])), str(_r(o[2023]))
    T["SB_NW_NA_rankw_wet"] = _wet(U["SB_NW"]["tri"]["NA"]["rank"], U["SB_NW"]["tri"]["NA"]["n"])
    three = ["FJ", "TO", "VU"]
    T["skill3"] = f"{min(U[u]['tri']['NA']['skill'] for u in three):.2f}–{max(U[u]['tri']['NA']['skill'] for u in three):.2f}"
    oi_ = lambda u: U[u]["enso_est"]["oisst_oct"]  # noqa: E731
    T["r2fc3_dt"] = _span([oi_(u)["r2_fc_dt"] * 100 for u in three])
    T["r2obs3_dt"] = _span([oi_(u)["r2_obs_dt"] * 100 for u in three])
    T["trend3"] = _span([U[u]["fc_trend_decade"] for u in three])
    # El Niño share: the Niño term over the gap between the neutral-year value and the forecast, six fits, capped at 100%
    def _shares(u):
        f = U[u]["tri"]["NA"]["fc"]
        return [min(1.0, -e[f"{v}_nino"] / (e[f"{v}_neutral"] - f)) for e in U[u]["enso_est"].values() for v in VAR]
    for u in ["FJ", "TO", "VU", "SB_SE"]:
        T[f"{u}_share"] = _span([s * 100 for s in _shares(u)])
        T[f"{u}_neutral"] = _span([e[f"{v}_neutral"] for e in U[u]["enso_est"].values() for v in VAR])
    if not (min(min(_shares(u)) for u in three) >= 0.6):
        problems.append("El Niño share of the gap no longer at least about 60% in Fiji, Tonga and Vanuatu ('most')")
    sd = [U[u]["enso_est"][k][f"{v}_sd"] for u in three for k in U[u]["enso_est"] for v in VAR]
    T["fit_sd"] = _span(sd)
    # index tile and method: OISST vs ERSST disagreement
    gap = ix["oisst_oct"]["v2026"] - ix["psl_oct"]["v2026"]
    T["oisst_ersst_gap"] = f"{gap:.1f}"
    if gap < 0.5:
        problems.append("OISST and ERSST no longer disagree by about 1 °C for October 2026")
    r2i = U["FJ"]["r2_obs_idx"]
    T["r2_fj_idx"] = f"{_r(r2i['n34a'] * 100)}% for Niño3.4, {_r(r2i['n3a'] * 100)}% for Niño3, {_r(r2i['n4a'] * 100)}% for Niño4 and {_r(r2i['n3m4'] * 100)}% for their difference"
    if abs(r2i["n34a"] - oi_("FJ")["r2_obs_dt"]) > 0.005:
        problems.append("Fiji observed share for Niño3.4 differs between the table and the method note")
    for u in three:
        r2u = U[u]["r2_obs_idx"]
        if max(r2u["n3a"], r2u["n3m4"]) > r2u["n34a"] + 0.01 or r2u["n4a"] > r2u["n34a"] + 0.03:
            problems.append(f"{u}: another index now explains observed rainfall clearly better than Niño3.4")
    n4 = dict(zip(d["n34"]["years"], d["n34"]["n4"]))
    T["n4_oct"] = f"{n4[2026]:.2f}"
    if n4[2026] > max(v for y, v in n4.items() if y < 2026):
        problems.append("October 2026 Niño4 no longer within its past range")
    for u in ("FJ", "TO"):
        if U[u]["r2_obs_idx"]["n4a"] > U[u]["r2_obs_idx"]["n34a"] - 0.03:
            problems.append(f"{u}: Niño4 now about as good as Niño3.4 (method note says only Vanuatu)")
    r2v = U["VU"]["r2_obs_idx"]
    T["r2_vu_n4"] = f"{_r(r2v['n4a'] * 100)}% for Niño4 against {_r(r2v['n34a'] * 100)}% for Niño3.4"

    # wording guards
    for u in ("FJ", "TO"):
        if U[u]["tri"]["NA"]["rank"] != 1:
            problems.append(f"{u} Nov–Apr forecast no longer the driest of all November forecasts")
    for u in three:
        tri = U[u]["tri"]
        if min(tri[k]["fc"] for k in ("NDJ", "DJF", "JFM", "FMA", "MAM")) != tri["NDJ"]["fc"]:
            problems.append(f"{u}: Nov–Jan no longer the driest trimester")
        if tri["NA"]["skill"] < 0.5:
            problems.append(f"{u}: skill below 0.5 ('moderate to high')")
    yrs = U["FJ"]["years"]
    for u in ("FJ", "TO"):
        obs = {y: v for y, v in zip(U[u]["years"], U[u]["obs"]) if v is not None}
        order = sorted(obs, key=obs.get)
        for y in (1982, 1997):
            if order.index(y) >= 5:
                problems.append(f"{u} {y}/{(y + 1) % 100:02d} no longer among the five driest observed")
        if not obs[2015] < 100 or not obs[2023] > 100:
            problems.append(f"{u}: 2015/16 no longer below or 2023/24 no longer above average")
    strong = [y for y, v in zip(d["n34"]["years"], d["n34"]["oct"]) if 1982 <= y <= 2025 and v >= 1.5]
    if strong != [1982, 1997, 2015, 2023]:
        problems.append(f"strong El Niño years changed: {strong}")
    if not U["SB_SE"]["tri"]["NA"]["fc"] < 90 or not U["SB_NW"]["tri"]["NA"]["fc"] > 105:
        problems.append("Solomon Islands south-east dry / north-west wet no longer holds")
    if U["SB_NW"]["tri"]["NA"]["skill"] >= 0.3:
        problems.append("SB north-west skill no longer low")
    nw_jfm = U["SB_NW"]["jfm_strong"]
    vals = sorted(nw_jfm.values())
    if not (sum(v > 105 for v in vals) == 3 and 95 <= vals[0] <= 105):
        problems.append(f"SB north-west Jan–Mar in strong El Niño years no longer 'three wetter, one about average': {nw_jfm}")
    T["nw_jfm_wet"] = "three of the four and about average in the other"
    T["nw_jfm_vals"] = ", ".join(f"{_r(v)}%" for v in nw_jfm.values())
    # band averages on the map
    B = d["band"]
    for k, v in B.items():
        T[f"band_{k}"] = str(_r(v))
    if not (B["era5_1982"] < 75 and B["era5_1997"] < 75 and B["era5_2015"] > B["era5_1997"] and B["era5_2023"] > 100 and B["seas5_2026"] < 65):
        problems.append(f"band averages no longer match the map caption: {B}")
    order = d["timing"]["order"]
    for u in three:
        strongest = order[min(range(len(order)), key=lambda i: d["timing"][u][i])]
        if strongest not in ("OND", "NDJ", "DJF"):
            problems.append(f"{u}: strongest El Niño drying no longer Oct–Feb ({strongest})")

    # CERF
    four = [c for c in d["cerf"] if c["iso3"] in ("FJI", "VUT", "TON", "SLB")]
    if any(c["type"] == "Drought" for c in four):
        problems.append("a CERF drought allocation to one of the four countries now exists")
    T["cerf_n"] = str(len(four))
    T["cerf_storm"] = str(sum(c["type"] == "Storm" for c in four))
    if sorted(c["type"] for c in four if c["type"] != "Storm") != ["Earthquake", "Flood", "Volcano"]:
        problems.append("CERF non-storm allocation types changed")
    if min(int(c["date"][:4]) for c in four) != 2014:
        problems.append("first CERF allocation to the four is no longer 2014")
    pac_drought = sorted((c["iso3"], c["date"][:4]) for c in d["cerf"] if c["type"] == "Drought")
    if pac_drought != [("MHL", "2013"), ("PNG", "2016")]:
        problems.append(f"Pacific CERF drought allocations changed: {pac_drought}")

    def obs_since(u, y0=2006):
        return {y: v for y, v in zip(U[u]["years"], U[u]["obs"]) if v is not None and y >= y0}

    def fc_at(u, y):
        return U[u]["fc"][U[u]["years"].index(y)]

    parts = []
    for u, nm in (("FJ", "Fiji"), ("TO", "Tonga"), ("VU", "Vanuatu")):
        o = obs_since(u)
        y = min(o, key=o.get)
        parts.append(f"{_r(o[y])}% in {nm} ({y}/{(y + 1) % 100:02d})")
        T[f"{u}_min06"] = f"{_r(o[y])}"
    for u in ("FJ", "TO"):
        fc06 = [v for yy, v in zip(U[u]["years"], U[u]["fc"]) if 2006 <= yy <= 2025 and v is not None]
        if not U[u]["tri"]["NA"]["fc"] < min(fc06):
            problems.append(f"{u}: 2026 forecast no longer below every 2006–2025 forecast")
    T["cerf_note"] = (
        "Since 2006 the driest observed wet seasons (ERA5) were " + ", ".join(parts[:-1]) + " and " + parts[-1] +
        ", with no CERF drought allocation. Past forecasts point to about "
        f"{T['FJ_oexp']}% in Fiji (80% range {T['FJ_olo']}–{T['FJ_ohi']}%), {T['TO_oexp']}% in Tonga "
        f"({T['TO_olo']}–{T['TO_ohi']}%) and {T['VU_oexp']}% in Vanuatu ({T['VU_olo']}–{T['VU_ohi']}%) this season. "
        "Papua New Guinea's 2016 allocation covered a dry-season drought (April–November 2015).")
    for u in three:
        if not U[u]["obs_equiv"]["exp"] < float(T[f"{u}_min06"]):
            problems.append(f"{u}: observed equivalent no longer below the driest season since 2006")

    # Tonga CHIRPS check of the rain so far
    if "chirps_jaso" not in d or "TO" not in d["chirps_jaso"]:
        raise SystemExit("pacific: page_data.json has no Tonga CHIRPS check: run chc_window.py and chc_units.py, then synth.py")
    C = d["chirps_jaso"]["TO"]
    T["TO_chirps"], T["TO_chirps_r"] = str(_r(C["pct"]["2026"])), f"{C['r_era5']:.2f}"
    T["TO_chirps_rankw"] = _dry(C["rank2026"])
    if C["rank2026"] > 3 or C["r_era5"] < 0.7:
        problems.append("Tonga CHIRPS check no longer supports 'among the driest'")

    # this season against 2015/16
    rows = []
    for u, nm in (("FJ", "Fiji"), ("TO", "Tonga"), ("VU", "Vanuatu"), ("SB_SE", "Solomon Is. south-east"), ("SB_NW", "Solomon Is. north-west")):
        j = U[u]["jaso_pct"]
        o15 = U[u]["obs"][U[u]["years"].index(2015)]
        j15, j26 = f"{_r(j['2015'])}%", f"{_r(j['2026'])}%"
        if u == "TO":
            cj = d["chirps_jaso"]["TO"]["pct"]
            j15 += f"<br><small>CHIRPS {_r(cj['2015'])}%</small>"
            j26 += f"<br><small>CHIRPS {_r(cj['2026'])}%</small>"
            if not cj["2026"] < cj["2015"]:
                problems.append("TO: CHIRPS no longer has Jul–Oct 2026 drier than 2015")
        if u == "FJ":
            g15 = [g["jaso_pct"]["2015"] for g in d["gauges"] if "2015" in g["jaso_pct"]]
            j15 += f"<br><small>gauges {_span(g15)}%</small>"
            T["FJ_g15"] = _span(g15)
            if not max(g15) < j["2015"] - 15:
                problems.append("Fiji gauges no longer much drier than ERA5 in Jul–Oct 2015")
        rows.append(f'<tr><td>{nm}</td><td class="n">{j15}</td><td class="n">{j26}</td>'
                    f'<td class="n">{_r(fc_at(u, 2015))}%</td><td class="n">{_r(U[u]["tri"]["NA"]["fc"])}%</td>'
                    f'<td class="n">{_r(o15)}%</td>' + ("" if u == "SB_NW" else f'<td{" rowspan=2" if u == "SB_SE" else ""}>{RESP_2015[u[:2]]}</td>') + '</tr>')
    T["cmp2015_rows"] = "\n".join(rows)
    if not (U["TO"]["jaso_pct"]["2026"] < U["TO"]["jaso_pct"]["2015"]):
        problems.append("TO: Jul–Oct 2026 no longer drier than Jul–Oct 2015")
    for u in ("FJ", "TO"):
        if not U[u]["tri"]["NA"]["fc"] < fc_at(u, 2015):
            problems.append(f"{u}: 2026 forecast no longer drier than the 2015 forecast")
    if not U["VU"]["jaso_pct"]["2026"] > U["VU"]["jaso_pct"]["2015"]:
        problems.append("VU: Jul–Oct 2026 no longer wetter than Jul–Oct 2015")

    # Fiji gauges
    G = {g["name"]: g for g in d["gauges"]}
    dry_side = [G[n] for n in ("Nadi", "Lautoka", "Labasa")]
    wet_side = [G[n] for n in ("Nausori", "Suva (Laucala Bay)")]
    T["gauge_note"] = (
        f"Fiji's rain gauges ({min(g['y0'] for g in d['gauges'])}–{max(g['y1'] for g in d['gauges'])}) follow ERA5 "
        f"(r = {_rng([g['r_era5'] for g in d['gauges']])}). SEAS5 matches the gauges at Nadi, Lautoka and Labasa "
        f"(r = {_rng([g['skill_gauge'] for g in dry_side])}) but not at Nausori and Suva ({_rng([g['skill_gauge'] for g in wet_side])}), "
        f"so the Central Division forecast is less reliable than its ERA5 skill ({U_prov_skill(d, 'FJ4'):.2f}) suggests. In 1997/98 "
        f"Nadi, Lautoka and Labasa recorded {_span([g['pct']['1997'] for g in dry_side])}% of average, against "
        f"{_span([G['Nadi']['era5_div_1997'], G['Labasa']['era5_div_1997']])}% in ERA5.")
    if not all(g["r_n34"] < w["r_n34"] for g in dry_side for w in wet_side):
        problems.append("Fiji gauges: dry side no longer more El Niño-sensitive than wet side")
    if not (min(g["skill_gauge"] for g in dry_side) > 0.6 and max(g["skill_gauge"] for g in wet_side) < 0.4):
        problems.append("Fiji gauge skill split (dry side good, Nausori/Suva poor) no longer holds")
    return T, problems


def render(spec: dict) -> str:
    tpl = (DATA / "template.html").read_text(encoding="utf-8")
    data = json.loads((DATA / "page_data.json").read_text(encoding="utf-8"))
    T, problems = tokens(data)
    for card_tok in re.findall(r"\{\{(\w+)\}\}", spec["assessment"]["one_line"]):
        problems.append(f"card one_line holds a token ({card_tok}): write the number")
    for k in ("FJ_NA", "TO_NA", "VU_NA", "SB_SE_NA", "SB_NW_NA"):
        if f"{T[k]}%" not in spec["assessment"]["one_line"]:
            problems.append(f"card one_line misses {k} = {T[k]}%")
    if problems:
        raise SystemExit("pacific: page text no longer matches the data: " + "; ".join(problems))
    page = tpl
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", page)) - set(T))
    if missing:
        raise SystemExit(f"pacific: template tokens without a value: {missing}")
    page = re.sub(r"\{\{(\w+)\}\}", lambda m: T[m.group(1)], page)
    for marker in ("__DATA__", "__OUTLOOKS__"):
        if page.count(marker) != 1:
            raise SystemExit(f"pacific: template must hold {marker} exactly once")
    page = page.replace("__OUTLOOKS__", (DATA / "outlooks.html").read_text(encoding="utf-8").strip())
    page = page.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
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
    ap.add_argument("--publish", action="store_true", help="also write docs/enso/pacific/ and the /enso/ card index")
    build(publish=ap.parse_args().publish)


if __name__ == "__main__":
    main()
