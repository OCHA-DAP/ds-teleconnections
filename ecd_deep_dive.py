"""Eastern Chad and Darfur — the 2026 rainy season and the 2027 lean season (builder = "ecd").

Restricted page (staticrypt, like the DRC page). The analysis was run on 17 November 2026 by the scripts in
deep_dives/data/eastern-chad-darfur/analysis/ (order and inputs in the README there); their one output, page_data.json,
is committed next to them. This builder fills deep_dives/data/eastern-chad-darfur/template.html with it. Charts, tiles
and the estimate box are computed in the browser from the data. The few sentences hard-written in the template are
guarded by _check(): the build stops when the data no longer support them.

    uv run python ecd_deep_dive.py                                   # build out/eastern-chad-darfur/ (gitignored)
    SITE_PASSWORD=... uv run python ecd_deep_dive.py --encrypt       # + docs/enso/eastern-chad-darfur/ and the card

Refresh: rerun the analysis scripts (they rewrite page_data.json), reword the template where the build complains, then
build with --encrypt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLUG = "eastern-chad-darfur"
SPEC = ROOT / "deep_dives" / f"{SLUG}.toml"
DATA = ROOT / "deep_dives" / "data" / SLUG
OUT = ROOT / "out" / SLUG
DOCS = ROOT / "docs" / "enso" / SLUG


def methods(D):
    alt = D["rain_alt"]
    im = {s: alt[s]["imerg_late"].get("2026") for s in ("chad", "darfur")}
    def det(summary, body, id_):
        return f'<details class="m" id="{id_}"><summary>{summary}</summary>{body}</details>'
    return "".join([
        det("Area", """<p>Eastern Chad = the provinces of Ouaddaï, Wadi Fira and Sila
        (COD-AB 2023). Darfur = North, South, West, East and Central Darfur (COD-AB). The two sides
        are kept apart throughout. Rainfall and vegetation are averaged only over 0.5° cells that get
        at least 100 mm in an average June–September (CHIRPS v2, 1991–2020). This drops the
        desert north of North Darfur and Wadi Fira. FAO and JRC report vegetation by admin unit, and
        units are weighted by their area inside that belt (cropland area per unit is not published).
        FAO still uses the three pre-2012 Darfur states and calls Wadi Fira by its old name, Biltine.</p>""", "m-area"),
        det("Rainfall", f"""<p>CHIRPS v2 and v3, monthly, June–September total, 1981–2026. History
        from the team's 0.5° cubes (ds-precip-intercomparison). August (v3), September and October 2026
        were read from the CHC files on 17 November 2026. Anomaly = % of the value the 1981–2025
        linear trend gives for that year; % of the 1991–2020 average is in the tooltips. Both
        versions are shown because they disagree, by up to about 20 points in some years
        (2007, 2008, 2022); in CHIRPS v2, 2008 was about as dry as 2026. Station input in Darfur is sparse
        and largely absent since 2023, so CHIRPS there leans on satellite estimates.</p>
        <p>Cross-checks, not used: IMERG Late puts June–September 2026 at {im['chad']}% (eastern Chad)
        and {im['darfur']}% (Darfur) of its 2001–2020 average; CHIRPS on the same baseline gives
        {alt['chad']['chirps_2001_2020']}% and {alt['darfur']['chirps_2001_2020']}%. ERA5 is much drier
        over the Sahel in 2025–2026 than any gauge-based product and is known to drift there, so it is
        left out.</p>""", "m-rain"),
        det("Vegetation", f"""<p><b>JRC ASAP biomass</b>: cumulative FPAR z-score on cropland
        (and rangeland in the CSV), during the growing cycle, by admin-1, read at the 21 October
        dekad. ASAP standardises against its 2001–present record. Its export labels the series MODIS,
        but JRC's current FPAR record joins MODIS (to 2023) with intercalibrated VIIRS
        (Seguini et al. 2026, ESSD 18:309). If recent dekads come from VIIRS, a step at the switch
        cannot be excluded, and 2024–2025 sit at the top of the record just before 2026 sits at
        the bottom. <b>FAO ASIS</b> (METOP-AVHRR, a different
        satellite): vegetation health index (VHI; the file has no land-type column, so it appears to be
        the admin-unit mean rather than a cropland mean) and agricultural stress index (ASI, % of cropland with VHI below 35), mean of the
        dekads 1 July–21 October, 1984–2026. FAO's annual ASI does not exist for 2026 yet, and the
        dekadal mean ranks years only roughly like it (rank correlation 0.57–0.87 by unit), so the
        page gives no ASI rank. FAO rates 2026 {D['ranks']['chad']['vhi']}th (eastern Chad) and
        {D['ranks']['darfur']['vhi']}th (Darfur) lowest since 2001 on VHI: the record low rests on ASAP
        alone. No detrending: extrapolating a 25-year line to 2026 would overstate the anomaly.
        The GeoSahel biomass used by the Chad drought framework covers West Africa only and its
        server did not respond.</p>""", "m-veg"),
        det("Yields", """<p>FEWS NET Data Warehouse crop production statistics for Sudan
        (Federal Ministry of Agriculture to 2015, FAO CFSAM from 2016, the latter kept in 2016 where
        both exist). Sorghum and millet, traditional rainfed sector, main season. Greater Darfur
        yield = summed production / summed harvested area of the states reported (3 states to 2011,
        5 from 2012; 4 in 2023). Expressed as % of the median yield of the surrounding years (within five
        years either side, same source, the year itself left out), because the level of the series
        shifts several times (high in 1975–1983, low 1985–2013, a jump in 2014–2015, and again with
        the source change in 2016). The series ends in 2023. Several values look unreliable (2014,
        2015, 2018–2019 well above anything earlier), so read single points with care. The Data
        Warehouse has no series for Chad.</p>""", "m-yield"),
        det("Food insecurity", """<p><b>FEWS NET</b>: the classification maps,
        2009–2026, from FEWS NET's regional shapefile archive (East and West Africa) to 2021 and
        the Data Warehouse round packages from 2022. Lean season of year Y = the June round's
        June–September projection (the July round's July–September projection before 2016).
        Post-harvest = the October round's current situation. Shares are of the population
        living in areas classified at that phase or worse, with WorldPop 2018 (1 km) as fixed
        weights. Refugee camps and IDP-camp layers are not in these figures. Rounds where more than a
        quarter of the population is unclassified are dropped. Maps start in July 2011: until
        April 2011 FEWS NET used its own severity scale. There are no maps from December 2024 to
        October 2025.</p>
        <p><b>Cadre Harmonisé</b> (eastern Chad): the public HDX consolidated file, rows
        marked for use. Resident population by département; refugees and returnees are analysed
        separately from 2024 and shown apart. Lean season = the June–August projection, made in
        March (July update in 2023). For 2026, Dar Tama and Iriba were not analysed in March and
        come from the November 2025 exercise. <b>IPC</b> (Darfur): the team's IPC mirror
        (IPC API), state level, the analysis covering August (current if any, else the latest
        projection). The June–September 2026 projection classified 48% of Darfur's population and
        the share is of those classified; 2019 covers 4 states; 2025 uses the September analysis.</p>""", "m-fs"),
        det("Reading the scatter plots and the estimate", """<p>One point per season and side.
        Seasons from 2023 on are drawn as open rings and left out of the correlations: Darfur is at
        war and eastern Chad has taken in about a million refugees, so outcomes there are not set
        by the rain. The correlations are weak (|r| up to about 0.5 for eastern Chad's FEWS NET
        series, about zero for Darfur's), consistent with the team's cross-country finding that
        indicators carry little skill at province scale. So the page does not fit a line to read
        off a 2027 number. The estimate uses analogue seasons (the five pre-2023 seasons with the
        lowest end-of-season biomass on each side, all less dry than 2026) and what followed them,
        plus FEWS NET's own projections, which already take the 2026 season into account. Darfur's
        yield record is noisy: harvested area is identical in 2018 and 2019, and 2014–2015 and
        2018–2019 are well above anything earlier.</p>""", "m-read"),
        det("Humanitarian needs", """<p>Team HPC mirror: JIAF intersectoral final
        severity by admin-2 from the 2025 and 2026 HNO workbooks on HDX; intersectoral people in
        need from HDX HAPI (2025; the 2026 plans publish it nationally only).</p>""", "m-hnrp"),
        det("Not included", """<p>Conflict events, displacement and market prices, which drive
        much of the outcome since 2023. Pasture and water for livestock beyond what the
        rangeland biomass series shows. Planted area: the yield estimate says nothing about how
        much was planted.</p>""", "m-not"),
    ])


def _check(D: dict) -> None:
    """Stop the build when a sentence hard-written in the template no longer matches the data."""
    bad = []
    a1 = {d["pc"]: (d["v2"] + d["v3"]) / 2 for d in D["adm1_2026"]}
    if not all(40 <= a1[pc] <= 60 for pc in ("SD02", "TCD17")):
        bad.append("lead: North Darfur and Wadi Fira 'about half their usual rain'")
    for s in ("chad", "darfur"):
        mon = {m["m"]: (m["v2"] + m["v3"]) / 2 / m["norm"] for m in D["rain_month"] if m["side"] == s}
        if min((6, 7, 8, 9), key=mon.get) != 8:
            bad.append(f"lead: August the weakest month ({s})")
        rk = D["ranks"][s]
        if rk["rain"] != 1 or rk["bio"] != 1 or rk["n"] != 26:
            bad.append(f"lead/estimate: driest since 2001 and lowest biomass of 26 ({s})")
        if not 2 <= rk["vhi"] <= 10:
            bad.append(f"lead: FAO VHI 'poor but not exceptional' ({s}, rank {rk['vhi']})")
    r = {(d["side"], d["y"]): d for d in D["rain"]}
    if min(r[(s, 2008)]["pt2"] - r[(s, 2026)]["pt2"] for s in ("chad", "darfur")) > 5:
        bad.append("lead: 'in CHIRPS v2, 2008 comes close'")
    chad = {pc: v for pc, v in a1.items() if pc.startswith("TCD")}
    if max(chad, key=chad.get) != "TCD21" or min(chad, key=chad.get) != "TCD17":
        bad.append("estimate: Sila got the most rain, Wadi Fira the least")
    ad = D["analogues"]["darfur"]
    near = [y for y, v in ad["rain_full"].items() if abs(v - r[("darfur", 2026)]["pt"]) <= 5]
    if near != ["1984"]:
        bad.append(f"estimate: 1984 the one earlier season about as dry ({near})")
    yl = [v for y, v in ad["yield_biomass"].items() if y != "2015"]
    if not (min(yl) >= 60 and ad["yield_rain_full"]["1984"] < 70 and ad["yield_2023"] < 50):
        bad.append("estimate: 'below about two-thirds of normal' no longer follows")
    if next(d for d in D["yield"] if d["y"] == 2023)["n"] != 4:
        bad.append("estimate: 2023 'four of five states reported'")
    ipc26 = next(d for d in D["ipcch"] if d["side"] == "darfur" and d["y"] == 2026 and d["kind"] == "lean")
    if "48%" not in ipc26["flag"]:
        bad.append(f"captions: IPC 2026 classified 48% ({ipc26['flag']})")
    if bad:
        raise SystemExit("ecd: the data no longer support:\n  " + "\n  ".join(bad))


def render(spec: dict) -> str:
    D = json.loads((DATA / "page_data.json").read_text(encoding="utf-8"))
    _check(D)
    tpl = (DATA / "template.html").read_text(encoding="utf-8")
    for token in ("/*__DATA__*/null", '"/*__STAMP__*/"', '/*__METHODS__*/""'):
        if tpl.count(token) != 1:
            raise SystemExit(f"ecd: template must hold {token} exactly once")
    stamp = spec.get("page", {}).get("stamp") or date.today().strftime("%-d %B %Y")
    return (tpl.replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":"), ensure_ascii=False))
               .replace('"/*__STAMP__*/"', json.dumps(stamp))
               .replace('/*__METHODS__*/""', json.dumps(methods(D))))


def write_index() -> None:
    """Re-render the /enso/ card index from every deep_dives/*.toml (restricted cards say so)."""
    import enso_deep_dive as edd
    specs = [tomllib.loads(p.read_text(encoding="utf-8")) | {"slug": p.stem} for p in sorted((ROOT / "deep_dives").glob("*.toml"))]
    (ROOT / "docs" / "enso" / "index.html").write_text(edd.render_index(specs, 0), encoding="utf-8")


def encrypt() -> None:
    """staticrypt the page into docs/enso/eastern-chad-darfur/ and check no plaintext survives (SITE_PASSWORD)."""
    pw = os.environ.get("SITE_PASSWORD")
    if not pw:
        raise SystemExit("set SITE_PASSWORD to encrypt")
    DOCS.mkdir(parents=True, exist_ok=True)
    subprocess.run(["npx", "-y", "staticrypt@3.5.4", str(OUT / "index.html"), "-d", str(DOCS), "--short", "--remember", "30",
                    "--template-title", "Eastern Chad and Darfur 2026: restricted",
                    "--template-instructions", "OCHA Centre for Humanitarian Data. Password shared internally."],
                   check=True, env=dict(os.environ, STATICRYPT_PASSWORD=pw), stdout=subprocess.DEVNULL)
    enc = (DOCS / "index.html").read_text(encoding="utf-8")
    if "staticrypt" not in enc:
        raise SystemExit(f"{DOCS}/index.html is not a staticrypt page")
    for marker in ('class="tile"', "CHIRPS", "Kimiti", "FEWS NET", "adm1_2026"):
        if marker in enc:
            raise SystemExit(f"{DOCS}/index.html contains plaintext ({marker})")
    print(f"  encrypted -> {DOCS}/index.html")
    write_index()


def build(spec: dict | None = None, *_args, encrypt_page: bool = False) -> None:
    spec = spec or tomllib.loads(SPEC.read_text(encoding="utf-8"))
    page = render(spec)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"  wrote {OUT}/index.html")
    if encrypt_page:
        encrypt()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encrypt", action="store_true", help="also staticrypt into docs/enso/eastern-chad-darfur/ (needs SITE_PASSWORD)")
    build(encrypt_page=ap.parse_args().encrypt)


if __name__ == "__main__":
    main()
