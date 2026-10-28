"""Venezuela El Niño 2026/27: a short slide deck, in English and Spanish, from the same analysis as the page (ven_deep_dive.py).

    uv run --with python-pptx python ven_slides.py [--publish]

For each language, writes out/ven/slides/ (English) and out/ven/slides/es/ (Spanish): simple slide figures, content.json
(every number on the slides, from the build) and an editable PowerPoint (native text, placeholders and a native chart, so it
can be edited). Where LibreOffice is installed it also renders the deck to PDF and one PNG per slide (pdftoppm), for checking
and for the slides page. The build stops if the two decks do not carry the same numbers. --publish copies the .pptx, the PNGs
and a page showing them to docs/enso/ven/slides/ and docs/enso/ven/slides/es/.

The same decks also live as Claude Slides artifacts, for comments; their slide files are written by hand from content.json, so
re-check them against it after a rebuild.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

import ven_deep_dive as V
from ven_deep_dive import _t

OUT = V.OUT / "slides"
DOCS = V.DOCS / "slides"
PPTX = {"en": "venezuela-el-nino-2026-27.pptx", "es": "venezuela-el-nino-2026-27-es.pptx"}
PAGE_URL = {"en": "https://ocha-dap.github.io/ds-teleconnections/enso/ven/",
            "es": "https://ocha-dap.github.io/ds-teleconnections/enso/ven/es/"}
FEWS_VEN = "https://fews.net/latin-america-and-caribbean/venezuela"
DARK, LIGHT, PAPER, INK, MUTED, ACCENT, PANEL = "3A2414", "FBF7F2", "FBF8F4", "1F2328", "5B6168", "B5541B", "F1ECE5"
VEN_XLIM, VEN_YLIM = (-73.6, -59.6), (0.6, 12.4)
TITLE_MAX = 125   # characters: two lines of the 26 pt bold title across 12.1 in
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15})
_DIR = OUT   # where the current language's files go (set in main)


def _out(lang: str) -> Path:
    return OUT / "es" if lang == "es" else OUT


def _save(fig, name: str) -> Path:
    p = _DIR / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return p


def _ven_axes(ax):
    ax.set_xlim(*VEN_XLIM)
    ax.set_ylim(*VEN_YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()


# --------------------------------------------------------------------------- figures: large type, one idea each
def fig_zone_map() -> Path:
    z, st = V.zones(), V.states()
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    z.plot(ax=ax, color=z.zone.map(V.ZONE_FILL), edgecolor="none")
    st.boundary.plot(ax=ax, color="#ffffff", lw=0.5)
    for _, row in z.iterrows():
        pt = max(getattr(row.geometry, "geoms", [row.geometry]), key=lambda q: q.area).representative_point()
        ax.text(pt.x, pt.y, row.zone, ha="center", va="center", fontsize=20, fontweight="bold", color="#0b0b0b")
    _ven_axes(ax)
    ax.legend(handles=[Patch(color=V.ZONE_FILL[k], label=f"{k}: {_t(*V.ZONE_ONSET_TXT[k])}") for k in V.ZONE_ONSET],
              title=_t("Rains start", "Inicio de las lluvias"), alignment="left", loc="lower left", fontsize=11, title_fontsize=11,
              frameon=False)
    return _save(fig, "zone_map")


def fig_vegnow() -> Path:
    """Cropland only (the page also maps rangeland)."""
    st = V.states()
    lookup = {V._norm(n): pc for pc, n in zip(st.pcode, st.name)}
    a = V.asap(220, 1, 2)
    z = a[a.date == a.date.max()].assign(pcode=lambda d: d.region_name.map(lambda n: lookup.get(V._norm(n)))).set_index("pcode")["value"]
    zb = [-3, -1.5, -1, -0.5, 0.5, 1, 1.5, 3]
    cm, nm = mcolors.ListedColormap(V.DIV), mcolors.BoundaryNorm(zb, len(V.DIV))
    fig, ax = plt.subplots(figsize=(6.4, 5.9))
    st.merge(z.rename("z"), left_on="pcode", right_index=True, how="left").plot(
        ax=ax, column="z", cmap=cm, norm=nm, edgecolor="#ffffff", lw=0.6, missing_kwds=dict(color=V.DRY_GREY))
    _ven_axes(ax)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), ax=ax, orientation="horizontal", fraction=0.05, pad=0.02)
    cb.set_ticks(zb[1:-1], labels=[V._num(f"{t:g}") for t in zb[1:-1]])
    cb.set_label(_t("cropland vegetation, z-score (warning level −1)", "vegetación de los cultivos, puntaje z (umbral de alerta −1)"),
                 fontsize=12)
    return _save(fig, "vegnow")


def fig_guri() -> Path:
    g = V.guri()
    gy = V.guri_years(V.phases())
    drops = gy.loc[gy.phase == "nino", "drop"].dropna()
    peak = float(g[f"{V.NOW}-08-01":].max())
    lo, hi = peak - drops.max(), peak - drops.min()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    s = g["2008-01-01":]
    ax.plot(s.index, s.values, color="#3b3a37", lw=1.3)
    low10 = float(gy.loc[2009, "low"])
    ax.axhline(low10, color=V.NINO_C, lw=1.2, ls=(0, (4, 3)))
    ax.text(pd.Timestamp("2008-03-01"), low10 - 0.5, _t(f"2010 low, power rationing: {low10:.0f} m",   # ends before the 2016 dip
                                                        f"mínimo de 2010: {low10:.0f} m\n(racionamiento eléctrico)"),
            color=V.NINO_C, fontsize=12, va="top")
    x = pd.Timestamp(V.NOW + 1, 5, 1)
    ax.plot([x, x], [lo, hi], color=V.NOW_C, lw=6, solid_capstyle="butt")
    ax.text(x - pd.Timedelta(days=60), low10 - 0.9,   # under the 2010 line, where 2018–2026 leave room
            _t(f"May {V.NOW + 1}, if it falls as\nafter past El Niño years:\n{lo:.0f}–{hi:.0f} m",
               f"Mayo de {V.NOW + 1}, si baja como\ntras otros años de El Niño:\n{lo:.0f}–{hi:.0f} m"),
            ha="right", va="top", fontsize=12, fontweight="bold")
    ax.set_ylabel(_t("Guri reservoir level, m (satellite)", "nivel del embalse de Guri, m (satélite)"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=V.GRID, lw=0.6)
    ax.set_xlim(pd.Timestamp("2008-01-01"), pd.Timestamp(V.NOW + 1, 12, 1))
    return _save(fig, "guri")


def fig_forecast() -> Path:
    """October–February only, the months the page's headline is about; same classes and colours as the page."""
    pr, sk, _, _ = V.seas5_rank()
    rainy = V.seas5_rainy()
    st = V.states()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    seen, mod = set(), False
    for ax, code in zip(axes, ("OND", "NDJ", "DJF")):
        cls = {pc: V.seas5_class(pr.loc[pc, code], sk.loc[pc, code], bool(rainy.loc[pc, code])) for pc in V.STATE_ORDER}
        g = st.assign(cls=st.pcode.map(lambda pc: cls[pc][0]), skill=st.pcode.map(lambda pc: cls[pc][1]))
        g.plot(ax=ax, color=g.cls.map(V.SEAS5_FILL), edgecolor="none")
        hatch = (g.skill == "mod") & (g.cls != "none")
        for mask, colour, h in ((g.cls == "low_skill", V.SEAS5_CROSS, "xxx"), (hatch, V.SEAS5_HATCH_WHITE, "////"),
                                ((g.skill == "mod") & (g.cls == "none"), V.SEAS5_HATCH_GREY, "////")):
            if mask.any():
                g[mask].plot(ax=ax, facecolor="none", edgecolor=colour, hatch=h, lw=0)
        st.boundary.plot(ax=ax, color="#ffffff", lw=0.5)
        seen |= set(g.cls)
        mod |= bool(hatch.any())
        m = V.FC_SEASONS[code]
        ax.set_title(V._cap(V._span(m[0], m[-1])), fontsize=16, fontweight="bold")
        _ven_axes(ax)
    labels = {"vsev_dry": _t("Strongly below normal (driest 10% of forecasts)", "Muy por debajo de lo normal (10% más seco de los pronósticos)"),
              "sev_dry": _t("Below normal", "Por debajo de lo normal"), "none": _t("Roughly normal", "Cerca de lo normal"),
              "sev_wet": _t("Above normal", "Por encima de lo normal"), "vsev_wet": _t("Strongly above normal", "Muy por encima de lo normal"),
              "off_season": _t("Outside the rainy season", "Fuera de la temporada de lluvias")}
    handles = [Patch(color=V.SEAS5_FILL[k], label=lab) for k, lab in labels.items() if k in seen]
    if "low_skill" in seen:
        handles.append(Patch(facecolor="#ffffff", edgecolor=V.SEAS5_CROSS, hatch="xxx", label=_t("Low skill, not shown",
                                                                                                 "Destreza baja, no se muestra")))
    if mod:
        handles.append(Patch(facecolor=V.SEAS5_FILL["vsev_dry"], edgecolor="#ffffff", hatch="////",
                             label=_t("Hatched: moderate skill", "Rayado: destreza moderada")))
    fig.legend(handles=handles, loc="lower center", ncols=min(len(handles), 3 if V.LANG == "en" else 2), fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(wspace=0.02, bottom=0.1)
    return _save(fig, "forecast")


def fig_rain2026(regions: list[str], values: list[int]) -> Path:
    """For the artifact decks; the .pptx has a native chart of the same numbers."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(range(len(values)), values, color=V.NINO_C, height=0.62)
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v}%", va="center", fontsize=13)
    ax.axvline(100, color="#3b3a37", lw=1.2)
    ax.text(100, len(values) - 0.45, "  normal", fontsize=12, va="bottom")
    ax.set_yticks(range(len(values)), regions, fontsize=13)
    ax.set_xlim(0, max(112, max(values) + 12))
    ax.set_xlabel(_t("% of the 1991–2020 normal", "% de lo normal de 1991–2020"))
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "rain2026")


# --------------------------------------------------------------------------- content: every number from the page's build
def content(v: dict, rain: pd.Series, lang: str) -> dict:
    """The deck's text in the current language (V.set_lang): numbers come from the page's values, already localised."""
    span, yr = v["now_span"], v["fc_year"]
    r = rain.sort_values()
    return {
        "lang": lang,
        "title": "Venezuela: El Niño 2026/27",
        "subtitle": _t("What past El Niño years brought, how this year has gone so far, and the forecast for the coming months",
                       "Lo que trajeron los años de El Niño anteriores, cómo va este año y el pronóstico para los próximos meses"),
        "byline": _t(f"OCHA Centre for Humanitarian Data · rain to {v['data_end']}, vegetation to {v['veg_now_date']}, "
                     f"SEAS5 issued October {yr}",
                     f"Centro de Datos Humanitarios de OCHA · lluvia hasta {v['data_end']}, vegetación hasta el {v['veg_now_date']}, "
                     f"SEAS5 emitido en octubre de {yr}"),
        "parts": [_t("1 · PAST EL NIÑO YEARS", "1 · AÑOS DE EL NIÑO ANTERIORES"), _t("2 · THIS YEAR SO FAR", "2 · ESTE AÑO HASTA AHORA"),
                  _t("3 · THE FORECAST", "3 · EL PRONÓSTICO")],
        "summary_title": _t("Summary", "Resumen"),
        "summary": [
            ["1", _t("Past El Niño years", "Años de El Niño anteriores"),
             _t(f"A drier dry season in the south ({v['ab_dry_lo']}–{v['ab_dry_hi']}% less rain) and weaker rains in the north "
                f"({v['cde_rain_lo']}–{v['cde_rain_hi']}% less); fewer flood reports; the Guri reservoir fell further.",
                f"Estación seca más seca en el sur ({v['ab_dry_lo']}–{v['ab_dry_hi']}% menos lluvia) y lluvias más débiles en el norte "
                f"({v['cde_rain_lo']}–{v['cde_rain_hi']}% menos); menos registros de inundaciones; el embalse de Guri bajó más.")],
            ["2", _t("This year so far", "Este año hasta ahora"),
             _t(f"{V._cap(span)} rain was {v['now_lo']}–{v['now_hi']}% of normal, among the {v['now_rank_txt']} driest since 1981. "
                f"Cropland is below the warning level in {v['veg_warn']}. Guri is near full.",
                f"La lluvia de {span} fue del {v['now_lo']}–{v['now_hi']}% de lo normal, entre las {v['now_rank_txt']} más bajas "
                f"desde 1981. Los cultivos están por debajo del umbral de alerta en {v['veg_warn']}. El Guri está casi lleno.")],
            ["3", _t("The forecast", "El pronóstico"),
             _t(f"SEAS5 (October {yr}): October–February among the driest 10% of forecasts since 1981, wherever it is the rainy "
                "season. For this season, plan on the forecast.",
                f"SEAS5 (octubre de {yr}): octubre–febrero entre el 10% más seco de los pronósticos desde 1981, allí donde es "
                "temporada de lluvias. Para esta temporada, hay que planificar con el pronóstico.")],
        ],
        "zones": {"title": _t("In past El Niño years, the dry season was drier in the south and the rains weaker in the north",
                              "En los años de El Niño anteriores, la estación seca fue más seca en el sur y las lluvias más débiles "
                              "en el norte"),
                  "big1": f"{v['ab_dry_lo']}–{v['ab_dry_hi']}%",
                  "lab1": _t("less dry-season rain in the south (zones A and B)", "menos lluvia en la estación seca en el sur (zonas A y B)"),
                  "big2": f"{v['cde_rain_lo']}–{v['cde_rain_hi']}%",
                  "lab2": _t("less rainy-season rain in the north (zones C to E)",
                             "menos lluvia en la temporada de lluvias en el norte (zonas C a E)"),
                  "note": _t(f"Medians over the {v['n_nino']} El Niño years since 1981. Zones: INAMEH’s map of rainy-season onset. "
                             "Rain: CHIRPS v3.",
                             f"Medianas de los {v['n_nino']} años de El Niño desde 1981. Zonas: mapa del INAMEH de inicio de la "
                             "temporada de lluvias. Lluvia: CHIRPS v3.")},
        "impacts": {"title": _t("El Niño years also had fewer flood reports, a larger fall in the Guri reservoir and poorer "
                                "dry-season vegetation",
                                "Con El Niño también hubo menos inundaciones registradas, mayor bajada del Guri y peor vegetación en "
                                "la estación seca"),
                    "cards": [[f"{v['di_fewer_lo']}–{v['di_fewer_hi']}%",
                               _t("fewer flood and landslide reports in a typical El Niño year (DesInventar, 1975–2013)",
                                  "menos registros de inundaciones y deslizamientos en un año típico de El Niño (DesInventar, 1975–2013)")],
                              [f"{v['guri_nino']} m",
                               _t(f"fall of the Guri reservoir through the dry season after a typical El Niño year, against "
                                  f"{v['guri_nina']}\u00a0m after La Niña",
                                  f"bajada del embalse de Guri durante la estación seca tras un año típico de El Niño, frente a "
                                  f"{v['guri_nina']}\u00a0m tras La Niña")],
                              [_t("3 poorest", "3 peores"),
                               _t(f"dry seasons for vegetation since 2000 all followed El Niño ({v['veg_worst']})",
                                  f"estaciones secas para la vegetación desde 2000 siguieron todas a un El Niño ({v['veg_worst']})")]],
                    "note": _t("Harvests show no clear link with El Niño (FAOSTAT maize and rice yields).",
                               "Las cosechas no muestran una relación clara con El Niño (rendimientos de maíz y arroz de FAOSTAT).")},
        "rain": {"title": _t(f"{V._cap(span)} 2026 rain was {v['now_lo']}–{v['now_hi']}% of normal, among the {v['now_rank_txt']} "
                             "driest since 1981",
                             f"La lluvia de {span} de 2026 fue del {v['now_lo']}–{v['now_hi']}% de lo normal, entre las "
                             f"{v['now_rank_txt']} más bajas desde 1981"),
                 "regions": [V._reg(x) for x in r.index], "values": [round(float(x)) for x in r.values],
                 "note": _t(f"CHIRPS v3 to {v['data_end']}, % of the 1991–2020 normal. ERA5 gives {v['era5_lo']}–{v['era5_hi']}%. "
                            "October data come out in November.",
                            f"CHIRPS v3 hasta {v['data_end']}, % de lo normal de 1991–2020. ERA5 da {v['era5_lo']}–{v['era5_hi']}%. "
                            "Los datos de octubre salen en noviembre.")},
        "veg": {"title": _t(f"Mid-October: vegetation near normal nationally, cropland below the warning level in {v['veg_warn']}",
                            f"Mediados de octubre: vegetación cerca de lo normal en el país, cultivos bajo el umbral de alerta en "
                            f"{v['veg_warn']}"),
                "bullets": [_t(f"Cropland below normal in {v['veg_now_below']} of {v['veg_now_n']} states",
                               f"Cultivos por debajo de lo normal en {v['veg_now_below']} de {v['veg_now_n']} estados"),
                            _t("FEWS NET (August): localized soil-moisture deficits in rainfed areas, no major crop losses",
                               "FEWS NET (agosto): déficits localizados de humedad del suelo en zonas de secano, sin pérdidas "
                               "importantes de cultivos")],
                "note": _t(f"JRC ASAP, {v['veg_now_date']}: MODIS FPAR z-score over cropland; ASAP’s warning level is −1.",
                           f"JRC ASAP, {v['veg_now_date']}: puntaje z de la FPAR de MODIS sobre cultivos; el umbral de alerta de ASAP "
                           "es −1.")},
        "guri": {"title": _t(f"The Guri reservoir is near full ({v['guri_peak']} m); falls like those after past El Niño years "
                             f"would take it to {v['guri_lo']}–{v['guri_hi']} m by May {v['next_year']}",
                             f"El Guri está casi lleno ({v['guri_peak']} m); bajadas como tras años de El Niño anteriores lo "
                             f"llevarían a {v['guri_lo']}–{v['guri_hi']} m en mayo de {v['next_year']}"),
                 "note": _t(f"The low end is {v['guri_gap10']} m above the 2010 low. How far it falls also depends on releases for "
                            "power. NASA G-REALM satellite altimetry.",
                            f"El extremo inferior queda {v['guri_gap10']} m por encima del mínimo de 2010. Cuánto baje depende "
                            "también del agua que se turbine para generar electricidad. Altimetría satelital NASA G-REALM.")},
        "forecast": {"title": _t(f"SEAS5, October {yr}: one of the driest October–February forecasts since 1981, wherever it is "
                                 "the rainy season",
                                 f"SEAS5, octubre de {yr}: uno de los pronósticos más secos para octubre–febrero desde 1981, allí "
                                 "donde es temporada de lluvias"),
                     "bullets": [_t(f"Every rainy-season state and season ({v['fc_cells']} in all) is among the driest 10% of "
                                    "forecasts since 1981",
                                    f"Todos los estados y trimestres de temporada de lluvias ({v['fc_cells']} en total) están entre el "
                                    "10% más seco de los pronósticos desde 1981"),
                                 _t(f"The forecast has skill for these months (correlation {v['fc_r_lo']}–{v['fc_r_hi']})",
                                    f"El pronóstico tiene destreza para estos meses (correlación {v['fc_r_lo']}–{v['fc_r_hi']})"),
                                 _t("ECMWF issues a new forecast every month", "ECMWF emite un pronóstico nuevo cada mes")],
                     "note": _t(f"ECMWF SEAS5, issued October {yr}, detrended; classes as in the team’s seas5-skill viewer.",
                                f"SEAS5 de ECMWF, emitido en octubre de {yr}, con la tendencia de largo plazo quitada; clases como en el "
                                "visor seas5-skill del equipo.")},
        "calendar": {"title": _t("The forecast months cover the summer planting of maize and rice and the coffee and sugar-cane "
                                 "harvest",
                                 "Los meses del pronóstico abarcan la siembra de verano de maíz y arroz y la cosecha de café y caña "
                                 "de azúcar"),
                     "bullets": [_t("FEWS NET expects Stressed (IPC Phase 2) outcomes for October 2026 to January 2027",
                                    "FEWS NET prevé resultados de Estrés (Fase 2 de la CIF) de octubre de 2026 a enero de 2027"),
                                 _t("It expects localized rainfed losses with limited impact, given irrigation and imports",
                                    "Prevé pérdidas localizadas en secano con impacto limitado, dado el riego y las importaciones"),
                                 _t("It puts peak needs in August–October 2026, driven by the earthquake and the economy",
                                    "Sitúa el pico de necesidades en agosto–octubre de 2026, por el terremoto y la economía")],
                     "note": _t("Calendar redrawn from FEWS NET’s Venezuela page; outlook from FEWS NET’s El Niño briefing, "
                                "August 2026.",
                                "Calendario redibujado de la página de Venezuela de FEWS NET; perspectivas del informe de FEWS NET "
                                "sobre El Niño, agosto de 2026.")},
        "watch": {"title": _t("What to watch next", "Qué vigilar en las próximas semanas"),
                  "bullets": [_t("The November SEAS5 forecast, out in early November",
                                 "El pronóstico SEAS5 de noviembre, a principios de noviembre"),
                              _t("October rain, from CHIRPS and ERA5 in November", "La lluvia de octubre, de CHIRPS y ERA5 en noviembre"),
                              _t("The Guri reservoir level through the dry season",
                                 "El nivel del embalse de Guri durante la estación seca"),
                              _t("Cropland warnings in the north-west and north-east (JRC ASAP, every ten days)",
                                 "Alertas en los cultivos del noroeste y del nororiente (JRC ASAP, cada diez días)")],
                  "link_label": _t("Full analysis, in English and Spanish: ", "Análisis completo, en español y en inglés: "),
                  "link": PAGE_URL[lang]},
        "notes": {
            "cover": _t("Three parts: what past El Niño years brought, how this year has gone so far, and the seasonal forecast "
                        "for the coming months. For this season, plan on the forecast; past years are context.",
                        "Tres partes: lo que trajeron los años de El Niño anteriores, cómo va este año y el pronóstico estacional "
                        "para los próximos meses. Para esta temporada, hay que planificar con el pronóstico; los años anteriores "
                        "son contexto."),
            "summary": _t("One line for each part of the deck: what past El Niño years brought, how this year has gone, and what "
                          "the forecast says. Past years are context; the forecast is what to plan on.",
                          "Una línea por cada parte: lo que trajeron los años de El Niño anteriores, cómo va este año y qué dice el "
                          "pronóstico. Los años anteriores son contexto; el pronóstico es con lo que hay que planificar."),
            "zones": _t("INAMEH’s zones by rainy-season onset run from early April in the far south (A) to late June on the "
                        "northern coast (E). Each zone’s rainy season runs from its onset to November; its dry season from "
                        "December to the onset.",
                        "Las zonas del INAMEH según el inicio de la temporada de lluvias van de inicios de abril en el extremo sur "
                        "(A) a fines de junio en la costa norte (E). La temporada de lluvias de cada zona va de su inicio a "
                        "noviembre; su estación seca, de diciembre al inicio."),
            "impacts": _t("Floods: DesInventar records, each year against the years around it. Guri: NASA G-REALM satellite "
                          "altimetry since 1992. Vegetation: JRC ASAP FPAR, November–April, since 2000; 2023/24, a strong El Niño, "
                          "was not among the poorest.",
                          "Inundaciones: registros de DesInventar, cada año frente a los años vecinos. Guri: altimetría "
                          "satelital NASA G-REALM desde 1992. Vegetación: FPAR de JRC ASAP, noviembre–abril, desde 2000; 2023/24, "
                          "un El Niño fuerte, no estuvo entre las peores."),
            "rain": _t("Each region’s total for the months so far this rain year, as a share of its 1991–2020 normal.",
                       "El total de cada región en los meses transcurridos de este año hidrológico, como porcentaje de su normal "
                       "de 1991–2020."),
            "veg": _t("The full page also maps rangeland. ASAP updates every ten days.",
                      "La página completa también cartografía los pastizales. ASAP se actualiza cada diez días."),
            "guri": _t("Satellite heights above the EGM2008 geoid, not the operator’s gauge datum: compare falls, not levels.",
                       "Alturas satelitales sobre el geoide EGM2008, no la cota de referencia del operador: comparar bajadas, "
                       "no niveles."),
            "forecast": _t("The October forecasts in 1997 and 2015 ranked about as dry; the rain that followed was among the "
                           "driest on record in 1997 and below normal in 2015. February–April leans wet in the west and centre, "
                           "where the forecast has low skill or the months fall outside the rainy season.",
                           "Los pronósticos de octubre de 1997 y 2015 salieron casi igual de secos; la lluvia que siguió estuvo "
                           "entre las más bajas registradas en 1997 y por debajo de lo normal en 2015. Febrero–abril tiende a "
                           "húmedo en el oeste y el centro, donde el pronóstico tiene destreza baja o esos meses quedan fuera de "
                           "la temporada de lluvias."),
            "calendar": _t("Calendar: ", "Calendario: ") + FEWS_VEN,
            "watch": _t("Things that will change this picture over the coming weeks.",
                        "Lo que cambiará este panorama en las próximas semanas."),
        },
    }


def _numbers(c: dict) -> list[str]:
    """Every number in a deck's text, with the decimal comma read as a point (keys and URLs are the same in both)."""
    txt = json.dumps({k: x for k, x in c.items() if k != "lang"}, ensure_ascii=False)
    pat = r"\d+(?:,\d+)?" if c["lang"] == "es" else r"\d+(?:\.\d+)?"
    return sorted(n.replace(",", ".") for n in re.findall(pat, txt))


def check_parity(en: dict, es: dict) -> None:
    if _numbers(en) != _numbers(es):
        a, b = _numbers(en), _numbers(es)
        raise ValueError(f"the English and Spanish decks carry different numbers: only EN {sorted(set(a) - set(b))}, "
                         f"only ES {sorted(set(b) - set(a))} (or different counts)")


# --------------------------------------------------------------------------- PowerPoint
def _fit_layouts(prs, k: float) -> None:
    """python-pptx's default template is 4:3: stretch its master and layout placeholders to the 16:9 width, and set left-aligned
    titles and smaller body text, so slides added in PowerPoint match the deck."""
    from pptx.oxml.ns import qn

    for shp in [*prs.slide_master.placeholders, *(p for lay in prs.slide_layouts for p in lay.placeholders)]:
        if shp._element.spPr.find(qn("a:xfrm")) is not None:   # a layout placeholder without one inherits the master's
            shp.left, shp.width = int(shp.left * k), int(shp.width * k)
    styles = prs.slide_master.element.find(qn("p:txStyles"))
    t = styles.find(qn("p:titleStyle")).find(qn("a:lvl1pPr"))
    t.set("algn", "l")
    t.find(qn("a:defRPr")).set("sz", "2600")
    t.find(qn("a:defRPr")).set("b", "1")
    for lvl, sz in (("a:lvl1pPr", "2000"), ("a:lvl2pPr", "1800"), ("a:lvl3pPr", "1600")):
        styles.find(qn("p:bodyStyle")).find(qn(lvl)).find(qn("a:defRPr")).set("sz", sz)


def _set_language(prs, lang_id) -> None:
    """Proofing language on every run, slides and notes, so PowerPoint spell-checks the Spanish deck as Spanish."""
    for s in prs.slides:
        frames = [sh.text_frame for sh in s.shapes if sh.has_text_frame]
        if s.has_notes_slide:
            frames.append(s.notes_slide.notes_text_frame)
        for tf in frames:
            for para in tf.paragraphs:
                for r in para.runs:
                    r.font.language_id = lang_id


def build_pptx(c: dict, figs: dict[str, Path], path: Path) -> None:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
    from pptx.enum.lang import MSO_LANGUAGE_ID
    from pptx.enum.shapes import MSO_SHAPE, PP_PLACEHOLDER
    from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Emu, Inches, Pt

    rgb = RGBColor.from_string
    prs = Presentation()
    prs.slide_width = Emu(12192000)   # PowerPoint's Widescreen, 13.333 × 7.5 in
    prs._element.find(qn("p:sldSz")).attrib.pop("type", None)
    _fit_layouts(prs, 12192000 / 9144000)
    master = prs.slide_master
    master.background.fill.solid()
    master.background.fill.fore_color.rgb = rgb(PAPER)
    mt = next(p for p in master.placeholders if p.placeholder_format.type == PP_PLACEHOLDER.TITLE)
    mt.left, mt.top, mt.width, mt.height = Inches(0.6), Inches(0.75), Inches(12.1), Inches(1.15)
    mt.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    TITLE, TITLE_ONLY = prs.slide_layouts[0], prs.slide_layouts[5]
    n = c["notes"]

    def text(slide, x, y, w, h, lines, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, name=None):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        if name:
            tb.name = name
        tf = tb.text_frame
        tf.word_wrap = True
        for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = align
            para.space_after = Pt(6)
            r = para.add_run()
            r.text = line
            r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold, rgb(color)
        return tb

    def bullets(slide, x, y, w, h, items, size=18):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tb.name = "Bullets"
        tf = tb.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            pPr = para._p.get_or_add_pPr()   # a real bullet (not a typed one), with a hanging indent
            pPr.set("marL", str(Inches(0.3)))
            pPr.set("indent", str(-Inches(0.3)))
            pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
            para.space_after = Pt(14)
            r = para.add_run()
            r.text = item
            r.font.size, r.font.color.rgb = Pt(size), rgb(INK)
        return tb

    def panel(slide, x, y, w, h, name):
        box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        box.name = name
        box.fill.solid()
        box.fill.fore_color.rgb = rgb(PANEL)
        box.line.fill.background()
        box.shadow.inherit = False
        box._element.find(qn("p:style")).find(qn("a:effectRef")).set("idx", "0")   # no theme shadow either
        return box

    def slide(title, eyebrow=None, note=None, notes=None):
        s = prs.slides.add_slide(TITLE_ONLY)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(PAPER)
        t = s.shapes.title
        t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.75), Inches(12.1), Inches(1.15)
        t.text_frame.vertical_anchor = MSO_ANCHOR.TOP
        t.text_frame.word_wrap = True
        if len(title) > TITLE_MAX:
            raise ValueError(f"slide title over {TITLE_MAX} characters would run to three lines: shorten it ({title!r})")
        t.text = title
        t.text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        for r in t.text_frame.paragraphs[0].runs:
            r.font.color.rgb = rgb(INK)
        if eyebrow:
            text(s, 0.6, 0.35, 8, 0.4, eyebrow, size=14, color=ACCENT, bold=True, name="Part")
        if note:
            text(s, 0.6, 6.85, 12.1, 0.45, note, size=12, color=MUTED, name="Source")
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    PART1, PART2, PART3 = c["parts"]

    # 1 title
    s = prs.slides.add_slide(TITLE)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(DARK)
    for ph, top, height in ((s.shapes.title, 2.1, 1.5), (s.placeholders[1], 3.75, 1.6)):   # all four, or the box collapses
        ph.left, ph.top, ph.width, ph.height = Inches(1.0), Inches(top), Inches(11.3), Inches(height)
    s.shapes.title.text = c["title"]
    s.shapes.title.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    for r in s.shapes.title.text_frame.paragraphs[0].runs:
        r.font.size, r.font.color.rgb = Pt(48), rgb(LIGHT)
    sub = s.placeholders[1]
    sub.text = c["subtitle"]
    sub.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    for r in sub.text_frame.paragraphs[0].runs:
        r.font.size, r.font.color.rgb = Pt(22), rgb("E9DCCD")
    text(s, 1.0, 6.4, 11.3, 0.5, c["byline"], size=14, color="CDB9A4", name="Byline")
    s.notes_slide.notes_text_frame.text = n["cover"]

    # 2 summary
    s = slide(c["summary_title"], notes=n["summary"])
    for i, (num, head, body) in enumerate(c["summary"]):
        y = 2.0 + i * 1.55
        panel(s, 0.6, y, 12.1, 1.35, f"Panel {num}")
        text(s, 0.85, y + 0.2, 0.7, 0.9, num, size=40, color=ACCENT, bold=True, name=f"Number {num}")
        text(s, 1.7, y + 0.15, 3.0, 1.1, head, size=20, bold=True, name=f"Heading {num}")
        text(s, 4.6, y + 0.15, 7.9, 1.15, body, size=17, name=f"Text {num}")

    # 3 zones
    z = c["zones"]
    s = slide(z["title"], PART1, z["note"], n["zones"])
    s.shapes.add_picture(str(figs["zone_map"]), Inches(0.6), Inches(2.0), height=Inches(4.75))
    for i, (big, lab) in enumerate(((z["big1"], z["lab1"]), (z["big2"], z["lab2"]))):
        y = 2.3 + i * 2.15
        text(s, 7.0, y, 5.7, 1.0, big, size=54, color=ACCENT, bold=True, name=f"Figure {i + 1}")
        text(s, 7.0, y + 1.0, 5.7, 0.9, lab, size=20, name=f"Label {i + 1}")

    # 4 impacts
    m = c["impacts"]
    s = slide(m["title"], PART1, m["note"], n["impacts"])
    for i, (big, lab) in enumerate(m["cards"]):
        x = 0.6 + i * 4.1
        panel(s, x, 2.3, 3.85, 3.9, f"Card {i + 1}")
        text(s, x + 0.3, 2.6, 3.3, 1.0, big, size=40, color=ACCENT, bold=True, name=f"Figure {i + 1}")
        text(s, x + 0.3, 3.75, 3.3, 2.3, lab, size=18, name=f"Label {i + 1}")

    # 5 rain 2026: a native chart
    r = c["rain"]
    s = slide(r["title"], PART2, r["note"], n["rain"])
    cd = CategoryChartData()
    cd.categories = r["regions"]
    cd.add_series(_t("% of normal", "% de lo normal"), r["values"])
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(2.0), Inches(12.1), Inches(4.75), cd).chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.size = Pt(14)
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.has_data_labels = True
    pl.data_labels.number_format, pl.data_labels.number_format_is_linked = '0"%"', False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.series[0].format.fill.solid()
    pl.series[0].format.fill.fore_color.rgb = rgb(ACCENT)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0, max(100, -(-max(r["values"]) // 20) * 20), 20
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("DCD8D2")
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = '0"%"', False
    va.format.line.fill.background()
    ch.category_axis.format.line.color.rgb = rgb("9A9993")

    # 6 vegetation now
    g = c["veg"]
    s = slide(g["title"], PART2, g["note"], n["veg"])
    s.shapes.add_picture(str(figs["vegnow"]), Inches(0.9), Inches(2.0), height=Inches(4.75))
    bullets(s, 7.0, 2.5, 5.7, 3.5, g["bullets"], size=20)

    # 7 Guri
    g = c["guri"]
    s = slide(g["title"], PART2, g["note"], n["guri"])
    s.shapes.add_picture(str(figs["guri"]), Inches(1.6), Inches(2.05), height=Inches(4.6))

    # 8 forecast
    f = c["forecast"]
    s = slide(f["title"], PART3, f["note"], n["forecast"])
    s.shapes.add_picture(str(figs["forecast"]), Inches(0.5), Inches(2.3), width=Inches(8.5))
    bullets(s, 9.3, 2.2, 3.45, 4.4, f["bullets"], size=17)

    # 9 calendar
    k = c["calendar"]
    s = slide(k["title"], PART3, k["note"], n["calendar"])
    s.shapes.add_picture(str(figs["calendar"]), Inches(0.45), Inches(2.1), width=Inches(8.9))
    bullets(s, 9.55, 2.2, 3.25, 4.5, k["bullets"], size=16)

    # 10 what to watch
    w = c["watch"]
    s = slide(w["title"], notes=n["watch"])
    bullets(s, 0.6, 2.1, 12.1, 3.8, w["bullets"], size=24)
    tb = text(s, 0.6, 6.2, 12.1, 0.5, w["link_label"], size=16, color=MUTED, name="Link")
    run = tb.text_frame.paragraphs[0].add_run()
    run.text = w["link"]
    run.hyperlink.address = w["link"]
    run.font.size = Pt(16)

    if c["lang"] == "es":
        _set_language(prs, MSO_LANGUAGE_ID.SPANISH_VENEZUELA)
        for part in (prs.slide_master, *prs.slide_layouts):   # defaults for text typed into new boxes and slides
            for el in part._element.iter(qn("a:defRPr"), qn("a:endParaRPr"), qn("a:rPr")):
                el.set("lang", "es-VE")
    prs.core_properties.title = c["title"]
    prs.core_properties.author = _t("OCHA Centre for Humanitarian Data", "Centro de Datos Humanitarios de OCHA")
    prs.core_properties.language = _t("en-GB", "es-VE")
    prs.save(path)


def render_pngs(pptx: Path) -> list[Path]:
    """LibreOffice (headless) → PDF → one PNG per slide, for checking the deck and for the slides page. Skipped, with a message,
    where LibreOffice or pdftoppm is missing. (Scripting PowerPoint or Keynote fails here: their sandbox refuses the file.)"""
    soffice = shutil.which("soffice") or "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    pdf = pptx.with_suffix(".pdf")
    try:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(pptx.parent), str(pptx)], check=True,
                       capture_output=True, timeout=300)
        for old in pptx.parent.glob("slide-*.png"):
            old.unlink()
        subprocess.run(["pdftoppm", "-png", "-r", "110", str(pdf), str(pptx.parent / "slide")], check=True, timeout=180)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"no slide images ({e}); check the deck by opening it")
        return []
    return sorted(pptx.parent.glob("slide-*.png"))


PAGE = """<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — {slides}</title><meta name="description" content="{desc}">
<style>
:root{{--ink:#1f2328;--muted:#5b6168;--accent:#b5541b;--paper:#f4f2ee;--line:#dedad3}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:20px 16px 48px}}
a{{color:#1f5fa8}}
.top{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:0 0 18px}}
.btn{{display:inline-block;padding:8px 14px;border:1px solid var(--line);border-radius:6px;background:#fff;text-decoration:none;font-size:14px}}
.btn.primary{{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}}
h1{{font-size:26px;line-height:1.25;margin:0 0 6px}}
p.meta{{color:var(--muted);margin:0 0 18px;font-size:14px}}
figure{{margin:0 0 18px;background:#fff;border:1px solid var(--line);border-radius:6px;overflow:hidden}}
figure img{{display:block;width:100%;height:auto}}
</style></head><body><main>
<div class="top"><a class="btn" href="{full}">{back}</a><a class="btn primary" href="{pptx}" download>{download}</a>
<a class="btn" href="{other}" lang="{other_lang}">{other_label}</a></div>
<h1>{title} — {slides}</h1>
<p class="meta">{meta}</p>
{figs}
</main></body></html>
"""
PAGE_TEXT = {
    "en": dict(slides="slides", full="../", back="← Full analysis", download="Download the PowerPoint (editable)", other="es/",
               other_lang="es", other_label="Versión en español",
               meta="{n} slides, the short version of <a href='../'>the full analysis</a>. The PowerPoint has editable text and a "
                    "native chart; sources are on each slide and in the speaker notes.", alt="Slide"),
    "es": dict(slides="diapositivas", full="../../es/", back="← Análisis completo", download="Descargar el PowerPoint (editable)",
               other="../", other_lang="en", other_label="English version",
               meta="{n} diapositivas, la versión corta del <a href='../../es/'>análisis completo</a>. El PowerPoint tiene texto "
                    "editable y un gráfico nativo; las fuentes están en cada diapositiva y en las notas del orador.",
               alt="Diapositiva"),
}


def write_page(c: dict, pngs: list[Path], dest: Path) -> None:
    lang = c["lang"]
    t = PAGE_TEXT[lang]
    figs = "\n".join(f'<figure><img src="{p.name}" alt="{t["alt"]} {i}" loading="lazy"></figure>' for i, p in enumerate(pngs, 1))
    dest.write_text(PAGE.format(lang=lang, title=html.escape(c["title"]), desc=html.escape(c["subtitle"]), pptx=PPTX[lang], figs=figs,
                                **{**t, "meta": t["meta"].format(n=len(pngs))}), encoding="utf-8")


def main() -> None:
    global _DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true",
                    help="also copy the decks, their slide images and pages to docs/enso/ven/slides/ and slides/es/")
    args = ap.parse_args()
    a = V.analyse()
    last = a["mon"].dropna(how="all").index.max()
    rain = V._region_pct(a["mon"], list(range(5, last.month + 1))).loc[V.NOW]
    decks, pngs = {}, {}
    try:
        for lang in ("en", "es"):
            V.set_lang(lang)
            _DIR = _out(lang)
            _DIR.mkdir(parents=True, exist_ok=True)
            _, v = V.figures(a)
            c = content(v, rain, lang)
            figs = dict(zone_map=fig_zone_map(), vegnow=fig_vegnow(), guri=fig_guri(), forecast=fig_forecast(),
                        rain2026=fig_rain2026(c["rain"]["regions"], c["rain"]["values"]),
                        calendar=V.fig_calendar(k=1.35))   # the page's figure, in larger type
            (_DIR / "content.json").write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
            build_pptx(c, figs, _DIR / PPTX[lang])
            pngs[lang] = render_pngs(_DIR / PPTX[lang])
            write_page(c, pngs[lang], _DIR / "index.html")
            decks[lang] = c
            print(f"wrote {_DIR / PPTX[lang]}, {len(pngs[lang])} slide images and {_DIR / 'index.html'}")
    finally:
        V.set_lang("en")
    check_parity(decks["en"], decks["es"])
    if args.publish:
        if not all(pngs.values()):
            raise SystemExit("no slide images: the slides pages would be empty")
        if DOCS.exists():
            shutil.rmtree(DOCS)
        for lang in ("en", "es"):
            d, src = (DOCS / "es" if lang == "es" else DOCS), _out(lang)
            d.mkdir(parents=True)
            for p in [src / PPTX[lang], src / "index.html", *pngs[lang]]:
                shutil.copy(p, d / p.name)
        print(f"copied both decks and their pages to {DOCS}")


if __name__ == "__main__":
    main()
