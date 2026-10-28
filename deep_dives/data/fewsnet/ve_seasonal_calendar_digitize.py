"""Digitize FEWS NET's seasonal calendar for rural Venezuela into a table, so ven_deep_dive.py can redraw it in English and
Spanish.

Input: ve_seasonal_calendar_rural.png (the rural panel of FEWS NET's calendar, https://fews.net/latin-america-and-caribbean/venezuela).
Output: ve_seasonal_calendar_rural.csv. Run from the repo root:

    uv run python deep_dives/data/fewsnet/ve_seasonal_calendar_digitize.py

How: the month axis comes from a straight-line fit to the twelve month-label centres; in each row, a pixel column is a bar
where any pixel within ±10 px of the row's centre is within 30 (RGB distance) of a bar colour; the red lean-season outline is
bridged where the same bar runs on both sides; runs are rounded to the nearest half month (raw ends land within ±0.1 month of
one; the check allows 0.12); each run's colour (pixels within 14 of a bar colour) must match the kind given in ROWS. The text
in each bar and the row names are read off the image (ROWS), with Spanish translations; Winter/Summer are Venezuela's own crop
cycles (ciclo de invierno, planted with the rains; ciclo de verano or norte-verano, from about October), so they become
Invierno/Verano. Months are counted from 0 = 1 January to 12 = 31 December. Only the coffee and sugar-cane harvest crosses the
year end (October–March: FEWS NET cuts both of its bars square at the chart edge); the other bars that reach December have
rounded ends and stop there. `wraps` marks the two halves of that harvest.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
COLOURS = {"hurricane": (29, 58, 95), "prep": (79, 53, 46), "harvest": (13, 93, 64), "labour": (67, 67, 67)}
MONTH_CENTRES = [541.5, 652.0, 762.0, 870.5, 982.0, 1088.5, 1198.5, 1307.0, 1417.0, 1526.0, 1635.5, 1744.0]   # px, label centres
X_START = 489   # left edge of the month grid (the row-name column ends here)
# row centre (px): (English row, Spanish row, [(kind, English text, Spanish text) per bar, left to right])
ROWS = {
    222: ("Hurricane season", "Temporada de huracanes", [("hurricane", "", "")]),
    279: ("Land preparation: maize", "Preparación de tierras: maíz", [("prep", "Winter", "Invierno"), ("prep", "Summer", "Verano")]),
    337: ("Planting: maize", "Siembra: maíz", [("prep", "Winter", "Invierno"), ("prep", "Summer", "Verano")]),
    394: ("Maize harvest", "Cosecha de maíz", [("harvest", "Summer", "Verano"), ("harvest", "Winter", "Invierno")]),
    452: ("Land preparation: rice", "Preparación de tierras: arroz", [("prep", "Winter", "Invierno"), ("prep", "Summer", "Verano")]),
    510: ("Planting: rice", "Siembra: arroz", [("prep", "Winter", "Invierno"), ("prep", "Summer", "Verano")]),
    567: ("Rice harvest", "Cosecha de arroz", [("harvest", "Summer", "Verano"), ("harvest", "Winter", "Invierno")]),
    627: ("Potatoes", "Papa", [("harvest", "Harvest", "Cosecha"), ("prep", "Land prep", "Preparación"),
                              ("prep", "Planting", "Siembra")]),
    684: ("Vegetable harvest", "Cosecha de hortalizas", [("harvest", "Summer", "Verano"), ("harvest", "Winter", "Invierno")]),
    741: ("Cash crop harvest", "Cosecha de cultivos comerciales", [("harvest", "Coffee and sugar cane", "Café y caña de azúcar"),
                                                                  ("harvest", "Coffee and sugar cane", "Café y caña de azúcar")]),
    798: ("Peak demand: agricultural labour", "Pico de demanda: mano de obra agrícola", [("labour", "", ""), ("labour", "", "")]),
}


def runs(im: np.ndarray, y: int) -> list[tuple[float, float]]:
    win = im[y - 10:y + 11, X_START:]
    bar = np.zeros(win.shape[1], bool)
    for c in COLOURS.values():
        bar |= (np.sqrt(((win - np.array(c)) ** 2).sum(axis=2)) < 30).any(axis=0)
    red = ((win[:, :, 0] > 200) & (win[:, :, 1] < 100)).any(axis=0)
    idx = np.where(red & ~bar)[0]
    for i in idx:   # bridge the red outline only inside a bar
        left, right = bar[:i][~red[:i]], bar[i:][~red[i:]]
        bar[i] = len(left) > 0 and len(right) > 0 and left[-1] and right[0]
    edges = np.flatnonzero(np.diff(np.r_[0, bar.astype(int), 0]))
    out = [(s, e) for s, e in zip(edges[::2], edges[1::2]) if e - s > 8]
    # adjacent bars (potatoes) are split by a light border, already a gap
    return [(X_START + s, X_START + e) for s, e in out]


def kind_of(im: np.ndarray, y: int, x0: int, x1: int) -> str:
    """The bar colour most of a run's pixels are close to (14 tells prep from labour, which are 28 apart)."""
    px = im[y - 6:y + 7, x0:x1].reshape(-1, 3)
    d = np.stack([np.sqrt(((px - np.array(c)) ** 2).sum(axis=1)) for c in COLOURS.values()])
    near = d.min(axis=0) < 14
    votes = np.bincount(d.argmin(axis=0)[near], minlength=len(COLOURS))
    return list(COLOURS)[int(votes.argmax())]


def main() -> None:
    im = np.asarray(Image.open(HERE / "ve_seasonal_calendar_rural.png").convert("RGB")).astype(int)
    width, centre0 = np.polyfit(range(12), MONTH_CENTRES, 1)
    x0 = centre0 - width / 2
    month = lambda x: (x - x0) / width   # noqa: E731
    rows = []
    for order, (y, (row_en, row_es, bars)) in enumerate(ROWS.items()):
        found = runs(im, y)
        if len(found) != len(bars):
            raise ValueError(f"row {row_en}: found {len(found)} bars, expected {len(bars)}")
        for (s, e), (kind, en, es) in zip(found, bars):
            if kind_of(im, y, s, e) != kind:
                raise ValueError(f"row {row_en}: a bar's colour is {kind_of(im, y, s, e)}, not {kind}")
            m0, m1 = month(s), month(e)
            r0, r1 = round(m0 * 2) / 2, round(m1 * 2) / 2
            if max(abs(m0 - r0), abs(m1 - r1)) > 0.12:
                raise ValueError(f"row {row_en}: bar {m0:.2f}–{m1:.2f} is not on a half month")
            rows.append(dict(order=order, row_en=row_en, row_es=row_es, kind=kind, start=r0, end=r1, text_en=en, text_es=es,
                             wraps=row_en == "Cash crop harvest"))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "ve_seasonal_calendar_rural.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
