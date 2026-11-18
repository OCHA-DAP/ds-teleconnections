"""Eastern Chad and Darfur: poor seasons and local violence (builder = "ecdc"). RESTRICTED page.

Built from ACLED events, whose terms limit redistribution, so nothing derived from them is committed: the
data (conflict_data.json) and the curated text (text.json: literature, forecasting models) live in the work
folder, <ECD_WORK>/processed/conflict/ (mirrored on the dev blob; see analysis/blob_sync.py), written by
deep_dives/data/eastern-chad-darfur/analysis/10-12_*.py; missing files are fetched from blob.
Only this code, the generic template (deep_dives/data/eastern-chad-darfur/conflict_template.html, which
words everything from the data in the browser) and the staticrypt-encrypted HTML are in the repo. Commit
messages must not carry results either.

    uv run python ecd_conflict.py                                    # build out/eastern-chad-darfur-conflict/ to review
    SITE_PASSWORD=... uv run python ecd_conflict.py --encrypt
                                       # + docs/enso/eastern-chad-darfur-conflict/ (encrypted) and the /enso/ card
In a full `enso_deep_dive.py` run the page is built and encrypted only with SITE_PASSWORD and ECDC_PUBLISH=1.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLUG = "eastern-chad-darfur-conflict"
SPEC = ROOT / "deep_dives" / f"{SLUG}.toml"
TPL = ROOT / "deep_dives" / "data" / "eastern-chad-darfur" / "conflict_template.html"
ANALYSIS = ROOT / "deep_dives" / "data" / "eastern-chad-darfur" / "analysis"


def work(rel: str) -> Path:
    """A work-folder file (ECD_WORK), fetched from the dev blob if it is not local."""
    import sys
    sys.path.insert(0, str(ANALYSIS))
    from blob_sync import fetch
    return fetch(rel)
OUT = ROOT / "out" / SLUG
DOCS = ROOT / "docs" / "enso" / SLUG


def _check(D: dict) -> None:
    """Sentences hard-written in the template that the data must still support."""
    bad = []
    yr = [r for r in D["yearly"] if r["side"] == "chad" and 2014 <= r["year"] <= 2022]
    if max(max(r["resource"], r["communal"]) for r in yr) > 15:
        bad.append("the eastern Chad notes in text.json (a year has > 15 events)")
    test = [r for r in D["season"] if r["kind"] == "test"]
    g26 = next(r for r in D["season"] if r["y"] == 2026)
    if not (g26["grn"] < min(r["grn"] for r in test) and g26["grn_pastoral"] < min(r["grn_pastoral"] for r in test)):
        bad.append("'2026 poorer than any season in the test' (Darfur or pastoral-zone greening)")
    if max(r["m"] for r in D["cast"]) >= "2026-10":
        bad.append("'no CAST forecast covers the months after this season'")
    for k in ("lit", "models", "notes", "handcheck", "sources"):
        if not D["text"].get(k):
            bad.append(f"text.json has no '{k}'")
    if bad:
        raise SystemExit("ecdc: the data no longer support:\n  " + "\n  ".join(bad))


def render(spec: dict) -> str:
    D = json.loads(work("processed/conflict/conflict_data.json").read_text(encoding="utf-8"))
    D["text"] = json.loads(work("processed/conflict/text.json").read_text(encoding="utf-8"))
    P = json.loads(work("processed/page_data.json").read_text(encoding="utf-8"))  # geometry + season series (public)
    D["geo"], D["adm1"], D["sides"] = P["geo"], P["adm1"], P["sides"]
    _check(D)
    tpl = TPL.read_text(encoding="utf-8")
    for token in ("/*__DATA__*/null", '"/*__STAMP__*/"'):
        if tpl.count(token) != 1:
            raise SystemExit(f"ecdc: template must hold {token} exactly once")
    stamp = spec.get("page", {}).get("stamp") or date.today().strftime("%-d %B %Y")
    return (tpl.replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":"), ensure_ascii=False))
               .replace('"/*__STAMP__*/"', json.dumps(stamp)))


def write_index() -> None:
    """Re-render the /enso/ card index (a restricted card appears once its encrypted page exists)."""
    import enso_deep_dive as edd
    specs = [tomllib.loads(p.read_text(encoding="utf-8")) | {"slug": p.stem}
             for p in sorted((ROOT / "deep_dives").glob("*.toml"))]
    (ROOT / "docs" / "enso" / "index.html").write_text(edd.render_index(specs, 0), encoding="utf-8")


def encrypt() -> None:
    """staticrypt the page into docs/enso/eastern-chad-darfur-conflict/ and check no plaintext survives."""
    pw = os.environ.get("SITE_PASSWORD")
    if not pw:
        raise SystemExit("set SITE_PASSWORD to encrypt")
    with tempfile.TemporaryDirectory() as tmp:  # encrypt aside; only a page that passes the checks reaches docs/
        subprocess.run(["npx", "-y", "staticrypt@3.5.4", str(OUT / "index.html"), "-d", tmp, "--short",
                        "--remember", "30", "--template-title", "Eastern Chad and Darfur: restricted",
                        "--template-instructions", "OCHA Centre for Humanitarian Data. Password shared internally."],
                       check=True, env=dict(os.environ, STATICRYPT_PASSWORD=pw), stdout=subprocess.DEVNULL)
        enc = (Path(tmp) / "index.html").read_text(encoding="utf-8")
        if "staticrypt" not in enc:
            raise SystemExit("the encrypted page is not a staticrypt page; nothing written")
        # strings that occur in the plaintext page (checked) and must not survive encryption
        for marker in ("const D =", "greening", "ACLED", "adm2_pcode", "resource-type", "McGuirk", "share_res"):
            if marker in enc:
                raise SystemExit(f"the encrypted page contains plaintext ({marker}); nothing written")
        enc = enc.replace("<head>", '<head>\n<meta name="robots" content="noindex">', 1)
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "index.html").write_text(enc, encoding="utf-8")
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
    ap.add_argument("--encrypt", action="store_true",
                    help="also staticrypt into docs/enso/eastern-chad-darfur-conflict/ (needs SITE_PASSWORD)")
    build(encrypt_page=ap.parse_args().encrypt)


if __name__ == "__main__":
    main()
