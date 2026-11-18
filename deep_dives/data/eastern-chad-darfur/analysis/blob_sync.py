"""The work folder (ECD_WORK) on the team's dev blob: projects/ds-teleconnections/eastern-chad-darfur/.

The repo holds code only. Everything the analysis downloads or writes - raw downloads, processed
tables, both pages' data (processed/page_data.json; the restricted processed/conflict/) - lives in
the work folder, mirrored on blob with the same relative paths.

    uv run python blob_sync.py push              # upload files that are new or changed (by size)
    uv run python blob_sync.py pull [prefix]     # download what is missing locally, e.g. `pull processed`

Builders call fetch("processed/page_data.json"), which downloads one file if it is not local.
Needs DSCI_AZ_BLOB_DEV_SAS (read) / DSCI_AZ_BLOB_DEV_SAS_WRITE (push), as ocha-stratus reads them.
"""

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]  # ds-teleconnections
DATA = Path(os.environ.get("ECD_WORK", REPO / "cache" / "eastern-chad-darfur"))
BLOB_PREFIX = "ds-teleconnections/eastern-chad-darfur"
# not mirrored: version-control folders inside downloaded repos, caches, logs, the UCDP csv
# (the analysis reads the zip it came in)
SKIP = re.compile(r"(^|/)(\.git|__pycache__)(/|$)|\.log$|\.DS_Store$|^raw/ucdp/GEDEvent_[^/]*\.csv$")


def _container(write=False):
    import ocha_stratus as stratus
    return stratus.get_container_client(container_name="projects", stage="dev", write=write)


def _remote(cc, prefix=""):
    root = f"{BLOB_PREFIX}/{prefix}".rstrip("/")
    return {b.name[len(BLOB_PREFIX) + 1:]: b.size for b in cc.list_blobs(name_starts_with=root)}


def fetch(rel: str) -> Path:
    """Local path of a work-folder file, downloaded from blob first if it is missing."""
    p = DATA / rel
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        data = _container().get_blob_client(f"{BLOB_PREFIX}/{rel}").download_blob().readall()
        p.write_bytes(data)
        print(f"  fetched {rel} from blob")
    return p


def push():
    cc = _container(write=True)
    have = _remote(cc)
    todo = [p for p in DATA.rglob("*") if p.is_file()
            and not SKIP.search(r := p.relative_to(DATA).as_posix()) and have.get(r) != p.stat().st_size]
    print(f"[blob] {len(todo)} files to upload to projects/{BLOB_PREFIX}/")

    def up(p):
        with open(p, "rb") as f:
            cc.upload_blob(f"{BLOB_PREFIX}/{p.relative_to(DATA).as_posix()}", f, overwrite=True)

    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(up, todo), 1):
            if i % 200 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)


def pull(prefix=""):
    cc = _container()
    todo = [r for r, size in _remote(cc, prefix).items()
            if not (DATA / r).exists() or (DATA / r).stat().st_size != size]
    print(f"[blob] {len(todo)} files to download into {DATA}")

    def down(r):
        (DATA / r).parent.mkdir(parents=True, exist_ok=True)
        (DATA / r).write_bytes(cc.get_blob_client(f"{BLOB_PREFIX}/{r}").download_blob().readall())

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(down, todo))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "push":
        push()
    elif cmd == "pull":
        pull(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        raise SystemExit(__doc__)
