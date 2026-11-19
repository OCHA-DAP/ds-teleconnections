#!/usr/bin/env bash
# Daily CHIRPS v2.0 for February–July 1981–2026 over Haiti, from the IRI Data Library, in three parallel year ranges.
# Usage: bash dl_all.sh <work dir SP>    (writes SP/chirps_fj/chirps_hti_fj_<year>.nc and SP/dl{1,2,3}.log)
set -euo pipefail
SP="$1"; HERE="$(cd "$(dirname "$0")" && pwd)"
uv run python -I "$HERE/dl_chirps_fj.py" "$SP/chirps_fj" 1981 1996 > "$SP/dl1.log" 2>&1 &
uv run python -I "$HERE/dl_chirps_fj.py" "$SP/chirps_fj" 1997 2011 > "$SP/dl2.log" 2>&1 &
uv run python -I "$HERE/dl_chirps_fj.py" "$SP/chirps_fj" 2012 2026 > "$SP/dl3.log" 2>&1 &
wait
