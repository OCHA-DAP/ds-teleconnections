#!/usr/bin/env bash
# Raw vegetation pulls: FAO GIEWS ASIS country CSVs and JRC ASAP admin-1 indicator series.
# Usage: scripts/fetch_veg.sh <data_dir>   (one request at a time; ASAP's server is slow)
set -u; D=${1:-data}; mkdir -p "$D/raw/asis" "$D/raw/asap"
B="https://www.fao.org/giews/earthobservation/asis/data/country"
for iso in TCD SDN; do
  for f in MAP_ASI/DATA/ASI_AnnualSummary_Season1_data.csv MAP_ASI/DATA/Mean-VHI_AnnualSummary_Season1_data.csv MAP_ASI/DATA/ASI_Dekad_Season1_data.csv MAP_NDVI_ANOMALY/DATA/vhi_adm1_dekad_data.csv; do
    o="$D/raw/asis/${iso}_$(basename "$f")"
    code=$(curl -s -L -A "Mozilla/5.0" --max-time 300 -o "$o" -w "%{http_code}" "$B/$iso/$f"); [ "$code" = 200 ] || rm -f "$o"; echo "asis $iso $(basename "$f") $code"
  done
done
A="https://agricultural-production-hotspots.ec.europa.eu/export/rum/export.php"
# ASAP country_id: Chad 184, Sudan 165 (asap0_id from getDataDownload.php; check the country column)
for ic in TCD:184 SDN:165; do iso=${ic%%:*}; cid=${ic##*:}
  for spec in "zfparc_crop 240 1 1 3" "zfparc_range 240 2 1 3" "temp_crop 140 1 1 4" "spi3_crop 40 1 1 4"; do
    read -r name vid cls csid sid <<< "$spec"; f="$D/raw/asap/${iso}_$name.csv"
    [ -s "$f" ] || curl -s -A "Mozilla/5.0 (OCHA-CHD data science)" --max-time 1500 "$A?gaul_level=1&country_id=$cid&variable_id=$vid&class_id=$cls&classesset_id=$csid&sensor_id=$sid" -o "$f"
    echo "asap $iso $name: $(wc -l < "$f") rows, country=$(sed -n '2p' "$f" | cut -d, -f1-3)"; sleep 3
  done
done
