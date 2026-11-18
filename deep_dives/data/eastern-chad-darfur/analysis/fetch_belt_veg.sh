#!/usr/bin/env bash
# Admin-1 vegetation series for the pooled belt test: FAO GIEWS ASIS (ASI cropland dekads, VHI dekads) and
# JRC ASAP cumulated FPAR z-scores (cropland, rangeland). Same endpoints as fetch_veg.sh.
# Usage: ./fetch_belt_veg.sh <work_dir>   (one ASAP request at a time; its server is slow)
set -u; D=${1:-data}; mkdir -p "$D/raw/belt/asis" "$D/raw/belt/asap"
B="https://www.fao.org/giews/earthobservation/asis/data/country"
for iso in MRT SEN GMB MLI BFA NER NGA CMR CAF SSD TCD SDN; do
  for f in MAP_ASI/DATA/ASI_Dekad_Season1_data.csv MAP_NDVI_ANOMALY/DATA/vhi_adm1_dekad_data.csv; do
    o="$D/raw/belt/asis/${iso}_$(basename "$f")"
    [ -s "$o" ] && continue
    code=$(curl -s -L -A "Mozilla/5.0" --max-time 300 -o "$o" -w "%{http_code}" "$B/$iso/$f"); [ "$code" = 200 ] || rm -f "$o"; echo "asis $iso $(basename "$f") $code"
  done
done
A="https://agricultural-production-hotspots.ec.europa.eu/export/rum/export.php"
# asap0_id from gaul1_asap_v06 (name0 -> asap0_id)
for ic in MRT:129 SEN:166 GMB:74 MLI:123 BFA:17 NER:137 NGA:138 CMR:37 CAF:31 SSD:176 TCD:184 SDN:165; do iso=${ic%%:*}; cid=${ic##*:}
  for spec in "zfparc_crop 240 1 1 3" "zfparc_range 240 2 1 3"; do
    read -r name vid cls csid sid <<< "$spec"; f="$D/raw/belt/asap/${iso}_$name.csv"
    [ -s "$f" ] || curl -s -A "Mozilla/5.0 (OCHA-CHD data science)" --max-time 1500 "$A?gaul_level=1&country_id=$cid&variable_id=$vid&class_id=$cls&classesset_id=$csid&sensor_id=$sid" -o "$f"
    echo "asap $iso $name: $(wc -l < "$f") rows, country=$(sed -n '2p' "$f" | cut -d, -f1-2)"; sleep 3
  done
done
