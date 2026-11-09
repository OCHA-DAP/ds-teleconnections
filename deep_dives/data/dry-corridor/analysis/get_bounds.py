import sys, ocha_stratus as stratus
out = sys.argv[1]
for iso in ("GTM", "HND", "SLV"):
    for lvl in (0, 1):
        g = stratus.codab.load_codab_from_blob(iso, lvl, stage="prod")
        print(iso, lvl, g.crs, g.shape, list(g.columns)[:12], g.total_bounds.round(2))
        g.to_file(f"{out}/{iso.lower()}_adm{lvl}.gpkg")
