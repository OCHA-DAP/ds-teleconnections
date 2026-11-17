"""Dev DB engine. From a laptop the DB is reached only through the Databricks SSH tunnel
(`~/bin/db-tunnel up`, dev on 127.0.0.1:15432). The shell profile usually sets
DSCI_AZ_DB_DEV_HOST to the server name, which has no route from here, so it is replaced
with the tunnel address (override with DB_TUNNEL_DEV). stratus reads the host at import."""

import os

os.environ["DSCI_AZ_DB_DEV_HOST"] = os.environ.get("DB_TUNNEL_DEV", "127.0.0.1:15432")

import ocha_stratus as stratus  # noqa: E402


def engine():
    return stratus.get_engine("dev")
