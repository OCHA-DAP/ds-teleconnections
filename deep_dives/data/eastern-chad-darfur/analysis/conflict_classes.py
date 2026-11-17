"""Event classes for the conflict page (ACLED, eastern Chad + Darfur). Definitions only, no results.

Every class is built from who fought and what the event notes say was fought over - never
from words like "drought" or "scarcity" (sources attribute violence to drought more often
in dry years, which would build the answer in).

  war        a party to the wider wars is involved: state forces (Sudan, Chad, other states),
             rebel groups (incl. the RSF since 2023, and the RSF while a state force), and
             foreign/peacekeeping forces. Formal parties only: ACLED's militia labels for the
             same kind of attack can change over time, and a filter on them would put coding
             changes into the series.
  war_broad  also pro-government militias (Janjaweed, "Militia (Pro-Government)", PDF,
             Musa Hilal's militia): sensitivity check, since these labels can cover both
             counterinsurgency and attacks on farmers' land.
  resource   not war, and ACLED tags Farmers / Pastoralists / Herders as an actor, or the
             notes mention livestock, grazing, crops, farms or water points.   (headline)
  communal   not war, and an identity (communal/ethnic) militia is a party.  (second line;
             exposed to label changes)
"""

import re

import pandas as pd

A1 = {"Ouaddai": "chad", "Wadi Fira": "chad", "Sila": "chad", "North Darfur": "darfur",
      "South Darfur": "darfur", "West Darfur": "darfur", "Central Darfur": "darfur",
      "East Darfur": "darfur"}
EVENT_TYPES = ["Battles", "Violence against civilians"]
WAR_INTER = {"State forces", "Rebel group", "External/Other forces"}
WAR_NAME = re.compile(r"Military Forces|Police Forces|Government of|Rapid Support|UNAMID|AMIS:|UNMIS|"
                      r"MINURCAT|EUFOR|Wagner")
PROGOV = re.compile(r"Janjaweed|Militia \(Pro-Government\)|PDF: Popular Defence|Militia \(Musa Hilal\)")
FARM_HERD = re.compile(r"\b(Pastoralists|Farmers|Herders)\b")
KW = re.compile(r"\b(?:cattle|cows?|camels?|livestock|goats?|sheep|herds?|herding|herders?|herdsm[ae]n|"
                r"pastoral\w*|nomads?|nomadic|graz\w*|pastures?|farms?|farmers?|farmland|farming|crops?|"
                r"harvest\w*|cultivat\w*|(?:farm|agricultural) ?fields?|water ?points?|(?:water )?wells|"
                r"boreholes?|water sources?|grain|sorghum|millet|groundnuts?)\b", re.I)
# camels and horses as the attackers' transport ("riding camels", "on camelback") are not livestock
TRANSPORT = re.compile(r"\b(?:riding|rode|on|mounted on|aboard)\b[^.;,]{0,25}?\bcamel\w*(?:\s*(?:and|/|or)\s*horse\w*)?"
                       r"|\bcamel\s*/\s*horse\w*|\bhorse\w* and camels?", re.I)


def classify(ev: pd.DataFrame) -> pd.DataFrame:
    """ACLED rows -> rows in the study area (Battles + Violence against civilians) with class flags."""
    a = ev[ev["admin1"].isin(A1) & ev["event_type"].isin(EVENT_TYPES)].copy()
    a["side"] = a["admin1"].map(A1)
    a["date"] = pd.to_datetime(a["event_date"])
    actors = a[["actor1", "assoc_actor_1", "actor2", "assoc_actor_2"]].fillna("").agg(" ; ".join, axis=1)
    a["war"] = (a["inter1"].isin(WAR_INTER) | a["inter2"].isin(WAR_INTER)
                | actors.str.contains(WAR_NAME))
    a["war_broad"] = a["war"] | actors.str.contains(PROGOV)
    a["farmherd"] = actors.str.contains(FARM_HERD)
    a["kw"] = a["notes"].fillna("").str.replace(TRANSPORT, " ", regex=True).str.contains(KW)
    a["resource"] = ~a["war"] & (a["farmherd"] | a["kw"])
    a["resource_strict"] = ~a["war_broad"] & (a["farmherd"] | a["kw"])
    a["communal"] = ~a["war"] & ((a["inter1"] == "Identity militia") | (a["inter2"] == "Identity militia"))
    a["fatalities"] = a["fatalities"].astype(int)
    return a
