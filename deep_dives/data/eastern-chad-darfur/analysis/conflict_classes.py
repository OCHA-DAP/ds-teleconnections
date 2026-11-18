"""Event classes for the conflict page (ACLED, eastern Chad + Darfur). Definitions only, no results.

Every class is built from who fought and what the event notes say was fought over - never
from words like "drought" or "scarcity" (sources attribute violence to drought more often
in dry years, which would build the answer in).

  war        a party to the wider wars is involved: state forces (Sudan, Chad, other states),
             rebel groups (incl. the RSF since 2023, and the RSF while a state force), and
             foreign/peacekeeping forces. Formal parties only: ACLED's militia labels for the
             same kind of attack can change over time, and a filter on them would put coding
             changes into the series.
  war_broad  also pro-government, civil-war and self-defence militias (Janjaweed, "Militia
             (Pro-Government)", PDF, Musa Hilal's militia; for the belt also Seleka, anti-Balaka,
             Dan Na Ambassagou, GATIA, CMFPR, Koglweogo, VDP, CJTF): sensitivity check, since these
             labels can cover both the wars and attacks over land and herds.
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
                      r"MINURCAT|EUFOR|Wagner|MINUSMA|MINUSCA|UNMISS|MNJTF|Barkhane|Takuba|Africa Corps|"
                      r"G5 Sahel")
PROGOV = re.compile(r"Janjaweed|Militia \(Pro-Government\)|PDF: Popular Defence|Militia \(Musa Hilal\)|"
                    r"Seleka|Anti-Balaka|Dan Na Ambassagou|GATIA|CMFPR|Koglweogo|VDP|Volunteers for the Defen|"
                    r"CJTF|Civilian JTF")  # strict filter: pro-government, civil-war and self-defence militias too
FARM_HERD = re.compile(r"\b(Pastoralists|Farmers|Herders)\b")
KW = re.compile(r"\b(?:cattle|cows?|camels?|livestock|goats?|sheep|herds?|herding|herders?|herdsm[ae]n|"
                r"pastoral\w*|nomads?|nomadic|graz\w*|pastures?|farms?|farmers?|farmland|farming|crops?|"
                r"harvest\w*|cultivat\w*|(?:farm|agricultural) ?fields?|water ?points?|(?:water )?wells|"
                r"boreholes?|water sources?|grain|sorghum|millet|groundnuts?)\b", re.I)
# camels and horses as the attackers' transport ("riding camels", "on camelback") are not livestock
TRANSPORT = re.compile(r"\b(?:riding|rode|on|mounted on|aboard)\b[^.;,]{0,25}?\bcamel\w*(?:\s*(?:and|/|or)\s*horse\w*)?"
                       r"|\bcamel\s*/\s*horse\w*|\bhorse\w* and camels?", re.I)


def classify(ev: pd.DataFrame, area: dict | None = A1) -> pd.DataFrame:
    """ACLED rows -> Battles + Violence against civilians with class flags; in the study area by default,
    everywhere with area=None (side = country)."""
    a = ev[ev["event_type"].isin(EVENT_TYPES)].copy()
    if area is not None:
        a = a[a["admin1"].isin(area)]
        a["side"] = a["admin1"].map(area)
    else:
        a["side"] = a["country"]
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
