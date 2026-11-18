"""Wikipedia 'List of Formula One circuits': length, turns, type, direction.

Rows are matched to Jolpica circuits by the Wikipedia link in each row
(Jolpica stores the circuit's Wikipedia URL), with a name-similarity fallback.
"""
from __future__ import annotations

import difflib
import io
import re
from urllib.parse import unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .common import USER_AGENT, cache_dir

URL = "https://en.wikipedia.org/wiki/List_of_Formula_One_circuits"

# Jolpica URL slug -> slug used in the Wikipedia table, when they differ
SLUG_OVERRIDES = {
    "Las_Vegas_Grand_Prix": "Las_Vegas_Strip_Circuit",
    "Suzuka_International_Racing_Course": "Suzuka_Circuit",
}


def _slug(href: str) -> str:
    return unquote(href.split("/wiki/")[-1].split("#")[0]).replace(" ", "_").lower()


def _fetch_html() -> str:
    path = cache_dir("wikipedia") / "list_of_f1_circuits.html"
    if not path.exists():
        r = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def circuits_table() -> pd.DataFrame:
    html = _fetch_html()
    tables = pd.read_html(io.StringIO(html))
    df = max(tables, key=len).copy()
    # collect the link of the first cell of each row of the same table
    soup = BeautifulSoup(html, "lxml")
    wt = max(soup.select("table.wikitable"), key=lambda t: len(t.select("tr")))
    links = []
    for tr in wt.select("tr")[1:]:
        cell = tr.find(["th", "td"])
        a = cell.find("a", href=True) if cell else None
        links.append(_slug(a["href"]) if a else "")
    df["wiki_slug"] = links[: len(df)]
    df["Circuit"] = df["Circuit"].str.replace(r"[\*†‡]", "", regex=True).str.strip()
    df["length_km"] = df["Last length used"].str.extract(r"([\d.]+)\s*km").astype(float)
    df["n_turns"] = pd.to_numeric(df["Turns"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["street_circuit"] = df["Type"].str.contains("Street", case=False).astype(int)
    df["clockwise"] = df["Direction"].str.contains("Anti|Counter", case=False).map({True: 0, False: 1})
    return df[["Circuit", "wiki_slug", "Type", "length_km", "n_turns", "street_circuit", "clockwise"]]


def match_circuits(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return one row per circuit_id with Wikipedia features."""
    wiki = circuits_table()
    out = []
    for _, c in schedule.drop_duplicates("circuit_id").iterrows():
        slug = c["circuit_url"].split("/wiki/")[-1].split("#")[0]
        slug = _slug("/wiki/" + SLUG_OVERRIDES.get(unquote(slug), slug))
        hit = wiki[wiki["wiki_slug"] == slug]
        how = "url"
        if hit.empty:
            names = wiki["Circuit"].tolist()
            best = difflib.get_close_matches(c["circuit_name"], names, n=1, cutoff=0.5)
            hit = wiki[wiki["Circuit"] == best[0]] if best else hit
            how = "name"
        row = {"circuit_id": c["circuit_id"], "wiki_match": how if not hit.empty else "none"}
        if not hit.empty:
            h = hit.iloc[0]
            row.update({k: h[k] for k in ["Circuit", "length_km", "n_turns", "street_circuit", "clockwise"]})
            row["wiki_circuit"] = row.pop("Circuit")
        out.append(row)
    return pd.DataFrame(out)
