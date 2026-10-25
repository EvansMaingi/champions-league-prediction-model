"""Parse match "football boxes" from Wikipedia season articles (second source).

Phase: Milestone 3 - data quality checks.

Previous stage: collect.py saved the Wikipedia API responses in data/raw/wikipedia/.
Next stage: the quality checks compare these matches with the openfootball table
(scores, kickoff times) and use them as evidence for venue flags.

Wikipedia is used only to *check* our main source, never as model input. Its text
is CC BY-SA 4.0, so raw pages stay git-ignored.

Each match on Wikipedia is a template such as:
    {{Football box
    |date       = {{Start date|2021|2|24|df=y}}
    |time       = 21:00
    |team1      = [[Borussia Mönchengladbach]] {{fbaicon|GER}}
    |score      = 0–2
    |stadium    = [[Puskás Aréna]], [[Budapest]], Hungary
    |attendance = 0
    ...}}
"""

import difflib
import json
import re
from pathlib import Path

import pandas as pd

from src.data.clean import ASSOCIATION_CODES

BOX_START = re.compile(r"\{\{\s*(?:#invoke:\s*football box\s*\|\s*main|football box)\s*(?=[|}\n])", re.I)
FIELDS = ["date", "time", "team1", "score", "team2", "stadium", "attendance", "aet", "penaltyscore"]


def load_wikitext(path: Path) -> tuple[str, int]:
    """Return (wikitext, revision id) from a saved Wikipedia API response."""
    page = json.loads(path.read_text(encoding="utf-8"))["query"]["pages"][0]
    revision = page["revisions"][0]
    return revision["slots"]["main"]["content"], revision["revid"]


def find_boxes(wikitext: str) -> list[str]:
    """Return the full text of every football box, matching nested {{ }} pairs."""
    boxes = []
    for start in BOX_START.finditer(wikitext):
        depth, i = 0, start.start()
        while i < len(wikitext):
            if wikitext.startswith("{{", i):
                depth, i = depth + 1, i + 2
            elif wikitext.startswith("}}", i):
                depth, i = depth - 1, i + 2
                if depth == 0:
                    boxes.append(wikitext[start.start():i])
                    break
            else:
                i += 1
    return boxes


def split_params(box: str) -> dict[str, str]:
    """Split a box into {name: raw value}, splitting only on top-level "|"."""
    body = box[2:-2]
    parts, depth, current = [], 0, []
    i = 0
    while i < len(body):
        pair = body[i:i + 2]
        if pair in ("{{", "[["):
            depth, chunk = depth + 1, pair
        elif pair in ("}}", "]]"):
            depth, chunk = depth - 1, pair
        elif body[i] == "|" and depth == 0:
            parts.append("".join(current))
            current, i = [], i + 1
            continue
        else:
            chunk = body[i]
        current.append(chunk)
        i += len(chunk)
    parts.append("".join(current))

    params = {}
    for part in parts[1:]:  # parts[0] is the template name
        if "=" in part:
            name, value = part.split("=", 1)
            params[name.strip().lower()] = value.strip()
    return params


def plain_text(value: str) -> str:
    """Strip refs, comments, templates and link markup: "[[A F.C.|A]] {{fbaicon|ENG}}" -> "A"."""
    text = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>|<!--.*?-->", "", value, flags=re.S)
    previous = None
    while previous != text:  # remove innermost templates until none are left
        previous = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>|'''?", "", text)
    return re.sub(r"\s+", " ", text).strip()


def flag_country(value: str) -> str | None:
    """Return the country code of a flag template, e.g. "{{fbaicon|GER}}" -> "GER".

    Also handles the module form used on newer pages: "{{#invoke:flag|fbaicon|SUI}}".
    """
    match = re.search(r"\{\{\s*(?:#invoke:\s*flag\s*\|\s*)?fb[a-z]*icon\s*\|\s*([A-Z]{3})", value, re.I)
    return match.group(1).upper() if match else None


def link_target(value: str) -> str | None:
    """Return the article a team field links to: "[[FC Bayern Munich|Bayern Munich]]" -> "FC Bayern Munich".

    The article title identifies a club uniquely, unlike its display name.
    """
    match = re.search(r"\[\[([^|\]#]+)", value)
    return match.group(1).strip().replace("_", " ") if match else None


def parse_box(box: str) -> dict[str, object]:
    """Turn one football box into a flat record (date, time, teams, goals, venue, attendance)."""
    params = split_params(box)
    raw = {name: params.get(name, "") for name in FIELDS}

    date_match = re.search(r"\{\{\s*start date\s*\|\s*(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})", raw["date"], re.I)
    time_match = re.search(r"\b(\d{1,2}):(\d{2})\b", raw["time"])
    score_match = re.search(r"(\d+)\s*[–\-−]\s*(\d+)", plain_text(raw["score"]))
    attendance_text = plain_text(raw["attendance"]).replace(",", "")
    attendance_match = re.match(r"^(\d+)", attendance_text)

    return {
        "date": pd.Timestamp(*map(int, date_match.groups())) if date_match else pd.NaT,
        "kickoff_cet": f"{int(time_match.group(1)):02d}:{time_match.group(2)}" if time_match else None,
        "team1": plain_text(raw["team1"]),
        "team1_country": flag_country(raw["team1"]),
        "team1_article": link_target(raw["team1"]),
        "team2": plain_text(raw["team2"]),
        "team2_country": flag_country(raw["team2"]),
        "team2_article": link_target(raw["team2"]),
        "goals1": int(score_match.group(1)) if score_match else None,
        "goals2": int(score_match.group(2)) if score_match else None,
        "stadium": plain_text(raw["stadium"]),
        "attendance": int(attendance_match.group(1)) if attendance_match else None,
        "attendance_text": attendance_text,
        "time_raw": plain_text(raw["time"]),
    }


def load_wikipedia_matches(raw_dir: Path) -> pd.DataFrame:
    """Parse every saved season page ("YYYY-YY_<kind>.json") into one table.

    Adds the season, page kind and revision id to every row.
    """
    rows = []
    for path in sorted(raw_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]_*.json")):
        season, page = path.stem.split("_", 1)
        wikitext, revid = load_wikitext(path)
        for box in find_boxes(wikitext):
            rows.append({"season": season, "page": page, "revid": revid, **parse_box(box)})
    return pd.DataFrame(rows)


# --- Matching Wikipedia boxes to openfootball rows ---------------------------


def name_similarity(a: str, b: str) -> float:
    """Similarity of two team names in [0, 1], ignoring case and punctuation."""
    clean = lambda s: re.sub(r"[^\w]+", " ", s.lower()).strip()  # noqa: E731
    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()


def match_to_openfootball(matches: pd.DataFrame, wiki: pd.DataFrame) -> pd.DataFrame:
    """Pair each openfootball match with at most one Wikipedia box, without using scores.

    Matching keys: date + home/away association codes; name similarity breaks ties.
    Unmatched rows get a second pass allowing a one-day date difference (matches
    that were suspended and completed the next day). Each box is used at most once.

    Returns:
        One row per openfootball match with the matched box's columns prefixed "wp_"
        (all <NA> when no box matched) and the date difference in days.
    """
    of = matches.reset_index(names="of_index")
    of["home_assoc"] = of["home_country"].replace(ASSOCIATION_CODES)
    of["away_assoc"] = of["away_country"].replace(ASSOCIATION_CODES)
    used: set[int] = set()
    pairs: dict[int, int] = {}

    for max_days in (0, 1):
        for row in of[~of["of_index"].isin(pairs)].itertuples():
            candidates = wiki[
                (wiki["team1_country"] == row.home_assoc)
                & (wiki["team2_country"] == row.away_assoc)
                & ((wiki["date"] - row.date).abs() <= pd.Timedelta(days=max_days))
                & ~wiki.index.isin(used)
            ]
            if candidates.empty:
                continue
            scores = candidates.apply(
                lambda c: name_similarity(row.home_team, c["team1"]) + name_similarity(row.away_team, c["team2"]),
                axis=1,
            )
            best = scores.idxmax()
            pairs[row.of_index] = best
            used.add(best)

    matched = wiki.loc[list(pairs.values())].add_prefix("wp_")
    matched.index = list(pairs.keys())
    result = of.set_index("of_index").join(matched)
    result["date_diff_days"] = (result["wp_date"] - result["date"]).dt.days
    return result.drop(columns=["home_assoc", "away_assoc"])
