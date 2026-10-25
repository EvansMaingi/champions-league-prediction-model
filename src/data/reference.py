"""Build hand-reviewable reference tables from the matched sources.

Phase: Milestone 3 - data cleaning.

Previous stage: wikipedia.py pairs every openfootball match with its Wikipedia box.
Next stage: clean.py applies the reviewed tables in data/reference/ to the match table.

The tables are *proposals*: they are written to data/reference/ as CSV, reviewed by a
person, and committed. The cleaning code then reads only these committed files, so the
model pipeline never depends on the git-ignored Wikipedia pages.

- team_names.csv: every team name in openfootball and football-data.org -> one
  standard name. Clubs are identified by the Wikipedia article they link to (found
  by matching on date and countries, or by fixture schedule), never by string
  similarity. The standard name is derived from the article title by
  standard_name_from_title (rule documented there and in docs/data_notes.md).
- match_context.csv: matches at a truly neutral venue, played abroad, and/or behind
  closed doors, each with the evidence it comes from (a documented rule or source data).

Run from the project root:  python -m src.data.reference
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import FOOTBALL_DATA_RAW_DIR, REFERENCE_DIR, WIKIPEDIA_RAW_DIR
from src.data.clean import ASSOCIATION_CODES, load_openfootball
from src.data.collect import collect_wikipedia_title_resolution
from src.data.wikipedia import load_wikipedia_matches, match_to_openfootball

TITLE_RESOLUTION_FILE = WIKIPEDIA_RAW_DIR / "title_resolution.json"
CET = ZoneInfo("Europe/Paris")  # UEFA lists kickoffs in Central European (Summer) Time

# Club-type / legal-form abbreviations removed from the start or end of an article
# title. Tokens that are part of a club's name are deliberately absent (e.g. RB, PSV,
# VfB, VfL, TSG, CFR, LASK), and so is "TC": "Ferencvárosi TC" without it would leave an
# adjective ("of Ferencváros"), not a name.
LEGAL_FORM_TOKENS = [
    "1. FC", "F.C.", "FC", "A.F.C.", "AFC", "C.F.", "CF", "S.C.", "SC", "S.K.", "SK", "ŠK",
    "F.K.", "FK", "NK", "GNK", "PFC", "BSC", "KRC", "KAA", "RSC", "RC", "AC", "AS", "ASC",
    "SS", "SSC", "S.L.", "J.K.", "BC", "KV", "OSC", "HSC", "FF",
]

# Countries (UEFA associations) for 2026-27 clubs with no openfootball history,
# entered by hand and reviewed (2026-10-25).
NEW_CLUB_COUNTRIES = {
    "Como": "ITA", "Fenerbahçe": "TUR", "LASK": "AUT",
    "Real Betis": "ESP", "Sabah": "AZE", "Viking": "NOR",
}


def standard_name_from_title(title: str) -> str:
    """Derive a club's standard name from its Wikipedia article title.

    Rule, repeated until nothing changes:
      1. drop a trailing disambiguator in parentheses: "Galatasaray S.K. (football)";
      2. drop a trailing founding year (1800-2099): "Bologna FC 1909" -> "Bologna FC";
      3. drop a LEGAL_FORM_TOKENS entry at the start and/or end, as a whole word:
         "FC Bayern Munich" -> "Bayern Munich", "Arsenal F.C." -> "Arsenal".
    Everything else in the title is kept as it is (spelling, accents, word order).
    """
    name = re.sub(r"\s*\([^)]*\)$", "", title).strip()
    tokens = sorted(LEGAL_FORM_TOKENS, key=len, reverse=True)  # "1. FC" before "FC"
    while True:
        before = name
        name = re.sub(r"\s+(18|19|20)\d{2}$", "", name)
        for token in tokens:
            if name.startswith(token + " "):
                name = name[len(token) + 1:]
            if name.endswith(" " + token):
                name = name[: -len(token) - 1]
        if name == before:
            return name


# --- Team names ---------------------------------------------------------------


def resolve_titles(path: Path) -> dict[str, str]:
    """Return {title: final article title} from saved title-resolution API responses."""
    final: dict[str, str] = {}
    for response in json.loads(path.read_text(encoding="utf-8")):
        query = response["query"]
        steps = {s["from"]: s["to"] for s in query.get("normalized", []) + query.get("redirects", [])}
        for page in query["pages"]:
            final[page["title"]] = page["title"]
        for start in steps:
            title = start
            while title in steps:
                title = steps[title]
            final[start] = title
    return final


def team_appearances(matched: pd.DataFrame) -> pd.DataFrame:
    """One row per (openfootball team, match): name, country, linked article, display name, date."""
    home = matched[["date", "home_team", "home_country", "wp_team1_article", "wp_team1"]]
    away = matched[["date", "away_team", "away_country", "wp_team2_article", "wp_team2"]]
    columns = ["date", "name", "country", "article", "display"]
    return pd.concat([home.set_axis(columns, axis=1), away.set_axis(columns, axis=1)], ignore_index=True)


def fixture_signatures(fixtures: list[dict]) -> dict[int, set[tuple[str, str]]]:
    """football-data.org team id -> {(CET date, "H"/"A")} over all its fixtures."""
    signatures: dict[int, set] = defaultdict(set)
    for match in fixtures:
        day = pd.Timestamp(match["utcDate"]).tz_convert(CET).date().isoformat()
        signatures[match["homeTeam"]["id"]].add((day, "H"))
        signatures[match["awayTeam"]["id"]].add((day, "A"))
    return signatures


def match_fixture_teams(fixtures: list[dict], wiki_season: pd.DataFrame) -> dict[int, tuple[str, float]]:
    """Match football-data.org teams to Wikipedia articles by fixture schedule, not by name.

    A team's set of (date, home/away) over the league phase is compared with every
    Wikipedia club's set (Jaccard similarity). A match is accepted only if it is the
    unique best and at least 0.75, which tolerates a postponed fixture or two.

    Returns:
        {team id: (article, similarity)} for accepted matches.
    """
    wiki_sigs: dict[str, set] = defaultdict(set)
    for row in wiki_season.itertuples():
        day = row.date.date().isoformat()
        wiki_sigs[row.team1_article].add((day, "H"))
        wiki_sigs[row.team2_article].add((day, "A"))

    accepted = {}
    for team_id, signature in fixture_signatures(fixtures).items():
        scores = sorted(
            ((len(signature & s) / len(signature | s), article) for article, s in wiki_sigs.items()),
            reverse=True,
        )
        (best, article), runner_up = scores[0], scores[1][0] if len(scores) > 1 else 0.0
        if best >= 0.75 and best > runner_up:
            accepted[team_id] = (article, round(best, 2))
    return accepted


def build_team_names(
    matched: pd.DataFrame, wiki: pd.DataFrame, fixtures: list[dict], resolve: dict[str, str]
) -> pd.DataFrame:
    """Propose the team-name table for both sources (see module docstring).

    Returns columns: source, source_id, source_name, country, standard_name,
    wikipedia_article, evidence. Countries are UEFA association codes.
    """
    apps = team_appearances(matched).dropna(subset=["article"])
    apps["article"] = apps["article"].map(lambda t: resolve.get(t, t))
    apps["country"] = apps["country"].replace(ASSOCIATION_CODES)

    rows = []
    for (name, country), group in apps.groupby(["name", "country"]):
        counts = group["article"].value_counts()
        rows.append({
            "source": "openfootball", "source_id": "", "source_name": name, "country": country,
            "standard_name": standard_name_from_title(counts.index[0]),
            "wikipedia_article": counts.index[0],
            "evidence": f"date+country match: {counts.iloc[0]}/{len(group)} matches"
            + ("" if len(counts) == 1 else f"; CONFLICT {dict(counts)}"),
        })

    club_country = {row["standard_name"]: row["country"] for row in rows}

    season_2627 = wiki[(wiki["season"] == "2026-27") & (wiki["page"] == "league_phase")]
    accepted = match_fixture_teams(fixtures, season_2627)
    names = {m[side]["id"]: m[side]["name"] for m in fixtures for side in ("homeTeam", "awayTeam")}
    for team_id, name in sorted(names.items(), key=lambda x: x[1]):
        article, score = accepted.get(team_id, (None, 0.0))
        article = resolve.get(article, article)
        standard = standard_name_from_title(article) if article else ""
        evidence = f"fixture schedule match: similarity {score}" if article else "NO SCHEDULE MATCH"
        country = club_country.get(standard, "")
        if not country and standard in NEW_CLUB_COUNTRIES:
            country = NEW_CLUB_COUNTRIES[standard]
            evidence += "; no history, country entered by hand"
        rows.append({
            "source": "football-data.org", "source_id": str(team_id), "source_name": name,
            "country": country, "standard_name": standard, "wikipedia_article": article or "",
            "evidence": evidence,
        })
    return pd.DataFrame(rows)


# --- Match context (venue and crowd) ------------------------------------------


def build_match_context(matched: pd.DataFrame) -> pd.DataFrame:
    """Flag venue and crowd context, each flag with its evidence.

    - neutral_venue (truly neutral: neither side at home), with neutral_reason:
        "final": every final is at a venue chosen in advance (rule);
        "lisbon_final_8": 2019-20 quarter- and semi-finals, single matches in Lisbon (rule).
    - played_abroad: the listed home side played outside its own country. Evidence:
      Wikipedia's stadium field names a country, which it does only in that case.
    - behind_closed_doors: Wikipedia attendance is 0 (source data).
    """
    m = matched.copy()
    stadium = m["wp_stadium"].fillna("")
    m["neutral_reason"] = None
    m.loc[(m["season"] == "2019-20") & m["stage"].isin(["quarterfinal", "semifinal"]), "neutral_reason"] = "lisbon_final_8"
    m.loc[m["stage"] == "final", "neutral_reason"] = "final"
    m["played_abroad"] = stadium.str.count(",") >= 2
    m["behind_closed_doors"] = m["wp_attendance"] == 0

    # A "neutral" final at one finalist's own usual stadium is worth knowing about.
    usual = m[m["stage"] != "final"].groupby("home_team")["wp_stadium"].agg(lambda s: s.value_counts().index[0])
    m["note"] = ""
    for idx, row in m[m["stage"] == "final"].iterrows():
        for team in (row["home_team"], row["away_team"]):
            if usual.get(team) == row["wp_stadium"]:
                m.loc[idx, "note"] = f"played at {team}'s usual home stadium"

    flagged = m[m["neutral_reason"].notna() | m["played_abroad"] | m["behind_closed_doors"]]
    return pd.DataFrame({
        "season": flagged["season"],
        "date": flagged["date"].dt.date,
        "home_team": flagged["home_team"],
        "away_team": flagged["away_team"],
        "neutral_venue": flagged["neutral_reason"].notna(),
        "neutral_reason": flagged["neutral_reason"].fillna(""),
        "neutral_evidence": flagged["neutral_reason"].notna().map({True: "rule", False: ""}),
        "played_abroad": flagged["played_abroad"],
        "abroad_evidence": flagged["played_abroad"].map({True: "Wikipedia stadium field", False: ""}),
        "behind_closed_doors": flagged["behind_closed_doors"],
        "bcd_evidence": flagged["behind_closed_doors"].map({True: "Wikipedia attendance = 0", False: ""}),
        "stadium": flagged["wp_stadium"],
        "attendance": flagged["wp_attendance"].astype("Int64"),
        "note": flagged["note"],
        "wikipedia_revid": flagged["wp_revid"].astype("Int64"),
    }).reset_index(drop=True)


def latest_fixtures(raw_dir: Path = FOOTBALL_DATA_RAW_DIR) -> list[dict]:
    """Return the matches of the newest football-data.org snapshot."""
    snapshots = sorted(raw_dir.glob("cl_matches_*.json"))
    if not snapshots:
        raise FileNotFoundError(f"No fixture snapshot in {raw_dir}; run: python -m src.data.collect fixtures")
    return json.loads(snapshots[-1].read_text(encoding="utf-8"))["matches"]


def main() -> None:
    """Write proposed data/reference/team_names.csv and match_context.csv."""
    matches = load_openfootball()
    wiki = load_wikipedia_matches(WIKIPEDIA_RAW_DIR)
    matched = match_to_openfootball(matches, wiki)
    titles = sorted(set(wiki["team1_article"].dropna()) | set(wiki["team2_article"].dropna()))
    resolve = resolve_titles(collect_wikipedia_title_resolution(titles, TITLE_RESOLUTION_FILE))

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    teams = build_team_names(matched, wiki, latest_fixtures(), resolve)
    teams.to_csv(REFERENCE_DIR / "team_names.csv", index=False, encoding="utf-8")
    context = build_match_context(matched)
    context.to_csv(REFERENCE_DIR / "match_context.csv", index=False, encoding="utf-8")
    print(f"team_names.csv: {len(teams)} rows | match_context.csv: {len(context)} rows")


if __name__ == "__main__":
    main()
