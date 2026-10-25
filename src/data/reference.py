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
  similarity, and the standard name is that article's most recent display name.
- match_context.csv: matches at a neutral venue and/or behind closed doors, each with
  the reason and the evidence it comes from (a documented rule or source data).

Run from the project root:  python -m src.data.reference
"""

import json
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import FOOTBALL_DATA_RAW_DIR, REFERENCE_DIR, WIKIPEDIA_RAW_DIR
from src.data.clean import load_openfootball
from src.data.collect import collect_wikipedia_title_resolution
from src.data.wikipedia import load_wikipedia_matches, match_to_openfootball

TITLE_RESOLUTION_FILE = WIKIPEDIA_RAW_DIR / "title_resolution.json"
CET = ZoneInfo("Europe/Paris")  # UEFA lists kickoffs in Central European (Summer) Time


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
    wikipedia_article, evidence.
    """
    apps = team_appearances(matched).dropna(subset=["article"])
    apps["article"] = apps["article"].map(lambda t: resolve.get(t, t))

    # Standard name = the article's most recent display name (any season, incl. 2026-27).
    linked = pd.concat([
        wiki[["date", "team1_article", "team1"]].set_axis(["date", "article", "display"], axis=1),
        wiki[["date", "team2_article", "team2"]].set_axis(["date", "article", "display"], axis=1),
    ]).dropna(subset=["article"])
    linked["article"] = linked["article"].map(lambda t: resolve.get(t, t))
    standard = linked.sort_values("date").groupby("article")["display"].last()

    rows = []
    for (name, country), group in apps.groupby(["name", "country"]):
        counts = group["article"].value_counts()
        rows.append({
            "source": "openfootball", "source_id": "", "source_name": name, "country": country,
            "standard_name": standard[counts.index[0]], "wikipedia_article": counts.index[0],
            "evidence": f"date+country match: {counts.iloc[0]}/{len(group)} matches"
            + ("" if len(counts) == 1 else f"; CONFLICT {dict(counts)}"),
        })

    season_2627 = wiki[(wiki["season"] == "2026-27") & (wiki["page"] == "league_phase")]
    accepted = match_fixture_teams(fixtures, season_2627)
    names = {m[side]["id"]: m[side]["name"] for m in fixtures for side in ("homeTeam", "awayTeam")}
    for team_id, name in sorted(names.items(), key=lambda x: x[1]):
        article, score = accepted.get(team_id, (None, 0.0))
        article = resolve.get(article, article)
        rows.append({
            "source": "football-data.org", "source_id": str(team_id), "source_name": name,
            "country": "", "standard_name": standard.get(article, "") if article else "",
            "wikipedia_article": article or "",
            "evidence": f"fixture schedule match: similarity {score}" if article else "NO SCHEDULE MATCH",
        })
    return pd.DataFrame(rows)


# --- Match context (venue and crowd) ------------------------------------------


def build_match_context(matched: pd.DataFrame) -> pd.DataFrame:
    """Flag neutral venues and behind-closed-doors matches, with reason and evidence.

    Neutral venue (one reason per match, first that applies):
      - "final": every final is at a venue chosen in advance (rule).
      - "lisbon_final_8": 2019-20 quarter- and semi-finals, single matches in Lisbon (rule).
      - "played_abroad": Wikipedia's stadium field names a country, which it does only
        when the "home" side played outside its own country (source data).
    Behind closed doors: Wikipedia attendance is 0 (source data).
    """
    m = matched.copy()
    stadium = m["wp_stadium"].fillna("")
    m["neutral_reason"] = None
    m.loc[stadium.str.count(",") >= 2, "neutral_reason"] = "played_abroad"
    m.loc[(m["season"] == "2019-20") & m["stage"].isin(["quarterfinal", "semifinal"]), "neutral_reason"] = "lisbon_final_8"
    m.loc[m["stage"] == "final", "neutral_reason"] = "final"
    m["behind_closed_doors"] = m["wp_attendance"] == 0

    # A "neutral" final at one finalist's own usual stadium is worth knowing about.
    usual = m[m["stage"] != "final"].groupby("home_team")["wp_stadium"].agg(lambda s: s.value_counts().index[0])
    m["note"] = ""
    for idx, row in m[m["stage"] == "final"].iterrows():
        for team in (row["home_team"], row["away_team"]):
            if usual.get(team) == row["wp_stadium"]:
                m.loc[idx, "note"] = f"played at {team}'s usual home stadium"

    flagged = m[m["neutral_reason"].notna() | m["behind_closed_doors"]]
    evidence = flagged["neutral_reason"].map({
        "final": "rule", "lisbon_final_8": "rule", "played_abroad": "Wikipedia stadium field",
    })
    return pd.DataFrame({
        "season": flagged["season"],
        "date": flagged["date"].dt.date,
        "home_team": flagged["home_team"],
        "away_team": flagged["away_team"],
        "neutral_venue": flagged["neutral_reason"].notna(),
        "neutral_reason": flagged["neutral_reason"].fillna(""),
        "neutral_evidence": evidence.fillna(""),
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
