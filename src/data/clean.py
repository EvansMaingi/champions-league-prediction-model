"""Turn raw match files into one tidy, consistent match table.

Phase: Milestone 3 - Exploratory data analysis and cleaning.

Previous stage: collect.py saved raw openfootball Football.TXT files in data/raw/.
Next stage: validate.py checks this table; build_features.py then uses it.

Step 1 (implemented): parse the raw text into one row per match. The parser fails
loudly on any line it does not recognise instead of skipping it, so no match can go
missing silently. It records every score the source gives (90 minutes, half-time,
after extra time, penalties) without deciding yet which one defines the result
(that is Milestone 4).

Football.TXT layout handled here:
    ▪ Group A                      stage header
      Wed Sep 14 2011              date (the year is shown only occasionally)
        20:45  Home (ENG)  v  Away (ITA)   1-1 (0-0)
               Home (ESP)  v  Away (GER)   0-2 (0-1)   <- inherits the 20:45 kickoff

Kickoff times are stored as written; they appear to be Central European local time
(assumption, to be verified during EDA).

TODO (later steps, shaped by EDA findings):
- normalise_team_names(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame
    Map spelling variants to one canonical team name.
- clean_matches(raw_dir: Path) -> pd.DataFrame
    Parse, standardise, save to PROCESSED_DATA_DIR, and return the clean table.
"""

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.config import OPENFOOTBALL_RAW_DIR

DATE_RE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?$")
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})\s+")
TEAM_RE = re.compile(r"^(.+?)\s+\(([A-Z]{3})\)$")
# Score formats, from most to least detailed. "ft" = after 90 minutes.
SCORE_PATTERNS = [
    re.compile(r"^(?P<pens>\d+-\d+) pen\. (?P<aet>\d+-\d+) a\.e\.t\. \((?P<ft>\d+-\d+)(?:, (?P<ht>\d+-\d+))?\)$"),
    re.compile(r"^(?P<aet>\d+-\d+) a\.e\.t\. \((?P<ft>\d+-\d+)(?:, (?P<ht>\d+-\d+))?\)$"),
    re.compile(r"^(?P<ft>\d+-\d+)(?: \((?P<ht>\d+-\d+)\))?$"),
]
# Ordered checks: "Round of 16" before "Final" etc. The source mixes English and German.
STAGE_RULES = [
    ("Round of 16", "round_of_16"),
    ("Quarterfinals", "quarterfinal"),
    ("Semifinals", "semifinal"),
    ("Final", "final"),
    ("Playoffs", "playoff"),
    ("League", "league"),
    ("Group", "group"),
    ("Gruppe", "group"),
]
SCORE_KINDS = ["ft", "ht", "aet", "pens"]
COLUMNS = (
    ["season", "date", "kickoff_local", "stage", "stage_raw",
     "home_team", "home_country", "away_team", "away_country"]
    + [f"{side}_{kind}" for kind in SCORE_KINDS for side in ("home", "away")]
    + ["source_file"]
)


def parse_score(text: str) -> dict[str, int | None]:
    """Parse a score string into home/away goals for each score kind present.

    Examples:
        "1-1 (0-0)"                       -> ft 1-1, ht 0-0
        "4-1 a.e.t. (3-1, 1-0)"           -> aet 4-1, ft 3-1, ht 1-0
        "4-3 pen. 1-0 a.e.t. (1-0, 1-0)"  -> pens 4-3, aet 1-0, ft 1-0, ht 1-0

    Returns:
        Keys like "home_ft", "away_ft", ... for all SCORE_KINDS; None when absent.

    Raises:
        ValueError: If the text matches no known score format.
    """
    for pattern in SCORE_PATTERNS:
        match = pattern.match(text.strip())
        if match:
            break
    else:
        raise ValueError(f"Unrecognised score format: {text!r}")

    result: dict[str, int | None] = {}
    for kind in SCORE_KINDS:
        value = match.groupdict().get(kind)
        home, away = (int(g) for g in value.split("-")) if value else (None, None)
        result[f"home_{kind}"], result[f"away_{kind}"] = home, away
    return result


def parse_date_line(line: str, current_year: int | None) -> tuple[date, int]:
    """Parse a date line such as "Wed Sep 14 2011" or "Tue Sep 30".

    Lines without a year use current_year (the last year stated in the file). The
    weekday is checked against the computed date, which catches a wrongly inferred year.

    Returns:
        (the date, the year to carry forward to the next date line).

    Raises:
        ValueError: If the line is not a date, no year is known yet, or the weekday
        does not match the date.
    """
    match = DATE_RE.match(line.strip())
    if not match:
        raise ValueError(f"Not a date line: {line!r}")
    weekday, month, day, year = match.groups()
    year = int(year) if year else current_year
    if year is None:
        raise ValueError(f"Date without a year before any year was given: {line!r}")
    parsed = datetime.strptime(f"{day} {month} {year}", "%d %b %Y").date()
    if parsed.strftime("%a") != weekday:
        raise ValueError(f"Weekday mismatch for {line!r}: {parsed} is a {parsed:%a}")
    return parsed, year


def normalise_stage(header: str) -> str:
    """Map a stage header such as "Finals, Round of 16" or "Gruppe G" to a stage code."""
    for keyword, stage in STAGE_RULES:
        if keyword in header:
            return stage
    raise ValueError(f"Unrecognised stage header: {header!r}")


def split_team(text: str) -> tuple[str, str]:
    """Split "Arsenal FC (ENG)" into ("Arsenal FC", "ENG")."""
    match = TEAM_RE.match(text.strip())
    if not match:
        raise ValueError(f"Team without a country code: {text!r}")
    return match.group(1), match.group(2)


def parse_match_line(line: str) -> dict[str, str | int | None]:
    """Parse a match line into kickoff time (or None), teams, countries and scores.

    Example: "20:45  Chelsea FC (ENG)  v SSC Napoli (ITA)  4-1 a.e.t. (3-1, 1-0)"
    """
    text = line.strip()
    kickoff = None
    time_match = TIME_RE.match(text)
    if time_match:
        kickoff = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
        text = text[time_match.end():]

    home_text, rest = text.split(" v ", 1)
    # The away team ends at its "(CCC)" country code; the score follows.
    away_match = re.match(r"^(.+?\([A-Z]{3}\))\s+(.+)$", rest.strip())
    if not away_match:
        raise ValueError(f"Cannot separate away team from score: {line!r}")
    home_team, home_country = split_team(home_text)
    away_team, away_country = split_team(away_match.group(1))
    return {
        "kickoff_local": kickoff,
        "home_team": home_team,
        "home_country": home_country,
        "away_team": away_team,
        "away_country": away_country,
        **parse_score(away_match.group(2)),
    }


def parse_season_text(text: str, season: str, source_file: str = "") -> pd.DataFrame:
    """Parse one season's Football.TXT content into a match table (one row per match).

    Raises:
        ValueError: On any unrecognised line, with the source file and line number.
    """
    rows: list[dict] = []
    stage = stage_raw = None
    current_date: date | None = None
    current_year: int | None = None
    current_time: str | None = None

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        try:
            if not stripped or stripped.startswith(("#", "= ")):
                continue  # blank line, comment or title
            if stripped.startswith("▪"):
                stage_raw = stripped.lstrip("▪").strip()
                stage = normalise_stage(stage_raw)
            elif DATE_RE.match(stripped):
                current_date, current_year = parse_date_line(stripped, current_year)
                current_time = None  # kickoff times never carry over to a new day
            elif " v " in stripped:
                if stage is None or current_date is None:
                    raise ValueError("Match line before any stage or date line")
                match = parse_match_line(stripped)
                current_time = match["kickoff_local"] or current_time
                match["kickoff_local"] = current_time
                rows.append({"season": season, "date": current_date, "stage": stage,
                             "stage_raw": stage_raw, **match, "source_file": source_file})
            else:
                raise ValueError("Unrecognised line")
        except ValueError as error:
            raise ValueError(f"{source_file or season} line {number}: {error} | {stripped!r}") from error

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    score_cols = [c for c in COLUMNS if c.split("_")[-1] in SCORE_KINDS]
    df[score_cols] = df[score_cols].astype("Int64")  # nullable: missing stays <NA>, not 0
    return df


def load_openfootball(raw_dir: Path = OPENFOOTBALL_RAW_DIR) -> pd.DataFrame:
    """Parse every "<season>_cl.txt" file in raw_dir into one chronologically sorted table."""
    paths = sorted(raw_dir.glob("*_cl.txt"))
    if not paths:
        raise FileNotFoundError(f"No *_cl.txt files in {raw_dir}; run: python -m src.data.collect")
    tables = [
        parse_season_text(p.read_text(encoding="utf-8"), p.name.removesuffix("_cl.txt"), p.name)
        for p in paths
    ]
    return (
        pd.concat(tables, ignore_index=True)
        .sort_values(["date", "kickoff_local"], kind="stable", na_position="last")
        .reset_index(drop=True)
    )
