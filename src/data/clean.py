"""Turn raw match files into one tidy, consistent match table.

Phase: Milestone 3 - Exploratory data analysis and cleaning.

Previous stage: collect.py saved raw openfootball Football.TXT files in data/raw/.
Next stage: validate.py checks this table; build_features.py then uses it.

Step 1 - parse the raw text into one row per match. The parser fails loudly on any
line it does not recognise instead of skipping it, so no match can go missing
silently. It records every score the source gives (90 minutes, half-time, after
extra time, penalties) without deciding yet which one defines the result
(that is Milestone 4).

Football.TXT layout handled here:
    ▪ Group A                      stage header
      Wed Sep 14 2011              date (the year is shown only occasionally)
        20:45  Home (ENG)  v  Away (ITA)   1-1 (0-0)
               Home (ESP)  v  Away (GER)   0-2 (0-1)   <- inherits the 20:45 kickoff

Step 2 - cleaning decisions from EDA (each documented in docs/data_notes.md):
- Kickoff times are Central European (Summer) Time, as listed by UEFA, not local
  venue time: 1,992 of 1,997 match Wikipedia's CET/CEST times exactly (the other 5
  are delayed or suspended matches). `kickoff_utc` is derived from them.
- A missing half-time score occurs only when the match ended 0-0, so it is filled
  with 0-0 and marked `ht_inferred`. Half-time scores are post-kickoff data: they may
  only feed rolling features built from EARLIER matches, never the current match.
- Venue and crowd context (`neutral_venue`, `neutral_reason`, `played_abroad`,
  `behind_closed_doors`) comes from the reviewed table data/reference/match_context.csv.
- Knockout matches get `tie_id` and `leg` ("1", "2" or "single").
- Country codes are UEFA associations: AS Monaco's MCO (used from 2024-25) becomes FRA.
- Team names are mapped to one standard name per club with the reviewed table
  data/reference/team_names.csv (see normalise_team_names).
"""

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.config import OPENFOOTBALL_RAW_DIR, REFERENCE_DIR

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
    ["season", "date", "kickoff_cet", "stage", "stage_raw",
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
        "kickoff_cet": kickoff,
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
                current_time = match["kickoff_cet"] or current_time
                match["kickoff_cet"] = current_time
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
        .sort_values(["date", "kickoff_cet"], kind="stable", na_position="last")
        .reset_index(drop=True)
    )


# --- Step 2: cleaning decisions from EDA --------------------------------------

KNOCKOUT_STAGES = ["playoff", "round_of_16", "quarterfinal", "semifinal", "final"]
UEFA_TIMEZONE = "Europe/Paris"  # CET/CEST, the zone UEFA lists kickoff times in
# Country code -> UEFA association. openfootball switched AS Monaco from FRA to MCO in
# 2024-25; Monaco plays in the French league and enters through France's association.
ASSOCIATION_CODES = {"MCO": "FRA"}


def normalise_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace country codes with UEFA association codes (see ASSOCIATION_CODES)."""
    out = df.copy()
    for column in ("home_country", "away_country"):
        out[column] = out[column].replace(ASSOCIATION_CODES)
    return out


def add_kickoff_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Add `kickoff_utc` (timezone-aware) from `date` + `kickoff_cet` (CET/CEST)."""
    out = df.copy()
    local = pd.to_datetime(out["date"].dt.strftime("%Y-%m-%d") + " " + out["kickoff_cet"])
    out["kickoff_utc"] = local.dt.tz_localize(UEFA_TIMEZONE).dt.tz_convert("UTC")
    return out


def add_ht_inferred(df: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing half-time score with 0-0 when the 90-minute score is 0-0.

    The source omits the half-time score only for goalless matches, where it cannot be
    anything but 0-0. `ht_inferred` marks the filled rows; other missing half-time
    scores (none at present) stay <NA>.
    """
    out = df.copy()
    inferable = out["home_ht"].isna() & out["away_ht"].isna() & (out["home_ft"] == 0) & (out["away_ft"] == 0)
    out.loc[inferable, ["home_ht", "away_ht"]] = 0
    out["ht_inferred"] = inferable.astype(bool)
    return out


def load_match_context(path: Path = REFERENCE_DIR / "match_context.csv") -> pd.DataFrame:
    """Read the reviewed venue/crowd table (one row per flagged match)."""
    context = pd.read_csv(path, parse_dates=["date"], keep_default_na=False)
    for column in ("neutral_venue", "played_abroad", "behind_closed_doors"):
        context[column] = context[column].astype(str).str.lower().eq("true")
    return context


def add_match_context(df: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Add venue and crowd flags from the reference table.

    Columns: `neutral_venue` (truly neutral: finals, Lisbon Final 8), `neutral_reason`,
    `played_abroad` (the home side relocated to another country) and
    `behind_closed_doors`. Matches not in the table get False / "". Raises if a
    reference row does not match exactly one match, so a typo in the table can never
    silently drop a flag. Uses the source team names, so run it before
    normalise_team_names.
    """
    key = ["season", "date", "home_team", "away_team"]
    columns = ["neutral_venue", "neutral_reason", "played_abroad", "behind_closed_doors"]
    if context.duplicated(subset=key).any():
        raise ValueError(f"Duplicate rows in match context: {context[context.duplicated(subset=key)][key].values.tolist()}")
    merged = df.merge(context[key + columns], on=key, how="left", indicator=True, validate="one_to_one")
    unmatched = context.merge(df[key], on=key, how="left", indicator=True)
    unmatched = unmatched[unmatched["_merge"] == "left_only"]
    if not unmatched.empty:
        raise ValueError(f"Match context rows match no match: {unmatched[key].values.tolist()}")
    for column in ("neutral_venue", "played_abroad", "behind_closed_doors"):
        merged[column] = merged[column].fillna(False).astype(bool)
    merged["neutral_reason"] = merged["neutral_reason"].fillna("")
    return merged.drop(columns="_merge")


def is_single_leg_stage(season: str, stage: str) -> bool:
    """True where a knockout stage is a single match: every final, and the 2019-20
    quarter- and semi-finals (Lisbon "Final 8")."""
    return stage == "final" or (season == "2019-20" and stage in ("quarterfinal", "semifinal"))


def add_ties(df: pd.DataFrame) -> pd.DataFrame:
    """Add `tie_id` and `leg` to knockout matches (both <NA> for group/league matches).

    A tie is the set of matches between the same two teams in the same season and
    stage. Legs are numbered by date. tie_id is "<season>_<stage>_<NN>", numbered in
    order of the first leg's date and home team.

    Raises:
        ValueError: If a two-legged stage has a lone match, a tie has more than two
            matches, or the second leg does not swap home and away.
    """
    out = df.copy()
    out["tie_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["leg"] = pd.Series(pd.NA, index=out.index, dtype="string")
    knockout = out[out["stage"].isin(KNOCKOUT_STAGES)]
    pair = knockout.apply(lambda r: tuple(sorted((r["home_team"], r["away_team"]))), axis=1)

    for (season, stage), stage_rows in knockout.groupby(["season", "stage"]):
        ties = stage_rows.groupby(pair.loc[stage_rows.index])
        ordered = sorted(ties, key=lambda t: (t[1]["date"].min(), t[1].sort_values("date")["home_team"].iloc[0]))
        for number, (teams, tie) in enumerate(ordered, start=1):
            tie = tie.sort_values(["date", "kickoff_cet"])
            tie_id = f"{season}_{stage}_{number:02d}"
            if len(tie) == 1:
                if not is_single_leg_stage(season, stage):
                    raise ValueError(f"{tie_id} {teams}: lone match in a two-legged stage")
                legs = ["single"]
            elif len(tie) == 2:
                first, second = tie.iloc[0], tie.iloc[1]
                if (first["home_team"], first["away_team"]) != (second["away_team"], second["home_team"]):
                    raise ValueError(f"{tie_id} {teams}: second leg does not swap home and away")
                legs = ["1", "2"]
            else:
                raise ValueError(f"{tie_id} {teams}: {len(tie)} matches in one tie")
            out.loc[tie.index, "tie_id"] = tie_id
            out.loc[tie.index, "leg"] = legs
    return out


def load_team_names(path: Path = REFERENCE_DIR / "team_names.csv", source: str = "openfootball") -> dict[str, str]:
    """Return {source name: standard name} for one source from the reviewed table."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = table[table["source"] == source]
    return dict(zip(rows["source_name"], rows["standard_name"]))


def normalise_team_names(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Replace home/away team names with standard names; keep the originals as *_source.

    Raises (via validate.check_teams_mapped) if any name is missing from the mapping,
    so a new spelling can never slip through as a "new club".
    """
    from src.data.validate import check_teams_mapped  # local import avoids a cycle

    check_teams_mapped(df, mapping)
    out = df.copy()
    for side in ("home", "away"):
        out[f"{side}_team_source"] = out[f"{side}_team"]
        out[f"{side}_team"] = out[f"{side}_team"].map(mapping)
    return out


# --- Full pipeline ---------------------------------------------------------------

PROCESSED_FILE = "matches.csv"


def clean_matches(
    raw_dir: Path = OPENFOOTBALL_RAW_DIR,
    reference_dir: Path = REFERENCE_DIR,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Run every cleaning step, validate the result, and optionally save it.

    Order matters: match context and ties are attached using the source team names,
    and names are standardised last.

    Args:
        out_dir: If given, the table is written to out_dir / "matches.csv".

    Returns:
        The validated match table, sorted chronologically.
    """
    from src.data.validate import validate_matches, validate_team_table  # avoids a cycle

    team_table = pd.read_csv(reference_dir / "team_names.csv", dtype=str, keep_default_na=False)
    validate_team_table(team_table)

    df = load_openfootball(raw_dir)
    df = normalise_country_codes(df)
    df = add_kickoff_utc(df)
    df = add_ht_inferred(df)
    df = add_match_context(df, load_match_context(reference_dir / "match_context.csv"))
    df = add_ties(df)
    df = normalise_team_names(df, load_team_names(reference_dir / "team_names.csv"))
    df = validate_matches(df)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / PROCESSED_FILE, index=False, encoding="utf-8")
    return df


def load_clean_matches(path: Path) -> pd.DataFrame:
    """Read a saved clean match table back with its original column types.

    CSV stores only text, so dates, nullable integers, booleans and empty strings are
    restored here; later milestones should load the table with this function.
    """
    text_columns = ["season", "kickoff_cet", "stage", "stage_raw", "home_team", "home_country",
                    "away_team", "away_country", "source_file", "neutral_reason",
                    "home_team_source", "away_team_source"]
    df = pd.read_csv(path, dtype={c: str for c in text_columns}, keep_default_na=False,
                     na_values={c: [""] for c in ("tie_id", "leg")})
    df["date"] = pd.to_datetime(df["date"])
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    score_cols = [c for c in COLUMNS if c.split("_")[-1] in SCORE_KINDS]
    df[score_cols] = df[score_cols].replace("", pd.NA).astype("Float64").astype("Int64")
    for column in ("ht_inferred", "neutral_venue", "played_abroad", "behind_closed_doors"):
        df[column] = df[column].astype(str).eq("True")
    for column in ("tie_id", "leg"):
        df[column] = df[column].astype("string")
    return df


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    from src.data.validate import validate_matches

    table = clean_matches(out_dir=PROCESSED_DATA_DIR)
    validate_matches(load_clean_matches(PROCESSED_DATA_DIR / PROCESSED_FILE))  # round trip
    clubs = pd.unique(table[["home_team", "away_team"]].values.ravel())
    print(f"Saved {PROCESSED_DATA_DIR / PROCESSED_FILE}: {len(table)} matches, "
          f"{len(clubs)} clubs, {table.shape[1]} columns")
