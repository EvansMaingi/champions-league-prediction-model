# Data notes: cleaning decisions and evidence

This file records every cleaning decision made in Milestone 3, why it was made, and the
real counts behind it. Code: `src/data/clean.py`, `src/data/validate.py`,
`src/data/wikipedia.py`, `src/data/reference.py`.

## Sources

| Source | Role | Licence | In git? |
|---|---|---|---|
| [openfootball/champions-league](https://github.com/openfootball/champions-league) | Main historical data, 2011-12 to 2025-26 (1,997 matches) | CC0-1.0 | Yes (`data/raw/openfootball/`) |
| [football-data.org](https://www.football-data.org) | Current-season fixtures | No published redistribution terms | No |
| [English Wikipedia](https://en.wikipedia.org) season, knockout and final articles | Second source for quality checks only (never a model input) | CC BY-SA 4.0 | Raw pages: no. Derived facts (stadium, attendance) in `data/reference/match_context.csv`, with revision IDs |

## Second-source check (openfootball vs Wikipedia)

**Matching.** Every openfootball match was paired with its Wikipedia "football box" by
date + home/away association codes, with name similarity only to break ties, and one
box per match. Scores are never used for matching, so the score check is not circular.

- 1,997 of 1,997 matches paired one-to-one.
- 2 pairs differ by one day: Galatasaray v Juventus (2013) and PSG v İstanbul
  Başakşehir (2020) were suspended and completed the next day. openfootball has the
  completion date and is kept, because that is when the result became known.

**Scores.** Wikipedia's score (after extra time when played) vs openfootball:

- Random sample of 50 matches (seed 42, all 15 seasons): **0 mismatches**.
- All 1,997 matches: **0 mismatches**.

## Kickoff timezone

openfootball's kickoff times are **Central European Time / Central European Summer Time
(CET/CEST)**, the times UEFA lists, not local venue time.

- Wikipedia's season articles state "Times are CET/CEST", and 1,992 of 1,997 kickoff
  times are identical. The 5 differences: the two suspended matches (openfootball gives
  the restart time), two kickoffs delayed by 5 and 20 minutes, and the delayed 2022 final
  (21:35 vs 21:36).
- football-data.org lists 2026-10-20 evening kickoffs as 19:00 UTC, which is 21:00 CEST.
- Columns: `kickoff_cet` (as written) and `kickoff_utc` (converted with Europe/Paris
  rules, so summer time is handled: 20:45 in May = 18:45 UTC, 20:45 in December = 19:45 UTC).

## Half-time scores

- 112 matches have no half-time score, and every one of them ended 0-0 after 90 minutes
  (there are exactly 112 goalless matches). The source omits a half-time score that can
  only be 0-0.
- Decision: fill 0-0 and set `ht_inferred = True` (112 rows). 0 half-time scores remain missing.
- **Leakage rule:** half-time scores are post-kickoff data. They may only be used in
  rolling features built from earlier matches, never as inputs for the match itself.

## Venue and crowd context (`data/reference/match_context.csv`, 142 rows)

| Column | Value | Evidence | Matches |
|---|---|---|---|
| `neutral_venue` / `neutral_reason` | `final` | Rule: finals are at a venue chosen in advance | 15 |
| | `lisbon_final_8` | Rule: 2019-20 quarter- and semi-finals were single matches in Lisbon | 6 |
| | `played_abroad` | Source data: Wikipedia's stadium field names a country, which it does only when the "home" side played abroad | 17 |
| | **total neutral** | | **38** |
| `behind_closed_doors` | `True` | Source data: Wikipedia attendance = 0 | **118** |
| both | | | 14 |

`played_abroad` (17):
- 2020-21, COVID travel restrictions (7): RB Leipzig v Liverpool and return leg (Budapest),
  Borussia Mönchengladbach v Manchester City and return leg (Budapest), Atlético Madrid v
  Chelsea (Bucharest), Porto v Chelsea and return leg (Seville).
- Shakhtar Donetsk home matches after the 2022 invasion of Ukraine (10): Warsaw (3, 2022-23),
  Hamburg (3, 2023-24), Gelsenkirchen (4, 2024-25). Same definition as the 2020-21 cases,
  so flagged for consistency.
- Not flagged: relocations *within* the home country (e.g. Tottenham at Wembley,
  Shakhtar in Lviv/Kharkiv/Kyiv 2014-2021, Barcelona at the Olympic stadium).

`behind_closed_doors` (118) by period:
- March 2020: 2 of 4 (Leipzig v Tottenham and Liverpool v Atlético still had full crowds).
- August 2020 (remaining round of 16 and Lisbon): 11 of 11.
- 2020-21: 96 of 125. 28 group matches had partial crowds (132 to 18,531) and the final
  had 14,110, so a date-window rule would have mislabelled 30 matches.
- Outside COVID (9): CSKA Moscow x3 (2014), Legia Warsaw (2016), Lyon (2018), most
  likely UEFA stadium closures; and four matches in December 2021, most likely Omicron
  restrictions in Germany, the Netherlands and Austria. (The causes are not stored in the
  data; the zero attendance is.)

Note: the 2012 final (Bayern v Chelsea) is neutral by rule but was played at Bayern's own
Allianz Arena (recorded in the `note` column).

`clean.add_match_context` raises an error if any reference row matches no match.

## Two-legged ties

- Knockout matches (play-offs, round of 16, quarter-finals, semi-finals, finals) get
  `tie_id` (`<season>_<stage>_<NN>`) and `leg` (`"1"`, `"2"`, `"single"`).
- 461 knockout matches -> **241 ties**: 220 two-legged + 21 single (15 finals, 2019-20
  quarter-finals x4 and semi-finals x2).
- `add_ties` raises if a two-legged stage has a lone match, a tie has more than two
  matches, or the second leg does not swap home and away.
- `validate.check_shootouts`: a shootout must follow a level single match, or a level
  **aggregate** in a second leg. (A first version required a level second-leg score and
  wrongly rejected 7 real shootouts, e.g. Porto 1-0 Arsenal, Arsenal 1-0 Porto, 2024.)

## Country codes

openfootball switched AS Monaco from `FRA` (2014-15 to 2018-19) to `MCO` (from 2024-25).
Country codes are treated as UEFA associations, so `MCO` -> `FRA`.

## Team names (`data/reference/team_names.csv`, 172 rows) - proposed, under review

- **Identity:** a club is identified by the Wikipedia article its name links to. openfootball
  names are linked through the matched boxes (date + countries); redirect titles are
  resolved (e.g. `R.S.C. Anderlecht` = `RSC Anderlecht`). football-data.org teams are linked
  by fixture schedule (dates + home/away of all 8 league-phase matches): 36 of 36 matched
  with similarity 1.0.
- **Standard name convention:** the article's most recent Wikipedia display name, i.e. the
  common English name (e.g. `Bayern Munich`, `Inter Milan`, `Red Star Belgrade`).
- **Result:** 136 openfootball names -> 108 clubs (28 clubs had two spellings, mostly from
  openfootball's switch to official names around 2023-24); plus 6 clubs in 2026-27 with no
  history (Como, Fenerbahçe, LASK, Real Betis, Sabah, Viking), to be flagged as unknown
  teams at prediction time.
- String similarity is not used to decide identity: it scores real variants as low as 0.62
  and different clubs as high as 0.81.
- `validate.check_teams_mapped` fails on any name missing from the table.
