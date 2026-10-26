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

Two separate venue flags: `neutral_venue` means *neither* side is at home; `played_abroad`
means the listed home side relocated to another country (the "away" side may still
be the stronger-supported one).

| Column | Value | Evidence | Matches |
|---|---|---|---|
| `neutral_venue` / `neutral_reason` | `final` | Rule: finals are at a venue chosen in advance | 15 |
| | `lisbon_final_8` | Rule: 2019-20 quarter- and semi-finals were single matches in Lisbon | 6 |
| | **total** | | **21** |
| `played_abroad` | `True` | Source data: Wikipedia's stadium field names a country, which it does only when the "home" side played abroad | **17** |
| `neutral_venue` and `played_abroad` | | | **0** |
| `behind_closed_doors` | `True` | Source data: Wikipedia attendance = 0 | **118** (7 also neutral, 7 also played abroad) |

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

## Team names (`data/reference/team_names.csv`, 172 rows) - reviewed and applied

**Identity.** A club is identified by the Wikipedia article its name links to, never by
string similarity (which scores real variants as low as 0.62 and different clubs as high
as 0.81).
- openfootball names are linked through the matched boxes (date + countries); redirect
  titles are resolved (e.g. `R.S.C. Anderlecht` = `RSC Anderlecht`).
- football-data.org teams are linked by fixture schedule (dates + home/away of all 8
  league-phase matches): 36 of 36 matched with similarity 1.0.

**Standard name rule** (`reference.standard_name_from_title`). Start from the resolved
article title and repeat until nothing changes:
1. drop a trailing disambiguator in parentheses: `Galatasaray S.K. (football)` -> `Galatasaray S.K.`;
2. drop a trailing founding year 1800-2099: `Bologna FC 1909` -> `Bologna FC`;
3. drop one of these club-type / legal-form tokens at the start and/or end, as a whole word:
   `1. FC`, `F.C.`, `FC`, `A.F.C.`, `AFC`, `C.F.`, `CF`, `S.C.`, `SC`, `S.K.`, `SK`, `ŠK`, `F.K.`,
   `FK`, `NK`, `GNK`, `PFC`, `BSC`, `KRC`, `KAA`, `RSC`, `RC`, `AC`, `AS`, `ASC`, `SS`, `SSC`,
   `S.L.`, `J.K.`, `BC`, `KV`, `OSC`, `HSC`, `FF`.

Nothing else is changed (spelling, accents, word order). Tokens that are part of a name
are not on the list (`RB`, `PSV`, `VfB`, `VfL`, `TSG`, `CFR`, `LASK`), and neither is `TC`:
`Ferencvárosi TC` without it would leave an adjective ("of Ferencváros").

Why titles: the article title is the club's *current* name. A first version used Wikipedia's
link text, which keeps the name a club had in the article's season: `FC Steaua Bucureşti`
(2013-14) links to the article **FCSB** (the club's name since 2017; "Steaua București" now
refers to a different club) but was displayed as "Steaua București". Switching to the
title rule changed 11 clubs (13 rows):

| Before (link text) | After (title rule) |
|---|---|
| Bayer Leverkusen | Bayer 04 Leverkusen |
| Steaua București | FCSB |
| Ferencváros | Ferencvárosi TC |
| Malmö FF | Malmö |
| Marseille | Olympique de Marseille |
| Lyon | Olympique Lyonnais |
| Antwerp | Royal Antwerp |
| Union Saint-Gilloise | Royale Union Saint-Gilloise |
| Brest | Stade Brestois 29 |
| Rennes | Stade Rennais |
| TSG Hoffenheim | TSG 1899 Hoffenheim |

**Countries.** Every row has a UEFA association code (`validate.validate_team_table` fails
otherwise). football-data.org rows take the club's openfootball country; the 6 clubs with
no history were entered by hand: Como ITA, Fenerbahçe TUR, LASK AUT, Real Betis ESP,
Sabah AZE, Viking NOR.

**Result.** 136 openfootball names -> 108 clubs (28 had two spellings, mostly from
openfootball's switch to official names around 2023-24), plus 6 clubs in 2026-27 with no
history, to be flagged as unknown teams at prediction time.

**Checks.** `validate.check_teams_mapped` fails on any name missing from the table;
`validate.validate_team_table` fails if a row lacks a standard name or country, or if a
club has more than one country or article, or an article more than one standard name.

## Processed output (`data/processed/matches.csv`)

`python -m src.data.clean` runs: parse -> association codes -> `kickoff_utc` ->
`ht_inferred` -> match context -> ties -> team names -> `validate.validate_matches`
(required columns, no duplicates, chronological order, score consistency, shootouts),
saves the table, then reloads it with `clean.load_clean_matches` and validates it again.

Result: 1,997 matches, 108 clubs, 28 columns; original spellings kept in
`home_team_source` / `away_team_source`.

## Features (Milestone 5)

### Part A: recent form (`build_features.add_form_features`)

- Per-team log (one row per team per match); `shift(1)` + rolling mean over the previous
  `FORM_WINDOW` = 5 CL matches: `form_points`, `form_goals_for`, `form_goals_against`,
  plus `form_n` (matches in the window) and `form_n_season` (how many are from this season).
- **Across seasons: carry over** (option 3, chosen 2026-10-25). Compared with resetting each
  season: 108 instead of 488 team-rows without history, and 22% instead of 100% of teams'
  first matches of a season without history; correlation with that match's points 0.197
  vs 0.190. Staleness is visible to the model through `form_n_season` (median gap before a
  team's first match of a season: 279 days, within a season: 15).
- **Cold start:** no history gives missing form values and `*_has_history = False`
  (50 home and 58 away values = each of the 108 clubs' first CL match).
- **Leakage tests:** truncation (removing later matches changes no earlier feature) and
  perturbation (changing a score changes only later matches' features). A deliberately
  leaky no-shift version passes the truncation test but correlates 0.565 with its own
  match's points (honest version: 0.197), so both tests are needed.

### Part B: Elo rating (`build_features.add_elo_features`)

- Champions-League-only Elo in kickoff order; features are the ratings **before** each
  match: `home_elo`, `away_elo`, `elo_diff` (home minus away), `home_elo_n`, `away_elo_n`
  (matches rated before this one). Expected score
  E = 1 / (1 + 10^((R_away - (R_home + HFA)) / 400)); update R += K (S - E), zero-sum.
- Parameters fixed in advance, not fitted to the data: start 1500, K 20, home advantage 60
  (0 when `neutral_venue` or `played_abroad`), and ratings move 1/3 of the way back to 1500
  between seasons. Milestone 7 may tune them on training/validation seasons only.
- Distribution: ratings 1,419 to 1,641 (sd 39); `elo_diff` -164 to +170; no missing values
  (a newcomer's rating is the 1500 default; `*_elo_n` = 0 and `*_has_history` show that).
- Leakage tests: truncation at 3 cut-offs identical; perturbing one score changes only the
  2 later matches involving those teams.
- **Burn-in:** end-of-season rating spread (sd) 30.4 in 2011-12, 36.1, 41.7, 44.2, then
  47.0 by 2015-16 and 47-52 afterwards. Rank order settles faster: Spearman correlation of
  consecutive end-of-season ratings 0.68 for 2011-12 -> 2012-13, then 0.81-0.93. So after
  the 2011-12 warm-up the ordering is reasonable, but the scale is compressed until about
  2015-16: an Elo gap means a little less in the early training seasons.
- **Newcomers are overrated:** for the 76 clubs whose first season in the data is after
  2011-12, Elo expected 0.473 points-share per match but they scored 0.360 (established
  clubs: expected 0.505, actual 0.524). Starting at the pool average flatters small clubs.
- Both findings were measured on all seasons, so they are **not** used to change the
  parameters now.
- **Known limitation: scale drift.** Because the spread is compressed until about 2015-16,
  the same `elo_diff` corresponds to a slightly larger real strength gap in the early
  training seasons than later. Models will see this as mild drift in the feature.
- **Milestone 7 tuning plan (agreed 2026-10-25).** Small grid: K in {15, 20, 30} x newcomer
  start in {1500, empirical}. The tuning score is computed on **validation seasons only**;
  the test seasons are never used.
- **Newcomers, two options to compare in Milestone 7:**
  1. keep 1500 and check whether the model learns the correction from `*_elo_n = 0` and
     `*_has_history = False`;
  2. if it does not, an empirical newcomer start: the average rating newcomers end their
     first season at, measured on **training seasons only**.
- External ratings (e.g. ClubElo, which also uses domestic matches) are an option, not
  used until their licence and the timing of their dates (a rating published after a
  match must never be used for it) have been checked.

### Part C: context features (`build_features.add_context_features`)

| Feature | Known before kickoff? | Decision |
|---|---|---|
| `neutral_venue`, `played_abroad` | Yes (competition rules; relocations announced in advance) | Included |
| `behind_closed_doors` | Yes (crowd rules announced in advance) | Included; rare (118, 5.9%) and False for all 2026-27 matches |
| `stage` | Yes | Included; one-hot encoded in the Milestone 7 Pipeline |
| `leg_type` | Yes | Included: "1", "2", "single", "none" (original `leg` kept unchanged) |
| `first_leg_goal_diff`, `first_leg_away_goals` | Yes: leg 1 always finished at least 5 days earlier (median 8; 164 days in 2019-20) | Included for the 220 second legs, missing otherwise |
| `away_goals_rule` | Yes (competition rule, up to 2020-21) | Included: True for 134 second legs |
| Rest days | Only partly: domestic matches in between are not in the data | **Dropped**: home and away rest equal in 83% of matches and within 1 day in 97% (a Tuesday/Wednesday calendar effect); the season-start gap is covered by `form_n_season` |
| Head-to-head | Yes | **Dropped**: 42% of matches are a pair's first meeting; 98% of single earlier meetings were the same-season reverse fixture, already in both form windows; only 290 matches have a pair that met in an earlier season, dominated by a few big-club pairs (Man City v Real Madrid: 17 meetings) |

- **Away goals encoding:** raw facts plus a flag (option 2 of 3). `first_leg_away_goals`
  matters only when `away_goals_rule` is True; tree models can learn that interaction,
  and an explicit interaction term can be added for logistic regression in Milestone 7.
- **Value is tested later:** Milestones 7-8 compare validation log loss with and without
  each feature group (ablation). Second-leg features affect only 220 matches, so expect
  wide confidence intervals.

### Final Milestone 5 feature table (`data/processed/features.csv`)

- 1,997 matches; identifiers (`season`, `date`, `kickoff_utc`, `stage`, `tie_id`,
  `home_team`, `away_team`), the target `result`, and **25 allowed features**:
  12 form, 5 Elo, 8 context (`ALLOWED_FEATURES`). Load with `build_features.load_feature_table`.
- **Not allowed:** `season` (per-season memorisation, useless for a new season); team names
  and countries (identity; a Milestone 7 experiment with unknown-team handling);
  `stage_raw`, `neutral_reason` (redundant); rest days and head-to-head (dropped above).
- Missing values: form 50 home / 58 away (cold starts); first-leg features 1,777 (not a
  second leg). Nothing else is missing.
- Leakage tests on the complete table: truncation at 3 cut-offs identical for all 25
  features; perturbing a first leg changes no earlier row, and changes the second leg's
  first-leg features; the largest absolute correlation of any single feature with the
  home side's points is 0.395 (`elo_diff`), so no feature mirrors the target.

## Chronological split and baselines (Milestone 6)

Whole seasons, set in `src/config.py` and applied by `src/training/split.py`
(`chronological_split` raises if a season has no role, two roles, or if the sets overlap
in time):

| Set | Seasons | Matches | Dates | H / D / A |
|---|---|---|---|---|
| Warm-up (history only) | 2011-12 | 125 | | |
| Train | 2012-13 to 2023-24 | 1,494 | 2012-09-18 to 2024-06-01 | 46.7 / 21.2 / 32.1% |
| Validation | 2024-25 | 189 | 2024-09-17 to 2025-05-31 | 51.3 / 13.8 / 34.9% |
| Test (sealed until end of M8) | 2025-26 | 189 | 2025-09-16 to 2026-05-30 | not reported |

- **Why:** validation and test are both 36-team-format seasons, so model choices are judged
  on matches like the ones we will predict. Cost: 189 validation matches is small;
  Milestone 8 adds bootstrap intervals. **Fallback if too noisy in Milestone 7:**
  walk-forward validation (train to season N, score N+1, over a few seasons), with 2025-26
  still untouched.
- **Training data cutoff:** 2024-06-01 for model selection. Milestone 9 refits on 2012-13
  to 2025-26 to predict 2026-27 and records that cutoff.
- **Validation baselines** (fitted on train, `python -m src.training.train`):

  | Baseline | Log loss | Brier | Accuracy | mean p(D) |
  |---|---|---|---|---|
  | prior (training shares) | 1.0011 | 0.6042 | 0.5132 | 0.212 |
  | uniform (1/3 each) | 1.0986 | 0.6667 | 0.5132 | 0.333 |
  | always home | 17.5451 | 0.9735 | 0.5132 | 0.000 |

  All three have the same accuracy (every argmax is H; uniform ties go to H), so accuracy
  cannot rank them; log loss and Brier can. Always-home's log loss is the clipped cost
  (about 36 per match) of the 92 non-home results it called impossible.
- **Draw-rate shift (known limitation):** the training draw rate is 21.2%, validation 13.8%.
  No feature can learn the new format's lower draw rate (`stage` maps the league phase
  into `group_or_league`, and a format flag would be constant in training), so expect
  draws to be over-predicted on validation and test. The prior baseline already shows it
  (mean p(D) 0.212 vs 0.138). Milestone 8 reports per-class calibration, especially D.
  The Milestone 9 refit, which includes two new-format seasons, partly corrects it.
- **Column order:** scikit-learn's `log_loss` and `brier_score_loss` assume probability
  columns are in sorted label order (A, D, H) and do not reorder them to match `labels`.
  `evaluate.predict_proba_frame` names columns by `model.classes_` and returns H, D, A;
  `evaluate_probabilities` sorts them before calling scikit-learn. A test pins this
  (the wrong order gave 1.7533 instead of 0.8574 on the hand example).

## Decisions for later milestones

- **Milestone 6: 2011-12 is a warm-up season.** The data starts in 2011-12, so every club's
  first match that season looks like a cold start even when it had years of CL history
  (left-censoring): 16 of the 84 matches with a side lacking history are in 2011-12, the
  most of any season, and so are 16 of the 24 matches where *neither* side has history
  (every other season has at most 2). Use 2011-12 to build
  history (form, Elo) but exclude it from training and evaluation.
- **Milestone 7: missing form values** (cold starts) are imputed inside the scikit-learn
  Pipeline, fitted on training data only, and the `*_has_history` flags are kept so the
  model knows a value was imputed. `HistGradientBoostingClassifier` handles NaN natively
  and needs no imputation.
- **Milestone 9: venue flags for upcoming fixtures.** football-data.org matches have no venue
  field, so `played_abroad` (e.g. Shakhtar Donetsk, which has played home matches abroad
  since 2022 and is in 2026-27) and `neutral_venue` must come from a verified source, not
  the fixture feed; if unverified, the prediction is flagged.
