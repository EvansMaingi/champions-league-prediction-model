"""Tests for src/data/wikipedia.py (second-source parsing and matching) and
src/data/reference.py (building the reviewed reference tables).

All inputs are small hand-written wikitext snippets and DataFrames.
"""

import json

import pandas as pd
import pytest

from src.data import reference, wikipedia

BOX = """{{Football box
|date       = {{Start date|2021|2|24|df=y}}
|time       = 21:00
|team1      = [[Borussia Mönchengladbach]] {{fbaicon|GER}}
|score      = 0–2
|team2      = {{fbaicon|ENG}} [[Manchester City F.C.|Manchester City]]
|stadium    = [[Puskás Aréna]], [[Budapest]], Hungary{{refn|group=note|Moved due to travel {{nowrap|restrictions}}.}}
|attendance = 0
}}"""


def test_find_boxes_handles_nested_templates():
    text = "Intro {{other}}\n" + BOX + "\nmiddle\n" + BOX.replace("0–2", "1–1")
    boxes = wikipedia.find_boxes(text)
    assert len(boxes) == 2
    assert boxes[0].endswith("}}") and "1–1" in boxes[1]


def test_split_params_ignores_pipes_inside_links_and_templates():
    params = wikipedia.split_params(BOX)
    assert params["team2"] == "{{fbaicon|ENG}} [[Manchester City F.C.|Manchester City]]"
    assert params["attendance"] == "0"


def test_plain_text_strips_markup():
    assert wikipedia.plain_text("[[Manchester City F.C.|Manchester City]] {{fbaicon|ENG}}") == "Manchester City"
    assert wikipedia.plain_text("[[Puskás Aréna]], Budapest{{refn|a {{b}} c}}<ref>x</ref>") == "Puskás Aréna, Budapest"


@pytest.mark.parametrize("value, code", [
    ("[[A]] {{fbaicon|GER}}", "GER"),
    ("{{fbaicon|FRA|1974}} [[B]]", "FRA"),
    ("[[C]] {{#invoke:flag|fbaicon|SUI}}", "SUI"),
    ("[[D]]", None),
])
def test_flag_country_forms(value, code):
    assert wikipedia.flag_country(value) == code


def test_link_target_returns_article_not_display_name():
    assert wikipedia.link_target("[[FC Bayern Munich|Bayern Munich]] {{fbaicon|GER}}") == "FC Bayern Munich"
    assert wikipedia.link_target("Bayern Munich {{fbaicon|GER}}") is None


def test_parse_box_extracts_match_record():
    row = wikipedia.parse_box(BOX)
    assert row["date"] == pd.Timestamp(2021, 2, 24) and row["kickoff_cet"] == "21:00"
    assert (row["team1"], row["team1_country"], row["team1_article"]) == (
        "Borussia Mönchengladbach", "GER", "Borussia Mönchengladbach")
    assert (row["team2"], row["team2_article"]) == ("Manchester City", "Manchester City F.C.")
    assert (row["goals1"], row["goals2"], row["attendance"]) == (0, 2, 0)
    assert row["stadium"] == "Puskás Aréna, Budapest, Hungary"


def wiki_row(date, team1, country1, team2, country2, goals1=0, goals2=0):
    return {"date": pd.Timestamp(date), "team1": team1, "team1_country": country1,
            "team2": team2, "team2_country": country2, "goals1": goals1, "goals2": goals2}


def of_row(date, home, home_country, away, away_country):
    return {"date": pd.Timestamp(date), "home_team": home, "home_country": home_country,
            "away_team": away, "away_country": away_country}


def test_match_uses_countries_and_names_not_scores():
    of = pd.DataFrame([of_row("2020-10-21", "Real Madrid", "ESP", "Shakhtar", "UKR"),
                       of_row("2020-10-21", "Atlético Madrid", "ESP", "Bayern", "GER")])
    wiki = pd.DataFrame([wiki_row("2020-10-21", "Atlético Madrid", "ESP", "Bayern Munich", "GER", 9, 9),
                         wiki_row("2020-10-21", "Real Madrid", "ESP", "Shakhtar Donetsk", "UKR", 2, 3)])
    matched = wikipedia.match_to_openfootball(of, wiki)
    assert matched["wp_team2"].tolist() == ["Shakhtar Donetsk", "Bayern Munich"]


def test_match_maps_monaco_and_allows_one_day_for_suspended_matches():
    of = pd.DataFrame([of_row("2013-12-11", "Galatasaray", "TUR", "Juventus", "ITA"),
                       of_row("2024-09-19", "AS Monaco FC", "MCO", "FC Barcelona", "ESP")])
    wiki = pd.DataFrame([wiki_row("2013-12-10", "Galatasaray", "TUR", "Juventus", "ITA"),
                         wiki_row("2024-09-19", "Monaco", "FRA", "Barcelona", "ESP")])
    matched = wikipedia.match_to_openfootball(of, wiki)
    assert matched["wp_team1"].tolist() == ["Galatasaray", "Monaco"]
    assert matched["date_diff_days"].tolist() == [-1, 0]


def test_match_uses_each_box_once():
    of = pd.DataFrame([of_row("2020-10-21", "A", "ESP", "B", "GER"),
                       of_row("2020-10-21", "A2", "ESP", "B2", "GER")])
    wiki = pd.DataFrame([wiki_row("2020-10-21", "A", "ESP", "B", "GER")])
    matched = wikipedia.match_to_openfootball(of, wiki)
    assert matched["wp_team1"].notna().sum() == 1


# --- reference.py ----------------------------------------------------------------


def test_resolve_titles_follows_normalisation_and_redirects(tmp_path):
    path = tmp_path / "titles.json"
    path.write_text(json.dumps([{"query": {
        "normalized": [{"from": "rSC Anderlecht", "to": "RSC Anderlecht"}],
        "redirects": [{"from": "R.S.C. Anderlecht", "to": "RSC Anderlecht"}],
        "pages": [{"title": "RSC Anderlecht"}],
    }}]), encoding="utf-8")
    resolved = reference.resolve_titles(path)
    assert resolved["R.S.C. Anderlecht"] == resolved["rSC Anderlecht"] == "RSC Anderlecht"


def fixture(day, home_id, away_id):
    return {"utcDate": f"{day}T19:00:00Z", "homeTeam": {"id": home_id, "name": f"T{home_id}"},
            "awayTeam": {"id": away_id, "name": f"T{away_id}"}}


def test_match_fixture_teams_by_schedule_not_name():
    fixtures = [fixture("2026-09-08", 1, 2), fixture("2026-09-30", 2, 1), fixture("2026-10-21", 1, 3)]
    wiki = pd.DataFrame([
        {"date": pd.Timestamp("2026-09-08"), "team1_article": "Club One", "team2_article": "Club Two"},
        {"date": pd.Timestamp("2026-09-30"), "team1_article": "Club Two", "team2_article": "Club One"},
        {"date": pd.Timestamp("2026-10-21"), "team1_article": "Club One", "team2_article": "Club Three"},
    ])
    matched = reference.match_fixture_teams(fixtures, wiki)
    assert matched[1] == ("Club One", 1.0) and matched[2] == ("Club Two", 1.0)
    assert matched[3][0] == "Club Three"


def test_build_match_context_reasons_and_evidence():
    base = {"home_team": "A", "away_team": "B", "wp_revid": 1}
    matched = pd.DataFrame([
        {**base, "season": "2015-16", "date": pd.Timestamp("2016-05-28"), "stage": "final",
         "wp_stadium": "San Siro, Milan", "wp_attendance": 71942},
        {**base, "season": "2019-20", "date": pd.Timestamp("2020-08-12"), "stage": "quarterfinal",
         "wp_stadium": "Estádio da Luz, Lisbon", "wp_attendance": 0},
        {**base, "season": "2020-21", "date": pd.Timestamp("2021-02-16"), "stage": "round_of_16",
         "wp_stadium": "Puskás Aréna, Budapest, Hungary", "wp_attendance": 0},
        {**base, "season": "2020-21", "date": pd.Timestamp("2020-10-21"), "stage": "group",
         "wp_stadium": "Anfield, Liverpool", "wp_attendance": 0},
        {**base, "season": "2021-22", "date": pd.Timestamp("2021-09-14"), "stage": "group",
         "wp_stadium": "Anfield, Liverpool", "wp_attendance": 52000},
    ])
    context = reference.build_match_context(matched)
    assert len(context) == 4  # the ordinary 2021-22 match is not listed
    assert context["neutral_venue"].tolist() == [True, True, False, False]
    assert context["neutral_reason"].tolist() == ["final", "lisbon_final_8", "", ""]
    assert context["neutral_evidence"].tolist() == ["rule", "rule", "", ""]
    assert context["played_abroad"].tolist() == [False, False, True, False]
    assert context["abroad_evidence"].tolist() == ["", "", "Wikipedia stadium field", ""]
    assert context["behind_closed_doors"].tolist() == [False, True, True, True]


@pytest.mark.parametrize("title, standard", [
    ("FC Bayern Munich", "Bayern Munich"),
    ("Arsenal F.C.", "Arsenal"),
    ("Galatasaray S.K. (football)", "Galatasaray"),
    ("Sabah FK (Azerbaijan)", "Sabah"),
    ("Bologna FC 1909", "Bologna"),
    ("Como 1907", "Como"),
    ("1. FC Union Berlin", "Union Berlin"),
    ("AS Monaco FC", "Monaco"),
    ("FCSB", "FCSB"),
    ("RB Leipzig", "RB Leipzig"),            # RB is part of the name
    ("PSV Eindhoven", "PSV Eindhoven"),
    ("FC Schalke 04", "Schalke 04"),          # "04" is not a founding year
    ("TSG 1899 Hoffenheim", "TSG 1899 Hoffenheim"),
    ("Ferencvárosi TC", "Ferencvárosi TC"),  # TC deliberately kept
    ("Inter Milan", "Inter Milan"),
])
def test_standard_name_from_title(title, standard):
    assert reference.standard_name_from_title(title) == standard
