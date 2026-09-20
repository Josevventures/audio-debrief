import json
from pathlib import Path

from audio_debrief import nouns

FIX = Path(__file__).parent / "fixtures"


def load():
    rec = (FIX / "recorder.txt").read_text(encoding="utf-8")
    ww = json.loads((FIX / "whisper_words.json").read_text(encoding="utf-8"))
    return rec, ww


def by_pair(rows):
    return {(r["recorder_spelling"], r["whisper_spelling"]): r for r in rows}


def test_pairs_known_errors():
    rec, ww = load()
    rows = by_pair(nouns.noun_corrections(rec, ww, hotwords=[]))
    assert ("Alec", "Alex") in rows
    assert ("bramble", "Bramwell") in rows
    assert ("Kerbao", "carve-out") in rows


def test_counts_and_first_timestamp():
    rec, ww = load()
    rows = by_pair(nouns.noun_corrections(rec, ww, hotwords=[]))
    assert rows[("Alec", "Alex")]["count"] == 2
    assert rows[("Alec", "Alex")]["first_timestamp"] == "00:00"
    assert rows[("bramble", "Bramwell")]["first_timestamp"] == "01:05"


def test_hotwords_preconfirm():
    rec, ww = load()
    rows = by_pair(nouns.noun_corrections(rec, ww, hotwords=["Alex Rivera", "carve-out"]))
    assert rows[("Alec", "Alex")]["confirmed"] == "yes"
    assert rows[("Kerbao", "carve-out")]["confirmed"] == "yes"
    assert rows[("bramble", "Bramwell")]["confirmed"] == ""


def test_sentence_starts_and_matching_tokens_excluded():
    rec, ww = load()
    rows = nouns.noun_corrections(rec, ww, hotwords=[])
    seen = {r["recorder_spelling"] for r in rows} | {r["whisper_spelling"] for r in rows}
    for tok in ("Hello", "Can", "I", "We", "Then", "Halcyon", "Meridian", "MBA"):
        assert tok not in seen


def test_multiword_hotword_merges_and_hyphen_normalizes():
    rec = "I met Alex Rivera and we did a carve out."
    ww = [{"text": t, "start": i * 0.3, "end": i * 0.3 + 0.2}
          for i, t in enumerate("I met Alex Rivera and we did a carve-out.".split())]
    assert nouns.noun_corrections(rec, ww, hotwords=["Alex Rivera", "carve-out"]) == []


HOT = ["Sam Carter", "Alex Rivera", "Northfield", "Bramwell", "MBA", "AI", "Meridian", "carve-out"]


def pairs(rec, whi, hotwords=HOT):
    ww = [{"text": t, "start": 60.0 + i * 0.3, "end": 60.0 + i * 0.3 + 0.2} for i, t in enumerate(whi.split())]
    return {(r["recorder_spelling"], r["whisper_spelling"]): r for r in nouns.noun_corrections(rec, ww, hotwords)}


def test_junk_pairs_are_dropped():
    assert pairs("I think Disney is fine now.", "I think this is fine now.") == {}
    assert pairs("Then um I left early.", "Then I I left early.") == {}
    assert pairs("Then we did it.", "Then I did it.") == {}
    assert pairs("Then I've done it.", "Then I done it.") == {}
    assert pairs("Based in the US now.", "Based in the U now.") == {}
    assert pairs("That is no problem.", "That is a problem.") == {}


def test_single_letters_survive_only_against_hotwords():
    rows = pairs("I have an M from Bramwell.", "I have an MBA from Bramwell.")
    assert rows[("M", "MBA")]["direction"] == "recorder→whisper" and rows[("M", "MBA")]["confirmed"] == "yes"
    rows = pairs("We used the A team there.", "We used the AI team there.")
    assert ("A", "AI") in rows
    assert pairs("We used the A team there.", "We used the B team there.") == {}


def test_reverse_direction_when_recorder_is_right():
    rows = pairs("I went to Northfield last week.", "I went to northfeld last week.")
    r = rows[("Northfield", "northfeld")]
    assert r["direction"] == "whisper→recorder" and r["confirmed"] == "yes"
    rows = pairs("We used Argent for that work.", "We used archit for that work.")
    r = rows[("Argent", "archit")]
    assert r["direction"] == "whisper→recorder" and r["confirmed"] == ""


def test_forward_direction_and_hotword_beats_stoplist():
    rows = pairs("I went to bramble for my MBA.", "I went to Bramwell for my MBA.")
    assert rows[("bramble", "Bramwell")]["direction"] == "recorder→whisper"
    rows = pairs("Say, how are you?", "Sam, how are you?")  # "say" is stoplisted, but Sam is a hotword
    assert rows[("Say", "Sam")]["direction"] == "recorder→whisper"
    rows = pairs("Hello, Alec. Nice to meet you.", "Hello, Alex. Nice to meet you.", hotwords=[])
    assert rows[("Alec", "Alex")]["direction"] == "unclear"


def test_parse_qa_questions():
    qs = nouns.parse_qa_questions(FIX / "qa_prep.md")
    assert qs == ["Walk me through your background.", "Why Northfield?",
                  "Tell me about a time you worked on a team under pressure."]


def test_parse_interviewer_names(tmp_path):
    d = tmp_path / "Northfield"
    d.mkdir()
    (d / "interviewer_alex_rivera.md").write_text("# Interviewer Profile — Alex Rivera\n**Role:** x\n", encoding="utf-8")
    (d / "interviewer_jane_doe.md").write_text("# Interviewer Profile - Jane Doe\n", encoding="utf-8")
    assert nouns.parse_interviewer_names(d) == ["Alex Rivera", "Jane Doe"]
    assert nouns.parse_interviewer_names(tmp_path) == []
