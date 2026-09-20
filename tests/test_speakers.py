import numpy as np

from audio_debrief import speakers


def test_clean_segments_skips_overlaps_and_caps_total():
    turns = [
        {"speaker": "S1", "start": 0, "end": 40},
        {"speaker": "S1", "start": 50, "end": 80},    # 30 s
        {"speaker": "S1", "start": 90, "end": 92},    # touches overlap -> excluded
        {"speaker": "S1", "start": 100, "end": 100.5},  # too short
        {"speaker": "S0", "start": 40, "end": 50},
    ]
    segs = speakers.clean_segments(turns, [{"start": 91, "end": 91.5}], "S1", max_total_s=60)
    assert [(s["start"], s["end"]) for s in segs] == [(0, 40), (50, 80)]


def test_match_self_threshold():
    vp = np.array([1.0, 0.0, 0.0])
    embs = {"S0": np.array([0.0, 1.0, 0.0]), "S1": np.array([0.9, 0.1, 0.0])}
    label, scores = speakers.match_self(embs, vp, 0.6)
    assert label == "S1" and scores["S1"] > 0.9
    assert speakers.match_self(embs, np.array([0.0, 0.0, 1.0]), 0.6)[0] is None


def test_name_speakers_by_first_appearance():
    turns = [{"speaker": "S2", "start": 0, "end": 1}, {"speaker": "S0", "start": 2, "end": 3},
             {"speaker": "S1", "start": 4, "end": 5}]
    m = speakers.name_speakers(turns, "S0", ["Alex Rivera"], "Sam")
    assert m == {"S2": "Alex Rivera", "S0": "Sam", "S1": "Speaker 2"}
    m = speakers.name_speakers(turns, "S0", [])
    assert m["S0"] == "Me"  # default display name when none is configured
    m = speakers.name_speakers(turns, None, [])
    assert m == {"S2": "Speaker 1", "S0": "Speaker 2", "S1": "Speaker 3"}


def test_voiceprint_path_defaults_to_self():
    assert speakers.voiceprint_path().name == "self.npy"
    assert speakers.voiceprint_path("alt").name == "alt.npy"


def test_sample_text_first_ten_seconds():
    words = [{"text": "a", "start": 3.0, "speaker": "S0"}, {"text": "b", "start": 12.9, "speaker": "S0"},
             {"text": "c", "start": 13.5, "speaker": "S0"}, {"text": "x", "start": 5.0, "speaker": "S1"}]
    assert speakers.sample_text(words, "S0") == "a b"
    assert speakers.sample_text(words, "S9") == ""
