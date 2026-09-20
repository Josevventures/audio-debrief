from audio_debrief import align

TURNS = [
    {"speaker": "S0", "start": 0.0, "end": 2.5},
    {"speaker": "S1", "start": 4.0, "end": 7.0},
]


def w(text, start, end):
    return {"text": text, "start": start, "end": end, "prob": 0.9}


def test_word_assigned_by_midpoint():
    words = [w("a", 0.0, 1.0), w("b", 1.2, 2.0), w("c", 5.0, 6.0)]
    out = align.assign_speakers(words, TURNS)
    assert [x["speaker"] for x in out] == ["S0", "S0", "S1"]


def test_uncovered_word_goes_to_nearest_turn():
    # midpoint 3.1: 0.6 s from S0's end, 0.9 s from S1's start -> S0
    out = align.assign_speakers([w("x", 3.0, 3.2)], TURNS)
    assert out[0]["speaker"] == "S0"
    # midpoint 3.6: closer to S1
    out = align.assign_speakers([w("y", 3.5, 3.7)], TURNS)
    assert out[0]["speaker"] == "S1"


def test_word_covered_by_two_turns_prefers_larger_overlap():
    turns = [{"speaker": "A", "start": 0.0, "end": 5.0}, {"speaker": "B", "start": 4.0, "end": 9.0}]
    out = align.assign_speakers([w("z", 4.2, 5.4)], turns)  # 0.8 s inside A, 1.2 s inside B
    assert out[0]["speaker"] == "B"


def test_consecutive_same_speaker_words_merge():
    words = [w("hello", 0.0, 0.5), w("there", 0.6, 1.0), w("friend", 5.0, 5.5)]
    words = align.assign_speakers(words, TURNS)
    utts = align.build_utterances(words)
    assert len(utts) == 2
    assert utts[0]["speaker"] == "S0" and utts[0]["text"] == "hello there"
    assert utts[0]["start"] == 0.0 and utts[0]["end"] == 1.0
    assert len(utts[0]["words"]) == 2
    assert utts[1]["speaker"] == "S1" and utts[1]["text"] == "friend"


def test_utterance_splits_on_gap_over_threshold():
    words = [w("a", 0.0, 0.5), w("b", 0.6, 1.0), w("c", 2.6, 3.0)]  # gap b->c = 1.6 s
    for x in words:
        x["speaker"] = "S0"
    utts = align.build_utterances(words, max_gap=1.5)
    assert [u["text"] for u in utts] == ["a b", "c"]
    utts = align.build_utterances(words, max_gap=2.0)
    assert [u["text"] for u in utts] == ["a b c"]


def test_align_end_to_end_on_fixture(words, diarization):
    utts = align.align(words, diarization["turns"])
    assert [u["speaker"] for u in utts] == [
        "SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01",
        "SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01",
    ]
    assert utts[0]["text"].startswith("Hi Sam")
    assert sum(len(u["words"]) for u in utts) == len(words)


def test_align_with_stats_counts_fallbacks():
    words = [w("a", 0.0, 1.5), w("x", 2.6, 2.8), w("c", 5.0, 6.0)]  # x: midpoint 2.7, nearest is S0
    utts, stats = align.align_with_stats(words, TURNS)
    assert stats == {"words": 3, "covered": 2, "fallback": 1}
    assert [u["speaker"] for u in utts] == ["S0", "S1"]


def test_rename_turns_and_utterances():
    turns = align.rename_turns(TURNS, {"S0": "Sam"})
    assert [t["speaker"] for t in turns] == ["Sam", "S1"]
    utts = [{"speaker": "S1", "words": [{"speaker": "S1"}]}]
    out = align.rename_utterances(utts, {"S1": "Alex"})
    assert out[0]["speaker"] == "Alex" and out[0]["words"][0]["speaker"] == "Alex"
