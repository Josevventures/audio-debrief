import pytest

from audio_debrief import metrics
ME, OTHER = "Sam", "Alex Rivera"  # matches conftest.speaker_map


def test_speaker_stats(named_call):
    utts, _, _ = named_call
    stats = metrics.speaker_stats(utts)
    assert set(stats) == {ME, OTHER}
    assert stats[ME]["turn_count"] == 5
    assert stats[OTHER]["turn_count"] == 4
    assert stats[ME]["talk_time_s"] > stats[OTHER]["talk_time_s"]
    assert abs(stats[ME]["share"] + stats[OTHER]["share"] - 1.0) < 1e-6
    assert stats[ME]["max_turn_s"] >= stats[ME]["mean_turn_s"] > 0


def test_answer_detection(named_call, config):
    utts, _, _ = named_call
    answers = metrics.find_answers(utts, ME, config)
    # block after "Why Northfield?" is too short; the block after a non-question statement still counts
    assert len(answers) == 3
    a1, a2, a3 = answers
    assert a1["question"] == "Hi Sam, can you walk me through your background?"
    assert a1["is_question"] is True
    assert a1["start"] == 5.0 and abs(a1["duration_s"] - 25.35) < 0.01
    assert a1["word_count"] == 28
    assert a1["latency_s"] == pytest.approx(1.05, abs=0.01)
    assert a2["question"] == "What is your favorite database engine?"
    assert a2["start"] == 49.0
    assert a3["question"] == "Great, thanks for sharing that context." and a3["is_question"] is False
    assert a3["start"] == 75.0


def U(speaker, start, end, text):
    toks = text.split()
    d = (end - start) / len(toks)
    words = [{"text": t, "start": round(start + i * d, 3), "end": round(start + (i + 1) * d - 0.05, 3), "speaker": speaker}
             for i, t in enumerate(toks)]
    return {"speaker": speaker, "start": words[0]["start"], "end": words[-1]["end"], "text": text, "words": words}


def test_backchannel_does_not_split_a_self_block(config):
    utts = [U("Alex", 0, 4, "So how did that project go for you?"),
            U("Sam", 5, 17, "It went well because we planned it carefully from the start."),
            U("Alex", 17.5, 18.0, "Mm-hmm."),
            U("Sam", 18.5, 30, "And then we shipped it two weeks early which the client liked a lot.")]
    answers = metrics.find_answers(utts, "Sam", config)
    assert len(answers) == 1
    assert answers[0]["start"] == 5.0 and answers[0]["end"] == utts[-1]["end"]
    assert answers[0]["word_count"] == len(utts[1]["words"]) + len(utts[3]["words"])


def test_trigger_is_tail_of_contiguous_other_speaker_run(config):
    long_intro = " ".join(f"w{i}" for i in range(50)) + " so walk me through your background"
    utts = [U("Alex", 0, 10, "Great to meet you."),
            U("Alex", 12, 40, long_intro),
            U("Sam", 41, 70, " ".join(["word"] * 40))]
    a = metrics.find_answers(utts, "Sam", config)[0]
    assert a["question"].endswith("so walk me through your background")
    assert len(a["question"].split()) == config["question_tail_words"]
    assert a["question"].startswith("w16")  # 4 + 56 = 60 tokens; last 40 start at token 20 = w16


def test_question_detection_rules():
    assert metrics.is_question("Why Northfield?")
    assert metrics.is_question("Tell me about a time you failed")
    assert metrics.is_question("what drives you")
    assert not metrics.is_question("Great, thanks.")


def test_filler_counting_with_phrases(config):
    words = [{"text": t} for t in "Um, I kind of like it, you know, right? I like tea".split()]
    out = metrics.count_fillers(words, config["fillers"])
    assert out["by_filler"]["um"] == 1
    assert out["by_filler"]["kind of like"] == 1
    assert out["by_filler"]["you know"] == 1
    assert out["by_filler"]["right?"] == 1
    assert out["by_filler"]["like"] == 1  # raw count; the "like" inside "kind of like" is consumed
    assert out["total"] == 5


def test_filler_rate_and_wpm_per_answer(named_call, config):
    utts, _, _ = named_call
    a1, a2, _ = metrics.find_answers(utts, ME, config)
    assert a1["fillers"]["total"] == 3  # um, you know, kind of like
    assert a1["fillers"]["rate_per_100"] == pytest.approx(3 / 28 * 100, abs=0.01)
    assert a1["wpm"] == pytest.approx(28 / 25.35 * 60, abs=0.1)
    assert a2["fillers"]["total"] == 4  # like, I mean, pretty much, right?


def test_wpm():
    assert metrics.words_per_minute(120, 60) == 120.0
    assert metrics.words_per_minute(10, 0) == 0.0


def test_pauses_with_context(named_call, config):
    utts, _, _ = named_call
    pauses = metrics.find_pauses(utts, ME, config["pause_threshold_s"])
    assert len(pauses) == 1
    p = pauses[0]
    assert p["gap_s"] == pytest.approx(2.55, abs=0.01)
    assert p["at"] == pytest.approx(19.95, abs=0.01)
    assert p["preceding_words"] == "built a practice at Meridian."
    assert p["timestamp"] == "00:19"


def test_pause_inside_single_utterance():
    words = [
        {"text": "one", "start": 0.0, "end": 0.3, "speaker": "Sam"},
        {"text": "two", "start": 3.0, "end": 3.3, "speaker": "Sam"},
    ]
    utts = [{"speaker": "Sam", "start": 0.0, "end": 3.3, "text": "one two", "words": words}]
    pauses = metrics.find_pauses(utts, "Sam", 2.0)
    assert len(pauses) == 1 and pauses[0]["preceding_words"] == "one"


def test_overlap_events_and_counts(named_call, config):
    _, turns, overlaps = named_call
    events = metrics.overlap_events(turns, overlaps, config["overlap_min_s"])
    assert len(events) == 1
    e = events[0]
    assert e["already_speaking"] == ME
    assert e["interrupter"] == OTHER
    assert e["yielded"] == ME
    assert e["start"] == 30.5
    counts = metrics.overlap_counts(events)
    assert counts[OTHER]["interrupted"] == 1
    assert counts[ME]["was_interrupted"] == 1
    assert counts[ME]["interrupted"] == 0


def test_short_overlap_ignored(named_call):
    _, turns, _ = named_call
    assert metrics.overlap_events(turns, [{"start": 30.5, "end": 30.8}], 0.5) == []


def test_mode_tagging_by_token_set_ratio(qa_questions, config):
    thr, targets = config["rehearsed_match_threshold"], config["answer_targets"]
    assert metrics.tag_mode("Hi Sam, can you walk me through your background?", qa_questions, thr, targets) == "rehearsed"
    loose = ("It would be great to hear your story. Maybe walk me through where you studied and the roles "
             "you have held since.")
    assert metrics.tag_mode(loose, qa_questions, thr, targets) == "rehearsed"
    assert metrics.tag_mode("Yeah, so why Northfield of all places?", qa_questions, thr, targets) == "rehearsed"
    assert metrics.tag_mode("What is your favorite database engine?", qa_questions, thr, targets) == "improvised"
    assert metrics.tag_mode("Great, thanks for sharing that context.", qa_questions, thr, targets) == "improvised"


def test_long_trigger_needs_a_shared_content_token(config):
    # A long trigger can score above the threshold on token_set_ratio purely from shared letters and
    # function words; with no content word in common it must not be tagged rehearsed.
    qa = ["Do you have any questions for me?"]
    trigger = ("and then maybe you could speak to your data platform and software experience "
               "in a bit more detail")
    assert metrics.qa_match(trigger, qa) == (0.0, "")
    assert metrics.tag_mode(trigger, qa, config["rehearsed_match_threshold"], {}) == "improvised"
    real = "Okay, that is all I had on my side. Do you have any questions for me on anything so far?"
    score, q = metrics.qa_match(real, qa)
    assert q == qa[0] and score >= config["rehearsed_match_threshold"]


def test_mode_tagging_by_target_keywords_and_missing_qa(config):
    thr, targets = config["rehearsed_match_threshold"], config["answer_targets"]
    # not in qa_prep, but a prepped category keyword ("compensation") -> rehearsed
    assert metrics.tag_mode("And what are you thinking on compensation?", ["Why Northfield?"], thr, targets) == "rehearsed"
    assert metrics.tag_mode("What is your favorite database engine?", None, thr, targets) == "unknown"
    assert metrics.tag_mode("Walk me through your background.", None, thr, targets) == "rehearsed"
    assert metrics.tag_mode("Walk me through your background.", ["Why Northfield?"], thr, {}) == "improvised"


def test_interruptions_not_measured_when_diarizer_cannot_see_overlaps(named_call, qa_questions, config):
    utts, turns, _ = named_call
    m = metrics.compute_metrics(utts, turns, [], ME, qa_questions, config, overlaps_measured=False)
    assert m["interruptions"]["status"] == "not_measured"
    assert m["interruptions"]["events"] == [] and m["interruptions"]["counts"] == {}
    assert "far-side" in m["interruptions"]["note"]
    m = metrics.compute_metrics(utts, turns, [], ME, qa_questions, config)
    assert m["interruptions"]["status"] == "measured"


def test_target_matching_and_flag(config):
    key, seconds = metrics.match_target("Can you walk me through your background?", config["answer_targets"])
    assert key == "background_walkthrough" and seconds == 90
    assert metrics.match_target("What is your favorite database engine?", config["answer_targets"]) is None


def test_compute_metrics_shape(named_call, qa_questions, config):
    utts, turns, overlaps = named_call
    m = metrics.compute_metrics(utts, turns, overlaps, ME, qa_questions, config)
    assert set(m) >= {"self", "speakers", "answers", "fillers", "pace", "pauses", "interruptions", "mode_comparison"}
    assert m["self"] == ME
    a1, a2, a3 = m["answers"]
    assert a1["mode"] == "rehearsed" and a2["mode"] == "improvised" and a3["mode"] == "improvised"
    assert a1["target"] == {"key": "background_walkthrough", "target_s": 90, "over": False}
    assert a2["target"] is None
    assert len(a1["pauses"]) == 1 and a2["pauses"] == []
    assert m["fillers"][ME]["total"] >= 7
    assert m["pace"][ME]["wpm"] > 0
    cmp = m["mode_comparison"]
    assert cmp["rehearsed"]["count"] == 1 and cmp["improvised"]["count"] == 2
    assert cmp["rehearsed"]["mean_duration_s"] == pytest.approx(25.35, abs=0.01)
    assert cmp["rehearsed"]["mean_pause_count"] == 1


def test_over_target_flag(config):
    long_answer = {"question": "Walk me through your background", "duration_s": 130}
    assert metrics.target_for(long_answer, config["answer_targets"])["over"] is True
