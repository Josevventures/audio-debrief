import json

from audio_debrief import __version__, pipeline, report
ME, OTHER = "Sam", "Alex Rivera"  # matches conftest.speaker_map

SECTIONS = [
    "## Analysis status", "## Talk time", "## Answer map", "## Fillers", "## Pace and pauses",
    "## Interruptions", "## Tone proxies", "## Proper-noun corrections", "## Prepared vs improvised",
]


def minimal_analysis(status):
    return {
        "date": "2026-01-15", "audio": "call.m4a", "self": ME,
        "analysis_status": status,
        "speaker_map": {"SPEAKER_00": OTHER, "SPEAKER_01": ME},
        "metrics": None, "prosody": None, "noun_corrections": None,
        "transcribe_info": {"model": "large-v3", "device": "cuda", "compute_type": "float16"},
    }


def test_status_graph_marks_dependents_skipped():
    status = pipeline.new_status()
    assert set(status) == set(pipeline.STAGES)
    pipeline.mark_failed(status, "diarize", "boom")
    assert status["diarize"] == {"status": "failed", "detail": "boom"}
    for dep in ("speakers", "align", "metrics", "prosody"):
        assert status[dep]["status"] == "skipped", dep
        assert "diarize" in status[dep]["detail"]
    assert status["nouns"]["status"] == "pending"
    assert status["convert"]["status"] == "pending"


def test_markdown_sections_in_order_when_everything_missing():
    status = pipeline.new_status()
    pipeline.mark_ok(status, "convert")
    pipeline.mark_failed(status, "diarize", "no token")
    md = report.render_markdown(minimal_analysis(status))
    positions = [md.index(s) for s in SECTIONS]
    assert positions == sorted(positions)
    assert "failed" in md and "no token" in md
    assert report.PROSODY_CONFIDENCE_NOTE in md


def test_markdown_with_full_metrics(named_call, prep_questions, config):
    from audio_debrief import metrics
    utts, turns, overlaps = named_call
    status = pipeline.new_status()
    for s in pipeline.STAGES:
        pipeline.mark_ok(status, s)
    a = minimal_analysis(status)
    a["metrics"] = metrics.compute_metrics(utts, turns, overlaps, ME, prep_questions, config)
    a["noun_corrections"] = [{"reference_spelling": "Alec", "whisper_spelling": "Alex",
                              "count": 2, "first_timestamp": "00:00", "confirmed": "yes"}]
    a["prosody"] = {"baseline": {"f0_median_hz": 120.0, "f0_std_hz": 20.0, "rms_mean": 0.05, "rms_var": 0.001},
                    "answers": [{"start": 5.0, "f0_median_hz": 125.0, "f0_std_hz": 18.0, "rms_mean": 0.06,
                                 "rms_var": 0.001, "z": {"f0_median": 0.4, "f0_std": -0.2, "rms_mean": 0.9, "rms_var": 0.0}}]}
    md = report.render_markdown(a)
    positions = [md.index(s) for s in SECTIONS]
    assert positions == sorted(positions)
    assert "Alec" in md and "Alex" in md
    assert "prepared" in md and "improvised" in md
    assert OTHER in md and ME in md
    assert f"{ME} baseline (whole call)" in md and f"{ME} by filler" in md
    # answer map rows really render (question text, mode, target flag), not just the header
    answer_section = md[md.index("## Answer map"):md.index("## Fillers")]
    assert "| 1 | 00:05 | 25 s | Hi Sam, can you walk me through your background? | prepared |" in answer_section
    assert "(background_walkthrough) — ok" in answer_section
    assert "| 2 | 00:49 |" in answer_section
    pause_section = md[md.index("## Pace and pauses"):md.index("## Interruptions")]
    assert 'after "…built a practice at Meridian."' in pause_section
    a["metrics"]["pauses"][0].update({"kind": "silence", "gap_rms_ratio": 0.12})
    md2 = report.render_markdown(a)
    assert '2.5 s silence after "…built a practice at Meridian." (gap RMS ratio 0.12)' in md2
    # a long trigger is shown as its tail in the table; the JSON keeps the full text
    a["metrics"]["answers"][0]["question"] = " ".join(f"w{i}" for i in range(40))
    md = report.render_markdown(a)
    assert "| … w25 w26" in md and "w24 w25" not in md
    assert report.build_json(a)["answers"][0]["question"].startswith("w0 w1")


def test_json_shape(named_call, prep_questions, config):
    from audio_debrief import metrics
    utts, turns, overlaps = named_call
    status = pipeline.new_status()
    pipeline.mark_failed(status, "transcribe", "x")
    a = minimal_analysis(status)
    a["metrics"] = metrics.compute_metrics(utts, turns, overlaps, ME, prep_questions, config)
    data = report.build_json(a)
    text = json.dumps(data)  # must be serializable
    data = json.loads(text)
    assert data["tool_version"] == __version__
    assert data["self"] == ME
    assert set(data["analysis_status"]) == set(pipeline.STAGES)
    assert data["analysis_status"]["transcribe"]["status"] == "failed"
    assert data["analysis_status"]["align"]["status"] == "skipped"
    for key in ("speakers", "answers", "fillers", "pace", "pauses", "interruptions", "mode_comparison"):
        assert key in data


def test_header_names_the_diarizer_and_interruptions_say_not_measured(named_call, prep_questions, config):
    from audio_debrief import metrics
    utts, turns, _ = named_call
    status = pipeline.new_status()
    a = minimal_analysis(status)
    a["diarization_method"] = "hybrid"
    a["metrics"] = metrics.compute_metrics(utts, turns, [], ME, prep_questions, config, overlaps_measured=False)
    md = report.render_markdown(a)
    assert "Diarization: hybrid (whisper-VAD + wespeaker)" in md.splitlines()[2]
    section = md[md.index("## Interruptions"):md.index("## Tone proxies")]
    assert "not measurable with this capture method (far-side audio)" in section
    assert "--diarizer pyannote" in section
    a["diarization_method"] = "pyannote"
    assert "Diarization: pyannote/speaker-diarization-3.1" in report.render_markdown(a).splitlines()[2]
    tr = report.render_transcript(utts, {"model": "large-v3", "device": "cuda", "diarization": report.diarizer_label("hybrid")})
    assert "hybrid (whisper-VAD + wespeaker)" in tr


def test_transcript_rendering(named_call):
    utts, _, _ = named_call
    md = report.render_transcript(
        utts, {"model": "large-v3", "device": "cuda", "diarization": "pyannote/speaker-diarization-3.1"})
    assert f"[00:00] **{OTHER}:** Hi Sam, can you walk me through your background?" in md
    assert f"[00:05] **{ME}:**" in md
    assert "large-v3" in md and "pyannote" in md


def test_write_outputs(tmp_path, named_call):
    utts, _, _ = named_call
    status = pipeline.new_status()
    a = minimal_analysis(status)
    paths = report.write_outputs(tmp_path, "2026-01-15", a, utts)
    assert (tmp_path / "transcript_audio_2026-01-15.md").exists()
    assert (tmp_path / "audio_analysis_2026-01-15.md").exists()
    assert (tmp_path / "audio_analysis_2026-01-15.json").exists()
    assert len(paths) == 3
    written = json.loads((tmp_path / "audio_analysis_2026-01-15.json").read_text(encoding="utf-8"))
    assert written["analysis_status"]["report"]["status"] == "ok"  # not left as "pending" in its own output
    # a report stage already marked failed (fallback write) must stay failed
    pipeline.mark_failed(status, "report", "render blew up")
    report.write_outputs(tmp_path, "2026-01-15", a, utts)
    written = json.loads((tmp_path / "audio_analysis_2026-01-15.json").read_text(encoding="utf-8"))
    assert written["analysis_status"]["report"]["status"] == "failed"


def test_mmss():
    assert report.mmss(0) == "00:00"
    assert report.mmss(65.4) == "01:05"
    assert report.mmss(3725) == "62:05"
