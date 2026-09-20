"""Markdown + JSON outputs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import __version__
from .pipeline import STAGES, mmss

PROSODY_CONFIDENCE_NOTE = (
    "Confidence: low. These are proxies computed from phone-call AAC at ~50 kbps, a single-channel "
    "mix of both speakers, with no calibration. Read direction, not magnitude."
)
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DIARIZER_LABELS = {"hybrid": "hybrid (whisper-VAD + wespeaker)", "pyannote": DIARIZATION_MODEL}


def diarizer_label(method: str | None) -> str:
    return DIARIZER_LABELS.get(method or "", method or "—")


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _unavailable(analysis: dict, stage: str) -> str:
    s = analysis["analysis_status"].get(stage, {})
    return f"_Not available — stage `{stage}` {s.get('status', 'pending')}" + (f": {s['detail']}_" if s.get("detail") else "_")


def _fmt(v, nd=2) -> str:
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _status_section(a: dict) -> str:
    rows = [[s, a["analysis_status"][s]["status"], a["analysis_status"][s]["detail"] or ""] for s in STAGES]
    return _table(["Stage", "Status", "Detail"], rows)


def _talk_time(m: dict) -> str:
    rows = [[sp, f"{mmss(v['talk_time_s'])} ({v['talk_time_s']:.0f} s)", f"{v['share'] * 100:.1f}%",
             v["turn_count"], f"{v['mean_turn_s']:.1f} s", f"{v['max_turn_s']:.1f} s"]
            for sp, v in m["speakers"].items()]
    return _table(["Speaker", "Talk time", "Share", "Turns", "Mean turn", "Max turn"], rows)


def _target_cell(t: dict | None) -> str:
    if t is None:
        return "—"
    return f"{t['target_s']} s ({t['key']}) — {'OVER' if t['over'] else 'ok'}"


def _tail(text: str, n_words: int = 15) -> str:
    words = text.split()
    return text if len(words) <= n_words else "… " + " ".join(words[-n_words:])


def _answer_map(m: dict, me: str) -> str:
    if not m["answers"]:
        return f"_No {me} answers detected (>= answer_min_s following another speaker)._"
    rows = [[i + 1, a["timestamp"], f"{a['duration_s']:.0f} s", _tail(a["question"]), a["mode"], a["word_count"],
             f"{a['wpm']:.0f}", f"{a['fillers']['total']} ({a['fillers']['rate_per_100']:.1f})",
             len(a["pauses"]), _target_cell(a["target"])]
            for i, a in enumerate(m["answers"])]
    table = _table(["#", "Start", "Duration", "Trigger (tail)", "Mode", "Words", "WPM", "Fillers (/100)", "Pauses", "Target"], rows)
    return table + "\n\n_Trigger = last words of the other speaker's run before the answer (full text in the JSON)._"


def _fillers(m: dict, me: str) -> str:
    rows = [[sp, v["total"], f"{v['rate_per_100']:.2f}"] for sp, v in m["fillers"].items()]
    parts = [_table(["Speaker", "Fillers", "Per 100 words"], rows)]
    if me in m["fillers"]:
        by = sorted(m["fillers"][me]["by_filler"].items(), key=lambda kv: -kv[1])
        parts.append(f"\n{me} by filler:\n\n" + _table(["Filler", "Count"], [[f, c] for f, c in by]))
    parts.append("\n_Filler \"like\" is a raw count; verb \"like\" is not distinguished._")
    return "\n".join(parts)


def _pace_pauses(m: dict, me: str) -> str:
    rows = [[sp, v["word_count"], f"{v['wpm']:.0f}"] for sp, v in m["pace"].items()]
    parts = [_table(["Speaker", "Words", "WPM"], rows), f"\n{me} pauses > threshold: {len(m['pauses'])}\n"]
    for p in m["pauses"]:
        kind = f" {p['kind']}" if p.get("kind") else ""
        ratio = f" (gap RMS ratio {p['gap_rms_ratio']:.2f})" if p.get("gap_rms_ratio") is not None else ""
        parts.append(f"- {p['timestamp']} — {p['gap_s']:.1f} s{kind} after \"…{p['preceding_words']}\"{ratio}")
    if m["pauses"]:
        parts.append("\n_Kind comes from the waveform: \"silence\" when the gap's RMS is well below the speaker's whole-call "
                     "speech level; otherwise energy Whisper did not transcribe (a backchannel, laugh, or breath)._")
    return "\n".join(parts)


def _interruptions(m: dict) -> str:
    if m["interruptions"].get("status") == "not_measured":
        return f"_{m['interruptions']['note']}_"
    counts = m["interruptions"]["counts"]
    rows = [[sp, v["interrupted"], v["was_interrupted"], v["yielded"]] for sp, v in counts.items()]
    parts = [_table(["Speaker", "Interrupted others", "Was interrupted", "Yielded"], rows) if rows
             else "_No overlaps >= threshold._"]
    for e in m["interruptions"]["events"]:
        parts.append(f"- {e['timestamp']} — {e['interrupter']} came in over {e['already_speaking']} "
                     f"({e['duration_s']:.1f} s); {e['yielded']} yielded")
    return "\n".join(parts)


def _prosody(p: dict | None, a: dict, me: str) -> str:
    parts = [PROSODY_CONFIDENCE_NOTE, ""]
    if not p:
        parts.append(_unavailable(a, "prosody"))
        return "\n".join(parts)
    b = p["baseline"]
    parts.append(f"{me} baseline (whole call): F0 median {_fmt(b['f0_median_hz'], 1)} Hz, F0 std "
                 f"{_fmt(b['f0_std_hz'], 1)} Hz, RMS mean {_fmt(b['rms_mean'], 4)}, RMS var {_fmt(b['rms_var'], 6)}\n")
    rows = [[mmss(x["start"]), _fmt(x["f0_median_hz"], 1), _fmt(x["z"]["f0_median"]), _fmt(x["f0_std_hz"], 1),
             _fmt(x["z"]["f0_std"]), _fmt(x["rms_mean"], 4), _fmt(x["z"]["rms_mean"]), _fmt(x["z"]["rms_var"])]
            for x in p["answers"]]
    parts.append(_table(["Answer", "F0 med (Hz)", "z", "F0 std (Hz)", "z", "RMS mean", "z", "RMS var z"], rows))
    return "\n".join(parts)


def _nouns(rows: list[dict] | None, a: dict) -> str:
    if rows is None:
        return _unavailable(a, "nouns")
    if not rows:
        return "_No proper-noun differences found._"
    return _table(["Reference spelling", "Whisper spelling", "Count", "First at", "Direction", "Confirmed"],
                  [[r["reference_spelling"], r["whisper_spelling"], r["count"], r["first_timestamp"],
                    r.get("direction", ""), r["confirmed"]] for r in rows])


def _modes(m: dict) -> str:
    c = m["mode_comparison"]
    rows = [[mode, v["count"], _fmt(v["mean_duration_s"]), _fmt(v["mean_wpm"]), _fmt(v["mean_filler_rate"]),
             _fmt(v["mean_pause_count"])] for mode, v in c.items()]
    note = "" if m.get("prep_available") else "\n\n_prep.md not available: all answers tagged `unknown`._"
    return _table(["Mode", "Answers", "Mean duration (s)", "Mean WPM", "Mean fillers /100", "Mean pauses"], rows) + note


def render_markdown(a: dict) -> str:
    m, me, info = a.get("metrics"), a.get("self", "Me"), a.get("transcribe_info") or {}
    has = m is not None
    sections = [
        ("Analysis status", _status_section(a)),
        ("Talk time", _talk_time(m) if has else _unavailable(a, "metrics")),
        ("Answer map", _answer_map(m, me) if has else _unavailable(a, "metrics")),
        ("Fillers", _fillers(m, me) if has else _unavailable(a, "metrics")),
        ("Pace and pauses", _pace_pauses(m, me) if has else _unavailable(a, "metrics")),
        ("Interruptions", _interruptions(m) if has else _unavailable(a, "metrics")),
        ("Tone proxies", _prosody(a.get("prosody"), a, me)),
        ("Proper-noun corrections", _nouns(a.get("noun_corrections"), a)),
        ("Prepared vs improvised", _modes(m) if has else _unavailable(a, "metrics")),
    ]
    head = [f"# Audio analysis — {a.get('date', '')}", "",
            f"Audio: `{a.get('audio', '')}` · Tool: audio_debrief v{__version__} · Whisper: "
            f"{info.get('model', '—')} ({info.get('device', '—')}, {info.get('compute_type', '—')}) · "
            f"Diarization: {diarizer_label(a.get('diarization_method'))}", ""]
    body = [f"## {title}\n\n{content}\n" for title, content in sections]
    return "\n".join(head + body)


def render_transcript(utterances: list[dict], info: dict) -> str:
    head = ["# Audio transcript", "",
            f"Model: {info.get('model', '—')} on {info.get('device', '—')} · Diarization: "
            f"{info.get('diarization', DIARIZATION_MODEL)} · Tool: audio_debrief v{__version__}", ""]
    lines = [f"[{mmss(u['start'])}] **{u['speaker']}:** {u['text']}" for u in utterances]
    return "\n".join(head + lines) + "\n"


def build_json(a: dict) -> dict:
    m = a.get("metrics") or {}
    return {
        "tool_version": __version__, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": a.get("date"), "audio": a.get("audio"), "self": a.get("self"),
        "speaker_map": a.get("speaker_map"), "transcribe_info": a.get("transcribe_info"),
        "analysis_status": a["analysis_status"],
        "speakers": m.get("speakers"), "answers": m.get("answers"), "fillers": m.get("fillers"),
        "pace": m.get("pace"), "pauses": m.get("pauses"), "interruptions": m.get("interruptions"),
        "mode_comparison": m.get("mode_comparison"), "prosody": a.get("prosody"),
        "noun_corrections": a.get("noun_corrections"),
    }


def write_outputs(out_dir: str | Path, date: str, analysis: dict, utterances: list[dict] | None) -> list[Path]:
    out_dir = Path(out_dir)
    status = analysis.get("analysis_status") or {}
    if status.get("report", {}).get("status") == "pending":  # we are the report stage; record success up front
        status["report"] = {"status": "ok", "detail": ""}
    info = dict(analysis.get("transcribe_info") or {})
    info.setdefault("diarization", diarizer_label(analysis.get("diarization_method")))
    paths = [out_dir / f"transcript_audio_{date}.md", out_dir / f"audio_analysis_{date}.md",
             out_dir / f"audio_analysis_{date}.json"]
    transcript = render_transcript(utterances or [], info) if utterances else \
        "# Audio transcript\n\n_Transcript unavailable — see audio_analysis status table._\n"
    paths[0].write_text(transcript, encoding="utf-8")
    paths[1].write_text(render_markdown(analysis), encoding="utf-8")
    paths[2].write_text(json.dumps(build_json(analysis), indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
