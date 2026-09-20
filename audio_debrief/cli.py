"""`python -m audio_debrief run ...` / `enroll ...`. Every stage is wrapped by
pipeline.run_stage so a failure lands in analysis_status and the report is still written."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__, align, metrics, nouns, report, speakers
from .pipeline import SetupError, cached_json, load_config, mark, new_status, run_stage

ENROLL_HINT = ("Pick your own label above and run:\n"
               "  uv run python -m audio_debrief enroll --audio \"<audio>\" --speaker-label <LABEL>\n"
               "then re-run (everything else is cached).")


def _args(argv):
    p = argparse.ArgumentParser(prog="audio_debrief",
                                description="Local, offline analysis of a recorded conversation")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyze a recording into the output folder")
    r.add_argument("--audio", required=True)
    r.add_argument("--out-dir", required=True, help="output folder; also holds .audio_cache/")
    r.add_argument("--date", required=True, help="YYYY-MM-DD used in output file names")
    r.add_argument("--me", help="your display name in the outputs (default: config self_name)")
    r.add_argument("--reference-transcript", help="a second transcript of the same recording, for the "
                   "proper-noun diff (default: first *.txt in out-dir)")
    r.add_argument("--speakers", nargs="*", help="the other participants' names, in order of first appearance")
    r.add_argument("--prep", help="prepared questions or talking points (### N. headings); "
                   "default: out-dir/prep.md")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--non-interactive", action="store_true", help="never prompt; exit 2 if no voiceprint")
    e = sub.add_parser("enroll", help="save a voiceprint for one diarized speaker")
    e.add_argument("--audio", required=True)
    e.add_argument("--speaker-label", required=True, help="e.g. SPEAKER_01 (see run's prompt output)")
    e.add_argument("--name", default="self", help="voiceprint file name under voiceprints/ (default: self)")
    e.add_argument("--out-dir", help="output folder; also holds .audio_cache/ (default: the audio's folder)")
    for sp in (r, e):
        sp.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
        sp.add_argument("--model", default="large-v3")
        sp.add_argument("--diarizer", choices=["hybrid", "pyannote"], help="default: config.yaml diarizer (hybrid)")
        sp.add_argument("--config")
    return p.parse_args(argv)


def _diarize(diarizer, cache, wav, whisper, voiceprint, cfg, device, token, use_cache, log=print):
    """Cached diarization for either method; the hybrid result carries per-speaker embeddings."""
    from . import diarize, hybrid

    path, emb_path = cache / "diarization.json", cache / "embeddings.json"
    if use_cache and path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        stale = d.get("method", "pyannote") != diarizer or (
            diarizer == "hybrid" and voiceprint is not None and d.get("self_label") is None)
        if not stale:
            if emb_path.exists():
                d["embeddings"] = json.loads(emb_path.read_text(encoding="utf-8"))
            log("[diarize] cached")
            return d
    if diarizer == "pyannote":
        d = {**diarize.diarize(wav, device, token), "method": "pyannote"}
    else:
        d = hybrid.diarize(wav, whisper["words"], voiceprint, cfg, device, token, cache, log)
        emb_path.write_text(json.dumps(d["embeddings"]), encoding="utf-8")
    path.write_text(json.dumps({k: v for k, v in d.items() if k != "embeddings"}, ensure_ascii=False), encoding="utf-8")
    return d


def _speaker_embeddings(d, cache, wav, device, use_cache, token):
    if d.get("embeddings"):
        return {k: np.asarray(v) for k, v in d["embeddings"].items()}
    raw = cached_json(cache / "embeddings.json",
                      lambda: speakers.speaker_embeddings(wav, d["turns"], d["overlaps"], device, token), use_cache)
    return {k: np.asarray(v) for k, v in raw.items()}


def _prompt_for_self(words, diar, embeddings, non_interactive: bool) -> str | None:
    tagged = align.assign_speakers(words, diar["turns"])
    print("\nNo voiceprint found. Diarized speakers (first ~10 s of each):")
    for label in diar["speakers"]:
        print(f"  {label}: {speakers.sample_text(tagged, label)[:300]}")
    if non_interactive or not sys.stdin.isatty():
        raise SetupError("No voiceprint and no interactive terminal. " + ENROLL_HINT)
    while True:
        try:
            ans = input("Which speaker is you? (label, or 'skip'): ").strip()
        except EOFError:
            # isatty() can be true under some Windows shells that still have no readable stdin.
            raise SetupError("No voiceprint and stdin closed. " + ENROLL_HINT)
        if ans.lower() == "skip":
            return None
        if ans in embeddings:
            speakers.save_voiceprint(embeddings[ans])
            print(f"Enrolled {ans} as you -> {speakers.voiceprint_path()}")
            return ans
        print(f"Unknown label; choose one of {list(embeddings)}")


def run(a) -> int:
    from . import convert, diarize, prosody, transcribe

    cfg = load_config(a.config)
    status, use_cache = new_status(), not a.no_cache
    diarizer = a.diarizer or cfg.get("diarizer", "hybrid")
    audio, out_dir = Path(a.audio), Path(a.out_dir)
    cache = out_dir / ".audio_cache"
    cache.mkdir(parents=True, exist_ok=True)
    token = diarize.require_token()  # exit 2 before anything else runs
    voiceprint = speakers.load_voiceprint()
    if voiceprint is None and a.non_interactive:
        raise SetupError(f"No voiceprint at {speakers.voiceprint_path()} and --non-interactive set. Run `enroll` first.")

    me = a.me or cfg.get("self_name", "Me")
    names = a.speakers or []
    hotwords = [*names, *(cfg.get("hotwords") or [])]
    context_prompt = cfg.get("context_prompt") or "Conversation."
    prep_path = Path(a.prep) if a.prep else out_dir / "prep.md"
    prep = nouns.parse_prep_questions(prep_path) if prep_path.exists() else None
    txts = [Path(a.reference_transcript)] if a.reference_transcript else sorted(out_dir.glob("*.txt"))
    reference = txts[0] if txts and txts[0].exists() else None

    wav = run_stage(status, "convert", lambda: convert.to_wav(audio, cache / "audio.wav", use_cache))
    whisper = run_stage(status, "transcribe", lambda: cached_json(
        cache / "whisper.json", lambda: transcribe.transcribe(wav, a.model, a.device, hotwords, context_prompt), use_cache))
    if whisper and whisper["info"].get("warning"):
        mark(status, "transcribe", "ok", whisper["info"]["warning"])
    diar = run_stage(status, "diarize", lambda: _diarize(
        diarizer, cache, wav, whisper, voiceprint, cfg, a.device, token, use_cache))
    if diar:
        mark(status, "diarize", "ok", f"{diar.get('method')}: {len(diar['turns'])} turns, "
                                      f"{len(diar['speakers'])} speakers, {diar.get('unit_count', '-')} units, "
                                      f"scores {diar.get('scores', '-')}")

    def speakers_stage():
        nonlocal voiceprint
        embs = _speaker_embeddings(diar, cache, wav, a.device, use_cache, token)
        self_label = diar.get("self_label")
        if voiceprint is None:
            self_label = _prompt_for_self(whisper["words"] if whisper else [], diar, embs, a.non_interactive)
            voiceprint = speakers.load_voiceprint()
        scores = {k: round(speakers.cosine(e, voiceprint), 4) for k, e in embs.items()} if voiceprint is not None else {}
        if self_label is None and voiceprint is not None:
            self_label, scores = speakers.match_self(embs, voiceprint, cfg["voiceprint"]["speaker_threshold"])
        mapping = speakers.name_speakers(diar["turns"], self_label, names, me)
        if self_label is None:
            mark(status, "speakers", "unmatched", f"no speaker matched the voiceprint: {scores}")
        else:
            mark(status, "speakers", "ok", f"{self_label} = {me} (mean-embedding cosine {scores.get(self_label)}); {mapping}")
        return mapping

    speaker_map = run_stage(status, "speakers", speakers_stage) or {}
    turns = align.rename_turns(diar["turns"], speaker_map) if diar else []

    def align_stage():
        utts, st = align.align_with_stats(whisper["words"], turns, cfg["utterance_gap_s"])
        mark(status, "align", "ok", f"{st['fallback']} of {st['words']} words outside any turn (nearest-turn fallback)")
        return utts

    utts = run_stage(status, "align", align_stage)

    def metrics_stage():
        result = metrics.compute_metrics(utts, turns, diar["overlaps"], me, prep, cfg,
                                         overlaps_measured=(diar.get("method") == "pyannote"))
        y, sr = convert.load_waveform(wav)
        prosody.annotate_pauses(result["pauses"], y, sr, cfg["pause_silence_ratio"],
                                baseline_rms=prosody.speech_rms(y, sr, utts, me))  # answers share these dicts
        return result

    m = run_stage(status, "metrics", metrics_stage)
    if prep is None and status["metrics"]["status"] == "ok":
        mark(status, "metrics", "ok", f"prep.md not found at {prep_path}: answer modes are 'unknown'")
    pros = run_stage(status, "prosody", lambda: prosody.analyze(wav, utts, m["answers"], me))
    if reference is None:
        mark(status, "nouns", "skipped", "no reference transcript (*.txt) found in out-dir")
    nn = run_stage(status, "nouns", lambda: nouns.noun_corrections(
        reference.read_text(encoding="utf-8", errors="replace"), whisper["words"], hotwords))

    analysis = {"date": a.date, "audio": audio.name, "self": me, "analysis_status": status,
                "diarization_method": diar.get("method") if diar else diarizer,
                "speaker_map": speaker_map, "metrics": m, "prosody": pros, "noun_corrections": nn,
                "transcribe_info": whisper["info"] if whisper else {}}
    paths = run_stage(status, "report", lambda: report.write_outputs(out_dir, a.date, analysis, utts))
    if paths is None:  # report itself failed: last-ditch write of the status JSON
        paths = report.write_outputs(out_dir, a.date, {**analysis, "metrics": None, "prosody": None}, None)
    for p in paths:
        print(f"wrote {p}")
    return 0 if all(v["status"] in ("ok", "unmatched") for v in status.values()) else 1


def enroll(a) -> int:
    from . import convert, diarize, transcribe

    cfg = load_config(a.config)
    diarizer = a.diarizer or cfg.get("diarizer", "hybrid")
    audio = Path(a.audio)
    cache = (Path(a.out_dir) if a.out_dir else audio.parent) / ".audio_cache"
    cache.mkdir(parents=True, exist_ok=True)
    token = diarize.require_token()
    wav = convert.to_wav(audio, cache / "audio.wav", True)
    whisper = None
    if diarizer == "hybrid":
        whisper = cached_json(cache / "whisper.json",
                              lambda: transcribe.transcribe(wav, a.model, a.device, cfg.get("hotwords") or [],
                                                            cfg.get("context_prompt") or "Conversation."), True)
    d = _diarize(diarizer, cache, wav, whisper, None, cfg, a.device, token, True)
    embs = _speaker_embeddings(d, cache, wav, a.device, True, token)
    if a.speaker_label not in embs:
        raise SetupError(f"{a.speaker_label} not in diarized speakers {list(embs)}")
    path = speakers.save_voiceprint(embs[a.speaker_label], a.name)
    print(f"saved voiceprint for {a.speaker_label} -> {path}")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows pipes default to cp1252; transcript text is UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    a = _args(argv if argv is not None else sys.argv[1:])
    try:
        return run(a) if a.cmd == "run" else enroll(a)
    except SetupError as exc:
        print(f"\nSETUP REQUIRED\n{exc}", file=sys.stderr)
        return 2
