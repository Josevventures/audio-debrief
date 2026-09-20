"""Regenerates words.json / turns.json for the synthetic two-speaker call.

Run: uv run python tests/fixtures/make_fixtures.py
Times are seconds. SPEAKER_00 = the other participant, SPEAKER_01 = the user ("Sam").
Every name and company in here is invented.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# (speaker, start, end, text) — words are spaced evenly inside [start, end].
UTTERANCES = [
    ("SPEAKER_00", 0.0, 4.0, "Hi Sam, can you walk me through your background?"),
    ("SPEAKER_01", 5.0, 20.0,
     "Sure. Um, I started, you know, in consulting and kind of like built a practice at Meridian."),
    # 2.5 s pause (no other speaker in between) -> one pause event
    ("SPEAKER_01", 22.5, 30.4, "Then I moved to Halcyon to lead the carve-out work there."),
    # the other speaker comes in while Sam is finishing -> overlap 30.5-31.0 in turns
    ("SPEAKER_00", 30.6, 33.0, "Why Northfield?"),
    ("SPEAKER_01", 33.5, 45.0, "Because of the technology practice and the people I met."),  # 11.5 s: too short
    ("SPEAKER_00", 46.0, 48.0, "What is your favorite database engine?"),
    ("SPEAKER_01", 49.0, 72.0,
     "I like Postgres, I mean it is pretty much the default, right? It scales well and the tooling "
     "around it is mature so most teams I have worked with pick it without much debate."),
    ("SPEAKER_00", 73.0, 74.0, "Great, thanks for sharing that context."),  # statement, not a question
    ("SPEAKER_01", 75.0, 100.0,
     "One more thing I wanted to add is that the carve-out at Halcyon taught me a lot about "
     "operating under uncertainty and building a team from scratch in a short time."),
]

# Diarization turns (may differ slightly from word times, as in real pyannote output).
TURNS = [
    ("SPEAKER_00", 0.0, 4.0),
    ("SPEAKER_01", 5.0, 20.0),
    ("SPEAKER_01", 22.5, 31.0),
    ("SPEAKER_00", 30.5, 33.0),
    ("SPEAKER_01", 33.5, 45.0),
    ("SPEAKER_00", 46.0, 48.0),
    ("SPEAKER_01", 49.0, 72.0),
    ("SPEAKER_00", 73.0, 74.0),
    ("SPEAKER_01", 75.0, 100.0),
]
OVERLAPS = [{"start": 30.5, "end": 31.0}]


def words_for(start, end, text):
    toks = text.split()
    d = (end - start) / len(toks)
    out = []
    for i, tok in enumerate(toks):
        s = round(start + i * d, 3)
        e = round(start + (i + 1) * d - 0.05, 3)
        out.append({"text": tok, "start": s, "end": e, "prob": 0.95})
    return out


def main():
    words = []
    for _, s, e, text in UTTERANCES:
        words.extend(words_for(s, e, text))
    (HERE / "words.json").write_text(json.dumps(words, indent=1), encoding="utf-8")
    turns = [{"speaker": sp, "start": s, "end": e} for sp, s, e in TURNS]
    (HERE / "turns.json").write_text(
        json.dumps({"turns": turns, "overlaps": OVERLAPS}, indent=1), encoding="utf-8")
    print(f"wrote {len(words)} words, {len(turns)} turns")


if __name__ == "__main__":
    main()
