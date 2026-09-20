import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
APP_ROOT = Path(__file__).resolve().parent.parent

# The synthetic call: SPEAKER_01 is the user ("Sam"), SPEAKER_00 the other participant.
ME = "Sam"
OTHER = "Alex Rivera"


@pytest.fixture
def words():
    return json.loads((FIXTURES / "words.json").read_text(encoding="utf-8"))


@pytest.fixture
def diarization():
    return json.loads((FIXTURES / "turns.json").read_text(encoding="utf-8"))


@pytest.fixture
def speaker_map():
    return {"SPEAKER_00": OTHER, "SPEAKER_01": ME}


@pytest.fixture
def config():
    from audio_debrief.pipeline import load_config
    return load_config(APP_ROOT / "config.yaml")


@pytest.fixture
def prep_questions():
    from audio_debrief.nouns import parse_prep_questions
    return parse_prep_questions(FIXTURES / "prep.md")


@pytest.fixture
def named_call(words, diarization, speaker_map):
    """Utterances and turns with real names applied, like cli.run produces."""
    from audio_debrief import align
    turns = align.rename_turns(diarization["turns"], speaker_map)
    utts = align.align(words, turns)
    return utts, turns, diarization["overlaps"]
