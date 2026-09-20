import pytest

from audio_debrief import prosody


def utt(f0, f0sd, rms, rmsv):
    return {"f0_median_hz": f0, "f0_std_hz": f0sd, "rms_mean": rms, "rms_var": rmsv}


def test_baseline_is_mean_and_spread_of_per_utterance_values():
    base = prosody.baseline_from_utterances([utt(100, 10, 0.1, 0.01), utt(120, 20, 0.2, 0.02), utt(140, 30, 0.3, 0.03)])
    assert base["f0_median_hz"] == 120 and base["rms_mean"] == pytest.approx(0.2)
    assert base["spread"]["f0_median_hz"] == pytest.approx(16.33, abs=0.01)
    assert base["spread"]["f0_std_hz"] == pytest.approx(8.165, abs=0.01)
    assert base["utterance_count"] == 3


def test_zscores_use_per_utterance_spread():
    base = prosody.baseline_from_utterances([utt(100, 10, 0.1, 0.01), utt(120, 20, 0.2, 0.02), utt(140, 30, 0.3, 0.03)])
    z = prosody.zscores(utt(140, 10, 0.2, 0.05), base)
    assert z["f0_median"] == pytest.approx(1.22, abs=0.01)
    assert z["f0_std"] == pytest.approx(-1.22, abs=0.01)
    assert z["rms_mean"] == pytest.approx(0.0, abs=0.01)
    assert z["rms_var"] > 2


def test_annotate_pauses_classifies_silence_vs_energy_in_gap():
    import numpy as np
    sr = 16000
    rng = np.random.default_rng(1)
    y = np.zeros(sr * 20, dtype=np.float32)
    y[0:sr * 5] = rng.uniform(-0.3, 0.3, sr * 5)            # speech, then a real silence 5.0-7.5 s
    y[sr * 5:int(sr * 7.5)] = rng.uniform(-0.01, 0.01, int(sr * 2.5))
    y[sr * 10:sr * 15] = rng.uniform(-0.3, 0.3, sr * 5)     # speech, then near-speech energy 15.0-17.5 s
    y[sr * 15:int(sr * 17.5)] = rng.uniform(-0.25, 0.25, int(sr * 2.5))
    pauses = [{"at": 5.0, "gap_s": 2.5}, {"at": 15.0, "gap_s": 2.5}, {"at": 0.0, "gap_s": 2.0}]
    out = prosody.annotate_pauses(pauses, y, sr, silence_ratio=0.6)
    assert out is pauses
    assert pauses[0]["gap_rms_ratio"] < 0.1 and pauses[0]["kind"] == "silence"
    assert pauses[0]["gap_rms"] < 0.01 and 0.1 < pauses[1]["gap_rms"] < 0.2  # absolute RMS inside the gap
    assert 0.7 < pauses[1]["gap_rms_ratio"] < 1.0 and pauses[1]["kind"] == "unvoiced/possible backchannel"
    assert pauses[2]["gap_rms_ratio"] is None and pauses[2]["kind"] == "unknown"  # nothing before it to compare


def test_annotate_pauses_uses_speaker_baseline_when_given():
    """A quiet trailing-off before the gap must not make a real silence look loud: with the
    speaker's whole-call RMS as baseline, the gap is judged against normal speech level."""
    import numpy as np
    sr = 16000
    rng = np.random.default_rng(2)
    y = np.zeros(sr * 12, dtype=np.float32)
    y[0:sr * 3] = rng.uniform(-0.3, 0.3, sr * 3)                 # normal speech 0-3 s
    y[sr * 3:sr * 5] = rng.uniform(-0.04, 0.04, sr * 2)          # trailing off 3-5 s (the whole "2 s before")
    y[sr * 5:int(sr * 7.5)] = rng.uniform(-0.03, 0.03, int(sr * 2.5))  # gap: quiet, ~0.75 of the tail but ~0.1 of speech
    utts = [{"speaker": "Sam", "start": 0.0, "end": 5.0}]
    base = prosody.speech_rms(y, sr, utts, "Sam")
    assert base is not None and 0.1 < base < 0.2
    p_tail = prosody.annotate_pauses([{"at": 5.0, "gap_s": 2.5}], y, sr, silence_ratio=0.6)[0]
    p_base = prosody.annotate_pauses([{"at": 5.0, "gap_s": 2.5}], y, sr, silence_ratio=0.6, baseline_rms=base)[0]
    assert p_tail["kind"] == "unvoiced/possible backchannel"   # the old rule misfires here
    assert p_base["kind"] == "silence" and p_base["gap_rms_ratio"] < 0.2
    assert prosody.speech_rms(y, sr, utts, "Nobody") is None


def test_baseline_ignores_utterances_without_voicing_and_handles_zero_spread():
    base = prosody.baseline_from_utterances([utt(None, None, 0.1, 0.01), utt(100, 5, 0.1, 0.01), utt(100, 5, 0.1, 0.01)])
    assert base["f0_median_hz"] == 100 and base["utterance_count"] == 3
    z = prosody.zscores(utt(110, 5, 0.1, 0.01), base)
    assert z["f0_median"] is None  # zero spread -> no z, never a division blow-up
