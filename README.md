# audio-debrief

Local, offline analysis of a recorded conversation. Feed it an audio file and it
produces a diarized, timestamped transcript plus delivery metrics for one speaker
(you): talk time, answer lengths against targets, pace, fillers, pauses, tone
proxies, and a proper-noun diff against a second transcript if you have one.
Interview practice is the case it was built for; any two-or-more-person recording
works.

Audio never leaves the machine. The only network traffic is the one-time model
download from HuggingFace.

## Why this exists

I record my interviews (with consent) and debrief afterwards. The debrief used to
run off the phone's live transcript, which tells you what was said but nothing
about how: how long each answer ran, where the fillers piled up, whether a pause
was a real hesitation or the other person nodding along. I wanted the debrief
grounded in what the audio shows rather than in what I remembered feeling, so I
built a pipeline around Whisper and pyannote and pointed it at a real recording.

It mostly failed. The recording was a phone sitting next to a laptop on a video
call, so my voice was captured directly and the interviewer's came out of the
laptop speaker. pyannote's segmentation model barely fired on that far-side
audio: only 54% of Whisper's words landed inside any speaker turn. The transcript
looked fine at a glance, and the interviewer had silently disappeared from the
speaker map. Nothing errored.

Whisper's VAD did hear the interviewer, though. So the fix was not a better
threshold; it was a different diarizer that takes Whisper's own words as the
speech evidence. Words are split into short units at silences, each unit is
embedded with WeSpeaker, units that match an enrolled voiceprint of my voice are
mine, and the rest are clustered into the other speakers. On the same recording
that put every word inside a turn.

The honest limit: the hybrid diarizer cannot see two people talking at once,
because Whisper only emits one word stream. So with the default diarizer the
report says interruptions were `not_measured`. It never estimates them. If you
have a recording where both sides are captured directly, `--diarizer pyannote`
runs the full pyannote pipeline and measures overlaps for real.

## How it works

```
convert -> transcribe -> diarize (hybrid | pyannote) -> align -> metrics -> prosody -> nouns -> report
```

Every stage is wrapped so a failure lands in an `analysis_status` table and the
report is still written with the sections that could be computed.

**convert.** ffmpeg to 16 kHz mono PCM WAV. Cached.

**transcribe.** faster-whisper (`large-v3` by default) with word timestamps and
VAD. Hotwords from `config.yaml`, `--speakers`, and the output folder's parent
name go into the initial prompt so names and jargon are spelled right.

**diarize (hybrid, default).** Whisper's words are split into units at silences
longer than `hybrid.unit_gap_s` and capped at `hybrid.unit_max_s`. Each unit is
embedded with `pyannote/wespeaker-voxceleb-resnet34-LM`. A unit is you when its
cosine similarity to `voiceprints/self.npy` is at least `voiceprint.match_threshold`;
an ambiguous band down to `voiceprint.ambiguous_low` inherits the previous unit's
label. The remaining units are clustered (average linkage, cosine distance
`voiceprint.cluster_threshold`) so panels work. Far-side audio fragments badly at
the unit level (one interviewer came out as 96 raw clusters), so only clusters
with at least `voiceprint.min_cluster_s` of speech count as speakers, speaker
clusters whose centroids agree above `voiceprint.centroid_merge_sim` are merged,
and every smaller fragment is absorbed into the most similar speaker. A final
nearest-centroid pass relabels units the absolute rule did not claim. Turns are
runs of same-speaker units. Unit embeddings are cached.

**diarize (pyannote).** `pyannote/speaker-diarization-3.1` end to end. Detects
overlaps; under-detects far-side speech.

**align.** Each word takes the speaker whose turn covers its midpoint; words
outside every turn go to the nearest one and are counted in
`analysis_status.align.detail` (near zero with hybrid). Consecutive same-speaker
words merge into utterances, split at silences over `utterance_gap_s`.

**metrics.** Per speaker: talk time, share, turn count. For you: an *answer map*,
one row per contiguous block of your speech of at least `answer_min_s` that
follows someone else. Short non-question backchannels ("Mm-hmm", "Right") do not
break a block. The "question" shown is the last `question_tail_words` words of the
other speaker's run, because people rarely phrase prompts as questions; the JSON
carries `is_question` as a flag and `latency_s` (silence before you start).
Each answer gets word count, WPM, filler count and rate, pause count, and a
length target with an `OVER` flag when its trigger matches an `answer_targets`
pattern. Answers are tagged `rehearsed` when the trigger fuzzy-matches a question
in `qa_prep.md` (rapidfuzz `token_set_ratio` at or above
`rehearsed_match_threshold`, and sharing at least one content word) or a prepped
category keyword, `improvised` otherwise, `unknown` with no `qa_prep.md`.
Pauses inside your speech longer than `pause_threshold_s` are listed with the five
words before them and a `kind` read from the waveform: `silence` when the RMS
inside the gap is well below your whole-call speech level, otherwise
`unvoiced/possible backchannel` (energy Whisper did not transcribe). Only
`silence` pauses are safe to read as hesitation.

**prosody.** Median F0 (librosa pyin), F0 spread, RMS mean and variance per
answer, z-scored against the spread of the same values across all of your
utterances on the call. See the caveat below.

**nouns.** If a second transcript exists (a phone recorder's `.txt` export, for
instance), a token-level diff surfaces proper-noun disagreements as a table:
`recorder | whisper | count | first at | direction | confirmed`. `direction` says
which side looks right (a hotword or a capitalized mid-sentence token wins);
`confirmed` is pre-filled when the better side is a known hotword.

**report.** Three files in the output folder: `transcript_audio_<date>.md`
(`[mm:ss] **Name:** text`), `audio_analysis_<date>.md` (the human-readable
analysis, status table first), and `audio_analysis_<date>.json` (everything as
data, plus `tool_version` and `analysis_status`). The JSON is what a downstream
debrief step should cite.

## Setup

Requirements: [`uv`](https://docs.astral.sh/uv/), `ffmpeg` on PATH, Python 3.12.
An NVIDIA GPU is strongly recommended; CPU works, but Whisper large-v3 on CPU
takes roughly real time or longer.

```
uv sync
```

This creates a `.venv` with torch (CUDA 12.8 wheels), faster-whisper, pyannote,
librosa and friends. It is large (several GB). To keep it outside the repo, set
`UV_PROJECT_ENVIRONMENT` to a path of your choosing before running `uv sync`.

pyannote's models are gated. Logged in on huggingface.co, accept the terms on all
three:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM

Then store a read token in the standard HF location:

```
uv run hf auth login
```

Without a token `run` exits with code 2 and prints these steps; nothing else runs.

First-run downloads: Whisper `large-v3` (about 3 GB, CTranslate2 format) and the
pyannote diarization, segmentation and embedding models (about 100 MB total),
cached in the HF hub cache under your user profile.

## Usage

```
uv run python -m audio_debrief run \
  --audio "recordings/2026-01-15 call.m4a" \
  --round-dir "recordings/2026-01-15" \
  --date 2026-01-15 \
  --me "Sam" \
  --speakers "Alex Rivera"
```

| Flag | Default | Meaning |
|---|---|---|
| `--me` | config `self_name` (`Me`) | Your display name in the outputs |
| `--speakers` | names parsed from `../interviewer_*.md` headings, if any | The other participants, in order of first appearance |
| `--recorder-text` | first `*.txt` in `--round-dir` | A second transcript for the proper-noun diff |
| `--qa-prep` | `round-dir/qa_prep.md` | `### N. Question` headings drive rehearsed/improvised tagging |
| `--diarizer` | `hybrid` (config `diarizer`) | `hybrid` or `pyannote` (see above) |
| `--device` | `cuda` | `cpu` forces CPU (Whisper int8) |
| `--model` | `large-v3` | Any faster-whisper model name |
| `--no-cache` | off | Ignore `.audio_cache/` and recompute every stage |
| `--non-interactive` | off | Never prompt; exit 2 if no voiceprint exists |
| `--config` | `config.yaml` in the repo root | Fillers, thresholds, answer targets, hotwords |

Exit code: 0 when every stage is `ok` (speakers may be `unmatched`), 1 when any
stage failed or was skipped (the report is still written), 2 for a setup problem.

### First run: enrolling your voice

The tool keeps a local voiceprint at `voiceprints/self.npy` (gitignored; it never
leaves the machine). If it does not exist, `run` prints the first ~10 seconds of
each diarized speaker and asks which one is you. Answering enrolls the voiceprint
and the run continues. Every later run matches you automatically.

If there is no interactive terminal, the samples are printed and the tool exits 2
with the exact command to run:

```
uv run python -m audio_debrief enroll --audio "<same audio>" --speaker-label SPEAKER_01
```

`enroll` reuses the cached conversion, transcription and embeddings, so it is
quick; then re-run `run`. Optional `--round-dir` tells `enroll` where
`.audio_cache/` lives (default: the audio's own folder).

### Prosody caveat

The tone numbers are proxies. The source is typically a phone-call AAC at around
50 kbps, a single-channel mix of both speakers, with no calibration and with
diarization boundaries deciding which frames count as yours. Each answer's value
is z-scored against the per-utterance values across your whole call, so a z of 1
means "one utterance-spread above your own average on this call", never anything
absolute. The report carries a fixed confidence note for this reason. Use them to
find moments worth listening to, not as findings on their own.

### Cache behavior

Every stage writes its intermediate to `<round-dir>/.audio_cache/` (`audio.wav`,
`whisper.json`, `diarization.json`, `embeddings.json`, `unit_embeddings.npz`) and
skips itself on re-run when the file exists. A cached diarization produced by the
other diarizer, or by hybrid before a voiceprint existed, is recomputed
automatically. A second run therefore finishes in seconds and only recomputes
alignment, metrics, prosody, nouns and the report, which is how you iterate on
`config.yaml` cheaply. `--no-cache` recomputes everything.

### Error handling

| Condition | Behavior |
|---|---|
| ffmpeg missing | exit 2 with install hint |
| No CUDA | Whisper runs CPU int8; warning in `analysis_status.transcribe` |
| No HF token / terms not accepted | exit 2 with the three setup steps; nothing else runs |
| No voiceprint | interactive enroll; with `--non-interactive` or no TTY, exit 2 with the enroll command |
| No speaker above the voiceprint threshold | `Speaker N` labels, `analysis_status.speakers = unmatched`; no guess |
| `qa_prep.md` missing | answers tagged `unknown`; noted in status |
| Second transcript missing | noun section skipped; noted in status |
| Any stage exception | stage `failed` with message; dependents `skipped`; report still written |

## Tests

```
uv run pytest
```

Unit tests run on synthetic fixtures (a short two-speaker call with invented
names); no audio and no models are needed. `tests/fixtures/make_fixtures.py`
regenerates the word and turn fixtures.

## License

MIT, see [`LICENSE`](LICENSE).
