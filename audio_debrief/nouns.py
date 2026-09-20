"""Proper-noun diff between the Recorder text and the Whisper transcript,
plus the two small markdown parsers (qa_prep questions, interviewer names)."""
from __future__ import annotations

import difflib
import re
from pathlib import Path

from .pipeline import mmss

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*|[.?!]")
_SPEAKER_LABEL_RE = re.compile(r"\[Speaker \d+\]")


def _norm(s: str) -> str:
    return s.lower().replace("-", " ").replace("’", "'")


def tokenize(chunks) -> list[dict]:
    """chunks: iterable of (text, start, new_sentence). Returns word tokens only;
    sentence_start is derived from .?! tokens and chunk boundaries flagged new_sentence."""
    tokens, boundary = [], True
    for text, start, new_sentence in chunks:
        boundary = boundary or new_sentence
        for m in _TOKEN_RE.finditer(text):
            tok = m.group(0)
            if tok in ".?!":
                boundary = True
                continue
            tokens.append({"text": tok, "lower": _norm(tok), "start": start,
                           "sentence_start": boundary, "hotword": False})
            boundary = False
    return tokens


def recorder_chunks(text: str):
    text = _SPEAKER_LABEL_RE.sub("", text)
    for line in text.splitlines():
        if line.strip():
            yield line, None, True


def whisper_chunks(words: list[dict]):
    for w in words:
        yield w["text"], w.get("start"), False


def _hotword_sets(hotwords: list[str]) -> tuple[list[list[str]], set[str]]:
    phrases = [_norm(h).split() for h in hotwords]
    singles = {" ".join(p) for p in phrases}
    for p in phrases:
        singles.update(w for w in p if len(w) >= 3)
    return [p for p in phrases if len(p) > 1], singles


def merge_hotwords(tokens: list[dict], hotwords: list[str]) -> list[dict]:
    phrases, singles = _hotword_sets(hotwords)
    phrases.sort(key=len, reverse=True)
    out, i = [], 0
    while i < len(tokens):
        merged = False
        for ph in phrases:
            n = len(ph)
            if [t["lower"] for t in tokens[i:i + n]] == ph:
                out.append({**tokens[i], "text": " ".join(t["text"] for t in tokens[i:i + n]),
                            "lower": " ".join(ph), "hotword": True})
                i += n
                merged = True
                break
        if not merged:
            t = tokens[i]
            out.append({**t, "hotword": t["lower"] in singles})
            i += 1
    return out


def _candidate(tok: dict) -> bool:
    return tok["hotword"] or (tok["text"][0].isupper() and not tok["sentence_start"])


# Pronouns, contractions, articles, fillers, and transcript noise that never name anything.
STOPLIST = set("""
i i've i'm i'd i'll i'am we we're we've we'd we'll you you're you've you'll he he's she she's it it's they
they're they've me my mine our ours us your yours him her his them their theirs this that these those there here
the a an and or but so if as of to in on at for with by from up down out no not yes yeah yep nope um uh hmm mm
mhm okay ok oh ah um-hum name say said says like just right well then now really very also kind sort mean know
think thing things is are was were be been being am do did does done have has had get got go going gonna wanna
one two three
""".split())


def _proper(tok: dict) -> bool:
    return tok["text"][0].isupper() and len(tok["text"]) >= 3 and not tok["sentence_start"]


def _weight(tok: dict) -> int:
    return 2 if tok["hotword"] else (1 if _proper(tok) else 0)


def _keep(l: dict, r: dict, stop: set) -> bool:
    for side, other in ((l, r), (r, l)):
        if other["hotword"]:
            continue  # a hotword on the other side rescues a stoplisted or single-letter token
        if side["lower"] in stop or len(side["text"]) == 1:
            return False
    return _weight(l) > 0 or _weight(r) > 0


def _direction(l: dict, r: dict) -> str:
    wl, wr = _weight(l), _weight(r)
    return "recorder→whisper" if wr > wl else ("whisper→recorder" if wl > wr else "unclear")


def noun_corrections(recorder_text: str, whisper_words: list[dict], hotwords: list[str],
                     stoplist: set | None = None) -> list[dict]:
    stop = STOPLIST if stoplist is None else set(stoplist)
    rec = merge_hotwords(tokenize(recorder_chunks(recorder_text)), hotwords)
    whi = merge_hotwords(tokenize(whisper_chunks(whisper_words)), hotwords)
    sm = difflib.SequenceMatcher(None, [t["lower"] for t in rec], [t["lower"] for t in whi], autojunk=False)
    rows: dict[tuple[str, str], dict] = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace":
            continue
        left, right = rec[i1:i2], whi[j1:j2]
        if len(left) == len(right):
            pairs = [(l, r) for l, r in zip(left, right) if _candidate(l) or _candidate(r)]
        else:
            pairs = list(zip([t for t in left if _candidate(t)], [t for t in right if _candidate(t)]))
        for l, r in pairs:
            if not _keep(l, r, stop):
                continue
            direction = _direction(l, r)
            right_side = r if direction == "recorder→whisper" else (l if direction == "whisper→recorder" else None)
            key = (l["text"], r["text"])
            row = rows.setdefault(key, {
                "recorder_spelling": l["text"], "whisper_spelling": r["text"], "count": 0,
                "first_timestamp": mmss(r["start"] or 0), "_start": r["start"] or 0, "direction": direction,
                "confirmed": "yes" if right_side is not None and right_side["hotword"] else ""})
            row["count"] += 1
            if (r["start"] or 0) < row["_start"]:
                row["_start"], row["first_timestamp"] = r["start"], mmss(r["start"])
    out = sorted(rows.values(), key=lambda r: r["_start"])
    for r in out:
        r.pop("_start")
    return out


# --- small parsers ------------------------------------------------------------

_QA_RE = re.compile(r"^###\s*\d+\.\s*(.+?)\s*$")
_INTERVIEWER_RE = re.compile(r"^#\s*Interviewer Profile\s*[—–-]\s*(.+?)\s*$")


def parse_qa_questions(path: str | Path) -> list[str]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _QA_RE.match(line.strip())
        if m:
            out.append(m.group(1))
    return out


def parse_interviewer_names(company_dir: str | Path) -> list[str]:
    names = []
    for p in sorted(Path(company_dir).glob("interviewer_*.md")):
        with open(p, encoding="utf-8") as fh:
            first = fh.readline().strip()
        m = _INTERVIEWER_RE.match(first)
        if m:
            names.append(m.group(1))
    return names
