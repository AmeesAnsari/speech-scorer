"""Compare a timed transcript with the script it should follow, and score the delivery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from arabic_toolkit import align, tokenize

from .transcript import Word, make_normalizer, tokens


@dataclass(frozen=True)
class Settings:
    pace_range_wpm: tuple[float, float] = (100.0, 160.0)  # comfortable reading pace
    pace_window_seconds: float = 10.0  # short enough to catch a rushed stretch
    long_pause_seconds: float = 2.0
    min_missed_run: int = 3  # consecutive script words missed before it counts as a skipped passage
    low_confidence: float = 0.5
    weights: tuple[float, float, float] = (50.0, 30.0, 20.0)  # accuracy, coverage, pace


@dataclass
class Passage:
    text: str  # as written in the script
    words: int
    near_seconds: Optional[float]  # where in the recording it should have been


@dataclass
class Pause:
    at: float
    seconds: float


@dataclass
class Report:
    score: float
    accuracy: float  # 1 - WER, floored at 0
    coverage: float  # share of script words spoken correctly
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    script_words: int
    spoken_words: int
    reading_start: Optional[float]
    reading_end: Optional[float]
    off_script_before: int  # words before the first script match (greeting, intro)
    off_script_after: int
    pace_wpm: Optional[float]
    pace_in_range: float  # share of windows within the target pace
    pace_windows: int
    slowest_window_wpm: Optional[float]
    fastest_window_wpm: Optional[float]
    missed_passages: list[Passage] = field(default_factory=list)
    long_pauses: list[Pause] = field(default_factory=list)
    low_confidence_words: list[str] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def score(script: str, words: list[Word], settings: Settings = Settings()) -> Report:
    normalizer = make_normalizer(script)
    ref = tokens(script, normalizer)
    ref_display = tokenize(script)  # original spelling, same tokenisation, for quoting passages
    if len(ref_display) != len(ref):  # normalisation changed the token count (rare): quote normalised text
        ref_display = ref
    hyp = [w.text for w in words]
    if not ref:
        raise ValueError("The script is empty.")

    ops = align(ref, hyp)

    # Words spoken before the first / after the last script match are off-script (greeting, Q&A),
    # not reading errors. Strip those leading/trailing insertions before counting.
    matched = [i for i, op in enumerate(ops) if op.kind == "equal"]
    first, last = (matched[0], matched[-1]) if matched else (0, len(ops) - 1)
    lead = sum(1 for op in ops[:first] if op.kind == "insert")
    tail = sum(1 for op in ops[last + 1:] if op.kind == "insert")
    core = [op for i, op in enumerate(ops)
            if not (op.kind == "insert" and (i < first or i > last))]

    subs = sum(op.kind == "substitute" for op in core)
    dels = sum(op.kind == "delete" for op in core)
    ins = sum(op.kind == "insert" for op in core)
    equal = sum(op.kind == "equal" for op in core)
    wer = (subs + dels + ins) / len(ref)
    accuracy = max(0.0, 1.0 - wer)
    coverage = equal / len(ref)

    # Reading span: from the first to the last word aligned to the script.
    in_script = [words[op.hyp_index] for op in core if op.hyp_index is not None]
    start = in_script[0].start if in_script else None
    end = in_script[-1].end if in_script else None

    report = Report(
        score=0.0, accuracy=accuracy, coverage=coverage, wer=wer, substitutions=subs, deletions=dels,
        insertions=ins, script_words=len(ref), spoken_words=len(words), reading_start=start, reading_end=end,
        off_script_before=lead, off_script_after=tail, pace_wpm=None, pace_in_range=0.0, pace_windows=0,
        slowest_window_wpm=None, fastest_window_wpm=None,
    )
    report.missed_passages = _missed(core, ref_display, words, settings.min_missed_run)
    report.long_pauses = [Pause(a.end, b.start - a.end) for a, b in zip(in_script, in_script[1:])
                          if b.start - a.end >= settings.long_pause_seconds]
    report.low_confidence_words = [w.raw for w in in_script
                                   if w.probability is not None and w.probability < settings.low_confidence]
    _pace(report, in_script, settings)

    wa, wc, wp = settings.weights
    report.score = round(wa * accuracy + wc * coverage + wp * report.pace_in_range, 1)
    report.feedback = _feedback(report, settings)
    return report


def _missed(core, ref_display, words, min_run) -> list[Passage]:
    """Runs of script words that were skipped or garbled (deleted or substituted in a row)."""
    passages, run, last_time = [], [], None

    def flush():
        if len(run) >= min_run:
            passages.append(Passage(" ".join(ref_display[i] for i in run), len(run), last_time))

    for op in core:
        if op.kind in ("delete", "substitute"):
            run.append(op.ref_index)
        elif op.kind == "equal":
            flush()
            run = []
        if op.hyp_index is not None and op.kind == "equal":
            last_time = words[op.hyp_index].end
    flush()
    return passages


def _pace(report: Report, spoken: list[Word], settings: Settings) -> None:
    if len(spoken) < 2:
        return
    duration = spoken[-1].end - spoken[0].start
    if duration <= 0:
        return
    report.pace_wpm = round(len(spoken) / duration * 60, 1)

    lo, hi = settings.pace_range_wpm
    size = settings.pace_window_seconds
    t, rates = spoken[0].start, []
    while t < spoken[-1].end:
        window_end = min(t + size, spoken[-1].end)
        n = sum(1 for w in spoken if t <= w.start < window_end)
        if window_end - t >= size / 2:  # ignore a short final fragment
            rates.append(n / (window_end - t) * 60)
        t += size
    if not rates:  # whole recording shorter than half a window
        rates = [report.pace_wpm]
    report.pace_windows = len(rates)
    report.pace_in_range = sum(lo <= r <= hi for r in rates) / len(rates)
    report.slowest_window_wpm = round(min(rates), 1)
    report.fastest_window_wpm = round(max(rates), 1)


def _feedback(r: Report, s: Settings) -> list[str]:
    tips = []
    if r.coverage >= 0.95 and r.wer <= 0.05:
        tips.append("Excellent accuracy: the script was followed closely.")
    elif r.wer > 0.2:
        tips.append(f"Word error rate is {r.wer:.0%}. Review the missed passages and practise the harder sections.")
    for p in r.missed_passages:
        where = f" (around {_fmt(p.near_seconds)})" if p.near_seconds is not None else ""
        tips.append(f"Skipped or unclear passage of {p.words} words{where}: \"{p.text}\"")
    lo, hi = s.pace_range_wpm
    if r.pace_wpm is not None:
        if r.pace_wpm > hi:
            tips.append(f"Average pace {r.pace_wpm:.0f} wpm is fast; aim for {lo:.0f}-{hi:.0f}.")
        elif r.pace_wpm < lo:
            tips.append(f"Average pace {r.pace_wpm:.0f} wpm is slow; aim for {lo:.0f}-{hi:.0f}.")
        if r.pace_windows > 1 and r.pace_in_range < 0.7 and r.fastest_window_wpm - r.slowest_window_wpm >= 20:
            tips.append(f"Pace varies: from {r.slowest_window_wpm:.0f} to {r.fastest_window_wpm:.0f} wpm "
                        f"across {s.pace_window_seconds:.0f}s windows.")
    for p in r.long_pauses[:3]:
        tips.append(f"Long pause of {p.seconds:.1f}s at {_fmt(p.at)}.")
    if r.off_script_before or r.off_script_after:
        tips.append(f"Off-script speech not scored: {r.off_script_before} words before and "
                    f"{r.off_script_after} after the script.")
    return tips
