import json
from pathlib import Path

import pytest

from speech_scorer import Settings, load_words, make_normalizer, score
from speech_scorer.__main__ import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def timed(text, rate=0.4, start=0.0, pauses=None, probs=None):
    """Build a flat transcript: one word every ``rate`` seconds, optional extra pause before word i."""
    t, words = start, []
    for i, w in enumerate(text.split()):
        t += (pauses or {}).get(i, 0.0)
        words.append({"word": " " + w, "start": round(t, 3), "end": round(t + rate * 0.8, 3),
                      "probability": (probs or {}).get(i, 0.95)})
        t += rate
    return {"words": words}


def run(script, transcript, **settings):
    return score(script, load_words(transcript, make_normalizer(script)), Settings(**settings))


SCRIPT = "one two three four five six seven eight nine ten"


# ---------------------------------------------------------------- loading


def test_loads_segments_and_splits_punctuation():
    data = {"segments": [{"words": [{"word": " Hello,", "start": 0, "end": 0.3}]},
                         {"words": [{"word": " world.", "start": 0.4, "end": 0.7, "probability": 0.8}]}]}
    words = load_words(data, make_normalizer("hello world"))
    assert [w.text for w in words] == ["hello", "world"] and words[1].probability == 0.8


def test_missing_word_timestamps_is_a_clear_error():
    with pytest.raises(ValueError, match="word_timestamps"):
        load_words({"segments": [{"text": "hi", "start": 0, "end": 1}]}, make_normalizer("hi"))


# ---------------------------------------------------------------- scoring


def test_perfect_reading():
    r = run(SCRIPT, timed(SCRIPT, rate=0.5))  # 120 wpm
    assert r.wer == 0 and r.coverage == 1 and r.missed_passages == [] and r.score == 100.0
    assert "Excellent accuracy" in r.feedback[0]


def test_skipped_passage_is_quoted_with_time():
    r = run(SCRIPT, timed("one two three seven eight nine ten", rate=0.5))
    assert r.deletions == 3 and r.coverage == pytest.approx(0.7)
    assert len(r.missed_passages) == 1
    p = r.missed_passages[0]
    assert p.text == "four five six" and p.words == 3 and p.near_seconds == pytest.approx(1.4)


def test_short_slips_are_not_passages():
    r = run(SCRIPT, timed("one two tree four five six seven eight nine ten"))
    assert r.substitutions == 1 and r.missed_passages == []


def test_off_script_speech_is_not_penalised_but_middle_extras_are():
    r = run(SCRIPT, timed("good morning all " + SCRIPT + " thank you"))
    assert (r.off_script_before, r.off_script_after) == (3, 2) and r.wer == 0
    r = run(SCRIPT, timed("one two three um four five six seven eight nine ten"))
    assert r.insertions == 1 and r.wer == pytest.approx(0.1)


def test_pace_and_pauses():
    fast = run(SCRIPT, timed(SCRIPT, rate=0.25))  # 240 wpm
    assert fast.pace_wpm > 160 and any("fast" in f for f in fast.feedback)
    paused = run(SCRIPT, timed(SCRIPT, rate=0.5, pauses={5: 3.0}))
    assert len(paused.long_pauses) == 1 and paused.long_pauses[0].seconds == pytest.approx(3.1)


def test_rushed_stretch_shows_as_pace_variation():
    script = " ".join(f"w{i}" for i in range(80))
    words = timed(script, rate=0.5)["words"]
    # Second half spoken twice as fast.
    t = words[39]["start"]
    for w in words[40:]:
        t += 0.25
        w["start"], w["end"] = round(t, 3), round(t + 0.2, 3)
    r = run(script, {"words": words})
    assert r.pace_windows > 1 and r.fastest_window_wpm >= 2 * r.slowest_window_wpm * 0.9
    assert any("Pace varies" in f for f in r.feedback)


def test_arabic_spelling_variants_count_as_correct():
    script = "\u0646\u0633\u062a\u062e\u062f\u0645 \u0627\u0644\u0645\u0627\u0621 \u0628\u062d\u0643\u0645\u0629"  # نستخدم الماء بحكمة
    spoken = "\u0646\u0633\u062a\u062e\u062f\u0645 \u0627\u0644\u0645\u0627\u0621 \u0628\u062d\u0643\u0645\u0647"  # ...بحكمه (ha)
    assert run(script, timed(spoken)).wer == 0
    assert run(script, timed(spoken, rate=0.4)).script_words == 3


def test_low_confidence_words_listed():
    r = run(SCRIPT, timed(SCRIPT, probs={4: 0.3}))
    assert r.low_confidence_words == ["five"]


def test_empty_script_rejected():
    with pytest.raises(ValueError):
        run("", timed("hello"))


# ---------------------------------------------------------------- samples and CLI


def test_arabic_sample_end_to_end():
    script = (EXAMPLES / "ar_script.txt").read_text(encoding="utf-8")
    r = score(script, load_words(EXAMPLES / "ar_transcript.json", make_normalizer(script)))
    assert r.deletions == 7 and r.substitutions == 1 and r.insertions == 0
    assert r.missed_passages[0].words == 7 and r.missed_passages[0].text.startswith("\u0648\u0647\u064a")  # وهي
    assert (r.off_script_before, r.off_script_after) == (3, 2)
    assert len(r.long_pauses) == 1


def test_cli_json(capsys):
    main([str(EXAMPLES / "en_transcript.json"), str(EXAMPLES / "en_script.txt"), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["missed_passages"][0]["text"] == "Before you write any code"
    assert 0 <= out["score"] <= 100
