# speech-scorer

Score a speech against the script it was meant to follow. Give it a speech-to-text transcript with word timestamps (Whisper, faster-whisper and similar) and the script text. It reports **accuracy, coverage, skipped passages, pace and long pauses**, with a 100-point score and plain-language feedback. Works for **Arabic** (with proper normalisation) and English.

It's a small, open version of the text-scoring stage of an Arabic speech-evaluation system I've built. Audio processing and recognition are left to your recogniser of choice.

```
$ speech-scorer examples/ar_transcript.json examples/ar_script.txt

Score: 82.7 / 100

  Accuracy    86.7%   (WER 13.3%: 1 substituted, 7 missed, 0 extra, of 60 script words)
  Coverage    86.7%   of the script spoken correctly
  Pace          116 wpm  (67% of time in range; windows 72-132 wpm)
  Reading    0:02 to 0:29

Feedback:
  - Skipped or unclear passage of 7 words (around 0:11): "وهي عملية مكلفة وتستهلك الكثير من الطاقة"
  - Pace varies: from 72 to 132 wpm across 10s windows.
  - Long pause of 3.3s at 0:22.
  - Off-script speech not scored: 3 words before and 2 after the script.

Low-confidence words from the recogniser: تسريب
```

## What it measures

| Measure | How |
|---|---|
| **Accuracy** | 1 − word error rate, from a minimum-edit word alignment of script vs. transcript |
| **Coverage** | Share of script words spoken correctly |
| **Skipped passages** | 3+ consecutive script words missed or garbled, quoted from the script with the time they should have come |
| **Pace** | Words per minute overall and in 10-second windows, against a target range (default 100–160 wpm) |
| **Long pauses** | Gaps of 2+ seconds inside the reading |
| **Off-script speech** | Greetings before the script and closing words after it are reported, not penalised |
| **Low confidence** | Words the recogniser itself was unsure about (probability < 0.5) |

**Score** = 50 × accuracy + 30 × coverage + 20 × share of time at a good pace. Weights and thresholds are in `Settings`.

## Arabic handling

Speech recognisers and scripts often spell the same Arabic word differently. Before comparing, both sides go through [arabic-nlp-toolkit](https://github.com/AmeesAnsari/arabic-nlp-toolkit)'s loose normalisation: diacritics, tatweel, alef forms, alef maksura, ta marbuta and hamza carriers are folded together. So `بحكمة` in the script and `بحكمه` from the recogniser count as a match, while a different word like `تسريب` for `تسرب` is still an error.

The alignment prefers exact matches when breaking ties, so a skipped phrase shows up as a skipped phrase, not as a chain of substitutions.

## Use it

```bash
pip install git+https://github.com/AmeesAnsari/speech-scorer

speech-scorer transcript.json script.txt            # report
speech-scorer transcript.json script.txt --json     # everything, machine-readable
speech-scorer transcript.json script.txt --pace 90 140
```

```python
from speech_scorer import score, load_words, make_normalizer

script = open("script.txt", encoding="utf-8").read()
report = score(script, load_words("transcript.json", make_normalizer(script)))
print(report.score, report.missed_passages)
```

### Getting a transcript

With [faster-whisper](https://github.com/SYSTRAN/faster-whisper):

```python
import json
from faster_whisper import WhisperModel

segments, _ = WhisperModel("large-v3").transcribe("speech.wav", language="ar", word_timestamps=True)
words = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
         for s in segments for w in s.words]
json.dump({"words": words}, open("transcript.json", "w"), ensure_ascii=False)
```

Both `{"words": [...]}` and Whisper's `{"segments": [{"words": [...]}]}` are accepted.

## Samples

[`examples/`](examples/) has an Arabic and an English script with **synthetic** transcripts built by [`make_samples.py`](examples/make_samples.py). Each includes a greeting, a skipped passage, a recognition error, a pause or a rushed stretch, and a sign-off, so you can see every part of the report.

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

13 tests: transcript loading, perfect reads, skipped passages and their timing, off-script vs. mid-script extra words, pace and pause detection, a rushed second half, Arabic spelling variants, low-confidence words, both samples end to end, and JSON output.

## Limits

- It scores what the recogniser heard. Recognition errors look like reading errors; the low-confidence list helps separate the two.
- Pace is words per minute, which varies by language and style. Set `--pace` for your context.
- Pronunciation and intonation need the audio itself and are out of scope here.

## License

MIT
