"""Generate the synthetic sample transcripts in this folder.

Real transcripts come from a speech recogniser (e.g. faster-whisper with word_timestamps=True).
These samples are built by hand so every behaviour is easy to see: a greeting before the
script, a skipped clause, a recognition error, a long pause, a fast stretch, and a sign-off.

    python examples/make_samples.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def build(spoken: list, seconds_per_word: float, out: str) -> None:
    """``spoken`` items are a word, or a dict with overrides: {"w": word, "pause": s, "p": prob, "rate": s}."""
    t, words, rate = 0.5, [], seconds_per_word
    for item in spoken:
        spec = item if isinstance(item, dict) else {"w": item}
        if "rate" in spec:  # change speaking rate from this word on
            rate = spec["rate"]
        t += spec.get("pause", 0.0)
        duration = round(rate * 0.8, 2)
        words.append({"word": " " + spec["w"], "start": round(t, 2), "end": round(t + duration, 2),
                      "probability": spec.get("p", 0.93)})
        t += rate
    (HERE / out).write_text(json.dumps({"segments": [{"words": words}]}, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
    print(f"wrote {out}: {len(words)} words, {t:.1f}s")


# Arabic: the script is ar_script.txt. The speaker greets the audience first, skips the clause
# "وهي عملية مكلفة وتستهلك الكثير من الطاقة", says "تسريب" for "تسرب", pauses 3.2s, then thanks the audience.
# The recogniser writes "بحكمه" with ha instead of ta marbuta; loose normalisation treats that as correct.
AR = (
    "مرحبا بكم جميعا "
    "الماء أساس الحياة والحفاظ عليه مسؤولية مشتركة بين الجميع "
    "في دولة الإمارات نعتمد بشكل كبير على تحلية مياه البحر "
    "لذلك يجب أن نستخدم الماء بحكمه في بيوتنا وأعمالنا "
    "أغلق الصنبور أثناء تنظيف الأسنان وأصلح أي"
).split()
AR += [{"w": "تسريب", "p": 0.41}]
AR += "فور اكتشافه واسق الحدائق في الصباح الباكر أو المساء".split()
AR += [{"w": "كل", "pause": 3.2}] + "قطرة نوفرها اليوم تصنع فرقا لأجيال الغد شكرا لكم".split()

# English: the script is en_script.txt. The speaker drops "before you write any code",
# then rushes through the last section.
EN = ("Good morning everyone "
      "Every reliable system starts with a clear picture of how it fails "
      "Write down what happens when a dependency is slow when it is down and when it returns "
      "something you did not expect Then decide").split()
EN += [{"w": "which", "rate": 0.26}]
EN += ("failures the system should absorb on its own and which ones need a person "
       "Automate the first group and page someone for the second").split()
EN += [{"w": "Thank", "rate": 0.4}, "you"]

if __name__ == "__main__":
    build(AR, 0.46, "ar_transcript.json")
    build(EN, 0.40, "en_transcript.json")
