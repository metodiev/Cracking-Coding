"""Task 973 - Implement automated hint generator using solved-solution decomposition.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

The canonical solution of a problem is stored as an ordered decomposition into named steps, each tagged
with a teaching phase (setup, core, return), a prose title and the concrete code line.  An attempt is the
sequence of step identifiers the learner has already produced.  The generator compares the attempt with
the decomposition - longest common prefix, first divergence, or a solved/overshoot classification - and
emits a hint about the *first step the learner still has to produce*.  Hints escalate with the number of
failed attempts: level 1 names only the phase, level 2 the step's title, level 3 the code line, so the
answer is never handed over before the learner has tried three times.  A diverging attempt gets a repair
hint naming the step where it went off script instead of a hint that would skip ahead.

Simplified versus a real hint engine: a hand-written step decomposition instead of AST/PDG analysis, no
per-learner learning of which hint helped.  Verified: prefix and classification against an independent
linear-scan oracle over a corpus, escalation and the no-leak guarantee (level 1/2 never contain the code
and no hint ever contains the whole solution), repair hints for diverging attempts, the already-solved and
overshoot terminals, empty/absent decompositions, corpus coverage of every step, and reproducibility.
"""
from collections import namedtuple

Step = namedtuple("Step", "ident phase title code")
CANONICAL = (
    Step("parse", "setup", "Read the integers and build a frequency map",
         "values = [int(token) for token in data.split()]"),
    Step("count", "setup", "Count how often each value occurs", "counts = Counter(values)"),
    Step("sort", "core", "Sort the distinct values", "keys = sorted(counts)"),
    Step("scan", "core", "Scan consecutive pairs of distinct values", "for left, right in zip(keys, keys[1:]):"),
    Step("check", "core", "Check whether the pair differs by exactly one", "if right - left == 1:"),
    Step("update", "core", "Track the longest such run", "best = max(best, run)"),
    Step("return", "return", "Return the tracked maximum", "return best"),
)
LEVELS = ("phase", "title", "code")
# named attempts: a full prefix, a single first step, nothing, an exact solution, a divergence, overshoot
ATTEMPTS = {
    "ada": ("parse", "count", "sort", "scan"),
    "bob": ("parse",),
    "cleo": (),
    "dev": tuple(step.ident for step in CANONICAL),
    "eve": ("parse", "count", "return"),
    "finn": tuple(step.ident for step in CANONICAL) + ("extra",),
    "gus": ("parse", "count"),
}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def corpus(canonical=CANONICAL):
    """The named attempts plus every proper prefix of the solution, so each step is exercised."""
    attempts = dict(ATTEMPTS)
    for length in range(len(canonical)):
        attempts[f"prefix-{length}"] = tuple(step.ident for step in canonical[:length])
    return attempts


def progress(attempt, canonical=CANONICAL):
    """(longest common prefix, first divergence index or None, overshoot flag)."""
    prefix = 0
    while (prefix < len(attempt) and prefix < len(canonical)
           and attempt[prefix] == canonical[prefix].ident):
        prefix += 1
    divergence = prefix if prefix < len(attempt) and prefix < len(canonical) else None
    return prefix, divergence, len(attempt) > len(canonical) and prefix == len(canonical)


def prefix_oracle(attempt, canonical=CANONICAL):
    """Independent oracle: test every candidate prefix length from the longest down."""
    for length in range(min(len(attempt), len(canonical)), -1, -1):
        if all(attempt[index] == canonical[index].ident for index in range(length)):
            return length
    return 0


def classify_oracle(attempt, canonical=CANONICAL):
    """Independent oracle: one linear walk over the decomposition, classifying the attempt."""
    for index, step in enumerate(canonical):
        if index >= len(attempt):
            return ("next", index)
        if attempt[index] != step.ident:
            return ("diverged", index)
    if len(attempt) == len(canonical):
        return ("complete", None)
    return ("extra", len(canonical))


def classify(attempt, canonical=CANONICAL):
    """The generator's own classification, expressed in the oracle's vocabulary."""
    prefix, divergence, extra = progress(attempt, canonical)
    if divergence is not None:
        return ("diverged", divergence)
    if prefix == len(canonical):
        return ("extra", len(canonical)) if extra else ("complete", None)
    return ("next", prefix)


def hint(attempt, attempts_made, canonical=CANONICAL):
    """One hint for the next thing to produce, escalating with the number of failed attempts."""
    if not canonical:
        return (1, "none", "This problem ships no canonical decomposition, so there is nothing to hint at.")
    kind, index = classify(attempt, canonical)
    level = min(max(attempts_made, 1), len(LEVELS))
    if kind == "complete":
        return (level, "none", "Your attempt already matches the canonical solution - try the follow-up.")
    if kind == "extra":
        return (level, "none", f"Unexpected step {attempt[len(canonical)]!r} after the solution is complete.")
    step = canonical[index]
    lead = f"Step {index + 1} ({step.ident})"
    if kind == "diverged":
        lead = f"Step {index + 1} should be {step.ident!r}, not {attempt[index]!r}"
    if level == 1:
        return (level, step.ident, f"{lead}: focus on the {step.phase} phase next.")
    if level == 2:
        return (level, step.ident, f"{lead}: {step.title}.")
    return (level, step.ident, f"{lead}: {step.code}")


def hint_sequence(attempt, attempts=3, canonical=CANONICAL):
    return [hint(attempt, made, canonical) for made in range(1, attempts + 1)]


def solved_text(canonical=CANONICAL):
    return "\n".join(step.code for step in canonical)


def main():
    attempts = corpus()
    # 1) the common prefix matches the independent down-from-the-longest oracle on the whole corpus.
    check(all(progress(attempt)[0] == prefix_oracle(attempt) for attempt in attempts.values())
          and progress(("parse", "count"))[0] == 2 and progress(("count",))[0] == 0,
          f"longest-common-prefix oracle mismatch: {[(name, progress(at)[0], prefix_oracle(at)) for name, at in attempts.items()]}")
    # 2) classification matches the independent linear-scan oracle, case by case.
    check(all(classify(attempt) == classify_oracle(attempt) for attempt in attempts.values())
          and classify(ATTEMPTS["dev"]) == ("complete", None) and classify(ATTEMPTS["finn"]) == ("extra", 7)
          and classify(ATTEMPTS["eve"]) == ("diverged", 2),
          f"classification mismatch: {[(name, classify(at)) for name, at in attempts.items()]}")
    # 3) the hint names the first unproduced step - never a later one - for every prefix attempt.
    for length in range(len(CANONICAL)):
        attempt = tuple(step.ident for step in CANONICAL[:length])
        level, ident, text = hint(attempt, 3)
        check(ident == CANONICAL[length].ident and f"Step {length + 1}" in text
              and str(length + 2) not in text and text.count(CANONICAL[length].code) <= 1,
              f"hint for a {length}-step prefix points at the wrong step: {text}")
    # 4) escalation: level rises with failed attempts and the code appears only at the last level.
    step = CANONICAL[2]
    sequence = hint_sequence(("parse", "count"))
    check([row[0] for row in sequence] == [1, 2, 3] and [row[1] for row in sequence] == [step.ident] * 3
          and step.phase in sequence[0][2] and step.title in sequence[1][2]
          and step.title not in sequence[0][2] and step.code in sequence[2][2] and step.code not in sequence[1][2]
          and hint(("parse", "count"), 1) == sequence[0] and hint(("parse", "count"), 9) == sequence[2],
          f"hint escalation is wrong: {sequence}")
    # 5) no leak: no hint at level 1 or 2 contains any code line, nor the whole canonical solution.
    whole = solved_text()
    level_one = [hint(attempt, 1)[2] for attempt in attempts.values()]
    level_two = [hint(attempt, 2)[2] for attempt in attempts.values()]
    check(all(whole not in text and all(step.code not in text for step in CANONICAL)
              for text in level_one + level_two)
          and len({row[2] for row in sequence}) == 3,
          "a hint leaked the solution")
    # 6) repair hints: a diverging attempt is told where it went off script, never to skip ahead.
    level, ident, text = hint(ATTEMPTS["eve"], 3)
    check(ident == "sort" and "should be 'sort'" in text and "'return'" in text and CANONICAL[2].code in text
          and classify(ATTEMPTS["eve"]) == ("diverged", 2),
          f"the repair hint is wrong: {text}")
    # 7) terminals and boundaries: solved, overshoot, empty attempt, absent decomposition, empty corpus.
    check(hint(ATTEMPTS["dev"], 2)[1] == "none" and "matches the canonical" in hint(ATTEMPTS["dev"], 2)[2]
          and "Unexpected step" in hint(ATTEMPTS["finn"], 1)[2]
          and hint(ATTEMPTS["cleo"], 1)[1] == CANONICAL[0].ident and "setup" in hint(ATTEMPTS["cleo"], 1)[2]
          and {row[2] for row in hint_sequence((), 3, ())} == {hint((), 1, ())[2]} and hint((), 1, ())[1] == "none"
          and progress((), ()) == (0, None, False) and classify((), ()) == ("complete", None),
          "the solved, overshoot and empty boundaries are wrong")
    # 8) corpus coverage: every canonical step is the target of some hint, and no attempt is left unhinted.
    targets = {ident for attempt in attempts.values() for _level, ident, _text in [hint(attempt, 3)] if ident != "none"}
    check(targets == {step.ident for step in CANONICAL}
          and all(len(hint_sequence(attempt)) == 3 for attempt in attempts.values())
          and len(attempts) == len(ATTEMPTS) + len(CANONICAL),
          f"hint coverage is incomplete: {sorted(targets)}")
    # 9) determinism: two independent runs over two independently built corpora agree everywhere.
    stored = [(name, progress(attempt), classify(attempt)) for name, attempt in attempts.items()]
    repeat = [(name, progress(attempt), classify(attempt)) for name, attempt in corpus().items()]
    check(stored == repeat and hint_sequence(ATTEMPTS["ada"]) == [hint(ATTEMPTS["ada"], made) for made in (1, 2, 3)]
          and solved_text() == "\n".join(step.code for step in CANONICAL)
          and prefix_oracle(ATTEMPTS["gus"]) == progress(ATTEMPTS["gus"])[0],
          "hint generation must be reproducible")
    print(f"steps={len(CANONICAL)} attempts={len(attempts)} targets={len(targets)} levels={len(LEVELS)} "
          f"escalation={[row[0] for row in sequence]}")
    print(f"terminal={classify(ATTEMPTS['dev'])[0]},{classify(ATTEMPTS['finn'])[0]} "
          f"diverged={classify(ATTEMPTS['eve'])} sample_hint={sequence[1][2]!r}")
    print("PASS")


if __name__ == "__main__":
    main()
