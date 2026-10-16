"""Task 979 - Implement certification exam generator and proctoring integration.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Two halves of a certification platform.  The generator assembles several equivalent exam forms from an item
bank: a blueprint fixes how many items each topic contributes, items are ranked per topic by how close they
sit to the target difficulty and dealt round-robin to the forms, so every form gets the same topic counts, the
same point total and a comparable mean difficulty while forms stay disjoint.  The proctoring half audits one
session's integrity stream - heartbeat cadence, focus losses, answer latency and identity verification - into
severity-weighted findings and flags the sessions that cross a threshold.

Simplified versus a real certification system: a small in-memory item bank instead of a calibrated item pool,
no exposure control over repeated sessions, and a rule-based proctor instead of video/keystroke forensics.
Verified: exact blueprint and point-count compliance, form disjointness and difficulty equivalence against a
brute-force mean, complete recall of the planted cheating sessions with no false positive on the honest
cohort, severity arithmetic and monotonicity, findings against an independent audit oracle, empty/short-bank
boundaries and reproducibility.
"""
from collections import namedtuple

Item = namedtuple("Item", "ident topic difficulty points")
TOPICS = ("arrays", "hashing", "sorting", "graphs", "dp")
ITEMS_PER_TOPIC, FORMS, POINTS = 9, 3, 4
BLUEPRINT = {topic: 3 for topic in TOPICS}
TARGET_DIFFICULTY, TOLERANCE = 1.4, 0.2   # the blueprint target equals the bank's mean difficulty
BANK = tuple(Item(f"{topic[:2]}{index:02d}", topic, round(0.6 + 0.2 * index, 3), POINTS)
             for topic in TOPICS for index in range(ITEMS_PER_TOPIC))
RULES = {"heartbeat_gap": 45, "focus_losses": 3, "min_seconds": 8}
RULE_NAMES = ("heartbeat_gap", "focus_losses", "identity_verified", "impossible_speed")
SEVERITY = {"medium": 1, "high": 3, "critical": 5}
FLAG_THRESHOLD = 3
Session = namedtuple("Session", "ident heartbeats focus_losses answers identity_verified")
Answer = namedtuple("Answer", "item seconds")
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect(error, call, message):
    try:
        call()
    except error:
        return
    raise AssertionError(message)


class Rng:
    """splitmix64 - fixed seeds keep both the forms and the simulated sessions reproducible."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) % bound


def build_forms(bank, blueprint=BLUEPRINT, forms=FORMS, target=TARGET_DIFFICULTY):
    """Serpentine dealing of each topic's items (easiest to hardest) over the forms.

    Serpentine - left to right, then right to left - gives every form one item from each difficulty tercile
    of the topic, so the forms come out with close to the same mean difficulty while staying disjoint.
    """
    if not bank:
        raise ValueError("the item bank is empty")
    counts = {topic: sum(1 for item in bank if item.topic == topic) for topic in blueprint}
    for topic, need in blueprint.items():
        if counts.get(topic, 0) < need * forms:
            raise ValueError(f"topic {topic!r} needs {need * forms} items but holds {counts.get(topic, 0)}")
    picked = [[] for _ in range(forms)]
    for topic in sorted(blueprint):
        pool = sorted((item for item in bank if item.topic == topic),
                      key=lambda item: (abs(item.difficulty - target), item.ident))
        selected = sorted(pool[:blueprint[topic] * forms], key=lambda item: (item.difficulty, item.ident))
        for position, item in enumerate(selected):
            row, column = divmod(position, forms)
            picked[column if row % 2 == 0 else forms - 1 - column].append(item)
    return [tuple(sorted(form, key=lambda item: item.ident)) for form in picked]


def topic_counts(form):
    counts = {}
    for item in form:
        counts[item.topic] = counts.get(item.topic, 0) + 1
    return counts


def mean_difficulty(form):
    total = 0.0
    for item in form:
        total += item.difficulty
    return total / len(form) if form else 0.0


def points(form):
    return sum(item.points for item in form)


def audit(session, rules=RULES):
    """Severity-weighted findings for one proctored session (at most one per rule), in rule order."""
    findings = []
    gaps = [session.heartbeats[index] - session.heartbeats[index - 1]
            for index in range(1, len(session.heartbeats))]
    worst = max(gaps) if gaps else 0
    if worst > rules["heartbeat_gap"]:
        findings.append(("heartbeat_gap", "medium", f"largest gap {worst}s"))
    if session.focus_losses > rules["focus_losses"]:
        level = "high" if session.focus_losses >= 2 * rules["focus_losses"] else "medium"
        findings.append(("focus_losses", level, f"{session.focus_losses} focus losses"))
    if not session.identity_verified:
        findings.append(("identity_verified", "critical", "identity check failed"))
    fast = sum(1 for answer in session.answers if answer.seconds < rules["min_seconds"])
    if fast:
        findings.append(("impossible_speed", "high", f"{fast} answers under {rules['min_seconds']}s"))
    return findings


def audit_oracle(session, rules=RULES):
    """Independent oracle: the same rules, evaluated by walking the raw fields in a different order."""
    findings = []
    biggest = 0
    for position in range(len(session.heartbeats) - 1):
        pace = session.heartbeats[position + 1] - session.heartbeats[position]
        biggest = pace if pace > biggest else biggest
    if biggest >= rules["heartbeat_gap"] + 1:
        findings.append(("heartbeat_gap", "medium", f"largest gap {biggest}s"))
    losses = session.focus_losses
    if losses >= rules["focus_losses"] + 1:
        findings.append(("focus_losses", "high" if losses >= rules["focus_losses"] * 2 else "medium",
                         f"{losses} focus losses"))
    if session.identity_verified is False:
        findings.append(("identity_verified", "critical", "identity check failed"))
    rushed = sum(1 for answer in session.answers if answer.seconds < rules["min_seconds"])
    if rushed > 0:
        findings.append(("impossible_speed", "high", f"{rushed} answers under {rules['min_seconds']}s"))
    return findings


def risk_score(findings, severity=SEVERITY):
    return sum(severity[level] for _rule, level, _detail in findings)


def flagged(session, threshold=FLAG_THRESHOLD):
    return risk_score(audit(session)) >= threshold


def make_session(rng, ident, cheater, steps=30):
    """A plausible honest session, or one with one to three planted integrity violations."""
    beats, clock = [], 0
    for _ in range(steps):
        clock += 8 + rng.below(6)
        beats.append(clock)
    if cheater == "identity_verified":
        return Session(ident, tuple(beats), 0, tuple(Answer(f"a{index}", 40 + rng.below(40))
                                              for index in range(steps)), False)
    if cheater == "speed":
        beats = list(beats)
        for position in range(0, len(beats), 5):
            beats[position] = beats[position - 1] + 90 if position else 90
        return Session(ident, tuple(beats), 1, tuple(Answer(f"a{index}", 2)
                                                     for index in range(steps)), True)
    if cheater == "focus":
        return Session(ident, tuple(beats), 7, tuple(Answer(f"a{index}", 40 + rng.below(40))
                                                     for index in range(steps)), True)
    return Session(ident, tuple(beats), 0, tuple(Answer(f"a{index}", 30 + rng.below(60))
                                                 for index in range(steps)), True)


def cohort(seed=41, honest=8):
    """Simulated sessions: honest examinees plus three planted cheaters of different kinds."""
    rng = Rng(seed)
    sessions = [make_session(rng, f"s{index:02d}", None) for index in range(honest)]
    for offset, kind in enumerate(("identity_verified", "speed", "focus")):
        sessions.append(make_session(rng, f"c{offset:02d}", kind))
    return sessions


def main():
    forms = build_forms(BANK)
    # 1) blueprint compliance: every form carries exactly the blueprint counts and the full item count.
    total = sum(BLUEPRINT.values())
    check(len(forms) == FORMS and all(topic_counts(form) == BLUEPRINT for form in forms)
          and all(len(form) == total for form in forms)
          and len(BANK) == len(TOPICS) * ITEMS_PER_TOPIC == 45,
          f"blueprint compliance is wrong: {[topic_counts(form) for form in forms]}")
    # 2) forms are disjoint and together use exactly blueprint*forms items per topic.
    used = [item.ident for form in forms for item in form]
    check(len(used) == len(set(used)) == total * FORMS
          and all(sum(1 for form in forms for item in form if item.topic == topic) == BLUEPRINT[topic] * FORMS
                  for topic in TOPICS),
          "forms overlap or do not use the expected item counts")
    # 3) difficulty equivalence against a brute-force mean, and identical point totals.
    brute = [sum(item.difficulty for item in form) / len(form) for form in forms]
    check(all(abs(value - TARGET_DIFFICULTY) <= TOLERANCE for value in brute)
          and all(abs(value - mean_difficulty(form)) < 1e-12 for value, form in zip(brute, forms))
          and max(brute) - min(brute) <= TOLERANCE
          and len({points(form) for form in forms}) == 1 and points(forms[0]) == total * POINTS,
          f"the forms are not equivalent: {[round(value, 3) for value in brute]}")
    # 4) proctoring: every planted cheater is flagged, no honest session is, and recall is complete.
    sessions = cohort()
    cheaters = [session for session in sessions if session.ident.startswith("c")]
    honest_sessions = [session for session in sessions if session.ident.startswith("s")]
    detected = [session.ident for session in sessions if flagged(session)]
    check(len(cheaters) == 3 and all(flagged(session) for session in cheaters)
          and not any(flagged(session) for session in honest_sessions)
          and sorted(detected) == sorted(session.ident for session in cheaters),
          f"proctoring recall/precision is wrong: flagged={detected}")
    # 5) severity arithmetic and the independent audit oracle: the weights add up, identity alone crosses
    #    the threshold, another finding increases the score, and a second implementation agrees exactly.
    identity = next(session for session in cheaters if session.ident == "c00")
    findings = audit(identity)
    check(risk_score(findings) == sum(SEVERITY[level] for _r, level, _d in findings)
          and risk_score([("identity_verified", "critical", "x")]) >= FLAG_THRESHOLD
          and risk_score(findings + [("extra", "high", "y")]) == risk_score(findings) + SEVERITY["high"]
          and risk_score([]) == 0 and not flagged(honest_sessions[0])
          and all(audit(session) == audit_oracle(session) for session in sessions)
          and all(set(rule for rule, _l, _d in audit(session)) <= set(RULE_NAMES) for session in sessions),
          f"severity scoring or the audit oracle is wrong: {findings}")
    # 6) boundaries: a single heartbeat with no answers and no focus losses is a clean session; a short or
    #    empty bank is rejected instead of silently producing a short form.
    quiet = Session("quiet", (0,), 0, (), True)
    check(audit(quiet) == [] and not flagged(quiet) and risk_score(audit(quiet)) == 0
          and audit(Session("x", (0, 100), 0, (Answer("a", 1),), False))
          == audit_oracle(Session("x", (0, 100), 0, (Answer("a", 1),), False))
          and len(audit(Session("e", (), 0, (), True))) == 0,
          f"the quiet-session boundary is wrong: {audit(quiet)}")
    expect(ValueError, lambda: build_forms(()), "an empty bank must be rejected")
    expect(ValueError, lambda: build_forms(BANK[:4]), "a bank without enough items must be rejected")
    # 9) a smaller number of forms is still satisfied exactly when the bank allows it, and the whole
    #    pipeline - forms, cohort, audits and means - reproduces exactly.
    two_forms = build_forms(BANK, forms=2)
    check(len(two_forms) == 2 and len(two_forms[0]) == total and all(topic_counts(form) == BLUEPRINT
                                                                     for form in two_forms)
          and not {item.ident for item in two_forms[0]} & {item.ident for item in two_forms[1]}
          and build_forms(BANK) == forms and cohort() == sessions and build_forms(BANK, forms=2) != forms
          and all(abs(left - right) < 1e-12 for left, right in
                  zip([mean_difficulty(form) for form in forms], brute)),
          "a smaller number of forms or the reproducibility is wrong")
    print(f"bank={len(BANK)} forms={len(forms)} items={total} points={points(forms[0])} "
          f"mean_difficulty={[round(value, 3) for value in brute]} sessions={len(sessions)} "
          f"flagged={len(detected)} cheaters={len(cheaters)} honest={len(honest_sessions)} "
          f"findings={[len(audit(session)) for session in sessions]}")
    print("PASS")


if __name__ == "__main__":
    main()
