"""Task 982 - Implement monitoring and alerting playbooks for platform incidents.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A miniature alerting stack.  Metric series are sampled on a fixed cadence; every rule names a metric, a
threshold and a comparison and demands that the condition hold continuously for a `for` duration before it may
fire, so a brief spike only ever reaches the pending state.  Firing starts the rule's playbook steps in order,
the alert leaves firing only after the condition has been clear for a resolve duration, long incidents
escalate to the next on-call level, and inhibit rules suppress the notifications of a dependent alert while a
more fundamental alert is firing - without losing that alert's own state.

Simplified versus Prometheus/Alertmanager: in-memory series with integer step indices instead of timestamps,
no alert grouping, silencing windows or cross-restart deduplication.  Verified: the for-duration requirement,
resolution only after the clear window, the transition graph and a hand-checked timeline, playbook ordering,
escalation levels, inhibition with recovery, an independent scan oracle for the firing windows, the
exact-threshold comparison, empty/single-sample/missing-metric boundaries, notification accounting and
reproducibility.
"""
from collections import namedtuple

Rule = namedtuple("Rule", "name metric severity threshold direction for_steps resolve_steps "
                          "escalation_steps steps")
SAMPLES = 60                     # one sample per 15s step: a 15-minute window
CPU_SPIKE, CPU_SHORT_SPIKE = tuple(range(10, 23)), (45, 46)
LATENCY_BREACH, DISK_BREACH = tuple(range(12, 45)), tuple(range(30, SAMPLES))
RULES = (
    Rule("cpu-saturated", "cpu", "critical", 90.0, "above", 4, 5, 10,
         ("page the db on-call", "check the query plan")),
    Rule("latency-high", "latency", "warning", 800.0, "above", 3, 4, 12, ("inspect the edge cache",)),
    Rule("disk-full", "disk", "warning", 85.0, "above", 6, 3, 20, ("rotate the logs",)),
    Rule("queue-empty", "queue", "warning", 5.0, "below", 2, 2, 20, ("check the consumers",)),
    Rule("cpu-short", "cpu", "info", 96.0, "above", 6, 2, 30, ("watch only",)),
)
INHIBIT = (("cpu-saturated", "latency-high"),)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def breaches(value, direction, threshold):
    """>= for 'above' and <= for 'below', so a value sitting exactly on the threshold counts as a breach."""
    return value >= threshold if direction == "above" else value <= threshold


def series():
    """Deterministic series with one planted incident per rule plus a spike that is too short to fire."""
    cpu = [40.0 + (step % 5) for step in range(SAMPLES)]
    for step in CPU_SPIKE:
        cpu[step] = 95.0 + (step % 3)
    for step in CPU_SHORT_SPIKE:
        cpu[step] = 98.0
    latency = [420.0 + 3 * step for step in range(SAMPLES)]
    for step in LATENCY_BREACH:
        latency[step] = 900.0
    disk = [60.0 + step * 0.5 for step in range(SAMPLES)]
    for step in DISK_BREACH:
        disk[step] = 88.0
    return {"cpu": cpu, "latency": latency, "disk": disk, "queue": [30.0] * SAMPLES}


def walk(rule, values):
    """Drive one rule's state machine: returns (per-step states, [(first breach, last breach)] episodes)."""
    timeline, episodes = [], []
    state, breach_run, clear_run, start = "inactive", 0, 0, None
    for step, value in enumerate(values):
        breached = breaches(value, rule.direction, rule.threshold)
        breach_run = breach_run + 1 if breached else 0
        clear_run = 0 if breached else clear_run + 1
        if state == "firing":
            if clear_run >= rule.resolve_steps:
                state, episodes, start = "inactive", episodes + [(start, step - rule.resolve_steps)], None
        elif breach_run >= rule.for_steps:
            state, start = "firing", step - rule.for_steps + 1
        else:
            state = "pending" if breached else "inactive"
        timeline.append(state)
    if state == "firing":
        episodes.append((start, len(values) - 1))
    return timeline, episodes


def evaluate(rules, metrics, inhibit=INHIBIT):
    """Walk every rule, run its playbook, escalate long incidents and suppress inhibited notifications."""
    log, timelines, windows, suppressed = [], {}, {}, []
    for rule in rules:
        values = metrics.get(rule.metric)
        if values is None:
            continue
        timeline, episodes = walk(rule, values)
        timelines[rule.name] = timeline
        windows[rule.name] = episodes
    for rule in rules:
        for start, stop in windows.get(rule.name, []):
            resolved_at = stop + rule.resolve_steps
            for order, step in enumerate(rule.steps, 1):
                log.append((start, rule.name, "playbook", order, step))
            blocker = next((source for source, target in inhibit if target == rule.name
                            and any(begin <= start <= finish for begin, finish in windows.get(source, []))), None)
            if blocker:
                suppressed.append((start, rule.name, blocker))
            else:
                log.append((start, rule.name, "firing", 0, rule.severity))
            for level in (1, 2):
                at = start + level * rule.escalation_steps
                if at <= stop:
                    log.append((at, rule.name, "escalate", level, f"level {level} on-call"))
            log.append((resolved_at, rule.name, "resolved", 0, rule.severity))
    return {"log": sorted(log), "timelines": timelines, "windows": windows, "suppressed": suppressed}


def windows_oracle(rule, values):
    """Independent oracle: count the consecutive breach run per step and keep the runs that reach the
    duration, reporting the first and last breached step of each such run."""
    windows, run = [], 0
    for step in range(len(values)):
        if breaches(values[step], rule.direction, rule.threshold):
            run += 1
            if run == rule.for_steps:
                windows.append([step - rule.for_steps + 1, step])
            elif run > rule.for_steps:
                windows[-1][1] = step
        else:
            run = 0
    return [tuple(window) for window in windows]


def rule_named(name):
    return next(rule for rule in RULES if rule.name == name)


def main():
    metrics = series()
    outcome = evaluate(RULES, metrics)
    windows, log, timelines = outcome["windows"], outcome["log"], outcome["timelines"]
    # 1) the for-duration requirement: the two-step spike and the healthy queue metric never fire, while
    #    each planted incident produces exactly one firing window.
    check(windows == {"cpu-saturated": [(10, 22)], "latency-high": [(12, 44)], "disk-full": [(30, 59)],
                      "queue-empty": [], "cpu-short": []},
          f"the firing windows are wrong: {windows}")
    # 2) the states around the boundary: pending while the breach run is shorter than the duration, firing
    #    once it reaches it, and firing until the clear window has elapsed.
    cpu = timelines["cpu-saturated"]
    check(cpu[:10] == ["inactive"] * 10 and cpu[10:13] == ["pending"] * 3 and cpu[13:27] == ["firing"] * 14
          and cpu[27:45] == ["inactive"] * 18 and cpu[45] == "pending" and cpu[47:] == ["inactive"] * 13
          and timelines["latency-high"][13] == "pending" and timelines["latency-high"][14] == "firing"
          and timelines["disk-full"][34] == "pending" and timelines["disk-full"][35] == "firing",
          f"the pending/firing boundary is wrong: {cpu[:32]}")
    # 3) the transition graph is respected for every rule, and the machine never skips a state.
    allowed = {("inactive", "inactive"), ("inactive", "pending"), ("inactive", "firing"),
               ("pending", "pending"), ("pending", "firing"), ("pending", "inactive"),
               ("firing", "firing"), ("firing", "inactive")}
    check(all((timeline[index - 1], timeline[index]) in allowed
              for timeline in timelines.values() for index in range(1, SAMPLES))
          and all(len(timeline) == SAMPLES for timeline in timelines.values())
          and set(timelines) == {rule.name for rule in RULES},
          "the alert state machine took an illegal transition")
    # 4) playbooks run in order at the firing start, and exactly one firing notification is logged per
    #    unsuppressed episode.
    playbook = [entry for entry in log if entry[1] == "cpu-saturated" and entry[2] == "playbook"]
    check([entry[3] for entry in playbook] == [1, 2] and {entry[0] for entry in playbook} == {10}
          and [entry[4] for entry in playbook] == ["page the db on-call", "check the query plan"]
          and len([entry for entry in log if entry[2] == "firing"]) == 2
          and len([entry for entry in log if entry[2] == "resolved"]) == 3,
          f"the playbook log is wrong: {playbook}")
    # 5) escalation: the long incident escalates through both levels, the short ones do not escalate.
    escalations = [(entry[1], entry[3]) for entry in log if entry[2] == "escalate"]
    check(("cpu-saturated", 1) in escalations and ("cpu-saturated", 2) not in escalations
          and ("latency-high", 1) in escalations and ("latency-high", 2) in escalations
          and ("disk-full", 1) in escalations and not any(name == "cpu-short" for name, _l in escalations),
          f"escalation is wrong: {escalations}")
    # 6) inhibition: the dependent alert is suppressed while the source fires, keeps its own state, and its
    #    inbox contains no firing notification for that episode.
    blocked = outcome["suppressed"]
    check(blocked == [(12, "latency-high", "cpu-saturated")] and timelines["latency-high"][14] == "firing"
          and not [entry for entry in log if entry[1] == "latency-high" and entry[2] == "firing"]
          and any(entry[1] == "latency-high" and entry[2] == "resolved" for entry in log),
          f"inhibition is wrong: suppressed={blocked}")
    # 7) oracle: an independent run-length scan reproduces every window, including one that never ends.
    check(all(windows_oracle(rule_named(rule.name), metrics[rule.metric]) == windows[rule.name]
              for rule in RULES)
          and windows_oracle(Rule("x", "m", "info", 1.0, "above", 1, 1, 5, ()), [0.0, 2.0, 0.0]) == [(1, 1)]
          and windows_oracle(Rule("x", "m", "info", 5.0, "above", 2, 1, 5, ()), [5.0, 5.0, 4.0]) == [(0, 1)],
          f"the windows disagree with the oracle: {windows}")
    # 8) boundaries: an empty metric map, a single sample, a missing metric, and the exact-threshold rule.
    empty = evaluate(RULES, {})
    single = evaluate((Rule("one", "m", "info", 5.0, "above", 1, 1, 5, ("act",)),), {"m": [5.0]})
    short = evaluate((Rule("two", "m", "info", 5.0, "above", 2, 1, 5, ()),), {"m": [5.0]})
    check(empty == {"log": [], "timelines": {}, "windows": {}, "suppressed": []}
          and single["windows"]["one"] == [(0, 0)] and len(single["log"]) == 3
          and short["windows"]["two"] == [] and short["log"] == []
          and breaches(5.0, "above", 5.0) and not breaches(4.999, "above", 5.0)
          and breaches(5.0, "below", 5.0) and walk(RULES[0], []) == ([], []),
          f"the empty and single-sample boundaries are wrong: {single}")
    # 9) accounting and reproducibility: every log entry belongs to a rule, is unique, names a rule that is
    #    pending, firing or inactive at that step, and the whole evaluation reproduces exactly.
    owned = all(entry[1] in timelines and timelines[entry[1]][min(entry[0], SAMPLES - 1)] in
                ("pending", "firing", "inactive") for entry in log)
    check(owned and len(log) == len(set(log)) == 13 and evaluate(RULES, metrics) == outcome
          and series() == metrics and len(outcome["log"]) == sum(1 for _ in outcome["log"]),
          f"the notification log is inconsistent: {len(log)} entries, {len(set(log))} unique")
    print(f"rules={len(RULES)} samples={SAMPLES} windows={ {name: windows[name] for name in sorted(windows)} }")
    print(f"log_entries={len(log)} suppressed={blocked} escalations={escalations}")
    print("PASS")


if __name__ == "__main__":
    main()
