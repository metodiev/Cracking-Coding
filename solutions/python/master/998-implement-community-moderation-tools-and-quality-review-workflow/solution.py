"""Task 998 - Implement community moderation tools and quality review workflow.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A moderation queue with an explicit state machine (NEW -> TRIAGED -> IN_REVIEW -> RESOLVED or
ESCALATED).  Reports are triaged to the least-loaded reviewer with matching expertise who is not
the report's own author, so a conflict of interest can never decide an outcome; resolutions need
a quorum of distinct votes plus a strict majority, and anything still open at its SLA deadline is
escalated automatically.  Every state change lands in a digest-chained audit log.

Simplified versus production: the clock is a caller-driven integer tick (no wall time) and
reviewer load is an in-memory counter.  Verified: conflict-of-interest exclusion, least-loaded
assignment against a brute-force oracle, quorum/majority boundaries, illegal-transition and
double-vote rejection, the exact SLA deadline, audit-chain tampering, determinism, and emptiness.
"""
import hashlib
import json

TRANSITIONS = {("NEW", "TRIAGED"), ("TRIAGED", "IN_REVIEW"), ("TRIAGED", "ESCALATED"),
               ("IN_REVIEW", "RESOLVED"), ("IN_REVIEW", "ESCALATED")}
TERMINAL = ("RESOLVED", "ESCALATED")
SLA_BY_SEVERITY = {1: 24, 2: 12, 3: 6}
QUORUM = 2


class ModerationError(ValueError):
    """Raised when a moderation action would break the workflow rules."""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, needle):
    try:
        action()
    except ModerationError as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected a ModerationError mentioning %r" % needle)


def digest_of(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ReviewQueue:
    def __init__(self, reviewers):
        self.reviewers = {reviewer["id"]: {"expertise": set(reviewer["expertise"]), "load": 0}
                          for reviewer in reviewers}
        self.reports = {}
        self.audit = []

    # -- workflow ---------------------------------------------------------------
    def submit(self, report):
        for field in ("id", "author", "topic", "severity", "created_tick"):
            if field not in report:
                raise ModerationError("report is missing %r" % (field,))
        if report["id"] in self.reports:
            raise ModerationError("report %r was already submitted" % (report["id"],))
        if report["severity"] not in SLA_BY_SEVERITY:
            raise ModerationError("report %r has unknown severity %r"
                                  % (report["id"], report["severity"]))
        self.reports[report["id"]] = {"report": dict(report), "state": "NEW",
                                      "assignee": None, "votes": {}, "decision": None}
        self._log(0, report["id"], "NEW", "system")
        return report["id"]

    def _log(self, tick, report_id, action, actor):
        previous = self.audit[-1]["digest"] if self.audit else "genesis"
        entry = {"seq": len(self.audit), "tick": tick, "report": report_id,
                 "action": action, "actor": actor, "prev": previous}
        entry["digest"] = digest_of({key: entry[key] for key in entry})
        self.audit.append(entry)
        return entry["digest"]

    def transition(self, report_id, target, actor, tick):
        record = self.reports[report_id]
        if (record["state"], target) not in TRANSITIONS:
            raise ModerationError("cannot move %s from %s to %s"
                                  % (report_id, record["state"], target))
        record["state"] = target
        self._log(tick, report_id, target, actor)
        return target

    def deadline(self, report_id):
        record = self.reports[report_id]["report"]
        return record["created_tick"] + SLA_BY_SEVERITY[record["severity"]]

    # -- assignment and review ---------------------------------------------------
    def candidates(self, report_id):
        record = self.reports[report_id]["report"]
        return [(reviewer_id, self.reviewers[reviewer_id]["load"])
                for reviewer_id, reviewer in sorted(self.reviewers.items())
                if record["topic"] in reviewer["expertise"] and reviewer_id != record["author"]
                and reviewer_id not in self.reports[report_id]["votes"]]

    def assign(self, report_id, tick):
        record = self.reports[report_id]
        if record["state"] == "NEW":
            self.transition(report_id, "TRIAGED", "system", tick)
        available = self.candidates(report_id)
        if not available:
            raise ModerationError("no eligible reviewer for %s (topic %r, author %r)"
                                  % (report_id, record["report"]["topic"], record["report"]["author"]))
        chosen = min(available, key=lambda pair: (pair[1], pair[0]))[0]
        self.reviewers[chosen]["load"] += 1
        record["assignee"] = chosen
        self.transition(report_id, "IN_REVIEW", chosen, tick)
        return chosen

    def vote(self, report_id, reviewer_id, decision, tick):
        record = self.reports[report_id]
        if record["state"] != "IN_REVIEW":
            raise ModerationError("cannot vote on %s while it is %s" % (report_id, record["state"]))
        if reviewer_id == record["report"]["author"]:
            raise ModerationError("the author of %s cannot vote on their own report" % (report_id,))
        if reviewer_id in record["votes"]:
            raise ModerationError("%s already voted on %s" % (reviewer_id, report_id))
        if decision not in ("uphold", "dismiss"):
            raise ModerationError("unknown decision %r" % (decision,))
        record["votes"][reviewer_id] = decision
        self._log(tick, report_id, "vote:%s:%s" % (reviewer_id, decision), reviewer_id)
        upheld = sum(1 for value in record["votes"].values() if value == "uphold")
        dismissed = len(record["votes"]) - upheld
        if len(record["votes"]) >= QUORUM and upheld != dismissed:
            record["decision"] = "uphold" if upheld > dismissed else "dismiss"
            self.transition(report_id, "RESOLVED", "quorum", tick)
        return record["state"], record["decision"]

    def escalate_due(self, tick):
        escalated = []
        for report_id in sorted(self.reports):
            record = self.reports[report_id]
            if record["state"] not in TERMINAL and self.deadline(report_id) <= tick:
                self.transition(report_id, "ESCALATED", "scheduler", tick)
                escalated.append(report_id)
        return escalated

    def verify_audit(self):
        """Recompute the chain: any rewritten or relinked entry is reported."""
        problems = []
        for index, entry in enumerate(self.audit):
            expected_prev = "genesis" if index == 0 else self.audit[index - 1]["digest"]
            if entry["seq"] != index or entry["prev"] != expected_prev:
                problems.append("%d: broken chain" % index)
            body = {key: entry[key] for key in entry if key != "digest"}
            if digest_of(body) != entry["digest"]:
                problems.append("%d: digest mismatch" % index)
        return problems

    def audit_digest(self):
        return digest_of([entry["digest"] for entry in self.audit])


def reviewers():
    return [{"id": "ada", "expertise": ["algorithms", "databases"]},
            {"id": "grace", "expertise": ["algorithms", "tooling"]},
            {"id": "linus", "expertise": ["algorithms"]},
            {"id": "barbara", "expertise": ["databases", "tooling"]}]


def report(rid, author, topic, severity, created_tick=0):
    return {"id": rid, "author": author, "topic": topic, "severity": severity,
            "created_tick": created_tick}


def main():
    queue = ReviewQueue(reviewers())
    check(queue.submit(report("r1", "mallory", "algorithms", 2, 0)) == "r1",
          "a well-formed report must be accepted")

    # Conflict of interest: the author is never a candidate for their own report.
    check("mallory" not in [name for name, _ in queue.candidates("r1")],
          "the author must not appear among the candidates")
    solo = ReviewQueue([{"id": "mallory", "expertise": ["algorithms"]}]); solo.submit(report("r9", "mallory", "algorithms", 1))
    expect_error(lambda: solo.assign("r9", 1), "no eligible reviewer")
    # Least-loaded assignment, cross-checked by a brute-force minimum over the candidates.
    queue.submit(report("r2", "mallory", "databases", 1, 0))
    first = queue.assign("r1", 1)
    check(first == "ada", "the alphabetical tie-break must pick ada, got %r" % (first,))
    oracle = min(queue.candidates("r2"), key=lambda pair: (pair[1], pair[0]))[0]
    second = queue.assign("r2", 1)
    check(second == "barbara" and second == oracle
          and queue.reviewers["ada"]["load"] == 1 and queue.reviewers["grace"]["load"] == 0
          and queue.reports["r1"]["state"] == "IN_REVIEW" and queue.reports["r1"]["assignee"] == first,
          "assignment must take the least-loaded eligible reviewer and charge only that load")

    # Audit log: every state change is recorded in order and the chain verifies.
    actions = [entry["action"] for entry in queue.audit]
    check(queue.verify_audit() == [] and actions ==
          ["NEW", "NEW", "TRIAGED", "IN_REVIEW", "TRIAGED", "IN_REVIEW"],
          "the audit log must verify and record submissions, triage and assignment: %r" % (actions,))
    tampered = [dict(entry) for entry in queue.audit]
    tampered[2]["actor"] = "mallory"
    probe = ReviewQueue(reviewers())
    probe.audit = tampered
    check(queue.verify_audit() == [] and probe.verify_audit() == ["2: digest mismatch"],
          "rewriting an audit actor must be detected, got %r" % (probe.verify_audit(),))

    # Quorum and majority: one vote keeps the report open, a tie blocks, a majority resolves.
    check(queue.vote("r1", "grace", "uphold", 2) == ("IN_REVIEW", None)          and queue.vote("r1", "linus", "dismiss", 2) == ("IN_REVIEW", None),
          "a single vote or a 1-1 split must not resolve the report")
    state, decision = queue.vote("r1", "barbara", "uphold", 2)
    check((state, decision) == ("RESOLVED", "uphold"), "a 2-1 majority must resolve as uphold")
    expect_error(lambda: queue.vote("r1", "ada", "uphold", 3), "while it is RESOLVED")
    expect_error(lambda: queue.vote("r2", "mallory", "uphold", 3), "cannot vote on their own")
    queue.vote("r2", "barbara", "dismiss", 3)
    expect_error(lambda: queue.vote("r2", "barbara", "uphold", 3), "already voted")

    # Illegal transitions are refused instead of silently applied.
    expect_error(lambda: queue.transition("r1", "IN_REVIEW", "ada", 4), "cannot move r1")
    expect_error(lambda: queue.transition("r2", "TRIAGED", "ada", 4), "cannot move r2")
    expect_error(lambda: queue.submit({"id": "r3", "author": "x", "topic": "y"}), "missing")
    expect_error(lambda: queue.submit(report("r2", "x", "algorithms", 1)), "already submitted")
    expect_error(lambda: queue.submit(report("r4", "x", "algorithms", 9)), "unknown severity")

    # SLA boundary: severity 3 escalates at exactly created_tick + 6, not one tick earlier.
    queue.submit(report("r5", "mallory", "tooling", 3, 10))
    queue.assign("r5", 10)
    check(queue.deadline("r5") == 16 and queue.escalate_due(15) == []
          and queue.escalate_due(16) == ["r5"] and queue.reports["r5"]["state"] == "ESCALATED",
          "a report must escalate exactly at its SLA deadline, never one tick earlier")
    check(queue.escalate_due(1000) == ["r2"] and queue.reports["r1"]["state"] == "RESOLVED",
          "a long gap must escalate the open report and leave the resolved one alone")

    # Determinism: replaying the same script produces the same audit digest.
    def script():
        replay = ReviewQueue(reviewers())
        replay.submit(report("a", "mallory", "algorithms", 2, 0)); replay.submit(report("b", "mallory", "databases", 1, 0))
        replay.assign("a", 1); replay.assign("b", 1)
        replay.vote("a", "grace", "uphold", 2); replay.vote("a", "linus", "uphold", 2)
        return replay

    check(script().audit_digest() == script().audit_digest() and len(queue.audit_digest()) == 64,
          "the workflow must be deterministic and the audit digest a full sha256")

    # Boundary: an empty queue accepts nothing, escalates nothing and has an empty audit.
    empty = ReviewQueue(reviewers())
    check(empty.reports == {} and empty.audit == [] and empty.escalate_due(10_000) == []
          and empty.verify_audit() == [],
          "an empty queue must be inert but well-formed")
    print("audit_entries=%d state_r1=%s decision=%s digest=%s"
          % (len(queue.audit), queue.reports["r1"]["state"], queue.reports["r1"]["decision"],
             queue.audit_digest()[:16]))
    print("PASS")


if __name__ == "__main__":
    main()
