"""Task 865 - Implement transparent model serving with versioning, canary, shadow tests, and rollback.  (Tier: Master, Beyond - Machine Learning & AI)

A serving cluster holds several model versions, routes every request to exactly one of them,
copies a deterministic sample of production traffic to shadow versions whose answers are thrown
away, and runs a monitor that rolls the canary back when its measured quality or latency
regresses.  Routing is sticky: a user is placed in one of 1000 buckets by an FNV-1a hash of the
user id, and the canary owns the lowest `weight * 1000` buckets, so raising or lowering the
canary weight only moves the users on the boundary and every decision is reproducible.

What is simplified: there are no processes, sockets or timers.  Traffic is an explicit list of
(user, features) requests replayed in order, "time" is the monotone request counter, latency is a
deterministic per-version cost model plus a per-request jitter drawn from a seeded splitmix64
stream, a model is a linear scorer, and the monitor is evaluated after every request instead of
on a wall-clock schedule.  Errors are drawn deterministically from (version, user) so that two
configurations see exactly the same failure pattern and can be compared fairly.

Invariants verified (each against an independent recomputation or a negative control):
  * versioning and atomicity - every request is served by exactly one live version, the version
    reported in the response is a deployed, non-retired one, and the scores equal the linear
    scorers evaluated independently in the test;
  * routing accuracy and stickiness - the measured canary share matches the analytic bucket
    expectation and the set of users the specification assigns to the canary, is monotone in the
    configured weight, and raising the weight never moves a user off the canary (nor lowering it
    off the primary);
  * shadow isolation - a shadow version's answers never reach the user: enabling shadow tests
    leaves every user-visible (version, score, outcome) unchanged, the shadow's own failures do
    not appear in the service level metrics, and shadow cost stays inside the configured sample;
  * rollback correctness - a canary that regresses in quality or latency is rolled back exactly
    once the minimum sample size is reached, a healthy canary is never rolled back (calibrated
    negative control), a canary below the minimum sample size is never rolled back, and after
    rollback no request ever reaches the retired version again;
  * promotion and auditability - promotion is refused while the gate is unhappy or the sample is
    too small, succeeds when the canary passes, swaps roles atomically, and the hash-chained
    audit log verifies (and detects tampering);
  * edge cases and determinism - weight 0, weight 1, unknown users, duplicate deploys, serving
    before any deploy, rollback without a canary, and a replayed workload that reproduces the
    same per-user routing and the same audit chain hash for hash.
"""

MASK64 = (1 << 64) - 1
BUCKETS = 1000
MIN_CANARY_SAMPLES = 200
MIN_BASELINE_SAMPLES = 200
ERROR_MARGIN = 0.02
LATENCY_SLO_MS = 60.0
SHADOW_SAMPLE_PER_MILLE = 200

LIVE = ("primary", "canary")
SHADOW = "shadow"
STANDBY = "standby"
RETIRED = "retired"


def fnv1a(*parts):
    """FNV-1a over the text of every part: stable across processes, unlike hash()."""
    digest = 0xCBF29CE484222325
    for part in parts:
        for character in str(part):
            digest = ((digest ^ ord(character)) * 0x100000001B3) & MASK64
    return digest


def bucket_of(user, salt="route"):
    return fnv1a(salt, user) % BUCKETS


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    def uniform(self):
        return self.next_u64() / float(1 << 64)


class Report:
    """One served request: which version answered, with what result, and what the shadows said."""

    def __init__(self, user, version, score, ok, latency, shadows):
        self.user = user
        self.version = version
        self.score = score
        self.ok = ok
        self.latency = latency
        self.shadows = shadows

    def visible(self):
        return (self.version, round(self.score, 9), self.ok)


def draw(user, salt):
    """The single random source of the data plane: a uniform in [0, 1) per (salt, user)."""
    return SplitMix64(fnv1a(salt, user)).uniform()


class ModelVersion:
    def __init__(self, name, weights, bias, error_rate, base_latency):
        self.name = name
        self.weights = list(weights)
        self.bias = bias
        self.error_rate = error_rate
        self.base_latency = base_latency

    def score(self, features):
        return self.bias + sum(weight * value for weight, value in zip(self.weights, features))


class ServingCluster:
    """Control plane (deploy / canary / promote / rollback) plus data plane (route / serve)."""

    def __init__(self):
        self.versions = {}
        self.roles = {}
        self.primary = None
        self.canary = None
        self.canary_cutoff = 0
        self.shadows = []
        self.stats = {}
        self.window = {}
        self.served = 0
        self.audit = []
        self.monitor_fired = []

    # -- accounting ----------------------------------------------------------
    def _counter(self, name):
        self.stats.setdefault(name, {"requests": 0, "errors": 0, "latency": 0.0})
        self.window.setdefault(name, {"requests": 0, "errors": 0, "latency": 0.0})
        return self.stats[name], self.window[name]

    def error_rate(self, name):
        counters = self.stats[name]
        return counters["errors"] / counters["requests"] if counters["requests"] else 0.0

    def mean_latency(self, name):
        counters = self.stats[name]
        return counters["latency"] / counters["requests"] if counters["requests"] else 0.0

    def window_requests(self, name):
        return self.window[name]["requests"]

    # -- audit ---------------------------------------------------------------
    def log(self, event, detail):
        previous = self.audit[-1]["hash"] if self.audit else "genesis"
        sequence = len(self.audit)
        digest = fnv1a(previous, event, detail, sequence)
        self.audit.append({"seq": sequence, "event": event, "detail": detail,
                           "prev": previous, "hash": digest})
        return digest

    def chain_valid(self):
        previous = "genesis"
        for index, entry in enumerate(self.audit):
            if entry["seq"] != index or entry["prev"] != previous:
                return False
            if entry["hash"] != fnv1a(previous, entry["event"], entry["detail"], index):
                return False
            previous = entry["hash"]
        return True

    def chain_head(self):
        return self.audit[-1]["hash"] if self.audit else "genesis"

    # -- control plane -------------------------------------------------------
    def deploy(self, model, role=STANDBY):
        if model.name in self.versions:
            raise ValueError("version %s is already deployed" % model.name)
        if role not in (STANDBY, SHADOW, "primary"):
            raise ValueError("unknown role %s" % role)
        self.versions[model.name] = model
        self.roles[model.name] = role
        self._counter(model.name)
        if role == SHADOW:
            self.shadows = sorted(self.shadows + [model.name])
        if role == "primary":
            self.activate(model.name)
        self.log("deploy", "%s as %s" % (model.name, role))
        return model.name

    def activate(self, name, role="primary"):
        if role not in LIVE:
            raise ValueError("only a primary or canary has live traffic")
        if name not in self.versions:
            raise ValueError("unknown version %s" % name)
        if self.roles[name] == RETIRED:
            raise ValueError("a retired version cannot be reactivated")
        if role == "primary":
            if self.primary is not None and self.primary != name:
                self.roles[self.primary] = STANDBY
            self.primary = name
        else:
            self.canary = name
        self.roles[name] = role
        return name

    def set_canary(self, name, weight):
        """Point the canary at `name` with `weight` of the traffic (0 <= weight <= 1)."""
        if not 0.0 <= weight <= 1.0:
            raise ValueError("canary weight must be a fraction")
        if name is not None:
            self.activate(name, "canary")
        self.canary = name
        self.canary_cutoff = int(round(weight * BUCKETS))
        if name is not None:
            self.window[name] = {"requests": 0, "errors": 0, "latency": 0.0}
        self.log("canary", "%s at %d per mille" % (name, self.canary_cutoff))
        return self.canary_cutoff

    def rollback(self, reason):
        if self.canary is None:
            self.log("rollback-skipped", reason)
            return False
        name = self.canary
        self.roles[name] = RETIRED
        self.canary = None
        self.canary_cutoff = 0
        self.log("rollback", "%s retired: %s" % (name, reason))
        return True

    def health_action(self):
        """The monitor.  Returns None while it is too early to judge, else an action."""
        if self.canary is None:
            return None
        window = self.window[self.canary]
        primary_window = self.window[self.primary]
        if window["requests"] < MIN_CANARY_SAMPLES:
            return None
        if primary_window["requests"] < MIN_BASELINE_SAMPLES:
            return None  # no baseline to compare against, so no judgement either way
        canary_rate = window["errors"] / window["requests"]
        primary_rate = primary_window["errors"] / primary_window["requests"]
        if canary_rate > primary_rate + ERROR_MARGIN:
            return "quality"
        if window["latency"] / window["requests"] > LATENCY_SLO_MS:
            return "latency"
        return "healthy"

    def promote(self):
        """Make the canary the primary, but only when the gate has enough evidence that it is
        healthy."""
        if self.canary is None:
            self.log("promote-refused", "no canary")
            return False
        verdict = self.health_action()
        if verdict != "healthy":
            self.log("promote-refused", "%s (%s)" % (self.canary, verdict))
            return False
        promoted = self.canary
        self.roles[self.primary] = STANDBY
        self.primary = promoted
        self.roles[promoted] = "primary"
        self.canary = None
        self.canary_cutoff = 0
        self.log("promote", promoted)
        return True

    # -- data plane ----------------------------------------------------------
    def route(self, user):
        """Sticky bucketed routing: the canary owns the lowest buckets."""
        if self.primary is None:
            raise RuntimeError("no primary version is deployed")
        if self.canary is not None and bucket_of(user) < self.canary_cutoff:
            return self.canary
        return self.primary

    def _execute(self, name, user, features):
        model = self.versions[name]
        if self.roles[name] == RETIRED:
            raise AssertionError("a retired version must never execute (%s)" % name)
        score = model.score(features)
        ok = draw(user, "error:" + name) >= model.error_rate
        latency = model.base_latency + 5.0 * draw(user, "latency:" + name)
        cumulative, window = self._counter(name)
        for counters in (cumulative, window):
            counters["requests"] += 1
            counters["errors"] += 0 if ok else 1
            counters["latency"] += latency
        return score, ok, latency

    def serve(self, user, features):
        """Serve one request; shadow copies are executed but never shown to the user."""
        name = self.route(user)
        shadow_reports = []
        for shadow in self.shadows:
            if bucket_of(user, "shadow:" + shadow) < SHADOW_SAMPLE_PER_MILLE:
                score, ok, latency = self._execute(shadow, user, features)
                shadow_reports.append((shadow, round(score, 9), ok, latency))
        score, ok, latency = self._execute(name, user, features)
        self.served += 1
        action = self.health_action()
        if action not in (None, "healthy"):
            self.monitor_fired.append((self.served, action))
            self.rollback("%s regression measured on %d requests"
                          % (action, self.window[self.canary]["requests"]
                             if self.canary else 0))
        return Report(user, name, score, ok, latency, tuple(shadow_reports))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def make_versions():
    return {
        "v1": ModelVersion("v1", (1.5, -0.5, 2.0), 0.25, 0.02, 25.0),
        "v2": ModelVersion("v2", (1.4, -0.4, 2.1), 0.30, 0.005, 22.0),
        "v3": ModelVersion("v3", (1.6, -0.6, 1.9), 0.20, 0.20, 24.0),
        "v4": ModelVersion("v4", (1.5, -0.5, 2.0), 0.25, 0.01, 90.0),
        "v5": ModelVersion("v5", (0.1, 0.1, 0.1), -3.0, 0.50, 30.0),
    }


def workload(users, rng):
    return [(user, [round(rng.uniform() * 4.0 - 2.0, 6) for _ in range(3)]) for user in users]


def users_of(count):
    return ["u%04d" % index for index in range(count)]


def run(cluster, requests):
    return [cluster.serve(user, features) for user, features in requests]


def main():
    versions = make_versions()
    people = users_of(600)
    requests = workload(people, SplitMix64(20250101))
    features_of = dict(requests)
    check(len(requests) == 600 and len(features_of) == 600, "the workload must contain every user once")

    # ---- 1. versioning: one live version per request, scores from the oracle ----
    cluster = ServingCluster()
    cluster.deploy(versions["v1"], role="primary")
    check(cluster.primary == "v1" and cluster.roles["v1"] == "primary",
          "deploying as primary must activate the version")
    duplicate_raised = False
    try:
        cluster.deploy(ModelVersion("v1", (0.0,), 0.0, 0.0, 1.0))
    except ValueError:
        duplicate_raised = True
    check(duplicate_raised, "deploying a version name twice must be refused")
    before_any = ServingCluster()
    raised = False
    try:
        before_any.serve("u0000", [0.0, 0.0, 0.0])
    except RuntimeError:
        raised = True
    check(raised, "serving before any primary deploy must fail loudly, not silently")

    cluster.deploy(versions["v2"], role=STANDBY)
    cluster.set_canary("v2", 0.25)
    reports = run(cluster, requests)
    check(len(reports) == len(requests) and len({report.user for report in reports}) == len(requests),
          "every request must be answered exactly once")
    check(cluster.served == len(requests)
          and sum(c["requests"] for c in cluster.stats.values()) == len(requests),
          "the served counter must match the per-version counters")
    live = {name for name, role in cluster.roles.items() if role in LIVE}
    check({report.version for report in reports} <= live,
          "no standby or retired version may answer a request")
    for report in reports:
        expected = versions[report.version].score(features_of[report.user])
        check(abs(report.score - expected) < 1e-9,
              "the served score must equal the linear scorer (%s)" % report.user)
    check(cluster.chain_valid(), "the audit chain must verify after deploys and a canary")
    print("versioning: %d requests served by %s, %d live versions, audit chain intact"
          % (cluster.served, sorted({report.version for report in reports}),
             len(live)))

    # ---- 2. routing accuracy and stickiness --------------------------------
    shares = {}
    for weight in (0.0, 0.1, 0.25, 0.5, 1.0):
        trial = ServingCluster()
        trial.deploy(versions["v1"], role="primary")
        trial.deploy(versions["v2"], role=STANDBY)
        cutoff = trial.set_canary("v2", weight)
        check(cutoff == int(round(weight * BUCKETS)), "the cutoff must track the weight")
        observed = run(trial, requests)
        on_canary = sorted(report.user for report in observed if report.version == "v2")
        promised = sorted(user for user, _features in requests if bucket_of(user) < cutoff)
        check(on_canary == promised,
              "only the users below the cutoff may reach the canary at weight %.2f" % weight)
        shares[weight] = len(on_canary) / len(requests)
        check(abs(shares[weight] - weight) <= 0.06,
              "the measured canary share must track the configured weight (%.3f vs %.2f)"
              % (shares[weight], weight))
    check(all(shares[a] <= shares[b] for a, b in zip((0.0, 0.1, 0.25, 0.5),
                                                     (0.1, 0.25, 0.5, 1.0))),
          "the canary share must be monotone in the weight: %s" % shares)
    check(shares[0.0] == 0.0 and shares[1.0] == 1.0,
          "weight 0 must send nothing and weight 1 everything to the canary")
    low = {report.user for report in run(cluster, requests) if report.version == "v2"}
    cluster.set_canary("v2", 0.5)
    high = {report.user for report in run(cluster, requests) if report.version == "v2"}
    check(low <= high, "raising the canary weight must never move a user off the canary")
    cluster.set_canary("v2", 0.1)
    lower = {report.user for report in run(cluster, requests) if report.version == "v2"}
    check(lower <= high, "lowering the weight must keep the users that remain on the canary")
    first = [report.version for report in run(cluster, requests)]
    second = [report.version for report in run(cluster, requests)]
    check(first == second, "routing must be sticky for an unchanged configuration")
    print("routing: shares %s, sticky across reruns, monotone and boundary-only on weight change"
          % {key: round(value, 3) for key, value in shares.items()})

    # ---- 3. shadow isolation and cost --------------------------------------
    plain = ServingCluster()
    plain.deploy(versions["v1"], role="primary")
    plain.deploy(versions["v2"], role=STANDBY)
    plain.set_canary("v2", 0.25)
    plain_reports = run(plain, requests)
    shadowed = ServingCluster()
    shadowed.deploy(versions["v1"], role="primary")
    shadowed.deploy(versions["v2"], role=STANDBY)
    shadowed.set_canary("v2", 0.25)
    shadowed.deploy(versions["v5"], role=SHADOW)
    shadowed_reports = run(shadowed, requests)
    check({report.version for report in shadowed_reports} == {"v1", "v2"},
          "a shadow version must never answer a user")
    check([report.visible() for report in plain_reports]
          == [report.visible() for report in shadowed_reports],
          "enabling shadow tests must not change any user-visible answer")
    check(all(report.shadows == () for report in plain_reports),
          "without a shadow version no shadow report is produced")
    shadow_copies = sum(len(report.shadows) for report in shadowed_reports)
    expected_copies = sum(1 for user, _f in requests
                          if bucket_of(user, "shadow:v5") < SHADOW_SAMPLE_PER_MILLE)
    check(shadow_copies == expected_copies,
          "the shadow must see exactly the sampled traffic (%d vs %d)"
          % (shadow_copies, expected_copies))
    check(0 < shadow_copies < len(requests) * (SHADOW_SAMPLE_PER_MILLE / BUCKETS) * 1.5,
          "shadow cost must stay inside the configured sample (%d copies)" % shadow_copies)
    check(all(copy[1] == round(versions[copy[0]].score(features_of[report.user]), 9)
              for report in shadowed_reports for copy in report.shadows),
          "the shadow report must carry the shadow model's own score")
    check(shadowed.stats["v5"]["errors"] >= shadow_copies * 0.3,
          "the shadow's own failures are counted against the shadow, not the service")
    check(shadowed.stats["v1"] == plain.stats["v1"] and shadowed.stats["v2"] == plain.stats["v2"],
          "shadow traffic must not enter the live versions' service level counters")
    check(shadowed.window["v2"] == plain.window["v2"],
          "shadow traffic must not reach the canary's rollback window")
    print("shadow: %d sampled copies of %d requests, user-visible answers identical, "
          "%d shadow failures kept out of the service level metrics"
          % (shadow_copies, len(requests), shadowed.stats["v5"]["errors"]))

    # ---- 4. rollback on a regression, and no false alarms ------------------
    bad = ServingCluster()
    bad.deploy(versions["v1"], role="primary")
    bad.deploy(versions["v3"], role=STANDBY)
    bad.set_canary("v3", 0.5)
    bad_reports = run(bad, requests)
    check(bad.canary is None and bad.roles["v3"] == RETIRED,
          "a canary whose error rate regresses must be rolled back")
    check(len(bad.monitor_fired) == 1 and bad.monitor_fired[0][1] == "quality",
          "the rollback must fire exactly once, for quality: %s" % bad.monitor_fired)
    check(bad.error_rate("v3") > bad.error_rate("v1") + ERROR_MARGIN,
          "the measured regression must really exceed the margin")
    check(bad.window_requests("v3") >= MIN_CANARY_SAMPLES,
          "the gate must have enough samples before it fires")
    fired_at = bad.monitor_fired[0][0]
    check(all(report.version == "v1" for report in bad_reports[fired_at:]),
          "after the rollback every request must go to the surviving primary")
    check(bad_reports[fired_at - 1].version == "v3",
          "the request that triggered the rollback was still served by the canary")
    check(not bad.rollback("nothing to roll back"),
          "a rollback without a canary must be a no-op")
    healthy = ServingCluster()
    healthy.deploy(versions["v1"], role="primary")
    healthy.deploy(versions["v2"], role=STANDBY)
    healthy.set_canary("v2", 0.5)
    run(healthy, requests)
    check(healthy.canary == "v2" and not healthy.monitor_fired,
          "a healthy canary must never be rolled back: %s" % healthy.monitor_fired)
    check(healthy.error_rate("v2") <= healthy.error_rate("v1") + ERROR_MARGIN,
          "the control canary must really be inside the margin")
    early = ServingCluster()
    early.deploy(versions["v1"], role="primary")
    early.deploy(versions["v3"], role=STANDBY)
    early.set_canary("v3", 0.5)
    for user, features in requests[:80]:
        early.serve(user, features)
    check(early.canary == "v3",
          "a canary below the minimum sample size must not be rolled back")
    slow = ServingCluster()
    slow.deploy(versions["v1"], role="primary")
    slow.deploy(versions["v4"], role=STANDBY)
    slow.set_canary("v4", 0.5)
    run(slow, requests)
    check(slow.monitor_fired and slow.monitor_fired[0][1] == "latency",
          "a canary that breaks the latency objective must be rolled back: %s"
          % slow.monitor_fired)
    check(slow.mean_latency("v4") > LATENCY_SLO_MS > slow.mean_latency("v1"),
          "the latency objective must be the discriminator, not the error rate")
    print("rollback: quality regression at request %d, latency at request %d, healthy canary kept "
          "(%d spurious fires)"
          % (bad.monitor_fired[0][0], slow.monitor_fired[0][0], len(healthy.monitor_fired)))

    # ---- 5. promotion and the audit chain ----------------------------------
    fresh = ServingCluster()
    fresh.deploy(versions["v1"], role="primary")
    fresh.deploy(versions["v2"], role=STANDBY)
    fresh.set_canary("v2", 0.5)
    check(not fresh.promote(), "promotion must be refused before the sample gate is satisfied")
    run(fresh, requests)
    check(fresh.roles["v2"] == "canary", "the canary must still be a canary before promotion")
    canary_requests = fresh.window_requests("v2")
    check(canary_requests >= MIN_CANARY_SAMPLES,
          "the promotion gate must have seen enough canary traffic (%d)" % canary_requests)
    check(fresh.promote(), "a healthy canary must be promotable")
    check(fresh.primary == "v2" and fresh.roles["v1"] == STANDBY and fresh.canary is None,
          "promotion must swap the roles atomically")
    after = [fresh.serve(user, features) for user, features in requests]
    check({report.version for report in after} == {"v2"},
          "after promotion all traffic must reach the new primary")
    check(fresh.chain_valid(), "the audit chain must verify after a promotion")
    tampered = ServingCluster()
    tampered.audit = [dict(entry) for entry in fresh.audit]
    tampered.audit[1]["detail"] = "tampered"
    check(not tampered.chain_valid(), "editing an audit entry must break the chain")
    swapped = [dict(entry) for entry in fresh.audit]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    probe = ServingCluster()
    probe.audit = swapped
    check(not probe.chain_valid(), "reordering audit entries must break the chain")
    check([entry["event"] for entry in fresh.audit]
          == ["deploy", "deploy", "canary", "promote-refused", "promote"],
          "the audit trail must record every control-plane event: %s"
          % [entry["event"] for entry in fresh.audit])
    print("promotion: v2 promoted once the gate had seen %d canary requests, audit chain of %d "
          "entries verifies (tamper and reorder detected)" % (canary_requests, len(fresh.audit)))

    # ---- 6. determinism ----------------------------------------------------
    replay_a = ServingCluster()
    replay_b = ServingCluster()
    for replay in (replay_a, replay_b):
        replay.deploy(versions["v1"], role="primary")
        replay.deploy(versions["v2"], role=STANDBY)
        replay.set_canary("v2", 0.25)
        replay.deploy(versions["v5"], role=SHADOW)
    replay_reports_a = run(replay_a, requests)
    replay_reports_b = run(replay_b, requests)
    check([report.visible() for report in replay_reports_a]
          == [report.visible() for report in replay_reports_b],
          "the same workload must produce the same user-visible answers")
    check([report.shadows for report in replay_reports_a]
          == [report.shadows for report in replay_reports_b],
          "the same workload must produce the same shadow decisions")
    check(replay_a.chain_head() == replay_b.chain_head(),
          "the audit chain must be reproducible hash for hash")
    check(replay_a.stats == replay_b.stats, "the per-version counters must be reproducible")
    odd = "user-with-a-very-long-name-and-symbols-!*()"
    check(0 <= bucket_of(odd) < BUCKETS, "any user id must land in a bucket")
    expected_odd = ("v2" if bucket_of(odd) < replay_a.canary_cutoff else "v1")
    check(replay_a.route(odd) == expected_odd,
          "an unknown user must be routed by the same bucket rule")
    print("determinism: identical routing, shadow sample and audit head %d for a replayed workload"
          % (replay_a.chain_head() % 1000000))
    print("PASS")


if __name__ == "__main__":
    main()

