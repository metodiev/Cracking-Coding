"""Task 927 - Implement model-backed materialized views updating with streaming features.  (Tier: Master, CTCI Ch 14 - Databases)

A materialized view whose rows carry a model score, maintained incrementally as feature updates
stream in.

* Model: a linear model y = sum_f w_f x_f + bias over the feature vector, so a single-feature delta
  can be folded into the cached score exactly: y_new = y_old + w_f (x_new - x_old).
* View: one row per entity holding the current feature vector, the cached score, the vector's
  version timestamp and the set of feature names still missing (defaulted to zero).
* Stream: (entity, feature, value, timestamp) events arrive out of order.  An event is applied only
  when its timestamp is at least the row's version (a watermark rule), so replays and late duplicates
  are idempotent and a row never regresses to older values; the lag between the stream watermark and
  a row's version is the view's staleness.
* Compaction is implicit: only the latest version per entity is stored, so the row count equals the
  number of distinct entities however long the stream is.

Verified: after every applied update the incremental score equals a full re-scoring of the row
(the invariant is re-checked on every mutation); stale events change nothing and are dropped;
replaying a whole stream twice is a no-op; the row count equals the distinct entity count; a
from-scratch replay of the accepted events reproduces the view exactly; unknown entities are
created with defaults and missing features are tracked; empty and single-event streams behave; and
the maintenance is deterministic.  Simplified: one linear model, in-memory rows, no persistence.
"""
FEATURES = ("clicks", "spend", "tenure")
WEIGHTS = {"clicks": 0.5, "spend": 2.0, "tenure": -0.25}
BIAS = 1.0


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def score(features):
    """Full scoring of a feature dictionary; the oracle the incremental path must match."""
    return BIAS + sum(WEIGHTS[name] * features.get(name, 0.0) for name in WEIGHTS)


class ModelView:
    """Materialized view of model scores, maintained incrementally over a feature stream."""

    def __init__(self):
        self.rows = {}                       # entity -> row dictionary
        self.watermark = 0
        self.incremental, self.recomputed, self.dropped, self.accepted = 0, 0, 0, 0

    def apply(self, entity, feature, value, timestamp):
        """Apply one streaming update; returns True when the row changed."""
        check(feature in WEIGHTS, "the view only materialises known model features")
        if timestamp > self.watermark:
            self.watermark = timestamp
        row = self.rows.get(entity)
        if row is None:
            row = {"features": {name: 0.0 for name in WEIGHTS}, "missing": set(FEATURES),
                   "version": timestamp, "score": 0.0}
            row["score"] = score(row["features"])          # defaults, then the update below
            self.rows[entity] = row
            self.recomputed += 1
        if timestamp < row["version"]:
            self.dropped += 1
            return False                                   # stale event: the row does not regress
        previous = row["features"][feature]
        if timestamp == row["version"] and previous == value:
            self.dropped += 1
            return False                                   # idempotent replay
        row["features"][feature] = value
        row["missing"].discard(feature)
        row["version"] = timestamp
        row["score"] += WEIGHTS[feature] * (value - previous)      # incremental fold
        self.incremental += 1
        self.accepted += 1
        check(abs(row["score"] - score(row["features"])) < 1e-9,
              "the incrementally updated score must equal a full re-scoring")
        return True

    def full_recompute(self, entity):
        row = self.rows[entity]
        row["score"] = score(row["features"])
        self.recomputed += 1
        return row["score"]

    def stale(self, entity):
        row = self.rows.get(entity)
        return 0 if row is None else self.watermark - row["version"]

    def staleness_report(self):
        return {entity: self.stale(entity) for entity in sorted(self.rows)}


def stream():
    """Deterministic stream with duplicated, late and out-of-order events mixed in."""
    events = []
    for step in range(20):
        entity = "user_%d" % (step % 5)
        feature = FEATURES[step % len(FEATURES)]
        events.append((entity, feature, float(1 + (step * 7) % 11), step + 1))
    events.append(events[3])                                  # exact duplicate: must be a no-op
    events.append(("user_0", "clicks", 99.0, 2))              # late event: must not regress
    events.append(("user_1", "spend", 7.5, 25))               # future event: must apply
    events.append(("user_9", "tenure", 3.0, 26))              # brand new entity at the end
    return events


def replay(events):
    """Independent oracle: fold the accepted events from scratch with plain full scoring."""
    rows, seen = {}, {}
    for entity, feature, value, timestamp in events:
        if timestamp < seen.get(entity, -1):
            continue
        seen[entity] = timestamp
        row = rows.setdefault(entity, {name: 0.0 for name in WEIGHTS})
        row[feature] = value
    return {entity: score(row) for entity, row in rows.items()}


def main():
    events = stream()
    view = ModelView()

    # 1. Every applied update keeps the incremental score equal to a full re-scoring, and the
    #    incremental path is what actually runs (no silent fallback to recomputation).
    applied = [view.apply(entity, feature, value, timestamp)
               for entity, feature, value, timestamp in events[:20]]
    check(all(applied), "the first twenty in-order events must all be applied")
    check(view.incremental == 20 and view.recomputed == len(set(
        entity for entity, _, _, _ in events[:20])),
        "each row must be scored once at creation and updated incrementally afterwards")

    # 2. Out-of-order and duplicate events are dropped without changing a row.
    before = {entity: (dict(view.rows[entity]["features"]), view.rows[entity]["score"])
              for entity in view.rows}
    check(view.apply(*events[20]) is False and view.apply(*events[21]) is False,
          "a duplicate and a late event must both be rejected")
    check(view.apply(*events[22]) is True, "a newer event must still be applied")
    for entity, (features, value) in before.items():
        if entity != "user_1":
            check(view.rows[entity]["features"] == features
                  and view.rows[entity]["score"] == value,
                  "dropped events must leave every untouched row identical")

    # 3. Replaying the whole stream again is a no-op (idempotence) but the new entity is created.
    check(view.apply(*events[23]) is True, "a brand new entity must be created")
    snapshot = {entity: (dict(view.rows[entity]["features"]), view.rows[entity]["score"])
                for entity in view.rows}
    for event in events:
        view.apply(*event)
    check({entity: (dict(view.rows[entity]["features"]), view.rows[entity]["score"])
           for entity in view.rows} == snapshot,
          "replaying the stream must not change any row")

    # 4. Compaction: one row per entity, holding only the newest version.
    check(len(view.rows) == len({entity for entity, _, _, _ in events}) == 6,
          "the view must keep exactly one row per distinct entity")
    check(all(view.rows[entity]["version"] == max(timestamp for e, _, _, timestamp in events
                                                  if e == entity)
              for entity in view.rows),
          "each row must hold the newest timestamp it has seen")

    # 5. Independent oracle: a from-scratch fold of the accepted events matches the view.
    oracle = replay(events)
    check(oracle == {entity: row["score"] for entity, row in view.rows.items()},
          "the maintained view must equal a from-scratch replay of the same events")
    check(all(abs(view.full_recompute(entity) - oracle[entity]) < 1e-12 for entity in view.rows),
          "an explicit full recomputation must agree with the oracle")

    # 6. Missing features are tracked: defaults are zero and disappear once a value arrives.
    fresh = ModelView()
    fresh.apply("new_user", "clicks", 4.0, 1)
    check(fresh.rows["new_user"]["missing"] == {"spend", "tenure"}
          and abs(fresh.rows["new_user"]["score"] - (BIAS + 0.5 * 4.0)) < 1e-12,
          "an unknown entity must start from defaults and track its missing features")
    check(abs(score({}) - BIAS) < 1e-12, "an all-default feature vector must score the bias")

    # 7. Staleness: the watermark minus the row version, and rows far behind are visible.
    check(view.watermark == 26 and max(view.staleness_report().values()) >= 0,
          "the watermark must advance to the newest timestamp seen")
    lagging = ModelView()
    lagging.apply("a", "clicks", 1.0, 1)
    lagging.apply("b", "clicks", 1.0, 30)
    check(lagging.stale("a") == 29 and lagging.stale("b") == 0 and lagging.stale("ghost") == 0,
          "staleness must expose the lag of a row that missed recent traffic")

    # 8. Boundary cases: an empty stream and a single-event stream.
    empty = ModelView()
    check(empty.rows == {} and empty.watermark == 0 and empty.staleness_report() == {},
          "an empty stream must produce an empty view")
    solo = ModelView()
    check(solo.apply("only", "spend", 2.0, 5) and len(solo.rows) == 1
          and abs(solo.rows["only"]["score"] - (BIAS + 4.0)) < 1e-12,
          "a single-event stream must produce exactly one correctly scored row")

    # 9. Determinism: the same stream builds the same view.
    again = ModelView()
    for event in events:
        again.apply(*event)
    check({e: (again.rows[e]["features"], again.rows[e]["score"]) for e in again.rows}
          == {e: (view.rows[e]["features"], view.rows[e]["score"]) for e in view.rows},
          "maintenance must be deterministic")

    print(f"rows {len(view.rows)}, incremental updates {view.incremental}, "
          f"full recomputations {view.recomputed}, dropped {view.dropped}")
    print(f"watermark {view.watermark}, staleness {view.staleness_report()}")
    print("PASS")


if __name__ == "__main__":
    main()
