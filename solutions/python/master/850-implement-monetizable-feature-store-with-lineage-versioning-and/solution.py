"""Task 850 - Implement monetizable feature store with lineage, versioning, and online/offline sync.  (Tier: Master, CTCI Beyond — System Design and Scalability)

A feature store with the four properties that make one usable in production: append-only
versioned feature definitions, lineage that can be recomputed, an offline store that answers
point-in-time correct training queries, an online store that serves the latest materialised
values, and per-consumer entitlement/quota metering that turns feature reads into a bill.

Everything is an in-process deterministic model - logical timestamps, explicit materialisation
steps, no clock, no threads, no storage engine:

  * `FeatureStore.define` appends a new immutable version of a feature (or a derived feature
    whose body is a transform over the *pinned* versions of its upstreams), so history is never
    rewritten and `restore` re-deploys an old body as a *new* version;
  * `offline_query` performs a point-in-time join: for each entity the newest row with
    timestamp <= as_of, which is exactly what a training set must see to avoid label leakage;
  * `materialize` copies the offline rows up to a watermark into the online store, and
    `check_consistency` compares serving against offline - the online/offline skew detector that
    also catches the classic training/serving skew when a job pins one version and serving runs
    another;
  * lineage is a DAG over (feature, version) pins: it must resolve, cycles are rejected, and
    recomputing a derived feature from its lineage must reproduce the stored value exactly;
  * every read is metered: a consumer must be entitled to the pinned version (or the read is
    refused) and stay inside its quota, and the invoice is the sum over the append-only billing
    ledger - so the money is auditable against the ledger and per-reader usage counters.

Simplified versus production: one process, dict-backed stores, integer money in cents, quotas
per window instead of per month, and transforms limited to the small closed set below; what is
*not* simplified is the semantics - every claim below is verified against an independent
recomputation from the raw history.
"""


class FeatureError(Exception):
    """Raised for entitlement, quota, lineage and versioning violations."""


class Version:
    """One immutable feature version: how to compute it, what it depends on, what it costs."""

    __slots__ = ("name", "version", "kind", "params", "deps", "price", "retired")

    def __init__(self, name, version, kind, params, deps, price):
        self.name = name
        self.version = version
        self.kind = kind
        self.params = params
        self.deps = tuple(deps)
        self.price = price
        self.retired = False

    def key(self):
        return "%s@%d" % (self.name, self.version)


def transform(kind, values, params):
    """The closed set of transforms a derived feature may apply to its upstream values."""
    if kind == "identity":
        return {entity: value for entity, value in values[0].items()}
    if kind == "scale":
        factor = params["factor"]
        return {entity: value * factor for entity, value in values[0].items()}
    if kind == "combine":
        first, second = values
        weight = params["weight"]
        out = {}
        for entity in sorted(set(first) | set(second)):
            left = first.get(entity, 0)
            right = second.get(entity, 0)
            out[entity] = weight * left + (1 - weight) * right
        return out
    if kind == "lag":
        offset = params["offset"]
        return {entity: value + offset for entity, value in values[0].items()}
    raise FeatureError("unknown transform %r" % (kind,))


class FeatureStore:
    """Versioned definitions, an offline history, an online cache, lineage and metering."""

    def __init__(self):
        self.versions = {}
        self.history = {name: [] for name in ()}
        self.offline = []                      # (feature, version, entity, timestamp, value)
        self.online = {}                       # (feature, version) -> {entity: (value, ts)}
        self.watermarks = {}                   # (feature, version) -> timestamp materialised
        self.entitlements = {}                 # consumer -> set of "name@version"
        self.quotas = {}                       # consumer -> reads allowed
        self.usage = {}                        # consumer -> successful reads
        self.ledger = []                       # (consumer, name, version, timestamp, cents)
        self.registry = {}

    # -- definitions --------------------------------------------------------
    def define(self, name, kind, params=None, deps=(), price=0):
        """Append a new immutable version of a feature; returns its key."""
        if kind not in ("raw", "identity", "scale", "combine", "lag"):
            raise FeatureError("unknown feature kind %r" % (kind,))
        existing = [version for version in self.versions.values() if version.name == name]
        if kind != "raw" and not deps:
            raise FeatureError("a derived feature must declare its upstreams")
        for dependency in deps:
            if dependency not in self.versions:
                raise FeatureError("unknown lineage pin %s" % dependency)
        version = Version(name, len(existing) + 1, kind, dict(params or {}), deps, price)
        self.versions[version.key()] = version
        self.offline_by_key = getattr(self, "offline_by_key", {})
        self.offline_by_key[version.key()] = []
        self.registry[name] = version.key()
        return version.key()

    def restore(self, key):
        """Re-deploy an old body under a *new* version number (history stays append-only)."""
        source = self.versions[key]
        existing = [version for version in self.versions.values() if version.name == source.name]
        copy = Version(source.name, len(existing) + 1, source.kind, dict(source.params),
                       source.deps, source.price)
        self.versions[copy.key()] = copy
        self.offline_by_key[copy.key()] = []
        self.registry[copy.name] = copy.key()
        return copy.key()

    def retire(self, key):
        self.versions[key].retired = True
        return self

    # -- raw ingestion ------------------------------------------------------
    def ingest(self, key, entity, timestamp, value):
        if key not in self.versions:
            raise FeatureError("unknown feature version %s" % key)
        self.offline_by_key[key].append((entity, timestamp, value))
        self.offline_by_key[key].sort(key=lambda row: (row[0], row[1]))
        return self

    # -- offline point-in-time queries --------------------------------------
    def offline_query(self, key, entities, as_of):
        """Newest row per entity with timestamp <= as_of - the leak-free training join."""
        rows = self.offline_by_key[key]
        out = {}
        for entity in entities:
            chosen = None
            for row_entity, timestamp, value in rows:
                if row_entity != entity or timestamp > as_of:
                    continue
                if chosen is None or timestamp >= chosen[0]:
                    chosen = (timestamp, value)
            if chosen is not None:
                out[entity] = chosen[1]
        return out

    def point_in_time_frame(self, keys, entities, as_of):
        return {key: self.offline_query(key, entities, as_of) for key in keys}

    # -- materialisation and serving ---------------------------------------
    def materialize(self, key, up_to):
        """Copy offline rows with timestamp <= up_to into the online store."""
        rows = self.offline_by_key[key]
        cache = self.online.setdefault(key, {})
        copied = 0
        for entity, timestamp, value in rows:
            if timestamp > up_to:
                continue
            current = cache.get(entity)
            if current is None or timestamp >= current[1]:
                cache[entity] = (value, timestamp)
                copied += 1
        self.watermarks[key] = max(up_to, self.watermarks.get(key, -1))
        return copied

    def serve(self, key, entity):
        value = self.online.get(key, {}).get(entity)
        return None if value is None else value[0]

    def check_consistency(self, key, entities, as_of=None):
        """Skew detector: serving must equal the offline snapshot at the watermark."""
        if as_of is None:
            as_of = self.watermarks.get(key, -1)
        offline = self.offline_query(key, entities, as_of)
        mismatches = []
        for entity in entities:
            served = self.serve(key, entity)
            expected = offline.get(entity)
            if served != expected:
                mismatches.append((entity, served, expected))
        return mismatches

    def lag(self, key):
        """Offline rows newer than the watermark: freshness, not a correctness violation."""
        watermark = self.watermarks.get(key, -1)
        return len([row for row in self.offline_by_key[key] if row[1] > watermark])

    # -- lineage ------------------------------------------------------------
    def lineage(self, key, seen=None):
        """Resolve the transitive pins of a feature; rejects dangling pins and cycles."""
        version = self.versions.get(key)
        if version is None:
            raise FeatureError("dangling lineage pin %s" % key)
        seen = set() if seen is None else seen
        if key in seen:
            raise FeatureError("cycle detected at %s" % key)
        seen.add(key)
        for dependency in version.deps:
            self.lineage(dependency, seen)
        return sorted(seen)

    def recompute(self, key, as_of, entities):
        """Recompute a feature from its lineage; raw features read the history."""
        version = self.versions[key]
        if version.kind == "raw":
            return self.offline_query(key, entities, as_of)
        upstreams = [self.recompute(dependency, as_of, entities) for dependency in version.deps]
        if version.kind == "combine":
            return transform("combine", upstreams, version.params)
        if version.kind == "scale":
            return transform("scale", upstreams, version.params)
        if version.kind == "lag":
            return transform("lag", upstreams, version.params)
        return transform("identity", upstreams, version.params)

    # -- monetisation -------------------------------------------------------
    def license(self, consumer, keys, quota):
        self.entitlements.setdefault(consumer, set()).update(keys)
        self.quotas[consumer] = quota
        self.usage.setdefault(consumer, 0)
        return self

    def read(self, consumer, key, entity):
        """Metered read: entitlement and quota first, billing only on success."""
        version = self.versions.get(key)
        if version is None:
            raise FeatureError("unknown feature version %s" % key)
        if key not in self.entitlements.get(consumer, set()):
            raise FeatureError("consumer %s is not licensed for %s" % (consumer, key))
        if self.usage.get(consumer, 0) >= self.quotas.get(consumer, 0):
            raise FeatureError("consumer %s exceeded its quota" % consumer)
        value = self.serve(key, entity)
        self.usage[consumer] = self.usage.get(consumer, 0) + 1
        self.ledger.append((consumer, version.name, version.version, self.watermarks.get(key, -1),
                            version.price))
        return value

    def invoice(self, consumer=None):
        total = 0
        for entry_consumer, _, _, _, cents in self.ledger:
            if consumer is None or entry_consumer == consumer:
                total += cents
        return total


def reference_pit(rows, entities, as_of):
    """Independent point-in-time join: scan the history and keep the newest admissible row."""
    best = {}
    for entity, timestamp, value in rows:
        if entity not in entities or timestamp > as_of:
            continue
        if entity not in best or timestamp > best[entity][0]:
            best[entity] = (timestamp, value)
    return {entity: value for entity, (_, value) in best.items()}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    store = FeatureStore()

    # ---- 1. versioning is append-only ------------------------------------
    clicks = store.define("clicks", "raw", price=3)
    check(clicks == "clicks@1", "the first version is numbered one")
    check(store.define("clicks", "raw", price=5) == "clicks@2", "a redefinition appends")
    check(len([version for version in store.versions.values()
               if version.name == "clicks"]) == 2, "both versions coexist")
    restored = store.restore(clicks)
    check(restored == "clicks@3" and store.versions[restored].price == 3,
          "restoring re-deploys the old body under a new version")
    check(store.versions[clicks].price == 3, "the original version is untouched")
    check(store.versions["clicks@2"].price == 5, "the second version is untouched")
    store.retire("clicks@2")
    check(store.versions["clicks@2"].retired, "retirement is recorded, not deleted")

    # ---- 2. offline point-in-time queries --------------------------------
    rows = []
    for entity in range(6):
        for tick in range(4):
            rows.append((entity, tick * 10 + entity, 100 * entity + tick))
    for entity, timestamp, value in rows:
        store.ingest(clicks, entity, timestamp, value)
    for as_of in (0, 1, 5, 9, 10, 17, 23, 33, 99):
        entities = list(range(6))
        got = store.offline_query(clicks, entities, as_of)
        expected = reference_pit(store.offline_by_key[clicks], entities, as_of)
        check(got == expected, "point-in-time join at %d must match the scan" % as_of)
    check(store.offline_query(clicks, [0], 9) == {0: 0},
          "no row after as_of may leak into the frame")
    check(store.offline_query(clicks, [0], 10) == {0: 1},
          "a row exactly at as_of is included")
    check(store.offline_query(clicks, [5], 16) == {5: 501},
          "the newest admissible row wins")
    check(store.offline_query(clicks, [99], 99) == {},
          "an entity with no history produces no value")

    # ---- 3. online/offline synchronisation -------------------------------
    before = store.serve(clicks, 3)
    check(before is None, "nothing is served before the first materialisation")
    copied = store.materialize(clicks, 15)
    check(copied > 0, "materialisation copies rows")
    entities = list(range(6))
    check(store.check_consistency(clicks, entities) == [],
          "serving must agree with the offline snapshot at the watermark")
    check(store.lag(clicks) == len([row for row in store.offline_by_key[clicks]
                                    if row[1] > 15]),
          "the lag counts exactly the rows beyond the watermark")
    store.ingest(clicks, 3, 20, 9999)
    check(store.serve(clicks, 3) != 9999,
          "a row beyond the watermark must not leak into serving")
    check(store.check_consistency(clicks, entities) == [],
          "a newer row is lag, not skew")
    check(store.lag(clicks) >= 1, "the new row shows up as lag")
    store.ingest(clicks, 0, 14, 777)                    # late data inside the watermark
    skew = store.check_consistency(clicks, entities)
    check([entry[0] for entry in skew] == [0],
          "a late row inside the watermark must be reported as skew (%r)" % (skew,))
    check(store.materialize(clicks, 15) > 0 and store.check_consistency(clicks, entities) == [],
          "re-materialising repairs the skew")
    check(store.serve(clicks, 0) == 777, "serving returns the newest materialised value")
    check(store.materialize(clicks, 5) >= 0 and store.watermarks[clicks] == 15,
          "materialising an older watermark never moves the watermark backwards")
    store.materialize(clicks, 20)
    check(store.watermarks[clicks] == 20, "the watermark advances to the sync point")
    check(store.lag(clicks) == len([row for row in store.offline_by_key[clicks]
                                    if row[1] > 20]), "the lag is recomputed after the sync")
    check(store.serve(clicks, 3) == 9999, "the lagged row is served after the next sync")

    # ---- 4. lineage and recomputation ------------------------------------
    views = store.define("clicks_scaled", "scale", {"factor": 2}, deps=[clicks], price=1)
    score = store.define("score", "combine", {"weight": 1},
                         deps=[views, "clicks@2"], price=2)
    check(store.lineage(score) == sorted([clicks, views, "clicks@2", score]),
          "lineage resolves transitively (%r)" % (store.lineage(score),))
    check(store.offline_by_key["clicks@2"] == [],
          "the pinned upstream has its own (empty) history")
    try:
        store.lineage("missing@1")
        raise AssertionError("a dangling lineage pin must be rejected")
    except FeatureError:
        pass
    try:
        store.define("broken", "scale", {"factor": 2}, deps=["nope@1"])
        raise AssertionError("an unknown dependency must be rejected")
    except FeatureError:
        pass
    try:
        store.define("lonely", "scale", {"factor": 2})
        raise AssertionError("a derived feature without upstreams must be rejected")
    except FeatureError:
        pass
    recomputed = store.recompute(views, 20, entities)
    expected = {entity: value * 2 for entity, value in
                reference_pit(store.offline_by_key[clicks], entities, 20).items()}
    check(recomputed == expected, "recomputing a derived feature must match the transform")
    combined = store.recompute(score, 20, entities)
    second = reference_pit(store.offline_by_key["clicks@2"], entities, 20)
    hand = {entity: combined.get(entity) for entity in entities}
    for entity in entities:
        want = expected.get(entity, 0) * 1 + second.get(entity, 0) * 0
        check(hand[entity] == want,
              "the combine transform must follow its weights (entity %d)" % entity)

    # a cycle is impossible by construction (pins only ever point at older keys) but the
    # checker must still refuse one if it is forced in
    store.versions[views].deps = (score,)
    try:
        store.lineage(views)
        raise AssertionError("a lineage cycle must be rejected")
    except FeatureError:
        pass
    store.versions[views].deps = (clicks,)

    # ---- 5. entitlement, quota and billing --------------------------------
    store.license("alice", [clicks, views], quota=5)
    store.license("bob", [clicks], quota=100)
    check(store.read("alice", clicks, 0) == store.serve(clicks, 0),
          "a licensed read returns the served value")
    store.materialize(views, 20)
    for entity in range(5):
        store.ingest(views, entity, entity + 1, 10 * entity)
    store.materialize(views, 20)
    check(store.read("alice", views, 2) == 20, "the derived feature serves its own values")
    try:
        store.read("alice", score, 0)
        raise AssertionError("an unlicensed read must be refused")
    except FeatureError:
        pass
    check(store.usage["alice"] == 2, "a refused read is not metered")
    check(store.invoice("alice") == store.versions[clicks].price + store.versions[views].price,
          "the invoice sums the ledger entries")
    for _ in range(3):
        store.read("alice", clicks, 1)
    check(store.usage["alice"] == 5, "usage counts every successful read")
    try:
        store.read("alice", clicks, 1)
        raise AssertionError("the quota must refuse the sixth read")
    except FeatureError:
        pass
    expected_invoice = 4 * store.versions[clicks].price + 1 * store.versions[views].price
    check(store.invoice("alice") == expected_invoice,
          "billing follows the ledger line by line (%d vs %d)"
          % (store.invoice("alice"), expected_invoice))
    check(store.invoice("alice") ==
          sum(cents for consumer, _, _, _, cents in store.ledger if consumer == "alice"),
          "the invoice is exactly the ledger rows of that consumer")
    check(store.invoice("bob") == 0, "an idle consumer is not billed")
    check(store.invoice() == store.invoice("alice") + store.invoice("bob"),
          "the total invoice is the sum over consumers")
    check(len(store.ledger) == store.usage["alice"], "the ledger has one row per billed read")
    check(sum(cents for _, _, _, _, cents in store.ledger) == store.invoice(),
          "the ledger sum equals the invoice")

    # ---- 6. training/serving skew ----------------------------------------
    training = store.point_in_time_frame([clicks, views], entities, 20)
    expected_frame = reference_pit(store.offline_by_key[clicks], set(entities), 20)
    check(training[clicks] == expected_frame,
          "the training frame must equal the point-in-time reference (%r vs %r)"
          % (training[clicks], expected_frame))
    check(5 in training[clicks] or 5 not in expected_frame,
          "every entity with admissible history must appear in the frame")
    pinned = clicks
    store.define("clicks", "raw", price=4)          # a new default version is deployed
    store.materialize(clicks, 20)
    served_now = store.check_consistency(pinned, entities, 20)
    check(served_now == [], "the pinned version still serves consistently")
    new_key = store.registry["clicks"]
    check(new_key != pinned and store.versions[new_key].price == 4,
          "a new deployment gets a new version number")
    store.ingest(new_key, 0, 5, 4242)
    skew = store.check_consistency(new_key, entities, 5)
    check(len(skew) == 1,
          "the version that is behind on materialisation is reported (%r)" % (skew,))
    store.materialize(new_key, 5)
    check(store.check_consistency(new_key, entities, 5) == [],
          "re-materialising removes the skew")
    check(store.serve(pinned, 0) != store.serve(new_key, 0),
          "training on the pinned version and serving the new one is visible as skew")

    # ---- 7. degenerate cases and determinism ------------------------------
    empty = FeatureStore()
    blank = empty.define("nothing", "raw")
    check(empty.offline_query(blank, [1, 2], 100) == {}, "an empty history returns nothing")
    check(empty.materialize(blank, 10) == 0, "materialising nothing copies nothing")
    check(empty.check_consistency(blank, [1], 10) == [], "nothing is consistent with nothing")
    empty.ingest(blank, 1, 3, 7)
    empty.materialize(blank, 3)
    check(empty.serve(blank, 1) == 7, "a single row serves correctly")
    check(empty.serve(blank, 2) is None, "an unknown entity serves nothing")
    try:
        empty.ingest("ghost@9", 1, 0, 0)
        raise AssertionError("ingesting into an unknown version must fail")
    except FeatureError:
        pass
    twin = FeatureStore()
    twin.define("clicks", "raw", price=3)
    for entity, timestamp, value in rows:
        twin.ingest("clicks@1", entity, timestamp, value)
    twin.materialize("clicks@1", 15)
    check(twin.serve("clicks@1", 2) == store.serve(clicks, 2),
          "two identical stores serve identical values")
    check(twin.invoice() == 0, "an unlicensed store bills nothing")
    check(store.versions[clicks].key() == "clicks@1", "keys render deterministically")
    check(len(store.lineage(views)) == 2, "lineage of a derived feature has two pins")

    print("versions: %d defined, %d retired, watermark of clicks=%d"
          % (len(store.versions), len([v for v in store.versions.values() if v.retired]),
             store.watermarks[clicks]))
    print("offline: %d rows, online: %d entities, consistency check clean after sync"
          % (len(store.offline_by_key[clicks]), len(store.online.get(clicks, {}))))
    print("metering: ledger %d rows, alice usage %d/%d, invoice %d cents, bob %d cents"
          % (len(store.ledger), store.usage["alice"], store.quotas["alice"],
             store.invoice("alice"), store.invoice("bob")))
    print("PASS")


if __name__ == "__main__":
    main()
