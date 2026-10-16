"""Task 888 - Implement incremental view maintenance with delta propagation and bounded memory.  (Tier: Master, CTCI Beyond - Distributed Systems & Storage)

The workload is a two level materialised view chain over two base relations:

    R(key, a)   and   S(key, b)
    V1 = { (key, a, b) : (key, a) in R, (key, b) in S, b >= b_min }   (bag semantics)
    V2 = { key : sum of a * b over V1 }

Both levels are maintained incrementally.  A batch is a list of signed deltas
`(relation, row, +1/-1)`.  Deltas are staged in a buffer whose size is hard bounded by
`buffer_limit` distinct entries; a full buffer is flushed, and a flush propagates in one pass:

    dV1 = dR |> S + dS |> R(updated)      (a delta probe, never a full join)
    dV2 = aggregate(dV1)                  (streamed, so no intermediate bag is materialised)

Delta propagation is exact because the view is bilinear in the two base relations: applying
the R deltas against the old S and then the S deltas against the new R yields the true view of
the new state.  Relations are indexed by key, so a delta of k rows costs O(k * rows-per-key)
probes instead of a full |R| * |S| rejoin - that ratio is measured, not assumed.

Simplified versus a production IVM engine: one aggregate type (sum/count - split-additive
only; min/max would need the whole group), no recursive or outer joins, no concurrency, no
spill-to-disk (bounded memory is enforced by flushing rather than by eviction), a single
two-level chain rather than a DAG of views, and small relations.  Verified: the materialised
views equal a from-scratch rebuild and an independent list-based brute-force join after every
single batch; the applied delta equals the difference of two rebuilds; bag multiplicities and
deletes are exact and an impossible delete is rejected without corrupting state; the staging
buffer never exceeds its bound; the incremental cost stays far below the cost of re-deriving
the view; and the whole pipeline is reproducible from its seed.
"""
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state, value = splitmix64(self.state)
        return value % bound


class UnderflowError(Exception):
    """Raised when a delete would drive a multiplicity below zero."""


class SchemaError(Exception):
    """Raised for malformed batches, relations or plans."""


class Bag:
    """Multiset with signed counts; zero counts are removed so the state is canonical."""

    def __init__(self, entries=None):
        self.counts = {}
        if entries:
            for item, count in entries:
                self.add(item, count)

    def add(self, item, delta=1):
        current = self.counts.get(item, 0) + delta
        if current < 0:
            raise UnderflowError("multiplicity of %r would become %d" % (item, current))
        if current:
            self.counts[item] = current
        else:
            self.counts.pop(item, None)
        return current

    def update(self, items, delta=1):
        for item, weight in items:
            self.add(item, weight * delta)

    def total(self):
        return sum(self.counts.values())

    def size(self):
        return len(self.counts)

    def expanded(self):
        out = []
        for item, weight in sorted(self.counts.items()):
            out.extend([item] * weight)
        return out


class BaseRelation:
    """A base table stored as a bag plus a key index used for delta probing."""

    def __init__(self, name, columns):
        self.name = name
        self.columns = columns
        self.bag = Bag()
        self.index = {}          # key -> {value: multiplicity}

    def validate(self, row):
        if not isinstance(row, tuple) or len(row) != len(self.columns):
            raise SchemaError("%s expects %d columns, got %r"
                              % (self.name, len(self.columns), row))
        key, value = row
        if not isinstance(key, int) or not isinstance(value, int):
            raise SchemaError("keys and values must be integers, got %r" % (row,))

    def apply_delta(self, row, delta):
        """Commit a validated delta to the bag and to the key index."""
        self.bag.add(row, delta)
        key, value = row
        bucket = self.index.setdefault(key, {})
        updated = bucket.get(value, 0) + delta
        if updated:
            bucket[value] = updated
        elif bucket:
            bucket.pop(value, None)
            if not bucket:
                self.index.pop(key, None)
        return self.bag.counts.get(row, 0)

    def add(self, row, delta=1):
        self.validate(row)
        return self.apply_delta(row, delta)

    def rows(self):
        return self.bag.expanded()

    def key_count(self):
        return len(self.index)


def brute_force_view(left_rows, right_rows, min_right):
    """Independent list-based oracle: nested loops, no indexes, no deltas."""
    out = {}
    for left in left_rows:
        for right in right_rows:
            if left[0] == right[0] and right[1] >= min_right:
                out[(left[0], left[1], right[1])] = out.get((left[0], left[1], right[1]), 0) + 1
    return out


def brute_force_sums(view):
    sums = {}
    for (key, left_value, right_value), weight in view.items():
        sums[key] = sums.get(key, 0) + weight * left_value * right_value
    return {key: value for key, value in sums.items() if value != 0}


class ViewPipeline:
    """Maintains V1 and V2 incrementally with a bounded staging buffer."""

    def __init__(self, left, right, min_right=0, buffer_limit=8):
        if buffer_limit <= 0:
            raise ValueError("the staging buffer limit must be positive")
        self.left = left
        self.right = right
        self.min_right = min_right
        self.buffer_limit = buffer_limit
        self.relations = {left.name: left, right.name: right}
        self.staged = {}                 # (relation, row) -> net delta
        self.v1 = Bag()
        self.v2 = {}
        self.flushes = 0
        self.incremental_probes = 0
        self.peak_staged = 0
        self.peak_accumulator = 0
        self.last_delta_v1 = {}
        self.last_delta_v2 = {}
        self.full_probes = 0
        self._materialise()

    # ------------------------------------------------------------ maintenance
    def _materialise(self):
        self.v1 = Bag()
        self.v2 = {}
        delta_v1 = {}
        for key, bucket in self.left.index.items():
            other = self.right.index.get(key, {})
            for left_value, left_count in bucket.items():
                for right_value, right_count in other.items():
                    self.full_probes += 1
                    if right_value < self.min_right:
                        continue
                    entry = (key, left_value, right_value)
                    delta_v1[entry] = delta_v1.get(entry, 0) + left_count * right_count
        self._apply_delta_v1(delta_v1)

    def _contribution(self, view_delta, key, left_value, right_value, weight):
        view_delta[(key, left_value, right_value)] = (
            view_delta.get((key, left_value, right_value), 0) + weight)

    def _apply_delta_v1(self, delta_v1):
        """Fold a delta of V1 into both materialised levels (streamed, O(groups) memory)."""
        aggregate = {}
        for (key, left_value, right_value), weight in sorted(delta_v1.items()):
            self.v1.add((key, left_value, right_value), weight)
            aggregate[key] = aggregate.get(key, 0) + weight * left_value * right_value
        self.peak_accumulator = max(self.peak_accumulator, len(aggregate))
        for key, weight in aggregate.items():
            total = self.v2.get(key, 0) + weight
            if total:
                self.v2[key] = total
            else:
                self.v2.pop(key, None)
        self.last_delta_v1 = {entry: weight for entry, weight in delta_v1.items() if weight}
        self.last_delta_v2 = {key: weight for key, weight in aggregate.items() if weight}

    def stage(self, relation, row, delta):
        """Buffer a delta for the base; the base itself is only touched on flush."""
        if relation not in self.relations:
            raise SchemaError("unknown relation %r" % (relation,))
        if delta not in (-1, 1):
            raise SchemaError("a delta must be +1 or -1, got %r" % (delta,))
        target = self.relations[relation]
        target.validate(row)
        pending = self.staged.get((relation, row), 0)
        if target.bag.counts.get(row, 0) + pending + delta < 0:
            raise UnderflowError("the delete of %r would drive its multiplicity below zero"
                                 % (row,))
        self.staged[(relation, row)] = pending + delta
        self.peak_staged = max(self.peak_staged, len(self.staged))
        if len(self.staged) >= self.buffer_limit:
            self.flush()
        return self.peak_staged

    def flush(self):
        """Propagate the staged deltas.

        The R deltas are probed against the *old* S state and are then committed, so the S
        deltas are probed against the *new* R state.  That sequence accounts for the cross
        term dR x dS exactly, which is why the net delta equals `new view - old view`.
        """
        if not self.staged:
            return 0
        left_deltas = {}
        right_deltas = {}
        for (relation, row), weight in sorted(self.staged.items()):
            if not weight:
                continue
            (left_deltas if relation == self.left.name else right_deltas)[row] = weight

        delta_v1 = {}
        for (key, left_value), weight in sorted(left_deltas.items()):
            for right_value, right_count in sorted(self.right.index.get(key, {}).items()):
                self.incremental_probes += 1
                if right_value < self.min_right:
                    continue
                self._contribution(delta_v1, key, left_value, right_value,
                                   weight * right_count)
        for (key, left_value), weight in sorted(left_deltas.items()):
            self.left.apply_delta((key, left_value), weight)

        for (key, right_value), weight in sorted(right_deltas.items()):
            for left_value, left_count in sorted(self.left.index.get(key, {}).items()):
                self.incremental_probes += 1
                if right_value < self.min_right:
                    continue
                self._contribution(delta_v1, key, left_value, right_value,
                                   weight * left_count)
        for (key, right_value), weight in sorted(right_deltas.items()):
            self.right.apply_delta((key, right_value), weight)

        self.staged.clear()
        self.flushes += 1
        self._apply_delta_v1(delta_v1)
        return len(delta_v1)

    def apply_batch(self, batch):
        """Apply a batch of (relation, row, delta) triples.

        The batch is validated as a whole before anything is staged, so a batch that would
        drive a multiplicity below zero is rejected without leaving partial state behind.
        """
        net = {}
        for entry in batch:
            if len(entry) != 3:
                raise SchemaError("a batch entry must be (relation, row, delta), got %r" % (entry,))
            relation, row, delta = entry
            if relation not in self.relations:
                raise SchemaError("unknown relation %r" % (relation,))
            if delta not in (-1, 1):
                raise SchemaError("a delta must be +1 or -1, got %r" % (delta,))
            if not isinstance(row, tuple) or len(row) != 2:
                raise SchemaError("a row must be a two element tuple, got %r" % (row,))
            net[(relation, row)] = net.get((relation, row), 0) + delta
        for (relation, row), weight in net.items():
            if self.relations[relation].bag.counts.get(row, 0) + weight < 0:
                raise UnderflowError("the batch would delete %r more times than it exists" % (row,))
        self.last_delta_v1 = {}
        self.last_delta_v2 = {}
        applied = 0
        for (relation, row), weight in sorted(net.items()):
            for _ in range(abs(weight)):
                self.stage(relation, row, 1 if weight > 0 else -1)
                applied += 1
        self.flush()
        return applied

    def committed_v1(self):
        return dict(self.v1.counts)

    def committed_v2(self):
        return dict(self.v2)

    def full_rebuild(self):
        """Oracle path: throw the materialisation away and re-derive it from the base."""
        self.v1 = Bag()
        self.v2 = {}
        self._materialise()
        return self.committed_v1(), self.committed_v2()


# --------------------------------------------------------------------- batches
def random_batch(rng, left, right, keys, operations=3):
    """A batch that is valid for the current base state (only deletes existing rows)."""
    batch = []
    used = set()
    attempts = 0
    while len(batch) < operations and attempts < operations * 8:
        attempts += 1
        relation = left if rng.below(2) == 0 else right
        rows = relation.rows()
        if rows and rng.below(3) == 0:
            row = rows[rng.below(len(rows))]
            if (relation.name, row) in used:
                continue
            used.add((relation.name, row))
            batch.append((relation.name, row, -1))
        elif relation is left:
            batch.append((relation.name, (keys[rng.below(len(keys))], rng.below(10)), +1))
        else:
            batch.append((relation.name, (keys[rng.below(len(keys))], rng.below(4)), +1))
    return batch


def build_bases(rng, keys, left_per_key, right_per_key, max_value=9, max_right=3):
    left = BaseRelation("R", ("key", "a"))
    right = BaseRelation("S", ("key", "b"))
    for key in keys:
        for _ in range(left_per_key):
            left.add((key, rng.below(max_value + 1)), 1)
        for _ in range(right_per_key):
            right.add((key, rng.below(max_right + 1)), 1)
    return left, right


def reference_rows(relations):
    """Independent copy of the base data, maintained by the check harness itself."""
    return {name: list(relation.rows()) for name, relation in relations.items()}


def reference_apply(reference, batch):
    """Apply a batch to the plain-list reference, validating multiplicity by counting."""
    for relation, row, delta in batch:
        rows = reference[relation]
        if delta > 0:
            rows.append(row)
        else:
            if row not in rows:
                raise UnderflowError("reference cannot delete %r" % (row,))
            rows.remove(row)
    return reference


# ----------------------------------------------------------------------- main
def main():
    keys = list(range(8))
    rng = Rng(888001)

    # --- 1. bag semantics ---------------------------------------------------
    bag = Bag()
    bag.add(("k", 1))
    bag.add(("k", 1))
    check(bag.counts[("k", 1)] == 2, "a bag must count multiplicities")
    bag.add(("k", 1), -1)
    check(bag.counts[("k", 1)] == 1, "a delete must decrement the multiplicity")
    bag.add(("k", 1), -1)
    check(bag.counts == {}, "a zero multiplicity must be dropped from the bag")
    raised = False
    try:
        bag.add(("k", 1), -1)
    except UnderflowError:
        raised = True
    check(raised, "deleting a row that is not present must be rejected")
    check(bag.counts == {}, "a rejected delete must not corrupt the bag")

    left, right = build_bases(rng, keys, 3, 2)
    pipeline = ViewPipeline(left, right, min_right=1, buffer_limit=16)
    reference = reference_rows({"R": left, "S": right})
    oracle_v1 = brute_force_view(reference["R"], reference["S"], 1)
    check(pipeline.committed_v1() == oracle_v1,
          "the initial materialisation must equal the brute-force join")
    check(pipeline.committed_v2() == brute_force_sums(oracle_v1),
          "the initial aggregate must equal the brute-force aggregate")
    check(all(value > 0 for value in pipeline.committed_v1().values()),
          "bag multiplicities must stay positive")

    # --- 2. batch by batch differential testing -----------------------------
    batches = 0
    delta_checks = 0
    for trial in range(30):
        batch = random_batch(rng, left, right, keys)
        reference_apply(reference, batch)
        before_v2 = dict(pipeline.v2)
        pipeline.apply_batch(batch)
        batches += 1
        expected_v1 = brute_force_view(reference["R"], reference["S"], 1)
        expected_v2 = brute_force_sums(expected_v1)
        check(pipeline.committed_v1() == expected_v1,
              "V1 must match the oracle after batch %d: %r vs %r"
              % (trial, pipeline.committed_v1(), expected_v1))
        check(pipeline.committed_v2() == expected_v2,
              "V2 must match the oracle after batch %d" % trial)
        observed = {}
        for key in set(before_v2) | set(pipeline.v2):
            difference = pipeline.v2.get(key, 0) - before_v2.get(key, 0)
            if difference:
                observed[key] = difference
        check(observed == pipeline.last_delta_v2,
              "the propagated V2 delta must equal the difference of the two states")
        delta_checks += 1
        rebuilt_v1, rebuilt_v2 = pipeline.full_rebuild()
        check(rebuilt_v1 == pipeline.committed_v1() and rebuilt_v2 == pipeline.committed_v2(),
              "a full rebuild must agree with the incrementally maintained view")
        check(pipeline.peak_staged <= pipeline.buffer_limit,
              "the staging buffer must never exceed its limit")

    # --- 3. delta propagation through the chain ------------------------------
    for trial in range(10):
        batch = random_batch(rng, left, right, keys)
        reference_apply(reference, batch)
        pipeline.apply_batch(batch)
        aggregate = {}
        for (key, left_value, right_value), weight in pipeline.last_delta_v1.items():
            aggregate[key] = aggregate.get(key, 0) + weight * left_value * right_value
        check({key: value for key, value in aggregate.items() if value} == pipeline.last_delta_v2,
              "the aggregate delta must be exactly the V1 delta folded by the payload")

    # --- 4. exact multiplicities and impossible deletes ---------------------
    tiny_left = BaseRelation("R", ("key", "a"))
    tiny_right = BaseRelation("S", ("key", "b"))
    tiny_left.add((0, 5), 1)
    tiny_left.add((0, 5), 1)
    tiny_right.add((0, 2), 1)
    tiny = ViewPipeline(tiny_left, tiny_right, min_right=0, buffer_limit=4)
    check(tiny.committed_v1()[(0, 5, 2)] == 2, "a doubled left row must double the join tuple")
    check(tiny.committed_v2() == {0: 20}, "two copies of (0,5) joined to (0,2) must sum to 20")
    tiny.apply_batch([("R", (0, 5), -1)])
    check(tiny.committed_v2() == {0: 10}, "one delete must halve the contribution")
    check(tiny.committed_v1()[(0, 5, 2)] == 1, "the view multiplicity must follow the base")
    snapshot = dict(tiny.v2)
    raised = False
    try:
        tiny.apply_batch([("R", (0, 5), -1), ("R", (0, 5), -1)])
    except UnderflowError:
        raised = True
    check(raised, "deleting more copies than exist must be rejected")
    check(tiny.committed_v2() == snapshot,
          "a rejected batch must leave the materialised view untouched")

    # --- 5. canonicalisation: empty groups disappear ------------------------
    tiny.apply_batch([("R", (0, 5), -1)])
    check(tiny.committed_v2() == {}, "a group whose support disappears must vanish from V2")
    check(tiny.committed_v1() == {}, "V1 must be empty once its input is gone")
    tiny.apply_batch([("R", (0, 7), +1)])
    check(tiny.committed_v2() == {0: 14}, "a re-inserted row must come back exactly")

    # --- 6. the filter is applied on every path -----------------------------
    filtered_left = BaseRelation("R", ("key", "a"))
    filtered_right = BaseRelation("S", ("key", "b"))
    for key in range(4):
        for value in range(3):
            filtered_left.add((key, value + 1), 1)
            filtered_right.add((key, value), 1)
    filtered = ViewPipeline(filtered_left, filtered_right, min_right=2, buffer_limit=8)
    expected = brute_force_view(filtered_left.rows(), filtered_right.rows(), 2)
    check(filtered.committed_v1() == expected, "the filter must be applied at build time")
    check(all(entry[2] >= 2 for entry in expected), "no filtered-out row may reach V1")
    filtered.apply_batch([("S", (0, 1), +1)])
    check(filtered.committed_v1() == brute_force_view(filtered_left.rows(), filtered_right.rows(), 2),
          "a delta below the filter threshold must not change the view")
    before = dict(filtered.v2)
    filtered.apply_batch([("S", (0, 3), +1)])
    check(filtered.v2 != before, "a delta above the filter threshold must change the view")

    # --- 7. degenerate shapes ----------------------------------------------
    empty_left = BaseRelation("R", ("key", "a"))
    empty_right = BaseRelation("S", ("key", "b"))
    empty = ViewPipeline(empty_left, empty_right, min_right=0, buffer_limit=4)
    check(empty.committed_v1() == {} and empty.committed_v2() == {},
          "an empty relation set must produce empty views")
    check(empty.full_rebuild() == ({}, {}), "rebuilding nothing must stay empty")
    flushes_before = empty.flushes
    empty.apply_batch([])
    check(empty.flushes == flushes_before and empty.incremental_probes == 0,
          "an empty batch must cost nothing")
    lonely = BaseRelation("R", ("key", "a"))
    lonely_s = BaseRelation("S", ("key", "b"))
    lonely.add((3, 4), 1)
    lone = ViewPipeline(lonely, lonely_s, min_right=0, buffer_limit=4)
    check(lone.committed_v2() == {}, "a left row with no partner must produce no group")
    lone.apply_batch([("S", (3, 5), +1)])
    check(lone.committed_v2() == {3: 20}, "the late partner must materialise the group")

    # --- 8. invalid input is rejected --------------------------------------
    raised = False
    try:
        pipeline.stage("T", (0, 1), +1)
    except SchemaError:
        raised = True
    check(raised, "an unknown relation must be rejected")
    raised = False
    try:
        pipeline.stage("R", (0,), +1)
    except SchemaError:
        raised = True
    check(raised, "a row with the wrong arity must be rejected")
    raised = False
    try:
        pipeline.stage("R", (0, 1), +2)
    except SchemaError:
        raised = True
    check(raised, "a delta other than +1/-1 must be rejected")
    raised = False
    try:
        ViewPipeline(left, right, buffer_limit=0)
    except ValueError:
        raised = True
    check(raised, "a zero staging buffer must be rejected")

    # --- 9. bounded memory under a long delta stream ------------------------
    stream_left, stream_right = build_bases(Rng(888002), list(range(12)), 4, 2)
    stream = ViewPipeline(stream_left, stream_right, min_right=0, buffer_limit=3)
    stream_reference = reference_rows({"R": stream_left, "S": stream_right})
    for _ in range(60):
        batch = random_batch(rng, stream_left, stream_right, list(range(12)), operations=5)
        reference_apply(stream_reference, batch)
        stream.apply_batch(batch)
        check(len(stream.staged) <= 3, "the staging buffer must stay bounded during the stream")
        check(stream.peak_staged <= 3, "the peak staging size must respect the limit")
    check(stream.peak_staged == 3, "the stream must actually reach the staging limit")
    check(stream.flushes > 20, "a small buffer must force frequent flushes, got %d" % stream.flushes)
    check(stream.committed_v1()
          == brute_force_view(stream_reference["R"], stream_reference["S"], 0),
          "the streamed maintenance must still be exact")
    check(stream.peak_accumulator <= stream_left.key_count() + 1,
          "the aggregate accumulator must stay proportional to the number of groups")

    # --- 10. incremental cost versus a full rebuild -------------------------
    big_left, big_right = build_bases(Rng(888003), list(range(40)), 30, 5)
    big = ViewPipeline(big_left, big_right, min_right=1, buffer_limit=64)
    big_reference = reference_rows({"R": big_left, "S": big_right})
    rebuild_cost = big.full_probes
    big.incremental_probes = 0
    workload = 40
    for _ in range(workload):
        batch = random_batch(rng, big_left, big_right, list(range(40)), operations=3)
        reference_apply(big_reference, batch)
        big.apply_batch(batch)
    check(big.committed_v1() == brute_force_view(big_reference["R"], big_reference["S"], 1),
          "the large incremental run must equal the brute-force oracle")
    check(big.committed_v2() == brute_force_sums(big.committed_v1()),
          "the large aggregate must match the oracle")
    full_cost = rebuild_cost * workload
    check(big.incremental_probes * 10 < full_cost,
          "delta propagation must be far cheaper than %d full rebuilds: %d vs %d probes"
          % (workload, big.incremental_probes, full_cost))

    # --- 11. scale ----------------------------------------------------------
    scale_left = BaseRelation("R", ("key", "a"))
    scale_right = BaseRelation("S", ("key", "b"))
    scale_rng = Rng(888004)
    for key in range(30):
        for _ in range(100):
            scale_left.add((key, scale_rng.below(100)), 1)
        for _ in range(6):
            scale_right.add((key, scale_rng.below(4)), 1)
    scale = ViewPipeline(scale_left, scale_right, min_right=1, buffer_limit=32)
    scale_oracle = brute_force_view(scale_left.rows(), scale_right.rows(), 1)
    check(scale.committed_v1() == scale_oracle,
          "the scaled materialisation must equal the brute-force join")
    check(len(scale_oracle) >= 2500, "the scaled join must be substantial, got %d"
          % len(scale_oracle))
    scale.incremental_probes = 0
    for _ in range(25):
        for _ in range(4):
            scale.apply_batch([("R", (scale_rng.below(30), scale_rng.below(100)), +1)])
    check(scale.incremental_probes * 20 < scale.full_probes * 25,
          "the scaled incremental run must stay far below the rebuild cost")
    check(scale.committed_v1() == brute_force_view(scale_left.rows(), scale_right.rows(), 1),
          "the scaled view must still agree with the oracle")
    check(scale.peak_staged <= 32, "the scaled staging buffer must stay bounded")

    # --- 12. reproducibility ------------------------------------------------
    def replay(seed):
        local = Rng(seed)
        local_left, local_right = build_bases(local, list(range(6)), 3, 2)
        local_pipeline = ViewPipeline(local_left, local_right, min_right=1, buffer_limit=5)
        for _ in range(20):
            local_pipeline.apply_batch(random_batch(local, local_left, local_right,
                                                    list(range(6)), operations=2))
        return (local_pipeline.committed_v1(), local_pipeline.committed_v2(),
                local_pipeline.flushes, local_pipeline.peak_staged)

    check(replay(888500) == replay(888500), "an identical seed must replay the whole stream")

    print("differential batches: %d, delta equalities verified: %d"
          % (batches + 10 + 60 + workload, delta_checks))
    print("bounded memory: buffer limit 3 held %d distinct delta entries, %d flushes"
          % (stream.peak_staged, stream.flushes))
    print("cost: %d delta probes versus %d probes for %d full rebuilds (%.1fx cheaper)"
          % (big.incremental_probes, full_cost, workload,
             full_cost / max(1, big.incremental_probes)))
    print("scale: %d V1 entries from %d base rows, peak staging %d"
          % (len(scale.committed_v1()), scale_left.bag.total() + scale_right.bag.total(),
             scale.peak_staged))
    print("PASS")


if __name__ == "__main__":
    main()
