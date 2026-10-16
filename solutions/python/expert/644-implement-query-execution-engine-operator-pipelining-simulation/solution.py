"""Task 644 - Implement query execution engine operator pipelining simulation.  (Tier: Expert, Ch 14 — Databases)

A volcano/iterator-model execution engine: every operator implements open()/next()/close()
and hands one tuple at a time to its parent, so tuples flow *through* the pipeline instead
of being materialized between operators.  Blocking operators (Sort, and the build side of
HashJoin) must drain their child before they emit anything; streaming operators (Scan,
Filter, Project, Limit) hold no tuples between calls.

Verified invariants: (a) LIMIT over a pipeline pulls only the source tuples it needs, and
the number needed for the *first* result row is independent of the table size, (b) Sort
really is blocking - a LIMIT above it cannot stop the scan early, (c) a hash join consumes
its build side in open() but probes lazily, (d) every plan returns exactly what a naive
list-comprehension evaluator returns, (e) the volcano invariant `next_calls == rows_out + 1`
holds for a fully drained scan, and (f) peak buffering separates streaming from
materializing plans.

Simplified: rows are plain tuples, the only join is an equality join on one column, and
there are no page or buffer-pool iterators - just the calling discipline of the model.
"""


class Operator:
    label = "Operator"

    def __init__(self, *children):
        self.children = list(children)
        self.next_calls = 0
        self.rows_out = 0
        self.buffered = 0            # tuples this operator holds at its peak

    def open(self):
        for child in self.children:
            child.open()

    def next(self):
        raise NotImplementedError

    def close(self):
        for child in self.children:
            child.close()


class SeqScan(Operator):
    label = "SeqScan"

    def __init__(self, rows):
        super().__init__()
        self.rows = list(rows)
        self.position = 0
        self.rows_read = 0

    def open(self):
        self.position = 0
        self.rows_read = 0

    def next(self):
        self.next_calls += 1
        if self.position >= len(self.rows):
            return None
        row = self.rows[self.position]
        self.position += 1
        self.rows_read += 1
        self.rows_out += 1
        return row


class Filter(Operator):
    label = "Filter"

    def __init__(self, child, predicate):
        super().__init__(child)
        self.predicate = predicate

    def next(self):
        while True:
            self.next_calls += 1
            row = self.children[0].next()
            if row is None:
                return None
            if self.predicate(row):
                self.rows_out += 1
                return row


class Project(Operator):
    label = "Project"

    def __init__(self, child, columns):
        super().__init__(child)
        self.columns = tuple(columns)

    def next(self):
        self.next_calls += 1
        row = self.children[0].next()
        if row is None:
            return None
        self.rows_out += 1
        return tuple(row[i] for i in self.columns)


class Limit(Operator):
    label = "Limit"

    def __init__(self, child, count):
        super().__init__(child)
        self.count = count
        self.emitted = 0

    def open(self):
        self.emitted = 0
        self.children[0].open()

    def next(self):
        if self.emitted >= self.count:
            return None                  # an exhausted LIMIT never touches its child
        self.next_calls += 1
        row = self.children[0].next()
        if row is None:
            return None
        self.emitted += 1
        self.rows_out += 1
        return row


class Sort(Operator):
    """Blocking: the whole input is materialized in open(), then emitted in key order."""

    label = "Sort"

    def __init__(self, child, key_column):
        super().__init__(child)
        self.key_column = key_column
        self.materialized = []
        self.position = 0

    def open(self):
        child = self.children[0]
        child.open()
        self.materialized = []
        while True:
            row = child.next()
            if row is None:
                break
            self.materialized.append(row)
        self.buffered = len(self.materialized)
        child.close()
        self.materialized.sort(key=lambda row: row[self.key_column])
        self.position = 0

    def next(self):
        self.next_calls += 1
        if self.position >= len(self.materialized):
            return None
        row = self.materialized[self.position]
        self.position += 1
        self.rows_out += 1
        return row


class HashJoin(Operator):
    """Builds a hash table on the right child in open(); probes the left child lazily."""

    label = "HashJoin"

    def __init__(self, build_child, probe_child, build_key, probe_key):
        super().__init__(build_child, probe_child)
        self.build_child, self.probe_child = build_child, probe_child
        self.build_key, self.probe_key = build_key, probe_key
        self.buckets = {}
        self.pending = []
        self.pending_index = 0

    def open(self):
        self.build_child.open()
        self.buckets = {}
        built = 0
        while True:
            row = self.build_child.next()
            if row is None:
                break
            self.buckets.setdefault(row[self.build_key], []).append(row)
            built += 1
        self.buffered = built
        self.build_child.close()
        self.probe_child.open()
        self.pending, self.pending_index = [], 0

    def next(self):
        self.next_calls += 1
        while True:
            if self.pending_index < len(self.pending):
                row = self.pending[self.pending_index]
                self.pending_index += 1
                self.rows_out += 1
                return row
            probe_row = self.probe_child.next()
            if probe_row is None:
                return None
            self.pending = [probe_row + build_row
                            for build_row in self.buckets.get(probe_row[self.probe_key], ())]
            self.pending_index = 0


def walk(operator, out=None):
    if out is None:
        out = []
    out.append(operator)
    for child in operator.children:
        walk(child, out)
    return out


def scans_of(plan):
    return [op for op in walk(plan) if op.label == "SeqScan"]


def drain(plan):
    plan.open()
    rows = []
    while True:
        row = plan.next()
        if row is None:
            break
        rows.append(row)
    plan.close()
    return rows


def lcg(state):
    return (state * 1103515245 + 12345) % (2 ** 31)


def make_rows(count, key_domain, seed):
    rows, state = [], seed
    for i in range(count):
        state = lcg(state)
        rows.append((i, state % key_domain + 1, i * 3 + 1))
    return rows, state


def naive_join(probe_rows, build_rows, probe_key, build_key):
    """Oracle in the same emission order: probe-major, build-minor."""
    return [probe + build for probe in probe_rows for build in build_rows
            if probe[probe_key] == build[build_key]]


def main():
    rows, _ = make_rows(200, 7, 90210)
    is_even = lambda row: row[2] % 2 == 0                 # noqa: E731 - short local predicate
    expected_even = [row for row in rows if is_even(row)]

    # (a) LIMIT pulls only what it needs -------------------------------------------------
    plan = Limit(Filter(SeqScan(rows), is_even), 1)
    first = drain(plan)
    scan, filt = scans_of(plan)[0], walk(plan)[1]
    check(first == expected_even[:1], "the pipelined LIMIT returns the first matching row")
    check(scan.rows_read == 2 and scan.next_calls == 2,
          "only two source rows were pulled for one result: %d" % scan.rows_read)
    check(filt.rows_out == 1 and filt.next_calls == 2, "the filter forwarded exactly one row")

    limit_all = Limit(Filter(SeqScan(rows), is_even), len(expected_even) + 5)
    check(drain(limit_all) == expected_even, "LIMIT over the result equals the naive filter")
    scan_all = scans_of(limit_all)[0]
    check(scan_all.rows_read == len(rows), "the scan reads every row when nothing stops it")
    check(scan_all.next_calls == scan_all.rows_out + 1,
          "volcano invariant: a drained scan is called rows_out + 1 times")

    big_rows, _ = make_rows(10000, 7, 90210)
    big_plan = Limit(Filter(SeqScan(big_rows), is_even), 1)
    check(drain(big_plan) == first, "the same first row appears in the ten-times larger table")
    check(scans_of(big_plan)[0].rows_read == scan.rows_read,
          "time-to-first-row is independent of the input size")

    zero_plan = Limit(SeqScan(rows), 0)
    check(drain(zero_plan) == [], "LIMIT 0 returns nothing")
    check(scans_of(zero_plan)[0].rows_read == 0, "LIMIT 0 never touches the scan")

    # (b) Sort is blocking ---------------------------------------------------------------
    sort_plan = Limit(Sort(Filter(SeqScan(rows), is_even), 2), 3)
    top3 = drain(sort_plan)
    sort_scan, sort_op = scans_of(sort_plan)[0], walk(sort_plan)[1]
    check(top3 == sorted(expected_even, key=lambda row: row[2])[:3],
          "the top rows come out in sorted order")
    check(sort_scan.rows_read == len(rows),
          "the blocking sort drained the entire input first: %d rows" % sort_scan.rows_read)
    check(sort_op.buffered == len(expected_even) and sort_op.next_calls == 3,
          "the sort materialized %d rows and was pulled exactly three times" % sort_op.buffered)

    # (c) hash join: build side eager, probe side lazy ------------------------------------
    probe_rows, _ = make_rows(150, 5, 555)
    build_rows, _ = make_rows(60, 5, 777)
    surviving_probes = [row for row in probe_rows if is_even(row)]
    join = HashJoin(SeqScan(build_rows), Filter(SeqScan(probe_rows), is_even), 1, 1)
    joined = drain(join)
    oracle = naive_join(surviving_probes, build_rows, 1, 1)
    check(joined == oracle,
          "hash join result equals the nested-loop oracle (%d rows)" % len(joined))

    lazy_plan = HashJoin(SeqScan(build_rows), Filter(SeqScan(probe_rows), is_even), 1, 1)
    probe_scan = scans_of(lazy_plan)[1]
    lazy_plan.open()
    check(lazy_plan.build_child.rows_read == len(build_rows),
          "the build side is fully consumed during open()")
    check(probe_scan.rows_read == 0, "the probe side is untouched right after open()")
    first_join_row = lazy_plan.next()
    lazy_plan.close()
    check(first_join_row == oracle[0], "the first joined row matches the oracle")
    check(probe_scan.rows_read == 2,
          "the probe side was pulled only until the first surviving row (2 rows)")

    limited_join = Limit(HashJoin(SeqScan(build_rows),
                                  Filter(SeqScan(probe_rows), is_even), 1, 1), 2)
    first_two = drain(limited_join)
    check(first_two == oracle[:2], "LIMIT over the join keeps the first two joined rows")
    check(scans_of(limited_join)[1].rows_read < len(probe_rows),
          "LIMIT stopped the probe scan early: %d of %d rows"
          % (scans_of(limited_join)[1].rows_read, len(probe_rows)))

    # (d) projection and empty input -----------------------------------------------------
    project_plan = Project(Limit(Filter(SeqScan(rows), is_even), 5), (0, 2))
    check(drain(project_plan) == [(row[0], row[2]) for row in expected_even[:5]],
          "projection passes the pipelined rows through unchanged")
    empty_plan = Project(Filter(SeqScan([]), is_even), (0, 2))
    check(drain(empty_plan) == [] and scans_of(empty_plan)[0].next_calls == 1,
          "an empty input yields no rows and exactly one end-of-stream call")

    # (e) peak buffering separates streaming from materializing plans ---------------------
    streaming = Limit(Project(Filter(SeqScan(rows), is_even), (0, 2)), 4)
    drain(streaming)
    materializing = Limit(Project(Sort(Filter(SeqScan(rows), is_even), 2), (0, 2)), 4)
    drain(materializing)
    check(max(op.buffered for op in walk(streaming)) == 0, "a streaming plan buffers nothing")
    check(max(op.buffered for op in walk(materializing)) == len(expected_even),
          "the sorting plan buffers the whole intermediate")

    again = Limit(Filter(SeqScan(rows), is_even), 1)
    check(drain(again) == first and scans_of(again)[0].rows_read == scan.rows_read,
          "the same plan returns the same rows with the same statistics")
    print("[644] pipelining: LIMIT 1 pulled %d of %d rows; Sort drained %d rows before emitting"
          % (scan.rows_read, len(rows), sort_scan.rows_read))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
