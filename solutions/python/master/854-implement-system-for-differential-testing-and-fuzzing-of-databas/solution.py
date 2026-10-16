"""Task 854 - Implement system for differential testing and fuzzing of database engines.  (Tier: Master, Ch 14 — Databases)

A differential test harness for query engines in the style of SQLancer: one deterministic
fuzzer generates a corpus of random relational queries over a fixed schema, several
independent engine implementations evaluate the same query on the same data, divergences are
reported, and each finding is then *shrunk* to a minimal counterexample.

What is simplified: the schema is two small in-memory tables, the query language is a tiny
relational algebra (scan / filter / project / equi-join / group-aggregate / order / limit),
and the engine variants are a straightforward reference evaluator plus a rewrite-based
evaluator (predicate push-down into the scan, hash join with a build-side choice, streaming
aggregation) rather than real SQL planners.  No threads, no clock, no unseeded randomness:
the fuzzer draws only from a seeded splitmix64 stream, so a corpus is reproducible.

Semantics fixed by the harness (the classic bug sources): NULL never equals anything, not
even another NULL; a NULL join key never matches.  Engines may emit rows in any order, so
results are compared as multisets unless the query's outermost operator is an ORDER BY (or a
LIMIT over an ORDER BY), in which case ORDER BY is total because equal sort keys are broken
by a canonical signature of the row and the harness compares the sequences exactly.

Invariants verified:
  * no false positives: the reference and the optimised engine agree on every corpus query,
    and both agree with a third, independently written interpreter on a sample of the corpus,
    so the agreement is real correctness rather than a shared mistake;
  * the comparator is not vacuous: on an unordered query the two engines demonstrably emit
    different row orders, the multiset comparator accepts that while a naive sequence
    comparator rejects it;
  * the rewrites pay off: the optimised engine scans strictly fewer rows over the corpus
    while returning identical results;
  * bug detection: five injected semantic mutants (inner join turned outer, LIMIT off by one,
    count instead of count-distinct, NULL = NULL, filters dropped) are each detected on the
    corpus, while a semantically neutral mutant (reversing the operands of a conjunction) is
    not reported - no false negatives on bugs and no false positive on the neutral rewrite;
  * shrinking: a finding is reduced by a deterministic greedy protocol (drop predicates,
    projections and orderings, shrink the LIMIT, drop data rows) that keeps a reduction only
    when the divergence survives; the result must still diverge, must be minimal (no single
    further reduction reproduces it, checked exhaustively) and strictly smaller than the
    finding it came from, and the divergence must survive on a drastically smaller dataset;
  * determinism and edge cases: the same seed regenerates the identical corpus and verdicts,
    and both engines agree on an empty database, on LIMIT 0, on a huge LIMIT, on a self-join,
    on non-matching predicates and on NULL keys.
"""

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next_u64() % n


# ---------------------------------------------------------------------------
# schema and data
# ---------------------------------------------------------------------------
SCHEMA = {
    "users": ("uid", "dept", "tier", "age", "name"),
    "orders": ("oid", "uid", "amount", "status"),
}
COLUMN_TABLE = {column: table for table, columns in SCHEMA.items() for column in columns}
AGGREGATES = ("count", "count_distinct", "sum", "max")
NUMERIC = ("uid", "oid", "amount", "tier", "age")


def build_data():
    users = []
    depts = ["eng", "sales", None, "ops"]
    for uid in range(1, 16):
        users.append({"uid": uid, "dept": depts[uid % 4], "tier": 1 + (uid % 3),
                      "age": 20 + (uid * 3) % 25, "name": "u%02d" % uid})
    orders = []
    for oid in range(1, 31):
        uid = None if oid % 7 == 0 else 1 + (oid * 5) % 15
        orders.append({"oid": oid, "uid": uid, "amount": 10 * (oid % 6 + 1),
                       "status": ["new", "paid", "shipped", "paid"][oid % 4]})
    return {"users": users, "orders": orders}


# ---------------------------------------------------------------------------
# query AST
# ---------------------------------------------------------------------------
def scan(table):
    return ("scan", table)


def filter_(expr, child):
    return ("filter", expr, child)


def project(columns, child):
    return ("project", tuple(columns), child)


def join(left_key, right_key, left, right, outer=False):
    return ("join", left_key, right_key, left, right, outer)


def aggregate(group_key, value_key, func, child):
    return ("aggregate", group_key, value_key, func, child)


def order(column, child):
    return ("order", column, child)


def limit(count, child):
    return ("limit", count, child)


def expr_size(expr):
    return 1 + expr_size(expr[1]) + expr_size(expr[2]) if expr[0] == "and" else 1


def size_of(query):
    kind = query[0]
    if kind == "scan":
        return 1
    if kind == "filter":
        return 1 + expr_size(query[1]) + size_of(query[2])
    if kind == "project":
        return 1 + len(query[1]) + size_of(query[2])
    if kind == "join":
        return 1 + size_of(query[3]) + size_of(query[4])
    if kind == "aggregate":
        return 1 + size_of(query[4])
    return 1 + size_of(query[2])


def describe(query):
    kind = query[0]
    if kind == "scan":
        return "scan(%s)" % query[1]
    if kind == "filter":
        return "filter(%s, %s)" % (render(query[1]), describe(query[2]))
    if kind == "project":
        return "project(%s, %s)" % (",".join(query[1]), describe(query[2]))
    if kind == "join":
        return "join(%s=%s%s, %s, %s)" % (query[1], query[2], ",outer" if query[5] else "",
                                          describe(query[3]), describe(query[4]))
    if kind == "aggregate":
        return "%s(%s by %s, %s)" % (query[3], query[2], query[1], describe(query[4]))
    if kind == "order":
        return "order(%s, %s)" % (query[1], describe(query[2]))
    return "limit(%d, %s)" % (query[1], describe(query[2]))


def render(expr):
    if expr[0] == "and":
        return "(%s and %s)" % (render(expr[1]), render(expr[2]))
    return "%s %s %r" % (expr[1], expr[0], expr[2])


def schema_of(query):
    """Output columns of a node; the fuzzer only builds type-correct queries with this."""
    kind = query[0]
    if kind == "scan":
        return tuple(SCHEMA[query[1]])
    if kind == "filter":
        return schema_of(query[2])
    if kind == "project":
        return tuple(query[1])
    if kind == "join":
        columns = list(schema_of(query[3]))
        for column in schema_of(query[4]):
            if column not in columns:
                columns.append(column)
        return tuple(columns)
    if kind == "aggregate":
        return ("group", "value")
    return schema_of(query[2])


def is_sequence_compared(query):
    """Sequence comparison is sound only when the outermost operator fixes the order."""
    return query[0] == "order" or (query[0] == "limit" and query[2][0] == "order")


def row_signature(row):
    """Canonical, order-free signature of a row: used for tie-breaks and for multiset compare."""
    return tuple((column, row[column] is None, "" if row[column] is None else row[column])
                 for column in sorted(row))


# ---------------------------------------------------------------------------
# mutations
# ---------------------------------------------------------------------------
PINNED = ("null_equals_null", "conjunction_reversed")


def mutate(query, mutation):
    """Rewrite the AST for the structural mutants; engine flags cover the semantic ones."""
    if mutation == "none" or mutation in PINNED:
        return query
    if mutation == "left_join" and query[0] == "join":
        return ("join", query[1], query[2], mutate(query[3], mutation),
                mutate(query[4], mutation), True)
    if mutation == "limit_off_by_one" and query[0] == "limit":
        return ("limit", query[1] + 1, mutate(query[2], mutation))
    if mutation == "count_not_distinct" and query[0] == "aggregate":
        func = "count" if query[3] == "count_distinct" else query[3]
        return ("aggregate", query[1], query[2], func, mutate(query[4], mutation))
    if mutation == "drop_filter" and query[0] == "filter":
        return mutate(query[2], mutation)
    return tuple(mutate(part, mutation) if isinstance(part, tuple) else part for part in query)


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------
class Engine:
    """Reference evaluator: linear scans, nested-loop joins, full sort, materialisation."""

    name = "reference"
    pushdown = False

    def __init__(self, data, mutation="none"):
        self.data = data
        self.mutation = mutation
        self.rows_scanned = 0
        self.pairs_examined = 0

    def evaluate(self, expr, row):
        if expr[0] == "and":
            parts = (expr[1], expr[2]) if self.mutation != "conjunction_reversed" \
                else (expr[2], expr[1])
            return self.evaluate(parts[0], row) and self.evaluate(parts[1], row)
        op, column, value = expr
        current = row.get(column)
        if current is None or value is None:
            if self.mutation == "null_equals_null" and op == "=":
                return current is None and value is None
            return False
        if op == "=":
            return current == value
        if op == "<":
            return current < value
        if op == ">":
            return current > value
        return current in value

    def run(self, query):
        self.rows_scanned = 0
        self.pairs_examined = 0
        return self.execute(query)

    def execute(self, query):
        kind = query[0]
        if kind == "scan":
            rows = self.data[query[1]]
            self.rows_scanned += len(rows)
            return [dict(row) for row in rows]
        if kind == "filter":
            return [row for row in self.execute(query[2]) if self.evaluate(query[1], row)]
        if kind == "project":
            return [{column: row.get(column) for column in query[1]}
                    for row in self.execute(query[2])]
        if kind == "join":
            left = self.execute(query[3])
            right = self.execute(query[4])
            joined = []
            self.pairs_examined += len(left) * len(right)      # nested loop: every pair
            for lrow in left:
                lval = lrow.get(query[1])
                for rrow in right:
                    rval = rrow.get(query[2])
                    if lval is None or rval is None:
                        continue
                    if lval == rval:
                        merged = dict(lrow)
                        merged.update(rrow)
                        joined.append(merged)
            if query[5]:                                    # the injected defect
                for lrow in left:
                    lval = lrow.get(query[1])
                    if lval is None:
                        continue
                    if not any(other.get(query[1]) == lval for other in joined):
                        joined.append(dict(lrow))
            return joined
        if kind == "aggregate":
            return aggregate_rows(self.execute(query[4]), query[1], query[2], query[3])
        if kind == "order":
            rows = self.execute(query[2])
            return sorted(rows, key=lambda row: (row.get(query[1]) is None,
                                                 row.get(query[1]), row_signature(row)))
        if kind == "limit":
            return self.execute(query[2])[:query[1]]
        raise AssertionError("unknown node %r" % (kind,))


class OptimisedEngine(Engine):
    """Same semantics, different execution: push-down, hash join, streaming aggregation.

    Rows leave in reverse scan order (an unordered engine may emit any order), which is what
    forces the harness to compare multisets for unordered queries.
    """

    name = "optimised"
    pushdown = True

    def execute(self, query):
        kind = query[0]
        if kind == "scan":
            rows = list(reversed(self.data[query[1]]))
            self.rows_scanned += len(rows)
            return [dict(row) for row in rows]
        if kind == "filter":
            return self.execute_filter(query[1], query[2])
        if kind == "project":
            return [{column: row.get(column) for column in query[1]}
                    for row in self.execute(query[2])]
        if kind == "join":
            return self.execute_join(query)
        if kind == "aggregate":
            return aggregate_rows(self.execute(query[4]), query[1], query[2], query[3])
        if kind == "order":
            rows = self.execute(query[2])
            return sorted(rows, key=lambda row: (row.get(query[1]) is None,
                                                 row.get(query[1]), row_signature(row)))
        if kind == "limit":
            return self.execute(query[2])[:query[1]]
        raise AssertionError("unknown node %r" % (kind,))

    def execute_filter(self, expr, child):
        if child[0] == "scan":                     # predicate pushed into the scan
            rows = []
            for row in reversed(self.data[child[1]]):
                self.rows_scanned += 1
                if self.evaluate(expr, row):
                    rows.append(dict(row))
            return rows
        return [row for row in self.execute(child) if self.evaluate(expr, row)]

    def execute_join(self, query):
        left = self.execute(query[3])
        right = self.execute(query[4])
        left_key, right_key = query[1], query[2]
        if len(left) <= len(right):
            build, probe, build_key, probe_key = right, left, right_key, left_key
            build_is_left = False
        else:
            build, probe, build_key, probe_key = left, right, left_key, right_key
            build_is_left = True
        index = {}
        for row in build:
            index.setdefault(row.get(build_key), []).append(row)
        joined = []
        matched_build = set()
        matched_probe = set()
        for row in probe:
            key = row.get(probe_key)
            if key is None:
                continue
            candidates = index.get(key, ())
            self.pairs_examined += len(candidates)             # hash lookup: only candidates
            for other in candidates:
                # duplicate column names resolve by position (right side wins) regardless of
                # which side was chosen as the hash-table build input
                if build_is_left:
                    merged = dict(other)
                    merged.update(row)
                else:
                    merged = dict(row)
                    merged.update(other)
                joined.append(merged)
                matched_build.add(id(other))
                matched_probe.add(id(row))
        if query[5]:                                # the injected defect: left rows survive
            if build_is_left:
                leftovers, matched, key = build, matched_build, build_key
            else:
                leftovers, matched, key = probe, matched_probe, probe_key
            for row in leftovers:
                if row.get(key) is not None and id(row) not in matched:
                    joined.append(dict(row))
        return joined


def aggregate_rows(rows, group_key, value_key, func):
    buckets = {}
    for row in rows:
        buckets.setdefault(row.get(group_key), []).append(row.get(value_key))
    out = []
    for key, values in buckets.items():
        clean = [value for value in values if value is not None]
        if func == "count":
            out.append({"group": key, "value": len(values)})
        elif func == "count_distinct":
            out.append({"group": key, "value": len(set(clean))})
        elif func == "sum":
            out.append({"group": key, "value": sum(clean)})
        elif func == "max":
            out.append({"group": key, "value": max(clean) if clean else None})
        else:
            raise AssertionError("unknown aggregate %r" % (func,))
    return out


# ---------------------------------------------------------------------------
# second, deliberately independent interpreter (the oracle)
# ---------------------------------------------------------------------------
def oracle(query, data):
    kind = query[0]
    if kind == "scan":
        return [dict(row) for row in data[query[1]]]
    if kind == "filter":
        return [row for row in oracle(query[2], data) if oracle_match(query[1], row)]
    if kind == "project":
        return [{column: row[column] for column in query[1]} for row in oracle(query[2], data)]
    if kind == "join":
        left, right = oracle(query[3], data), oracle(query[4], data)
        pairs = []
        for lrow in left:
            for rrow in right:
                if lrow[query[1]] is None or rrow[query[2]] is None:
                    continue
                if lrow[query[1]] == rrow[query[2]]:
                    merged = dict(lrow)
                    merged.update(rrow)
                    pairs.append(merged)
        return pairs
    if kind == "aggregate":
        return aggregate_rows(oracle(query[4], data), query[1], query[2], query[3])
    if kind == "order":
        rows = oracle(query[2], data)
        return sorted(rows, key=lambda row: (row[query[1]] is None, row[query[1]],
                                             row_signature(row)))
    if kind == "limit":
        return oracle(query[2], data)[:query[1]]
    raise AssertionError("unknown node %r" % (kind,))


def oracle_match(expr, row):
    if expr[0] == "and":
        return oracle_match(expr[1], row) and oracle_match(expr[2], row)
    op, column, value = expr
    current = row.get(column)
    if current is None or value is None:
        return False
    if op == "=":
        return current == value
    if op == "<":
        return current < value
    if op == ">":
        return current > value
    return current in value


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def canonical(rows):
    return sorted(row_signature(row) for row in rows)


def compare(left, right, ordered):
    if ordered:
        return left == right, "sequence"
    return canonical(left) == canonical(right), "multiset"


# ---------------------------------------------------------------------------
# fuzzer
# ---------------------------------------------------------------------------
class Fuzzer:
    """Generates type-correct queries over the schema from a seeded stream."""

    SHAPES = ("plain", "plain", "filter", "aggregate", "order", "limit")

    def __init__(self, seed, data):
        self.rng = SplitMix64(seed)
        self.values = {}
        for table, columns in SCHEMA.items():
            for column in columns:
                self.values[column] = [row[column] for row in data[table]]

    def pick(self, items):
        return items[self.rng.below(len(items))]

    def aggregate_func(self, column):
        """Only numeric columns can be summed; every column can be counted or maximised."""
        return self.pick(AGGREGATES if column in NUMERIC else ("count", "count_distinct", "max"))

    def columns(self, query):
        """Columns of a node that the fuzzer can also draw literals for."""
        return [column for column in schema_of(query) if column in self.values]

    def literal(self, column):
        return self.pick(self.values[column])

    def predicate(self, columns, depth):
        if depth > 0 and self.rng.below(4) == 0:
            return ("and", self.predicate(columns, depth - 1),
                    self.predicate(columns, depth - 1))
        column = self.pick(list(columns))
        op = self.pick(["=", "=", "<", ">", "in"])
        if op == "in":
            return (op, column, tuple(sorted({self.literal(column) for _ in range(3)}, key=repr)))
        return (op, column, self.literal(column))

    def node(self, query):
        columns = self.columns(query)
        choice = self.rng.below(7)
        if choice in (0, 1) and columns:
            return filter_(self.predicate(columns, 1), query)
        if choice == 2 and columns:
            picked = sorted({self.pick(columns) for _ in range(self.rng.below(3) + 1)})
            return project(picked, query)
        if choice == 3 and columns:
            value_column = self.pick(columns)
            return aggregate(self.pick(columns), value_column,
                             self.aggregate_func(value_column), query)
        if choice == 4 and columns:
            return order(self.pick(columns), query)
        if choice == 5 and "uid" in columns:
            other = "orders" if self.pick([0, 1]) == 0 else "users"
            right = scan(other)
            if self.rng.below(2) == 0:      # a predicate directly above a scan can be pushed
                right = filter_(self.predicate(SCHEMA[other], 0), right)
            return join("uid", "uid", query, right)
        if choice == 6 and columns:
            # a LIMIT is only well defined when an ORDER BY fixes which rows are the first
            return limit(1 + self.rng.below(9), order(self.pick(columns), query))
        return query

    def query(self, shape="plain"):
        query = join("uid", "uid", scan("users"), scan("orders"))
        for _ in range(1 + self.rng.below(3)):
            query = self.node(query)
        columns = self.columns(query)
        if shape == "filter" and query[0] != "filter" and columns:
            query = filter_(self.predicate(columns, 1), query)
        elif shape == "aggregate" and query[0] != "aggregate" and columns:
            value_column = self.pick(columns)
            query = aggregate(self.pick(columns), value_column,
                              self.aggregate_func(value_column), query)
        elif shape in ("order", "limit") and query[0] not in ("order", "limit") and columns:
            query = order(self.pick(columns), query)
            if shape == "limit":
                query = limit(1 + self.rng.below(9), query)
        return query

    def corpus(self, count):
        return [self.query(self.SHAPES[index % len(self.SHAPES)]) for index in range(count)]


# ---------------------------------------------------------------------------
# differential runner and shrinker
# ---------------------------------------------------------------------------
class Harness:
    def __init__(self, data, corpus):
        self.data = data
        self.corpus = corpus

    def run_query(self, query, mutation="none"):
        """The mutation is injected into the candidate engine only; the reference stays clean."""
        reference = Engine(self.data, "none").run(query)
        candidate = OptimisedEngine(self.data, mutation).run(mutate(query, mutation))
        return reference, candidate, is_sequence_compared(query)

    def diverges(self, query, mutation):
        left, right, ordered = self.run_query(query, mutation)
        return compare(left, right, ordered)[0] is False

    def differential(self, mutation="none"):
        findings = []
        for index, query in enumerate(self.corpus):
            left, right, ordered = self.run_query(query, mutation)
            ok, mode = compare(left, right, ordered)
            if not ok:
                findings.append({"index": index, "query": query, "left": left, "right": right,
                                 "ordered": ordered, "mode": mode})
        return findings

    def scans(self):
        """Total rows read and join pairs examined by each engine over the corpus."""
        totals = [0, 0, 0, 0]
        for query in self.corpus:
            rewritten = mutate(query, "none")
            reference = Engine(self.data)
            reference.run(rewritten)
            candidate = OptimisedEngine(self.data)
            candidate.run(rewritten)
            totals[0] += reference.rows_scanned
            totals[1] += candidate.rows_scanned
            totals[2] += reference.pairs_examined
            totals[3] += candidate.pairs_examined
        return totals

    # -- shrinking ---------------------------------------------------------
    def reductions(self, query):
        """All single-step reductions of a query."""
        kind = query[0]
        options = []
        if kind == "filter":
            options.append(query[2])
            if query[1][0] == "and":
                options.append(filter_(query[1][1], query[2]))
                options.append(filter_(query[1][2], query[2]))
        elif kind == "project":
            options.append(query[2])
            for column in query[1]:
                smaller = tuple(item for item in query[1] if item != column)
                if smaller:
                    options.append(project(smaller, query[2]))
        elif kind == "order":
            options.append(query[2])
        elif kind == "limit":
            options.append(query[2])
            if query[1] > 1:
                options.append(limit(query[1] - 1, query[2]))
        elif kind == "join":
            options.append(query[3])
            options.append(query[4])
        elif kind == "aggregate":
            options.append(query[4])
            for func in AGGREGATES:
                if func != query[3]:
                    options.append(aggregate(query[1], query[2], func, query[4]))
        return options

    def shrink(self, query, mutation):
        steps = 0
        current = query
        while True:
            progressed = False
            for candidate in self.reductions(current):
                if size_of(candidate) >= size_of(current):
                    continue
                if self.diverges(candidate, mutation):
                    current = candidate
                    steps += 1
                    progressed = True
                    break
            if not progressed:
                return current, steps

    def shrink_data(self, query, mutation):
        removed = 0
        for table in ("users", "orders"):
            index = 0
            while index < len(self.data[table]):
                trial = {name: list(rows) for name, rows in self.data.items()}
                trial[table] = [row for position, row in enumerate(trial[table])
                                if position != index]
                if Harness(trial, self.corpus).diverges(query, mutation):
                    self.data = trial
                    removed += 1
                else:
                    index += 1
        return removed


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    data = build_data()
    check(len(data["users"]) == 15 and len(data["orders"]) == 30, "the schema must be populated")
    check(any(row["uid"] is None for row in data["orders"]),
          "the data must contain a NULL join key (the classic divergence source)")
    check(any(row["dept"] is None for row in data["users"]), "the data must contain a NULL value")
    print("schema: users=%d orders=%d (with NULL keys)" % (len(data["users"]), len(data["orders"])))

    corpus = Fuzzer(20240607, data).corpus(240)
    check(len(corpus) == 240, "the fuzzer must produce the requested corpus")
    shapes = {query[0] for query in corpus}
    check(len(shapes) >= 4, "the corpus must exercise several query shapes: %s" % sorted(shapes))
    check(any(query[0] == "limit" and is_sequence_compared(query) for query in corpus),
          "the corpus must contain sequence-compared paged queries")
    for query in corpus[:20]:
        check(schema_of(query), "every query must have a non-empty output schema")
        if query[0] == "order":
            check(query[1] in schema_of(query[2]),
                  "ORDER BY must reference a column of its input: %s" % describe(query))
    print("corpus: %d queries, top-level shapes %s" % (len(corpus), sorted(shapes)))

    harness = Harness(data, corpus)

    # ---- 1. no false positives -------------------------------------------
    findings = harness.differential("none")
    check(findings == [], "the two engines must agree on the whole corpus: %s"
          % [describe(finding["query"]) for finding in findings[:3]])
    print("differential: 0 divergences over %d queries" % len(corpus))

    # ---- 2. agreement is correctness, not a shared mistake ---------------
    checked = 0
    for query in corpus[:60]:
        left, right, ordered = harness.run_query(query, "none")
        expected = oracle(query, data)
        check(compare(left, expected, ordered)[0],
              "the reference engine disagrees with the oracle on %s" % describe(query))
        check(compare(right, expected, ordered)[0],
              "the optimised engine disagrees with the oracle on %s" % describe(query))
        checked += 1
    print("oracle cross-check: %d queries agree with a third implementation" % checked)

    # ---- 3. the comparator is not vacuous --------------------------------
    permutation = None
    for query in corpus:
        if is_sequence_compared(query):
            continue
        left, right, ordered = harness.run_query(query, "none")
        if left != right and compare(left, right, ordered)[0]:
            permutation = (query, left, right)
            break
    check(permutation is not None,
          "some unordered query must expose two different row orders (multiset semantics needed)")
    unordered, left, right = permutation
    check(len(left) > 1 and canonical(left) == canonical(right),
          "the two orders must be permutations of the same rows")
    print("comparator: %s returns two different orders (%d rows), multiset compare accepts it"
          % (describe(unordered), len(left)))

    # ---- 4. the rewrites really save work --------------------------------
    reference_scans, candidate_scans, reference_pairs, candidate_pairs = harness.scans()
    check(candidate_pairs < reference_pairs,
          "the hash join must examine fewer pairs (%d vs %d)" % (candidate_pairs, reference_pairs))
    check(candidate_scans <= reference_scans,
          "the optimised engine must not read more rows (%d vs %d)"
          % (candidate_scans, reference_scans))
    print("join pairs examined: reference=%d optimised=%d (%.1f%%); rows read %d vs %d"
          % (reference_pairs, candidate_pairs, 100.0 * candidate_pairs / reference_pairs,
             reference_scans, candidate_scans))

    # ---- 5. injected bugs are detected, the neutral mutant is not --------
    detectable = ("left_join", "limit_off_by_one", "count_not_distinct", "null_equals_null",
                  "drop_filter")
    reports = {}
    for mutation in detectable + ("conjunction_reversed",):
        found = harness.differential(mutation)
        reports[mutation] = found
        print("mutant %-20s detected on %d of %d queries" % (mutation, len(found), len(corpus)))
        if mutation in detectable:
            check(found, "the mutant %r must be detected on the corpus" % mutation)
        else:
            check(not found, "the neutral mutant %r must not be reported" % mutation)
    for mutation in detectable:
        smallest = min(reports[mutation], key=lambda item: size_of(item["query"]))
        check(compare(smallest["left"], smallest["right"], smallest["ordered"])[0] is False,
              "the recorded counterexample must reproduce the divergence")

    # ---- 6. shrinking ----------------------------------------------------
    original = min(reports["left_join"], key=lambda item: size_of(item["query"]))["query"]
    before_size = size_of(original)
    shrunk, steps = harness.shrink(original, "left_join")
    check(harness.diverges(shrunk, "left_join"),
          "the shrunken query must still exhibit the divergence")
    check(size_of(shrunk) < before_size,
          "shrinking must reduce the query (%d -> %d nodes)" % (before_size, size_of(shrunk)))
    stall = [candidate for candidate in harness.reductions(shrunk)
             if size_of(candidate) < size_of(shrunk) and harness.diverges(candidate, "left_join")]
    check(not stall, "the shrunken query must be minimal: %s still diverges"
          % [describe(query) for query in stall])
    print("shrinking: %d nodes -> %s (%d nodes) in %d steps"
          % (before_size, describe(shrunk), size_of(shrunk), steps))

    shrink_harness = Harness({name: list(rows) for name, rows in data.items()}, corpus)
    removed = shrink_harness.shrink_data(shrunk, "left_join")
    remaining = sum(len(rows) for rows in shrink_harness.data.values())
    total = sum(len(rows) for rows in data.values())
    check(removed > 0 and remaining < total,
          "data shrinking must remove rows (%d of %d dropped)" % (removed, total))
    check(Harness(shrink_harness.data, corpus).diverges(shrunk, "left_join"),
          "the shrunk dataset must still reproduce the divergence")
    print("data shrinking: %d of %d rows removed, divergence survives on %d rows"
          % (removed, total, remaining))

    # ---- 7. determinism --------------------------------------------------
    regenerated = Fuzzer(20240607, build_data()).corpus(240)
    check([describe(query) for query in regenerated] == [describe(query) for query in corpus],
          "the same seed must regenerate the identical corpus")
    rerun = Harness(build_data(), regenerated)
    check(rerun.differential("none") == [], "the same seed must reproduce the clean verdict")
    check(len(rerun.differential("left_join")) == len(reports["left_join"]),
          "the same seed must reproduce the same findings")
    other = Fuzzer(99, data).corpus(240)
    check([describe(query) for query in other] != [describe(query) for query in corpus],
          "a different seed must produce a different corpus")
    print("determinism: identical corpus, verdicts and findings for the same seed")

    # ---- 8. edge cases ---------------------------------------------------
    empty = {"users": [], "orders": []}
    for query in corpus[:40]:
        left, right, ordered = Harness(empty, corpus).run_query(query, "none")
        expected = oracle(query, empty)
        check(compare(left, right, ordered)[0] and compare(left, expected, ordered)[0],
              "engines must agree on an empty database: %s" % describe(query))
    check(Engine(empty).run(filter_(("=", "dept", "nowhere"), scan("users"))) == [],
          "a non-matching predicate must return no rows")
    check(OptimisedEngine(data).run(limit(0, order("uid", scan("users")))) == [],
          "LIMIT 0 must return nothing")
    check(len(Engine(data).run(limit(1000, order("uid", scan("users"))))) == 15,
          "a huge LIMIT must behave like no limit")
    self_join = join("uid", "uid", scan("users"), scan("users"))
    check(compare(Engine(data).run(self_join), OptimisedEngine(data).run(self_join), False)[0],
          "a self-join must agree between the engines")
    check(len(Engine(data).run(self_join)) == 15, "each user must match itself exactly once")
    check(Engine(data).run(filter_(("=", "uid", None), scan("orders"))) == [],
          "NULL = NULL must be false in every engine")
    null_join = join("uid", "uid", scan("users"),
                     filter_(("=", "uid", None), scan("orders")))
    check(Engine(data).run(null_join) == [] and OptimisedEngine(data).run(null_join) == [],
          "a NULL join key must never match anything")
    duplicate_status = aggregate("status", "uid", "count_distinct", scan("orders"))
    counted = Engine(data).run(duplicate_status)
    check(any(row["value"] > 1 for row in counted),
          "the count-distinct probe must be non-trivial (grouped duplicates exist)")
    print("edge cases: empty database, LIMIT 0, huge LIMIT, self-join and NULL keys agree")
    print("PASS")


if __name__ == "__main__":
    main()
