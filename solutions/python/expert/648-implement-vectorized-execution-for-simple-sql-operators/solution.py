"""Task 648 - Implement vectorized execution for simple SQL operators.  (Tier: Expert, Ch 14 — Databases)

A batch-at-a-time ("vectorized") execution engine: the scan emits column batches of fixed
width, a selection builds a boolean mask for the whole batch and *compacts* the surviving
slots, projection is a per-column slice, and the hash aggregate dictionary-encodes the group
key of every batch (one slot per distinct key seen in that batch), accumulates per-batch
partials and merges them into the global groups.  Each of those steps is one operator call
per batch instead of one call per tuple, which is what vectorization buys; the checks count
the calls and compare against a tuple-at-a-time engine computing the same answer.

Documented simplification: there is no SIMD, so the number of *predicate evaluations* is the
same as in the row engine - the checks assert that too rather than pretending otherwise -
and only the operator invocation count (the interpretation overhead) drops.  Batch
boundaries are handled correctly: a group split across two batches is partially accumulated
in each and merged.

Invariants verified: identical results for scan/filter/project/aggregate against the
row-at-a-time engine (empty input, one row, exact multiples of the batch size, a straddling
last batch), a dense (hole-free) filtered batch, per-batch dictionary slots bounded by the
rows of that batch, call counts monotone in the batch size, and deterministic output.
"""

BATCH = 8
AGGREGATES = ("sum", "count_star", "count", "min", "max")


class Aggregator:
    def __init__(self, name):
        if name not in AGGREGATES:
            raise AssertionError("unknown aggregate %r" % (name,))
        self.name = name

    def identity(self):
        return 0 if self.name in ("sum", "count_star", "count") else None

    def step(self, acc, value):
        if self.name == "count_star":
            return acc + 1
        if value is None:                       # SQL aggregates skip NULLs
            return acc
        if self.name == "sum":
            return acc + value
        if self.name == "count":
            return acc + 1
        if self.name == "min":
            return value if acc is None or value < acc else acc
        return value if acc is None or value > acc else acc

    def merge(self, left, right):
        if right is None:
            return left
        if left is None:
            return right
        if self.name in ("sum", "count", "count_star"):
            return left + right
        return min(left, right) if self.name == "min" else max(left, right)


def to_batch(rows, columns=("id", "group", "value")):
    return {column: [row[index] for row in rows] for index, column in enumerate(columns)}


def scan(rows, batch_size, stats):
    """Yield column batches; one operator call per batch, one tuple read per row."""
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        stats["calls"] += 1
        stats["rows_read"] += len(chunk)
        yield to_batch(chunk)
    if not rows:
        stats["calls"] += 1                      # the empty stream still costs one call


def vector_filter(batch, predicate, stats):
    """One call per batch: build the mask, then compact the selected slots."""
    stats["calls"] += 1
    mask = [predicate(value) for value in batch["value"]]
    stats["comparisons"] += len(mask)
    keep = [index for index, selected in enumerate(mask) if selected]
    return {column: [values[index] for index in keep] for column, values in batch.items()}, mask


def vector_project(batch, columns, stats):
    stats["calls"] += 1
    return {column: batch[column] for column in columns}


def vector_aggregate(batches, key_column, value_column, aggregator, stats):
    """Per-batch dictionary encoding, per-batch partials, then a merge into global groups."""
    groups, batch_slots = {}, 0
    for batch in batches:
        stats["calls"] += 1
        slots, keys, partials = {}, [], []
        for index, key in enumerate(batch[key_column]):
            slot = slots.get(key)
            if slot is None:
                slot = slots[key] = len(keys)
                keys.append(key)
                partials.append(aggregator.identity())
            partials[slot] = aggregator.step(partials[slot], batch[value_column][index])
        batch_slots += len(keys)
        for key, partial in zip(keys, partials):
            groups[key] = aggregator.merge(groups.get(key), partial)
    stats["slots"] += batch_slots
    return groups


def row_filter(rows, predicate, stats):
    kept = []
    for row in rows:
        stats["calls"] += 1
        stats["comparisons"] += 1
        if predicate(row[2]):
            kept.append(row)
    return kept


def row_aggregate(rows, key_column, value_column, aggregator, stats):
    groups = {}
    for row in rows:
        stats["calls"] += 1
        key = row[key_column]
        groups[key] = aggregator.step(groups.get(key, aggregator.identity()), row[value_column])
    return groups


def lcg(state):
    return (state * 1103515245 + 12345) % (2 ** 31)


def make_table(row_count, groups, seed, null_every=0):
    table, state = [], seed
    for index in range(row_count):
        state = lcg(state)
        value = None if (null_every and state % null_every == 0) else state % 97
        table.append((index, state % groups, value))
    return table


def vector_pipeline(table, predicate, key_column, value_column, aggregator, batch_size):
    stats = {"calls": 0, "rows_read": 0, "comparisons": 0, "slots": 0}
    projected_batches = []
    for batch in scan(table, batch_size, stats):
        kept, mask = vector_filter(batch, predicate, stats)
        check(len(kept["id"]) == sum(mask), "the filtered batch is dense (no holes)")
        check(len(kept["id"]) <= len(batch["id"]), "a filter never adds rows")
        projected_batches.append(vector_project(kept, (key_column, value_column), stats))
    return vector_aggregate(projected_batches, key_column, value_column, aggregator, stats), stats


def flatten(batches):
    return [(batches[b]["id"][i], batches[b]["group"][i], batches[b]["value"][i])
            for b in range(len(batches)) for i in range(len(batches[b]["id"]))]


def main():
    table = make_table(200, 6, 24680, null_every=11)
    predicate = lambda value: value is not None and value % 3 == 0     # noqa: E731
    batches_expected = (len(table) + BATCH - 1) // BATCH

    for name in AGGREGATES:
        aggregator = Aggregator(name)
        vector_result, vector_stats = vector_pipeline(table, predicate, "group", "value", aggregator, BATCH)
        filter_stats = {"calls": 0, "comparisons": 0}
        matches = row_filter(table, predicate, filter_stats)
        aggregate_stats = {"calls": 0}
        row_result = row_aggregate(matches, 1, 2, aggregator, aggregate_stats)
        check(vector_result == row_result,
              "%s: vectorized and row-at-a-time results differ: %r" % (name, vector_result))
        check(len(vector_result) == 6, "every group has a result for %s" % name)
        check(vector_stats["comparisons"] == filter_stats["comparisons"],
              "%s: the same number of predicate evaluations (no SIMD), %d vs %d"
              % (name, vector_stats["comparisons"], filter_stats["comparisons"]))
        row_calls = filter_stats["calls"] + aggregate_stats["calls"]
        check(vector_stats["calls"] * 2 < row_calls,
              "%s: %d vector calls against %d tuple calls"
              % (name, vector_stats["calls"], row_calls))
        check(len(vector_result) < vector_stats["slots"] <= vector_stats["rows_read"],
              "%s: per-batch dictionaries rebuild slots: %d slots for %d groups"
              % (name, vector_stats["slots"], len(vector_result)))
        check(vector_stats["calls"] == 4 * batches_expected,
              "%s: four operator calls per batch: %d" % (name, vector_stats["calls"]))

    # the filter alone matches the row engine row for row ---------------------------------
    stats = {"calls": 0, "rows_read": 0, "comparisons": 0, "slots": 0}
    batches = []
    for batch in scan(table, BATCH, stats):
        kept, _ = vector_filter(batch, predicate, stats)
        batches.append(kept)
    row_stats = {"calls": 0, "comparisons": 0}
    check(flatten(batches) == row_filter(table, predicate, row_stats),
          "the batch filter selects exactly the rows the row engine selects")
    check(stats["calls"] == 2 * batches_expected,
          "scan plus filter cost one call per batch each: %d" % stats["calls"])
    check(stats["rows_read"] == len(table), "every row was read exactly once")

    # boundary cases ----------------------------------------------------------------------
    for count in (0, 1, BATCH, BATCH + 1, 3 * BATCH):
        small = make_table(count, 3, 1000 + count)
        for name in ("sum", "count_star", "min"):
            aggregator = Aggregator(name)
            vector_result, _ = vector_pipeline(small, predicate, "group", "value", aggregator, BATCH)
            expected = row_aggregate([row for row in small if predicate(row[2])], 1, 2,
                                     aggregator, {"calls": 0})
            check(vector_result == expected,
                  "%s: mismatch for %d rows: %r vs %r" % (name, count, vector_result, expected))
    empty_vector, empty_stats = vector_pipeline([], predicate, "group", "value", Aggregator("sum"), BATCH)
    check(empty_vector == {} and empty_stats["rows_read"] == 0 and empty_stats["calls"] == 1,
          "the vector engine handles the empty input with a single scan call")
    check(row_aggregate([], 1, 2, Aggregator("sum"), {"calls": 0}) == {},
          "an empty input aggregates to nothing")

    # a group straddling a batch boundary still merges correctly --------------------------
    straddle = [(i, 0, 1) for i in range(BATCH + 3)]
    straddle_result, straddle_stats = vector_pipeline(straddle, lambda v: True, "group", "value",
                                                      Aggregator("sum"), BATCH)
    check(straddle_result == {0: BATCH + 3},
          "a group split across batches merges to the full sum: %r" % (straddle_result,))
    check(straddle_stats["slots"] == 2, "one slot per batch for the single group")
    check(sum(1 for _ in scan(straddle, BATCH, {"calls": 0, "rows_read": 0})) == 2,
          "eleven rows span exactly two batches")

    # a bigger batch does strictly less operator work -------------------------------------
    calls_by_batch = {}
    for batch_size in (1, 2, 4, 8, 16, 32):
        _, batch_stats = vector_pipeline(table, predicate, "group", "value", Aggregator("sum"), batch_size)
        calls_by_batch[batch_size] = batch_stats["calls"]
    check(all(calls_by_batch[a] > calls_by_batch[b]
              for a, b in zip((1, 2, 4, 8, 16), (2, 4, 8, 16, 32))),
          "call counts fall as the batch grows: %r" % (calls_by_batch,))

    first, first_stats = vector_pipeline(table, predicate, "group", "value", Aggregator("sum"), BATCH)
    again, again_stats = vector_pipeline(table, predicate, "group", "value", Aggregator("sum"), BATCH)
    check(first == again and first_stats == again_stats, "the vector engine is deterministic")
    print("[648] vectorized: batch %d, %d calls for %d rows (%d per batch), %d dictionary slots"
          % (BATCH, first_stats["calls"], len(table), batches_expected, first_stats["slots"]))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
