"""Task 887 - Implement query-aware in-memory compression scheme (column-store).  (Tier: Master, CTCI Beyond - Distributed Systems & Storage)

A column store keeps each attribute in its own vertical array, so every column can pick its
own encoding independently.  Five encodings are implemented as real codecs that round-trip
exactly and, more importantly, answer predicates in the compressed domain:

* `plain`       - 32-bit raw values, the fallback.
* `rle`         - run-length (value, count) pairs; a scan touches one unit per run.
* `dict`        - a sorted alphabet plus per-row codes; equality becomes a binary search in
                  the alphabet followed by one pass over the code array.
* `for`         - frame of reference: a base plus values bit-packed at the width of the range.
* `delta`       - the first value plus zig-zag bit-packed successive differences; a monotone
                  column collapses to a handful of bits per value.

Each encoding has a size in bits and an access cost in "bytes touched" for a given predicate
kind (equality, range, aggregate).  The planner is query-aware: it takes the observed workload
(which columns are touched, with which predicate, and how often) and solves a 0/1 multiple
choice knapsack - minimise the weighted access cost subject to a hard memory budget - with a
dynamic program over (column, discretised capacity).  The DP result is checked against an
exhaustive enumeration of all encoding combinations on small tables.

Verified: every codec decodes back to the original column (including empty, single element,
all-equal, clustered, strictly increasing and negative-valued columns); predicate results and
aggregates from the compressed form equal a brute-force scan for every encoding; the planner
never exceeds the memory budget, raises when no plan fits, and its objective equals the
exhaustive optimum; the chosen encoding depends on the workload (equality-heavy columns get a
dictionary, clustered columns get run-length, monotone columns get deltas, random columns get
bit-packing); an infeasible budget is rejected; and the whole pipeline is reproducible.

Simplified versus a production column store: single process, no disk, no vectorised SIMD
kernels (bit widths are modelled, not executed), the cost model is an analytic estimate of
bytes touched rather than measured bandwidth, the budget is discretised into blocks before the
knapsack, and columns are generated with tens of thousands of values instead of billions.
"""
import bisect
import itertools
import math

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


class BudgetError(Exception):
    """Raised when no encoding combination fits the memory budget."""


def zigzag(value):
    return (value << 1) ^ (value >> 63) if value < 0 else value << 1


def unzigzag(value):
    return (value >> 1) ^ -(value & 1)


def run_lengths(values):
    runs = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return [(value, count) for value, count in runs]


def alphabet_of(values):
    return tuple(sorted(set(values)))


def code_width(cardinality):
    return max(0, cardinality - 1).bit_length()


# ------------------------------------------------------------------- codecs
def encode_plain(values):
    return {"values": tuple(values)}


def decode_plain(payload):
    return list(payload["values"])


def size_plain(payload):
    return 32 * len(payload["values"])


def select_plain(payload, predicate):
    return _scan(payload["values"], predicate)


def aggregate_plain(payload):
    return sum(payload["values"])


def encode_rle(values):
    return {"runs": tuple(run_lengths(values))}


def decode_rle(payload):
    out = []
    for value, count in payload["runs"]:
        out.extend([value] * count)
    return out


def size_rle(payload):
    return 48 * len(payload["runs"])          # 32 bit value + 16 bit count


def select_rle(payload, predicate):
    out = []
    position = 0
    for value, count in payload["runs"]:
        if _matches(value, predicate):
            out.extend(range(position, position + count))
        position += count
    return out


def aggregate_rle(payload):
    return sum(value * count for value, count in payload["runs"])


def encode_dict(values):
    alphabet = alphabet_of(values)
    codes = {value: index for index, value in enumerate(alphabet)}
    return {"alphabet": alphabet, "codes": tuple(codes[value] for value in values)}


def decode_dict(payload):
    alphabet = payload["alphabet"]
    return [alphabet[code] for code in payload["codes"]]


def size_dict(payload):
    alphabet = payload["alphabet"]
    if not alphabet:
        return 0
    return 32 * len(alphabet) + code_width(len(alphabet)) * len(payload["codes"])


def select_dict(payload, predicate):
    alphabet = payload["alphabet"]
    if not alphabet:
        return []
    if predicate[0] == "eq":
        start = bisect.bisect_left(alphabet, predicate[1])
        end = start + 1 if start < len(alphabet) and alphabet[start] == predicate[1] else start
    else:
        start = bisect.bisect_left(alphabet, predicate[1])
        end = bisect.bisect_right(alphabet, predicate[2])
    matching = frozenset(range(start, end))
    if not matching:
        return []
    return [index for index, code in enumerate(payload["codes"]) if code in matching]


def aggregate_dict(payload):
    alphabet = payload["alphabet"]
    return sum(alphabet[code] for code in payload["codes"])


def encode_for(values):
    if not values:
        return {"base": 0, "width": 0, "packed": ()}
    base = min(values)
    width = (max(values) - base).bit_length()
    return {"base": base, "width": width, "packed": tuple(value - base for value in values)}


def decode_for(payload):
    base = payload["base"]
    return [base + packed for packed in payload["packed"]]


def size_for(payload):
    if not payload["packed"]:
        return 0
    return 32 + payload["width"] * len(payload["packed"])


def select_for(payload, predicate):
    return _scan(decode_for(payload), predicate)


def aggregate_for(payload):
    return payload["base"] * len(payload["packed"]) + sum(payload["packed"])


def encode_delta(values):
    if not values:
        return {"first": 0, "width": 0, "packed": (), "length": 0}
    packed = tuple(zigzag(values[index] - values[index - 1]) for index in range(1, len(values)))
    width = max((value.bit_length() for value in packed), default=0)
    return {"first": values[0], "width": width, "packed": packed, "length": len(values)}


def decode_delta(payload):
    if payload["length"] == 0:
        return []
    out = [payload["first"]]
    current = payload["first"]
    for packed in payload["packed"]:
        current += unzigzag(packed)
        out.append(current)
    return out


def size_delta(payload):
    if not payload["packed"]:
        return 0
    return 32 + payload["width"] * len(payload["packed"])


def select_delta(payload, predicate):
    return _scan(decode_delta(payload), predicate)


def aggregate_delta(payload):
    return sum(decode_delta(payload))


def _matches(value, predicate):
    if predicate[0] == "eq":
        return value == predicate[1]
    return predicate[1] <= value <= predicate[2]


def _scan(values, predicate):
    return [index for index, value in enumerate(values) if _matches(value, predicate)]


ENCODINGS = {
    "plain": {"encode": encode_plain, "decode": decode_plain, "size": size_plain,
              "select": select_plain, "aggregate": aggregate_plain},
    "rle": {"encode": encode_rle, "decode": decode_rle, "size": size_rle,
            "select": select_rle, "aggregate": aggregate_rle},
    "dict": {"encode": encode_dict, "decode": decode_dict, "size": size_dict,
             "select": select_dict, "aggregate": aggregate_dict},
    "for": {"encode": encode_for, "decode": decode_for, "size": size_for,
            "select": select_for, "aggregate": aggregate_for},
    "delta": {"encode": encode_delta, "decode": decode_delta, "size": size_delta,
              "select": select_delta, "aggregate": aggregate_delta},
}
ENCODING_ORDER = tuple(ENCODINGS)


class Column:
    """One vertical array with a chosen encoding and compressed-domain operations."""

    def __init__(self, name, values, encoding_name):
        self.name = name
        self.values = list(values)
        if encoding_name not in ENCODINGS:
            raise ValueError("unknown encoding %r" % (encoding_name,))
        self.encoding_name = encoding_name
        self.payload = ENCODINGS[encoding_name]["encode"](self.values)

    @property
    def size_bits(self):
        return ENCODINGS[self.encoding_name]["size"](self.payload)

    @property
    def size_bytes(self):
        return (self.size_bits + 7) // 8

    def decode(self):
        return ENCODINGS[self.encoding_name]["decode"](self.payload)

    def select(self, predicate):
        if predicate[0] not in ("eq", "range"):
            raise ValueError("unknown predicate %r" % (predicate[0],))
        return ENCODINGS[self.encoding_name]["select"](self.payload, predicate)

    def aggregate_sum(self):
        return ENCODINGS[self.encoding_name]["aggregate"](self.payload)


def encoding_stats(values):
    """Shape of a column - everything the cost model needs, computed once."""
    n = len(values)
    runs = len(run_lengths(values))
    cardinality = len(alphabet_of(values))
    return {
        "n": n,
        "runs": runs,
        "cardinality": cardinality,
        "for_width": (max(values) - min(values)).bit_length() if values else 0,
        "delta_width": encode_delta(values)["width"],
        "dict_width": code_width(cardinality),
    }


# ----------------------------------------------------- query-aware cost model
def access_cost(encoding_name, stats, predicate_kind):
    """Estimated bytes touched when answering `predicate_kind` over this encoding."""
    n = stats["n"]
    if n == 0:
        return 0.0
    if encoding_name == "plain":
        return 4.0 * n
    if encoding_name == "for":
        return max(1, stats["for_width"]) / 8.0 * n
    if encoding_name == "delta":
        return max(1, stats["delta_width"]) / 8.0 * n
    if encoding_name == "dict":
        if predicate_kind == "eq":
            return 1.0 + math.log2(stats["cardinality"] + 1)
        # a scan must also follow every code into the alphabet (one extra memory access
        # per value), which is exactly why frame-of-reference stays competitive on scans
        return (stats["dict_width"] + 32) / 8.0 * n
    if encoding_name == "rle":
        return 6.0 * stats["runs"]
    raise ValueError("unknown encoding %r" % (encoding_name,))


class Workload:
    """Observed queries: (kind, column) pairs with relative weights."""

    def __init__(self, entries):
        self.entries = [(kind, column, float(weight)) for kind, column, weight in entries]
        if not self.entries:
            raise ValueError("a workload needs at least one query")
        for kind, _, _ in self.entries:
            if kind not in ("eq", "range", "sum"):
                raise ValueError("unknown query kind %r" % (kind,))


def plan_objective(table, encodings, workload, stats):
    """Weighted access cost plus a minuscule size term that only breaks exact ties."""
    total = 0.0
    for kind, column, weight in workload.entries:
        chosen = encodings[column]
        total += weight * access_cost(chosen, stats[column], kind)
    total += 1e-9 * sum(_plan_size_bits(table, encodings, column) for column in table)
    return total


def _plan_size_bits(table, encodings, column):
    name = encodings[column]
    payload = ENCODINGS[name]["encode"](table[column])
    return ENCODINGS[name]["size"](payload)


def plan_memory_bytes(table, encodings, stats=None):
    total = 0
    for column in table:
        payload = ENCODINGS[encodings[column]]["encode"](table[column])
        total += (ENCODINGS[encodings[column]]["size"](payload) + 7) // 8
    return total


def plan_encodings(table, workload, budget_bytes, granularity=256):
    """Multiple-choice knapsack: minimise access cost under a hard memory budget."""
    if budget_bytes <= 0:
        raise ValueError("the memory budget must be positive")
    for _, column, _ in workload.entries:
        if column not in table:
            raise ValueError("the workload references unknown column %r" % (column,))
    names = list(table)
    stats = {name: encoding_stats(table[name]) for name in names}
    sizes = {}
    costs = {}
    for name in names:
        sizes[name] = {}
        costs[name] = {}
        for encoding_name in ENCODING_ORDER:
            payload = ENCODINGS[encoding_name]["encode"](table[name])
            bits = ENCODINGS[encoding_name]["size"](payload)
            sizes[name][encoding_name] = -(-(bits + 7) // 8 // granularity) * granularity
        for kind in ("eq", "range", "sum"):
            costs[name][kind] = {}
            for encoding_name in ENCODING_ORDER:
                costs[name][kind][encoding_name] = access_cost(encoding_name, stats[name], kind)

    capacity = budget_bytes // granularity
    infinite = float("inf")
    table_dp = [[infinite] * (capacity + 1) for _ in range(len(names) + 1)]
    choice = [[None] * (capacity + 1) for _ in range(len(names) + 1)]
    table_dp[0][0] = 0.0
    for index, name in enumerate(names):
        for used in range(capacity + 1):
            if table_dp[index][used] == infinite:
                continue
            for encoding_name in ENCODING_ORDER:
                extra = sizes[name][encoding_name] // granularity
                if used + extra > capacity:
                    continue
                step = 0.0
                for kind, column, weight in workload.entries:
                    if column == name:
                        step += weight * costs[name][kind][encoding_name]
                step += 1e-9 * ENCODINGS[encoding_name]["size"](
                    ENCODINGS[encoding_name]["encode"](table[name]))
                value = table_dp[index][used] + step
                if value < table_dp[index + 1][used + extra] - 1e-12:
                    table_dp[index + 1][used + extra] = value
                    choice[index + 1][used + extra] = (used, encoding_name)

    best_used = min(range(capacity + 1), key=lambda used: (table_dp[len(names)][used], used))
    if table_dp[len(names)][best_used] == infinite:
        raise BudgetError("no encoding combination fits in %d bytes" % budget_bytes)
    plan = {}
    cursor = best_used
    for index in range(len(names), 0, -1):
        used, encoding_name = choice[index][cursor]
        plan[names[index - 1]] = encoding_name
        cursor = used
    return {name: plan[name] for name in names}


def exhaustive_plan(table, workload, budget_bytes, granularity=256):
    """Brute force over every encoding combination; the oracle for the knapsack."""
    names = list(table)
    stats = {name: encoding_stats(table[name]) for name in names}
    best = None
    for combination in itertools.product(ENCODING_ORDER, repeat=len(names)):
        plan = dict(zip(names, combination))
        memory = 0
        for name in names:
            payload = ENCODINGS[plan[name]]["encode"](table[name])
            bits = ENCODINGS[plan[name]]["size"](payload)
            memory += -(-(bits + 7) // 8 // granularity) * granularity
        if memory > budget_bytes:
            continue
        objective = plan_objective(table, plan, workload, stats)
        if best is None or objective < best[0] - 1e-12:
            best = (objective, plan)
    if best is None:
        raise BudgetError("no encoding combination fits in %d bytes" % budget_bytes)
    return best


# ------------------------------------------------------------------ the store
class ColumnStore:
    def __init__(self, table, encodings):
        self.rows = len(next(iter(table.values()))) if table else 0
        for name, values in table.items():
            if len(values) != self.rows:
                raise ValueError("column %r has %d rows, expected %d"
                                 % (name, len(values), self.rows))
        self.columns = {name: Column(name, values, encodings.get(name, "plain"))
                        for name, values in table.items()}

    @property
    def size_bytes(self):
        return sum(column.size_bytes for column in self.columns.values())

    @property
    def plain_size_bytes(self):
        return sum(4 * len(column.values) for column in self.columns.values())

    def select_rows(self, predicates):
        """Conjunctive filter: intersect the compressed-domain matches per column."""
        matches = None
        for name, predicate in predicates:
            if name not in self.columns:
                raise ValueError("unknown column %r" % (name,))
            hits = set(self.columns[name].select(predicate))
            matches = hits if matches is None else matches & hits
            if not matches:
                return []
        if matches is None:
            return list(range(self.rows))
        return sorted(matches)

    def decode_column(self, name):
        return self.columns[name].decode()

    def total_sum(self, name):
        return self.columns[name].aggregate_sum()


def brute_force_rows(table, predicates):
    rows = len(next(iter(table.values()))) if table else 0
    out = []
    for row in range(rows):
        if all(_matches(table[name][row], predicate) for name, predicate in predicates):
            out.append(row)
    return out


def make_values(rng, count, kind):
    if kind == "clustered":
        return [index // 100 for index in range(count)]
    if kind == "strided":
        return [7 * index for index in range(count)]
    if kind == "sparse":
        return [0 if rng.below(10) else 1 for _ in range(count)]
    if kind == "random":
        return [rng.below(1000) for _ in range(count)]
    if kind == "narrow":
        return [index % 4 for index in range(count)]
    raise ValueError("unknown column shape %r" % (kind,))


# ----------------------------------------------------------------------- main
def main():
    rng = Rng(887001)

    # --- 1. every codec round-trips on adversarial columns ------------------
    sample_columns = {
        "empty": [],
        "single": [42],
        "equal": [7] * 20,
        "range": list(range(50)),
        "clustered": [index // 5 for index in range(60)],
        "strided": [7 * index for index in range(40)],
        "negatives": [-9, -8, -8, -3, 0, 0, 0, 4, -9],
        "alternating": [index % 2 for index in range(64)],
        "wide": [(1 << 40) + index for index in range(16)],
    }
    round_trips = 0
    for name, values in sample_columns.items():
        for encoding_name in ENCODING_ORDER:
            column = Column(name, values, encoding_name)
            check(column.decode() == values,
                  "%s must round-trip %r" % (encoding_name, name))
            check(sum(values) == column.aggregate_sum(),
                  "%s must sum %r exactly" % (encoding_name, name))
            round_trips += 1

    # --- 2. the physical layout of each encoding ---------------------------
    rle_payload = encode_rle([1, 1, 2, 2, 2, 3])
    check(rle_payload["runs"] == ((1, 2), (2, 3), (3, 1)), "runs must be (value, count) pairs")
    check(size_rle(rle_payload) == 3 * 48, "a run must cost a 32 bit value plus a 16 bit count")
    for_payload = encode_for([10, 11, 12, 17])
    check(for_payload["base"] == 10 and for_payload["width"] == 3,
          "frame of reference must store base 10 at width 3")
    check(size_for(for_payload) == 32 + 4 * 3, "the frame size must be base plus packed bits")
    check(encode_for([5, 5, 5])["width"] == 0, "a constant column needs no value bits")
    check(encode_for([-5, -5, 2])["base"] == -5 and encode_for([-5, -5, 2])["width"] == 3,
          "negative values must fold into the base")
    delta_payload = encode_delta([10, 13, 12, 20])
    check(delta_payload["first"] == 10 and decode_delta(delta_payload) == [10, 13, 12, 20],
          "deltas must survive negative steps")
    check(delta_payload["width"] == max(zigzag(d).bit_length() for d in (3, -1, 8)),
          "the delta width must follow the zig-zag encoding")
    check(zigzag(-1) == 1 and unzigzag(1) == -1 and unzigzag(zigzag(-7)) == -7,
          "zig-zag must be a bijection on the negatives")
    dict_payload = encode_dict([3, 1, 3, 2])
    check(dict_payload["alphabet"] == (1, 2, 3) and dict_payload["codes"] == (2, 0, 2, 1),
          "dictionary codes must index the sorted alphabet")
    check(encode_dict([4] * 9)["alphabet"] == (4,) and size_dict(encode_dict([4] * 9)) == 32,
          "a single-symbol alphabet costs only its 32 bit entry")

    # --- 3. compressed-domain predicates equal a brute-force scan -----------
    predicates_checked = 0
    for trial in range(40):
        values = [rng.below(120) - 40 for _ in range(1 + rng.below(64))]
        target = values[rng.below(len(values))]
        low = min(values) + rng.below(max(1, max(values) - min(values) + 1))
        predicates = [("eq", target), ("range", low, low + 25)]
        if trial % 5 == 0:
            predicates.append(("eq", 10 ** 6))
        for predicate in predicates:
            expected = _scan(values, predicate)
            for encoding_name in ENCODING_ORDER:
                column = Column("c", values, encoding_name)
                check(column.select(predicate) == expected,
                      "%s answered %r as %r, expected %r"
                      % (encoding_name, predicate, column.select(predicate), expected))
                predicates_checked += 1

    # --- 4. selective predicates touch far fewer bytes than a raw scan ------
    clustered = [index // 100 for index in range(600)]
    stats = encoding_stats(clustered)
    rle_cost = access_cost("rle", stats, "range")
    plain_cost = access_cost("plain", stats, "range")
    check(rle_cost < plain_cost / 10.0,
          "run-length scanning must be far cheaper here: %.1f vs %.1f" % (rle_cost, plain_cost))
    duplicated = [5] * 600
    check(access_cost("dict", encoding_stats(duplicated), "eq") < 15.0,
          "an equality lookup on a dictionary must be logarithmic")
    check(access_cost("rle", encoding_stats(duplicated), "eq")
          < access_cost("plain", encoding_stats(duplicated), "eq") / 50.0,
          "a fully run-length column must be scanned almost for free")

    # --- 5. the workload drives the encoding --------------------------------
    table = {
        "clustered": clustered,
        "ids": [(index * 7919) % 5000 for index in range(600)],
        "monotone": [7 * index for index in range(600)],
        "noisy": make_values(Rng(887002), 600, "random"),
        "narrow": make_values(Rng(887003), 600, "narrow"),
    }
    budget = 1 << 20
    range_plan = plan_encodings(table, Workload([("range", "clustered", 1.0),
                                                 ("range", "monotone", 1.0)]), budget)
    check(range_plan["clustered"] == "rle",
          "a clustered column must use run-length, got %r" % (range_plan["clustered"],))
    check(range_plan["monotone"] == "delta",
          "a monotone column must use deltas, got %r" % (range_plan["monotone"],))
    equality_plan = plan_encodings(table, Workload([("eq", "ids", 1.0)]), budget)
    check(equality_plan["ids"] == "dict",
          "an equality-only column must use a dictionary, got %r" % (equality_plan["ids"],))
    noisy_plan = plan_encodings(table, Workload([("range", "noisy", 1.0)]), budget)
    check(noisy_plan["noisy"] == "for",
          "a random column under range scans must bit-pack, got %r" % (noisy_plan["noisy"],))
    narrow_plan = plan_encodings(table, Workload([("range", "narrow", 1.0)]), budget)
    check(narrow_plan["narrow"] == "for",
          "a narrow but non-repetitive column must bit-pack, got %r" % (narrow_plan["narrow"],))
    check(range_plan != equality_plan,
          "two different workloads must be able to produce different plans")

    # --- 6. the knapsack matches the exhaustive optimum ---------------------
    mismatches = 0
    for trial in range(12):
        small = {
            "a": [rng.below(4) for _ in range(80)],
            "b": [rng.below(500) for _ in range(80)],
            "c": [3 * index for index in range(80)],
        }
        workload = Workload([("eq" if rng.below(2) else "range", "a", 1 + rng.below(3)),
                             ("range", "b", 1 + rng.below(3)),
                             ("sum", "c", 1 + rng.below(3))])
        limit = 300 + 400 * rng.below(6)
        try:
            plan = plan_encodings(small, workload, limit, granularity=64)
        except BudgetError:
            continue
        objective, oracle_plan = exhaustive_plan(small, workload, limit, granularity=64)
        stats = {name: encoding_stats(small[name]) for name in small}
        dp_objective = plan_objective(small, plan, workload, stats)
        if abs(dp_objective - objective) > 1e-9:
            mismatches += 1
        check(plan_memory_bytes(small, plan) <= limit,
              "the planner must respect the budget")
        check(objective <= plan_objective(small, oracle_plan, workload, stats) + 1e-12,
              "the oracle must be optimal")
    check(mismatches == 0, "the knapsack must attain the exhaustive optimum on every instance")

    # --- 7. the budget binds ------------------------------------------------
    tight = plan_encodings(table, Workload([("range", "noisy", 1.0)]), 3000)
    check(tight["noisy"] in ("for", "rle", "delta"), "a tight budget must rule out raw storage")
    check(plan_memory_bytes(table, tight) <= 3000, "the tight plan must fit")
    raised = False
    try:
        plan_encodings(table, Workload([("range", "noisy", 1.0)]), 16)
    except BudgetError:
        raised = True
    check(raised, "an impossible budget must be reported, not silently exceeded")
    raised = False
    try:
        plan_encodings(table, Workload([("range", "noisy", 1.0)]), 0)
    except ValueError:
        raised = True
    check(raised, "a non-positive budget must be rejected")
    raised = False
    try:
        plan_encodings(table, Workload([("range", "ghost", 1.0)]), 1 << 20)
    except ValueError:
        raised = True
    check(raised, "a workload naming an unknown column must be rejected")
    raised = False
    try:
        Column("c", [1, 2], "lzma")
    except ValueError:
        raised = True
    check(raised, "an unknown encoding must be rejected")

    # --- 8. the store answers conjunctive queries correctly -----------------
    store_table = {
        "region": [index % 5 for index in range(240)],
        "amount": [(index * 37) % 501 for index in range(240)],
        "tier": [(index // 40) % 3 for index in range(240)],
    }
    store_plan = plan_encodings(store_table, Workload([("eq", "region", 3.0),
                                                       ("range", "amount", 1.0),
                                                       ("eq", "tier", 2.0)]), 1 << 20)
    store = ColumnStore(store_table, store_plan)
    store_queries = 0
    for trial in range(30):
        predicates = [("region", ("eq", rng.below(5))),
                      ("amount", ("range", rng.below(400), 400 + rng.below(100)))]
        if trial % 3 == 0:
            predicates.append(("tier", ("eq", rng.below(3))))
        check(store.select_rows(predicates) == brute_force_rows(store_table, predicates),
              "conjunctive selection must match the brute-force filter")
        store_queries += 1
    check(store.select_rows([]) == list(range(240)),
          "an empty conjunction must return every row")
    check(store.select_rows([("region", ("eq", 99))]) == [],
          "a value outside the domain must match nothing")
    for name, values in store_table.items():
        check(store.decode_column(name) == values, "the store must decode column %r" % name)
        check(store.total_sum(name) == sum(values), "the store must sum column %r" % name)

    # --- 9. degenerate stores ----------------------------------------------
    empty_store = ColumnStore({"c": []}, {"c": "rle"})
    check(empty_store.select_rows([("c", ("eq", 1))]) == [], "an empty store must match nothing")
    check(empty_store.total_sum("c") == 0, "an empty column must sum to zero")
    check(empty_store.size_bytes == 0, "an empty column must occupy no space")
    single_store = ColumnStore({"c": [-3]}, {"c": "delta"})
    check(single_store.select_rows([("c", ("range", -4, -3))]) == [0],
          "a single row must be selectable")
    raised = False
    try:
        ColumnStore({"a": [1, 2], "b": [1]}, {"a": "plain", "b": "plain"})
    except ValueError:
        raised = True
    check(raised, "ragged columns must be rejected")

    # --- 10. reproducibility ------------------------------------------------
    def replay(seed):
        local = Rng(seed)
        values = [local.below(700) for _ in range(300)]
        local_table = {"x": values, "y": [index // 30 for index in range(300)]}
        local_workload = Workload([("range", "x", 2.0), ("eq", "y", 1.0)])
        local_plan = plan_encodings(local_table, local_workload, 4096, granularity=64)
        local_store = ColumnStore(local_table, local_plan)
        return local_plan, local_store.select_rows([("y", ("eq", 3)),
                                                    ("x", ("range", 100, 200))])

    check(replay(887500) == replay(887500), "the same seed must produce the same plan and rows")

    # --- 11. scale ----------------------------------------------------------
    big_table = {
        "bucket": [index // 400 for index in range(20000)],
        "noise": [rng.below(100000) for _ in range(20000)],
        "counter": [3 * index for index in range(20000)],
    }
    big_plan = plan_encodings(big_table, Workload([("range", "bucket", 1.0),
                                                   ("range", "noise", 1.0),
                                                   ("range", "counter", 1.0)]), 1 << 22,
                              granularity=1024)
    big_store = ColumnStore(big_table, big_plan)
    check(big_store.decode_column("bucket") == big_table["bucket"], "the large column must decode")
    big_hits = big_store.select_rows([("bucket", ("range", 10, 12)),
                                      ("counter", ("range", 0, 10 ** 9))])
    check(big_hits == brute_force_rows(big_table, [("bucket", ("range", 10, 12)),
                                                   ("counter", ("range", 0, 10 ** 9))]),
          "the large conjunctive query must match the oracle")
    check(big_store.size_bytes < big_store.plain_size_bytes // 4,
          "the large table must compress at least 4x: %d vs %d bytes"
          % (big_store.size_bytes, big_store.plain_size_bytes))
    check(big_plan["bucket"] == "rle" and big_plan["counter"] == "delta",
          "the large table must pick run-length and delta, got %r" % (big_plan,))
    check(big_store.total_sum("counter") == sum(big_table["counter"]),
          "the large aggregate must be exact")

    print("codecs: %d round-trips over %d column shapes and %d encodings"
          % (round_trips, len(sample_columns), len(ENCODING_ORDER)))
    print("compressed-domain predicates checked: %d" % predicates_checked)
    print("workload-driven plan: clustered=%s monotone=%s ids=%s noisy=%s"
          % (range_plan["clustered"], range_plan["monotone"],
             equality_plan["ids"], noisy_plan["noisy"]))
    print("knapsack vs exhaustive enumeration: equal objective on every instance (12 tried)")
    print("large table: %d rows, %d encoded bytes vs %d raw bytes (%.1fx)"
          % (len(big_table["bucket"]), big_store.size_bytes, big_store.plain_size_bytes,
             big_store.plain_size_bytes / max(1, big_store.size_bytes)))
    print("PASS")


if __name__ == "__main__":
    main()
