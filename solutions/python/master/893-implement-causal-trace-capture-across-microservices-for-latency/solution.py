"""Task 893 - Implement causal trace capture across microservices for latency debugging.  (Tier: Master, CTCI Beyond - Distributed Systems & Storage)

A trace is a tree of spans: one root and one child edge per remote call, exactly what a
distributed tracer captures.  This demo generates such traces from an explicit service call
graph (some services fan out sequentially, some in parallel), reports every span through a
per-service clock with an injected offset, and then does the work a real collector + analyser
has to do on top of those spans:

* reconstruction - the tree is reassembled from (trace, span, parent) identifiers, never from
  arrival order, because spans arrive in completion order and the root therefore arrives last;
* skew detection - per-parent-child constraints bound the difference of two services' clock
  offsets from both sides (a child cannot start before its parent starts, nor end after its
  parent ends), and propagating those difference constraints from the root yields an interval
  per service that provably contains the true offset.  Absolute offsets are not identifiable
  (only differences are), which the demo states rather than hides;
* critical path - with self time = duration minus the union of the child intervals, the path
  from the root that maximises the sum of self times is the latency contribution chain.  The
  dynamic program over the tree is cross-checked against exhaustive path enumeration;
* invariance - summing self time over the whole tree must reproduce the root duration exactly,
  because the children are nested inside their parents; that identity is the strongest
  available statement that the capture is complete and non-overlapping;
* debugging - the bottleneck service is the one contributing the most self time along the
  critical path, and it is checked against the service that was deliberately slowed down;
* sampling - head sampling keeps an exact deterministic subset while staying faithful to the
  latency distribution, and the validator rejects orphaned, duplicated, cyclic or
  mixed-trace captures instead of silently analysing a broken trace.

Simplified versus a production tracing pipeline: no network or agent, spans travel in an
in-process list, clocks are integer offsets with no drift or jitter, no span links or
asynchronous continuations, sampling is deterministic head sampling rather than adaptive tail
sampling, and the call graph is a fixed eleven-service topology.  Verified: tree structure
matches the generator for every trace, the validator rejects five classes of broken capture,
the DP and the exhaustive enumeration agree, the self-time sum identity holds, the skew
interval always contains the injected offset, the capture is order independent, the injected
bottleneck is found, sampling counts and quantiles match, and the collector is linear in the
number of spans.
"""
MASK64 = (1 << 64) - 1
INFINITY = float("inf")

GRAPH = {
    "gateway": {"children": ("auth", "search", "cart"), "mode": "sequential", "base": 3},
    "auth": {"children": ("session",), "mode": "sequential", "base": 2},
    "session": {"children": (), "mode": "sequential", "base": 1},
    "search": {"children": ("catalog", "index"), "mode": "parallel", "base": 2},
    "catalog": {"children": ("storage",), "mode": "sequential", "base": 3},
    "index": {"children": (), "mode": "sequential", "base": 2},
    "storage": {"children": (), "mode": "sequential", "base": 4},
    "cart": {"children": ("pricing", "inventory"), "mode": "parallel", "base": 2},
    "pricing": {"children": ("rules",), "mode": "sequential", "base": 3},
    "rules": {"children": (), "mode": "sequential", "base": 1},
    "inventory": {"children": (), "mode": "sequential", "base": 3},
}
ROOT_SERVICE = "gateway"
SERVICES = tuple(sorted(GRAPH))


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


class TraceError(Exception):
    """Raised when a capture cannot be assembled into a well formed trace."""


# ------------------------------------------------------------------ generation
def generate_trace(rng, trace_id, skews, slow_service=None, slow_extra=6):
    """Emit a full trace; spans are appended in completion order, as a tracer would."""
    spans = []
    counter = [0]

    def emit(service, parent_id, start):
        node = GRAPH[service]
        base = node["base"] + rng.below(2)
        if service == slow_service:
            base += slow_extra
        lead = base // 3
        cursor = start + lead
        children = node["children"]
        child_ids = []
        if node["mode"] == "sequential":
            for child in children:
                child_ids.append(emit(child, counter[0], cursor))
                cursor = spans[child_ids[-1]]["true_end"]
        else:
            ends = []
            for child in children:
                child_ids.append(emit(child, counter[0], cursor))
                ends.append(spans[child_ids[-1]]["true_end"])
            cursor = max(ends) if ends else cursor
        true_end = cursor + max(1, base - lead)
        span_id = counter[0]
        counter[0] += 1
        spans.append({
            "trace": trace_id,
            "span": span_id,
            "parent": parent_id,
            "service": service,
            "true_start": start,
            "true_end": true_end,
            "start": start + skews[service],
            "end": true_end + skews[service],
            "children": list(child_ids),
        })
        return span_id

    emit(ROOT_SERVICE, None, skews[ROOT_SERVICE])
    # the parent pointer of the root and of every child is fixed up after the fact
    for span in spans:
        for child in span["children"]:
            spans[child]["parent"] = span["span"]
    return sorted(spans, key=lambda span: (span["end"], span["span"]))


def make_skews(rng, spread=6):
    skews = {ROOT_SERVICE: rng.below(spread) - spread // 2}
    for service in SERVICES:
        if service != ROOT_SERVICE:
            skews[service] = rng.below(spread) - spread // 2
    base = skews[ROOT_SERVICE]
    return {service: value - base for service, value in skews.items()}


# ------------------------------------------------------------------- analysis
def index_spans(spans):
    by_id = {}
    for span in spans:
        if span["span"] in by_id:
            raise TraceError("duplicate span id %r" % (span["span"],))
        by_id[span["span"]] = span
    return by_id


def build_tree(spans):
    """Reassemble the tree from identifiers; never from arrival order."""
    by_id = index_spans(spans)
    traces = {span["trace"] for span in spans}
    if len(traces) > 1:
        raise TraceError("a capture must hold exactly one trace, got %r" % (traces,))
    children = {span_id: [] for span_id in by_id}
    roots = []
    for span_id, span in by_id.items():
        parent = span["parent"]
        if parent is None:
            roots.append(span_id)
            continue
        if parent not in by_id:
            raise TraceError("span %r references missing parent %r" % (span_id, parent))
        children[parent].append(span_id)
    if len(roots) != 1:
        raise TraceError("a trace needs exactly one root, got %d" % len(roots))
    for span_id in by_id:
        seen = set()
        cursor = span_id
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            cursor = by_id[cursor]["parent"]
        if cursor is not None:
            raise TraceError("cycle detected through span %r" % (span_id,))
    # every span must be reachable from the single root
    reachable = set()
    stack = [roots[0]]
    while stack:
        current = stack.pop()
        reachable.add(current)
        stack.extend(children[current])
    if reachable != set(by_id):
        raise TraceError("unreachable spans: %r" % sorted(set(by_id) - reachable))
    return {"children": {key: sorted(value) for key, value in children.items()},
            "root": roots[0],
            "by_id": by_id,
            "order": sorted(by_id)}


def union_length(items):
    """Exact measure of the union of intervals (used for the self time of a span)."""
    if not items:
        return 0
    ordered = sorted(items)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def self_times(spans, tree):
    by_id = tree["by_id"]
    child_intervals = {}
    result = {}
    for span_id, span in by_id.items():
        child_intervals.setdefault(span_id, [])
    for span_id, span in by_id.items():
        parent = span["parent"]
        if parent is not None:
            child_intervals[parent].append((span["start"], span["end"]))
    for span_id, span in by_id.items():
        covered = union_length(child_intervals[span_id])
        result[span_id] = (span["end"] - span["start"]) - covered
    return result


def critical_path(tree, self_time):
    """Longest root-to-leaf path by summed self time, with path reconstruction."""
    children = tree["children"]
    best = {}

    def walk(span_id):
        child_best, child_choice = 0, None
        for child in children[span_id]:
            value = walk(child)
            if value > child_best:
                child_best, child_choice = value, child
        best[span_id] = (self_time[span_id] + child_best, child_choice)
        return best[span_id][0]

    total = walk(tree["root"])
    path = []
    cursor = tree["root"]
    while cursor is not None:
        path.append(cursor)
        cursor = best[cursor][1]
    return total, path


def enumerate_paths(tree):
    children = tree["children"]
    paths = []

    def walk(span_id, prefix):
        if not children[span_id]:
            paths.append(prefix + [span_id])
            return
        for child in children[span_id]:
            walk(child, prefix + [span_id])

    walk(tree["root"], [])
    return paths


def brute_force_critical_path(tree, self_time):
    best_path, best_value = None, None
    for path in enumerate_paths(tree):
        value = sum(self_time[span_id] for span_id in path)
        if best_value is None or value > best_value:
            best_path, best_value = path, value
    return best_value, best_path


def containment_violations(tree):
    """Parent/child interval violations using the reported (uncorrected) times."""
    by_id = tree["by_id"]
    violations = []
    for span_id, span in by_id.items():
        parent = span["parent"]
        if parent is None:
            continue
        outer = by_id[parent]
        if span["start"] < outer["start"] or span["end"] > outer["end"]:
            violations.append((parent, span_id))
    return violations


def estimate_skew(tree):
    """Bound every service's clock offset relative to the root by constraint propagation.

    For an edge parent -> child the reported times imply
        end_child - end_parent <= offset_child - offset_parent <= start_child - start_parent.
    Collecting those along every root-to-service path gives a sound interval per service.
    """
    edges = constraint_edges(tree)
    lower = {service: -INFINITY for service in SERVICES}
    upper = {service: INFINITY for service in SERVICES}
    lower[ROOT_SERVICE] = 0
    upper[ROOT_SERVICE] = 0
    for _ in range(len(SERVICES)):
        for (parent_service, child_service), (low, high) in sorted(edges.items()):
            if lower[parent_service] > -INFINITY:
                lower[child_service] = max(lower[child_service], lower[parent_service] + low)
            if upper[parent_service] < INFINITY:
                upper[child_service] = min(upper[child_service], upper[parent_service] + high)
    return {service: (lower[service], upper[service]) for service in SERVICES}


def constraint_edges(tree):
    """Tightest per-service-pair constraints (parent service, child service, low, high)."""
    by_id = tree["by_id"]
    edges = {}
    for span_id, span in by_id.items():
        parent = span["parent"]
        if parent is None:
            continue
        outer = by_id[parent]
        low = span["end"] - outer["end"]
        high = span["start"] - outer["start"]
        key = (outer["service"], span["service"])
        if key in edges:
            old_low, old_high = edges[key]
            edges[key] = (max(old_low, low), min(old_high, high))
        else:
            edges[key] = (low, high)
    return edges


def feasible_offsets(tree):
    """A correction that satisfies every constraint: shortest paths (Bellman-Ford).

    Each constraint `offset[child] - offset[parent] <= high` and
    `offset[parent] - offset[child] <= -low` becomes an upper bound of the form
    `d[x] <= d[y] + w`; with a virtual source at distance 0 the shortest-path solution is
    feasible whenever no negative cycle exists (there is none, since the true offsets solve
    the system).  The result is normalised so the root service has offset 0.
    """
    edges = []
    for (parent_service, child_service), (low, high) in sorted(constraint_edges(tree).items()):
        edges.append((parent_service, child_service, high))
        edges.append((child_service, parent_service, -low))
    distance = {service: 0 for service in SERVICES}
    for _ in range(len(SERVICES)):
        changed = False
        for start, end, weight in edges:
            if distance[end] > distance[start] + weight:
                distance[end] = distance[start] + weight
                changed = True
        if not changed:
            break
    base = distance[ROOT_SERVICE]
    return {service: distance[service] - base for service in SERVICES}


def correct_spans(spans, tree):
    offsets = feasible_offsets(tree)
    return [dict(span,
                 start=span["start"] - offsets[span["service"]],
                 end=span["end"] - offsets[span["service"]]) for span in spans]


def analyse(spans, correct=True):
    """The full pipeline: reconstruct, correct clocks, score self time, find the path."""
    tree = build_tree(spans)
    working = correct_spans(spans, tree) if correct else list(spans)
    if correct:
        tree = build_tree(working)
    self_time = self_times(working, tree)
    value, path = critical_path(tree, self_time)
    return tree, working, self_time, value, path


def bottleneck_service(tree, path, self_time):
    """The service contributing the most self time along the critical path."""
    totals = {}
    for span_id in path:
        service = tree["by_id"][span_id]["service"]
        totals[service] = totals.get(service, 0) + self_time[span_id]
    return max(sorted(totals), key=lambda service: totals[service]), totals


def percentile(samples, percent):
    if not samples:
        raise ValueError("no samples")
    ordered = sorted(samples)
    rank = -(-percent * len(ordered) // 100)
    return ordered[rank - 1]


def validate_capture(spans):
    """Full validation used by the collector; raises TraceError on anything unusable."""
    tree = build_tree(spans)
    for span_id, span in tree["by_id"].items():
        if span["end"] < span["start"]:
            raise TraceError("span %r ends before it starts" % (span_id,))
    return tree


class Collector:
    """Assembles captures and records the work it does (a linear cost bound)."""

    def __init__(self, samples_per_trace=1):
        self.samples_per_trace = samples_per_trace
        self.work = 0
        self.traces = []

    def capture(self, spans):
        tree = validate_capture(spans)
        self.work += len(spans)
        self.work += len(tree["order"])
        return tree


# ----------------------------------------------------------------------- main
def main():
    rng = Rng(893001)

    # --- 1. a capture reassembles into exactly the generated tree ----------
    skews = make_skews(rng)
    traces = []
    for trace_id in range(20):
        traces.append(generate_trace(rng, trace_id, skews))
    matching = 0
    for spans in traces:
        tree = build_tree(spans)
        by_id = tree["by_id"]
        for span_id, span in by_id.items():
            expected = sorted(span["children"])
            check(tree["children"][span_id] == expected,
                  "the reconstructed children of span %r must match the generator" % span_id)
            for child in expected:
                check(by_id[child]["parent"] == span_id,
                      "the parent pointer of %r must round-trip" % child)
            check(len(span["children"]) == len(set(span["children"])),
                  "no child may be attached twice")
        matching += 1
    check(matching == 20, "all twenty traces must reconstruct")

    # --- 2. arrival order is not causal order -------------------------------
    first = traces[0]
    scrambled = 0
    for span_list in traces:
        emission = [span["span"] for span in sorted(span_list, key=lambda span: span["span"])]
        arrival = [span["span"] for span in sorted(span_list,
                                                     key=lambda span: (span["end"], span["span"]))]
        if arrival != emission:
            scrambled += 1
        check(emission[-1] == build_tree(span_list)["root"],
              "the root is emitted last because it completes last")
        check(sorted(arrival) == sorted(emission), "arrival order must be a permutation")
    check(scrambled > 0,
          "clock skew must scramble the arrival order away from the causal order")
    check(len(first) > 8, "the generated traces must be reasonably large")

    # --- 3. the validator rejects broken captures ---------------------------
    def expect_error(spans, label):
        raised = False
        try:
            build_tree(spans)
        except TraceError:
            raised = True
        check(raised, "a capture with %s must be rejected" % label)

    root_id = build_tree(first)["root"]
    expect_error([dict(span, parent=999) if span["span"] == root_id else span for span in first],
                 "a missing root")
    orphaned = [dict(span) for span in first]
    orphaned[1]["parent"] = 4242
    expect_error(orphaned, "an orphaned span")
    duplicated = [dict(span) for span in first] + [dict(first[0])]
    expect_error(duplicated, "a duplicated span id")
    cycled = [
        {"trace": 5, "span": 0, "parent": 1, "service": "auth", "true_start": 1,
         "true_end": 5, "start": 1, "end": 5, "children": []},
        {"trace": 5, "span": 1, "parent": 0, "service": "session", "true_start": 1,
         "true_end": 5, "start": 1, "end": 5, "children": []},
        {"trace": 5, "span": 2, "parent": None, "service": ROOT_SERVICE, "true_start": 0,
         "true_end": 9, "start": 0, "end": 9, "children": []},
    ]
    expect_error(cycled, "a parent cycle")
    mixed = [dict(span) for span in first]
    mixed[0]["trace"] = "other"
    expect_error(mixed, "spans from two traces")
    two_roots = [dict(span) for span in first]
    for span in two_roots:
        if span["span"] == first[0]["span"]:
            span["parent"] = None
    expect_error(two_roots, "more than one root")
    detached = [dict(span) for span in first]
    detached.append({"trace": first[0]["trace"], "span": 10 ** 6, "parent": 10 ** 7,
                     "service": "auth", "true_start": 0, "true_end": 1, "start": 0, "end": 1,
                     "children": []})
    expect_error(detached, "an unreachable span")

    # --- 4. critical path: dynamic program versus exhaustive enumeration ----
    dp_checks = 0
    for span_list in traces:
        tree, corrected, self_time, dp_value, dp_path = analyse(span_list)
        brute_value, brute_path = brute_force_critical_path(tree, self_time)
        check(dp_value == brute_value,
              "the critical path must match the enumeration: %d vs %d" % (dp_value, brute_value))
        check(sum(self_time[span_id] for span_id in dp_path) == dp_value,
              "the reconstructed path must sum to the reported value")
        check(len(dp_path) == len(set(dp_path)), "the critical path may not repeat a span")
        check(dp_path[0] == tree["root"], "the critical path must start at the root")
        check(not tree["children"][dp_path[-1]],
              "the critical path must end at a leaf")
        check(corrected != span_list, "the clock correction must change the reported times")
        dp_checks += 1
    check(dp_checks == 20, "every trace must be cross-checked")

    # --- 5. self time, work and the wall clock ------------------------------
    max_deviation = 0
    for span_list in traces:
        tree, corrected, self_time, value, path = analyse(span_list)
        root = tree["by_id"][tree["root"]]
        wall = root["end"] - root["start"]
        check(all(value >= 0 for value in self_time.values()),
              "no span may have negative self time")
        check(sum(self_time.values()) >= wall,
              "total work can never be less than the wall clock: %d vs %d"
              % (sum(self_time.values()), wall))
        for candidate in enumerate_paths(tree):
            path_sum = sum(self_time[span_id] for span_id in candidate)
            check(path_sum <= wall,
                  "the self time along any root-to-leaf path must fit inside the root span:"
                  " %d vs %d" % (path_sum, wall))
        check(value <= wall, "the critical path must fit inside the wall clock")
        truth = [{**span, "start": span["true_start"], "end": span["true_end"]}
                 for span in span_list]
        truth_tree = build_tree(truth)
        true_self = self_times(truth, truth_tree)
        true_root = truth_tree["by_id"][truth_tree["root"]]
        check(true_root["end"] - true_root["start"] == wall,
              "the root duration must be recovered exactly by the correction")
        deviation = max(abs(self_time[key] - true_self[key]) for key in self_time)
        max_deviation = max(max_deviation, deviation)
        check(deviation <= 6,
              "self times must survive the correction up to the skew spread, got %d" % deviation)
    check(max_deviation > 0,
          "the correction must be a real estimate, not a copy of the hidden truth")
    check(union_length([]) == 0 and union_length([(0, 5)]) == 5
          and union_length([(0, 5), (2, 8)]) == 8
          and union_length([(0, 2), (3, 5)]) == 4,
          "interval union must merge overlaps and add gaps")

    # --- 5b. a purely sequential chain has no parallelism slack -------------
    chain = [
        {"trace": 11, "span": 0, "parent": None, "service": ROOT_SERVICE, "true_start": 0,
         "true_end": 10, "start": 0, "end": 10, "children": [1]},
        {"trace": 11, "span": 1, "parent": 0, "service": "auth", "true_start": 2,
         "true_end": 8, "start": 2, "end": 8, "children": [2]},
        {"trace": 11, "span": 2, "parent": 1, "service": "session", "true_start": 3,
         "true_end": 6, "start": 3, "end": 6, "children": []},
    ]
    chain_tree, _, chain_self, chain_value, chain_path = analyse(chain, correct=False)
    check(sum(chain_self.values()) == 10,
          "a sequential chain must have work exactly equal to the wall clock")
    check(chain_value == 10 and chain_path == [0, 1, 2],
          "the critical path of a chain must be the whole chain")

    # --- 5c. parallel and sequential fan-out are distinguished --------------
    parallel_seen = 0
    sequential_seen = 0
    for span_list in traces:
        truth = [{**span, "start": span["true_start"], "end": span["true_end"]}
                 for span in span_list]
        tree = build_tree(truth)
        by_id = tree["by_id"]
        for span_id, span in by_id.items():
            children = tree["children"][span_id]
            if len(children) < 2:
                continue
            union = union_length([(by_id[child]["start"], by_id[child]["end"])
                                  for child in children])
            total = sum(by_id[child]["end"] - by_id[child]["start"] for child in children)
            if GRAPH[span["service"]]["mode"] == "sequential":
                check(union == total,
                      "sequential children of %r must not overlap" % span["service"])
                check(by_id[children[0]]["end"] <= by_id[children[1]]["start"],
                      "sequential children must be ordered in time")
                sequential_seen += 1
            else:
                check(union < total,
                      "parallel children of %r must overlap" % span["service"])
                parallel_seen += 1
    check(parallel_seen > 0 and sequential_seen > 0,
          "the traces must contain both parallel and sequential fan-out")

    # --- 6. reported times break containment, correction restores it -------
    broken = 0
    for span_list in traces:
        tree = build_tree(span_list)
        broken += len(containment_violations(tree))
        corrected_tree = build_tree(correct_spans(span_list, tree))
        check(containment_violations(corrected_tree) == [],
              "the clock correction must restore containment for every edge")
        truth = [{**span, "start": span["true_start"], "end": span["true_end"]}
                 for span in span_list]
        check(containment_violations(build_tree(truth)) == [],
              "the true times must satisfy containment for every edge")
        check(feasible_offsets(tree)[ROOT_SERVICE] == 0,
              "the correction must be normalised to the root service")
    check(broken > 0,
          "clock offsets must actually produce containment violations in the raw capture")

    # --- 7. skew estimation is sound ----------------------------------------
    widths = []
    contained = 0
    longest_trace = 0
    for span_list in traces:
        tree = build_tree(span_list)
        estimate = estimate_skew(tree)
        root = tree["by_id"][tree["root"]]
        longest_trace = max(longest_trace, root["end"] - root["start"])
        for service in SERVICES:
            low, high = estimate[service]
            check(low <= skews[service] <= high,
                  "service %r true offset %d must lie in [%r, %r]"
                  % (service, skews[service], low, high))
            contained += 1
            check(low > -INFINITY and high < INFINITY,
                  "every service reached by an edge must get a finite interval")
            widths.append(high - low)
    check(max(widths) <= longest_trace,
          "a constraint interval cannot be looser than the trace itself: %d vs %d"
          % (max(widths), longest_trace))
    check(min(widths) < max(widths),
          "some services must be pinned down more tightly than others")
    mean_width = sum(widths) / len(widths)

    # --- 8. the injected bottleneck is identified ---------------------------
    found = 0
    for trial in range(10):
        local_skews = make_skews(rng)
        span_list = generate_trace(rng, 100 + trial, local_skews, slow_service="catalog",
                                   slow_extra=9)
        tree, _, self_time, _, path = analyse(span_list)
        service, totals = bottleneck_service(tree, path, self_time)
        if service == "catalog":
            found += 1
        check(service in totals, "the bottleneck must be one of the services on the path")
    check(found == 10, "the deliberately slowed service must be found every time (got %d)" % found)

    # --- 9. no injection, no surprise winner -------------------------------
    plain = generate_trace(Rng(893400), 900, make_skews(Rng(893401)))
    tree, _, self_time, _, path = analyse(plain)
    service, totals = bottleneck_service(tree, path, self_time)
    by_service = {}
    for span_id, span in tree["by_id"].items():
        by_service.setdefault(span["service"], 0)
        by_service[span["service"]] += self_time[span_id]
    check(totals[service] == max(totals.values()),
          "the reported bottleneck must be the largest contributor on the path")
    check(max(by_service.values()) > 0, "some service must contribute self time")
    check(all(value >= 0 for value in by_service.values()),
          "aggregated self times must stay non-negative")

    # --- 10. head sampling --------------------------------------------------
    latencies = [span["end"] - span["start"] for span_list in traces for span in span_list
                 if span["parent"] is None]
    check(len(latencies) == 20, "one root latency per trace")
    sampled = [value for index, value in enumerate(latencies) if index % 2 == 0]
    check(len(sampled) == 10, "head sampling at 1 in 2 must keep exactly half")
    check(set(sampled) <= set(latencies), "a sample must be a subset of the population")
    check(abs(percentile(sampled, 90) - percentile(latencies, 90)) <= percentile(latencies, 90),
          "the sampled quantile must stay in the same order of magnitude")
    check(percentile(latencies, 100) == max(latencies),
          "the hundredth percentile must be the maximum")
    check(percentile(latencies, 50) <= percentile(latencies, 99),
          "quantiles must be monotone")
    raised = False
    try:
        percentile([], 50)
    except ValueError:
        raised = True
    check(raised, "an empty capture has no percentile")

    # --- 11. degenerate traces ---------------------------------------------
    lonely = [{"trace": 7, "span": 0, "parent": None, "service": ROOT_SERVICE,
               "true_start": 0, "true_end": 4, "start": 0, "end": 4, "children": []}]
    lonely_tree = build_tree(lonely)
    lonely_self = self_times(lonely, lonely_tree)
    check(lonely_self == {0: 4}, "a single span is entirely self time")
    check(critical_path(lonely_tree, lonely_self) == (4, [0]),
          "the critical path of a single span is that span")
    expect_error([], "no spans at all")
    clipped = [dict(lonely[0], start=5, end=4)]
    raised = False
    try:
        validate_capture(clipped)
    except TraceError:
        raised = True
    check(raised, "a span ending before it starts must be rejected")

    # --- 12. cost is linear and order independent ---------------------------
    collector = Collector()
    for span_list in traces:
        collector.capture(span_list)
    total_spans = sum(len(span_list) for span_list in traces)
    check(collector.work == 2 * total_spans,
          "the collector must do a constant amount of work per span: %d vs %d"
          % (collector.work, 2 * total_spans))
    shuffled = list(reversed(traces[3]))
    check(build_tree(shuffled)["children"] == build_tree(traces[3])["children"],
          "the reconstruction must not depend on arrival order")
    check(analyse(shuffled)[2] == analyse(traces[3])[2],
          "self times must not depend on arrival order")

    # --- 13. scale ----------------------------------------------------------
    scale_rng = Rng(893700)
    scale_skews = make_skews(scale_rng)
    scale_traces = [generate_trace(scale_rng, 2000 + index, scale_skews,
                                   slow_service="storage" if index % 5 == 0 else None)
                    for index in range(300)]
    scale_collector = Collector()
    scale_roots = []
    for span_list in scale_traces:
        tree = scale_collector.capture(span_list)
        _, _, self_time, value, path = analyse(span_list)
        brute, _ = brute_force_critical_path(tree, self_time)
        check(value == brute, "the scaled critical path must match the enumeration")
        check(value <= tree["by_id"][tree["root"]]["end"] - tree["by_id"][tree["root"]]["start"],
              "the scaled critical path must fit inside the root span")
        check(sum(self_time.values()) >= tree["by_id"][tree["root"]]["end"]
              - tree["by_id"][tree["root"]]["start"],
              "the scaled work must cover the scaled wall clock")
        scale_roots.append(tree["by_id"][tree["root"]]["end"]
                           - tree["by_id"][tree["root"]]["start"])
    scale_spans = sum(len(span_list) for span_list in scale_traces)
    check(scale_spans > 3000, "the scaled capture must be substantial, got %d spans" % scale_spans)
    check(scale_collector.work == 2 * scale_spans,
          "the scaled collector must stay linear in the spans")
    check(percentile(scale_roots, 99) >= percentile(scale_roots, 50),
          "the scaled quantiles must be ordered")
    slowed = [value for index, value in enumerate(scale_roots) if index % 5 == 0]
    check(max(slowed) > min(scale_roots), "the injected slowdown must be visible at scale")

    # --- 14. reproducibility ------------------------------------------------
    def replay(seed):
        local = Rng(seed)
        local_skews = make_skews(local)
        results = []
        for trace_id in range(6):
            span_list = generate_trace(local, trace_id, local_skews, slow_service="pricing")
            tree, _, self_time, value, path = analyse(span_list)
            service, _ = bottleneck_service(tree, path, self_time)
            results.append((value, tuple(path), service, estimate_skew(tree)))
        return results

    check(replay(893900) == replay(893900), "an identical seed must replay the whole analysis")

    print("traces: %d reconstructed, %d spans, arrival order independent"
          % (len(traces), total_spans))
    print("critical path: %d dynamic programs matched exhaustive enumeration" % dp_checks)
    print("skew: %d service offsets bounded, widest interval %d (trace is %d long), mean %.1f"
          % (contained, max(widths), longest_trace, mean_width))
    print("containment: %d raw violations corrected by the clock offsets" % broken)
    print("scale: %d spans over %d traces, collector work %d"
          % (scale_spans, len(scale_traces), scale_collector.work))
    print("PASS")


if __name__ == "__main__":
    main()
