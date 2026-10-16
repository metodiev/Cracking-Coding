"""Task 941 - Implement continuous profiling pipeline that attributes CPU/ memory hotspots to code and inputs.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A deterministic sampling profiler over a fixed set of call paths.  Samples (stack, window, input class,
allocation bytes) are collected per time window; the pipeline aggregates them into (a) a call tree with
inclusive and self costs, (b) a per-frame view, (c) an attribution matrix frame x input class and (d) a
memory view that charges every allocation to its leaf frame.  Comparing two windows yields the
regression list - frames whose inclusive cost grew by at least a delta and by more than a ratio - which
is how a newly hot function is surfaced.

Simplified versus a production profiler (pprof/Parca): short fixed paths instead of real symbols, one
sample per unit of work instead of wall-clock sampling, no DWARF unwinding.  Verified: cost
conservation; the call tree's inclusive costs equal an independent brute-force prefix count and its
self costs equal the brute-force whole-stack counts; every frame's inclusive cost dominates its self
cost; attribution follows the declared input mix and conserves visits; an injected hot path is the only
flagged regression while identical windows flag nothing; memory attribution is conserved and charged to
the allocating leaf; drilling into a frame returns exactly the samples that ran through it; empty and
single-sample profiles work; generation and ranking are deterministic.
"""
FRAME_PATHS = (
    (("main", "request_handler", "db_query"), 30),
    (("main", "request_handler", "render"), 20),
    (("main", "request_handler", "cache_lookup"), 25),
    (("main", "background", "serialize"), 15),
    (("main", "background", "db_query"), 10),
)
INPUT_MIX = (("tenant-a", 40), ("tenant-b", 35), ("tenant-c", 25))
HOT_PATH = ("main", "request_handler", "serialize", "json_dumps")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - fixed seed, so the sampled profile is reproducible."""

    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)

    def below(self, bound):
        return self.next() % bound


def draw(rng, choices):
    point = rng.below(sum(weight for _, weight in choices))
    for value, weight in choices:
        if point < weight:
            return value
        point -= weight
    return choices[-1][0]


class Sample:
    __slots__ = ("stack", "window", "input_class", "alloc")

    def __init__(self, stack, window, input_class, alloc=0):
        self.stack = stack
        self.window = window
        self.input_class = input_class
        self.alloc = alloc


def generate(rng, count, window, hot_weight=0):
    """Sample `count` stacks in one window; hot_weight is the per-mille share of the hot path."""
    samples = []
    for _ in range(count):
        stack = HOT_PATH if rng.below(1000) < hot_weight else draw(rng, FRAME_PATHS)
        samples.append(Sample(stack, window, draw(rng, INPUT_MIX),
                              rng.below(4096) if rng.below(4) == 0 else 0))
    return samples


def self_costs(samples):
    counts = {}
    for sample in samples:
        counts[sample.stack[-1]] = counts.get(sample.stack[-1], 0) + 1
    return counts


def inclusive_costs(samples):
    counts = {}
    for sample in samples:
        for frame in sample.stack:
            counts[frame] = counts.get(frame, 0) + 1
    return counts


def prefix_costs(samples):
    """Independent oracle for the call tree: every stack prefix counted once per sample."""
    counts = {}
    for sample in samples:
        for length in range(1, len(sample.stack) + 1):
            counts[sample.stack[:length]] = counts.get(sample.stack[:length], 0) + 1
    return counts


def stack_costs(samples):
    """Independent oracle for the tree's self costs: whole stacks counted once per sample."""
    counts = {}
    for sample in samples:
        counts[sample.stack] = counts.get(sample.stack, 0) + 1
    return counts


def call_tree(samples):
    root = {"self": 0, "inclusive": 0, "children": {}}
    for sample in samples:
        node = root
        node["inclusive"] += 1
        for frame in sample.stack:
            node = node["children"].setdefault(frame, {"self": 0, "inclusive": 0, "children": {}})
            node["inclusive"] += 1
        node["self"] += 1
    return root


def flatten(node, field="inclusive", path=(), out=None):
    """Path -> cost, for comparison with the brute-force prefix/stack oracles."""
    if out is None:
        out = {}
    for frame, child in node["children"].items():
        out[path + (frame,)] = child[field]
        flatten(child, field, path + (frame,), out)
    return out


def attribution(samples, frame):
    """Input-class mix of the samples that ran through `frame`."""
    counts = {}
    for sample in samples:
        if frame in sample.stack:
            counts[sample.input_class] = counts.get(sample.input_class, 0) + 1
    return counts


def drilldown(samples, frame):
    return [sample for sample in samples if frame in sample.stack]


def memory_by_leaf(samples):
    counts = {}
    for sample in samples:
        if sample.alloc:
            leaf = sample.stack[-1]
            counts[leaf] = counts.get(leaf, 0) + sample.alloc
    return counts


def regressions(before, after, ratio=1.5, delta=4):
    """Frames whose inclusive cost grew by >= delta samples and by more than `ratio` times."""
    old, new = inclusive_costs(before), inclusive_costs(after)
    grown = [frame for frame in new
             if new[frame] - old.get(frame, 0) >= delta
             and new[frame] > ratio * max(old.get(frame, 1), 1)]
    return sorted(grown, key=lambda frame: (-(new[frame] - old.get(frame, 0)), frame))


def main():
    generator = Rng(4242)
    window_one = generate(generator, 4000, 1)
    window_two = generate(generator, 4000, 2, hot_weight=120)
    tree = call_tree(window_one)
    self_cost, inclusive = self_costs(window_one), inclusive_costs(window_one)

    # 1) conservation: one unit of self cost and one root visit per sample.
    check(sum(self_cost.values()) == len(window_one) and tree["inclusive"] == len(window_one),
          f"self costs {sum(self_cost.values())} must account for {len(window_one)} samples")
    # 2) the call tree matches the independent prefix-count oracle, node by node.
    tree_self = {path: cost for path, cost in flatten(tree, "self").items() if cost}
    check(flatten(tree) == prefix_costs(window_one) and tree_self == stack_costs(window_one),
          "tree inclusive costs must equal the prefix counts and self costs the stack counts")
    # 3) per-frame inclusive cost dominates self cost, and all paths share one root frame.
    check(all(inclusive.get(frame, 0) >= cost for frame, cost in self_cost.items())
          and list(tree["children"]) == ["main"],
          f"inclusive costs {inclusive} must dominate self costs {self_cost} and share a root")

    # 4) attribution: the frame's input mix follows the declared mix and conserves visits.
    attributed = attribution(window_one, "db_query")
    total = sum(attributed.values())
    declared = {name: weight / sum(w for _, w in INPUT_MIX) for name, weight in INPUT_MIX}
    worst = max(abs(attributed.get(name, 0) / total - share) for name, share in declared.items())
    check(total == inclusive["db_query"] and worst <= 0.05,
          f"attribution total {total} vs inclusive {inclusive['db_query']}, drift {worst:.4f}")

    # 5) regression detection: only the injected hot path is flagged; identical windows flag none.
    flagged = regressions(window_one, window_two)
    check(flagged and all(frame in HOT_PATH for frame in flagged)
          and regressions(window_one, window_one) == []
          and regressions(window_two, window_two) == [],
          f"the regression list must be the injected hot path only: {flagged}")

    # 6) memory attribution: conserved in total and charged to the leaf that allocated.
    memory = memory_by_leaf(window_one)
    observed = sum(sample.alloc for sample in window_one)
    check(sum(memory.values()) == observed and memory and len(memory) == len(set(FRAME_PATHS[i][0][-1]
                                                                               for i in range(5))),
          f"memory attribution {sum(memory.values())} vs observed {observed} across "
          f"{len(memory)} leaves")

    # 7) drilldown returns exactly the samples that ran through the frame, in generation order.
    subset = drilldown(window_one, "render")
    check(len(subset) == inclusive["render"] and subset
          and all(sample.stack[-2:] == ("request_handler", "render") for sample in subset)
          and sum(self_costs(subset).values()) == len(subset),
          f"drilldown returned {len(subset)} samples for {inclusive['render']} visits")

    # 8) boundaries: an empty profile aggregates to nothing, a single sample to unit costs.
    single = [Sample(("main", "db_query"), 3, "tenant-a", 512)]
    check(self_costs([]) == {} and inclusive_costs([]) == {} and regressions([], []) == []
          and memory_by_leaf([]) == {} and flatten(call_tree([])) == {}
          and self_costs(single) == {"db_query": 1} and inclusive_costs(single)["main"] == 1
          and memory_by_leaf(single) == {"db_query": 512},
          "the empty and single-sample boundaries must aggregate correctly")

    # 9) determinism, ranking by self cost, and the injected frame visible only in window two.
    repeat = generate(Rng(4242), 4000, 1)
    ranked = sorted(self_cost, key=lambda frame: (-self_cost[frame], frame))
    brute = sorted({sample.stack[-1] for sample in window_one},
                   key=lambda frame: (-self_cost[frame], frame))
    second_self = self_costs(window_two)
    check([sample.stack for sample in repeat] == [sample.stack for sample in window_one]
          and ranked == brute and "json_dumps" not in self_cost
          and second_self.get("json_dumps", 0) >= 400,
          f"generation and ranking must be deterministic; injected self cost "
          f"{second_self.get('json_dumps', 0)}")

    print(f"samples={len(window_one)}+{len(window_two)} tree_nodes={len(flatten(tree))} "
          f"db_query_inclusive={inclusive['db_query']} attribution={dict(sorted(attributed.items()))}")
    print(f"regressions={flagged} memory_total={sum(memory.values())} "
          f"injected_self_cost={second_self.get('json_dumps', 0)}")
    print("PASS")


if __name__ == "__main__":
    main()
