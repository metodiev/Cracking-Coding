"""Task 674 - Implement register allocation via graph coloring (simplified).  (Tier: Expert, Beyond — Compilers, Languages & Solvers)

Chaitin-style global register allocation on an interference graph.

  * `interference(program)` builds the graph for straight-line code: a variable is live
    from its definition until its last use, and two variables that are live at the same
    program point interfere (they cannot share a register). Liveness is computed with a
    backward scan and re-checked against an independent forward per-value scan.
  * `color_graph(graph, k)` is simplify / spill / select: repeatedly remove a node of degree
    < k (push it on the stack); when every remaining node has degree >= k, pick the highest
    degree one as a *potential* spill and keep going (optimistic spilling). Popping the
    stack assigns the lowest free colour, and a node that finds no free colour becomes an
    *actual* spill, which the caller would turn into memory traffic.

Verified: every colouring is checked to be proper (no interfering pair shares a colour) and
to use at most k colours; the number of colours used is never below the brute-force
chromatic number of the graph; and the spill count is counted together with the extra
load/store traffic it implies.

The strongest cross-check uses a theorem instead of a second implementation: when every
value has a single live range (each variable is defined once - live ranges are then
intervals), the interference graph is an interval graph, where the clique number omega (the
maximum number of simultaneously live values) equals the chromatic number, so the allocator
must spill if and only if max_live > k. That is verified exactly for 100 programs and
k = 1..4, which pins the graph construction and the colouring together. Programs that reuse
variables have values with several disjoint live ranges, so there only validity and
"spill-free implies chi <= k" are claimed, and no optimality.

Simplified: spilling is not followed by live-range splitting and re-colouring - spilled
values are accounted for as reload traffic only, which is stated where the cost is reported.
"""

from itertools import permutations  # noqa: F401  (kept out of the hot path; unused on purpose)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def liveness_points(program):
    """Live-out set at every program point, by a single backward scan."""
    points = [set()] * (len(program) + 1)
    live = set()
    for i in range(len(program) - 1, -1, -1):
        op, var = program[i]
        if op == "def":
            live.discard(var)
        else:
            live.add(var)
        points[i] = set(live)
    return points


def live_oracle(program):
    """Independent formulation: v is live at point i iff some later use reads a value that
    was already available at i, i.e. no redefinition of v lies in [i, use)."""
    n = len(program)
    variables = sorted({var for _, var in program})
    result = []
    for i in range(n + 1):
        live = set()
        for var in variables:
            for j in range(i, n):
                if program[j] != ("use", var):
                    continue
                if all(program[k] != ("def", var) for k in range(i, j)):
                    live.add(var)
                break
        result.append(live)
    return result


def max_live(program):
    """Clique number omega: the maximum number of simultaneously live variables."""
    points = liveness_points(program)
    return max(len(s) for s in points)


def interference(program):
    """Interference graph as an adjacency map: a definition clashes with everything that is
    live right after it. Dead definitions (whose value is never read) are ignored, exactly
    as dead code elimination would have removed them, so the graph stays exact for the
    values that really need a register."""
    points = liveness_points(program)
    graph = {var: set() for _, var in program}
    for i, (op, var) in enumerate(program):
        if op != "def" or var not in points[i + 1]:
            continue                     # this value is overwritten / never read again
        for other in points[i + 1]:
            if other != var:
                graph[var].add(other)
                graph[other].add(var)
    for node, neighbours in graph.items():
        neighbours.discard(node)
    return graph


def needed_points(program):
    """Oracle for the graph: for each value, the program points where it must be held."""
    needed = {var: set() for _, var in program}
    for j, (op, var) in enumerate(program):
        if op != "use":
            continue
        producer = max((k for k in range(j) if program[k] == ("def", var)), default=None)
        for point in range(producer if producer is not None else 0, j):
            needed[var].add(point)
    return needed


def overlap_edges(program):
    """Independent edge oracle: two values clash when both are needed at the same point."""
    needed = needed_points(program)
    nodes = sorted(needed)
    return {(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]
            if needed[a] & needed[b]}


def edge_set(graph):
    return {tuple(sorted((a, b))) for a, neighbours in graph.items() for b in neighbours if a < b}


def colive_edges(program):
    """Pairs of variables that are live at some common program point."""
    edges = set()
    for live in liveness_points(program):
        nodes = sorted(live)
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                edges.add((a, b))
    return edges


def max_degree(graph):
    return max((len(v) for v in graph.values()), default=0)


def color_graph(graph, k):
    """Optimistic Chaitin-Briggs. Returns (colors, spills, spill_candidates, rounds)."""
    if k < 1:
        raise ValueError("at least one register is required")
    for node, neighbours in graph.items():
        if node in neighbours:
            raise ValueError("self loops are not colourings")
    work = {node: set(neighbours) for node, neighbours in graph.items()}
    stack, candidates = [], []
    while work:
        low = sorted(node for node in work if len(work[node]) < k)
        if low:
            node = low[0]
        else:
            node = max(sorted(work), key=lambda n: (len(work[n]), n))
            candidates.append(node)
        stack.append(node)
        for other in work[node]:
            work[other].discard(node)
        del work[node]
    colors, spills = {}, []
    while stack:
        node = stack.pop()
        taken = {colors[other] for other in graph[node] if other in colors}
        free = [c for c in range(k) if c not in taken]
        if free:
            colors[node] = free[0]
        else:
            spills.append(node)
    return colors, sorted(spills), sorted(candidates), len(stack)


def proper_coloring(graph, colors, k):
    if any(color >= k or color < 0 for color in colors.values()):
        return False
    return all(colors[a] != colors[b] for a, b in edge_set(graph) if a in colors and b in colors)


def chromatic_number(graph):
    """Brute-force chi via backtracking over colours (small graphs only)."""
    if not graph:
        return 0
    order = sorted(graph, key=lambda n: -len(graph[n]))
    for k in range(1, len(order) + 1):
        def place(index, colors):
            if index == len(order):
                return True
            node = order[index]
            taken = {colors[other] for other in graph[node] if other in colors}
            for color in range(k):
                if color not in taken:
                    colors[node] = color
                    if place(index + 1, colors):
                        return True
                    del colors[node]
            return False

        if place(0, {}):
            return k
    return len(order)


def spill_traffic(program, spills):
    """Memory operations implied by the spilled values: one store per def, one load per use."""
    return sum(1 for op, var in program if var in spills)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_program(seed, length=14, variables=5):
    """Straight-line code: every use reads a value that was defined earlier."""
    state = seed
    program = []
    defined = []
    for i in range(length):
        state, raw = splitmix64(state)
        var = "v%d" % (raw % variables)
        state, raw = splitmix64(state)
        if defined and raw % 3 == 0:
            program.append(("use", defined[raw % len(defined)]))
        else:
            program.append(("def", var))
            defined.append(var)
    return program


def single_assignment_program(seed, length=12):
    """Each value is defined exactly once, so every live range is a single interval."""
    state = seed
    program = []
    live_names = []
    for _ in range(length):
        state, raw = splitmix64(state)
        if live_names and raw % 3 == 0:
            program.append(("use", live_names[raw % len(live_names)]))
        else:
            name = "s%d" % len(live_names)
            program.append(("def", name))
            live_names.append(name)
    return program


def random_graph(seed, nodes=6, density=3):
    state = seed
    names = ["n%d" % i for i in range(nodes)]
    graph = {name: set() for name in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            state, raw = splitmix64(state)
            if raw % density == 0:
                graph[a].add(b)
                graph[b].add(a)
    return graph


def main():
    # ---- intervals and liveness -------------------------------------------------------
    program = [("def", "a"), ("def", "b"), ("use", "a"), ("use", "b"), ("def", "b"),
               ("use", "b")]
    points = liveness_points(program)
    check(points[0] == set() and points[len(program)] == set(),
          "nothing is live before the first definition or after the last use: %r" % (points,))
    check(max_live(program) == 2, "two values are live at the peak: %d" % max_live(program))
    for i in range(len(program) + 1):
        check(points[i] == live_oracle(program)[i],
              "backward scan and forward oracle disagree at point %d" % i)
    graph = interference(program)
    check(graph["a"] == {"b"} and graph["b"] == {"a"},
          "a and b overlap, and a stops interfering once overwritten: %r" % (graph,))
    check(edge_set(graph) == overlap_edges(program),
          "the value-point oracle must reproduce the same edges: %r" % (edge_set(graph),))
    check(interference(program) == graph, "graph construction must be deterministic")

    disjoint = [("def", "a"), ("use", "a"), ("def", "b"), ("use", "b")]
    check(edge_set(interference(disjoint)) == set(),
          "values whose live ranges do not overlap must not interfere")

    # ---- interval-graph exactness (single live range per value) -----------------------
    across_k = 0
    for i in range(100):
        code = single_assignment_program(0x5741 + i * 7919)
        built = interference(code)
        check(edge_set(built) == overlap_edges(code), "edge oracle mismatch on program %d" % i)
        peak = max_live(code)
        if peak == 0:
            check(edge_set(built) == set() and color_graph(built, 1)[1] == [],
                  "with nothing live there is no interference and nothing to spill (%d)" % i)
            continue
        check(chromatic_number(built) == peak,
              "chi must equal the peak for interval graphs (%d): %d vs %d"
              % (i, chromatic_number(built), peak))
        for k in range(1, 5):
            colors, spills, candidates, _ = color_graph(built, k)
            check(proper_coloring(built, colors, k), "improper colouring (%d, k=%d)" % (i, k))
            check(len(colors) + len(spills) == len(built),
                  "every value is either coloured or spilled (%d, k=%d)" % (i, k))
            # interval graphs satisfy chi == omega, so spilling happens iff the peak
            # exceeds the register file; this is exact, not a heuristic observation.
            check((spills == []) == (peak <= k),
                  "interval-graph exactness broken on program %d with k=%d (peak=%d, spills=%r)"
                  % (i, k, peak, spills))
            if spills:
                across_k += 1
                check(spill_traffic(code, spills) >= len(spills),
                      "each spill costs at least one memory operation")
    check(across_k > 20, "the sweep must exercise spilling (%d cases)" % across_k)

    # ---- programs that reuse variables: validity only --------------------------------
    reused = 0
    for i in range(60):
        code = random_program(0xBEE5 + i * 7919)
        built = interference(code)
        check(edge_set(built) == overlap_edges(code), "reused-value oracle mismatch (%d)" % i)
        check(edge_set(built) <= colive_edges(code),
              "every interference edge joins two values live at a common point (%d)" % i)
        chi = chromatic_number(built)
        for k in range(1, 4):
            colors, spills, _, _ = color_graph(built, k)
            check(proper_coloring(built, colors, k), "improper colouring (%d, k=%d)" % (i, k))
            check(len(colors) + len(spills) == len(built), "coverage (%d, k=%d)" % (i, k))
            if not spills:
                check(chi <= k, "a spill-free colouring proves chi <= k (%d, k=%d)" % (i, k))
            else:
                reused += 1
    check(reused > 0, "the reused-variable sweep must spill somewhere (%d)" % reused)

    # ---- general graphs: validity, chromatic bound, spilling ---------------------------
    spills_seen = 0
    optimal_greedy = 0
    for i in range(120):
        graph = random_graph(0x1CE1 + i * 104729, 6, 4 + i % 3)
        chi = chromatic_number(graph)
        colors, spills, candidates, _ = color_graph(graph, 3)
        check(proper_coloring(graph, colors, 3), "improper colouring on random graph %d" % i)
        if not spills:
            check(chi <= 3, "a spill-free colouring proves the graph is 3-colourable (%d)" % i)
        else:
            spills_seen += 1
        if not spills and chi == 3:
            optimal_greedy += 1
        check(len(colors) + len(spills) == len(graph), "coloured plus spilled must cover all")
        check(all(node not in colors for node in spills), "a spilled value gets no register")
    check(spills_seen > 0 and optimal_greedy > 0,
          "the sweep must contain both spilled graphs (%d) and 3-colourable ones (%d)"
          % (spills_seen, optimal_greedy))

    triangle = {"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"}}
    check(chromatic_number(triangle) == 3, "a triangle needs three colours")
    two, spills_two, _, _ = color_graph(triangle, 2)
    check(proper_coloring(triangle, two, 2) and len(spills_two) == 1,
          "with two registers exactly one value must spill: %r" % (spills_two,))
    three, spills_three, _, _ = color_graph(triangle, 3)
    check(spills_three == [] and proper_coloring(triangle, three, 3),
          "with three registers the triangle fits")
    check(len(set(three.values())) == 3,
          "all three registers must be used by the triangle: %r" % (three,))

    odd_cycle = {"a": {"b", "e"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"c", "e"},
                 "e": {"d", "a"}}
    check(chromatic_number(odd_cycle) == 3, "an odd cycle needs three colours")
    check(color_graph(odd_cycle, 3)[1] == [], "three registers suffice for a 5-cycle")
    check(len(color_graph(odd_cycle, 2)[1]) >= 1, "two registers cannot hold a 5-cycle")

    # ---- boundary cases, errors, determinism -----------------------------------------
    empty, empty_spills, _, _ = color_graph({}, 2)
    check(empty == {} and empty_spills == [] and chromatic_number({}) == 0,
          "an empty graph needs nothing")
    lone, lone_spills, _, _ = color_graph({"solo": set()}, 1)
    check(lone == {"solo": 0} and lone_spills == [], "an isolated value takes colour 0")
    check(interference([("def", "x")]) == {"x": set()},
          "a definition with no use interferes with nothing")
    rejected = 0
    for bad in (lambda: color_graph({"a": set()}, 0),
                lambda: color_graph({"a": {"a"}}, 2),
                lambda: color_graph({"a": {"b"}}, -1)):
        try:
            bad()
        except ValueError:
            rejected += 1
    check(rejected == 3, "invalid register files and graphs must be rejected (%d)" % rejected)
    check(color_graph(odd_cycle, 3) == color_graph(odd_cycle, 3),
          "allocation must be deterministic")
    check(spill_traffic(program, ["b"]) == 4,
          "spill traffic counts the defs and uses of the spilled value: %d"
          % spill_traffic(program, ["b"]))

    print("interval graphs: 100 programs x 4 register counts, spill <=> peak > k exactly")
    print("random graphs: %d spilled, %d optimally 3-coloured of 120; triangle spills with "
          "k=2: %r" % (spills_seen, optimal_greedy, spills_two))
    print("PASS")


if __name__ == "__main__":
    main()
