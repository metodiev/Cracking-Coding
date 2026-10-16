"""Task 845 - Implement approximate personalized PageRank with bookmark coloring.  (Tier: Master, CTCI Beyond — System Design and Scalability)

Berkhin's bookmark-colouring algorithm (BCA) for *approximate* personalized PageRank: instead
of solving (I - (1-a)P^T) p = a e_s, mass is pushed from a single bookmark along the graph and
each vertex keeps a residual ("colour") that is only spent when it is large enough to matter.
Everything is deterministic - no random walk, no wall clock - and the approximation is verified
against an exact dense solve of the same linear system.

Algorithm, as implemented (`bookmark_coloring`):
  * residuals start as r[s] = 1 and approximate scores as p = 0;
  * repeatedly pick the vertex with the largest residual, tie-broken by id, and push it:
    p[v] += alpha * r[v], then (1 - alpha) * r[v] is spread uniformly over the out-neighbours
    and r[v] is cleared.  A dangling vertex (no out-edges) absorbs its part, which is exactly
    what the reference system does when P has a zero row;
  * stopping: the maximum residual drops below `epsilon` (the prefix-epsilon rule), with a
    separate variant that pushes while r[v] > epsilon * outdeg(v) (the alpha-residual rule).

The exact identity that makes the approximation auditable is linearity of PageRank:
    exact_ppr(s -> v) == p[v] + sum_u r[u] * exact_ppr(u -> v)   for every vertex v,
so the residual, weighted by the exact PPR of every residual holder, reconstructs the exact
score.  The checks below verify that identity numerically, verify that p is a *lower bound* of
the exact score, and verify that the error is bounded by the total residual mass - which is
what "epsilon-approximate" means here.

Simplified versus production: the graph is a fixed 8-vertex toy, the exact reference is dense
Gaussian elimination, and the "randomized" variant of BCA is replaced by the deterministic
largest-residual bookmark selection (so runs are reproducible).  Locality is measured rather
than assumed: the number of vertices that ever receive mass, and the number of pushes, are
reported and bounded.

Verified: the sparse-touch work bound (pushes grow like 1/epsilon while the error and the
error bound shrink like epsilon), the residual invariant after every push, exactness in the
limit epsilon -> 0, the lower-bound and correction identities, top-k ranking agreement with the
exact scores, dangling and unreachable vertices, alpha = 1, a single vertex, a two-vertex graph,
and determinism.
"""

GRAPH = {
    0: [1, 2],
    1: [2, 3],
    2: [0, 4],
    3: [4, 5],
    4: [5, 6],
    5: [6, 7],
    6: [7, 0],
    7: [0],
}


def solve(matrix, rhs):
    """Dense Gaussian elimination with partial pivoting (toy sizes only)."""
    size = len(rhs)
    work = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-14:
            raise ValueError("singular system")
        work[column], work[pivot] = work[pivot], work[column]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column] / work[column][column]
            if factor:
                for index in range(column, size + 1):
                    work[row][index] -= factor * work[column][index]
    return [work[index][size] / work[index][index] for index in range(size)]


def exact_ppr(graph, source, alpha):
    """Exact personalized PageRank: the solution of (I - (1-a)P^T) p = a e_source."""
    nodes = sorted(graph)
    position = {node: index for index, node in enumerate(nodes)}
    size = len(nodes)
    matrix = [[0.0] * size for _ in range(size)]
    for index in range(size):
        matrix[index][index] = 1.0
    for node in nodes:
        targets = graph[node]
        if not targets:
            continue
        share = (1.0 - alpha) / len(targets)
        for target in targets:
            matrix[position[target]][position[node]] -= share
    rhs = [0.0] * size
    rhs[position[source]] = alpha
    return dict(zip(nodes, solve(matrix, rhs)))


class ApproximatePPR:
    """The result of one bookmark-colouring run, with its residual vector kept for auditing."""

    def __init__(self, graph, source, alpha):
        self.graph = {node: list(targets) for node, targets in graph.items()}
        self.source = source
        self.alpha = alpha
        self.nodes = sorted(self.graph)
        self.approx = {node: 0.0 for node in self.nodes}
        self.residual = {node: 0.0 for node in self.nodes}
        self.residual[source] = 1.0
        self.pushes = 0
        self.touched = {source}
        self.trace = []

    def _push(self, vertex):
        """Spend a vertex's residual: bank the alpha share, spread the rest to its neighbours."""
        mass = self.residual[vertex]
        self.approx[vertex] += self.alpha * mass
        self.residual[vertex] = 0.0
        targets = self.graph[vertex]
        if targets:
            share = (1.0 - self.alpha) * mass / len(targets)
            for target in targets:
                self.residual[target] += share
                self.touched.add(target)
        self.pushes += 1

    def _bookmark(self):
        """Deterministic bookmark selection: the largest residual, ties broken by vertex id."""
        best, best_value = None, 0.0
        for node in self.nodes:
            if self.residual[node] > best_value:
                best, best_value = node, self.residual[node]
        return best

    def run(self, epsilon, rule="max"):
        """Push bookmarks until the residual is small enough; returns self."""
        while True:
            vertex = self._bookmark()
            if vertex is None:
                break
            if rule == "max":
                if self.residual[vertex] <= epsilon:
                    break
            elif rule == "alpha-residual":
                degree = max(1, len(self.graph[vertex]))
                if self.residual[vertex] <= epsilon * degree:
                    break
            else:
                raise ValueError("unknown stopping rule %r" % (rule,))
            self._push(vertex)
            self.trace.append((vertex, self.approx[vertex], self.residual_mass()))
        return self

    def run_to_completion(self, tolerance=1e-18):
        """Push until the residual is numerically gone: the approximation becomes exact.

        The tolerance is not zero because the residual only decays geometrically, so it would
        otherwise crawl through denormals forever; at 1e-18 every score is correct to far more
        digits than the checks below require.
        """
        while True:
            vertex = self._bookmark()
            if vertex is None or self.residual[vertex] <= tolerance:
                break
            self._push(vertex)
        return self

    def residual_mass(self):
        return sum(self.residual.values())

    def exact(self):
        return exact_ppr(self.graph, self.source, self.alpha)

    def max_residual(self):
        return max(self.residual.values())

    def correction(self, exact_scores=None):
        """The residual's exact contribution to every score: sum_u r[u] * exact_ppr(u -> v)."""
        scores = exact_scores or self.exact()
        total = {node: 0.0 for node in self.nodes}
        for holder, mass in self.residual.items():
            if mass <= 0.0:
                continue
            contribution = exact_ppr(self.graph, holder, self.alpha)
            for node in self.nodes:
                total[node] += mass * contribution[node]
        return total


def top_k(scores, k):
    return [node for node, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    alpha = 0.15
    source = 0

    # ---- 1. the exact reference ------------------------------------------
    exact = exact_ppr(GRAPH, source, alpha)
    check(abs(sum(exact.values()) - 1.0) < 1e-9,
          "exact PPR sums to one on a graph without dangling vertices")
    check(abs(exact[source] - max(exact.values())) < 1e-12,
          "the source has the largest exact score")
    check(all(value > 0 for value in exact.values()),
          "every vertex is reachable from the source here")
    for target in sorted(GRAPH):
        row = exact_ppr(GRAPH, target, alpha)
        check(abs(sum(row.values()) - 1.0) < 1e-9,
              "exact PPR from %d must also sum to one" % target)

    # ---- 2. the approximation against the exact solve ---------------------
    run = ApproximatePPR(GRAPH, source, alpha).run(1e-3)
    check(run.max_residual() <= 1e-3, "the run must satisfy its own stopping rule")
    correction = run.correction(exact)
    for node in sorted(GRAPH):
        identity = run.approx[node] + correction[node]
        check(abs(identity - exact[node]) < 1e-9,
              "the correction identity must hold at vertex %d (%.12f vs %.12f)"
              % (node, identity, exact[node]))
        check(run.approx[node] <= exact[node] + 1e-12,
              "the approximation is a lower bound at vertex %d" % node)
    error = max(abs(run.approx[node] - exact[node]) for node in GRAPH)
    check(error <= run.residual_mass() + 1e-12,
          "the error must be bounded by the residual mass (%.3g vs %.3g)"
          % (error, run.residual_mass()))
    check(error < 1e-2, "a 1e-3 residual budget must give a small error (%.3g)" % error)

    # ---- 3. the invariant after every single push -------------------------
    walk = ApproximatePPR(GRAPH, source, alpha)
    for _ in range(12):
        vertex = walk._bookmark()
        walk._push(vertex)
    fresh_correction = walk.correction(exact)
    for node in sorted(GRAPH):
        check(abs(walk.approx[node] + fresh_correction[node] - exact[node]) < 1e-9,
              "the invariant must survive every push (vertex %d)" % node)
    check(all(value >= 0.0 for value in walk.residual.values()),
          "residuals are never negative")
    check(abs(sum(walk.approx.values()) + walk.residual_mass() - 1.0) < 1e-9,
          "pushed mass plus residual mass must account for all of the source's unit mass")

    # ---- 4. exactness in the limit and the lower-bound structure ----------
    full = ApproximatePPR(GRAPH, source, alpha).run_to_completion()
    for node in sorted(GRAPH):
        check(abs(full.approx[node] - exact[node]) < 1e-12,
              "pushing every bookmark must reproduce the exact score at %d" % node)
    check(full.max_residual() <= 1e-18, "no meaningful residual may be left when the run completes")

    # ---- 5. epsilon drives work and error ---------------------------------
    work, errors = {}, {}
    for epsilon in (1e-1, 1e-2, 1e-3, 1e-4):
        small = ApproximatePPR(GRAPH, source, alpha).run(epsilon)
        error = max(abs(small.approx[node] - exact[node]) for node in GRAPH)
        work[epsilon] = small.pushes
        errors[epsilon] = error
        check(error <= 10.0 * epsilon, "error must shrink with epsilon (%.3g at %.0e)"
              % (error, epsilon))
        check(small.pushes > 0, "a push must happen for every epsilon")
    for coarse, fine in ((1e-1, 1e-2), (1e-2, 1e-3), (1e-3, 1e-4)):
        check(errors[fine] < errors[coarse],
              "halving epsilon must reduce the error (%0.e vs %0.e)" % (coarse, fine))
        check(work[fine] > work[coarse],
              "a tighter epsilon must cost more pushes (%d -> %d)"
              % (work[coarse], work[fine]))
    check(work[1e-4] <= 2000, "the work must stay bounded by about 1/epsilon (%d pushes)"
          % work[1e-4])
    check(work[1e-4] <= 40 * work[1e-1], "the cost grows polynomially, not explosively")

    # locality: only a small part of the graph is ever coloured
    check(len(run.touched) <= len(GRAPH), "the colouring stays inside the graph")
    sparse = ApproximatePPR(GRAPH, source, alpha).run(1e-1)
    check(len(sparse.touched) <= len(run.touched),
          "a looser budget touches no more vertices than a tighter one")

    # the alpha-residual stopping rule is a different budget but the same invariant
    alternative = ApproximatePPR(GRAPH, source, alpha).run(1e-3, rule="alpha-residual")
    check(alternative.max_residual() <= 1e-3 * max(1, max(len(GRAPH[node])
                                                          for node in alternative.touched)),
          "the alpha-residual rule bounds the residual by epsilon * degree")
    alternative_correction = alternative.correction(exact)
    for node in sorted(GRAPH):
        check(abs(alternative.approx[node] + alternative_correction[node] - exact[node]) < 1e-9,
              "the invariant holds under the alpha-residual rule too (vertex %d)" % node)

    # ---- 6. rankings ------------------------------------------------------
    check(top_k(run.approx, 3) == top_k(exact, 3),
          "the approximate top three must match the exact ranking (%r vs %r)"
          % (top_k(run.approx, 3), top_k(exact, 3)))
    coarse = ApproximatePPR(GRAPH, source, alpha).run(1e-1)
    check(top_k(coarse.approx, 1) == top_k(exact, 1),
          "even a loose budget must find the source as the top result")

    # ---- 7. degenerate cases ---------------------------------------------
    single = exact_ppr({0: [0]}, 0, alpha)
    check(abs(single[0] - 1.0) < 1e-12, "a self-looping single vertex scores one")
    pair = exact_ppr({0: [1], 1: [0]}, 0, alpha)
    check(abs(pair[0] - alpha / (1.0 - (1.0 - alpha) ** 2)) < 1e-12,
          "the two-vertex cycle matches its closed form")
    check(pair[0] > pair[1], "the source beats its neighbour on a two-vertex cycle")
    check(abs(sum(pair.values()) - 1.0) < 1e-9, "the two-vertex cycle conserves mass")
    dangling = ApproximatePPR({0: [1], 1: []}, 0, alpha).run(1e-6)
    dangling_exact = exact_ppr({0: [1], 1: []}, 0, alpha)
    for node in (0, 1):
        check(abs(dangling.approx[node] - dangling_exact[node]) < 1e-6,
              "a dangling vertex must absorb its mass exactly as the reference does")
    check(sum(dangling_exact.values()) < 1.0,
          "a dangling vertex leaks mass, which the reference and the colouring agree on")
    disconnected = exact_ppr({0: [1], 1: [0], 2: [2], 3: [2]}, 0, alpha)
    check(disconnected[2] == 0.0 and disconnected[3] == 0.0,
          "vertices unreachable from the source score exactly zero")
    greedy = ApproximatePPR(GRAPH, source, 1.0).run(1e-6)
    unit = exact_ppr(GRAPH, source, 1.0)
    check(abs(greedy.approx[source] - unit[source]) < 1e-12 and
          sum(unit.values()) - unit[source] < 1e-12,
          "alpha = 1 leaves all the mass on the source")
    tiny = ApproximatePPR(GRAPH, source, alpha).run(10.0)
    check(tiny.pushes == 0 and sum(tiny.approx.values()) == 0.0,
          "an epsilon above one stops before the first push and stays a valid lower bound")

    # ---- 8. determinism ---------------------------------------------------
    left = ApproximatePPR(GRAPH, source, alpha).run(1e-4)
    right = ApproximatePPR(GRAPH, source, alpha).run(1e-4)
    check(left.approx == right.approx and left.pushes == right.pushes,
          "the colouring must be reproducible")
    check(left.trace == right.trace, "the push trace must be reproducible")
    check(all(0 <= entry[0] < len(GRAPH) for entry in left.trace),
          "the trace records only real vertices")
    check(len(left.trace) == left.pushes, "the trace has one entry per push")

    print("exact: source %.6f, max %.6f, sum %.12f"
          % (exact[source], max(exact.values()), sum(exact.values())))
    print("approx: epsilon 1e-3 -> %d pushes, error %.3g, residual mass %.3g, %d vertices touched"
          % (run.pushes, error, run.residual_mass(), len(run.touched)))
    print("work: pushes %s, errors %s"
          % ({key: value for key, value in sorted(work.items())},
             {key: "%.2g" % value for key, value in sorted(errors.items())}))
    print("PASS")


if __name__ == "__main__":
    main()
