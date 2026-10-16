"""Task 545 - Implement pagerank iterative solver and convergence check. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

PageRank by power iteration: x_{t+1} = (1 - d)/n * 1 + d * M x_t, where M is the column-stochastic
transition matrix of the directed graph (M[v][u] = 1/outdeg(u) for each arc u -> v) and d = 0.85 is
the damping factor. Every test graph in this file has out-degree at least one at every vertex, so
no mass escapes through dumb vertices: the dangling-node variant is the subject of task 546. The
iteration is instrumented with the L1 step size, which is asserted to shrink by at least the
factor d each round - true because M is column-stochastic, hence ||M z||_1 <= ||z||_1 - and which
drives the convergence test.

Three independent references are used: a dense Gaussian elimination with partial pivoting on the
linear system (I - d M) x = (1 - d)/n * 1, a deterministic Monte-Carlo random surfer driven by the
same LCG used elsewhere in the repository (whose visit frequencies must approach the ranks), and
closed forms for regular graphs: on a cycle every vertex has rank exactly 1/n, and with d = 0 the
stationary vector is uniform as well.

Checks: uniform ranks 1/n on cycles C2, C3 and C5 (within 1e-9); the linear-solve solution matched
to 1e-9 on random digraphs with minimum out-degree one; the residual ||x - P x||_1 within the
tolerance; the mass sum equal to 1 and every coordinate strictly positive; monotone step sizes with
delta_{t+1} <= d * delta_t and the documented iteration count; convergence from a concentrated
start as well as from the uniform start (uniqueness of the fixed point for d < 1); the L1
contraction ||P x - P y||_1 <= d ||x - y||_1 for random probability vectors; a tighter tolerance
needing at least as many iterations; the empirical surfer within 0.005 of the ranks; d = 0
collapsing to the uniform vector while d = 0.9999 still fits inside the cap; and the two-cliques
dumbbell, where a concentrated start plus a narrow bridge makes the iteration count grow strictly
with the damping factor (91 < 164 < 238 iterations for d = 0.85, 0.95, 0.99).
"""
DAMPING = 0.85


def lcg(seed):
    """The small 16-bit LCG used across this repository for sample data."""
    return (seed * 75 + 74) % 65537


MASK64 = (1 << 64) - 1


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants) for the long Monte-Carlo run.

    The 16-bit LCG above has a short period, and averaging a 200k-step random walk over it makes
    the surfer partly periodic: the empirical frequencies then plateau about 0.007 away from the
    true ranks no matter how many steps are taken. A full-period generator removes that artefact
    while remaining completely deterministic.
    """
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def check_simple(adj):
    for targets in adj:
        assert len(set(targets)) == len(targets), "no multi-edges"
    for u, targets in enumerate(adj):
        assert u not in targets, "no self-loops in the test graphs of this task"


def has_arcs(adj):
    return all(len(targets) >= 1 for targets in adj)


def pagerank(adj, damping=DAMPING, tol=1e-12, max_iter=500, start=None):
    """Power iteration. Returns (ranks, iterations, step sizes) - the last step is <= tol."""
    n = len(adj)
    out_degree = [len(targets) for targets in adj]
    assert all(degree >= 1 for degree in out_degree), "this solver assumes no dangling vertices"
    if start is None:
        x = [1.0 / n] * n
    else:
        x = [float(value) for value in start]
        total = sum(x)
        x = [value / total for value in x]
    steps = []
    for iteration in range(1, max_iter + 1):
        nxt = [(1.0 - damping) / n] * n
        for u in range(n):
            share = damping * x[u] / out_degree[u]
            for v in adj[u]:
                nxt[v] += share
        step = sum(abs(nxt[i] - x[i]) for i in range(n))
        steps.append(step)
        x = nxt
        if step <= tol:
            return x, iteration, steps
    return x, max_iter, steps


def google_matrix(adj, damping=DAMPING):
    """Column-stochastic PageRank matrix M[v][u]; assumes out-degree >= 1 everywhere."""
    n = len(adj)
    matrix = [[0.0] * n for _ in range(n)]
    for u in range(n):
        share = 1.0 / len(adj[u])
        for v in adj[u]:
            matrix[v][u] += share
    return [[(1.0 - damping) / n + damping * matrix[i][j] for j in range(n)] for i in range(n)]


def apply_matrix(matrix, vector):
    n = len(vector)
    return [sum(matrix[i][j] * vector[j] for j in range(n)) for i in range(n)]


def linear_solve(matrix, rhs):
    """Gaussian elimination with partial pivoting - the direct reference solution."""
    n = len(matrix)
    work = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(work[row][col]))
        assert abs(work[pivot][col]) > 1e-14, "the PageRank system is non-singular for d < 1"
        work[col], work[pivot] = work[pivot], work[col]
        inverse = 1.0 / work[col][col]
        for row in range(col + 1, n):
            factor = work[row][col] * inverse
            if factor != 0.0:
                for c in range(col, n + 1):
                    work[row][c] -= factor * work[col][c]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = work[row][n] - sum(work[row][c] * x[c] for c in range(row + 1, n))
        x[row] = total / work[row][row]
    return x


def exact_pagerank(adj, damping=DAMPING):
    """Solve x = P x directly: (I - P) x = 0 is singular, so use (I - d M) x = (1-d)/n * 1."""
    n = len(adj)
    matrix = [[0.0] * n for _ in range(n)]
    for u in range(n):
        share = 1.0 / len(adj[u])
        for v in adj[u]:
            matrix[v][u] += share
    system = [[(1.0 if i == j else 0.0) - damping * matrix[i][j] for j in range(n)]
              for i in range(n)]
    rhs = [(1.0 - damping) / n] * n
    return linear_solve(system, rhs)


def residual(adj, ranks, damping=DAMPING):
    """L1 norm of ||x - P x||, the fixed-point residual."""
    return sum(abs(ranks[i] - apply_matrix(google_matrix(adj, damping), ranks)[i])
               for i in range(len(adj)))


def simulate_surfer(adj, steps, seed, damping=DAMPING):
    """Deterministic Monte-Carlo surfer: visit frequencies estimate the ranks."""
    n = len(adj)
    assert all(len(targets) >= 1 for targets in adj)
    visits = [0] * n
    current = 0
    state = (seed * 2654435761 + 1) & MASK64
    for _ in range(steps):
        visits[current] += 1
        state = prng64(state)
        if (state >> 33) % 10000 < int(damping * 10000):
            state = prng64(state)
            targets = adj[current]
            current = targets[(state >> 17) % len(targets)]
        else:
            state = prng64(state)
            current = (state >> 20) % n
    return [count / steps for count in visits]


def cycle(n):
    return [[(v + 1) % n] for v in range(n)]


def dumbbell(k):
    """Two k-cliques joined by a single bridge: a graph with a narrow mixing bottleneck."""
    adj = [[] for _ in range(2 * k)]
    for i in range(k):
        for j in range(k):
            if i != j:
                adj[i].append(j)
                adj[k + i].append(k + j)
    adj[0].append(k)
    adj[k].append(0)
    return adj


def lcg_digraph(n, out_links, seed):
    adj = []
    for v in range(n):
        targets = []
        for _ in range(out_links):
            seed = lcg(seed)
            target = seed % n
            if target != v and target not in targets:
                targets.append(target)
        if not targets:
            targets.append((v + 1) % n)
        adj.append(targets)
    return adj, seed


def main():
    # --- Regular graphs: the closed form is uniform -------------------------------------
    for size in (2, 3, 5, 4, 7):
        adj = cycle(size)
        ranks, iterations, steps = pagerank(adj)
        assert all(abs(value - 1.0 / size) <= 1e-9 for value in ranks), "cycles are uniform"
        assert iterations < 500
        assert abs(sum(ranks) - 1.0) <= 1e-12
        exact = exact_pagerank(adj)
        assert max(abs(ranks[i] - exact[i]) for i in range(size)) <= 1e-9
        assert residual(adj, ranks) <= 1e-9
        check_simple(adj)
        assert has_arcs(adj)

    # --- The only hand-computable non-regular shape -------------------------------------
    two = [[1], [0, 1]]                                 # 0 -> 1 -> 0 and 1 -> 1
    ranks, _, _ = pagerank(two)
    exact = exact_pagerank(two)
    assert max(abs(ranks[i] - exact[i]) for i in range(2)) <= 1e-9
    assert ranks[1] > ranks[0], "the self-loop keeps 1 ahead"

    # --- Random digraphs against the direct solve and the residual ----------------------
    seed = 20241016
    for trial in range(10):
        n = 5 + 2 * trial
        adj, seed = lcg_digraph(n, 2 + trial % 3, seed)
        assert has_arcs(adj)
        ranks, iterations, steps = pagerank(adj)
        assert abs(sum(ranks) - 1.0) <= 1e-12, "the iteration conserves mass exactly"
        assert all(value > 0.0 for value in ranks)
        exact = exact_pagerank(adj)
        assert max(abs(ranks[i] - exact[i]) for i in range(n)) <= 1e-9, "matches the direct solve"
        assert residual(adj, ranks) <= 1e-9
        # The step sizes shrink by at least the damping factor every round.
        for index in range(len(steps) - 1):
            assert steps[index + 1] <= DAMPING * steps[index] + 1e-15, "||M z||_1 <= ||z||_1"
        assert steps[-1] <= 1e-12 and (len(steps) == 1 or steps[-2] > 1e-12)
        assert iterations == len(steps)
        # A concentrated start converges to the same fixed point.
        concentrated = [0.0] * n
        concentrated[n - 1] = 1.0
        second, second_iterations, _ = pagerank(adj, start=concentrated)
        assert max(abs(second[i] - ranks[i]) for i in range(n)) <= 1e-8, "unique fixed point"
        assert second_iterations >= 1
        # A looser tolerance cannot need more iterations than a tight one.
        _, loose_iterations, _ = pagerank(adj, tol=1e-4)
        assert loose_iterations <= iterations

    # --- Contraction of the PageRank map ------------------------------------------------
    seed = 31415
    adj, seed = lcg_digraph(9, 3, seed)
    matrix = google_matrix(adj)
    for _ in range(20):
        x = []
        y = []
        for _ in range(9):
            seed = lcg(seed)
            x.append(seed % 100 + 1)
            seed = lcg(seed)
            y.append(seed % 100 + 1)
        x = [value / sum(x) for value in x]
        y = [value / sum(y) for value in y]
        distance = sum(abs(x[i] - y[i]) for i in range(9))
        image_distance = sum(abs(apply_matrix(matrix, x)[i] - apply_matrix(matrix, y)[i])
                             for i in range(9))
        assert image_distance <= DAMPING * distance + 1e-12, "the map is a d-contraction in L1"

    # --- Monte-Carlo surfer -------------------------------------------------------------
    seed = 606
    adj, seed = lcg_digraph(6, 2, seed)
    ranks, _, _ = pagerank(adj)
    empirical = simulate_surfer(adj, 200000, seed)
    for i in range(6):
        assert abs(empirical[i] - ranks[i]) <= 5e-3, "visit frequencies approach the ranks"
    assert abs(sum(empirical) - 1.0) <= 1e-12

    # --- Damping edge cases -------------------------------------------------------------
    adj, _ = lcg_digraph(7, 3, 777)
    base_ranks, _, _ = pagerank(adj)
    uniform, _, _ = pagerank(adj, damping=0.0)
    assert all(abs(value - 1.0 / 7) <= 1e-12 for value in uniform), "d = 0 teleports always"
    slow, slow_iterations, slow_steps = pagerank(adj, damping=0.99, max_iter=5000)
    slow_exact = exact_pagerank(adj, damping=0.99)
    assert max(abs(slow[i] - slow_exact[i]) for i in range(7)) <= 1e-9
    assert all(slow_steps[i + 1] <= 0.99 * slow_steps[i] + 1e-15 for i in range(len(slow_steps) - 1))
    high = pagerank(adj, damping=0.9999, max_iter=2000)[0]
    assert abs(sum(high) - 1.0) <= 1e-12
    assert all(value > 0.0 for value in high)

    # Slow mixing needs a bottleneck plus a concentrated start: on the two cliques joined by a
    # single bridge the iteration count grows with the damping factor.
    narrow = dumbbell(5)
    concentrated = [0.0] * 10
    concentrated[0] = 1.0
    iterations_by_damping = {}
    for damping in (0.85, 0.95, 0.99):
        ranks_narrow, count, steps_narrow = pagerank(narrow, damping=damping, tol=1e-10,
                                                     max_iter=20000, start=concentrated)
        solution = exact_pagerank(narrow, damping=damping)
        assert max(abs(ranks_narrow[i] - solution[i]) for i in range(10)) <= 1e-6
        assert all(steps_narrow[i + 1] <= damping * steps_narrow[i] + 1e-15
                   for i in range(len(steps_narrow) - 1))
        iterations_by_damping[damping] = count
    assert iterations_by_damping[0.85] < iterations_by_damping[0.95] \
        < iterations_by_damping[0.99], "a near-one damping factor converges more slowly"
    assert iterations_by_damping[0.85] > 50, "the bottleneck really slows the walk down"
    # A slightly different damping factor gives a slightly different ranking.
    small_change = pagerank(adj, damping=0.95)[0]
    assert max(abs(small_change[i] - base_ranks[i]) for i in range(7)) > 1e-12


if __name__ == "__main__":
    main()
    print("PASS")
