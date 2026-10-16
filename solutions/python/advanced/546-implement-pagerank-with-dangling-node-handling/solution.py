"""Task 546 - Implement pagerank with dangling node handling. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

PageRank when the graph contains dangling vertices (out-degree zero). The random surfer standing on
a dangling vertex cannot follow a link, so by convention its whole share is teleported uniformly to
all n vertices: the iteration becomes
x_{t+1} = (1 - d)/n * 1 + d * (sum over dangling v of x_v)/n * 1 + d * M x_t, with M[v][u] =
1/outdeg(u) only for non-dangling u. Two other formulations of exactly the same semantics are
implemented and cross-checked: completing the graph by adding one arc from every dangling vertex to
*every* vertex (out-degree n) and then solving with the plain solver of task 545, and a dense
Gaussian elimination of (I - d M) x = (1 - d)/n * 1 with M's dangling columns uniform. The
hand-checked instance 0 -> 1 with 1 dangling has the exact solution (20/57, 37/57): node 1 collects
all of node 0's mass, then spreads it over everything including itself.

Checks: the exact 20/57 + 37/57 split within 1e-9, with the dangling vertex outranking its only
predecessor; the handled iteration matching the completed-graph solver to 1e-12 and the linear
solve to 1e-9 on random digraphs with randomly chosen dangling sets; mass conservation (sum exactly
1) and strict positivity; the all-dangling graph and the single-vertex graph returning the uniform
vector immediately (zero step size in the first round); self-loops making a vertex non-dangling;
the *naive* variant that simply drops the dangling mass quantified rather than asserted away - its
total mass is exactly 1 - d * (dangling rank) and, in the hand-checked example, node 0 is stuck at
the teleport value (1 - d)/n = 0.075 against the correct 0.351; a deterministic Monte-Carlo surfer
that teleports instead of walking when it steps on a dangling vertex, matching the ranks to 5e-3;
and determinism plus the monotone step sizes that the d-contraction guarantees.
"""
MASK64 = (1 << 64) - 1
DAMPING = 0.85


def lcg(seed):
    return (seed * 75 + 74) % 65537


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants), used for the long Monte-Carlo run."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def dangling_nodes(adj):
    return [v for v in range(len(adj)) if len(adj[v]) == 0]


def pagerank_dangling(adj, damping=DAMPING, tol=1e-12, max_iter=500):
    """Power iteration with uniform redistribution of the dangling mass."""
    n = len(adj)
    out_degree = [len(targets) for targets in adj]
    dangling = [v for v in range(n) if out_degree[v] == 0]
    x = [1.0 / n] * n
    steps = []
    for iteration in range(1, max_iter + 1):
        nxt = [(1.0 - damping) / n] * n
        if dangling:
            share = damping * sum(x[v] for v in dangling) / n
            for i in range(n):
                nxt[i] += share
        for u in range(n):
            if out_degree[u] == 0:
                continue
            contribution = damping * x[u] / out_degree[u]
            for v in adj[u]:
                nxt[v] += contribution
        step = sum(abs(nxt[i] - x[i]) for i in range(n))
        steps.append(step)
        x = nxt
        if step <= tol:
            return x, iteration, steps
    return x, max_iter, steps


def pagerank_naive(adj, damping=DAMPING, tol=1e-12, max_iter=500):
    """The broken variant: the dangling mass is silently dropped instead of redistributed."""
    n = len(adj)
    out_degree = [len(targets) for targets in adj]
    x = [1.0 / n] * n
    for _ in range(max_iter):
        nxt = [(1.0 - damping) / n] * n
        for u in range(n):
            if out_degree[u] == 0:
                continue
            contribution = damping * x[u] / out_degree[u]
            for v in adj[u]:
                nxt[v] += contribution
        step = sum(abs(nxt[i] - x[i]) for i in range(n))
        x = nxt
        if step <= tol:
            break
    return x


def complete_dangling(adj):
    """Add one arc from every dangling vertex to every vertex - the equivalent dense graph."""
    n = len(adj)
    completed = []
    for u in range(n):
        if len(adj[u]) == 0:
            completed.append(list(range(n)))
        else:
            completed.append(list(adj[u]))
    return completed


def pagerank_plain(adj, damping=DAMPING, tol=1e-12, max_iter=500):
    """Task 545's solver, duplicated locally so this file runs standalone."""
    n = len(adj)
    assert all(len(targets) >= 1 for targets in adj), "no dangling vertices here"
    out_degree = [len(targets) for targets in adj]
    x = [1.0 / n] * n
    steps = []
    for iteration in range(1, max_iter + 1):
        nxt = [(1.0 - damping) / n] * n
        for u in range(n):
            contribution = damping * x[u] / out_degree[u]
            for v in adj[u]:
                nxt[v] += contribution
        step = sum(abs(nxt[i] - x[i]) for i in range(n))
        steps.append(step)
        x = nxt
        if step <= tol:
            return x, iteration, steps
    return x, max_iter, steps


def transition_matrix(adj, damping=DAMPING):
    n = len(adj)
    matrix = [[0.0] * n for _ in range(n)]
    for u in range(n):
        if not adj[u]:
            for v in range(n):
                matrix[v][u] += 1.0 / n
        else:
            share = 1.0 / len(adj[u])
            for v in adj[u]:
                matrix[v][u] += share
    return matrix


def exact_ranks(adj, damping=DAMPING):
    """Dense solve of (I - d M) x = (1 - d)/n * 1 with M's dangling columns uniform."""
    n = len(adj)
    base = transition_matrix(adj, damping)
    system = [[(1.0 if i == j else 0.0) - damping * base[i][j] for j in range(n)]
              for i in range(n)]
    rhs = [(1.0 - damping) / n] * n
    work = [list(system[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(work[row][col]))
        assert abs(work[pivot][col]) > 1e-14
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


def simulate_surfer(adj, steps, seed, damping=DAMPING):
    """The surfer teleports instead of walking when it lands on a dangling vertex."""
    n = len(adj)
    visits = [0] * n
    current = 0
    state = (seed * 2654435761 + 1) & MASK64
    for _ in range(steps):
        visits[current] += 1
        state = prng64(state)
        targets = adj[current]
        if targets and (state >> 33) % 10000 < int(damping * 10000):
            state = prng64(state)
            current = targets[(state >> 17) % len(targets)]
        else:
            state = prng64(state)
            current = (state >> 20) % n
    return [count / steps for count in visits]


def lcg_digraph_with_dangling(n, arcs, dangling_count, seed):
    """Random digraph; the last `dangling_count` vertices are left without outgoing arcs."""
    live = n - dangling_count
    adj = [[] for _ in range(n)]
    for u in range(live):
        targets = []
        for _ in range(arcs):
            seed = lcg(seed)
            target = seed % n
            if target != u and target not in targets:
                targets.append(target)
        if not targets:
            targets.append((u + 1) % n)
        adj[u] = targets
    return adj, seed


def main():
    # --- Hand-checked two-vertex instance ----------------------------------------------
    hand = [[1], []]
    ranks, iterations, steps = pagerank_dangling(hand)
    assert abs(ranks[0] - 20.0 / 57.0) <= 1e-9, "solving x = P x by hand gives 20/57"
    assert abs(ranks[1] - 37.0 / 57.0) <= 1e-9, "and 37/57 for the dangling vertex"
    assert abs(sum(ranks) - 1.0) <= 1e-12
    assert ranks[1] > ranks[0], "the dangling vertex outranks its only predecessor"
    assert dangling_nodes(hand) == [1]
    assert abs(steps[-1]) <= 1e-12 and iterations >= 1
    completed = complete_dangling(hand)
    assert completed == [[1], [0, 1]]
    same, _, _ = pagerank_plain(completed)
    assert max(abs(same[i] - ranks[i]) for i in range(2)) <= 1e-12, \
        "completing the graph with uniform arcs is the same model"
    exact = exact_ranks(hand)
    assert max(abs(exact[i] - ranks[i]) for i in range(2)) <= 1e-9

    # The naive variant is quantified, not hand-waved: node 0 is stranded at the teleport value
    # (1 - d)/n = 0.075, node 1 merely collects 0.075 + d * 0.075, and the vector sums to
    # (1 - d) + d * (non-dangling mass) rather than to one.
    naive = pagerank_naive(hand)
    assert abs(naive[0] - 0.075) <= 1e-9, "with no in-arcs node 0 only ever sees teleports"
    assert abs(naive[1] - (0.075 + DAMPING * 0.075)) <= 1e-9
    preserved = (1.0 - DAMPING) + DAMPING * sum(naive[v] for v in range(2) if v not in (1,))
    assert abs(sum(naive) - preserved) <= 1e-9, "the naive fixed point keeps only that much mass"
    assert sum(naive) < 0.25
    assert naive[0] < ranks[0] - 0.2, "dropping the dangling mass badly understates node 0"

    # --- Degenerate graphs --------------------------------------------------------------
    all_dangling = [[], [], []]
    uniform, iterations_all, steps_all = pagerank_dangling(all_dangling)
    assert uniform == [1.0 / 3] * 3, "with no arcs at all the ranks are uniform"
    assert steps_all[0] <= 1e-15, "the uniform vector is already a fixed point"
    assert iterations_all == 1
    assert pagerank_dangling([[]])[0] == [1.0]
    assert pagerank_dangling([[]], damping=0.0)[0] == [1.0]
    selfloop = [[0], []]
    assert dangling_nodes(selfloop) == [1], "a self-loop is an out-arc, so vertex 0 is fine"
    loop_ranks, _, _ = pagerank_dangling(selfloop)
    loop_same, _, _ = pagerank_plain([[0], [0, 1]])
    assert max(abs(loop_ranks[i] - loop_same[i]) for i in range(2)) <= 1e-12
    chain = [[1], [2], []]
    assert dangling_nodes(chain) == [2]
    chain_ranks, _, _ = pagerank_dangling(chain)
    assert chain_ranks[2] > chain_ranks[1] > chain_ranks[0], "mass flows down the chain"
    assert abs(sum(chain_ranks) - 1.0) <= 1e-12

    # --- Random digraphs with dangling vertices ----------------------------------------
    seed = 20241016
    for trial in range(12):
        n = 5 + trial % 6
        dangling_count = 1 + trial % max(1, n - 2)
        adj, seed = lcg_digraph_with_dangling(n, 2, dangling_count, seed)
        assert len(dangling_nodes(adj)) == dangling_count
        ranks, iterations, steps = pagerank_dangling(adj)
        assert abs(sum(ranks) - 1.0) <= 1e-12, "mass is conserved when dangles are handled"
        assert all(value > 0.0 for value in ranks)
        assert all(steps[i + 1] <= DAMPING * steps[i] + 1e-15 for i in range(len(steps) - 1))
        completed_ranks, _, _ = pagerank_plain(complete_dangling(adj))
        assert max(abs(completed_ranks[i] - ranks[i]) for i in range(n)) <= 1e-12, \
            "the completed-graph solver is the same algorithm"
        exact = exact_ranks(adj)
        assert max(abs(exact[i] - ranks[i]) for i in range(n)) <= 1e-9, "matches the direct solve"
        naive = pagerank_naive(adj)
        live_mass = sum(naive[v] for v in range(n) if v not in dangling_nodes(adj))
        assert abs(sum(naive) - ((1.0 - DAMPING) + DAMPING * live_mass)) <= 1e-9, \
            "the naive fixed point keeps (1 - d) + d * (non-dangling mass)"
        repeat, _, _ = pagerank_dangling(adj)
        assert repeat == ranks, "deterministic"

    # --- Monte-Carlo surfer with dangling vertices -------------------------------------
    seed = 8080
    adj, seed = lcg_digraph_with_dangling(6, 2, 2, seed)
    ranks, _, _ = pagerank_dangling(adj)
    empirical = simulate_surfer(adj, 200000, seed)
    assert max(abs(empirical[i] - ranks[i]) for i in range(6)) <= 5e-3, \
        "teleporting from dangling vertices reproduces the ranks"
    assert abs(sum(empirical) - 1.0) <= 1e-12
    all_dangling_empirical = simulate_surfer([[], [], [], []], 100000, 91)
    assert max(abs(all_dangling_empirical[i] - 0.25) for i in range(4)) <= 5e-3, \
        "with no arcs the surfer is uniform"
    assert simulate_surfer([[]], 1000, 5) == [1.0]

    # --- Damping extremes ---------------------------------------------------------------
    adj, _ = lcg_digraph_with_dangling(7, 3, 3, 4242)
    zero_damping, _, _ = pagerank_dangling(adj, damping=0.0)
    assert all(abs(value - 1.0 / 7) <= 1e-12 for value in zero_damping), "d = 0 is uniform"
    nearly_one, _, _ = pagerank_dangling(adj, damping=0.999, tol=1e-10, max_iter=20000)
    solution = exact_ranks(adj, damping=0.999)
    assert max(abs(nearly_one[i] - solution[i]) for i in range(7)) <= 1e-8
    assert abs(sum(nearly_one) - 1.0) <= 1e-12
    assert dangling_nodes(adj) == [4, 5, 6]


if __name__ == "__main__":
    main()
    print("PASS")
