"""Task 719 - Implement parameter server simulation for distributed ML training.  (Tier: Expert, Ch 9 — System Design and Scalability)

A parameter server (PS) holds the model; workers pull the current parameters, compute a
gradient on their shard and push it back; the server applies it and bumps a version counter.
The design axis is consistency:

* synchronous SGD - every worker uses the same barrier version, the round ends at the slowest
  worker (a straggler stalls everyone) and the server applies the *sum* of the shard
  gradients.  Each shard gradient is scaled by 1/n over the whole dataset, so the sum is
  exactly the full-batch gradient and sync SGD reproduces centralised gradient descent;
* asynchronous SGD with bounded staleness - a gradient is computed on the version its worker
  pulled, so it may arrive s versions late; updates staler than the bound are dropped and
  accepted ones are damped by lr/(1 + staleness), the usual guard against overshoot.

The simulation is event driven with deterministic per-worker compute times (worker i pulls at
its previous push time, so its first push lands at i + latency) and a (time, worker)
tie-break, so every counter and the whole update log are reproducible.  The workload is a
deterministic linear-regression problem whose optimum comes from the normal equations solved
by our own Gaussian elimination with partial pivoting - an oracle sharing no code with the
trainers.

Simplified versus a production PS: no network model beyond a flat per-message byte count, no
quantization, no server-side sharding of the parameter vector, no straggler mitigation beyond
the staleness bound, one gradient per pull rather than local mini-batches.  Verified: sync SGD
equals centralised full-batch SGD to machine precision, single-worker async equals plain
gradient descent, the staleness bound and the lr/(1+staleness) damping hold exactly, every
logged gradient really is the shard gradient at the parameters its worker pulled, the log
replays to the same parameters, and async finishes far more updates per unit of wall clock.
"""
import heapq
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    """A rejected invalid configuration is a pass."""
    try: call()
    except ValueError: return
    raise AssertionError(message)


def splitmix64(state):
    """Deterministic 64-bit PRNG step."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state, low, high):
    state, r = splitmix64(state)
    return state, low + (high - low) * ((r >> 11) / float(1 << 53))


def solve_linear_system(a, b):
    """Gaussian elimination with partial pivoting; raises on a singular system."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


class RegressionProblem:
    """Least squares on a deterministic dataset: L(w) = ||Xw - y||^2 / (2 n)."""

    def __init__(self, samples, dim, seed):
        self.dim = dim
        self.samples = samples
        self.truth, self.rows, self.target = [], [], []
        state = seed & MASK64
        for _ in range(dim):
            state, v = uniform(state, -0.5, 0.5)
            self.truth.append(v)
        for _ in range(samples):
            row = []
            for _ in range(dim):
                state, v = uniform(state, -1.0, 1.0)
                row.append(v)
            self.rows.append(row)
            state, noise = uniform(state, -0.05, 0.05)
            self.target.append(sum(row[i] * self.truth[i] for i in range(dim)) + noise)

    def gradient(self, indices, w):
        """(1/n) X_s^T (X_s w - y_s): shard gradient scaled by the *global* sample count."""
        grad = [0.0] * self.dim
        for i in indices:
            row = self.rows[i]
            residual = sum(row[j] * w[j] for j in range(self.dim)) - self.target[i]
            for j in range(self.dim):
                grad[j] += row[j] * residual
        return [g / self.samples for g in grad]

    def loss(self, w):
        total = 0.0
        for i, row in enumerate(self.rows):
            residual = sum(row[j] * w[j] for j in range(self.dim)) - self.target[i]
            total += residual * residual
        return total / (2.0 * self.samples)

    def optimum(self):
        """Solve the normal equations X^T X w = X^T y with an independent routine."""
        normal = [[0.0] * self.dim for _ in range(self.dim)]
        rhs = [0.0] * self.dim
        for i, row in enumerate(self.rows):
            for a in range(self.dim):
                rhs[a] += row[a] * self.target[i]
                for b in range(self.dim):
                    normal[a][b] += row[a] * row[b]
        return solve_linear_system(normal, rhs)


def make_shards(samples, workers):
    """Contiguous partition of the sample indices; every worker needs at least one row."""
    if samples < workers:
        raise ValueError("every worker needs at least one sample")
    size, remainder = divmod(samples, workers)
    shards, start = [], 0
    for w in range(workers):
        end = start + size + (1 if w < remainder else 0)
        shards.append(list(range(start, end)))
        start = end
    return shards

class ParameterServer:
    """Version-countered parameter store that applies (possibly stale) gradient updates."""

    def __init__(self, problem, latencies, mode, learning_rate, staleness_bound):
        if mode not in ("sync", "async"):
            raise ValueError("mode must be sync or async")
        if learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if staleness_bound < 0:
            raise ValueError("staleness bound must be non-negative")
        self.problem = problem
        self.shards = make_shards(problem.samples, len(latencies))
        self.latencies = list(latencies)
        self.mode = mode
        self.learning_rate = learning_rate
        self.staleness_bound = staleness_bound
        self.w = [0.0] * problem.dim
        self.version = self.applied = self.dropped = 0
        self.log = []                 # (worker, pulled_version, pulled_w, gradient, weight, staleness)
        self.max_staleness = 0        # deepest staleness among the accepted updates
        self.rejected_staleness = 0   # deepest staleness among the rejected updates
        self.staleness_histogram = {}  # accepted updates only
        self.vector_transfers = 0      # pulls plus pushes, dropped pushes included
        self.wall_time = 0

    def pull(self):
        self.vector_transfers += 1
        return self.version, list(self.w)

    def push(self, worker, pulled_version, pulled_w, gradient):
        """Apply or drop one worker update; returns True when it was applied."""
        self.vector_transfers += 1
        staleness = self.version - pulled_version
        if self.mode == "async" and staleness > self.staleness_bound:
            self.dropped += 1
            self.rejected_staleness = max(self.rejected_staleness, staleness)
            return False
        self.max_staleness = max(self.max_staleness, staleness)
        self.staleness_histogram[staleness] = self.staleness_histogram.get(staleness, 0) + 1
        weight = self.learning_rate if self.mode == "sync" else self.learning_rate / (1 + staleness)
        self.log.append((worker, pulled_version, pulled_w, gradient, weight, staleness))
        self._apply(gradient, weight)
        return True

    def apply_aggregate(self, gradient):
        """Sync round after the barrier: the summed shard gradient is perfectly fresh."""
        self.staleness_histogram[0] = self.staleness_histogram.get(0, 0) + 1
        self.log.append((None, self.version, None, gradient, self.learning_rate, 0))
        self._apply(gradient, self.learning_rate)

    def _apply(self, gradient, weight):
        for j in range(self.problem.dim):
            self.w[j] -= weight * gradient[j]
        self.version += 1
        self.applied += 1

    def communication_bytes(self):
        """Every transfer ships the whole parameter vector in full precision (8 bytes each)."""
        return self.vector_transfers * self.problem.dim * 8


def run_sync(server, rounds):
    """One round: all workers use the same barrier snapshot; the round ends at the slowest."""
    round_cost = max(server.latencies)
    dim = server.problem.dim
    for _ in range(rounds):
        w_pinned = list(server.w)                    # the barrier snapshot, shared by all
        total = [0.0] * dim
        for worker in range(len(server.shards)):
            for j, value in enumerate(server.problem.gradient(server.shards[worker], w_pinned)):
                total[j] += value                     # each pull+push of the shard gradient
        server.vector_transfers += 2 * len(server.shards)
        server.apply_aggregate(total)
        server.wall_time += round_cost
    return server


def run_async(server, horizon):
    """Worker i pulls, computes for its latency, pushes, then immediately pulls again."""
    pending = []
    for worker, latency in enumerate(server.latencies):
        pulled_version, w_snapshot = server.pull()
        heapq.heappush(pending, (float(worker) + latency, worker, latency, pulled_version, w_snapshot))
    while pending:
        now, worker, latency, pulled_version, w_snapshot = heapq.heappop(pending)
        if now > horizon:
            break
        server.wall_time = now
        grad = server.problem.gradient(server.shards[worker], w_snapshot)
        server.push(worker, pulled_version, w_snapshot, grad)
        if now + latency <= horizon:
            next_version, next_snapshot = server.pull()
            heapq.heappush(pending, (now + latency, worker, latency, next_version, next_snapshot))
    return server


def scheduled_pushes(latencies, horizon):
    """Independent count of the pushes the async loop must perform inside the horizon."""
    total = 0
    for worker, latency in enumerate(latencies):
        time = float(worker) + latency
        while time <= horizon:
            total += 1
            time += latency
    return total


def central_gd(problem, steps, learning_rate):
    """Reference: plain full-batch gradient descent, one step per round."""
    w = [0.0] * problem.dim
    losses = []
    for _ in range(steps):
        grad = problem.gradient(list(range(problem.samples)), w)
        w = [w[j] - learning_rate * grad[j] for j in range(problem.dim)]
        losses.append(problem.loss(w))
    return w, losses


def main():
    problem = RegressionProblem(240, 6, 0x7190)
    optimum = problem.optimum()
    best_loss = problem.loss(optimum)
    check(math.dist(problem.gradient(list(range(240)), optimum), [0.0] * 6) < 1e-9,
          "the normal-equation optimum must have a vanishing gradient")

    # 1. sharding: contiguous, exhaustive, disjoint; more workers than rows is an error
    #    rather than a silently empty shard.
    shards = make_shards(240, 4)
    covered = sorted(i for s in shards for i in s)
    check([len(s) for s in make_shards(242, 4)] == [61, 61, 60, 60] and covered == list(range(240)),
          "shards must be contiguous, disjoint and cover every sample")
    expect_value_error(lambda: make_shards(3, 4), "more workers than samples is invalid")

    # 2. the identity that makes sync SGD exact: shard gradients scaled by 1/n sum to the
    #    full-batch gradient.
    probe = [0.1 * i for i in range(6)]
    total = [0.0] * 6
    for s in shards:
        for j, value in enumerate(problem.gradient(s, probe)):
            total[j] += value
    check(math.dist(problem.gradient(list(range(240)), probe), total) < 1e-12,
          "sum of shard gradients must equal the full-batch gradient")

    # 3. synchronous SGD: one server update per round, a round as slow as the straggler, and
    #    a trajectory identical to centralised full-batch gradient descent.
    rounds, sync_lr, latencies = 100, 0.6, [3, 4, 5, 3]
    sync = run_sync(ParameterServer(problem, latencies, "sync", sync_lr, 0), rounds)
    ref_w, ref_losses = central_gd(problem, rounds, sync_lr)
    check(sync.applied == rounds and sync.dropped == 0
          and math.dist(sync.w, ref_w) < 1e-12,
          "one sync round applies one update and equals centralised full-batch GD")
    check(problem.loss(sync.w) - best_loss < 1e-12 and problem.loss(sync.w) < ref_losses[0] / 20.0
          and sync.wall_time == rounds * max(latencies),
          "sync SGD reaches the optimum and a round costs the straggler's time")
    check(sync.staleness_histogram == {0: rounds} and
          sync.communication_bytes() == 2 * rounds * 4 * 6 * 8,
          "sync pushes are perfectly fresh and cost one pull plus one push per worker")

    # 4. a single worker can never go stale, so async collapses onto plain gradient descent.
    solo = run_async(ParameterServer(problem, [3], "async", sync_lr, 3), horizon=3 * rounds)
    check(solo.applied == rounds and solo.dropped == 0 and solo.max_staleness == 0
          and math.dist(solo.w, ref_w) < 1e-12, "one worker: nothing stale, dropped or reordered")

    # 5. bounded-staleness async: the bound holds, the damping is lr/(1+s), the schedule is
    #    the one we predicted, and the update log replays to the same parameters.
    horizon, async_lr, bound = 600, 0.5, 2
    server = run_async(ParameterServer(problem, latencies, "async", async_lr, bound), horizon)
    pushes = scheduled_pushes(latencies, horizon)
    check(pushes == 667 and server.applied + server.dropped == pushes
          and server.rejected_staleness > bound and 0 < server.dropped < pushes,
          "the schedule matches the hand-computed 667 pushes and drops only the stale ones")
    check(server.max_staleness == bound and max(server.staleness_histogram) <= bound,
          "accepted updates never exceed the staleness bound")
    check(all(math.dist(problem.gradient(server.shards[w], pw), g) < 1e-15
              and abs(wt - async_lr / (1 + s)) < 1e-15 for w, _, pw, g, wt, s in server.log),
          "a logged update must use its own stale shard gradient and lr/(1+staleness)")
    replay = [sum(-weight * grad[j] for _, _, _, grad, weight, _ in server.log)
              for j in range(6)]
    check(math.dist(replay, server.w) < 1e-12,
          "replaying the update log must reproduce the final parameters")
    async_gap = problem.loss(server.w) - best_loss
    check(async_gap < 1e-4 and problem.loss(server.w) < problem.loss([0.0] * 6) / 100.0 and
          async_gap > 1e5 * (problem.loss(sync.w) - best_loss),
          "damped async converges near - but plateaus above - the exact sync fixed point")

    # 6. throughput and staleness control under the same simulated wall clock.
    budget = sync.wall_time
    fast = run_async(ParameterServer(problem, latencies, "async", async_lr, 3), horizon=budget)
    check(fast.applied + fast.dropped == scheduled_pushes(latencies, budget)
          and fast.applied > 3 * rounds
          and fast.communication_bytes() == 2 * (fast.applied + fast.dropped) * 6 * 8,
          "async applies >3x the sync updates for the same bytes per scheduled push")
    tight = run_async(ParameterServer(problem, latencies, "async", async_lr, 0), horizon=horizon)
    check(tight.dropped > server.dropped and tight.max_staleness == 0 and tight.applied > 0
          and tight.rejected_staleness > 0, "a zero bound admits only fresh updates")

    # 7. rejected configurations and determinism.
    expect_value_error(lambda: ParameterServer(problem, [3, 3], "asynchronous", 0.1, 1), "bad mode")
    expect_value_error(lambda: ParameterServer(problem, [3, 3], "sync", 0.0, 1), "bad step")
    expect_value_error(lambda: ParameterServer(problem, [3, 3], "sync", 0.1, -1), "bad bound")
    left = run_async(ParameterServer(problem, latencies, "async", async_lr, bound), horizon)
    right = run_async(ParameterServer(problem, latencies, "async", async_lr, bound), horizon)
    check(left.log == right.log and left.w == right.w,
          "the same schedule and seed must produce the same log and parameters")

    print(f"719 sync gap={problem.loss(sync.w) - best_loss:.2e} wall={sync.wall_time} | async "
          f"applied={server.applied} dropped={server.dropped} max_stale={server.max_staleness} "
          f"gap={async_gap:.2e} | {budget} units: sync={rounds} async={fast.applied} updates")
    print("PASS")


if __name__ == "__main__":
    main()
