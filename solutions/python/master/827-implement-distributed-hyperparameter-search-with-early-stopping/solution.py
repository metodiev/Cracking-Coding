"""Task 827 - Implement distributed hyperparameter search with early stopping and resource allocation (BOHB). (Tier: Master, CTCI Beyond — Machine Learning & AI)

BOHB = Hyperband + a kernel-density proposal. Hyperband allocates a *fidelity* budget
successively -- many cheap configurations, few expensive ones, each rung keeping only the
best 1/eta -- and a KDE model replaces Hyperband's uniform sampling so that most cheap
configurations are drawn near the configurations that looked good at the same fidelity.
This file implements the whole loop: the fidelity schedule, the successive-halving rungs,
the density-ratio sampler, and a tick-driven cluster of workers that executes the tasks.

What is implemented
-------------------
* The discrete space is the full grid of `DIMS` (4 learning rates x 2 depths).
* `objective(kind)` is the "training run", loss(config, budget), deterministic. `aligned`
  keeps the *ranking* identical at every fidelity, so the optimum is provably
  recoverable; `misleading` deliberately reverses the cheap-fidelity ranking so that the
  cost of a bad low-fidelity signal can be exhibited instead of asserted away.
* `KDESampler`: two 1-D Gaussian KDEs over the *grid index* (exact for a finite grid, so
  no multi-variate tree is needed), one fitted to the best `gamma` fraction of the
  observations available at that fidelity and one to the rest; the proposal is the
  argmax of the density ratio over the grid (the TPE/BOHB rule, made deterministic).
  Until `n_min` observations exist, and for a `1 - quota` fraction of every batch, the
  proposal comes from a splitmix64 stream instead.
* `InMemoryNetwork` + `Cluster`: a message bus with a fixed latency and `W` workers. A
  round of tasks becomes TASK messages, a worker is occupied for `cost(budget)` ticks,
  and the result returns as a RESULT message; the coordinator learns nothing before the
  message arrives. The cluster reports concurrency, charged budget and the message log.
* `hyperband(kind, workers)`: brackets s = s_max ... 0, each running successive halving
  (`n = ceil((s_max+1)/(s+1) eta^s)` configurations at `r = r_max eta^-s`, promoting
  `max(1, k//eta)` per rung), feeding every observation back into the sampler.

What is simplified
------------------
* Fidelity is one integer `budget` (epochs) and a run costs `budget` ticks, so simulated
  time is a faithful proxy for cost without any wall-clock.
* The model is a 1-D KDE over grid indices rather than BOHB's adaptive multi-variate KDE
  over a configuration tree; the density-ratio proposal and the exploration quota -- the
  parts that make BOHB BOHB -- are unchanged.
* No threads, sockets, sleeping or randomness beyond the seeded splitmix64 stream: one
  process, one deterministic event loop, messages only through `InMemoryNetwork`.

Invariants verified
-------------------
1.  The fidelity schedule matches the closed form n_s = ceil((s_max+1)/(s+1) eta^s),
    r_s = r_max eta^-s, brackets ordered from s_max down to 0, so the cheapest fidelity
    explores the most configurations.
2.  Resource allocation is exact: the budget the cluster charges equals the independently
    enumerated Hyperband rung schedule (nothing lost, nothing double-charged), stays under
    the closed-form bracket bound, and is well below the cost of running every sampled
    configuration at the maximum fidelity.
3.  Every rung promotes exactly max(1, floor(k/eta)) configurations, and that set is the
    rung's own loss order (the promotion rule, re-derived independently).
4.  Early stopping is real: only promoted configurations are re-evaluated, at a strictly
    higher fidelity, and the widest rung of the whole search is the cheapest one.
5.  The cluster never exceeds `W` concurrent tasks, does reach W, emits exactly one result
    message per task, and never delivers a message before it was sent.
6.  On the ranking-aligned instance the incumbent (best run at the maximum fidelity) is
    exactly the brute-force optimum of the grid.
7.  On the misleading instance the cheap-fidelity argmin provably differs from the
    full-fidelity argmin, and the search is fooled: the incumbent is *not* the optimum
    (measured, not hidden) yet stays within the documented factor of it.
8.  The KDE proposal maximises an independently recomputed density ratio over the grid,
    prefers the neighbourhood of good observations, returns duplicate-free batches, stays
    cold until `n_min` observations exist, and warm-up sampling is reproducible.
9.  The whole search is bit-reproducible and worker-count invariant: 1 worker (fully
    serialised) charges exactly the same budget and returns the same incumbent.
"""
import itertools
import math

ETA = 3
R_MIN, R_MAX = 1, 9
S_MAX = 2                                     # = log_eta(R_MAX / R_MIN)
DIMS = ((0.5, 0.25, 0.125, 0.0625), (1, 2))
GRID = list(itertools.product(*DIMS))
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# the objective ("training run")
# --------------------------------------------------------------------------------------
def base_loss(config):
    lr, depth = config
    x = math.log2(lr) + 3.0                   # 0.0625 -> 1, 0.5 -> -2
    return 0.05 * (x + 0.6) ** 2 + 0.02 * (depth - 1.6) ** 2


def objective(kind):
    """Returns loss(config, budget); deterministic, and the budget is the fidelity."""
    def evaluate(config, budget):
        base = base_loss(config)
        if kind == "aligned":
            return base + base / budget        # identical ranking at every fidelity
        lr, depth = config
        bonus = (0.9 - 0.5 * depth) / budget   # cheap runs love depth 2 ...
        trap = 0.25 * (math.log2(lr) + 1.0) ** 2 / budget   # ... and a big learning rate
        return base + bonus + trap
    return evaluate


def fidelity_schedule(s_max=S_MAX, eta=ETA, r_max=R_MAX):
    """Hyperband brackets, longest (most configurations, cheapest fidelity) first."""
    brackets = []
    for s in range(s_max, -1, -1):
        brackets.append({"s": s,
                         "n": math.ceil((s_max + 1) / (s + 1) * eta ** s),
                         "r": int(r_max * eta ** (-s)),
                         "rungs": s + 1})
    return brackets


def rung_sizes(n, rungs):
    """Configuration count per rung of one successive-halving bracket."""
    sizes = [n]
    while len(sizes) < rungs and sizes[-1] > 1:
        sizes.append(max(1, sizes[-1] // ETA))
    return sizes


def rung_schedule(n, rungs, r, r_max=R_MAX):
    """(configurations, budget) for each rung -- the independent resource oracle."""
    plan, budget = [], r
    for size in rung_sizes(n, rungs):
        plan.append((size, budget))
        if size == 1 or budget * ETA > r_max:
            break
        budget *= ETA
    return plan


# --------------------------------------------------------------------------------------
# deterministic PRNG + the KDE proposal
# --------------------------------------------------------------------------------------
class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


class KDESampler:
    """Density-ratio proposal over grid indices (the BOHB/TPE sampling rule)."""

    def __init__(self, grid=None, bandwidth=1.0, gamma=0.2, n_min=3, quota=0.7, seed=12345):
        self.grid = GRID if grid is None else grid
        self.bandwidth = bandwidth
        self.gamma = gamma
        self.n_min = n_min
        self.quota = quota
        self.observations = []                # (index, loss, budget)
        self.rng = SplitMix64(seed)

    def observe(self, index, loss, budget):
        self.observations.append((index, loss, budget))

    def split(self, budget):
        """Best gamma fraction of the observations at (or below) `budget`, and the rest."""
        pool = [(i, l) for i, l, b in self.observations if b <= budget]
        if len(pool) < self.n_min:
            return [], []
        pool.sort(key=lambda pair: (pair[1], pair[0]))
        cut = max(1, int(math.floor(self.gamma * len(pool))))
        return [i for i, _ in pool[:cut]], [i for i, _ in pool[cut:]]

    def density(self, x, points):
        """Mean Gaussian density of `points` at x (a plain KDE with one bandwidth)."""
        if not points:
            return 0.0
        h = self.bandwidth
        weight = 1.0 / (len(points) * h * math.sqrt(2.0 * math.pi))
        return weight * sum(math.exp(-0.5 * ((x - p) / h) ** 2) for p in points)

    def ratio(self, x, budget):
        good, bad = self.split(budget)
        if not good:
            return None
        if not bad:
            return math.inf                  # every observation is "good"
        return self.density(x, good) / self.density(x, bad)

    def random_index(self, exclude):
        while True:
            candidate = self.rng.next_u64() % len(self.grid)
            if candidate not in exclude:
                return candidate

    def propose(self, budget, exclude=()):
        """argmax of the density ratio over the grid; None while the model is cold."""
        best, best_score = None, None
        for index in range(len(self.grid)):
            if index in exclude:
                continue
            score = self.ratio(index, budget)
            if score is None:
                return None
            if best_score is None or score > best_score:
                best, best_score = index, score
        return best

    def sample_batch(self, count, budget):
        """Distinct configurations: a `quota` share from the model, the rest uniform."""
        picked = []
        target = min(count, len(self.grid))
        if len(self.observations) >= self.n_min:
            for _ in range(min(target, int(math.ceil(self.quota * count)))):
                index = self.propose(budget, set(picked))
                if index is None:
                    break
                picked.append(index)
        while len(picked) < target:
            picked.append(self.random_index(set(picked)))
        return picked


# --------------------------------------------------------------------------------------
# the distributed executor
# --------------------------------------------------------------------------------------
class InMemoryNetwork:
    """Deterministic message bus: a message sent at tick T arrives at T + latency."""

    def __init__(self, latency=1):
        self.latency = latency
        self.inflight = []
        self.log = []

    def send(self, tick, sender, receiver, kind, payload):
        self.inflight.append((tick + self.latency, sender, receiver, kind, payload))
        self.log.append((tick, tick + self.latency, sender, receiver, kind))

    def deliver(self, tick):
        due = sorted([m for m in self.inflight if m[0] == tick],
                     key=lambda m: (m[1], m[3], repr(m[4])))
        self.inflight = [m for m in self.inflight if m[0] != tick]
        return due


class Cluster:
    """`workers` workers pulling tasks off a queue, driven by an explicit tick loop."""

    def __init__(self, workers, evaluate, network=None, cost=None):
        self.workers = tuple(f"w{i}" for i in range(workers))
        self.evaluate = evaluate
        self.network = network if network is not None else InMemoryNetwork(latency=1)
        self.cost = cost if cost is not None else (lambda budget: budget)
        self.budget_used = 0
        self.runs = []                       # (task_id, config, budget, dispatch tick)
        self.peak_concurrency = 0

    def run_round(self, tasks):
        """Executes one synchronous round of (task_id, config, budget); returns the losses."""
        pending = list(tasks)
        running, results, tick = {}, {}, 0
        while pending or running:
            tick += 1
            for _, worker, _, kind, payload in self.network.deliver(tick):
                if kind == "RESULT":
                    task_id, config, budget = payload
                    results[task_id] = self.evaluate(config, budget)
                    del running[worker]
            for worker in self.workers:
                if worker in running or not pending:
                    continue
                task_id, config, budget = pending.pop(0)
                self.budget_used += budget
                self.runs.append((task_id, config, budget, tick))
                self.network.send(tick, "coordinator", worker, "TASK", (task_id, config, budget))
                self.network.send(tick + self.cost(budget), worker, "coordinator", "RESULT",
                                  (task_id, config, budget))
                running[worker] = task_id
            self.peak_concurrency = max(self.peak_concurrency, len(running))
        return results


# --------------------------------------------------------------------------------------
# the search driver
# --------------------------------------------------------------------------------------
def hyperband(kind, workers=3, seed=12345):
    """Runs every bracket; returns the incumbent, the event log and the cluster."""
    evaluate = objective(kind)
    cluster = Cluster(workers, evaluate)
    sampler = KDESampler(seed=seed)
    events = []
    incumbent = None                          # best run at the maximum fidelity only
    for bracket in fidelity_schedule():
        indices = sampler.sample_batch(bracket["n"], bracket["r"])
        budget = bracket["r"]
        rung = 0
        while True:
            tasks = [(f"s{bracket['s']}-r{budget}-{i}", GRID[index], budget)
                     for i, index in enumerate(indices)]
            losses = cluster.run_round(tasks)
            ranked = sorted((losses[task_id], index, config)
                            for (task_id, config, _), index in zip(tasks, indices))
            for loss, index, _ in ranked:
                sampler.observe(index, loss, budget)
                if budget == R_MAX and (incumbent is None
                                        or (loss, index) < (incumbent[0], incumbent[1])):
                    incumbent = (loss, index, GRID[index], budget)
            keep = max(1, len(ranked) // ETA)
            events.append({"bracket": bracket["s"], "rung": rung, "budget": budget,
                           "size": len(indices), "ranked": ranked,
                           "promoted": [row[1] for row in ranked[:keep]]})
            if len(indices) == 1 or budget * ETA > R_MAX:
                break
            indices = [row[1] for row in ranked[:keep]]
            budget *= ETA
            rung += 1
    return {"incumbent": incumbent, "events": events, "sampler": sampler, "cluster": cluster}


def full_budget_optimum(kind):
    evaluate = objective(kind)
    return sorted((evaluate(config, R_MAX), index, config)
                  for index, config in enumerate(GRID))[0]


def brute_force_at(kind, budget):
    evaluate = objective(kind)
    return sorted((evaluate(config, budget), index) for index, config in enumerate(GRID))


def main():
    # -- 1. the fidelity schedule --------------------------------------------------------
    schedule = fidelity_schedule()
    check([b["s"] for b in schedule] == [2, 1, 0], "brackets run from s_max down to 0")
    check([b["r"] for b in schedule] == [R_MIN, 3, R_MAX], "fidelity is r_max * eta^-s")
    check([b["n"] for b in schedule] == [9, 5, 3],
          "n_s = ceil((s_max+1)/(s+1) * eta^s) gives 9, 5 and 3 configurations")
    check(schedule[0]["n"] > schedule[-1]["n"] and schedule[0]["r"] < schedule[-1]["r"],
          "the cheapest fidelity gets the most configurations")
    check(rung_sizes(9, 3) == [9, 3, 1] and rung_sizes(5, 2) == [5, 1]
          and rung_sizes(1, 1) == [1] and rung_sizes(2, 2) == [2, 1],
          "successive halving keeps max(1, floor(k/eta)) configurations per rung")

    # -- 2. resource accounting ----------------------------------------------------------
    aligned = hyperband("aligned", workers=3)
    cluster = aligned["cluster"]
    expected = 0
    chains = {}
    for event in aligned["events"]:
        chains.setdefault(event["bracket"], []).append(event)
    for chain in chains.values():
        chain.sort(key=lambda e: e["budget"])
        expected += sum(size * budget for size, budget in
                        rung_schedule(chain[0]["size"], 99, chain[0]["budget"]))
    check(cluster.budget_used == expected,
          f"charged {cluster.budget_used} units, the Hyperband rung schedule says {expected}")
    check(cluster.budget_used == 74, "8*1 + 2*3 + 1*9 and 5*3 + 1*9 and 3*9 = 74 units")
    bound = sum(b["rungs"] * b["n"] * b["r"] for b in schedule)
    check(cluster.budget_used <= bound,
          f"{cluster.budget_used} units stay under the closed-form bracket bound {bound}")
    slots = sum(event["size"] for event in aligned["events"])
    naive = slots * R_MAX
    check(cluster.budget_used < naive,
          f"{cluster.budget_used} units instead of {naive} for the same {slots} runs at r_max")

    # -- 3. promotion is by rank ---------------------------------------------------------
    for event in aligned["events"]:
        keep = max(1, event["size"] // ETA)
        independent = [index for _, index in sorted((loss, index)
                                                    for loss, index, _ in event["ranked"])]
        check(event["promoted"] == independent[:keep],
              "promotion keeps exactly the max(1, k//eta) best configurations")
        check(len(event["promoted"]) >= 1 and len(event["promoted"]) <=
              max(1, event["size"] // ETA), "the promotion count is the eta-thinning rule")

    # -- 4. early stopping ---------------------------------------------------------------
    for chain in chains.values():
        for previous, following in zip(chain, chain[1:]):
            survivors = {index for _, index, _ in following["ranked"]}
            check(survivors <= set(previous["promoted"]),
                  "only promoted configurations are re-evaluated at a higher fidelity")
            check(following["budget"] > previous["budget"]
                  and following["size"] < previous["size"],
                  "fidelity rises while the population shrinks")
        check(chain[0]["size"] >= chain[-1]["size"], "populations shrink along a bracket")
        if len(chain) > 1:
            check(chain[-1]["size"] == 1, "a bracket that promotes ends with a single full run")
        else:
            check(chain[0]["budget"] == R_MAX and chain[0]["size"] < 9,
                  "a one-rung bracket runs straight at the maximum fidelity")
    widest = {budget: max(event["size"] for event in aligned["events"]
                          if event["budget"] == budget)
              for budget in (R_MIN, 3, R_MAX)}
    check(widest[R_MIN] > widest[3] > widest[R_MAX],
          f"the widest rung of the whole search is at the lowest fidelity {widest}")
    check(schedule[0]["n"] == 9 > len(GRID) == chains[2][0]["size"] == 8,
          "the widest bracket asks for 9 configurations, only 8 exist, so the grid is "
          "screened exhaustively at the cheapest fidelity")

    # -- 5. the cluster ------------------------------------------------------------------
    check(cluster.peak_concurrency == 3,
          f"concurrency reaches the worker limit exactly ({cluster.peak_concurrency})")
    sends = [m for m in cluster.network.log if m[4] == "TASK"]
    results = [m for m in cluster.network.log if m[4] == "RESULT"]
    check(len(sends) == len(results) == len(cluster.runs),
          "every task produced exactly one result message")
    check(all(deliver >= send for send, deliver, _, _, _ in cluster.network.log),
          "no message is delivered before it was sent")
    check(cluster.network.latency >= 1 and all(m[1] > m[0] for m in cluster.network.log),
          "the bus models a strictly positive latency")
    check(all(budget == R_MIN for _, _, budget, _ in cluster.runs[:8])
          and cluster.runs[8][2] == 3 and cluster.runs[-1][2] == R_MAX,
          "the search opens with a full round at the cheapest fidelity and ends at r_max")
    check(len(cluster.runs) == slots, "the cluster ran exactly the scheduled number of runs")

    # -- 6. the aligned instance ---------------------------------------------------------
    best_loss, best_index, best_config, best_budget = aligned["incumbent"]
    oracle = full_budget_optimum("aligned")
    check(best_index == oracle[1] and abs(best_loss - oracle[0]) < 1e-12,
          f"the incumbent is the brute-force optimum {oracle[2]} at fidelity {best_budget}")
    check(best_budget == R_MAX, "the winning configuration was confirmed at full fidelity")

    # -- 7. the misleading instance ------------------------------------------------------
    cheap_best = brute_force_at("misleading", R_MIN)[0][1]
    full_best = full_budget_optimum("misleading")[1]
    check(cheap_best != full_best,
          f"cheap argmin {GRID[cheap_best]} differs from the full-fidelity argmin "
          f"{GRID[full_best]}")
    misled = hyperband("misleading", workers=2)
    incumbent = misled["incumbent"]
    optimum = full_budget_optimum("misleading")
    check(incumbent[1] != optimum[1],
          "the misled search picks a different configuration than the exhaustive optimum")
    check(incumbent[0] <= 5.0 * optimum[0],
          f"it stays within a factor of five of the optimum "
          f"({incumbent[0]:.5f} vs {optimum[0]:.5f})")

    # -- 8. the sampler ------------------------------------------------------------------
    sampler = KDESampler(seed=7)
    sampler.observe(0, 0.01, R_MIN)           # the good observation
    sampler.observe(3, 2.0, R_MIN)
    sampler.observe(7, 5.0, R_MIN)
    good, bad = sampler.split(R_MIN)
    check(good == [0] and sorted(bad) == [3, 7],
          "the good set is the best gamma fraction, the rest is the bad set")
    check(sampler.ratio(0, R_MIN) > sampler.ratio(3, R_MIN) > sampler.ratio(6, R_MIN),
          "the density ratio decays away from the good observation")

    def independent_ratio(x, h=1.0):
        good_w = math.exp(-0.5 * ((x - 0) / h) ** 2)
        bad_w = 0.5 * (math.exp(-0.5 * ((x - 3) / h) ** 2)
                       + math.exp(-0.5 * ((x - 7) / h) ** 2))
        return good_w / bad_w

    chosen = sampler.propose(R_MIN)
    check(chosen == max(range(len(GRID)), key=independent_ratio),
          "the proposal maximises an independently recomputed density ratio")
    batch = sampler.sample_batch(4, R_MIN)
    check(len(set(batch)) == len(batch) == 4 and all(0 <= i < len(GRID) for i in batch),
          "a batch holds four distinct configurations on the grid")
    cold = KDESampler(seed=7)
    check(cold.propose(R_MIN) is None,
          "the model proposes nothing before n_min observations exist")
    check(KDESampler(seed=99).sample_batch(5, R_MIN)
          == KDESampler(seed=99).sample_batch(5, R_MIN),
          "warm-up sampling is reproducible from the seed")
    degenerate = KDESampler(seed=1)
    for i in range(4):
        degenerate.observe(i, float(i), R_MIN)
    full_set = degenerate.sample_batch(len(GRID) + 3, R_MIN)
    check(sorted(full_set) == list(range(len(GRID))),
          "asking for more configurations than exist returns each one exactly once")

    # -- 9. determinism and serialisation ------------------------------------------------
    repeat = hyperband("aligned", workers=3)
    serial = hyperband("aligned", workers=1)
    check(repeat["incumbent"] == aligned["incumbent"]
          and [event["promoted"] for event in repeat["events"]]
          == [event["promoted"] for event in aligned["events"]],
          "the whole search is bit-reproducible")
    check(serial["incumbent"] == aligned["incumbent"]
          and serial["cluster"].budget_used == cluster.budget_used,
          "one worker serialises the same schedule and returns the same incumbent")
    check(serial["cluster"].peak_concurrency == 1, "a one-worker cluster never overlaps")
    check(serial["cluster"].budget_used == cluster.budget_used == expected,
          "the charged budget does not depend on the worker count")

    print(f"fidelity schedule (s, n, r): {[(b['s'], b['n'], b['r']) for b in schedule]}")
    print(f"resource: {cluster.budget_used} units (bracket bound {bound}; {slots} runs at "
          f"max fidelity would cost {naive}); incumbent {best_config}")
    print(f"misleading instance: cheap argmin {GRID[cheap_best]}, full argmin {GRID[full_best]}, "
          f"incumbent {GRID[incumbent[1]]} loss {incumbent[0]:.5f} vs optimum {optimum[0]:.5f}")
    print("PASS")


if __name__ == "__main__":
    main()
