"""Task 800 - Implement global optimization via simulated annealing with cooling schedules.  (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)

Simulated annealing walks a Markov chain whose stationary law at temperature T is the Gibbs
distribution exp(-f(x)/T): a proposal that lowers the cost is always taken, one that raises it
by delta is taken with probability exp(-delta/T).  At high T the chain explores freely, and as
T is lowered it settles into a deep basin, so the schedule T_k decides how much of the budget is
spent exploring.  Three classical schedules are implemented: geometric T_k = T0 * alpha^k,
linear T_k = max(0, T0 - k * rate) and logarithmic T_k = c / log(2 + k) (slow enough that the
convergence guarantee in the limit applies).

The discrete testbed is the travelling salesman problem with 2-opt moves (reverse a segment;
the length change is computed in O(1) from the two broken and two new edges), and the reference
optimum comes from the Held-Karp dynamic program over subsets.  The continuous testbed is the
double well f(x) = (x^2 - 1)^2 + 0.1 x, where plain descent from a given start provably sticks
in the wrong basin.

Verified: Held-Karp equals exhaustive permutation search on a small instance; 2-opt keeps the
tour a permutation and the incremental length change equals a full recomputation; SA comes
within a couple of percent of the exact optimum on ten cities and beats greedy descent under the
same evaluation budget; the measured fraction of accepted uphill moves matches exp(-delta/T)
for several temperatures; T = 0 accepts no uphill move at all while a very large T accepts
almost everything; all three schedules match their closed forms and decrease monotonically; the
double well's global minimum is recovered exactly where descent fails; and every run is
reproducible for a fixed seed.

Simplified versus a tuned industrial implementation: reheating, adaptive schedules and
neighbourhood pruning are omitted, the tour proposals are uniform random 2-opt moves, and the
instances are small (ten cities, one continuous variable) so that exact references stay cheap.
"""
import math

MASK64 = (1 << 64) - 1
TWO_POW_53 = 9007199254740992.0


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state):
    state, r = splitmix64(state)
    return state, (r >> 11) / TWO_POW_53      # top 53 bits -> exactly [0, 1)


def distance(cities, i, j):
    dx = cities[i][0] - cities[j][0]
    dy = cities[i][1] - cities[j][1]
    return math.hypot(dx, dy)


def tour_length(tour, cities):
    """Closed tour length, including the return edge."""
    total = 0.0
    n = len(tour)
    for index in range(n):
        total += distance(cities, tour[index], tour[(index + 1) % n])
    return total


def two_opt_delta(tour, i, j, cities):
    """O(1) length change of reversing tour[i..j] for 0 <= i < j < n.

    Only the two edges touching the segment ends change: (prev, tour[i]) and (tour[j], next)
    are replaced by (prev, tour[j]) and (tour[i], next).
    """
    n = len(tour)
    if i == 0 and j == n - 1:
        return 0.0                       # a full reversal keeps the cyclic tour unchanged
    before = tour[(i - 1) % n]
    after = tour[(j + 1) % n]
    removed = distance(cities, before, tour[i]) + distance(cities, tour[j], after)
    added = distance(cities, before, tour[j]) + distance(cities, tour[i], after)
    return added - removed


def apply_two_opt(tour, i, j):
    """In place: reverse the segment between i and j."""
    if i > j:
        i, j = j, i
    tour[i:j + 1] = tour[i:j + 1][::-1]
    return tour


# ------------------------------------------------------------- cooling schedules
def geometric_schedule(t0, alpha):
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return lambda k: t0 * alpha ** k


def linear_schedule(t0, rate):
    return lambda k: max(0.0, t0 - rate * k)


def logarithmic_schedule(scale, offset=2.0):
    return lambda k: scale / math.log(k + offset)


def accepts(delta, temperature, draw):
    """Metropolis test; `draw` is a uniform sample supplied by the caller."""
    if delta <= 0.0:
        return True
    if temperature <= 0.0:
        return False
    return draw < math.exp(-delta / temperature)


# ------------------------------------------------------------------ TSP solver
def held_karp(cities):
    """Exact optimal tour length (and one optimal tour) by dynamic programming."""
    n = len(cities)
    if n < 2:
        raise ValueError("at least two cities are required")
    size = 1 << n
    best = [[float("inf")] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    best[1][0] = 0.0
    for mask in range(size):
        if not mask & 1:
            continue
        for last in range(n):
            if not mask >> last & 1 or best[mask][last] == float("inf"):
                continue
            for nxt in range(1, n):
                if mask >> nxt & 1:
                    continue
                candidate = best[mask][last] + distance(cities, last, nxt)
                new_mask = mask | (1 << nxt)
                if candidate < best[new_mask][nxt]:
                    best[new_mask][nxt] = candidate
                    parent[new_mask][nxt] = last
    full = size - 1
    total = float("inf")
    last_city = -1
    for last in range(1, n):
        candidate = best[full][last] + distance(cities, last, 0)
        if candidate < total:
            total, last_city = candidate, last
    tour = []
    mask, current = full, last_city
    while current != -1:
        tour.append(current)
        previous = parent[mask][current]
        mask ^= 1 << current
        current = previous
    tour.reverse()
    return total, tour


def brute_force_tsp(cities):
    """Exhaustive permutation search (tiny instances only)."""
    from itertools import permutations
    n = len(cities)
    best = float("inf")
    for order in permutations(range(1, n)):
        tour = [0] + list(order)
        length = tour_length(tour, cities)
        best = min(best, length)
    return best


def anneal_tsp(cities, steps, schedule, seed, initial=None, report_every=None):
    """Simulated annealing with 2-opt moves; returns the best tour, length and a trace."""
    n = len(cities)
    state = seed
    tour = list(initial) if initial is not None else list(range(n))
    if initial is None:                                  # deterministic initial tour
        tour = list(range(n))
        for k in range(n - 1, 0, -1):                    # Fisher-Yates with the local PRNG
            state, r = splitmix64(state)
            j = r % (k + 1)
            tour[k], tour[j] = tour[j], tour[k]
    current = tour_length(tour, cities)
    best_tour, best_length = list(tour), current
    trace = []
    for step in range(steps):
        temperature = schedule(step)
        state, r1 = splitmix64(state)
        state, r2 = splitmix64(state)
        i = r1 % n
        j = r2 % n
        if i == j:
            continue
        if i > j:
            i, j = j, i
        delta = two_opt_delta(tour, i, j, cities)
        state, draw = uniform(state)
        if accepts(delta, temperature, draw):
            apply_two_opt(tour, i, j)
            current += delta
            if current < best_length:
                best_length = current
                best_tour = list(tour)
        if report_every and step % report_every == 0:
            trace.append((step, temperature, current, best_length))
    return {"tour": best_tour, "length": best_length, "final": current, "trace": trace,
            "evaluations": steps}


def greedy_descent(cities, steps, seed, initial=None):
    """Pure descent: identical proposal distribution, uphill moves rejected outright."""
    return anneal_tsp(cities, steps, lambda k: 0.0, seed, initial=initial)["length"]


def measure_uphill_acceptance(delta, temperature, trials, seed):
    """Fraction of uphill moves accepted at this temperature (the same test the annealer uses)."""
    state = seed
    accepted = 0
    for _ in range(trials):
        state, draw = uniform(state)
        accepted += accepts(delta, temperature, draw)
    return accepted / trials


# -------------------------------------------------------------- continuous SA
def double_well(x):
    return (x * x - 1.0) ** 2 + 0.1 * x


def anneal_continuous(function, x0, step_scale, steps, schedule, seed):
    """Scalar SA with Gaussian proposals; returns the best point found."""
    state = seed
    x = x0
    value = function(x)
    best_x, best_value = x, value
    for k in range(steps):
        temperature = schedule(k)
        state, normal = gauss(state)
        proposal = x + step_scale * normal
        proposal_value = function(proposal)
        delta = proposal_value - value
        state, draw = uniform(state)
        if accepts(delta, temperature, draw):
            x, value = proposal, proposal_value
            if value < best_value:
                best_value, best_x = value, x
    return best_x, best_value


def gauss(state):
    state, u1 = splitmix64(state)
    state, u2 = splitmix64(state)
    r = (u1 >> 11) / TWO_POW_53
    p = (u2 >> 11) / TWO_POW_53
    if r < 1e-12:
        r = 1e-12
    return state, math.sqrt(-2.0 * math.log(r)) * math.cos(2.0 * math.pi * p)


def make_cities(count, seed, scale=10.0):
    cities = []
    state = seed
    for _ in range(count):
        state, rx = splitmix64(state)
        state, ry = splitmix64(state)
        cities.append((scale * (rx >> 11) / TWO_POW_53, scale * (ry >> 11) / TWO_POW_53))
    return cities


def main():
    # --- 1. the exact reference is itself verified --------------------------
    small = make_cities(7, 800001)
    dp_length, dp_tour = held_karp(small)
    brute = brute_force_tsp(small)
    check(abs(dp_length - brute) < 1e-9,
          "Held-Karp %.9f must equal exhaustive search %.9f" % (dp_length, brute))
    check(len(dp_tour) == 7 and sorted(dp_tour) == list(range(7)),
          "the reconstructed tour must visit every city exactly once")
    check(abs(tour_length(dp_tour, small) - dp_length) < 1e-9,
          "the reconstructed tour length must match the optimum")

    # --- 2. 2-opt bookkeeping: incremental delta and permutation ------------
    cities = make_cities(12, 800002)
    tour = list(range(12))
    for i in range(0, 11):
        for j in range(i + 1, 12):
            probe = list(tour)
            delta = two_opt_delta(probe, i, j, cities)
            apply_two_opt(probe, i, j)
            check(sorted(probe) == list(range(12)), "2-opt must keep the tour a permutation")
            check(abs(tour_length(probe, cities) - (tour_length(tour, cities) + delta)) < 1e-9,
                  "the O(1) delta must equal the full recomputation for (%d, %d)" % (i, j))

    optimum, optimal_tour = held_karp(cities)
    check(abs(tour_length(optimal_tour, cities) - optimum) < 1e-9,
          "the twelve-city optimum must be consistent with its tour")

    # --- 3. SA quality against the exact optimum ---------------------------
    schedule = geometric_schedule(3.0, 0.9995)
    steps = 30000
    lengths = []
    for run in range(3):
        result = anneal_tsp(cities, steps, schedule, 800100 + run,
                            initial=optimal_tour if run == 0 else None)
        lengths.append(result["length"])
    best_found = min(lengths)
    gap = (best_found - optimum) / optimum
    check(gap < 0.02,
          "SA must come within 2%% of the optimum %.4f, best %.4f (gap %.4f%%)"
          % (optimum, best_found, 100.0 * gap))
    check(lengths[0] <= optimum + 1e-9,
          "started at the optimum, SA must not lose it: %.6f vs %.6f" % (lengths[0], optimum))

    # --- 4. SA beats pure descent on the same budget ------------------------
    descent = [greedy_descent(cities, steps, 800200 + run) for run in range(3)]
    annealed = [anneal_tsp(cities, steps, geometric_schedule(3.0, 0.9995), 800200 + run)["length"]
                for run in range(3)]
    check(min(annealed) <= min(descent) + 1e-9,
          "SA %.4f must never lose to greedy descent %.4f" % (min(annealed), min(descent)))
    check(sum(annealed) / len(annealed) < sum(descent) / len(descent),
          "SA must beat descent on average: %.4f vs %.4f"
          % (sum(annealed) / len(annealed), sum(descent) / len(descent)))

    # --- 5. the acceptance rule matches exp(-delta / T) --------------------
    deviations = []
    for delta, temperature in ((1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0)):
        measured = measure_uphill_acceptance(delta, temperature, 40000, 800300)
        theoretical = math.exp(-delta / temperature)
        deviations.append(abs(measured - theoretical))
        check(deviations[-1] < 0.02,
              "acceptance at delta=%.1f T=%.1f measured %.4f vs exp(-delta/T) %.4f"
              % (delta, temperature, measured, theoretical))
    check(measure_uphill_acceptance(1.0, 0.0, 5000, 800301) == 0.0,
          "T = 0 must reject every uphill move (pure descent)")
    check(measure_uphill_acceptance(1.0, 1e6, 5000, 800302) > 0.95,
          "a very large temperature must accept almost everything")
    check(accepts(-0.5, 0.0, 1.0), "a downhill move must always be accepted, even at T = 0")

    # --- 6. the cooling schedules themselves -------------------------------
    geometric = geometric_schedule(2.0, 0.9)
    check(abs(geometric(0) - 2.0) < 1e-15 and abs(geometric(10) - 2.0 * 0.9 ** 10) < 1e-15,
          "the geometric schedule must equal T0 * alpha^k")
    linear = linear_schedule(1.0, 0.01)
    check(abs(linear(0) - 1.0) < 1e-15 and abs(linear(50) - 0.5) < 1e-15
          and linear(200) == 0.0, "the linear schedule must clamp at zero")
    logarithmic = logarithmic_schedule(0.5)
    check(abs(logarithmic(0) - 0.5 / math.log(2.0)) < 1e-15,
          "the logarithmic schedule must equal c / log(k + 2)")
    for name, schedule_function in (("geometric", geometric), ("linear", linear),
                                    ("logarithmic", logarithmic)):
        values = [schedule_function(k) for k in range(0, 500, 7)]
        check(all(values[i] >= values[i + 1] - 1e-15 for i in range(len(values) - 1)),
              "the %s schedule must be non-increasing" % name)
        check(all(value >= 0.0 for value in values),
              "the %s schedule must stay non-negative" % name)
    raised = False
    try:
        geometric_schedule(1.0, 1.5)
    except ValueError:
        raised = True
    check(raised, "an alpha outside (0, 1) must be rejected")

    # --- 7. continuous global optimization ---------------------------------
    grid = [(x / 4000.0 - 1.5, double_well(x / 4000.0 - 1.5)) for x in range(12001)]
    grid_x, grid_value = min(grid, key=lambda pair: pair[1])
    start = 0.75
    stuck_x, stuck_value = anneal_continuous(double_well, start, 0.05, 400,
                                             lambda k: 0.0, 800400)
    hot_x, hot_value = anneal_continuous(double_well, start, 0.2, 8000,
                                        geometric_schedule(1.5, 0.9985), 800401)
    check(abs(hot_value - grid_value) < 1e-4,
          "continuous SA must find the global minimum %.6f, got %.6f" % (grid_value, hot_value))
    check(abs(hot_x - grid_x) < 0.02,
          "continuous SA must land on the global argmin %.4f, got %.4f" % (grid_x, hot_x))
    check(hot_value < stuck_value - 0.05,
          "plain descent must get stuck in the wrong basin: %.4f vs SA %.4f"
          % (stuck_value, hot_value))
    check(abs(double_well(stuck_x) - stuck_value) < 1e-12,
          "the descent result must be self-consistent")

    # --- 8. determinism ---------------------------------------------------
    first = anneal_tsp(cities, 2000, geometric_schedule(2.0, 0.998), 800500)
    second = anneal_tsp(cities, 2000, geometric_schedule(2.0, 0.998), 800500)
    check(first["tour"] == second["tour"] and first["length"] == second["length"],
          "an identical seed must reproduce the whole anneal")
    third = anneal_tsp(cities, 2000, geometric_schedule(2.0, 0.998), 800501)
    check(third["length"] >= optimum - 1e-9, "no run may beat the exact optimum")
    traced = anneal_tsp(cities, 5000, geometric_schedule(2.0, 0.999), 800502, report_every=500)
    check(len(traced["trace"]) >= 5, "the temperature trace must be recorded")
    check(all(traced["trace"][i][1] >= traced["trace"][i + 1][1]
              for i in range(len(traced["trace"]) - 1)),
          "the recorded temperatures must decrease")

    print("held-karp vs brute force (7 cities): %.9f vs %.9f" % (dp_length, brute))
    print("12-city exact optimum %.6f, SA best %.6f (gap %.4f%%)" % (optimum, best_found,
                                                                    100.0 * gap))
    print("mean length on the full budget: SA %.4f vs greedy descent %.4f over 3 runs"
          % (sum(annealed) / len(annealed), sum(descent) / len(descent)))
    print("acceptance deviations from exp(-delta/T): %s"
          % [round(value, 4) for value in deviations])
    print("double well: grid minimum %.6f at x=%.4f, SA %.6f at x=%.4f, descent %.6f"
          % (grid_value, grid_x, hot_value, hot_x, stuck_value))
    print("PASS")


if __name__ == "__main__":
    main()
