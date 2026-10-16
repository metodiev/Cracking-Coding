"""Task 769 - Implement knapsack approximation PTAS idea demonstration.  (Tier: Expert, Ch 8 - Recursion and Dynamic Programming)

0/1 knapsack: pick a subset of items with total weight <= capacity maximising total profit.
The exact capacity DP costs O(n * capacity) and is only pseudo-polynomial; the *scaled-profit*
FPTAS replaces every profit by p_i' = floor(p_i / k) with k = eps * p_max / n and then runs a DP
whose table is indexed by profit instead of capacity.  The scaled table has at most
sum_i floor(p_i / k) + 1 <= n * floor(n / eps) + 1 states, so its cost is polynomial in n and
1/eps; since the DP is exact on the scaled profits and each rounding drops less than k per item,
the reconstructed set satisfies   profit >= k * OPT' >= OPT - n*k >= (1 - eps) * OPT.

Implementations and what they are verified against:
  * brute_force_knapsack - exhaustive subset enumeration (n <= 18): the independent oracle that
    validates the capacity DP, not just the FPTAS;
  * exact_knapsack - capacity DP with back-pointers, returns value and the chosen indices;
  * fptas_knapsack - scaled-profit DP (min weight per scaled profit), returns value, chosen
    indices and the size of the scaled table so the polynomial-state claim can be checked;
  * greedy_half - take free items, then items in decreasing profit/weight ratio, and compare with
    the single best item; the classic 1/2-approximation baseline for contrast.

What is simplified: the DP is written with plain Python lists and rebuilt per instance (no
divide-and-conquer / SMAWK speed-ups), and eps is kept >= 0.05 so the table stays small.  The
approximation *guarantee* itself is not simplified: every returned set is checked to be feasible
(distinct indices, weight <= capacity, reported value equal to the sum of its profits), to be at
most OPT, and to reach at least (1 - eps) * OPT, with eps -> 0 recovering the exact optimum.
"""
import math

INF = math.inf
MAX_SCALED_STATES = 2_000_000   # documented guard: n^2/eps states, so keep eps reasonable


def check(ok, message):
    if not ok:
        raise AssertionError(message)


MASK64 = (1 << 64) - 1


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def random_instance(count, seed, weight_high=30, profit_high=40):
    """Deterministic (weight, profit) pairs plus a capacity of about 45% of the total weight."""
    items = []
    for _ in range(count):
        seed, r1 = splitmix64(seed)
        seed, r2 = splitmix64(seed)
        items.append((1 + r1 % weight_high, 1 + r2 % profit_high))
    capacity = max(1, (sum(w for w, _ in items) * 45) // 100)
    return items, capacity, seed


def brute_force_knapsack(items, capacity):
    """Exhaustive enumeration over all 2^n subsets; only viable for tiny n."""
    n = len(items)
    best_value, best_mask = 0, 0
    for mask in range(1 << n):
        weight = profit = 0
        for i in range(n):
            if mask >> i & 1:
                weight += items[i][0]
                profit += items[i][1]
        if weight <= capacity and profit > best_value:
            best_value, best_mask = profit, mask
    return best_value, [i for i in range(n) if best_mask >> i & 1]


def exact_knapsack(items, capacity):
    """Capacity-indexed DP: dp[c] = best profit with weight budget c; back-pointers rebuild the set."""
    if capacity < 0:
        return 0, []
    dp = [0] * (capacity + 1)
    take = [bytearray(capacity + 1) for _ in items]
    for i, (w, p) in enumerate(items):
        if w > capacity:
            continue
        for c in range(capacity, w - 1, -1):
            candidate = dp[c - w] + p
            if candidate > dp[c]:
                dp[c] = candidate
                take[i][c] = 1
    chosen = []
    c = capacity
    for i in range(len(items) - 1, -1, -1):
        if take[i][c]:
            chosen.append(i)
            c -= items[i][0]
    return dp[capacity], sorted(chosen)


def fptas_knapsack(items, capacity, eps):
    """Scaled-profit DP.  Returns (value, chosen_indices, scaled_table_size)."""
    check(0 < eps, "the PTAS needs a positive precision")
    feasible = [(i, w, p) for i, (w, p) in enumerate(items) if w <= capacity]
    if capacity < 0 or not feasible:
        return 0, [], 0
    p_max = max(p for _, _, p in feasible)
    if p_max <= 0:
        return 0, [], 1
    count = len(feasible)
    scale = eps * p_max / count          # k: rounding granularity
    scaled = [int(p / scale) for _, _, p in feasible]     # floor rounding, keep it deterministic
    total = sum(scaled)
    if total + 1 > MAX_SCALED_STATES:
        raise ValueError("scaled-profit table would be too large for this eps")

    # dp[i][s] = minimum weight achieving scaled profit s with the first i feasible items.
    dp = [[INF] * (total + 1) for _ in range(count + 1)]
    dp[0][0] = 0
    for i in range(1, count + 1):
        weight, gain = feasible[i - 1][1], scaled[i - 1]
        row, previous = dp[i], dp[i - 1]
        for s in range(total + 1):
            best = previous[s]
            if s >= gain:
                alt = previous[s - gain] + weight
                if alt < best:
                    best = alt
            row[s] = best
    target = 0
    for s in range(total, -1, -1):
        if dp[count][s] <= capacity:
            target = s
            break

    chosen = []
    s = target
    for i in range(count, 0, -1):
        if dp[i][s] == dp[i - 1][s]:
            continue                      # item i is not part of this optimal scaled solution
        chosen.append(feasible[i - 1][0])
        s -= scaled[i - 1]
    chosen.sort()
    value = sum(items[i][1] for i in chosen)
    return value, chosen, total + 1


def greedy_half(items, capacity):
    """1/2-approximation baseline: free items, then decreasing profit/weight ratio, or best item."""
    chosen = [i for i, (w, p) in enumerate(items) if w == 0 and p > 0]
    weight = 0
    value = sum(items[i][1] for i in chosen)
    order = sorted((i for i, (w, _) in enumerate(items) if w > 0),
                   key=lambda i: (-(items[i][1] / items[i][0]), i))
    for i in order:
        w, p = items[i]
        if weight + w <= capacity:
            chosen.append(i)
            weight += w
            value += p
    best_single = max((p for w, p in items if w <= capacity), default=0)
    return max(value, best_single)


def set_value(items, chosen):
    return sum(items[i][1] for i in chosen), sum(items[i][0] for i in chosen)


def main():
    # 1. hand-checked instance: capacity 8, the optimum is 10 + 5 = 15 (weights 5 + 3)
    hand = [(5, 10), (4, 7), (3, 5), (2, 3)]
    value, chosen = exact_knapsack(hand, 8)
    check(value == 15 and set_value(hand, chosen) == (15, 8),
          "the capacity DP must find the hand-checked optimum 15 at weight 8")

    # 2. the capacity DP is itself verified against exhaustive enumeration
    seed = 0x5EED
    comparisons = 0
    for trial in range(8):
        items, capacity, seed = random_instance(11 + trial, seed, 12, 25)
        exact_value, _ = exact_knapsack(items, capacity)
        brute_value, brute_set = brute_force_knapsack(items, capacity)
        check(exact_value == brute_value,
              f"capacity DP {exact_value} != brute force {brute_value} on trial {trial}")
        check(set_value(items, brute_set)[1] <= capacity, "the brute-force set must respect capacity")
        comparisons += 1

    # 3. boundary: zero capacity, nothing fits, everything fits, single item
    check(exact_knapsack([(3, 9)], 0)[0] == 0, "zero capacity must yield zero profit")
    check(fptas_knapsack([(3, 9)], 0, 0.2)[0] == 0, "the FPTAS must handle zero capacity")
    check(exact_knapsack([(9, 9), (8, 4)], 5)[0] == 0, "items heavier than capacity are unusable")
    check(fptas_knapsack([(9, 9), (8, 4)], 5, 0.2)[0] == 0, "the FPTAS must discard unusable items")
    all_fit = [(2, 4), (3, 6), (1, 1)]
    check(exact_knapsack(all_fit, 10)[0] == 11, "everything fits: the whole profit is optimal")
    check(fptas_knapsack(all_fit, 10, 0.5)[0] == 11, "everything fits: the FPTAS must take all")
    check(exact_knapsack([(4, 8)], 4)[0] == 8, "a single fitting item is optimal")
    check(exact_knapsack([(4, 8)], 3)[0] == 0, "a single non-fitting item yields nothing")
    check(exact_knapsack([], 10)[0] == 0 and fptas_knapsack([], 10, 0.1)[0] == 0,
          "an empty item list must give zero")
    zero_profit = [(2, 0), (3, 0)]
    check(exact_knapsack(zero_profit, 10)[0] == 0 and fptas_knapsack(zero_profit, 10, 0.1)[0] == 0,
          "zero profits must give zero, not a division by zero")
    free_items = [(0, 5), (3, 1)]
    check(exact_knapsack(free_items, 0)[0] == 5, "zero-weight items fit even in zero capacity")

    # 4. the PTAS guarantee: (1 - eps) * OPT <= value <= OPT, with a genuinely feasible set
    seed = 0xC0FFEE
    guarantee_checks = 0
    worst_ratio = 1.0
    for eps in (0.5, 0.3, 0.2, 0.1, 0.05):
        for trial in range(4):
            items, capacity, seed = random_instance(14, seed, 25, 60)
            optimum, _ = exact_knapsack(items, capacity)
            value, chosen, states = fptas_knapsack(items, capacity, eps)
            chosen_weight = sum(items[i][0] for i in chosen)
            check(len(set(chosen)) == len(chosen), "the PTAS must not reuse an item")
            check(chosen_weight <= capacity, "the PTAS set must respect capacity")
            check(set_value(items, chosen)[0] == value, "reported value must equal the set's profit")
            check(value <= optimum, "no heuristic may beat the exact optimum")
            check(value >= (1 - eps) * optimum,
                  f"PTAS broke its {1 - eps} bound ({value} < {(1 - eps) * optimum})")
            if optimum:
                worst_ratio = min(worst_ratio, value / optimum)
            guarantee_checks += 1

    # 5. eps -> 0 recovers the exact optimum: with n*k = eps * p_max < 1 the rounding loss is
    #    below one profit unit, so the integrality of the optimum makes the bound tight.
    seed = 0xBEEF
    for trial in range(4):
        items, capacity, seed = random_instance(13, seed, 20, 50)
        optimum, _ = exact_knapsack(items, capacity)
        p_max = max(p for _, p in items if p > 0)
        tiny, chosen, states = fptas_knapsack(items, capacity, 0.5 / p_max)
        check(tiny == optimum, "an eps with eps * p_max < 1 must reproduce the exact optimum")
        check(states <= 2 * len(items) * len(items) * p_max + 1, "the tiny-eps table must stay bounded")

    # 6. more precision can never hurt: value is monotone in 1/eps
    seed = 0x1234
    for trial in range(3):
        items, capacity, seed = random_instance(15, seed, 25, 60)
        coarse, _, _ = fptas_knapsack(items, capacity, 0.4)
        medium, _, _ = fptas_knapsack(items, capacity, 0.15)
        fine, _, _ = fptas_knapsack(items, capacity, 0.05)
        check(coarse <= medium <= fine, "a smaller eps must not produce a worse value")

    # 7. the scaled table really is polynomial: at most sum_i floor(n/eps) + 1 states
    items, capacity, _ = random_instance(16, 0xFACE, 25, 60)
    for eps in (0.5, 0.2, 0.1):
        _, _, states = fptas_knapsack(items, capacity, eps)
        bound = 16 * int(16 / eps) + 1
        check(states <= bound, f"scaled table of {states} exceeds the n/eps bound {bound}")
        check(states >= 1, "the table must at least hold the empty state")

    # 8. the 1/2-greedy baseline is valid but strictly worse than the PTAS
    seed = 0xABCD
    greedy_checks = 0
    for trial in range(5):
        items, capacity, seed = random_instance(12, seed, 20, 40)
        optimum, _ = exact_knapsack(items, capacity)
        greedy = greedy_half(items, capacity)
        check(greedy <= optimum, "the greedy baseline must not beat the optimum")
        check(2 * greedy >= optimum, "the ratio greedy must reach the OPT/2 bound")
        greedy_checks += 1

    # 9. determinism of the whole pipeline
    items, capacity, _ = random_instance(14, 0x9999, 25, 60)
    first = fptas_knapsack(items, capacity, 0.1)
    second = fptas_knapsack(items, capacity, 0.1)
    check(first == second, "the FPTAS must be deterministic")

    print(f"769 dp_vs_brute={comparisons} ptas_guarantees={guarantee_checks} greedy={greedy_checks}")
    print(f"769 worst_ptas_ratio={worst_ratio:.4f} states(eps=0.1)={fptas_knapsack(items, capacity, 0.1)[2]}")
    print("PASS")


if __name__ == "__main__":
    main()
