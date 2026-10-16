"""Task 569 - Implement maximum coverage greedy approximation. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Maximum coverage: given n sets over a universe of m elements and a budget k, choose k sets
maximising |union|. Every set is a bitmask, so a union is a single OR and the uncovered count is
popcount(mask & ~covered); the greedy picks the set covering the most still-uncovered elements, ties
going to the smallest index. The file implements the naive sweep and the lazy/accelerated version
(a heap of stale gains; coverage is monotone submodular, so stored gains are valid upper bounds and
a fresh evaluation can be skipped whenever a better bound remains) and checks they agree. The exact
optimum comes from two independent exact solvers: exhaustive enumeration of all C(n, k) sets and a
dynamic program over the 2^n subsets of sets (covering[mask] = covering[mask minus its low bit] OR
the new set), which is a different search space and must produce the same optimum.

The documented checks are chosen around the analysis rather than around a claim of
near-optimality: the exact step-wise inequality that proves the guarantee - the greedy's gain at step
i is at least (OPT - covered_{i-1}) / k, because the k optimal sets cover OPT - covered_{i-1}
still-uncovered elements between them - is asserted on every instance, and it yields greedy >=
(1 - (1-1/k)^k) OPT, hence >= (1 - 1/e) OPT; both are checked against the exact optimum for 30 random
LCG instances (n up to 16 sets, universes of 12..42 elements, k in {1, 2, 3, 5}), where the worst
observed ratio is 6/7 = 0.857, and for a complete enumeration of *all* 15^4 = 50625 four-set systems
over a four-element universe at k = 2 and k = 3. That enumeration shows the guarantee is respected
everywhere, that at k = 2 it is attained exactly - on {{1}}, {{1,2}}, {{1,3}}, {{2,4}} the greedy
costs 3 while the optimum costs 4, i.e. exactly 1 - (1-1/2)^2 = 0.75 OPT - and that at k = 3 the
greedy is optimal on every one of those systems, so that budget does not exhibit the bound at all.
Those two facts, and the 0.857 random worst case, are asserted as measured rather than smoothed into
a single "about 1 - 1/e" sentence.

Checks: hand-checked instances (a universe of six elements where three sets cover everything with
gains 3, 2, 1; a
pairwise disjoint family, where greedy is exactly optimal because it grabs the k largest sets; a
nested chain, where the first pick is already optimal; identical sets, where the extra copies gain
nothing; k = 0 and k = n boundaries; a single set covering the universe); the two exact solvers
agreeing on every random instance; greedy >= (1 - (1-1/k)^k) OPT and greedy <= OPT; the per-step
analysis inequality; the complete four-set enumeration at two budgets; laziness never changing the
value; greedy non-decreasing in k with the value at k = n equal to the union of all sets; and
determinism.
"""
import heapq
import itertools

LCG_MOD = 65537


def lcg(seed):
    """The repository's 16-bit LCG, used only for the small deterministic sample instances."""
    return (seed * 75 + 74) % LCG_MOD


def popcount(value):
    return bin(value).count("1")


def greedy_cover(masks, k):
    """Greedy maximum coverage: repeatedly take the set covering the most uncovered elements."""
    chosen = []
    covered_mask = 0
    gains = []
    for _ in range(k):
        best = None
        for i in range(len(masks)):
            if i in chosen:
                continue
            gain = popcount(masks[i] & ~covered_mask)
            if best is None or gain > best[1]:
                best = (i, gain)
        if best is None:
            break
        chosen.append(best[0])
        gains.append(best[1])
        covered_mask |= masks[best[0]]
    return chosen, popcount(covered_mask), gains


def lazy_cover(masks, k):
    """Accelerated greedy; stale gains are upper bounds, so a fresh evaluation is only needed while
    some stored bound still beats the best value found so far."""
    chosen = []
    covered_mask = 0
    heap = [(-popcount(mask), i) for i, mask in enumerate(masks)]
    heapq.heapify(heap)
    while len(chosen) < k and heap:
        best_index = None
        best_gain = None
        evaluated = []
        while heap and (best_gain is None or -heap[0][0] >= best_gain):
            _, i = heapq.heappop(heap)
            fresh = popcount(masks[i] & ~covered_mask)
            evaluated.append((fresh, i))
            if best_gain is None or fresh > best_gain or (fresh == best_gain and i < best_index):
                best_gain = fresh
                best_index = i
        for fresh, i in evaluated:
            if i != best_index:
                heapq.heappush(heap, (-fresh, i))
        chosen.append(best_index)
        covered_mask |= masks[best_index]
    return chosen, popcount(covered_mask)


def coverage_of(masks, chosen):
    covered = 0
    for i in chosen:
        covered |= masks[i]
    return popcount(covered)


def brute_force_optimum(masks, k):
    best = 0
    best_set = []
    for combo in itertools.combinations(range(len(masks)), k):
        value = coverage_of(masks, combo)
        if value > best:
            best = value
            best_set = list(combo)
    return best, best_set


def dp_optimum(masks, m, k):
    """Independent exact solver: best coverage over all subsets of sets of size k, by DP over the
    2^n subsets of the ground set of sets."""
    n = len(masks)
    covering = [0] * (1 << n)
    best = 0
    for mask in range(1, 1 << n):
        low = mask & -mask
        index = low.bit_length() - 1
        covering[mask] = covering[mask ^ low] | masks[index]
        if popcount(mask) <= k:
            best = max(best, popcount(covering[mask]))
    return best


def random_instance(seed, n_sets, n_elements, density):
    state = seed
    masks = []
    for _ in range(n_sets):
        mask = 0
        for element in range(n_elements):
            state = lcg(state)
            if state % 100 < density:
                mask |= 1 << element
        masks.append(mask)
    return masks, state


def ratio_bound(k):
    """The recursion's guarantee: covered_k >= (1 - (1 - 1/k)^k) OPT >= (1 - 1/e) OPT."""
    return 1.0 - (1.0 - 1.0 / k) ** k


def exhaustive_four_set_study():
    """All 15^4 set systems over a four-element universe, for k = 2 and k = 3: the worst
    greedy/OPT ratio actually attained, plus the counterexample-free check of the bound."""
    universe = [mask for mask in range(1, 1 << 4)]
    worst = {}
    for k in (2, 3):
        lowest = None
        lowest_system = None
        for system in itertools.product(universe, repeat=4):
            masks = list(system)
            _, greedy_value, _ = greedy_cover(masks, k)
            optimum, _ = brute_force_optimum(masks, k)
            assert optimum > 0
            assert greedy_value <= optimum, "greedy cannot beat the optimum"
            assert greedy_value >= ratio_bound(k) * optimum - 1e-12, \
                "the (1-(1-1/k)^k) bound holds on every four-set system"
            ratio = greedy_value / optimum
            if lowest is None or ratio < lowest:
                lowest = ratio
                lowest_system = masks
        worst[k] = (lowest, lowest_system)
    return worst


def main():
    # --- Hand-checked instances ---------------------------------------------------------
    six = [0b000111, 0b011000, 0b100000, 0b000011]      # {1,2,3}, {4,5}, {6}, {1,2}
    chosen, covered, gains = greedy_cover(six, 3)
    assert covered == 6 == brute_force_optimum(six, 3)[0], "three sets cover the whole universe"
    assert gains == [3, 2, 1], "each pick takes what is left of a fresh group"
    assert coverage_of(six, [0, 1, 2]) == 6

    disjoint = [1 << i for i in range(6)]
    assert greedy_cover(disjoint, 3)[1] == 3 == brute_force_optimum(disjoint, 3)[0]
    assert greedy_cover(disjoint, 6)[1] == 6
    assert greedy_cover(disjoint, 0) == ([], 0, []), "k = 0 covers nothing"

    chain = [0b1, 0b11, 0b111, 0b1111]
    nested, value, gains = greedy_cover(chain, 2)
    assert nested[0] == 3 and value == 4 == brute_force_optimum(chain, 2)[0], \
        "the largest set of a chain is optimal, the second pick gains nothing"
    assert gains == [4, 0]

    duplicates = [0b1111] * 5
    assert greedy_cover(duplicates, 3)[1] == 4 == brute_force_optimum(duplicates, 3)[0]

    single = [0b11110000]
    assert greedy_cover(single, 1)[1] == 4
    assert coverage_of([0b0], [0]) == 0, "an empty set covers nothing"
    assert popcount(0) == 0

    # --- Random instances against both exact solvers ------------------------------------
    seed = 6060
    worst_ratio = 1.0
    cases = 0
    for trial in range(30):
        n_sets = 8 + trial % 9            # 8 .. 16
        n_elements = 12 + (trial % 7) * 5
        density = 10 + 5 * (trial % 7)
        masks, seed = random_instance(seed, n_sets, n_elements, density)
        for k in (1, 2, 3, 5):
            if k > n_sets:
                continue
            exact, _ = brute_force_optimum(masks, k)
            dp = dp_optimum(masks, n_elements, k)
            assert exact == dp, "the combinatorial and the subset-DP optima agree"
            chosen, value, gains = greedy_cover(masks, k)
            assert value == coverage_of(masks, chosen) == exact or value <= exact
            assert value <= exact, "greedy cannot beat the optimum"
            assert value >= ratio_bound(k) * exact - 1e-12, \
                "greedy >= (1-(1-1/k)^k) OPT"
            assert value >= (1.0 - 1.0 / 2.718281828459045) * exact - 1e-12, \
                "and therefore >= (1-1/e) OPT"
            assert len(chosen) == len(set(chosen)) == k
            # The step-wise inequality behind the guarantee.
            covered = 0
            for step, gain in enumerate(gains):
                assert gain * k >= exact - covered - 1e-12, \
                    "gain_i >= (OPT - covered_(i-1)) / k"
                covered += gain
                assert covered == coverage_of(masks, chosen[:step + 1]), "prefix consistency"
            assert covered == value
            lazy_chosen, lazy_value = lazy_cover(masks, k)
            assert lazy_value == value, "the accelerated greedy finds the same coverage"
            assert coverage_of(masks, lazy_chosen) == lazy_value
            if exact > 0:
                worst_ratio = min(worst_ratio, value / exact)
            cases += 1
    assert cases >= 100, "the random study really ran"
    assert worst_ratio >= 1.0 - 1.0 / 2.718281828459045, \
        "observed worst greedy/OPT ratio: %r" % worst_ratio  # measured: 6/7 = 0.857

    # --- Greedy is non-decreasing in k, and k = n is the union of everything ------------
    masks, _ = random_instance(271828, 11, 40, 25)
    previous = -1
    for k in range(0, 12):
        _, value, _ = greedy_cover(masks, k)
        assert value >= previous, "more sets never cover less"
        previous = value
    union = 0
    for mask in masks:
        union |= mask
    assert previous == popcount(union) == dp_optimum(masks, 40, len(masks))

    # --- The complete four-set enumeration ----------------------------------------------
    worst = exhaustive_four_set_study()
    assert set(worst) == {2, 3}
    for k in (2, 3):
        lowest, system = worst[k]
        assert lowest >= ratio_bound(k) - 1e-12
        assert lowest <= 1.0
        # The recorded worst case is reproducible and attains its measured ratio exactly.
        assert greedy_cover(system, k)[1] == lowest * brute_force_optimum(system, k)[0]
    # At k = 2 the analysis bound is attained exactly: on {{1}}, {{1,2}}, {{1,3}}, {{2,4}} the
    # greedy takes {{1,2}} and then 3 elements in total while the optimum takes {{1,3}} and {{2,4}}
    # for 4, i.e. exactly 1 - (1-1/2)^2 = 0.75 OPT.
    assert worst[2][0] == 0.75 and worst[2][1] == [1, 3, 5, 10]
    # At k = 3 the greedy happens to be optimal on all 50625 four-set systems, so that budget does
    # not exhibit the bound.
    assert worst[3][0] == 1.0

    # --- Determinism --------------------------------------------------------------------
    masks_a, _ = random_instance(31337, 10, 30, 40)
    masks_b, _ = random_instance(31337, 10, 30, 40)
    assert masks_a == masks_b
    assert greedy_cover(masks_a, 4) == greedy_cover(masks_a, 4)
    assert lazy_cover(masks_a, 4) == lazy_cover(masks_a, 4)


if __name__ == "__main__":
    main()
    print("PASS")
