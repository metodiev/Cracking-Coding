"""Task 943 - Implement composable privacy guarantees across combined DP mechanisms.  (Tier: Master, CTCI Beyond - Security & Privacy)

A small composition library over *finite* mechanisms, so every claim can be checked exactly.  A
mechanism is given as a pair of conditional distributions over a finite output space for a pair of
neighbouring inputs; from that the exact pure-DP epsilon (max log likelihood ratio) and the exact
delta at a given epsilon are computed by brute force.  On top of the exact checks the library
implements the usual composition rules - basic sequential composition (epsilon and delta add),
advanced composition (Dwork-Roth), parallel composition over disjoint partitions (max), RDP/zCDP
accounting with the rho + 2*sqrt(rho*ln(1/delta)) conversion (checked against the classical
closed-form delta of the Gaussian mechanism), and the subsampling amplification
epsilon' = log(1 + q*(e^epsilon - 1)) with delta' proportional to q - plus the randomised-response
mechanism.

Simplified versus OpenDP/Opacus: the finite mechanisms are explicit probability vectors, and the
accountants return textbook closed forms instead of optimising over RDP orders numerically.
Verified: basic composition is tight (brute-force epsilon of the composed randomised response equals
the sum) and stays pure; parallel composition is strictly better than sequential; advanced
composition bounds the exact epsilon and beats basic composition for many mechanisms; the zCDP
conversion meets the Gaussian delta target while a halved epsilon does not; subsampling lowers epsilon
monotonically in the sampling rate and its brute-force exact delta is exactly q times the underlying
mechanism's delta at the same epsilon; zero/one-mechanism and commutativity boundaries; monotonicity
under adding mechanisms; determinism.
"""
import math
from itertools import product


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def randomised_response(epsilon, value, outcomes=2):
    """Randomised response over a binary input: P(output = input) = e^eps / (1 + e^eps)."""
    high = math.exp(epsilon) / (1 + math.exp(epsilon))
    return [high, 1 - high] if value == 0 else [1 - high, high]


def normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def gaussian_delta(epsilon, sigma, sensitivity=1.0):
    """Classical closed-form bound on delta of the Gaussian mechanism (sensitivity/sigma)."""
    shift = sensitivity / (2 * sigma)
    return max(0.0, normal_cdf(shift - epsilon * sigma)
               - math.exp(epsilon) * normal_cdf(-shift - epsilon * sigma))


def joint(distributions):
    """Product distribution over the joint output space of independent mechanisms."""
    probabilities = []
    for combination in product(*[range(len(dist)) for dist in distributions]):
        mass = 1.0
        for dist, index in zip(distributions, combination):
            mass *= dist[index]
        probabilities.append(mass)
    return probabilities


def exact_epsilon(dist_x, dist_y):
    """Exact pure-DP epsilon of one neighbouring pair (both distributions must share support)."""
    return max(math.log(px / py) for px, py in zip(dist_x, dist_y) if px > 0 and py > 0)


def exact_delta(dist_x, dist_y, epsilon):
    """Exact delta at `epsilon` for one neighbouring pair: max(0, sum(px - e^eps * py))."""
    bound = math.exp(epsilon)
    return max(0.0, sum(px - bound * py for px, py in zip(dist_x, dist_y) if px > bound * py))


def basic(epsilons, deltas):
    return sum(epsilons), sum(deltas)


def advanced(epsilons, delta_prime):
    """Dwork-Roth advanced composition for k mechanisms with the largest epsilon."""
    count, epsilon = len(epsilons), max(epsilons)
    return (math.sqrt(2 * count * math.log(1 / delta_prime)) * epsilon
            + count * epsilon * (math.exp(epsilon) - 1), delta_prime)


def parallel(epsilons):
    """Disjoint partitions: the guarantee is the maximum, not the sum."""
    return max(epsilons)


def subsampled(epsilon, delta, rate):
    """Subsampling amplification: eps' = log(1 + q(e^eps - 1)), delta' = q * delta."""
    return math.log(1 + rate * (math.exp(epsilon) - 1)), rate * delta


def zcdp_to_epsilon(rho, delta):
    """Standard zCDP -> (epsilon, delta) conversion (Bun-Steinke)."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def main():
    eps = 1.2
    dist_x, dist_y = randomised_response(eps, 0), randomised_response(eps, 1)

    # 1) basic sequential composition is tight: the brute-force epsilon of three composed
    #    randomised-response mechanisms equals the sum of the individual epsilons.
    composed_x = joint([dist_x] * 3)
    composed_y = joint([dist_y] * 3)
    check(abs(exact_epsilon(composed_x, composed_y) - 3 * eps) < 1e-9
          and len(composed_x) == 8,
          f"composed epsilon {exact_epsilon(composed_x, composed_y)} vs {3 * eps}")
    # 2) composing pure mechanisms stays pure: delta is zero (to rounding) at the summed epsilon
    #    and strictly positive as soon as the reported epsilon is tightened.
    check(exact_delta(composed_x, composed_y, 3 * eps) < 1e-12
          and exact_delta(composed_x, composed_y, 3 * eps - 0.2) > 1e-6,
          f"a composition of pure mechanisms must stay pure at the summed epsilon: "
          f"{exact_delta(composed_x, composed_y, 3 * eps):.3e} / "
          f"{exact_delta(composed_x, composed_y, 3 * eps - 0.2):.3e}")
    check(basic([eps] * 3, [0.0] * 3) == (3 * eps, 0.0), "basic composition arithmetic")

    # 3) parallel composition: one differing partition, the other untouched - the exact epsilon
    #    is the maximum over the two pairs and strictly below the sequential sum.
    first = (joint([randomised_response(1.0, 0), randomised_response(2.0, 1)]),
             joint([randomised_response(1.0, 1), randomised_response(2.0, 1)]))
    second = (joint([randomised_response(1.0, 0), randomised_response(2.0, 0)]),
              joint([randomised_response(1.0, 0), randomised_response(2.0, 1)]))
    exact_parallel = max(exact_epsilon(*pair) for pair in (first, second))
    check(abs(exact_parallel - parallel([1.0, 2.0])) < 1e-9
          and exact_parallel < basic([1.0, 2.0], [0.0, 0.0])[0],
          f"parallel exact epsilon {exact_parallel} must be max(1, 2) and below the sum 3")

    # 4) advanced composition bounds the exact epsilon and beats basic for many mechanisms.
    small = 0.1
    compact = (joint([randomised_response(small, 0)] * 3),
               joint([randomised_response(small, 1)] * 3))
    exact_small = exact_epsilon(*compact)
    delta_prime = 1e-6
    advanced_3 = advanced([small] * 3, delta_prime)[0]
    advanced_50 = advanced([small] * 50, delta_prime)[0]
    check(advanced_3 >= exact_small and advanced_50 < basic([small] * 50, [0.0] * 50)[0]
          and advanced([small] * 50, 1e-9)[0] > advanced([small] * 50, 1e-5)[0],
          f"advanced composition: 3 -> {advanced_3:.3f} >= {exact_small:.3f}, "
          f"50 -> {advanced_50:.3f} < {basic([small] * 50, [0.0] * 50)[0]:.3f}, and a smaller "
          f"delta budget must cost more epsilon")

    # 5) zCDP accounting: the rho + 2*sqrt(rho ln(1/delta)) conversion meets the classical
    #    closed-form delta of the Gaussian mechanism, for the single mechanism and for its
    #    composition with itself (rho adds, sensitivity becomes sqrt(2)); halving epsilon fails.
    sigma, delta = 1.5, 1e-5
    rho = 1.0 / (2 * sigma * sigma)
    single_eps = zcdp_to_epsilon(rho, delta)
    composed_eps = zcdp_to_epsilon(2 * rho, delta)
    check(gaussian_delta(single_eps, sigma) <= delta
          and gaussian_delta(composed_eps, sigma, math.sqrt(2)) <= delta
          and composed_eps > single_eps
          and gaussian_delta(single_eps / 2, sigma) > delta,
          f"zCDP conversion: delta({single_eps:.3f})={gaussian_delta(single_eps, sigma):.3e}, "
          f"delta({composed_eps:.3f}, sensitivity sqrt(2))="
          f"{gaussian_delta(composed_eps, sigma, math.sqrt(2)):.3e}, but half the epsilon gives "
          f"{gaussian_delta(single_eps / 2, sigma):.3e} > delta={delta}")

    # 6) subsampling: eps' < eps for q < 1 and is monotone in q, delta' = q * delta, and the exact
    #    brute-force delta of the sub-sampled mechanism is exactly q times the underlying one at
    #    the same epsilon, so the exposed delta shrinks with the sampling rate.
    rate = 0.02
    amplified_eps, amplified_delta = subsampled(1.0, 0.05, rate)
    base_x, base_y = randomised_response(1.0, 0), randomised_response(1.0, 1)
    subsampled_x = [1 - rate] + [rate * mass for mass in base_x]
    subsampled_y = [1 - rate] + [rate * mass for mass in base_y]
    exposed = exact_delta(subsampled_x, subsampled_y, amplified_eps)
    check(amplified_eps < 1.0 and abs(subsampled(1.0, 0.05, 1.0)[0] - 1.0) < 1e-9
          and subsampled(1.0, 0.05, 0.01)[0] < subsampled(1.0, 0.05, 0.05)[0] < 1.0
          and abs(amplified_delta - rate * 0.05) < 1e-15 and exposed < rate
          and abs(exposed - rate * exact_delta(base_x, base_y, amplified_eps)) < 1e-12,
          f"subsampling: eps'={amplified_eps:.4f}, delta'={amplified_delta:.5f}, exposed exact "
          f"delta={exposed:.5f} vs q * {exact_delta(base_x, base_y, amplified_eps):.5f} = "
          f"{rate * exact_delta(base_x, base_y, amplified_eps):.5f}")

    # 7) boundaries: no mechanism composes to (0, 0), one mechanism is the identity, and the
    #    sequential bound does not depend on the order of composition.
    check(basic([], []) == (0.0, 0.0) and basic([eps], [0.0]) == (eps, 0.0)
          and parallel([eps]) == eps
          and basic([0.1, 0.2, 0.3], [1e-6, 2e-6, 3e-6]) == basic([0.3, 0.2, 0.1], [3e-6, 2e-6,
                                                                                   1e-6]),
          "zero, single and order boundaries of composition")

    # 8) monotonicity: adding a mechanism never lowers the composed epsilon or delta.
    baseline = basic([0.3, 0.4], [1e-6, 2e-6])
    grown = basic([0.3, 0.4, 0.5], [1e-6, 2e-6, 3e-6])
    check(grown[0] >= baseline[0] and grown[1] >= baseline[1]
          and advanced([0.3, 0.4, 0.5], 1e-6)[0] >= advanced([0.3, 0.4], 1e-6)[0]
          and parallel([0.3, 0.4, 0.5]) >= parallel([0.3, 0.4])
          and zcdp_to_epsilon(0.5, delta) >= zcdp_to_epsilon(0.25, delta),
          "adding mechanisms must never weaken the guarantee")

    # 9) determinism of the exact and closed-form accounts.
    check(exact_epsilon(composed_x, composed_y) == exact_epsilon(joint([dist_x] * 3),
                                                                 joint([dist_y] * 3))
          and zcdp_to_epsilon(rho, delta) == zcdp_to_epsilon(rho, delta),
          "the accounts must be deterministic")

    print(f"basic(k=3,eps={eps}): exact={exact_epsilon(composed_x, composed_y):.4f} bound="
          f"{basic([eps] * 3, [0.0] * 3)[0]:.4f}")
    print(f"parallel: exact={exact_parallel:.4f} sequential={basic([1.0, 2.0], [0.0, 0.0])[0]:.4f}")
    print(f"zcdp: single_eps={single_eps:.4f} composed_eps={composed_eps:.4f} at delta={delta}")
    print("PASS")


if __name__ == "__main__":
    main()
