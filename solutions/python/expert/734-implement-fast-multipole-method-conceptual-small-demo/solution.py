"""Task 734 - Implement fast multipole method conceptual small demo.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

The fast multipole method evaluates N sources at N targets in O(N log N) by summarising clusters:
a distant group of charges is replaced by a short *multipole expansion* about the group's centre,
and that expansion is translated into a *local expansion* about the target's centre, which is then
evaluated once per target in the cluster.  Three translation operators make the hierarchy work:

    M2M  shift a multipole from a child centre to its parent centre  (upward pass)
    M2L  convert a well-separated source multipole into a local expansion  (downward pass)
    L2L  shift a local expansion from a parent centre to a child centre

All three are implemented here for the 1D log kernel phi(x) = sum_j q_j log|x - x_j|, using the
complex representation F(z) = sum_j q_j log(z - z_j) whose real part is exactly that potential on
the real axis (the branch jump of the logarithm is purely imaginary, so it cancels under Re).  A
multipole about c is (Q, a_1..a_p) with a_k = (1/k) sum_j q_j (z_j - c)^k and
F(z) = Q log(z - c) - sum_k a_k (z - c)^{-k}; a local expansion about c is the Taylor polynomial
sum_n b_n (z - c)^n about the target centre.

Verified invariants: the truncated multipole series converges to the direct sum as the order
grows, M2M reproduces a multipole computed directly about the parent centre (and merging two
shifted multipoles equals the multipole of the union), M2L reproduces the source multipole's own
values near the target centre, L2L preserves the local expansion's values, the end-to-end
upward/downward pass matches the direct sum and improves with order, the near field is handled
exactly, no node is ever its neighbour's far-field source, and every source sits in exactly one
leaf.

Simplified: a 1D layout on a binary median-split tree, one fixed order for the whole hierarchy, no
adaptive order selection, no periodic images and no multipole acceptance criterion beyond strict
separation - the goal is to exercise every translation operator against a direct oracle, not to
beat a tuned treecode.
"""
import cmath
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state, low, high):
    state, r = splitmix64(state)
    return state, low + (high - low) * ((r >> 11) / float(1 << 53))


def binomials(limit):
    table = [[0] * (limit + 1) for _ in range(limit + 1)]
    for n in range(limit + 1):
        table[n][0] = 1
        for k in range(1, n + 1):
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k]
    return table


BINOMIAL = binomials(64)


def choose(n, k):
    if n < 0 or k < 0 or k > n:
        return 0
    return BINOMIAL[n][k]


class Source:
    """A point charge on the real axis, stored as a complex coordinate."""

    def __init__(self, position, charge):
        self.z = complex(position, 0.0)
        self.charge = charge


def zero_net_load(count, seed, extent=1.0):
    """Charges summing to zero: the log kernel needs that for a decaying potential."""
    sources, state = [], seed & MASK64
    for _ in range(count // 2):
        state, magnitude = uniform(state, 0.5, 1.5)
        for charge in (magnitude, -magnitude):
            state, x = uniform(state, -extent, extent)
            sources.append(Source(x, charge))
    sources.sort(key=lambda source: source.z.real)
    return sources, state


# ---------------------------------------------------------------------------- expansions
def midpoint_centre(points):
    """Safest multipole centre: the midpoint of the cluster's extent (works for neutral clusters)."""
    positions = [position.real for position, _ in points]
    return complex(0.5 * (min(positions) + max(positions)), 0.0)


def multipole(points, order, centre=None):
    """Coefficients (Q, a_1..a_p) of the multipole about `centre`; the direct definition."""
    if not points:
        raise ValueError("an empty cluster has no multipole")
    if centre is None:
        centre = midpoint_centre(points)
    centre = complex(centre, 0.0)
    coefficients = [sum(charge for _, charge in points)]
    for k in range(1, order + 1):
        coefficients.append(sum(charge * (complex(position, 0.0) - centre) ** k
                               for position, charge in points) / k)
    return centre, coefficients


def multipole_value(centre, coefficients, z):
    """F(z) = Q log(z - c) - sum_k a_k (z - c)^{-k}."""
    return coefficients[0] * cmath.log(z - centre) - sum(
        coefficients[k] / (z - centre) ** k for k in range(1, len(coefficients)))


def m2m(child_centre, coefficients, parent_centre, order):
    """Shift a multipole from the child centre to the parent centre (upward pass)."""
    shift = parent_centre - child_centre
    out = [coefficients[0]]
    for m in range(1, order + 1):
        term = coefficients[0] * (-shift) ** m / m
        for t in range(1, m + 1):
            term += choose(m - 1, t - 1) * coefficients[t] * (-shift) ** (m - t)
        out.append(term)
    return out


def m2l(source_centre, coefficients, target_centre, order):
    """Convert a well-separated source multipole into a local expansion about the target."""
    d = target_centre - source_centre
    local = [coefficients[0] * cmath.log(d)
             - sum(coefficients[k] / d ** k for k in range(1, len(coefficients)))]
    for n in range(1, order + 1):
        term = coefficients[0] * (-1) ** (n + 1) / (n * d ** n)
        for k in range(1, len(coefficients)):
            term -= coefficients[k] * (-1) ** n * choose(k + n - 1, n) * d ** (-(k + n))
        local.append(term)
    return local


def l2l(parent_centre, coefficients, child_centre, order):
    """Shift a local expansion from the parent centre to the child centre (downward pass)."""
    shift = child_centre - parent_centre
    return [sum(choose(m, n) * coefficients[m] * shift ** (m - n)
                for m in range(n, len(coefficients))) for n in range(order + 1)]


def local_value(centre, coefficients, z):
    return sum(coefficients[n] * (z - centre) ** n for n in range(len(coefficients)))


# ---------------------------------------------------------------------------- tree
class Node:
    """Binary tree node over a sorted slice of sources; leaves hold up to `leaf_size` charges."""

    def __init__(self, sources, leaf_size, depth=0):
        if leaf_size < 1:
            raise ValueError("leaf size must be positive")
        self.sources = list(sources)
        self.depth = depth
        self.children = []
        self.parent = None
        self.centre = None
        self.multipole = None
        self.interaction = []
        if len(self.sources) > leaf_size:
            half = len(self.sources) // 2
            self.children = [Node(self.sources[:half], leaf_size, depth + 1),
                             Node(self.sources[half:], leaf_size, depth + 1)]
            for child in self.children:
                child.parent = self
            self.sources = []

    def is_leaf(self):
        return not self.children

    def leaves(self):
        if self.is_leaf():
            return [self]
        out = []
        for child in self.children:
            out.extend(child.leaves())
        return out

    def nodes(self):
        out = [self]
        for child in self.children:
            out.extend(child.nodes())
        return out

    def all_sources(self):
        if self.is_leaf():
            return list(self.sources)
        out = []
        for child in self.children:
            out.extend(child.all_sources())
        return out


def level_lists(tree):
    """Nodes bucketed by depth and ordered left to right (that order defines neighbours)."""
    buckets = {}
    for node in tree.nodes():
        buckets.setdefault(node.depth, []).append(node)
    for bucket in buckets.values():
        bucket.sort(key=lambda node: min((source.z.real for source in node.all_sources()),
                                         default=node.centre.real if node.centre is not None else 0.0))
    return buckets


def prepare(tree, order):
    """Upward pass (M2M into the parents) and the interaction lists for the downward pass.

    Greengard's rule: the interaction list of a child is the set of children of the *parent's
    neighbours* that are not neighbours of the child itself, which in 1D leaves the far cousin
    of every node.
    """
    for node in reversed(tree.nodes()):
        if node.is_leaf():
            points = [(source.z, source.charge) for source in node.sources]
            if points:
                node.centre, node.multipole = multipole(points, order, midpoint_centre(points))
            else:
                node.centre, node.multipole = 0j, [0j] * (order + 1)
        else:
            centre = (node.children[0].centre + node.children[1].centre) / 2.0
            shifted = [m2m(child.centre, child.multipole, centre, order)
                       for child in node.children]
            node.centre = centre
            node.multipole = [sum(part[k] for part in shifted) for k in range(order + 1)]
    buckets = level_lists(tree)
    neighbours_of = {}
    for bucket in buckets.values():
        for index, node in enumerate(bucket):
            neighbours_of[id(node)] = [bucket[j] for j in (index - 1, index + 1)
                                       if 0 <= j < len(bucket)]
    for bucket in buckets.values():
        for node in bucket:
            if node.parent is None:
                node.interaction = []
                continue
            allowed = [child for other in neighbours_of[id(node.parent)]
                       for child in other.children]
            near = {id(other) for other in neighbours_of[id(node)]} | {id(node)}
            node.interaction = [other for other in allowed if id(other) not in near]
    return tree


def fmm_potentials(tree, order):
    """Downward pass; returns {id(source): potential} for every source."""
    buckets = level_lists(tree)
    near_particles = {}
    for bucket in buckets.values():
        for index, node in enumerate(bucket):
            if not node.is_leaf():
                continue
            near = []
            for j in (index - 1, index, index + 1):
                if 0 <= j < len(bucket):
                    near.extend(bucket[j].all_sources())
            near_particles[id(node)] = near
    result = {}

    def descend(node, local):
        if node.is_leaf():
            near = near_particles[id(node)]
            for target in node.sources:
                value = local_value(node.centre, local, target.z).real
                for other in near:
                    if other is not target:
                        value += other.charge * math.log(abs(target.z.real - other.z.real))
                result[id(target)] = value
            return
        for child in node.children:
            shifted = list(l2l(node.centre, local, child.centre, order))
            for source_node in child.interaction:
                translated = m2l(source_node.centre, source_node.multipole, child.centre, order)
                shifted = [shifted[n] + translated[n] for n in range(order + 1)]
            descend(child, shifted)

    descend(tree, [0j] * (order + 1))
    return result


def reference_potentials(tree):
    """Direct O(N^2) oracle over every source in the tree."""
    everything = tree.all_sources()
    return {id(target): sum(other.charge * math.log(abs(target.z.real - other.z.real))
                            for other in everything if other is not target)
            for target in everything}


def relative_difference(left, right):
    numerator = math.sqrt(sum(abs(left[key] - right[key]) ** 2 for key in left))
    denominator = math.sqrt(sum(abs(value) ** 2 for value in right.values()))
    return numerator / denominator


def main():
    sources, seed = zero_net_load(24, 0x7341)
    check(abs(sum(source.charge for source in sources)) < 1e-12,
          "the test load must have zero net charge")

    # 1. the multipole series: truncating at order p leaves an error that falls with p, and the
    #    series must match the direct sum for a well-separated target.
    cluster = [(source.z, source.charge) for source in sources[:6]]
    centre, _ = multipole(cluster, 1)
    far_targets = [complex(4.0 + 0.5 * i, 0.0) for i in range(4)]
    errors = []
    for order in (2, 4, 6, 10):
        _, coefficients = multipole(cluster, order)
        error = 0.0
        for target in far_targets:
            exact = sum(charge * cmath.log(target - position) for position, charge in cluster)
            error = max(error, abs(multipole_value(centre, coefficients, target) - exact))
        errors.append(error)
    check(all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)),
          "the truncation error must fall monotonically with the order")
    check(errors[-1] < 1e-6 * errors[0], "a ten-term expansion must be far more accurate")
    check(errors[0] > 0.0, "the lowest order must not accidentally be exact")

    # 2. M2M: the shifted multipole must equal a multipole computed directly about the parent
    #    centre, and merging two shifted multipoles must equal the multipole of the union.
    left = [(source.z, source.charge) for source in sources[:3]]
    right = [(source.z, source.charge) for source in sources[3:6]]
    child_centre, child_coefficients = multipole(left, 8)
    right_centre, right_coefficients = multipole(right, 8)
    parent_centre = child_centre + 0.75
    shifted = m2m(child_centre, child_coefficients, parent_centre, 8)
    _, direct_coefficients = multipole(left, 8, parent_centre)
    check(max(abs(shifted[k] - direct_coefficients[k]) for k in range(1, 9)) < 1e-9,
          "M2M must reproduce a direct multipole about the parent centre")
    union_centre = 0.5 * (child_centre + right_centre)
    merged = [m2m(child_centre, child_coefficients, union_centre, 8)[k]
              + m2m(right_centre, right_coefficients, union_centre, 8)[k] for k in range(9)]
    _, union_direct = multipole(left + right, 8, union_centre)
    check(max(abs(merged[k] - union_direct[k]) for k in range(9)) < 1e-9,
          "merging two shifted multipoles must equal the multipole of the union")

    # 3. M2L: the translated local expansion must reproduce the source multipole near the target
    #    centre - exactly what the downward pass relies on.
    source_centre, source_coefficients = multipole(cluster, 8)
    target_centre = source_centre + 3.0
    local = m2l(source_centre, source_coefficients, target_centre, 8)
    probes = [target_centre + complex(0.05 * i, 0.0) for i in range(-2, 3)]
    worst = max(abs(local_value(target_centre, local, probe)
                    - multipole_value(source_centre, source_coefficients, probe))
                for probe in probes)
    check(worst < 1e-9, "M2L must reproduce the source multipole near the target centre")

    # 4. L2L: shifting a local expansion must preserve its values at the child centre.
    child_local = l2l(target_centre, local, target_centre + 0.25, 8)
    probe = target_centre + complex(0.3, 0.0)
    check(abs(local_value(target_centre + 0.25, child_local, probe)
              - local_value(target_centre, local, probe)) < 1e-9,
          "L2L must preserve the local expansion's values")

    # 5. end to end: the upward/downward pass must match the direct sum, improve with the order,
    #    never use a neighbour as a far-field source, and store every source once.
    tree = prepare(Node(sources, 3), 10)
    reference = reference_potentials(tree)
    potentials = fmm_potentials(tree, 10)
    check(len(potentials) == len(sources), "every source must receive a potential")
    error = relative_difference(potentials, reference)
    check(error < 1e-4, "the FMM potential must agree with the direct sum")
    quality = [relative_difference(fmm_potentials(prepare(Node(sources, 3), order), order),
                                  reference) for order in (2, 6, 10)]
    check(quality[-1] < quality[0], "a higher order must improve the end-to-end accuracy")
    check(all(value < 5e-2 for value in quality), "even order two must stay within a few percent")
    for bucket in level_lists(tree).values():
        for index, node in enumerate(bucket):
            near = {id(bucket[j]) for j in (index - 1, index + 1) if 0 <= j < len(bucket)}
            check(not near.intersection({id(other) for other in node.interaction}),
                  "a node may never be its neighbour's far-field source")
    stored = [source for leaf in tree.leaves() for source in leaf.sources]
    check(len(stored) == len(sources) and len({id(source) for source in stored}) == len(sources),
          "every source must sit in exactly one leaf")
    check(all(node.multipole is not None for node in tree.nodes()),
          "the upward pass must fill every node")

    # 6. boundary cases and rejected inputs.
    lone = [Source(-0.25, 1.0), Source(0.25, -1.0)]
    lone_values = fmm_potentials(prepare(Node(lone, 1), 4), 4)
    check(abs(lone_values[id(lone[0])] + math.log(0.5)) < 1e-9
          and abs(lone_values[id(lone[1])] - math.log(0.5)) < 1e-9,
          "the two-charge pair must reproduce the exact Coulomb logarithms")
    single = [Source(0.0, 1.0), Source(0.5, -1.0), Source(1.0, 1.0)]
    single_values = fmm_potentials(prepare(Node(single, 3), 6), 6)
    check(len(single_values) == 3, "a single leaf must still produce every potential")
    empty = prepare(Node([], 2), 4)
    check(fmm_potentials(empty, 4) == {}, "an empty load must produce no potentials")
    expect_value_error(lambda: Node(sources, 0), "a zero leaf size is invalid")
    expect_value_error(lambda: multipole([], 4), "an empty cluster has no multipole")

    # 7. determinism.
    again = fmm_potentials(prepare(Node(sources, 3), 10), 10)
    check(all(abs(again[key] - potentials[key]) < 1e-15 for key in potentials),
          "the same load and order must give the same potentials")

    print(f"734 sources={len(sources)} relative_error={error:.3e} "
          f"order_sweep={[f'{value:.3e}' for value in quality]}")
    print(f"734 multipole sweep={[f'{value:.3e}' for value in errors]} "
          f"m2l_error={worst:.2e} leaves={len(tree.leaves())}")
    print("PASS")


if __name__ == "__main__":
    main()
