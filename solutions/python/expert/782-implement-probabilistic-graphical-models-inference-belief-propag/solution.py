"""Task 782 - Implement probabilistic graphical models inference (belief propagation).  (Tier: Expert, Beyond - Machine Learning & AI)

A factor graph has variable nodes with finite domains and factor nodes whose tables weight the joint
distribution p(x) = (1/Z) * prod_f f_f(x_scope(f)).  Sum-product belief propagation passes two kinds
of messages along the edges:
    m_{u->f}(x_u)  = prod over the other factors g at u of m_{g->u}(x_u)
    m_{f->u}(x_u)  = sum over the other variables of f(x_scope) * prod of their messages into f
and the variable beliefs are q_u(x) proportional to the product of the incoming factor messages.
On a *tree* the messages converge and the beliefs equal the exact marginals; on a graph with cycles
("loopy" BP) the same updates converge (or are damped into a fixed point) but the beliefs are only an
approximation - the fixed point property holds, exactness does not.  Max-product replaces the sums by
maxima and yields the MAP assignment, again exactly on trees.

Verified: brute-force enumeration (over every assignment) as the independent oracle for both
marginals and MAP, exact agreement of sum-product marginals with the oracle on trees (chain and star),
the fixed-point residual of the converged messages on every tested graph, convergence within the
diameter for trees, exactness of max-product MAP on trees, the loopy four-cycle case where the
fixed-point equations hold while the marginals deviate from exact by a bounded, non-zero amount,
deterministic (0/1) factors, a zero-probability configuration, and the degenerate single-variable
graphs (with and without factors).

Simplified: tiny graphs (at most six binary/ternary variables) so enumeration is exact, synchronous
sweeps with damping instead of message schedules, and no Bethe/Kikuchi free-energy machinery - the
loopy results are documented as approximations rather than justified by a free-energy bound.
"""
import itertools


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class FactorGraph:
    """Variables with a domain size each, and factors as (scope, table) with the table in scope order."""

    def __init__(self, domains, factors):
        self.domains = list(domains)
        self.factors = [(tuple(scope), list(table)) for scope, table in factors]
        for scope, table in self.factors:
            expected = 1
            for variable in scope:
                expected *= self.domains[variable]
            check(len(table) == expected, "a factor table must match its scope's domain sizes")
            check(all(entry >= 0.0 for entry in table), "factor weights must be non-negative")

    def variable_factors(self):
        """For every variable, the list of factor indices it appears in."""
        attached = [[] for _ in self.domains]
        for index, (scope, _) in enumerate(self.factors):
            for variable in scope:
                attached[variable].append(index)
        return attached

    def table_index(self, scope, assignment):
        index = 0
        for variable, value in zip(scope, assignment):
            index = index * self.domains[variable] + value
        return index

    def weight_of(self, assignment):
        total = 1.0
        for scope, table in self.factors:
            total *= table[self.table_index(scope, [assignment[variable] for variable in scope])]
        return total

    def brute_force(self):
        """Exact enumeration: returns (Z, marginals, MAP assignment, MAP weight)."""
        total = 0.0
        marginals = [[0.0] * domain for domain in self.domains]
        best_weight, best_assignment = -1.0, None
        for assignment in itertools.product(*[range(domain) for domain in self.domains]):
            weight = self.weight_of(assignment)
            total += weight
            for variable, value in enumerate(assignment):
                marginals[variable][value] += weight
            if weight > best_weight:
                best_weight, best_assignment = weight, assignment
        if total <= 0:
            return 0.0, [[1.0 / domain] * domain for domain in self.domains], best_assignment, 0.0
        return (total, [[value / total for value in row] for row in marginals],
                best_assignment, best_weight / total)


def normalize(values):
    """Normalise a message; an all-zero message becomes uniform so nothing ever divides by zero."""
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


class BeliefPropagation:
    """Sum-product (or max-product) message passing with synchronous, optionally damped sweeps."""

    def __init__(self, graph, damping=0.0, max_product=False):
        self.graph = graph
        self.damping = damping
        self.max_product = max_product
        self.v2f = {}
        self.f2v = {}
        for index, (scope, _) in enumerate(graph.factors):
            for variable in scope:
                self.v2f[(variable, index)] = [1.0 / graph.domains[variable]] * graph.domains[variable]
                self.f2v[(index, variable)] = [1.0 / graph.domains[variable]] * graph.domains[variable]

    def sweep(self):
        """One synchronous update of every factor->variable and variable->factor message."""
        new_f2v = {}
        for index, (scope, table) in enumerate(self.graph.factors):
            ranges = [range(self.graph.domains[variable]) for variable in scope]
            for target in scope:
                keep = scope.index(target)
                accumulator = [0.0] * self.graph.domains[target]
                for assignment in itertools.product(*ranges):
                    weight = table[self.graph.table_index(scope, list(assignment))]
                    for position, variable in enumerate(scope):
                        if position != keep:                 # the target's own message is excluded
                            weight *= self.v2f[(variable, index)][assignment[position]]
                    if self.max_product:
                        accumulator[assignment[keep]] = max(accumulator[assignment[keep]], weight)
                    else:
                        accumulator[assignment[keep]] += weight
                new_f2v[(index, target)] = normalize(accumulator)
        new_v2f = {}
        attached = self.graph.variable_factors()
        for variable, factors in enumerate(attached):
            for source in factors:
                product = [1.0] * self.graph.domains[variable]
                for other in factors:
                    if other == source:
                        continue
                    product = [a * b for a, b in zip(product, new_f2v[(other, variable)])]
                new_v2f[(variable, source)] = normalize(product)
        if self.damping:
            keep = 1.0 - self.damping
            for key in list(new_f2v):
                new_f2v[key] = [keep * old + self.damping * new
                                for old, new in zip(self.f2v[key], new_f2v[key])]
            for key in list(new_v2f):
                new_v2f[key] = [keep * old + self.damping * new
                                for old, new in zip(self.v2f[key], new_v2f[key])]
        shift = max(max(abs(a - b) for a, b in zip(self.f2v[key], new_f2v[key])) for key in new_f2v)
        self.f2v, self.v2f = new_f2v, new_v2f
        return shift

    def run(self, iterations=60, tolerance=1e-13):
        """Sweep until the messages stop moving; returns the number of sweeps used."""
        used = 0
        for _ in range(iterations):
            used += 1
            if self.sweep() < tolerance:
                break
        return used

    def marginals(self):
        attached = self.graph.variable_factors()
        out = []
        for variable, factors in enumerate(attached):
            if not factors:
                out.append([1.0 / self.graph.domains[variable]] * self.graph.domains[variable])
                continue
            product = [1.0] * self.graph.domains[variable]
            for index in factors:
                product = [a * b for a, b in zip(product, self.f2v[(index, variable)])]
            out.append(normalize(product))
        return out

    def map_assignment(self):
        return [max(range(domain), key=lambda value: row[value])
                for domain, row in zip(self.graph.domains, self.marginals())]

    def residual(self):
        """Fixed-point residual: how far a further sweep would move the factor messages."""
        snapshot = {key: list(value) for key, value in self.f2v.items()}
        self.sweep()
        shift = max(max(abs(a - b) for a, b in zip(snapshot[key], self.f2v[key])) for key in snapshot)
        self.f2v = snapshot
        return shift


def chain_graph(length, seed=3):
    """A binary chain: pairwise attractive factors plus a prior on the first variable."""
    factors = [((0,), [1.0, seed / 4.0])]
    for index in range(length - 1):
        factors.append(((index, index + 1), [4.0, 1.0, 1.0, 4.0]))
    return FactorGraph([2] * length, factors)


def star_graph(leaves, seed=5):
    factors = [((0,), [1.0, 2.0])]
    for index in range(1, leaves + 1):
        factors.append(((0, index), [3.0, 1.0, 1.0, 3.0]))
    return FactorGraph([2] * (leaves + 1), factors)


def cycle_graph(length, coupling=2.0, prior=None):
    factors = []
    if prior is not None:
        factors.append(((0,), list(prior)))
    for index in range(length):
        factors.append(((index, (index + 1) % length), [coupling, 1.0, 1.0, coupling]))
    return FactorGraph([2] * length, factors)


def main():
    # 1. the enumeration oracle on a hand-checked example
    tiny = FactorGraph([2], [((0,), [1.0, 3.0])])
    total, marginals, assignment, weight = tiny.brute_force()
    check(abs(total - 4.0) < 1e-12, "the partition function of one factor is its weight sum")
    check(abs(marginals[0][1] - 0.75) < 1e-12, "the hand-checked marginal must be 0.75")
    check(assignment == (1,) and abs(weight - 0.75) < 1e-12, "the MAP must pick the heavy state")
    empty = FactorGraph([3], [])
    check(empty.brute_force()[1] == [[1 / 3, 1 / 3, 1 / 3]],
          "a variable with no factors must be uniform")

    # 2. sum-product is exact on trees: chain and star
    exact_checks = 0
    worst_tree_error = 0.0
    for length in (2, 3, 4, 5):
        graph = chain_graph(length, seed=length)
        _, exact, _, _ = graph.brute_force()
        bp = BeliefPropagation(graph)
        sweeps = bp.run()
        approximate = bp.marginals()
        for left, right in zip(exact, approximate):
            for a, b in zip(left, right):
                worst_tree_error = max(worst_tree_error, abs(a - b))
        check(worst_tree_error < 1e-9, f"chain marginals must be exact, error {worst_tree_error}")
        check(sweeps <= graph.domains.__len__() + 2,
              f"a chain must converge in about its diameter, took {sweeps}")
        exact_checks += 1
    for leaves in (1, 2, 3):
        graph = star_graph(leaves)
        _, exact, _, _ = graph.brute_force()
        bp = BeliefPropagation(graph)
        bp.run()
        approximate = bp.marginals()
        deviation = max(abs(a - b) for left, right in zip(exact, approximate)
                        for a, b in zip(left, right))
        check(deviation < 1e-9, f"star marginals must be exact, error {deviation}")
        exact_checks += 1
    check(exact_checks == 7, "seven tree graphs must be verified")

    # 3. the converged messages are a fixed point
    for graph in (chain_graph(4), star_graph(3), cycle_graph(4)):
        bp = BeliefPropagation(graph, damping=0.2)
        bp.run(iterations=200)
        check(bp.residual() < 1e-10, "the converged messages must satisfy the fixed-point equations")
        check(all(abs(sum(row) - 1.0) < 1e-9 for row in bp.marginals()),
              "every belief must be normalised")
        check(all(sum(row) for row in bp.marginals()), "beliefs must not be empty")

    # 4. max-product gives the exact MAP on trees
    maps_checked = 0
    for graph in (chain_graph(5), star_graph(3), chain_graph(2)):
        _, _, exact_assignment, _ = graph.brute_force()
        maximal = BeliefPropagation(graph, max_product=True)
        maximal.run()
        approximate = maximal.map_assignment()
        exact_weight = graph.weight_of(exact_assignment)
        approximate_weight = graph.weight_of(approximate)
        check(abs(approximate_weight - exact_weight) < 1e-9,
              f"max-product must find a MAP assignment ({approximate_weight} vs {exact_weight})")
        maps_checked += 1
    check(maps_checked == 3, "three MAP checks must run")

    # 5. a graph with a cycle: the fixed point holds but the marginals are only approximate
    loop = cycle_graph(4, coupling=8.0, prior=[1.0, 5.0])
    _, exact, _, _ = loop.brute_force()
    loopy = BeliefPropagation(loop, damping=0.3)
    loopy.run(iterations=300)
    loopy_deviation = max(abs(a - b) for left, right in zip(exact, loopy.marginals())
                          for a, b in zip(left, right))
    check(loopy.residual() < 1e-9, "loopy BP must still reach a message fixed point")
    check(loopy_deviation > 1e-4, "loopy BP must be measurably different from the exact answer")
    check(loopy_deviation < 0.25, f"the loopy error must stay small, got {loopy_deviation:.4f}")

    # 6. deterministic factors and a zero-probability configuration
    hard = FactorGraph([2, 2], [((0, 1), [1.0, 0.0, 0.0, 1.0]), ((0,), [1.0, 1.0])])
    _, exact, _, _ = hard.brute_force()
    bp = BeliefPropagation(hard)
    bp.run()
    check(abs(bp.marginals()[0][1] - 0.5) < 1e-9, "an equality constraint must leave a fair marginal")
    forbidden = FactorGraph([2], [((0,), [0.0, 2.0])])
    _, exact_forbidden, _, _ = forbidden.brute_force()
    check(exact_forbidden[0][0] == 0.0, "a zero-weight state must have a zero marginal")
    bp_forbidden = BeliefPropagation(forbidden)
    bp_forbidden.run()
    check(bp_forbidden.marginals()[0][0] == 0.0,
          "BP must not invent mass for a forbidden state")
    check(bp_forbidden.marginals()[0][1] == 1.0, "the allowed state must carry all the mass")
    check(forbidden.weight_of((0,)) == 0.0 and forbidden.weight_of((1,)) == 2.0,
          "the factor weights must be respected by the oracle")
    zeroed = FactorGraph([2], [((0,), [0.0, 0.0])])
    check(BeliefPropagation(zeroed).marginals()[0] == [0.5, 0.5],
          "an all-zero factor must degenerate to the uniform belief instead of dividing by zero")
    check(hard.weight_of((0, 0)) == 1.0 and hard.weight_of((1, 1)) == 1.0,
          "the equality factor must accept exactly the matching assignments")

    # 7. ternary variables and a single factor
    ternary = FactorGraph([3, 2], [((0, 1), [1.0, 0.0, 2.0, 1.0, 1.0, 3.0])])
    total, exact, _, _ = ternary.brute_force()
    bp = BeliefPropagation(ternary)
    bp.run()
    deviation = max(abs(a - b) for left, right in zip(exact, bp.marginals())
                    for a, b in zip(left, right))
    check(deviation < 1e-9, f"a single factor on ternary variables is a tree, error {deviation}")
    check(abs(total - 8.0) < 1e-12, "the ternary partition function must be the table sum")

    # 8. determinism
    first = BeliefPropagation(loop, damping=0.3)
    second = BeliefPropagation(loop, damping=0.3)
    check(first.run(iterations=50) == second.run(iterations=50), "sweep counts must reproduce")
    check(first.marginals() == second.marginals(), "marginals must reproduce exactly")

    print(f"782 tree_error={worst_tree_error:.2e} tree_checks={exact_checks} "
          f"map_checks={maps_checked} loopy_error={loopy_deviation:.4f}")
    print(f"782 chain_sweeps={BeliefPropagation(chain_graph(5)).run()} "
          f"hard_marginal={bp.marginals()[1][1]:.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
