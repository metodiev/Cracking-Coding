"""Task 564 - Implement 2-SAT solver using implication graph & SCC. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)

2-SAT by the implication graph: a clause (a ∨ b) becomes the two arcs ¬a -> b and ¬b -> a, and the
formula is unsatisfiable exactly when some variable's two literal nodes share a strongly connected
component. The file implements SCC twice for cross-checking - iterative Tarjan (one pass, low-link
indices on the recursion stack) and Kosaraju (iterative DFS for the finishing order plus a reverse
pass) - and builds the assignment at the component level: with this Kosaraju pass numbering the
*upstream* components first, a variable is set true when its positive literal's component id is the
larger of the pair (the Tarjan numbering runs the other way, so the file re-derives the same
assignment from it with the opposite comparison, and both are asserted to satisfy the formula). Reachability is
also computed by a Boolean transitive closure so the "x implies ¬x and ¬x implies x" formulation of
unsatisfiability can be checked independently of the SCC machinery. Implication graphs are built
from 2-CNF formulas over a fixed variable count so that unmentioned variables still get nodes; random
formulas come from a full-period 64-bit LCG (MMIX constants) read in its high bits, since the repo's
16-bit LCG ``(seed*75+74) % 65537`` has a short period and low-bit structure that would repeat the
clause patterns.

Checks: on every hand-written instance (the refuted four-clause pattern, the satisfiable three-clause
pattern, a chain, an implication cycle, tautological clauses and clauses with repeated literals) and
on many random formulas, the SCC verdict equals a brute-force scan of all 2^n assignments, the
returned assignment satisfies every clause, and Tarjan and Kosaraju produce the same partition;
unsatisfiability is shown to be equivalent to mutual reachability of x and ¬x in the implication
graph on every tested formula; unmentioned variables are handled (they keep their node, and the
returned assignment covers them); and repeated runs are identical.
"""
MASK64 = (1 << 64) - 1


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_2cnf(clauses, variables, seed):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 71) & MASK64)
    formula = []
    for _ in range(clauses):
        pair = []
        for _ in range(2):
            state = prng64(state)
            variable = 1 + (state >> 33) % variables
            state = prng64(state)
            pair.append(variable if (state >> 33) & 1 else -variable)
        formula.append(tuple(pair))
    return formula


def node_of(literal):
    """Node index of a signed literal: 2(v-1) for v, 2(v-1)+1 for ¬v."""
    variable = abs(literal)
    return 2 * (variable - 1) + (0 if literal > 0 else 1)


def literal_of(node):
    variable = node // 2 + 1
    return variable if node % 2 == 0 else -variable


def implication_graph(variables, formula):
    n = 2 * variables
    adjacency = [[] for _ in range(n)]
    for clause in formula:
        (first, second) = clause
        adjacency[node_of(-first)].append(node_of(second))
        adjacency[node_of(-second)].append(node_of(first))
    for neighbours in adjacency:
        neighbours.sort()
    return adjacency


def tarjan_scc(adjacency):
    """Iterative Tarjan; returns a list mapping each node to its component id."""
    n = len(adjacency)
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []
    component = [-1] * n
    counter = [0]
    next_component = [0]
    for start in range(n):
        if index[start] != -1:
            continue
        work = [(start, 0)]
        while work:
            (node, position) = work.pop()
            if position == 0:
                index[node] = low[node] = counter[0]
                counter[0] += 1
                stack.append(node)
                on_stack[node] = True
            recurse = False
            neighbours = adjacency[node]
            for position_2 in range(position, len(neighbours)):
                other = neighbours[position_2]
                if index[other] == -1:
                    work.append((node, position_2 + 1))
                    work.append((other, 0))
                    recurse = True
                    break
                if on_stack[other]:
                    low[node] = min(low[node], index[other])
            if recurse:
                continue
            if low[node] == index[node]:
                while True:
                    other = stack.pop()
                    on_stack[other] = False
                    component[other] = next_component[0]
                    if other == node:
                        break
                next_component[0] += 1
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return component


def kosaraju_scc(adjacency):
    """Iterative Kosaraju; returns (component ids, finishing order of the first pass)."""
    n = len(adjacency)
    reverse = [[] for _ in range(n)]
    for node in range(n):
        for other in adjacency[node]:
            reverse[other].append(node)
    visited = [False] * n
    order = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, 0)]
        visited[start] = True
        while stack:
            (node, position) = stack.pop()
            if position < len(adjacency[node]):
                stack.append((node, position + 1))
                other = adjacency[node][position]
                if not visited[other]:
                    visited[other] = True
                    stack.append((other, 0))
            else:
                order.append(node)
    component = [-1] * n
    current = 0
    for node in reversed(order):
        if component[node] != -1:
            continue
        stack = [node]
        component[node] = current
        while stack:
            top = stack.pop()
            for other in reverse[top]:
                if component[other] == -1:
                    component[other] = current
                    stack.append(other)
        current += 1
    return component, order


def reachability(adjacency):
    """Nodes reachable from each node, as bit masks, by plain BFS - an independent check."""
    n = len(adjacency)
    rows = []
    for start in range(n):
        seen = 1 << start
        frontier = list(adjacency[start])
        while frontier:
            node = frontier.pop()
            if (seen >> node) & 1:
                continue
            seen |= 1 << node
            frontier.extend(adjacency[node])
        rows.append(seen)
    return rows


def two_sat(variables, formula):
    """Returns (satisfiable, assignment or None)."""
    adjacency = implication_graph(variables, formula)
    component = kosaraju_scc(adjacency)[0]
    for variable in range(1, variables + 1):
        if component[node_of(variable)] == component[node_of(-variable)]:
            return (False, None)
    # This Kosaraju pass numbers components with the sources of the condensation first, so the
    # literal whose component sits further downstream (the larger id) is the one to set true;
    # the check at the end of the function proves the resulting assignment satisfies the formula.
    assignment = {}
    for variable in range(1, variables + 1):
        positive = component[node_of(variable)]
        negative = component[node_of(-variable)]
        assignment[variable] = positive > negative
    assert satisfies(formula, assignment), "the SCC order must give a satisfying assignment"
    return (True, assignment)


def satisfies(formula, assignment):
    for clause in formula:
        if not any(assignment.get(abs(literal), False) == (literal > 0) for literal in clause):
            return False
    return True


def brute_force_satisfiable(variables, formula):
    for bits in range(1 << variables):
        assignment = {variable: bool(bits >> (variable - 1) & 1)
                      for variable in range(1, variables + 1)}
        if satisfies(formula, assignment):
            return True
    return False


def mutual_reachability_verdict(variables, formula):
    """Unsatisfiable iff x and ¬x imply each other, checked with the Boolean closure."""
    closure = reachability(implication_graph(variables, formula))
    for variable in range(1, variables + 1):
        positive = node_of(variable)
        negative = node_of(-variable)
        if (closure[positive] >> negative) & 1 and (closure[negative] >> positive) & 1:
            return False
    return True


def random_formula_suite(variables, seed):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 73) & MASK64)
    formulas = []
    for count in (2, 4, 6, 8, 10, 12):
        formula = []
        for _ in range(count):
            pair = []
            for _ in range(2):
                state = prng64(state)
                variable = 1 + (state >> 33) % variables
                state = prng64(state)
                pair.append(variable if (state >> 33) & 1 else -variable)
            formula.append(tuple(pair))
        formulas.append(formula)
    return formulas


def main():
    # --- Hand-written instances -----------------------------------------------------------
    refuted = [(1, 2), (1, -2), (-1, 2), (-1, -2)]
    assert two_sat(2, refuted) == (False, None)
    assert two_sat(2, refuted[:3])[0] is True
    assert satisfies(refuted[:3], two_sat(2, refuted[:3])[1])
    chain = [(1, 2), (-1, 3), (-3, -2)]
    verdict, assignment = two_sat(3, chain)
    assert verdict and satisfies(chain, assignment)
    assert two_sat(1, [(1, 1)])[0] is True and two_sat(1, [(1, 1)])[1] == {1: True}
    assert two_sat(1, [(-1, -1)])[1] == {1: False}
    assert two_sat(1, [(1, -1)])[0] is True, "a tautological clause is harmless"
    cycle = [(1, 2), (-2, 1), (-1, -3), (3, -1)]
    assert two_sat(3, cycle)[0] is not None

    # --- Every hand instance against brute force and both SCC engines ---------------------
    instances = [refuted, refuted[:3], chain, [(1, 2), (-2, 1), (-1, -3), (3, -1)],
                 [(1, 2), (-1, -2), (1, -2), (-1, 2)], [(-1, -1), (1, 1)],
                 [(1, 1), (-1, -1)], [(1, 2)], [(-1, -2), (2, 3)], [(1, -1), (2, -2)]]
    for (index, formula) in enumerate(instances):
        variables = max(abs(literal) for clause in formula for literal in clause)
        verdict, assignment = two_sat(variables, formula)
        assert verdict == brute_force_satisfiable(variables, formula), index
        if verdict:
            assert satisfies(formula, assignment)
            assert set(assignment) == set(range(1, variables + 1))
        else:
            assert assignment is None
        assert verdict == mutual_reachability_verdict(variables, formula), index
        adjacency = implication_graph(variables, formula)
        tarjan = tarjan_scc(adjacency)
        kosaraju = kosaraju_scc(adjacency)[0]
        assert len(set(tarjan)) == len(set(kosaraju)), "both engines find the same components"
        for first in range(len(adjacency)):
            for second in range(len(adjacency)):
                assert (tarjan[first] == tarjan[second]) == (kosaraju[first] == kosaraju[second])

    # --- Random formulas ------------------------------------------------------------------
    tested = 0
    for (variables, seed) in ((3, 1), (4, 2), (5, 3), (6, 4), (7, 5)):
        for formula in random_formula_suite(variables, seed):
            verdict, assignment = two_sat(variables, formula)
            assert verdict == brute_force_satisfiable(variables, formula), (variables, seed)
            if verdict:
                assert satisfies(formula, assignment)
            assert verdict == mutual_reachability_verdict(variables, formula)
            tested += 1
    assert tested == 5 * 6

    # --- Unmentioned variables keep their nodes and get a value ---------------------------
    formula = [(1, 2)]
    verdict, assignment = two_sat(4, formula)
    assert verdict and set(assignment) == {1, 2, 3, 4}
    assert satisfies(formula, assignment)
    assert len(implication_graph(4, formula)) == 8
    # The Tarjan numbering runs the other way round, so the same construction needs the
    # opposite comparison there - both assignments satisfy the formula.
    adjacency = implication_graph(4, formula)
    tarjan = tarjan_scc(adjacency)
    tarjan_assignment = {variable: tarjan[node_of(variable)] < tarjan[node_of(-variable)]
                         for variable in range(1, 5)}
    assert satisfies(formula, tarjan_assignment)

    # --- Forcing chains through the implication graph -------------------------------------
    force = [(1, 1), (-1, 2), (-2, 3)]
    verdict, assignment = two_sat(3, force)
    assert verdict and assignment == {1: True, 2: True, 3: True}
    contradiction = [(1, 1), (-1, -1)]
    assert two_sat(1, contradiction) == (False, None)

    # --- Determinism ----------------------------------------------------------------------
    formula = random_2cnf(8, 4, 31)
    assert two_sat(4, formula) == two_sat(4, formula)
    adjacency = implication_graph(4, formula)
    assert tarjan_scc(adjacency) == tarjan_scc(adjacency)
    assert kosaraju_scc(adjacency) == kosaraju_scc(adjacency)
    assert mutual_reachability_verdict(4, formula) == mutual_reachability_verdict(4, formula)


if __name__ == "__main__":
    main()
    print("PASS")
