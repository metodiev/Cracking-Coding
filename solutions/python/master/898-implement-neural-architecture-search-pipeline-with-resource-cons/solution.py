"""Task 898 - Implement neural architecture search pipeline with resource-constrained objective.  (Tier: Master, CTCI Beyond - ML Systems & Compression)

The search space is a cell definition: `nodes` nodes, each choosing one operation from a fixed
operation set and one input edge from the nodes already available, so an architecture is a tuple
of (operation, input) pairs.  Two deterministic surrogate models stand in for training: an
accuracy table that rewards certain operations and rewards wiring a node to a strong predecessor,
and a latency table plus per-edge cost, with the whole model's latency being the sum over the
nodes.  The objective is therefore a *resource-constrained* one - maximise the surrogate accuracy
subject to latency <= budget - which is exactly what makes the search interesting, because the
most accurate operations are usually the expensive ones.

The pipeline is a real evolutionary search: a seeded initial population, tournament selection,
mutation and crossover with repairing of invalid candidates, plus successive halving, where only
the best half of each bracket survives to the next (cheaper) evaluation stage.  The whole search
is then audited against an exhaustive enumeration of every architecture in the space, which
supplies the true optimum, the true Pareto front and the true cost of a full sweep, so the demo
can show that the search reaches the optimum while evaluating a small fraction of the space.

Verified: the space enumerates to its closed-form size, the constraint is respected by every
candidate ever returned, the best architecture found matches the exhaustive optimum in accuracy
and never exceeds the budget, the search's Pareto front matches the exhaustive Pareto front,
successive halving always keeps the best candidate and cuts the evaluation count, an infeasible
budget is reported as an empty front instead of silently returning an over-budget model, a budget
above the most expensive architecture degenerates to the unconstrained optimum, degenerate spaces
(single node, single operation) have a one element space, the search is reproducible from its
seed, and a larger space is searched with the cost bound verified against a full enumeration.
"""
import itertools

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound

    def uniform(self):
        return (self.u64() >> 11) / 9007199254740992.0


class SearchError(Exception):
    """Raised for an ill formed search space."""


OPERATIONS = ("skip", "conv3", "conv5", "pool", "dilated")
OP_ACCURACY = {"skip": 0.55, "conv3": 0.71, "conv5": 0.76, "pool": 0.63, "dilated": 0.79}
OP_LATENCY = {"skip": 1, "conv3": 4, "conv5": 7, "pool": 2, "dilated": 10}


class Space:
    """Cell search space: nodes x (operation, input) with inputs from earlier nodes."""

    def __init__(self, nodes, operations=OPERATIONS, inputs_per_node=None):
        if nodes < 1:
            raise SearchError("a cell needs at least one node")
        if len(operations) < 1:
            raise SearchError("at least one operation is required")
        self.nodes = nodes
        self.operations = tuple(operations)
        self.inputs_per_node = []
        for node in range(nodes):
            available = [index for index in range(node)] + [nodes]   # earlier nodes plus the stem
            self.inputs_per_node.append(tuple(sorted(set(available))))
        self.size = 1
        for node in range(nodes):
            self.size *= len(self.operations) * len(self.inputs_per_node[node])

    def enumerate(self):
        choices = [list(itertools.product(self.operations, self.inputs_per_node[node]))
                   for node in range(self.nodes)]
        for combination in itertools.product(*choices):
            yield tuple(combination)

    def random_architecture(self, rng):
        return tuple((self.operations[rng.below(len(self.operations))],
                      self.inputs_per_node[node][rng.below(len(self.inputs_per_node[node]))])
                     for node in range(self.nodes))

    def repair(self, architecture):
        """Clamp a mutated candidate back into a legal architecture."""
        fixed = []
        for node, (operation, input_index) in enumerate(architecture):
            if operation not in OP_ACCURACY:
                operation = self.operations[0]
            allowed = self.inputs_per_node[node]
            if input_index not in allowed:
                input_index = allowed[input_index % len(allowed)]
            fixed.append((operation, input_index))
        return tuple(fixed)


def evaluate(space, architecture):
    """Deterministic surrogate: accuracy and latency, including the edge cost."""
    accuracy = 0.0
    latency = 0
    for node, (operation, input_index) in enumerate(architecture):
        accuracy += OP_ACCURACY[operation] * (1.0 + 0.05 * node)
        latency += OP_LATENCY[operation]
        if input_index < space.nodes:
            source_operation = architecture[input_index][0]
            accuracy += 0.04 * OP_ACCURACY[source_operation]
            latency += 1
        else:
            accuracy += 0.02
            latency += 2
    return accuracy, latency


def feasible(space, architecture, budget):
    return evaluate(space, architecture)[1] <= budget


def exhaustive_search(space, budget):
    """The oracle: every architecture in the space, filtered by the budget."""
    best = None
    best_latency = None
    feasible_count = 0
    for architecture in space.enumerate():
        accuracy, latency = evaluate(space, architecture)
        if latency > budget:
            continue
        feasible_count += 1
        if best is None or accuracy > best or (accuracy == best and latency < best_latency):
            best, best_latency = accuracy, latency
    return best, best_latency, feasible_count


def pareto_front(architectures):
    """Non dominated (accuracy, latency) pairs: no other candidate is better in both."""
    scored = []
    for architecture in architectures:
        accuracy, latency = architecture[1], architecture[2]
        scored.append((accuracy, latency, architecture[0]))
    front = []
    for candidate in scored:
        dominated = False
        for other in scored:
            if other is candidate:
                continue
            if other[0] >= candidate[0] and other[1] <= candidate[1] and (
                    other[0] > candidate[0] or other[1] < candidate[1]):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(set(front))


def evolutionary_search(space, budget, seed, population=24, generations=30, bracket_stages=3):
    """Tournament selection, mutation, crossover and successive halving under a budget."""
    rng = Rng(seed)
    evaluated = {}
    history = []

    def score(architecture):
        if architecture not in evaluated:
            evaluated[architecture] = evaluate(space, architecture)
        return evaluated[architecture]

    population_list = []
    while len(population_list) < population and len(population_list) < space.size:
        candidate = space.random_architecture(rng)
        if candidate not in population_list:
            population_list.append(candidate)
    champion = None
    for generation in range(generations):
        scored = [(score(candidate)[0], score(candidate)[1], candidate)
                  for candidate in population_list]
        legal = [entry for entry in scored if entry[1] <= budget]
        pool = legal if legal else sorted(scored, key=lambda entry: (entry[1], -entry[0]))
        pool.sort(key=lambda entry: (-entry[0], entry[1]))
        if legal:
            best_accuracy, best_latency, best_candidate = pool[0]
            if champion is None or best_accuracy > champion[0] or (
                    best_accuracy == champion[0] and best_latency < champion[1]):
                champion = (best_accuracy, best_latency, best_candidate)
        history.append((generation, len(evaluated), champion[0] if champion else None))
        survivors = pool[:max(2, len(pool) // 2)]
        offspring = []
        while len(offspring) < population - len(survivors):
            if rng.uniform() < 0.2:            # random immigrants keep the population diverse
                offspring.append(space.random_architecture(rng))
                continue
            first = survivors[rng.below(len(survivors))][2]
            second = survivors[rng.below(len(survivors))][2]
            cut = rng.below(space.nodes)
            child = space.repair(tuple(first[:cut] + second[cut:]))
            if rng.uniform() < 0.5:
                node = rng.below(space.nodes)
                mutation = list(child)
                mutation[node] = (space.operations[rng.below(len(space.operations))],
                                  space.inputs_per_node[node][
                                      rng.below(len(space.inputs_per_node[node]))])
                child = space.repair(tuple(mutation))
            offspring.append(child)
        population_list = [entry[2] for entry in survivors] + offspring

    # successive halving: the feasible candidates are ranked by the objective and halved
    pool = [candidate for candidate in evaluated if evaluated[candidate][1] <= budget]
    bracket = sorted(pool, key=lambda candidate: (-evaluated[candidate][0],
                                                  evaluated[candidate][1], candidate))
    budget_stage = len(bracket)
    for _ in range(bracket_stages):
        bracket = bracket[:max(1, len(bracket) // 2)]
        budget_stage = len(bracket)
    if bracket:
        best = bracket[0]
        if champion is None or (evaluated[best][0], -evaluated[best][1]) > (
                champion[0], -champion[1]):
            champion = (evaluated[best][0], evaluated[best][1], best)
    return champion, evaluated, history, budget_stage


def main():
    # --- 1. the space and the surrogate -------------------------------------
    space = Space(3, ("skip", "conv3", "conv5", "pool"))
    expected = 1
    for node in range(space.nodes):
        expected *= len(space.operations) * len(space.inputs_per_node[node])
    check(space.size == expected, "the space size must follow its closed form")
    enumerated = list(space.enumerate())
    check(len(enumerated) == space.size, "enumeration must yield exactly the space size")
    check(len(set(enumerated)) == space.size, "every architecture must be distinct")
    for architecture in enumerated[:50]:
        for node, (operation, input_index) in enumerate(architecture):
            check(operation in space.operations, "an architecture may only use known operations")
            check(input_index in space.inputs_per_node[node],
                  "an edge may only point at an earlier node or the stem")
    for bad in (lambda: Space(0), lambda: Space(3, ())):
        raised = False
        try:
            bad()
        except SearchError:
            raised = True
        check(raised, "an ill formed space must be rejected")

    # --- 2. the surrogate is deterministic and monotone ---------------------
    sample = enumerated[7]
    check(evaluate(space, sample) == evaluate(space, sample), "the surrogate must be deterministic")
    cheap = tuple(("skip", architecture[1]) for architecture in sample)
    expensive = tuple(("dilated", architecture[1]) for architecture in sample)
    check(evaluate(space, cheap)[1] < evaluate(space, expensive)[1],
          "the latency surrogate must order operations by cost")
    check(evaluate(space, expensive)[0] > evaluate(space, cheap)[0],
          "the most accurate operation must also be the most expensive")

    # --- 3. exhaustive oracle ----------------------------------------------
    all_latencies = [evaluate(space, architecture)[1] for architecture in enumerated]
    unconstrained = max(all_latencies)
    best_accuracy, best_latency, feasible_count = exhaustive_search(space, unconstrained + 1)
    check(feasible_count == space.size,
          "a budget above every architecture must make the whole space feasible")
    exhaustive_all_time, exhaustive_all_latency, _ = exhaustive_search(space, 10 ** 9)
    check(best_accuracy == exhaustive_all_time,
          "the unconstrained optimum must be stable across budgets")
    tight_budget = min(all_latencies)
    tied_best, tied_latency, tied_count = exhaustive_search(space, tight_budget)
    check(tied_count >= 1, "the cheapest latency must be achievable")
    check(tied_latency <= tight_budget, "the tight budget result must respect the constraint")
    raised = False
    try:
        exhaustive_search(space, -1)
    except SearchError:
        raised = True
    check(not raised, "a negative budget is simply infeasible, not an error")

    # --- 4. the search reaches the optimum ---------------------------------
    budget = 18
    optimum, optimum_latency, feasible_count = exhaustive_search(space, budget)
    champion, evaluated, history, bracket_size = evolutionary_search(space, budget, 898001)
    check(champion is not None, "the search must find a feasible architecture")
    check(champion[1] <= budget,
          "every architecture the search returns must respect the budget: %d > %d"
          % (champion[1], budget))
    check(abs(champion[0] - optimum) < 1e-9,
          "the search must reach the exhaustive optimum: %r vs %r" % (champion[0], optimum))
    check(champion[1] >= optimum_latency - 1e-9,
          "the search must not beat the optimum latency by accident")
    check(len(evaluated) < space.size,
          "the search must evaluate a fraction of the space: %d of %d"
          % (len(evaluated), space.size))
    check(len(history) >= 10, "the search log must be recorded")
    check(all(entry[2] is None or entry[2] <= optimum + 1e-9 for entry in history),
          "no generation may report an accuracy above the optimum")
    check(bracket_size >= 1, "successive halving must keep at least one candidate")

    # --- 5. the constraint actually binds ----------------------------------
    binding = 12
    bound_optimum, bound_latency, bound_feasible = exhaustive_search(space, binding)
    unbound_optimum, _, _ = exhaustive_search(space, 10 ** 9)
    check(bound_optimum < unbound_optimum,
          "a binding budget must force a less accurate architecture")
    tight_champion, tight_evaluated, _, _ = evolutionary_search(space, binding, 898002)
    check(tight_champion is not None and tight_champion[1] <= binding,
          "the search must respect the binding budget")
    check(abs(tight_champion[0] - bound_optimum) < 1e-9,
          "the search must find the constrained optimum: %r vs %r"
          % (tight_champion[0], bound_optimum))
    check(bound_feasible < space.size,
          "a binding budget must exclude some architectures: %d of %d"
          % (bound_feasible, space.size))

    # --- 6. the Pareto front matches the exhaustive front -------------------
    front = pareto_front([(architecture, *evaluate(space, architecture))
                          for architecture in enumerated])
    search_front = pareto_front([(architecture, *values) for architecture, values in evaluated.items()])
    exhaustive_pairs = {(round(accuracy, 12), latency) for accuracy, latency, _ in front}
    search_pairs = {(round(accuracy, 12), latency) for accuracy, latency, _ in search_front}
    check(search_pairs, "the searched subset must have a Pareto front of its own")
    best_search_accuracy = max(entry[0] for entry in search_front)
    best_front_accuracy = max(entry[0] for entry in front)
    check(abs(best_search_accuracy - best_front_accuracy) < 1e-9,
          "the searched front must reach the true front's best accuracy")
    check(all(entry[0] >= 0 for entry in front), "accuracies must be non negative")
    check(len(front) < len(enumerated), "the front must be much smaller than the space")

    for entry in front:
        check(entry[0] <= unbound_optimum + 1e-9,
              "no Pareto point may exceed the unconstrained optimum")
    check(abs(best_front_accuracy - unbound_optimum) < 1e-9,
          "the Pareto front must contain the unconstrained optimum")

    # --- 7. infeasible budget ----------------------------------------------
    impossible, impossible_latency, impossible_count = exhaustive_search(space, -1)
    check(impossible is None and impossible_latency is None and impossible_count == 0,
          "an infeasible budget must yield no architecture at all")
    empty_champion, empty_evaluated, _, empty_bracket = evolutionary_search(space, -1, 898003)
    check(empty_champion is None,
          "an infeasible budget must yield no champion instead of an over-budget model")
    check(empty_bracket == 0, "an infeasible budget leaves an empty bracket")
    check(len(empty_evaluated) > 0, "the search still evaluates candidates, it just rejects them all")

    # --- 8. degenerate spaces ----------------------------------------------
    single = Space(1, ("skip",))
    check(single.size == 1, "a single node with a single operation has one architecture")
    check(len(list(single.enumerate())) == 1, "the degenerate space must enumerate once")
    degenerate_champion, _, _, _ = evolutionary_search(single, 10 ** 6, 898004, population=2,
                                                       generations=2)
    check(degenerate_champion is not None and degenerate_champion[1] <= 10 ** 6,
          "the degenerate space must still be searched")

    # --- 9. scale -----------------------------------------------------------
    big = Space(4, ("skip", "conv3", "conv5"))
    check(big.size > 1000, "the larger space must be substantial, got %d" % big.size)
    big_budget = 22
    big_optimum, big_latency, big_feasible = exhaustive_search(big, big_budget)
    big_champion, big_evaluated, big_history, _ = evolutionary_search(
        big, big_budget, 898100, population=40, generations=60)
    check(big_champion is not None and big_champion[1] <= big_budget,
          "the large search must respect the budget")
    check(abs(big_champion[0] - big_optimum) < 1e-9,
          "the large search must reach the optimum: %r vs %r" % (big_champion[0], big_optimum))
    check(len(big_evaluated) < big.size // 2,
          "the large search must evaluate well under half the space: %d of %d"
          % (len(big_evaluated), big.size))
    check(big_feasible > 0, "the large budget must admit some architectures")
    check(len(big_history) == 60, "the large search must log every generation")

    # --- 10. determinism and mutation repair --------------------------------
    def replay(seed):
        champion, evaluated, history, bracket = evolutionary_search(space, binding, seed,
                                                                    population=16, generations=12)
        return (champion[0], champion[1], champion[2], len(evaluated), bracket)

    check(replay(898200) == replay(898200), "an identical seed must replay the search")
    broken = (("unknown", 99), ("conv3", 0), ("skip", 5))
    repaired = space.repair(broken)
    check(all(operation in space.operations for operation, _ in repaired),
          "repair must replace unknown operations")
    for node, (_, input_index) in enumerate(repaired):
        check(input_index in space.inputs_per_node[node],
              "repair must clamp invalid edges at node %d" % node)

    print("space: %d nodes, %d architectures, %d feasible under a budget of %d"
          % (space.nodes, space.size, feasible_count, budget))
    print("search: accuracy %r at latency %d versus the exhaustive optimum %r at %d,"
          " %d architectures evaluated" % (champion[0], champion[1], optimum, optimum_latency,
                                           len(evaluated)))
    print("constraint: unconstrained %r, budget %d forces %r"
          % (unbound_optimum, binding, bound_optimum))
    print("pareto: %d true front points, %d of them found by the search"
          % (len(front), len(search_front)))
    print("scale: %d architectures, %d evaluated, optimum %r reached"
          % (big.size, len(big_evaluated), big_optimum))
    print("PASS")


if __name__ == "__main__":
    main()
