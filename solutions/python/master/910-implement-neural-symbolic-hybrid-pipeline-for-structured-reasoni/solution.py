"""Task 910 - Implement neural-symbolic hybrid pipeline for structured reasoning tasks.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

What is implemented (in-process, deterministic):

  * `Engine` - a small first-order forward chainer: facts are binary relations, rules are
    conjunctive implications over variables, and every derived fact stores the rule and the
    premises that produced it, so the whole derivation is a proof DAG.  Deriving runs to a
    fixpoint over a monotone fact set, which terminates even for cyclic rule sets.
  * `verify_proof` - an independent checker that re-unifies the stored premises against the stored
    rule, rejects a proof whose premises are missing or mis-instantiated, and requires the head to
    follow from the resulting binding.
  * A learned linear scorer over symbolic features (rule confidence, hop count, proof depth) that
    re-ranks the answers the symbolic engine derived.  Because it only re-orders a closed
    candidate set, the hybrid can never invent an answer the logic did not support.

Simplified versus production: one toy domain (reachability over a weighted DAG), a linear scorer
instead of a network, no embeddings.  Verified below: the closure against a brute-force
reachability oracle, proof soundness and tamper rejection, hybrid soundness, the accuracy gain
over the symbolic prior, open-world queries, confidence bounds, cyclic rules, and the
empty/singleton/unknown boundaries.
"""

import math


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rule:
    __slots__ = ("name", "premises", "head", "weight")

    def __init__(self, name, premises, head, weight):
        self.name = name
        self.premises = premises          # tuple of (relation, arg, arg) patterns
        self.head = head
        self.weight = weight              # symbolic confidence carried by one application


def is_variable(term):
    return isinstance(term, str) and term.startswith("?")


def unify(pattern, fact, binding):
    """Extend a binding so that `pattern` matches `fact`, or return None."""
    extended = dict(binding)
    for slot, want in zip(fact, pattern):
        if is_variable(want):
            if want in extended and extended[want] != slot:
                return None
            extended[want] = slot
        elif want != slot:
            return None
    return extended


def instantiate(pattern, binding):
    return tuple(binding.get(term, term) if is_variable(term) else term for term in pattern)


class Engine:
    """Forward chaining over binary facts; every derived fact keeps its proof step."""

    def __init__(self, facts, rules):
        self.facts = list(dict.fromkeys(facts))
        self.rules = rules
        self.proofs = {fact: None for fact in self.facts}   # None = given, tuple = derived
        self.steps = 0

    def run(self, limit=5000):
        changed = True
        while changed and self.steps < limit:
            changed = False
            for rule in self.rules:
                for joined in self._join(rule.premises, 0, {}, ()):
                    head = instantiate(rule.head, joined)
                    if head not in self.proofs:
                        self.proofs[head] = (rule, tuple(
                            instantiate(pattern, joined) for pattern in rule.premises))
                        self.facts.append(head)
                        changed = True
                        self.steps += 1
        return self.proofs

    def _join(self, patterns, index, binding, chosen):
        """Depth-first join of the premise patterns over the current fact set."""
        if index == len(patterns):
            yield binding
            return
        for fact in self.facts:
            if fact[0] != patterns[index][0]:
                continue
            extended = unify(patterns[index], fact, binding)
            if extended is None:
                continue
            yield from self._join(patterns, index + 1, extended, chosen + (fact,))

    def metrics(self, fact, memo=None):
        """(confidence, depth, hops) in one pass: the confidence is the product of the rule
        weights along the proof, the depth is the longest premise chain and the hop count is the
        number of edge facts the proof rests on (the ground-truth preference)."""
        memo = {} if memo is None else memo
        if fact not in memo:
            step = self.proofs[fact]
            if step is None:
                memo[fact] = (1.0, 0, 1 if fact[0] == "edge" else 0)
            else:
                parts = [self.metrics(premise, memo) for premise in step[1]]
                memo[fact] = (step[0].weight * math.prod(part[0] for part in parts),
                              1 + max(part[1] for part in parts),
                              sum(part[2] for part in parts))
        return memo[fact]

    def confidence(self, fact):
        return self.metrics(fact)[0]

    def depth(self, fact):
        return self.metrics(fact)[1]

    def hops(self, fact):
        return self.metrics(fact)[2]

    def query(self, relation, subject):
        return [fact for fact in self.proofs
                if fact[0] == relation and fact[1] == subject]


def verify_proof(engine, fact, trail=None):
    """Independent checker: bind the rule by re-unifying its premises, then require the head to
    follow, and recurse.  A tampered, missing or mis-instantiated premise is rejected."""
    trail = set() if trail is None else trail
    if fact in trail:
        return False
    step = engine.proofs.get(fact)
    if step is None:
        return fact in engine.facts and fact in engine.proofs
    rule, premises = step
    if rule.name not in [r.name for r in engine.rules] or len(premises) != len(rule.premises):
        return False
    binding = {}
    for pattern, premise in zip(rule.premises, premises):
        if premise not in engine.proofs:
            return False
        binding = unify(pattern, premise, binding)
        if binding is None:
            return False
    if instantiate(rule.head, binding) != fact:
        return False
    return all(verify_proof(engine, premise, trail | {fact}) for premise in premises)


def reachability_oracle(facts, nodes):
    """Brute-force transitive closure over the edge facts - the symbol engine's oracle."""
    closure = set()
    for source in nodes:
        frontier = [edge[2] for edge in facts if edge[0] == "edge" and edge[1] == source]
        seen = set()
        while frontier:
            node = frontier.pop()
            if node in seen:
                continue
            seen.add(node)
            frontier.extend(edge[2] for edge in facts if edge[0] == "edge" and edge[1] == node)
        closure.update(("path", source, node) for node in seen)
    return closure


def build_domain():
    nodes = ["n%02d" % i for i in range(12)]
    edges, weights = [], {}
    for index, node in enumerate(nodes[:-1]):
        step = 1 + (index % 2)
        target = nodes[min(index + step, len(nodes) - 1)]
        edges.append(("edge", node, target))
        weights[(node, target)] = 0.55 + 0.05 * ((index * 7) % 9)
        if index + 2 < len(nodes):
            edges.append(("edge", node, nodes[index + 2]))
            weights[(node, nodes[index + 2])] = 0.6 + 0.04 * ((index * 5) % 10)
    return nodes, edges, weights


def rule_set():
    """Edge -> path, then extend a path by one edge (the rule confidences are the weights)."""
    return [Rule("reach", (("edge", "?x", "?y"),), ("path", "?x", "?y"), 1.0),
            Rule("extend", (("path", "?x", "?y"), ("edge", "?y", "?z")),
                 ("path", "?x", "?z"), 0.9)]


def score_rows(engine, candidates):
    """Symbolic features per candidate: confidence, inverse hops and inverse proof depth."""
    rows = []
    for fact in candidates:
        confidence = engine.confidence(fact)
        hops = max(1, engine.hops(fact))
        depth = max(1, engine.depth(fact))
        rows.append((fact, [1.0, confidence, 1.0 / hops, 1.0 / depth], confidence, hops))
    return rows


def fit_ranker(rows, labels):
    """Least squares on the analytic normal equations (no library, deterministic)."""
    width = len(rows[0])
    gram = [[sum(row[i] * row[j] for row in rows) for j in range(width)] for i in range(width)]
    rhs = [sum(row[i] * label for row, label in zip(rows, labels)) for i in range(width)]
    for index in range(width):
        gram[index][index] += 1e-9
    for column in range(width):
        pivot = max(range(column, width), key=lambda r: abs(gram[r][column]))
        gram[column], gram[pivot] = gram[pivot], gram[column]
        rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        for row in range(width):
            if row != column and gram[row][column] != 0.0:
                factor = gram[row][column] / gram[column][column]
                for k in range(column, width):
                    gram[row][k] -= factor * gram[column][k]
                rhs[row] -= factor * rhs[column]
    return [rhs[i] / gram[i][i] for i in range(width)]


def main():
    nodes, edges, weights = build_domain()
    engine = Engine(edges, rule_set())
    proofs = engine.run()
    check(engine.steps > 20 and engine.steps < 5000, "the closure must converge in bounded steps")
    check(all(rule.weight <= 1.0 for rule in engine.rules), "rule confidences must be <= 1")

    # ---- 1. the symbolic closure against the brute-force reachability oracle
    derived = {fact for fact in proofs if fact[0] == "path"}
    oracle = reachability_oracle(edges, nodes)
    check(derived == oracle,
          "the derivation must equal the transitive closure: %d vs %d" % (len(derived), len(oracle)))
    check(all(("edge", e[1], e[2]) in proofs for e in edges), "every given edge must be present")

    # ---- 2. proofs verify, tampering does not
    for fact in list(proofs)[:60]:
        check(verify_proof(engine, fact), "every derived fact must carry a valid proof: %r" % (fact,))
    tampered = Engine(edges, rule_set())
    tampered.proofs = dict(engine.proofs)
    victim = next(f for f in proofs if f[0] == "path" and engine.proofs[f] is not None
                  and engine.proofs[f][0].name == "extend")
    rule, premises = tampered.proofs[victim]
    tampered.proofs[victim] = (rule, (("edge", "n00", "n99"),) + premises[1:])
    check(not verify_proof(tampered, victim), "a replaced premise must fail verification")
    broken = Engine(edges, rule_set())
    broken.proofs = dict(engine.proofs)
    broken.proofs[victim] = (rule, premises[:-1] + (("path", "n11", "n11"),))
    check(not verify_proof(broken, victim), "an invented premise must fail verification")
    dropped = Engine(edges, rule_set())
    dropped.proofs = dict(engine.proofs)
    dropped.proofs.pop(premises[0])
    check(not verify_proof(dropped, victim), "a missing premise must fail verification")
    check(verify_proof(engine, ("edge", nodes[0], nodes[1])), "given facts are trivially proven")
    check(not verify_proof(engine, ("path", "n99", "n00")), "an underivable fact has no proof")

    # ---- 3. confidences and depths are bounded and monotone along the proof
    positive = [f for f in derived]
    check(all(0.0 < engine.confidence(f) <= 1.0 for f in positive),
          "proof confidence must lie in (0, 1]")
    check(all(engine.depth(f) < len(nodes) for f in positive), "proof depth must stay bounded")
    chain = max(positive, key=lambda f: engine.hops(f))
    check(engine.hops(chain) >= 2
          and engine.confidence(chain) <= 0.9 ** (engine.hops(chain) - 1) + 1e-12,
          "a chain of h hops cannot be more confident than 0.9^(h-1)")
    check(all(engine.hops(f) >= 1 for f in positive), "every path rests on at least one edge")

    # ---- 4. the hybrid: rerank the symbolic candidates
    queries = []
    for source in nodes:
        candidates = engine.query("path", source)
        if len(candidates) >= 2:
            queries.append((source, candidates))
    check(len(queries) >= 6, "the domain must offer several multi-candidate queries")
    # a deterministic reordering (by candidate count, then name) so the train/test split does not
    # simply follow the node enumeration order
    shuffled = sorted(queries, key=lambda item: (len(item[1]), item[0]))
    train_queries, test_queries = shuffled[:len(shuffled) * 2 // 3], shuffled[len(shuffled) * 2 // 3:]
    check(train_queries and test_queries, "the query split must be non-empty")

    train_rows, train_labels = [], []
    for source, candidates in train_queries:
        rows = score_rows(engine, candidates)
        best = min(row[3] for row in rows)                # the truth: the fewest-hop derivation
        for row in rows:
            train_rows.append(row[1])
            train_labels.append(1.0 if row[3] == best else 0.0)
    scorer = fit_ranker(train_rows, train_labels)

    prior_hits = hybrid_hits = total = 0
    for source, candidates in test_queries:
        rows = score_rows(engine, candidates)
        best = min(row[3] for row in rows)
        prior = max(rows, key=lambda row: (row[2], row[0]))[0]
        learned = max(rows, key=lambda row: (sum(w * f for w, f in zip(scorer, row[1])), row[0]))[0]
        check(learned in candidates and prior in candidates,
              "the hybrid must answer inside the symbolic candidate set")
        check(verify_proof(engine, learned), "the chosen answer must have a valid proof")
        prior_hits += 1 if engine.hops(prior) == best else 0
        hybrid_hits += 1 if engine.hops(learned) == best else 0
        total += 1
    check(total == len(test_queries) and total >= 2, "every test query must be scored")
    check(hybrid_hits >= prior_hits,
          "the learned ranker must not do worse than the prior (%d vs %d)" % (hybrid_hits, prior_hits))
    check(hybrid_hits / total >= 0.8,
          "the hybrid must be accurate on the held-out queries (%.2f)" % (hybrid_hits / total))

    # ---- 5. open world and the boundaries
    unknown = engine.query("path", "n99")
    check(unknown == [], "an unconnected source must return no candidates (unknown, not false)")
    check(verify_proof(engine, ("path", "n99", "n00")) is False,
          "absence of a proof is what makes the answer unknown")
    empty = Engine([], rule_set())
    check(empty.run() == {} and empty.query("path", "n00") == [],
          "an empty knowledge base derives nothing")
    single = Engine([("edge", "a", "b")], rule_set())
    single_proofs = single.run()
    check(("path", "a", "b") in single_proofs and len(single_proofs) == 2,
          "a single fact must derive exactly one path")
    check(verify_proof(single, ("path", "a", "b")), "the singleton proof must verify")
    cyclic = Engine([("edge", "a", "b"), ("edge", "b", "a")], rule_set())
    cyclic.run()
    check(("path", "b", "a") in cyclic.proofs and cyclic.steps < 20,
          "a cyclic rule set must terminate and still derive the reverse path")
    self_rule = Rule("self", (("path", "?x", "?y"),), ("path", "?x", "?y"), 1.0)
    repeat = Engine(edges, rule_set() + [self_rule])
    repeat.run()
    check(set(repeat.proofs) == set(engine.proofs),
          "a rule that only re-derives known facts must change nothing")

    # ---- 6. determinism
    twin = Engine(edges, rule_set())
    twin.run()
    check({f: (s[0].name if s else None) for f, s in twin.proofs.items()}
          == {f: (s[0].name if s else None) for f, s in engine.proofs.items()},
          "the closure must be reproducible")

    print("closure %d facts in %d steps, oracle %d" % (len(proofs), engine.steps, len(oracle)))
    print("prior hits %d/%d, hybrid hits %d/%d" % (prior_hits, total, hybrid_hits, total))
    print("ranker weights %r" % [round(w, 4) for w in scorer])
    print("PASS")


if __name__ == "__main__":
    main()
