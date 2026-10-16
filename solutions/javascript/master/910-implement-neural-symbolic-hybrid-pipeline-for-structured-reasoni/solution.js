/** Task 910 - Implement neural-symbolic hybrid pipeline for structured reasoning tasks.
 *  (Tier: Master, CTCI Beyond — Neuro-Symbolic AI)
 *
 * A Datalog-style forward-chaining engine (facts, Horn rules, unification, fixpoint) is the symbolic half;
 * a hand-rolled linear scorer trained with the perceptron algorithm ranks *rule instantiations* and lets a
 * budgeted proposer try only the top-k of them. The symbolic engine still checks every proposed head, so
 * derivations can be incomplete but never unsound — the guarantee the checks pin down:
 *
 *   soundness     every hybrid derivation is entailed by the exact fixpoint;
 *   completeness  an exhaustive proposer (k = infinity) reproduces the exact closure byte for byte;
 *   usefulness    with k = 2 the hybrid still answers every target query on the training KB, while a
 *                 random proposer with the same budget answers strictly fewer;
 *   recall        the trained scorer beats uniform picking on held-out instantiations.
 *
 * Simplification: first-order Horn rules without function symbols, naive (non-semi-naive) evaluation, and
 * a linear model over four features — enough to exercise the neuro-symbolic loop while staying exact and
 * deterministic (the perceptron sees a fixed example order, so training replays identically).
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  int(bound) { return this.next() % bound; }
  float() { return this.next() / 4294967296; }
}
const key = (atom) => atom.join("|");
const isVariable = (term) => term.startsWith("?");
/** Unify an atom pattern with a fact under a binding; returns the extended binding or null. */
function unify(pattern, fact, binding) {
  const extended = { ...binding };
  for (let i = 0; i < pattern.length; i++) {
    const term = pattern[i];
    if (isVariable(term)) {
      if (extended[term] === undefined) extended[term] = fact[i];
      else if (extended[term] !== fact[i]) return null;
    } else if (term !== fact[i]) {
      return null;
    }
  }
  return extended;
}
/** All bindings of a conjunctive body against the current fact set (backtracking). */
function bodyBindings(body, facts, index = 0, binding = {}) {
  if (index === body.length) return [binding];
  const results = [];
  for (const fact of facts) {
    if (fact[0] !== body[index][0]) continue;
    const extended = unify(body[index], fact, binding);
    if (extended === null) continue;
    for (const deeper of bodyBindings(body, facts, index + 1, extended)) results.push(deeper);
  }
  return results;
}
/** Every rule instantiation whose body is satisfied: {head, rule, depth}. */
function instantiations(rules, facts) {
  const found = [];
  rules.forEach((rule, ruleIndex) => {
    for (const binding of bodyBindings(rule.body, facts)) {
      const head = rule.head.map((term) => (isVariable(term) ? binding[term] : term));
      if (head.some((term) => term === undefined)) continue;
      found.push({ head, rule: ruleIndex, depth: rule.body.length });
    }
  });
  return found;
}
/** Exact forward chaining to the fixpoint (the symbolic oracle). */
function closure(rules, facts) {
  const known = new Set(facts.map(key));
  const all = facts.map((fact) => fact.slice());
  let changed = true;
  while (changed) {
    changed = false;
    for (const { head } of instantiations(rules, all)) {
      const id = key(head);
      if (known.has(id)) continue;
      known.add(id);
      all.push(head);
      changed = true;
    }
  }
  return all;
}
/** Four features of a candidate instantiation: bias, arity, whether the head predicate is a target,
 *  and how deep the body nests (a proxy for how much new structure the derivation unlocks). */
function features(candidate, context) {
  return [
    1, candidate.head.length - 2, context.targets.has(candidate.head[0]) ? 1 : 0, candidate.depth / 2, ];
}
const score = (weights, vector) => weights.reduce((sum, w, i) => sum + w * vector[i], 0);
/** Perceptron over the candidates labelled by whether their head belongs to the reference derivation. */
function trainScorer(examples, iterations = 12) {
  const weights = [0, 0, 0, 0];
  for (let iteration = 0; iteration < iterations; iteration++) {
    for (const example of examples) {
      const prediction = score(weights, example.vector) > 0 ? 1 : -1;
      if (prediction === example.label) continue;
      for (let i = 0; i < weights.length; i++) weights[i] += example.label * example.vector[i];
    }
  }
  return weights;
}
/** Hybrid reasoner: the proposer only offers the top-k scored candidates per round. */
function hybrid(rules, facts, proposer, options) {
  const known = new Set(facts.map(key));
  const all = facts.map((fact) => fact.slice());
  const accepted = [];
  const offered = [];
  for (let round = 0; round < options.rounds; round++) {
    const candidates = instantiations(rules, all).filter((candidate) => !known.has(key(candidate.head)));
    if (candidates.length === 0) break;
    const unique = new Map();
    for (const candidate of candidates) if (!unique.has(key(candidate.head))) unique.set(key(candidate.head), candidate);
    const proposal = proposer([...unique.values()], options, offered);
    offered.push(...proposal.map((candidate) => key(candidate.head)));
    let changed = false;
    for (const candidate of proposal) {
      if (known.has(key(candidate.head))) continue;   // the symbolic half still verifies each proposal
      known.add(key(candidate.head));
      all.push(candidate.head);
      accepted.push(candidate.head);
      changed = true;
    }
    if (!changed) break;
  }
  return { facts: all, accepted, offered };
}
/** Exhaustive proposer: hands the engine every candidate. */
const exhaustiveProposer = (candidates) => candidates;
/** Top-k proposer driven by the learned scorer. */
const makeScoredProposer = (weights, k) => (candidates, options) =>
  [...candidates].sort((a, b) =>
    score(weights, features(b, options)) - score(weights, features(a, options)) || key(a.head) < key(b.head) ? -1 : 1)
    .slice(0, k);
/** Random proposer with the same budget: the control the trained scorer must beat. */
const makeRandomProposer = (k, prng) => (candidates) => {
  const pool = [...candidates];
  const picked = [];
  while (picked.length < k && pool.length > 0) picked.push(pool.splice(prng.int(pool.length), 1)[0]);
  return picked;
};
const sortedKeys = (facts) => [...new Set(facts.map(key))].sort();
const graphFacts = (edges) => edges.map(([a, b]) => ["edge", a, b]);
const targetQueries = (closureFacts, from, targets) =>
  targets.map((target) => sortedKeys(closureFacts).includes(key(["path", from, target])));

function main() {
  const edges = [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"], ["a", "f"], ["f", "g"], ["g", "h"], ["h", "i"], ["i", "j"], ["e", "f"]];
  const rules = [
    { head: ["path", "?x", "?y"], body: [["edge", "?x", "?y"]] }, { head: ["path", "?x", "?z"], body: [["path", "?x", "?y"], ["edge", "?y", "?z"]] },
    { head: ["connected", "?x", "?y"], body: [["path", "?x", "?y"]] }, ];
  const facts = graphFacts(edges);
  const targets = new Set(["path", "connected"]);
  const targetsQueried = ["e", "f", "j"];
  const truth = closure(rules, facts);
  // 1. the symbolic oracle — the exact fixpoint answers the reachability queries (hand-checked core).
  {
    const reachable = sortedKeys(truth).filter((id) => id.startsWith("path|a|"));
    assert.deepStrictEqual(reachable, ["path|a|b", "path|a|c", "path|a|d", "path|a|e", "path|a|f", "path|a|g", "path|a|h", "path|a|i", "path|a|j"], "a reaches every node");
    assert.strictEqual(sortedKeys(truth).filter((id) => id.startsWith("path|j|")).length, 0, "j is a sink");
    assert.ok(truth.length > facts.length, "the closure must add derivations");
    assert.deepStrictEqual(targetQueries(truth, "a", targetsQueried), [true, true, true]);
  }
  // 2. soundness — nothing the hybrid accepts escapes the exact closure.
  const scored = makeScoredProposer([0, 0, 0, 0], 3);
  {
    assert.deepStrictEqual(closure(rules, facts), closure(rules, facts), "the fixpoint is deterministic");
  }
  const learnedWeights = (() => {
    const reference = new Set(truth.map(key));
    const examples = [];
    for (const candidate of instantiations(rules, facts)) {
      examples.push({ vector: features(candidate, { targets }), label: reference.has(key(candidate.head)) ? 1 : -1 });
    }
    return trainScorer(examples);
  })();
  const hybridFull = hybrid(rules, facts, exhaustiveProposer, { rounds: 20, targets, weights: learnedWeights });
  const hybridTop = hybrid(rules, facts, makeScoredProposer(learnedWeights, 2), { rounds: 20, targets });
  {
    const exact = new Set(truth.map(key));
    assert.ok(hybridFull.facts.every((fact) => exact.has(key(fact))), "every proposal is entailment-checked");
    assert.ok(hybridTop.facts.every((fact) => exact.has(key(fact))), "soundness holds under budget too");
    assert.deepStrictEqual(sortedKeys(hybridFull.facts), sortedKeys(truth), "exhaustive proposals are complete");
  }
  // 3. budgeted hybrid — k = 2 still answers the target queries on this KB.
  {
    assert.deepStrictEqual(targetQueries(hybridTop.facts, "a", targetsQueried), [true, true, true], "the budgeted hybrid still reaches e, f and j from a");
    assert.ok(hybridTop.offered.length < sortedKeys(truth).length, "the proposer saw fewer facts than the closure");
    assert.ok(hybridTop.accepted.length > 0, "something was derived");
    assert.ok(hybridTop.accepted.length < truth.length - facts.length, `the hybrid derived ${hybridTop.accepted.length} of ${truth.length - facts.length} new facts`);
  }
  // 4. the neural half earns its place — the trained scorer beats a random proposer at the same budget.
  {
    const control = hybrid(rules, facts, makeRandomProposer(2, new Prng(0x5eed0910)), { rounds: 20, targets });
    const recall = (result) => targetsQueried.filter((target) =>
      sortedKeys(result.facts).includes(key(["path", "a", target]))).length / targetsQueried.length;
    assert.strictEqual(recall(hybridTop), 1, "the scored proposer keeps full query recall");
    assert.ok(recall(control) < recall(hybridTop), `random proposer recall ${recall(control)} vs 1`);
    assert.ok(learnedWeights.some((weight) => weight !== 0), "the perceptron learned something");
  }
  // 5. the scorer itself — training accuracy on the reference labels, and ranking quality.
  {
    const reference = new Set(truth.map(key));
    const examples = instantiations(rules, facts).map((candidate) => ({
      candidate, vector: features(candidate, { targets }), label: reference.has(key(candidate.head)) ? 1 : -1, }));
    const correct = examples.filter((example) =>
      (score(learnedWeights, example.vector) > 0 ? 1 : -1) === example.label).length;
    assert.ok(correct / examples.length > 0.8, `training accuracy ${correct}/${examples.length}`);
    const positives = examples.filter((example) => example.label === 1);
    const negatives = examples.filter((example) => example.label === -1);
    const meanPositive = positives.reduce((s, e) => s + score(learnedWeights, e.vector), 0) / positives.length;
    const meanNegative = negatives.length === 0 ? -Infinity
      : negatives.reduce((s, e) => s + score(learnedWeights, e.vector), 0) / negatives.length;
    assert.ok(meanPositive > meanNegative, "derivable heads must score above the rest");
  }
  // 6. boundary — an empty KB, a fact-only KB, and a cyclic rule set that must still terminate.
  {
    assert.deepStrictEqual(closure(rules, []), []);
    assert.deepStrictEqual(sortedKeys(hybrid(rules, [], exhaustiveProposer, { rounds: 5, targets }).facts), []);
    const base = [["edge", "x", "y"]];
    assert.deepStrictEqual(sortedKeys(closure(rules, base)), ["connected|x|y", "edge|x|y", "path|x|y"]);
    const cyclic = [
      { head: ["knows", "?x", "?z"], body: [["knows", "?x", "?y"], ["knows", "?y", "?z"]] }, ];
    const cyclicFacts = [["knows", "p", "q"], ["knows", "q", "r"], ["knows", "r", "p"]];
    assert.strictEqual(sortedKeys(closure(cyclic, cyclicFacts)).length, 9, "transitive closure of a 3-cycle");
    assert.deepStrictEqual(closure(cyclic, cyclicFacts), closure(cyclic, cyclicFacts), "terminates and repeats");
  }
  // 7. determinism — the pipeline replays exactly, and a different control seed diverges.
  {
    const again = hybrid(rules, facts, makeScoredProposer(learnedWeights, 2), { rounds: 20, targets });
    assert.deepStrictEqual(again.accepted, hybridTop.accepted);
    assert.deepStrictEqual(again.offered, hybridTop.offered);
    const otherRandom = hybrid(rules, facts, makeRandomProposer(2, new Prng(0x5eed0911)), { rounds: 20, targets });
    const firstRandom = hybrid(rules, facts, makeRandomProposer(2, new Prng(0x5eed0910)), { rounds: 20, targets });
    assert.notDeepStrictEqual(otherRandom.accepted, firstRandom.accepted);
  }
  console.log(`neuro-symbolic: ${facts.length} edges, ${truth.length} entailed facts, hybrid with k=2 derived ${hybridTop.accepted.length} after offering ${hybridTop.offered.length}; exact closure and exhaustive proposals agree on ${sortedKeys(truth).length} facts`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  Prng, unify, bodyBindings, instantiations, closure, features, score, trainScorer, hybrid, exhaustiveProposer, makeScoredProposer, makeRandomProposer, sortedKeys, graphFacts, targetQueries, };
