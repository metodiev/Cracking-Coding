/** Task 867 - Implement counterfactual explanation generator for classifiers (optimization-based).
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * Model. A counterfactual explainer for a tabular classifier. Every feature carries an actionability
 * specification — immutable, bounded numeric with a step and an optional monotone direction, or
 * categorical with an allowed value set — and the explainer searches for the cheapest instance x' that
 * the classifier labels with the requested class:
 *
 *     minimise   cost(x, x') = lambda0 * |{j : x'_j != x_j}| + lambda1 * sum_j d_j(x_j, x'_j)
 *     subject to argmax class(x') = target, all actionability constraints respected.
 *
 * The search is optimisation based: a greedy coordinate pass that repeatedly moves the single feature
 * with the best (cost, margin) trade-off, followed by seeded random restarts that spend the sparsity
 * budget in different ways; the best valid candidate wins, and the explainer reports the cost, the
 * changed features and the number of classifier calls it spent. An exhaustive enumerator over the same
 * discrete candidate space provides the optimal reference, so the search can be measured for both validity
 * and optimality rather than just plausibility.
 *
 * What is simplified: a linear-threshold classifier instead of a gradient-boosted model, a discrete
 * candidate grid per feature instead of continuous optimisation with gradient descent, greedy plus random
 * restarts instead of a mixed-integer program or a learned generator, and no diversity or plausibility
 * metric beyond actionability. Randomness is one seeded mulberry32 stream — no wall clock, no threads.
 *
 * Invariants verified:
 *   validity    every returned counterfactual really is labelled with the target class, respects every
 *               actionability constraint (immutability, bounds, step, monotone direction, categorical
 *               domain), and honours the sparsity budget; a null return therefore means "no counterfactual
 *               exists inside the budget", which is checked against the exhaustive oracle.
 *   optimality  on small discrete instances the search matches the exhaustive optimum exactly, and across a
 *               larger corpus it never exceeds it and stays within a documented factor of it.
 *   efficiency  the produced counterfactual is strictly closer to the instance than random valid edits
 *               found by a baseline that draws candidates from the same candidate space.
 *   cost        the number of classifier evaluations per explanation stays inside the documented budget.
 *   determinism identical seeds reproduce identical counterfactuals, costs and evaluation counts.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — the only source of randomness (restarts and the random-edit baseline). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Feature specification: everything the search has to respect. */
function numericFeature({ name, min, max, step = 1, immutable = false, monotone = "any", weight = 1 }) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min >= max) throw new RangeError(`bad range for ${name}`);
  if (!(step > 0)) throw new RangeError(`bad step for ${name}`);
  if (!["any", "increase", "decrease"].includes(monotone)) throw new RangeError(`bad monotone for ${name}`);
  return { name, type: "numeric", min, max, step, immutable, monotone, weight };
}

function categoricalFeature({ name, values, immutable = false, weight = 1 }) {
  if (!Array.isArray(values) || values.length < 2) throw new RangeError(`bad values for ${name}`);
  return { name, type: "categorical", values: values.slice(), immutable, weight };
}

const featureOf = (specs, name) => specs.find((spec) => spec.name === name);

/** Linear-threshold classifier over the featurised instance (numeric as-is, categorical one-hot). */
function makeClassifier({ specs, coefficients, bias = 0, threshold = 0 }) {
  const categorical = specs.filter((spec) => spec.type === "categorical");
  const dimension =
    specs.filter((spec) => spec.type === "numeric").length + categorical.reduce((acc, spec) => acc + spec.values.length, 0);
  if (coefficients.length !== dimension) throw new RangeError("coefficient count must match the featurised dimension");
  const encode = (instance) => {
    const parts = [];
    for (const spec of specs) if (spec.type === "numeric") parts.push(instance[spec.name]);
    for (const spec of categorical) {
      for (const value of spec.values) parts.push(instance[spec.name] === value ? 1 : 0);
    }
    return parts;
  };
  const score = (instance) => {
    const encoded = encode(instance);
    let total = bias;
    for (let i = 0; i < encoded.length; i++) total += coefficients[i] * encoded[i];
    return total;
  };
  return { score, classify: (instance) => (score(instance) > threshold ? 1 : 0), threshold, encode, dimension };
}

/** Distance contributions of one feature between two values. */
function featureDistance(spec, from, to) {
  if (spec.type === "categorical") return from === to ? 0 : spec.weight;
  return spec.weight * Math.abs(to - from);
}

function costOf(specs, from, to, { lambda0 = 1, lambda1 = 0.05 } = {}) {
  let l0 = 0;
  let l1 = 0;
  for (const spec of specs) {
    if (from[spec.name] !== to[spec.name]) l0 += 1;
    l1 += featureDistance(spec, from[spec.name], to[spec.name]);
  }
  return { cost: lambda0 * l0 + lambda1 * l1, l0, l1 };
}

/** Snapshots the value into the feature domain (bounds, step, allowed values, monotone direction). */
function candidateValues(spec, current) {
  if (spec.type === "categorical") return spec.values.filter((value) => value !== current);
  const out = [];
  const steps = Math.max(1, Math.ceil((spec.max - spec.min) / spec.step));
  for (let i = 0; i <= steps; i++) {
    let value = spec.min + i * spec.step;
    if (value > spec.max) value = spec.max;
    if (value === current) continue;
    if (spec.monotone === "increase" && value < current) continue;
    if (spec.monotone === "decrease" && value > current) continue;
    if (!out.includes(value)) out.push(value);
  }
  return out.sort((a, b) => Math.abs(a - current) - Math.abs(b - current));
}

/** The actionability constraints that a counterfactual must still satisfy. */
function violatesConstraints(specs, from, to) {
  for (const spec of specs) {
    const value = to[spec.name];
    if (spec.type === "categorical") {
      if (!spec.values.includes(value)) return `illegal category for ${spec.name}`;
    } else {
      if (typeof value !== "number" || !Number.isFinite(value)) return `non-numeric ${spec.name}`;
      if (value < spec.min - 1e-9 || value > spec.max + 1e-9) return `${spec.name} outside its bounds`;
      if (spec.monotone === "increase" && value < from[spec.name]) return `${spec.name} may only increase`;
      if (spec.monotone === "decrease" && value > from[spec.name]) return `${spec.name} may only decrease`;
    }
    if (spec.immutable && value !== from[spec.name]) return `${spec.name} is immutable`;
  }
  return null;
}

/**
 * The optimization-based explainer: a greedy coordinate pass followed by random restarts, all scoring
 * candidates with the same objective and counting every classifier call.
 */
function explain({ specs, classifier, instance, target, budget = 3, lambda0 = 1, lambda1 = 0.05, restarts = 60, randomRestartEdits = 2, seed = 867 }) {
  if (!Array.isArray(specs) || specs.length === 0) throw new RangeError("specs must be a non-empty array");
  if (typeof classifier !== "function") throw new TypeError("classifier must be a function");
  if (instance === null || typeof instance !== "object") throw new TypeError("instance must be an object");
  if (!Number.isInteger(budget) || budget < 0) throw new RangeError("budget must be >= 0");
  if (!Number.isInteger(restarts) || restarts < 0) throw new RangeError("restarts must be >= 0");
  for (const spec of specs) {
    if (!(spec.name in instance)) throw new RangeError(`instance is missing ${spec.name}`);
  }
  let calls = 0;
  const classify = (candidate) => {
    calls += 1;
    return classifier(candidate);
  };
  const evaluate = (candidate) => {
    const violations = violatesConstraints(specs, instance, candidate);
    if (violations !== null) return null;
    const cost = costOf(specs, instance, candidate, { lambda0, lambda1 });
    if (cost.l0 > budget) return null;
    if (classify(candidate) !== target) return null;
    return cost;
  };
  const emptyCost = (candidate) => costOf(specs, instance, candidate, { lambda0, lambda1 });
  if (classify(instance) === target) {
    return { instance: { ...instance }, ...emptyCost(instance), target, changed: [], calls, found: true, method: "already-there" };
  }
  let best = null;

  /** Greedy pass: keep moving the feature that improves the objective the most. */
  const greedy = (start, maxEdits) => {
    let current = { ...start };
    let edits = 0;
    const used = new Set();
    while (edits < maxEdits) {
      let improvement = null;
      for (const spec of specs) {
        if (spec.immutable || used.has(spec.name)) continue;
        for (const value of candidateValues(spec, current[spec.name])) {
          const candidate = { ...current, [spec.name]: value };
          const result = evaluate(candidate);
          if (result !== null && (improvement === null || result.cost < improvement.cost)) {
            improvement = { candidate, cost: result.cost, kind: result, name: spec.name };
          }
        }
      }
      if (improvement === null) break;
      current = improvement.candidate;
      edits += 1;
      used.add(improvement.name);
      const total = emptyCost(current);
      if (best === null || total.cost < best.cost) best = { instance: { ...current }, ...total, changed: changedFeatures(specs, instance, current) };
    }
    return current;
  };
  greedy(instance, budget);

  /** Random restarts: spend the budget on different feature subsets and keep the cheapest valid result. */
  const rand = mulberry32(seed);
  for (let attempt = 0; attempt < restarts; attempt++) {
    const mutable = specs.filter((spec) => !spec.immutable);
    if (mutable.length === 0) break;
    const edits = 1 + Math.floor(rand() * Math.min(randomRestartEdits, budget));
    let candidate = { ...instance };
    for (let e = 0; e < edits; e++) {
      const spec = mutable[Math.floor(rand() * mutable.length)];
      const options = candidateValues(spec, instance[spec.name]);
      if (options.length === 0) continue;
      candidate[spec.name] = options[Math.floor(rand() * options.length)];
    }
    const cost = emptyCost(candidate);
    if (cost.l0 > budget) continue;
    if (classify(candidate) === target) {
      if (best === null || cost.cost < best.cost - 1e-12) {
        best = { instance: { ...candidate }, ...cost, changed: changedFeatures(specs, instance, candidate) };
      }
      continue;
    }
    const refined = greedy(candidate, Math.max(0, budget - cost.l0));
    const refinedCost = emptyCost(refined);
    if (refinedCost.l0 <= budget && classify(refined) === target) {
      if (best === null || refinedCost.cost < best.cost - 1e-12) {
        best = { instance: { ...refined }, ...refinedCost, changed: changedFeatures(specs, instance, refined) };
      }
    }
  }
  if (best === null) {
    return { instance: null, cost: Infinity, l0: 0, l1: 0, target, changed: [], calls, found: false, method: "none" };
  }
  return { ...best, target, calls, found: true, method: "search" };
}

function changedFeatures(specs, from, to) {
  return specs.filter((spec) => from[spec.name] !== to[spec.name]).map((spec) => spec.name);
}

/** Exhaustive oracle: enumerate every combination of up to `budget` feature edits from the candidate grid. */
function exhaustiveCounterfactual({ specs, classifier, instance, target, budget = 3, lambda0 = 1, lambda1 = 0.05 }) {
  const mutable = specs.filter((spec) => !spec.immutable);
  const options = mutable.map((spec) => ({
    name: spec.name,
    values: candidateValues(spec, instance[spec.name]),
  }));
  let calls = 0;
  let best = null;
  const evaluate = (candidate) => {
    const cost = costOf(specs, instance, candidate, { lambda0, lambda1 });
    if (cost.l0 > budget) return;
    calls += 1;
    if (classifier(candidate) === target && (best === null || cost.cost < best.cost)) {
      best = { instance: { ...candidate }, ...cost, changed: changedFeatures(specs, instance, candidate) };
    }
  };
  evaluate({ ...instance });
  const walk = (index, candidate, edits) => {
    if (index >= options.length || edits > budget) return;
    walk(index + 1, candidate, edits);
    for (const value of options[index].values) {
      candidate[options[index].name] = value;
      evaluate({ ...candidate });
      walk(index + 1, candidate, edits + 1);
      candidate[options[index].name] = instance[options[index].name];
    }
  };
  walk(0, { ...instance }, 0);
  return best === null ? { found: false, calls } : { ...best, found: true, calls };
}

/** Baseline: random valid edits inside the candidate space, used to show the search is not just lucky. */
function randomEditBaseline({ specs, classifier, instance, target, budget = 3, lambda0 = 1, lambda1 = 0.05, attempts = 200, seed = 5 }) {
  const rand = mulberry32(seed);
  const mutable = specs.filter((spec) => !spec.immutable);
  let calls = 0;
  const valid = [];
  for (let i = 0; i < attempts; i++) {
    const candidate = { ...instance };
    const edits = 1 + Math.floor(rand() * budget);
    for (let e = 0; e < edits; e++) {
      if (mutable.length === 0) break;
      const spec = mutable[Math.floor(rand() * mutable.length)];
      const options = candidateValues(spec, instance[spec.name]);
      if (options.length === 0) continue;
      candidate[spec.name] = options[Math.floor(rand() * options.length)];
    }
    const cost = costOf(specs, instance, candidate, { lambda0, lambda1 });
    if (cost.l0 > budget) continue;
    calls += 1;
    if (classifier(candidate) === target) valid.push(cost.cost);
  }
  return { validCount: valid.length, calls, averageCost: valid.length === 0 ? Infinity : valid.reduce((a, b) => a + b, 0) / valid.length };
}

/** The fixture: a small credit-approval model with mixed actionable features. */
function makeFixture() {
  const specs = [
    numericFeature({ name: "age", min: 18, max: 70, step: 5, monotone: "increase", weight: 0.5 }),
    numericFeature({ name: "income", min: 0, max: 100, step: 10, weight: 1 }),
    numericFeature({ name: "debt", min: 0, max: 100, step: 10, monotone: "decrease", weight: 1 }),
    numericFeature({ name: "history", min: 0, max: 10, step: 1, immutable: true, weight: 1 }),
    categoricalFeature({ name: "employment", values: ["none", "part", "full"], weight: 1 }),
  ];
  // numeric features first, then the one-hot employment block
  const coefficients = [0.4, 1.6, -1.4, 0.9, -1.0, 0.5, 2.0];
  const classifier = makeClassifier({ specs, coefficients, bias: -30, threshold: 0 });
  return { specs, classifier: classifier.classify, score: classifier.score, encode: classifier.encode };
}

function selfCheck() {
  // --- validation -----------------------------------------------------------------------------------------
  assert.throws(() => numericFeature({ name: "a", min: 5, max: 1 }), /bad range/);
  assert.throws(() => numericFeature({ name: "a", min: 0, max: 10, step: 0 }), /bad step/);
  assert.throws(() => numericFeature({ name: "a", min: 0, max: 10, monotone: "up" }), /bad monotone/);
  assert.throws(() => categoricalFeature({ name: "c", values: ["only"] }), /bad values/);
  assert.throws(() => makeClassifier({ specs: [numericFeature({ name: "a", min: 0, max: 1 })], coefficients: [1, 2] }), /coefficient count/);
  const fixture = makeFixture();
  assert.throws(() => explain({ specs: fixture.specs, classifier: fixture.classifier, instance: {}, target: 1 }), /missing/);
  assert.throws(() => explain({ specs: [], classifier: fixture.classifier, instance: {}, target: 1 }), /specs/);
  assert.throws(() => explain({ specs: fixture.specs, classifier: null, instance: {}, target: 1 }), /classifier must be/);
  assert.throws(
    () => explain({ specs: fixture.specs, classifier: fixture.classifier, instance: { age: 30, income: 40, debt: 20, history: 5, employment: "part" }, target: 1, budget: -1 }),
    /budget/,
  );

  // --- a hand-checked example -------------------------------------------------------------------------------
  const { specs, classifier } = fixture;
  const base = { age: 25, income: 20, debt: 60, history: 4, employment: "none" };
  assert.strictEqual(classifier(base), 0, "the fixture instance must start in the negative class");
  const explanation = explain({ specs, classifier, instance: base, target: 1, budget: 3 });
  assert.strictEqual(explanation.found, true, "a counterfactual must exist for this fixture");
  assert.strictEqual(classifier(explanation.instance), 1, "the counterfactual must be labelled with the target class");
  assert.ok(explanation.l0 <= 3 && explanation.l0 === explanation.changed.length);
  assert.ok(explanation.l1 >= 0);
  assert.ok(Math.abs(explanation.cost - (1 * explanation.l0 + 0.05 * explanation.l1)) < 1e-12, "the reported cost must match its definition");
  assert.strictEqual(violatesConstraints(specs, base, explanation.instance), null, "constraints must be respected");
  assert.strictEqual(explanation.instance.history, 4, "an immutable feature must not move");
  assert.ok(["none", "part", "full"].includes(explanation.instance.employment));
  assert.ok(explanation.instance.age >= base.age, "a monotone feature may only move in the allowed direction");
  assert.ok(explanation.instance.debt <= base.debt);
  const recomputed = costOf(specs, base, explanation.instance);
  assert.strictEqual(recomputed.l0, explanation.l0);
  assert.ok(Math.abs(recomputed.cost - explanation.cost) < 1e-12, "L0/L1 must be recomputable from the returned instance");

  // --- optimality against the exhaustive oracle on small instances ---------------------------------------------
  const instances = [
    { age: 25, income: 20, debt: 60, history: 4, employment: "none" },
    { age: 30, income: 30, debt: 50, history: 6, employment: "part" },
    { age: 22, income: 10, debt: 70, history: 2, employment: "none" },
    { age: 45, income: 50, debt: 30, history: 8, employment: "part" },
    { age: 35, income: 40, debt: 40, history: 5, employment: "full" },
  ];
  let optimalMatches = 0;
  let searched = 0;
  for (const instance of instances) {
    for (const target of [1, 0]) {
      if (classifier(instance) === target) continue;
      const brute = exhaustiveCounterfactual({ specs, classifier, instance, target, budget: 2 });
      const searchedResult = explain({ specs, classifier, instance, target, budget: 2 });
      searched += 1;
      assert.strictEqual(searchedResult.found, brute.found, "the search and the oracle must agree on existence");
      if (!brute.found) {
        assert.strictEqual(searchedResult.instance, null, "no counterfactual means no instance");
        continue;
      }
      assert.ok(searchedResult.cost >= brute.cost - 1e-12, "the search cannot beat the optimum");
      assert.ok(Math.abs(searchedResult.cost - brute.cost) < 1e-9, `the search must find the optimum (${searchedResult.cost} vs ${brute.cost})`);
      assert.strictEqual(searchedResult.l0, brute.l0);
      optimalMatches += 1;
    }
  }
  assert.ok(searched >= 5 && optimalMatches === searched, `every search must reach the optimum (${searched} searches, ${optimalMatches} optima)`);

  // --- a constrained instance where no counterfactual exists inside the budget ---------------------------------
  const stuck = { age: 70, income: 0, debt: 100, history: 0, employment: "none" };
  const impossible = explain({ specs, classifier, instance: stuck, target: 1, budget: 1 });
  const impossibleOracle = exhaustiveCounterfactual({ specs, classifier, instance: stuck, target: 1, budget: 1 });
  assert.strictEqual(impossible.found, impossibleOracle.found);
  if (!impossible.found) {
    assert.strictEqual(impossible.instance, null, "an unreachable target must return null, never a fake explanation");
    assert.strictEqual(impossible.cost, Infinity);
  }
  const smallBudget = explain({ specs, classifier, instance: base, target: 1, budget: 0 });
  assert.strictEqual(smallBudget.found, false, "with a zero budget only an already-conforming instance works");
  const already = explain({ specs, classifier, instance: { age: 60, income: 100, debt: 0, history: 10, employment: "full" }, target: 1, budget: 3 });
  assert.strictEqual(already.found, true);
  assert.strictEqual(already.l0, 0, "an instance already in the target class needs no change");
  assert.strictEqual(already.changed.length, 0);
  assert.strictEqual(already.method, "already-there");

  // --- validity rate, constraint safety and proximity over a random corpus -----------------------------------------
  const rand = mulberry32(867);
  let found = 0;
  let attempts = 0;
  let totalSearchCost = 0;
  let totalBaselineCost = 0;
  let comparable = 0;
  let maxCalls = 0;
  for (let i = 0; i < 60; i++) {
    const instance = {
      age: 18 + Math.floor(rand() * 11) * 5,
      income: Math.floor(rand() * 11) * 10,
      debt: Math.floor(rand() * 11) * 10,
      history: Math.floor(rand() * 11),
      employment: ["none", "part", "full"][Math.floor(rand() * 3)],
    };
    const target = classifier(instance) === 1 ? 0 : 1;
    attempts += 1;
    const result = explain({ specs, classifier, instance, target, budget: 3 });
    maxCalls = Math.max(maxCalls, result.calls);
    const oracle = exhaustiveCounterfactual({ specs, classifier, instance, target, budget: 3 });
    assert.strictEqual(result.found, oracle.found, `existence must agree for instance ${JSON.stringify(instance)}`);
    if (!result.found) continue;
    found += 1;
    assert.strictEqual(classifier(result.instance), target, "every returned counterfactual must be valid");
    assert.strictEqual(violatesConstraints(specs, instance, result.instance), null, "every returned counterfactual must be legal");
    assert.ok(result.l0 <= 3);
    if (oracle.found) {
      assert.ok(result.cost >= oracle.cost - 1e-12);
      assert.ok(result.cost <= oracle.cost * 1.05 + 1e-9, `the search must stay near optimal (${result.cost} vs ${oracle.cost})`);
      totalSearchCost += result.cost;
      const baseline = randomEditBaseline({ specs, classifier, instance, target, budget: 3, seed: 100 + i });
      if (Number.isFinite(baseline.averageCost)) {
        totalBaselineCost += baseline.averageCost;
        comparable += 1;
      }
    }
  }
  assert.ok(found >= attempts * 0.7, `most instances must have a counterfactual in this fixture (${found}/${attempts})`);
  assert.ok(comparable > 20, `the proximity comparison needs enough instances (${comparable})`);
  assert.ok(totalSearchCost < totalBaselineCost, `the search must beat random valid edits (${totalSearchCost.toFixed(2)} vs ${totalBaselineCost.toFixed(2)})`);
  assert.ok(maxCalls < 40000, `the classifier-call budget must stay bounded (${maxCalls} worst case)`);

  // --- a categorical-only action and an immutable-heavy fixture ------------------------------------------------------
  const rigidSpecs = [
    categoricalFeature({ name: "plan", values: ["basic", "plus", "premium"] }),
    numericFeature({ name: "seats", min: 1, max: 9, step: 1, immutable: true }),
  ];
  const rigid = makeClassifier({ specs: rigidSpecs, coefficients: [0, 1, 3, 2], bias: -1.5, threshold: 0 });
  const rigidInstance = { plan: "basic", seats: 4 };
  assert.strictEqual(rigid.classify(rigidInstance), 0);
  const rigidExplanation = explain({ specs: rigidSpecs, classifier: rigid.classify, instance: rigidInstance, target: 1, budget: 1 });
  assert.strictEqual(rigidExplanation.found, true);
  assert.ok(["plus", "premium"].includes(rigidExplanation.instance.plan), "the plan must move to a class-flipping category");
  assert.strictEqual(rigid.classify(rigidExplanation.instance), 1);
  assert.strictEqual(rigidExplanation.instance.seats, 4, "the immutable feature stays put");
  assert.strictEqual(rigidExplanation.l0, 1);
  const rigidOracle = exhaustiveCounterfactual({ specs: rigidSpecs, classifier: rigid.classify, instance: rigidInstance, target: 1, budget: 1 });
  assert.ok(Math.abs(rigidExplanation.cost - rigidOracle.cost) < 1e-12, "the categorical edit must also be optimal");

  // --- determinism ------------------------------------------------------------------------------------------------------
  const runA = explain({ specs, classifier, instance: base, target: 1, budget: 3, seed: 42 });
  const runB = explain({ specs, classifier, instance: base, target: 1, budget: 3, seed: 42 });
  assert.deepStrictEqual(runA, runB);
  const runC = explain({ specs, classifier, instance: base, target: 1, budget: 3, seed: 43 });
  assert.strictEqual(runC.found, true);
  assert.strictEqual(runC.cost, runA.cost, "another seed still reaches the optimum on this fixture");
  const orbitA = randomEditBaseline({ specs, classifier, instance: base, target: 1, seed: 7 });
  const orbitB = randomEditBaseline({ specs, classifier, instance: base, target: 1, seed: 7 });
  assert.deepStrictEqual(orbitA, orbitB);
  const oracleA = exhaustiveCounterfactual({ specs, classifier, instance: base, target: 1, budget: 2 });
  const oracleB = exhaustiveCounterfactual({ specs, classifier, instance: base, target: 1, budget: 2 });
  assert.deepStrictEqual(oracleA, oracleB);

  console.log(`fixture: ${specs.length} features (${specs.filter((spec) => spec.immutable).length} immutable), instance ${JSON.stringify(base)} -> target ${1}`);
  console.log(`explanation: ${JSON.stringify(explanation.instance)} changed ${explanation.changed.join(",")} cost ${explanation.cost.toFixed(3)} (L0 ${explanation.l0}, L1 ${explanation.l1}) in ${explanation.calls} classifier calls`);
  console.log(`corpus: ${found}/${attempts} explainable, search cost ${totalSearchCost.toFixed(2)} vs random-edit ${totalBaselineCost.toFixed(2)}, worst-case calls ${maxCalls}`);
  console.log("PASS");
}

module.exports = { explain, exhaustiveCounterfactual, randomEditBaseline, makeFixture, makeClassifier, numericFeature, categoricalFeature, candidateValues, violatesConstraints, costOf, changedFeatures, mulberry32 };

if (require.main === module) {
  selfCheck();
}
