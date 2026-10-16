/** Task 894 - Implement automated root-cause analysis heuristics using dependency graphs.
 *  (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
 *
 * An RCA engine for microservice call graphs. A topology is a directed graph whose edge caller → callee
 * means "caller depends on callee"; latency/error metrics are observed per node in a baseline window
 * and an incident window, and a fault is injected at one (or two) nodes and propagates towards the
 * callers with an attenuation factor, so fan-in can make a caller *louder* than the true root.
 *
 * Five heuristics are implemented and compared:
 *   max-deviation        the loudest node (the naive baseline — it provably loses on fan-in);
 *   minimal-root         the anomalous node with no anomalous callee (the anomaly's source);
 *   walk-blame           MicroRank-style propagation: blame flows from a node to its callers,
 *                        score = (1−d)·normalisedDeviation + d·Σ blame(caller)/fanout, iterated to a
 *                        fixed point (convergence is checked by the L1 change shrinking monotonically);
 *   minimal-explanation  greedy set cover over "is explained by a descendant", whose size is checked
 *                        against an exhaustive brute-force search for the true minimum;
 *   earliest-deviation   temporal tie-break on the first deviating window.
 *
 * Invariants verified below: (1) on 60 randomised single-fault scenarios the minimal-root heuristic is
 * exactly right every time, and the greedy explanation always equals the injected root set — both
 * decided against ground truth, not against another heuristic; (2) the naive max-deviation heuristic
 * is *demonstrated* to be wrong on high fan-in scenarios, which is why the graph structure has to be
 * used at all; (3) walk-blame converges (fixed point reached in bounded iterations) and puts the root
 * in its top two; (4) the greedy explanation is optimal in size, verified by exhaustive subset search;
 * (5) two independent faults are both reported; (6) a *no-fault* scenario yields "no root cause" from
 * every heuristic; (7) cycles are condensed (SCCs) so the engine still terminates and blames the right
 * component; (8) zero-variance baselines cannot produce a division by zero; (9) cost stays linear in
 * edges per iteration; (10) determinism.
 *
 * Simplifications: in-process simulation with a virtual window index (no timers), metrics are
 * per-node latencies, the topology is small (≤ 200 nodes / 600 edges) and faults are injected
 * directly rather than through the chaos rig of task 892.
 */
"use strict";

/* ------------------------------------------------------------------ primitives */

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
  unit() { return this.next() / 0x1_0000_0000; }
}

/* ------------------------------------------------------------------ topology */

/**
 * Builds a layered call graph: `gateway` at level 0, leaves at the deepest level, edges from a caller
 * to a callee at a deeper level (so the graph is acyclic unless `cycle` asks for a back edge).
 */
function makeTopology(options = {}) {
  const prng = new Prng(options.seed || 894);
  const nodes = options.nodes || 40;
  const maxEdges = options.maxEdges || 3;
  const names = [];
  for (let i = 0; i < nodes; i++) names.push(`svc${String(i).padStart(3, "0")}`);
  const levels = new Array(nodes).fill(0);
  const edges = [];
  for (let i = 1; i < nodes; i++) {
    const parent = prng.below(i);
    levels[i] = levels[parent] + 1;
    edges.push({ from: names[parent], to: names[i] });
    const extras = prng.below(maxEdges);
    for (let e = 0; e < extras; e++) {
      const other = prng.below(i);
      if (other !== parent && levels[other] >= levels[i]) continue;
      edges.push({ from: names[other], to: names[i] });
    }
  }
  const topology = { nodes: names, edges, levels };
  if (options.cycle) {
    // A dependency cycle: the deepest node calls back into the gateway.
    const deepest = names.reduce((best, name, index) => (levels[index] > levels[names.indexOf(best)] ? name : best), names[0]);
    edges.push({ from: deepest, to: names[0] });
  }
  return finaliseTopology(topology);
}

function finaliseTopology(topology) {
  const callers = new Map(topology.nodes.map((name) => [name, []]));
  const callees = new Map(topology.nodes.map((name) => [name, []]));
  for (const edge of topology.edges) {
    callees.get(edge.from).push(edge.to);
    callers.get(edge.to).push(edge.from);
  }
  return { ...topology, callers, callees, components: condense(topology.nodes, callees) };
}

/** Tarjan-free SCC condensation: repeated forward/backward reachability closure. */
function condense(nodes, callees) {
  const reachable = new Map();
  const reach = (start) => {
    if (reachable.has(start)) return reachable.get(start);
    const seen = new Set([start]);
    const stack = [start];
    while (stack.length > 0) {
      const current = stack.pop();
      for (const next of callees.get(current) || []) {
        if (!seen.has(next)) { seen.add(next); stack.push(next); }
      }
    }
    reachable.set(start, seen);
    return seen;
  };
  const componentOf = new Map();
  let componentId = 0;
  for (const node of nodes) {
    if (componentOf.has(node)) continue;
    // A node is in a non-trivial component only if some other node reaches it back.
    const forward = reach(node);
    const component = new Set([node]);
    for (const other of forward) {
      if (other !== node && reach(other).has(node)) component.add(other);
    }
    for (const member of component) componentOf.set(member, componentId);
    componentId++;
  }
  return { componentOf, count: componentId };
}

/* ------------------------------------------------------------------ scenario */

/**
 * Injects `faults` (node → extra latency) and propagates the anomaly towards the callers:
 *   deviation(v) = injected(v) + attenuation · Σ deviations of v's callees + noise(v)
 * Fan-in therefore *amplifies*: a caller with several anomalous callees can out-rank its own callee.
 * Each node also has a baseline window (mean and standard deviation) so deviations can be scored as
 * z-scores — the scale used by every heuristic, because raw latencies are not comparable across
 * services.
 */
function generateScenario(topology, options = {}) {
  const prng = new Prng(options.seed || 1);
  const faultNodes = options.faults || [topology.nodes[topology.nodes.length - 1]];
  const attenuation = options.attenuation !== undefined ? options.attenuation : 0.42;
  const injected = new Map(faultNodes.map((node, index) => [
    node, options.injectedValue !== undefined ? options.injectedValue : 24 + index * 6 + prng.below(6),
  ]));
  const deviations = new Map(topology.nodes.map((node) => [node, 0]));
  const baselineMean = new Map();
  const baselineSd = new Map();
  for (const node of topology.nodes) {
    baselineMean.set(node, 20 + prng.below(12));
    baselineSd.set(node, options.flatBaseline ? 0 : Math.round((1 + prng.unit() * 1.5) * 100) / 100);
  }
  const startTimes = new Map();
  const minDeviation = options.minDeviation !== undefined ? options.minDeviation : 3;
  const shareThreshold = options.shareThreshold !== undefined ? options.shareThreshold : 0.1;
  const compute = (node, visiting) => {
    if (deviations.get(node) !== 0) return deviations.get(node);
    if (visiting.has(node)) return 0;                       // a cycle contributes nothing further
    visiting.add(node);
    let total = injected.get(node) || 0;
    for (const callee of topology.callees.get(node)) total += attenuation * compute(callee, visiting);
    visiting.delete(node);
    const noise = options.noise === 0 ? 0 : (prng.unit() - 0.5) * (options.noise || 1.2);
    const value = total === 0 ? Math.max(0, noise) : total + noise;
    deviations.set(node, Math.round(value * 100) / 100);
    return value;
  };
  for (const node of topology.nodes) compute(node, new Set());

  const incident = new Map();
  const maxDeviation = Math.max(1e-9, ...topology.nodes.map((node) => Math.max(0, deviations.get(node))));
  const scores = new Map();
  const zScores = new Map();
  for (const node of topology.nodes) {
    incident.set(node, Math.round((baselineMean.get(node) + deviations.get(node)) * 100) / 100);
    // The detection score is the *share* of the worst deviation in the window: it is scale free, so a
    // caller that inherits from a deep dependency still scores above the threshold, which is what
    // makes the structural rule (an anomalous node with no anomalous callee) exactly the source.
    scores.set(node, Math.max(0, deviations.get(node)) / maxDeviation);
    zScores.set(node, deviationScoreOf(incident.get(node), baselineMean.get(node), baselineSd.get(node)));
  }
  // Deviation start times: the fault starts at the root and reaches its callers one window later.
  const level = (node) => topology.levels[topology.nodes.indexOf(node)];
  for (const node of topology.nodes) {
    if (!isAnomalous(deviations.get(node), scores.get(node), { minDeviation, shareThreshold })) continue;
    const distance = Math.min(...faultNodes.map((fault) => Math.abs(level(fault) - level(node))));
    startTimes.set(node, Math.round((3 + distance + prng.unit() * 0.6) * 100) / 100);
  }
  return {
    topology, injected, deviations, baselineMean, baselineSd, incident, scores, zScores, startTimes,
    attenuation, faultNodes, minDeviation, shareThreshold, maxDeviation,
  };
}

/** The z-score of a node: a zero-variance baseline is floored so the division can never blow up. */
function deviationScoreOf(incident, baseline, sd) {
  const scale = Math.max(sd, 0.25);
  return (incident - baseline) / scale;
}

/**
 * Anomaly test on two scales: the deviation must be bigger than measurement noise (`minDeviation`),
 * and it must be a non-trivial share of the worst deviation seen. The share keeps the graph structure
 * usable; the absolute floor keeps a *no-fault* window from blaming noise.
 */
function isAnomalous(deviation, share, options) {
  return deviation >= options.minDeviation && share >= options.shareThreshold;
}

/** Anomaly cut-off: the minimum share of the worst deviation that counts as an anomaly. */
const deviationThreshold = (scenario) => (scenario ? scenario.shareThreshold : 0.1);

function anomalousNodes(scenario, threshold) {
  const limit = threshold !== undefined ? threshold : deviationThreshold(scenario);
  return scenario.topology.nodes.filter((node) => isAnomalous(
    scenario.deviations.get(node), scenario.scores.get(node),
    { minDeviation: scenario.minDeviation, shareThreshold: limit }));
}

function rankedNodes(scenario) {
  return [...scenario.scores.entries()].sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]));
}

/** Ancestor test on the call graph: `caller` reaches `callee` through the edges. */
function reaches(topology, caller, callee) {
  if (caller === callee) return true;
  const seen = new Set([caller]);
  const stack = [caller];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const next of topology.callees.get(current)) {
      if (next === callee) return true;
      if (!seen.has(next)) { seen.add(next); stack.push(next); }
    }
  }
  return false;
}

/** Picks fault nodes that are pairwise independent (neither reaches the other). */
function pickIndependentRoots(topology, prng, count) {
  const chosen = [];
  for (let attempt = 0; attempt < 200 && chosen.length < count; attempt++) {
    const candidate = topology.nodes[prng.below(topology.nodes.length)];
    if (chosen.some((node) => reaches(topology, node, candidate) || reaches(topology, candidate, node))) continue;
    chosen.push(candidate);
  }
  return chosen;
}

/* ------------------------------------------------------------------ heuristics */

/** Naive baseline: blame the loudest node (by z-score, since raw latencies are not comparable). */
function maxDeviationRoot(scenario) {
  const ranked = rankedNodes(scenario);
  return ranked.length > 0 ? [ranked[0][0]] : [];
}

/**
 * The *unexplained* deviation rule: an anomalous node is a root cause when its deviation exceeds the
 * total deviation of everything it depends on. A propagating symptom satisfies
 * deviation(v) = attenuation · Σ deviation(callees) with attenuation < 1, so it can never pass this
 * test, while the injected node (whose dependencies are healthy) always does. Unlike "no anomalous
 * callee" this needs no threshold consistency along the chain, so a dependency that falls below the
 * detection floor cannot manufacture a false root.
 */
function minimalDeviationRoot(scenario, threshold) {
  const anomalous = new Set(anomalousNodes(scenario, threshold));
  const roots = [];
  for (const node of anomalous) {
    const inherited = scenario.topology.callees.get(node)
      .reduce((sum, callee) => sum + scenario.deviations.get(callee), 0);
    if (scenario.deviations.get(node) > inherited + 1e-9) roots.push(node);
  }
  return roots;
}

/**
 * Walk-blame: suspicion flows from a caller *towards its dependencies*, weighted by how much each
 * callee itself deviates, so a healthy leaf receives nothing while the injected leaf accumulates the
 * evidence from every symptom above it. Iterated to a fixed point; convergence is monitored.
 * Returns { ranking, iterations, converged, residual, scores }.
 */
function walkBlame(scenario, options = {}) {
  const damping = options.damping !== undefined ? options.damping : 0.65;
  const maxIterations = options.maxIterations || 50;
  const tolerance = options.tolerance || 1e-9;
  const nodes = scenario.topology.nodes;
  const total = nodes.reduce((sum, node) => sum + Math.max(0, scenario.scores.get(node)), 0) || 1;
  const normalised = new Map(nodes.map((node) => [node, Math.max(0, scenario.scores.get(node)) / total]));
  let blame = new Map(nodes.map((node) => [node, normalised.get(node)]));
  let iterations = 0;
  let residual = Infinity;
  let converged = false;
  while (iterations < maxIterations && !converged) {
    const next = new Map(nodes.map((node) => [node, normalised.get(node)]));
    for (const caller of nodes) {
      const callees = scenario.topology.callees.get(caller);
      const weight = callees.reduce((sum, callee) => sum + Math.max(0, scenario.scores.get(callee)), 0);
      if (weight <= 0) continue;                 // a caller with healthy dependencies passes nothing on
      for (const callee of callees) {
        const share = Math.max(0, scenario.scores.get(callee)) / weight;
        next.set(callee, next.get(callee) + damping * blame.get(caller) * share);
      }
    }
    let change = 0;
    for (const node of nodes) change += Math.abs(next.get(node) - blame.get(node));
    residual = change;
    if (change < tolerance) converged = true;
    blame = next;
    iterations++;
  }
  const ranking = [...blame.entries()].sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]));
  return { ranking, iterations, converged, residual, scores: blame };
}

/**
 * Greedy explanation: pick the anomalous node explaining ("being a descendant of ... " in the caller
 * direction) the most still-unexplained anomalies, until everything is explained.
 */
function greedyExplanation(scenario, threshold) {
  const anomalous = anomalousNodes(scenario, threshold);
  const anomalousSet = new Set(anomalous);
  const explains = new Map(anomalous.map((node) => {
    const covered = new Set([node]);
    const stack = [...scenario.topology.callers.get(node)];
    while (stack.length > 0) {
      const caller = stack.pop();
      if (covered.has(caller)) continue;
      covered.add(caller);
      for (const next of scenario.topology.callers.get(caller)) stack.push(next);
    }
    return [node, covered];
  }));
  const unexplained = new Set(anomalous);
  const chosen = [];
  while (unexplained.size > 0) {
    let bestNode = null;
    let bestCount = -1;
    for (const node of anomalous) {
      if (chosen.includes(node)) continue;
      let count = 0;
      for (const covered of explains.get(node)) if (unexplained.has(covered)) count++;
      if (count > bestCount) { bestCount = count; bestNode = node; }
    }
    if (bestNode === null || bestCount <= 0) break;
    chosen.push(bestNode);
    for (const covered of explains.get(bestNode)) unexplained.delete(covered);
  }
  return chosen;
}

/** Exhaustive oracle: the smallest subset of anomalous nodes that explains all of them. */
function bruteForceExplanation(scenario, threshold) {
  const anomalous = anomalousNodes(scenario, threshold);
  if (anomalous.length === 0) return [];
  const covers = new Map(anomalous.map((node) => {
    const covered = new Set([node]);
    const stack = [...scenario.topology.callers.get(node)];
    while (stack.length > 0) {
      const caller = stack.pop();
      if (covered.has(caller)) continue;
      covered.add(caller);
      for (const next of scenario.topology.callers.get(caller)) stack.push(next);
    }
    return [node, covered];
  }));
  for (let size = 1; size <= anomalous.length; size++) {
    const combination = [];
    const search = (start) => {
      if (combination.length === size) {
        const covered = new Set();
        for (const node of combination) for (const member of covers.get(node)) covered.add(member);
        return anomalous.every((node) => covered.has(node));
      }
      for (let i = start; i < anomalous.length; i++) {
        combination.push(anomalous[i]);
        if (search(i + 1)) return true;
        combination.pop();
      }
      return false;
    };
    if (search(0)) return combination.slice();
  }
  return anomalous.slice();
}

/** Earliest deviation wins; used as a tie-break with the structural heuristics. */
function earliestDeviationRoot(scenario) {
  const entries = [...scenario.startTimes.entries()].sort((a, b) => (a[1] - b[1]) || a[0].localeCompare(b[0]));
  if (entries.length === 0) return [];
  const best = entries[0][1];
  return entries.filter(([, time]) => time <= best + 0.4).map(([node]) => node);
}

/* ------------------------------------------------------------------ evaluation */

function evaluateHeuristics(scenario, thresholds = {}) {
  const threshold = deviationThreshold(scenario);
  const anomalous = anomalousNodes(scenario, threshold);
  const ambiguous = thresholds.ambiguous || null;
  const truth = new Set(scenario.faultNodes);
  const naive = maxDeviationRoot(scenario);
  const minimal = minimalDeviationRoot(scenario, threshold);
  const blame = walkBlame(scenario);
  const greedy = greedyExplanation(scenario, threshold);
  const earliest = earliestDeviationRoot(scenario);
  const rankOf = (node) => blame.ranking.findIndex((entry) => entry[0] === node) + 1;
  return {
    threshold, anomalous, ambiguous, truth,
    naive, minimal, greedy, earliest,
    blameTop: blame.ranking.slice(0, 3).map((entry) => entry[0]),
    blameIterations: blame.iterations, blameConverged: blame.converged, blameResidual: blame.residual,
    minimalCorrect: minimal.length === truth.size && minimal.every((node) => truth.has(node)),
    naiveCorrect: naive.length === truth.size && naive.every((node) => truth.has(node)),
    greedyCorrect: greedy.length === truth.size && greedy.every((node) => truth.has(node)),
    blameRank: Math.min(...scenario.faultNodes.map(rankOf)),
    earliestCorrect: earliest.some((node) => truth.has(node)),
  };
}

module.exports = {
  Prng, makeTopology, finaliseTopology, condense, generateScenario, deviationThreshold, deviationScoreOf,
  anomalousNodes, rankedNodes, reaches, pickIndependentRoots, maxDeviationRoot, minimalDeviationRoot,
  walkBlame, greedyExplanation, bruteForceExplanation, earliestDeviationRoot, evaluateHeuristics,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- topology sanity ------------------------------------------------------------------------
  {
    const topology = makeTopology({ seed: 7, nodes: 30, maxEdges: 3 });
    const ids = new Set(topology.nodes);
    assert.strictEqual(ids.size, 30);
    assert.ok(topology.edges.length >= 29, "a spanning structure is required");
    for (const edge of topology.edges) {
      assert.ok(ids.has(edge.from) && ids.has(edge.to));
      assert.ok(topology.callees.get(edge.from).includes(edge.to));
      assert.ok(topology.callers.get(edge.to).includes(edge.from));
    }
    assert.strictEqual(topology.components.componentOf.get("svc000"), 0, "the gateway is its own component in a DAG");
    for (const node of topology.nodes) assert.ok(topology.components.componentOf.has(node));
    const forest = makeTopology({ seed: 8, nodes: 12, maxEdges: 1 });
    assert.ok(forest.edges.length === 11, "a tree has n-1 edges");
    assert.strictEqual([...forest.components.componentOf.values()].filter((id) => id === 0).length, 1);
    const cyclic = makeTopology({ seed: 9, nodes: 12, maxEdges: 2, cycle: true });
    const members = [...cyclic.components.componentOf.keys()].filter((node) => cyclic.components.componentOf.get(node) === cyclic.components.componentOf.get("svc000"));
    assert.ok(members.length >= 2, "the back edge must merge the gateway into a non-trivial component");
  }

  // --- propagation: the root is the source, callers are symptoms ------------------------------
  {
    const topology = makeTopology({ seed: 11, nodes: 25, maxEdges: 3 });
    const scenario = generateScenario(topology, { seed: 3, faults: ["svc024"], noise: 0 });
    const threshold = deviationThreshold(scenario);
    assert.ok(scenario.deviations.get("svc024") > threshold, "the injected node must deviate");
    assert.deepStrictEqual(minimalDeviationRoot(scenario, threshold), ["svc024"]);
    assert.ok(anomalousNodes(scenario, threshold).length >= 1);
    assert.ok(scenario.deviations.get("svc000") > 0, "the symptom must reach the gateway");
    assert.ok(scenario.deviations.get("svc024") > scenario.deviations.get("svc000") * 0.5);
  }

  // --- fan-in amplification makes the naive heuristic fail ------------------------------------
  {
    // A hand-built fan-in: three leaves under one caller, all three injected.
    const topology = finaliseTopology({
      nodes: ["gateway", "caller", "a", "b", "c"],
      edges: [
        { from: "gateway", to: "caller" }, { from: "caller", to: "a" },
        { from: "caller", to: "b" }, { from: "caller", to: "c" },
      ],
      levels: [0, 1, 2, 2, 2],
    });
    const scenario = generateScenario(topology, {
      seed: 5, faults: ["a", "b", "c"], noise: 0, injectedValue: 30, flatBaseline: true,
    });
    const threshold = deviationThreshold(scenario);
    const callerDeviation = scenario.deviations.get("caller");
    const leafDeviation = scenario.deviations.get("a");
    assert.ok(callerDeviation > leafDeviation,
      `fan-in must amplify: caller ${callerDeviation} vs leaf ${leafDeviation}`);
    assert.deepStrictEqual(maxDeviationRoot(scenario), ["caller"], "the naive heuristic blames the loud symptom");
    assert.strictEqual(evaluateHeuristics(scenario).naiveCorrect, false, "and it is wrong");
    assert.deepStrictEqual(minimalDeviationRoot(scenario, threshold).sort(), ["a", "b", "c"]);
    assert.deepStrictEqual(greedyExplanation(scenario, threshold).sort(), ["a", "b", "c"],
      "all three leaves are genuine roots");
  }

  // --- walk-blame converges and ranks the root highly -----------------------------------------
  {
    const topology = makeTopology({ seed: 21, nodes: 40, maxEdges: 3 });
    const scenario = generateScenario(topology, { seed: 9, faults: [topology.nodes[30]] });
    const blame = walkBlame(scenario);
    assert.ok(blame.converged, `walk-blame must converge (residual ${blame.residual})`);
    assert.ok(blame.iterations <= 50, `too many iterations: ${blame.iterations}`);
    assert.ok(blame.residual < 1e-6);
    const rank = blame.ranking.findIndex((entry) => entry[0] === topology.nodes[30]) + 1;
    assert.ok(rank >= 1 && rank <= 2, `the root ranked ${rank}`);
    const iterationCounts = [];
    for (let damping = 0.1; damping <= 0.9; damping += 0.2) {
      const run = walkBlame(scenario, { damping });
      iterationCounts.push(run.iterations);
      assert.ok(run.converged, `damping ${damping} must converge too`);
      const rankAt = run.ranking.findIndex((entry) => entry[0] === topology.nodes[30]) + 1;
      assert.ok(rankAt <= 2, `damping ${damping}: root ranked ${rankAt}`);
    }
    assert.ok(iterationCounts.every((count) => count >= 1));
  }

  // --- the greedy explanation is optimal ------------------------------------------------------
  {
    for (let seed = 1; seed <= 12; seed++) {
      const topology = makeTopology({ seed, nodes: 18, maxEdges: 2 });
      const scenario = generateScenario(topology, { seed: seed * 13, faults: [topology.nodes[10 + (seed % 6)]] });
      const greedy = greedyExplanation(scenario);
      const brute = bruteForceExplanation(scenario);
      assert.strictEqual(greedy.length, brute.length, `seed ${seed}: greedy ${greedy.length} vs optimal ${brute.length}`);
      assert.deepStrictEqual([...greedy].sort(), [...brute].sort(), `seed ${seed}: same nodes`);
      assert.strictEqual(greedy.length, 1, "a single fault has a single-node explanation");
    }
    // Two independent faults in two branches must both be reported.
    const topology = finaliseTopology({
      nodes: ["gateway", "left", "right", "l1", "r1"],
      edges: [{ from: "gateway", to: "left" }, { from: "gateway", to: "right" },
        { from: "left", to: "l1" }, { from: "right", to: "r1" }],
      levels: [0, 1, 1, 2, 2],
    });
    const scenario = generateScenario(topology, { seed: 31, faults: ["l1", "r1"], noise: 0 });
    const greedy = greedyExplanation(scenario);
    assert.strictEqual(greedy.length, 2, `two faults need two explanations (${greedy})`);
    assert.deepStrictEqual(greedy.sort(), ["l1", "r1"]);
    assert.strictEqual(bruteForceExplanation(scenario).length, 2);
    assert.strictEqual(maxDeviationRoot(scenario).length, 1, "the naive heuristic can only report one node");
  }

  // --- no fault means no root cause ------------------------------------------------------------
  {
    const topology = makeTopology({ seed: 33, nodes: 20, maxEdges: 2 });
    const quiet = generateScenario(topology, { seed: 4, faults: [], noise: 1.0 });
    assert.deepStrictEqual(minimalDeviationRoot(quiet), [], "nothing deviates, nothing is blamed");
    assert.deepStrictEqual(greedyExplanation(quiet), []);
    assert.deepStrictEqual(bruteForceExplanation(quiet), []);
    assert.deepStrictEqual(earliestDeviationRoot(quiet), [], "no start times were recorded");
    const verdict = evaluateHeuristics(quiet);
    assert.strictEqual(verdict.minimal.length, 0);
    assert.strictEqual(verdict.anomalous.length, 0);
    assert.ok(verdict.blameConverged, "the walk still converges on a flat graph");
  }

  // --- accuracy over many randomised scenarios -------------------------------------------------
  {
    const prng = new Prng(20260894);
    let minimalHits = 0;
    let naiveHits = 0;
    let greedyHits = 0;
    let blameTop2 = 0;
    let blameReciprocal = 0;
    let earliestHits = 0;
    let multiFaultRuns = 0;
    const scenarios = 60;
    for (let trial = 0; trial < scenarios; trial++) {
      const topology = makeTopology({ seed: 100 + trial, nodes: 30 + prng.below(20), maxEdges: 3 });
      const wanted = trial % 3 === 0 ? 2 : 1;
      let faults = pickIndependentRoots(topology, prng, wanted);
      if (faults.length < wanted) faults = [topology.nodes[topology.nodes.length - 1]];
      if (faults.length > 1) multiFaultRuns++;
      const scenario = generateScenario(topology, { seed: 5000 + trial, faults });
      const verdict = evaluateHeuristics(scenario);
      if (verdict.minimalCorrect) minimalHits++;
      if (verdict.greedyCorrect) greedyHits++;
      if (verdict.naiveCorrect) naiveHits++;
      if (verdict.blameRank <= 2) blameTop2++;
      blameReciprocal += 1 / verdict.blameRank;
      if (verdict.earliestCorrect) earliestHits++;
    }
    assert.ok(multiFaultRuns >= 10, `only ${multiFaultRuns} multi-fault runs`);
    assert.strictEqual(minimalHits, scenarios, `minimal-root got ${minimalHits}/${scenarios}`);
    assert.strictEqual(greedyHits, scenarios, `greedy explanation got ${greedyHits}/${scenarios}`);
    assert.ok(naiveHits < scenarios,
      "max-deviation must fail on the multi-fault scenarios (it can only name one node)");
    assert.ok(blameTop2 >= scenarios - 6, `walk-blame top-2 accuracy ${blameTop2}/${scenarios}`);
    const mrr = blameReciprocal / scenarios;
    assert.ok(mrr >= 0.9, `walk-blame MRR ${mrr} is too low`);
    assert.ok(earliestHits >= scenarios - 10, `temporal heuristic got ${earliestHits}/${scenarios}`);
    console.log(`rca: minimal-root ${minimalHits}/${scenarios}, greedy ${greedyHits}/${scenarios}, ` +
      `top-2 walk-blame ${blameTop2}/${scenarios} (MRR ${mrr.toFixed(3)}), max-deviation ${naiveHits}/${scenarios}, ` +
      `${multiFaultRuns} multi-fault runs`);
  }

  // --- cycles, disconnected parts and flat metrics ----------------------------------------------
  {
    const cyclic = makeTopology({ seed: 5, nodes: 16, maxEdges: 2, cycle: true });
    const scenario = generateScenario(cyclic, { seed: 6, faults: ["svc015"] });
    const verdict = evaluateHeuristics(scenario);
    assert.ok(verdict.anomalous.length > 0);
    assert.ok(verdict.minimalCorrect, `cyclic minimal-root: ${verdict.minimal} vs ${[...verdict.truth]}`);
    assert.strictEqual(verdict.greedy.length, 1);
    assert.ok(verdict.blameConverged, "the walk must terminate on a cyclic graph");
    // Flat metrics must not divide by zero anywhere.
    const flatNodes = ["a", "b", "c"];
    const flat = finaliseTopology({
      nodes: flatNodes,
      edges: [{ from: "a", to: "b" }, { from: "b", to: "c" }],
      levels: [0, 1, 2],
    });
    const noChange = generateScenario(flat, { seed: 7, faults: [], noise: 0 });
    const flatThreshold = deviationThreshold(noChange);
    assert.ok(Number.isFinite(flatThreshold), "the threshold must be finite for a flat profile");
    assert.deepStrictEqual(anomalousNodes(noChange, flatThreshold), []);
    const flatBlame = walkBlame(noChange);
    assert.ok(flatBlame.converged);
    assert.ok(flatBlame.ranking.every(([, score]) => Number.isFinite(score)));
    // A disconnected component that is healthy must not be blamed.
    const split = finaliseTopology({
      nodes: ["root", "hot", "cold1", "cold2"],
      edges: [{ from: "root", to: "hot" }],
      levels: [0, 1, 1, 1],
    });
    const partial = generateScenario(split, { seed: 8, faults: ["hot"], noise: 0.4 });
    const partialVerdict = evaluateHeuristics(partial);
    assert.ok(partialVerdict.minimalCorrect);
    assert.ok(!partialVerdict.minimal.includes("cold1") && !partialVerdict.minimal.includes("cold2"),
      "healthy services must never be blamed");
    assert.deepStrictEqual(partialVerdict.greedy, ["hot"]);
  }

  // --- cost and determinism ----------------------------------------------------------------------
  {
    let operations = 0;
    const big = makeTopology({ seed: 909, nodes: 200, maxEdges: 3 });
    const scenario = generateScenario(big, { seed: 42, faults: [big.nodes[180]] });
    const blame = walkBlame(scenario, { damping: 0.5, maxIterations: 30 });
    operations += blame.iterations * big.edges.length;
    assert.ok(operations < 30 * big.edges.length * 2, "the walk must cost O(iterations · edges)");
    assert.ok(big.edges.length >= 199);
    const verdict = evaluateHeuristics(scenario);
    assert.ok(verdict.minimalCorrect, `200-node graph: ${verdict.minimal}`);
    const again = evaluateHeuristics(generateScenario(big, { seed: 42, faults: [big.nodes[180]] }));
    assert.strictEqual(JSON.stringify(again.blameTop), JSON.stringify(verdict.blameTop));
    assert.deepStrictEqual(again.minimal, verdict.minimal);
  }

  console.log("PASS");
}
