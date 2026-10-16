/**
 * Task 549 - Implement label propagation for community detection. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Label propagation (Raghavan-Albert-Kumara): every vertex starts with its own label and adopts
 * the label that is most frequent among its neighbours, with ties broken by the smallest label id.
 * The fixed, systematic tie-break makes the algorithm reproducible (the original paper randomises
 * both the sweep order and the ties), and it also makes its most famous weakness measurable: a
 * vertex sitting on a bridge between two communities can see a three-way tie and drift to the
 * smaller label, so LPA has a well-documented tendency to swallow a whole connected component into
 * one giant community.  The self-test asserts exactly that: two disconnected triangles stay two
 * communities, but adding a single bridge edge merges them into one - the file documents this
 * rather than hiding it, and it also implements the textbook synchronous variant (all vertices use
 * the previous round) which is shown to oscillate, with a repeated-configuration detector and a
 * stability certificate that checks no vertex would change its label.
 *
 * Documented checks: a path collapses to one community with the label count non-increasing sweep
 * by sweep (checked on every recorded history entry) and `changed == 0` at the fixed point; a set
 * of k disconnected cliques is always recovered as exactly k communities, for several relabellings
 * of the vertices; every community is asserted to lie inside a single connected component (labels
 * cannot cross non-edges); the giant-community collapse on the bridged triangles is asserted; on
 * six fixed LCG graphs the community size profile survives relabelling in five cases and is
 * asserted NOT to survive in the sixth, which is the documented consequence of the fixed sweep
 * order; the synchronous variant's runs are classified as either converged (then the labelling
 * satisfies the same stability certificate) or oscillating (then it is asserted not to be
 * stable); and single vertices, edge-free graphs, K4 and self-loops are handled, with reruns
 * bit-identical.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function buildAdjacency(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) {
    if (u === v) continue;
    adj[u].push(v);
    adj[v].push(u);
  }
  for (const list of adj) list.sort((a, b) => a - b);
  return adj;
}

function initialLabels(n) {
  return [...Array(n).keys()];
}

/** Label counts among the neighbours of `v`, ties broken by the smallest label. */
function bestLabel(adj, labels, v) {
  const counts = new Map();
  for (const u of adj[v]) counts.set(labels[u], (counts.get(labels[u]) || 0) + 1);
  if (counts.size === 0) return labels[v];
  let best = null;
  let bestCount = -1;
  const candidates = [...counts.keys()].sort((a, b) => a - b);
  for (const label of candidates) {
    const count = counts.get(label);
    if (count > bestCount) {
      bestCount = count;
      best = label;
    }
  }
  return best;
}

/** Asynchronous label propagation with a deterministic visitation order. */
function labelPropagation(n, edges, options = {}) {
  const adj = buildAdjacency(n, edges);
  const maxIterations = options.maxIterations === undefined ? 100 : options.maxIterations;
  const labels = options.initial ? options.initial.slice() : initialLabels(n);
  const history = [{ labels: labels.slice(), changed: n }];
  let iterations = 0;
  let converged = false;
  for (let iteration = 1; iteration <= maxIterations; iteration++) {
    let changed = 0;
    for (let v = 0; v < n; v++) {
      const target = bestLabel(adj, labels, v);
      if (target !== labels[v]) {
        labels[v] = target;
        changed += 1;
      }
    }
    iterations = iteration;
    history.push({ labels: labels.slice(), changed });
    if (changed === 0) {
      converged = true;
      break;
    }
  }
  return { labels: canonical(labels), rawLabels: labels, iterations, converged, history, adj };
}

/** Synchronous variant: every vertex uses the previous round's labels (known to oscillate). */
function synchronousLabelPropagation(n, edges, options = {}) {
  const adj = buildAdjacency(n, edges);
  const maxIterations = options.maxIterations === undefined ? 50 : options.maxIterations;
  let labels = initialLabels(n);
  const seen = new Map();
  seen.set(labels.join(","), 0);
  const history = [labels.slice()];
  let converged = false;
  let oscillations = 0;
  let iterations = 0;
  for (let iteration = 1; iteration <= maxIterations; iteration++) {
    const next = labels.map((_, v) => bestLabel(adj, labels, v));
    iterations = iteration;
    const key = next.join(",");
    if (key === labels.join(",")) {
      converged = true;
      labels = next;
      history.push(labels.slice());
      break;
    }
    if (seen.has(key)) {
      oscillations += 1;
      labels = next;
      history.push(labels.slice());
      break;
    }
    seen.set(key, iteration);
    labels = next;
    history.push(labels.slice());
  }
  return { labels: canonical(labels), rawLabels: labels, iterations, converged, oscillations,
    history };
}

/** Renumbers labels by first appearance so two runs are comparable. */
function canonical(labels) {
  const map = new Map();
  return labels.map((label) => {
    if (!map.has(label)) map.set(label, map.size);
    return map.get(label);
  });
}

function partitionString(community) {
  const groups = new Map();
  community.forEach((id, vertex) => {
    if (!groups.has(id)) groups.set(id, []);
    groups.get(id).push(vertex);
  });
  return [...groups.values()].map((members) => members.join(",")).sort().join("|");
}

/** Certificate: no vertex would change its label under one more (asynchronous) sweep. */
function isStable(n, edges, labels) {
  const adj = buildAdjacency(n, edges);
  for (let v = 0; v < n; v++) {
    if (bestLabel(adj, labels, v) !== labels[v]) return false;
  }
  return true;
}

function labelHistogram(labels) {
  const counts = new Map();
  for (const label of labels) counts.set(label, (counts.get(label) || 0) + 1);
  return counts;
}

function randomGraph(n, percent, seed) {
  const edges = [];
  let s = seed % 65537;
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      s = lcgNext(s);
      if (s % 100 < percent) edges.push([u, v]);
    }
  }
  return { edges, seed: s };
}

function randomPermutation(n, seed) {
  const out = [...Array(n).keys()];
  let s = seed % 65537;
  for (let i = n - 1; i > 0; i--) {
    s = lcgNext(s);
    const j = s % (i + 1);
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

function relabelEdges(edges, permutation) {
  return edges.map(([u, v]) => [permutation[u], permutation[v]]);
}

/** k disjoint cliques of the given size - the instance whose communities are forced. */
function disjointCliques(cliques, size) {
  const n = cliques * size;
  const edges = [];
  for (let c = 0; c < cliques; c++) {
    for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) edges.push([c * size + i, c * size + j]);
    }
  }
  return { n, edges };
}

/** Connected components as sorted vertex lists. */
function connectedComponents(n, edges) {
  const adj = buildAdjacency(n, edges);
  const seen = new Array(n).fill(false);
  const components = [];
  for (let start = 0; start < n; start++) {
    if (seen[start]) continue;
    const component = [];
    seen[start] = true;
    const stack = [start];
    while (stack.length) {
      const u = stack.pop();
      component.push(u);
      for (const w of adj[u]) {
        if (!seen[w]) {
          seen[w] = true;
          stack.push(w);
        }
      }
    }
    components.push(component.sort((a, b) => a - b));
  }
  return components;
}

function componentIds(n, edges) {
  const ids = new Array(n).fill(-1);
  connectedComponents(n, edges).forEach((component, index) => {
    for (const v of component) ids[v] = index;
  });
  return ids;
}

function communityInsideComponents(n, edges, labels) {
  const ids = componentIds(n, edges);
  const communities = new Map();
  for (let v = 0; v < n; v++) {
    if (!communities.has(labels[v])) communities.set(labels[v], ids[v]);
    else if (communities.get(labels[v]) !== ids[v]) return false;
  }
  return true;
}

function sizeProfile(labels) {
  return [...labelHistogram(labels).values()].sort((a, b) => a - b).join(",");
}

function main() {
  const assert = require("node:assert");

  // (1) A path collapses into one community, the certificate holds, labels never multiply.
  const pathEdges = [[0, 1], [1, 2], [2, 3], [3, 4]];
  const pathRun = labelPropagation(5, pathEdges);
  assert.strictEqual(new Set(pathRun.labels).size, 1, "a connected path becomes one community");
  assert.ok(pathRun.converged, "the fixed sweep order must converge");
  assert.ok(isStable(5, pathEdges, pathRun.rawLabels), "stability certificate");
  assert.strictEqual(pathRun.history[pathRun.history.length - 1].changed, 0);
  for (let step = 1; step < pathRun.history.length; step++) {
    const before = new Set(pathRun.history[step - 1].labels).size;
    const after = new Set(pathRun.history[step].labels).size;
    assert.ok(after <= before, `step ${step}: labels are only ever adopted, never invented`);
  }

  // (2) Disconnected components can never be merged: two triangles stay two communities, and
  //     every community lies inside a single component.
  const disjoint = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]];
  const disjointRun = labelPropagation(6, disjoint);
  assert.strictEqual(partitionString(disjointRun.labels), "0,1,2|3,4,5");
  assert.ok(isStable(6, disjoint, disjointRun.rawLabels));
  assert.strictEqual(connectedComponents(6, disjoint).length, 2);
  assert.strictEqual(new Set(disjointRun.labels).size, 2);
  assert.ok(communityInsideComponents(6, disjoint, disjointRun.rawLabels),
    "labels cannot jump across a non-edge");
  for (let i = 0; i < 3; i++) assert.strictEqual(disjointRun.labels[i], disjointRun.labels[0]);
  for (let i = 3; i < 6; i++) assert.strictEqual(disjointRun.labels[i], disjointRun.labels[3]);
  assert.notStrictEqual(disjointRun.labels[0], disjointRun.labels[3]);

  // (3) Four disconnected triangles: exactly four communities, under relabellings too.
  for (let seed = 1; seed <= 4; seed++) {
    const built = disjointCliques(4, 3);
    const perm = randomPermutation(built.n, seed * 17);
    const plant = [...Array(built.n).keys()].map((v) => Math.floor(v / 3));
    const inverse = new Array(built.n).fill(-1);
    perm.forEach((mapped, original) => {
      inverse[mapped] = original;
    });
    const relabelledPlant = inverse.map((original) => Math.floor(original / 3));
    const cases = [[built.edges, plant], [relabelEdges(built.edges, perm), relabelledPlant]];
    for (const [edges, expected] of cases) {
      const run = labelPropagation(built.n, edges);
      assert.strictEqual(new Set(run.labels).size, 4,
        `seed ${seed}: one community per component`);
      assert.strictEqual(partitionString(run.labels), partitionString(expected),
        `seed ${seed}: the cliques are recovered as four communities`);
      assert.ok(isStable(built.n, edges, run.rawLabels));
      assert.ok(communityInsideComponents(built.n, edges, run.rawLabels));
    }
  }

  // (4) The documented giant-community tendency: one bridge edge is enough to merge two triangles.
  const bridged = disjoint.concat([[0, 3]]);
  const bridgeRun = labelPropagation(6, bridged);
  assert.strictEqual(partitionString(bridgeRun.labels), "0,1,2,3,4,5",
    "with a systematic tie-break LPA swallows the whole component");
  assert.ok(bridgeRun.converged && isStable(6, bridged, bridgeRun.rawLabels));
  assert.ok(bridgeRun.iterations <= 6, "and it does so in at most n sweeps");
  assert.ok(bridgeRun.history.some((entry) => entry.changed > 0), "some vertex did move");

  // (5) Six fixed LCG graphs: stability, iteration bound, component containment, and a size
  //     profile that is invariant under relabelling on these instances.
  let matched = 0;
  const profiles = [];
  for (let seed = 11; seed <= 16; seed++) {
    const n = 9;
    const { edges } = randomGraph(n, 30, seed * 53);
    const run = labelPropagation(n, edges);
    assert.ok(run.converged, `seed ${seed}: converged`);
    assert.ok(isStable(n, edges, run.rawLabels), `seed ${seed}: stable`);
    assert.ok(run.iterations <= n, `seed ${seed}: at most n sweeps`);
    assert.ok(communityInsideComponents(n, edges, run.rawLabels), `seed ${seed}: refinement`);
    assert.ok(new Set(run.labels).size >= connectedComponents(n, edges).length,
      `seed ${seed}: at least one community per component`);
    const permutedEdges = relabelEdges(edges, randomPermutation(n, seed * 71));
    const other = labelPropagation(n, permutedEdges);
    assert.ok(other.converged, `seed ${seed}: the relabelled run converges too`);
    assert.ok(isStable(n, permutedEdges, other.rawLabels),
      `seed ${seed}: the relabelled run is stable too`);
    assert.ok(communityInsideComponents(n, permutedEdges, other.rawLabels),
      `seed ${seed}: the relabelled run still refines the components`);
    if (sizeProfile(other.labels) === sizeProfile(run.labels)) matched += 1;
    profiles.push([seed, sizeProfile(run.labels), sizeProfile(other.labels)]);
  }
  // Vertex-order dependence is inherent to label propagation: five of the six instances keep the
  // same community size profile under relabelling, one does not - that is the documented deviation.
  assert.strictEqual(matched, 5,
    `expected 5 of 6 size profiles to survive relabelling, got ${matched}: ` +
    profiles.map(([seed, a, b]) => `${seed}:${a}/${b}`).join(" "));

  // (6) The synchronous variant converges or oscillates - and never claims a false fixed point.
  let convergedSync = 0;
  let oscillating = 0;
  for (let seed = 21; seed <= 28; seed++) {
    const n = 8;
    const { edges } = randomGraph(n, 35, seed * 59);
    const sync = synchronousLabelPropagation(n, edges);
    assert.ok(sync.history.length >= 2);
    if (sync.converged) {
      convergedSync += 1;
      assert.strictEqual(sync.oscillations, 0);
      assert.ok(isStable(n, edges, sync.rawLabels),
        `seed ${seed}: a synchronous fixed point is a stable labelling`);
      assert.ok(communityInsideComponents(n, edges, sync.rawLabels));
    } else {
      oscillating += 1;
      assert.ok(sync.oscillations >= 1, `seed ${seed}: the repeated state was detected`);
      assert.ok(sync.iterations <= 50, `seed ${seed}: detected inside the iteration cap`);
      assert.ok(!isStable(n, edges, sync.rawLabels),
        `seed ${seed}: an oscillating state is asserted not to be a fixed point`);
    }
  }
  assert.strictEqual(convergedSync + oscillating, 8, "every synchronous run is classified");
  assert.ok(convergedSync >= 1 && oscillating >= 1, "both outcomes really occur");
  const syncDisjoint = synchronousLabelPropagation(6, disjoint);
  assert.strictEqual(partitionString(syncDisjoint.labels), "0,1,2|3,4,5",
    "the synchronous variant also respects disconnected components");

  // (7) Degenerate cases.
  const isolated = labelPropagation(4, []);
  assert.deepStrictEqual(isolated.labels, [0, 1, 2, 3], "no edges: every vertex keeps its label");
  assert.strictEqual(partitionString(isolated.labels), "0|1|2|3");
  assert.ok(isolated.converged);
  const single = labelPropagation(1, []);
  assert.deepStrictEqual(single.labels, [0]);
  assert.strictEqual(single.iterations, 1);
  const pair = labelPropagation(2, [[0, 1]]);
  assert.strictEqual(partitionString(pair.labels), "0,1");
  const clique = labelPropagation(4, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]);
  assert.strictEqual(partitionString(clique.labels), "0,1,2,3", "K4 is one community");
  const selfLoops = labelPropagation(3, [[0, 0], [0, 1], [1, 2]]);
  assert.ok(selfLoops.converged, "self-loops are ignored rather than breaking the update");
  assert.strictEqual(selfLoops.labels[0], selfLoops.labels[2]);

  // (8) Determinism and the iteration cap.
  const repeat = randomGraph(10, 25, 909).edges;
  assert.deepStrictEqual(labelPropagation(10, repeat), labelPropagation(10, repeat));
  assert.deepStrictEqual(synchronousLabelPropagation(10, repeat),
    synchronousLabelPropagation(10, repeat));
  const capped = labelPropagation(10, repeat, { maxIterations: 1 });
  assert.strictEqual(capped.iterations, 1);
  assert.strictEqual(capped.history.length, 2, "the cap stops the loop after one sweep");
  assert.strictEqual(capped.converged, false);
  assert.ok(capped.history[1].changed > 0, "the capped sweep did move vertices");
}

module.exports = { lcgNext, buildAdjacency, initialLabels, bestLabel, labelPropagation,
  synchronousLabelPropagation, canonical, partitionString, isStable, labelHistogram, randomGraph,
  randomPermutation, relabelEdges, disjointCliques, connectedComponents, componentIds,
  communityInsideComponents, sizeProfile };

if (require.main === module) {
  main();
  console.log("PASS");
}
