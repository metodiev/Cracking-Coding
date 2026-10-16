/** Task 972 - Implement personalized learning paths using performance clustering.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * Learners are points in mastery space (one coordinate per curriculum topic). The path engine runs
 * Lloyd's k-means with k-means++ seeding over a fixed-seed PRNG, then turns each cluster centroid
 * into a profile - topics sorted from weakest to strongest - and emits a personalized path: the
 * weakest not-yet-mastered topic first, but only once its prerequisites are already mastered.
 * Simplified: mastery is a synthetic 5-dimensional vector (three latent archetypes plus bounded
 * noise) and the path is a plain prerequisite-respecting ordering instead of a cost-optimized plan.
 * Verified: cluster purity against the latent labels, centroids recomputed as member means,
 * assignments against a brute-force nearest-centroid oracle, monotone sse per iteration and per k,
 * convergence, the k=1 and duplicate-point boundaries, path legality and determinism.
 */
"use strict";
const assert = require("node:assert");

const TOPICS = ["arrays", "strings", "hashing", "graphs", "dp"];
const PREREQS = { arrays: [], strings: [], hashing: ["arrays"], graphs: ["hashing"], dp: ["hashing"] };
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const dist2 = (a, b) => a.reduce((sum, x, i) => sum + (x - b[i]) ** 2, 0);

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, unit: () => Number(next() >> 11n) / 2 ** 53, int: (n) => Number(next() % BigInt(n)) };
}

/** Three latent archetypes: array-first, graph-first and dynamic-programming-first learners. */
const ARCHETYPES = [
  [0.9, 0.85, 0.35, 0.2, 0.15],
  [0.3, 0.35, 0.9, 0.85, 0.35],
  [0.5, 0.45, 0.55, 0.6, 0.9],
];
function makeLearners(seed, count = 36) {
  const rng = splitmix64(seed);
  return Array.from({ length: count }, (_, i) => {
    const archetype = i % ARCHETYPES.length;
    return { id: `L${String(i).padStart(2, "0")}`, archetype,
      vector: ARCHETYPES[archetype].map((value) => clamp01(value + (rng.unit() - 0.5) * 0.12)) };
  });
}

const mean = (vectors) => vectors[0].map((_, i) => vectors.reduce((sum, v) => sum + v[i], 0) / vectors.length);
const nearest = (vector, centroids) => centroids
  .map((centroid) => dist2(vector, centroid))
  .reduce((best, d, i) => (d < best.d ? { d, i } : best), { d: Infinity, i: -1 }).i;

/** k-means++ seeding: the first centre is random, later ones favour distant points. */
function seedCentroids(vectors, k, rng) {
  const centroids = [vectors[rng.int(vectors.length)]];
  while (centroids.length < k) {
    const weights = vectors.map((v) => Math.min(...centroids.map((c) => dist2(v, c))));
    const total = weights.reduce((a, b) => a + b, 0);
    if (total === 0) return centroids.concat(vectors.slice(0, k - centroids.length).map((v) => [...v]));
    let target = rng.unit() * total;
    let index = 0;
    while (index < weights.length - 1 && (target -= weights[index]) > 0) index += 1;
    centroids.push([...vectors[index]]);
  }
  return centroids;
}

/** Lloyd's algorithm; every iteration must not increase the sum of squared errors. */
function kmeans(vectors, k, { seed = 972, maxIterations = 40 } = {}) {
  assert(Number.isInteger(k) && k >= 1 && k <= vectors.length, `k=${k} needs between 1 and ${vectors.length} points`);
  const rng = splitmix64(seed);
  let centroids = seedCentroids(vectors, k, rng);
  const sseHistory = [];
  let iterations = 0;
  for (let iter = 0; iter < maxIterations; iter += 1) {
    const assign = vectors.map((v) => nearest(v, centroids));
    sseHistory.push(vectors.reduce((sum, v, i) => sum + dist2(v, centroids[assign[i]]), 0));
    iterations += 1;
    const moved = Array.from({ length: k }, (_, c) => {
      const members = vectors.filter((_, i) => assign[i] === c);
      return members.length ? mean(members) : [...centroids[c]];
    });
    const settled = moved.every((c, i) => c.every((x, j) => Math.abs(x - centroids[i][j]) < 1e-12));
    centroids = moved;
    if (settled) break;
  }
  const assign = vectors.map((v) => nearest(v, centroids));
  const sse = vectors.reduce((sum, v, i) => sum + dist2(v, centroids[assign[i]]), 0);
  return { centroids, assign, sse, sseHistory, iterations };
}

/** Weakest-first topic ranking derived from a cluster centroid. */
const clusterProfile = (centroid) => TOPICS
  .map((topic, i) => ({ topic, mastery: centroid[i] }))
  .sort((a, b) => (a.mastery - b.mastery) || (a.topic < b.topic ? -1 : 1));

/** Personalized path: weakest eligible topic first, prerequisites always satisfied first. */
function learningPath(profile, mastered = []) {
  const done = new Set(mastered);
  const pending = profile.map((entry) => entry.topic).filter((topic) => !done.has(topic));
  const path = [];
  while (path.length < pending.length) {
    const next = pending.find((topic) => !done.has(topic) && PREREQS[topic].every((pre) => done.has(pre)));
    if (next === undefined) break;
    path.push(next);
    done.add(next);
  }
  return path;
}

/** Cluster purity against the latent archetype labels - the external quality oracle. */
function purity(learners, assignment, k) {
  let correct = 0;
  for (let c = 0; c < k; c += 1) {
    const members = learners.filter((_, i) => assignment[i] === c);
    const counts = new Map();
    for (const member of members) counts.set(member.archetype, (counts.get(member.archetype) || 0) + 1);
    correct += Math.max(0, ...counts.values());
  }
  return learners.length ? correct / learners.length : 0;
}

function main() {
  const learners = makeLearners(972, 36);
  const vectors = learners.map((l) => l.vector);

  // 1. Happy path: three well-separated archetypes are recovered, every learner is assigned.
  const run = kmeans(vectors, 3, { seed: 972 });
  assert.strictEqual(run.assign.length, 36, "every learner is assigned");
  assert(new Set(run.assign).size === 3, "all three clusters are populated");
  assert(purity(learners, run.assign, 3) >= 0.95, `clusters match the archetypes: ${purity(learners, run.assign, 3)}`);
  assert(run.sse < 1.0, `within-cluster error stays tiny: ${run.sse.toFixed(4)}`);
  assert(run.iterations <= 40, "the loop terminates");

  // 2. Independent oracle: centroids are the means of their members, assignments the nearest one.
  for (let c = 0; c < 3; c += 1) {
    const members = vectors.filter((_, i) => run.assign[i] === c);
    const expected = members[0].map((_, j) => members.reduce((sum, v) => sum + v[j], 0) / members.length);
    expected.forEach((value, j) => assert(Math.abs(value - run.centroids[c][j]) < 1e-9, `centroid ${c} coord ${j}`));
  }
  const bruteForce = vectors.map((v) => {
    let best = 0;
    for (let c = 1; c < run.centroids.length; c += 1) {
      if (dist2(v, run.centroids[c]) < dist2(v, run.centroids[best])) best = c;
    }
    return best;
  });
  assert.deepStrictEqual(bruteForce, run.assign, "assignments match the brute-force oracle");

  // 3. Invariant: the sse never increases while iterating, and it ends at the reported value.
  for (let i = 1; i < run.sseHistory.length; i += 1) {
    assert(run.sseHistory[i] <= run.sseHistory[i - 1] + 1e-9, `sse rises at iteration ${i}`);
  }
  assert(run.sse <= run.sseHistory.at(-1) + 1e-9, "the final sse is the best seen");
  assert(run.sseHistory.length === run.iterations, "history matches the iteration count");

  // 4. Refinement monotonicity: more clusters never explain less variance (k = 1..5).
  const ladder = [1, 2, 3, 4, 5].map((k) => kmeans(vectors, k, { seed: 972 }).sse);
  for (let i = 1; i < ladder.length; i += 1) {
    assert(ladder[i] <= ladder[i - 1] + 1e-9, `k=${i + 1} must not be worse than k=${i}: ${ladder}`);
  }
  assert(ladder[0] > ladder[4], "clustering explains progressively more variance");

  // 5. Boundary: k=1 collapses to the global mean; k above the population size is rejected.
  const single = kmeans(vectors, 1, { seed: 972 });
  const global = vectors[0].map((_, j) => vectors.reduce((sum, v) => sum + v[j], 0) / vectors.length);
  global.forEach((value, j) => assert(Math.abs(single.centroids[0][j] - value) < 1e-9, `global mean coord ${j}`));
  assert(single.assign.every((a) => a === 0), "one cluster holds everyone");
  assert.strictEqual(mean(vectors).length, TOPICS.length, "the mean keeps the topic dimension");
  assert.throws(() => kmeans(vectors, 37), /between 1 and 36/, "k larger than the cohort is rejected");
  assert.throws(() => kmeans([], 1), /between 1 and 0/, "an empty cohort is rejected");

  // 6. Degenerate input: identical points must not divide by zero, and produce zero error.
  const clones = Array.from({ length: 10 }, () => [0.4, 0.4, 0.4, 0.4, 0.4]);
  const duplicateRun = kmeans(clones, 3, { seed: 7 });
  assert.strictEqual(duplicateRun.sse, 0, "identical points leave no residual error");
  assert(duplicateRun.centroids.every((c) => c.every((x) => Math.abs(x - 0.4) < 1e-12)), "centroids sit on the duplicate");
  assert(duplicateRun.iterations <= 40 && duplicateRun.sseHistory.every((s) => s === 0), "degenerate run is stable");

  // 7. Profiles and personalized paths: weakest first, prerequisites honoured, mastered topics skipped.
  const profile = clusterProfile(run.centroids[0]);
  assert.strictEqual(profile.length, TOPICS.length, "a profile ranks every topic");
  for (let i = 1; i < profile.length; i += 1) {
    assert(profile[i].mastery >= profile[i - 1].mastery, "profiles are sorted weakest first");
  }
  const path = learningPath(profile);
  assert.strictEqual(path.length, TOPICS.length, "a fresh learner gets the whole curriculum");
  for (const topic of path) {
    for (const pre of PREREQS[topic]) assert(path.indexOf(pre) < path.indexOf(topic), `${pre} before ${topic}`);
  }
  assert.strictEqual(new Set(path).size, path.length, "no topic is repeated");
  const advanced = learningPath(profile, ["arrays", "hashing"]);
  assert(!advanced.includes("arrays") && !advanced.includes("hashing"), "mastered topics leave the path");
  assert.deepStrictEqual(advanced, profile.map((e) => e.topic).filter((t) => !["arrays", "hashing"].includes(t)),
    "with prerequisites mastered the path follows the weakness ranking");
  assert.deepStrictEqual(learningPath(profile, TOPICS), [], "a complete learner has an empty path");

  // 8. Determinism: the same seed gives the same clustering, a different seed still converges.
  const again = kmeans(vectors, 3, { seed: 972 });
  assert(JSON.stringify(again.centroids) === JSON.stringify(run.centroids), "centroids are reproducible");
  assert.deepStrictEqual(again.assign, run.assign, "assignments are reproducible");
  const other = kmeans(vectors, 3, { seed: 973 });
  assert(other.sse <= run.sse + 1e-9 && purity(learners, other.assign, 3) >= 0.95, "another seed also separates");

  console.log(`clusters: k=3 purity ${purity(learners, run.assign, 3).toFixed(2)}, sse ${run.sse.toFixed(4)} ` +
    `in ${run.iterations} iterations, sse ladder ${ladder.map((s) => s.toFixed(2)).join(" > ")}`);
  console.log("PASS");
}
module.exports = { TOPICS, PREREQS, splitmix64, makeLearners, kmeans, seedCentroids, clusterProfile,
  learningPath, purity, dist2 };
if (require.main === module) main();
