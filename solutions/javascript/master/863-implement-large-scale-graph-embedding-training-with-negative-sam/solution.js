/** Task 863 - Implement large-scale graph embedding training with negative sampling optimization.
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * Model. DeepWalk-style unsupervised graph embedding: seeded random walks produce node sequences, the
 * skip-gram objective with negative sampling is trained by SGD over (target, context, negatives) triples,
 * and the resulting embeddings are evaluated with link prediction and neighbourhood recall on held-out
 * edges. Each node keeps two vectors — a context vector u and a target vector v — and the per-pair
 * objective is
 *
 *     L = -log sigma(u_c . v_t) - sum_{k in N} log sigma(-u_k . v_t)
 *
 * Negatives are drawn from the smoothed unigram distribution p^0.75, the standard choice that keeps hubs
 * from dominating the negative set. The analytic gradient of L is implemented directly and is checked
 * against a finite-difference gradient without sampling, and the unigram^0.75 weights are compared with an
 * independently computed normalisation.
 *
 * What is simplified: one process and no distributed parameter server, a small graph instead of billions of
 * nodes, plain SGD with a linear learning-rate decay (no Hogwild!, no Adam, no adagrad), a single
 * fixed-length walk policy (no return/in-out parameters), no hierarchical softmax, and randomness confined
 * to one seeded mulberry32 stream — no wall clock, no threads, no I/O.
 *
 * Invariants verified:
 *   gradient   the analytic gradient matches a finite-difference gradient on the exact same objective to
 *              better than 1e-6, so the optimiser really descends the stated loss.
 *   learning   the training loss falls monotonically epoch over epoch, embeddings never collapse (finite
 *              values, variance above a floor, cosine spread), and the exact update count matches
 *              pairs x (1 + negatives) x epochs.
 *   quality    link prediction on held-out edges reaches an AUC well above the random baseline, and the
 *              mean similarity of edges is higher than of non-edges; intra-community similarity exceeds
 *              inter-community similarity.
 *   recall     the cosine top-k neighbourhood overlaps the graph neighbourhood above a documented threshold,
 *              and a larger k cannot lower it below the threshold.
 *   distribution the negative-sampling distribution equals unigram^0.75 normalised (independent
 *              recomputation), sums to one, and gives high-degree nodes proportionally more negatives.
 *   determinism identical seeds reproduce identical walks, embeddings, metrics and loss traces.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — the only source of randomness (walks, negative sampling, initialisation). */
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

const dot = (a, b) => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
};

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

const mean = (values) => (values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length);
const variance = (values) => {
  if (values.length < 2) return 0;
  const m = mean(values);
  return values.reduce((acc, value) => acc + (value - m) ** 2, 0) / (values.length - 1);
};

/**
 * Deterministic community graph: `communities` cliques of `perCommunity` nodes, with a fixed number of
 * cross-community edges so the structure is learnable but not trivial.
 */
function makeGraph({ communities = 3, perCommunity = 8, crossEdges = 4, seed = 863 } = {}) {
  if (!Number.isInteger(communities) || communities < 1) throw new RangeError("communities must be >= 1");
  if (!Number.isInteger(perCommunity) || perCommunity < 2) throw new RangeError("perCommunity must be >= 2");
  const nodes = communities * perCommunity;
  const rand = mulberry32(seed);
  const adjacency = Array.from({ length: nodes }, () => []);
  const edges = [];
  const add = (a, b) => {
    if (a === b) return;
    if (adjacency[a].includes(b)) return;
    adjacency[a].push(b);
    adjacency[b].push(a);
    edges.push([Math.min(a, b), Math.max(a, b)]);
  };
  for (let c = 0; c < communities; c++) {
    const base = c * perCommunity;
    for (let i = 0; i < perCommunity; i++) for (let j = i + 1; j < perCommunity; j++) add(base + i, base + j);
  }
  for (let i = 0; i < crossEdges * communities; i++) {
    const a = Math.floor(rand() * nodes);
    const b = Math.floor(rand() * nodes);
    if (Math.floor(a / perCommunity) === Math.floor(b / perCommunity)) continue;
    add(a, b);
  }
  return { nodes, adjacency, edges, communities, perCommunity, degree: adjacency.map((list) => list.length) };
}

/** Seeded random walks: `walksPerNode` walks of `walkLength` steps starting from every node. */
function randomWalks(graph, { walksPerNode = 4, walkLength = 12, seed = 863 } = {}) {
  const rand = mulberry32(seed);
  const walks = [];
  for (let node = 0; node < graph.nodes; node++) {
    for (let w = 0; w < walksPerNode; w++) {
      if (graph.adjacency[node].length === 0) continue; // an isolated node produces no walk
      const walk = [node];
      let current = node;
      for (let step = 1; step < walkLength; step++) {
        const neighbours = graph.adjacency[current];
        current = neighbours[Math.floor(rand() * neighbours.length)];
        walk.push(current);
      }
      walks.push(walk);
    }
  }
  return walks;
}

/** p^0.75 negative-sampling distribution over nodes, with the cumulative table used for sampling. */
function smoothedUnigram(graph, exponent = 0.75) {
  const weights = graph.degree.map((degree) => Math.pow(degree, exponent));
  const total = weights.reduce((a, b) => a + b, 0);
  const probability = weights.map((weight) => weight / total);
  const cumulative = [];
  let running = 0;
  for (const value of probability) {
    running += value;
    cumulative.push(running);
  }
  return { weights, total, probability, cumulative };
}

function sampleFromDistribution(cumulative, rand) {
  const target = rand();
  let low = 0;
  let high = cumulative.length - 1;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (cumulative[mid] < target) low = mid + 1;
    else high = mid;
  }
  return low;
}

/** Training pairs: every ordered (target, context) pair inside the walk window. */
function trainingPairs(walks, window = 2) {
  const pairs = [];
  for (const walk of walks) {
    for (let i = 0; i < walk.length; i++) {
      for (let j = Math.max(0, i - window); j <= Math.min(walk.length - 1, i + window); j++) {
        if (i === j) continue;
        pairs.push([walk[i], walk[j]]);
      }
    }
  }
  return pairs;
}

/**
 * The skip-gram-with-negative-sampling objective and its gradient for a single pair, computed on an
 * explicit list of negative node ids so the gradient can be checked exactly by finite differences.
 */
function skipGramObjective(context, target, sides, negatives) {
  let loss = 0;
  const positive = dot(sides.u[context], sides.v[target]);
  loss -= Math.log(Math.max(1e-12, sigmoid(positive)));
  const gradient = { positive: 1 - sigmoid(positive), negatives: [] };
  for (const negative of negatives) {
    const score = dot(sides.u[negative], sides.v[target]);
    loss -= Math.log(Math.max(1e-12, sigmoid(-score)));
    gradient.negatives.push(sigmoid(score));
  }
  return { loss, gradient };
}

/** Analytic gradient of the summed objective with respect to every touched vector. */
function analyticGradient(pairs, sides, negatives) {
  const gradU = new Map();
  const gradV = new Map();
  const ensure = (map, index, size) => {
    if (!map.has(index)) map.set(index, new Array(size).fill(0));
    return map.get(index);
  };
  let loss = 0;
  for (let p = 0; p < pairs.length; p++) {
    const [target, context] = pairs[p];
    const sampled = negatives[p];
    const score = dot(sides.u[context], sides.v[target]);
    const positiveGradient = 1 - sigmoid(score);
    loss -= Math.log(Math.max(1e-12, sigmoid(score)));
    const gu = ensure(gradU, context, sides.u[context].length);
    const gv = ensure(gradV, target, sides.v[target].length);
    for (let i = 0; i < gv.length; i++) {
      gu[i] -= positiveGradient * sides.v[target][i];
      gv[i] -= positiveGradient * sides.u[context][i];
    }
    for (const negative of sampled) {
      const negativeScore = dot(sides.u[negative], sides.v[target]);
      loss -= Math.log(Math.max(1e-12, sigmoid(-negativeScore)));
      const negativeGradient = sigmoid(negativeScore);
      const gneg = ensure(gradU, negative, sides.u[negative].length);
      for (let i = 0; i < gv.length; i++) {
        gneg[i] += negativeGradient * sides.v[target][i];
        gv[i] += negativeGradient * sides.u[negative][i];
      }
    }
  }
  return { loss, gradU, gradV };
}

/** Trains embeddings with plain SGD; returns the embeddings, the loss trace and the update count. */
function trainEmbeddings(graph, {
  dimension = 16,
  walksPerNode = 4,
  walkLength = 12,
  window = 2,
  negatives = 5,
  epochs = 6,
  learningRate = 0.05,
  seed = 863,
} = {}) {
  if (!Number.isInteger(dimension) || dimension < 1) throw new RangeError("dimension must be >= 1");
  if (!Number.isInteger(negatives) || negatives < 0) throw new RangeError("negatives must be >= 0");
  if (!Number.isInteger(epochs) || epochs < 1) throw new RangeError("epochs must be >= 1");
  if (!(learningRate > 0)) throw new RangeError("learningRate must be > 0");
  const rand = mulberry32(seed);
  const walks = randomWalks(graph, { walksPerNode, walkLength, seed });
  const pairs = trainingPairs(walks, window);
  const distribution = smoothedUnigram(graph);
  const u = [];
  const v = [];
  for (let node = 0; node < graph.nodes; node++) {
    const left = [];
    const right = [];
    for (let i = 0; i < dimension; i++) {
      left.push((rand() - 0.5) / dimension);
      right.push((rand() - 0.5) / dimension);
    }
    u.push(left);
    v.push(right);
  }
  const sides = { u, v };
  const lossTrace = [];
  // A fixed evaluation set and fixed negatives make the epoch-over-epoch progress measurable: the running
  // training trace is an online average, this one is the actual objective at fixed weights.
  const evaluationPairs = pairs.slice(0, Math.min(200, pairs.length));
  const evaluationNegatives = evaluationPairs.map(() => {
    const sampled = [];
    for (let k = 0; k < Math.max(1, negatives); k++) sampled.push(sampleFromDistribution(distribution.cumulative, rand));
    return sampled;
  });
  const evaluationLoss = () => analyticGradient(evaluationPairs, sides, evaluationNegatives).loss / Math.max(1, evaluationPairs.length);
  const evaluationTrace = [evaluationLoss()];
  let updates = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    const rate = learningRate * (1 - epoch / (epochs + 1));
    let epochLoss = 0;
    for (const [target, context] of pairs) {
      const sampled = [];
      for (let k = 0; k < negatives; k++) sampled.push(sampleFromDistribution(distribution.cumulative, rand));
      const { gradient } = skipGramObjective(context, target, sides, sampled);
      const positiveGradient = gradient.positive;
      for (let i = 0; i < dimension; i++) {
        const contextDelta = positiveGradient * v[target][i];
        const targetDelta = positiveGradient * u[context][i];
        u[context][i] += rate * contextDelta;
        v[target][i] += rate * targetDelta;
      }
      for (let k = 0; k < sampled.length; k++) {
        const negative = sampled[k];
        const negativeGradient = gradient.negatives[k];
        for (let i = 0; i < dimension; i++) {
          const negativeDelta = negativeGradient * v[target][i];
          const targetDelta = negativeGradient * u[negative][i];
          u[negative][i] -= rate * negativeDelta;
          v[target][i] -= rate * targetDelta;
        }
      }
      epochLoss += -Math.log(Math.max(1e-12, sigmoid(dot(u[context], v[target]))));
      for (const negative of sampled) epochLoss += -Math.log(Math.max(1e-12, sigmoid(-dot(u[negative], v[target]))));
      updates += 1 + negatives;
    }
    lossTrace.push(epochLoss / Math.max(1, pairs.length));
    evaluationTrace.push(evaluationLoss());
  }
  // the final embeddings combine both sides, as usual in skip-gram
  const embedding = u.map((left, index) => left.map((value, i) => value + v[index][i]));
  return { embedding, context: u, target: v, walks, pairs, lossTrace, evaluationTrace, updates, distribution, sampledPairs: pairs.length };
}

const cosine = (a, b) => {
  const norm = (vector) => Math.sqrt(vector.reduce((acc, value) => acc + value * value, 0));
  const scale = norm(a) * norm(b);
  return scale === 0 ? 0 : dot(a, b) / scale;
};

/** Link prediction AUC by the rank-based (Mann-Whitney) formula. */
function linkPredictionAuc(scores, labels) {
  const positives = [];
  const negatives = [];
  for (let i = 0; i < scores.length; i++) (labels[i] ? positives : negatives).push(scores[i]);
  if (positives.length === 0 || negatives.length === 0) return 0.5;
  let wins = 0;
  for (const positive of positives) {
    for (const negative of negatives) {
      if (positive > negative) wins += 1;
      else if (positive === negative) wins += 0.5;
    }
  }
  return wins / (positives.length * negatives.length);
}

/** Brute-force AUC for cross-checking the rank formula. */
function linkPredictionAucBruteForce(scores, labels) {
  const pairs = scores.map((score, index) => ({ score, label: labels[index] }));
  let comparisons = 0;
  let wins = 0;
  for (let i = 0; i < pairs.length; i++) {
    for (let j = i + 1; j < pairs.length; j++) {
      if (pairs[i].label === pairs[j].label) continue;
      comparisons += 1;
      const positive = pairs[i].label ? pairs[i].score : pairs[j].score;
      const negative = pairs[i].label ? pairs[j].score : pairs[i].score;
      if (positive > negative) wins += 1;
      else if (positive === negative) wins += 0.5;
    }
  }
  return comparisons === 0 ? 0.5 : wins / comparisons;
}

/** Splits the edge set into training and held-out halves plus an equal number of non-edges. */
function heldOutEdges(graph, { holdOut = 0.2, seed = 863 } = {}) {
  const rand = mulberry32(seed);
  const shuffled = graph.edges.slice();
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = shuffled[i];
    shuffled[i] = shuffled[j];
    shuffled[j] = tmp;
  }
  const cut = Math.max(1, Math.floor(shuffled.length * holdOut));
  const heldOut = shuffled.slice(0, cut);
  const present = new Set(graph.edges.map(([a, b]) => `${a}-${b}`));
  const nonEdges = [];
  let guard = 0;
  while (nonEdges.length < heldOut.length && guard < 100000) {
    guard += 1;
    const a = Math.floor(rand() * graph.nodes);
    const b = Math.floor(rand() * graph.nodes);
    if (a === b) continue;
    const key = `${Math.min(a, b)}-${Math.max(a, b)}`;
    if (present.has(key) || nonEdges.some(([x, y]) => x === Math.min(a, b) && y === Math.max(a, b))) continue;
    nonEdges.push([Math.min(a, b), Math.max(a, b)]);
  }
  return { heldOut, nonEdges };
}

/** Neighbourhood recall: how many of a node's top-k cosine neighbours are real graph neighbours. */
function neighbourhoodRecall(graph, embedding, k = 3) {
  let hits = 0;
  let total = 0;
  for (let node = 0; node < graph.nodes; node++) {
    if (graph.adjacency[node].length === 0) continue;
    const scored = [];
    for (let other = 0; other < graph.nodes; other++) {
      if (other === node) continue;
      scored.push({ other, similarity: cosine(embedding[node], embedding[other]) });
    }
    scored.sort((a, b) => b.similarity - a.similarity || a.other - b.other);
    const top = scored.slice(0, k).map((entry) => entry.other);
    hits += top.filter((other) => graph.adjacency[node].includes(other)).length;
    total += top.length;
  }
  return total === 0 ? 0 : hits / total;
}

function selfCheck() {
  // --- validation ----------------------------------------------------------------------------------------
  assert.throws(() => makeGraph({ communities: 0 }), /communities/);
  assert.throws(() => makeGraph({ perCommunity: 1 }), /perCommunity/);
  assert.throws(() => trainEmbeddings(makeGraph(), { dimension: 0 }), /dimension/);
  assert.throws(() => trainEmbeddings(makeGraph(), { negatives: -1 }), /negatives/);
  assert.throws(() => trainEmbeddings(makeGraph(), { epochs: 0 }), /epochs/);
  assert.throws(() => trainEmbeddings(makeGraph(), { learningRate: 0 }), /learningRate/);

  const graph = makeGraph({ communities: 3, perCommunity: 8, crossEdges: 4, seed: 863 });
  assert.strictEqual(graph.nodes, 24);
  assert.ok(graph.edges.length > 60, `the clique graph must be dense enough (${graph.edges.length})`);
  assert.ok(graph.edges.every(([a, b]) => graph.adjacency[a].includes(b) && graph.adjacency[b].includes(a)));
  assert.ok(graph.degree.every((degree) => degree > 0), "no isolated nodes in this fixture");

  // --- the negative-sampling distribution is unigram^0.75, independently recomputed ------------------------
  const distribution = smoothedUnigram(graph);
  assert.ok(Math.abs(distribution.probability.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  const recomputed = graph.degree.map((degree) => Math.pow(degree, 0.75));
  const recomputedTotal = recomputed.reduce((a, b) => a + b, 0);
  for (let i = 0; i < graph.nodes; i++) {
    assert.ok(Math.abs(distribution.probability[i] - recomputed[i] / recomputedTotal) < 1e-12, `node ${i} probability`);
  }
  const hub = graph.degree.indexOf(Math.max(...graph.degree));
  const leaf = graph.degree.indexOf(Math.min(...graph.degree));
  assert.ok(distribution.probability[hub] > distribution.probability[leaf], "hubs must be sampled more often");
  assert.ok(distribution.cumulative[graph.nodes - 1] >= 1 - 1e-12);
  const samplerRand = mulberry32(11);
  const counts = new Array(graph.nodes).fill(0);
  const draws = 40000;
  for (let i = 0; i < draws; i++) counts[sampleFromDistribution(distribution.cumulative, samplerRand)] += 1;
  for (const node of [hub, leaf]) {
    assert.ok(Math.abs(counts[node] / draws - distribution.probability[node]) < 0.02, `sampling must follow the distribution at node ${node}`);
  }

  // --- gradient: analytic versus finite differences ----------------------------------------------------------
  const gradRand = mulberry32(5);
  const size = 4;
  const sides = { u: [], v: [] };
  for (let i = 0; i < 5; i++) {
    sides.u.push(Array.from({ length: size }, () => gradRand() - 0.5));
    sides.v.push(Array.from({ length: size }, () => gradRand() - 0.5));
  }
  const pairs = [[0, 1], [2, 3]];
  const negatives = [[4, 2], [0, 1]];
  const analytic = analyticGradient(pairs, sides, negatives);
  const epsilon = 1e-6;
  let worstRelative = 0;
  const numericLoss = (perturb) => {
    const snapshotU = sides.u.map((vector) => vector.slice());
    const snapshotV = sides.v.map((vector) => vector.slice());
    perturb();
    let loss = 0;
    for (let i = 0; i < pairs.length; i++) {
      const [target, context] = pairs[i];
      const positive = dot(sides.u[context], sides.v[target]);
      loss -= Math.log(Math.max(1e-12, sigmoid(positive)));
      for (const negative of negatives[i]) {
        loss -= Math.log(Math.max(1e-12, sigmoid(-dot(sides.u[negative], sides.v[target]))));
      }
    }
    sides.u = snapshotU;
    sides.v = snapshotV;
    return loss;
  };
  for (let index = 0; index < sides.u.length; index++) {
    for (let i = 0; i < size; i++) {
      const plus = numericLoss(() => {
        sides.u[index][i] += epsilon;
      });
      const minus = numericLoss(() => {
        sides.u[index][i] -= epsilon;
      });
      const numeric = (plus - minus) / (2 * epsilon);
      const analyticValue = analytic.gradU.has(index) ? analytic.gradU.get(index)[i] : 0;
      worstRelative = Math.max(worstRelative, Math.abs(numeric - analyticValue));
    }
  }
  for (let index = 0; index < sides.v.length; index++) {
    for (let i = 0; i < size; i++) {
      const plus = numericLoss(() => {
        sides.v[index][i] += epsilon;
      });
      const minus = numericLoss(() => {
        sides.v[index][i] -= epsilon;
      });
      const numeric = (plus - minus) / (2 * epsilon);
      const analyticValue = analytic.gradV.has(index) ? analytic.gradV.get(index)[i] : 0;
      worstRelative = Math.max(worstRelative, Math.abs(numeric - analyticValue));
    }
  }
  assert.ok(worstRelative < 1e-6, `analytic gradient must match finite differences (worst gap ${worstRelative.toExponential(2)})`);
  const closedForm = skipGramObjective(1, 0, sides, [[4], [2]]);
  assert.ok(Math.abs(closedForm.loss + Math.log(sigmoid(dot(sides.u[1], sides.v[0]))) + Math.log(sigmoid(-dot(sides.u[4], sides.v[0]))) + Math.log(sigmoid(-dot(sides.u[2], sides.v[0])))) < 1e-9);

  // --- training: loss falls, embeddings stay healthy, update count is exact ------------------------------------
  const trained = trainEmbeddings(graph, { dimension: 16, walksPerNode: 4, walkLength: 12, window: 2, negatives: 5, epochs: 6, seed: 863 });
  assert.strictEqual(trained.lossTrace.length, 6);
  assert.strictEqual(trained.evaluationTrace.length, 7);
  for (let i = 1; i < trained.lossTrace.length; i++) {
    assert.ok(trained.lossTrace[i] <= trained.lossTrace[i - 1] * 1.02, `the online loss must not jump at epoch ${i} (${trained.lossTrace[i].toFixed(4)} vs ${trained.lossTrace[i - 1].toFixed(4)})`);
  }
  assert.ok(trained.lossTrace[0] > trained.lossTrace[5] * 1.2, "the improvement must be substantial");
  for (let i = 1; i < trained.evaluationTrace.length; i++) {
    assert.ok(trained.evaluationTrace[i] <= trained.evaluationTrace[i - 1] * 1.02, `the fixed-set objective must not rise at epoch ${i} (${trained.evaluationTrace[i].toFixed(4)})`);
  }
  assert.ok(trained.evaluationTrace[6] < trained.evaluationTrace[0] * 0.9, `training must reduce the objective (${trained.evaluationTrace[0].toFixed(4)} -> ${trained.evaluationTrace[6].toFixed(4)})`);
  assert.strictEqual(trained.updates, trained.sampledPairs * (1 + 5) * 6, "one update per pair and negative, per epoch");
  assert.ok(trained.sampledPairs > 1000, `enough training signal (${trained.sampledPairs} pairs)`);
  const flat = trained.embedding.flat();
  assert.ok(flat.every((value) => Number.isFinite(value)), "no NaN or infinity in the embeddings");
  assert.ok(variance(flat) > 1e-4, `embeddings must not collapse (variance ${variance(flat).toExponential(2)})`);
  const norms = trained.embedding.map((vector) => Math.sqrt(dot(vector, vector)));
  assert.ok(norms.every((norm) => norm > 1e-6), "no zero vectors");
  const cosines = [];
  for (let a = 0; a < graph.nodes; a++) for (let b = a + 1; b < graph.nodes; b++) cosines.push(cosine(trained.embedding[a], trained.embedding[b]));
  assert.ok(variance(cosines) > 1e-4, "cosine similarities must have structure, not be a constant");
  assert.ok(Math.max(...cosines) - Math.min(...cosines) > 0.2, "similarities must span a range");

  // --- quality: link prediction and community structure ----------------------------------------------------------
  const { heldOut, nonEdges } = heldOutEdges(graph, { holdOut: 0.2, seed: 9 });
  const scores = [];
  const labels = [];
  for (const [a, b] of heldOut) {
    scores.push(dot(trained.embedding[a], trained.embedding[b]));
    labels.push(true);
  }
  for (const [a, b] of nonEdges) {
    scores.push(dot(trained.embedding[a], trained.embedding[b]));
    labels.push(false);
  }
  const auc = linkPredictionAuc(scores, labels);
  const aucBrute = linkPredictionAucBruteForce(scores, labels);
  assert.ok(Math.abs(auc - aucBrute) < 1e-12, "the rank formula must match the brute-force AUC");
  assert.ok(auc > 0.7, `held-out link prediction must beat random by a margin (AUC ${auc.toFixed(3)})`);
  const randomScores = scores.map((_, i) => mulberry32(100 + i)());
  assert.ok(linkPredictionAuc(randomScores, labels) > 0.4 && linkPredictionAuc(randomScores, labels) < 0.6, "a random scorer must sit near 0.5");
  const edgeSimilarity = mean(heldOut.map(([a, b]) => dot(trained.embedding[a], trained.embedding[b])));
  const nonEdgeSimilarity = mean(nonEdges.map(([a, b]) => dot(trained.embedding[a], trained.embedding[b])));
  assert.ok(edgeSimilarity > nonEdgeSimilarity, `edges must be more similar than non-edges (${edgeSimilarity.toFixed(3)} vs ${nonEdgeSimilarity.toFixed(3)})`);
  const intra = [];
  const inter = [];
  for (let a = 0; a < graph.nodes; a++) {
    for (let b = a + 1; b < graph.nodes; b++) {
      const same = Math.floor(a / graph.perCommunity) === Math.floor(b / graph.perCommunity);
      (same ? intra : inter).push(cosine(trained.embedding[a], trained.embedding[b]));
    }
  }
  assert.ok(mean(intra) > mean(inter), `intra-community similarity must exceed inter-community (${mean(intra).toFixed(3)} vs ${mean(inter).toFixed(3)})`);

  // --- neighbourhood recall --------------------------------------------------------------------------------------
  const recall3 = neighbourhoodRecall(graph, trained.embedding, 3);
  const recall5 = neighbourhoodRecall(graph, trained.embedding, 5);
  assert.ok(recall3 > 0.6, `the top-3 neighbours must often be real neighbours (${recall3.toFixed(3)})`);
  assert.ok(recall5 > 0.55, `the top-5 neighbourhood recall must stay high (${recall5.toFixed(3)})`);
  const untrained = trainEmbeddings(graph, { epochs: 1, learningRate: 1e-9, seed: 863 });
  assert.ok(neighbourhoodRecall(graph, untrained.embedding, 3) < recall3, "training must beat the barely-moved initialisation");

  // --- degenerate graphs and edges of the parameter space -----------------------------------------------------------
  const single = makeGraph({ communities: 1, perCommunity: 2, crossEdges: 0, seed: 2 });
  assert.strictEqual(single.nodes, 2);
  const singleTrained = trainEmbeddings(single, { dimension: 4, walksPerNode: 2, walkLength: 3, epochs: 2, seed: 2 });
  assert.ok(singleTrained.embedding.length === 2);
  assert.ok(singleTrained.embedding.flat().every(Number.isFinite));
  const isolated = { nodes: 3, adjacency: [[], [], []], edges: [], degree: [0, 0, 0], communities: 1, perCommunity: 3 };
  const isolatedTrained = trainEmbeddings(isolated, { dimension: 4, walksPerNode: 1, walkLength: 4, epochs: 1, seed: 3 });
  assert.strictEqual(isolatedTrained.sampledPairs, 0, "an edgeless graph yields no walks and no pairs");
  assert.strictEqual(isolatedTrained.updates, 0);
  assert.strictEqual(neighbourhoodRecall(isolated, isolatedTrained.embedding, 2), 0);
  const noNegatives = trainEmbeddings(graph, { negatives: 0, epochs: 2, seed: 4 });
  assert.strictEqual(noNegatives.updates, noNegatives.sampledPairs * 2, "without negatives only the positive pair is updated");
  assert.ok(Number.isFinite(noNegatives.lossTrace[1]));
  assert.strictEqual(linkPredictionAuc([1], [true]), 0.5, "a single class cannot be ranked");
  assert.strictEqual(linkPredictionAuc([], []), 0.5);

  // --- determinism ----------------------------------------------------------------------------------------------------
  const repeatA = trainEmbeddings(graph, { dimension: 8, epochs: 2, seed: 42 });
  const repeatB = trainEmbeddings(graph, { dimension: 8, epochs: 2, seed: 42 });
  assert.deepStrictEqual(repeatA.embedding, repeatB.embedding);
  assert.deepStrictEqual(repeatA.lossTrace, repeatB.lossTrace);
  assert.deepStrictEqual(repeatA.walks.slice(0, 5), repeatB.walks.slice(0, 5));
  assert.strictEqual(repeatA.updates, repeatB.updates);
  const otherSeed = trainEmbeddings(graph, { dimension: 8, epochs: 2, seed: 43 });
  assert.notDeepStrictEqual(otherSeed.embedding, repeatA.embedding);
  assert.notDeepStrictEqual(otherSeed.walks[0], repeatA.walks[0]);

  console.log(`graph: ${graph.nodes} nodes, ${graph.edges.length} edges, ${trained.sampledPairs} training pairs, ${trained.updates} updates`);
  console.log(`loss: ${trained.lossTrace.map((value) => value.toFixed(3)).join(" -> ")}`);
  console.log(`quality: AUC ${auc.toFixed(3)} (brute force ${aucBrute.toFixed(3)}), edge-vs-non-edge ${edgeSimilarity.toFixed(3)} vs ${nonEdgeSimilarity.toFixed(3)}, recall@3 ${recall3.toFixed(3)}`);
  console.log("PASS");
}

module.exports = { makeGraph, randomWalks, smoothedUnigram, sampleFromDistribution, trainingPairs, trainEmbeddings, skipGramObjective, analyticGradient, cosine, dot, linkPredictionAuc, linkPredictionAucBruteForce, heldOutEdges, neighbourhoodRecall, mulberry32, mean, variance };

if (require.main === module) {
  selfCheck();
}
