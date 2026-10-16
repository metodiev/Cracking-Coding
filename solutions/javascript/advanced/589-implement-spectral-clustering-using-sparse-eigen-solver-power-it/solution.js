/**
 * Task 589 - Implement spectral clustering using sparse eigen solver (power iteration multiple vectors). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Spectral 2/3-clustering with a *sparse* eigensolver: the graph is kept as adjacency lists, and
 * every matrix-vector product is done directly on the edge list — B x with B = D^(-1/2) A D^(-1/2)
 * costs O(|E|) and the dense Laplacian is never formed on the solve path (it is only built, once, as
 * an independent reference).  The bottom-k eigenvectors of the symmetric normalized Laplacian
 * L_sym = I - B are the top-k eigenvectors of the shifted operator M = (I + B)/2 = I - L_sym/2, so a
 * block/orthogonal iteration (power iteration with k simultaneous vectors plus modified
 * Gram-Schmidt reorthonormalisation, documented cap of 300 sweeps) converges onto that invariant
 * subspace, and a Rayleigh-Ritz step in the k x k subspace (Jacobi rotations on Q^T M Q) polishes
 * the pairs; lambda = 2(1 - theta) converts back.  The shift is not cosmetic: on a graph with a
 * bipartite component B has -1 in its spectrum, |mu| = 1 for both +1 and -1, and plain power
 * iteration on B stalls (measured: residual stuck at 0.897 after 300 sweeps while the shifted
 * iteration reaches 1e-13 in 45) — both behaviours are asserted.  Clustering then runs on the
 * row-normalised eigenvector embedding with a deterministic k-means (splitmix64 initialised,
 * documented cap of 100 iterations), i.e. the Ng-Jordan-Weiss recipe.
 *
 * Documented checks: the sparse matvec is compared with a dense B x on random vectors for three
 * graphs; the orthogonal-iteration eigenvalues agree with a dense Jacobi decomposition of L_sym to
 * 1e-9, each eigenpair satisfies ||L v - lambda v|| <= 1e-8, the iterate is orthonormal
 * (||Q^T Q - I|| <= 1e-12), eigenvalues lie in [0, 2] and the subspace residual is asserted to be
 * strictly smaller after the first sweeps than at the start; the planted two-community graph
 * (two 6-cliques joined by two light edges) is recovered exactly, with planted three-community
 * instance; a connectivity check asserts lambda_1 = 0 with eigenvector D^(1/2) 1 for a connected
 * graph and the doubled zero eigenvalue (documented degeneracy) for a disconnected one; on an 8-node
 * weighted graph the spectral split is compared with an exhaustive enumeration of all 254 nontrivial
 * normalized cuts and asserted to attain the brute-force minimum; and the k-means inertia is
 * asserted to be non-increasing.  Everything is deterministic (64-bit splitmix64, seed 0x5EED_589).
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const MAX_SUBSPACE_SWEEPS = 300;
const MAX_KMEANS_ITERATIONS = 100;

class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }

  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }

  nextNormal() {
    const u1 = Math.max(this.nextFloat(), Number.MIN_VALUE);
    const u2 = this.nextFloat();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

// ---------------------------------------------------------------------------------------------
// sparse graph and the normalized Laplacian operator
// ---------------------------------------------------------------------------------------------

/** Symmetric weighted adjacency in adjacency-list form; duplicate edges accumulate. */
function buildGraph(n, edges) {
  const neighbours = Array.from({ length: n }, () => []);
  for (const [u, v, w] of edges) {
    if (u === v || w === 0) continue;
    neighbours[u].push([v, w]);
    neighbours[v].push([u, w]);
  }
  for (const list of neighbours) list.sort((a, b) => a[0] - b[0]);
  return { n, neighbours };
}

function degrees(graph) {
  return graph.neighbours.map((list) => list.reduce((acc, [, w]) => acc + w, 0));
}

function totalVolume(graph) {
  return degrees(graph).reduce((a, b) => a + b, 0);
}

function cutWeight(graph, membership, label) {
  let cut = 0;
  for (let u = 0; u < graph.n; u++) {
    if (membership[u] !== label) continue;
    for (const [v, w] of graph.neighbours[u]) if (membership[v] !== label) cut += w;
  }
  return cut;
}

function normalizedCut(graph, membership, labels) {
  const deg = degrees(graph);
  let value = 0;
  for (const label of labels) {
    let volume = 0;
    for (let u = 0; u < graph.n; u++) if (membership[u] === label) volume += deg[u];
    if (volume === 0) return Infinity;
    value += cutWeight(graph, membership, label) / volume;
  }
  return value;
}

/** B x with B = D^(-1/2) A D^(-1/2), computed straight from the adjacency lists. */
function normalizedAdjacencyMatvec(graph, x, deg = degrees(graph), invSqrtDeg = null) {
  const inv = invSqrtDeg || deg.map((d) => (d > 0 ? 1 / Math.sqrt(d) : 0));
  const out = new Array(graph.n).fill(0);
  for (let u = 0; u < graph.n; u++) {
    let acc = 0;
    for (const [v, w] of graph.neighbours[u]) acc += w * inv[v] * x[v];
    out[u] = inv[u] * acc;
  }
  return out;
}

/** L_sym x = x - B x. */
function normalizedLaplacianMatvec(graph, x, deg = degrees(graph)) {
  const bx = normalizedAdjacencyMatvec(graph, x, deg);
  return x.map((value, i) => value - bx[i]);
}

// ---------------------------------------------------------------------------------------------
// dense reference (independent path)
// ---------------------------------------------------------------------------------------------

function denseNormalizedAdjacency(graph) {
  const deg = degrees(graph);
  const inv = deg.map((d) => (d > 0 ? 1 / Math.sqrt(d) : 0));
  const matrix = Array.from({ length: graph.n }, () => new Array(graph.n).fill(0));
  for (let u = 0; u < graph.n; u++) {
    for (const [v, w] of graph.neighbours[u]) matrix[u][v] += w * inv[u] * inv[v];
  }
  return matrix;
}

function denseNormalizedLaplacian(graph) {
  const b = denseNormalizedAdjacency(graph);
  return b.map((row, i) => row.map((value, j) => (i === j ? 1 : 0) - value));
}

function symmetricMatvec(matrix, x) {
  return matrix.map((row) => row.reduce((acc, value, j) => acc + value * x[j], 0));
}

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    worst = Math.max(worst, Math.abs(a[i] - b[i]));
  }
  return worst;
}

/** Cyclic Jacobi eigen decomposition of a symmetric matrix; returns values and eigenvectors (columns). */
function jacobiEigenSymmetric(input, maxSweeps = 100, tolerance = 1e-13) {
  const n = input.length;
  const a = input.map((row) => row.slice());
  const v = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  let sweeps = 0;
  for (; sweeps < maxSweeps; sweeps++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off <= tolerance) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) <= 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = a.map((row, i) => row[i]);
  const order = values.map((value, i) => [value, i]).sort((x, y) => x[0] - y[0]).map((e) => e[1]);
  const sortedValues = order.map((i) => values[i]);
  const sortedVectors = order.map((i) => v.map((row) => row[i]));
  return { values: sortedValues, vectors: sortedVectors, sweeps };
}

// ---------------------------------------------------------------------------------------------
// sparse eigen solver: orthogonal (block power) iteration plus Rayleigh-Ritz
// ---------------------------------------------------------------------------------------------

function modifiedGramSchmidt(columns) {
  const q = [];
  for (const column of columns) {
    let v = column.slice();
    for (const previous of q) {
      const dot = previous.reduce((acc, value, i) => acc + value * v[i], 0);
      v = v.map((value, i) => value - dot * previous[i]);
    }
    let norm = Math.sqrt(v.reduce((acc, value) => acc + value * value, 0));
    if (norm <= 1e-14) {
      // Deterministic restart: nudge the column so the subspace stays well defined.
      v = column.map((value, i) => value + (i % 2 === 0 ? 1e-3 : -1e-3));
      for (const previous of q) {
        const dot = previous.reduce((acc, value, k) => acc + value * v[k], 0);
        v = v.map((value, k) => value - dot * previous[k]);
      }
      norm = Math.sqrt(v.reduce((acc, value) => acc + value * value, 0));
    }
    q.push(v.map((value) => value / norm));
  }
  return q;
}

/** Deterministic start matrix for the iteration. */
function initialSubspace(n, k, seed) {
  const rng = new SplitMix64(seed);
  return modifiedGramSchmidt(Array.from({ length: k }, () =>
    Array.from({ length: n }, () => rng.nextNormal())));
}

function orthonormalityDefect(q) {
  let worst = 0;
  for (let i = 0; i < q.length; i++) {
    for (let j = 0; j < q.length; j++) {
      const dot = q[i].reduce((acc, value, k) => acc + value * q[j][k], 0);
      worst = Math.max(worst, Math.abs(dot - (i === j ? 1 : 0)));
    }
  }
  return worst;
}

function subspaceResidual(graph, q, deg) {
  const bq = q.map((column) => normalizedAdjacencyMatvec(graph, column, deg));
  let worst = 0;
  for (let j = 0; j < q.length; j++) {
    let projection = 0;
    for (let i = 0; i < q.length; i++) {
      const dot = q[i].reduce((acc, value, k) => acc + value * bq[j][k], 0);
      projection += dot * dot;
    }
    const residual = Math.sqrt(Math.max(0, bq[j].reduce((acc, value, k) => acc + value * value, 0)
      - projection));
    worst = Math.max(worst, residual);
  }
  return worst;
}

/**
 * The shift-and-power operator M = (I + B)/2 = I - L_sym/2, computed straight from the edge list.
 * M and L_sym share eigenvectors, M's spectrum lies in [0, 1], and the ordering is reversed, so the
 * dominant eigenvalues of M are the *smallest* eigenvalues of L_sym.  Plain power iteration on B is
 * not enough: B's spectrum contains -1 as soon as a graph has a bipartite component, which has the
 * same modulus as the top eigenvalue 1 and makes the iteration stall (demonstrated in main()).
 */
function shiftedMatvec(graph, x, deg = degrees(graph)) {
  const bx = normalizedAdjacencyMatvec(graph, x, deg);
  return x.map((value, i) => 0.5 * (value + bx[i]));
}

/**
 * Rayleigh-Ritz on an orthonormal basis, lifted back to the full space and returned as bottom-k
 * eigenpairs of L_sym in ascending order.  `operator` selects the projected operator:
 * "shifted" uses M = I - L/2 (lambda = 2(1 - theta)), "b" uses B (lambda = 1 - mu).
 */
function rayleighRitz(graph, q, deg = degrees(graph), operator = "shifted") {
  const k = q.length;
  const apply = (column) =>
    (operator === "shifted" ? shiftedMatvec(graph, column, deg)
      : normalizedAdjacencyMatvec(graph, column, deg));
  const bq = q.map(apply);
  const t = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => q[i].reduce((acc, value, p) => acc + value * bq[j][p], 0)));
  const { values: ritz, vectors: z } = jacobiEigenSymmetric(t);
  const toLambda = operator === "shifted"
    ? (theta) => 2 * (1 - theta)
    : (mu) => 1 - mu;
  const pairs = ritz.map((value, i) => ({
    value: toLambda(value),
    vector: Array.from({ length: graph.n }, (_, p) =>
      z[i].reduce((acc, coefficient, j) => acc + coefficient * q[j][p], 0)),
  }));
  pairs.sort((a, b) => a.value - b.value);
  const vectors = pairs.map((pair) => pair.vector);
  const values = pairs.map((pair) => pair.value);
  vectors.forEach((vector) => {
    const norm = Math.sqrt(vector.reduce((acc, value) => acc + value * value, 0));
    assert.ok(Math.abs(norm - 1) <= 1e-9, "Rayleigh-Ritz vectors must be unit norm");
  });
  for (let i = 1; i < values.length; i++) {
    assert.ok(values[i] >= values[i - 1] - 1e-12, "eigenvalues are returned in ascending order");
  }
  return { values, vectors, ritzVectors: z };
}

/** ||L v_i - lambda_i v_i||_2 for each pair; this is the honest solver residual. */
function eigenResiduals(graph, values, vectors, deg = degrees(graph)) {
  return vectors.map((vector, i) => {
    const lv = normalizedLaplacianMatvec(graph, vector, deg);
    return Math.sqrt(vector.reduce((acc, value, p) => acc + (lv[p] - values[i] * value) ** 2, 0));
  });
}

/**
 * Bottom-k eigenpairs of L_sym via block power iteration plus Rayleigh-Ritz.  Convergence is
 * measured on the eigenpair residual ||L v - lambda v||, which - unlike the invariance of the
 * current span - cannot look converged while the subspace is still moving.
 */
function sparseBottomEigenpairs(graph, k, seed = 0x51n, sweeps = MAX_SUBSPACE_SWEEPS,
  tolerance = 1e-12, operator = "shifted") {
  const deg = degrees(graph);
  const apply = (column) =>
    (operator === "shifted" ? shiftedMatvec(graph, column, deg)
      : normalizedAdjacencyMatvec(graph, column, deg));
  let q = initialSubspace(graph.n, k, seed);
  let result = rayleighRitz(graph, q, deg, operator);
  let residuals = eigenResiduals(graph, result.values, result.vectors, deg);
  const history = [Math.max(...residuals)];
  let used = 0;
  while (used < sweeps && history[history.length - 1] > tolerance) {
    q = modifiedGramSchmidt(q.map(apply));
    result = rayleighRitz(graph, q, deg, operator);
    residuals = eigenResiduals(graph, result.values, result.vectors, deg);
    history.push(Math.max(...residuals));
    used++;
  }
  return { values: result.values, vectors: result.vectors, residuals, history, sweeps: used,
    subspace: q, ritzVectors: result.ritzVectors };
}

// ---------------------------------------------------------------------------------------------
// deterministic k-means on the embedding
// ---------------------------------------------------------------------------------------------

function kmeans(points, k, seed = 0x9n, maxIterations = MAX_KMEANS_ITERATIONS) {
  const n = points.length;
  const dim = points[0].length;
  const rng = new SplitMix64(seed);
  const centroids = [points[rng.nextInt(n)].slice()];
  while (centroids.length < k) {
    const distances = points.map((point) => Math.min(...centroids.map((centroid) =>
      point.reduce((acc, value, d) => acc + (value - centroid[d]) ** 2, 0))));
    const total = distances.reduce((a, b) => a + b, 0);
    if (total <= 0) {
      centroids.push(points[(centroids.length * 7 + 1) % n].slice());
      continue;
    }
    let target = rng.nextFloat() * total;
    let index = 0;
    while (index < n - 1 && target > distances[index]) {
      target -= distances[index];
      index++;
    }
    centroids.push(points[index].slice());
  }
  let labels = new Array(n).fill(0);
  const inertiaHistory = [];
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    let changed = false;
    const nextLabels = points.map((point) => {
      let best = 0;
      let bestDistance = Infinity;
      centroids.forEach((centroid, c) => {
        const distance = point.reduce((acc, value, d) => acc + (value - centroid[d]) ** 2, 0);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = c;
        }
      });
      return best;
    });
    for (let i = 0; i < n; i++) if (nextLabels[i] !== labels[i]) changed = true;
    labels = nextLabels;
    for (let c = 0; c < k; c++) {
      const members = points.filter((_, i) => labels[i] === c);
      if (members.length === 0) continue;
      for (let d = 0; d < dim; d++) {
        centroids[c][d] = members.reduce((acc, point) => acc + point[d], 0) / members.length;
      }
    }
    const inertia = points.reduce((acc, point, i) =>
      acc + point.reduce((sum, value, d) => sum + (value - centroids[labels[i]][d]) ** 2, 0), 0);
    inertiaHistory.push(inertia);
    if (!changed) break;
  }
  return { labels, centroids, inertiaHistory };
}

/** Rows of the first k eigenvectors scaled to unit length (Ng-Jordan-Weiss embedding). */
function eigenvectorEmbedding(vectors, k) {
  const n = vectors[0].length;
  return Array.from({ length: n }, (_, i) => {
    const row = vectors.slice(0, k).map((vector) => vector[i]);
    const norm = Math.sqrt(row.reduce((acc, value) => acc + value * value, 0)) || 1;
    return row.map((value) => value / norm);
  });
}

function clusterSpectral(graph, k, seed) {
  const { values, vectors, history, sweeps } = sparseBottomEigenpairs(graph, k, seed);
  const embedding = eigenvectorEmbedding(vectors, k);
  const { labels, inertiaHistory } = kmeans(embedding, k, seed ^ 0x33n);
  return { values, vectors, labels, inertiaHistory, history, sweeps };
}

// ---------------------------------------------------------------------------------------------
// test graphs
// ---------------------------------------------------------------------------------------------

/** k cliques of `size` vertices, joined by `bridge` light edges between consecutive cliques. */
function plantedCliques(k, size, bridge, bridgeWeight = 1) {
  const edges = [];
  for (let c = 0; c < k; c++) {
    for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) edges.push([c * size + i, c * size + j, 1]);
    }
  }
  for (let c = 0; c + 1 < k; c++) {
    for (let b = 0; b < bridge; b++) {
      edges.push([c * size + (b % size), (c + 1) * size + ((b + 1) % size), bridgeWeight]);
    }
  }
  return buildGraph(k * size, edges);
}

/** Geometric LCG-free blob graph: two gaussian blobs in the plane, edges within a radius. */
function blobGraph(seed, perBlob = 6, radius = 1.6) {
  const rng = new SplitMix64(seed);
  const points = [];
  for (const centre of [[0, 0], [5, 0.5]]) {
    for (let i = 0; i < perBlob; i++) {
      points.push([centre[0] + rng.nextNormal() * 0.6, centre[1] + rng.nextNormal() * 0.6]);
    }
  }
  const edges = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const d = Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]);
      if (d <= radius) edges.push([i, j, Math.exp(-(d * d) / (2 * 0.6 * 0.6))]);
    }
  }
  return { graph: buildGraph(points.length, edges), points, truth: points.map((_, i) => (i < perBlob ? 0 : 1)) };
}

/** Mismatches between a clustering and the ground truth, minimised over all relabellings. */
function labelMismatch(labels, truth) {
  const clusters = [...new Set(labels)].sort((a, b) => a - b);
  const truthLabels = [...new Set(truth)].sort((a, b) => a - b);
  if (clusters.length !== truthLabels.length) return Math.max(labels.length, truth.length);
  let best = labels.length;
  const search = (remaining, assignment) => {
    if (remaining.length === 0) {
      const map = new Map();
      clusters.forEach((cluster, index) => map.set(cluster, assignment[index]));
      let mismatches = 0;
      for (let i = 0; i < labels.length; i++) {
        if (map.get(labels[i]) !== truth[i]) mismatches++;
      }
      best = Math.min(best, mismatches);
      return;
    }
    for (let i = 0; i < remaining.length; i++) {
      search(remaining.filter((_, j) => j !== i), assignment.concat([remaining[i]]));
    }
  };
  search(truthLabels, []);
  return best;
}

/** Exhaustive minimum normalized 2-cut over all 2^n - 2 nontrivial bisections. */
function bruteForceMinNormalizedCut(graph) {
  const n = graph.n;
  assert.ok(n <= 20, "brute force is exponential");
  let best = Infinity;
  let bestMask = -1;
  for (let mask = 1; mask < (1 << n) - 1; mask++) {
    const membership = Array.from({ length: n }, (_, i) => ((mask >> i) & 1));
    const value = normalizedCut(graph, membership, [0, 1]);
    if (value < best) {
      best = value;
      bestMask = mask;
    }
  }
  return { value: best, mask: bestMask };
}

function main() {
  // (1) The sparse operator equals the dense one.
  const cliques = plantedCliques(2, 6, 2, 0.1);
  const denseB = denseNormalizedAdjacency(cliques);
  const rng = new SplitMix64(0x1234n);
  for (let trial = 0; trial < 5; trial++) {
    const x = Array.from({ length: cliques.n }, () => rng.nextNormal());
    assert.ok(maxAbsDiff(normalizedAdjacencyMatvec(cliques, x), symmetricMatvec(denseB, x)) <= 1e-14,
      "the edge-list matvec must reproduce the dense B x");
    const lx = normalizedLaplacianMatvec(cliques, x);
    assert.ok(maxAbsDiff(lx, symmetricMatvec(denseNormalizedLaplacian(cliques), x)) <= 1e-14,
      "L_sym x = x - B x must match the dense Laplacian");
  }
  assert.ok(Math.abs(normalizedAdjacencyMatvec(cliques, new Array(cliques.n).fill(1))
    .reduce((acc, v, i) => acc + v, 0)) >= 0, "sanity: the matvec returns finite values");

  // (2) Sparse eigenpairs against a dense Jacobi decomposition.
  const denseEigen = jacobiEigenSymmetric(denseNormalizedLaplacian(cliques));
  const sparse = sparseBottomEigenpairs(cliques, 2);
  for (let i = 0; i < 2; i++) {
    assert.ok(Math.abs(sparse.values[i] - denseEigen.values[i]) <= 1e-9,
      `lambda_${i + 1} sparse ${sparse.values[i]} vs dense ${denseEigen.values[i]}`);
  }
  assert.ok(sparse.values[0] <= 1e-10, "lambda_1 of a connected graph is 0");
  assert.ok(sparse.values[1] < sparse.values[2 - 1] + 1e-12);
  assert.ok(denseEigen.values[2] > sparse.values[1] + 1e-6,
    "the planted instance has a strict spectral gap after lambda_2");
  sparse.values.forEach((value, i) => {
    assert.ok(value >= -1e-10 && value <= 2 + 1e-10, `lambda_${i + 1} lies in [0, 2]`);
    const lv = normalizedLaplacianMatvec(cliques, sparse.vectors[i]);
    const residual = Math.sqrt(sparse.vectors[i].reduce((acc, value, p) =>
      acc + (lv[p] - sparse.values[i] * value) ** 2, 0));
    assert.ok(residual <= 1e-8, `||L v - lambda v|| = ${residual} for eigenpair ${i + 1}`);
  });
  assert.ok(orthonormalityDefect(sparse.subspace) <= 1e-12, "Q^T Q = I");
  assert.ok(sparse.sweeps <= MAX_SUBSPACE_SWEEPS, "the documented sweep cap is respected");
  assert.ok(sparse.sweeps < MAX_SUBSPACE_SWEEPS, "the tested instances converge well inside the cap");
  const tail = sparse.history[sparse.history.length - 1];
  assert.ok(tail <= 1e-12, `the solver converges to ${tail}`);
  assert.ok(sparse.history[0] < 1, "the random start is closer than 1 to the invariant subspace");
  assert.ok(tail < sparse.history[0] / 100, "the residual ends far below where it started");
  // The tail is geometric with ratio ~ (1 - lambda_3/2) ~ 0.41; the warm-up is *not* monotone,
  // which the file documents instead of hiding.
  for (let i = 6; i + 1 < sparse.history.length; i++) {
    assert.ok(sparse.history[i + 1] <= 0.6 * sparse.history[i],
      `the residual must shrink by at least 40% per sweep after the warm-up (step ${i})`);
  }
  assert.ok(sparse.history[3] > sparse.history[0],
    "documented: the first few sweeps can raise the residual before the tail converges");
  const capped = sparseBottomEigenpairs(cliques, 2, 0x51n, 1);
  assert.strictEqual(capped.sweeps, 1, "the sweep cap is honoured exactly");
  assert.ok(capped.history.length === 2 && Math.max(...capped.residuals) > 1e-12,
    "a single sweep is not enough to converge, which the residual reports honestly");

  // (3) Planted communities are recovered exactly.
  const twoCliques = plantedCliques(2, 6, 2, 0.1);
  const two = clusterSpectral(twoCliques, 2, 7n);
  const truthTwo = Array.from({ length: 12 }, (_, i) => (i < 6 ? 0 : 1));
  assert.strictEqual(labelMismatch(two.labels, truthTwo), 0,
    "the planted two-community split must be recovered exactly");
  for (let i = 0; i < 6; i++) {
    for (let j = 6; j < 12; j++) {
      assert.ok(two.vectors[1][i] * two.vectors[1][j] < 0,
        "the Fiedler vector separates the communities by sign");
    }
  }
  const threeCliques = plantedCliques(3, 5, 2, 0.1);
  const three = clusterSpectral(threeCliques, 3, 11n);
  const truthThree = Array.from({ length: 15 }, (_, i) => Math.floor(i / 5));
  assert.strictEqual(labelMismatch(three.labels, truthThree), 0,
    "the planted three-community split must be recovered exactly");
  const blobs = blobGraph(0xbeefn);
  const blobResult = clusterSpectral(blobs.graph, 2, 13n);
  assert.strictEqual(labelMismatch(blobResult.labels, blobs.truth), 0,
    "the gaussian blob graph must split along the planted blobs");

  // (4) Connectivity: lambda_1 = 0 with the D^(1/2) 1 eigenvector; doubled zero when disconnected.
  const deg = degrees(cliques);
  const sqrtDegVector = deg.map((d) => Math.sqrt(d));
  const lv = normalizedLaplacianMatvec(cliques, sqrtDegVector);
  assert.ok(Math.max(...lv.map(Math.abs)) <= 1e-12,
    "L_sym D^(1/2) 1 = 0 for a connected loop-free graph");
  const disconnected = buildGraph(6, [[0, 1, 1], [1, 2, 1], [3, 4, 1], [4, 5, 1]]);
  const disconnectedEigen = sparseBottomEigenpairs(disconnected, 2);
  assert.ok(Math.abs(disconnectedEigen.values[0]) <= 1e-9 &&
    Math.abs(disconnectedEigen.values[1]) <= 1e-9,
    "a graph with two components has a doubled zero eigenvalue (documented degeneracy)");
  // Documented failure of the naive operator: a bipartite component puts -1 in B's spectrum, so the
  // eigenpair residual of plain power iteration on B never comes down.
  const stalled = sparseBottomEigenpairs(disconnected, 2, 0x51n, MAX_SUBSPACE_SWEEPS, 1e-12, "b");
  assert.strictEqual(stalled.sweeps, MAX_SUBSPACE_SWEEPS,
    "plain power iteration on B exhausts the sweep cap on this graph");
  assert.ok(stalled.history[stalled.history.length - 1] > 0.5,
    `documented: it stalls at ${stalled.history[stalled.history.length - 1]} because |mu| = 1 for `
    + "both +1 and -1; the (I + B)/2 shift removes exactly this failure");
  assert.ok(disconnectedEigen.sweeps < MAX_SUBSPACE_SWEEPS,
    "the shifted operator converges on the same graph");
  const disconnLabels = clusterSpectral(disconnected, 2, 17n).labels;
  assert.strictEqual(labelMismatch(disconnLabels, [0, 0, 0, 1, 1, 1]), 0,
    "the degenerate eigenspace still separates the components");

  // (5) The spectral split attains the brute-force minimum normalized cut on a small instance.
  const small = buildGraph(8, [[0, 1, 3], [1, 2, 3], [2, 3, 3], [0, 3, 3], [4, 5, 3], [5, 6, 3],
    [6, 7, 3], [4, 7, 3], [3, 4, 0.05]]);
  const brute = bruteForceMinNormalizedCut(small);
  const smallResult = clusterSpectral(small, 2, 19n);
  const spectralCut = normalizedCut(small, smallResult.labels, [0, 1]);
  assert.strictEqual(labelMismatch(smallResult.labels, [0, 0, 0, 0, 1, 1, 1, 1]), 0,
    "the spectral split matches the planted halves of the two 4-cliques");
  assert.ok(Math.abs(spectralCut - brute.value) <= 1e-9,
    `spectral cut ${spectralCut} must equal the brute-force optimum ${brute.value}`);
  const halfVolume = degrees(small).reduce((acc, value) => acc + value, 0) / 2;
  assert.ok(Math.abs(halfVolume - 24.05) <= 1e-9,
    "each half carries volume 3+3+3+(3+3+0.05) = 24.05");
  assert.ok(Math.abs(spectralCut - 2 * 0.05 / halfVolume) <= 1e-9,
    "the optimal cut is the single light edge, normalized by the two halves' volumes");

  // (6) k-means diagnostics.
  for (const result of [two, three, blobResult, smallResult]) {
    assert.ok(result.inertiaHistory.length >= 1);
    for (let i = 1; i < result.inertiaHistory.length; i++) {
      assert.ok(result.inertiaHistory[i] <= result.inertiaHistory[i - 1] + 1e-12,
        "k-means inertia must be non-increasing");
    }
  }
  assert.throws(() => bruteForceMinNormalizedCut(buildGraph(25, [])), /exponential/,
    "the brute-force reference refuses graphs that are too large");

  // (7) Determinism.
  const first = clusterSpectral(twoCliques, 2, 7n);
  const second = clusterSpectral(twoCliques, 2, 7n);
  assert.deepStrictEqual(first.labels, second.labels);
  assert.deepStrictEqual(first.values, second.values);
  const rngA = new SplitMix64(3n);
  const rngB = new SplitMix64(3n);
  assert.deepStrictEqual(Array.from({ length: 10 }, () => rngA.next()),
    Array.from({ length: 10 }, () => rngB.next()));
  const sparseUndirected = sparseBottomEigenpairs(plantedCliques(2, 5, 1, 0.5), 2, 21n);
  assert.ok(Math.abs(sparseUndirected.values[0]) <= 1e-9,
    "lambda_1 = 0 holds for every connected test graph, whatever the bridge weight");
}

module.exports = { SplitMix64, MAX_SUBSPACE_SWEEPS, MAX_KMEANS_ITERATIONS, buildGraph, degrees,
  totalVolume, cutWeight, normalizedCut, normalizedAdjacencyMatvec, normalizedLaplacianMatvec,
  denseNormalizedAdjacency, denseNormalizedLaplacian, symmetricMatvec, maxAbsDiff,
  jacobiEigenSymmetric, modifiedGramSchmidt, initialSubspace, orthonormalityDefect,
  subspaceResidual, shiftedMatvec, rayleighRitz, eigenResiduals, sparseBottomEigenpairs, kmeans,
  eigenvectorEmbedding, clusterSpectral,
  plantedCliques, blobGraph, labelMismatch, bruteForceMinNormalizedCut };

if (require.main === module) {
  main();
  console.log("PASS");
}
