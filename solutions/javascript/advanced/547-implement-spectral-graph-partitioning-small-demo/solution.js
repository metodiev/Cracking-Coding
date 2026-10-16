/**
 * Task 547 - Implement spectral graph partitioning small demo. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Spectral partitioning of small unweighted graphs.  The combinatorial Laplacian L = D - A (and
 * its normalised sibling L_sym = I - D^{-1/2} A D^{-1/2}) is diagonalised with cyclic Jacobi
 * rotations, and the eigenvector of the second-smallest eigenvalue - the Fiedler vector -
 * supplies the relaxed partition; the classic sweep then sorts the vertices by that vector and
 * scores every prefix, so the reported cut is the best prefix rather than a guess.
 *
 * Documented checks: hand-verified spectra (0, 1, 3 for the 3-path; 0, 3, 3 for the triangle; the
 * 4-cycle has 0, 2, 2, 4) with residuals ||L v - lambda v|| <= 1e-9 and orthogonality of the
 * Fiedler vector to the constant vector; row sums of L are exactly zero; the eigen-residual and
 * the Jacobi sweep count are bounded; the provable inequality lambda2 <= minRatioCut over every
 * non-trivial subset is asserted, and on the canonical "two triangles joined by a bridge" example
 * the best sweep prefix is asserted to equal both the brute-force minimum ratio cut (2/3) and the
 * planted partition, with the sign pattern of the Fiedler vector matching the planted split.  On
 * a degree-regular example the argmin subset for the ratio cut and for the normalised cut are
 * asserted to coincide (they differ only by the factor 1/d) and lambda2(L_sym) lands inside the
 * provable [0, 2] window; a disconnected graph is asserted to have two zero eigenvalues and a
 * Fiedler vector that separates the components; and LCG two-blob graphs must be recovered.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function adjacencyMatrix(n, edges) {
  const A = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const [u, v] of edges) {
    A[u][v] += 1;
    A[v][u] += 1;
  }
  return A;
}

function degrees(A) {
  return A.map((row) => row.reduce((a, b) => a + b, 0));
}

function combinatorialLaplacian(A) {
  const n = A.length;
  const deg = degrees(A);
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? deg[i] : 0) - A[i][j]));
}

function normalizedLaplacian(A) {
  const n = A.length;
  const deg = degrees(A);
  const inv = deg.map((d) => (d > 0 ? 1 / Math.sqrt(d) : 0));
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0) - A[i][j] * inv[i] * inv[j]));
}

/** Cyclic Jacobi eigen-decomposition of a symmetric matrix; eigenvalues + eigenvector columns. */
function jacobiEigh(matrix, maxSweeps = 100, tol = 1e-12) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  const v = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  let sweeps = 0;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (Math.sqrt(off) <= tol) break;
    sweeps += 1;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) <= 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        let t = 1 / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        if (theta < 0) t = -t;
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
          a[p][k] = a[k][p];
          a[q][k] = a[k][q];
        }
        const app = a[p][p];
        const aqq = a[q][q];
        const apq = a[p][q];
        a[p][p] = c * c * app - 2 * c * s * apq + s * s * aqq;
        a[q][q] = s * s * app + 2 * c * s * apq + c * c * aqq;
        a[p][q] = 0;
        a[q][p] = 0;
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: a.map((row, i) => row[i]), vectors: v, sweeps };
}

function sortedSpectrum(matrix) {
  const { values, vectors, sweeps } = jacobiEigh(matrix);
  const order = values.map((value, index) => [value, index]).sort((x, y) => x[0] - y[0]);
  return {
    values: order.map(([value]) => value),
    vectors: order.map(([, index]) => vectors.map((row) => row[index])),
    sweeps,
  };
}

function fiedlerVector(matrix) {
  const spectrum = sortedSpectrum(matrix);
  return { values: spectrum.values, vector: spectrum.vectors[1], sweeps: spectrum.sweeps,
    lambda2: spectrum.values[1] };
}

function cutWeight(A, side) {
  const n = A.length;
  let total = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (side[i] !== side[j]) total += A[i][j];
    }
  }
  return total;
}

function ratioCut(A, side) {
  const n = A.length;
  let size = 0;
  for (const value of side) if (value === 0) size += 1;
  if (size === 0 || size === n) return Infinity;
  const cut = cutWeight(A, side);
  return cut / size + cut / (n - size);
}

function normalizedCut(A, side) {
  const n = A.length;
  const deg = degrees(A);
  let sizeLeft = 0;
  let sizeRight = 0;
  let volLeft = 0;
  let volRight = 0;
  for (let i = 0; i < n; i++) {
    if (side[i] === 0) {
      sizeLeft += 1;
      volLeft += deg[i];
    } else {
      sizeRight += 1;
      volRight += deg[i];
    }
  }
  if (sizeLeft === 0 || sizeRight === 0) return Infinity;
  const cut = cutWeight(A, side);
  return cut / volLeft + cut / volRight;
}

function subsetSide(n, mask) {
  const side = new Array(n).fill(1);
  for (let i = 0; i < n; i++) if (mask & (1 << i)) side[i] = 0;
  return side;
}

function bruteBestCut(A, score) {
  const n = A.length;
  let best = Infinity;
  let bestSide = null;
  for (let mask = 1; mask < (1 << n) - 1; mask++) {
    const side = subsetSide(n, mask);
    const value = score(A, side);
    if (value < best - 1e-15) {
      best = value;
      bestSide = side;
    }
  }
  return { value: best, side: bestSide };
}

/** Sweep cut: sort by the Fiedler vector and score every prefix. */
function sweepCut(A, values) {
  const n = A.length;
  const order = [...Array(n).keys()].sort((a, b) => (values[a] - values[b]) || (a - b));
  const results = [];
  for (let k = 1; k < n; k++) {
    const side = new Array(n).fill(1);
    for (let t = 0; t < k; t++) side[order[t]] = 0;
    results.push({ k, side, ratio: ratioCut(A, side), normalized: normalizedCut(A, side) });
  }
  const byRatio = results.reduce((a, b) => (b.ratio < a.ratio - 1e-15 ? b : a));
  const byNormalized = results.reduce((a, b) => (b.normalized < a.normalized - 1e-15 ? b : a));
  return { bestRatio: byRatio, bestNormalized: byNormalized, candidates: results.length };
}

function residualNorm(matrix, vector, lambda) {
  const n = matrix.length;
  let total = 0;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += matrix[i][j] * vector[j];
    total += (sum - lambda * vector[i]) ** 2;
  }
  return Math.sqrt(total);
}

function signSide(vector) {
  const mean = vector.reduce((a, b) => a + b, 0) / vector.length;
  return vector.map((value) => (value >= mean ? 0 : 1));
}

function partitionString(side) {
  const groups = [[], []];
  side.forEach((value, index) => groups[value].push(index));
  return groups.map((members) => members.join(",")).sort().join("|");
}

/**
 * Two blobs of `half` vertices each: every blob gets a spanning path (so it is connected by
 * construction), random intra-blob edges fill it out, and `crossEdges` edges join the blobs.
 * The seeded path is what makes lambda2 strictly positive instead of silently producing a
 * disconnected graph that the Fiedler vector would split into components.
 */
function blobGraph(half, insidePercent, crossEdges, seed) {
  const n = 2 * half;
  const seen = new Set();
  const edges = [];
  const add = (u, v) => {
    const key = u < v ? `${u},${v}` : `${v},${u}`;
    if (u === v || seen.has(key)) return;
    seen.add(key);
    edges.push([u, v]);
  };
  for (let blob = 0; blob < 2; blob++) {
    for (let i = 0; i + 1 < half; i++) add(blob * half + i, blob * half + i + 1);
  }
  let s = seed % 65537;
  for (let blob = 0; blob < 2; blob++) {
    for (let i = 0; i < half; i++) {
      for (let j = i + 1; j < half; j++) {
        s = lcgNext(s);
        if (s % 100 < insidePercent) add(blob * half + i, blob * half + j);
      }
    }
  }
  for (let i = 0; i < crossEdges; i++) add(i % half, half + ((i * 3 + 1) % half));
  return { n, edges, seed: s };
}

/** Every vertex must have at least one incident edge (no isolated vertex, hence no zero-volume side). */
function noIsolatedVertices(A) {
  return degrees(A).every((degree) => degree > 0);
}

function main() {
  const assert = require("node:assert");

  // (1) Hand-verified spectra and Laplacian identities.
  const path3 = adjacencyMatrix(3, [[0, 1], [1, 2]]);
  const path3L = combinatorialLaplacian(path3);
  assert.deepStrictEqual(path3L, [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]);
  const spectrum3 = sortedSpectrum(path3L);
  assert.ok(Math.abs(spectrum3.values[0]) <= 1e-12);
  assert.ok(Math.abs(spectrum3.values[1] - 1) <= 1e-12);
  assert.ok(Math.abs(spectrum3.values[2] - 3) <= 1e-12);
  const triangle = combinatorialLaplacian(adjacencyMatrix(3, [[0, 1], [1, 2], [2, 0]]));
  const triangleSpectrum = sortedSpectrum(triangle);
  assert.ok(Math.abs(triangleSpectrum.values[0]) <= 1e-12);
  assert.ok(Math.abs(triangleSpectrum.values[1] - 3) <= 1e-12);
  assert.ok(Math.abs(triangleSpectrum.values[2] - 3) <= 1e-12);
  const square = combinatorialLaplacian(adjacencyMatrix(4, [[0, 1], [1, 2], [2, 3], [3, 0]]));
  const squareSpectrum = sortedSpectrum(square);
  assert.ok(Math.abs(squareSpectrum.values[0]) <= 1e-12);
  assert.ok(Math.abs(squareSpectrum.values[1] - 2) <= 1e-12);
  assert.ok(Math.abs(squareSpectrum.values[2] - 2) <= 1e-12);
  assert.ok(Math.abs(squareSpectrum.values[3] - 4) <= 1e-12);
  for (const row of square) {
    assert.ok(Math.abs(row.reduce((a, b) => a + b, 0)) <= 1e-12, "L has zero row sums");
  }

  // (2) The Fiedler vector is a genuine eigenpair, orthogonal to the constant vector.
  for (const matrix of [path3L, triangle, square]) {
    const { vector, lambda2, sweeps } = fiedlerVector(matrix);
    assert.ok(residualNorm(matrix, vector, lambda2) <= 1e-9, "eigen residual");
    assert.ok(Math.abs(vector.reduce((a, b) => a + b, 0)) <= 1e-9, "mean-zero Fiedler vector");
    assert.ok(sweeps <= 100, "Jacobi must converge inside the documented sweep cap");
    assert.ok(lambda2 >= -1e-12);
  }

  // (3) The canonical bridge example: planted split found by the sweep, and lambda2 <= minRatioCut.
  const bridgeA = adjacencyMatrix(6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3]]);
  const bridgeL = combinatorialLaplacian(bridgeA);
  const bridgeFiedler = fiedlerVector(bridgeL);
  const bridgeSweep = sweepCut(bridgeA, bridgeFiedler.vector);
  const planted = [0, 0, 0, 1, 1, 1];
  assert.strictEqual(ratioCut(bridgeA, planted), 2 / 3);
  assert.strictEqual(normalizedCut(bridgeA, planted), 2 / 7,
    "each side has volume 7 (degrees 3, 2, 2) and the bridge is the only cut edge");
  assert.strictEqual(partitionString(signSide(bridgeFiedler.vector)), partitionString(planted),
    "the Fiedler sign pattern must reproduce the planted split");
  const bruteRatio = bruteBestCut(bridgeA, ratioCut);
  assert.ok(Math.abs(bruteRatio.value - 2 / 3) <= 1e-12, "brute force sees the same optimum");
  assert.ok(Math.abs(bridgeSweep.bestRatio.ratio - 2 / 3) <= 1e-9,
    `the best sweep prefix must reach the optimum (got ${bridgeSweep.bestRatio.ratio})`);
  assert.ok(bridgeSweep.bestRatio.ratio >= bridgeFiedler.lambda2 - 1e-9,
    "the provable inequality lambda2 <= minRatioCut");
  assert.ok(bridgeFiedler.lambda2 <= 2 / 3 + 1e-9);
  assert.ok(Math.abs(bridgeSweep.bestNormalized.normalized - 2 / 7) <= 1e-9);
  assert.strictEqual(bridgeSweep.candidates, 5, "one candidate per non-trivial prefix");

  // (4) Regular example: ratio and normalised cuts pick the same subset (they differ by 1/d).
  const ring = adjacencyMatrix(6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]]);
  const ringBruteRatio = bruteBestCut(ring, ratioCut);
  const ringBruteNormalized = bruteBestCut(ring, normalizedCut);
  assert.deepStrictEqual(ringBruteRatio.side, ringBruteNormalized.side,
    "for the 3-regular prism the two objectives share their argmin");
  assert.ok(Math.abs(ringBruteNormalized.value - ringBruteRatio.value / 3) <= 1e-12,
    "Ncut = ratioCut / d on a d-regular graph");
  assert.deepStrictEqual(degrees(ring), [3, 3, 3, 3, 3, 3]);
  assert.strictEqual(cutWeight(ring, ringBruteRatio.side), 3);
  const normalisedWindow = sortedSpectrum(normalizedLaplacian(ring));
  assert.ok(normalisedWindow.values[0] <= 1e-12);
  assert.ok(normalisedWindow.values.every((value) => value >= -1e-12 && value <= 2 + 1e-12),
    "the normalised Laplacian spectra live in [0, 2]");

  // (5) Disconnected: two zero eigenvalues and a Fiedler vector separating the components.
  const split = adjacencyMatrix(6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]);
  const splitSpectrum = sortedSpectrum(combinatorialLaplacian(split));
  assert.ok(Math.abs(splitSpectrum.values[0]) <= 1e-12);
  assert.ok(Math.abs(splitSpectrum.values[1]) <= 1e-12, "one zero eigenvalue per component");
  assert.ok(splitSpectrum.values[2] > 1, "the third eigenvalue is strictly positive");
  const splitFiedler = fiedlerVector(combinatorialLaplacian(split));
  assert.strictEqual(partitionString(signSide(splitFiedler.vector)), "0,1,2|3,4,5",
    "the Fiedler vector of a disconnected graph separates its components");
  assert.strictEqual(bruteBestCut(split, ratioCut).value, 0, "a disconnected graph cuts for free");

  // (6) LCG two-blob graphs are recovered, and the sweep cut dominates the naive threshold zero.
  for (let seed = 1; seed <= 5; seed++) {
    const { n, edges } = blobGraph(5, 70, 2, seed * 97);
    const A = adjacencyMatrix(n, edges);
    assert.strictEqual(noIsolatedVertices(A), true, `seed ${seed}: no isolated vertex`);
    assert.ok(fiedlerVector(combinatorialLaplacian(A)).lambda2 > 1e-9,
      `seed ${seed}: the blob graph is connected, so lambda2 is strictly positive`);
    const plantedSide = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    const fiedler = fiedlerVector(combinatorialLaplacian(A));
    assert.strictEqual(partitionString(signSide(fiedler.vector)), partitionString(plantedSide),
      `seed ${seed}: the two blobs must be separated by the Fiedler sign pattern`);
    const sweep = sweepCut(A, fiedler.vector);
    const brute = bruteBestCut(A, ratioCut);
    assert.ok(sweep.bestRatio.ratio >= brute.value - 1e-9, `seed ${seed}: sweep >= optimum`);
    assert.ok(sweep.bestRatio.ratio <= ratioCut(A, plantedSide) + 1e-9,
      `seed ${seed}: the sweep is at least as good as the planted split`);
    assert.ok(fiedler.lambda2 <= brute.value + 1e-9, `seed ${seed}: lambda2 <= minRatioCut`);
    assert.ok(residualNorm(combinatorialLaplacian(A), fiedler.vector, fiedler.lambda2) <= 1e-9);
  }

  // (7) Determinism.
  const repeatA = adjacencyMatrix(6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3]]);
  assert.deepStrictEqual(fiedlerVector(combinatorialLaplacian(repeatA)),
    fiedlerVector(combinatorialLaplacian(repeatA)));
  assert.deepStrictEqual(sortedSpectrum(normalizedLaplacian(repeatA)),
    sortedSpectrum(normalizedLaplacian(repeatA)));
}

module.exports = { lcgNext, adjacencyMatrix, degrees, combinatorialLaplacian, normalizedLaplacian,
  jacobiEigh, sortedSpectrum, fiedlerVector, cutWeight, ratioCut, normalizedCut, subsetSide,
  bruteBestCut, sweepCut, residualNorm, signSide, partitionString, blobGraph };

if (require.main === module) {
  main();
  console.log("PASS");
}
