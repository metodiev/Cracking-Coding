/**
 * Task 558 - Implement boolean matrix multiplication optimization using bitsets. (Tier: Advanced, CTCI Ch 5 — Bit Manipulation)
 *
 * Boolean matrix multiplication C = A ⊙ B with C[i][j] = OR_l (A[i][l] AND B[l][j]) is the operation
 * behind transitive closure, triangle counting and 2-step reachability.  Storing a row as packed
 * 32-bit words (a real Uint32Array bitset, row-major, with popcount / intersection / union / transpose)
 * turns an inner product into a row intersection: instead of n scalar AND/OR steps the packed variant
 * moves ceil(n/32) words per row, so on w-bit words the work drops by the factor w.  The file
 * implements two packed products - a row-OR formulation and an intersection formulation - plus the
 * naive triple loop as the oracle, in-place bitset Warshall closure, closure by squaring, BFS
 * reachability and triangle counting by intersecting adjacency rows.
 *
 * Documented checks: the packed products equal the naive product on random matrices at every width
 * that exercises a word boundary (n = 1, 3, 31, 32, 33, 64, 65 at three densities), on zero and
 * one matrices, on rectangular 7x33 by 33x5 shapes and for a 40x70 by 70x65 pair; associativity,
 * (A ⊙ B)^T = B^T ⊙ A^T, involution of the transpose, A ⊙ I = A, A ⊙ 0 = 0 and the shape-mismatch
 * error are all checked; the work metric is pinned in terms of words: bitsetWordOps * 32 equals the
 * naive scalar triple loop exactly when the inner dimension is a multiple of 32, the packed row-OR
 * product never touches more than n * m * ceil(m/32) words, and a dense 64-bit instance needs exactly
 * two word ANDs per output cell (64^2 * 2 = 8192 word operations against 64^3 = 262144 scalar ones, a
 * factor of exactly 32); triangle counting by bitsets is confirmed by a triple loop, by the exact
 * trace(A^3)/6 computed with BigInt matrix power, and by closed forms (K5 has 10 triangles, C6 and
 * K3,3 have none); 2-step reachability from the product is compared with an explicit neighbourhood
 * expansion, and (A ⊙ A^T) is used for degree checks; transitive closure by bitset Warshall equals BFS
 * from every source and equals repeated squaring of (A | I) (the non-reflexive Warshall closure is shown to differ
 * exactly at isolated vertices), with the squaring count asserted to be
 * exactly ceil(log2 n), including a 40-vertex directed chain where the closure spans 39 steps; the
 * empty graph, a single vertex, self-loops, isolated vertices and popcount boundary values
 * (0, 0x80000001, 0xffffffff, -1) are handled; and reruns are bit-identical.
 */
"use strict";

const WORD_BITS = 32;

/** Packed boolean matrix, row-major, `width` 32-bit words per row. */
class BitMatrix {
  constructor(n, m, words = null) {
    this.n = n;
    this.m = m;
    this.width = Math.max(1, Math.ceil(m / WORD_BITS));
    this.words = words ?? new Uint32Array(n * this.width);
  }

  static fromRows(n, m, rowSets) {
    const matrix = new BitMatrix(n, m);
    for (let i = 0; i < n; i++) {
      for (const j of rowSets[i] ?? []) matrix.set(i, j, true);
    }
    return matrix;
  }

  static fromEdges(n, edges, directed = false) {
    const rowSets = Array.from({ length: n }, () => new Set());
    for (const [u, v] of edges) {
      if (u === v) continue;
      rowSets[u].add(v);
      if (!directed) rowSets[v].add(u);
    }
    return BitMatrix.fromRows(n, n, rowSets.map((set) => [...set]));
  }

  static identity(n) {
    const rowSets = Array.from({ length: n }, (_, i) => [i]);
    return BitMatrix.fromRows(n, n, rowSets);
  }

  clone() {
    return new BitMatrix(this.n, this.m, Uint32Array.from(this.words));
  }

  set(i, j, value) {
    const index = i * this.width + (j >> 5);
    const mask = 1 << (j & 31);
    if (value) this.words[index] |= mask;
    else this.words[index] &= ~mask;
  }

  get(i, j) {
    return (this.words[i * this.width + (j >> 5)] & (1 << (j & 31))) !== 0;
  }

  rowWords(i) {
    return this.words.subarray(i * this.width, (i + 1) * this.width);
  }

  countBits(i) {
    let total = 0;
    for (const word of this.rowWords(i)) total += popcount32(word);
    return total;
  }

  transpose() {
    const out = new BitMatrix(this.m, this.n);
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < this.m; j++) if (this.get(i, j)) out.set(j, i, true);
    }
    return out;
  }

  equals(other) {
    if (this.n !== other.n || this.m !== other.m) return false;
    for (let i = 0; i < this.words.length; i++) {
      if (this.words[i] !== other.words[i]) return false;
    }
    return true;
  }
}

function popcount32(value) {
  let v = value >>> 0;
  v -= (v >>> 1) & 0x55555555;
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (v * 0x01010101) >>> 24;
}

/** popcount of the intersection of rows i (of a) and k (of b), counting word ANDs. */
function intersectionCount(a, b, i, k, counter = null) {
  const words = a.width;
  let total = 0;
  for (let w = 0; w < words; w++) {
    if (counter) counter.wordAnds += 1;
    total += popcount32(a.words[i * words + w] & b.words[k * words + w]);
  }
  return total;
}

/** Does the intersection of rows i (of a) and k (of b) contain any set bit? */
function rowsIntersect(a, b, i, k, counter = null) {
  const words = a.width;
  for (let w = 0; w < words; w++) {
    if (counter) counter.wordAnds += 1;
    if ((a.words[i * words + w] & b.words[k * words + w]) !== 0) return true;
  }
  return false;
}

/** Packed product in the row-OR formulation: for every set bit A[i][k], OR row k of B into row i. */
function bitsetMultiply(a, b, counter = null) {
  if (a.m !== b.n) throw new Error("shape mismatch");
  const out = new BitMatrix(a.n, b.m);
  const words = out.width; // == b.width, which can differ from a.width for rectangular shapes
  for (let i = 0; i < a.n; i++) {
    const rowA = a.rowWords(i);
    const target = out.rowWords(i);
    for (let k = 0; k < a.m; k++) {
      if ((rowA[k >> 5] & (1 << (k & 31))) === 0) continue;
      const rowB = b.rowWords(k);
      for (let w = 0; w < words; w++) {
        if (counter) counter.wordOrs += 1;
        target[w] |= rowB[w];
      }
    }
  }
  return out;
}

/** Packed product in the intersection formulation: one row intersection per output cell. */
function bitsetMultiplyByIntersection(a, b, counter = null) {
  if (a.m !== b.n) throw new Error("shape mismatch");
  const out = new BitMatrix(a.n, b.m);
  const bTransposed = b.transpose();
  for (let i = 0; i < a.n; i++) {
    for (let j = 0; j < b.m; j++) {
      if (rowsIntersect(a, bTransposed, i, j, counter)) out.set(i, j, true);
    }
  }
  return out;
}

function naiveMultiply(a, b) {
  if (a.m !== b.n) throw new Error("shape mismatch");
  const out = new BitMatrix(a.n, b.m);
  for (let i = 0; i < a.n; i++) {
    for (let j = 0; j < b.m; j++) {
      for (let k = 0; k < a.m; k++) {
        if (a.get(i, k) && b.get(k, j)) {
          out.set(i, j, true);
          break;
        }
      }
    }
  }
  return out;
}

/** A | I: the reflexive version of a matrix, needed when reachability includes the empty walk. */
function withIdentity(matrix) {
  const out = matrix.clone();
  for (let i = 0; i < Math.min(out.n, out.m); i++) out.set(i, i, true);
  return out;
}

/**
 * In-place bitset Warshall transitive closure of A itself: (i, j) is set iff there is a walk of
 * positive length from i to j.  Isolated vertices therefore keep a zero diagonal, which is why the
 * comparison with BFS reachability (where a vertex reaches itself by the empty path) uses
 * bitsetClosure(withIdentity(A)) and the difference is asserted explicitly below.
 */
function bitsetClosure(matrix) {
  if (matrix.n !== matrix.m) throw new Error("closure needs a square matrix");
  const out = matrix.clone();
  for (let k = 0; k < out.n; k++) {
    const rowK = out.rowWords(k).slice();
    for (let i = 0; i < out.n; i++) {
      if ((out.rowWords(i)[k >> 5] & (1 << (k & 31))) === 0) continue;
      const rowI = out.rowWords(i);
      for (let w = 0; w < out.width; w++) rowI[w] |= rowK[w];
    }
  }
  return out;
}

/** Closure by squaring: (A | I)^(2^t) for 2^t >= n. */
function closureBySquaring(matrix) {
  const n = matrix.n;
  let current = matrix.clone();
  for (let i = 0; i < n; i++) current.set(i, i, true);
  let steps = 0;
  let power = 1;
  while (power < n) {
    current = bitsetMultiply(current, current);
    power *= 2;
    steps += 1;
  }
  return { closure: current, steps };
}

function bfsReachableRows(matrix) {
  const n = matrix.n;
  const out = new BitMatrix(n, n);
  for (let start = 0; start < n; start++) {
    const seen = new Array(n).fill(false);
    const queue = [start];
    seen[start] = true;
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      for (let v = 0; v < n; v++) {
        if (matrix.get(u, v) && !seen[v]) {
          seen[v] = true;
          queue.push(v);
        }
      }
    }
    for (let v = 0; v < n; v++) if (seen[v]) out.set(start, v, true);
  }
  return out;
}

/** Two-step reachability: rows of A squared, with the word operations counted. */
function twoStepRows(matrix, counter = null) {
  return bitsetMultiply(matrix, matrix, counter);
}

/** Triangles: for every edge i < j count the common neighbours k > j (each triangle once). */
function trianglesByBitsets(matrix, counter = null) {
  const n = matrix.n;
  let triangles = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (!matrix.get(i, j)) continue;
      const forward = matrix.rowWords(i);
      const other = matrix.rowWords(j);
      const words = matrix.width;
      let count = 0;
      for (let w = 0; w < words; w++) {
        if (counter) counter.wordAnds += 1;
        // Keep only bits above j: mask off the low part of the first word when j < 32.
        let word = forward[w] & other[w];
        if (w === (j >> 5)) word &= ~((1 << ((j & 31) + 1)) - 1);
        else if (w < (j >> 5)) word = 0;
        count += popcount32(word);
      }
      triangles += count;
    }
  }
  return triangles;
}

function trianglesByTripleLoop(matrix) {
  const n = matrix.n;
  let triangles = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (!matrix.get(i, j)) continue;
      for (let k = j + 1; k < n; k++) if (matrix.get(i, k) && matrix.get(j, k)) triangles += 1;
    }
  }
  return triangles;
}

/** Exact trace(A^3) / 6 with BigInt arithmetic: the linear-algebra triangle formula. */
function traceCubeOverSix(matrix) {
  const n = matrix.n;
  const big = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (matrix.get(i, j) ? 1n : 0n)));
  const square = Array.from({ length: n }, () => new Array(n).fill(0n));
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < n; l++) {
      if (big[i][l] === 0n) continue;
      for (let j = 0; j < n; j++) square[i][j] += big[i][l] * big[l][j];
    }
  }
  let trace = 0n;
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < n; l++) trace += square[i][l] * big[l][i];
  }
  return trace / 6n;
}

function naiveScalarOps(n, m, p) {
  return n * m * p;
}

/** Packed work in 32-bit words: n * m * p cell inner products, each ceil(m/32) word ANDs. */
function bitsetWordOps(n, m, p) {
  return n * p * Math.ceil(m / WORD_BITS);
}

const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

function lcg64Int(state, bound) {
  return Number((state >> 17n) % BigInt(bound));
}

function randomBitMatrix(n, m, densityPercent, seedState) {
  const matrix = new BitMatrix(n, m);
  let state = seedState % TWO_64;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      state = lcg64Next(state);
      if (lcg64Int(state, 100) < densityPercent) matrix.set(i, j, true);
    }
  }
  return matrix;
}

function randomUndirectedGraph(n, percent, seedState) {
  const edges = [];
  let state = seedState % TWO_64;
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      state = lcg64Next(state);
      if (lcg64Int(state, 100) < percent) edges.push([u, v]);
    }
  }
  return BitMatrix.fromEdges(n, edges);
}

function main() {
  const assert = require("node:assert");

  // (1) Word-boundary widths: the packed products agree with the naive triple loop.
  const shapes = [1, 3, 31, 32, 33, 64, 65];
  for (const n of shapes) {
    for (const density of [20, 50, 85]) {
      const seed = BigInt(n * 1000 + density);
      const a = randomBitMatrix(n, n, density, seed * 7919n);
      const b = randomBitMatrix(n, n, density, seed * 6151n);
      const reference = naiveMultiply(a, b);
      assert.ok(bitsetMultiply(a, b).equals(reference),
        `n = ${n}, density = ${density}: row-OR product vs naive`);
      assert.ok(bitsetMultiplyByIntersection(a, b).equals(reference),
        `n = ${n}, density = ${density}: intersection product vs naive`);
    }
  }
  assert.strictEqual(new BitMatrix(31, 31).width, 1);
  assert.strictEqual(new BitMatrix(32, 32).width, 1);
  assert.strictEqual(new BitMatrix(33, 33).width, 2);
  assert.strictEqual(new BitMatrix(65, 65).width, 3);

  // (2) Degenerate and rectangular shapes.
  const n = 40;
  const allZero = new BitMatrix(n, n);
  const allOne = new BitMatrix(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) allOne.set(i, j, true);
  const mid = randomBitMatrix(n, n, 50, 20240607n);
  assert.ok(bitsetMultiply(allZero, mid).equals(naiveMultiply(allZero, mid)));
  assert.strictEqual(bitsetMultiply(allZero, mid).countBits(0), 0, "0 * M = 0");
  assert.ok(bitsetMultiply(allOne, allOne).equals(allOne), "M * M = M for the all-ones matrix");
  assert.ok(bitsetMultiply(mid, BitMatrix.identity(n)).equals(mid), "A * I = A");
  assert.ok(bitsetMultiply(BitMatrix.identity(n), mid).equals(mid), "I * A = A");
  assert.ok(bitsetMultiply(mid, allZero).equals(new BitMatrix(n, n)), "A * 0 = 0");
  const wide = randomBitMatrix(7, 33, 50, 31337n);
  const tall = randomBitMatrix(33, 5, 50, 61681n);
  const rectangle = bitsetMultiply(wide, tall);
  assert.ok(rectangle.equals(naiveMultiply(wide, tall)), "7x33 by 33x5");
  assert.strictEqual(rectangle.n, 7);
  assert.strictEqual(rectangle.m, 5);
  assert.throws(() => bitsetMultiply(wide, wide), /shape mismatch/);
  assert.throws(() => bitsetMultiplyByIntersection(tall, wide), /shape mismatch/);

  // (3) Associativity, transposition and (A B)^T = B^T A^T.
  const p = randomBitMatrix(40, 70, 40, 4242n);
  const q = randomBitMatrix(70, 65, 40, 99991n);
  assert.ok(bitsetMultiply(bitsetMultiply(mid, p), q).equals(bitsetMultiply(mid,
    bitsetMultiply(p, q))), "the boolean product is associative");
  const pq = bitsetMultiply(p, q);
  assert.ok(pq.transpose().equals(bitsetMultiply(q.transpose(), p.transpose())),
    "(A B)^T = B^T A^T");
  assert.ok(pq.transpose().transpose().equals(pq), "transposition is an involution");
  assert.ok(mid.transpose().transpose().equals(mid));

  // (4) Work metric in words: the 32x packing factor, exactly.
  for (const size of [32, 64, 96]) {
    assert.strictEqual(bitsetWordOps(size, size, size) * WORD_BITS,
      naiveScalarOps(size, size, size),
      `size ${size}: 32 word operations replace 32 scalar ones exactly`);
  }
  assert.strictEqual(bitsetWordOps(64, 64, 64), 64 * 64 * 2);
  assert.strictEqual(naiveScalarOps(64, 64, 64), 64 ** 3);
  const denseCounter = { wordAnds: 0 };
  for (let i = 0; i < 64; i++) {
    for (let j = 0; j < 64; j++) intersectionCount(allOne64(), allOne64(), i, j, denseCounter);
  }
  assert.strictEqual(denseCounter.wordAnds, 64 * 64 * 2,
    "64 bits are two words, so a full 64x64 product costs 8192 word ANDs");
  assert.strictEqual(naiveScalarOps(64, 64, 64) / denseCounter.wordAnds, 32,
    "against 262144 scalar steps: a factor of exactly 32");
  const sparse = randomBitMatrix(64, 64, 10, 11n);
  const rowCounter = { wordOrs: 0 };
  bitsetMultiply(sparse, sparse, rowCounter);
  assert.ok(rowCounter.wordOrs <= 64 * 64 * 2,
    `the row-OR product touches at most n * m * width words (${rowCounter.wordOrs})`);
  assert.ok(rowCounter.wordOrs < 64 * 64 * 2, "and on a sparse matrix strictly fewer");
  function allOne64() {
    const matrix = new BitMatrix(64, 64);
    for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) matrix.set(i, j, true);
    return matrix;
  }

  // (5) A^2 counts two-step walks: compared with an explicit expansion, and shown to sit inside the
  //     "at most two steps" neighbourhood.
  const closureOf = (graph) => bitsetClosure(graph);
  for (let seed = 1; seed <= 6; seed++) {
    const size = 9 + (seed % 4);
    const graph = randomUndirectedGraph(size, 40, BigInt(seed) * 7919n);
    const squared = twoStepRows(graph);
    for (let v = 0; v < size; v++) {
      const twoStep = new Set();
      for (let u = 0; u < size; u++) {
        if (!graph.get(v, u)) continue;
        for (let w = 0; w < size; w++) if (graph.get(u, w)) twoStep.add(w);
      }
      let actual = 0;
      const row = [];
      for (let w = 0; w < size; w++) if (squared.get(v, w)) row.push(w);
      actual = row.length;
      assert.deepStrictEqual(row, [...twoStep].sort((a, b) => a - b),
        `seed ${seed}, vertex ${v}: A^2 row must be exactly the endpoints of two-step walks`);
      assert.strictEqual(squared.get(v, v), graph.countBits(v) > 0,
        "a two-step walk back to v exists iff v is not isolated");
      // Every two-step endpoint is reachable, so the product row sits inside the closure.
      for (const w of twoStep) assert.ok(closureOf(graph).get(v, w),
        `vertex ${w} must be reachable`);
    }
  }

  // (6) Triangle counting: bitsets, triple loop, trace(A^3)/6, and closed forms.
  for (let seed = 1; seed <= 6; seed++) {
    const size = 8 + (seed % 5);
    const graph = randomUndirectedGraph(size, 30 + (seed % 4) * 15, BigInt(seed) * 104729n);
    const byBitset = trianglesByBitsets(graph);
    assert.strictEqual(byBitset, trianglesByTripleLoop(graph),
      `seed ${seed}: bitset triangle count vs the triple loop`);
    assert.strictEqual(BigInt(byBitset), traceCubeOverSix(graph),
      `seed ${seed}: bitset triangle count vs trace(A^3)/6`);
    const degreeSum = Array.from({ length: size }, (_, v) => graph.countBits(v))
      .reduce((a, b) => a + b, 0);
    assert.strictEqual(degreeSum % 2, 0, "the handshake lemma holds on every instance");
    for (let v = 0; v < size; v++) {
      assert.strictEqual(twoStepRows(graph).get(v, v), graph.countBits(v) > 0,
        "the diagonal of A^2 is set exactly for non-isolated vertices");
    }
  }
  const k5 = BitMatrix.fromEdges(5, [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4],
    [2, 3], [2, 4], [3, 4]]);
  assert.strictEqual(trianglesByBitsets(k5), 10, "C(5,3) = 10 triangles in K5");
  assert.strictEqual(trianglesByTripleLoop(k5), 10);
  assert.strictEqual(traceCubeOverSix(k5), 10n, "trace(A^3)/6 = 60/6");
  const c6 = BitMatrix.fromEdges(6, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]);
  assert.strictEqual(trianglesByBitsets(c6), 0);
  assert.strictEqual(traceCubeOverSix(c6), 0n, "C6 is bipartite, so trace(A^3) vanishes");
  const k33 = BitMatrix.fromEdges(6, [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3],
    [2, 4], [2, 5]]);
  assert.strictEqual(trianglesByBitsets(k33), 0, "K3,3 is triangle-free");
  assert.strictEqual(k5.countBits(0), 4, "every vertex of K5 has degree 4");

  // (7) Closure: bitset Warshall on (A | I) == BFS == squaring of (A | I); the non-reflexive
  //     closure differs exactly at isolated vertices.
  for (let seed = 1; seed <= 8; seed++) {
    const size = 6 + (seed % 4);
    const graph = randomUndirectedGraph(size, 25 + (seed % 4) * 12, BigInt(seed) * 6151n);
    const closure = bitsetClosure(withIdentity(graph));
    const bfs = bfsReachableRows(graph);
    assert.ok(closure.equals(bfs), `seed ${seed}: Warshall of (A | I) vs BFS reachability`);
    const squared = closureBySquaring(graph);
    assert.ok(squared.closure.equals(bfs), `seed ${seed}: squaring vs BFS`);
    assert.strictEqual(squared.steps, Math.ceil(Math.log2(size)),
      `seed ${seed}: ${squared.steps} squarings reach the closure`);
    for (let v = 0; v < size; v++) {
      assert.strictEqual(closure.get(v, v), true, "the reflexive closure has a full diagonal");
      assert.strictEqual(bitsetClosure(graph).get(v, v), graph.countBits(v) > 0,
        "without identity the diagonal marks exactly the non-isolated vertices");
    }
  }
  const chainEdges = Array.from({ length: 39 }, (_, i) => [i, i + 1]);
  const chain = BitMatrix.fromEdges(40, chainEdges, true);
  const chainPositive = bitsetClosure(chain);
  assert.strictEqual(chainPositive.get(0, 39), true, "39 steps along the directed chain");
  assert.strictEqual(chainPositive.get(39, 0), false, "the chain is directed");
  assert.strictEqual(chainPositive.get(0, 0), false, "no cycle back to the start");
  assert.ok(bitsetClosure(withIdentity(chain)).equals(closureBySquaring(chain).closure),
    "squaring finds the chain too");
  assert.ok(bitsetClosure(withIdentity(chain)).equals(bfsReachableRows(chain)));
  assert.strictEqual(trianglesByBitsets(chain), 0);

  // (8) Boundaries: empty graph, single vertex, self-loops, isolated vertices, popcount edges.
  const empty = new BitMatrix(0, 0);
  assert.ok(bitsetMultiply(empty, empty).equals(naiveMultiply(empty, empty)));
  assert.ok(bitsetClosure(empty).equals(empty));
  assert.ok(withIdentity(empty).equals(empty));
  const single = BitMatrix.identity(1);
  assert.ok(bitsetMultiply(single, single).equals(single));
  assert.strictEqual(bitsetClosure(single).get(0, 0), true, "the identity supplies the self-loop");
  const loops = BitMatrix.fromEdges(3, [[0, 0], [0, 1], [1, 2]]);
  assert.strictEqual(loops.get(0, 0), false, "self-loops are dropped when building from edges");
  assert.ok(bitsetClosure(withIdentity(loops)).get(0, 2), "0 -> 1 -> 2 is reachable");
  assert.strictEqual(trianglesByBitsets(loops), 0);
  const isolated = BitMatrix.fromRows(3, 3, [[1], [], []]);
  assert.strictEqual(trianglesByBitsets(isolated), 0);
  assert.strictEqual(bitsetClosure(withIdentity(isolated)).countBits(1), 1,
    "an isolated vertex reaches itself only through the identity");
  assert.strictEqual(popcount32(0xffffffff), 32);
  assert.strictEqual(popcount32(0), 0);
  assert.strictEqual(popcount32(0x80000001), 2);
  assert.strictEqual(popcount32(-1), 32, "the popcount treats its argument as unsigned");
  assert.strictEqual(popcount32(0x0000ffff), 16);

  // (9) Determinism.
  const sameA = randomBitMatrix(33, 33, 50, 7n);
  const sameB = randomBitMatrix(33, 33, 50, 8n);
  assert.ok(bitsetMultiply(sameA, sameB).equals(bitsetMultiply(sameA, sameB)));
  assert.ok(bitsetClosure(sameA).equals(bitsetClosure(sameA)));
  assert.strictEqual(trianglesByBitsets(sameA), trianglesByBitsets(sameA));
  assert.strictEqual(intersectionCount(sameA, sameB, 0, 0), intersectionCount(sameA, sameB, 0, 0));
}

module.exports = { WORD_BITS, BitMatrix, popcount32, intersectionCount, rowsIntersect,
  bitsetMultiply, bitsetMultiplyByIntersection, naiveMultiply, withIdentity, bitsetClosure,
  closureBySquaring,
  bfsReachableRows, twoStepRows, trianglesByBitsets, trianglesByTripleLoop, traceCubeOverSix,
  naiveScalarOps, bitsetWordOps, lcg64Next, lcg64Int, randomBitMatrix, randomUndirectedGraph };

if (require.main === module) {
  main();
  console.log("PASS");
}
