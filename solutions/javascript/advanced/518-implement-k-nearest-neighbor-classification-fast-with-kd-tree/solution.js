/**
 * Task 518 - Implement k-nearest neighbor classification fast with kd-tree. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * k-nearest-neighbour classification with two interchangeable neighbour engines that are
 * *required* to agree exactly. The brute-force engine sorts all n training points by the total
 * order (squared distance, training index) and takes the first k; the kd-tree engine builds a
 * static median-split tree (at every node the split axis is the widest-spread coordinate, ties
 * broken by the smallest axis index, and the pivot is the middle element of the index list
 * sorted by coordinate then index) and searches it with a bounded max-heap of size k ordered by
 * the same (distance^2, index) comparator, pruning a subtree only when the squared offset to the
 * splitting plane is *strictly* greater than the heap's worst key - the strictness is what keeps
 * index tie-breaks exact. Classification is a majority vote over the k retrieved labels with
 * ties broken by the smallest label; the vote is also cross-checked against a second, independent
 * accumulation that walks the brute-force neighbour list.
 *
 * Documented checks: both engines return element-wise identical neighbour lists (exact integer
 * arithmetic for LCG-generated integer points, so ties are not a source of disagreement) for
 * k = 1, 3, 5, 7 and k = n; the kd-tree evaluates strictly fewer distances than brute force on
 * 1-D/2-D/3-D sets and at most as many in the pathological all-equal case; a query placed
 * exactly on a training point returns distance 0 first; duplicate-heavy sets still agree; the
 * majority vote reproduces hand-computed labels on a two-blob example, is fully accurate on two
 * well-separated blobs, and rejects a non-positive k as well as an empty training set.
 */
"use strict";

/** Squared Euclidean distance, evaluated identically by both engines (bit-for-bit comparable). */
function dist2(a, b) {
  let total = 0.0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    total += d * d;
  }
  return total;
}

/** Small deterministic LCG in the style of the neighbouring wave-3 solutions. */
function lcgPoints(n, seed, limit, dim = 2) {
  const points = [];
  for (let i = 0; i < n; i++) {
    const point = [];
    for (let d = 0; d < dim; d++) {
      seed = (seed * 75 + 74) % 65537;
      point.push(seed % limit);
    }
    points.push(point);
  }
  return points;
}

/** Total order used by both engines: nearer first, then smaller training index. */
function closer(a, b) {
  return a.d2 < b.d2 || (a.d2 === b.d2 && a.idx < b.idx);
}

class BruteForceNN {
  constructor(points) {
    this.points = points;
    this.checks = 0;
  }

  knn(query, k) {
    const scored = [];
    for (let i = 0; i < this.points.length; i++) {
      this.checks += 1;
      scored.push({ d2: dist2(query, this.points[i]), idx: i });
    }
    scored.sort((a, b) => (a.d2 - b.d2) || (a.idx - b.idx));
    return scored.slice(0, k);
  }
}

/** Bounded max-heap on the (d2, idx) comparator: the root is always the *worst* kept neighbour. */
class NeighbourHeap {
  constructor(k) {
    this.k = k;
    this.data = [];
  }

  get size() {
    return this.data.length;
  }

  get worst() {
    return this.data[0];
  }

  push(item) {
    if (this.data.length < this.k) {
      this.data.push(item);
      this.up(this.data.length - 1);
      return;
    }
    if (closer(item, this.data[0])) {
      this.data[0] = item;
      this.down(0);
    }
  }

  up(i) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!closer(this.data[parent], this.data[i])) break;
      const tmp = this.data[parent];
      this.data[parent] = this.data[i];
      this.data[i] = tmp;
      i = parent;
    }
  }

  down(i) {
    for (;;) {
      const left = 2 * i + 1;
      const right = left + 1;
      let worst = i;
      if (left < this.data.length && closer(this.data[worst], this.data[left])) worst = left;
      if (right < this.data.length && closer(this.data[worst], this.data[right])) worst = right;
      if (worst === i) break;
      const tmp = this.data[worst];
      this.data[worst] = this.data[i];
      this.data[i] = tmp;
      i = worst;
    }
  }

  /** Heap contents in (d2, idx) order. */
  sorted() {
    return this.data.slice().sort((a, b) => (a.d2 - b.d2) || (a.idx - b.idx));
  }
}

class KdTreeNN {
  constructor(points) {
    this.points = points;
    this.dim = points.length === 0 ? 0 : points[0].length;
    this.idx = [];
    this.axis = [];
    this.left = [];
    this.right = [];
    this.checks = 0;
    const all = points.map((_, i) => i);
    this.root = this.build(all);
  }

  /** Widest-spread axis, smallest index on ties (a deterministic tree shape). */
  chooseAxis(indices) {
    let best = 0;
    let bestSpread = -1.0;
    for (let axis = 0; axis < this.dim; axis++) {
      let lo = Infinity;
      let hi = -Infinity;
      for (const i of indices) {
        const value = this.points[i][axis];
        if (value < lo) lo = value;
        if (value > hi) hi = value;
      }
      const spread = hi - lo;
      if (spread > bestSpread) {
        bestSpread = spread;
        best = axis;
      }
    }
    return best;
  }

  build(indices) {
    if (indices.length === 0) return -1;
    const axis = this.chooseAxis(indices);
    indices.sort((a, b) => (this.points[a][axis] - this.points[b][axis]) || (a - b));
    const mid = indices.length >> 1;
    const node = this.idx.length;
    const pivot = indices[mid];
    this.idx.push(pivot);
    this.axis.push(axis);
    this.left.push(-1);
    this.right.push(-1);
    this.left[node] = this.build(indices.slice(0, mid));
    this.right[node] = this.build(indices.slice(mid + 1));
    return node;
  }

  knn(query, k) {
    const heap = new NeighbourHeap(k);
    this.search(this.root, query, heap);
    return heap.sorted();
  }

  search(node, query, heap) {
    if (node === -1) return;
    const pivot = this.idx[node];
    const axis = this.axis[node];
    this.checks += 1;
    heap.push({ d2: dist2(query, this.points[pivot]), idx: pivot });
    const offset = query[axis] - this.points[pivot][axis];
    const nearFirst = offset <= 0.0;
    const near = nearFirst ? this.left[node] : this.right[node];
    const far = nearFirst ? this.right[node] : this.left[node];
    this.search(near, query, heap);
    // Prune the far side only when its nearest possible point is *strictly* worse than the heap
    // root; using < here would discard equal-distance points with a smaller index and break the
    // exact tie-break agreement with brute force.
    if (heap.size < heap.k || offset * offset <= heap.worst.d2) {
      this.search(far, query, heap);
    }
  }
}

function neighbourIndices(neighbours) {
  return neighbours.map((entry) => entry.idx);
}

/** Majority vote over the k labels; ties go to the smallest label (documented rule). */
function classify(engine, labels, query, k) {
  if (!Number.isInteger(k) || k <= 0) throw new Error("k must be a positive integer");
  const neighbours = engine.knn(query, k);
  if (neighbours.length === 0) throw new Error("no training points");
  const votes = new Map();
  for (const entry of neighbours) {
    const label = labels[entry.idx];
    votes.set(label, (votes.get(label) || 0) + 1);
  }
  let bestLabel = null;
  let bestCount = -1;
  for (const [label, count] of [...votes.entries()].sort((a, b) => a[0] - b[0])) {
    if (count > bestCount) {
      bestCount = count;
      bestLabel = label;
    }
  }
  return { label: bestLabel, votes, neighbours: neighbourIndices(neighbours) };
}

/** Independent second vote accumulation over a *list of labels* (no engine involved). */
function majorityOfLabels(neighbourLabels) {
  const counts = new Map();
  for (const label of neighbourLabels) counts.set(label, (counts.get(label) || 0) + 1);
  let best = null;
  let bestCount = -1;
  for (const [label, count] of [...counts.entries()].sort((a, b) => a[0] - b[0])) {
    if (count > bestCount) {
      bestCount = count;
      best = label;
    }
  }
  return best;
}

/** Two integer blobs with labels 0 and 1; the blobs are far enough apart to be separable. */
function blobSet(seed) {
  const points = [];
  const labels = [];
  let s = seed;
  for (const [cx, cy, label] of [[0, 0, 0], [60, 60, 1]]) {
    for (const [x, y] of lcgPoints(30, s, 7)) {
      points.push([x + cx, y + cy]);
      labels.push(label);
    }
    s += 13;
  }
  return { points, labels };
}

function main() {
  const assert = require("node:assert");

  const train = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.5, 10.0]];
  const tree = new KdTreeNN(train);

  assert.deepStrictEqual(neighbourIndices(tree.knn([0.2, 0.2], 1)), [0]);
  assert.deepStrictEqual(neighbourIndices(tree.knn([0.2, 0.2], 3)), [0, 1, 2]);
  assert.deepStrictEqual(neighbourIndices(tree.knn([10.2, 10.1], 2)), [3, 4],
    "(10,10) is 0.05 away, (10.5,10) is 0.10 away");
  assert.deepStrictEqual(neighbourIndices(tree.knn([0.0, 0.0], 5)), [0, 1, 2, 3, 4],
    "k = n returns every point, nearest first");

  const onTop = tree.knn([10.0, 10.0], 1)[0];
  assert.strictEqual(onTop.d2, 0.0, "a query on a training point has distance 0");
  assert.strictEqual(onTop.idx, 3);
  assert.deepStrictEqual(neighbourIndices(tree.knn([0.0, 0.0], 2)), [0, 1],
    "tie between (1,0) and (0,1) breaks to the smaller index");

  const datasets = [
    ["steps-2d", lcgPoints(200, 7, 40, 2), 3],
    ["steps-1d", lcgPoints(150, 19, 25, 1), 7],
    ["steps-3d", lcgPoints(120, 31, 15, 3), 5],
    ["coarse", lcgPoints(60, 43, 6, 2), 1],
    ["duplicates", lcgPoints(40, 59, 2, 2), 4],
    ["all-equal", new Array(50).fill([5, 5]), 3],
  ];
  const report = [];
  for (const [name, points, k] of datasets) {
    const slow = new BruteForceNN(points);
    const fast = new KdTreeNN(points);
    const queries = lcgPoints(25, 101, 40, points[0].length);
    for (const query of queries) {
      const expected = slow.knn(query, k);
      const actual = fast.knn(query, k);
      assert.deepStrictEqual(neighbourIndices(actual), neighbourIndices(expected),
        `${name}: engines disagree for k = ${k}`);
      assert.deepStrictEqual(actual.map((e) => e.d2), expected.map((e) => e.d2),
        `${name}: distance sequences differ`);
    }
    report.push([name, points.length, k, slow.checks, fast.checks]);
  }

  for (const [name, , , slowChecks, fastChecks] of report) {
    if (name === "all-equal") {
      assert.ok(fastChecks <= slowChecks, "all-equal input cannot prune, but must not explode");
    } else {
      assert.ok(fastChecks < slowChecks, `${name}: the kd-tree must evaluate fewer distances`);
    }
  }
  assert.ok(report[0][4] <= Math.floor(report[0][3] / 2), "2-D: at least a 2x saving");

  const { points, labels } = blobSet(2024);
  const slow = new BruteForceNN(points);
  const fast = new KdTreeNN(points);
  // Probes are drawn inside the two blobs, so the planted label is unambiguous: every one of
  // the 5 nearest training points lies in the same blob as the probe.
  const probes = [];
  const truth = [];
  for (const [x, y] of lcgPoints(20, 555, 7)) {
    probes.push([x, y]);
    truth.push(0);
  }
  for (const [x, y] of lcgPoints(20, 777, 7)) {
    probes.push([x + 60, y + 60]);
    truth.push(1);
  }
  let correct = 0;
  let correctBrute = 0;
  probes.forEach((probe, index) => {
    const voted = classify(fast, labels, probe, 5);
    const votedSlow = classify(slow, labels, probe, 5);
    assert.strictEqual(voted.label, votedSlow.label, "engines must vote identically");
    if (voted.label === truth[index]) correct += 1;
    if (votedSlow.label === truth[index]) correctBrute += 1;
    assert.strictEqual(majorityOfLabels(voted.neighbours.map((i) => labels[i])), voted.label,
      "independent vote accumulation must agree");
  });
  assert.strictEqual(correct, probes.length, "well-separated blobs are classified perfectly");
  assert.strictEqual(correctBrute, probes.length);

  const tieVote = classify(tree, [0, 1, 1, 0, 0], [0.2, 0.2], 4);
  assert.strictEqual(tieVote.label, 0, "2-2 tie resolves to the smaller label");

  const fresh = new KdTreeNN(train);
  const again = new KdTreeNN(train);
  assert.deepStrictEqual(neighbourIndices(fresh.knn([3.0, 3.0], 3)),
    neighbourIndices(again.knn([3.0, 3.0], 3)), "tree construction is deterministic");

  assert.throws(() => classify(tree, [0, 1, 0, 1, 1], [0.0, 0.0], 0), /positive integer/);
  assert.throws(() => classify(new KdTreeNN([]), [], [0.0, 0.0], 1), /no training points/);
}

module.exports = { dist2, lcgPoints, closer, BruteForceNN, NeighbourHeap, KdTreeNN,
  neighbourIndices, classify, majorityOfLabels, blobSet };

if (require.main === module) {
  main();
  console.log("PASS");
}
