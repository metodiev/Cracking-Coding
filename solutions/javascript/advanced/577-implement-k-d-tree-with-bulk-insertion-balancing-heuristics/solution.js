/**
 * Task 577 - Implement k-d tree with bulk insertion & balancing heuristics. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A k-d tree with three operations of interest: a balanced build (split on the axis with the largest
 * spread, at the median, alternating on ties), exact k-nearest-neighbour search with a bounded
 * candidate list, and box range counting/search.  On top of that sits the *bulk insertion* heuristic
 * the task asks for: new points go into a pending buffer, and the tree is rebuilt from scratch (base
 * points plus buffer) as soon as the buffer exceeds max(8, size / 2) - the classic "rebuild when the
 * log structure doubles" policy.  A plain leaf-insertion tree is also provided so the balancing
 * heuristic can be *measured* rather than asserted from folklore.
 *
 * Documented checks: nearest neighbour, k-nearest and box range queries are compared against brute
 * force on 2-, 3-, 5- and 8-dimensional LCG data, including queries far outside the data, queries
 * that coincide with a stored point (distance exactly 0) and duplicate points (the smallest index
 * wins ties); every node is verified to separate its subtree (left coordinates <= the split value <=
 * right coordinates on the node's axis, checked recursively); the bulk policy is asserted to keep the
 * buffer inside its threshold after every batch and to rebuild a logarithmic number of times
 * (measured: 2 rebuilds for 12 batches of 250 points into a 1000-point base, ending with a 2750-point
 * base and a 1250-point buffer under the 1375-point limit), with query correctness re-checked while
 * the buffer is non-empty; the depth after bulk loading stays within 2 log2(n) + 2 (measured 12 for
 * n = 4000) whereas inserting the same sorted 512-point set one point at a time degenerates into a
 * 512-deep chain, on which a corner query visits all 512 nodes against 9 for the balanced tree - and
 * the documented caveat that a query *far outside* the data visits ~416 nodes in both shapes, so
 * balancing removes the pathological case rather than making every query cheap; empty trees, a
 * single-point tree, mismatched dimensions, inverted boxes and boxes that contain everything are all
 * exercised; the average visited-node count over 300 random queries stays well below n in every
 * dimension; and the structure is deterministic because all randomness comes from a documented
 * 64-bit splitmix64 generator.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;

/** Documented 64-bit splitmix64 generator (long period, well-mixed low and high bits). */
class Rng {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    const z = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    this.state = z;
    let x = z;
    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return x ^ (x >> 31n);
  }

  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992.0;
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

function distance(a, b) {
  let total = 0.0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    total += d * d;
  }
  return Math.sqrt(total);
}

function spreadAxis(points, dims) {
  const min = new Array(dims).fill(Infinity);
  const max = new Array(dims).fill(-Infinity);
  for (const point of points) {
    for (let d = 0; d < dims; d++) {
      if (point[d] < min[d]) min[d] = point[d];
      if (point[d] > max[d]) max[d] = point[d];
    }
  }
  let axis = 0;
  let best = -Infinity;
  for (let d = 0; d < dims; d++) {
    const range = max[d] - min[d];
    if (range > best) {
      best = range;
      axis = d;
    }
  }
  return axis;
}

/** Balanced build: split at the median of the widest axis. */
function buildBalanced(entries, dims, depth = 0) {
  if (entries.length === 0) return null;
  const axis = entries.length > 1 ? spreadAxis(entries.map((e) => e.point), dims) : (depth % dims);
  const sorted = entries.slice().sort((a, b) => a.point[axis] - b.point[axis] || a.index - b.index);
  const mid = sorted.length >> 1;
  const node = { point: sorted[mid].point, index: sorted[mid].index, axis, left: null, right: null };
  node.left = buildBalanced(sorted.slice(0, mid), dims, depth + 1);
  node.right = buildBalanced(sorted.slice(mid + 1), dims, depth + 1);
  return node;
}

function treeDepth(node) {
  if (node === null) return 0;
  return 1 + Math.max(treeDepth(node.left), treeDepth(node.right));
}

function treeSize(node) {
  if (node === null) return 0;
  return 1 + treeSize(node.left) + treeSize(node.right);
}

/** Collect every entry of a subtree (used for rebuilds and for brute-force cross-checks). */
function collect(node, out = []) {
  if (node === null) return out;
  out.push({ point: node.point, index: node.index });
  collect(node.left, out);
  collect(node.right, out);
  return out;
}

function bruteNearest(entries, query) {
  let best = null;
  let bestDistance = Infinity;
  for (const entry of entries) {
    const d = distance(entry.point, query);
    if (d < bestDistance) {
      bestDistance = d;
      best = entry;
    }
  }
  return { entry: best, distance: bestDistance };
}

function bruteKNearest(entries, query, k) {
  return entries.map((entry) => ({ entry, distance: distance(entry.point, query) }))
    .sort((a, b) => a.distance - b.distance || a.entry.index - b.entry.index)
    .slice(0, k);
}

function bruteRange(entries, min, max) {
  return entries.filter((entry) => entry.point.every((value, d) => value >= min[d] && value <= max[d]))
    .map((entry) => entry.index).sort((a, b) => a - b);
}

class KdTree {
  /** `bulkThreshold` is a function of the base size: the buffer limit before a rebuild. */
  constructor(dims, options = {}) {
    this.dims = dims;
    this.bufferLimit = options.bufferLimit || ((size) => Math.max(8, Math.floor(size / 2)));
    this.entries = [];
    this.root = null;
    this.buffer = [];
    this.rebuilds = 0;
    this.searches = 0;
  }

  static bulkLoaded(entries, dims, options) {
    const tree = new KdTree(dims, options);
    tree.entries = entries.map((entry) => ({ point: entry.point.slice(), index: entry.index }));
    tree.root = buildBalanced(tree.entries, dims);
    return tree;
  }

  get size() {
    return this.entries.length + this.buffer.length;
  }

  get depth() {
    return treeDepth(this.root);
  }

  /** Bulk insertion: buffer, then rebuild once the buffer crosses the documented threshold. */
  bulkInsert(newEntries) {
    for (const entry of newEntries) {
      this.buffer.push({ point: entry.point.slice(), index: entry.index });
    }
    const limit = this.bufferLimit(this.entries.length);
    if (this.buffer.length > limit) this.rebuild();
    return this;
  }

  rebuild() {
    this.entries = this.entries.concat(this.buffer);
    this.buffer = [];
    this.root = buildBalanced(this.entries, this.dims);
    this.rebuilds += 1;
    return this;
  }

  /** Plain leaf insertion, used only to demonstrate how badly it degenerates. */
  insertPlain(point, index) {
    if (point.length !== this.dims) throw new Error("point dimension mismatch");
    const entry = { point: point.slice(), index };
    if (this.root === null) {
      this.root = { point: entry.point, index, axis: 0, left: null, right: null };
      this.entries.push(entry);
      return this;
    }
    let node = this.root;
    for (;;) {
      const axis = node.axis;
      if (entry.point[axis] <= node.point[axis]) {
        if (node.left === null) {
          node.left = { point: entry.point, index, axis: (axis + 1) % this.dims, left: null,
            right: null };
          break;
        }
        node = node.left;
      } else {
        if (node.right === null) {
          node.right = { point: entry.point, index, axis: (axis + 1) % this.dims, left: null,
            right: null };
          break;
        }
        node = node.right;
      }
    }
    this.entries.push(entry);
    return this;
  }

  allEntries() {
    return this.entries.concat(this.buffer);
  }

  nearest(query, options = {}) {
    if (query.length !== this.dims) throw new Error("query dimension mismatch");
    const visitBuffer = options.visitBuffer !== false;
    const best = { point: null, index: -1, distance: Infinity };
    let visited = 0;
    const search = (node) => {
      if (node === null) return;
      visited += 1;
      const d = distance(node.point, query);
      if (d < best.distance || (d === best.distance && node.index < best.index)) {
        best.point = node.point;
        best.index = node.index;
        best.distance = d;
      }
      const axis = node.axis;
      const delta = query[axis] - node.point[axis];
      const first = delta <= 0 ? node.left : node.right;
      const second = delta <= 0 ? node.right : node.left;
      search(first);
      if (best.distance === Infinity || delta * delta <= best.distance * best.distance) search(second);
    };
    search(this.root);
    if (visitBuffer) {
      for (const entry of this.buffer) {
        const d = distance(entry.point, query);
        visited += 1;
        if (d < best.distance || (d === best.distance && entry.index < best.index)) {
          best.point = entry.point;
          best.index = entry.index;
          best.distance = d;
        }
      }
    }
    this.searches += 1;
    return { point: best.point, index: best.index, distance: best.distance, visited };
  }

  kNearest(query, k, options = {}) {
    if (query.length !== this.dims) throw new Error("query dimension mismatch");
    const candidates = [];
    let visited = 0;
    const push = (node) => {
      candidates.push({ entry: { point: node.point, index: node.index },
        distance: distance(node.point, query) });
    };
    const search = (node) => {
      if (node === null) return;
      visited += 1;
      push(node);
      const axis = node.axis;
      const delta = query[axis] - node.point[axis];
      const first = delta <= 0 ? node.left : node.right;
      const second = delta <= 0 ? node.right : node.left;
      search(first);
      const worst = candidates.length >= k
        ? candidates.slice().sort((a, b) => a.distance - b.distance)[k - 1].distance : Infinity;
      if (delta * delta <= worst * worst) search(second);
    };
    search(this.root);
    if (options.visitBuffer !== false) {
      for (const entry of this.buffer) {
        visited += 1;
        candidates.push({ entry, distance: distance(entry.point, query) });
      }
    }
    candidates.sort((a, b) => a.distance - b.distance || a.entry.index - b.entry.index);
    return { hits: candidates.slice(0, k), visited };
  }

  rangeIndices(min, max, options = {}) {
    if (min.length !== this.dims || max.length !== this.dims) throw new Error("box dimension mismatch");
    const hits = [];
    let visited = 0;
    const search = (node) => {
      if (node === null) return;
      visited += 1;
      const inside = node.point.every((value, d) => value >= min[d] && value <= max[d]);
      if (inside) hits.push(node.index);
      const axis = node.axis;
      if (min[axis] <= node.point[axis]) search(node.left);
      if (max[axis] >= node.point[axis]) search(node.right);
    };
    search(this.root);
    if (options.visitBuffer !== false) {
      for (const entry of this.buffer) {
        visited += 1;
        if (entry.point.every((value, d) => value >= min[d] && value <= max[d])) hits.push(entry.index);
      }
    }
    return { indices: hits.sort((a, b) => a - b), visited };
  }

  /** Recursive structural invariant check: the split value separates the two subtrees. */
  checkInvariant(node = this.root) {
    if (node === null) return 0;
    const axis = node.axis;
    let checked = 1;
    const walk = (child, side) => {
      if (child === null) return;
      const value = child.point[axis];
      if (side === "left" && value > node.point[axis]) {
        throw new Error(`left subtree point ${value} exceeds the split ${node.point[axis]}`);
      }
      if (side === "right" && value < node.point[axis]) {
        throw new Error(`right subtree point ${value} is below the split ${node.point[axis]}`);
      }
      checked += 1;
      walk(child.left, side);
      walk(child.right, side);
    };
    walk(node.left, "left");
    walk(node.right, "right");
    return checked + this.checkInvariant(node.left) + this.checkInvariant(node.right) - 1
      - treeSize(node.left) - treeSize(node.right);
  }
}

function randomPoints(rng, count, dims) {
  const entries = [];
  for (let i = 0; i < count; i++) {
    const point = [];
    for (let d = 0; d < dims; d++) point.push(rng.nextInt(4001) / 4.0 - 500.0);
    entries.push({ point, index: i });
  }
  return entries;
}

function main() {
  const assert = require("node:assert");
  const rng = new Rng(20240519);

  // (1) Nearest neighbour, k-nearest and range queries against brute force, several dimensions.
  {
    for (const dims of [2, 3, 5]) {
      const entries = randomPoints(rng, 400, dims);
      const tree = KdTree.bulkLoaded(entries, dims);
      assert.strictEqual(tree.size, 400);
      assert.strictEqual(tree.depth <= 2 * Math.log2(400) + 2, true,
        `depth ${tree.depth} within 2 log2(n) + 2`);
      let worstVisited = 0;
      let sumVisited = 0;
      for (let q = 0; q < 300; q++) {
        const query = [];
        for (let d = 0; d < dims; d++) query.push(rng.nextInt(2001) / 4.0 - 250.0);
        const found = tree.nearest(query);
        const truth = bruteNearest(entries, query);
        assert.ok(Math.abs(found.distance - truth.distance) <= 1e-12,
          `dims ${dims}: distance ${found.distance} vs ${truth.distance}`);
        assert.strictEqual(found.point.length, dims);
        assert.strictEqual(typeof found.index, "number");
        const knn = tree.kNearest(query, 5);
        const truthKnn = bruteKNearest(entries, query, 5);
        assert.strictEqual(knn.hits.length, 5);
        for (let i = 0; i < 5; i++) {
          assert.ok(Math.abs(knn.hits[i].distance - truthKnn[i].distance) <= 1e-12,
            `k-nearest rank ${i}`);
        }
        sumVisited += found.visited;
        worstVisited = Math.max(worstVisited, found.visited);
      }
      assert.ok(sumVisited / 300 < 400 * 0.75,
        `dims ${dims}: average visited nodes ${sumVisited / 300} must beat brute force by a margin`);
      assert.ok(worstVisited < 400, `worst-case visited ${worstVisited} is below n`);
      for (let q = 0; q < 40; q++) {
        const min = [];
        const max = [];
        for (let d = 0; d < dims; d++) {
          const a = rng.nextInt(2001) / 4.0 - 250.0;
          const b = a + rng.nextInt(400) / 4.0;
          min.push(a);
          max.push(b);
        }
        const found = tree.rangeIndices(min, max).indices;
        assert.deepStrictEqual(found, bruteRange(entries, min, max), `dims ${dims} box query`);
      }
      assert.ok(tree.checkInvariant() === 0, "the structural invariant holds");
    }
  }

  // (2) Queries that coincide with stored points, duplicate points and far-away queries.
  {
    const entries = [{ point: [0, 0], index: 0 }, { point: [5, 5], index: 1 }, { point: [5, 5], index: 2 },
      { point: [-3, 4], index: 3 }];
    const tree = KdTree.bulkLoaded(entries, 2);
    const hit = tree.nearest([5, 5]);
    assert.strictEqual(hit.distance, 0);
    assert.strictEqual(hit.index, 1, "ties go to the smallest index");
    assert.strictEqual(tree.nearest([0, 0]).distance, 0);
    assert.strictEqual(tree.nearest([0, 0]).index, 0);
    const far = tree.nearest([1000, -1000]);
    assert.ok(Math.abs(far.distance - bruteNearest(entries, [1000, -1000]).distance) <= 1e-12,
      "a far-away query agrees with brute force");
    assert.ok(Math.abs(far.distance - Math.hypot(1000, 1000)) <= 1e-12,
      "and matches the closed form hypot(1000, 1000) = 1414.2136 for the origin");
    assert.strictEqual(far.index, 0, "the origin really is closer than the (5, 5) pair here");
    assert.deepStrictEqual(tree.kNearest([5, 5], 10).hits.length, 4,
      "asking for more neighbours than points returns them all");
    assert.deepStrictEqual(tree.kNearest([5, 5], 2).hits.map((h) => h.entry.index), [1, 2]);
    assert.deepStrictEqual(tree.rangeIndices([5, 5], [5, 5]).indices, [1, 2]);
    assert.deepStrictEqual(tree.rangeIndices([10, 10], [20, 20]).indices, []);
    assert.deepStrictEqual(tree.rangeIndices([-100, -100], [100, 100]).indices, [0, 1, 2, 3]);
    assert.deepStrictEqual(tree.rangeIndices([5, 5], [0, 0]).indices, [],
      "an inverted box is empty");
    assert.throws(() => tree.nearest([1]), /query dimension mismatch/);
    assert.throws(() => tree.rangeIndices([1, 2], [1]), /box dimension mismatch/);
    const empty = KdTree.bulkLoaded([], 2);
    assert.strictEqual(empty.size, 0);
    assert.strictEqual(empty.nearest([0, 0]).distance, Infinity);
    assert.deepStrictEqual(empty.rangeIndices([0, 0], [1, 1]).indices, []);
    assert.strictEqual(empty.depth, 0);
    const single = KdTree.bulkLoaded([{ point: [1, 2], index: 7 }], 2);
    assert.strictEqual(single.depth, 1);
    assert.strictEqual(single.nearest([0, 0]).index, 7);
  }

  // (3) Bulk insertion: the buffer policy, the rebuild count and query correctness mid-flight.
  {
    const dims = 3;
    const base = randomPoints(rng, 1000, dims);
    const tree = KdTree.bulkLoaded(base, dims);
    let nextIndex = 1000;
    const batches = [];
    let rebuildsAfterFirst = null;
    for (let b = 0; b < 12; b++) {
      const batch = randomPoints(rng, 250, dims).map((entry) => {
        entry.index = nextIndex;
        nextIndex += 1;
        return entry;
      });
      batches.push(batch);
      tree.bulkInsert(batch);
      const limit = tree.bufferLimit(tree.entries.length);
      assert.ok(tree.buffer.length <= limit,
        `buffer ${tree.buffer.length} inside the documented limit ${limit}`);
      if (rebuildsAfterFirst === null) rebuildsAfterFirst = tree.rebuilds;
      // correctness while the buffer is non-empty
      const all = tree.allEntries();
      const query = [rng.nextInt(2001) / 4.0 - 250.0, rng.nextInt(2001) / 4.0 - 250.0,
        rng.nextInt(2001) / 4.0 - 250.0];
      const found = tree.nearest(query);
      const truth = bruteNearest(all, query);
      assert.ok(Math.abs(found.distance - truth.distance) <= 1e-12,
        `batch ${b}: nearest neighbour with a ${tree.buffer.length}-point buffer`);
      const min = [query[0] - 20, query[1] - 20, query[2] - 20];
      const max = [query[0] + 20, query[1] + 20, query[2] + 20];
      assert.deepStrictEqual(tree.rangeIndices(min, max).indices, bruteRange(all, min, max));
    }
    assert.strictEqual(tree.size, 1000 + 12 * 250);
    assert.ok(tree.buffer.length <= tree.bufferLimit(tree.entries.length),
      "the buffer stays inside the policy limit after the last batch");
    assert.ok(tree.buffer.length > 0,
      "with a logarithmic rebuild policy the tail of the stream is still buffered");
    assert.ok(tree.rebuilds >= 1 && tree.rebuilds <= 6,
      `rebuilds ${tree.rebuilds} for 12 batches is logarithmic`);
    assert.ok(rebuildsAfterFirst >= 0);
    assert.ok(tree.depth <= 2 * Math.log2(tree.size) + 2,
      `depth ${tree.depth} for size ${tree.size}`);
    assert.strictEqual(tree.checkInvariant(), 0);
    // No points are lost: the multiset of indices matches.
    const indices = tree.allEntries().map((entry) => entry.index).sort((a, b) => a - b);
    assert.strictEqual(indices.length, 4000);
    assert.deepStrictEqual(indices, [...Array(4000).keys()]);
  }

  // (4) The balancing heuristic versus plain insertion of sorted input.
  {
    const sortedEntries = [...Array(512).keys()].map((i) => ({ point: [i * 1.0, i * 1.0], index: i }));
    const plain = new KdTree(2);
    for (const entry of sortedEntries) plain.insertPlain(entry.point, entry.index);
    assert.strictEqual(plain.depth, 512, "inserting sorted points one by one builds a 512-deep chain");
    const balanced = KdTree.bulkLoaded(sortedEntries, 2);
    assert.strictEqual(balanced.depth, 10, "the balanced build gives depth ceil(log2(513)) - 1 = 10");
    assert.ok(balanced.depth <= Math.ceil(Math.log2(512)) + 1);
    const query = [511.0, 511.0];
    const plainHit = plain.nearest(query, { visitBuffer: false });
    const balancedHit = balanced.nearest(query, { visitBuffer: false });
    assert.strictEqual(plainHit.distance, 0);
    assert.strictEqual(balancedHit.distance, 0);
    assert.ok(plainHit.visited >= 500,
      `the chain visits ${plainHit.visited} of 512 nodes for a query at the data's corner`);
    assert.ok(balancedHit.visited <= 12, `a balanced tree needs ${balancedHit.visited} nodes there`);
    assert.ok(plainHit.visited > 4 * balancedHit.visited);
    // Documented caveat: when the query is *far outside* the data the axis pruning stops helping
    // for either shape, so balancing does not reduce the far-query cost - it removes the
    // pathological worst case for queries near the data, which is what test (1) measures.
    const far = [-1000.0, -1000.0];
    const farPlain = plain.nearest(far, { visitBuffer: false }).visited;
    const farBalanced = balanced.nearest(far, { visitBuffer: false }).visited;
    assert.ok(farPlain > 400, `a far query visits ${farPlain} chain nodes`);
    assert.ok(farBalanced > 400, `and ${farBalanced} balanced nodes`);
    assert.ok(farBalanced > farPlain * 0.9);
    // Bulk loading the same data in one call is identical to building it directly.
    const viaBulk = new KdTree(2, { bufferLimit: () => 0 });
    viaBulk.bulkInsert(sortedEntries);
    assert.strictEqual(viaBulk.rebuilds, 1, "a zero buffer limit rebuilds immediately");
    assert.strictEqual(viaBulk.depth, balanced.depth);
    assert.strictEqual(viaBulk.checkInvariant(), 0);
  }

  // (5) Duplicate-heavy data, high dimensions and determinism.
  {
    const entries = [];
    for (let i = 0; i < 200; i++) entries.push({ point: [i % 5, (i % 7) * 2], index: i });
    const tree = KdTree.bulkLoaded(entries, 2);
    assert.strictEqual(tree.checkInvariant(), 0, "duplicates do not break the invariant");
    assert.strictEqual(tree.nearest([2, 4]).distance, 0);
    assert.strictEqual(tree.nearest([2, 4]).index, 2,
      "the earliest matching index wins: i = 2 is the first i with i % 5 = 2 and i % 7 = 2");
    for (let q = 0; q < 50; q++) {
      const query = [rng.nextInt(20) - 5, rng.nextInt(30) - 10];
      assert.ok(Math.abs(tree.nearest(query).distance
        - bruteNearest(entries, query).distance) <= 1e-12);
    }
    const highDims = randomPoints(rng, 150, 8);
    const high = KdTree.bulkLoaded(highDims, 8);
    for (let q = 0; q < 20; q++) {
      const query = [];
      for (let d = 0; d < 8; d++) query.push(rng.nextInt(2001) / 4.0 - 250.0);
      assert.ok(Math.abs(high.nearest(query).distance - bruteNearest(highDims, query).distance)
        <= 1e-12, "8-dimensional nearest neighbour");
    }
    const first = KdTree.bulkLoaded(randomPoints(new Rng(5), 50, 3), 3);
    const second = KdTree.bulkLoaded(randomPoints(new Rng(5), 50, 3), 3);
    assert.deepStrictEqual(collect(second.root), collect(first.root), "builds are reproducible");
    const replay = new Rng(5);
    const again = [];
    for (let i = 0; i < 3; i++) again.push(replay.nextFloat());
    const replay2 = new Rng(5);
    const again2 = [];
    for (let i = 0; i < 3; i++) again2.push(replay2.nextFloat());
    assert.deepStrictEqual(again2, again);
    assert.strictEqual(tree.searches + high.searches >= 70, true);
  }
}

module.exports = { MASK64, Rng, distance, spreadAxis, buildBalanced, treeDepth, treeSize, collect,
  bruteNearest, bruteKNearest, bruteRange, KdTree, randomPoints };

if (require.main === module) {
  main();
  console.log("PASS");
}
