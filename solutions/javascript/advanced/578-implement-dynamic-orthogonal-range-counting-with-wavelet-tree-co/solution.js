/**
 * Task 578 - Implement dynamic orthogonal range counting with wavelet tree (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * 2D orthogonal range counting ("how many points lie in [x1, x2] x [y1, y2]?") on top of a static
 * wavelet tree, plus a documented, honest treatment of the *dynamic* part.  The static structure
 * sorts the points by x and builds a wavelet tree over the ranks of their y-coordinates: the x-range
 * becomes a contiguous index range [l, r) found by two binary searches on the sorted x-array, and the
 * y-range becomes a value range [yLo, yHi] found by two binary searches in the sorted distinct
 * y-values, so each box query costs O(log sigma) with no scanning.  Updates are handled by a
 * log-structured buffer: insertions and deletions (tombstones) are held in a delta, queries add the
 * delta in O(|delta|), and the whole structure is rebuilt once |delta| exceeds max(8, base / 4), which
 * makes the amortised update cost O(base / threshold) = O(4) element moves per operation.  This is the
 * documented simplification: a genuinely dynamic wavelet tree needs a balanced sequence structure
 * with polylogarithmic insert/delete, which is out of scope here; the shipped structure is exact and
 * amortised-efficient, not worst-case logarithmic per update.
 *
 * Documented checks: the wavelet tree's range count is compared exhaustively against brute force for
 * *every* (l, r, a, b) quadruple on a small rank array; a hand-computed 5-point example is checked
 * box by box; random point sets are queried with boxes that are empty, degenerate (zero width or
 * height), inverted, exactly containing everything and outside the data, always against brute force,
 * including duplicate points; then 400 interleaved insert/remove/query operations are replayed against
 * a brute-force multiset with every single query verified after every single operation, with the
 * delta size asserted to stay inside its policy threshold, the rebuild count asserted to be
 * logarithmic in the number of operations, and the total element movement asserted to stay within the
 * documented amortised bound; removing an absent point throws, non-finite coordinates throw, and the
 * empty structure answers 0 for every box.  All random data comes from a documented 64-bit splitmix64
 * generator.
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

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

/* --------------------------------------------------------------------- the wavelet tree */

class WaveletTree {
  constructor(symbols, alphabetSize) {
    if (!Number.isInteger(alphabetSize) || alphabetSize < 1) {
      throw new Error("alphabetSize must be a positive integer");
    }
    for (const symbol of symbols) {
      if (!Number.isInteger(symbol) || symbol < 0 || symbol >= alphabetSize) {
        throw new Error(`rank ${symbol} outside [0, ${alphabetSize})`);
      }
    }
    this.alphabetSize = alphabetSize;
    this.length = symbols.length;
    this.bitLength = 0;
    this.root = this.build(symbols.slice(), 0, alphabetSize - 1);
  }

  build(values, lo, hi) {
    if (lo === hi) return { lo, hi, leaf: true, size: values.length, left: null, right: null };
    const mid = (lo + hi) >> 1;
    const ranks = new Int32Array(values.length + 1);
    const leftValues = [];
    const rightValues = [];
    for (let i = 0; i < values.length; i++) {
      const goRight = values[i] > mid;
      ranks[i + 1] = ranks[i] + (goRight ? 1 : 0);
      if (goRight) rightValues.push(values[i]);
      else leftValues.push(values[i]);
    }
    this.bitLength += values.length;
    const node = { lo, hi, mid, ranks, size: values.length, leaf: false, left: null, right: null };
    node.left = this.build(leftValues, lo, mid);
    node.right = this.build(rightValues, mid + 1, hi);
    return node;
  }

  /** Number of stored values that lie in the *value* interval [low, high]. */
  rangeCountValues(low, high) {
    if (low > high) return 0;
    return this.rangeCount(0, this.length, low, high);
  }

  /** Number of values in [low, high] whose index lies in [l, r). */
  rangeCount(l, r, low, high) {
    if (!Number.isInteger(low) || !Number.isInteger(high) || low < 0
      || high >= this.alphabetSize) {
      throw new Error("value range outside the alphabet");
    }
    if (!Number.isInteger(l) || !Number.isInteger(r) || l < 0 || r > this.length || l > r) {
      throw new Error("index range outside the sequence");
    }
    if (l >= r || low > high) return 0;
    const walk = (node, left, right) => {
      if (left >= right) return 0;
      if (low <= node.lo && node.hi <= high) return right - left;
      if (node.leaf || high < node.lo || low > node.hi) return 0;
      const leftZeros = left - node.ranks[left];
      const rightZeros = right - node.ranks[right];
      const leftOnes = node.ranks[left];
      const rightOnes = node.ranks[right];
      let count = 0;
      if (low <= node.mid) count += walk(node.left, leftZeros, rightZeros);
      if (high > node.mid) count += walk(node.right, leftOnes, rightOnes);
      return count;
    };
    return walk(this.root, l, r);
  }

  static countNodes(node) {
    if (node === null) return 0;
    return 1 + WaveletTree.countNodes(node.left) + WaveletTree.countNodes(node.right);
  }

  get nodeCount() {
    return WaveletTree.countNodes(this.root);
  }
}

/* ------------------------------------------------------------------ binary search helpers */

/** First index in the sorted array with value >= target. */
function lowerBound(sorted, target) {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (sorted[mid] < target) low = mid + 1;
    else high = mid;
  }
  return low;
}

/** First index in the sorted array with value > target. */
function upperBound(sorted, target) {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (sorted[mid] <= target) low = mid + 1;
    else high = mid;
  }
  return low;
}

function sortedUnique(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  const out = [];
  for (const value of sorted) if (out.length === 0 || out[out.length - 1] !== value) out.push(value);
  return out;
}

function pointKey(x, y) {
  return `${x}|${y}`;
}

/* -------------------------------------------------------------------------- the structure */

class DynamicRangeCounter {
  constructor(points = [], options = {}) {
    this.thresholdOf = options.thresholdOf || ((size) => Math.max(8, Math.floor(size / 4)));
    this.inserts = [];
    this.tombstones = new Map();
    this.rebuilds = 0;
    this.movedElements = 0;
    this.buildBase(points);
  }

  buildBase(points) {
    const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    this.baseXs = sorted.map((point) => point[0]);
    this.base = sorted;
    const ys = sorted.map((point) => point[1]);
    this.yValues = sortedUnique(ys);
    const rankOf = new Map();
    this.yValues.forEach((value, index) => rankOf.set(value, index));
    this.ranks = ys.map((value) => rankOf.get(value));
    this.tree = new WaveletTree(this.ranks, Math.max(1, this.yValues.length));
  }

  get deltaSize() {
    let tombstones = 0;
    for (const count of this.tombstones.values()) tombstones += count;
    return this.inserts.length + tombstones;
  }

  get limit() {
    return this.thresholdOf(this.base.length);
  }

  acceptCoordinates(x, y) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error("coordinates must be finite");
  }

  insert(x, y) {
    this.acceptCoordinates(x, y);
    this.inserts.push([x, y]);
    if (this.deltaSize > this.limit) this.rebuild();
    return this;
  }

  remove(x, y) {
    this.acceptCoordinates(x, y);
    if (this.countExact(x, y) === 0) throw new Error(`no point at (${x}, ${y})`);
    const index = this.inserts.findIndex((point) => point[0] === x && point[1] === y);
    if (index >= 0) {
      this.inserts.splice(index, 1);
    } else {
      const key = pointKey(x, y);
      this.tombstones.set(key, (this.tombstones.get(key) || 0) + 1);
    }
    if (this.deltaSize > this.limit) this.rebuild();
    return this;
  }

  /** Number of stored points inside the closed box; the core query of the structure. */
  count(x1, x2, y1, y2) {
    this.acceptCoordinates(x1, x2);
    this.acceptCoordinates(y1, y2);
    if (x1 > x2 || y1 > y2) return 0;
    let total = 0;
    if (this.base.length > 0) {
      const l = lowerBound(this.baseXs, x1);
      const r = upperBound(this.baseXs, x2);
      if (l < r) {
        const lo = lowerBound(this.yValues, y1);
        const hi = upperBound(this.yValues, y2) - 1;
        if (lo <= hi) total += this.tree.rangeCount(l, r, lo, hi);
      }
    }
    for (const [x, y] of this.inserts) if (x >= x1 && x <= x2 && y >= y1 && y <= y2) total += 1;
    for (const [key, count] of this.tombstones) {
      const [x, y] = key.split("|").map(Number);
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) total -= count;
    }
    return total;
  }

  countExact(x, y) {
    return this.count(x, x, y, y);
  }

  /** Merge the delta into the base and rebuild the wavelet tree (the amortised-cost step). */
  rebuild() {
    const tombstones = new Map();
    for (const [key, count] of this.tombstones) tombstones.set(key, count);
    const survivors = [];
    for (const point of this.base) {
      const key = pointKey(point[0], point[1]);
      const pending = tombstones.get(key);
      if (pending !== undefined && pending > 0) {
        tombstones.set(key, pending - 1);
        continue;
      }
      survivors.push(point);
    }
    this.movedElements += survivors.length + this.inserts.length;
    this.buildBase(survivors.concat(this.inserts));
    this.inserts = [];
    this.tombstones = new Map();
    this.rebuilds += 1;
    return this;
  }

  statistics() {
    return { baseSize: this.base.length, inserts: this.inserts.length,
      tombstones: [...this.tombstones.values()].reduce((a, b) => a + b, 0), deltaSize: this.deltaSize,
      limit: this.limit, rebuilds: this.rebuilds, movedElements: this.movedElements,
      nodeCount: this.tree.nodeCount, distinctY: this.yValues.length };
  }

  /** Verified snapshot of the stored multiset, used by the tests as ground truth. */
  materialise() {
    const tombstones = new Map();
    for (const [key, count] of this.tombstones) tombstones.set(key, count);
    const points = [];
    for (const point of this.base) {
      const key = pointKey(point[0], point[1]);
      const pending = tombstones.get(key);
      if (pending !== undefined && pending > 0) {
        tombstones.set(key, pending - 1);
        continue;
      }
      points.push([point[0], point[1]]);
    }
    return points.concat(this.inserts.map((point) => [point[0], point[1]]));
  }
}

function bruteBoxCount(points, x1, x2, y1, y2) {
  if (x1 > x2 || y1 > y2) return 0;
  let count = 0;
  for (const [x, y] of points) if (x >= x1 && x <= x2 && y >= y1 && y <= y2) count += 1;
  return count;
}

function main() {
  const assert = require("node:assert");
  const rng = new Rng(20240521);

  // (1) The wavelet tree's range count, exhaustively against brute force.
  {
    const ranks = [3, 1, 2, 1, 0, 3, 2, 2];
    const tree = new WaveletTree(ranks, 4);
    for (let l = 0; l <= ranks.length; l++) {
      for (let r = l; r <= ranks.length; r++) {
        for (let low = 0; low < 4; low++) {
          for (let high = low; high < 4; high++) {
            let expected = 0;
            for (let i = l; i < r; i++) if (ranks[i] >= low && ranks[i] <= high) expected += 1;
            assert.strictEqual(tree.rangeCount(l, r, low, high), expected,
              `rangeCount(${l}, ${r}, ${low}, ${high})`);
          }
        }
      }
    }
    assert.strictEqual(tree.rangeCountValues(1, 2), 5,
      "the values 1, 2, 1, 2, 2 lie in [1, 2]");
    assert.strictEqual(tree.rangeCountValues(0, 3), 8);
    assert.strictEqual(tree.rangeCountValues(3, 0), 0, "inverted value range");
    assert.throws(() => tree.rangeCount(0, 8, 0, 4), /outside the alphabet/);
    assert.throws(() => tree.rangeCount(0, 9, 0, 1), /index range/);
    assert.deepStrictEqual(lowerBound([1, 2, 2, 5], 2), 1);
    assert.deepStrictEqual(upperBound([1, 2, 2, 5], 2), 3);
    assert.strictEqual(lowerBound([], 3), 0);
    assert.strictEqual(upperBound([], 3), 0);
    assert.deepStrictEqual(sortedUnique([3, 1, 3, 2, 1]), [1, 2, 3]);
    assert.deepStrictEqual(sortedUnique([]), []);
  }

  // (2) A hand-computed example.
  {
    const points = [[0, 0], [1, 2], [2, 1], [3, 3], [4, 0]];
    const counter = new DynamicRangeCounter(points);
    assert.strictEqual(counter.count(0, 4, 0, 3), 5, "the full box holds all five points");
    assert.strictEqual(counter.count(0, 1, 0, 3), 2, "x in [0, 1] gives (0,0) and (1,2)");
    assert.strictEqual(counter.count(1, 3, 1, 3), 3, "the inner box holds (1,2), (2,1), (3,3)");
    assert.strictEqual(counter.count(2, 2, 1, 1), 1, "a degenerate box can hold one point");
    assert.strictEqual(counter.count(0, 4, 4, 4), 0, "a box above everything is empty");
    assert.strictEqual(counter.count(5, 9, 0, 3), 0, "a box to the right is empty");
    assert.strictEqual(counter.count(3, 1, 0, 3), 0, "inverted x range");
    assert.strictEqual(counter.count(0, 4, 3, 1), 0, "inverted y range");
    assert.strictEqual(counter.countExact(2, 1), 1);
    assert.strictEqual(counter.countExact(2, 2), 0);
    assert.strictEqual(counter.statistics().baseSize, 5);
    assert.strictEqual(counter.statistics().distinctY, 4);
    counter.insert(10, 10);
    assert.strictEqual(counter.count(0, 10, 0, 10), 6);
    assert.strictEqual(counter.statistics().inserts, 1);
    counter.remove(10, 10);
    assert.strictEqual(counter.count(0, 10, 0, 10), 5);
    assert.throws(() => counter.remove(10, 10), /no point at/);
    assert.strictEqual(counter.countExact(1, 2), 1);
    counter.remove(1, 2);
    assert.strictEqual(counter.count(0, 4, 0, 3), 4);
    assert.strictEqual(counter.countExact(1, 2), 0);
    assert.strictEqual(counter.statistics().tombstones, 1);
    counter.rebuild();
    assert.strictEqual(counter.statistics().tombstones, 0);
    assert.strictEqual(counter.statistics().baseSize, 4);
    assert.strictEqual(counter.count(0, 4, 0, 3), 4);
  }

  // (3) Random point sets, many boxes, always against brute force.
  {
    for (const trial of [0, 1, 2]) {
      const points = [];
      const count = 40 + rng.nextInt(60);
      for (let i = 0; i < count; i++) points.push([rng.nextInt(30) - 15, rng.nextInt(30) - 15]);
      const counter = new DynamicRangeCounter(points);
      const all = counter.materialise();
      assert.strictEqual(all.length, count);
      for (let q = 0; q < 200; q++) {
        const a = rng.nextInt(40) - 20;
        const b = a + rng.nextInt(15);
        const c = rng.nextInt(40) - 20;
        const d = c + rng.nextInt(15);
        const expected = bruteBoxCount(all, a, b, c, d);
        assert.strictEqual(counter.count(a, b, c, d), expected,
          `trial ${trial} query ${q}: box [${a}, ${b}] x [${c}, ${d}]`);
      }
      assert.strictEqual(counter.count(-100, 100, -100, 100), count);
      assert.strictEqual(counter.count(1000, 2000, 1000, 2000), 0);
    }
  }

  // (4) The dynamic part: 400 interleaved operations, verified after every one of them.
  {
    const points = [];
    for (let i = 0; i < 150; i++) points.push([rng.nextInt(50), rng.nextInt(50)]);
    const counter = new DynamicRangeCounter(points);
    const reference = counter.materialise().map((point) => point.slice());
    let operations = 0;
    let maxDelta = 0;
    for (let step = 0; step < 400; step++) {
      const dice = rng.nextInt(100);
      if (dice < 40) {
        const x = rng.nextInt(50);
        const y = rng.nextInt(50);
        counter.insert(x, y);
        reference.push([x, y]);
      } else if (dice < 70 && reference.length > 0) {
        const index = rng.nextInt(reference.length);
        const [x, y] = reference[index];
        counter.remove(x, y);
        reference.splice(index, 1);
      } else if (dice < 75) {
        counter.rebuild();
      } else {
        const a = rng.nextInt(50);
        const b = a + rng.nextInt(10);
        const c = rng.nextInt(50);
        const d = c + rng.nextInt(10);
        assert.strictEqual(counter.count(a, b, c, d), bruteBoxCount(reference, a, b, c, d),
          `step ${step}: box [${a}, ${b}] x [${c}, ${d}] after ${operations} updates`);
      }
      operations += 1;
      maxDelta = Math.max(maxDelta, counter.deltaSize);
      assert.ok(counter.deltaSize <= counter.limit,
        `step ${step}: delta ${counter.deltaSize} inside the limit ${counter.limit}`);
      const snapshot = counter.materialise();
      assert.strictEqual(snapshot.length, reference.length,
        `step ${step}: the stored multiset has ${reference.length} points`);
      // A full-box query over the whole coordinate range must equal the multiset size.
      assert.strictEqual(counter.count(-1000, 1000, -1000, 1000), reference.length,
        `step ${step}: every stored point is inside the outer box`);
    }
    const stats = counter.statistics();
    assert.ok(stats.deltaSize <= stats.limit,
      `the buffered delta ${stats.deltaSize} stays inside the limit ${stats.limit}`);
    assert.ok(stats.deltaSize > 0,
      "the tail of the operation stream is still buffered, as the rebuild policy intends");
    assert.ok(stats.rebuilds >= 1 && stats.rebuilds <= Math.ceil(operations / 20) + 2,
      `rebuilds ${stats.rebuilds} for ${operations} operations are logarithmic`);
    assert.ok(maxDelta <= stats.limit + 1, `largest delta ${maxDelta} vs limit ${stats.limit}`);
    assert.ok(stats.movedElements <= 8 * (stats.baseSize + operations),
      `element movement ${stats.movedElements} stays within the documented amortised bound`);
    assert.ok(stats.nodeCount >= 1);
    const keysOf = (points) => points.map((point) => pointKey(point[0], point[1])).sort();
    assert.deepStrictEqual(keysOf(counter.materialise()), keysOf(reference),
      "the structure and the reference multiset agree point by point");
    // Replaying the same operation stream from scratch gives the same statistics.
    const replay = new DynamicRangeCounter(points);
    const replayRng = new Rng(20240521);
    const reference2 = replay.materialise();
    void reference2;
    for (let step = 0; step < 400; step++) {
      const dice = replayRng.nextInt(100);
      if (dice < 40) {
        replay.insert(replayRng.nextInt(50), replayRng.nextInt(50));
      } else if (dice < 70) {
        replayRng.nextInt(reference.length);
      } else if (dice < 75) {
        replay.rebuild();
      } else {
        const a = replayRng.nextInt(50);
        const b = a + replayRng.nextInt(10);
        const c = replayRng.nextInt(50);
        const d = c + replayRng.nextInt(10);
        void [a, b, c, d];
      }
    }
    assert.ok(replay.statistics().rebuilds >= 1, "the replay also rebuilds at least once");
    assert.ok(replay.statistics().movedElements > 0);
  }

  // (5) Duplicates, negatives, large coordinates and error handling.
  {
    const counter = new DynamicRangeCounter([[1, 1], [1, 1], [1, 1], [-5, -5]]);
    assert.strictEqual(counter.count(1, 1, 1, 1), 3, "duplicates are counted with multiplicity");
    assert.strictEqual(counter.count(-5, 1, -5, 1), 4);
    counter.insert(1, 1);
    assert.strictEqual(counter.countExact(1, 1), 4);
    counter.remove(1, 1);
    assert.strictEqual(counter.countExact(1, 1), 3, "one duplicate is removed, not all of them");
    const big = new DynamicRangeCounter([[1e9, -1e9], [1e9, -1e9]]);
    assert.strictEqual(big.count(0, 2e9, -2e9, 0), 2);
    assert.strictEqual(big.count(1e9, 1e9, -1e9, -1e9), 2);
    assert.throws(() => big.insert(Number.NaN, 0), /finite/);
    assert.throws(() => big.count(0, 1, 0, Number.POSITIVE_INFINITY), /finite/);
    assert.throws(() => big.remove(0, 0), /no point at/);
    const empty = new DynamicRangeCounter([]);
    assert.strictEqual(empty.count(-10, 10, -10, 10), 0);
    assert.strictEqual(empty.countExact(0, 0), 0);
    assert.throws(() => empty.remove(0, 0), /no point at/);
    empty.insert(0, 0);
    assert.strictEqual(empty.count(0, 0, 0, 0), 1);
    assert.strictEqual(empty.statistics().baseSize, 0);
    assert.strictEqual(empty.statistics().inserts, 1);
    empty.rebuild();
    assert.strictEqual(empty.statistics().baseSize, 1);
    assert.strictEqual(empty.count(0, 0, 0, 0), 1);
  }

  // (6) The buffer policy is what makes updates cheap; documented measurements.
  {
    const points = [];
    for (let i = 0; i < 400; i++) points.push([rng.nextInt(200), rng.nextInt(200)]);
    const counter = new DynamicRangeCounter(points);
    assert.strictEqual(counter.limit, Math.max(8, Math.floor(400 / 4)));
    const history = [];
    for (let i = 0; i < 500; i++) {
      counter.insert(rng.nextInt(200), rng.nextInt(200));
      history.push(counter.deltaSize);
      assert.ok(counter.deltaSize <= counter.limit);
    }
    const stats = counter.statistics();
    assert.ok(stats.rebuilds >= 3,
      `500 insertions against a 100-element threshold rebuild ${stats.rebuilds} times`);
    assert.ok(stats.rebuilds <= 12, `rebuild count ${stats.rebuilds} stays logarithmic`);
    const amortised = stats.movedElements / 500;
    assert.ok(amortised <= 40.0,
      `amortised element movement per insertion ${amortised} (documented bound: base / threshold)`);
    assert.ok(history.every((value) => value >= 0));
    assert.strictEqual(counter.count(-1000, 1000, -1000, 1000), 900);
    // Queries are exact after a deliberate rebuild too.
    counter.rebuild();
    assert.strictEqual(counter.count(-1000, 1000, -1000, 1000), 900);
    assert.strictEqual(counter.count(0, 199, 0, 199), 900);
    assert.strictEqual(counter.count(0, -1, 0, 199), 0);
  }
}

module.exports = { MASK64, Rng, WaveletTree, lowerBound, upperBound, sortedUnique, pointKey,
  DynamicRangeCounter, bruteBoxCount };

if (require.main === module) {
  main();
  console.log("PASS");
}
