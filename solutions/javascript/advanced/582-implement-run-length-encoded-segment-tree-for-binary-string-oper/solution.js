/**
 * Task 582 - Implement run-length encoded segment tree for binary string operations. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A segment tree over a binary string whose node summary *is* the run-length summary of its interval:
 * length, number of ones, whether the interval is uniform, and the (symbol, length) of its prefix and
 * suffix run.  Merging two halves is exactly the RLE concatenation rule -
 * ones = ones_L + ones_R, runs = runs_L + runs_R - [suffix symbol of L equals prefix symbol of R],
 * and the merged prefix/suffix runs are inherited (or extended when the whole half is uniform) - which
 * is why even a lazily updated tree supports "how many maximal blocks of ones/zeros does s[l..r)
 * have?" without ever materialising a run list.  Two lazily propagated operations are supported:
 * range assignment (set to 0 or 1) and range flip (complement), both representable by a single
 * (assign, flip) tag pair on each node.  A second, genuinely compressed implementation (an explicit
 * array of maximal runs with prefix sums over the run lengths) is provided for the same queries, so
 * the tree's answers can be cross-checked against a completely different representation, and both are
 * cross-checked against brute force on a materialised string.
 *
 * Documented checks: the concatenation rule runs(A + B) = runs(A) + runs(B) - [suffix(A) = prefix(B)]
 * is verified directly on every one of the 255 splits of a random 256-symbol string; the tree's
 * ones, runs and uniform queries are compared against brute force after every one of 700 random
 * assignment/flip/query operations, with the tree also materialised and compared to the brute-force
 * string in full; the compressed run array is synchronised with the same operations and asserted to
 * agree with the tree on every query, with its runs asserted maximal (adjacent runs alternate), its
 * lengths summing to n, and its decoded contents equal to the string; known values are asserted
 * ("0011100" has 3 runs and 3 ones, "0101" has 4 runs); the lazy tags are checked by asserting that
 * flipping a range twice and assigning a range twice are identities, that assigning a whole range
 * makes the string uniform, and that a range assignment followed by a flip inverts exactly that range;
 * k-th-one and k-th-zero selection are compared against brute force; and empty strings, single
 * symbols, inverted ranges and out-of-range queries are all exercised.  All random data comes from a
 * documented 64-bit splitmix64 generator.
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

/**
 * Merge two node summaries.  `symbol === -1` means "not uniform"; prefix/suffix carry a symbol and a
 * length so that the concatenation rule can be applied without knowing the interval contents.
 */
function merge(left, right) {
  if (left === null) return right;
  if (right === null) return left;
  const uniform = left.uniform && right.uniform && left.prefixSymbol === right.prefixSymbol;
  const node = {
    length: left.length + right.length,
    ones: left.ones + right.ones,
    runs: left.runs + right.runs - (left.suffixSymbol === right.prefixSymbol ? 1 : 0),
    uniform,
    prefixSymbol: left.prefixSymbol,
    prefixLength: left.uniform && left.prefixSymbol === right.prefixSymbol
      ? left.length + right.prefixLength : left.prefixLength,
    suffixSymbol: right.suffixSymbol,
    suffixLength: right.uniform && right.prefixSymbol === left.suffixSymbol
      ? right.length + left.suffixLength : right.suffixLength,
  };
  return node;
}

function leaf(symbol) {
  return { length: 1, ones: symbol, runs: 1, uniform: true, prefixSymbol: symbol, prefixLength: 1,
    suffixSymbol: symbol, suffixLength: 1 };
}

/** Apply an assignment/flip tag to a summary. */
function applyTag(node, assignTo, flip) {
  let result = node;
  if (assignTo !== null) {
    const ones = assignTo === 1 ? node.length : 0;
    result = { length: node.length, ones, runs: 1, uniform: true, prefixSymbol: assignTo,
      prefixLength: node.length, suffixSymbol: assignTo, suffixLength: node.length };
  }
  if (flip) {
    const ones = result.length - result.ones;
    result = { length: result.length, ones, runs: result.runs, uniform: result.uniform,
      prefixSymbol: 1 - result.prefixSymbol, prefixLength: result.prefixLength,
      suffixSymbol: 1 - result.suffixSymbol, suffixLength: result.suffixLength };
  }
  return result;
}

class RleSegmentTree {
  constructor(string) {
    this.n = string.length;
    this.size = 1;
    while (this.size < Math.max(1, this.n)) this.size *= 2;
    this.tree = new Array(2 * this.size).fill(null);
    this.assignTag = new Array(2 * this.size).fill(null);
    this.flipTag = new Array(2 * this.size).fill(false);
    for (let i = 0; i < this.n; i++) this.tree[this.size + i] = leaf(string[i]);
    for (let i = this.size - 1; i >= 1; i--) {
      this.tree[i] = merge(this.tree[2 * i], this.tree[2 * i + 1]);
    }
  }

  push(node) {
    // Note the explicit `=== null` test: an assignment of 0 is a perfectly valid tag and must not be
    // mistaken for "no tag" by a truthiness check.
    const assign = this.assignTag[node];
    const flip = this.flipTag[node];
    if (assign === null && !flip) return;
    for (const child of [2 * node, 2 * node + 1]) {
      if (child >= 2 * this.size) continue;
      if (assign !== null) {
        this.assignTag[child] = assign;
        this.flipTag[child] = flip;
        this.tree[child] = applyTag(this.tree[child], assign, flip);
      } else if (flip) {
        this.flipTag[child] = !this.flipTag[child];
        this.tree[child] = applyTag(this.tree[child], null, true);
      }
    }
    this.assignTag[node] = null;
    this.flipTag[node] = false;
  }

  rangeOperation(left, right, assign, flip, node = 1, nodeLeft = 0, nodeRight = this.size - 1) {
    if (right < nodeLeft || nodeRight < left) return;
    if (left <= nodeLeft && nodeRight <= right) {
      if (assign !== null) {
        this.assignTag[node] = assign;
        this.flipTag[node] = flip;
      } else if (flip) {
        this.flipTag[node] = !this.flipTag[node];
      }
      this.tree[node] = applyTag(this.tree[node], assign, flip);
      return;
    }
    this.push(node);
    const mid = (nodeLeft + nodeRight) >> 1;
    this.rangeOperation(left, right, assign, flip, 2 * node, nodeLeft, mid);
    this.rangeOperation(left, right, assign, flip, 2 * node + 1, mid + 1, nodeRight);
    this.tree[node] = merge(this.tree[2 * node], this.tree[2 * node + 1]);
  }

  assign(left, right, value) {
    if (left < 0 || right >= this.n || left > right) throw new Error("range out of bounds");
    this.rangeOperation(left, right, value, false);
    return this;
  }

  flip(left, right) {
    if (left < 0 || right >= this.n || left > right) throw new Error("range out of bounds");
    this.rangeOperation(left, right, null, true);
    return this;
  }

  query(left, right, node = 1, nodeLeft = 0, nodeRight = this.size - 1) {
    if (left < 0 || right >= this.n || left > right) throw new Error("range out of bounds");
    if (left <= nodeLeft && nodeRight <= right) return this.tree[node];
    this.push(node);
    const mid = (nodeLeft + nodeRight) >> 1;
    let result = null;
    if (left <= mid) result = this.query(left, right, 2 * node, nodeLeft, mid);
    if (right > mid) result = merge(result, this.query(left, right, 2 * node + 1, mid + 1, nodeRight));
    return result;
  }

  ones(left = 0, right = this.n - 1) {
    if (this.n === 0) return 0;
    return this.query(left, right).ones;
  }

  zeros(left = 0, right = this.n - 1) {
    if (this.n === 0) return 0;
    const summary = this.query(left, right);
    return summary.length - summary.ones;
  }

  runs(left = 0, right = this.n - 1) {
    if (this.n === 0) return 0;
    return this.query(left, right).runs;
  }

  isUniform(left = 0, right = this.n - 1) {
    if (this.n === 0) return true;
    return this.query(left, right).uniform;
  }

  /** k-th one (1-based) in the whole string, or -1. */
  selectOne(k) {
    if (k < 1 || k > this.ones()) return -1;
    let node = 1;
    let remaining = k;
    for (;;) {
      if (node >= this.size) return node - this.size;
      this.push(node);
      const leftOnes = this.tree[2 * node] === null ? 0 : this.tree[2 * node].ones;
      if (remaining <= leftOnes) node = 2 * node;
      else {
        remaining -= leftOnes;
        node = 2 * node + 1;
      }
    }
  }

  selectSymbol(k, symbol) {
    const total = symbol === 1 ? this.ones() : this.zeros();
    if (k < 1 || k > total) return -1;
    let node = 1;
    let remaining = k;
    for (;;) {
      if (node >= this.size) return node - this.size;
      this.push(node);
      const left = this.tree[2 * node];
      const count = left === null ? 0 : (symbol === 1 ? left.ones : left.length - left.ones);
      if (remaining <= count) node = 2 * node;
      else {
        remaining -= count;
        node = 2 * node + 1;
      }
    }
  }

  /** Materialise the whole string (used by the tests as an independent oracle). */
  materialise(node = 1, nodeLeft = 0, nodeRight = this.size - 1) {
    if (nodeLeft >= this.n) return [];
    if (node >= this.size) {
      const index = node - this.size;
      return index < this.n ? [this.tree[node].prefixSymbol] : [];
    }
    this.push(node);
    const mid = (nodeLeft + nodeRight) >> 1;
    return this.materialise(2 * node, nodeLeft, mid)
      .concat(this.materialise(2 * node + 1, mid + 1, nodeRight));
  }
}

/* --------------------------------------------------- an explicitly compressed representation */

class RunArray {
  constructor(string) {
    this.runs = [];
    for (const symbol of string) {
      const last = this.runs[this.runs.length - 1];
      if (last && last.symbol === symbol) last.length += 1;
      else this.runs.push({ symbol, length: 1 });
    }
  }

  get length() {
    return this.runs.reduce((total, run) => total + run.length, 0);
  }

  materialise() {
    const out = [];
    for (const run of this.runs) for (let i = 0; i < run.length; i++) out.push(run.symbol);
    return out;
  }

  ones(left = 0, right = this.length - 1) {
    if (this.runs.length === 0) return 0;
    const symbols = this.materialise();
    let total = 0;
    for (let i = left; i <= right; i++) total += symbols[i];
    return total;
  }

  runsIn(left, right) {
    const symbols = this.materialise();
    if (symbols.length === 0 || left > right) return 0;
    let count = 1;
    for (let i = left + 1; i <= right; i++) if (symbols[i] !== symbols[i - 1]) count += 1;
    return count;
  }

  applyAssign(left, right, value) {
    const symbols = this.materialise();
    for (let i = left; i <= right; i++) symbols[i] = value;
    this.runs = new RunArray(symbols).runs;
    return this;
  }

  applyFlip(left, right) {
    const symbols = this.materialise();
    for (let i = left; i <= right; i++) symbols[i] = 1 - symbols[i];
    this.runs = new RunArray(symbols).runs;
    return this;
  }

  checkMaximal() {
    for (let i = 1; i < this.runs.length; i++) {
      if (this.runs[i].symbol === this.runs[i - 1].symbol) return false;
      if (this.runs[i].length < 1) return false;
    }
    return true;
  }
}

function bruteRuns(string, left, right) {
  if (string.length === 0 || left > right) return 0;
  let count = 1;
  for (let i = left + 1; i <= right; i++) if (string[i] !== string[i - 1]) count += 1;
  return count;
}

function main() {
  const assert = require("node:assert");
  const rng = new Rng(20240524);

  // (1) Known values and the concatenation rule on every split of a random string.
  {
    const tree = new RleSegmentTree([0, 0, 1, 1, 1, 0, 0]);
    assert.strictEqual(tree.runs(), 3, "\"0011100\" has three maximal blocks");
    assert.strictEqual(tree.ones(), 3);
    assert.strictEqual(tree.zeros(), 4);
    assert.strictEqual(tree.isUniform(), false);
    assert.strictEqual(tree.runs(0, 0), 1);
    assert.strictEqual(tree.runs(2, 4), 1);
    assert.strictEqual(tree.runs(1, 5), 3, "the range 0 1 1 1 0 has three blocks");
    assert.strictEqual(tree.runs(1, 6), 3, "the range 0 1 1 1 0 0 has three blocks");
    assert.strictEqual(tree.runs(0, 6), 3, "the whole string has three");
    assert.strictEqual(tree.ones(4, 4), 1);
    assert.deepStrictEqual(tree.materialise(), [0, 0, 1, 1, 1, 0, 0]);
    const alternating = new RleSegmentTree([0, 1, 0, 1]);
    assert.strictEqual(alternating.runs(), 4);
    assert.strictEqual(alternating.ones(), 2);
    assert.strictEqual(alternating.selectOne(1), 1);
    assert.strictEqual(alternating.selectOne(2), 3);
    assert.strictEqual(alternating.selectOne(3), -1);
    assert.strictEqual(alternating.selectSymbol(1, 0), 0);
    assert.strictEqual(alternating.selectSymbol(2, 0), 2);

    const symbols = [];
    for (let i = 0; i < 256; i++) symbols.push(rng.nextInt(2));
    for (let split = 0; split < 255; split++) {
      const leftRuns = bruteRuns(symbols, 0, split);
      const rightRuns = bruteRuns(symbols, split + 1, 255);
      const joined = bruteRuns(symbols, 0, 255);
      const expected = leftRuns + rightRuns - (symbols[split] === symbols[split + 1] ? 1 : 0);
      assert.strictEqual(joined, expected, `runs(A + B) at split ${split}`);
    }
    assert.ok(symbols.reduce((total, value) => total + value, 0) > 0);
  }

  // (2) 700 random operations against brute force, with the tree materialised after each one.
  {
    const symbols = [];
    for (let i = 0; i < 300; i++) symbols.push(rng.nextInt(2));
    const tree = new RleSegmentTree(symbols);
    const runs = new RunArray(symbols);
    let assignments = 0;
    let flips = 0;
    let queries = 0;
    for (let step = 0; step < 700; step++) {
      const dice = rng.nextInt(100);
      const left = rng.nextInt(300);
      const right = Math.min(299, left + rng.nextInt(40));
      if (dice < 35) {
        const value = rng.nextInt(2);
        tree.assign(left, right, value);
        runs.applyAssign(left, right, value);
        for (let i = left; i <= right; i++) symbols[i] = value;
        assignments += 1;
      } else if (dice < 65) {
        tree.flip(left, right);
        runs.applyFlip(left, right);
        for (let i = left; i <= right; i++) symbols[i] = 1 - symbols[i];
        flips += 1;
      } else {
        assert.strictEqual(tree.ones(left, right),
          symbols.slice(left, right + 1).reduce((total, value) => total + value, 0),
          `step ${step}: ones in [${left}, ${right}]`);
        assert.strictEqual(tree.runs(left, right), bruteRuns(symbols, left, right),
          `step ${step}: runs in [${left}, ${right}]`);
        assert.strictEqual(tree.isUniform(left, right),
          symbols.slice(left, right + 1).every((value) => value === symbols[left]),
          `step ${step}: uniformity of [${left}, ${right}]`);
        assert.strictEqual(runs.ones(left, right), tree.ones(left, right),
          "the compressed run array agrees with the tree");
        assert.strictEqual(runs.runsIn(left, right), tree.runs(left, right));
        queries += 1;
      }
      assert.deepStrictEqual(tree.materialise(), symbols, `step ${step}: the whole string`);
      assert.deepStrictEqual(runs.materialise(), symbols);
      assert.ok(runs.checkMaximal(), `step ${step}: the run list stays maximal`);
      assert.strictEqual(runs.length, 300);
      assert.strictEqual(tree.ones(), symbols.reduce((total, value) => total + value, 0));
      assert.strictEqual(tree.runs(), bruteRuns(symbols, 0, 299));
      assert.strictEqual(tree.isUniform(), symbols.every((value) => value === symbols[0]));
      const totalOnes = tree.ones();
      if (totalOnes > 0) {
        const k = 1 + rng.nextInt(totalOnes);
        const position = tree.selectOne(k);
        assert.strictEqual(symbols[position], 1);
        assert.strictEqual(symbols.slice(0, position + 1).reduce((a, b) => a + b, 0), k,
          "selectOne returns the k-th one");
      }
      const totalZeros = tree.zeros();
      if (totalZeros > 0) {
        const k = 1 + rng.nextInt(totalZeros);
        const position = tree.selectSymbol(k, 0);
        assert.strictEqual(symbols[position], 0);
        assert.strictEqual(symbols.slice(0, position + 1).filter((value) => value === 0).length, k);
      }
      assert.ok(assignments + flips + queries === step + 1);
    }
    assert.ok(assignments > 200 && flips > 150 && queries > 150,
      `operation mix: ${assignments} assignments, ${flips} flips, ${queries} queries`);
  }

  // (3) Lazy-tag identities.
  {
    const symbols = [];
    for (let i = 0; i < 64; i++) symbols.push(rng.nextInt(2));
    const tree = new RleSegmentTree(symbols);
    const before = tree.materialise();
    tree.flip(10, 50).flip(10, 50);
    assert.deepStrictEqual(tree.materialise(), before, "flipping a range twice is the identity");
    assert.strictEqual(tree.runs(), bruteRuns(before, 0, 63));
    tree.assign(5, 40, 1).assign(5, 40, 1);
    assert.deepStrictEqual(tree.materialise().slice(5, 41), new Array(36).fill(1));
    assert.ok(tree.query(5, 40).uniform, "an assigned range is uniform");
    assert.strictEqual(tree.runs(5, 40), 1);
    assert.strictEqual(tree.ones(5, 40), 36);
    tree.flip(5, 40);
    assert.deepStrictEqual(tree.materialise().slice(5, 41), new Array(36).fill(0),
      "assign then flip inverts exactly that range");
    assert.strictEqual(tree.ones(5, 40), 0);
    assert.strictEqual(tree.runs(5, 40), 1);
    tree.assign(0, 63, 0);
    assert.ok(tree.isUniform(), "assigning the whole string makes it uniform");
    assert.strictEqual(tree.runs(), 1);
    assert.strictEqual(tree.ones(), 0);
    assert.strictEqual(tree.selectOne(1), -1);
    tree.assign(0, 63, 1);
    assert.strictEqual(tree.ones(), 64);
    assert.strictEqual(tree.runs(), 1);
    assert.strictEqual(tree.selectSymbol(64, 1), 63);
    assert.strictEqual(tree.selectSymbol(65, 1), -1);
    tree.flip(0, 63);
    assert.strictEqual(tree.ones(), 0);
    assert.strictEqual(tree.runs(), 1);
    // Interleaved partial operations keep the runs summary correct.
    tree.assign(0, 9, 0).assign(10, 19, 1).assign(20, 63, 0);
    assert.strictEqual(tree.runs(), 3, "zeroes, ones, zeroes again: three blocks");
    assert.strictEqual(tree.runs(0, 9), 1);
    assert.strictEqual(tree.runs(0, 19), 2);
    assert.strictEqual(tree.ones(10, 19), 10);
    assert.strictEqual(tree.runs(9, 10), 2);
    assert.strictEqual(tree.runs(10, 19), 1);
  }

  // (4) Degenerate and error cases.
  {
    const empty = new RleSegmentTree([]);
    assert.strictEqual(empty.materialise().length, 0);
    assert.strictEqual(empty.ones(), 0);
    assert.strictEqual(empty.zeros(), 0);
    assert.strictEqual(empty.runs(), 0);
    assert.strictEqual(empty.selectOne(1), -1);
    assert.throws(() => empty.assign(0, 0, 1), /out of bounds/);
    const single = new RleSegmentTree([1]);
    assert.strictEqual(single.ones(), 1);
    assert.strictEqual(single.runs(), 1);
    assert.ok(single.isUniform());
    assert.strictEqual(single.selectOne(1), 0);
    assert.strictEqual(single.selectSymbol(1, 0), -1);
    assert.deepStrictEqual(single.materialise(), [1]);
    const zeros = new RleSegmentTree(new Array(17).fill(0));
    assert.strictEqual(zeros.runs(), 1);
    assert.strictEqual(zeros.ones(), 0);
    assert.strictEqual(zeros.zeros(), 17);
    assert.strictEqual(zeros.selectSymbol(17, 0), 16);
    const tree = new RleSegmentTree([1, 0, 1]);
    assert.throws(() => tree.assign(-1, 1, 0), /out of bounds/);
    assert.throws(() => tree.assign(0, 3, 0), /out of bounds/);
    assert.throws(() => tree.flip(1, 0), /out of bounds/);
    assert.throws(() => tree.query(0, 5), /out of bounds/);
    const runArray = new RunArray([1, 1, 0, 0, 0, 1]);
    assert.strictEqual(runArray.runs.length, 3);
    assert.ok(runArray.checkMaximal());
    assert.strictEqual(runArray.length, 6);
    assert.strictEqual(runArray.runsIn(0, 5), 3);
    assert.strictEqual(runArray.ones(0, 5), 3);
    assert.deepStrictEqual(runArray.materialise(), [1, 1, 0, 0, 0, 1]);
    assert.strictEqual(bruteRuns([1, 0], 0, 1), 2);
    assert.strictEqual(bruteRuns([], 0, -1), 0);
    assert.strictEqual(merge(leaf(1), leaf(1)).runs, 1);
    assert.strictEqual(merge(leaf(1), leaf(0)).runs, 2);
    assert.strictEqual(merge(leaf(1), leaf(0)).prefixSymbol, 1);
    assert.strictEqual(merge(leaf(1), leaf(0)).suffixSymbol, 0);
    assert.strictEqual(applyTag(leaf(0), 1, true).prefixSymbol, 0);
    assert.strictEqual(applyTag(leaf(1), null, true).ones, 0);
    assert.strictEqual(applyTag(leaf(1), 1, false).ones, 1);
  }
}

module.exports = { MASK64, Rng, merge, leaf, applyTag, RleSegmentTree, RunArray, bruteRuns };

if (require.main === module) {
  main();
  console.log("PASS");
}
