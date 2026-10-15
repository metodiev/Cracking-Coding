/** Task 395 - Implement the dynamic convex hull trick for max queries (Li Chao tree). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

class LiChao {
  constructor(xLo, xHi) {
    this.xLo = xLo;
    this.xHi = xHi;
    const size = 4 * (xHi - xLo + 1);
    this.has = new Array(size).fill(false);
    this.m = new Array(size).fill(0);
    this.b = new Array(size).fill(0);
  }

  addLine(slope, intercept) {
    this.#add(1, this.xLo, this.xHi, slope, intercept);
  }

  #add(node, lo, hi, slope, intercept) {
    if (!this.has[node]) {
      this.has[node] = true;
      this.m[node] = slope;
      this.b[node] = intercept;
      return;
    }
    const mid = Math.floor((lo + hi) / 2);
    if (slope * mid + intercept > this.m[node] * mid + this.b[node]) {
      [slope, this.m[node]] = [this.m[node], slope];
      [intercept, this.b[node]] = [this.b[node], intercept];
    }
    if (lo === hi) return;
    if (slope * lo + intercept > this.m[node] * lo + this.b[node]) {
      this.#add(2 * node, lo, mid, slope, intercept);
    } else if (slope * hi + intercept > this.m[node] * hi + this.b[node]) {
      this.#add(2 * node + 1, mid + 1, hi, slope, intercept);
    }
  }

  query(x) {
    let node = 1;
    let lo = this.xLo;
    let hi = this.xHi;
    let best = null;
    for (;;) {
      if (this.has[node]) {
        const value = this.m[node] * x + this.b[node];
        if (best === null || value > best) best = value;
      }
      if (lo === hi) break;
      const mid = Math.floor((lo + hi) / 2);
      if (x <= mid) {
        node = 2 * node;
        hi = mid;
      } else {
        node = 2 * node + 1;
        lo = mid + 1;
      }
    }
    if (best === null) throw new Error("query on an empty tree");
    return best;
  }
}

function bruteForce(lines, x) {
  return Math.max(...lines.map(([m, b]) => m * x + b));
}

function lcgLines(count, seed = 4242, mRange = 50, bRange = 1000) {
  const lines = [];
  for (let k = 0; k < count; k++) {
    seed = (seed * 75 + 74) % 65537;
    const slope = (seed % (2 * mRange + 1)) - mRange;
    seed = (seed * 75 + 74) % 65537;
    const intercept = (seed % (2 * bRange + 1)) - bRange;
    lines.push([slope, intercept]);
  }
  return lines;
}

module.exports = { LiChao, bruteForce, lcgLines };

if (require.main === module) {
  const assert = require("node:assert");
  const tree = new LiChao(-500, 500);
  for (const [m, b] of [[1, 0], [-1, 0], [0, 0]]) tree.addLine(m, b);
  assert.strictEqual(tree.query(5), 5);
  assert.strictEqual(tree.query(-5), 5);
  assert.strictEqual(tree.query(0), 0);
  assert.strictEqual(tree.query(3), 3);

  const cross = new LiChao(-500, 500);
  cross.addLine(2, 1);
  cross.addLine(-3, 10);
  assert.strictEqual(cross.query(2), 5);
  assert.strictEqual(cross.query(1), 7);
  assert.strictEqual(cross.query(0), 10);
  assert.strictEqual(cross.query(5), 11);

  const fixed = [[1, 0], [-1, 3], [0, -2], [5, -10], [-4, 6]];
  const checker = new LiChao(-20, 20);
  for (const [m, b] of fixed) checker.addLine(m, b);
  for (let x = -20; x <= 20; x++) assert.strictEqual(checker.query(x), bruteForce(fixed, x), String(x));

  const lines = lcgLines(200);
  const big = new LiChao(-500, 500);
  for (const [m, b] of lines) big.addLine(m, b);
  for (let step = 0; step <= 100; step++) {
    const x = -500 + step * 10;
    assert.strictEqual(big.query(x), bruteForce(lines, x), String(x));
  }

  const duplicate = new LiChao(-10, 10);
  for (let i = 0; i < 4; i++) duplicate.addLine(3, -1);
  assert.strictEqual(duplicate.query(7), 20);

  assert.throws(() => new LiChao(-10, 10).query(0), /empty tree/);
  console.log("PASS");
}
