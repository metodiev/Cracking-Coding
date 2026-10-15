/** Task 370 - Line sweep for rectangle area union. (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
"use strict";

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/**
 * Segment tree over compressed y edges.  count == how many active rectangles fully cover
 * the node, length == covered length of its span.  Counts (not booleans) are what make
 * overlaps of three or more rectangles and exact touching edges work.
 */
class CoverLengthTree {
  constructor(sortedYs) {
    this.ys = sortedYs;
    this.m = sortedYs.length - 1;
    const size = 4 * Math.max(this.m, 1);
    this.count = new Array(size).fill(0);
    this.length = new Array(size).fill(0);
  }

  rangeAdd(yLo, yHi, delta) {
    if (this.m === 0 || yLo >= yHi) return;
    this._update(1, 0, this.m, yLo, yHi, delta);
  }

  _update(node, lo, hi, ql, qr, delta) {
    if (qr <= lo || hi <= ql) return;
    if (ql <= lo && hi <= qr) {
      this.count[node] += delta;
    } else {
      const mid = (lo + hi) >> 1;
      this._update(2 * node, lo, mid, ql, qr, delta);
      this._update(2 * node + 1, mid, hi, ql, qr, delta);
    }
    if (this.count[node] > 0) {
      this.length[node] = this.ys[hi] - this.ys[lo];
    } else if (hi - lo === 1) {
      this.length[node] = 0;
    } else {
      this.length[node] = this.length[2 * node] + this.length[2 * node + 1];
    }
  }

  measure() {
    return this.m === 0 ? 0 : this.length[1];
  }
}

/** Area of the union of axis-aligned rectangles [x1, y1, x2, y2]. */
function unionArea(rectangles) {
  const events = [];
  const ySet = new Set();
  for (const [x1, y1, x2, y2] of rectangles) {
    if (x1 >= x2 || y1 >= y2) continue;                  // zero/negative area: ignored
    events.push([x1, 1, y1, y2]);
    events.push([x2, -1, y1, y2]);
    ySet.add(y1);
    ySet.add(y2);
  }
  if (events.length === 0) return 0;
  const ys = [...ySet].sort((a, b) => a - b);
  const index = new Map();
  ys.forEach((y, i) => index.set(y, i));
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const tree = new CoverLengthTree(ys);
  let area = 0;
  let prevX = events[0][0];
  let i = 0;
  while (i < events.length) {
    const x = events[i][0];
    area += tree.measure() * (x - prevX);
    while (i < events.length && events[i][0] === x) {
      const [, delta, y1, y2] = events[i];
      tree.rangeAdd(index.get(y1), index.get(y2), delta);
      i++;
    }
    prevX = x;
  }
  return area;
}

/** Independent oracle: sum the fully covered cells of the compressed coordinate grid. */
function unionAreaOracle(rectangles) {
  const rects = rectangles.filter(([x1, y1, x2, y2]) => x1 < x2 && y1 < y2);
  if (rects.length === 0) return 0;
  const xs = [...new Set(rects.flatMap((r) => [r[0], r[2]]))].sort((a, b) => a - b);
  const ys = [...new Set(rects.flatMap((r) => [r[1], r[3]]))].sort((a, b) => a - b);
  let area = 0;
  for (let i = 0; i + 1 < xs.length; i++) {
    for (let j = 0; j + 1 < ys.length; j++) {
      for (const [x1, y1, x2, y2] of rects) {
        if (x1 <= xs[i] && x2 >= xs[i + 1] && y1 <= ys[j] && y2 >= ys[j + 1]) {
          area += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j]);
          break;
        }
      }
    }
  }
  return area;
}

module.exports = { unionArea, unionAreaOracle, CoverLengthTree, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  // Canonical example: [0,0,2,2] and [1,1,3,3] overlap in a 1x1 square -> 7.
  const canonical = [[0, 0, 2, 2], [1, 1, 3, 3]];
  assert.strictEqual(unionArea(canonical), 7);
  assert.strictEqual(unionAreaOracle(canonical), 7);

  assert.strictEqual(unionArea([[0, 0, 1, 1], [2, 2, 3, 3]]), 2, "disjoint");
  assert.strictEqual(unionArea([[0, 0, 10, 10], [2, 2, 4, 4]]), 100, "nested");
  assert.strictEqual(unionArea([[1, 1, 3, 3], [1, 1, 3, 3]]), 4, "identical");
  assert.strictEqual(unionArea([[0, 0, 2, 2], [2, 0, 4, 2]]), 8, "touching edge");
  assert.strictEqual(unionArea([[0, 0, 1, 1], [1, 1, 2, 2]]), 2, "touching corner");
  assert.strictEqual(unionArea([[5, 5, 5, 9]]), 0, "zero width");
  assert.strictEqual(unionArea([[5, 5, 5, 5]]), 0, "point rectangle");
  assert.strictEqual(unionArea([[0, 0, 3, 3], [1, 1, 1, 4]]), 9, "zero-area among valid");
  assert.strictEqual(unionArea([]), 0, "empty");
  assert.strictEqual(unionArea([[2, 3, 7, 9]]), 30, "single rectangle");
  assert.strictEqual(unionArea([[0, 0, 4, 4], [0, 0, 4, 4], [1, 1, 3, 3]]), 16, "repeats");

  // Triple overlap: y in [0,1) spans x 0..4 (length 4), y in [1,3) also 4, y in [3,4) 3.
  const triple = [[0, 0, 3, 3], [1, 0, 4, 3], [0, 1, 3, 4]];
  assert.strictEqual(unionArea(triple), 15, "triple union");
  assert.strictEqual(unionAreaOracle(triple), 15, "triple union oracle");

  // LCG cross-check, including degenerate rectangles and negative coordinates.
  const lcg = new Lcg(1618);
  for (let trial = 0; trial < 40; trial++) {
    const count = 1 + lcg.next(8);
    const modulus = 8 + lcg.next(10);
    const rectangles = [];
    for (let i = 0; i < count; i++) {
      const x1 = lcg.next(modulus);
      const width = lcg.next(Math.floor(modulus / 2));
      const y1 = lcg.next(modulus);
      const height = lcg.next(Math.floor(modulus / 2));
      rectangles.push([x1, y1, x1 + width, y1 + height]);
    }
    const offset = -lcg.next(7);
    const shifted = rectangles.map((r) => [r[0] + offset, r[1] + offset, r[2] + offset, r[3] + offset]);
    assert.strictEqual(unionArea(shifted), unionAreaOracle(shifted), `trial ${trial}`);
  }
  console.log("PASS");
}
