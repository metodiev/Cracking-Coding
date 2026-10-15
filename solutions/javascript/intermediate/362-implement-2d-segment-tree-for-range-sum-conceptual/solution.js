/** Task 362 - Implement 2D segment tree for range sum (conceptual). (Tier: Intermediate, CTCI Ch 4) */
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

function nextPowerOfTwo(value) {
  let size = 1;
  while (size < value) size <<= 1;
  return size;
}

/**
 * A segment tree over rows whose every node stores an inner segment tree over columns;
 * internal outer nodes are the element-wise sum of their two children.
 */
class SegmentTree2D {
  constructor(grid) {
    const rows = grid.length;
    const cols = rows === 0 ? 0 : grid[0].length;
    this.sizeR = nextPowerOfTwo(Math.max(rows, 1));
    this.sizeC = nextPowerOfTwo(Math.max(cols, 1));
    this.trees = [];
    for (let i = 0; i < 2 * this.sizeR; i++) {
      this.trees.push(new Array(2 * this.sizeC).fill(0));
    }
    for (let r = 0; r < rows; r++) {
      const rowTree = this.trees[this.sizeR + r];
      for (let c = 0; c < cols; c++) rowTree[this.sizeC + c] = grid[r][c];
      for (let node = this.sizeC - 1; node >= 1; node--) {
        rowTree[node] = rowTree[2 * node] + rowTree[2 * node + 1];
      }
    }
    for (let node = this.sizeR - 1; node >= 1; node--) {
      const left = this.trees[2 * node];
      const right = this.trees[2 * node + 1];
      const merged = this.trees[node];
      for (let j = 1; j < 2 * this.sizeC; j++) merged[j] = left[j] + right[j];
    }
  }

  /** Sum over the inclusive rectangle [rowLo..rowHi] x [colLo..colHi]. */
  rangeSum(rowLo, rowHi, colLo, colHi) {
    if (rowLo > rowHi || colLo > colHi) return 0;
    let total = 0;
    let l = rowLo + this.sizeR;
    let r = rowHi + 1 + this.sizeR;
    while (l < r) {
      if (l & 1) total += this.innerSum(this.trees[l++], colLo, colHi + 1);
      if (r & 1) total += this.innerSum(this.trees[--r], colLo, colHi + 1);
      l >>= 1;
      r >>= 1;
    }
    return total;
  }

  innerSum(tree, colLo, colHi) {
    let total = 0;
    let l = colLo + this.sizeC;
    let r = colHi + this.sizeC;
    while (l < r) {
      if (l & 1) total += tree[l++];
      if (r & 1) total += tree[--r];
      l >>= 1;
      r >>= 1;
    }
    return total;
  }
}

function naiveRectSum(grid, rowLo, rowHi, colLo, colHi) {
  let total = 0;
  for (let r = rowLo; r <= rowHi; r++) {
    for (let c = colLo; c <= colHi; c++) total += grid[r][c];
  }
  return total;
}

function lcgGrid(rows, cols, lcg) {
  const grid = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) row.push(lcg.next(20) - 5);
    grid.push(row);
  }
  return grid;
}

module.exports = { SegmentTree2D, naiveRectSum, lcgGrid, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-checked sums on the 4x4 grid 1..16.
  const hand = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
  ];
  const tree = new SegmentTree2D(hand);
  assert.strictEqual(tree.rangeSum(0, 3, 0, 3), 136);
  assert.strictEqual(tree.rangeSum(0, 1, 0, 1), 14);
  assert.strictEqual(tree.rangeSum(2, 3, 1, 2), 50);
  assert.strictEqual(tree.rangeSum(1, 1, 0, 3), 26);
  assert.strictEqual(tree.rangeSum(0, 3, 0, 0), 28);
  assert.strictEqual(tree.rangeSum(2, 2, 3, 3), 12);
  assert.strictEqual(tree.rangeSum(3, 0, 0, 3), 0);
  assert.strictEqual(tree.rangeSum(0, 3, 3, 2), 0);

  // Degenerate shapes: one cell, one row, one column.
  const single = new SegmentTree2D([[7]]);
  assert.strictEqual(single.rangeSum(0, 0, 0, 0), 7);
  const row = new SegmentTree2D([[1, -2, 3, 4, 5]]);
  assert.strictEqual(row.rangeSum(0, 0, 0, 4), 11);
  assert.strictEqual(row.rangeSum(0, 0, 1, 3), 5);
  const col = new SegmentTree2D([[1], [2], [3], [4], [5]]);
  assert.strictEqual(col.rangeSum(0, 4, 0, 0), 15);
  assert.strictEqual(col.rangeSum(2, 4, 0, 0), 12);

  // 8x8 LCG grids: every rectangle cross-checked against the naive double loop.
  const lcg = new Lcg(31337);
  for (let trial = 0; trial < 6; trial++) {
    const grid = lcgGrid(8, 8, lcg);
    const randomTree = new SegmentTree2D(grid);
    for (let rowLo = 0; rowLo < 8; rowLo++) {
      for (let rowHi = rowLo; rowHi < 8; rowHi++) {
        for (let colLo = 0; colLo < 8; colLo++) {
          for (let colHi = colLo; colHi < 8; colHi++) {
            assert.strictEqual(randomTree.rangeSum(rowLo, rowHi, colLo, colHi),
              naiveRectSum(grid, rowLo, rowHi, colLo, colHi),
              `trial ${trial} rectangle ${rowLo},${rowHi},${colLo},${colHi}`);
          }
        }
      }
    }
  }

  // Non power-of-two shapes (3x5, 5x3, 7x1) still work.
  for (const [rows, cols] of [[3, 5], [5, 3], [7, 1]]) {
    const grid = lcgGrid(rows, cols, lcg);
    const shapeTree = new SegmentTree2D(grid);
    for (let rowLo = 0; rowLo < rows; rowLo++) {
      for (let rowHi = rowLo; rowHi < rows; rowHi++) {
        for (let colLo = 0; colLo < cols; colLo++) {
          for (let colHi = colLo; colHi < cols; colHi++) {
            assert.strictEqual(shapeTree.rangeSum(rowLo, rowHi, colLo, colHi),
              naiveRectSum(grid, rowLo, rowHi, colLo, colHi), `shape ${rows}x${cols}`);
          }
        }
      }
    }
  }
  console.log("PASS");
}
