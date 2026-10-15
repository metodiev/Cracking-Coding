/** Task 364 - Implement quadtree for 2D region queries. (Tier: Intermediate, CTCI Ch 4) */
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
 * Split rule: a node is a leaf when size == 1 or all of its cells are equal; otherwise
 * it splits into four quadrants of size / 2 and stores the total count of its region.
 */
class QuadTree {
  constructor(grid) {
    const size = grid.length;
    if (size === 0 || (size & (size - 1)) !== 0) {
      throw new Error("grid side must be a power of two");
    }
    if (grid.some((row) => row.length !== size)) throw new Error("grid must be square");
    this.grid = grid;
    this.root = this._build(0, 0, size);
  }

  _build(row, col, size) {
    const first = this.grid[row][col];
    let total = 0;
    let uniform = true;
    for (let r = row; r < row + size; r++) {
      for (let c = col; c < col + size; c++) {
        const value = this.grid[r][c];
        total += value;
        if (value !== first) uniform = false;
      }
    }
    if (uniform || size === 1) {
      return { row, col, size, count: total, value: first, children: null };
    }
    const half = size >> 1;
    return {
      row,
      col,
      size,
      count: total,
      value: null,
      children: [
        this._build(row, col, half),
        this._build(row, col + half, half),
        this._build(row + half, col, half),
        this._build(row + half, col + half, half),
      ],
    };
  }

  /** Number of ones in the inclusive rectangle [rowLo..rowHi] x [colLo..colHi]. */
  countRect(rowLo, rowHi, colLo, colHi) {
    return this._count(this.root, rowLo, rowHi, colLo, colHi);
  }

  _count(node, rowLo, rowHi, colLo, colHi) {
    if (node.row > rowHi || node.row + node.size - 1 < rowLo ||
        node.col > colHi || node.col + node.size - 1 < colLo) {
      return 0;                                          // disjoint
    }
    if (rowLo <= node.row && node.row + node.size - 1 <= rowHi &&
        colLo <= node.col && node.col + node.size - 1 <= colHi) {
      return node.count;                                 // fully contained
    }
    if (node.children === null) {                        // uniform leaf, partial overlap
      const overlapRows = Math.min(node.row + node.size - 1, rowHi) - Math.max(node.row, rowLo) + 1;
      const overlapCols = Math.min(node.col + node.size - 1, colHi) - Math.max(node.col, colLo) + 1;
      return node.value * overlapRows * overlapCols;
    }
    let total = 0;
    for (const child of node.children) {
      total += this._count(child, rowLo, rowHi, colLo, colHi);
    }
    return total;
  }
}

function naiveRectCount(grid, rowLo, rowHi, colLo, colHi) {
  let total = 0;
  for (let r = rowLo; r <= rowHi; r++) {
    for (let c = colLo; c <= colHi; c++) total += grid[r][c];
  }
  return total;
}

function lcgBinaryGrid(size, lcg) {
  const grid = [];
  for (let r = 0; r < size; r++) {
    const row = [];
    for (let c = 0; c < size; c++) row.push(lcg.next(2));
    grid.push(row);
  }
  return grid;
}

function diagonalGrid(size) {
  const grid = [];
  for (let r = 0; r < size; r++) {
    const row = [];
    for (let c = 0; c < size; c++) row.push(r === c ? 1 : 0);
    grid.push(row);
  }
  return grid;
}

module.exports = { QuadTree, naiveRectCount, lcgBinaryGrid, diagonalGrid, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-checked diagonal grid: ones only on the main diagonal.
  const tree = new QuadTree(diagonalGrid(8));
  assert.strictEqual(tree.countRect(0, 7, 0, 7), 8);
  assert.strictEqual(tree.countRect(0, 3, 0, 3), 4);
  assert.strictEqual(tree.countRect(2, 5, 2, 5), 4);
  assert.strictEqual(tree.countRect(1, 6, 0, 7), 6);
  assert.strictEqual(tree.countRect(0, 7, 7, 7), 1);
  assert.strictEqual(tree.countRect(0, 0, 1, 7), 0);
  assert.strictEqual(tree.countRect(3, 4, 3, 4), 2);
  assert.strictEqual(tree.countRect(2, 2, 0, 1), 0);

  // Uniform grids: all ones and all zeros.
  const ones = new QuadTree(Array.from({ length: 8 }, () => new Array(8).fill(1)));
  assert.strictEqual(ones.countRect(0, 7, 0, 7), 64);
  assert.strictEqual(ones.countRect(2, 4, 1, 3), 9);
  const zeros = new QuadTree(Array.from({ length: 8 }, () => new Array(8).fill(0)));
  assert.strictEqual(zeros.countRect(0, 7, 0, 7), 0);

  const cell = new QuadTree([[1]]);
  assert.strictEqual(cell.countRect(0, 0, 0, 0), 1);

  // 8x8 LCG grids: every possible rectangle against the naive loop.
  const lcg = new Lcg(4919);
  for (let trial = 0; trial < 8; trial++) {
    const grid = lcgBinaryGrid(8, lcg);
    const randomTree = new QuadTree(grid);
    for (let rowLo = 0; rowLo < 8; rowLo++) {
      for (let rowHi = rowLo; rowHi < 8; rowHi++) {
        for (let colLo = 0; colLo < 8; colLo++) {
          for (let colHi = colLo; colHi < 8; colHi++) {
            assert.strictEqual(randomTree.countRect(rowLo, rowHi, colLo, colHi),
              naiveRectCount(grid, rowLo, rowHi, colLo, colHi),
              `trial ${trial} rectangle ${rowLo},${rowHi},${colLo},${colHi}`);
          }
        }
      }
    }
  }

  // A 16x16 grid (deeper splits) with random rectangles.
  for (let trial = 0; trial < 3; trial++) {
    const grid = lcgBinaryGrid(16, lcg);
    const deepTree = new QuadTree(grid);
    for (let q = 0; q < 60; q++) {
      const rowLo = lcg.next(16);
      const rowHi = rowLo + lcg.next(16 - rowLo);
      const colLo = lcg.next(16);
      const colHi = colLo + lcg.next(16 - colLo);
      assert.strictEqual(deepTree.countRect(rowLo, rowHi, colLo, colHi),
        naiveRectCount(grid, rowLo, rowHi, colLo, colHi), `16x16 trial ${trial}`);
    }
  }
  console.log("PASS");
}
