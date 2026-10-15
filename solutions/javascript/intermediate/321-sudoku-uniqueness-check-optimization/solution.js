/** Task 321 - Sudoku uniqueness check optimization. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

const FULL = 0x1ff; // nine candidate bits

function boxOf(cell) {
  return Math.floor(cell / 27) * 3 + Math.floor((cell % 9) / 3);
}

function peersOf(cell) {
  const peers = [];
  for (let other = 0; other < 81; other++) {
    if (other === cell) continue;
    const sameUnit =
      Math.floor(other / 9) === Math.floor(cell / 9) ||
      other % 9 === cell % 9 ||
      boxOf(other) === boxOf(cell);
    if (sameUnit) peers.push(other);
  }
  return peers;
}

const PEERS = Array.from({ length: 81 }, (_, cell) => peersOf(cell));

function bitCount(mask) {
  let count = 0;
  while (mask) {
    mask &= mask - 1;
    count++;
  }
  return count;
}

function search(board, limit) {
  const grid = board.slice();
  const rows = new Array(9).fill(0);
  const cols = new Array(9).fill(0);
  const boxes = new Array(9).fill(0);
  const trail = [];
  const found = [];

  const place = (cell, bit) => {
    rows[Math.floor(cell / 9)] |= bit;
    cols[cell % 9] |= bit;
    boxes[boxOf(cell)] |= bit;
    grid[cell] = 32 - Math.clz32(bit); // bit length
    trail.push([cell, bit]);
  };

  const undoTo = (mark) => {
    while (trail.length > mark) {
      const [cell, bit] = trail.pop();
      grid[cell] = 0;
      rows[Math.floor(cell / 9)] ^= bit;
      cols[cell % 9] ^= bit;
      boxes[boxOf(cell)] ^= bit;
    }
  };

  const candidates = (cell) =>
    FULL & ~(rows[Math.floor(cell / 9)] | cols[cell % 9] | boxes[boxOf(cell)]);

  // Naked singles: a peer left with a single candidate is forced; false on contradiction.
  const propagate = (seeds) => {
    const work = seeds.slice();
    while (work.length) {
      const cell = work.pop();
      for (const peer of PEERS[cell]) {
        if (grid[peer] !== 0) continue;
        const cand = candidates(peer);
        if (cand === 0) return false;
        if ((cand & (cand - 1)) === 0) {
          place(peer, cand);
          work.push(peer);
        }
      }
    }
    return true;
  };

  for (let cell = 0; cell < 81; cell++) {
    if (grid[cell] !== 0) {
      const bit = 1 << (grid[cell] - 1);
      if ((candidates(cell) & bit) === 0) return { count: 0, first: null };
      place(cell, bit);
    }
  }
  const allSeeds = Array.from({ length: 81 }, (_, i) => i);
  if (!propagate(allSeeds)) return { count: 0, first: null };

  const dive = () => {
    let bestCell = -1;
    let bestCand = 0;
    let bestSize = 10;
    for (let cell = 0; cell < 81; cell++) {
      if (grid[cell] !== 0) continue;
      const cand = candidates(cell);
      if (cand === 0) return; // dead end
      const size = bitCount(cand);
      if (size < bestSize) {
        bestCell = cell;
        bestCand = cand;
        bestSize = size;
        if (size === 2) break; // cannot do better
      }
    }
    if (bestCell === -1) {
      found.push(grid.slice());
      return;
    }
    while (bestCand) {
      const bit = bestCand & -bestCand;
      bestCand ^= bit;
      const mark = trail.length;
      place(bestCell, bit);
      if (propagate([bestCell])) dive();
      undoTo(mark);
      if (found.length >= limit) return;
    }
  };

  dive();
  return { count: found.length, first: found.length ? found[0] : null };
}

function countSolutions(board, limit = 2) {
  return search(board, limit).count;
}

function uniqueness(board) {
  return search(board, 2).count;
}

function firstSolution(board) {
  return search(board, 1).first;
}

/** The 27 units (rows, columns, boxes) as arrays of nine values. */
function units(board) {
  const result = [];
  for (let r = 0; r < 9; r++) {
    const unit = [];
    for (let c = 0; c < 9; c++) unit.push(board[r * 9 + c]);
    result.push(unit);
  }
  for (let c = 0; c < 9; c++) {
    const unit = [];
    for (let r = 0; r < 9; r++) unit.push(board[r * 9 + c]);
    result.push(unit);
  }
  for (let boxRow = 0; boxRow < 3; boxRow++) {
    for (let boxCol = 0; boxCol < 3; boxCol++) {
      const unit = [];
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) unit.push(board[(boxRow * 3 + r) * 9 + boxCol * 3 + c]);
      }
      result.push(unit);
    }
  }
  return result;
}

const PUZZLE = [
  5, 3, 0, 0, 7, 0, 0, 0, 0,
  6, 0, 0, 1, 9, 5, 0, 0, 0,
  0, 9, 8, 0, 0, 0, 0, 6, 0,
  8, 0, 0, 0, 6, 0, 0, 0, 3,
  4, 0, 0, 8, 0, 3, 0, 0, 1,
  7, 0, 0, 0, 2, 0, 0, 0, 6,
  0, 6, 0, 0, 0, 0, 2, 8, 0,
  0, 0, 0, 4, 1, 9, 0, 0, 5,
  0, 0, 0, 0, 8, 0, 0, 7, 9,
];

const KNOWN_SOLUTION = [
  5, 3, 4, 6, 7, 8, 9, 1, 2,
  6, 7, 2, 1, 9, 5, 3, 4, 8,
  1, 9, 8, 3, 4, 2, 5, 6, 7,
  8, 5, 9, 7, 6, 1, 4, 2, 3,
  4, 2, 6, 8, 5, 3, 7, 9, 1,
  7, 1, 3, 9, 2, 4, 8, 5, 6,
  9, 6, 1, 5, 3, 7, 2, 8, 4,
  2, 8, 7, 4, 1, 9, 6, 3, 5,
  3, 4, 5, 2, 8, 6, 1, 7, 9,
];

module.exports = { countSolutions, uniqueness, firstSolution, units, boxOf };

if (require.main === module) {
  const assert = require("node:assert");
  const empty = new Array(81).fill(0);

  // empty board: plenty of solutions, so the capped counter reports "many" (2)
  assert.strictEqual(countSolutions(empty), 2);
  assert.strictEqual(uniqueness(empty), 2);
  assert.strictEqual(countSolutions(empty, 4), 4);

  // proper puzzle: exactly one completion; with limit 2 the whole tree is explored
  assert.strictEqual(countSolutions(PUZZLE, 2), 1);
  assert.strictEqual(uniqueness(PUZZLE), 1);
  const solution = firstSolution(PUZZLE);
  assert.deepStrictEqual(solution, KNOWN_SOLUTION);
  for (const unit of units(solution)) {
    assert.deepStrictEqual(unit.slice().sort((a, b) => a - b), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  }

  // unsatisfiable board: no completion, and it is not a trivially conflicting one
  const unsat = PUZZLE.slice();
  unsat[1] = 1;
  for (const unit of units(unsat)) {
    const givens = unit.filter((v) => v !== 0);
    assert.strictEqual(new Set(givens).size, givens.length);
  }
  assert.strictEqual(countSolutions(unsat, 2), 0);
  assert.strictEqual(uniqueness(unsat), 0);
  assert.strictEqual(firstSolution(unsat), null);

  // a plainly conflicting board is rejected while loading the givens
  const conflicting = PUZZLE.slice();
  conflicting[0] = 6; // two 6s in the top-left box
  assert.strictEqual(countSolutions(conflicting, 2), 0);

  // sparse valid board (a few clues of the known solution) has multiple completions
  const sparse = new Array(81).fill(0);
  for (let index = 0; index < 81; index += 10) sparse[index] = KNOWN_SOLUTION[index];
  assert.strictEqual(uniqueness(sparse), 2);

  console.log("PASS");
}
