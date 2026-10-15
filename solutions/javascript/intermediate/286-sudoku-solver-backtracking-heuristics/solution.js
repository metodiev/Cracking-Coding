/** Task 286 - Sudoku solver with backtracking and constraint propagation. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

const FULL = 0x1FF;                          // bits 0..8 for digits 1..9

const PUZZLE = [
  [5, 3, 0, 0, 7, 0, 0, 0, 0],
  [6, 0, 0, 1, 9, 5, 0, 0, 0],
  [0, 9, 8, 0, 0, 0, 0, 6, 0],
  [8, 0, 0, 0, 6, 0, 0, 0, 3],
  [4, 0, 0, 8, 0, 3, 0, 0, 1],
  [7, 0, 0, 0, 2, 0, 0, 0, 6],
  [0, 6, 0, 0, 0, 0, 2, 8, 0],
  [0, 0, 0, 4, 1, 9, 0, 0, 5],
  [0, 0, 0, 0, 8, 0, 0, 7, 9],
];

const SOLUTION = [
  [5, 3, 4, 6, 7, 8, 9, 1, 2],
  [6, 7, 2, 1, 9, 5, 3, 4, 8],
  [1, 9, 8, 3, 4, 2, 5, 6, 7],
  [8, 5, 9, 7, 6, 1, 4, 2, 3],
  [4, 2, 6, 8, 5, 3, 7, 9, 1],
  [7, 1, 3, 9, 2, 4, 8, 5, 6],
  [9, 6, 1, 5, 3, 7, 2, 8, 4],
  [2, 8, 7, 4, 1, 9, 6, 3, 5],
  [3, 4, 5, 2, 8, 6, 1, 7, 9],
];

/**
 * Candidates come from row/column/box bitmasks and the search always branches on the
 * most-constrained empty cell (MRV). Returns the solved board, or null if unsolvable.
 */
function solveSudoku(board) {
  const grid = board.map((row) => [...row]);
  const rowsMask = new Array(9).fill(0);
  const colsMask = new Array(9).fill(0);
  const boxesMask = new Array(9).fill(0);

  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const value = grid[r][c];
      if (value < 0 || value > 9) return null;
      if (value === 0) continue;
      const bit = 1 << (value - 1);
      const box = ((r / 3) | 0) * 3 + ((c / 3) | 0);
      if (rowsMask[r] & bit || colsMask[c] & bit || boxesMask[box] & bit) return null; // duplicate clue
      rowsMask[r] |= bit;
      colsMask[c] |= bit;
      boxesMask[box] |= bit;
    }
  }

  const search = () => {
    let bestR = -1, bestC = -1, bestMask = 0, bestCount = 10;
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        if (grid[r][c] !== 0) continue;
        const box = ((r / 3) | 0) * 3 + ((c / 3) | 0);
        const mask = ~(rowsMask[r] | colsMask[c] | boxesMask[box]) & FULL;
        let count = 0;
        for (let v = 0; v < 9; v++) if ((mask >> v) & 1) count++;
        if (count === 0) return false;        // dead end found by propagation
        if (count < bestCount) {              // most-constrained cell
          bestCount = count;
          bestR = r;
          bestC = c;
          bestMask = mask;
        }
      }
    }
    if (bestR === -1) return true;            // no empty cell left
    const box = ((bestR / 3) | 0) * 3 + ((bestC / 3) | 0);
    for (let value = 1; value <= 9; value++) {
      const bit = 1 << (value - 1);
      if (!(bestMask & bit)) continue;
      grid[bestR][bestC] = value;
      rowsMask[bestR] |= bit;
      colsMask[bestC] |= bit;
      boxesMask[box] |= bit;
      if (search()) return true;
      grid[bestR][bestC] = 0;
      rowsMask[bestR] &= ~bit;
      colsMask[bestC] &= ~bit;
      boxesMask[box] &= ~bit;
    }
    return false;
  };

  return search() ? grid : null;
}

/** Every row, column and 3x3 box must contain the digits 1..9 exactly once. */
function isValidSolution(board) {
  const complete = (digits) => digits.slice().sort((a, b) => a - b).join(",") === "1,2,3,4,5,6,7,8,9";
  for (let r = 0; r < 9; r++) {
    if (!complete(Array.from({ length: 9 }, (_, c) => board[r][c]))) return false;
  }
  for (let c = 0; c < 9; c++) {
    if (!complete(Array.from({ length: 9 }, (_, r) => board[r][c]))) return false;
  }
  for (let boxR = 0; boxR < 9; boxR += 3) {
    for (let boxC = 0; boxC < 9; boxC += 3) {
      const digits = [];
      for (let dr = 0; dr < 3; dr++)
        for (let dc = 0; dc < 3; dc++) digits.push(board[boxR + dr][boxC + dc]);
      if (!complete(digits)) return false;
    }
  }
  return true;
}

const deepEquals = (a, b) => {
  if (a === null || b === null) return a === b;
  for (let r = 0; r < 9; r++)
    for (let c = 0; c < 9; c++)
      if (a[r][c] !== b[r][c]) return false;
  return true;
};

const rng = {
  state: 13579,
  next() {
    this.state = (this.state * 75 + 74) % 65537;
    return this.state;
  },
};

function randomPermutation(size) {
  const values = Array.from({ length: size }, (_, i) => i);
  for (let i = size - 1; i > 0; i--) {
    const j = rng.next() % (i + 1);
    [values[i], values[j]] = [values[j], values[i]];
  }
  return values;
}

function transform(source, digitMap, rowOrder, colOrder) {
  return Array.from({ length: 9 }, (_, r) =>
    Array.from({ length: 9 }, (_, c) => digitMap[source[rowOrder[r]][colOrder[c]]])
  );
}

module.exports = { solveSudoku, isValidSolution, PUZZLE, SOLUTION, deepEquals, transform, randomPermutation };

if (require.main === module) {
  const assert = require("node:assert");
  const copy = (board) => board.map((row) => [...row]);

  const solved = solveSudoku(PUZZLE);
  assert.deepStrictEqual(solved, SOLUTION);
  assert.ok(isValidSolution(solved));
  for (let r = 0; r < 9; r++)
    for (let c = 0; c < 9; c++)
      assert.ok(PUZZLE[r][c] === 0 || solved[r][c] === PUZZLE[r][c], `clue changed ${r},${c}`);

  assert.deepStrictEqual(solveSudoku(SOLUTION), SOLUTION);   // already solved input

  const duplicateInColumn = copy(SOLUTION);
  duplicateInColumn[0][0] = 6;              // column 0 already has a 6
  assert.strictEqual(solveSudoku(duplicateInColumn), null);

  const duplicateInRow = copy(SOLUTION);
  duplicateInRow[0][1] = 5;                 // row 0 already has a 5
  assert.strictEqual(solveSudoku(duplicateInRow), null);

  const duplicateInBox = copy(SOLUTION);
  duplicateInBox[1][1] = 5;                 // box 0 already has a 5
  assert.strictEqual(solveSudoku(duplicateInBox), null);

  const forcedConflict = Array.from({ length: 9 }, () => new Array(9).fill(0));
  forcedConflict[0] = [1, 2, 3, 4, 5, 6, 0, 8, 9];   // (0,6) can only be 7
  forcedConflict[3][6] = 7;                          // but column 6 already has a 7
  assert.strictEqual(solveSudoku(forcedConflict), null);

  const filled = solveSudoku(Array.from({ length: 9 }, () => new Array(9).fill(0)));
  assert.ok(filled !== null && isValidSolution(filled));

  for (let trial = 0; trial < 25; trial++) {
    const permutation = randomPermutation(9);
    const digitMap = [0, ...permutation.map((v) => v + 1)];
    const bands = randomPermutation(3);
    const stacks = randomPermutation(3);
    const rowOrder = [];
    const colOrder = [];
    for (let group = 0; group < 3; group++) {
      const inner = randomPermutation(3);
      for (let i = 0; i < 3; i++) rowOrder.push(bands[group] * 3 + inner[i]);
      const inner2 = randomPermutation(3);
      for (let i = 0; i < 3; i++) colOrder.push(stacks[group] * 3 + inner2[i]);
    }
    const puzzleVariant = transform(PUZZLE, digitMap, rowOrder, colOrder);
    const solutionVariant = transform(SOLUTION, digitMap, rowOrder, colOrder);
    assert.ok(isValidSolution(solutionVariant), `variant valid ${trial}`);
    assert.deepStrictEqual(solveSudoku(puzzleVariant), solutionVariant, `variant solved ${trial}`);
  }

  console.log("PASS");
}
