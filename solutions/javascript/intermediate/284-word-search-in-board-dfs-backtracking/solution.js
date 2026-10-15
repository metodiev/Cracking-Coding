/** Task 284 - Word search in a board with DFS backtracking. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

const DIRECTIONS = [[1, 0], [-1, 0], [0, 1], [0, -1]];

/**
 * Backtracking DFS: the word must be spelled by a 4-directionally adjacent path with no
 * cell reuse. A cell must be unmarked when a branch fails, otherwise a dead-end branch
 * can poison later branches (see the "ABCESEEEFS" trap case). Cross-checked against an
 * independent state-space BFS oracle over (cell, visited-mask) pairs.
 */
function exist(board, word) {
  if (!word) return false;
  const rows = board.length, cols = board[0].length;
  const visited = Array.from({ length: rows }, () => new Array(cols).fill(false));

  const search = (r, c, i) => {
    if (i === word.length) return true;
    if (r < 0 || r >= rows || c < 0 || c >= cols) return false;
    if (visited[r][c] || board[r][c] !== word[i]) return false;
    visited[r][c] = true;
    let found = false;
    for (const [dr, dc] of DIRECTIONS) {
      if (search(r + dr, c + dc, i + 1)) { found = true; break; }
    }
    visited[r][c] = false;                     // unmark: later branches may need this cell
    return found;
  };

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (search(r, c, 0)) return true;
    }
  }
  return false;
}

/** Deliberately never unmarks cells; only used to prove the trap case needs backtracking. */
function buggyExistWithoutUnmarking(board, word) {
  const rows = board.length, cols = board[0].length;
  const visited = Array.from({ length: rows }, () => new Array(cols).fill(false));

  const search = (r, c, i) => {
    if (i === word.length) return true;
    if (r < 0 || r >= rows || c < 0 || c >= cols) return false;
    if (visited[r][c] || board[r][c] !== word[i]) return false;
    visited[r][c] = true;                      // never unmarked
    for (const [dr, dc] of DIRECTIONS) {
      if (search(r + dr, c + dc, i + 1)) return true;
    }
    return false;
  };

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (search(r, c, 0)) return true;
    }
  }
  return false;
}

/** Independent BFS over states (row, col, visited-cell mask); popcount = prefix length. */
function oracleExist(board, word) {
  const rows = board.length, cols = board[0].length;
  const shift = rows * cols;
  const queue = [];
  const seen = new Set();
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (board[r][c] === word[0]) {
        const state = ((r * cols + c) << shift) | (1 << (r * cols + c));
        if (!seen.has(state)) {
          seen.add(state);
          queue.push([r, c, 1 << (r * cols + c)]);
        }
      }
    }
  }
  for (let head = 0; head < queue.length; head++) {
    const [r, c, mask] = queue[head];
    const i = popcount(mask);
    if (i === word.length) return true;
    for (const [dr, dc] of DIRECTIONS) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
      if ((mask >> (nr * cols + nc)) & 1) continue;
      if (board[nr][nc] !== word[i]) continue;
      const nextMask = mask | (1 << (nr * cols + nc));
      const state = ((nr * cols + nc) << shift) | nextMask;
      if (!seen.has(state)) {
        seen.add(state);
        queue.push([nr, nc, nextMask]);
      }
    }
  }
  return false;
}

function popcount(mask) {
  let count = 0;
  while (mask) {
    mask &= mask - 1;
    count++;
  }
  return count;
}

const rng = {
  state: 13579,
  next() {
    this.state = (this.state * 75 + 74) % 65537;
    return this.state;
  },
};

function randomBoard(rows, cols) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => "ABC"[rng.next() % 3]).join("")
  );
}

function randomCase(rows, cols, maxLen) {
  const board = randomBoard(rows, cols);
  const length = 1 + (rng.next() % maxLen);
  let word = "";
  for (let i = 0; i < length; i++) word += "ABC"[rng.next() % 3];
  return { board, word };
}

/** Reads a random simple path, so the word is guaranteed to exist. */
function plantedCase(rows, cols, maxLen) {
  const board = randomBoard(rows, cols);
  let r = rng.next() % rows, c = rng.next() % cols;
  const used = new Set([`${r},${c}`]);
  let word = board[r][c];
  const length = 1 + (rng.next() % maxLen);
  while (word.length < length) {
    const candidates = DIRECTIONS
      .map(([dr, dc]) => [r + dr, c + dc])
      .filter(([nr, nc]) => nr >= 0 && nr < rows && nc >= 0 && nc < cols && !used.has(`${nr},${nc}`));
    if (candidates.length === 0) break;
    [r, c] = candidates[rng.next() % candidates.length];
    used.add(`${r},${c}`);
    word += board[r][c];
  }
  return { board, word };
}

module.exports = { exist, buggyExistWithoutUnmarking, oracleExist, randomCase, plantedCase };

if (require.main === module) {
  const assert = require("node:assert");

  const classic = ["ABCE", "SFCS", "ADEE"];
  assert.strictEqual(exist(classic, "ABCCED"), true);
  assert.strictEqual(exist(classic, "SEE"), true);
  assert.strictEqual(exist(classic, "ABCB"), false);

  const trap = ["ABCE", "SFES", "ADEE"];       // needs the visited-unmarking
  assert.strictEqual(exist(trap, "ABCESEEEFS"), true);
  assert.strictEqual(buggyExistWithoutUnmarking(trap, "ABCESEEEFS"), false);

  assert.strictEqual(exist(["A"], "A"), true);          // single cell
  assert.strictEqual(exist(["A"], "B"), false);
  assert.strictEqual(exist(["A"], "AA"), false);        // a cell cannot be reused

  assert.strictEqual(exist(["AB", "CD"], "ABDC"), true);
  assert.strictEqual(exist(["AB", "CD"], "ABCD"), false);

  for (let trial = 0; trial < 150; trial++) {
    const { board, word } = randomCase(3, 3, 4);
    assert.strictEqual(exist(board, word), oracleExist(board, word), `random ${trial} ${word} ${board}`);
  }

  for (let trial = 0; trial < 50; trial++) {
    const { board, word } = plantedCase(3, 3, 5);
    assert.strictEqual(exist(board, word), true, `planted ${trial} ${word} ${board}`);
    assert.strictEqual(oracleExist(board, word), true, `planted oracle ${trial}`);
  }

  console.log("PASS");
}
