/** Task 285 - N-Queens via backtracking. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/**
 * Queens are placed row by row (one per row) with columns tried 0..n-1, skipping
 * attacked columns and diagonals. Solutions are therefore emitted in lexicographic
 * order, so the first solution is deterministic. Counts are asserted against the known
 * sequence and re-derived with an independent permutation oracle.
 */
function solveNQueens(n) {
  const solutions = [];
  const columns = [];

  const place = (row) => {
    if (row === n) {
      solutions.push([...columns]);
      return;
    }
    for (let col = 0; col < n; col++) {
      let ok = true;
      for (let pr = 0; pr < columns.length; pr++) {
        const pc = columns[pr];
        if (col === pc || Math.abs(col - pc) === row - pr) { ok = false; break; }
      }
      if (ok) {
        columns.push(col);
        place(row + 1);
        columns.pop();
      }
    }
  };

  place(0);
  return solutions;
}

const countSolutions = (n) => solveNQueens(n).length;

/** Lexicographically first solution, or null when no placement exists. */
function firstSolution(n) {
  const solutions = solveNQueens(n);
  return solutions.length ? solutions[0] : null;
}

/** One queen per row is structural; check columns are a permutation and no shared diagonal. */
function isValidSolution(solution) {
  const n = solution.length;
  const seen = new Array(n).fill(false);
  for (const col of solution) {
    if (col < 0 || col >= n || seen[col]) return false;
    seen[col] = true;
  }
  for (let r = 0; r < n; r++)
    for (let s = r + 1; s < n; s++)
      if (Math.abs(solution[r] - solution[s]) === s - r) return false;
  return true;
}

/** Permutation oracle: count column permutations with no shared diagonal. */
function bruteForceCount(n) {
  const permutation = new Array(n).fill(0);
  const used = new Array(n).fill(false);
  const countPermutations = (row) => {
    if (row === n) {
      for (let r = 0; r < n; r++)
        for (let s = r + 1; s < n; s++)
          if (Math.abs(permutation[r] - permutation[s]) === s - r) return 0;
      return 1;
    }
    let total = 0;
    for (let col = 0; col < n; col++) {
      if (used[col]) continue;
      used[col] = true;
      permutation[row] = col;
      total += countPermutations(row + 1);
      used[col] = false;
    }
    return total;
  };
  return countPermutations(0);
}

const lexLess = (a, b) => {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return a[i] < b[i];
  }
  return false;
};

module.exports = { solveNQueens, countSolutions, firstSolution, isValidSolution, bruteForceCount };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(countSolutions(1), 1);
  assert.strictEqual(countSolutions(2), 0);
  assert.strictEqual(countSolutions(3), 0);
  assert.strictEqual(countSolutions(4), 2);
  assert.strictEqual(countSolutions(5), 10);
  assert.strictEqual(countSolutions(6), 4);
  assert.strictEqual(countSolutions(7), 40);
  assert.strictEqual(countSolutions(8), 92);

  assert.deepStrictEqual(firstSolution(1), [0]);
  assert.deepStrictEqual(firstSolution(4), [1, 3, 0, 2]);   // hand-verified lexicographic first
  assert.strictEqual(firstSolution(2), null);
  assert.strictEqual(firstSolution(3), null);

  for (let n = 1; n <= 8; n++) {
    const solutions = solveNQueens(n);
    assert.strictEqual(solutions.length, countSolutions(n));
    for (let i = 1; i < solutions.length; i++) {
      assert.ok(!lexLess(solutions[i], solutions[i - 1]), `lexicographic order n=${n}`);
    }
    for (const solution of solutions) {
      assert.ok(isValidSolution(solution), `valid n=${n} ${solution}`);
    }
    if (solutions.length) assert.deepStrictEqual(firstSolution(n), solutions[0]);
    assert.strictEqual(bruteForceCount(n), solutions.length, `oracle n=${n}`);
  }

  console.log("PASS");
}
