/** Task 276 - Implement the Hungarian algorithm for assignment (basic). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function hungarian(cost) {
  const n = cost.length;
  if (n === 0) return { total: 0, assignment: [] };
  const infinity = Number.POSITIVE_INFINITY;
  const u = new Array(n + 1).fill(0);
  const v = new Array(n + 1).fill(0);
  const p = new Array(n + 1).fill(0);
  const way = new Array(n + 1).fill(0);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(n + 1).fill(infinity);
    const used = new Array(n + 1).fill(false);
    for (;;) {
      used[j0] = true;
      const i0 = p[j0];
      let delta = infinity;
      let j1 = -1;
      for (let j = 1; j <= n; j++) {
        if (!used[j]) {
          const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else minv[j] -= delta;
      }
      j0 = j1;
      if (p[j0] === 0) break;
    }
    while (j0 !== 0) {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    }
  }

  const assignment = new Array(n).fill(-1);
  for (let j = 1; j <= n; j++) if (p[j] !== 0) assignment[p[j] - 1] = j - 1;
  let total = 0;
  for (let i = 0; i < n; i++) total += cost[i][assignment[i]];
  return { total, assignment };
}

function bruteForce(cost) {
  const n = cost.length;
  const perm = Array.from({ length: n }, (_, i) => i);
  let best = Infinity;
  const permute = (start) => {
    if (start === n) {
      let total = 0;
      for (let i = 0; i < n; i++) total += cost[i][perm[i]];
      best = Math.min(best, total);
      return;
    }
    for (let i = start; i < n; i++) {
      [perm[start], perm[i]] = [perm[i], perm[start]];
      permute(start + 1);
      [perm[start], perm[i]] = [perm[i], perm[start]];
    }
  };
  permute(0);
  return best;
}

function lcgMatrices(sizes, perSize, seed = 4242) {
  const matrices = [];
  for (const n of sizes) {
    for (let k = 0; k < perSize; k++) {
      const matrix = [];
      for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
          seed = (seed * 75 + 74) % 65537;
          row.push(seed % 20);
        }
        matrix.push(row);
      }
      matrices.push(matrix);
    }
  }
  return matrices;
}

module.exports = { hungarian, bruteForce, lcgMatrices };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(hungarian([[1, 2], [3, 4]]).total, 5);
  assert.deepStrictEqual(hungarian([[1, 2], [3, 4]]).assignment, [0, 1]);
  assert.strictEqual(hungarian([[10, 19], [20, 11]]).total, 21);
  assert.strictEqual(hungarian([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).total, 15);
  assert.deepStrictEqual(hungarian([[7]]).assignment, [0]);
  assert.strictEqual(hungarian([]).total, 0);
  assert.strictEqual(hungarian([[9, 1, 1], [1, 9, 1], [1, 1, 9]]).total, 3);

  let trial = 0;
  for (const matrix of lcgMatrices([2, 3, 4, 5], 15)) {
    const { total, assignment } = hungarian(matrix);
    assert.deepStrictEqual([...assignment].sort((a, b) => a - b), assignment.map((_, i) => i));
    assert.strictEqual(total, assignment.reduce((acc, column, i) => acc + matrix[i][column], 0));
    assert.strictEqual(total, bruteForce(matrix), `trial ${trial}`);
    trial++;
  }
  console.log("PASS");
}
