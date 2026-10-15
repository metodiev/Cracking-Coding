/** Task 297 - Edit distance variants (multi-dimensional DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function editDistance(a, b, insertCost = 1, deleteCost = 1, replaceCost = 1) {
  const n = a.length;
  const m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 1; i <= n; i++) dp[i][0] = i * deleteCost;
  for (let j = 1; j <= m; j++) dp[0][j] = j * insertCost;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const substitution = a[i - 1] === b[j - 1] ? 0 : replaceCost;
      dp[i][j] = Math.min(
        dp[i - 1][j - 1] + substitution,
        dp[i - 1][j] + deleteCost,
        dp[i][j - 1] + insertCost
      );
    }
  }
  return dp[n][m];
}

function bruteEdit(a, b, insertCost, deleteCost, replaceCost) {
  const memo = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(-1));
  const rec = (i, j) => {
    if (i === 0) return j * insertCost;
    if (j === 0) return i * deleteCost;
    if (memo[i][j] >= 0) return memo[i][j];
    const substitution = a[i - 1] === b[j - 1] ? 0 : replaceCost;
    const best = Math.min(
      rec(i - 1, j - 1) + substitution,
      rec(i - 1, j) + deleteCost,
      rec(i, j - 1) + insertCost
    );
    memo[i][j] = best;
    return best;
  };
  return rec(a.length, b.length);
}

function lcgString(maxLength, seed) {
  seed = (seed * 75 + 74) % 65537;
  const length = seed % (maxLength + 1);
  let text = "";
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    text += "abc"[seed % 3];
  }
  return [text, seed];
}

module.exports = { editDistance, bruteEdit, lcgString };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(editDistance("kitten", "sitting"), 3);
  assert.strictEqual(editDistance("flaw", "lawn"), 2);
  assert.strictEqual(editDistance("abc", "abc"), 0);
  assert.strictEqual(editDistance("", "abcd"), 4);
  assert.strictEqual(editDistance("abcd", ""), 4);
  assert.strictEqual(editDistance("", ""), 0);

  assert.strictEqual(editDistance("ab", "ba", 1, 1, 2), 2);
  assert.strictEqual(editDistance("abc", "axc", 1, 1, 3), 2);
  assert.strictEqual(editDistance("abc", "axc", 1, 1, 1), 1);

  assert.strictEqual(editDistance("abc", "abcd", 1, 5, 1), 1);
  assert.strictEqual(editDistance("abcd", "abc", 1, 5, 1), 5);
  assert.strictEqual(editDistance("abc", "abcd", 7, 1, 1), 7);

  let seed = 31337;
  for (let trial = 0; trial < 200; trial++) {
    let a;
    let b;
    [a, seed] = lcgString(6, seed);
    [b, seed] = lcgString(6, seed);
    seed = (seed * 75 + 74) % 65537;
    const insertCost = 1 + (seed % 3);
    seed = (seed * 75 + 74) % 65537;
    const deleteCost = 1 + (seed % 3);
    seed = (seed * 75 + 74) % 65537;
    const replaceCost = 1 + (seed % 3);
    assert.strictEqual(
      editDistance(a, b, insertCost, deleteCost, replaceCost),
      bruteEdit(a, b, insertCost, deleteCost, replaceCost),
      `trial ${trial}`
    );
  }
  console.log("PASS");
}
