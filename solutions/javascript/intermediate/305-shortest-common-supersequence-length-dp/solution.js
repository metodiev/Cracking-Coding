/** Task 305 - Shortest common supersequence length (DP). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function lcsLength(a, b) {
  let previous = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    const current = new Array(b.length + 1).fill(0);
    for (let j = 1; j <= b.length; j++) {
      current[j] = a[i - 1] === b[j - 1] ? previous[j - 1] + 1 : Math.max(previous[j], current[j - 1]);
    }
    previous = current;
  }
  return previous[b.length];
}

function scsLength(a, b) {
  const n = a.length;
  const m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.min(dp[i - 1][j], dp[i][j - 1]) + 1;
    }
  }
  return dp[n][m];
}

function lcgStrings(length, count, seed = 4682) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += "abc"[seed % 3];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { lcsLength, scsLength, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(scsLength("AGGTAB", "GXTXAYB"), 9);
  assert.strictEqual(scsLength("ab", "b"), 2);
  assert.strictEqual(scsLength("", "abc"), 3);
  assert.strictEqual(scsLength("abc", ""), 3);
  assert.strictEqual(scsLength("", ""), 0);
  assert.strictEqual(scsLength("abc", "abc"), 3);
  assert.strictEqual(scsLength("geek", "eke"), 5);

  for (const [a, b] of [["AGGTAB", "GXTXAYB"], ["geek", "eke"], ["ab", "b"]]) {
    assert.strictEqual(scsLength(a, b), a.length + b.length - lcsLength(a, b), `${a}/${b}`);
  }
  const strings = lcgStrings(6, 40);
  for (let i = 0; i + 1 < strings.length; i += 2) {
    const a = strings[i];
    const b = strings[i + 1];
    assert.strictEqual(scsLength(a, b), a.length + b.length - lcsLength(a, b), `${a}/${b}`);
    assert.ok(scsLength(a, b) >= Math.max(a.length, b.length));
  }
  console.log("PASS");
}
