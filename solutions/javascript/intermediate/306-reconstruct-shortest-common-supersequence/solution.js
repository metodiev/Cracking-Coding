/** Task 306 - Reconstruct the shortest common supersequence. (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function shortestCommonSupersequence(a, b) {
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

  const chars = [];
  let i = n;
  let j = m;
  while (i > 0 && j > 0) {
    if (a[i - 1] === b[j - 1]) {
      chars.push(a[i - 1]);
      i--;
      j--;
    } else if (dp[i - 1][j] <= dp[i][j - 1]) {
      chars.push(a[i - 1]);
      i--;
    } else {
      chars.push(b[j - 1]);
      j--;
    }
  }
  while (i > 0) chars.push(a[--i]);
  while (j > 0) chars.push(b[--j]);
  return chars.reverse().join("");
}

function isSubsequence(candidate, text) {
  let idx = 0;
  for (let i = 0; i < text.length && idx < candidate.length; i++) {
    if (text[i] === candidate[idx]) idx++;
  }
  return idx === candidate.length;
}

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

function lcgStrings(length, count, seed = 3141) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += "ab"[seed % 2];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { shortestCommonSupersequence, isSubsequence, lcsLength, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(shortestCommonSupersequence("ab", "b"), "ab");
  assert.strictEqual(shortestCommonSupersequence("abc", "abc"), "abc");
  assert.strictEqual(shortestCommonSupersequence("", "abc"), "abc");
  assert.strictEqual(shortestCommonSupersequence("abc", ""), "abc");
  assert.strictEqual(shortestCommonSupersequence("", ""), "");

  const [x, y] = ["AGGTAB", "GXTXAYB"];
  const scs = shortestCommonSupersequence(x, y);
  assert.strictEqual(scs.length, 9);
  assert.strictEqual(scs.length, x.length + y.length - lcsLength(x, y));
  assert.ok(isSubsequence(x, scs) && isSubsequence(y, scs));

  const strings = lcgStrings(5, 40);
  for (let i = 0; i + 1 < strings.length; i += 2) {
    const a = strings[i];
    const b = strings[i + 1];
    const candidate = shortestCommonSupersequence(a, b);
    assert.strictEqual(candidate.length, a.length + b.length - lcsLength(a, b), `${a}/${b}`);
    assert.ok(isSubsequence(a, candidate) && isSubsequence(b, candidate), `${a}/${b}`);
  }
  console.log("PASS");
}
