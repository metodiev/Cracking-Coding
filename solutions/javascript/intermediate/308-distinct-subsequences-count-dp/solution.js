/** Task 308 - Count distinct subsequences. (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function countDistinctSubsequences(s, t) {
  const n = s.length;
  const m = t.length;
  if (m === 0) return 1n;
  if (n < m) return 0n;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0n));
  for (let i = 0; i <= n; i++) dp[i][0] = 1n;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i][j] = dp[i - 1][j];
      if (s[i - 1] === t[j - 1]) dp[i][j] += dp[i - 1][j - 1];
    }
  }
  return dp[n][m];
}

function bruteForce(s, t) {
  let count = 0;
  for (let mask = 0; mask < 1 << s.length; mask++) {
    let candidate = "";
    for (let i = 0; i < s.length; i++) if ((mask >> i) & 1) candidate += s[i];
    if (candidate === t) count++;
  }
  return BigInt(count);
}

function lcgStrings(length, count, seed = 8642) {
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

module.exports = { countDistinctSubsequences, bruteForce, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countDistinctSubsequences("rabbbit", "rabbit"), 3n);
  assert.strictEqual(countDistinctSubsequences("babgbag", "bag"), 5n);
  assert.strictEqual(countDistinctSubsequences("abc", ""), 1n);
  assert.strictEqual(countDistinctSubsequences("", "a"), 0n);
  assert.strictEqual(countDistinctSubsequences("aaa", "a"), 3n);
  assert.strictEqual(countDistinctSubsequences("aaa", "aa"), 3n);
  assert.strictEqual(countDistinctSubsequences("abc", "abcd"), 0n);

  const strings = lcgStrings(8, 40);
  for (let i = 0; i + 1 < strings.length; i += 2) {
    const s = strings[i];
    const t = strings[i + 1].slice(0, 4);
    assert.strictEqual(countDistinctSubsequences(s, t), bruteForce(s, t), `${s}/${t}`);
  }
  console.log("PASS");
}
