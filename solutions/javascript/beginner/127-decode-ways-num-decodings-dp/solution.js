/** Task 127 - Decode ways (num decodings) DP. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function numDecodings(s) {
  if (s.length === 0) return 0;
  const dp = new Array(s.length + 1).fill(0);
  dp[0] = 1;
  dp[1] = s[0] === "0" ? 0 : 1;
  for (let i = 2; i <= s.length; i++) {
    const one = Number(s[i - 1]);
    const two = Number(s.slice(i - 2, i));
    if (one >= 1 && one <= 9) dp[i] += dp[i - 1];
    if (two >= 10 && two <= 26) dp[i] += dp[i - 2];
  }
  return dp[s.length];
}

module.exports = { numDecodings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(numDecodings("12"), 2);
  assert.strictEqual(numDecodings("226"), 3);
  assert.strictEqual(numDecodings("06"), 0);
  assert.strictEqual(numDecodings("10"), 1);
  assert.strictEqual(numDecodings("27"), 1);
  assert.strictEqual(numDecodings("11106"), 2);
  console.log("PASS");
}
