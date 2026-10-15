/** Task 171 - Minimum cuts for palindrome partitioning (DP). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function minPalindromeCuts(s) {
  const n = s.length;
  if (n <= 1) return 0;
  const isPal = Array.from({ length: n }, () => new Array(n).fill(false));
  const cuts = new Array(n).fill(0);
  for (let end = 0; end < n; end++) {
    let best = end;
    for (let start = 0; start <= end; start++) {
      if (s[start] === s[end] && (end - start <= 1 || isPal[start + 1][end - 1])) {
        isPal[start][end] = true;
        best = start === 0 ? 0 : Math.min(best, cuts[start - 1] + 1);
      }
    }
    cuts[end] = best;
  }
  return cuts[n - 1];
}

module.exports = { minPalindromeCuts };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(minPalindromeCuts("aab"), 1);
  assert.strictEqual(minPalindromeCuts("a"), 0);
  assert.strictEqual(minPalindromeCuts(""), 0);
  assert.strictEqual(minPalindromeCuts("ab"), 1);
  assert.strictEqual(minPalindromeCuts("aabb"), 1);
  assert.strictEqual(minPalindromeCuts("abcba"), 0);
  console.log("PASS");
}
