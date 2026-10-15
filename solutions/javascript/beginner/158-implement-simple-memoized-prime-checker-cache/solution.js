/** Task 158 - Implement simple memoized prime checker cache. (Tier: Beginner, CTCI Ch 6) */
"use strict";

/**
 * Trial division up to sqrt(n) with a verdict cache; hits/misses counters prove
 * that repeated queries for the same number are served from the cache.
 */
class PrimeChecker {
  constructor() {
    this.cache = new Map();
    this.hits = 0;
    this.misses = 0;
  }

  /** Memoized primality test: n < 2 is never prime. */
  isPrime(n) {
    if (this.cache.has(n)) {
      this.hits++;
      return this.cache.get(n);
    }
    this.misses++;
    const verdict = PrimeChecker.check(n);
    this.cache.set(n, verdict);
    return verdict;
  }

  get size() {
    return this.cache.size;
  }

  /** Trial division up to sqrt(n): O(sqrt(n)) time, O(1) space. */
  static check(n) {
    if (n < 2) return false;
    if (n % 2 === 0) return n === 2;
    for (let d = 3; d * d <= n; d += 2) {
      if (n % d === 0) return false;
    }
    return true;
  }
}

module.exports = { PrimeChecker };

if (require.main === module) {
  const assert = require("node:assert");
  const checker = new PrimeChecker();
  assert.strictEqual(checker.isPrime(97), true);
  assert.deepStrictEqual([checker.hits, checker.misses, checker.size], [0, 1, 1]);

  // repeated queries for the same number hit the cache
  assert.strictEqual(checker.isPrime(97), true);
  assert.strictEqual(checker.isPrime(97), true);
  assert.deepStrictEqual([checker.hits, checker.misses, checker.size], [2, 1, 1]);

  const cases = [[0, false], [1, false], [2, true], [100, false], [7919, true], [7917, false]];
  for (const [n, expected] of cases) assert.strictEqual(checker.isPrime(n), expected, `case ${n}`);
  assert.deepStrictEqual([checker.hits, checker.misses, checker.size], [2, 7, 7]);

  // repeat the whole set: every answer now comes from the cache
  const again = [[0, false], [1, false], [2, true], [97, true], [100, false], [7919, true], [7917, false]];
  for (const [n, expected] of again) assert.strictEqual(checker.isPrime(n), expected, `again ${n}`);
  assert.deepStrictEqual([checker.hits, checker.misses, checker.size], [9, 7, 7]);
  console.log("PASS");
}
