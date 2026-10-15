/** Task 039 - Simple linear congruential pseudorandom generator. (Tier: Beginner, CTCI Ch 6) */
function lcg(seed, n, a = 1103515245n, c = 12345n, m = 1n << 31n) {
  let x = BigInt(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    x = (a * x + c) % m;
    out.push(x);
  }
  return out;
}

module.exports = { lcg };

if (require.main === module) {
  const assert = require("node:assert");
  const seq = lcg(42, 5);
  assert.strictEqual(seq.length, 5);
  assert.deepStrictEqual(seq, lcg(42, 5));
  assert.notDeepStrictEqual(seq, lcg(43, 5));
  console.log("PASS");
}
