/** Task 029 - Check if integer is power of two. (Tier: Beginner, CTCI Ch 5) */
function isPowerOfTwo(x) {
  return x > 0 && (x & (x - 1)) === 0;
}

module.exports = { isPowerOfTwo };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isPowerOfTwo(1), true);
  assert.strictEqual(isPowerOfTwo(1024), true);
  assert.strictEqual(isPowerOfTwo(0), false);
  assert.strictEqual(isPowerOfTwo(12), false);
  console.log("PASS");
}
