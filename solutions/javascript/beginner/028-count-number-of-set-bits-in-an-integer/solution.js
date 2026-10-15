/** Task 028 - Count number of set bits in an integer. (Tier: Beginner, CTCI Ch 5) */
function countSetBits(x) {
  let count = 0;
  x = x >>> 0;              // view as unsigned 32-bit
  while (x !== 0) { x &= x - 1; count++; }
  return count;
}

module.exports = { countSetBits };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countSetBits(0), 0);
  assert.strictEqual(countSetBits(255), 8);
  assert.strictEqual(countSetBits(0b101101), 4);
  console.log("PASS");
}
