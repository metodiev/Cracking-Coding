/** Task 026 - Check if array contains any duplicates (boolean). (Tier: Beginner, CTCI Ch 1) */
function containsDuplicates(items) {
  const seen = new Set();
  for (const x of items) {
    if (seen.has(x)) return true;
    seen.add(x);
  }
  return false;
}

module.exports = { containsDuplicates };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(containsDuplicates([1, 2, 3, 1]), true);
  assert.strictEqual(containsDuplicates([1, 2, 3]), false);
  console.log("PASS");
}
