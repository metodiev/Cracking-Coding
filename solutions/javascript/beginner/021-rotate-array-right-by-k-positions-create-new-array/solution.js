/** Task 021 - Rotate array right by k positions (create new array). (Tier: Beginner, CTCI Ch 1) */
function rotateRight(items, k) {
  if (items.length === 0) return [];
  k %= items.length;
  return [...items.slice(items.length - k), ...items.slice(0, items.length - k)];
}

module.exports = { rotateRight };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(rotateRight([1, 2, 3, 4, 5], 2), [4, 5, 1, 2, 3]);
  assert.deepStrictEqual(rotateRight([1, 2], 3), [2, 1]);
  console.log("PASS");
}
