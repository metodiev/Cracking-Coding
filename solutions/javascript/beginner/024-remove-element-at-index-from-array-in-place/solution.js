/** Task 024 - Remove element at index from array (in-place). (Tier: Beginner, CTCI Ch 1) */
function removeAt(items, index) {
  if (index < 0 || index >= items.length) throw new RangeError("index out of range");
  for (let i = index; i + 1 < items.length; i++) items[i] = items[i + 1];
  return items.length - 1;
}

module.exports = { removeAt };

if (require.main === module) {
  const assert = require("node:assert");
  const a = [10, 20, 30, 40];
  assert.strictEqual(removeAt(a, 1), 3);
  assert.deepStrictEqual(a.slice(0, 3), [10, 30, 40]);
  console.log("PASS");
}
