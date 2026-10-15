/** Task 048 - Find most frequent element in array. (Tier: Beginner, CTCI Ch 1) */
function mostFrequent(items) {
  if (items.length === 0) throw new Error("empty array");
  const counts = new Map();
  for (const x of items) counts.set(x, (counts.get(x) ?? 0) + 1);
  let best = items[0], bestCount = 0;
  for (const x of items) {
    if (counts.get(x) > bestCount) { best = x; bestCount = counts.get(x); }
  }
  return best;
}

module.exports = { mostFrequent };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(mostFrequent([1, 3, 2, 3, 4, 3]), 3);
  assert.strictEqual(mostFrequent([1, 2, 1, 2]), 1);
  console.log("PASS");
}
