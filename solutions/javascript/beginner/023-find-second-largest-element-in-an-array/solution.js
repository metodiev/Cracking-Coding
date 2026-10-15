/** Task 023 - Find second largest element in an array. (Tier: Beginner, CTCI Ch 1) */
function secondLargest(items) {
  let best = null, second = null;
  for (const x of items) {
    if (best === null || x > best) { second = best; best = x; }
    else if (x !== best && (second === null || x > second)) second = x;
  }
  if (second === null) throw new Error("no second largest");
  return second;
}

module.exports = { secondLargest };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(secondLargest([3, 7, 2, 9, 4]), 7);
  assert.strictEqual(secondLargest([5, 5, 4]), 4);
  assert.strictEqual(secondLargest([-1, -2, -3]), -2);
  console.log("PASS");
}
