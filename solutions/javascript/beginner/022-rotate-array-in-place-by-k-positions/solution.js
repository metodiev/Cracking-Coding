/** Task 022 - Rotate array in-place by k positions. (Tier: Beginner, CTCI Ch 1) */
function rotateInPlace(items, k) {
  const n = items.length;
  if (n === 0) return items;
  k %= n;
  const reverse = (lo, hi) => {
    while (lo < hi) { [items[lo], items[hi]] = [items[hi], items[lo]]; lo++; hi--; }
  };
  reverse(0, n - 1);
  reverse(0, k - 1);
  reverse(k, n - 1);
  return items;
}

module.exports = { rotateInPlace };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(rotateInPlace([1, 2, 3, 4, 5], 2), [4, 5, 1, 2, 3]);
  assert.deepStrictEqual(rotateInPlace([1, 2], 0), [1, 2]);
  console.log("PASS");
}
