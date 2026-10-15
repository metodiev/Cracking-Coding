/** Task 134 - Next lexicographical permutation. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function nextPermutation(a) {
  const n = a.length;
  let pivot = n - 2;
  while (pivot >= 0 && a[pivot] >= a[pivot + 1]) pivot--;
  if (pivot < 0) {
    a.reverse();
    return false;
  }
  let successor = n - 1;
  while (a[successor] <= a[pivot]) successor--;
  [a[pivot], a[successor]] = [a[successor], a[pivot]];
  for (let i = pivot + 1, j = n - 1; i < j; i++, j--) {
    [a[i], a[j]] = [a[j], a[i]];
  }
  return true;
}

module.exports = { nextPermutation };

if (require.main === module) {
  const assert = require("node:assert");
  const a = [1, 2, 3];
  assert.strictEqual(nextPermutation(a), true);
  assert.deepStrictEqual(a, [1, 3, 2]);
  assert.strictEqual(nextPermutation(a), true);
  assert.deepStrictEqual(a, [2, 1, 3]);

  const last = [3, 2, 1];
  assert.strictEqual(nextPermutation(last), false);
  assert.deepStrictEqual(last, [1, 2, 3]);

  const b = [1, 2, 3, 4];
  let seen = 1;
  while (nextPermutation(b)) seen++;
  assert.strictEqual(seen, 24);
  console.log("PASS");
}
