/** Task 155 - Josephus problem simulation. (Tier: Beginner, CTCI Ch 2) */
"use strict";

/** Eliminate every k-th person in a circle of n; returns the 0-based survivor. O(n^2). */
function josephusSimulate(n, k) {
  if (n < 1 || k < 1) throw new RangeError("n and k must be >= 1");
  const people = Array.from({ length: n }, (_, i) => i);
  let idx = 0;
  while (people.length > 1) {
    idx = (idx + k - 1) % people.length;    // the k-th person counted from idx
    people.splice(idx, 1);
  }
  return people[0];
}

/** Bottom-up J(n,k) = (J(n-1,k) + k) % n starting at J(1,k) = 0. O(n) time, O(1) space. */
function josephusRecurrence(n, k) {
  if (n < 1 || k < 1) throw new RangeError("n and k must be >= 1");
  let survivor = 0;
  for (let i = 2; i <= n; i++) survivor = (survivor + k) % i;
  return survivor;
}

module.exports = { josephusSimulate, josephusRecurrence };

if (require.main === module) {
  const assert = require("node:assert");
  // known checkpoints
  assert.strictEqual(josephusSimulate(5, 2), 2);
  assert.strictEqual(josephusSimulate(7, 3), 3);
  assert.strictEqual(josephusSimulate(41, 3), 30);

  // both variants agree, including the n = 1 base case
  for (const n of [1, 2, 3, 5, 7, 10, 20, 41]) {
    for (const k of [1, 2, 3, 4, 7, 11]) {
      assert.strictEqual(josephusRecurrence(n, k), josephusSimulate(n, k), `n=${n} k=${k}`);
    }
  }
  assert.strictEqual(josephusSimulate(1, 7), 0);
  assert.strictEqual(josephusRecurrence(1, 999), 0);

  // k = 1 eliminates in order 0, 1, 2, ... so the last index survives
  assert.strictEqual(josephusSimulate(10, 1), 9);
  assert.strictEqual(josephusRecurrence(10, 1), 9);

  assert.strictEqual(josephusRecurrence(5, 2), 2);
  assert.strictEqual(josephusRecurrence(7, 3), 3);
  assert.strictEqual(josephusRecurrence(41, 3), 30);

  assert.throws(() => josephusSimulate(0, 2), RangeError);
  assert.throws(() => josephusRecurrence(3, -1), RangeError);
  console.log("PASS");
}
