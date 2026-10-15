/** Task 167 - Find all elements appearing > n/3 (generalized). (Tier: Beginner, CTCI Ch 6) */
"use strict";

function majorityElementsN3(nums) {
  let c1 = null, c2 = null, n1 = 0, n2 = 0;
  for (const x of nums) {
    if (c1 !== null && x === c1) n1++;
    else if (c2 !== null && x === c2) n2++;
    else if (n1 === 0) { c1 = x; n1 = 1; }
    else if (n2 === 0) { c2 = x; n2 = 1; }
    else { n1--; n2--; }
  }
  const out = [];
  for (const cand of new Set([c1, c2].filter((x) => x !== null))) {
    let occurrences = 0;
    for (const x of nums) if (x === cand) occurrences++;
    if (occurrences * 3 > nums.length) out.push(cand);
  }
  return out.sort((a, b) => a - b);
}

module.exports = { majorityElementsN3 };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(majorityElementsN3([3, 2, 3]), [3]);
  assert.deepStrictEqual(majorityElementsN3([1, 1, 1, 3, 3, 2, 2, 2]), [1, 2]);
  assert.deepStrictEqual(majorityElementsN3([1, 2, 3]), []);
  assert.deepStrictEqual(majorityElementsN3([]), []);
  assert.deepStrictEqual(majorityElementsN3([4]), [4]);
  assert.deepStrictEqual(majorityElementsN3([1, 1, 2, 2, 3]), [1, 2]);   // 2/5 = 0.4 > 1/3
  console.log("PASS");
}
