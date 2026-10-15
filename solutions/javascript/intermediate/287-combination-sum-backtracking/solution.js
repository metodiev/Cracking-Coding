/** Task 287 - Combination sum with repetition via backtracking. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/**
 * All unique combinations of positive candidates that sum to target, each candidate
 * reusable. Candidates are walked in increasing order and the DFS continues from the same
 * index after taking a value, so every multiset appears exactly once; results are sorted
 * deterministically. Cross-checked against a per-candidate copy-counting oracle.
 */
function combinationSum(candidates, target) {
  const numbers = [...candidates].sort((a, b) => a - b);
  const results = [];
  const current = [];

  const backtrack = (start, remaining) => {
    if (remaining === 0) {
      results.push([...current]);
      return;
    }
    for (let i = start; i < numbers.length; i++) {
      if (numbers[i] > remaining) break;
      current.push(numbers[i]);
      backtrack(i, remaining - numbers[i]);   // same index: repetition allowed
      current.pop();
    }
  };

  backtrack(0, target);
  results.sort(compareCombinations);
  return results;
}

function compareCombinations(a, b) {
  const size = Math.min(a.length, b.length);
  for (let i = 0; i < size; i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
}

/** Oracle: choose how many copies of each candidate to take (counts per candidate). */
function bruteForceCombinations(candidates, target) {
  const numbers = [...candidates].sort((a, b) => a - b);
  const results = [];

  const build = (index, remaining, chosen) => {
    if (index === numbers.length) {
      if (remaining === 0) results.push([...chosen]);
      return;
    }
    const value = numbers[index];
    for (let copies = 0; value * copies <= remaining; copies++) {
      build(index + 1, remaining - value * copies, chosen.concat(new Array(copies).fill(value)));
    }
  };

  build(0, target, []);
  results.sort(compareCombinations);
  return results;
}

const rng = {
  state: 13579,
  next() {
    this.state = (this.state * 75 + 74) % 65537;
    return this.state;
  },
};

function randomCase() {
  const size = 1 + (rng.next() % 4);
  const candidates = [];
  while (candidates.length < size) {
    const value = 1 + (rng.next() % 8);
    if (!candidates.includes(value)) candidates.push(value);
  }
  const target = rng.next() % 13;
  return { candidates, target };
}

module.exports = { combinationSum, bruteForceCombinations, compareCombinations };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(combinationSum([2, 3, 6, 7], 7), [[2, 2, 3], [7]]);
  assert.deepStrictEqual(combinationSum([2, 3, 5], 8), [[2, 2, 2, 2], [2, 3, 3], [3, 5]]);
  assert.deepStrictEqual(combinationSum([7, 2, 3, 6], 7), [[2, 2, 3], [7]]);   // input order does not matter

  assert.deepStrictEqual(combinationSum([2, 3], 0), [[]]);                     // empty combination sums to 0
  assert.deepStrictEqual(combinationSum([2, 3], -1), []);                      // negative target has none
  assert.deepStrictEqual(combinationSum([2, 4], 7), []);                       // odd target, even values
  assert.deepStrictEqual(combinationSum([2], 1), []);                          // target below every candidate
  assert.deepStrictEqual(combinationSum([5], 10), [[5, 5]]);                   // repetition allowed
  assert.deepStrictEqual(combinationSum([8], 4), []);

  for (let trial = 0; trial < 200; trial++) {
    const { candidates, target } = randomCase();
    const results = combinationSum(candidates, target);
    assert.deepStrictEqual(results, bruteForceCombinations(candidates, target), `${candidates} target ${target}`);
    const seen = new Set();
    for (const combination of results) {
      const sum = combination.reduce((a, b) => a + b, 0);
      assert.strictEqual(sum, target, `sum ${candidates} target ${target}`);
      for (const value of combination) assert.ok(candidates.includes(value), `value ${value}`);
      for (let i = 1; i < combination.length; i++) assert.ok(combination[i - 1] <= combination[i], "sorted");
      const key = combination.join(",");
      assert.ok(!seen.has(key), `duplicate ${key}`);
      seen.add(key);
    }
    assert.strictEqual(results.length, seen.size);
  }

  console.log("PASS");
}
