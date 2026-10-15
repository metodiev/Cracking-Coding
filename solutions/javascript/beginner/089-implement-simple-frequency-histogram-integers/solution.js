/** Task 089 - Implement simple frequency histogram (integers). (Tier: Beginner, CTCI Ch 1)
 *
 * histogramCounts -> Map value -> count, built in one pass.
 * renderHistogram -> one text line per distinct value, sorted ascending,
 * rendered as "value: " followed by one "#" per occurrence, e.g. "3: ###".
 */
"use strict";

function histogramCounts(values) {
  const counts = new Map();
  for (const value of values) {
    counts.set(value, (counts.get(value) || 0) + 1);
  }
  return counts;
}

function renderHistogram(counts) {
  const keys = [...counts.keys()].sort((a, b) => a - b);
  return keys.map((value) => `${value}: ${"#".repeat(counts.get(value))}`);
}

module.exports = { histogramCounts, renderHistogram };

if (require.main === module) {
  const assert = require("node:assert");
  const counts = histogramCounts([1, 2, 2, 3, 3, 3]);
  assert.strictEqual(counts.size, 3);
  assert.strictEqual(counts.get(1), 1);
  assert.strictEqual(counts.get(2), 2);
  assert.strictEqual(counts.get(3), 3);

  const lines = renderHistogram(counts);
  assert.strictEqual(lines.length, 3);
  assert.deepStrictEqual(lines, ["1: #", "2: ##", "3: ###"]);

  const empty = histogramCounts([]);
  assert.strictEqual(empty.size, 0);
  assert.deepStrictEqual(renderHistogram(empty), []);

  const negatives = histogramCounts([-1, -1, 0, 2, 2, 2]);
  assert.deepStrictEqual([...negatives.entries()], [[-1, 2], [0, 1], [2, 3]]);
  assert.deepStrictEqual(renderHistogram(negatives), ["-1: ##", "0: #", "2: ###"]);

  const single = histogramCounts([7]);
  assert.deepStrictEqual(renderHistogram(single), ["7: #"]);
  console.log("PASS");
}
