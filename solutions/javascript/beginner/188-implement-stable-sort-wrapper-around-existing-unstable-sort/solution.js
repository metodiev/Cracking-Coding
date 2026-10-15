/** Task 188 - Implement stable sort wrapper around existing unstable sort. (Tier: Beginner, CTCI Beyond — Software Engineering & Tooling) */
"use strict";

/**
 * Array.prototype.sort is stable since ES2019, so the *unstable* black box here is a
 * hand-rolled Lomuto quicksort: its strict "< 0" comparison lets records with equal keys
 * swap places. The wrapper then uses the decorate-sort-undecorate trick: sort
 * (key, originalIndex) pairs, which makes every comparison key unique, then undecorate.
 */
function unstableSort(records, compare) {
  const out = records.slice();
  const quick = (lo, hi) => {
    while (lo < hi) {
      const pivot = out[hi];
      let store = lo;
      for (let j = lo; j < hi; j++) {
        if (compare(out[j], pivot) < 0) {
          [out[store], out[j]] = [out[j], out[store]];
          store++;
        }
      }
      [out[store], out[hi]] = [out[hi], out[store]];
      quick(lo, store - 1);
      lo = store + 1;
    }
  };
  if (out.length) quick(0, out.length - 1);
  return out;
}

/** Stable sort by `key`, built on the unstable black box above. */
function stableSort(records) {
  const decorated = records.map((record, index) => ({ key: record.key, tag: record.tag, order: index }));
  const ordered = unstableSort(decorated, (a, b) => (a.key !== b.key ? a.key - b.key : a.order - b.order));
  return ordered.map((record) => ({ key: record.key, tag: record.tag }));
}

const tags = (records) => records.map((record) => record.tag).join(",");
const groupTags = (records, key) => records.filter((r) => r.key === key).map((r) => r.tag).join(",");
const keysSorted = (records) => records.every((r, i) => i === 0 || records[i - 1].key <= r.key);

module.exports = { unstableSort, stableSort };

if (require.main === module) {
  const assert = require("node:assert");

  const records = [
    { key: 3, tag: "a" },
    { key: 1, tag: "x" },
    { key: 3, tag: "b" },
    { key: 2, tag: "y" },
    { key: 3, tag: "c" },
    { key: 1, tag: "w" },
    { key: 2, tag: "z" },
  ];
  const compareByKey = (a, b) => a.key - b.key;

  assert.strictEqual(tags(stableSort(records)), "x,w,y,z,a,b,c");
  assert.strictEqual(tags(records), "a,x,b,y,c,w,z"); // input untouched

  // the black box only promises key order, not tie order; it must not lose records
  const raw = unstableSort(records, compareByKey);
  assert.ok(keysSorted(raw), "raw sort ordered by key");
  assert.deepStrictEqual(
    raw.map((r) => `${r.key}:${r.tag}`).sort(),
    records.map((r) => `${r.key}:${r.tag}`).sort(),
    "raw sort is a permutation",
  );

  // one distinct key: the stable wrapper keeps the original order exactly
  const sameKeys = [
    { key: 1, tag: "p" },
    { key: 1, tag: "q" },
    { key: 1, tag: "r" },
    { key: 1, tag: "s" },
  ];
  assert.deepStrictEqual(stableSort(sameKeys), sameKeys);
  assert.deepStrictEqual(stableSort([]), []);
  assert.deepStrictEqual(stableSort([{ key: 9, tag: "only" }]), [{ key: 9, tag: "only" }]);

  // negative keys work the same way
  const mixed = [
    { key: 0, tag: "p" },
    { key: -1, tag: "q" },
    { key: 0, tag: "r" },
    { key: -1, tag: "s" },
  ];
  assert.strictEqual(tags(stableSort(mixed)), "q,s,p,r");

  // larger pseudo-random batch: equal keys must stay in their original order
  let state = 41;
  const nextInt = (bound) => {
    state = (state * 48271) % 2147483647;
    return state % bound;
  };
  const batch = Array.from({ length: 120 }, (_, i) => ({ key: nextInt(5), tag: `tag-${String(i).padStart(3, "0")}` }));
  const ordered = stableSort(batch);
  assert.ok(keysSorted(ordered), "batch keys sorted");
  const describe = (list) => list.map((r) => `${r.key}:${r.tag}`).sort().join("|");
  assert.strictEqual(describe(ordered), describe(batch), "batch is a permutation");
  for (let key = 0; key < 5; key++) {
    assert.strictEqual(groupTags(ordered, key), groupTags(batch, key), `batch group ${key}`);
  }
  console.log("PASS");
}
