/** Task 257 - Top K frequent words (heap + lexicographic tie). (Tier: Intermediate, CTCI Ch 10) */
"use strict";

/** Tiny binary heap; compare(a, b) < 0 means a ranks nearer the root. */
class MinHeap {
  constructor(compare) {
    this.data = [];
    this.compare = compare;
  }

  get size() {
    return this.data.length;
  }

  peek() {
    return this.data[0];
  }

  push(value) {
    this.data.push(value);
    let i = this.data.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compare(this.data[parent], this.data[i]) <= 0) break;
      [this.data[parent], this.data[i]] = [this.data[i], this.data[parent]];
      i = parent;
    }
  }

  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) {
      this.data[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        let best = i;
        if (l < this.data.length && this.compare(this.data[l], this.data[best]) < 0) best = l;
        if (r < this.data.length && this.compare(this.data[r], this.data[best]) < 0) best = r;
        if (best === i) break;
        [this.data[best], this.data[i]] = [this.data[i], this.data[best]];
        i = best;
      }
    }
    return top;
  }
}

function wordOrder(a, b) {                 // LeetCode 692: count desc, then word ascending
  if (a.count !== b.count) return b.count - a.count;
  if (a.word < b.word) return -1;
  if (a.word > b.word) return 1;
  return 0;
}

/**
 * LeetCode 692 ordering: more frequent first, ties broken lexicographically ascending.
 * The heap root is the worst kept word (lower count, or on a tie the lexicographically
 * larger word), so eviction is one pop. O(n + d log k) for d distinct words.
 */
function topKFrequentWords(words, k) {
  if (k <= 0) return [];
  const freq = new Map();
  for (const w of words) freq.set(w, (freq.get(w) || 0) + 1);
  const heap = new MinHeap((a, b) => {
    if (a.count !== b.count) return a.count - b.count;   // lowest count = worst = first
    if (a.word > b.word) return -1;                      // larger word = worse
    if (a.word < b.word) return 1;
    return 0;
  });
  for (const [word, count] of freq) {
    const entry = { count, word };
    if (heap.size < k) {
      heap.push(entry);
    } else if (isBetter(entry, heap.peek())) {
      heap.pop();
      heap.push(entry);
    }
  }
  return heap.data.slice().sort(wordOrder).map((e) => e.word);
}

function isBetter(candidate, worst) {
  if (candidate.count !== worst.count) return candidate.count > worst.count;
  return candidate.word < worst.word;
}

module.exports = { MinHeap, topKFrequentWords };

if (require.main === module) {
  const assert = require("node:assert");
  const classic = ["i", "love", "leetcode", "i", "love", "coding"];   // i:2 love:2 others:1
  assert.deepStrictEqual(topKFrequentWords(classic, 1), ["i"]);
  assert.deepStrictEqual(topKFrequentWords(classic, 2), ["i", "love"]);
  assert.deepStrictEqual(topKFrequentWords(classic, 3), ["i", "love", "coding"]);
  assert.deepStrictEqual(topKFrequentWords(classic, 4), ["i", "love", "coding", "leetcode"]);
  assert.deepStrictEqual(topKFrequentWords(classic, 9), ["i", "love", "coding", "leetcode"]);
  assert.deepStrictEqual(topKFrequentWords(classic, 0), []);

  const other = ["the", "day", "is", "sunny", "the", "the", "sunny", "is", "is"];
  assert.deepStrictEqual(topKFrequentWords(other, 1), ["is"]);        // is:3 the:3 sunny:2 day:1
  assert.deepStrictEqual(topKFrequentWords(other, 2), ["is", "the"]);
  assert.deepStrictEqual(topKFrequentWords(other, 4), ["is", "the", "sunny", "day"]);

  assert.deepStrictEqual(topKFrequentWords(["a", "aa", "aaa"], 2), ["a", "aa"]);
  assert.deepStrictEqual(topKFrequentWords(["a", "aa", "aaa"], 3), ["a", "aa", "aaa"]);
  const mixed = ["b", "b", "a", "a", "c", "c", "d"];                  // a:2 b:2 c:2 d:1
  assert.deepStrictEqual(topKFrequentWords(mixed, 1), ["a"]);
  assert.deepStrictEqual(topKFrequentWords(mixed, 2), ["a", "b"]);
  assert.deepStrictEqual(topKFrequentWords(mixed, 3), ["a", "b", "c"]);
  assert.deepStrictEqual(topKFrequentWords(["only"], 1), ["only"]);
  assert.deepStrictEqual(topKFrequentWords([], 3), []);
  console.log("PASS");
}
