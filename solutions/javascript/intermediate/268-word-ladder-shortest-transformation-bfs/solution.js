/** Task 268 - Word ladder shortest transformation (BFS). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

/**
 * BFS over the implicit one-letter-difference graph with a wildcard bucket index
 * (h*t -> [hot, hit, ...]) built once, so all neighbours are found in O(L) instead of
 * scanning the whole list. Returns the sequence LENGTH in words (begin and end
 * included), or 0 when no sequence exists (documented, as in LeetCode 127).
 * end not in the list -> 0; begin == end -> 1; mismatched lengths -> 0; duplicate
 * entries collapse to one word.
 */
function ladderLength(begin, end, wordList) {
  const words = new Set(wordList);
  if (!words.has(end) || begin.length !== end.length) return 0;
  if (begin === end) return 1;
  const length = begin.length;
  const buckets = new Map();
  for (const word of words) {
    if (word.length !== length) continue;
    for (let i = 0; i < length; i++) {
      const pattern = word.slice(0, i) + "*" + word.slice(i + 1);
      if (!buckets.has(pattern)) buckets.set(pattern, []);
      buckets.get(pattern).push(word);
    }
  }
  let queue = [begin];
  const seen = new Set([begin]);
  let steps = 1;
  while (queue.length > 0) {
    const next = [];
    for (const word of queue) {
      for (let i = 0; i < length; i++) {
        const pattern = word.slice(0, i) + "*" + word.slice(i + 1);
        for (const candidate of buckets.get(pattern) || []) {
          if (candidate === end) return steps + 1;
          if (!seen.has(candidate)) {
            seen.add(candidate);
            next.push(candidate);
          }
        }
      }
    }
    queue = next;
    steps++;
  }
  return 0;
}

/** Naive BFS that scans every word for a one-letter difference; cross-check only. */
function ladderLengthOracle(begin, end, wordList) {
  const words = new Set(wordList);
  if (!words.has(end) || begin.length !== end.length) return 0;
  if (begin === end) return 1;
  const oneApart = (a, b) => {
    if (a.length !== b.length) return false;
    let diff = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i] && ++diff > 1) return false;
    return diff === 1;
  };
  let queue = [[begin, 1]];
  const seen = new Set([begin]);
  while (queue.length > 0) {
    const next = [];
    for (const [word, dist] of queue) {
      for (const candidate of words) {
        if (seen.has(candidate) || !oneApart(word, candidate)) continue;
        if (candidate === end) return dist + 1;
        seen.add(candidate);
        next.push([candidate, dist + 1]);
      }
    }
    queue = next;
  }
  return 0;
}

module.exports = { ladderLength, ladderLengthOracle };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]), 5);
  assert.strictEqual(ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"]), 0);   // end not in list
  assert.strictEqual(ladderLength("a", "c", ["a", "b", "c"]), 2);
  assert.strictEqual(ladderLength("hit", "hot", ["hot", "dot", "dog"]), 2);
  assert.strictEqual(ladderLength("hot", "hot", ["hot"]), 1);                               // same word -> 1
  assert.strictEqual(ladderLength("ab", "abc", ["abc"]), 0);                                // mismatched lengths
  assert.strictEqual(ladderLength("abc", "ab", ["ab"]), 0);
  assert.strictEqual(ladderLength("lost", "cost", ["most", "fost", "cost", "cast"]), 2);    // lost -> cost direct
  assert.strictEqual(ladderLength("ab", "cd", ["ac", "bd"]), 0);                            // unreachable
  assert.strictEqual(ladderLength("hit", "cog", ["hot", "hot", "dot", "dog", "lot", "log", "cog"]), 5);
  assert.strictEqual(ladderLength("hot", "hot", ["dog"]), 0);                               // end not in list wins
  assert.strictEqual(ladderLength("aaaa", "aaab", ["aaaa", "aaab"]), 2);

  // deterministic LCG so the cross-check is reproducible
  let seed = 268;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  const randWord = () => "abc"[randInt(0, 2)] + "abc"[randInt(0, 2)] + "abc"[randInt(0, 2)];
  for (let test = 0; test < 300; test++) {      // cross-check vs the naive-scan BFS
    const pool = new Set();
    const poolSize = randInt(1, 6);
    while (pool.size < poolSize) pool.add(randWord());
    const wordList = [...pool];
    if (rand() < 0.5) wordList.push(wordList[0]);           // duplicate entry
    const begin = randWord();
    const end = pool.size > 0 && rand() < 0.5 ? wordList[randInt(0, wordList.length - 1)] : randWord();
    assert.strictEqual(ladderLength(begin, end, wordList), ladderLengthOracle(begin, end, wordList),
      `random agreement begin=${begin} end=${end} list=${wordList}`);
  }
  console.log("PASS");
}
