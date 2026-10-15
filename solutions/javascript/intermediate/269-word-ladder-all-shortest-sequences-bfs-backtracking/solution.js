/** Task 269 - Word ladder all shortest sequences (BFS + backtracking). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

/**
 * All shortest transformation sequences (each step changes one letter; every
 * intermediate word must be in the word list). BFS over the wildcard bucket index
 * assigns minimum distances; the DFS then walks the LAYERED DAG, following only edges
 * u -> v with dist[v] === dist[u] + 1. That keeps every shortest path exactly once
 * (dist strictly increases, so no word repeats within a path) and prunes all
 * non-shortest continuations. Returned sequences are sorted element-wise
 * lexicographically so hash-iteration order never leaks into the result.
 * [] when end is not in the list, the lengths differ, or end is unreachable;
 * [[begin]] when begin === end (and end is in the list).
 */
function findLadders(begin, end, wordList) {
  const words = new Set(wordList);
  if (!words.has(end) || begin.length !== end.length) return [];
  if (begin === end) return [[begin]];
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

  const dist = new Map([[begin, 0]]);                 // BFS: minimum distance per word
  let queue = [begin];
  while (queue.length > 0) {
    const next = [];
    for (const word of queue) {
      for (let i = 0; i < length; i++) {
        for (const candidate of buckets.get(word.slice(0, i) + "*" + word.slice(i + 1)) || []) {
          if (!dist.has(candidate)) {
            dist.set(candidate, dist.get(word) + 1);
            next.push(candidate);
          }
        }
      }
    }
    queue = next;
  }
  if (!dist.has(end)) return [];

  const target = dist.get(end);
  const result = [];
  const path = [begin];
  const dfs = (word) => {                             // backtrack along the layered DAG
    if (word === end) {
      result.push(path.slice());
      return;
    }
    if (dist.get(word) >= target) return;
    for (let i = 0; i < length; i++) {
      for (const candidate of buckets.get(word.slice(0, i) + "*" + word.slice(i + 1)) || []) {
        if (dist.get(candidate) === dist.get(word) + 1) {
          path.push(candidate);
          dfs(candidate);
          path.pop();
        }
      }
    }
  };
  dfs(begin);
  result.sort(comparePaths);
  return result;
}

/** Element-wise lexicographic order (deterministic output). */
function comparePaths(p, q) {
  const common = Math.min(p.length, q.length);
  for (let i = 0; i < common; i++) {
    if (p[i] < q[i]) return -1;
    if (p[i] > q[i]) return 1;
  }
  return p.length - q.length;
}

/** Enumerate all simple paths, keep the shortest; cross-check only. */
function findLaddersBrute(begin, end, wordList) {
  const words = new Set(wordList);
  if (!words.has(end) || begin.length !== end.length) return [];
  if (begin === end) return [[begin]];
  const found = [];
  const dfs = (word, path, seen) => {
    if (word === end) {
      found.push(path.slice());
      return;
    }
    if (path.length > words.size + 1) return;         // simple paths cannot exceed this
    for (const candidate of words) {
      if (seen.has(candidate) || candidate.length !== word.length) continue;
      let diff = 0;
      for (let i = 0; i < word.length && diff <= 1; i++) if (word[i] !== candidate[i]) diff++;
      if (diff === 1) {
        seen.add(candidate);
        path.push(candidate);
        dfs(candidate, path, seen);
        path.pop();
        seen.delete(candidate);
      }
    }
  };
  dfs(begin, [begin], new Set([begin]));
  if (found.length === 0) return [];
  const best = Math.min(...found.map((p) => p.length));
  return found.filter((p) => p.length === best).sort(comparePaths);
}

module.exports = { findLadders, findLaddersBrute, comparePaths };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]), [
    ["hit", "hot", "dot", "dog", "cog"],
    ["hit", "hot", "lot", "log", "cog"],
  ]);
  assert.deepStrictEqual(findLadders("red", "tax", ["ted", "tex", "red", "tax", "tad", "den", "rex", "pee"]), [
    ["red", "rex", "tex", "tax"],
    ["red", "ted", "tad", "tax"],
    ["red", "ted", "tex", "tax"],
  ]);
  assert.deepStrictEqual(findLadders("ab", "ba", ["bb", "aa", "ba"]), [["ab", "aa", "ba"], ["ab", "bb", "ba"]]);
  assert.deepStrictEqual(findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log"]), []);   // unreachable
  assert.deepStrictEqual(findLadders("hot", "hot", ["hot"]), [["hot"]]);                        // same word
  assert.deepStrictEqual(findLadders("a", "c", ["a", "b", "c"]), [["a", "c"]]);                 // direct hop
  assert.deepStrictEqual(findLadders("ab", "abc", ["abc"]), []);                                // mismatched lengths
  assert.deepStrictEqual(findLadders("hot", "hot", ["dog"]), []);                               // end not in list wins
  assert.deepStrictEqual(findLadders("lost", "cost", ["most", "fost", "cost", "cast"]), [["lost", "cost"]]);

  // deterministic LCG so the cross-check is reproducible
  let seed = 269;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  const randWord = () => "abc"[randInt(0, 2)] + "abc"[randInt(0, 2)] + "abc"[randInt(0, 2)];
  for (let test = 0; test < 200; test++) {      // cross-check vs the path-enumeration brute
    const pool = new Set();
    const poolSize = randInt(1, 6);
    while (pool.size < poolSize) pool.add(randWord());
    const wordList = [...pool];
    const begin = randWord();
    const end = rand() < 0.5 ? wordList[randInt(0, wordList.length - 1)] : randWord();
    assert.deepStrictEqual(findLadders(begin, end, wordList), findLaddersBrute(begin, end, wordList),
      `random agreement begin=${begin} end=${end} list=${wordList}`);
  }
  console.log("PASS");
}
