/** Task 290 - Palindrome partitioning with backtracking. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/**
 * All ways to cut a string into palindromic parts. Pieces are extended shortest-first, so
 * partition lists come out in lexicographic order and results are deterministic. Every
 * piece is validated with an independent palindrome checker and concatenation must
 * restore the input; cross-checked against an oracle over all 2^(n-1) cut sets.
 */
function isPalindrome(text) {
  let left = 0, right = text.length - 1;
  while (left < right) {
    if (text[left] !== text[right]) return false;
    left++;
    right--;
  }
  return true;
}

function palindromePartitioning(s) {
  const results = [];
  const current = [];

  const backtrack = (start) => {
    if (start === s.length) {
      results.push([...current]);
      return;
    }
    for (let end = start + 1; end <= s.length; end++) {
      const piece = s.slice(start, end);
      if (!isPalindrome(piece)) continue;
      current.push(piece);
      backtrack(end);
      current.pop();
    }
  };

  backtrack(0);
  return results;
}

/** Oracle: enumerate all 2^(n-1) cut sets and keep the palindrome-only partitions. */
function bruteForcePartitions(s) {
  const n = s.length;
  if (n === 0) return [[]];
  const results = [];
  for (let mask = 0; mask < 1 << (n - 1); mask++) {
    const parts = [];
    let start = 0;
    for (let i = 0; i < n - 1; i++) {
      if ((mask >> i) & 1) {
        parts.push(s.slice(start, i + 1));
        start = i + 1;
      }
    }
    parts.push(s.slice(start));
    if (parts.every(isPalindrome)) results.push(parts);
  }
  results.sort((a, b) => {
    const ka = a.join(","), kb = b.join(",");
    return ka < kb ? -1 : ka > kb ? 1 : 0;
  });
  return results;
}

const keyOf = (parts) => parts.join(",");
const canonical = (partitions) => partitions.map(keyOf).sort();
const keyList = (partitions) => partitions.map(keyOf);

const rng = {
  state: 13579,
  next() {
    this.state = (this.state * 75 + 74) % 65537;
    return this.state;
  },
};

module.exports = { palindromePartitioning, isPalindrome, bruteForcePartitions };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(palindromePartitioning("aab"), [["a", "a", "b"], ["aa", "b"]]);
  assert.deepStrictEqual(palindromePartitioning("a"), [["a"]]);
  assert.deepStrictEqual(palindromePartitioning("aba"), [["a", "b", "a"], ["aba"]]);
  assert.deepStrictEqual(palindromePartitioning("aaa"),
    [["a", "a", "a"], ["a", "aa"], ["aa", "a"], ["aaa"]]);
  assert.deepStrictEqual(palindromePartitioning("aaaa"), bruteForcePartitions("aaaa"));
  assert.strictEqual(palindromePartitioning("aaaa").length, 8);   // every substring is palindromic
  assert.strictEqual(palindromePartitioning("abba").length, 3);
  assert.deepStrictEqual(palindromePartitioning(""), [[]]);       // zero parts concatenate to ""

  for (let trial = 0; trial < 200; trial++) {
    const length = 1 + (rng.next() % 7);
    let s = "";
    for (let i = 0; i < length; i++) s += "abc"[rng.next() % 3];
    const partitions = palindromePartitioning(s);
    assert.deepStrictEqual(partitions, bruteForcePartitions(s), s);
    assert.deepStrictEqual(keyList(partitions), canonical(partitions), `lexicographic ${s}`);
    const seen = new Set();
    for (const parts of partitions) {
      assert.strictEqual(parts.join(""), s, `concatenation ${s}`);
      for (const piece of parts) assert.ok(piece.length > 0 && isPalindrome(piece), `palindrome ${s}`);
      const key = keyOf(parts);
      assert.ok(!seen.has(key), `duplicate ${s} ${key}`);
      seen.add(key);
    }
    assert.strictEqual(partitions.length, seen.size, `count ${s}`);
  }

  console.log("PASS");
}
