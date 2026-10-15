/** Task 232 - Minimum window substring covering all chars of pattern. (Tier: Intermediate, CTCI Ch 1) */
"use strict";

/**
 * Sliding window with a need/missing counter: grow with `right`; when the window
 * stops missing characters, shrink from the left while still valid, keeping the
 * smallest window seen. O(|s| + |t|) time. Leftmost window wins ties; "" if none.
 */
function minWindow(s, t) {
  if (s.length === 0 || t.length === 0 || t.length > s.length) return "";
  const need = new Map();
  for (const ch of t) need.set(ch, (need.get(ch) || 0) + 1);
  let missing = t.length;
  let left = 0;
  let bestStart = 0;
  let bestLen = Infinity;
  for (let right = 0; right < s.length; right++) {
    const ch = s[right];
    if ((need.get(ch) || 0) > 0) missing--;
    need.set(ch, (need.get(ch) || 0) - 1);
    while (missing === 0) {
      if (right - left + 1 < bestLen) {       // strict: ties keep the leftmost window
        bestStart = left;
        bestLen = right - left + 1;
      }
      const lc = s[left];
      need.set(lc, need.get(lc) + 1);
      if (need.get(lc) > 0) missing++;
      left++;
    }
  }
  return bestLen === Infinity ? "" : s.slice(bestStart, bestStart + bestLen);
}

/** O(n^2 * k) reference used only for cross-checks; same leftmost tie-break. */
function bruteMinWindow(s, t) {
  if (s.length === 0 || t.length === 0 || t.length > s.length) return "";
  const want = new Array(128).fill(0);
  for (const ch of t) want[ch.charCodeAt(0)]++;
  let best = "";
  for (let left = 0; left < s.length; left++) {
    for (let right = left + t.length; right <= s.length; right++) {
      const have = new Array(128).fill(0);
      for (let i = left; i < right; i++) have[s.charCodeAt(i)]++;
      let ok = true;
      for (let c = 0; c < 128 && ok; c++) if (have[c] < want[c]) ok = false;
      if (ok) {
        const window = s.slice(left, right);
        if (best === "" || window.length < best.length) best = window;
        break;
      }
    }
  }
  return best;
}

module.exports = { minWindow, bruteMinWindow };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(minWindow("ADOBECODEBANC", "ABC"), "BANC");
  assert.strictEqual(minWindow("cabwefgewcwaefgcf", "cae"), "cwae");
  assert.strictEqual(minWindow("a", "a"), "a");
  assert.strictEqual(minWindow("a", "aa"), "");
  assert.strictEqual(minWindow("abc", "abcd"), "");
  assert.strictEqual(minWindow("ab", "cd"), "");
  assert.strictEqual(minWindow("aa", "aa"), "aa");
  assert.strictEqual(minWindow("abaab", "aa"), "aa");
  assert.strictEqual(minWindow("bba", "ab"), "ba");
  assert.strictEqual(minWindow("aabc", "aabc"), "aabc");
  assert.strictEqual(minWindow("abc", ""), "");
  assert.strictEqual(minWindow("", "a"), "");

  // deterministic LCG so the cross-check is reproducible
  let seed = 232;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  for (let test = 0; test < 200; test++) {
    const sLen = Math.floor(rand() * 15);
    let s = "";
    for (let i = 0; i < sLen; i++) s += rand() < 0.5 ? "a" : "b";
    const tLen = Math.floor(rand() * 5);
    let t = "";
    for (let i = 0; i < tLen; i++) t += rand() < 0.5 ? "a" : "b";
    assert.strictEqual(minWindow(s, t), bruteMinWindow(s, t), `matches brute force: ${s} / ${t}`);
  }
  console.log("PASS");
}
