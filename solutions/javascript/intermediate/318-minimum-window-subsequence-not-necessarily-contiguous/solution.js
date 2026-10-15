/** Task 318 - Minimum window subsequence (not necessarily contiguous). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

/** Shortest window of s containing t as a subsequence, "" if none exists. */
function minWindowSubsequence(s, t) {
  const n = s.length;
  const m = t.length;
  if (m === 0 || m > n) return "";
  const bestStart = new Array(m).fill(-1);       // bestStart[j] = latest start covering t[0..j]
  let bestLength = n + 1;
  let begin = -1;
  let end = -1;
  for (let i = 0; i < n; i++) {
    const ch = s[i];
    for (let j = m - 1; j >= 0; j--) {           // descending keeps row i-1 intact
      if (ch === t[j]) bestStart[j] = j === 0 ? i : bestStart[j - 1];
    }
    if (bestStart[m - 1] !== -1) {
      const start = bestStart[m - 1];
      if (i - start + 1 < bestLength) {          // strictly better keeps the leftmost window
        bestLength = i - start + 1;
        begin = start;
        end = i;
      }
    }
  }
  return begin === -1 ? "" : s.slice(begin, end + 1);
}

function isSubsequence(needle, haystack) {
  let k = 0;
  for (let i = 0; i < haystack.length && k < needle.length; i++) {
    if (haystack[i] === needle[k]) k++;
  }
  return k === needle.length;
}

/** Tries every window; the independent oracle. Returns the window or null when none exists. */
function bruteMinWindow(s, t) {
  let best = null;
  for (let left = 0; left < s.length; left++) {
    for (let right = left; right < s.length; right++) {
      const window = s.slice(left, right + 1);
      if (isSubsequence(t, window) && (best === null || window.length < best.length)) best = window;
    }
  }
  return best;
}

function lcgStrings(length, count, alphabet = "ab", seed = 3180) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += alphabet[seed % alphabet.length];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { minWindowSubsequence, isSubsequence, bruteMinWindow, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(minWindowSubsequence("abcdebdde", "bde"), "bcde");   // [1..4] beats [5..8]
  assert.strictEqual(minWindowSubsequence("abc", "abc"), "abc");
  assert.strictEqual(minWindowSubsequence("abc", "d"), "");
  assert.strictEqual(minWindowSubsequence("bab", "ab"), "ab");
  assert.strictEqual(minWindowSubsequence("a", "a"), "a");
  assert.strictEqual(minWindowSubsequence("a", "aa"), "");
  assert.strictEqual(minWindowSubsequence("", "a"), "");
  assert.strictEqual(minWindowSubsequence("", ""), "");
  assert.strictEqual(minWindowSubsequence("ab", ""), "");

  // Hand-verified absent case: "ezu" needs e < z < u, but every u (indices 41, 42) precedes
  // every z (indices 43, 44), and no u follows a z.
  const longText = "jmeqksfrsdcmsiwvaovxtkwmamwhvlbajfmbfmecwuuzzaqq";
  assert.strictEqual(minWindowSubsequence(longText, "ezu"), "");
  assert.strictEqual(bruteMinWindow(longText, "ezu"), null);

  const texts = ["abcdebdde", "bab", "abc", "aab", "zzz", ...lcgStrings(30, 20, "ab"), ...lcgStrings(24, 10, "abc")];
  const patterns = ["", "ab", "ba", "z", "cc", "bca",
    ...lcgStrings(1, 20, "abc", 777), ...lcgStrings(2, 20, "abc", 778), ...lcgStrings(3, 20, "abc", 779)];

  for (const text of texts) {
    for (const pattern of patterns) {
      const window = minWindowSubsequence(text, pattern);
      const oracle = bruteMinWindow(text, pattern);
      if (oracle === null) {
        assert.strictEqual(window, "", `${text} / ${pattern}`);
      } else {
        assert.ok(text.includes(window), `${text} / ${pattern}`);            // a real contiguous window
        assert.ok(isSubsequence(pattern, window), `${text} / ${pattern}`);   // containing t as subsequence
        assert.strictEqual(window.length, oracle.length, `${text} / ${pattern}`);
      }
    }
  }

  console.log("PASS");
}
