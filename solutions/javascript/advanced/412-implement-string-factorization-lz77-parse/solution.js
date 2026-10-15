/** Task 412 - Implement string factorization (LZ77 parse). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Greedy LZ77 factorization. Every token is a triple [offset, length, next]:
 *   * offset = distance back to the match source (0 for a pure literal),
 *   * length = matched characters copied from that earlier position,
 *   * next   = the literal character code that follows the match (-1 when the match
 *              consumes the end of the string).
 * Rules (documented, deterministic):
 *   * a match source j must satisfy max(0, i - window) <= j < i (window = 8),
 *   * the greedy step takes the LONGEST match, ties broken by the SMALLEST offset,
 *   * matches shorter than min_len = 3 are not worth a token, so a literal is emitted,
 *   * matches may overlap the source region (s[j+L] is read from the original string).
 * The brute-force oracle recomputes the same token list with explicit enumeration, and
 * a DP computes a lower bound on the token count of any parse.
 */
"use strict";

const MIN_LEN = 3;
const WINDOW = 8;

function lz77Parse(s, window = WINDOW, minLen = MIN_LEN) {
  const tokens = [];
  const n = s.length;
  let i = 0;
  while (i < n) {
    let bestLen = 0, bestOff = 0;
    for (let j = Math.max(0, i - window); j < i; j++) {
      let L = 0;
      while (i + L < n && s[j + L] === s[i + L]) L++;
      if (L > bestLen) { bestLen = L; bestOff = i - j; }
    }
    if (bestLen >= minLen) {
      const atEnd = i + bestLen === n;
      tokens.push([bestOff, bestLen, atEnd ? -1 : s.charCodeAt(i + bestLen)]);
      i += bestLen + (atEnd ? 0 : 1);
    } else {
      tokens.push([0, 0, s.charCodeAt(i)]);
      i += 1;
    }
  }
  return tokens;
}

function lz77Decode(tokens) {
  let out = "";
  for (const [off, ln, nxt] of tokens) {
    if (ln > 0) {
      const start = out.length - off;
      for (let k = 0; k < ln; k++) out += out[start + k];
    }
    if (nxt !== -1) out += String.fromCharCode(nxt);
  }
  return out;
}

function lz77Bruteforce(s, window = WINDOW, minLen = MIN_LEN) {
  const tokens = [];
  const n = s.length;
  let i = 0;
  while (i < n) {
    let bestLen = 0, bestOff = 0;
    for (let length = n - i; length >= 1 && bestLen === 0; length--) {
      for (let j = Math.max(0, i - window); j < i; j++) {
        if (s.slice(j, j + length) === s.slice(i, i + length)) {
          bestLen = length;
          bestOff = i - j;
          break;
        }
      }
    }
    if (bestLen >= minLen) {
      const atEnd = i + bestLen === n;
      tokens.push([bestOff, bestLen, atEnd ? -1 : s.charCodeAt(i + bestLen)]);
      i += bestLen + (atEnd ? 0 : 1);
    } else {
      tokens.push([0, 0, s.charCodeAt(i)]);
      i += 1;
    }
  }
  return tokens;
}

function optimalParseLen(s, window = WINDOW, minLen = MIN_LEN) {
  const n = s.length;
  const dp = new Array(n + 1).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let best = 1 + dp[i + 1];
    for (let j = Math.max(0, i - window); j < i; j++) {
      let L = 0;
      while (i + L < n && s[j + L] === s[i + L]) L++;
      for (let length = minLen; length <= L; length++) {
        const cand = i + length === n ? 1 : 1 + dp[i + length + 1];
        if (cand < best) best = cand;
      }
    }
    dp[i] = best;
  }
  return dp[0];
}

let seed = 777;

function lcgString(length, alphabet) {
  let out = "";
  for (let k = 0; k < length; k++) {
    seed = (seed * 75 + 74) % 65537;
    out += alphabet[seed % alphabet.length];
  }
  return out;
}

function sameTokens(a, b) {
  if (a.length !== b.length) return false;
  return a.every((t, i) => t[0] === b[i][0] && t[1] === b[i][1] && t[2] === b[i][2]);
}

module.exports = { lz77Parse, lz77Decode, lz77Bruteforce, optimalParseLen, lcgString, sameTokens, MIN_LEN, WINDOW };

if (require.main === module) {
  const assert = require("node:assert");

  const a = (ch) => ch.charCodeAt(0);

  // exact greedy factorizations demanded by the task
  assert.ok(sameTokens(lz77Parse("abcabcabc", 8, 3),
    [[0, 0, a("a")], [0, 0, a("b")], [0, 0, a("c")], [3, 6, -1]]), "abcabcabc");
  assert.ok(sameTokens(lz77Parse("aaaaaaaa", 8, 3), [[0, 0, a("a")], [1, 7, -1]]), "aaaaaaaa");

  // round trips and hand-built corner cases
  assert.strictEqual(lz77Decode(lz77Parse("abcabcabc", 8, 3)), "abcabcabc");
  assert.strictEqual(lz77Decode(lz77Parse("aaaaaaaa", 8, 3)), "aaaaaaaa");
  assert.deepStrictEqual(lz77Parse("", 8, 3), []);
  assert.strictEqual(lz77Decode([]), "");
  assert.ok(sameTokens(lz77Parse("ab", 8, 3), [[0, 0, a("a")], [0, 0, a("b")]]), "too short to match");
  assert.ok(sameTokens(lz77Parse("abcabcx", 8, 3),
    [[0, 0, a("a")], [0, 0, a("b")], [0, 0, a("c")], [3, 3, a("x")]]), "match then next char");
  assert.ok(sameTokens(lz77Parse("aabaaba", 8, 3),
    [[0, 0, a("a")], [0, 0, a("a")], [0, 0, a("b")], [3, 4, -1]]), "consume tail");

  // window bounds the match source: distance-3 source needs window >= 3
  assert.ok(sameTokens(lz77Parse("aabaab", 3, 3),
    [[0, 0, a("a")], [0, 0, a("a")], [0, 0, a("b")], [3, 3, -1]]), "window 3");
  assert.ok(sameTokens(lz77Parse("aabaab", 2, 3),
    [[0, 0, a("a")], [0, 0, a("a")], [0, 0, a("b")], [0, 0, a("a")], [0, 0, a("a")], [0, 0, a("b")]]), "window 2");

  // min length boundary: length-2 match is rejected, length-3 accepted
  assert.ok(sameTokens(lz77Parse("aaab", 8, 3),
    [[0, 0, a("a")], [0, 0, a("a")], [0, 0, a("a")], [0, 0, a("b")]]), "min len 3");
  assert.ok(sameTokens(lz77Parse("aaab", 8, 2), [[0, 0, a("a")], [1, 2, a("b")]]), "min len 2");

  // optimal parses of the two hand strings
  assert.strictEqual(optimalParseLen("abcabcabc", 8, 3), 4);
  assert.strictEqual(optimalParseLen("aaaaaaaa", 8, 3), 2);
  assert.strictEqual(optimalParseLen("aaab", 8, 3), 4);
  assert.strictEqual(optimalParseLen("aabaab", 3, 3), 4);

  for (let round = 0; round < 8; round++) {
    const length = 60 + (seed % 100);
    const alphabet = seed % 2 === 0 ? "ab" : "abc";
    const text = lcgString(length, alphabet);
    const window = 4 + (seed % 29);
    const greedy = lz77Parse(text, window, 3);
    const slow = lz77Bruteforce(text, window, 3);
    assert.ok(sameTokens(greedy, slow), `oracle equal round ${round}`);
    assert.strictEqual(lz77Decode(greedy), text, `round trip round ${round}`);
    for (const [off, ln] of greedy) {
      assert.ok(off <= window && (ln === 0 || ln >= 3), `invariants round ${round}`);
    }
    const optimal = optimalParseLen(text, window, 3);
    assert.ok(optimal <= greedy.length && greedy.length <= text.length, `bound round ${round}`);
  }

  // greedy re-derivation: at every token boundary the token must be the longest
  // match, ties resolved to the smallest offset (or a literal when it is shorter)
  const text = "abababababab";
  const tokens = lz77Parse(text, 8, 3);
  let pos = 0;
  for (const [off, ln, nxt] of tokens) {
    let bestLen = 0, bestJ = -1;
    for (let j = Math.max(0, pos - 8); j < pos; j++) {
      let L = 0;
      while (pos + L < text.length && text[j + L] === text[pos + L]) L++;
      if (L > bestLen) { bestLen = L; bestJ = j; }
    }
    if (ln > 0) {
      assert.ok(ln === bestLen && off === pos - bestJ, `rederive match at ${pos}`);
    } else {
      assert.ok(bestLen < 3, `rederive literal at ${pos}`);
    }
    pos += ln + (nxt === -1 ? 0 : 1);
  }

  console.log("PASS");
}
