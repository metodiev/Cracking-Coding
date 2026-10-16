/** Task 701 - Implement approximate string matching with edit distance threshold using Ukkonen. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Ukkonen's cut-off ("banded") dynamic program. Every alignment step moves i - j by at most
 * one, so in the edit-distance table only the |i - j| <= k band can hold a value <= k. The band
 * is stored with out-of-band cells left at the sentinel k+1 ("more than k"); a row whose
 * minimum is the sentinel proves no later row can reach k, which is the cut-off that reduces
 * the O(n m) table to O(k m) work.
 *
 * Search (every end position where some substring of the text is within distance k of the
 * pattern) has a free start, so the anchored band cannot be laid over the whole text. The same
 * argument gives the reduction used here: such a window has length in [m-k, m+k], so the banded
 * DP runs on those (at most 2k+1) candidate windows ending at each position. Two independent
 * algorithms are cross-checked against it: the bit-parallel k-error automaton (Ukkonen's 1985
 * automaton in the R-mask/Wu-Manber packaging - k+1 bitmasks over the pattern, the free start
 * supplied by the 1 shifted into bit 0 at every step) and an exhaustive O(n^2 m^2) scan.
 *
 * Invariant verified: banded == full-table value whenever that value is <= k and "> k"
 * otherwise; and for every text/pattern/threshold all three searches agree exactly.
 */
"use strict";

const { strictEqual, deepStrictEqual, ok, throws } = require("node:assert");

/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

function assertThreshold(k) {
  if (!Number.isInteger(k) || k < 0) {
    throw new RangeError(`threshold k must be a non-negative integer (got ${k})`);
  }
}

/** Plain O(m n) Levenshtein distance - the reference oracle, no cut-off, no band. */
function editDistance(a, b) {
  const m = a.length; const n = b.length;
  let prev = new Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    const row = new Array(n + 1);
    row[0] = i;
    for (let j = 1; j <= n; j++) {
      const substitute = a[i - 1] === b[j - 1] ? 0 : 1;
      row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + substitute);
    }
    prev = row;
  }
  return prev[n];
}

/**
 * Ukkonen's banded distance: returns the exact edit distance when it is <= k, else -1.
 * Only the 2k+1 diagonals |i - j| <= k are materialised; out-of-band cells count as k+1.
 */
function distanceWithinK(a, b, k) {
  assertThreshold(k);
  if (typeof a !== "string" || typeof b !== "string") {
    throw new TypeError("distanceWithinK expects two strings");
  }
  const m = a.length; const n = b.length;
  if (Math.abs(m - n) > k) return -1; // each edit changes the length by at most one
  const width = 2 * k + 1;
  const inf = k + 1; // sentinel for "strictly more than k"
  const at = (row, base, j) => {
    const idx = j - base;
    return idx >= 0 && idx < width ? row[idx] : inf;
  };
  let prev = new Array(width).fill(inf); // row 0: base -k holds D[0][j] = j for j <= k
  for (let j = 0; j <= Math.min(n, k); j++) prev[j + k] = j;
  for (let i = 1; i <= m; i++) {
    const lo = Math.max(0, i - k);
    const hi = Math.min(n, i + k);
    if (lo > hi) return -1; // the band has left the table: everything is > k
    const base = i - k;
    const row = new Array(width).fill(inf);
    let rowMin = inf;
    if (lo === 0) {
      row[0 - base] = i <= k ? i : inf; // D[i][0] = i deletions
      rowMin = row[0 - base];
    }
    for (let j = Math.max(1, lo); j <= hi; j++) {
      const substitute = a[i - 1] === b[j - 1] ? 0 : 1;
      const diag = at(prev, i - 1 - k, j - 1) + substitute;
      const up = at(prev, i - 1 - k, j) + 1;
      const left = at(row, base, j - 1) + 1;
      let value = Math.min(diag, up, left);
      if (value > k) value = inf;
      row[j - base] = value;
      if (value < rowMin) rowMin = value;
    }
    if (rowMin > k) return -1; // cut-off: no cell of this row can lead to a value <= k
    prev = row;
  }
  const result = at(prev, m - k, n);
  return result <= k ? result : -1;
}

/**
 * Approximate pattern search with Ukkonen's cut-off. Returns the sorted end positions j >= 1
 * such that some window text[start, j) is within distance k of the pattern. The empty window
 * only qualifies when k >= m, so an empty pattern reports every position.
 */
function approximateSearchUkkonen(text, pattern, k) {
  assertThreshold(k);
  if (typeof text !== "string" || typeof pattern !== "string") {
    throw new TypeError("approximateSearchUkkonen expects two strings");
  }
  const n = text.length; const m = pattern.length;
  const minLength = Math.max(0, m - k);
  const maxLength = Math.min(m + k, n);
  const hits = [];
  for (let end = 1; end <= n; end++) {
    let best = -1;
    for (let len = minLength; len <= maxLength; len++) {
      const start = end - len;
      if (start < 0) break; // longer windows only start earlier still
      const d = distanceWithinK(text.slice(start, end), pattern, k);
      if (d >= 0 && (best < 0 || d < best)) best = d;
      if (best === 0) break;
    }
    if (best >= 0) hits.push(end);
  }
  return hits;
}

/**
 * Bit-parallel k-error automaton: Ukkonen's 1985 automaton in the R-mask (Wu-Manber)
 * formulation. Bit i of R[d] is set when the pattern prefix pattern[0, i+1) can be matched
 * against some suffix of the text read so far with at most d errors, so one text character
 * updates all k+1 masks with a shift, a mask and three ORs - O(n * ceil(m/64) * k) word
 * operations instead of a DP table. Column -1 (no text read) satisfies prefix i with at most
 * d errors exactly when i <= d, which is the initialisation R[d] = (1 << d) - 1; the "| 1"
 * inside every shift is the free start (the injected suffix beginning). Entirely different
 * from the banded tabulation, which is what makes it worth cross-checking against.
 */
function approximateSearchAutomaton(text, pattern, k) {
  assertThreshold(k);
  if (typeof text !== "string" || typeof pattern !== "string") {
    throw new TypeError("approximateSearchAutomaton expects two strings");
  }
  const m = pattern.length; const n = text.length;
  const hits = [];
  if (m === 0) {
    for (let j = 1; j <= n; j++) hits.push(j); // the empty window always matches
    return hits;
  }
  const masks = new Map();
  for (let i = 0; i < m; i++) {
    const bit = 1n << BigInt(i);
    masks.set(pattern[i], (masks.get(pattern[i]) ?? 0n) | bit);
  }
  const ones = (1n << BigInt(m)) - 1n;
  const finalBit = 1n << BigInt(m - 1);
  const shift = (x) => ((x << 1n) | 1n) & ones;
  const current = new Array(k + 1);
  for (let d = 0; d <= k; d++) current[d] = (1n << BigInt(Math.min(d, m))) - 1n & ones;
  for (let j = 0; j < n; j++) {
    const mask = masks.get(text[j]) ?? 0n;
    const next = new Array(k + 1).fill(0n);
    next[0] = shift(current[0]) & mask;
    for (let d = 1; d <= k; d++) {
      // d-1 errors then: consume text char and pattern char (substitution), consume only the
      // text char (insertion), consume only the pattern char (deletion), or delete one more
      // pattern character inside this very column (the left-to-right DP sweep).
      next[d] = (shift(current[d]) & mask)
        | shift(current[d - 1])
        | current[d - 1]
        | ((next[d - 1] << 1n) & ones);
      next[d] &= ones;
    }
    for (let d = 0; d <= k; d++) current[d] = next[d];
    if ((current[k] & finalBit) !== 0n) hits.push(j + 1);
  }
  return hits;
}

/** Exhaustive oracle: full Levenshtein distance of every window of the text. */
function approximateSearchNaive(text, pattern, k) {
  assertThreshold(k);
  const hits = [];
  for (let end = 1; end <= text.length; end++) {
    let best = Infinity;
    for (let start = 0; start <= end && best > 0; start++) {
      best = Math.min(best, editDistance(text.slice(start, end), pattern));
    }
    if (best <= k) hits.push(end);
  }
  return hits;
}

/** Ends reported by indexOf for k = 0, used as an independent exact-match oracle. */
function exactMatchEnds(text, pattern) {
  const ends = new Set();
  if (pattern.length === 0) {
    for (let j = 1; j <= text.length; j++) ends.add(j);
    return [...ends].sort((x, y) => x - y);
  }
  let from = 0;
  for (;;) {
    const at = text.indexOf(pattern, from);
    if (at < 0) break;
    ends.add(at + pattern.length);
    from = at + 1;
  }
  return [...ends].sort((x, y) => x - y);
}

/** All strings of length <= maxLength over `alphabet`, used for exhaustive sweeps. */
function stringsUpTo(alphabet, maxLength) {
  const out = [""];
  for (let len = 1; len <= maxLength; len++) {
    const previous = out.filter((s) => s.length === len - 1);
    for (const head of previous) for (const ch of alphabet) out.push(head + ch);
  }
  return out;
}

function randomString(rng, length, alphabet) {
  let out = "";
  for (let i = 0; i < length; i++) out += alphabet[rng(alphabet.length)];
  return out;
}

module.exports = {
  makeRng, editDistance, distanceWithinK, approximateSearchUkkonen,
  approximateSearchAutomaton, approximateSearchNaive, exactMatchEnds, stringsUpTo, randomString,
};

if (require.main === module) {
  // -- reference distances and rejected inputs -------------------------------------------------
  for (const [a, b, want] of [["kitten", "sitting", 3], ["flaw", "lawn", 2], ["", "abc", 3],
    ["abc", "abc", 0], ["", "", 0]]) {
    strictEqual(editDistance(a, b), want, `editDistance("${a}","${b}")`);
  }
  throws(() => distanceWithinK("a", "b", -1), RangeError);
  throws(() => distanceWithinK("a", "b", 1.5), RangeError);
  throws(() => distanceWithinK(1, "b", 1), TypeError);
  throws(() => approximateSearchUkkonen("abc", "a", -1), RangeError);

  // -- boundary cases: over-long gap, degenerate strings, distant strings ----------------------
  for (const [a, b, k, want] of [["", "abc", 2, -1], ["", "abc", 3, 3], ["", "", 0, 0],
    ["abcdef", "uvwxyz", 6, 6], ["abcdef", "uvwxyz", 2, -1], ["abc", "abc", 0, 0]]) {
    strictEqual(distanceWithinK(a, b, k), want, `distanceWithinK("${a}","${b}",${k})`);
  }

  // -- exhaustive agreement of the banded DP with the full table, both argument orders ---------
  const binary = ["a", "b"];
  const shortStrings = stringsUpTo(binary, 4);
  let comparisons = 0;
  for (const a of shortStrings) for (const b of shortStrings) {
    const truth = editDistance(a, b);
    for (let k = 0; k <= 3; k++) {
      const banded = distanceWithinK(a, b, k);
      strictEqual(banded, truth <= k ? truth : -1, `distanceWithinK("${a}","${b}",${k})`);
      strictEqual(banded, distanceWithinK(b, a, k), "distance must be symmetric");
      comparisons++;
    }
  }
  ok(comparisons === shortStrings.length * shortStrings.length * 4);

  // -- k = 0 must behave exactly like substring search (independent indexOf oracle) ------------
  const rng = makeRng(701);
  const alphabet = "abcd";
  for (let trial = 0; trial < 40; trial++) {
    const text = randomString(rng, 5 + rng(14), alphabet);
    const pattern = randomString(rng, 1 + rng(4), alphabet);
    const expected = exactMatchEnds(text, pattern);
    deepStrictEqual(approximateSearchUkkonen(text, pattern, 0), expected, `ukkonen "${text}"/"${pattern}"`);
    deepStrictEqual(approximateSearchAutomaton(text, pattern, 0), expected);
  }

  // -- exhaustive three-way agreement of the three search implementations ----------------------
  const universe = stringsUpTo(binary, 3).filter((s) => s.length > 0);
  let searchChecks = 0;
  for (const text of universe) for (const pattern of stringsUpTo(binary, 3)) {
    for (let k = 0; k <= 3; k++) {
      const expected = approximateSearchNaive(text, pattern, k);
      deepStrictEqual(approximateSearchUkkonen(text, pattern, k), expected, `ukkonen "${text}"/"${pattern}" k=${k}`);
      deepStrictEqual(approximateSearchAutomaton(text, pattern, k), expected, `automaton "${text}"/"${pattern}" k=${k}`);
      searchChecks++;
    }
  }
  ok(searchChecks === universe.length * stringsUpTo(binary, 3).length * 4);

  // -- hand-checked approximate occurrences ----------------------------------------------------
  const sentence = "approximate string matching with ukkonen";
  const hits = approximateSearchUkkonen(sentence, "strng", 1);
  ok(hits.includes(sentence.indexOf("string") + "string".length), "one deletion must be tolerated");
  deepStrictEqual(hits, approximateSearchNaive(sentence, "strng", 1));
  ok(hits.length >= 1 && hits.length < sentence.length, "not everything matches");
  const oneEdit = approximateSearchUkkonen("abcdef", "bcd", 1);
  ok(oneEdit.includes(3) && oneEdit.includes(4), "exact window and one insertion");
  deepStrictEqual(oneEdit, approximateSearchNaive("abcdef", "bcd", 1));
  deepStrictEqual(approximateSearchUkkonen("axcdef", "bcd", 2), approximateSearchNaive("axcdef", "bcd", 2));

  // -- monotonicity in k, and every reported end justified by a window inside the threshold ----
  const monotoneText = "the quick brown fox jumps over the lazy dog";
  let previous = [];
  for (let k = 0; k <= 4; k++) {
    const current = approximateSearchUkkonen(monotoneText, "quick", k);
    for (const end of previous) ok(current.includes(end), `k=${k} lost end ${end}`);
    ok(current.length >= previous.length, "more tolerance never loses an occurrence");
    previous = current;
  }
  ok(previous.length > approximateSearchUkkonen(monotoneText, "quick", 0).length,
    "higher thresholds must add occurrences");
  for (let trial = 0; trial < 30; trial++) {
    const text = randomString(rng, 6 + rng(12), alphabet);
    const pattern = randomString(rng, 1 + rng(5), alphabet);
    const k = rng(3);
    for (const end of approximateSearchUkkonen(text, pattern, k)) {
      const windows = Array.from({ length: end + 1 }, (_, s) => editDistance(text.slice(s, end), pattern));
      ok(Math.min(...windows) <= k, `end ${end} at k=${k} has no window within the threshold`);
    }
  }

  // -- degenerate inputs: empty text, empty pattern, pattern longer than the text --------------
  for (const [text, pattern, k, want] of [["", "", 0, []], ["abc", "", 0, [1, 2, 3]],
    ["abc", "abcdef", 0, []], ["abc", "abcdef", 3, [3]]]) {
    for (const search of [approximateSearchUkkonen, approximateSearchAutomaton]) {
      deepStrictEqual(search(text, pattern, k), want, `search("${text}","${pattern}",${k})`);
    }
    deepStrictEqual(approximateSearchNaive(text, pattern, k), want, "the brute force must agree");
  }
  deepStrictEqual(approximateSearchAutomaton("", "abc", 2), []);

  // -- deterministic replay --------------------------------------------------------------------
  function replay(seed) {
    const r = makeRng(seed);
    const text = randomString(r, 60, "abcde");
    const out = [];
    for (let q = 0; q < 8; q++) {
      const pattern = randomString(r, 1 + r(4), "abcde");
      const k = r(3);
      out.push(`${pattern}:${k}:${approximateSearchUkkonen(text, pattern, k).join(",")}`);
    }
    return out.join("|");
  }
  strictEqual(replay(7), replay(7), "identical seeds must replay identically");

  console.log(`banded/oracle distance comparisons=${comparisons}`);
  console.log(`three-way search agreements=${searchChecks}`);
  console.log("PASS");
}
