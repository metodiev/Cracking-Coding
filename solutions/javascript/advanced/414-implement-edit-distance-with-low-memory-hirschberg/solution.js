/** Task 414 - Implement edit distance with low memory (Hirschberg). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Classic Levenshtein DP uses an O(n*m) table. Hirschberg's algorithm recovers an
 * optimal alignment with O(n) auxiliary memory and O(n*m) time:
 *   * split b in half (Myers-Miller variant of the split), compute the forward row
 *     f[i] = dist(a[:i], b[:mid]) and the backward row g[i] = dist(a[i:], b[mid:])
 *     with two rolling rows each — the backward row reverses both strings because
 *     reversal preserves edit distance and turns suffixes into prefixes,
 *   * pick the split k minimizing f[k] + g[k] (smallest k on ties), recurse on
 *     (a[:k], b[:mid]) and (a[k:], b[mid:]); base cases solve directly.
 * An alignment is a list of ops ['=', x, y] match; ['!', x, y] substitution;
 * ['-', x, ''] deletion; ['+', '', y] insertion. Replaying the ops from a rebuilds b.
 */
"use strict";

const BASE = 16;

/** Oracle: full O(n*m) table over rolling rows. */
function fullDpDist(a, b) {
  const m = b.length;
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = new Array(m + 1).fill(0);
    cur[0] = i;
    for (let j = 1; j <= m; j++) {
      const sub = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub);
    }
    prev = cur;
  }
  return prev[m];
}

/** f[i] = dist(a[:i], prefix) for every i; one rolling row over the prefix. */
function forwardRow(a, prefix) {
  const m = prefix.length;
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  const out = [prev[m]];
  for (let i = 1; i <= a.length; i++) {
    const cur = new Array(m + 1).fill(0);
    cur[0] = i;
    for (let j = 1; j <= m; j++) {
      const sub = a[i - 1] === prefix[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub);
    }
    prev = cur;
    out.push(prev[m]);
  }
  return out;
}

/** g[i] = dist(a[i:], suffix) for every i (reversal turns suffixes into prefixes). */
function backwardRow(a, suffix) {
  const ra = a.split("").reverse().join("");
  const rs = suffix.split("").reverse().join("");
  const m = rs.length;
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  const collected = [prev[m]];
  for (let p = 1; p <= ra.length; p++) {
    const cur = new Array(m + 1).fill(0);
    cur[0] = p;
    for (let j = 1; j <= m; j++) {
      const sub = ra[p - 1] === rs[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub);
    }
    prev = cur;
    collected.push(prev[m]);
  }
  const n = a.length;
  const out = new Array(n + 1);
  for (let i = 0; i <= n; i++) out[i] = collected[n - i];
  return out;
}

/** Full-table DP with backtracking; used for the recursion base cases. */
function directAlign(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 1; i <= n; i++) dp[i][0] = i;
  for (let j = 1; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const sub = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + sub);
    }
  }
  const ops = [];
  let i = n, j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1] && dp[i][j] === dp[i - 1][j - 1]) {
      ops.push(["=", a[i - 1], b[j - 1]]);
      i--; j--;
    } else if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) {
      ops.push(["!", a[i - 1], b[j - 1]]);
      i--; j--;
    } else if (i > 0 && dp[i][j] === dp[i - 1][j] + 1) {
      ops.push(["-", a[i - 1], ""]);
      i--;
    } else {
      ops.push(["+", "", b[j - 1]]);
      j--;
    }
  }
  ops.reverse();
  return { dist: dp[n][m], ops };
}

function hirschbergAlign(a, b, base = BASE) {
  if (a.length === 0) {
    return { dist: b.length, ops: [...b].map((c) => ["+", "", c]) };
  }
  if (b.length === 0) {
    return { dist: a.length, ops: [...a].map((c) => ["-", c, ""]) };
  }
  if (Math.min(a.length, b.length) <= Math.max(base, 1)) return directAlign(a, b);
  const mid = Math.floor(b.length / 2);
  const f = forwardRow(a, b.slice(0, mid));
  const g = backwardRow(a, b.slice(mid));
  let split = 0, best = Infinity;
  for (let i = 0; i <= a.length; i++) {
    if (f[i] + g[i] < best) { best = f[i] + g[i]; split = i; }
  }
  const left = hirschbergAlign(a.slice(0, split), b.slice(0, mid), base);
  const right = hirschbergAlign(a.slice(split), b.slice(mid), base);
  return { dist: left.dist + right.dist, ops: left.ops.concat(right.ops) };
}

/** Apply an alignment to a; returns the produced string. */
function replay(a, ops) {
  let out = "";
  let ai = 0;
  for (const [kind, x, y] of ops) {
    if (kind === "=" || kind === "!" || kind === "-") {
      if (a[ai] !== x) throw new Error("bad alignment column");
      ai++;
    }
    if (kind === "=" || kind === "!" || kind === "+") out += y;
  }
  if (ai !== a.length) throw new Error("source not fully consumed");
  return out;
}

function countKind(ops, kind) {
  return ops.filter((op) => op[0] === kind).length;
}

let seed = 4242;

function lcgString(length, alphabet) {
  let out = "";
  for (let k = 0; k < length; k++) {
    seed = (seed * 75 + 74) % 65537;
    out += alphabet[seed % alphabet.length];
  }
  return out;
}

module.exports = { fullDpDist, forwardRow, backwardRow, directAlign, hirschbergAlign, replay, countKind, lcgString, BASE };

if (require.main === module) {
  const assert = require("node:assert");

  // hand-derived distances (Levenshtein, substitution cost 1)
  assert.strictEqual(fullDpDist("kitten", "sitting"), 3);
  assert.strictEqual(fullDpDist("flaw", "lawn"), 2);
  assert.strictEqual(fullDpDist("sunday", "saturday"), 3);
  assert.strictEqual(fullDpDist("", ""), 0);
  assert.strictEqual(fullDpDist("a", "b"), 1);
  assert.strictEqual(fullDpDist("abc", ""), 3);
  assert.strictEqual(fullDpDist("abc", "abc"), 0);
  assert.strictEqual(fullDpDist("abc", "yabd"), 2);

  let res = hirschbergAlign("kitten", "sitting");
  assert.strictEqual(res.dist, 3);
  assert.strictEqual(replay("kitten", res.ops), "sitting");
  assert.strictEqual(countKind(res.ops, "="), 4);
  assert.strictEqual(countKind(res.ops, "!"), 2);
  assert.strictEqual(countKind(res.ops, "+"), 1);
  assert.strictEqual(countKind(res.ops, "-"), 0);

  res = hirschbergAlign("flaw", "lawn", 1);
  assert.strictEqual(res.dist, 2);
  assert.strictEqual(replay("flaw", res.ops), "lawn");
  assert.strictEqual(countKind(res.ops, "!") + countKind(res.ops, "-") + countKind(res.ops, "+"), 2);
  assert.strictEqual(countKind(res.ops, "="), 3);

  // base-case paths: empty strings, pure insertions/deletions, equal strings
  const ins = hirschbergAlign("", "abc");
  assert.strictEqual(ins.dist, 3);
  assert.strictEqual(ins.ops.length, 3);
  const del = hirschbergAlign("abc", "");
  assert.strictEqual(del.dist, 3);
  assert.strictEqual(del.ops.length, 3);
  const none = hirschbergAlign("", "");
  assert.strictEqual(none.dist, 0);
  assert.strictEqual(none.ops.length, 0);
  const same = hirschbergAlign("same", "same");
  assert.strictEqual(same.dist, 0);
  assert.strictEqual(replay("same", same.ops), "same");
  assert.strictEqual(countKind(same.ops, "="), 4);

  // both base settings must agree with the full table on hand and random pairs
  const pairs = [["kitten", "sitting"], ["flaw", "lawn"], ["sunday", "saturday"], ["abc", "yabd"]];
  for (const [a, b] of pairs) {
    for (const base of [0, 1, 3, 16, 1000]) {
      const r2 = hirschbergAlign(a, b, base);
      assert.strictEqual(r2.dist, fullDpDist(a, b), `pair ${a}/${b} base ${base}`);
      assert.strictEqual(replay(a, r2.ops), b, `pair replay ${a} base ${base}`);
    }
  }

  const alphabet = "abcd";
  for (let round = 0; round < 10; round++) {
    const la = 30 + (seed % 60);
    const a = lcgString(la, alphabet);
    const lb = 25 + (seed % 75);
    const b = lcgString(lb, alphabet);
    const want = fullDpDist(a, b);
    for (const base of [2, 16, 400]) {
      const r2 = hirschbergAlign(a, b, base);
      assert.strictEqual(r2.dist, want, `lcg round ${round} base ${base}`);
      assert.strictEqual(replay(a, r2.ops), b, `lcg replay ${round} base ${base}`);
      assert.strictEqual(countKind(r2.ops, "=") + countKind(r2.ops, "!") + countKind(r2.ops, "-"), a.length);
      assert.strictEqual(countKind(r2.ops, "=") + countKind(r2.ops, "!") + countKind(r2.ops, "+"), b.length);
      assert.strictEqual(countKind(r2.ops, "!") + countKind(r2.ops, "-") + countKind(r2.ops, "+"), want);
      for (const [kind, x, y] of r2.ops) {
        if (kind === "=") assert.strictEqual(x, y);
        if (kind === "!") assert.notStrictEqual(x, y);
      }
    }
  }

  // recurrence sanity: mid rows of the recursion match the full-table boundary values
  const a2 = "alignment", b2 = "alignmentx";
  const mid = Math.floor(b2.length / 2);
  const f = forwardRow(a2, b2.slice(0, mid));
  const g = backwardRow(a2, b2.slice(mid));
  assert.strictEqual(f[a2.length], fullDpDist(a2, b2.slice(0, mid)));
  assert.strictEqual(g[0], fullDpDist(a2, b2.slice(mid)));
  for (let i = 0; i <= a2.length; i++) {
    assert.strictEqual(f[i], fullDpDist(a2.slice(0, i), b2.slice(0, mid)), `f row ${i}`);
    assert.strictEqual(g[i], fullDpDist(a2.slice(i), b2.slice(mid)), `g row ${i}`);
  }
  let split = 0, best = Infinity;
  for (let i = 0; i <= a2.length; i++) {
    if (f[i] + g[i] < best) { best = f[i] + g[i]; split = i; }
  }
  assert.strictEqual(best, fullDpDist(a2, b2));

  console.log("PASS");
}
