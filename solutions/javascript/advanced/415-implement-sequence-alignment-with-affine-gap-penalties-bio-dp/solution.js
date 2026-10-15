/** Task 415 - Implement sequence alignment with affine gap penalties (bio-DP). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Gotoh's algorithm aligns two strings under affine gap costs. Cost model (documented):
 *   * substitution: MISMATCH = 2, matched characters cost 0,
 *   * a gap of length L costs OPEN + (L - 1) * EXTEND, with OPEN = 2 and EXTEND = 1:
 *     opening the run costs OPEN, every extra character costs EXTEND.
 * Three DP states drive the recurrence:
 *   * X - the column ends with a (mis)match of the last two characters,
 *   * Y - the column ends with a gap in b (a character of a is deleted),
 *   * Z - the column ends with a gap in a (a character of b is inserted).
 * Gap-to-gap switches are allowed so that with EXTEND == OPEN the optimum equals
 * Needleman-Wunsch with a linear gap cost of OPEN per character (asserted below).
 * An alignment is a list of ops ['=', x, y] match; ['!', x, y] substitution;
 * ['-', x, ''] deletion; ['+', '', y] insertion; replaying the ops rebuilds b.
 */
"use strict";

const MISMATCH = 2;
const OPEN = 2;
const EXTEND = 1;
const INF = Number.MAX_SAFE_INTEGER / 8;

function argMin3(a, b, c) {
  if (a <= b && a <= c) return 0;
  if (b <= c) return 1;
  return 2;
}

/** Returns { cost, ops } for the optimal affine-gap global alignment. */
function gotoh(a, b, mismatch = MISMATCH, openCost = OPEN, extend = EXTEND) {
  const n = a.length, m = b.length;
  const x = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(INF));
  const y = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(INF));
  const z = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(INF));
  x[0][0] = 0;
  for (let i = 1; i <= n; i++) y[i][0] = openCost + (i - 1) * extend;
  for (let j = 1; j <= m; j++) z[0][j] = openCost + (j - 1) * extend;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const sub = a[i - 1] === b[j - 1] ? 0 : mismatch;
      x[i][j] = Math.min(x[i - 1][j - 1], y[i - 1][j - 1], z[i - 1][j - 1]) + sub;
      y[i][j] = Math.min(x[i - 1][j] + openCost, z[i - 1][j] + openCost, y[i - 1][j] + extend);
      z[i][j] = Math.min(x[i][j - 1] + openCost, y[i][j - 1] + openCost, z[i][j - 1] + extend);
    }
  }
  let state = argMin3(x[n][m], y[n][m], z[n][m]);
  const cost = state === 0 ? x[n][m] : state === 1 ? y[n][m] : z[n][m];
  const ops = [];
  let i = n, j = m;
  while (i > 0 || j > 0) {
    if (state === 0) {
      const sub = a[i - 1] === b[j - 1] ? 0 : mismatch;
      ops.push([sub === 0 ? "=" : "!", a[i - 1], b[j - 1]]);
      const k = argMin3(x[i - 1][j - 1], y[i - 1][j - 1], z[i - 1][j - 1]);
      const cand = k === 0 ? x[i - 1][j - 1] : k === 1 ? y[i - 1][j - 1] : z[i - 1][j - 1];
      if (cand + sub !== x[i][j]) throw new Error("X backtrack");
      state = k;
      i--;
      j--;
    } else if (state === 1) {
      ops.push(["-", a[i - 1], ""]);
      const k = argMin3(x[i - 1][j] + openCost, z[i - 1][j] + openCost, y[i - 1][j] + extend);
      const cand = k === 0 ? x[i - 1][j] + openCost : k === 1 ? z[i - 1][j] + openCost : y[i - 1][j] + extend;
      if (cand !== y[i][j]) throw new Error("Y backtrack");
      state = k === 0 ? 0 : k === 1 ? 2 : 1;
      i--;
    } else {
      ops.push(["+", "", b[j - 1]]);
      const k = argMin3(x[i][j - 1] + openCost, y[i][j - 1] + openCost, z[i][j - 1] + extend);
      const cand = k === 0 ? x[i][j - 1] + openCost : k === 1 ? y[i][j - 1] + openCost : z[i][j - 1] + extend;
      if (cand !== z[i][j]) throw new Error("Z backtrack");
      state = k;
      j--;
    }
  }
  ops.reverse();
  return { cost, ops };
}

/** Linear gap cost: every gap character costs `gap` regardless of run length. */
function needlemanWunsch(a, b, gap, mismatch = MISMATCH) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 1; i <= n; i++) dp[i][0] = i * gap;
  for (let j = 1; j <= m; j++) dp[0][j] = j * gap;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const sub = a[i - 1] === b[j - 1] ? 0 : mismatch;
      dp[i][j] = Math.min(dp[i - 1][j] + gap, dp[i][j - 1] + gap, dp[i - 1][j - 1] + sub);
    }
  }
  return dp[n][m];
}

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

/** Recompute the cost from the op pattern alone, validating the cost model. */
function opsCost(ops, mismatch = MISMATCH, openCost = OPEN, extend = EXTEND) {
  let total = 0;
  let i = 0;
  while (i < ops.length) {
    const kind = ops[i][0];
    if (kind === "!") {
      total += mismatch;
      i++;
    } else if (kind === "-" || kind === "+") {
      let run = 0;
      while (i < ops.length && ops[i][0] === kind) { run++; i++; }
      total += openCost + (run - 1) * extend;
    } else {
      i++;
    }
  }
  return total;
}

let seed = 20270;

function lcgString(length, alphabet) {
  let out = "";
  for (let k = 0; k < length; k++) {
    seed = (seed * 75 + 74) % 65537;
    out += alphabet[seed % alphabet.length];
  }
  return out;
}

module.exports = { gotoh, needlemanWunsch, replay, opsCost, lcgString, argMin3, MISMATCH, OPEN, EXTEND };

if (require.main === module) {
  const assert = require("node:assert");

  // equal strings align for free
  let res = gotoh("GATTACA", "GATTACA");
  assert.strictEqual(res.cost, 0);
  assert.strictEqual(res.ops.length, 7);
  for (const op of res.ops) assert.strictEqual(op[0], "=");

  // a single substitution (2) beats delete+insert (2 + 2)
  res = gotoh("a", "b");
  assert.strictEqual(res.cost, 2);
  assert.strictEqual(res.ops.length, 1);
  assert.strictEqual(res.ops[0][0], "!");

  // one insertion: a single gap of length 1 costs OPEN = 2
  res = gotoh("ACGT", "ACCGT");
  assert.strictEqual(res.cost, 2);
  assert.strictEqual(opsCost(res.ops), 2);
  assert.strictEqual(replay("ACGT", res.ops), "ACCGT");

  // one gap of length 2 costs OPEN + EXTEND = 3; two separate gaps would cost 4
  res = gotoh("acgt", "acgggt");
  assert.strictEqual(res.cost, 3);
  assert.strictEqual(replay("acgt", res.ops), "acgggt");
  assert.strictEqual(opsCost(res.ops), 3);
  assert.strictEqual(needlemanWunsch("acgt", "acgggt", 2), 4);
  assert.strictEqual(needlemanWunsch("ab", "aZZb", 2), 4);
  const ab = gotoh("ab", "aZZb");
  assert.strictEqual(ab.cost, 3);
  assert.strictEqual(opsCost(ab.ops), 3);
  assert.strictEqual(gotoh("acgt", "acgggt", 2, 2, 2).cost, needlemanWunsch("acgt", "acgggt", 2, 2));

  // empty-string cases: a gap of length L costs OPEN + (L-1)*EXTEND
  const empty = gotoh("", "");
  assert.strictEqual(empty.cost, 0);
  assert.strictEqual(empty.ops.length, 0);
  res = gotoh("abc", "");
  assert.strictEqual(res.cost, OPEN + 2 * EXTEND);
  assert.strictEqual(res.cost, 4);
  assert.strictEqual(res.ops.length, 3);
  for (const op of res.ops) assert.strictEqual(op[0], "-");
  res = gotoh("", "abc");
  assert.strictEqual(res.cost, 4);
  assert.strictEqual(res.ops.length, 3);
  assert.strictEqual(gotoh("", "AAAAA").cost, OPEN + 4 * EXTEND);
  assert.strictEqual(gotoh("", "AAAAA").cost, 6);
  assert.strictEqual(needlemanWunsch("", "AAAAA", 2), 10);

  // longer hand case: 3-character insertion in the middle
  res = gotoh("ABCD", "ABXYZCD");
  assert.strictEqual(res.cost, OPEN + 2 * EXTEND);
  assert.strictEqual(res.cost, 4);
  assert.strictEqual(replay("ABCD", res.ops), "ABXYZCD");
  assert.strictEqual(opsCost(res.ops), 4);

  // replay + cost-model consistency on the hand pairs
  for (const [a, b] of [["kitten", "sitting"], ["sunday", "saturday"], ["flaw", "lawn"]]) {
    const r2 = gotoh(a, b);
    assert.strictEqual(replay(a, r2.ops), b, `hand replay ${a}`);
    assert.strictEqual(opsCost(r2.ops), r2.cost, `hand ops cost ${a}`);
  }

  // with EXTEND == OPEN the affine optimum equals Needleman-Wunsch
  const alphabet = "acgt";
  for (let round = 0; round < 8; round++) {
    const la = 25 + (seed % 40);
    const a = lcgString(la, alphabet);
    const lb = 20 + (seed % 45);
    const b = lcgString(lb, alphabet);
    const affine = gotoh(a, b, 2, 3, 3);
    const linear = needlemanWunsch(a, b, 3, 2);
    assert.strictEqual(affine.cost, linear, `affine == linear round ${round}`);
    assert.strictEqual(replay(a, affine.ops), b, `lcg replay ${round}`);
    assert.strictEqual(opsCost(affine.ops, 2, 3, 3), affine.cost, `lcg ops cost ${round}`);
    // a cheaper affine extension can never lose against the linear model
    assert.ok(gotoh(a, b, 2, 2, 1).cost <= linear, `affine bound round ${round}`);
  }

  // gap-run counting sanity: a long deletion is one open plus many extends
  res = gotoh("ABCDEFGH", "AB");
  assert.strictEqual(res.cost, OPEN + 5 * EXTEND);
  assert.strictEqual(res.cost, 7);
  assert.strictEqual(res.ops.filter((op) => op[0] === "-").length, 6);
  assert.strictEqual(opsCost(res.ops), 7);
  assert.strictEqual(replay("ABCDEFGH", res.ops), "AB");

  console.log("PASS");
}
