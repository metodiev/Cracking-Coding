/** Task 958 - Implement plagiarism detector comparing code structure and logic patterns.
 * (Tier: Master, CTCI Ch 11 - Testing)
 *
 * A MOSS-style detector over a small JavaScript subset. Comments are stripped and the token stream is
 * abstracted (identifiers -> `id`, numbers -> `num`, strings -> `str`), so renaming a variable or
 * reformatting a file cannot change a fingerprint. Three independent families are compared by Jaccard
 * similarity: winnowed 4-gram token hashes (copied code), 2-gram "shape" hashes (structure under
 * renaming) and 2-grams of the control-keyword skeleton (`if`/`for`/`while`/`return` order = the logic
 * pattern). Similarity is the weighted blend; pairs at or above the threshold are flagged and clustered
 * with union-find into plagiarism groups.
 *
 * Simplified: the parser is a tokenizer plus keyword skeleton, not a real AST -- nesting is encoded only
 * through the keyword sequence, and the winnowing window is fixed at 4. Verified: verbatim copies score
 * 1.0, a renamed+reformatted clone has byte-identical fingerprints, an independent merge-based Jaccard
 * oracle agrees, unrelated submissions stay below the threshold, clustering yields the expected groups,
 * a logic mutation drops the control similarity while leaving the shape intact, the empty/comment-only
 * boundary, threshold semantics and deterministic re-analysis.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const KEYWORDS = new Set(["if", "else", "while", "for", "return", "do", "break", "continue", "function", "const", "let", "var", "new"]);
const TOKEN = /[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|"[^"]*"|'[^']*'|=>|===|!==|==|!=|<=|>=|&&|\|\||\S/g;
const stripComments = (src) => src.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");
const tokenize = (src) => stripComments(src).match(TOKEN) || [];
/** Abstract away naming: identifiers, numbers and strings collapse to one symbol each. */
const abstract = (tokens) => tokens.map((token) => (KEYWORDS.has(token) ? token
  : /^\d/.test(token) ? "num" : /^["']/.test(token) ? "str" : /^[A-Za-z_]/.test(token) ? "id" : token));
function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
const kGrams = (tokens, k) => tokens.slice(0, Math.max(0, tokens.length - k + 1)).map((_, i) => tokens.slice(i, i + k).join("\u0001"));
/** Winnowing (Schleimer et al.): keep the minimum k-gram hash of every window, so the fingerprint is
 *  O(n/w) sized while any shared run of >= k tokens still contributes a common hash. */
function winnow(grams, windowSize = 4) {
  const hashes = grams.map(fnv1a);
  const kept = new Set();
  if (hashes.length <= windowSize) { hashes.forEach((h) => kept.add(h)); return kept; }
  for (let i = 0; i + windowSize <= hashes.length; i += 1) kept.add(Math.min(...hashes.slice(i, i + windowSize)));
  return kept;
}
/** Compute the three fingerprints of one submission. */
function fingerprints(src) {
  const abs = abstract(tokenize(src));
  const control = abs.filter((token) => KEYWORDS.has(token));
  return { tokens: winnow(kGrams(abs, 4)), shape: new Set(kGrams(abs, 2)), control: new Set(kGrams(control, 2)), length: abs.length };
}
const WEIGHTS = { tokens: 0.5, shape: 0.3, control: 0.2 };
const THRESHOLD = 0.6;
function jaccard(a, b) {
  if (a.size === 0 || b.size === 0) return a.size === 0 && b.size === 0 ? 1 : 0;
  let intersection = 0;
  for (const item of a) if (b.has(item)) intersection += 1;
  return intersection / (a.size + b.size - intersection);
}
/** Independent Jaccard oracle: sorted-array merge instead of Set membership. */
function jaccardByMerge(a, b) {
  const left = [...a].sort(), right = [...b].sort();
  let i = 0, j = 0, intersection = 0;
  while (i < left.length && j < right.length) {
    if (left[i] === right[j]) { intersection += 1; i += 1; j += 1; } else if (left[i] < right[j]) i += 1; else j += 1;
  }
  return left.length + right.length === intersection ? 1 : intersection / (left.length + right.length - intersection);
}
const similarity = (a, b) => WEIGHTS.tokens * jaccard(a.tokens, b.tokens) + WEIGHTS.shape * jaccard(a.shape, b.shape)
  + WEIGHTS.control * jaccard(a.control, b.control);
const isFlagged = (score) => score >= THRESHOLD;

/** All flagged pairs plus the plagiarism groups they induce (union-find, group = connected component). */
function detect(pack, { threshold = THRESHOLD } = {}) {
  const ids = Object.keys(pack).sort();
  const prints = Object.fromEntries(ids.map((id) => [id, fingerprints(pack[id])]));
  const pairs = [];
  for (let i = 0; i < ids.length; i += 1) for (let j = i + 1; j < ids.length; j += 1) {
    const score = similarity(prints[ids[i]], prints[ids[j]]);
    if (score >= threshold) pairs.push({ a: ids[i], b: ids[j], score: Number(score.toFixed(4)) });
  }
  const parent = new Map(ids.map((id) => [id, id]));
  const find = (id) => (parent.get(id) === id ? id : (parent.set(id, find(parent.get(id))).get(id)));
  for (const pair of pairs) parent.set(find(pair.a), find(pair.b));
  const groups = new Map();
  for (const id of ids) groups.set(find(id), [...(groups.get(find(id)) || []), id]);
  return { prints, pairs, groups: [...groups.values()].sort((a, b) => b.length - a.length || (a[0] < b[0] ? -1 : 1)) };
}

const PACK = {
  alpha: "function sumEven(xs) {\n  let total = 0;\n  for (let i = 0; i < xs.length; i++) {\n    if (xs[i] % 2 === 0) { total += xs[i]; }\n  }\n  return total;\n}",
  beta: "// sums the even entries\nfunction sumEven(list) {\n  /* running total */\n  let acc = 0;\n  for (let k = 0; k < list.length; k++) {\n    if (list[k] % 2 === 0) {\n      acc += list[k];\n    }\n  }\n  return acc;\n}",
  gamma: "function search(a, t) {\n  let lo = 0, hi = a.length - 1;\n  while (lo <= hi) {\n    const mid = (lo + hi) >> 1;\n    if (a[mid] === t) return mid;\n    if (a[mid] < t) lo = mid + 1; else hi = mid - 1;\n  }\n  return -1;\n}",
  delta: "function matmul(a, b) {\n  const out = [];\n  for (let i = 0; i < a.length; i++) {\n    out.push([]);\n    for (let j = 0; j < b[0].length; j++) {\n      let cell = 0;\n      for (let k = 0; k < b.length; k++) cell += a[i][k] * b[k][j];\n      out[i].push(cell);\n    }\n  }\n  return out;\n}",
  epsilon: "function isPalindrome(s) {\n  let i = 0, j = s.length - 1;\n  while (i < j) {\n    if (s[i] !== s[j]) return false;\n    i += 1; j -= 1;\n  }\n  return true;\n}",
};

function main() {
  const rng = makeRng(20240958);
  const prints = Object.fromEntries(Object.entries(PACK).map(([id, src]) => [id, fingerprints(src)]));
  // 1. Happy path: a verbatim copy is a perfect match in every family.
  const clonePrints = fingerprints(PACK.alpha);
  assert.strictEqual(similarity(clonePrints, prints.alpha), 1, "identical sources score exactly 1");
  assert(["tokens", "shape", "control"].every((family) => jaccard(clonePrints[family], prints.alpha[family]) === 1),
    "every fingerprint family matches for a verbatim copy");
  // 2. Renaming and reformatting cannot move a fingerprint: beta is alpha with new names and comments.
  const setEquals = (a, b) => a.size === b.size && [...a].every((x) => b.has(x));
  assert(["tokens", "shape", "control"].every((family) => setEquals(prints.beta[family], prints.alpha[family])),
    "renaming, recommenting and reformatting leave the fingerprints identical");
  assert.strictEqual(similarity(prints.beta, prints.alpha), 1, "the renamed clone still scores 1.0");
  // 3. Independent oracle: merge-based Jaccard equals the Set-based one on seeded subsets.
  let oracleAgreements = 0;
  const universe = [...new Set([...prints.alpha.tokens, ...prints.gamma.tokens].map(String))];
  for (let trial = 0; trial < 60; trial += 1) {
    const pick = () => new Set(universe.filter(() => rng.int(2) === 0));
    const left = pick(), right = pick();
    if (Math.abs(jaccard(left, right) - jaccardByMerge(left, right)) < 1e-12) oracleAgreements += 1;
  }
  assert.strictEqual(oracleAgreements, 60, `jaccard oracle disagreements ${60 - oracleAgreements}`);
  // 4. False-positive control: unrelated submissions stay below the threshold.
  const unrelated = [["alpha", "gamma"], ["alpha", "delta"], ["alpha", "epsilon"], ["gamma", "delta"]];
  assert(unrelated.every(([a, b]) => similarity(prints[a], prints[b]) < THRESHOLD),
    `unrelated pairs: ${unrelated.map(([a, b]) => `${a}/${b}=${similarity(prints[a], prints[b]).toFixed(2)}`).join(" ")}`);
  // 5. Clustering: alpha+beta+its clone form one group, the other submissions stay singletons.
  const pack = { ...PACK, zeta: PACK.alpha };
  const report = detect(pack);
  assert.deepStrictEqual(report.groups.map((g) => g.length), [3, 1, 1, 1], `groups ${JSON.stringify(report.groups)}`);
  assert(report.groups[0].join(",") === "alpha,beta,zeta", `the clone group is ${report.groups[0].join(",")}`);
  assert(report.pairs.length === 3 && report.pairs.every((p) => p.score === 1), "three perfect pairs flagged");
  // 6. Logic mutation: swapping the branch for a loop keeps the shape but changes the control skeleton.
  const mutated = PACK.alpha.replace("if (xs[i] % 2 === 0) { total += xs[i]; }", "while (xs[i] % 2 !== 0) { xs[i] = xs[i] + 1; }");
  const mutatedPrints = fingerprints(mutated);
  assert(jaccard(mutatedPrints.control, prints.alpha.control) < 1 && jaccard(mutatedPrints.shape, prints.alpha.shape) > 0.5,
    `logic change visible in the control family (shape ${jaccard(mutatedPrints.shape, prints.alpha.shape).toFixed(2)})`);
  assert(similarity(mutatedPrints, prints.alpha) < 1, "a logic mutation is no longer a perfect match");
  // 7. Boundaries: empty and comment-only sources carry no fingerprints, and score 0 against real code.
  const empty = fingerprints("");
  const commentOnly = fingerprints("// nothing but a note\n/* and a block */");
  assert(empty.tokens.size === 0 && empty.shape.size === 0 && empty.control.size === 0 && empty.length === 0, "empty fingerprints");
  assert(commentOnly.shape.size === 0 && jaccard(empty.shape, commentOnly.shape) === 1, "comment-only equals empty");
  assert(similarity(empty, prints.alpha) === 0, "an empty submission matches nothing");
  assert(isFlagged(THRESHOLD) && !isFlagged(THRESHOLD - 0.0001), "the threshold itself is flagged (inclusive)");
  // 8. Determinism: fingerprinting and detection replay byte for byte.
  const again = detect({ ...PACK, zeta: PACK.alpha });
  assert.deepStrictEqual(again.pairs, report.pairs, "the flagged pairs replay");
  assert.deepStrictEqual(again.groups, report.groups, "the groups replay");
  assert.deepStrictEqual(Object.values(again.prints), Object.values(report.prints), "fingerprints are stable");
  console.log(`plagiarism: submissions=${Object.keys(pack).length} pairs=${report.pairs.length} `
    + `groups=${report.groups.map((g) => g.length).join("/")} threshold=${THRESHOLD}`);
  console.log("PASS");
}

module.exports = { makeRng, KEYWORDS, stripComments, tokenize, abstract, fnv1a, kGrams, winnow, fingerprints,
  WEIGHTS, THRESHOLD, jaccard, jaccardByMerge, similarity, isFlagged, detect, PACK };
if (require.main === module) main();
