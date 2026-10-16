/** Task 956 - Implement grading rubric generator that assigns difficulty and expected complexity.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A static rubric: the source is tokenised and structurally parsed (blocks, branches, loops, simple
 * statements), then the loop nest is turned into a cost term. Nesting multiplies factors -- a linear
 * loop adds a polynomial degree, a halving loop (a `/`, `>>` or `*` in the update/clamp) adds a
 * logarithmic factor -- and sibling statements take the dominant term, so the analysis reports
 * O(1)/O(log n)/O(n)/O(n log n)/O(n^2)/O(n^3) from shape alone. The weighted criteria (branches,
 * nesting depth, lines, expected complexity, logarithmic loops) produce a score that maps to a tier,
 * and the reported per-criterion contributions must add up to that score exactly.
 *
 * Simplified: expressions are never evaluated -- only block structure and the presence of a
 * halving/doubling update are recognised, recursion and library calls are invisible, and "lines" is a
 * physical line count. Verified: six hand-labelled snippets, a lexical cross-check of the branch count
 * against the structural parse, monotonicity when a body is wrapped in another loop, the empty-program
 * boundary, the tier threshold semantics, contribution arithmetic, and deterministic re-analysis.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const tokenize = (src) => (src.replace(/\/\/[^\n]*/g, " ").match(/[A-Za-z_][A-Za-z0-9_]*|\d+|\S/g) || []);

/** Structural parse: `if`/`while`/`for` plus braced blocks; everything else is one simple statement. */
function parseProgram(tokens) {
  let index = 0;
  const parseSimple = () => {
    const start = index;
    while (index < tokens.length && tokens[index] !== ";" && tokens[index] !== "}") index += 1;
    if (tokens[index] === ";") index += 1;
    return { kind: "stmt", tokens: tokens.slice(start, index) };
  };
  const block = () => {
    const nodes = [];
    while (index < tokens.length && tokens[index] !== "}") {
      if (["if", "while", "for"].includes(tokens[index])) nodes.push(branch(tokens[index]));
      else if (tokens[index] === ";") index += 1;
      else nodes.push(parseSimple());
    }
    if (tokens[index] === "}") index += 1;
    return nodes;
  };
  const braced = () => { if (tokens[index] === "{") index += 1; return block(); };
  const branch = (keyword) => {
    index += 1;
    const head = [];
    while (index < tokens.length && tokens[index] !== "{") head.push(tokens[index++]);
    const node = { kind: keyword, head, body: braced(), alt: null };
    if (keyword === "if" && tokens[index] === "else") {
      index += 1;
      node.alt = tokens[index] === "if" ? [branch("if")] : braced();
    }
    return node;
  };
  return block();
}
/** A loop is logarithmic when its update (for) or clamp (while) halves, doubles or shifts. */
function isLogLoop(node) {
  const head = node.head.join(" ");
  if (node.kind === "for") return /[/]|>>|\*/.test(head.split(";").slice(2).join(";"));
  const first = node.body.find((n) => n.kind === "stmt");
  return /[/]|>>/.test(`${head} ${first ? first.tokens.join(" ") : ""}`);
}
const dominant = (a, b) => (b.poly > a.poly || (b.poly === a.poly && b.logs > a.logs) ? b : a);
/** Cost term of a statement list: loops multiply their body's term, siblings keep the dominant one. */
function costOf(nodes, metrics, depth) {
  let best = { poly: 0, logs: 0 };
  for (const node of nodes) {
    if (node.kind === "stmt") { metrics.statements += 1; continue; }
    metrics.decisions += 1;
    if (node.kind === "if") {
      best = dominant(best, costOf(node.body, metrics, depth));
      if (node.alt) best = dominant(best, costOf(node.alt, metrics, depth));
      continue;
    }
    metrics.maxNesting = Math.max(metrics.maxNesting, depth + 1);
    const inner = costOf(node.body, metrics, depth + 1);
    const log = isLogLoop(node);
    if (log) metrics.logLoops += 1;
    best = dominant(best, { poly: inner.poly + (log ? 0 : 1), logs: inner.logs + (log ? 1 : 0) });
  }
  return best;
}
function costLabel({ poly, logs }) {
  if (poly === 0) return logs ? "O(log n)" : "O(1)";
  return `O(n${poly === 1 ? "" : `^${poly}`}${logs ? " log n" : ""})`;
}
const COMPLEXITY_LADDER = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)"];
const complexityRank = (label) => COMPLEXITY_LADDER.indexOf(label);
const WEIGHTS = { branches: 2, nesting: 2.5, loc: 0.05, complexity: 4, logLoops: 1.5 };
const TIERS = [["Beginner", 8], ["Intermediate", 14], ["Advanced", 22], ["Expert", 30], ["Master / Challenge", Infinity]];
const tierFor = (score) => TIERS.find(([, limit]) => score < limit)[0];

/** Analyse one submission: structural metrics plus the complexity implied by the loop nest. */
function analyze(src) {
  const metrics = { decisions: 0, maxNesting: 0, statements: 0, logLoops: 0, loc: src.trim() ? src.trim().split("\n").length : 0, cyclomatic: 1, complexity: "O(1)" };
  metrics.complexity = costLabel(costOf(parseProgram(tokenize(src)), metrics, 0));
  metrics.cyclomatic = metrics.decisions + 1;
  return metrics;
}
/** Score the metrics and report the contribution of every criterion, so the rubric is auditable. */
function rubric(src) {
  const metrics = analyze(src);
  const contributions = [
    { criterion: "branches", weight: WEIGHTS.branches, units: metrics.cyclomatic - 1, value: WEIGHTS.branches * (metrics.cyclomatic - 1) },
    { criterion: "nesting", weight: WEIGHTS.nesting, units: metrics.maxNesting, value: WEIGHTS.nesting * metrics.maxNesting },
    { criterion: "loc", weight: WEIGHTS.loc, units: metrics.loc, value: WEIGHTS.loc * metrics.loc },
    { criterion: "complexity", weight: WEIGHTS.complexity, units: complexityRank(metrics.complexity), value: WEIGHTS.complexity * complexityRank(metrics.complexity) },
    { criterion: "logLoops", weight: WEIGHTS.logLoops, units: metrics.logLoops, value: WEIGHTS.logLoops * metrics.logLoops },
  ];
  const score = Number(contributions.reduce((a, c) => a + c.value, 0).toFixed(2));
  return { metrics, contributions, score, tier: tierFor(score), expectedComplexity: metrics.complexity };
}

function main() {
  const snippets = {
    constant: "x = 1;\ny = x + 2;",
    binarySearch: "while (lo <= hi) {\n  mid = (lo + hi) / 2;\n  if (a[mid] < t) { lo = mid + 1; } else { hi = mid - 1; }\n}",
    linear: "for (i = 0; i < n; i++) {\n  sum = sum + a[i];\n}",
    bubble: "for (i = 0; i < n; i++) {\n  for (j = 0; j < n - i; j++) {\n    if (a[j] > a[j + 1]) { swap(a[j], a[j + 1]); }\n  }\n}",
    nLogN: "for (i = 0; i < n; i++) {\n  while (x > 0) { x = x / 2; }\n}",
    cubic: "for (i = 0; i < n; i++) {\n  for (j = 0; j < n; j++) {\n    for (k = 0; k < n; k++) { s = s + a[i] * a[j] * a[k]; }\n  }\n}",
    empty: "",
  };
  const graded = Object.fromEntries(Object.entries(snippets).map(([name, src]) => [name, rubric(src)]));
  // 1. Happy path: six hand-labelled snippets must land on the complexity they were written for.
  const expected = { constant: "O(1)", binarySearch: "O(log n)", linear: "O(n)", nLogN: "O(n log n)", bubble: "O(n^2)", cubic: "O(n^3)" };
  for (const [name, label] of Object.entries(expected)) {
    assert.strictEqual(graded[name].expectedComplexity, label, `${name}: ${graded[name].expectedComplexity} != ${label}`);
  }
  assert(graded.binarySearch.metrics.logLoops === 1 && graded.nLogN.metrics.logLoops === 1, "halving loops detected");
  // 2. Independent lexical cross-check: keyword counts must match the structural parse.
  const keywordCount = (src) => (src.match(/\b(if|while|for)\b/g) || []).length;
  assert(Object.values(snippets).every((src) => keywordCount(src) === analyze(src).decisions),
    "branch count agrees lexically and structurally");
  // 3. Monotonicity: wrapping a body in another loop never lowers the rank or the score.
  const wrapped = `for (i = 0; i < n; i++) {\n${snippets.linear}\n}`;
  assert(complexityRank(graded.bubble.expectedComplexity) > complexityRank(graded.linear.expectedComplexity)
    && graded.bubble.score > graded.linear.score, "a second nest level is harder");
  assert(complexityRank(rubric(wrapped).expectedComplexity) > complexityRank(graded.linear.expectedComplexity)
    && rubric(wrapped).score > graded.linear.score && rubric(wrapped).metrics.maxNesting === 2, "wrapping raises the rank");
  // 4. Boundary: the empty program is trivial on every metric and lands in the lowest tier.
  assert.deepStrictEqual(graded.empty.metrics, { decisions: 0, maxNesting: 0, statements: 0, logLoops: 0, loc: 0, cyclomatic: 1, complexity: "O(1)" },
    "empty program metrics");
  assert(graded.empty.score === 0 && graded.empty.tier === "Beginner" && graded.empty.contributions.length === 5, "empty score");
  // 5. Oracle: the contribution rows must add up to the reported score, recomputed independently.
  const sum = (r) => Number(r.contributions.reduce((a, c) => a + c.value, 0).toFixed(2));
  assert(Object.values(graded).every((r) => Number(r.score.toFixed(2)) === sum(r)), "contributions sum to the score");
  const hand = 2 * (graded.bubble.metrics.cyclomatic - 1) + 2.5 * 2 + 0.05 * graded.bubble.metrics.loc + 4 * 4;
  assert.strictEqual(graded.bubble.score, Number(hand.toFixed(2)), `bubble score ${graded.bubble.score} == ${hand}`);
  assert(Object.values(WEIGHTS).every((w) => w > 0) && Object.keys(WEIGHTS).length === graded.bubble.contributions.length,
    "every criterion has a positive weight");
  // 6. Tier thresholds: a score exactly at a limit moves up, one hundredth below stays down.
  assert(tierFor(7.99) === "Beginner" && tierFor(8) === "Intermediate" && tierFor(30) === "Master / Challenge", "threshold semantics");
  assert(tierFor(graded.cubic.score) !== tierFor(graded.constant.score) && graded.cubic.score > graded.nLogN.score,
    "harder snippets rank into higher tiers");
  // 7. Seeded fuzz: hand-built nests must land on the label their own shape implies.
  const rng = makeRng(20240956);
  let fuzzed = 0;
  for (let trial = 0; trial < 40; trial += 1) {
    const halve = trial % 4 === 0;
    const depth = halve ? 1 : 1 + rng.int(3);
    const loop = halve ? "for (i = 1; i < n; i = i * 2) { %BODY% }" : "for (i = 0; i < n; i++) { %BODY% }";
    let src = "x = x + 1;";
    for (let level = 0; level < depth; level += 1) src = loop.replace("%BODY%", src);
    const result = rubric(src);
    const wanted = halve ? "O(log n)" : ["O(n)", "O(n^2)", "O(n^3)"][depth - 1];
    if (result.expectedComplexity === wanted && result.metrics.maxNesting === depth
      && result.metrics.logLoops === (halve ? 1 : 0)) fuzzed += 1;
  }
  assert.strictEqual(fuzzed, 40, `fuzz mismatches ${40 - fuzzed}`);
  // 8. Determinism: re-analysing every snippet reproduces the same rubric byte for byte.
  assert.deepStrictEqual(Object.fromEntries(Object.entries(snippets).map(([n, s]) => [n, rubric(s)])), graded, "re-analysis identical");
  console.log(`rubric: snippets=${Object.keys(graded).length} bubbleScore=${graded.bubble.score} tier=${graded.bubble.tier} `
    + `cubicTier=${graded.cubic.tier} fuzz=${fuzzed}/${40}`);
  console.log("PASS");
}

module.exports = { makeRng, tokenize, parseProgram, isLogLoop, dominant, costOf, costLabel, COMPLEXITY_LADDER,
  complexityRank, WEIGHTS, TIERS, tierFor, analyze, rubric };
if (require.main === module) main();
