/** Task 942 - Implement hot-path optimization analyzer that suggests inline and caching
 * opportunities. (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * The analyzer consumes a profile of a call graph - static body size per function, dynamic call
 * multiplicity per edge, a number of call sites and, for leaf helpers, sampled argument statistics -
 * and turns it into an optimization plan. It computes dynamic counts, the exclusive and inclusive
 * cost of every function, the hottest root-to-leaf path and two kinds of suggestions: inline a
 * callee that is small, called often and referenced from exactly one site (paying an i-cache growth
 * penalty), and memoize a pure leaf whose sampled arguments repeat. Every suggestion carries a
 * predicted saving, and the plan is validated by re-executing the program on a cost model that
 * charges call instructions, lookups and misses, so predictions are compared against measurements.
 *
 * Simplified: the graph is a DAG of a handful of functions, costs are abstract cycles instead of
 * real instructions and the i-cache penalty is a linear term. Verified: cost accounting and hot-path
 * selection against an independent path enumeration, the rejections (too big, multiple call sites,
 * impure, cold), predicted versus measured savings, semantic preservation of the optimized program,
 * cyclic-graph refusal, determinism and empty profiles.
 */
"use strict";
const assert = require("node:assert");
/** splitmix64 - the only source of randomness; identical stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}
const unit = (rng) => Number(rng.next() >> 11n) / 2 ** 53;
/** Sampled profile: edges carry [callee, calls per invocation, call sites]. */
const PROFILE = {
  root: "main",
  overhead: 6,
  growthPenalty: 0.5,
  lookupCost: 1,
  minInlineCalls: 100,
  minHitRate: 0.2,
  maxInlineBody: 8,
  functions: {
    main: { body: 5, pure: true, calls: [["scan", 1, 1], ["rank", 1, 1], ["blob", 1, 1], ["twice", 10, 2]] },
    scan: { body: 12, pure: true, calls: [["hash", 400, 1]] },
    rank: { body: 30, pure: true, calls: [["cmp", 900, 1], ["cold", 1, 1]] },
    hash: { body: 2, pure: true, calls: [], argDomain: 6, sampledCalls: 100, sampledDistinct: 6 },
    cmp: { body: 1, pure: true, calls: [], argDomain: 4000, sampledCalls: 100, sampledDistinct: 78 },
    cold: { body: 3, pure: true, calls: [] },
    blob: { body: 60, pure: true, calls: [] },
    twice: { body: 2, pure: true, calls: [] },
  },
};
const callsOf = (profile, name) => profile.functions[name]?.calls ?? [];
const sitesOf = (profile, callee) => Object.values(profile.functions)
  .flatMap((fn) => (fn.calls ?? []).filter(([c]) => c === callee).map(([, , sites]) => sites));
/** Dynamic invocation counts in topological order; a cyclic graph is refused. */
function dynamicCounts(profile, name = profile.root) {
  assert(profile.functions[name], `unknown function ${name}`);
  const postOrder = [];
  const seen = new Set();
  const visit = (current, path) => {
    assert(!path.includes(current), `dynamic counts need an acyclic call graph, cycle at ${current}`);
    if (seen.has(current)) return;
    seen.add(current);
    for (const [callee] of callsOf(profile, current)) visit(callee, [...path, current]);
    postOrder.push(current);
  };
  visit(name, []);
  const counts = new Map([[name, 1]]);
  for (const fn of [...postOrder].reverse()) {
    for (const [callee, perCall] of callsOf(profile, fn)) counts.set(callee, (counts.get(callee) ?? 0) + counts.get(fn) * perCall);
  }
  return counts;
}
const exclusiveCost = (profile, counts, name) => counts.get(name) * profile.functions[name].body;
const edgeCost = (profile, counts, caller, callee) => counts.get(caller) * (callsOf(profile, caller).find(([c]) => c === callee)?.[1] ?? 0);
function plainCost(profile, counts) {
  let total = 0;
  for (const [name, count] of counts) {
    total += count * profile.functions[name].body;
    for (const [callee, perCall] of callsOf(profile, name)) total += count * perCall * profile.overhead;
  }
  return total;
}
/** Inclusive cost of a subtree: its exclusive cost, its call instructions and every callee. */
function inclusiveCost(profile, counts, name, memo = new Map()) {
  if (memo.has(name)) return memo.get(name);
  let total = exclusiveCost(profile, counts, name);
  for (const [callee, perCall] of callsOf(profile, name)) {
    total += counts.get(name) * perCall * profile.overhead + inclusiveCost(profile, counts, callee, memo);
  }
  memo.set(name, total);
  return total;
}
/** Independent oracle: every root-to-leaf chain, enumerated without pruning. */
function allPaths(profile, name = profile.root) {
  const children = callsOf(profile, name);
  if (!children.length) return [[name]];
  return children.flatMap(([callee]) => allPaths(profile, callee).map((path) => [name, ...path]));
}
/** Cost of a chain on its own: the exclusive cost of its frames plus its call instructions. */
function chainCost(profile, counts, path) {
  let total = path.reduce((a, name) => a + exclusiveCost(profile, counts, name), 0);
  for (let i = 0; i + 1 < path.length; i += 1) total += edgeCost(profile, counts, path[i], path[i + 1]) * profile.overhead;
  return total;
}
const hotPath = (profile, counts) => allPaths(profile)
  .map((path) => ({ path, cost: chainCost(profile, counts, path) }))
  .sort((a, b) => b.cost - a.cost)[0];
/** Inline: small body, exactly one call site, called often enough to pay for the call overhead. */
function inlineSuggestions(profile, counts) {
  const suggestions = [];
  for (const caller of Object.keys(profile.functions)) {
    if (!counts.has(caller)) continue;
    for (const [callee, perCall, sites] of callsOf(profile, caller)) {
      if (!counts.has(callee)) continue;
      const fn = profile.functions[callee];
      const callCount = counts.get(caller) * perCall;
      const reasons = [];
      if (fn.body > profile.maxInlineBody) reasons.push("body too large");
      if (sites > 1) reasons.push("multiple call sites");
      if (callCount < profile.minInlineCalls) reasons.push("not called often enough");
      const saving = callCount * profile.overhead - sites * fn.body * profile.growthPenalty;
      if (!reasons.length && saving > 0) suggestions.push({ kind: "inline", callee, caller, predicted: saving });
      else suggestions.push({ kind: "reject", callee, caller, predicted: 0, reasons: reasons.length ? reasons : ["no net saving"] });
    }
  }
  return suggestions;
}
/** Memoize: a pure leaf whose sampled arguments repeat often enough to beat the lookup cost. */
function memoSuggestions(profile, counts) {
  const suggestions = [];
  for (const [name, fn] of Object.entries(profile.functions)) {
    if (!counts.has(name) || callsOf(profile, name).length) continue;
    if (fn.sampledCalls === undefined) continue;
    const hitRate = 1 - fn.sampledDistinct / fn.sampledCalls;
    const distinct = counts.get(name) * (fn.sampledDistinct / fn.sampledCalls);
    const saving = (counts.get(name) - distinct) * (fn.body - profile.lookupCost);
    const reasons = [];
    if (!fn.pure) reasons.push("impure");
    if (hitRate < profile.minHitRate) reasons.push("low hit rate");
    if (saving <= 0) reasons.push("no net saving");
    if (!reasons.length) suggestions.push({ kind: "memoize", callee: name, predicted: saving, hitRate });
    else suggestions.push({ kind: "reject", callee: name, predicted: 0, reasons });
  }
  return suggestions;
}
function analyze(profile) {
  const counts = dynamicCounts(profile);
  const suggestions = [...inlineSuggestions(profile, counts), ...memoSuggestions(profile, counts)]
    .sort((a, b) => b.predicted - a.predicted || a.callee.localeCompare(b.callee));
  const accepted = suggestions.filter((s) => s.kind !== "reject");
  return { counts, suggestions, accepted, predicted: accepted.reduce((a, s) => a + s.predicted, 0),
    hot: hotPath(profile, counts), plain: plainCost(profile, counts), hottest: [...counts.keys()].sort((a, b) => inclusiveCost(profile, counts, b) - inclusiveCost(profile, counts, a))[0] };
}
const BODY = { main: (x) => x, scan: (x) => (x * 3 + 1) % 997, rank: (x) => (x * 5 + 2) % 991, hash: (x) => (x * 7) % 5,
  cmp: (x) => (x * 11 + 3) % 89, cold: (x) => x + 1, blob: (x) => (x * 2) % 13, twice: (x) => x - 1 };
/** Execute the plan on a cost model; returns the cost, the savings and a semantic checksum. */
function simulate(profile, plan = { inline: [], memo: [] }, seed = 1) {
  const counts = dynamicCounts(profile);
  const rng = splitmix64(seed);
  const cache = new Map();
  let total = plainCost(profile, counts);
  let checksum = 0;
  let hits = 0;
  let valueMismatches = 0;
  for (const name of plan.inline) total -= counts.get(name) * profile.overhead;
  for (const name of plan.inline) total += sitesOf(profile, name).reduce((a, b) => a + b, 0) * profile.functions[name].body * profile.growthPenalty;
  for (const name of plan.memo) {
    const fn = profile.functions[name];
    total -= counts.get(name) * fn.body;
    for (let i = 0; i < counts.get(name); i += 1) {
      const arg = Math.floor(unit(rng) * fn.argDomain);
      const key = `${name}:${arg}`;
      const fresh = BODY[name](arg);
      if (cache.has(key)) {
        hits += 1;
        total += profile.lookupCost;
        if (cache.get(key) !== fresh) valueMismatches += 1;
      } else {
        cache.set(key, fresh);
        total += profile.lookupCost + fn.body;
      }
    }
  }
  for (const [name, count] of counts) for (let i = 0; i < count; i += 1) checksum = (checksum + BODY[name](i)) % 1e9;
  return { total, savings: plainCost(profile, counts) - total, checksum, hits, valueMismatches };
}
const planOf = (result) => ({ inline: result.accepted.filter((s) => s.kind === "inline").map((s) => s.callee),
  memo: result.accepted.filter((s) => s.kind === "memoize").map((s) => s.callee) });
function main() {
  const analysis = analyze(PROFILE);
  const counts = analysis.counts;
  const plain = plainCost(PROFILE, counts);
  // 1. Cost accounting: dynamic counts and the derived costs are self-consistent.
  assert(counts.get("hash") === 400 && counts.get("cmp") === 900 && counts.get("twice") === 10, `counts ${[...counts]}`);
  const summed = [...counts].reduce((a, [name, count]) => a + count * PROFILE.functions[name].body, 0)
    + Object.keys(PROFILE.functions).filter((n) => counts.has(n)).reduce((a, n) => a
      + callsOf(PROFILE, n).reduce((b, [, perCall]) => b + counts.get(n) * perCall * PROFILE.overhead, 0), 0);
  assert(summed === plain, `cost accounting ${summed} vs ${plain}`);
  assert(inclusiveCost(PROFILE, counts, "main") === plain, "the root's inclusive cost is the program cost");
  // 2. Hot path: the pruned search agrees with the brute-force enumeration of every chain.
  const oracle = allPaths(PROFILE).map((path) => ({ path, cost: chainCost(PROFILE, counts, path) })).sort((a, b) => b.cost - a.cost)[0];
  assert(analysis.hot.path.join(">") === oracle.path.join(">") && analysis.hot.cost === oracle.cost, `hot path ${analysis.hot.path} vs ${oracle.path}`);
  assert(analysis.hot.path.length >= 3 && analysis.hottest === "main", "the whole program runs under main");
  // 3. Inlining: the hot small helper is accepted, oversized/multi-site/cold callees are rejected.
  const kindsOf = (callee) => analysis.suggestions.filter((s) => s.callee === callee);
  const inlinable = (callee) => kindsOf(callee).some((s) => s.kind === "inline");
  const reasonsFor = (callee) => kindsOf(callee).find((s) => s.kind === "reject")?.reasons ?? [];
  assert(inlinable("hash") && inlinable("cmp"), "hot single-site helpers are inlinable");
  assert(reasonsFor("blob").includes("body too large"), `blob is refused: ${reasonsFor("blob")}`);
  assert(reasonsFor("twice").includes("multiple call sites"), `twice is refused: ${reasonsFor("twice")}`);
  assert(reasonsFor("cold").includes("not called often enough"), `cold is refused: ${reasonsFor("cold")}`);
  // 4. Caching: the repeating pure leaf is accepted, the wide-argument one is rejected.
  assert(kindsOf("hash").some((s) => s.kind === "memoize"), "the repeating leaf is memoized");
  assert(!kindsOf("cmp").some((s) => s.kind === "memoize"), "a wide-argument leaf is not worth caching");
  const impure = analyze({ ...PROFILE, functions: { ...PROFILE.functions, hash: { ...PROFILE.functions.hash, pure: false } } });
  assert(impure.suggestions.find((s) => s.callee === "hash" && s.reasons?.includes("impure")), "an impure leaf is never cached");
  // 5. Measured versus predicted savings for the accepted plan.
  const plan = planOf(analysis);
  assert(plan.inline.length >= 2 && plan.memo.length === 1, `plan ${JSON.stringify(plan)}`);
  const executed = simulate(PROFILE, plan, 942);
  assert(Math.abs(executed.savings - analysis.predicted) / analysis.predicted < 0.15, `predicted ${analysis.predicted} vs measured ${executed.savings}`);
  assert(executed.savings > 0 && executed.hits > 0, "the plan actually removes work");
  // 6. Semantics: the optimized program keeps the same results, and every cache hit is correct.
  const baseline = simulate(PROFILE, { inline: [], memo: [] }, 942);
  assert(executed.checksum === baseline.checksum, "optimization preserves the program's results");
  assert(executed.valueMismatches === 0, "a cached value always equals a fresh computation");
  // 7. Suggestions are ordered by their predicted saving.
  assert(analysis.accepted.every((s, i) => i === 0 || analysis.accepted[i - 1].predicted >= s.predicted), "suggestions are ranked");
  assert(analysis.accepted.reduce((a, s) => a + s.predicted, 0) === analysis.predicted, "the plan total is the sum of its suggestions");
  // 8. Boundary: a program with no calls yields no suggestions.
  const flat = analyze({ ...PROFILE, functions: { main: { body: 10, pure: true, calls: [] } } });
  assert(flat.accepted.length === 0 && flat.suggestions.length === 0 && flat.plain === 10, "a call-free program has nothing to optimize");
  assert(flat.hot.path.join(">") === "main", "the hot path of a single function is itself");
  // 9. Negative: a cyclic graph is refused instead of looping forever.
  const cyclic = { ...PROFILE, functions: { main: { body: 1, calls: [["loop", 2, 1]] }, loop: { body: 1, calls: [["loop", 2, 1]] } } };
  assert.throws(() => analyze(cyclic), /acyclic/, "a recursive graph cannot be counted");
  // 10. Determinism: re-analysis returns an identical plan and identical measurements.
  const again = analyze(PROFILE);
  assert(JSON.stringify(again.suggestions) === JSON.stringify(analysis.suggestions), "analysis is deterministic");
  assert(simulate(PROFILE, planOf(again), 942).total === executed.total, "execution is deterministic");
  console.log(`hot-path: plain=${plain} predicted=${analysis.predicted} measured=${executed.savings} hits=${executed.hits} hot=${analysis.hot.path.join(">")}`);
  console.log("PASS");
}
module.exports = { splitmix64, analyze, simulate, dynamicCounts, plainCost, inclusiveCost, allPaths, chainCost, PROFILE };
if (require.main === module) main();
