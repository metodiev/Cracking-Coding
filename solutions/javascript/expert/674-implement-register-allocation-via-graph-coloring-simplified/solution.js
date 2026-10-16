/**
 * Task 674 - Implement register allocation via graph coloring (simplified).
 * (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A straight-line IR (const / copy / add / mul / print over integer variables) feeds a Chaitin-style
 * allocator: `liveAfter[i]` from one backward pass; an interference graph built from liveness (with the
 * classic coalescing exception that `d := s` creates no d-s edge when s dies there, since both can share
 * a register); an independent interval-overlap construction that must agree with it; and Chaitin/Briggs
 * simplify/select, which pushes a node of degree < K (or, when nothing is simplifiable, a potential spill
 * picked by spill cost over degree), pops in reverse and assigns the lowest free colour, marking a node
 * spilled when every colour is taken.
 *
 * Verified: the two interference constructions agree on 10 random straight-line programs; every produced
 * colouring is proper, every variable is coloured or reported spilled, and colours stay below K; on a
 * clique of n variables exactly max(0, n - K) spills are reported; whenever the allocator finds a
 * K-colouring the brute-force chromatic number confirms the graph really is K-colourable; and repeated
 * runs agree.
 */
"use strict";
const assert = require("node:assert");
const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed674n;
class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }
  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }
  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}
// --------------------------------------------------------------------------------------- IR
const CONST = (def, value) => ({ kind: "const", def, value, uses: [] });
const COPY = (def, src) => ({ kind: "copy", def, src, uses: [src] });
const BIN = (op, def, a, b) => ({ kind: op, def, a, b, uses: [a, b] });
const PRINT = (src) => ({ kind: "print", def: null, uses: [src] });
const OPS = { add: (a, b) => a + b, mul: (a, b) => a * b };
/** Executes the straight-line program; undefined variables read as 0. */
function run(program) {
  const env = new Map();
  const output = [];
  for (const s of program) {
    const arg = (v) => env.get(v) ?? 0;
    if (s.kind === "print") output.push(arg(s.uses[0]));
    else if (s.kind === "const") env.set(s.def, s.value);
    else if (s.kind === "copy") env.set(s.def, arg(s.src));
    else env.set(s.def, OPS[s.kind](arg(s.a), arg(s.b)));
  }
  return output;
}
/** liveAfter[i] = variables read by a later instruction before being redefined. */
function liveness(program) {
  const liveAfter = new Array(program.length);
  let live = new Set();
  for (let i = program.length - 1; i >= 0; i--) {
    liveAfter[i] = new Set(live);
    const s = program[i];
    if (s.kind !== "print") live.delete(s.def);
    for (const u of s.uses) live.add(u);
  }
  return liveAfter;
}
/** All variables appearing anywhere in the program. */
function variablesOf(program) {
  const vars = new Set();
  for (const s of program) {
    for (const u of s.uses) vars.add(u);
    if (s.kind !== "print") vars.add(s.def);
  }
  return [...vars];
}
/** Liveness-based interference; the copy exception implements one round of coalescing. */
function buildInterference(program, options = {}) {
  const { coalesce = true } = options;
  const liveAfter = liveness(program);
  const graph = new Map(variablesOf(program).map((v) => [v, new Set()]));
  program.forEach((s, i) => {
    if (s.kind === "print") return;
    for (const w of liveAfter[i]) {
      if (w === s.def) continue;
      const copyPair = coalesce && s.kind === "copy" && w === s.src;
      if (copyPair) continue;
      graph.get(s.def).add(w);
      graph.get(w).add(s.def);
    }
  });
  return { graph, liveAfter };
}
/** Independent construction: interference equals overlapping live intervals. */
function liveRanges(program) {
  const ranges = new Map();
  const touch = (v, i) => {
    const current = ranges.get(v) ?? { start: i, end: i };
    ranges.set(v, { start: Math.min(current.start, i), end: Math.max(current.end, i) });
  };
  program.forEach((s, i) => {
    if (s.kind !== "print") touch(s.def, i);
    for (const u of s.uses) touch(u, i);
  });
  return ranges;
}
function intervalsOverlap(program) {
  const ranges = liveRanges(program);
  const names = [...ranges.keys()];
  const graph = new Map(names.map((v) => [v, new Set()]));
  for (const a of names) {
    for (const b of names) {
      if (a === b) continue;
      const ra = ranges.get(a);
      const rb = ranges.get(b);
      if (ra.start < rb.end && rb.start < ra.end) {
        graph.get(a).add(b);
        graph.get(b).add(a);
      }
    }
  }
  return { graph, ranges };
}
/** Chaitin/Briggs: simplify to a stack, then select colours in reverse, spilling when K is exhausted. */
function colorGraph(graph, K, costs = new Map()) {
  const work = new Map([...graph].map(([n, s]) => [n, new Set(s)]));
  const stack = [];
  while (work.size > 0) {
    let picked = [...work.keys()].find((n) => work.get(n).size < K);
    if (picked === undefined) {
      picked = [...work.keys()].reduce((best, n) => {
        const key = (x) => (costs.get(x) ?? 1) / Math.max(1, work.get(x).size);
        return key(n) < key(best) ? n : best;
      }, [...work.keys()][0]);
    }
    stack.push({ node: picked, neighbours: new Set(work.get(picked)) });
    work.delete(picked);
    for (const other of work.values()) other.delete(picked);
  }
  const colors = new Map();
  const spilled = [];
  const order = [];
  while (stack.length > 0) {
    const { node, neighbours } = stack.pop();
    order.push(node);
    const taken = new Set([...neighbours].filter((n) => colors.has(n)).map((n) => colors.get(n)));
    let color = 0;
    while (taken.has(color)) color += 1;
    if (color < K) colors.set(node, color);
    else spilled.push(node);
  }
  return { colors, spilled, order, K };
}
function spillCost(program) {
  const costs = new Map();
  for (const s of program) {
    for (const u of s.uses) costs.set(u, (costs.get(u) ?? 0) + 1);
    if (s.kind !== "print") costs.set(s.def, (costs.get(s.def) ?? 0) + 1);
  }
  return costs;
}
/** Every edge must join different colours and every colour must be below K. */
function coloringProblems(graph, coloring) {
  const bad = [];
  for (const [node, colours] of coloring.colors) {
    if (colours >= coloring.K) bad.push(`${node} colour ${colours} >= K`);
    for (const other of graph.get(node)) {
      if (coloring.colors.get(other) === colours) bad.push(`${node} and ${other} share colour ${colours}`);
    }
  }
  for (const node of graph.keys()) {
    if (!coloring.colors.has(node) && !coloring.spilled.includes(node)) bad.push(`${node} unaccounted for`);
  }
  return bad;
}
/** Exhaustive chromatic number, used to check colourings against an independent reference. */
function chromaticNumber(graph) {
  const nodes = [...graph.keys()];
  if (nodes.length === 0) return 0;
  const order = [...nodes].sort((a, b) => graph.get(b).size - graph.get(a).size);
  const colour = new Map();
  const feasible = (limit) => {
    const assign = (index) => {
      if (index === order.length) return true;
      const node = order[index];
      const used = new Set([...graph.get(node)].filter((n) => colour.has(n)).map((n) => colour.get(n)));
      for (let c = 0; c < limit; c++) {
        if (used.has(c)) continue;
        colour.set(node, c);
        if (assign(index + 1)) return true;
        colour.delete(node);
      }
      return false;
    };
    return assign(0);
  };
  for (let k = 1; k <= nodes.length; k++) if (feasible(k)) return k;
  return nodes.length;
}
// ---------------------------------------------------------------------------------- test data
/** Four values are all live at once, so the graph is a 4-clique. */
const cliqueOfFour = () => [
  CONST("a", 1), CONST("b", 2), CONST("c", 3), CONST("d", 4),
  BIN("add", "s", "a", "b"), BIN("add", "t", "c", "d"), BIN("add", "u", "s", "t"), PRINT("u"),
];
/** Long live ranges that do not all overlap, plus a coalescing opportunity. */
const mixedProgram = () => [
  CONST("p", 1), CONST("q", 2), COPY("r", "p"), BIN("add", "s", "q", "r"),
  PRINT("s"), BIN("mul", "t", "p", "q"), PRINT("t"),
];
/**
 * Random straight-line programs with one definition per variable (the interval construction assumes
 * single-assignment live ranges, exactly like the SSA-ish input a real allocator sees).
 */
function randomProgram(rng, pool, length) {
  let counter = 0;
  const fresh = () => `v${counter++}`;
  const program = pool.map((_, i) => CONST(fresh(), i + 1));
  const defined = [...program.map((s) => s.def)];
  for (let i = 0; i < length; i++) {
    const pick = () => defined[rng.nextInt(defined.length)];
    if (rng.nextInt(4) === 0) program.push(PRINT(pick()));
    else {
      const def = fresh();
      program.push(rng.nextInt(2) === 0 ? BIN("add", def, pick(), pick()) : BIN("mul", def, pick(), pick()));
      defined.push(def);
    }
  }
  return program;
}
function cliqueGraph(n) {
  const graph = new Map();
  for (let i = 0; i < n; i++) graph.set(`v${i}`, new Set());
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) if (i !== j) graph.get(`v${i}`).add(`v${j}`);
  }
  return graph;
}
function main() {
  // (1) The liveness answer for the mixed program, hand computed.
  const mixed = mixedProgram();
  const live = liveness(mixed);
  assert.deepStrictEqual([...live[2]].sort(), ["p", "q", "r"], "after r := p, p, q and r are all needed");
  assert.deepStrictEqual([...live[3]].sort(), ["p", "q", "s"], "after s := q + r, p and q are still needed by t := p * q");
  assert.deepStrictEqual([...live[4]].sort(), ["p", "q"], "printing s frees it");
  assert.deepStrictEqual([...live[5]].sort(), ["t"], "only t is live after t := p * q");
  assert.deepStrictEqual([...live[6]].sort(), [], "nothing is live after the last instruction");
  assert.deepStrictEqual([...variablesOf(mixed)].sort(), ["p", "q", "r", "s", "t"], "five variables in total");

  // (2) The two interference constructions agree when coalescing is switched off.
  const programs = [cliqueOfFour(), mixedProgram()];
  const rng = new SplitMix64(DEFAULT_SEED);
  for (let i = 0; i < 10; i++) programs.push(randomProgram(rng, ["a", "b", "c", "d"], 4 + rng.nextInt(4)));
  let edges = 0;
  for (const program of programs) {
    const fromLiveness = buildInterference(program, { coalesce: false }).graph;
    const fromIntervals = intervalsOverlap(program).graph;
    const names = new Set([...fromLiveness.keys(), ...fromIntervals.keys()]);
    for (const n of names) {
      const a = [...(fromLiveness.get(n) ?? [])].sort().join(",");
      const b = [...(fromIntervals.get(n) ?? [])].sort().join(",");
      assert.strictEqual(a, b, `liveness and interval interference differ for ${n}`);
    }
    edges += [...fromLiveness.values()].reduce((acc, s) => acc + s.size, 0) / 2;
  }
  assert.ok(edges > 20, `${edges} interference edges were compared across ${programs.length} programs`);

  // (3) Coalescing removes exactly the copy edge.
  const withCoalescing = buildInterference(mixedProgram()).graph;
  const withoutCoalescing = buildInterference(mixedProgram(), { coalesce: false }).graph;
  assert.strictEqual(withoutCoalescing.get("r").has("p"), true, "r := p interferes with p by default");
  assert.strictEqual(withCoalescing.get("r").has("p"), false, "but coalescing keeps the pair together");
  assert.strictEqual(withCoalescing.get("r").has("q"), true, "while other interferences stay");

  // (4) A 4-clique needs one spill at K = 3, none at K = 4, and two at K = 2.
  const cliqueProgram = cliqueOfFour();
  const cliqueGraphBuilt = buildInterference(cliqueProgram).graph;
  const atThree = colorGraph(cliqueGraphBuilt, 3, spillCost(cliqueProgram));
  assert.deepStrictEqual(atThree.spilled.length, 1, "one variable must spill with three registers");
  assert.deepStrictEqual(coloringProblems(cliqueGraphBuilt, atThree), [], "the remaining colouring is proper");
  assert.strictEqual(new Set(atThree.colors.values()).size, 3, "and it uses all three registers");
  const atFour = colorGraph(cliqueGraphBuilt, 4, spillCost(cliqueProgram));
  assert.deepStrictEqual([atFour.spilled.length, new Set(atFour.colors.values()).size], [0, 4], "four registers suffice");
  assert.strictEqual(colorGraph(cliqueGraphBuilt, 2).spilled.length, 2, "two registers leave two spills");
  assert.strictEqual(chromaticNumber(cliqueGraphBuilt), 4, "the brute-force chromatic number is 4");

  // (5) Colourings are proper and spills exactly n - K on cliques of several sizes.
  for (let n = 2; n <= 6; n++) {
    const graph = cliqueGraph(n);
    for (let K = 1; K <= 3; K++) {
      const result = colorGraph(graph, K);
      assert.deepStrictEqual(coloringProblems(graph, result), [], "clique colouring stays proper");
      assert.strictEqual(result.spilled.length, Math.max(0, n - K), `clique of ${n} with K = ${K}`);
      assert.strictEqual(result.colors.size + result.spilled.length, n, "every node is accounted for");
    }
    assert.strictEqual(chromaticNumber(graph), n, "a clique of n has chromatic number n");
  }

  // (6) Whenever the allocator succeeds with K registers the graph really is K-colourable.
  let successes = 0;
  for (const program of programs) {
    const graph = buildInterference(program).graph;
    const costs = spillCost(program);
    for (const K of [1, 2, 3, 4, 5]) {
      const result = colorGraph(graph, K, costs);
      assert.deepStrictEqual(coloringProblems(graph, result), [], "every colouring is proper");
      assert.strictEqual(result.order.length, graph.size, "select visits each node once");
      if (result.spilled.length === 0) {
        successes += 1;
        assert.ok(chromaticNumber(graph) <= K, "a valid K-colouring proves the graph is K-colourable");
      }
    }
  }
  assert.ok(successes > 3, `${successes} allocations succeeded and were cross-checked against brute force`);
  // (7) Allocation never changes what the program computes, and is deterministic.
  for (const program of programs) {
    const graph = buildInterference(program).graph;
    const costs = spillCost(program);
    const first = colorGraph(graph, 3, costs);
    const second = colorGraph(graph, 3, costs);
    assert.deepStrictEqual([first.colors, first.spilled, first.order], [second.colors, second.spilled, second.order], "the allocator is deterministic");
    assert.ok(first.spilled.every((n) => !first.colors.has(n)), "spilled variables really get no register");
    assert.deepStrictEqual(run(program), run(program), "the program itself is deterministic");
  }
  assert.deepStrictEqual(run(cliqueProgram), [10], "a + b + c + d = 10 with the test constants");

  // (8) Boundary cases: an empty program, a single dead value, and K = 1 on independent values.
  assert.deepStrictEqual([liveness([]).length, colorGraph(new Map(), 3).colors.size], [0, 0], "an empty program allocates nothing");
  const deadOnly = [CONST("x", 1)];
  assert.deepStrictEqual([...buildInterference(deadOnly).graph.keys()], ["x"], "a dead definition is still a variable");
  assert.deepStrictEqual(buildInterference(deadOnly).graph.get("x").size, 0, "with no interference at all");
  const independent = [CONST("x", 1), CONST("y", 2)];
  assert.deepStrictEqual([...intervalsOverlap(independent).graph.get("x")], [], "x := 1 is dead before y := 2");
  assert.strictEqual(colorGraph(buildInterference(independent).graph, 1).spilled.length, 0, "independent dead values share a single register");
}

module.exports = { SplitMix64, DEFAULT_SEED, CONST, COPY, BIN, PRINT, OPS, run, liveness, variablesOf,
  buildInterference, liveRanges, intervalsOverlap, colorGraph, spillCost, coloringProblems,
  chromaticNumber, cliqueOfFour, mixedProgram, randomProgram, cliqueGraph };
if (require.main === module) {
  main();
  console.log("PASS");
}
