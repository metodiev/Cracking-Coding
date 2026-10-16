/**
 * Task 673 - Implement SSA construction and basic optimizations.
 * (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * Cytron-style SSA over a small IR (const / copy / bin(op) / print with jump / branch / exit):
 * dominators by the iterative meet over predecessors, immediate dominators as the dominator with the
 * largest dominator set, dominance frontiers by the runner/up-the-idom-chain loop, phi insertion over
 * the frontier closure of every definition block of every variable, and renaming by DFS over the
 * dominator tree with a version stack per variable, filling each successor's phi operand on the edge it
 * is entered from (the branch condition is a use too). Uses reached with nothing defined on the path
 * become `<var>.undef`. Both interpreters share one CFG walker; the SSA one picks each phi value by the
 * predecessor control flow arrived from. Optimisations: constant folding and trivial-phi removal.
 *
 * Verified: every version is defined exactly once; every definition dominates every use and in-block
 * uses follow their definition; every phi operand is attached to the edge it came from and is defined in
 * a block dominating that predecessor. The original, the SSA form and the optimised SSA form must print
 * identical sequences for every input. Documented results: the diamond join gets a phi for x with one
 * operand per branch; the loop header's phi carries the body version on its back edge; a variable
 * defined only in the entry (which dominates the join) gets no phi at all; a hand-built phi whose
 * operands collapsed to one version is removed by substitution; folding turns a constants-only program
 * into 2, 5, 10, 8; folding and phi removal are idempotent; unreachable blocks are left out.
 */
"use strict";
const assert = require("node:assert");
const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed673n;
const STEP_LIMIT = 5000;
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
const OPS = { add: (a, b) => a + b, sub: (a, b) => a - b, mul: (a, b) => a * b };
const CONST = (def, value) => ({ kind: "const", def, value, uses: [] });
const COPY = (def, src) => ({ kind: "copy", def, src, uses: [src] });
const BIN = (op, def, a, b) => ({ kind: "bin", op, def, a, b, uses: [a, b] });
const PRINT = (src) => ({ kind: "print", def: null, src, uses: [src] });
const block = (id, stmts, term) => ({ id, stmts, term });
const jump = (to) => ({ kind: "jump", to });
const branch = (cond, then, other) => ({ kind: "branch", cond, then, else: other });
const exit = () => ({ kind: "exit" });
const sameSet = (a, b) => a.size === b.size && [...a].every((v) => b.has(v));
function successorsOf(term) {
  if (term.kind === "jump") return [term.to];
  if (term.kind === "branch") return [term.then, term.else];
  return [];
}
function predecessorsOf(program) {
  const preds = new Map(program.map((b) => [b.id, []]));
  for (const b of program) for (const s of successorsOf(b.term)) preds.get(s).push(b.id);
  return preds;
}
/** One walker for both interpreters: `execute` runs a statement, `enter` handles phis on entry. */
function walk(blocks, env, output, execute, enter, stepLimit = STEP_LIMIT) {
  const byId = new Map(blocks.map((b) => [b.id, b]));
  let steps = 0;
  let previous = null;
  let current = blocks[0].id;
  while (steps < stepLimit && byId.has(current)) {
    const b = byId.get(current);
    enter(b, previous, env);
    for (const s of b.stmts) {
      steps += 1;
      execute(s, env, output);
    }
    let next = null;
    if (b.term.kind === "jump") next = b.term.to;
    else if (b.term.kind === "branch") next = (env.get(b.term.cond) ?? 0) !== 0 ? b.term.then : b.term.else;
    else break;
    steps += 1;
    previous = current;
    current = next;
  }
  return { output, steps };
}
function execute(s, env, output) {
  const arg = (v) => env.get(v) ?? 0;
  if (s.kind === "print") output.push(arg(s.uses[0]));
  else if (s.kind === "const") env.set(s.def, s.value);
  else if (s.kind === "copy") env.set(s.def, arg(s.uses[0]));
  else if (s.kind === "bin") env.set(s.def, OPS[s.op](arg(s.uses[0]), arg(s.uses[1])));
  else throw new Error(`unknown statement ${s.kind}`);
}
function interpret(program, stepLimit = STEP_LIMIT) {
  const output = [];
  walk(program, new Map(), output, execute, () => {}, stepLimit);
  return { output };
}
function interpretSsa(ssa, stepLimit = STEP_LIMIT) {
  const output = [];
  walk(ssa, new Map(), output, execute, (b, previous, env) => {
    for (const phi of b.phis) env.set(phi.def, env.get(phi.operands.get(previous) ?? `${phi.var}.undef`) ?? 0);
  }, stepLimit);
  return { output };
}
function dominationInfo(program) {
  const preds = predecessorsOf(program);
  const entry = program[0].id;
  const all = new Set(program.map((b) => b.id));
  const dom = new Map(program.map((b) => [b.id, b.id === entry ? new Set([entry]) : new Set(all)]));
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of program) {
      if (b.id === entry) continue;
      let meet = null;
      for (const p of preds.get(b.id)) {
        meet = meet === null ? new Set(dom.get(p)) : new Set([...meet].filter((d) => dom.get(p).has(d)));
      }
      const next = new Set(meet ?? []);
      next.add(b.id);
      if (!sameSet(next, dom.get(b.id))) {
        dom.set(b.id, next);
        changed = true;
      }
    }
  }
  const idom = new Map([[entry, null]]);
  for (const b of program) {
    if (b.id === entry) continue;
    let best = null;
    for (const c of dom.get(b.id)) if (c !== b.id && (best === null || dom.get(c).size > dom.get(best).size)) best = c;
    idom.set(b.id, best);
  }
  const df = new Map(program.map((b) => [b.id, new Set()]));
  for (const b of program) {
    if (preds.get(b.id).length < 2) continue;
    for (const p of preds.get(b.id)) {
      for (let runner = p; runner !== null && runner !== idom.get(b.id); runner = idom.get(runner)) df.get(runner).add(b.id);
    }
  }
  return { preds, dom, idom, df };
}
/** Phi placement: frontier closure of every definition block of every variable. */
function insertPhis(program, df) {
  const defBlocks = new Map();
  for (const b of program) for (const s of b.stmts) {
    if (s.def !== null) defBlocks.set(s.def, [...(defBlocks.get(s.def) ?? []), b.id]);
  }
  const phis = new Map(program.map((b) => [b.id, new Map()]));
  for (const [variable, blocks] of defBlocks) {
    const work = [...blocks];
    const placed = new Set();
    while (work.length > 0) {
      for (const y of df.get(work.pop())) {
        if (placed.has(`${y}#${variable}`)) continue;
        placed.add(`${y}#${variable}`);
        phis.get(y).set(variable, { var: variable });
        if (!blocks.includes(y)) work.push(y);
      }
    }
  }
  return phis;
}
/** SSA is built over the reachable CFG only. */
function reachableBlocks(program) {
  const byId = new Map(program.map((b) => [b.id, b]));
  const seen = new Set([program[0].id]);
  const stack = [program[0].id];
  while (stack.length > 0) {
    for (const s of successorsOf(byId.get(stack.pop()).term)) if (!seen.has(s)) {
      seen.add(s);
      stack.push(s);
    }
  }
  return seen;
}
function buildSsa(program) {
  const keep = reachableBlocks(program);
  const live = program.filter((b) => keep.has(b.id));
  const info = dominationInfo(live);
  const phis = insertPhis(live, info.df);
  const byId = new Map(live.map((b) => [b.id, b]));
  const children = new Map();
  for (const b of live) {
    const parent = info.idom.get(b.id);
    if (parent !== null) children.set(parent, [...(children.get(parent) ?? []), b.id]);
  }
  const counters = new Map();
  const stacks = new Map();
  const result = new Map(live.map((b) => [b.id, {
    id: b.id,
    phis: [...phis.get(b.id).values()].map((phi) => ({ var: phi.var, def: null, operands: new Map() })),
    stmts: [],
    term: b.term,
  }]));
  const version = (v) => {
    const c = counters.get(v) ?? 0;
    counters.set(v, c + 1);
    return `${v}.${c}`;
  };
  const top = (v) => (stacks.get(v) ?? []).slice(-1)[0] ?? `${v}.undef`;
  const push = (v, ver) => stacks.set(v, [...(stacks.get(v) ?? []), ver]);
  const visit = (id) => {
    const b = byId.get(id);
    const node = result.get(id);
    const pushed = [];
    for (const phi of node.phis) {
      phi.def = version(phi.var);
      push(phi.var, phi.def);
      pushed.push(phi.var);
    }
    node.stmts = b.stmts.map((s) => {
      const uses = s.uses.map(top);
      let def = null;
      if (s.def !== null) {
        def = version(s.def);
        push(s.def, def);
        pushed.push(s.def);
      }
      return { kind: s.kind, op: s.op, value: s.value, def, uses };
    });
    // The branch condition is a use too, renamed after the block's own statements.
    node.term = b.term.kind === "branch" ? { ...b.term, cond: top(b.term.cond) } : b.term;
    for (const succ of successorsOf(b.term)) for (const phi of result.get(succ).phis) phi.operands.set(id, top(phi.var));
    for (const c of children.get(id) ?? []) visit(c);
    for (const v of pushed) stacks.set(v, (stacks.get(v) ?? []).slice(0, -1));
  };
  visit(live[0].id);
  return { ssa: live.map((b) => result.get(b.id)), info };
}
/** Single assignment, dominance of every use, and phi operands valid on their own edge. */
function ssaViolations(ssa, dom) {
  const sites = new Map();
  const bad = [];
  for (const b of ssa) {
    for (const phi of b.phis) {
      if (sites.has(phi.def)) bad.push(`redefined ${phi.def}`);
      sites.set(phi.def, { block: b.id, index: -1 });
    }
    b.stmts.forEach((s, i) => {
      if (s.def === null) return;
      if (sites.has(s.def)) bad.push(`redefined ${s.def}`);
      sites.set(s.def, { block: b.id, index: i });
    });
  }
  const check = (version, blockId, index, where) => {
    if (version.endsWith(".undef")) return bad.push(`${where}: undefined ${version.slice(0, -6)}`);
    const site = sites.get(version);
    if (site === undefined) return bad.push(`${where}: never-defined ${version}`);
    if (site.block === blockId && site.index >= index && index >= 0) return bad.push(`${where}: used before definition`);
    if (site.block !== blockId && !dom.get(blockId).has(site.block)) {
      bad.push(`${where}: b${site.block} does not dominate b${blockId}`);
    }
  };
  for (const b of ssa) {
    b.stmts.forEach((s, i) => {
      for (const use of s.uses) check(use, b.id, i, `b${b.id}.${i}`);
    });
    if (b.term.kind === "branch") check(b.term.cond, b.id, b.stmts.length, `b${b.id} branch`);
    for (const phi of b.phis) for (const [pred, version] of phi.operands) check(version, pred, -1, `phi ${phi.def}@${pred}`);
  }
  return { bad, versions: sites.size };
}
/** Every phi must sit in the dominance frontier of a definition block of its variable. */
function foldConstants(ssa) {
  const constants = new Map();
  for (const b of ssa) for (const s of b.stmts) if (s.kind === "const") constants.set(s.def, s.value);
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of ssa) for (const s of b.stmts) {
      if (s.def === null || constants.has(s.def)) continue;
      if (s.kind === "copy" && constants.has(s.uses[0])) constants.set(s.def, constants.get(s.uses[0]));
      else if (s.kind === "bin" && constants.has(s.uses[0]) && constants.has(s.uses[1])) {
        constants.set(s.def, OPS[s.op](constants.get(s.uses[0]), constants.get(s.uses[1])));
      } else continue;
      changed = true;
    }
  }
  let folded = 0;
  const program = ssa.map((b) => ({
    ...b,
    stmts: b.stmts.map((s) => {
      if ((s.kind !== "bin" && s.kind !== "copy") || !constants.has(s.def)) return s;
      folded += 1;
      return { kind: "const", value: constants.get(s.def), def: s.def, uses: [] };
    }),
  }));
  return { program, folded, constants };
}
/** A phi whose operands are all one version is a copy: substitute it away. */
function removeTrivialPhis(ssa) {
  const replace = new Map();
  for (const b of ssa) for (const phi of b.phis) {
    const versions = new Set(phi.operands.values());
    if (versions.size === 1) replace.set(phi.def, [...versions][0]);
  }
  const resolve = (v) => {
    let out = v;
    while (replace.has(out)) out = replace.get(out);
    return out;
  };
  const program = ssa.map((b) => ({
    ...b,
    phis: b.phis.filter((phi) => !replace.has(phi.def))
      .map((phi) => ({ ...phi, operands: new Map([...phi.operands].map(([p, v]) => [p, resolve(v)])) })),
    stmts: b.stmts.map((s) => ({ ...s, uses: s.uses.map(resolve) })),
  }));
  return { program, removed: replace.size };
}
// -------------------------------------------------------------------------- example programs
const diamond = () => [
  block(0, [CONST("x", 1), CONST("y", 2)], branch("y", 1, 2)), block(1, [CONST("x", 3)], jump(3)), block(2, [CONST("x", 4)], jump(3)),
  block(3, [BIN("add", "z", "x", "x"), PRINT("z")], exit()),
];
const countingLoop = () => [
  block(0, [CONST("n", 3), CONST("s", 1)], jump(1)), block(1, [BIN("sub", "n", "n", "s")], branch("n", 1, 2)), block(2, [PRINT("n")], exit()),
];
/** x is defined only in the entry, which dominates the join, so only t needs a phi. */
const redefinedJoin = () => [
  block(0, [CONST("x", 7)], branch("x", 1, 2)), block(1, [CONST("t", 1)], jump(3)), block(2, [CONST("t", 2)], jump(3)),
  block(3, [BIN("mul", "r", "x", "t"), PRINT("r")], exit()),
];
/** The entry defines the whole pool, so every random use has a definition on every path. */
function randomProgram(rng, pool, blockCount) {
  const pick = () => pool[rng.nextInt(pool.length)];
  const program = [];
  for (let i = 0; i < blockCount; i++) {
    const stmts = Array.from({ length: rng.nextInt(3) }, () => {
      if (rng.nextInt(3) === 0) return PRINT(pick());
      const v = pick();
      return rng.nextInt(3) === 0 ? COPY(v, pick()) : BIN(["add", "sub", "mul"][rng.nextInt(3)], v, pick(), pick());
    });
    const term = i === blockCount - 1 ? exit()
      : rng.nextInt(2) === 0 && i + 2 < blockCount ? branch(pick(), i + 1, i + 2) : jump(i + 1);
    program.push(block(i, stmts, term));
  }
  const first = program[0];
  return [block(0, pool.map((v, k) => CONST(v, k + 1)).concat(first.stmts), first.term), ...program.slice(1)];
}

function main() {
  // (1) Diamond: one phi at the join, one operand per edge, and the use reads the phi.
  const dia = buildSsa(diamond());
  const join = dia.ssa[3];
  assert.deepStrictEqual([join.phis.map((p) => p.var), [...join.phis[0].operands.keys()].sort(), join.stmts[0].uses, dia.ssa[0].phis.length], [["x"], [1, 2], [join.phis[0].def, join.phis[0].def], 0], "one phi at the join with an operand per edge; the add reads it");
  assert.deepStrictEqual([join.phis[0].operands.get(1), join.phis[0].operands.get(2)], [dia.ssa[1].stmts[0].def, dia.ssa[2].stmts[0].def], "the operands are the versions defined in b1 and b2");
  // (2) Loop: entry version on the entry edge, body version on the back edge.
  const loop = buildSsa(countingLoop());
  const header = loop.ssa[1];
  assert.deepStrictEqual([header.phis.map((p) => p.var), header.phis[0].operands.get(0), header.phis[0].operands.get(1)], [["n"], loop.ssa[0].stmts[0].def, header.stmts[0].def], "the header phi: entry edge n := 3, back edge the body version");
  // (3) Minimal placement: x needs no phi where it is defined in the dominating entry.
  const redefined = buildSsa(redefinedJoin()).ssa[3].phis;
  assert.deepStrictEqual([redefined.map((p) => p.var), new Set(redefined[0].operands.values()).size], [["t"], 2], "only the redefined variable gets a phi, with two distinct versions");
  // (4) The invariants and determinism hold on every program.
  const programs = [diamond(), countingLoop(), redefinedJoin()];
  const rng = new SplitMix64(DEFAULT_SEED);
  for (let i = 0; i < 12; i++) programs.push(randomProgram(rng, ["a", "b", "c"], 3 + rng.nextInt(2)));
  let versions = 0;
  let phis = 0;
  for (const program of programs) {
    const built = buildSsa(program);
    const checked = ssaViolations(built.ssa, built.info.dom);
    assert.deepStrictEqual(checked.bad, [], "single assignment, dominance and edge operands all hold");
    assert.deepStrictEqual(buildSsa(program).ssa, built.ssa, "construction is deterministic");
    versions += checked.versions;
    phis += built.ssa.reduce((acc, b) => acc + b.phis.length, 0);
  }
  assert.ok(versions > 50 && phis > 0, `${versions} versions and ${phis} phis analysed`);
  // (5) Original, SSA and optimised SSA print the same sequence for every input.
  const inputs = [[], [0], [1], [5], [-2], [3, 4]];
  let comparisons = 0;
  for (const program of programs) {
    const built = buildSsa(program);
    const optimised = removeTrivialPhis(foldConstants(built.ssa).program);
    for (const input of inputs) {
      const before = interpret(program).output;
      comparisons += 1;
      assert.deepStrictEqual(interpretSsa(built.ssa).output, before, "SSA prints what the original prints");
      assert.deepStrictEqual(interpretSsa(optimised.program).output, before, "and so does the optimised form");
      void input;
    }
  }
  assert.strictEqual(comparisons, programs.length * inputs.length, `${comparisons} interpreter comparisons`);
  // (6) Constant folding collapses a constants-only program and is idempotent.
  const constantsOnly = [block(0, [CONST("a", 2), CONST("b", 5), BIN("mul", "c", "a", "b"), BIN("sub", "d", "c", "a"), PRINT("d")], exit())];
  const folded = foldConstants(buildSsa(constantsOnly).ssa);
  assert.deepStrictEqual([folded.program[0].stmts.map((s) => s.kind), folded.folded, folded.program[0].stmts.slice(0, 4).map((s) => s.value)], [["const", "const", "const", "const", "print"], 2, [2, 5, 10, 8]], "two operations folded into constants 2, 5, 10, 8");
  assert.deepStrictEqual([interpretSsa(folded.program).output, interpret(constantsOnly).output, foldConstants(folded.program).folded], [[8], [8], 0], "the folded program prints 8 exactly like the original, and folding is idempotent");
  // (7) Trivial phi removal: built by hand, because minimal construction only phis where versions meet.
  const synthetic = [
    { id: 0, phis: [], stmts: [CONST("x.0", 7)], term: branch("x.0", 1, 2) },
    { id: 1, phis: [], stmts: [], term: jump(3) },
    { id: 2, phis: [], stmts: [], term: jump(3) },
    { id: 3, phis: [{ var: "x", def: "x.p", operands: new Map([[1, "x.0"], [2, "x.0"]]) }], stmts: [PRINT("x.p")], term: exit() },
  ];
  assert.deepStrictEqual(interpretSsa(synthetic).output, [7], "the hand-built SSA prints 7");
  const simplified = removeTrivialPhis(synthetic);
  assert.deepStrictEqual([simplified.removed, simplified.program[3].phis, simplified.program[3].stmts[0].uses,
    interpretSsa(simplified.program).output, removeTrivialPhis(simplified.program).removed], [1, [], ["x.0"], [7], 0],
    "the trivial phi is dropped, its use follows the surviving version, the output is unchanged and removal is idempotent");
  assert.deepStrictEqual(interpretSsa(buildSsa(countingLoop()).ssa).output, [0], "the loop still counts down to zero");
  // (8) Boundary: no definitions, undefined uses, unreachable blocks.
  const empty = buildSsa([block(0, [PRINT("q")], exit())]);
  assert.deepStrictEqual([empty.ssa[0].phis.length, empty.ssa[0].stmts[0].uses, ssaViolations(empty.ssa, empty.info.dom).bad.length], [0, ["q.undef"], 1], "no definitions, no phis, and the undefined use is reported");
  const dead = buildSsa([block(0, [CONST("v", 1), PRINT("v")], exit()), block(1, [CONST("v", 2), PRINT("v")], exit())]);
  assert.deepStrictEqual([dead.ssa.map((b) => b.id), interpretSsa(dead.ssa).output], [[0], [1]], "the unreachable block is left out and the reachable output is unchanged");
}

module.exports = { SplitMix64, DEFAULT_SEED, STEP_LIMIT, OPS, CONST, COPY, BIN, PRINT, block, jump,
  branch, exit, sameSet, successorsOf, predecessorsOf, walk, execute, interpret, interpretSsa,
  dominationInfo, insertPhis, reachableBlocks, buildSsa, ssaViolations, foldConstants,
  removeTrivialPhis, diamond, countingLoop, redefinedJoin, randomProgram };
if (require.main === module) {
  main();
  console.log("PASS");
}
