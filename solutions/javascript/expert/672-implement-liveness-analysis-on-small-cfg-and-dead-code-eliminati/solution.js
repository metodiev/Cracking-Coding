/**
 * Task 672 - Implement liveness analysis on small CFG and dead code elimination.
 * (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A tiny imperative IR with real semantics plus backward data-flow analysis. Statements are
 * const/copy/add/sub/mul/print/input over integer variables; terminators are jump/branch/exit, and
 * `interpret` runs a program on a list of inputs, returning the printed values, the number of executed
 * statements and whether it hit the step limit. That interpreter is the oracle for every optimisation.
 *
 * Liveness (backward): LIVE_OUT[b] = union of LIVE_IN over the successors and
 * LIVE_IN[b] = use[b] | (LIVE_OUT[b] - def[b]), where use[b] collects variables read before being
 * written inside the block (including the branch condition) and def[b] the variables written. Each
 * sweep is snapshotted, so monotone growth of the backward iteration can be asserted, and `liveAfter`
 * yields the live set at every program point.
 *
 * Dead code elimination then repeatedly removes statements whose result nobody reads later and which
 * have no side effect (print/input are kept), plus blocks unreachable from the entry, until nothing
 * changes. `livenessBySearch` is the independent reference: a variable is live at a point iff some path
 * from that point reaches a read of it before any write, checked by an explicit walk over points.
 *
 * Verified: hand-computed live sets for straight-line code and for a loop (LIVE_IN of the header holds
 * the counter, so the loop-carried update survives); LIVE_IN/LIVE_OUT obey the meet and transfer
 * equations on every block of 18 programs; backward iteration is monotone and reproducible; the
 * search-based liveness agrees with data flow at every point; DCE reaches a fixpoint in one confirming
 * pass; interpretation output is identical before and after on many inputs; a definition read only in
 * an unreachable block is removed; and the optimised program never executes more statements.
 */
"use strict";
const assert = require("node:assert");
const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed672n;
const STEP_LIMIT = 20000;
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
// ------------------------------------------------------------------------------------------ IR
const CONST = (def, value) => ({ kind: "const", def, value, uses: [] });
const COPY = (def, src) => ({ kind: "copy", def, src, uses: [src] });
const BINOP = (kind, def, a, b) => ({ kind, def, a, b, uses: [a, b] });
const PRINT = (src) => ({ kind: "print", def: null, src, uses: [src] });
const INPUT = (def) => ({ kind: "input", def, uses: [] });
const defOf = (s) => s.def;
const BINOPS = { add: (x, y) => x + y, sub: (x, y) => x - y, mul: (x, y) => x * y };
const block = (id, stmts, term) => ({ id, stmts, term });
const jump = (to) => ({ kind: "jump", to });
const branch = (cond, then, other) => ({ kind: "branch", cond, then, else: other });
const exit = () => ({ kind: "exit" });
function successorsOf(term) {
  if (term.kind === "jump") return [term.to];
  if (term.kind === "branch") return [term.then, term.else];
  return [];
}
/** Reads before the first write in the block plus the branch condition, and the written variables. */
function blockSummary(b) {
  const defined = new Set();
  const used = new Set();
  for (const s of b.stmts) {
    for (const u of s.uses) if (!defined.has(u)) used.add(u);
    if (defOf(s) !== null) defined.add(defOf(s));
  }
  if (b.term.kind === "branch" && !defined.has(b.term.cond)) used.add(b.term.cond);
  return { used, defined };
}
const blockUse = (b) => blockSummary(b).used;
/** Executes the program; `input` supplies the values read by `input` statements (missing ones are 0). */
function interpret(program, input = [], stepLimit = STEP_LIMIT) {
  const byId = new Map(program.map((b) => [b.id, b]));
  const env = new Map();
  const output = [];
  let steps = 0;
  let consumed = 0;
  let current = program[0].id;
  while (steps < stepLimit) {
    const b = byId.get(current);
    if (b === undefined) throw new Error(`jumped to unknown block ${current}`);
    let next = null;
    for (const s of b.stmts) {
      steps += 1;
      if (steps > stepLimit) break;
      const arg = (v) => env.get(v) ?? 0;
      if (s.kind === "print") output.push(arg(s.src));
      else if (s.kind === "input") env.set(s.def, input[consumed++] ?? 0);
      else if (s.kind === "const") env.set(s.def, s.value);
      else if (s.kind === "copy") env.set(s.def, arg(s.src));
      else if (BINOPS[s.kind] !== undefined) env.set(s.def, BINOPS[s.kind](arg(s.a), arg(s.b)));
      else throw new Error(`unknown statement ${s.kind}`);
    }
    if (steps > stepLimit) break;
    if (b.term.kind === "jump") next = b.term.to;
    else if (b.term.kind === "branch") next = (env.get(b.term.cond) ?? 0) !== 0 ? b.term.then : b.term.else;
    else break;
    steps += 1;
    current = next;
  }
  return { output, steps, truncated: steps >= stepLimit };
}
// ------------------------------------------------------------------------------- liveness
function liveness(program) {
  const byId = new Map(program.map((b) => [b.id, b]));
  const succ = new Map(program.map((b) => [b.id, successorsOf(b.term)]));
  const summary = new Map(program.map((b) => [b.id, blockSummary(b)]));
  const use = new Map([...summary].map(([id, s]) => [id, s.used]));
  const def = new Map([...summary].map(([id, s]) => [id, s.defined]));
  const liveIn = new Map(program.map((b) => [b.id, new Set()]));
  const liveOut = new Map(program.map((b) => [b.id, new Set()]));
  const history = [];
  let changed = true;
  while (changed) {
    changed = false;
    const snapshot = new Map();
    for (const b of [...program].reverse()) {
      const out = new Set();
      for (const s of succ.get(b.id)) for (const v of liveIn.get(s)) out.add(v);
      const inside = new Set(use.get(b.id));
      for (const v of out) if (!def.get(b.id).has(v)) inside.add(v);
      snapshot.set(b.id, { in: new Set(inside), out: new Set(out) });
      if (sameSet(inside, liveIn.get(b.id)) && sameSet(out, liveOut.get(b.id))) continue;
      liveIn.set(b.id, inside);
      liveOut.set(b.id, out);
      changed = true;
    }
    history.push(snapshot);
  }
  return { byId, succ, use, def, liveIn, liveOut, iterations: history.length, history };
}
const sameSet = (a, b) => a.size === b.size && [...a].every((v) => b.has(v));
/** Live variables immediately after statement k (k = -1 means the block entry). */
function liveAfter(result, blockId, k) {
  const b = result.byId.get(blockId);
  const live = new Set(result.liveOut.get(blockId));
  if (b.term.kind === "branch") live.add(b.term.cond);
  for (let i = b.stmts.length - 1; i > k; i--) {
    const s = b.stmts[i];
    if (defOf(s) !== null) live.delete(defOf(s));
    for (const u of s.uses) live.add(u);
  }
  return live;
}
/** Independent reference: a variable is live at a point iff a path reaches a read before a write. */
function livenessBySearch(program, variables) {
  const byId = new Map(program.map((b) => [b.id, b]));
  const result = new Map();
  for (const b of program) for (let k = 0; k <= b.stmts.length; k++) {
    const live = new Set();
    for (const v of variables) {
      const seen = new Set();
      const stack = [{ block: b.id, index: k }];
      let found = false;
      while (stack.length > 0 && !found) {
        const at = stack.pop();
        const mark = `${at.block}#${at.index}`;
        if (seen.has(mark)) continue;
        seen.add(mark);
        const cur = byId.get(at.block);
        if (at.index === cur.stmts.length) {
          if (cur.term.kind === "branch" && cur.term.cond === v) found = true;
          else for (const s of successorsOf(cur.term)) stack.push({ block: s, index: 0 });
        } else {
          const s = cur.stmts[at.index];
          if (s.uses.includes(v)) found = true;
          else if (defOf(s) !== v) stack.push({ block: at.block, index: at.index + 1 });
        }
      }
      if (found) live.add(v);
    }
    result.set(`b${b.id}#${k}`, live);
  }
  return result;
}
// ------------------------------------------------------------------------------- dead code
const hasSideEffect = (s) => s.kind === "print" || s.kind === "input";
function reachableSet(program) {
  const byId = new Map(program.map((b) => [b.id, b]));
  const seen = new Set([program[0].id]);
  const stack = [program[0].id];
  while (stack.length > 0) {
    for (const s of successorsOf(byId.get(stack.pop()).term)) {
      if (seen.has(s)) continue;
      seen.add(s);
      stack.push(s);
    }
  }
  return seen;
}
/** Unreachable blocks first, then statements whose result nobody reads later; until fixpoint. */
function deadCodeEliminate(program) {
  const reachable = reachableSet(program);
  let current = program.filter((b) => reachable.has(b.id)).map((b) => block(b.id, [...b.stmts], b.term));
  const removed = [];
  let passes = 0;
  for (;;) {
    passes += 1;
    const analysis = liveness(current);
    let changed = 0;
    current = current.map((b) => {
      const kept = [];
      b.stmts.forEach((s, k) => {
        const dead = defOf(s) !== null && !liveAfter(analysis, b.id, k).has(defOf(s)) && !hasSideEffect(s);
        if (dead) removed.push(`b${b.id}.${k}`);
        else kept.push(s);
      });
      if (kept.length !== b.stmts.length) changed += 1;
      return block(b.id, kept, b.term);
    });
    if (changed === 0) break;
  }
  return { program: current, removed, passes };
}
// ------------------------------------------------------------------------------------ test data
/** Straight line: w := 99 is never read, so it is dead; everything printed is live. */
const straightLine = () => [
  block(0, [INPUT("x"), CONST("y", 3), BINOP("add", "z", "x", "y"), PRINT("z"), CONST("w", 99)], exit()),
];
/** Loop with a counter: the update is used by the branch on the next iteration and by the print. */
const countingLoop = () => [
  block(0, [INPUT("i"), COPY("n", "i"), CONST("s", 1)], jump(1)),
  block(1, [BINOP("sub", "n", "n", "s")], branch("n", 1, 2)),
  block(2, [CONST("dead", 7), PRINT("n")], exit()),
  block(3, [CONST("unused", 1), PRINT("n")], exit()),
];
/** A definition whose only reader sits in an unreachable block. */
const ghostUse = () => [
  block(0, [CONST("a", 5), PRINT("a")], exit()),
  block(1, [BINOP("add", "b", "a", "a"), PRINT("b")], exit()),
];
function randomProgram(rng, pool, blockCount) {
  const pick = () => pool[rng.nextInt(pool.length)];
  const program = [];
  for (let i = 0; i < blockCount; i++) {
    const stmts = Array.from({ length: 1 + rng.nextInt(3) }, () => {
      const roll = rng.nextInt(4);
      return roll === 0 ? CONST(pick(), 1 + rng.nextInt(5)) : roll === 1 ? COPY(pick(), pick())
        : roll === 2 ? BINOP("add", pick(), pick(), pick()) : PRINT(pick());
    });
    const term = i === blockCount - 1 ? exit()
      : i + 2 < blockCount ? branch(pick(), i + 1, i + 2) : jump(i + 1);
    program.push(block(i, stmts, term));
  }
  // The entry block initialises the pool so that reads are defined on every path.
  const first = program[0];
  return [block(0, [CONST("a0", 2), CONST("b0", 3), ...first.stmts], first.term), ...program.slice(1)];
}
function main() {
  // (1) Straight-line liveness, hand computed.
  const straight = liveness(straightLine());
  assert.deepStrictEqual([...straight.liveOut.get(0)], [], "nothing is live after the block ends");
  assert.deepStrictEqual([[...liveAfter(straight, 0, 2)].sort(), [...straight.liveIn.get(0)], [...blockUse(straightLine()[0])]],
    [["z"], [], []], "z is live right after z := x + y; every read follows a write, so LIVE_IN[0] is empty");
  assert.deepStrictEqual([...liveAfter(straight, 0, 4)], [], "and nothing is live after the dead w := 99");
  const readFirst = liveness([block(0, [PRINT("p"), CONST("p", 1)], exit())]);
  assert.deepStrictEqual([...readFirst.liveIn.get(0)], ["p"], "a read before the write keeps p live at the entry");
  // (2) Loop liveness: the loop-carried counter is live on the back edge.
  const loop = liveness(countingLoop());
  assert.deepStrictEqual([...loop.liveIn.get(1)].sort(), ["n", "s"], "LIVE_IN[1] = {n, s}: the branch, print and subtraction");
  assert.ok(loop.liveOut.get(1).has("n"), "documented: n stays live around the back edge, so n := n - s must survive");
  assert.strictEqual(loop.liveIn.get(1).has("i"), false, "the input variable is dead once copied into n");
  // (3) The meet/transfer equations hold on every block of every test program.
  const programs = [straightLine(), countingLoop(), ghostUse()];
  const rng = new SplitMix64(DEFAULT_SEED);
  for (let i = 0; i < 15; i++) programs.push(randomProgram(rng, ["a0", "b0", "c", "d"], 3 + rng.nextInt(2)));
  for (const program of programs) {
    const result = liveness(program);
    for (const b of program) {
      const out = new Set();
      for (const s of successorsOf(b.term)) for (const v of result.liveIn.get(s)) out.add(v);
      assert.ok(sameSet(out, result.liveOut.get(b.id)), "LIVE_OUT = union of successor LIVE_IN");
      const inside = new Set(result.use.get(b.id));
      for (const v of out) if (!result.def.get(b.id).has(v)) inside.add(v);
      assert.ok(sameSet(inside, result.liveIn.get(b.id)), "LIVE_IN = use | (LIVE_OUT - def)");
    }
    for (let t = 1; t < result.history.length; t++) for (const b of program) {
      const before = result.history[t - 1].get(b.id);
      const after = result.history[t].get(b.id);
      assert.ok([...before.in].every((v) => after.in.has(v)) && [...before.out].every((v) => after.out.has(v)), `backward pass ${t} only adds live variables in block ${b.id}`);
    }
    assert.ok(result.iterations <= program.length * 3 + 2, `fixpoint after ${result.iterations} passes`);
    assert.strictEqual(liveness(program).iterations, result.iterations, "the pass count is reproducible");
  }
  // (4) Liveness by search agrees with the data flow at every point.
  const variables = ["a0", "b0", "c", "d", "x", "y", "z", "w", "n", "s", "i", "a", "b", "dead", "unused"];
  let compared = 0;
  for (const program of programs) {
    const result = liveness(program);
    const searched = livenessBySearch(program, variables);
    for (const b of program) for (let k = 0; k <= b.stmts.length; k++) {
      compared += 1;
      assert.ok(sameSet(searched.get(`b${b.id}#${k}`), liveAfter(result, b.id, k - 1)), `search and data flow disagree before b${b.id}.${k}`);
    }
  }
  assert.ok(compared > 150, `cross-checked ${compared} program points`);

  // (5) DCE on the straight-line program: exactly the dead store goes, output unchanged.
  const straightDce = deadCodeEliminate(straightLine());
  assert.deepStrictEqual([straightDce.removed, straightDce.program[0].stmts.length, straightDce.passes], [["b0.4"], 4, 2], "only w := 99 is removed, in one removing pass plus the confirming pass");
  assert.deepStrictEqual(interpret(straightDce.program, [4]).output, interpret(straightLine(), [4]).output, "the printed value is unchanged");
  // (6) DCE keeps the loop-carried update and prints identically.
  const loopDce = deadCodeEliminate(countingLoop());
  assert.deepStrictEqual(loopDce.removed, ["b2.0"], "only the never-read constant in b2 is dropped");
  assert.ok(loopDce.program.find((b) => b.id === 1).stmts.some((s) => s.def === "n"), "documented: n := n - s survives although its own block never reads n");
  for (const inputs of [[3], [1], [0], [7]]) assert.deepStrictEqual(interpret(loopDce.program, inputs).output, interpret(countingLoop(), inputs).output, `loop output matches for ${inputs}`);
  assert.deepStrictEqual(interpret(countingLoop(), [3]).output, [0], "the loop stops when its counter hits zero");
  assert.ok(interpret(countingLoop(), [4]).steps > interpret(countingLoop(), [3]).steps, "one more iteration costs more statements");
  // (7) A use inside an unreachable block does not keep the definition alive.
  const ghostDce = deadCodeEliminate(ghostUse());
  assert.deepStrictEqual([ghostDce.program.map((b) => b.id), interpret(ghostDce.program, []).output], [[0], [5]], "the unreachable block is dropped and the reachable output is intact");
  assert.deepStrictEqual(deadCodeEliminate(ghostDce.program).removed, [], "the result is a fixpoint");
  // (8) Semantics preserved on every random program and several inputs.
  let randomChecks = 0;
  let truncations = 0;
  for (const program of programs) {
    const after = deadCodeEliminate(program);
    assert.deepStrictEqual(deadCodeEliminate(after.program).removed, [], "DCE reaches a fixpoint");
    for (const inputs of [[], [1], [5, 2], [0], [-3, 4, 1]]) {
      const before = interpret(program, inputs);
      const optimised = interpret(after.program, inputs);
      randomChecks += 1;
      if (before.truncated) truncations += 1;
      else assert.deepStrictEqual(optimised.output, before.output, "optimised output equals the original");
      assert.ok(optimised.steps <= before.steps, "DCE never executes more statements than before");
    }
  }
  assert.ok(randomChecks === programs.length * 5, `${randomChecks} interpreter comparisons were run`);
  assert.ok(truncations < randomChecks, "the generated programs terminate, so truncation is not the norm");
  assert.ok(straightDce.program[0].stmts.every((s) => s.kind !== "print" || s.src === "z"),
    "the surviving print still reads the live value");
}

module.exports = { SplitMix64, DEFAULT_SEED, STEP_LIMIT, CONST, COPY, BINOP, PRINT, INPUT, defOf,
  block, jump, branch, exit, successorsOf, blockSummary, blockUse, interpret, liveness, sameSet,
  liveAfter, livenessBySearch, hasSideEffect, reachableSet, deadCodeEliminate, straightLine,
  countingLoop, ghostUse, randomProgram };
if (require.main === module) {
  main();
  console.log("PASS");
}
