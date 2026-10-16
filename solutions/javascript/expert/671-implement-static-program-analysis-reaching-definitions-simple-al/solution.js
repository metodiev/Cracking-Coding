/**
 * Task 671 - Implement static program analysis: reaching definitions simple algorithm.
 * (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A tiny three-address IR plus the classic Kildall data-flow analysis. A program is a list of basic
 * blocks; a block is a list of assignments (var <- f(uses)) and a terminator (jump / branch / exit),
 * enough CFG structure for diamonds and loops. Every assignment gets the stable id `b<block>.<index>`.
 *
 * Transfer function: OUT[b] = gen[b] | (IN[b] - kill[b]), gen[b] being the block's stores that later
 * statements of the block do not overwrite and kill[b] every store (anywhere) of a variable the block
 * redefines minus gen[b]; IN[b] is the union of OUT over the predecessors. One driver computes the
 * fixpoint either by sweeping all blocks until a sweep changes nothing (snapshotting each sweep so
 * Kildall's ascending-chain monotonicity can be asserted) or by a worklist that only re-checks the
 * successors of a block whose sets grew. The independent reference is pathBasedReaching, which walks
 * forward from each definition marking the program points it reaches and stopping where a store to the
 * same variable kills it - the meets-over-all-paths definition, so it must agree with data flow
 * everywhere (points are read before a statement, so a killing store is still reported).
 *
 * Verified: a diamond (x := 3 and x := 4 reach the join, the entry store x := 1 is killed) and a loop
 * (the use of i is reached by i := 0 and by the loop-carried i := i + 1); gen and kill disjoint with
 * OUT = gen | (IN - kill) on every block of 15 programs; monotone, bounded, reproducible iteration;
 * worklist and sweep agree while the worklist visits fewer blocks; >800 definition/point pairs
 * cross-checked against path enumeration; exact def-use chains that flag uses nobody defines; an
 * unreachable store still flows (reachability is a separate pass, shown by reachableBlocks); and a
 * statement-free program has empty sets with an immediate fixpoint.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed671n;

/** Deterministic 64-bit splitmix64, used for the random CFGs. */
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
const def = (v, uses = []) => ({ var: v, uses: [...new Set(uses)] });
const block = (id, stmts, term) => ({ id, stmts, term });
const jump = (to) => ({ kind: "jump", to });
const branch = (v, then, other) => ({ kind: "branch", var: v, then, else: other });
const exit = () => ({ kind: "exit" });
function successorsOf(term) {
  if (term.kind === "jump") return [term.to];
  if (term.kind === "branch") return [term.then, term.else];
  return [];
}

/** Every variable a block reads: statement operands plus the branch condition. */
function usesOf(b) {
  const used = new Set(b.stmts.flatMap((s) => s.uses));
  if (b.term.kind === "branch") used.add(b.term.var);
  return used;
}
/** Static index: definitions, per-variable definition lists, predecessors. */
function indexProgram(program) {
  const byId = new Map(program.map((b) => [b.id, b]));
  const defs = [];
  const defsByVar = new Map();
  const varOfDef = new Map();
  for (const b of program) b.stmts.forEach((s, k) => {
    const id = `b${b.id}.${k}`;
    defs.push({ id, block: b.id, index: k, var: s.var, uses: s.uses });
    varOfDef.set(id, s.var);
    defsByVar.set(s.var, [...(defsByVar.get(s.var) ?? []), id]);
  });
  const preds = new Map(program.map((b) => [b.id, []]));
  for (const b of program) for (const s of successorsOf(b.term)) {
    if (!preds.has(s)) throw new Error(`undefined block ${s}`);
    preds.get(s).push(b.id);
  }
  return { program, byId, defs, defsByVar, varOfDef, preds };
}

/** gen/kill: the block generates its last store per variable, and kills every store it overwrites. */
function genKill(info) {
  const gen = new Map();
  const kill = new Map();
  for (const b of info.program) {
    const generates = new Set();
    const kills = new Set();
    b.stmts.forEach((s, k) => {
      for (const other of [...generates]) if (info.varOfDef.get(other) === s.var) generates.delete(other);
      generates.add(`b${b.id}.${k}`);
      for (const d of info.defsByVar.get(s.var) ?? []) kills.add(d);
    });
    for (const d of generates) kills.delete(d);
    gen.set(b.id, generates);
    kill.set(b.id, kills);
  }
  return { gen, kill };
}
const sameSet = (a, b) => a.size === b.size && [...a].every((v) => b.has(v));
/** Meet over predecessors followed by the transfer function. */
function transfer(b, info, gen, kill, outSets) {
  const incoming = new Set();
  for (const p of info.preds.get(b.id)) for (const d of outSets.get(p)) incoming.add(d);
  const outgoing = new Set(gen.get(b.id));
  for (const d of incoming) if (!kill.get(b.id).has(d)) outgoing.add(d);
  return { incoming, outgoing };
}
/**
 * Fixpoint driver: "round-robin" sweeps every block until a sweep changes nothing (snapshotting each
 * sweep for the monotonicity check), "worklist" re-enqueues only successors of blocks that grew.
 */
function reachingDefinitions(program, strategy = "round-robin") {
  const info = indexProgram(program);
  const { gen, kill } = genKill(info);
  const inSets = new Map(program.map((b) => [b.id, new Set()]));
  const outSets = new Map(program.map((b) => [b.id, new Set()]));
  const pending = strategy === "worklist" ? program.map((b) => b.id) : [];
  const queued = new Set(pending);
  const history = [];
  let visits = 0;
  let changed = true;
  while (strategy === "worklist" ? pending.length > 0 : changed) {
    changed = false;
    const snapshot = new Map();
    const batch = strategy === "worklist" ? [pending.shift()] : program.map((b) => b.id);
    queued.delete(batch[0]);
    for (const id of batch) {
      visits += 1;
      const b = info.byId.get(id);
      const { incoming, outgoing } = transfer(b, info, gen, kill, outSets);
      snapshot.set(id, { in: new Set(incoming), out: new Set(outgoing) });
      if (sameSet(incoming, inSets.get(id)) && sameSet(outgoing, outSets.get(id))) continue;
      inSets.set(id, incoming);
      outSets.set(id, outgoing);
      changed = true;
      for (const s of successorsOf(b.term)) if (!queued.has(s)) {
        pending.push(s);
        queued.add(s);
      }
    }
    history.push(snapshot);
  }
  return { info, gen, kill, inSets, outSets, iterations: history.length, visits, history };
}
function reachableBlocks(program) {  const info = indexProgram(program);
  const seen = new Set([0]);
  const stack = [0];
  while (stack.length > 0) {
    const next = successorsOf(info.byId.get(stack.pop()).term).filter((s) => !seen.has(s));
    for (const s of next) {
      seen.add(s);
      stack.push(s);
    }
  }
  return [...seen].sort((a, b) => a - b);
}
/** Successors of a program point: the next statement, or the successors' entry points. Blocks with no
 *  statements are walked through, with a guard so a cycle of empty blocks cannot loop forever. */
function pointSuccessors(info, blockId, index) {
  const b = info.byId.get(blockId);
  if (index + 1 < b.stmts.length) return [`b${blockId}.${index + 1}`];
  const out = [];
  const queue = successorsOf(b.term).map((id) => [id, new Set([id])]);
  while (queue.length > 0) {
    const [id, guard] = queue.shift();
    const next = info.byId.get(id);
    if (next.stmts.length > 0) {
      out.push(`b${id}.0`);
      continue;
    }
    for (const s of successorsOf(next.term)) if (!guard.has(s)) queue.push([s, new Set([...guard, s])]);
  }
  return out;
}

const pointOf = (key) => ({
  block: Number(key.slice(1, key.indexOf("."))),
  index: Number(key.slice(key.indexOf(".") + 1)),
});
/** Meets-over-all-paths reference: mark reached points, stop propagating at a killing store. */
function pathBasedReaching(program) {
  const info = indexProgram(program);
  const reaches = new Map();
  for (const d of info.defs) {
    const visited = new Set();
    const reached = new Set();
    const stack = pointSuccessors(info, d.block, d.index);
    while (stack.length > 0) {
      const key = stack.pop();
      if (visited.has(key)) continue;
      visited.add(key);
      reached.add(key); // the point before this statement is reached by d
      const p = pointOf(key);
      if (info.byId.get(p.block).stmts[p.index].var === d.var) continue; // killed, nothing flows past
      for (const next of pointSuccessors(info, p.block, p.index)) stack.push(next);
    }
    reaches.set(d.id, reached);
  }
  return { info, reaches };
}
/** Definitions valid at a program point: after the first k statements of the block. */
function reachingAt(result, blockId, k) {
  const b = result.info.byId.get(blockId);
  const set = new Set(result.inSets.get(blockId));
  for (let i = 0; i < k; i++) {
    for (const d of result.info.defsByVar.get(b.stmts[i].var) ?? []) set.delete(d);
    set.add(`b${blockId}.${i}`);
  }
  return set;
}

/** Per use: the definitions reaching it, plus the uses that no definition reaches. */
function defUseChains(result) {
  const chains = new Map();
  const undefinedUses = [];
  for (const b of result.info.program) b.stmts.forEach((s, k) => {
    const at = reachingAt(result, b.id, k);
    for (const v of s.uses) {
      const reaching = [...at].filter((d) => result.info.varOfDef.get(d) === v).sort();
      chains.set(`b${b.id}.${k}:${v}`, { defs: reaching, block: b.id, index: k });
      if (reaching.length === 0) undefinedUses.push(`b${b.id}.${k}:${v}`);
    }
  });
  return { chains, undefinedUses };
}
/** Diamond join: x := 3 and x := 4 reach z := x, the entry store x := 1 does not. */
const diamond = () => [
  block(0, [def("x", []), def("y", [])], branch("y", 1, 2)),
  block(1, [def("x", ["y"])], jump(3)),
  block(2, [def("x", [])], jump(3)),
  block(3, [def("z", ["x"])], exit()),
];
/** Loop: the use of i in b1 is reached by i := 0 and by the loop-carried i := i + 1. */
const loopProgram = () => [
  block(0, [def("i", [])], jump(1)),
  block(1, [def("i", ["i"])], branch("i", 1, 2)),
  block(2, [def("out", ["i"])], exit()),
];
/** A block nobody branches to, which still jumps into block 2. */
const unreachableBlock = () => [
  block(0, [def("a", [])], jump(2)),
  block(1, [def("ghost", ["a"])], jump(2)),
  block(2, [def("b", ["a"])], exit()),
];
function randomProgram(rng, blockCount, pool, maxStmts) {
  const pick = () => pool[rng.nextInt(pool.length)];
  const program = [];
  for (let i = 0; i < blockCount; i++) {
    const stmts = Array.from({ length: 1 + rng.nextInt(maxStmts) }, () => def(pick(), [pick(), pick()]));
    const roll = rng.nextInt(4);
    const twoMore = i + 2 < blockCount;
    const term = i === blockCount - 1 ? exit()
      : roll === 0 ? jump(i === 0 ? 1 : rng.nextInt(i + 1)) // back edge -> loop
        : roll === 1 && twoMore ? branch(pick(), i + 1, i + 2) : jump(i + 1);
    program.push(block(i, stmts, term));
  }
  return program;
}

function main() {
  // (1) Diamond and loop, hand computed.
  const dia = reachingDefinitions(diamond());
  assert.deepStrictEqual([...dia.inSets.get(3)].sort(), ["b0.1", "b1.0", "b2.0"], "IN[3] = {y:=2, x:=3, x:=4}");
  assert.deepStrictEqual([...reachingAt(dia, 3, 0)].sort(), ["b0.1", "b1.0", "b2.0"], "both branch stores reach the join");
  assert.deepStrictEqual([...dia.outSets.get(1)].sort(), ["b0.1", "b1.0"], "OUT[1] = {y:=2, x:=3}: x:=1 is killed");
  assert.strictEqual(dia.inSets.get(3).has("b0.0"), false, "the killed store x:=1 is gone at the join");
  assert.deepStrictEqual([...usesOf(diamond()[0])], ["y"], "the branch condition counts as a block use");
  const loop = reachingDefinitions(loopProgram());
  assert.deepStrictEqual([...loop.inSets.get(1)].sort(), ["b0.0", "b1.0"], "IN[1] = {i:=0, i:=i+1}");
  assert.ok(loop.inSets.get(1).has("b1.0"), "the loop-carried store reaches its own use: one pass is not enough");
  const chains = defUseChains(loop);
  assert.deepStrictEqual([chains.chains.get("b1.0:i").defs, chains.chains.get("b2.0:i").defs, chains.undefinedUses], [["b0.0", "b1.0"], ["b1.0"], []], "the use in b1 has two defs, the use after the loop one, none is undefined");
  // (2) gen/kill laws and the transfer function on every test program.
  const programs = [diamond(), loopProgram(), unreachableBlock()];
  const rng = new SplitMix64(DEFAULT_SEED);
  for (let i = 0; i < 12; i++) programs.push(randomProgram(rng, 4 + rng.nextInt(3), ["a", "b", "c"], 3));
  for (const program of programs) {
    const result = reachingDefinitions(program);
    for (const b of program) {
      const g = result.gen.get(b.id);
      const k = result.kill.get(b.id);
      assert.ok([...g].every((d) => !k.has(d)), "gen and kill are disjoint");
      const recomputed = new Set([...g, ...[...result.inSets.get(b.id)].filter((d) => !k.has(d))]);
      assert.ok(sameSet(recomputed, result.outSets.get(b.id)), "OUT = gen | (IN - kill) holds");
      assert.ok([...result.inSets.get(b.id)].every((d) => result.info.varOfDef.has(d)), "no unknown ids in IN");
    }
  }
  // (3) The sweep is monotone, bounded and reproducible.
  for (const program of programs) {
    const result = reachingDefinitions(program);
    for (let t = 1; t < result.history.length; t++) for (const b of program) {
      const before = result.history[t - 1].get(b.id);
      const after = result.history[t].get(b.id);
      assert.ok([...before.in].every((d) => after.in.has(d)) && [...before.out].every((d) => after.out.has(d)), `sweep ${t} adds only in block ${b.id}`);
    }
    assert.ok(result.iterations <= result.info.defs.length + 2, `fixpoint after ${result.iterations} sweeps`);
    const again = reachingDefinitions(program);
    assert.strictEqual(again.iterations, result.iterations, "the sweep count is reproducible");
    for (const b of program) assert.ok(sameSet(again.inSets.get(b.id), result.inSets.get(b.id)), "identical IN sets");
  }
  // (4) The worklist driver agrees with the sweep and visits fewer blocks overall.
  const sweepVisits = programs.reduce((acc, p) => acc + reachingDefinitions(p).visits, 0);
  const worklistVisits = programs.reduce((acc, p) => acc + reachingDefinitions(p, "worklist").visits, 0);
  for (const program of programs) {
    const sweep = reachingDefinitions(program);
    const worklist = reachingDefinitions(program, "worklist");
    for (const b of program) assert.ok(sameSet(sweep.inSets.get(b.id), worklist.inSets.get(b.id))
      && sameSet(sweep.outSets.get(b.id), worklist.outSets.get(b.id)), "worklist sets match the sweep");
  }
  assert.ok(worklistVisits < sweepVisits, `worklist visits ${worklistVisits} blocks, the sweep ${sweepVisits}`);

  // (5) Meets-over-all-paths agreement (program points are read before the statement).
  let compared = 0;
  for (const program of programs) {
    const result = reachingDefinitions(program);
    const paths = pathBasedReaching(program);
    for (const d of result.info.defs) for (const b of program) b.stmts.forEach((_, k) => {
      compared += 1;
      assert.strictEqual(paths.reaches.get(d.id).has(`b${b.id}.${k}`), reachingAt(result, b.id, k).has(d.id), `definition ${d.id} vs point before b${b.id}.${k}`);
    });
  }
  assert.ok(compared > 800, `cross-checked ${compared} definition/point pairs`);
  // (6) Reachability is a separate pass; undefined uses are flagged.
  const unreachable = reachingDefinitions(unreachableBlock());
  assert.deepStrictEqual([unreachable.inSets.get(2).has("b1.0"), unreachable.inSets.get(2).has("b0.0"), reachableBlocks(unreachableBlock())],
    [true, true, [0, 2]], "documented: the unreachable store still flows into b2, while reachability drops block 1");
  const undefinedChains = defUseChains(reachingDefinitions([block(0, [def("q", ["p"])], exit())]));
  assert.deepStrictEqual([undefinedChains.undefinedUses, undefinedChains.chains.get("b0.0:p").defs], [["b0.0:p"], []], "a use no definition reaches is flagged, with an empty chain");
  // (7) Boundary: statement-free blocks, and two stores of the same variable in one block.
  const emptyResult = reachingDefinitions([block(0, [], branch("c", 1, 1)), block(1, [], exit())]);
  assert.deepStrictEqual([emptyResult.info.defs.length, [...emptyResult.inSets.get(1)], emptyResult.iterations], [0, [], 1], "no definitions means empty IN sets and an immediate fixpoint");
  const straight = reachingDefinitions([block(0, [def("v", []), def("v", ["v"])], exit())]);
  assert.deepStrictEqual([[...straight.outSets.get(0)], straight.iterations], [["b0.1"], 2], "the second store kills the first in-line, two sweeps reach the fixpoint");
}

module.exports = { SplitMix64, DEFAULT_SEED, def, block, jump, branch, exit, successorsOf, usesOf,
  indexProgram, genKill, sameSet, transfer, reachingDefinitions, reachableBlocks, pointSuccessors,
  pointOf, pathBasedReaching, reachingAt, defUseChains, diamond, loopProgram, unreachableBlock,
  randomProgram };

if (require.main === module) {
  main();
  console.log("PASS");
}
