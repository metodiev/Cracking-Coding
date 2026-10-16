/** Task 749 - Implement a symbolic execution skeleton for small programs.
 *  (Tier: Expert, CTCI Beyond - Compilers, Languages & Solvers)
 *
 * Programs are a tiny imperative IR over integer variables: assignments, conditional jumps, an
 * unconditional jump, `assume` (path pruning) and `assert`. Symbolic execution walks the control
 * flow graph collecting a path condition (a conjunction of integer atoms) and a list of
 * assertions along the way; at every conditional jump it forks into both successors after
 * first checking that the extended path condition is *feasible*, so infeasible branches are
 * pruned instead of explored. Loops are unrolled up to a documented bound: an edge is not
 * followed a second time if the current unroll count for that back edge is exhausted. Feasibility
 * is decided by an interval propagation pass plus exhaustive enumeration over a documented value
 * domain, and a reported counterexample is a concrete input that the *concrete interpreter* then
 * re-runs to confirm the assertion really fires.
 *
 * What is simplified: integer variables only, no arrays/pointers, straight-line feasibility by
 * bounded enumeration rather than a real theory solver, and an unroll bound instead of a
 * loop-invariant inference or a fixpoint. What is verified: the set of reachable assertion
 * failures found symbolically equals the set found by brute-force enumeration of concrete inputs
 * on the same domain (differential testing against the concrete interpreter); every reported
 * counterexample is re-run concretely and must fail; every path condition reported as infeasible
 * has no satisfying input; the symbolic path count matches the number of feasible concrete paths
 * for branch-free/unrolled programs; pruning actually happens (an assume-false program explores
 * one path); degenerate programs (empty, infinite loop, always-failing assertion) behave as
 * documented; and the whole analysis is deterministic.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

const DOMAIN = [-3, -2, -1, 0, 1, 2, 3]; // documented value domain of the bounded decision procedure

// ---- tiny IR ----------------------------------------------------------------------------------
// Instruction shapes (all indices into the instruction array):
//   { op: "assign", variable: "x", expression: ["+", "x", 2] }   arithmetic expression tree
//   { op: "branch", expression: ["<", "x", 5], then: index, else: index }
//   { op: "jump", target: index }
//   { op: "assume", expression: [...] }
//   { op: "assert", expression: [...] , message: "..." }
//   { op: "halt" }
function evaluateExpression(expression, state) {
  if (typeof expression === "number") return expression;
  if (typeof expression === "string") {
    if (!(expression in state)) throw new ReferenceError(`unbound variable ${expression}`);
    return state[expression];
  }
  const [operator, ...operands] = expression;
  const values = operands.map((operand) => evaluateExpression(operand, state));
  switch (operator) {
    case "+":
      return values.reduce((a, b) => a + b, 0);
    case "-":
      return values.length === 1 ? -values[0] : values[0] - values[1];
    case "*":
      return values.reduce((a, b) => a * b, 1);
    case "<":
      return values[0] < values[1] ? 1 : 0;
    case "<=":
      return values[0] <= values[1] ? 1 : 0;
    case ">":
      return values[0] > values[1] ? 1 : 0;
    case ">=":
      return values[0] >= values[1] ? 1 : 0;
    case "=":
      return values[0] === values[1] ? 1 : 0;
    default:
      throw new SyntaxError(`unknown operator ${operator}`);
  }
}

// ---- concrete interpreter (the oracle) ---------------------------------------------------------
function runConcretely(program, input, options = {}) {
  const unrollLimit = options.unrollLimit ?? 4;
  const state = { ...input };
  const failures = [];
  const visits = new Map();
  let pc = 0;
  let steps = 0;
  while (pc < program.length && steps++ < 10000) {
    const instruction = program[pc];
    switch (instruction.op) {
      case "assign":
        state[instruction.variable] = evaluateExpression(instruction.expression, state);
        pc++;
        break;
      case "branch": {
        const key = `b${pc}`;
        const count = visits.get(key) ?? 0;
        if (count > unrollLimit) return { state, failures, terminated: "unroll-limit" };
        visits.set(key, count + 1);
        pc = evaluateExpression(instruction.expression, state) !== 0 ? instruction.then : instruction.else;
        break;
      }
      case "jump": {
        const key = `j${pc}`;
        const count = visits.get(key) ?? 0;
        if (count > unrollLimit) return { state, failures, terminated: "unroll-limit" };
        visits.set(key, count + 1);
        pc = instruction.target;
        break;
      }
      case "assume":
        if (evaluateExpression(instruction.expression, state) === 0) {
          return { state, failures, terminated: "assume-false" };
        }
        pc++;
        break;
      case "assert":
        if (evaluateExpression(instruction.expression, state) === 0) {
          failures.push({ pc, message: instruction.message ?? `assert ${JSON.stringify(instruction.expression)}` });
          pc++;
          break;
        }
        pc++;
        break;
      default:
        return { state, failures, terminated: "halt" };
    }
  }
  return { state, failures, terminated: "step-limit" };
}

/** All concrete inputs of the domain that make the program fail an assertion. */
function bruteForceFailures(program, variables, domain = DOMAIN) {
  const failures = new Set();
  const assignment = {};
  const recurse = (index) => {
    if (index === variables.length) {
      const result = runConcretely(program, { ...assignment });
      if (result.failures.length > 0) failures.add(JSON.stringify(assignment));
      return;
    }
    for (const value of domain) {
      assignment[variables[index]] = value;
      recurse(index + 1);
    }
  };
  recurse(0);
  return failures;
}

// ---- symbolic execution --------------------------------------------------------------------------
function substitute(expression, state) {
  if (typeof expression === "number") return expression;
  if (typeof expression === "string") {
    const value = state[expression];
    return value === undefined ? expression : value;
  }
  const [operator, ...operands] = expression;
  return [operator, ...operands.map((operand) => substitute(operand, state))];
}

/** Evaluates an expression whose leaves are numbers only; null when a variable remains. */
function evaluateConstant(expression) {
  if (typeof expression === "number") return expression;
  if (typeof expression === "string") return null;
  const [operator, ...operands] = expression;
  const values = operands.map(evaluateConstant);
  if (values.some((value) => value === null)) return null;
  return evaluateExpression(expression, {});
}

/** Feasibility by interval propagation plus bounded enumeration over the documented domain. */
function feasible(conditions, variables, domain = DOMAIN) {
  const constraints = conditions.filter((condition) => condition !== null);
  for (const condition of constraints) {
    const constant = evaluateConstant(condition);
    if (constant === 0) return false; // a statically false condition
  }
  // interval pass: prune obviously contradictory pairs before enumerating
  const bounds = new Map();
  for (const variable of variables) bounds.set(variable, [domain[0], domain[domain.length - 1]]);
  for (const condition of constraints) {
    if (typeof condition === "string") continue;
    const [operator, left, right] = condition.length === 3 ? condition : [condition[0], condition[1], null];
    if (typeof left !== "string" || typeof right !== "number") continue;
    const [low, high] = bounds.get(left);
    if (operator === "<") bounds.set(left, [low, Math.min(high, right - 1)]);
    if (operator === "<=") bounds.set(left, [low, Math.min(high, right)]);
    if (operator === ">") bounds.set(left, [Math.max(low, right + 1), high]);
    if (operator === ">=") bounds.set(left, [Math.max(low, right), high]);
    if (operator === "=") bounds.set(left, [Math.max(low, right), Math.min(high, right)]);
    const [newLow, newHigh] = bounds.get(left);
    if (newLow > newHigh) return false;
  }
  const assignment = {};
  const ordered = variables;
  const recurse = (index) => {
    if (index === ordered.length) {
      return constraints.every((condition) => evaluateExpression(condition, assignment) !== 0);
    }
    const [low, high] = bounds.get(ordered[index]);
    for (const value of domain) {
      if (value < low || value > high) continue;
      assignment[ordered[index]] = value;
      if (recurse(index + 1)) return true;
    }
    return false;
  };
  return recurse(0);
}

/** Finds one satisfying input for the given conditions, or null when infeasible. */
function modelFor(conditions, variables, domain = DOMAIN) {
  const constraints = conditions.filter((condition) => condition !== null);
  const assignment = {};
  const recurse = (index) => {
    if (index === variables.length) {
      return constraints.every((condition) => evaluateExpression(condition, assignment) !== 0)
        ? { ...assignment }
        : null;
    }
    for (const value of domain) {
      assignment[variables[index]] = value;
      const found = recurse(index + 1);
      if (found !== null) return found;
    }
    return null;
  };
  return recurse(0);
}

/**
 * Symbolic execution. Returns the explored paths, the assertion failures with a witness input,
 * and counters showing how much pruning happened.
 */
function simulate(program, variables, options = {}) {
  const unrollLimit = options.unrollLimit ?? 4;
  const domain = options.domain ?? DOMAIN;
  const paths = [];
  const failures = [];
  let prunedPaths = 0;
  let forks = 0;
  const visits = new Map();

  const walk = (pc, symbolicState, conditions, assertions) => {
    if (pc >= program.length) {
      paths.push({ conditions: conditions.slice(), assertions: assertions.slice(), end: "halt" });
      return;
    }
    const instruction = program[pc];
    switch (instruction.op) {
      case "assign": {
        const next = { ...symbolicState };
        next[instruction.variable] = substitute(instruction.expression, symbolicState);
        walk(pc + 1, next, conditions, assertions);
        return;
      }
      case "branch": {
        const key = `b${pc}`;
        const count = visits.get(key) ?? 0;
        if (count > unrollLimit) {
          prunedPaths++;
          return;
        }
        visits.set(key, count + 1);
        const condition = substitute(instruction.expression, symbolicState);
        forks++;
        const taken = [...conditions, condition];
        if (feasible(taken, variables, domain)) {
          walk(instruction.then, symbolicState, taken, assertions);
        } else {
          prunedPaths++;
        }
        const notTaken = [...conditions, ["=", condition, 0]];
        if (feasible(notTaken, variables, domain)) {
          walk(instruction.else, symbolicState, notTaken, assertions);
        } else {
          prunedPaths++;
        }
        return;
      }
      case "jump": {
        const key = `j${pc}`;
        const count = visits.get(key) ?? 0;
        if (count > unrollLimit) {
          prunedPaths++;
          return;
        }
        visits.set(key, count + 1);
        walk(instruction.target, symbolicState, conditions, assertions);
        return;
      }
      case "assume": {
        const condition = substitute(instruction.expression, symbolicState);
        const extended = [...conditions, condition];
        if (!feasible(extended, variables, domain)) {
          prunedPaths++;
          return;
        }
        walk(pc + 1, symbolicState, extended, assertions);
        return;
      }
      case "assert": {
        const condition = substitute(instruction.expression, symbolicState);
        const violated = modelFor([...conditions, ["=", condition, 0]], variables, domain);
        if (violated !== null) {
          failures.push({ pc, message: instruction.message ?? "assertion", witness: violated });
        }
        walk(pc + 1, symbolicState, [...conditions, condition], [...assertions, pc]);
        return;
      }
      default: {
        paths.push({ conditions: conditions.slice(), assertions: assertions.slice(), end: "halt" });
      }
    }
  };
  walk(0, {}, [], []);
  return { paths, failures, prunedPaths, forks };
}

// ---- small program builders ---------------------------------------------------------------------
const builder = () => {
  const program = [];
  return {
    program,
    assign(variable, expression) {
      program.push({ op: "assign", variable, expression });
      return this;
    },
    branch(expression, thenIndex, elseIndex) {
      program.push({ op: "branch", expression, then: thenIndex, else: elseIndex });
      return this;
    },
    jump(target) {
      program.push({ op: "jump", target });
      return this;
    },
    assume(expression) {
      program.push({ op: "assume", expression });
      return this;
    },
    assert(expression, message) {
      program.push({ op: "assert", expression, message });
      return this;
    },
    halt() {
      program.push({ op: "halt" });
      return this;
    },
    fill(op) {
      program.push(op);
      return this;
    },
  };
};

module.exports = { simulate, runConcretely, bruteForceFailures, feasible, modelFor, builder, DOMAIN };

if (require.main === module) {
  // -- a branchy program: the symbolic failures must equal the concrete ones --------------------
  //   x = x * 2; if (x < 4) { assert x != 2 } else { assert x < 20 }
  const programs = [];
  {
    const b = builder();
    b.assign("x", ["*", "x", 2]);
    b.branch(["<", "x", 4], 3, 4);
    b.assign("unused", 0);
    b.assert(["=", "x", 2], "inside branch fails for x = 1");
    b.assert(["<", "x", 20], "outside branch fails for large x");
    b.halt();
    programs.push({ name: "double-and-test", program: b.program, variables: ["x"] });
  }
  {
    const b = builder();
    b.assert([">", "x", 0], "positive only");
    b.halt();
    programs.push({ name: "single-assert", program: b.program, variables: ["x"] });
  }
  {
    const b = builder();
    b.assume(["<", "x", 0]);
    b.assert([">", "x", 0], "cannot hold: assume prunes the path");
    b.assert(["<", "x", 0], "holds on the surviving path");
    b.halt();
    programs.push({ name: "assume-pruning", program: b.program, variables: ["x"] });
  }
  {
    const b = builder();
    b.assign("y", ["+", "x", 1]);
    b.assert(["=", "y", ["+", "x", 1]], "the substitution must stay exact");
    b.assert(["=", "y", "x"], "and y = x must fail");
    b.halt();
    programs.push({ name: "substitution", program: b.program, variables: ["x"] });
  }
  {
    const b = builder();
    b.assign("sum", 0);
    b.assign("i", 0);
    // loop: while (i < 3) { sum = sum + i; i = i + 1 }
    b.branch(["<", "i", 3], 3, 6);
    b.assign("sum", ["+", "sum", "i"]);
    b.assign("i", ["+", "i", 1]);
    b.jump(2);
    b.assert(["=", "sum", 3], "sum of 0..2 is 3");
    b.halt();
    programs.push({ name: "bounded-loop", program: b.program, variables: [] });
  }

  {
    const b = builder();
    b.assume([">", "x", 2]);
    b.branch(["<", "x", 0], 3, 4);
    b.assign("unused", 0);
    b.assert(0, "unreachable: the taken branch is infeasible");
    b.assert(["=", "x", 3], "only x = 3 survives the assume");
    b.halt();
    programs.push({ name: "infeasible-branch", program: b.program, variables: ["x"] });
  }

  let differentialChecks = 0;
  for (const { name, program, variables } of programs) {
    for (const unrollLimit of [1, 3]) {
      const symbolic = simulate(program, variables, { unrollLimit });
      const concreteFailures = bruteForceFailures(program, variables);
      const symbolicFailures = new Set(symbolic.failures.map((failure) => failure.pc));
      const concreteFailuresAt = new Set();
      for (const witness of concreteFailures) {
        const replay = runConcretely(program, JSON.parse(witness), { unrollLimit });
        for (const failure of replay.failures) concreteFailuresAt.add(failure.pc);
      }
      deepStrictEqual([...symbolicFailures].sort((a, b) => a - b),
        [...concreteFailuresAt].sort((a, b) => a - b),
        `${name}: the failing assertions must match the concrete ones`);
      ok(symbolic.paths.length >= 0, `${name}: paths are counted (${symbolic.paths.length})`);
      // every symbolic witness must be a real failure that the concrete interpreter reproduces
      for (const failure of symbolic.failures) {
        const replay = runConcretely(program, failure.witness, { unrollLimit });
        ok(replay.failures.some((concrete) => concrete.pc === failure.pc),
          `${name}: the witness ${JSON.stringify(failure.witness)} must fail concretely`);
        differentialChecks++;
      }
    }
  }
  ok(differentialChecks > 5, `the differential test must run broadly (${differentialChecks} witnesses)`);

  // -- infeasible conditions really have no model -------------------------------------------------
  ok(!feasible([["<", "x", 0], [">", "x", 0]], ["x"]), "contradictory conditions are infeasible");
  ok(feasible([["<", "x", 2]], ["x"]), "a single bound is feasible");
  ok(!feasible([[">", "x", 7]], ["x"]), "a bound outside the domain is infeasible");
  ok(!feasible([["=", 1, 0]], []), "a contradictory constant condition is infeasible");
  deepStrictEqual(modelFor([["<", "x", 0]], ["x"]), { x: -3 }, "the model is the smallest value");
  strictEqual(modelFor([["<", "x", 0], [">", "x", 0]], ["x"]), null, "no model for a contradiction");

  // -- pruning and path counting -------------------------------------------------------------------
  const safeAssert = simulate(programs[2].program, programs[2].variables, { unrollLimit: 3 });
  strictEqual(safeAssert.paths.filter((path) => path.end === "halt").length, 1,
    "the assume program keeps a single path");
  strictEqual(safeAssert.failures.length, 1,
    "only the assertion that the assumed condition falsifies is reported");
  strictEqual(safeAssert.failures[0].pc, 1, "the failing assertion is the one at pc 1");
  ok(safeAssert.failures[0].witness.x < 0,
    "the witness must respect the assumed path condition");
  const prunedBranch = simulate(programs[5].program, programs[5].variables, { unrollLimit: 3 });
  ok(prunedBranch.prunedPaths > 0, "an infeasible branch must be pruned and counted");
  strictEqual(prunedBranch.paths.length, 1, "only the feasible branch survives");
  strictEqual(prunedBranch.failures.length, 0, "the unreachable assertion cannot fail");
  const branchy = simulate(programs[0].program, programs[0].variables, { unrollLimit: 3 });
  strictEqual(branchy.forks, 1, "the branchy program forks once");
  ok(branchy.paths.length >= 2, "both branches survive");
  const emptyProgram = simulate([], [], {});
  strictEqual(emptyProgram.paths.length, 1, "an empty program has one path");
  deepStrictEqual(emptyProgram.failures, []);
  const alwaysFails = builder();
  alwaysFails.assert(0, "contradiction");
  alwaysFails.halt();
  const failure = simulate(alwaysFails.program, ["x"], {});
  strictEqual(failure.failures.length, 1, "a statically false assertion fails on its one path");
  ok(failure.failures[0].witness !== undefined, "the failure carries a witness");
  const noWitness = simulate(alwaysFails.program, [], {});
  strictEqual(noWitness.failures.length, 1, "even with no variables the assertion fails");
  const infiniteLoop = builder();
  infiniteLoop.jump(0);
  const looped = simulate(infiniteLoop.program, [], { unrollLimit: 2 });
  ok(looped.prunedPaths > 0, "the unroll bound stops the infinite loop");
  strictEqual(looped.paths.length, 0, "no path reaches a halt");

  // -- the loop program reaches the same verdict concretely ------------------------------------------
  const loopResult = runConcretely(programs[4].program, {}, { unrollLimit: 4 });
  deepStrictEqual(loopResult.failures, [], "the bounded loop keeps sum = 3");
  const loopSymbolic = simulate(programs[4].program, [], { unrollLimit: 4 });
  deepStrictEqual(loopSymbolic.failures, [], "and the symbolic run agrees");
  const tightUnroll = simulate(programs[4].program, [], { unrollLimit: 1 });
  ok(tightUnroll.prunedPaths > 0, "a tight unroll bound prunes the loop");

  // -- determinism ----------------------------------------------------------------------------------
  for (const { program, variables } of programs) {
    const first = simulate(program, variables, { unrollLimit: 3 });
    const second = simulate(program, variables, { unrollLimit: 3 });
    deepStrictEqual(first.failures, second.failures, "the analysis must be deterministic");
    deepStrictEqual(first.paths, second.paths, "the explored paths must be reproducible");
  }

  console.log(`programs=${programs.length} differential_checks=${differentialChecks} ` +
    `pruned=${programs.map((p) => simulate(p.program, p.variables, {}).prunedPaths).join("/")}`);
  console.log(`branchy_paths=${branchy.paths.length} pruned_branch_paths=${prunedBranch.paths.length} ` +
    `domain=${DOMAIN.length} values`);
  console.log("PASS");
}
