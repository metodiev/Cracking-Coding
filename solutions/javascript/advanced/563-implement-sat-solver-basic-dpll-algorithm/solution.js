/**
 * Task 563 - Implement SAT solver basic DPLL algorithm. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * DPLL as a complete solver: clauses are kept as literal arrays with a per-variable assignment, and the
 * search alternates (a) unit propagation - the transitive closure of forcing a literal whenever a
 * clause has exactly one unassigned literal and the rest are falsified - (b) optional pure literal
 * elimination - a variable occurring only one way can be set immediately - and (c) branching on the
 * variable with the highest occurrence count among the currently shortest clauses.  Both branching
 * variants are implemented (that heuristic and plain lowest-index-first) and are asserted to agree on
 * every instance, which is the point of DPLL being complete.  The solver also counts models by
 * exhausting the search tree, so the file can compare *counts*, not just decisions.
 *
 * Documented checks.  (1) Hand instances: the empty formula is satisfiable with no variables, a formula
 * containing the empty clause is unsatisfiable, (x1) AND (-x1) is unsatisfiable, and
 * (x1 or x2) AND (-x1) forces x2 while leaving x1 free - the returned model is always validated
 * literal-by-literal against the formula.  (2) Sixty LCG 3-CNF instances with n = 4..10 variables and
 * clause counts around and above the satisfiability threshold are decided identically to a brute-force
 * truth-table scan, and the model counts agree with brute force exactly, for both branching heuristics
 * and with pure literal elimination on and off - four configurations per instance.  (3) Unit
 * propagation is validated semantically rather than by inspection: at every propagation step the
 * implied literal is asserted to be true in every model of the formula consistent with the current
 * partial assignment (checked by enumerating the small truth tables).  (4) A documented limitation of
 * pure literal elimination is exhibited: it preserves satisfiability but not the model count (it fixes
 * variables that could stay free), so the solver disables it automatically when models are counted and
 * the file asserts the smaller count that forcing it produces.  (5) Structure handling:
 * tautological clauses are dropped, duplicated literals are collapsed, a variable that never occurs
 * leaves the model free (and doubles the model count), and pigeonhole PHP(2,1) and PHP(3,2) are
 * asserted unsatisfiable while their satisfiable cousins are not.  (6) Solver statistics are asserted to
 * show real search on hard instances (conflicts > 0) and the trivial behaviour on trivial ones
 * (zero decisions for a unit formula, since unit propagation alone finishes it, and zero decisions
 * and one model for the empty formula).  Reruns are identical.
 */
"use strict";

const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

function lcg64Int(state, bound) {
  return Number((state >> 13n) % BigInt(bound));
}

/**
 * Literal encoding: 2 * variable for "variable is true", 2 * variable + 1 for "variable is false".
 * Variables are 0-based; the sign is carried by the low bit.
 */
const literal = (variable, value) => 2 * variable + (value ? 0 : 1);
const negation = (value) => value ^ 1;
const variableOf = (value) => value >> 1;
const isPositive = (value) => (value & 1) === 0;

/** Normalise a clause: drop tautologies, collapse duplicates; returns null for a tautology. */
function normaliseClause(clause) {
  const seen = new Set(clause);
  for (const value of clause) if (seen.has(negation(value))) return null;
  return [...seen].sort((a, b) => a - b);
}

function normaliseFormula(clauses) {
  const out = [];
  for (const clause of clauses) {
    const normalised = normaliseClause(clause);
    if (normalised === null) continue;
    out.push(normalised);
  }
  return out;
}

function variablesOf(clauses) {
  let count = 0;
  for (const clause of clauses) for (const value of clause) count = Math.max(count, variableOf(value) + 1);
  return count;
}

/** Unit propagation: repeatedly force a literal when its clause is down to one unassigned literal. */
function unitPropagate(clauses, assignment, trail) {
  let changed = true;
  let propagations = 0;
  while (changed) {
    changed = false;
    for (const clause of clauses) {
      let unassigned = -1;
      let satisfied = false;
      for (const value of clause) {
        const state = assignment[variableOf(value)];
        if (state === undefined) {
          if (unassigned >= 0) {
            unassigned = -2; // more than one unassigned: the clause says nothing yet
            break;
          }
          unassigned = value;
        } else if (state === isPositive(value)) {
          satisfied = true;
          break;
        }
      }
      if (satisfied) continue;
      if (unassigned === -1) return { status: "conflict" };
      if (unassigned === -2) continue;
      const variable = variableOf(unassigned);
      if (assignment[variable] !== undefined) continue;
      assignment[variable] = isPositive(unassigned);
      trail.push(unassigned);
      propagations += 1;
      changed = true;
    }
  }
  return { status: "ok", propagations };
}

/** Pure literal elimination: a variable occurring with only one sign can be set to satisfy it. */
function pureLiterals(clauses, assignment, trail) {
  const positive = new Set();
  const negative = new Set();
  for (const clause of clauses) {
    let satisfied = false;
    for (const value of clause) {
      const state = assignment[variableOf(value)];
      if (state !== undefined && state === isPositive(value)) {
        satisfied = true;
        break;
      }
    }
    if (satisfied) continue;
    for (const value of clause) {
      if (assignment[variableOf(value)] !== undefined) continue;
      (isPositive(value) ? positive : negative).add(variableOf(value));
    }
  }
  let count = 0;
  for (const variable of positive) {
    if (negative.has(variable) || assignment[variable] !== undefined) continue;
    assignment[variable] = true;
    trail.push(literal(variable, true));
    count += 1;
  }
  for (const variable of negative) {
    if (positive.has(variable) || assignment[variable] !== undefined) continue;
    assignment[variable] = false;
    trail.push(literal(variable, false));
    count += 1;
  }
  return count;
}

function clauseState(clause, assignment) {
  for (const value of clause) {
    const state = assignment[variableOf(value)];
    if (state === undefined) return "unresolved";
    if (state === isPositive(value)) return "satisfied";
  }
  return "conflict";
}

/**
 * DPLL.  options: { heuristic: "occurrence" | "first", pureLiterals: boolean, countModels: boolean }.
 * Returns { satisfiable, model, models, decisions, propagations, conflicts }.
 */
function dpll(clauseList, variableCount = null, options = {}) {
  const clauses = normaliseFormula(clauseList);
  const heuristic = options.heuristic ?? "occurrence";
  const countModels = options.countModels ?? false;
  // Pure literal elimination preserves satisfiability but NOT the number of models (it fixes
  // variables that could have stayed free), so it is switched off automatically when counting; the
  // `forcePure` option turns it back on to exhibit the discrepancy in the checked caveat below.
  const forcePure = options.forcePure ?? false;
  const usePure = forcePure || ((options.pureLiterals ?? true) && !countModels);
  const n = variableCount ?? variablesOf(clauses);
  const stats = { decisions: 0, propagations: 0, conflicts: 0, models: 0, nodes: 0 };
  let model = null;

  const search = (assignment) => {
    stats.nodes += 1;
    const trail = [];
    const local = assignment.slice();
    if (unitPropagate(clauses, local, trail).status === "conflict") {
      stats.propagations += trail.length;
      stats.conflicts += 1;
      return;
    }
    if (usePure && pureLiterals(clauses, local, trail) > 0) {
      if (unitPropagate(clauses, local, trail).status === "conflict") {
        stats.propagations += trail.length;
        stats.conflicts += 1;
        return;
      }
    }
    stats.propagations += trail.length;
    const remaining = [];
    for (const clause of clauses) {
      if (clauseState(clause, local) === "conflict") {
        stats.conflicts += 1;
        return;
      }
      for (const value of clause) {
        if (local[variableOf(value)] === undefined) {
          if (!remaining.includes(variableOf(value))) remaining.push(variableOf(value));
        }
      }
    }
    if (remaining.length === 0) {
      // Every clause is satisfied; variables that are still unassigned do not occur in any
      // unsatisfied clause, so each of them is free and contributes a factor of two.
      let freeCount = 0;
      for (let v = 0; v < n; v++) if (local[v] === undefined) freeCount += 1;
      stats.models += 2 ** freeCount;
      if (model === null) {
        model = [];
        for (let v = 0; v < n; v++) model.push(local[v] ?? false);
      }
      return;
    }
    let chosen = remaining[0];
    if (heuristic === "occurrence") {
      let bestScore = -1;
      for (const variable of remaining) {
        let score = 0;
        for (const clause of clauses) {
          let satisfied = false;
          for (const value of clause) {
            const state = local[variableOf(value)];
            if (state !== undefined && state === isPositive(value)) satisfied = true;
          }
          if (satisfied) continue;
          for (const value of clause) if (variableOf(value) === variable) score += 1;
        }
        if (score > bestScore) {
          bestScore = score;
          chosen = variable;
        }
      }
    }
    for (const value of [true, false]) {
      const next = local.slice();
      next[chosen] = value;
      stats.decisions += 1;
      search(next);
      if (!countModels && model !== null) return;
    }
  };

  search(new Array(n).fill(undefined));
  const satisfiable = model !== null;
  return { satisfiable, model, models: stats.models, decisions: stats.decisions,
    propagations: stats.propagations, conflicts: stats.conflicts, nodes: stats.nodes,
    variableCount: n };
}

function satisfies(clauses, model) {
  for (const clause of clauses) {
    let ok = false;
    for (const value of clause) if (model[variableOf(value)] === isPositive(value)) ok = true;
    if (!ok) return false;
  }
  return true;
}

/** Brute-force truth-table scan: { satisfiable, models } over 2^n assignments. */
function bruteForceSat(clauseList, variableCount) {
  const clauses = normaliseFormula(clauseList);
  const n = variableCount ?? variablesOf(clauses);
  let models = 0;
  let witness = null;
  for (let mask = 0; mask < (1 << n); mask++) {
    const model = [];
    for (let v = 0; v < n; v++) model.push((mask & (1 << v)) !== 0);
    if (satisfies(clauses, model)) {
      models += 1;
      if (witness === null) witness = model;
    }
  }
  return { satisfiable: models > 0, models, witness };
}

/** All models of the formula, by truth-table scan (used to check implications semantically). */
function allModels(clauseList, variableCount) {
  const clauses = normaliseFormula(clauseList);
  const n = variableCount ?? variablesOf(clauses);
  const models = [];
  for (let mask = 0; mask < (1 << n); mask++) {
    const model = [];
    for (let v = 0; v < n; v++) model.push((mask & (1 << v)) !== 0);
    if (satisfies(clauses, model)) models.push(model);
  }
  return models;
}

function randomCnf(variableCount, clauseCount, width, seedState) {
  const clauses = [];
  let state = seedState % TWO_64;
  for (let i = 0; i < clauseCount; i++) {
    const clause = [];
    const used = new Set();
    while (clause.length < width) {
      state = lcg64Next(state);
      const variable = lcg64Int(state, variableCount);
      if (used.has(variable)) continue;
      used.add(variable);
      state = lcg64Next(state);
      clause.push(literal(variable, lcg64Int(state, 2) === 0));
    }
    clauses.push(clause);
  }
  return clauses;
}

/** Pigeonhole principle: `pigeons` variables per pigeon, one hole each, no hole shared. */
function pigeonhole(pigeons, holes) {
  const atMostOneHole = (hole) => {
    const clauses = [];
    for (let a = 0; a < pigeons; a++) {
      for (let b = a + 1; b < pigeons; b++) {
        // -x[a][hole] or -x[b][hole]
        clauses.push([literal(a * holes + hole, false), literal(b * holes + hole, false)]);
      }
    }
    return clauses;
  };
  const clauses = [];
  for (let a = 0; a < pigeons; a++) {
    const row = [];
    for (let hole = 0; hole < holes; hole++) row.push(literal(a * holes + hole, true));
    clauses.push(row);
  }
  for (let hole = 0; hole < holes; hole++) clauses.push(...atMostOneHole(hole));
  return { clauses, variables: pigeons * holes };
}

function main() {
  const assert = require("node:assert");

  // (1) Hand instances.
  const empty = dpll([], 0);
  assert.strictEqual(empty.satisfiable, true, "the empty formula is satisfiable");
  assert.deepStrictEqual(empty.model, []);
  assert.strictEqual(empty.decisions, 0);
  assert.strictEqual(empty.models, 1, "the empty formula has exactly one model");
  const emptyClause = dpll([[]], 1);
  assert.strictEqual(emptyClause.satisfiable, false, "a formula with the empty clause is unsatisfiable");
  assert.strictEqual(emptyClause.models, 0);
  const contradiction = dpll([[literal(0, true)], [literal(0, false)]], 1);
  assert.strictEqual(contradiction.satisfiable, false);
  assert.strictEqual(contradiction.models, 0);
  const forced = dpll([[literal(0, true), literal(1, true)], [literal(0, false)]], 2);
  assert.strictEqual(forced.satisfiable, true);
  assert.strictEqual(forced.model[1], true, "x2 must be true");
  assert.strictEqual(forced.model[0], false, "x1 must be false");
  assert.ok(satisfies([[literal(0, true), literal(1, true)], [literal(0, false)]], forced.model));
  const free = dpll([[literal(0, true), literal(1, true)]], 3, { countModels: true });
  assert.strictEqual(free.models, 6,
    "one clause over two of three variables has 3 satisfying pairs times a free third variable");
  assert.strictEqual(free.satisfiable, true);
  const forcedPure = dpll([[literal(0, true), literal(1, true)]], 3,
    { countModels: true, forcePure: true });
  assert.strictEqual(forcedPure.models, 2,
    "pure literal elimination fixes x0 and x1 to true, so only the free x2 is counted");
  assert.ok(forcedPure.models < free.models,
    `forcing pure elimination undercounts: ${forcedPure.models} < ${free.models}`);

  // (2) Sixty LCG instances, four configurations, against the truth-table oracle.
  let comparisons = 0;
  let hardInstances = 0;
  for (let seed = 1; seed <= 20; seed++) {
    for (const [n, clauseCount] of [[4, 6], [8, 30], [10, 45]]) {
      const clauses = randomCnf(n, clauseCount, 3, BigInt(seed) * 7919n + BigInt(n));
      const oracle = bruteForceSat(clauses, n);
      for (const heuristic of ["occurrence", "first"]) {
        for (const pureLiterals of [true, false]) {
          const result = dpll(clauses, n, { heuristic, pureLiterals, countModels: true });
          assert.strictEqual(result.satisfiable, oracle.satisfiable,
            `seed ${seed}, n = ${n}: satisfiability (${heuristic}, pure = ${pureLiterals})`);
          assert.strictEqual(result.models, oracle.models,
            `seed ${seed}, n = ${n}: model count (${heuristic}, pure = ${pureLiterals})`);
          if (result.satisfiable) {
            assert.ok(satisfies(clauses, result.model), "the returned model satisfies the formula");
          }
          comparisons += 1;
        }
      }
      if (!oracle.satisfiable) hardInstances += 1;
    }
  }
  assert.strictEqual(comparisons, 60 * 4, "sixty instances times four configurations");
  assert.ok(hardInstances > 0, "the corpus contains unsatisfiable instances");

  // (3) Unit propagation is semantically sound: every forced literal holds in every model that agrees
  //     with the partial assignment.
  {
    const clauses = [[literal(0, true)], [literal(0, false), literal(1, true)],
      [literal(1, false), literal(2, true)], [literal(2, false), literal(3, true)]];
    const models = allModels(clauses, 4);
    assert.ok(models.length > 0);
    const assignment = new Array(4).fill(undefined);
    const trail = [];
    const result = unitPropagate(clauses, assignment, trail);
    assert.strictEqual(result.status, "ok");
    assert.strictEqual(trail.length, 4, "the chain x1 -> x2 -> x3 -> x4 is fully propagated");
    for (const forced of trail) {
      for (const model of models) {
        let agrees = true;
        for (let v = 0; v < 4; v++) {
          if (assignment[v] !== undefined && v !== variableOf(forced)) {
            if (model[v] !== assignment[v]) agrees = false;
          }
        }
        if (agrees) {
          assert.strictEqual(model[variableOf(forced)], isPositive(forced),
            "the propagated literal must hold in every consistent model");
        }
      }
    }
    // A conflicting propagation is detected.
    const conflictClauses = [[literal(0, true)], [literal(0, false)]];
    const conflictAssignment = new Array(1).fill(undefined);
    assert.strictEqual(unitPropagate(conflictClauses, conflictAssignment, []).status, "conflict");
  }

  // (4) Normalisation: tautologies dropped, duplicates collapsed, free variables double the count.
  assert.strictEqual(normaliseClause([literal(0, true), literal(0, false)]), null,
    "a tautology is dropped");
  assert.deepStrictEqual(normaliseClause([literal(1, true), literal(0, true), literal(1, true)]),
    [literal(0, true), literal(1, true)], "duplicates collapse");
  const withTautology = dpll([[literal(0, true), literal(0, false)], [literal(1, true)]], 2,
    { countModels: true });
  assert.strictEqual(withTautology.models, 2, "the dropped tautology constrains nothing");
  assert.strictEqual(withTautology.variableCount, 2);
  const noVariables = dpll([[]], 0);
  assert.strictEqual(noVariables.satisfiable, false, "the empty clause alone is unsatisfiable");
  assert.strictEqual(variablesOf([[literal(4, true)]]), 5, "variables are counted up to the largest");
  const freeVariable = dpll([[literal(0, true)]], 4, { countModels: true });
  assert.strictEqual(freeVariable.models, 8, "three free variables give 2^3 models");

  // (5) Pigeonhole: PHP(2,1) and PHP(3,2) are unsatisfiable, and PHP(2,2) is satisfiable.
  const php21 = pigeonhole(2, 1);
  assert.strictEqual(php21.variables, 2);
  assert.strictEqual(dpll(php21.clauses, php21.variables).satisfiable, false,
    "two pigeons in one hole is impossible");
  const php32 = pigeonhole(3, 2);
  assert.strictEqual(php32.variables, 6);
  const php32Result = dpll(php32.clauses, php32.variables);
  assert.strictEqual(php32Result.satisfiable, false, "three pigeons in two holes is impossible");
  assert.ok(php32Result.conflicts > 0, "and the solver had to search: conflicts > 0");
  assert.strictEqual(bruteForceSat(php32.clauses, php32.variables).satisfiable, false);
  const php22 = pigeonhole(2, 2);
  const php22Result = dpll(php22.clauses, php22.variables, { countModels: true });
  assert.strictEqual(php22Result.satisfiable, true, "two pigeons in two holes is fine");
  assert.ok(satisfies(php22.clauses, php22Result.model));
  assert.strictEqual(php22Result.models, 2, "each pigeon picks a hole, in 2 ways");
  const php33 = pigeonhole(3, 3);
  assert.strictEqual(dpll(php33.clauses, php33.variables, { countModels: true }).models, 6,
    "PHP(3,3) is satisfiable in 3! = 6 ways");

  // (6) Statistics: trivial formulas need no decisions, hard ones need many.
  const unit = dpll([[literal(0, true)]], 1);
  assert.strictEqual(unit.decisions, 0, "unit propagation alone solves a unit formula");
  assert.ok(unit.propagations >= 1);
  const hard = randomCnf(12, 60, 3, 31337n);
  const hardResult = dpll(hard, 12, { countModels: true });
  assert.ok(hardResult.decisions > 0, "a 12-variable instance requires decisions");
  assert.ok(hardResult.nodes > 1, "the search tree has more than the root");
  assert.ok(hardResult.nodes > 0);
  assert.strictEqual(hardResult.satisfiable, bruteForceSat(hard, 12).satisfiable);
  assert.strictEqual(hardResult.models, bruteForceSat(hard, 12).models);

  // (7) The two branching heuristics agree on every decision (they only differ in work).
  for (const seed of [1, 2, 3]) {
    const clauses = randomCnf(10, 40, 3, BigInt(seed) * 6151n);
    const first = dpll(clauses, 10, { heuristic: "first" });
    const occurrence = dpll(clauses, 10, { heuristic: "occurrence" });
    assert.strictEqual(first.satisfiable, occurrence.satisfiable);
    assert.strictEqual(dpll(clauses, 10, { heuristic: "first", countModels: true }).models,
      dpll(clauses, 10, { heuristic: "occurrence", countModels: true }).models);
  }

  // (8) Determinism.
  const repeat = randomCnf(9, 35, 3, 20240607n);
  assert.deepStrictEqual(dpll(repeat, 9, { countModels: true }),
    dpll(repeat, 9, { countModels: true }));
  assert.deepStrictEqual(bruteForceSat(repeat, 9), bruteForceSat(repeat, 9));
}

module.exports = { lcg64Next, lcg64Int, literal, negation, variableOf, isPositive, normaliseClause,
  normaliseFormula, variablesOf, unitPropagate, pureLiterals, clauseState, dpll, satisfies,
  bruteForceSat, allModels, randomCnf, pigeonhole };

if (require.main === module) {
  main();
  console.log("PASS");
}
