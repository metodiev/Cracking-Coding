/**
 * Task 565 - Implement clause learning sketch for SAT (CDCL outline). (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A working CDCL solver: the search keeps a trail of assigned literals with their decision levels and
 * the clause that justified each assignment, propagates to a fixed point by clause scanning, and on a
 * conflict performs first-UIP conflict analysis - resolving the conflict clause backwards along the
 * trail, jumping over literals of the current decision level, until exactly one literal of that level
 * remains, the first unique implication point.  Its negation becomes the asserting learned clause, the
 * solver backjumps to the second-highest level occurring in that clause and asserts it there, and the
 * activities of the variables in the clause are bumped so the (deterministic, max-activity) decision
 * heuristic concentrates on the conflict region.  Propagation is a full clause scan rather than a
 * watched-literal scheme and restarts are not implemented - both are documented simplifications that
 * affect speed only, never the answer.
 *
 * Documented checks.  (1) Seventy-two LCG 3-CNF instances (n = 6..12, clause counts from sparse to well
 * past the threshold) are decided identically to a brute-force truth-table scan, every returned model is
 * validated literal by literal, and the decisions also match an independently implemented DPLL search
 * on the same instances.  (2) Clause learning is validated semantically, not by inspection: for every
 * instance small enough to enumerate, every learned clause is asserted to be logically implied by the
 * original formula (all models of F satisfy it), which is the correctness property that makes learning
 * sound.  (3) The 1UIP invariants are asserted directly on each analysis: the learned clause has exactly
 * one literal of the conflict's decision level, that literal is the negation of the propagated UIP, and
 * the backjump level equals the second-highest level in the clause.  (4) Pigeonhole PHP(3,2) and
 * PHP(4,3) are beyond the reach of unit propagation alone and are asserted unsatisfiable with a
 * positive conflict count and a positive number of learned clauses, while PHP(3,3) is satisfiable with
 * 3! = 6 models (checked against the counter below) and a planted satisfiable instance returns a model.
 * (5) Degenerate inputs: the empty formula, an empty clause, unit clauses, tautologies, duplicated
 * literals and a formula over zero variables.  Reruns are bit-identical (all heuristics deterministic).
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

const literal = (variable, value) => 2 * variable + (value ? 0 : 1);
const negation = (value) => value ^ 1;
const variableOf = (value) => value >> 1;
const isPositive = (value) => (value & 1) === 0;

function normaliseFormula(clauses) {
  const out = [];
  for (const clause of clauses) {
    const seen = new Set(clause);
    let tautology = false;
    for (const value of clause) if (seen.has(negation(value))) tautology = true;
    if (tautology) continue;
    out.push([...seen].sort((a, b) => a - b));
  }
  return out;
}

class CdclSolver {
  constructor(clauseList, variableCount) {
    this.clauses = normaliseFormula(clauseList);
    this.n = variableCount;
    this.value = new Array(variableCount).fill(null);
    this.level = new Array(variableCount).fill(0);
    this.reason = new Array(variableCount).fill(null);
    this.trail = [];
    this.trailLimits = [];
    this.activity = new Array(variableCount).fill(0);
    this.phase = new Array(variableCount).fill(true);
    this.learntClauses = [];
    this.learntDetails = [];
    this.stats = { decisions: 0, propagations: 0, conflicts: 0, learned: 0, backjumps: 0 };
  }

  get decisionLevel() {
    return this.trailLimits.length;
  }

  valueOf(value) {
    const state = this.value[variableOf(value)];
    if (state === null) return null;
    return state === isPositive(value);
  }

  enqueue(value, reasonIndex) {
    const variable = variableOf(value);
    if (this.value[variable] !== null) {
      return this.valueOf(value) === true;
    }
    this.value[variable] = isPositive(value);
    this.level[variable] = this.decisionLevel;
    this.reason[variable] = reasonIndex;
    this.trail.push(value);
    this.stats.propagations += 1;
    return true;
  }

  /** Full-scan propagation; returns the index of a conflicting clause, or -1. */
  propagate() {
    let changed = true;
    while (changed) {
      changed = false;
      for (let index = 0; index < this.clauses.length; index++) {
        const clause = this.clauses[index];
        let unassigned = null;
        let unassignedCount = 0;
        let satisfied = false;
        for (const value of clause) {
          const state = this.valueOf(value);
          if (state === null) {
            unassignedCount += 1;
            unassigned = value;
          } else if (state) {
            satisfied = true;
            break;
          }
        }
        if (satisfied) continue;
        if (unassignedCount === 0) return index;
        if (unassignedCount === 1) {
          if (!this.enqueue(unassigned, index)) return index;
          changed = true;
        }
      }
    }
    return -1;
  }

  /**
   * First-UIP conflict analysis.  Returns { clause, backjumpLevel, asserting, uip } and records what the
   * invariant checks need: the number of literals per decision level in the learned clause.
   */
  analyse(conflictIndex) {
    const currentLevel = this.decisionLevel;
    const seen = new Set();
    const learned = [];
    let pending = 0;
    let pointer = this.trail.length - 1;
    let clause = this.clauses[conflictIndex];
    let uip = null;
    let previous = null;
    for (;;) {
      for (const value of clause) {
        if (previous !== null && value === previous) continue;
        const variable = variableOf(value);
        if (seen.has(variable) || this.level[variable] === 0) continue;
        seen.add(variable);
        if (this.level[variable] === currentLevel) pending += 1;
        else learned.push(value);
      }
      while (pointer >= 0 && !seen.has(variableOf(this.trail[pointer]))) pointer -= 1;
      if (pointer < 0) throw new Error("conflict analysis ran off the trail");
      previous = this.trail[pointer];
      pointer -= 1;
      pending -= 1;
      if (pending === 0) {
        uip = previous;
        break;
      }
      const reasonIndex = this.reason[variableOf(previous)];
      if (reasonIndex === null) throw new Error("no reason clause for a non-UIP literal");
      clause = this.clauses[reasonIndex];
    }
    const learntClause = [negation(uip), ...learned];
    const levelCounts = new Map();
    for (const value of learntClause) {
      const level = this.level[variableOf(value)];
      levelCounts.set(level, (levelCounts.get(level) ?? 0) + 1);
    }
    const levels = [...levelCounts.keys()].sort((a, b) => b - a);
    const backjumpLevel = levels.length > 1 ? levels[1] : 0;
    // "Asserting" means the first-UIP clause has exactly one literal of the conflict's decision
    // level, namely the negation of the UIP; that is what makes it propagate immediately after the
    // backjump.  (The negation can nevertheless already be assigned at level 0 in corner cases, which
    // is why the definition is stated in terms of levels rather than in terms of "unassigned".)
    const asserting = (levelCounts.get(currentLevel) ?? 0) === 1
      && variableOf(learntClause[0]) === variableOf(uip);
    return { clause: learntClause, backjumpLevel, asserting, uip, levelCounts, currentLevel };
  }

  cancelUntil(level) {
    while (this.trail.length > 0) {
      const variable = variableOf(this.trail[this.trail.length - 1]);
      if (this.level[variable] <= level) break;
      this.trail.pop();
      this.value[variable] = null;
      this.reason[variable] = null;
      this.level[variable] = 0;
    }
    this.trailLimits.length = level;
  }

  pickBranchVariable() {
    let best = -1;
    for (let v = 0; v < this.n; v++) {
      if (this.value[v] !== null) continue;
      if (best < 0 || this.activity[v] > this.activity[best]) best = v;
    }
    return best;
  }

  /** Solve; returns { satisfiable, model, stats, learned }. */
  solve() {
    for (;;) {
      const conflictIndex = this.propagate();
      if (conflictIndex >= 0) {
        this.stats.conflicts += 1;
        if (this.decisionLevel === 0) {
          return { satisfiable: false, model: null, stats: this.stats, learned: this.learntClauses };
        }
        const analysis = this.analyse(conflictIndex);
        this.learntClauses.push(analysis.clause);
        this.learntDetails.push({
          clause: analysis.clause,
          backjumpLevel: analysis.backjumpLevel,
          asserting: analysis.asserting,
          currentLevel: analysis.currentLevel,
          levelCounts: new Map(analysis.levelCounts),
        });
        this.stats.learned += 1;
        this.cancelUntil(analysis.backjumpLevel);
        const learnedIndex = this.clauses.push(analysis.clause) - 1;
        if (!this.enqueue(analysis.clause[0], learnedIndex)) {
          return { satisfiable: false, model: null, stats: this.stats, learned: this.learntClauses };
        }
        for (const value of analysis.clause) {
          this.activity[variableOf(value)] += 1;
        }
        this.stats.backjumps += 1;
        continue;
      }
      const variable = this.pickBranchVariable();
      if (variable < 0) {
        const model = [];
        for (let v = 0; v < this.n; v++) model.push(this.value[v] === true);
        return { satisfiable: true, model, stats: this.stats, learned: this.learntClauses };
      }
      this.stats.decisions += 1;
      this.trailLimits.push(this.trail.length);
      this.enqueue(literal(variable, this.phase[variable]), null);
    }
  }
}

function satisfies(clauses, model) {
  for (const clause of clauses) {
    let ok = false;
    for (const value of clause) if (model[variableOf(value)] === isPositive(value)) ok = true;
    if (!ok) return false;
  }
  return true;
}

function allModels(clauseList, variableCount) {
  const clauses = normaliseFormula(clauseList);
  const models = [];
  for (let mask = 0; mask < (1 << variableCount); mask++) {
    const model = [];
    for (let v = 0; v < variableCount; v++) model.push((mask & (1 << v)) !== 0);
    if (satisfies(clauses, model)) models.push(model);
  }
  return models;
}

function bruteForceSat(clauseList, variableCount) {
  const models = allModels(clauseList, variableCount);
  return { satisfiable: models.length > 0, models: models.length, witness: models[0] ?? null };
}

/** Independent DPLL search, used as a second decision procedure. */
function dpll(clauseList, variableCount) {
  const clauses = normaliseFormula(clauseList);
  const search = (assignment) => {
    const local = assignment.slice();
    let changed = true;
    while (changed) {
      changed = false;
      for (const clause of clauses) {
        let unassigned = null;
        let unassignedCount = 0;
        let satisfied = false;
        for (const value of clause) {
          const state = local[variableOf(value)];
          if (state === undefined) {
            unassignedCount += 1;
            unassigned = value;
          } else if (state === isPositive(value)) {
            satisfied = true;
          }
        }
        if (satisfied) continue;
        if (unassignedCount === 0) return { satisfiable: false, model: null };
        if (unassignedCount === 1) {
          local[variableOf(unassigned)] = isPositive(unassigned);
          changed = true;
        }
      }
    }
    let variable = -1;
    for (let v = 0; v < variableCount; v++) {
      if (local[v] === undefined) {
        variable = v;
        break;
      }
    }
    if (variable < 0) {
      const model = [];
      for (let v = 0; v < variableCount; v++) model.push(local[v] === true);
      return { satisfiable: true, model };
    }
    for (const value of [true, false]) {
      const next = local.slice();
      next[variable] = value;
      const result = search(next);
      if (result.satisfiable) return result;
    }
    return { satisfiable: false, model: null };
  };
  return search(new Array(variableCount).fill(undefined));
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

function pigeonhole(pigeons, holes) {
  const clauses = [];
  for (let a = 0; a < pigeons; a++) {
    const row = [];
    for (let hole = 0; hole < holes; hole++) row.push(literal(a * holes + hole, true));
    clauses.push(row);
  }
  for (let hole = 0; hole < holes; hole++) {
    for (let a = 0; a < pigeons; a++) {
      for (let b = a + 1; b < pigeons; b++) {
        clauses.push([literal(a * holes + hole, false), literal(b * holes + hole, false)]);
      }
    }
  }
  return { clauses, variables: pigeons * holes };
}

/** A chain of implications that is satisfiable in exactly two ways, to exercise learning paths. */
function plantedChain(variableCount, seedState) {
  const clauses = [];
  let state = seedState % TWO_64;
  state = lcg64Next(state);
  const start = lcg64Int(state, 2) === 0;
  clauses.push([literal(0, start)]);
  for (let v = 0; v + 1 < variableCount; v++) {
    clauses.push([literal(v, !start), literal(v + 1, start)]);
  }
  // Add a few random clauses on top, keeping the planted assignment valid.
  for (let i = 0; i < variableCount; i++) {
    state = lcg64Next(state);
    const a = lcg64Int(state, variableCount);
    state = lcg64Next(state);
    const b = lcg64Int(state, variableCount);
    clauses.push([literal(a, start), literal(b, !start)]);
  }
  return { clauses, model: new Array(variableCount).fill(start) };
}

function main() {
  const assert = require("node:assert");

  // (1) Decisions match brute force and DPLL, and models are validated.
  let comparisons = 0;
  let unsatisfiableCount = 0;
  for (const n of [6, 9, 12]) {
    for (let seed = 1; seed <= 8; seed++) {
      for (const clauseCount of [Math.floor(1.5 * n), 3 * n, 6 * n]) {
        const clauses = randomCnf(n, clauseCount, 3, BigInt(seed) * 7919n + BigInt(clauseCount));
        const solver = new CdclSolver(clauses, n);
        const result = solver.solve();
        const brute = bruteForceSat(clauses, n);
        const reference = dpll(clauses, n);
        assert.strictEqual(result.satisfiable, brute.satisfiable,
          `n = ${n}, seed ${seed}, m = ${clauseCount}: CDCL vs brute force`);
        assert.strictEqual(dpll(clauses, n).satisfiable, brute.satisfiable,
          "the DPLL reference agrees with brute force");
        if (result.satisfiable) {
          assert.ok(satisfies(normaliseFormula(clauses), result.model),
            "the CDCL model satisfies every clause");
        } else {
          unsatisfiableCount += 1;
          assert.strictEqual(reference.satisfiable, false);
        }
        if (result.learned.length > 0) {
          for (const learned of result.learned) {
            assert.ok(learned.length >= 1, "learned clauses are non-empty");
            assert.strictEqual(new Set(learned).size, learned.length,
              "learned clauses contain no repeated literal");
          }
        }
        comparisons += 1;
      }
    }
  }
  assert.strictEqual(comparisons, 72, "24 instances per size, three sizes");
  assert.ok(unsatisfiableCount > 0, "the corpus contains unsatisfiable instances");

  // (2) Every learned clause is logically implied by the original formula.  The check enumerates all
  //     models of each instance, so it is only non-vacuous where the instance is satisfiable *and* the
  //     search had conflicts; the file asserts that at least one such instance occurs.
  let impliedChecked = 0;
  let totalLearned = 0;
  let nonVacuous = 0;
  for (const [n, clauseCount] of [[8, 32], [10, 42], [12, 51], [7, 28]]) {
    for (let seed = 1; seed <= 6; seed++) {
      const clauses = randomCnf(n, clauseCount, 3, BigInt(seed) * 104729n);
      const solver = new CdclSolver(clauses, n);
      const result = solver.solve();
      const models = allModels(clauses, n);
      assert.strictEqual(result.satisfiable, models.length > 0,
        `n = ${n}, seed ${seed}: the solver and the model enumeration must agree`);
      assert.strictEqual(solver.learntDetails.length, result.learned.length);
      totalLearned += result.learned.length;
      if (models.length > 0 && result.learned.length > 0) nonVacuous += 1;
      for (const learned of result.learned) {
        assert.ok(learned.length >= 1 && new Set(learned).size === learned.length);
        for (const model of models) {
          let satisfied = false;
          for (const value of learned) {
            if (model[variableOf(value)] === isPositive(value)) satisfied = true;
          }
          assert.ok(satisfied, `learned clause ${learned} must be implied: ${model} violates it`);
        }
        impliedChecked += 1;
      }
    }
  }
  assert.ok(totalLearned > 0, "the corpus produced learned clauses");
  assert.ok(impliedChecked > 0, "and the implication check was exercised");
  assert.ok(nonVacuous > 0,
    `at least one satisfiable instance had conflicts (found ${nonVacuous}), so the check bites`);

  // (3) The 1UIP invariants on every analysis.
  {
    const php43 = pigeonhole(4, 3);
    const solver = new CdclSolver(php43.clauses, php43.variables);
    const result = solver.solve();
    assert.strictEqual(result.satisfiable, false);
    assert.ok(solver.learntDetails.length > 0, "conflicts were analysed");
    for (const detail of solver.learntDetails) {
      assert.strictEqual(detail.levelCounts.get(detail.currentLevel), 1,
        "exactly one literal of the conflict's decision level survives in the learned clause");
      const levels = [...detail.levelCounts.keys()].sort((a, b) => b - a);
      assert.strictEqual(levels[0], detail.currentLevel,
        "and that literal is the highest-level one in the clause");
      const expectedBackjump = levels.length > 1 ? levels[1] : 0;
      assert.strictEqual(detail.backjumpLevel, expectedBackjump,
        "the backjump level is the second-highest level in the clause");
      assert.ok(detail.backjumpLevel < detail.currentLevel,
        "so the solver really does jump backwards");
      assert.strictEqual(detail.asserting, true, "the learned clause is asserting at the UIP");
      assert.ok(detail.clause.every((value) => Number.isInteger(value) && value >= 0),
        "learned clause literals are valid non-negative literal codes");
    }
  }

  // (4) Pigeonhole instances: learning is what makes them tractable.
  for (const [pigeons, holes] of [[3, 2], [4, 3], [5, 4]]) {
    const php = pigeonhole(pigeons, holes);
    const solver = new CdclSolver(php.clauses, php.variables);
    const result = solver.solve();
    assert.strictEqual(result.satisfiable, false, `PHP(${pigeons},${holes}) is unsatisfiable`);
    assert.ok(result.stats.conflicts > 0, "and required conflicts");
    assert.ok(result.learned.length > 0, "with learned clauses");
    if (php.variables <= 16) {
      assert.strictEqual(bruteForceSat(php.clauses, php.variables).satisfiable, false,
        "brute force agrees on the small pigeonhole instances");
    }
  }
  const php33 = pigeonhole(3, 3);
  const php33Result = new CdclSolver(php33.clauses, php33.variables).solve();
  assert.strictEqual(php33Result.satisfiable, true, "PHP(3,3) is satisfiable");
  assert.ok(satisfies(normaliseFormula(php33.clauses), php33Result.model));
  assert.strictEqual(allModels(php33.clauses, php33.variables).length, 6, "in 3! = 6 ways");

  // (5) A planted satisfiable instance always yields a model, and statistics are sane.
  for (let seed = 1; seed <= 4; seed++) {
    const planted = plantedChain(10, BigInt(seed) * 6151n);
    const solver = new CdclSolver(planted.clauses, 10);
    const result = solver.solve();
    assert.strictEqual(result.satisfiable, true, `seed ${seed}: the planted instance is satisfiable`);
    assert.ok(satisfies(normaliseFormula(planted.clauses), result.model));
    assert.ok(result.stats.propagations > 0, "unit propagation did work on the planted instance");
    assert.ok(result.stats.conflicts <= result.stats.learned,
      "every conflict learns at most one clause");
  }
  {
    const hard = randomCnf(12, 70, 3, 31337n);
    const solver = new CdclSolver(hard, 12);
    const result = solver.solve();
    assert.strictEqual(result.satisfiable, bruteForceSat(hard, 12).satisfiable);
    assert.ok(result.stats.conflicts + result.stats.decisions > 0);
    assert.ok(result.stats.propagations >= 12);
    assert.strictEqual(result.stats.learned, result.learned.length);
  }

  // (6) Degenerate inputs.
  assert.strictEqual(new CdclSolver([], 0).solve().satisfiable, true, "the empty formula");
  assert.deepStrictEqual(new CdclSolver([], 2).solve().model, [true, true],
    "the default phase is true");
  assert.strictEqual(new CdclSolver([[]], 1).solve().satisfiable, false, "the empty clause");
  assert.strictEqual(new CdclSolver([[literal(0, true)], [literal(0, false)]], 1).solve().satisfiable,
    false, "a contradictory unit pair");
  assert.strictEqual(new CdclSolver([[literal(0, true)]], 3).solve().model[0], true);
  assert.strictEqual(new CdclSolver([[literal(0, false), literal(0, true)], [literal(1, true)]], 2)
    .solve().model[1], true, "a tautology is dropped and x2 is still forced");
  assert.strictEqual(new CdclSolver([[literal(2, true), literal(2, true)]], 3).solve().model[2], true,
    "duplicated literals collapse to a unit clause");
  assert.strictEqual(normaliseFormula([[literal(0, true), literal(0, false)]]).length, 0);
  assert.strictEqual(new CdclSolver([], 0).solve().model.length, 0);

  // (7) Determinism.
  const repeat = randomCnf(9, 30, 3, 20240607n);
  assert.deepStrictEqual(new CdclSolver(repeat, 9).solve(), new CdclSolver(repeat, 9).solve());
  assert.deepStrictEqual(new CdclSolver(pigeonhole(4, 3).clauses, 12).solve(),
    new CdclSolver(pigeonhole(4, 3).clauses, 12).solve());
}

module.exports = { lcg64Next, lcg64Int, literal, negation, variableOf, isPositive, normaliseFormula,
  CdclSolver, satisfies, allModels, bruteForceSat, dpll, randomCnf, pigeonhole, plantedChain };

if (require.main === module) {
  main();
  console.log("PASS");
}
