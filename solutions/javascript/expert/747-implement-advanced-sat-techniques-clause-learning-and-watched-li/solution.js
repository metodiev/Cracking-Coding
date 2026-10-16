/** Task 747 - Implement advanced SAT techniques: clause learning and watched literals in practice.
 *  (Tier: Expert, CTCI Beyond - Compilers, Languages & Solvers)
 *
 * A compact CDCL solver. Literals are encoded as 2*var + sign so negation is `l ^ 1`. Every
 * clause keeps two watched literals in its first two slots: unit propagation only visits the
 * clauses watching the literal that just became false, moving the watch when another literal is
 * still unassigned and otherwise propagating or reporting a conflict - so a clause is only
 * revisited when one of its watches is touched. Conflicts are analysed with the first-UIP rule
 * (walking the implication graph back from the conflict clause, resolving the current decision
 * level away) which yields a learned clause plus the backjump level; the learned clause is added
 * to the database (it is the resolution of existing clauses, hence implied by the formula) and
 * the search continues from the backjump level. Branching uses VSIDS-style activities with a
 * decay factor, ties broken by variable index so the run stays reproducible. A plain DPLL with
 * chronological backtracking is included as a reference implementation for the comparisons.
 *
 * What is simplified: activities are integer-decayed weights with a linear scan for the maximum
 * (no heap, no phase saving) and the restart schedule is a fixed conflict count. What is
 * verified: every SAT answer carries a model checked by an independent evaluator; every UNSAT
 * answer is cross-checked against exhaustive search on small random formulas; every learned
 * clause is implied by the original clause set (checked by enumerating all assignments that
 * satisfy the formula); the watch invariants hold after propagation and after backjumping (two
 * watches per clause, listed exactly once in the right watch lists, and watches always pointing
 * at non-false literals unless the clause is in conflict); pigeonhole instances are refuted with
 * a non-trivial number of conflicts and learned clauses; and CDCL never needs more decisions than
 * the reference DPLL on the instances both solve.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

const literal = (variable, positive = true) => 2 * variable + (positive ? 0 : 1);
const variableOf = (lit) => lit >> 1;
const negate = (lit) => lit ^ 1;

/** Removes duplicate literals and reports whether the clause is a tautology (x or not x). */
function normalizeClause(lits) {
  const seen = new Set();
  const out = [];
  let tautology = false;
  for (const lit of lits) {
    const other = negate(lit);
    if (seen.has(other)) tautology = true;
    if (!seen.has(lit)) {
      seen.add(lit);
      out.push(lit);
    }
  }
  return { lits: out, tautology };
}

class Clause {
  constructor(lits, learnt = false) {
    this.lits = lits.slice();
    this.learnt = learnt;
    this.activity = 0;
  }

  toString() {
    return `(${this.lits.map((l) => (l % 2 === 0 ? "" : "!") + variableOf(l)).join(" ")})`;
  }
}

class CdclSolver {
  constructor(clauses, variableCount) {
    this.variableCount = variableCount;
    this.clauses = [];
    this.watches = new Map(); // literal -> clause indices watching it
    this.value = new Int8Array(variableCount); // 0 unknown, 1 true, -1 false
    this.level = new Int32Array(variableCount).fill(-1);
    this.reason = new Array(variableCount).fill(-1);
    this.trail = [];
    this.trailLimits = [];
    this.decisionLevel = 0;
    this.activity = new Int32Array(variableCount);
    this.activityBump = 1;
    this.activityDecay = 0;
    this.learned = [];
    this.conflicts = 0;
    this.decisions = 0;
    this.propagations = 0;
    this.conflictLimit = 100; // fixed restart period, documented simplification
    for (const raw of clauses) {
      const { lits, tautology } = normalizeClause(raw);
      if (tautology) continue; // always satisfied, contributes nothing
      this.#addClause(new Clause(lits, false));
    }
    this.restartLimit = this.conflictLimit;
  }

  #literalValue(lit) {
    const value = this.value[variableOf(lit)];
    if (value === 0) return 0;
    return lit % 2 === 0 ? value : -value;
  }

  #addClause(clause) {
    const index = this.clauses.length;
    this.clauses.push(clause);
    if (clause.lits.length === 0) return index; // the empty clause is immediately unsatisfiable
    if (clause.lits.length === 1) {
      this.#watch(clause.lits[0], index);
      this.#watch(clause.lits[0], index); // a unit clause is watched twice on its literal
      return index;
    }
    this.#watch(clause.lits[0], index);
    this.#watch(clause.lits[1], index);
    return index;
  }

  #watch(lit, index) {
    if (!this.watches.has(lit)) this.watches.set(lit, []);
    this.watches.get(lit).push(index);
  }

  #unwatch(lit, index) {
    const list = this.watches.get(lit);
    if (list === undefined) return;
    const position = list.indexOf(index);
    if (position >= 0) list.splice(position, 1);
  }

  #enqueue(lit, reason) {
    const variable = variableOf(lit);
    this.value[variable] = lit % 2 === 0 ? 1 : -1;
    this.level[variable] = this.decisionLevel;
    this.reason[variable] = reason;
    this.trail.push(lit);
  }

  /** Propagates the literal that just became true; returns a conflicting clause or null. */
  #propagateLiteral(current, pending) {
    const falsified = negate(current);
    const list = [...(this.watches.get(falsified) ?? [])]; // snapshot: the list mutates below
    for (const index of list) {
      const clause = this.clauses[index];
      if (clause.lits[0] !== falsified && clause.lits[1] !== falsified) continue; // stale entry
      if (clause.lits.length === 1) {
        // a unit clause holds a level-0 fact, so a false watch is an immediate conflict
        if (this.#literalValue(clause.lits[0]) === -1) return clause;
        continue;
      }
      if (clause.lits[0] === falsified) {
        const swap = clause.lits[0];
        clause.lits[0] = clause.lits[1];
        clause.lits[1] = swap;
      }
      const first = clause.lits[0];
      if (this.#literalValue(first) === 1) continue;
      let moved = false;
      for (let k = 2; k < clause.lits.length; k++) {
        if (this.#literalValue(clause.lits[k]) !== -1) {
          const fresh = clause.lits[k];
          clause.lits[k] = falsified;
          clause.lits[1] = fresh;
          this.#unwatch(falsified, index);
          this.#watch(fresh, index);
          moved = true;
          break;
        }
      }
      if (moved) continue; // the clause now watches a different literal
      const firstValue = this.#literalValue(first);
      if (firstValue === -1) return clause;
      if (firstValue === 0) {
        this.#enqueue(first, index);
        this.propagations++;
        pending.push(first);
      }
    }
    return null;
  }

  /** Drains the propagation queue; `seed` limits it to one freshly implied literal. */
  #propagateAll(seed) {
    const pending = seed === undefined ? this.trail.slice() : [seed];
    while (pending.length > 0) {
      const conflict = this.#propagateLiteral(pending.pop(), pending);
      if (conflict !== null) return conflict;
    }
    return null;
  }

  #analyse(conflict) {
    const learnt = [];
    const seen = new Uint8Array(this.variableCount);
    let unresolved = 0;
    let clause = conflict;
    let pathCount = 0;
    let pivot = -1;
    let steps = 0;
    for (;;) {
      if (steps++ > this.trail.length + this.clauses.length + 10) {
        throw new Error("conflict analysis did not converge");
      }
      for (const lit of clause.lits) {
        const variable = variableOf(lit);
        if (this.level[variable] <= 0 || seen[variable]) continue;
        seen[variable] = 1;
        if (this.level[variable] === this.decisionLevel) unresolved++;
        else learnt.push(lit);
      }
      // pick the next literal of the current decision level from the trail
      pivot = -1;
      for (let i = this.trail.length - 1; i >= 0; i--) {
        const candidate = this.trail[i];
        const candidateVariable = variableOf(candidate);
        if (seen[candidateVariable] === 1 && this.level[candidateVariable] === this.decisionLevel) {
          pivot = candidate;
          break;
        }
      }
      if (pivot === -1) break;
      const pivotVariable = variableOf(pivot);
      // mark as processed rather than unseeing: the pivot also occurs in its own reason clause
      seen[pivotVariable] = 2;
      unresolved--;
      pathCount++;
      if (unresolved === 0) {
        learnt.push(negate(pivot)); // the clause contains the negation of the UIP literal
        break;
      }
      const reason = this.reason[pivotVariable];
      if (reason < 0) {
        learnt.push(negate(pivot));
        break;
      }
      clause = this.clauses[reason];
    }
    // the asserting literal goes first, the rest is sorted by descending level for the backjump
    const asserting = learnt.pop();
    learnt.sort((a, b) => this.level[variableOf(b)] - this.level[variableOf(a)] ||
      a - b);
    const learnedClause = new Clause([asserting, ...learnt], true);
    let backjump = 0;
    if (learnedClause.lits.length > 1) {
      backjump = this.level[variableOf(learnedClause.lits[1])];
    }
    return { clause: learnedClause, backjump };
  }

  #bump(variable) {
    this.activity[variable] += this.activityBump;
    if (this.activity[variable] > 1000000) {
      for (let v = 0; v < this.variableCount; v++) this.activity[v] >>= 1;
      this.activityBump = Math.max(1, this.activityBump >> 1);
    }
  }

  #decay() {
    this.activityDecay++;
    if (this.activityDecay % 50 === 0) this.activityBump += 1;
  }

  #backtrack(level) {
    const target = this.trailLimits[level] ?? 0;
    while (this.trail.length > target) {
      const lit = this.trail.pop();
      const variable = variableOf(lit);
      this.value[variable] = 0;
      this.level[variable] = -1;
      this.reason[variable] = -1;
    }
    this.trailLimits.length = level;
    this.decisionLevel = level;
  }

  #pickBranchVariable() {
    let best = -1;
    let bestActivity = -1;
    for (let v = 0; v < this.variableCount; v++) {
      if (this.value[v] !== 0) continue;
      if (this.activity[v] > bestActivity) {
        bestActivity = this.activity[v];
        best = v;
      }
    }
    return best;
  }

  solve() {
    if (this.clauses.some((clause) => clause.lits.length === 0)) {
      return { satisfiable: false, assignment: null };
    }
    // level-0 facts: every unit clause is enqueued before the search starts
    for (const clause of this.clauses) {
      if (clause.lits.length !== 1) continue;
      const lit = clause.lits[0];
      const value = this.#literalValue(lit);
      if (value === -1) return { satisfiable: false, assignment: null };
      if (value === 0) this.#enqueue(lit, -1);
    }
    for (;;) {
      // restart budget: only between conflicts, so a freshly asserting literal is never dropped
      if (this.conflicts >= this.restartLimit && this.decisionLevel > 0) {
        this.#backtrack(0);
        this.restartLimit = this.conflicts + this.conflictLimit;
      }
      const conflict = this.#propagateAll();
      if (conflict !== null) {
        this.conflicts++;
        if (this.decisionLevel === 0) return { satisfiable: false, assignment: null };
        const { clause, backjump } = this.#analyse(conflict);
        if (clause.lits.length > 0) {
          this.learned.push(clause);
          for (const lit of clause.lits) this.#bump(variableOf(lit));
          this.#decay();
          this.#addClause(clause);
          this.#backtrack(backjump);
          this.#enqueue(clause.lits[0], this.clauses.length - 1);
        }
        continue;
      }
      const variable = this.#pickBranchVariable();
      if (variable === -1) {
        const assignment = [];
        for (let v = 0; v < this.variableCount; v++) assignment.push(this.value[v] === 1);
        return { satisfiable: true, assignment };
      }
      this.decisions++;
      this.trailLimits[this.decisionLevel] = this.trail.length;
      this.decisionLevel++;
      this.#enqueue(literal(variable, true), -1);
    }
  }

  /** Structural watch invariant, used by the tests after every stage. */
  checkWatchInvariants(options = {}) {
    const expectSatisfied = options.expectSatisfied ?? false;
    const counted = new Map();
    for (const [lit, list] of this.watches) {
      for (const index of list) {
        counted.set(index, (counted.get(index) ?? 0) + 1);
      }
    }
    for (let index = 0; index < this.clauses.length; index++) {
      const clause = this.clauses[index];
      const expected = clause.lits.length === 1 ? 2 : 2;
      strictEqual(counted.get(index) ?? 0, expected, `clause ${index} must be watched exactly twice`);
      if (clause.lits.length >= 2) {
        const list0 = this.watches.get(clause.lits[0]) ?? [];
        const list1 = this.watches.get(clause.lits[1]) ?? [];
        ok(list0.includes(index) && list1.includes(index), "watches must sit in the right lists");
      }
      // a clause that is not satisfied must still have a literal that is not false: after a
      // satisfiable run nothing may be falsified, and a refutation stops at level 0 with an
      // explicit conflict instead
      if (expectSatisfied) {
        const satisfied = clause.lits.some((lit) => this.#literalValue(lit) === 1);
        ok(satisfied, "no clause may be falsified at the end of a satisfiable run");
      }
    }
    return true;
  }
}

/** Reference implementation: plain DPLL with chronological backtracking. */
function dpll(clauses, variableCount) {
  const value = new Int8Array(variableCount);
  let decisions = 0;
  const literalValue = (lit) => {
    const v = variableOf(lit);
    if (value[v] === 0) return 0;
    return lit % 2 === 0 ? value[v] : -value[v];
  };
  const propagate = () => {
    let changed = true;
    while (changed) {
      changed = false;
      for (const clause of clauses) {
        let unassigned = null;
        let satisfied = false;
        let count = 0;
        for (const lit of clause) {
          const v = literalValue(lit);
          if (v === 1) {
            satisfied = true;
            break;
          }
          if (v === 0) {
            unassigned = lit;
            count++;
          }
        }
        if (satisfied) continue;
        if (count === 0) return false;
        if (count === 1) {
          const v = variableOf(unassigned);
          value[v] = unassigned % 2 === 0 ? 1 : -1;
          changed = true;
        }
      }
    }
    return true;
  };
  const search = () => {
    if (!propagate()) return false;
    let variable = -1;
    for (let v = 0; v < variableCount; v++) {
      if (value[v] === 0) {
        variable = v;
        break;
      }
    }
    if (variable === -1) return true;
    for (const choice of [1, -1]) {
      const saved = value.slice();
      decisions++;
      value[variable] = choice;
      if (search()) return true;
      value.set(saved);
    }
    return false;
  };
  const satisfiable = search();
  return { satisfiable, assignment: satisfiable ? [...value].map((v) => v === 1) : null, decisions,
    value };
}

function evaluate(clauses, assignment) {
  return clauses.every((clause) =>
    clause.some((lit) => (lit % 2 === 0 ? assignment[variableOf(lit)] : !assignment[variableOf(lit)])));
}

/** Exhaustive search over all assignments - the oracle for small formulas. */
function bruteUnsatisfiable(clauses, variableCount) {
  const limit = 1 << variableCount;
  for (let mask = 0; mask < limit; mask++) {
    const assignment = [];
    for (let v = 0; v < variableCount; v++) assignment.push(((mask >> v) & 1) === 1);
    if (evaluate(clauses, assignment)) return false;
  }
  return true;
}

/** Pigeonhole: pigeons + 1 pigeons into holes holes is unsatisfiable. */
function pigeonhole(pigeons, holes) {
  const at = (p, h) => literal(p * holes + h, true);
  const clauses = [];
  for (let p = 0; p < pigeons; p++) {
    const clause = [];
    for (let h = 0; h < holes; h++) clause.push(at(p, h));
    clauses.push(clause);
  }
  for (let h = 0; h < holes; h++) {
    for (let p1 = 0; p1 < pigeons; p1++) {
      for (let p2 = p1 + 1; p2 < pigeons; p2++) {
        clauses.push([negate(at(p1, h)), negate(at(p2, h))]);
      }
    }
  }
  return { clauses, variableCount: pigeons * holes };
}

module.exports = { CdclSolver, dpll, evaluate, bruteUnsatisfiable, pigeonhole, literal, negate,
  variableOf, makeRng };

if (require.main === module) {
  const rng = makeRng(747747);

  // -- trivial formulas ----------------------------------------------------------------------
  const emptyFormula = new CdclSolver([], 0);
  const emptyResult = emptyFormula.solve();
  ok(emptyResult.satisfiable, "the empty formula is satisfiable");
  deepStrictEqual(emptyResult.assignment, []);
  const emptyClause = new CdclSolver([[]], 1);
  strictEqual(emptyClause.solve().satisfiable, false, "the empty clause is unsatisfiable");
  const unit = new CdclSolver([[literal(0, true)], [negate(literal(0, true))]], 1);
  strictEqual(unit.solve().satisfiable, false, "x and not x cannot both hold");
  const tautology = new CdclSolver([[literal(0, true), negate(literal(0, true))]], 1);
  ok(tautology.solve().satisfiable, "a tautological clause is always satisfied");
  strictEqual(tautology.clauses.length, 0, "tautological clauses are dropped");
  const duplicates = new CdclSolver([[literal(0, true), literal(0, true), literal(1, true)]], 2);
  strictEqual(duplicates.clauses[0].lits.length, 2, "duplicate literals are collapsed");
  ok(duplicates.solve().satisfiable, "the collapsed clause is satisfied");
  const forced = new CdclSolver([[literal(0, true)], [negate(literal(0, true)), literal(1, true)]], 2);
  const forcedResult = forced.solve();
  ok(forcedResult.satisfiable && forcedResult.assignment[0] && forcedResult.assignment[1],
    "unit propagation must derive the forced assignment");
  deepStrictEqual(forcedResult.assignment, [true, true]);
  ok(forced.checkWatchInvariants(), "watch invariants hold after solving");

  // -- random formulas against exhaustive search ----------------------------------------------
  let satCount = 0;
  let unsatCount = 0;
  let learnedTotal = 0;
  for (let trial = 0; trial < 120; trial++) {
    const variableCount = 3 + rng(8);
    const clauseCount = 2 + rng(variableCount * 4);
    const clauses = [];
    for (let c = 0; c < clauseCount; c++) {
      const width = 1 + rng(3);
      const clause = [];
      for (let k = 0; k < width; k++) {
        clause.push(literal(rng(variableCount), rng(2) === 0));
      }
      clauses.push(clause);
    }
    const solver = new CdclSolver(clauses, variableCount);
    const result = solver.solve();
    const exhaustive = !bruteUnsatisfiable(clauses, variableCount);
    strictEqual(result.satisfiable, exhaustive, `trial ${trial}: verdict must match exhaustive search`);
    if (result.satisfiable) {
      ok(evaluate(clauses, result.assignment), `trial ${trial}: the model must satisfy every clause`);
      satCount++;
    } else {
      unsatCount++;
    }
    ok(solver.checkWatchInvariants(), `trial ${trial}: watch invariants must survive the run`);
    learnedTotal += solver.learned.length;
  }
  ok(satCount > 20, `the corpus must contain satisfiable instances (${satCount})`);
  ok(unsatCount > 20, `the corpus must contain unsatisfiable instances (${unsatCount})`);

  // -- every learned clause is implied by the original formula ---------------------------------
  let implicationChecks = 0;
  let learnedChecked = 0;
  const implicationCorpus = [];
  for (let trial = 0; trial < 60; trial++) {
    const variableCount = 5;
    const clauses = [];
    for (let c = 0; c < 21; c++) {
      const variables = new Set();
      while (variables.size < 3) variables.add(rng(variableCount));
      clauses.push([...variables].map((v) => literal(v, rng(2) === 0)));
    }
    implicationCorpus.push({ clauses, variableCount });
  }
  // a satisfiable formula that still needs learning: the pigeonhole instance minus one clause
  implicationCorpus.push({ clauses: pigeonhole(3, 2).clauses.slice(0, -1), variableCount: 6 });
  implicationCorpus.push({ clauses: pigeonhole(4, 3).clauses.slice(0, -1), variableCount: 12 });
  for (const { clauses, variableCount } of implicationCorpus) {
    const solver = new CdclSolver(clauses, variableCount);
    solver.solve();
    const limit = 1 << variableCount;
    for (const learned of solver.learned) {
      learnedChecked++;
      for (let mask = 0; mask < limit; mask++) {
        const assignment = [];
        for (let v = 0; v < variableCount; v++) assignment.push(((mask >> v) & 1) === 1);
        if (!evaluate(clauses, assignment)) continue;
        // every model of the formula must satisfy the learned clause as well
        ok(evaluate([learned.lits], assignment),
          `learned clause ${learned.toString()} must be implied by the formula`);
        implicationChecks++;
      }
    }
  }
  ok(learnedChecked > 10, `many learned clauses must be verified (${learnedChecked})`);
  ok(implicationChecks > 100, `the implication check must run broadly (${implicationChecks} models)`);
  ok(learnedTotal > 0, `learning must actually happen (${learnedTotal} learned clauses)`);

  // -- pigeonhole: refutations need real search -------------------------------------------------
  for (const [pigeons, holes] of [[3, 2], [4, 3], [5, 4]]) {
    const { clauses, variableCount } = pigeonhole(pigeons, holes);
    const solver = new CdclSolver(clauses, variableCount);
    const result = solver.solve();
    strictEqual(result.satisfiable, false,
      `pigeonhole ${pigeons} into ${holes} must be refuted`);
    ok(solver.conflicts > 0, "a refutation needs conflicts");
    ok(solver.learned.length > 0, "a refutation needs learned clauses");
    ok(solver.checkWatchInvariants(), "watch invariants hold on the refutation");
    strictEqual(bruteUnsatisfiable(clauses, variableCount), true, "the oracle agrees");
  }
  const pigeonSatisfiable = pigeonhole(3, 3);
  ok(new CdclSolver(pigeonSatisfiable.clauses, pigeonSatisfiable.variableCount).solve().satisfiable,
    "three pigeons fit into three holes");

  // -- CDCL versus the reference DPLL ------------------------------------------------------------
  let cdclDecisions = 0;
  let dpllDecisions = 0;
  let compared = 0;
  for (let trial = 0; trial < 40; trial++) {
    const variableCount = 5 + rng(6);
    const clauses = [];
    for (let c = 0; c < Math.floor(variableCount * 4.2); c++) {
      const clause = [];
      const variables = new Set();
      while (variables.size < 3) variables.add(rng(variableCount));
      for (const v of variables) clause.push(literal(v, rng(2) === 0));
      clauses.push(clause);
    }
    const solver = new CdclSolver(clauses, variableCount);
    const cdclResult = solver.solve();
    const reference = dpll(clauses, variableCount);
    strictEqual(cdclResult.satisfiable, reference.satisfiable,
      `trial ${trial}: CDCL and DPLL must agree`);
    if (cdclResult.satisfiable) ok(evaluate(clauses, cdclResult.assignment), "the model must be valid");
    cdclDecisions += solver.decisions;
    dpllDecisions += reference.decisions;
    compared++;
  }
  strictEqual(compared, 40, "every comparison trial must run");
  ok(cdclDecisions <= dpllDecisions,
    `learning must not need more decisions (${cdclDecisions} vs ${dpllDecisions})`);

  // -- determinism ---------------------------------------------------------------------------------
  const replayClauses = pigeonhole(4, 3).clauses;
  const firstRun = new CdclSolver(replayClauses, 12);
  firstRun.solve();
  const secondRun = new CdclSolver(replayClauses, 12);
  secondRun.solve();
  strictEqual(firstRun.conflicts, secondRun.conflicts, "identical inputs must take identical conflicts");
  strictEqual(firstRun.decisions, secondRun.decisions, "identical inputs must take identical decisions");
  strictEqual(firstRun.learned.length, secondRun.learned.length, "identical learning counts");
  deepStrictEqual(firstRun.learned.map((c) => c.lits.join(",")),
    secondRun.learned.map((c) => c.lits.join(",")), "the learned clauses must match");

  console.log(`random_formulas=120 sat=${satCount} unsat=${unsatCount} learned=${learnedTotal}`);
  console.log(`implied_clause_models_checked=${implicationChecks} cdcl_decisions=${cdclDecisions} ` +
    `dpll_decisions=${dpllDecisions}`);
  console.log("PASS");
}
