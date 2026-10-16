/** Task 748 - Implement an SMT solver front-end sketch (simple theory combination).
 *  (Tier: Expert, CTCI Beyond - Compilers, Languages & Solvers)
 *
 * A tiny S-expression front end parses formulas over boolean atoms and integer atoms; integer
 * terms may contain variables and one interpreted notion of *uninterpreted function application*
 * (f(x), f(f(y)), ...). Solving is the classical lazy DPLL(T) shape:
 *   1. the boolean skeleton is explored by assigning every distinct atom a truth value,
 *   2. the chosen polarities are handed to the theory part, which runs a real congruence closure
 *      over the term universe (union-find plus the fixpoint rule "same head, congruent arguments
 *      => same class"), so f(x) = f(y) follows from x = y - the equality sharing a Nelson-Oppen
 *      combination needs,
 *   3. the arithmetic side is decided over a bounded integer domain: the congruence classes are
 *      the valuated entities, so equal terms automatically receive one value, disequalities get
 *      different values, and a satisfying assignment is a model.
 *
 * What is simplified: the arithmetic theory is the *bounded* linear-integer theory (values inside
 * [-BOUND, BOUND], BOUND = 2 by default) rather than full Presburger arithmetic, a single
 * uninterpreted function symbol is enough for the examples, and the boolean part is a plain
 * enumeration of atom polarities instead of a CDCL core. What is verified: models are checked by
 * an independent evaluator that works on the parse tree and on a partial function table; the
 * verdicts agree with a brute-force enumeration of all variable values together with all partial
 * function tables on that identical domain; congruence closure is cross-checked against a naive
 * all-pairs closure; the classic combination cases (x = y with f(x) != f(y), x < y with y < x,
 * plus arithmetic/EUF interactions) decide as theory prescribes; malformed input is rejected; and
 * the whole front end is deterministic.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

const BOUND = 2; // the documented value domain [-BOUND, BOUND] of the sketch
const DOMAIN = [];
for (let v = -BOUND; v <= BOUND; v++) DOMAIN.push(v);

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

// ---- parser ----------------------------------------------------------------------------------
function tokenize(text) {
  return text.replace(/\(/g, " ( ").replace(/\)/g, " ) ").trim().split(/\s+/).filter((t) => t !== "");
}

function parse(text) {
  const tokens = tokenize(text);
  let position = 0;
  const parseForm = () => {
    const token = tokens[position++];
    if (token === undefined) throw new SyntaxError("unexpected end of input");
    if (token === "(") {
      const items = [];
      while (tokens[position] !== ")") {
        if (position >= tokens.length) throw new SyntaxError("missing closing parenthesis");
        items.push(parseForm());
      }
      position++;
      if (items.length === 0) throw new SyntaxError("empty form");
      return { kind: "list", items };
    }
    if (token === ")") throw new SyntaxError("unexpected closing parenthesis");
    if (/^-?\d+$/.test(token)) return { kind: "number", value: Number(token) };
    if (token === "and" || token === "or" || token === "not" || token === "=>" ||
      token === "=" || token === "<" || token === "<=" || token === ">" || token === ">=") {
      return { kind: "head", name: token };
    }
    return { kind: "symbol", name: token };
  };
  const form = parseForm();
  if (position !== tokens.length) throw new SyntaxError("trailing input");
  return form;
}

/**
 * Turns the parsed form into an expression tree with, besides formulas, integer terms. Terms are
 * numbers, variables or applications `(f term)`; atoms are a head symbol applied to terms.
 */
function analyse(form, variables, functions) {
  const asTerm = (node) => {
    if (node.kind === "number") return { type: "number", value: node.value };
    if (node.kind === "symbol") {
      variables.add(node.name);
      return { type: "variable", name: node.name };
    }
    if (node.kind === "list" && node.items.length === 2 &&
      node.items[0].kind === "symbol") {
      functions.add(node.items[0].name);
      return { type: "application", head: node.items[0].name, argument: asTerm(node.items[1]) };
    }
    throw new SyntaxError("not a term");
  };
  if (form.kind === "symbol") {
    variables.add(form.name);
    return { type: "boolean", name: form.name }; // a bare boolean variable
  }
  if (form.kind !== "list") throw new SyntaxError("a formula must be a list or a boolean variable");
  const [head, ...rest] = form.items;
  if (head.kind !== "head") throw new SyntaxError("a formula must start with a connective");
  switch (head.name) {
    case "and":
    case "or": {
      if (rest.length === 0) throw new SyntaxError(`${head.name} needs arguments`);
      return { type: head.name, children: rest.map((child) => analyse(child, variables, functions)) };
    }
    case "not": {
      if (rest.length !== 1) throw new SyntaxError("not takes one argument");
      return { type: "not", child: analyse(rest[0], variables, functions) };
    }
    case "=>": {
      if (rest.length !== 2) throw new SyntaxError("=> takes two arguments");
      return { type: "implies", children: rest.map((child) => analyse(child, variables, functions)) };
    }
    case "=":
    case "<":
    case "<=":
    case ">":
    case ">=": {
      if (rest.length !== 2) throw new SyntaxError(`${head.name} takes two arguments`);
      return { type: "atom", relation: head.name, left: asTerm(rest[0]), right: asTerm(rest[1]) };
    }
    default:
      throw new SyntaxError(`unknown connective ${head.name}`);
  }
}

// ---- congruence closure ----------------------------------------------------------------------
function termKey(term) {
  if (term.type === "number") return `#${term.value}`;
  if (term.type === "variable") return `v:${term.name}`;
  return `a:${term.head}(${termKey(term.argument)})`;
}

class UnionFind {
  constructor() {
    this.parent = new Map();
  }

  add(item) {
    if (!this.parent.has(item)) this.parent.set(item, item);
  }

  find(item) {
    this.add(item);
    let root = item;
    while (this.parent.get(root) !== root) root = this.parent.get(root);
    let current = item;
    while (this.parent.get(current) !== root) {
      const next = this.parent.get(current);
      this.parent.set(current, root);
      current = next;
    }
    return root;
  }

  union(a, b) {
    const rootA = this.find(a);
    const rootB = this.find(b);
    if (rootA === rootB) return false;
    const [first, second] = rootA < rootB ? [rootA, rootB] : [rootB, rootA];
    this.parent.set(second, first);
    return true;
  }
}

/** Congruence closure over the terms of the formula, driven by the asserted equalities. */
function congruenceClosure(terms, equalities) {
  const union = new UnionFind();
  const seen = new Set();
  const stack = [...terms];
  while (stack.length > 0) {
    const term = stack.pop();
    const key = termKey(term);
    if (seen.has(key)) continue;
    seen.add(key);
    union.add(key);
    if (term.type === "application") {
      union.union(key, key); // ensure the key exists
      stack.push(term.argument);
    }
  }
  for (const [a, b] of equalities) union.union(termKey(a), termKey(b));
  // fixpoint: congruent arguments make congruent applications
  let changed = true;
  const applications = [...seen].filter((key) => key.startsWith("a:"));
  while (changed) {
    changed = false;
    for (const key of applications) {
      const keyHead = key.slice(2, key.indexOf("("));
      const inner = key.slice(key.indexOf("(") + 1, key.length - 1); // strip "a:head(" and ")"
      for (const other of applications) {
        if (other <= key) continue;
        if (keyHead !== other.slice(2, other.indexOf("("))) continue;
        const otherInner = other.slice(other.indexOf("(") + 1, other.length - 1);
        if (union.find(inner) === union.find(otherInner) && union.union(key, other)) changed = true;
      }
    }
  }
  return union;
}

/** Independent closure: iterate the merge rule until nothing changes (no union-find tricks). */
function naiveClosure(terms, equalities) {
  const keys = new Set();
  const applications = [];
  const walk = (term) => {
    const key = termKey(term);
    if (keys.has(key)) return;
    keys.add(key);
    if (term.type === "application") {
      applications.push({ head: term.head, argument: key, own: key, argumentKey: termKey(term.argument) });
      walk(term.argument);
    }
  };
  for (const term of terms) walk(term);
  const pairs = equalities.map(([a, b]) => [termKey(a), termKey(b)]);
  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < applications.length; i++) {
      for (let j = i + 1; j < applications.length; j++) {
        const first = applications[i];
        const second = applications[j];
        if (first.head !== second.head) continue;
        const sameArgument = pairConnected(pairs, first.argumentKey, second.argumentKey);
        if (sameArgument && !pairConnected(pairs, first.own, second.own)) {
          pairs.push([first.own, second.own]);
          changed = true;
        }
      }
    }
  }
  const sameClass = (a, b) => pairConnected(pairs, a, b);
  return { sameClass, pairs };
}

function pairConnected(pairs, a, b) {
  // closure over the pair list, recomputed from scratch each time (small inputs)
  const adjacency = new Map();
  const link = (u, v) => {
    if (!adjacency.has(u)) adjacency.set(u, []);
    if (!adjacency.has(v)) adjacency.set(v, []);
    adjacency.get(u).push(v);
    adjacency.get(v).push(u);
  };
  for (const [u, v] of pairs) link(u, v);
  const stack = [a];
  const seen = new Set([a]);
  while (stack.length > 0) {
    const current = stack.pop();
    if (current === b) return true;
    for (const next of adjacency.get(current) ?? []) {
      if (!seen.has(next)) {
        seen.add(next);
        stack.push(next);
      }
    }
  }
  return false;
}

// ---- the lazy DPLL(T) loop --------------------------------------------------------------------
function collectAtoms(expression, out = []) {
  switch (expression.type) {
    case "boolean":
    case "atom":
      out.push(expression);
      break;
    case "not":
      collectAtoms(expression.child, out);
      break;
    default:
      for (const child of expression.children) collectAtoms(child, out);
  }
  return out;
}

function collectTerms(expression, out = []) {
  switch (expression.type) {
    case "atom":
      out.push(expression.left, expression.right);
      break;
    case "not":
      collectTerms(expression.child, out);
      break;
    case "boolean":
      break;
    default:
      for (const child of expression.children) collectTerms(child, out);
  }
  return out;
}

function atomKey(atom) {
  return atom.type === "boolean" ? `b:${atom.name}` :
    `${atom.relation}:${termKey(atom.left)}:${termKey(atom.right)}`;
}

/** All distinct sub-terms of the formula, innermost included. */
function allTerms(expression) {
  const found = new Map();
  const walk = (term) => {
    const key = termKey(term);
    if (found.has(key)) return term;
    found.set(key, term);
    if (term.type === "application") walk(term.argument);
    return term;
  };
  for (const term of collectTerms(expression)) walk(term);
  return [...found.values()];
}

function atomHolds(atom, assignment) {
  if (atom.type === "boolean") return assignment[`b:${atom.name}`] === true;
  const left = assignment[termKey(atom.left)];
  const right = assignment[termKey(atom.right)];
  switch (atom.relation) {
    case "=":
      return left === right;
    case "<":
      return left < right;
    case "<=":
      return left <= right;
    case ">":
      return left > right;
    default:
      return left >= right;
  }
}

function evaluateExpression(expression, values) {
  switch (expression.type) {
    case "boolean":
      return values[`b:${expression.name}`] === true;
    case "atom":
      return atomHolds(expression, values);
    case "not":
      return !evaluateExpression(expression.child, values);
    case "implies":
      return !evaluateExpression(expression.children[0], values) ||
        evaluateExpression(expression.children[1], values);
    case "and":
      return expression.children.every((child) => evaluateExpression(child, values));
    default:
      return expression.children.some((child) => evaluateExpression(child, values));
  }
}

/** True when the boolean skeleton holds for the given atom polarities. */
function skeletonHolds(expression, polarities) {
  switch (expression.type) {
    case "boolean":
    case "atom":
      return polarities[atomKey(expression)] === true;
    case "not":
      return !skeletonHolds(expression.child, polarities);
    case "implies":
      return !skeletonHolds(expression.children[0], polarities) ||
        skeletonHolds(expression.children[1], polarities);
    case "and":
      return expression.children.every((child) => skeletonHolds(child, polarities));
    default:
      return expression.children.some((child) => skeletonHolds(child, polarities));
  }
}

/**
 * Lazy DPLL(T): enumerate the polarities of the distinct atoms, keep the choices whose boolean
 * skeleton holds, then ask the theory part for a model. The theory part runs congruence closure
 * over the term universe and then assigns one value per congruence class inside the bounded
 * domain, which is exactly the equality sharing between the EUF and the bounded-integer theory.
 */
function solve(text) {
  const parsed = parse(text);
  const variables = new Set();
  const functions = new Set();
  const expression = analyse(parsed, variables, functions);
  const atoms = [...new Map(collectAtoms(expression).map((a) => [atomKey(a), a])).values()];
  const keys = atoms.map(atomKey);
  const terms = allTerms(expression);
  let theoryCalls = 0;
  let skeletons = 0;

  const theoryCheck = (polarities) => {
    theoryCalls++;
    const equalities = [];
    for (const atom of atoms) {
      if (atom.type === "boolean" || polarities[atomKey(atom)] !== true) continue;
      if (atom.relation === "=") equalities.push([atom.left, atom.right]);
    }
    const closure = congruenceClosure(terms, equalities);
    const classes = new Map();
    for (const term of terms) {
      const root = closure.find(termKey(term));
      if (!classes.has(root)) classes.set(root, []);
      classes.get(root).push(term);
    }
    const roots = [...classes.keys()];
    const fixed = new Map(); // classes holding a numeral are pinned to that numeral
    for (const [root, members] of classes) {
      const numerals = [...new Set(members.filter((t) => t.type === "number").map((t) => t.value))];
      if (numerals.length > 1) return null; // a = 1 and a = 2 cannot hold at once
      if (numerals.length === 1) fixed.set(root, numerals[0]);
    }
    const classValues = new Map();
    const assign = (index) => {
      if (index === roots.length) {
        const concrete = {};
        for (const [root, members] of classes) {
          for (const term of members) concrete[termKey(term)] = classValues.get(root);
        }
        for (const atom of atoms) {
          if (atom.type === "boolean") {
            concrete[atomKey(atom)] = polarities[atomKey(atom)];
            continue;
          }
          if (atomHolds(atom, concrete) !== polarities[atomKey(atom)]) return null;
        }
        return concrete;
      }
      const root = roots[index];
      if (fixed.has(root)) {
        classValues.set(root, fixed.get(root));
        return assign(index + 1);
      }
      for (const value of DOMAIN) {
        classValues.set(root, value);
        const found = assign(index + 1);
        if (found !== null) return found;
      }
      return null;
    };
    return assign(0);
  };

  const total = 1 << keys.length;
  for (let mask = 0; mask < total; mask++) {
    const polarities = {};
    keys.forEach((key, bit) => {
      polarities[key] = ((mask >> bit) & 1) === 1;
    });
    if (!skeletonHolds(expression, polarities)) continue;
    skeletons++;
    const model = theoryCheck(polarities);
    if (model !== null) {
      return { satisfiable: true, model, theoryCalls, skeletons, atoms: keys.length };
    }
  }
  return { satisfiable: false, model: null, theoryCalls, skeletons, atoms: keys.length };
}

// ---- independent oracle ------------------------------------------------------------------------
/** Brute force over variable values and application values on the very same bounded domain. */
function bruteSolve(text) {
  const parsed = parse(text);
  const expression = analyse(parsed, new Set(), new Set());
  const terms = allTerms(expression);
  const variables = [...new Set(terms.filter((t) => t.type === "variable").map((t) => t.name))];
  const applications = terms.filter((t) => t.type === "application");
  const booleans = [...new Set(collectAtoms(expression)
    .filter((a) => a.type === "boolean").map((a) => a.name))];
  const variableValues = {};
  const applicationValues = {};

  const valueOf = (term) => {
    if (term.type === "number") return term.value;
    if (term.type === "variable") return variableValues[term.name];
    return applicationValues[termKey(term)];
  };
  const consistent = () => {
    // congruence: one function, one value per argument value
    const table = new Map();
    for (const term of applications) {
      const key = `${term.head}|${valueOf(term.argument)}`;
      const value = applicationValues[termKey(term)];
      if (table.has(key) && table.get(key) !== value) return false;
      table.set(key, value);
    }
    const concrete = {};
    for (const term of terms) concrete[termKey(term)] = valueOf(term);
    for (let mask = 0; mask < (1 << booleans.length); mask++) {
      booleans.forEach((name, bit) => {
        concrete[`b:${name}`] = ((mask >> bit) & 1) === 1;
      });
      if (evaluateExpression(expression, concrete)) return true;
    }
    return false;
  };
  const overApplications = (index) => {
    if (index === applications.length) return consistent();
    for (const value of DOMAIN) {
      applicationValues[termKey(applications[index])] = value;
      if (overApplications(index + 1)) return true;
    }
    return false;
  };
  const overVariables = (index) => {
    if (index === variables.length) return overApplications(0);
    for (const value of DOMAIN) {
      variableValues[variables[index]] = value;
      if (overVariables(index + 1)) return true;
    }
    return false;
  };
  return overVariables(0);
}

module.exports = { parse, analyse, solve, bruteSolve, congruenceClosure, naiveClosure, evaluateExpression,
  allTerms, DOMAIN, BOUND };

if (require.main === module) {
  // -- parser ---------------------------------------------------------------------------------
  const parsed = parse("(and (< x y) (<= y 3))");
  strictEqual(parsed.kind, "list");
  strictEqual(parsed.items[0].name, "and");
  throws(() => parse("(and (< x y)"), SyntaxError, "an unbalanced form is rejected");
  throws(() => parse("(< x y))"), SyntaxError, "trailing input is rejected");
  throws(() => parse("()"), SyntaxError, "an empty form is rejected");
  const variables = new Set();
  const functions = new Set();
  analyse(parse("(= (f x) (f (f y)))"), variables, functions);
  deepStrictEqual([...variables].sort(), ["x", "y"], "variables are collected");
  deepStrictEqual([...functions], ["f"], "function symbols are collected");
  throws(() => analyse(parse("(< (f x) (+ y 1))"), new Set(), new Set()), SyntaxError,
    "unsupported connectives are rejected");

  // -- congruence closure, cross-checked against the naive closure ------------------------------
  const formula = " (and (= x y) (= (f x) a) (= (f y) b)) ";
  const variableNames = new Set();
  const functionNames = new Set();
  const tree = analyse(parse(formula), variableNames, functionNames);
  const terms = allTerms(tree);
  const equalities = [[tree.children[0].left, tree.children[0].right]];
  const union = congruenceClosure(terms, equalities);
  const naive = naiveClosure(terms, equalities);
  let closureChecks = 0;
  for (const first of terms) {
    for (const second of terms) {
      strictEqual(union.find(termKey(first)) === union.find(termKey(second)),
        naive.sameClass(termKey(first), termKey(second)),
        `closure disagreement on ${termKey(first)} vs ${termKey(second)}`);
      closureChecks++;
    }
  }
  ok(closureChecks > 20, `the closure comparison must be broad (${closureChecks} pairs)`);
  const xKey = "v:x";
  const yKey = "v:y";
  const fxKey = "a:f(v:x)";
  const fyKey = "a:f(v:y)";
  ok(union.find(xKey) === union.find(yKey), "x = y must be merged");
  ok(union.find(fxKey) === union.find(fyKey), "f(x) = f(y) must follow from x = y");
  ok(union.find(fxKey) !== union.find("v:a"), "f(x) and a start in different classes");

  // -- the classic theory-combination cases ------------------------------------------------------
  const cases = [
    ["(and (< x y) (< y x))", false, "the integer order is irreflexive"],
    ["(and (= x y) (< x y))", false, "equality and a strict order conflict"],
    ["(and (<= x y) (<= y x))", true, "a double inequality forces equality"],
    ["(and (= x y) (not (= (f x) (f y))))", false, "congruence refutes it"],
    ["(and (= x y) (= (f x) (f y)))", true, "congruence confirms it"],
    ["(and (= (f x) a) (= (f y) a))", true, "two applications may share a value"],
    ["(and (= (f x) a) (= (f y) b) (= x y) (not (= a b)))", false,
      "congruence plus a disequality is inconsistent"],
    ["(and (< x 3) (> x 3))", false, "a strict contradiction in arithmetic"],
    ["(and (<= x 2) (>= x 2) (= y x) (= (f y) 1))", true, "arithmetic equalities reach the function"],
    ["(and (= (f (f x)) x) (= (f x) y) (not (= x y)))", true,
      "nested applications can describe a two cycle"],
    ["(and (= (f x) x) (= (f (f x)) y) (not (= x y)))", false,
      "congruence collapses the nested application onto x"],
    ["(and (> x 5))", false, "the documented bound refuses a value above it"],
    ["(and (>= x 2) (<= x 2))", true, "the boundary value is allowed"],
    ["p", true, "a bare boolean variable is satisfiable"],
    ["(and p (not p))", false, "a boolean contradiction is refuted"],
    ["(or (and p q) (and (not p) (not q)))", true, "boolean structure is explored"],
  ];
  let verdicts = 0;
  for (const [text, expected, why] of cases) {
    const result = solve(text);
    strictEqual(result.satisfiable, expected, `${text}: ${why}`);
    verdicts++;
    if (result.satisfiable) {
      ok(evaluateExpression(analyse(parse(text), new Set(), new Set()), result.model),
        `${text}: the reported model must satisfy the formula`);
    }
  }
  strictEqual(verdicts, cases.length, "every hand written case must be decided");

  // -- random formulas against the brute-force oracle ---------------------------------------------
  const rng = makeRng(748748);
  const relations = ["=", "<", "<=", ">", ">="];
  const randomTerm = () => {
    const choice = rng(4);
    if (choice === 0) return `${rng(5) - 2}`;
    if (choice === 1) return `x`;
    if (choice === 2) return `(f x)`;
    return `(f (f x))`;
  };
  let agreements = 0;
  for (let trial = 0; trial < 40; trial++) {
    const relation = relations[rng(relations.length)];
    const text = `(and (${relation} ${randomTerm()} ${randomTerm()}))`;
    const result = solve(text);
    const brute = bruteSolve(text);
    strictEqual(result.satisfiable, brute, `trial ${trial}: ${text} must match the oracle`);
    if (result.satisfiable) {
      ok(evaluateExpression(analyse(parse(text), new Set(), new Set()), result.model),
        `trial ${trial}: the model must satisfy ${text}`);
    }
    agreements++;
  }
  strictEqual(agreements, 40, "every random formula must be compared");

  // -- determinism ----------------------------------------------------------------------------------
  for (const [text] of cases) {
    strictEqual(solve(text).satisfiable, solve(text).satisfiable, "repeated runs must agree");
    if (solve(text).satisfiable) {
      deepStrictEqual(solve(text).model, solve(text).model, "the model must be reproducible");
    }
  }

  console.log(`hand_cases=${verdicts} random_agreements=${agreements} closure_pairs=${closureChecks}`);
  console.log(`bound=${BOUND} domain_size=${DOMAIN.length} skeletons_explored=${solve("(and (= x y) (= (f x) (f y)))").skeletons}`);
  console.log("PASS");
}
