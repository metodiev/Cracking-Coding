/** Task 803 - Build a small SQL engine with parser, planner, and executor supporting joins and aggregates. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A real (small) SQL front end and back end: a hand written tokenizer and recursive-descent parser
 * for SELECT with INNER/LEFT JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT and the aggregates
 * COUNT/SUM/AVG/MIN/MAX; a rule-based planner that resolves every column reference against the
 * scope (so unknown and ambiguous columns are plan-time errors), folds constants, pushes WHERE
 * conjuncts and single-side ON conjuncts below the join, prunes the columns each scan materialises,
 * and chooses a hash join for equi-joins and a nested loop otherwise; and a volcano-style iterator
 * executor (open/next) whose per-operator row counters expose the effect of every planning decision.
 *
 * What is simplified: no indexes or sampled statistics (only small exact tables, and estimates are
 * used for reporting, never for correctness), no disk pages, WHERE uses two-valued logic (a missing
 * value simply fails a predicate) instead of SQL's three-valued logic, grouping is in-memory hashing
 * and ORDER BY may be solved with an internal hidden projection column that is stripped again, and
 * the supported grammar is exactly the one above. Sizes are typed parameters (row counts, key
 * counts), so the same engine runs the "at scale" numbers.
 *
 * Invariants verified:
 *   - result equivalence: every query is executed twice, once through the planner/executor and once
 *     through an independent brute-force evaluator that forms the full cross product, folds
 *     aggregates straight over the group's rows and never touches a plan (exact order when ORDER BY
 *     is present, multiset comparison otherwise); 25 randomly generated queries are included;
 *   - hand-computed golden results for joins, GROUP BY, HAVING, LEFT JOIN, DISTINCT and LIMIT;
 *   - the planner really optimises: the filter lands inside the join's input, the equi-join becomes a
 *     hash join, the non-equi join falls back to the nested loop, the join emits fewer rows than the
 *     cross product, constants are folded, and column pruning materialises fewer values than a full
 *     scan while SELECT * materialises more;
 *   - SQL edge semantics: COUNT(*) over an empty table is 0, SUM over no rows is NULL, GROUP BY over
 *     an empty table yields no groups, LIMIT 0 and LIMIT past the end behave, IS NULL finds exactly
 *     the unmatched LEFT JOIN rows, and integer division by zero yields NULL;
 *   - errors instead of wrong answers: unknown table/column, ambiguous column, aggregate in WHERE,
 *     non-grouped column, trailing comma, unterminated literal and truncated expression all raise a
 *     descriptive SqlError;
 *   - determinism: identical plan text, row counts and results across repeated compilation.
 */
"use strict";

const assert = require("node:assert");

const KEYWORDS = new Set(["select", "distinct", "from", "as", "join", "inner", "left", "on", "where",
  "group", "by", "having", "order", "asc", "desc", "limit", "and", "or", "not", "is", "null",
  "count", "sum", "avg", "min", "max"]);
const AGGREGATES = new Set(["count", "sum", "avg", "min", "max"]);
const COMPARISONS = new Set(["=", "<>", "!=", "<", ">", "<=", ">="]);

const SCHEMA = {
  users: ["id", "name", "city", "age"],
  orders: ["id", "user_id", "amount"],
  cities: ["name", "country", "population"],
  empty: ["id", "amount"],
};

class SqlError extends Error {}

// --------------------------------------------------------------------------- tokenizer

function tokenize(sql) {
  const tokens = [];
  let i = 0;
  while (i < sql.length) {
    const ch = sql[i];
    if (/\s/.test(ch)) { i++; continue; }
    if ("(),*.".includes(ch)) { tokens.push({ type: "punct", value: ch }); i++; continue; }
    if ("<>=!".includes(ch)) {
      const two = sql.slice(i, i + 2);
      if (["<>", "!=", "<=", ">="].includes(two)) { tokens.push({ type: "op", value: two }); i += 2; continue; }
      tokens.push({ type: "op", value: ch }); i++; continue;
    }
    if ("+-/".includes(ch)) { tokens.push({ type: "op", value: ch }); i++; continue; }
    if (ch === "'") {
      const end = sql.indexOf("'", i + 1);
      if (end < 0) throw new SqlError("unterminated string literal");
      tokens.push({ type: "str", value: sql.slice(i + 1, end) });
      i = end + 1;
      continue;
    }
    if (/[0-9]/.test(ch)) {
      let j = i;
      while (j < sql.length && /[0-9]/.test(sql[j])) j++;
      tokens.push({ type: "num", value: Number(sql.slice(i, j)) });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      let j = i;
      while (j < sql.length && /[A-Za-z0-9_]/.test(sql[j])) j++;
      const text = sql.slice(i, j);
      tokens.push(KEYWORDS.has(text.toLowerCase()) ? { type: "kw", value: text.toLowerCase() } : { type: "id", value: text });
      i = j;
      continue;
    }
    throw new SqlError(`unexpected character ${ch}`);
  }
  tokens.push({ type: "eof", value: null });
  return tokens;
}

// --------------------------------------------------------------------------- parser

class Parser {
  constructor(sql) { this.tokens = tokenize(sql); this.pos = 0; }

  peek(offset = 0) { return this.tokens[this.pos + offset]; }

  is(type, value) { const token = this.peek(); return token.type === type && (value === undefined || token.value === value); }

  take() { return this.tokens[this.pos++]; }

  expect(type, value) {
    if (!this.is(type, value)) throw new SqlError(`expected ${value ?? type} but found ${JSON.stringify(this.peek().value)}`);
    return this.take();
  }

  parseSelect() {
    this.expect("kw", "select");
    const distinct = this.is("kw", "distinct") ? (this.take(), true) : false;
    const items = [this.parseSelectItem()];
    while (this.is("punct", ",")) { this.take(); items.push(this.parseSelectItem()); }
    this.expect("kw", "from");
    const from = this.parseTableRef();
    const joins = [];
    while (this.is("kw", "join") || this.is("kw", "inner") || this.is("kw", "left")) {
      let type = "inner";
      if (this.is("kw", "left")) { type = "left"; this.take(); }
      else if (this.is("kw", "inner")) this.take();
      this.expect("kw", "join");
      const table = this.parseTableRef();
      this.expect("kw", "on");
      joins.push({ type, table, on: this.parseExpr() });
    }
    const where = this.is("kw", "where") ? (this.take(), this.parseExpr()) : null;
    const groupBy = [];
    if (this.is("kw", "group")) {
      this.take();
      this.expect("kw", "by");
      groupBy.push(this.parseExpr());
      while (this.is("punct", ",")) { this.take(); groupBy.push(this.parseExpr()); }
    }
    const having = this.is("kw", "having") ? (this.take(), this.parseExpr()) : null;
    const orderBy = [];
    if (this.is("kw", "order")) {
      this.take();
      this.expect("kw", "by");
      const parseKey = () => {
        const expr = this.parseExpr();
        let dir = "asc";
        if (this.is("kw", "asc") || this.is("kw", "desc")) dir = this.take().value;
        return { expr, dir };
      };
      orderBy.push(parseKey());
      while (this.is("punct", ",")) { this.take(); orderBy.push(parseKey()); }
    }
    const limit = this.is("kw", "limit") ? (this.take(), this.expect("num").value) : null;
    this.expect("eof");
    return { kind: "select", distinct, items, from, joins, where, groupBy, having, orderBy, limit };
  }

  parseSelectItem() {
    if (this.is("punct", "*")) { this.take(); return { star: true, expr: null, alias: null }; }
    const expr = this.parseExpr();
    const alias = this.is("kw", "as") ? (this.take(), this.expect("id").value) : null;
    return { star: false, expr, alias };
  }

  parseTableRef() {
    const table = this.expect("id").value;
    let alias = table;
    if (this.is("kw", "as")) { this.take(); alias = this.expect("id").value; }
    else if (this.is("id")) alias = this.take().value;
    return { table, alias };
  }

  parseExpr() { return this.parseOr(); }

  parseOr() {
    let left = this.parseAnd();
    while (this.is("kw", "or")) { this.take(); left = { kind: "bin", op: "or", left, right: this.parseAnd() }; }
    return left;
  }

  parseAnd() {
    let left = this.parseNot();
    while (this.is("kw", "and")) { this.take(); left = { kind: "bin", op: "and", left, right: this.parseNot() }; }
    return left;
  }

  parseNot() {
    if (this.is("kw", "not")) { this.take(); return { kind: "un", op: "not", operand: this.parseNot() }; }
    return this.parseComparison();
  }

  parseComparison() {
    const left = this.parseAdditive();
    if (this.is("kw", "is")) {
      this.take();
      const negated = this.is("kw", "not") ? (this.take(), true) : false;
      this.expect("kw", "null");
      return { kind: "isnull", negated, operand: left };
    }
    const token = this.peek();
    if (token.type === "op" && COMPARISONS.has(token.value)) {
      this.take();
      return { kind: "bin", op: token.value === "!=" ? "<>" : token.value, left, right: this.parseAdditive() };
    }
    return left;
  }

  parseAdditive() {
    let left = this.parseMultiplicative();
    while (this.is("op", "+") || this.is("op", "-")) {
      const op = this.take().value;
      left = { kind: "bin", op, left, right: this.parseMultiplicative() };
    }
    return left;
  }

  parseMultiplicative() {
    let left = this.parseUnary();
    while (this.is("op", "*") || this.is("op", "/")) {
      const op = this.take().value;
      left = { kind: "bin", op, left, right: this.parseUnary() };
    }
    return left;
  }

  parseUnary() {
    if (this.is("op", "-")) { this.take(); return { kind: "un", op: "-", operand: this.parseUnary() }; }
    return this.parsePrimary();
  }

  parsePrimary() {
    const token = this.peek();
    if (token.type === "num") { this.take(); return { kind: "num", value: token.value }; }
    if (token.type === "str") { this.take(); return { kind: "str", value: token.value }; }
    if (token.type === "punct" && token.value === "(") {
      this.take();
      const inner = this.parseExpr();
      this.expect("punct", ")");
      return inner;
    }
    if (token.type === "id" || token.type === "kw") {
      const name = this.take().value;
      if (AGGREGATES.has(String(name).toLowerCase()) && this.is("punct", "(")) {
        this.take();
        const arg = this.is("punct", "*") ? (this.take(), { kind: "star" }) : this.parseExpr();
        this.expect("punct", ")");
        return { kind: "agg", func: String(name).toLowerCase(), arg };
      }
      if (this.is("punct", ".")) {
        this.take();
        return { kind: "ref", qualifier: name, name: this.expect("id").value };
      }
      return { kind: "ref", qualifier: null, name };
    }
    throw new SqlError(`unexpected token ${JSON.stringify(token.value)}`);
  }
}

function parse(sql) { return new Parser(sql).parseSelect(); }

// --------------------------------------------------------------------------- expressions

function exprText(expr) {
  switch (expr.kind) {
    case "num": return String(expr.value);
    case "str": return `'${expr.value}'`;
    case "alias": case "ref": return expr.kind === "alias" || expr.qualifier === null || expr.key === expr.name ? expr.name : `${expr.qualifier}.${expr.name}`;
    case "star": return "*";
    case "agg": return `${expr.func}(${exprText(expr.arg)})`;
    case "isnull": return `${exprText(expr.operand)} is ${expr.negated ? "not " : ""}null`;
    case "un": return `${expr.op}${exprText(expr.operand)}`;
    default: return `(${exprText(expr.left)} ${expr.op} ${exprText(expr.right)})`;
  }
}

function containsAggregate(expr) {
  if (expr === null || expr === undefined) return false;
  if (expr.kind === "agg") return true;
  if (expr.kind === "bin") return containsAggregate(expr.left) || containsAggregate(expr.right);
  if (expr.kind === "un" || expr.kind === "isnull") return containsAggregate(expr.operand);
  return false;
}

function refsOf(expr, out = []) {
  if (expr === null || expr === undefined) return out;
  if (expr.kind === "ref") out.push(expr);
  else if (expr.kind === "bin") { refsOf(expr.left, out); refsOf(expr.right, out); }
  else if (expr.kind === "un" || expr.kind === "isnull") refsOf(expr.operand, out);
  else if (expr.kind === "agg") refsOf(expr.arg, out);
  return out;
}

/** Resolve every bare column reference against the scope; ambiguity and unknown columns fail here. */
function resolveRefs(expr, scope) {
  if (expr === null || expr === undefined) return null;
  switch (expr.kind) {
    case "num": case "str": case "star": case "alias": return { ...expr };
    case "ref": {
      if (expr.qualifier !== null) {
        const entry = scope.find((item) => item.alias === expr.qualifier);
        if (entry === undefined) throw new SqlError(`unknown table alias ${expr.qualifier}`);
        if (!SCHEMA[entry.table].includes(expr.name)) throw new SqlError(`unknown column ${expr.name} in table ${entry.table}`);
        return { kind: "ref", qualifier: expr.qualifier, name: expr.name, key: `${expr.qualifier}.${expr.name}` };
      }
      const matches = scope.filter((item) => SCHEMA[item.table].includes(expr.name));
      if (matches.length === 0) throw new SqlError(`unknown column ${expr.name}`);
      if (matches.length > 1) throw new SqlError(`ambiguous column ${expr.name}`);
      return { kind: "ref", qualifier: matches[0].alias, name: expr.name, key: `${matches[0].alias}.${expr.name}` };
    }
    case "agg": return { ...expr, arg: resolveRefs(expr.arg, scope) };
    case "isnull": return { ...expr, operand: resolveRefs(expr.operand, scope) };
    case "un": return { ...expr, operand: resolveRefs(expr.operand, scope) };
    default: return { ...expr, left: resolveRefs(expr.left, scope), right: resolveRefs(expr.right, scope) };
  }
}

function applyBinary(op, left, right) {
  if (left === undefined || right === undefined) return undefined;
  switch (op) {
    case "and": return truthy(left) && truthy(right);
    case "or": return truthy(left) || truthy(right);
    case "+": return left + right;
    case "-": return left - right;
    case "*": return left * right;
    case "/": return right === 0 ? undefined : Math.trunc(left / right);
    case "=": return left === right;
    case "<>": return left !== right;
    case "<": return left < right;
    case ">": return left > right;
    case "<=": return left <= right;
    case ">=": return left >= right;
    default: throw new SqlError(`unsupported operator ${op}`);
  }
}

const truthy = (value) => value === true || (typeof value === "number" && value !== 0);

function evalExpr(expr, row) {
  switch (expr.kind) {
    case "num": case "str": return expr.value;
    case "alias": return row[expr.name];
    case "ref": return row[expr.key !== undefined ? expr.key : expr.name];
    case "bin": {
      const left = evalExpr(expr.left, row);
      if (expr.op === "and" && !truthy(left)) return false;
      if (expr.op === "or" && truthy(left)) return true;
      return applyBinary(expr.op, left, evalExpr(expr.right, row));
    }
    case "un": return expr.op === "not" ? !truthy(evalExpr(expr.operand, row)) : -evalExpr(expr.operand, row);
    case "isnull": {
      const value = evalExpr(expr.operand, row);
      return expr.negated ? value !== undefined : value === undefined;
    }
    default: throw new SqlError(`cannot evaluate ${expr.kind} row by row`);
  }
}

function foldConstants(expr) {
  if (expr === null || expr === undefined) return expr;
  if (expr.kind === "bin") {
    const left = foldConstants(expr.left);
    const right = foldConstants(expr.right);
    if ((left.kind === "num" || left.kind === "str") && (right.kind === "num" || right.kind === "str")) {
      const value = applyBinary(expr.op, left.value, right.value);
      if (typeof value === "number") return { kind: "num", value };
      if (typeof value === "string") return { kind: "str", value };
    }
    return { ...expr, left, right };
  }
  if (expr.kind === "un") return { ...expr, operand: foldConstants(expr.operand) };
  if (expr.kind === "isnull") return { ...expr, operand: foldConstants(expr.operand) };
  if (expr.kind === "agg") return { ...expr, arg: foldConstants(expr.arg) };
  return expr;
}

/** Split an expression into AND conjuncts; the planner pushes each one as far down as it can. */
function conjuncts(expr, out = []) {
  if (expr === null || expr === undefined) return out;
  if (expr.kind === "bin" && expr.op === "and") { conjuncts(expr.left, out); conjuncts(expr.right, out); return out; }
  out.push(expr);
  return out;
}

const combineAnd = (parts) => parts.reduce((left, right) => (left === null ? right : { kind: "bin", op: "and", left, right }), null);

const refNode = (key) => ({ kind: "ref", qualifier: null, name: key, key });
const rowKey = (row) => JSON.stringify(Object.entries(row).map(([k, v]) => [k, v === undefined ? null : v]).sort());

// --------------------------------------------------------------------------- operators

class ScanOp {
  constructor(table, alias, columns = null) {
    this.kind = "scan";
    this.table = table;
    this.alias = alias;
    this.columns = columns;
    this.rows = [];
    this.stats = { rowsOut: 0, columnsLoaded: 0 };
  }

  open(database) {
    const columns = this.columns === null ? [...SCHEMA[this.table]] : this.columns;
    this.rows = database[this.table].map((row) => {
      const projected = {};
      for (const column of columns) projected[`${this.alias}.${column}`] = row[column];
      return projected;
    });
    this.pos = 0;
    this.stats.columnsLoaded = this.rows.length * columns.length;
  }

  next() {
    if (this.pos >= this.rows.length) return null;
    this.stats.rowsOut++;
    return this.rows[this.pos++];
  }
}

class FilterOp {
  constructor(child, predicate) { this.kind = "filter"; this.child = child; this.predicate = predicate; this.stats = { rowsOut: 0 }; }
  open(database) { this.child.open(database); }
  next() {
    for (;;) {
      const row = this.child.next();
      if (row === null) return null;
      if (truthy(evalExpr(this.predicate, row))) { this.stats.rowsOut++; return row; }
    }
  }
}

class ProjectOp {
  constructor(child, items) { this.kind = "project"; this.child = child; this.items = items; this.stats = { rowsOut: 0 }; }
  open(database) { this.child.open(database); }
  next() {
    const row = this.child.next();
    if (row === null) return null;
    this.stats.rowsOut++;
    const out = {};
    for (const item of this.items) out[item.alias] = evalExpr(item.expr, row);
    return out;
  }
}

class HashJoinOp {
  constructor(left, right, predicate, hashKey) {
    this.kind = "join";
    this.strategy = "hash";
    this.joinType = "inner";
    this.left = left;
    this.right = right;
    this.predicate = predicate;
    this.hashKey = hashKey;
    this.stats = { rowsOut: 0, probeRows: 0 };
  }

  open(database) {
    this.left.open(database);
    this.right.open(database);
    this.buckets = new Map();
    for (;;) {
      const row = this.right.next();
      if (row === null) break;
      const key = evalExpr(this.hashKey.right, row);
      if (key === undefined) continue;
      if (!this.buckets.has(key)) this.buckets.set(key, []);
      this.buckets.get(key).push(row);
    }
    this.pending = [];
  }

  next() {
    for (;;) {
      if (this.pending.length > 0) {
        const row = this.pending.shift();
        this.stats.rowsOut++;
        return row;
      }
      const leftRow = this.left.next();
      if (leftRow === null) return null;
      this.stats.probeRows++;
      const candidates = this.buckets.get(evalExpr(this.hashKey.left, leftRow)) ?? [];
      const matches = [];
      for (const rightRow of candidates) {
        const merged = { ...leftRow, ...rightRow };
        if (this.predicate === null || truthy(evalExpr(this.predicate, merged))) matches.push(merged);
      }
      this.pending = matches;
    }
  }
}

class NestedLoopJoinOp {
  constructor(left, right, predicate, joinType) {
    this.kind = "join";
    this.strategy = "nested";
    this.joinType = joinType;
    this.left = left;
    this.right = right;
    this.predicate = predicate;
    this.hashKey = null;
    this.stats = { rowsOut: 0, probeRows: 0 };
  }

  open(database) {
    this.left.open(database);
    this.right.open(database);
    this.rightRows = [];
    this.rightColumns = [];
    for (;;) {
      const row = this.right.next();
      if (row === null) break;
      this.rightRows.push(row);
      for (const column of Object.keys(row)) if (!this.rightColumns.includes(column)) this.rightColumns.push(column);
    }
    if (this.rightRows.length === 0) {
      for (const column of SCHEMA[this.right.table]) this.rightColumns.push(`${this.right.alias}.${column}`);
    }
    this.current = null;
  }

  next() {
    for (;;) {
      if (this.current === null) {
        const leftRow = this.left.next();
        if (leftRow === null) return null;
        this.stats.probeRows++;
        this.current = { leftRow, index: 0, matched: false };
      }
      const current = this.current;
      if (current.index >= this.rightRows.length) {
        const emitUnmatched = this.joinType === "left" && !current.matched;
        this.current = null;
        if (emitUnmatched) {
          const merged = { ...current.leftRow };
          for (const column of this.rightColumns) merged[column] = undefined;
          this.stats.rowsOut++;
          return merged;
        }
        continue;
      }
      const rightRow = this.rightRows[current.index++];
      const merged = { ...current.leftRow, ...rightRow };
      if (this.predicate === null || truthy(evalExpr(this.predicate, merged))) {
        current.matched = true;
        this.stats.rowsOut++;
        return merged;
      }
    }
  }
}

function aggregateValue(agg, rows) {
  if (agg.func === "count" && agg.arg.kind === "star") return rows.length;
  const values = [];
  for (const row of rows) {
    const value = evalExpr(agg.arg, row);
    if (value !== undefined) values.push(value);
  }
  switch (agg.func) {
    case "count": return values.length;
    case "sum": return values.length === 0 ? undefined : values.reduce((a, b) => a + b, 0);
    case "avg": return values.length === 0 ? undefined : values.reduce((a, b) => a + b, 0) / values.length;
    case "min": return values.length === 0 ? undefined : values.reduce((a, b) => (b < a ? b : a));
    default: return values.length === 0 ? undefined : values.reduce((a, b) => (b > a ? b : a));
  }
}

class AggregateOp {
  constructor(child, groupExprs, aggs, having) {
    this.kind = "aggregate";
    this.child = child;
    this.groupExprs = groupExprs;
    this.aggs = aggs;
    this.having = having;
    this.stats = { rowsOut: 0, inputRows: 0 };
  }

  open(database) {
    this.child.open(database);
    const groups = new Map();
    for (;;) {
      const row = this.child.next();
      if (row === null) break;
      this.stats.inputRows++;
      const parts = this.groupExprs.map((expr) => {
        const value = evalExpr(expr, row);
        return value === undefined ? null : value;
      });
      const key = JSON.stringify(parts);
      if (!groups.has(key)) groups.set(key, { parts, rows: [] });
      groups.get(key).rows.push(row);
    }
    if (groups.size === 0 && this.groupExprs.length === 0) groups.set("[]", { parts: [], rows: [] });
    this.results = [];
    for (const [key, group] of [...groups.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))) {
      const out = {};
      this.groupExprs.forEach((expr, index) => { out[exprText(expr)] = group.parts[index]; });
      for (const agg of this.aggs) out[exprText(agg)] = aggregateValue(agg, group.rows);
      if (this.having !== null && !truthy(evalExpr(this.having, out))) continue;
      this.results.push(out);
      this.stats.rowsOut++;
    }
    this.pos = 0;
  }

  next() {
    if (this.pos >= this.results.length) return null;
    return this.results[this.pos++];
  }
}

class SortOp {
  constructor(child, keys) { this.kind = "sort"; this.child = child; this.keys = keys; this.stats = { rowsOut: 0 }; }
  open(database) {
    this.child.open(database);
    const rows = [];
    for (;;) {
      const row = this.child.next();
      if (row === null) break;
      rows.push(row);
    }
    rows.sort((a, b) => {
      for (const key of this.keys) {
        const left = evalExpr(key.expr, a);
        const right = evalExpr(key.expr, b);
        if (left === right) continue;
        const cmp = left === undefined ? -1 : right === undefined ? 1 : left < right ? -1 : 1;
        return key.dir === "desc" ? -cmp : cmp;
      }
      return 0;
    });
    this.rows = rows.map((row) => {
      const copy = {};
      for (const [column, value] of Object.entries(row)) if (!column.startsWith("__sort")) copy[column] = value;
      return copy;
    });
    this.pos = 0;
  }
  next() { if (this.pos >= this.rows.length) return null; this.stats.rowsOut++; return this.rows[this.pos++]; }
}

class DistinctOp {
  constructor(child) { this.kind = "distinct"; this.child = child; this.stats = { rowsOut: 0 }; }
  open(database) { this.child.open(database); this.seen = new Set(); }
  next() {
    for (;;) {
      const row = this.child.next();
      if (row === null) return null;
      const key = rowKey(row);
      if (this.seen.has(key)) continue;
      this.seen.add(key);
      this.stats.rowsOut++;
      return row;
    }
  }
}

class LimitOp {
  constructor(child, limit) { this.kind = "limit"; this.child = child; this.limit = limit; this.stats = { rowsOut: 0 }; }
  open(database) { this.child.open(database); this.emitted = 0; }
  next() {
    if (this.emitted >= this.limit) return null;
    const row = this.child.next();
    if (row === null) return null;
    this.emitted++;
    this.stats.rowsOut++;
    return row;
  }
}

function childrenOf(node) {
  const out = [];
  if (node.child !== undefined) out.push(node.child);
  if (node.left !== undefined) out.push(node.left);
  if (node.right !== undefined) out.push(node.right);
  return out;
}

function mapChildren(node, fn) {
  const clone = Object.create(Object.getPrototypeOf(node));
  Object.assign(clone, node);
  if (node.child !== undefined) clone.child = fn(node.child);
  if (node.left !== undefined) clone.left = fn(node.left);
  if (node.right !== undefined) clone.right = fn(node.right);
  return clone;
}

function findScanByAlias(node, alias) {
  if (node.kind === "scan") return node.alias === alias ? node : null;
  for (const child of childrenOf(node)) {
    const found = findScanByAlias(child, alias);
    if (found !== null) return found;
  }
  return null;
}

function planText(node, depth = 0) {
  const pad = "  ".repeat(depth);
  let line;
  switch (node.kind) {
    case "scan": line = `Scan(${node.table} as ${node.alias}, [${(node.columns ?? SCHEMA[node.table]).join(",")}])`; break;
    case "filter": line = `Filter(${exprText(node.predicate)})`; break;
    case "project": line = `Project([${node.items.map((item) => `${exprText(item.expr)} as ${item.alias}`).join(", ")}])`; break;
    case "join": line = `${node.strategy === "hash" ? "HashJoin" : "NestedLoopJoin"}(${node.joinType}, ${node.predicate === null ? "true" : exprText(node.predicate)})`; break;
    case "aggregate": line = `Aggregate(group=[${node.groupExprs.map(exprText).join(", ")}], aggs=[${node.aggs.map(exprText).join(", ")}]${node.having ? `, having ${exprText(node.having)}` : ""})`; break;
    case "sort": line = `Sort([${node.keys.map((key) => `${exprText(key.expr)} ${key.dir}`).join(", ")}])`; break;
    case "distinct": line = "Distinct"; break;
    case "limit": line = `Limit(${node.limit})`; break;
    default: line = node.kind;
  }
  return [pad + line, ...childrenOf(node).map((child) => planText(child, depth + 1))].join("\n");
}

function statsOf(node, out = {}) {
  const stats = node.stats ?? {};
  out[node.kind] = (out[node.kind] ?? 0) + (stats.rowsOut ?? 0);
  if (stats.columnsLoaded !== undefined) out.columnsLoaded = (out.columnsLoaded ?? 0) + stats.columnsLoaded;
  if (stats.inputRows !== undefined) out.aggregateInputRows = (out.aggregateInputRows ?? 0) + stats.inputRows;
  if (node.strategy !== undefined && stats.probeRows !== undefined) out.probeRows = (out.probeRows ?? 0) + stats.probeRows;
  for (const child of childrenOf(node)) statsOf(child, out);
  return out;
}

function execute(plan, database) {
  plan.open(database);
  const rows = [];
  for (;;) {
    const row = plan.next();
    if (row === null) break;
    rows.push(row);
  }
  return { rows, stats: statsOf(plan), planText: planText(plan) };
}

function resultMatrix(result) {
  const columns = result.rows.length === 0 ? [] : Object.keys(result.rows[0]);
  return result.rows.map((row) => columns.map((column) => (row[column] === undefined ? null : row[column])));
}

// --------------------------------------------------------------------------- planner

function allColumnsOf(scope) {
  const keys = [];
  for (const item of scope) for (const column of SCHEMA[item.table]) keys.push(`${item.alias}.${column}`);
  return keys;
}

function equalityKey(predicate, leftAliases, rightAliases) {
  for (const part of conjuncts(predicate)) {
    if (part.kind !== "bin" || part.op !== "=") continue;
    if (part.left.kind !== "ref" || part.right.kind !== "ref") continue;
    if (leftAliases.includes(part.left.qualifier) && rightAliases.includes(part.right.qualifier)) {
      return { left: part.left, right: part.right };
    }
    if (rightAliases.includes(part.left.qualifier) && leftAliases.includes(part.right.qualifier)) {
      return { left: part.right, right: part.left };
    }
  }
  return null;
}

class Planner {
  constructor(database) { this.database = database; }

  compile(sql) {
    const ast = parse(sql);
    const scope = [{ ...ast.from }];
    if (!(ast.from.table in this.database)) throw new SqlError(`unknown table ${ast.from.table}`);
    for (const join of ast.joins) {
      if (!(join.table.table in this.database)) throw new SqlError(`unknown table ${join.table.table}`);
      scope.push({ ...join.table });
    }
    const resolve = (expr) => resolveRefs(foldConstants(expr), scope);
    const aliasNames = scope.map((item) => item.alias);
    const duplicate = aliasNames.find((alias, index) => aliasNames.indexOf(alias) !== index);
    if (duplicate !== undefined) throw new SqlError(`duplicate table alias ${duplicate}; use AS to give each occurrence its own name`);
    // ORDER BY may name an output column, so an explicit select alias wins over a table column there.
    const explicitAliases = new Set(ast.items.filter((item) => !item.star && item.alias !== null).map((item) => item.alias));
    const resolveOrderKey = (expr) => (expr.kind === "ref" && expr.qualifier === null && explicitAliases.has(expr.name)
      ? { kind: "alias", name: expr.name }
      : resolve(expr));
    const resolved = {
      ...ast,
      joins: ast.joins.map((join) => ({ ...join, on: resolve(join.on) })),
      where: resolve(ast.where),
      groupBy: ast.groupBy.map(resolve),
      having: ast.having === null ? null : resolve(ast.having),
      orderBy: ast.orderBy.map((key) => ({ dir: key.dir, expr: resolveOrderKey(key.expr) })),
      items: ast.items.map((item) => (item.star ? { ...item } : { ...item, expr: resolve(item.expr) })),
    };
    if (containsAggregate(resolved.where)) throw new SqlError("aggregates are not allowed in WHERE");
    for (const join of resolved.joins) if (containsAggregate(join.on)) throw new SqlError("aggregates are not allowed in ON");
    const aggregate = resolved.groupBy.length > 0
      || resolved.items.some((item) => !item.star && containsAggregate(item.expr))
      || containsAggregate(resolved.having);
    const plan = this.build(resolved, scope, aggregate);
    return { ast: resolved, plan, text: planText(plan) };
  }

  build(ast, scope, aggregate) {
    // --- join tree: left-deep in written order, with single-side conjuncts pushed into the inputs -
    const nullable = new Set(ast.joins.filter((join) => join.type === "left").map((join) => join.table.alias));
    let node = new ScanOp(ast.from.table, ast.from.alias);
    let leftAliases = [ast.from.alias];
    for (const join of ast.joins) {
      const rightAliases = [join.table.alias];
      const leftOnly = [];
      const rightOnly = [];
      const bothSides = [];
      for (const part of conjuncts(join.on)) {
        const aliases = [...new Set(refsOf(part).map((ref) => ref.qualifier))];
        const inLeft = aliases.every((alias) => leftAliases.includes(alias));
        const inRight = aliases.every((alias) => rightAliases.includes(alias));
        if (inLeft && !inRight && join.type === "inner") leftOnly.push(part);
        else if (inRight && !inLeft && join.type === "inner") rightOnly.push(part);
        else bothSides.push(part);
      }
      const leftInput = leftOnly.length === 0 ? node : new FilterOp(node, combineAnd(leftOnly));
      let rightInput = new ScanOp(join.table.table, join.table.alias);
      if (rightOnly.length > 0) rightInput = new FilterOp(rightInput, combineAnd(rightOnly));
      const predicate = combineAnd(bothSides);
      const hashKey = join.type === "inner" ? equalityKey(predicate, leftAliases, rightAliases) : null;
      node = hashKey !== null
        ? new HashJoinOp(leftInput, rightInput, predicate, hashKey)
        : new NestedLoopJoinOp(leftInput, rightInput, predicate, join.type);
      leftAliases = [...leftAliases, ...rightAliases];
    }
    // --- WHERE: conjuncts that name exactly one non-nullable alias sink into that scan -----------
    const remaining = [];
    for (const part of conjuncts(ast.where)) {
      const aliases = [...new Set(refsOf(part).map((ref) => ref.qualifier))];
      const target = aliases.length === 1 && !nullable.has(aliases[0]) ? findScanByAlias(node, aliases[0]) : null;
      if (target !== null) node = replaceScan(node, target, new FilterOp(target, part));
      else remaining.push(part);
    }
    if (remaining.length > 0) node = new FilterOp(node, combineAnd(remaining));

    // --- aggregation: rewrite select/HAVING/ORDER BY onto group and aggregate keys ---------------
    if (aggregate) {
      const groupTexts = ast.groupBy.map(exprText);
      const aggregates = [];
      const collect = (expr) => {
        if (expr === null || expr === undefined) return;
        if (expr.kind === "agg") { if (!aggregates.some((agg) => exprText(agg) === exprText(expr))) aggregates.push(expr); return; }
        if (expr.kind === "bin") { collect(expr.left); collect(expr.right); return; }
        if (expr.kind === "un" || expr.kind === "isnull") collect(expr.operand);
      };
      for (const item of ast.items) if (!item.star) collect(item.expr);
      collect(ast.having);
      const rewrite = (expr) => {
        const text = exprText(expr);
        if (groupTexts.includes(text) || expr.kind === "agg") return refNode(text);
        if (expr.kind === "bin") return { ...expr, left: rewrite(expr.left), right: rewrite(expr.right) };
        if (expr.kind === "un" || expr.kind === "isnull") return { ...expr, operand: rewrite(expr.operand) };
        if (expr.kind === "ref") throw new SqlError(`column ${text} must appear in GROUP BY or be aggregated`);
        return expr;
      };
      const items = ast.items.map((item) => (item.star ? { star: true, expr: null, alias: null } : { ...item, expr: rewrite(item.expr) }));
      ast = {
        ...ast,
        items,
        having: ast.having === null ? null : rewrite(ast.having),
        orderBy: ast.orderBy.map((key) => ({ ...key, expr: rewrite(key.expr) })),
      };
      node = new AggregateOp(node, ast.groupBy, aggregates, ast.having);
    }

    // --- projection, DISTINCT, ORDER BY (hidden column when a key is not selected) ---------------
    const items = ast.items.some((item) => item.star)
      ? allColumnsOf(scope).map((key) => ({ alias: key, expr: refNode(key) }))
      : ast.items.map((item) => ({ alias: item.alias ?? exprText(item.expr), expr: item.expr }));
    const byText = new Map(items.map((item) => [exprText(item.expr), item.alias]));
    const sortKeys = [];
    for (const key of ast.orderBy) {
      const text = exprText(key.expr);
      let alias = key.expr.kind === "alias" ? key.expr.name : byText.get(text);
      if (alias === undefined && !ast.distinct) {
        alias = `__sort${sortKeys.length}`;
        items.push({ alias, expr: key.expr });
      }
      sortKeys.push({ dir: key.dir, expr: alias === undefined ? key.expr : (key.expr.kind === "alias" ? key.expr : refNode(alias)) });
    }
    node = new ProjectOp(node, items);
    if (ast.distinct) node = new DistinctOp(node);
    if (sortKeys.length > 0) node = new SortOp(node, sortKeys);
    if (ast.limit !== null) node = new LimitOp(node, ast.limit);
    return pruneColumns(node, ast.items.some((item) => item.star) ? null : neededColumns(node));
  }
}

function replaceScan(node, target, replacement) {
  if (node === target) return replacement;
  if (node.kind === "scan") return node;
  return mapChildren(node, (child) => replaceScan(child, target, replacement));
}

function neededColumns(node, out = new Set()) {
  const collect = (expr) => { for (const ref of refsOf(expr)) out.add(ref.key); };
  if (node.kind === "scan") return out;
  collect(node.predicate);
  if (node.hashKey) { collect(node.hashKey.left); collect(node.hashKey.right); }
  if (node.items) for (const item of node.items) collect(item.expr);
  if (node.groupExprs) for (const expr of node.groupExprs) collect(expr);
  if (node.aggs) for (const agg of node.aggs) collect(agg);
  if (node.keys) for (const key of node.keys) collect(key.expr);
  for (const child of childrenOf(node)) neededColumns(child, out);
  return out;
}

function pruneColumns(node, needed) {
  if (needed === null) return node;
  if (node.kind === "scan") {
    const clone = Object.create(Object.getPrototypeOf(node));
    Object.assign(clone, node);
    const columns = [...needed].filter((key) => key.startsWith(`${node.alias}.`)).map((key) => key.slice(node.alias.length + 1));
    clone.columns = columns.length === 0 ? [SCHEMA[node.table][0]] : columns;
    return clone;
  }
  return mapChildren(node, (child) => pruneColumns(child, needed));
}

// --------------------------------------------------------------------------- brute-force oracle

/** Independent evaluator: full cross product, aggregates folded straight over the group's rows. */
function naiveExecute(sql, database) {
  const ast = parse(sql);
  const tables = [{ ...ast.from }, ...ast.joins.map((join) => ({ ...join.table }))];
  const resolve = (expr) => resolveRefs(expr, tables);
  const where = resolve(ast.where);
  const ons = ast.joins.map((join) => ({ type: join.type, on: resolve(join.on) }));
  const items = ast.items.map((item) => (item.star ? { star: true } : { ...item, expr: resolve(item.expr) }));
  const groupBy = ast.groupBy.map(resolve);
  const having = ast.having === null ? null : resolve(ast.having);
  const aliases = new Set(ast.items.filter((item) => !item.star && item.alias !== null).map((item) => item.alias));
  const orderBy = ast.orderBy.map((key) => ({
    dir: key.dir,
    expr: key.expr.kind === "ref" && key.expr.qualifier === null && aliases.has(key.expr.name)
      ? { kind: "alias", name: key.expr.name }
      : resolve(key.expr),
  }));
  const materialize = (ref) => database[ref.table].map((row) => {
    const out = {};
    for (const column of SCHEMA[ref.table]) out[`${ref.alias}.${column}`] = row[column];
    return out;
  });
  let rows = materialize(tables[0]);
  for (let i = 0; i < ons.length; i++) {
    const right = materialize(tables[i + 1]);
    const merged = [];
    for (const left of rows) {
      let matched = false;
      for (const rightRow of right) {
        const candidate = { ...left, ...rightRow };
        if (truthy(evalExpr(ons[i].on, candidate))) { merged.push(candidate); matched = true; }
      }
      if (ons[i].type === "left" && !matched) {
        const candidate = { ...left };
        for (const column of SCHEMA[tables[i + 1].table]) candidate[`${tables[i + 1].alias}.${column}`] = undefined;
        merged.push(candidate);
      }
    }
    rows = merged;
  }
  if (where !== null) rows = rows.filter((row) => truthy(evalExpr(where, row)));

  const fold = (expr, groupRows) => {
    if (expr.kind === "agg") return aggregateValue(expr, groupRows);
    if (expr.kind === "alias") return groupRows.length === 0 ? undefined : groupRows[0][expr.name];
    if (expr.kind === "bin") {
      const left = fold(expr.left, groupRows);
      if (expr.op === "and" && !truthy(left)) return false;
      if (expr.op === "or" && truthy(left)) return true;
      return applyBinary(expr.op, left, fold(expr.right, groupRows));
    }
    if (expr.kind === "un") return expr.op === "not" ? !truthy(fold(expr.operand, groupRows)) : -fold(expr.operand, groupRows);
    if (expr.kind === "isnull") { const value = fold(expr.operand, groupRows); return expr.negated ? value !== undefined : value === undefined; }
    if (expr.kind === "num" || expr.kind === "str") return expr.value;
    return groupRows.length === 0 ? undefined : evalExpr(expr, groupRows[0]);
  };

  const grouped = groupBy.length > 0 || items.some((item) => !item.star && containsAggregate(item.expr)) || containsAggregate(having);
  let outRows;
  if (grouped) {
    const groups = new Map();
    for (const row of rows) {
      const parts = groupBy.map((expr) => { const value = evalExpr(expr, row); return value === undefined ? null : value; });
      const key = JSON.stringify(parts);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    }
    if (groups.size === 0 && groupBy.length === 0) groups.set("[]", []);
    outRows = [...groups.entries()]
      .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
      .filter(([, groupRows]) => having === null || truthy(fold(having, groupRows)))
      .map(([, groupRows]) => {
        const out = {};
        if (items.some((item) => item.star)) {
          for (const key of allColumnsOf(tables)) out[key] = groupRows.length === 0 ? undefined : evalExpr(refNode(key), groupRows[0]);
        } else {
          for (const item of items) out[item.alias ?? exprText(item.expr)] = fold(item.expr, groupRows);
        }
        return out;
      });
  } else {
    outRows = rows.map((row) => {
      const out = {};
      if (items.some((item) => item.star)) {
        for (const key of allColumnsOf(tables)) out[key] = row[key];
      } else {
        for (const item of items) out[item.alias ?? exprText(item.expr)] = evalExpr(item.expr, row);
      }
      return out;
    });
  }
  if (ast.distinct) {
    const seen = new Set();
    outRows = outRows.filter((row) => { const key = rowKey(row); if (seen.has(key)) return false; seen.add(key); return true; });
  }
  if (orderBy.length > 0) {
    outRows = [...outRows].sort((a, b) => {
      for (const key of orderBy) {
        const text = exprText(key.expr);
        const left = a[text] !== undefined ? a[text] : fold(key.expr, [a]);
        const right = b[text] !== undefined ? b[text] : fold(key.expr, [b]);
        if (left === right) continue;
        const cmp = left === undefined ? -1 : right === undefined ? 1 : left < right ? -1 : 1;
        return key.dir === "desc" ? -cmp : cmp;
      }
      return 0;
    });
  }
  if (ast.limit !== null) outRows = outRows.slice(0, ast.limit);
  return outRows;
}

function canonicalRows(rows) { return rows.map((row) => rowKey(row)); }

function compareResults(executed, expected, ordered) {
  const actual = canonicalRows(executed.rows);
  const oracle = canonicalRows(expected);
  if (ordered) return JSON.stringify(actual) === JSON.stringify(oracle);
  return JSON.stringify([...actual].sort()) === JSON.stringify([...oracle].sort());
}

// --------------------------------------------------------------------------- fixture data

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

function buildDatabase(rng, userCount = 24, orderCount = 40) {
  const cityNames = ["Paris", "Berlin", "Lyon", "Porto", "Hanoi"];
  const countries = ["FR", "DE", "FR", "PT", "VN"];
  return {
    users: Array.from({ length: userCount }, (_, index) => ({
      id: index + 1,
      name: `user${String(index + 1).padStart(2, "0")}`,
      city: cityNames[rng.int(cityNames.length)],
      age: 18 + rng.int(40),
    })),
    orders: Array.from({ length: orderCount }, (_, index) => ({
      id: index + 1,
      user_id: 1 + rng.int(userCount),
      amount: 5 + rng.int(95),
    })),
    cities: cityNames.map((name, index) => ({ name, country: countries[index], population: 100000 * (index + 1) + 7 })),
    empty: [],
  };
}

if (require.main === module) {
  const database = buildDatabase(splitmix64(0x5011));
  const planner = new Planner(database);

  // --- 1. hand-computed golden results -----------------------------------------------------------
  const goldenTable = {
    users: [{ id: 1, name: "ann", city: "Paris", age: 30 }, { id: 2, name: "bob", city: "Lyon", age: 40 },
      { id: 3, name: "cid", city: "Paris", age: 50 }],
    orders: [{ id: 10, user_id: 1, amount: 5 }, { id: 11, user_id: 1, amount: 7 }, { id: 12, user_id: 3, amount: 1 }],
    cities: [{ name: "Paris", country: "FR", population: 100 }, { name: "Lyon", country: "FR", population: 50 }],
    empty: [],
  };
  const golden = new Planner(goldenTable);
  const goldenJoin = execute(golden.compile("SELECT users.name, orders.amount FROM users JOIN orders ON users.id = orders.user_id WHERE orders.amount > 4").plan, goldenTable);
  assert.deepStrictEqual(resultMatrix(goldenJoin), [["ann", 5], ["ann", 7]], "join + where golden result");
  const goldenAgg = execute(golden.compile("SELECT city, COUNT(*) AS n, SUM(age) AS total FROM users GROUP BY city HAVING COUNT(*) >= 1 ORDER BY city").plan, goldenTable);
  assert.deepStrictEqual(resultMatrix(goldenAgg), [["Lyon", 1, 40], ["Paris", 2, 80]], "group by + having golden result");
  const goldenLeft = execute(golden.compile("SELECT users.name, orders.amount FROM users LEFT JOIN orders ON users.id = orders.user_id ORDER BY users.name, orders.amount").plan, goldenTable);
  assert.strictEqual(goldenLeft.rows.length, 4, "LEFT JOIN keeps the unmatched user");
  assert.strictEqual(goldenLeft.rows.filter((row) => row["orders.amount"] === undefined).length, 1, "with NULLs on the right side");
  assert.deepStrictEqual(resultMatrix(execute(golden.compile("SELECT DISTINCT city FROM users ORDER BY city").plan, goldenTable)), [["Lyon"], ["Paris"]], "DISTINCT + ORDER BY");

  // --- 2. planner decisions are real optimisations ----------------------------------------------
  const pushed = golden.compile("SELECT users.name FROM users JOIN orders ON users.id = orders.user_id WHERE orders.amount > 4");
  const pushedJoin = pushed.plan.child;
  execute(pushed.plan, goldenTable);
  assert.strictEqual(pushedJoin.kind, "join", "the join sits directly under the project");
  assert.strictEqual(pushedJoin.right.kind, "filter", "the WHERE conjunct was pushed into the join's input");
  assert.strictEqual(pushed.text.split("\n")[1].trim().startsWith("HashJoin"), true, `an equi-join becomes a hash join:\n${pushed.text}`);
  const pushedStats = statsOf(pushed.plan);
  assert.strictEqual(pushedJoin.stats.rowsOut, 2, "the join emits exactly the two matching pairs");
  assert.strictEqual(pushedJoin.right.stats.rowsOut, 2, "the pushed filter removed one order before the join");
  assert.strictEqual(pushedJoin.stats.probeRows, 3, "the probe side still reads every user");
  assert.ok(pushedStats.join < goldenTable.users.length * goldenTable.orders.length, `fewer rows than the cross product (${pushedStats.join})`);

  const nonEqui = golden.compile("SELECT users.name FROM users JOIN orders ON users.age > orders.amount");
  assert.strictEqual(nonEqui.plan.child.strategy, "nested", "a non-equi join falls back to the nested loop");
  const cross = execute(nonEqui.plan, goldenTable);
  const expectedCross = goldenTable.users.reduce((sum, user) => sum
    + goldenTable.orders.filter((order) => user.age > order.amount).length, 0);
  assert.strictEqual(cross.rows.length, expectedCross, "non-equi join row count matches a brute-force count");
  assert.strictEqual(expectedCross, 9, "in this fixture every user is older than every order amount");
  assert.strictEqual(nonEqui.plan.child.stats.probeRows, goldenTable.users.length, "the nested loop probes every left row");

  // --- 3. column pruning and constant folding ---------------------------------------------------
  const pruned = golden.compile("SELECT users.name FROM users JOIN orders ON users.id = orders.user_id");
  execute(pruned.plan, goldenTable);
  const prunedStats = statsOf(pruned.plan);
  const fullScanValues = goldenTable.users.length * 4 + goldenTable.orders.length * 3;
  assert.ok(prunedStats.columnsLoaded < fullScanValues, `pruning materialises fewer values (${prunedStats.columnsLoaded} < ${fullScanValues})`);
  const full = golden.compile("SELECT * FROM users JOIN orders ON users.id = orders.user_id");
  execute(full.plan, goldenTable);
  assert.strictEqual(statsOf(full.plan).columnsLoaded, fullScanValues, "SELECT * materialises everything");
  const folded = golden.compile("SELECT users.name FROM users WHERE users.age > 20 + 20");
  assert.ok(/Filter\(\(users\.age > 40\)\)/.test(folded.text), `constant folding shows up in the plan text:\n${folded.text}`);

  // --- 4. the oracle agrees on a curated query set ----------------------------------------------
  const curated = [
    "SELECT * FROM users WHERE age > 30",
    "SELECT name, age FROM users WHERE city = 'Paris' AND age <= 40 ORDER BY age DESC",
    "SELECT DISTINCT city FROM users",
    "SELECT COUNT(*) AS n FROM orders WHERE amount > 50",
    "SELECT SUM(amount) AS total, MIN(amount) AS lo, MAX(amount) AS hi, AVG(amount) AS average FROM orders",
    "SELECT users.city, COUNT(*) AS orders, SUM(orders.amount) AS spend FROM users JOIN orders ON users.id = orders.user_id GROUP BY users.city ORDER BY spend DESC LIMIT 3",
    "SELECT users.name, cities.country FROM users JOIN cities ON users.city = cities.name WHERE cities.population > 250000 ORDER BY users.name",
    "SELECT users.name, orders.amount FROM users LEFT JOIN orders ON users.id = orders.user_id WHERE orders.amount IS NULL",
    "SELECT city, COUNT(*) AS n FROM users GROUP BY city HAVING COUNT(*) > 3 ORDER BY n DESC, city ASC",
    "SELECT COUNT(*) AS n FROM empty",
    "SELECT COUNT(amount) AS n FROM empty",
    "SELECT id FROM users ORDER BY id LIMIT 0",
    "SELECT id FROM users ORDER BY id LIMIT 1000",
    "SELECT DISTINCT users.city, cities.country FROM users JOIN cities ON users.city = cities.name ORDER BY users.city",
    "SELECT users.name, users.age - 18 AS adult_years FROM users WHERE users.age >= 40 ORDER BY adult_years DESC, users.name ASC LIMIT 5",
    "SELECT u.city, COUNT(*) AS n FROM users AS u JOIN orders AS o ON u.id = o.user_id GROUP BY u.city ORDER BY u.city",
    "SELECT users.name, orders.amount FROM users LEFT JOIN orders ON users.id = orders.user_id ORDER BY users.name, orders.amount LIMIT 8",
    "SELECT COUNT(*) AS n FROM users WHERE NOT (city = 'Paris') AND age > 20",
    "SELECT cities.name, COUNT(*) AS n FROM cities LEFT JOIN users ON users.city = cities.name GROUP BY cities.name ORDER BY cities.name",
    "SELECT id, amount FROM orders WHERE amount >= 50 AND amount <= 90 ORDER BY amount DESC, id ASC",
  ];
  for (const sql of curated) {
    const compiled = planner.compile(sql);
    const executed = execute(compiled.plan, database);
    const expected = naiveExecute(sql, database);
    assert.strictEqual(compareResults(executed, expected, compiled.ast.orderBy.length > 0), true, `oracle mismatch for: ${sql}`);
    assert.strictEqual(executed.rows.length, expected.length, `row count mismatch for: ${sql}`);
  }

  // --- 5. randomised queries against the oracle --------------------------------------------------
  const random = splitmix64(0xabcdef);
  const columns = ["users.id", "users.name", "users.city", "users.age", "orders.amount", "orders.user_id"];
  const literals = ["'Paris'", "'Berlin'", "'Lyon'", "'Porto'", "'Hanoi'"];
  let randomChecked = 0;
  for (let i = 0; i < 25; i++) {
    const pick = () => columns[random.int(columns.length)];
    const column = pick();
    const op = ["=", "<", ">", "<=", ">=", "<>"][random.int(6)];
    const literal = random.int(2) === 0 ? literals[random.int(literals.length)] : String(random.int(60));
    const sql = random.int(3) === 0
      ? `SELECT users.city, COUNT(*) AS n, SUM(orders.amount) AS total FROM users JOIN orders ON users.id = orders.user_id WHERE ${column} ${op} ${literal} GROUP BY users.city ORDER BY users.city`
      : `SELECT users.name, ${pick()} FROM users JOIN orders ON users.id = orders.user_id WHERE ${column} ${op} ${literal} ORDER BY users.name LIMIT ${1 + random.int(20)}`;
    const compiled = planner.compile(sql);
    const executed = execute(compiled.plan, database);
    const expected = naiveExecute(sql, database);
    assert.strictEqual(compareResults(executed, expected, compiled.ast.orderBy.length > 0), true, `random oracle mismatch for: ${sql}`);
    randomChecked++;
  }

  // --- 6. edge semantics required by SQL ---------------------------------------------------------
  assert.deepStrictEqual(resultMatrix(execute(planner.compile("SELECT COUNT(*) AS n FROM empty").plan, database)), [[0]], "COUNT(*) over an empty table is 0");
  assert.deepStrictEqual(resultMatrix(execute(planner.compile("SELECT SUM(amount) AS total FROM empty").plan, database)), [[null]], "SUM over no rows is NULL");
  assert.strictEqual(execute(planner.compile("SELECT amount, COUNT(*) AS n FROM empty GROUP BY amount").plan, database).rows.length, 0, "GROUP BY over an empty table yields no groups");
  assert.strictEqual(execute(planner.compile("SELECT id FROM users ORDER BY id LIMIT 0").plan, database).rows.length, 0, "LIMIT 0 returns nothing");
  assert.strictEqual(execute(planner.compile("SELECT id FROM users ORDER BY id LIMIT 1000").plan, database).rows.length, database.users.length, "LIMIT past the end returns everything");
  const distinct = execute(planner.compile("SELECT DISTINCT city FROM users").plan, database);
  assert.strictEqual(distinct.rows.length, new Set(database.users.map((user) => user.city)).size, "DISTINCT collapses duplicates");
  assert.deepStrictEqual(resultMatrix(execute(planner.compile("SELECT DISTINCT city FROM users").plan, database)), resultMatrix(distinct), "no ORDER BY is still deterministic");
  const nullFilter = execute(planner.compile("SELECT users.name FROM users LEFT JOIN orders ON users.id = orders.user_id WHERE orders.amount IS NULL").plan, database);
  assert.strictEqual(nullFilter.rows.length, database.users.filter((user) => !database.orders.some((order) => order.user_id === user.id)).length, "IS NULL finds exactly the unmatched users");
  assert.deepStrictEqual(resultMatrix(execute(planner.compile("SELECT id, age / 0 AS bad FROM users LIMIT 1").plan, database)), [[1, null]], "integer division by zero is NULL");

  // --- 7. errors instead of wrong answers -------------------------------------------------------
  const failures = [
    ["SELECT FROM users", /expected/],
    ["SELECT id, FROM users", /expected from/],
    ["SELECT bogus FROM users", /unknown column bogus/],
    ["SELECT * FROM nope", /unknown table nope/],
    ["SELECT city, COUNT(*) FROM users", /must appear in GROUP BY/],
    ["SELECT COUNT(*) FROM users WHERE COUNT(*) > 1", /not allowed in WHERE/],
    ["SELECT id FROM users JOIN orders ON users.id = orders.user_id", /ambiguous column id/],
    ["SELECT users.id FROM users JOIN users ON users.id = users.id", /duplicate table alias/],
    ["SELECT id FROM users, orders", /expected eof/],
    ["SELECT id FROM users WHERE age >", /unexpected token/],
    ["SELECT id FROM users WHERE name = 'oops", /unterminated string/],
    ["SELECT users.name FROM users JOIN orders ON users.id = orders.user_id WHERE orders.zzz = 1", /unknown column zzz/],
  ];
  for (const [sql, pattern] of failures) {
    assert.throws(() => planner.compile(sql), (error) => {
      assert.ok(error instanceof SqlError, `${sql} must raise SqlError, saw ${error.constructor.name}`);
      assert.match(error.message, pattern, `message for ${sql}`);
      return true;
    }, `${sql} must be rejected`);
  }

  // --- 8. determinism and cost reporting ---------------------------------------------------------
  const repeated = planner.compile("SELECT users.city, COUNT(*) AS n FROM users JOIN orders ON users.id = orders.user_id GROUP BY users.city ORDER BY users.city");
  const first = execute(repeated.plan, database);
  const second = execute(new Planner(database).compile("SELECT users.city, COUNT(*) AS n FROM users JOIN orders ON users.id = orders.user_id GROUP BY users.city ORDER BY users.city").plan, database);
  assert.strictEqual(first.planText, second.planText, "the planner is deterministic");
  assert.deepStrictEqual(resultMatrix(first), resultMatrix(second), "and so is the executor");
  assert.strictEqual(first.stats.scan, database.users.length + database.orders.length, "both scans read their table once");
  const crossProduct = database.users.length * database.orders.length;
  assert.ok(first.stats.aggregateInputRows < crossProduct, `the join output (${first.stats.aggregateInputRows}) is below the cross product (${crossProduct})`);
  const ratio = first.stats.aggregateInputRows / crossProduct;
  assert.ok(ratio < 0.35, `the join really filtered (ratio ${ratio.toFixed(3)})`);

  console.log(`sql: ${curated.length} curated + ${randomChecked} random queries cross-checked against the brute-force evaluator`);
  console.log(`sql: golden plans use hash joins, pushed predicates and pruned scans (${prunedStats.columnsLoaded}/${fullScanValues} values materialised)`);
  console.log(`sql: ${failures.length} malformed queries rejected, join output/cross product = ${ratio.toFixed(3)}, plan rows = ${JSON.stringify(first.stats)}`);
  console.log("PASS");
}

module.exports = {
  tokenize, parse, Parser, Planner, execute, naiveExecute, planText, statsOf, SCHEMA, SqlError,
  evalExpr, resolveRefs, containsAggregate, compareResults, resultMatrix, buildDatabase, splitmix64,
};
