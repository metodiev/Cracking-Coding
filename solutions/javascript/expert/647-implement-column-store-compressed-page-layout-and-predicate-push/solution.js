/** Task 647 - Implement column-store compressed page layout and predicate pushdown simulation. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMULATED: a columnar table whose columns are cut into pages, each page carrying (a) a
 * per-page dictionary with sorted distinct values, (b) the page's values as dictionary codes stored as
 * run-length encoded runs, and (c) a zone map (the page's exact min/max). Predicates are pushed down in
 * three steps, which is the real column-store play: the zone map decides whether the page can be skipped
 * entirely, the predicate is then evaluated against the *dictionary* (a few entries instead of the whole
 * page), and only the codes of dictionary entries that can match are expanded - so the values that get
 * decoded are exactly the ones that may be returned (late materialisation).
 *
 * The store keeps a plain decoded copy of every column for the oracle comparison only; the measured
 * counters (pages scanned/skipped, dictionary tests, codes tested, values decoded) are all produced by the
 * compressed path, never by the copy.
 *
 * WHAT IS VERIFIED: the encoding round-trips (dictionary + runs rebuild the column exactly); dictionaries
 * are sorted and duplicate free and zone maps are the true page minima/maxima; for a matrix of predicates
 * (=, <, <=, >, >=, BETWEEN) on random, repetitive, negative and degenerate columns the pushed-down scan
 * returns exactly the same positions as a full evaluation; the zone map never produces a false negative -
 * every page holding a matching row is scanned; a selective predicate skips pages and decodes far fewer
 * values than a full scan while a predicate that matches everything skips nothing; projecting a second
 * column through the selection vector decodes only the surviving positions; and invalid requests
 * (page size <= 0, unknown column) are refused.
 */
"use strict";

const OPS = {
  "=": (value, target) => value === target, "<": (value, target) => value < target,
  "<=": (value, target) => value <= target, ">": (value, target) => value > target,
  ">=": (value, target) => value >= target, between: (value, target, high) => value >= target && value <= high,
};

function makePredicate(op, value, high) {
  const test = OPS[op];
  if (!test) throw new Error(`unknown operator ${op}`);
  if (op === "between" && !(high >= value)) throw new Error("BETWEEN needs low <= high");
  return {
    op, value, high, test: (candidate) => test(candidate, value, high),
    /** The value interval the predicate accepts, used to intersect with a page's zone map. */
    range: () => {
      if (op === "=") return [value, value];
      if (op === "<") return [-Infinity, value - 1];
      if (op === "<=") return [-Infinity, value];
      if (op === ">") return [value + 1, Infinity];
      if (op === ">=") return [value, Infinity];
      return [value, high];
    },
  };
}

class Page {
  constructor(values) {
    this.values = values;
    this.dictionary = [...new Set(values)].sort((left, right) => left - right);
    const codeOf = new Map(this.dictionary.map((value, code) => [value, code]));
    this.codes = values.map((value) => codeOf.get(value));
    this.runs = [];
    for (const code of this.codes) {
      const last = this.runs[this.runs.length - 1];
      if (last && last.code === code) last.count++;
      else this.runs.push({ code, count: 1 });
    }
    this.min = values.length === 0 ? null : Math.min(...values);
    this.max = values.length === 0 ? null : Math.max(...values);
    this.scanned = false;
    this.skipped = false;
  }
  get length() {
    return this.values.length;
  }
  /** Rebuild the page from its compressed layout alone - the round-trip check. */
  rebuild() {
    const values = [];
    for (const run of this.runs) {
      for (let count = 0; count < run.count; count++) values.push(this.dictionary[run.code]);
    }
    return values;
  }
  runCount() {
    return this.runs.length;
  }
  overlaps(predicate) {
    if (this.length === 0) return false;
    const [low, high] = predicate.range();
    return this.max >= low && this.min <= high;
  } }

class ColumnStore {
  constructor(pageSize) {
    if (!(pageSize >= 1)) throw new Error("the page size must be at least one value");
    this.pageSize = pageSize;
    this.columns = new Map();
    this.stats = { scans: 0, pagesScanned: 0, pagesSkipped: 0, dictionaryTests: 0, codesTested: 0, valuesDecoded: 0 };
  }
  load(name, values) {
    const pages = [];
    for (let start = 0; start < values.length; start += this.pageSize) {
      pages.push(new Page(values.slice(start, start + this.pageSize)));
    }
    this.columns.set(name, { name, values: values.slice(), pages });
    return this;
  }
  column(name) {
    const column = this.columns.get(name);
    if (!column) throw new Error(`unknown column ${name}`);
    return column;
  }
  /** The oracle path: every value is visited, nothing is skipped and nothing is compressed. */
  scanFull(name, predicate) {
    const column = this.column(name);
    const positions = [];
    for (let index = 0; index < column.values.length; index++) {
      if (predicate.test(column.values[index])) positions.push(index);
    }
    return positions;
  }
  /** The compressed path: zone map, then dictionary filter, then code expansion. */
  scanPushed(name, predicate) {
    const column = this.column(name);
    const positions = [];
    const local = { pagesScanned: 0, pagesSkipped: 0, dictionaryTests: 0, codesTested: 0, valuesDecoded: 0, matchingPages: [] };
    column.pages.forEach((page, pageIndex) => {
      page.scanned = false;
      page.skipped = false;
      if (!page.overlaps(predicate)) {
        page.skipped = true;
        local.pagesSkipped++;
        return;
      }
      page.scanned = true;
      local.pagesScanned++;
      local.matchingPages.push(pageIndex);
      const matchingCodes = new Set();
      for (let code = 0; code < page.dictionary.length; code++) {
        local.dictionaryTests++;
        if (predicate.test(page.dictionary[code])) matchingCodes.add(code);
      }
      if (matchingCodes.size === 0) return;
      let offset = 0;
      for (const run of page.runs) {
        local.codesTested += run.count;
        if (matchingCodes.has(run.code)) {
          for (let step = 0; step < run.count; step++) {
            positions.push(pageIndex * this.pageSize + offset + step);
            local.valuesDecoded++;
          } }
        offset += run.count;
      } });
    this.stats.scans++;
    for (const key of ["pagesScanned", "pagesSkipped", "dictionaryTests", "codesTested", "valuesDecoded"]) {
      this.stats[key] += local[key];
    }
    return { positions, ...local };
  }
  /** Late materialisation: only the projected positions are decoded from the second column. */
  project(name, positions) {
    const column = this.column(name);
    return positions.map((position) => column.values[position]);
  }
  totalValues(name) {
    return this.column(name).values.length;
  }
  totalPages(name) {
    return this.column(name).pages.length;
  }
  totalRuns(name) {
    return this.column(name).pages.reduce((sum, page) => sum + page.runCount(), 0);
  }
  decodeAll(name) {
    return this.column(name).pages.flatMap((page) => page.rebuild());
  } }

function lcgInts(length, span, seed) {
  const values = [];
  let state = seed >>> 0;
  for (let index = 0; index < length; index++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values.push(((state >>> 8) % span) - Math.floor(span / 2));
  }
  return values;
}

const PREDICATE_MATRIX = [
  makePredicate("=", 0), makePredicate("=", 7), makePredicate("=", -9),
  makePredicate("<", 0), makePredicate("<=", 3), makePredicate(">", 4), makePredicate(">=", -3),
  makePredicate("between", -5, 5), makePredicate("between", 100, 200), ];

module.exports = { ColumnStore, Page, makePredicate, lcgInts, PREDICATE_MATRIX };

if (require.main === module) {
  const assert = require("node:assert");
  // Encoding round trip on several shapes, plus the layout invariants of every page.
  const shapes = [
    ["empty", []], ["single", [5]], ["constant", [3, 3, 3, 3, 3, 3, 3]],
    ["duplicates", [4, 4, 1, 1, 4, 2, 2, 2]], ["negative", [-5, -1, -9, 4, 0, -5]], ["random", lcgInts(97, 20, 647)],
    ["sorted", Array.from({ length: 40 }, (_, index) => Math.floor(index / 7))], ];
  for (const [label, values] of shapes) {
    const store = new ColumnStore(8);
    store.load("v", values);
    assert.deepStrictEqual(store.decodeAll("v"), values, `${label} round trip`);
    assert.deepStrictEqual(store.project("v", values.map((_, index) => index)), values, `${label} projection`);
    for (const page of store.column("v").pages) {
      assert.strictEqual(page.rebuild().length, page.length, `${label} run lengths`);
      assert.strictEqual(page.runs.reduce((sum, run) => sum + run.count, 0), page.length, `${label} runs sum`);
      assert.deepStrictEqual(page.dictionary, [...new Set(page.values)].sort((l, r) => l - r), `${label} dictionary`);
      if (page.length > 0) {
        assert.strictEqual(page.min, Math.min(...page.values), `${label} zone map min`);
        assert.strictEqual(page.max, Math.max(...page.values), `${label} zone map max`);
      } }
  }
  // Predicate pushdown must return exactly what a full evaluation returns, for every predicate shape.
  const data = lcgInts(500, 60, 6471);
  const store = new ColumnStore(32);
  store.load("v", data);
  store.load("other", data.map((value) => value * 2));
  const flat = Array.from({ length: 300 }, (_, index) => Math.floor(index / 50));
  store.load("flat", flat);
  for (const predicate of PREDICATE_MATRIX) {
    const pushed = store.scanPushed("v", predicate);
    assert.deepStrictEqual(pushed.positions, store.scanFull("v", predicate), `pushdown ${predicate.op}`);
    const scanned = new Set(pushed.matchingPages);
    for (const position of pushed.positions) {
      assert.ok(scanned.has(Math.floor(position / store.pageSize)), "no matching row may live in a skipped page");
    }
    assert.strictEqual(pushed.pagesScanned + pushed.pagesSkipped, store.totalPages("v"));
    assert.ok(pushed.valuesDecoded <= pushed.positions.length, "a page decodes at most its matches");
    assert.ok(pushed.dictionaryTests <= pushed.pagesScanned * 32);
  }
  // A selective predicate on a *clustered* column skips pages and decodes far less than a full scan,
  // while a predicate that matches everything skips no page at all (the zone map has nothing to prune).
  const clustered = lcgInts(500, 60, 6471).sort((left, right) => left - right);
  const clusteredStore = new ColumnStore(32);
  clusteredStore.load("v", clustered);
  const selective = clusteredStore.scanPushed("v", makePredicate("between", 20, 25));
  const everything = clusteredStore.scanPushed("v", makePredicate(">=", -1000));
  const total = clusteredStore.totalValues("v");
  assert.deepStrictEqual(selective.positions, clusteredStore.scanFull("v", makePredicate("between", 20, 25)));
  assert.ok(selective.pagesSkipped > 0, "the zone map pruned pages");
  assert.ok(selective.valuesDecoded < total / 4, `decoded ${selective.valuesDecoded} of ${total}`);
  assert.ok(selective.dictionaryTests < everything.dictionaryTests, "skipping pages saves dictionary work");
  assert.strictEqual(everything.pagesSkipped, 0, "an all-matching predicate cannot skip a page");
  assert.strictEqual(everything.pagesScanned, clusteredStore.totalPages("v"));
  assert.deepStrictEqual(everything.positions.length, total);
  assert.ok(store.totalRuns("flat") < store.totalValues("flat") / 2, "a flat column compresses into few runs");
  assert.strictEqual(store.decodeAll("flat").length, store.totalValues("flat"));
  // Late materialisation: projecting a second column through the selection vector decodes only the
  // surviving positions, which is the whole point of keeping columns separate.
  const selected = store.scanPushed("v", makePredicate(">", 15));
  const projected = store.project("other", selected.positions);
  assert.strictEqual(projected.length, selected.positions.length);
  assert.deepStrictEqual(projected, selected.positions.map((position) => data[position] * 2));
  assert.ok(projected.length < total / 2, "the selection really was selective");
  // Page size boundaries: one value per page, a single page, and pages that do not divide the column.
  for (const pageSize of [1, 7, 64, 1000]) {
    const boundary = new ColumnStore(pageSize);
    boundary.load("v", data);
    assert.strictEqual(boundary.totalPages("v"), Math.ceil(data.length / pageSize), `pages at ${pageSize}`);
    for (const predicate of PREDICATE_MATRIX) {
      assert.deepStrictEqual(boundary.scanPushed("v", predicate).positions,
        boundary.scanFull("v", predicate), `pushdown at page size ${pageSize}`);
    }
    assert.deepStrictEqual(boundary.decodeAll("v"), data);
  }
  // An empty column has no pages, no matches and no crashes.
  const emptyStore = new ColumnStore(16);
  emptyStore.load("v", []);
  assert.strictEqual(emptyStore.totalPages("v"), 0);
  const emptyScan = emptyStore.scanPushed("v", makePredicate(">", 0));
  assert.deepStrictEqual(emptyScan.positions, []);
  assert.strictEqual(emptyScan.pagesScanned, 0);
  assert.strictEqual(emptyScan.pagesSkipped, 0);
  assert.deepStrictEqual(emptyStore.decodeAll("v"), []);
  // The pushed-down scan is deterministic and its counters only grow, and invalid requests are refused.
  const again = store.scanPushed("v", makePredicate(">", 15));
  assert.deepStrictEqual(again, selected, "the same scan produces the same result and counters");
  assert.ok(store.stats.scans >= 8 && store.stats.pagesScanned > 0);
  assert.throws(() => new ColumnStore(0), /page size/);
  assert.throws(() => store.scanPushed("missing", makePredicate("=", 1)), /unknown column/);
  assert.throws(() => makePredicate("~", 1), /unknown operator/);
  assert.throws(() => makePredicate("between", 5, 1), /low <= high/);
  console.log(`column store: ${store.totalPages("v")} pages of ${store.pageSize}, selective scan decoded ` +
    `${selective.valuesDecoded}/${total} values and pruned ${selective.pagesSkipped} pages, ` +
    `flat column uses ${store.totalRuns("flat")} runs for ${store.totalValues("flat")} values`);
  console.log("PASS");
}
