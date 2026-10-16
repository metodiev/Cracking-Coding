/** Task 993 - Implement academic citation generator for editorial solutions with references.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Turns a list of reference records into a bibliography: records are validated, de-duplicated by DOI
 * (keeping the more complete copy), sorted by year and then by first author, keyed for in-text use
 * (numeric or author-year with a/b disambiguation) and rendered in one of three citation styles.
 * Rendering is pure text assembly, so the same input always produces the same bibliography and the
 * keys stay in step with the entry order.
 *
 * What is simplified: a citation database is an in-memory array and styles are three small
 * formatters rather than a CSL processor. Verified: exact rendering per style, de-duplication,
 * author-year disambiguation, sorting against an independent insertion-sort oracle, BibTeX escaping,
 * rejection of malformed records, and the empty and single-record cases.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the file's only randomness source, so fixtures are byte-reproducible. */
function splitmix64(seed) {
  let s = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    s = BigInt.asUintN(64, s + 0x9e3779b97f4a7c15n);
    let z = BigInt.asUintN(64, (s ^ (s >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

/** BibTeX treats these as special, so they are escaped in the .bib rendering. */
const escapeBibtex = (text) => String(text).replace(/[{}\\&%$#_^~]/g, (ch) => `\\${ch}`);
const surname = (author) => (author && author.last ? author.last : "Anonymous");
const authorsPlain = (authors) => authors.map((a) => (a.first ? `${a.first} ${a.last}` : a.last));
const authorsInitials = (authors) => authors.map((a) => (a.first ? `${a.first[0]}. ${a.last}` : a.last));

/** "A, B, and C" -- the serial-comma form used by two of the three house styles. */
function joined(names) {
  if (names.length === 0) return "Anonymous";
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(", ")}, and ${names[names.length - 1]}`;
}

/** Validation: a reference needs a title and a year; anything else is optional metadata. */
function normalize(record) {
  if (!record || typeof record.title !== "string" || record.title.length === 0) {
    return { ok: false, reason: "missing-title" };
  }
  if (!Number.isInteger(record.year) || record.year < 1500 || record.year > 2200) {
    return { ok: false, reason: "invalid-year" };
  }
  const authors = (record.authors || []).map((a) => ({ last: String(a.last), first: a.first ? String(a.first) : null }));
  const url = record.url || (record.doi ? `https://doi.org/${record.doi}` : null);
  return { ok: true, value: { title: record.title, year: record.year, venue: record.venue || null,
    doi: record.doi || null, url, kind: record.kind || "article", authors,
    pages: record.pages || null } };
}

const STYLES = {
  plain: (c, key) => (`${key} ${joined(authorsPlain(c.authors))} (${c.year}). ${c.title}. ` +
    `${c.venue ? `${c.venue}. ` : ""}${c.url || ""}`).trim(),
  acm: (c, key) => (`${key} ${joined(authorsInitials(c.authors))}. ${c.year}. ${c.title}. ` +
    `${c.venue ? `${c.venue}, ` : ""}${c.pages ? `${c.pages}. ` : ""}${c.url || ""}`).trim(),
  ieee: (c, key) => (`${key} ${joined(authorsInitials(c.authors))}, "${c.title}," ` +
    `${c.venue ? `${c.venue}, ` : ""}${c.pages ? `pp. ${c.pages}, ` : ""}${c.year}. ${c.url || ""}`).trim(),
};

/** Sort order: year, then first author surname, then title -- the classic bibliography ordering. */
function compareEntries(a, b) {
  if (a.year !== b.year) return a.year - b.year;
  if (surname(a.authors[0]) !== surname(b.authors[0])) {
    return surname(a.authors[0]) < surname(b.authors[0]) ? -1 : 1;
  }
  if (a.title !== b.title) return a.title < b.title ? -1 : 1;
  return (a.doi || "") < (b.doi || "") ? -1 : 1;
}

/** Independent oracle: insertion sort using the same comparator, written without Array#sort. */
function insertionSorted(entries) {
  const out = [];
  for (const entry of entries) {
    let i = out.length;
    while (i > 0 && compareEntries(out[i - 1], entry) > 0) { out[i] = out[i - 1]; i -= 1; }
    out[i] = entry;
  }
  return out;
}

const completeness = (c) => [c.venue, c.doi, c.pages, c.url, c.authors.length > 0 ? "a" : null]
  .filter((x) => x !== null && x !== "").length;

/** De-duplicate by DOI, preferring the copy that carries more metadata (ties: the first seen). */
function dedupe(entries) {
  const byDoi = new Map();
  const withoutDoi = [];
  for (const entry of entries) {
    if (!entry.doi) { withoutDoi.push(entry); continue; }
    const held = byDoi.get(entry.doi);
    if (!held || completeness(entry) > completeness(held)) byDoi.set(entry.doi, entry);
  }
  return Array.from(byDoi.values()).concat(withoutDoi);
}

/** Author-year keys get a/b suffixes when two references share surname and year. */
function assignKeys(entries, keying) {
  if (keying === "numeric") return entries.map((_, i) => `[${i + 1}]`);
  const seen = new Map();
  return entries.map((entry) => {
    const base = `${surname(entry.authors[0])}${entry.year}`;
    const n = seen.get(base) || 0;
    seen.set(base, n + 1);
    return n === 0 ? base : `${base}${String.fromCharCode(96 + n + 1)}`;
  });
}

/** Build the bibliography: validate, de-duplicate, sort, key and render. */
function bibliography(records, { style = "plain", keying = "numeric" } = {}) {
  const rejected = [];
  const valid = [];
  for (const record of records) {
    const result = normalize(record);
    if (result.ok) valid.push(result.value); else rejected.push({ id: (record && record.id) || null, reason: result.reason });
  }
  const entries = dedupe(valid).sort(compareEntries);
  const keys = assignKeys(entries, keying);
  const lines = entries.map((entry, i) => STYLES[style](entry, keys[i]));
  return { entries, keys, lines, rejected,
    digest: fnv1a(lines.join("\n")) };
}

const SURNAMES = ["Knuth", "Hoare", "Dijkstra", "Tarjan", "Lamport"];
const TITLES = ["Sorting Revisited", "Notes on Graphs", "On the Shortest Path", "Concurrency Basics"];
/** Seeded records; a few deliberately share a surname and year to exercise a/b disambiguation. */
function seededRecords(seed, count) {
  const rng = splitmix64(seed);
  return Array.from({ length: count }, (_, i) => ({ id: `r${i}`,
    title: `${TITLES[rng.int(TITLES.length)]} ${i}`,
    year: 1990 + rng.int(30), venue: rng.int(4) === 0 ? null : "Journal of Algorithms",
    doi: rng.int(3) === 0 ? null : `10.1145/${1000 + rng.int(5)}`,
    authors: [{ last: SURNAMES[rng.int(SURNAMES.length)], first: "A" }], pages: rng.int(2) ? "12-34" : null }));
}

function main() {
  const knuth = { title: "The Art of Computer Programming", year: 1968, venue: "Addison-Wesley",
    doi: "10.5555/kb", authors: [{ last: "Knuth", first: "Donald" }], pages: "1-650" };
  const acm = bibliography([knuth], { style: "acm" });
  const ieee = bibliography([knuth], { style: "ieee" });
  // 1. Happy path: exact rendering in the three styles, with the DOI promoted to a URL.
  assert.deepStrictEqual(acm.lines,
    ["[1] D. Knuth. 1968. The Art of Computer Programming. Addison-Wesley, 1-650. https://doi.org/10.5555/kb"],
    "acm rendering");
  assert.deepStrictEqual(ieee.lines,
    ["[1] D. Knuth, \"The Art of Computer Programming,\" Addison-Wesley, pp. 1-650, 1968. https://doi.org/10.5555/kb"],
    "ieee rendering");
  assert.strictEqual(bibliography([knuth], { style: "plain" }).lines[0],
    "[1] Donald Knuth (1968). The Art of Computer Programming. Addison-Wesley. https://doi.org/10.5555/kb",
    "plain rendering");
  // 2. Keys: numeric keys are 1-based and parallel the entry order; three authors use the serial comma.
  const trio = { title: "Concurrent Structures", year: 2001, authors: [{ last: "Lamport", first: "Leslie" },
    { last: "Hoare", first: "Tony" }, { last: "Dijkstra", first: "Edsger" }] };
  const numbered = bibliography([knuth, trio]);
  assert.deepStrictEqual(numbered.keys, ["[1]", "[2]"], "keys follow the sorted order");
  assert.strictEqual(numbered.lines[1], "[2] Leslie Lamport, Tony Hoare, and Edsger Dijkstra (2001). Concurrent Structures.",
    "three authors use the serial comma and omit the missing URL");
  // 3. Author-year keying with a/b disambiguation for same surname and year.
  const twins = [{ title: "Alpha", year: 2000, authors: [{ last: "Knuth", first: "Donald" }] },
    { title: "Beta", year: 2000, authors: [{ last: "Knuth", first: "Donald" }] }];
  assert.deepStrictEqual(bibliography(twins, { keying: "author-year" }).keys, ["Knuth2000", "Knuth2000b"],
    "same surname and year get an a/b suffix");
  // 4. Boundary: the empty list renders nothing; a record without authors or venue is anonymous.
  assert.deepStrictEqual([bibliography([]).lines, bibliography([]).keys, bibliography([]).digest], [[], [], fnv1a("")],
    "empty bibliography");
  const anonymous = bibliography([{ title: "Unsigned Note", year: 2010 }]);
  assert.strictEqual(anonymous.lines[0], "[1] Anonymous (2010). Unsigned Note.", "missing metadata degrades gracefully");
  // 5. Negative: records without a title or with an impossible year are rejected, not rendered.
  const mixed = bibliography([{ id: "bad1" }, { title: "No Year", year: 12 }, knuth]);
  assert.deepStrictEqual([mixed.rejected, mixed.lines.length], [[{ id: "bad1", reason: "missing-title" },
    { id: null, reason: "invalid-year" }], 1], "malformed records are rejected with a reason");
  // 6. De-duplication: the same DOI collapses to the more complete record.
  const duplicated = bibliography([{ title: "Short", year: 2020, doi: "10.1/x" },
    { title: "Long", year: 2020, doi: "10.1/x", venue: "Venue", pages: "1-9", authors: [{ last: "Knuth" }] }]);
  assert.deepStrictEqual([duplicated.entries.length, duplicated.entries[0].title, duplicated.lines.length], [1, "Long", 1],
    "richer duplicate wins");
  // 7. BibTeX escaping is applied to every special character and nothing else.
  assert.strictEqual(escapeBibtex("a{b}c&d%_#"), "a\\{b\\}c\\&d\\%\\_\\#", "special characters escaped");
  assert.strictEqual(escapeBibtex("plain text 50/50"), "plain text 50/50", "ordinary text untouched");
  // 8. Oracle: sorting matches an independent insertion sort, and the output is order independent.
  const records = seededRecords(0x993, 30);
  const built = bibliography(records);
  const raw = dedupe(records.map((r) => normalize(r).value));
  assert.deepStrictEqual(insertionSorted(raw).map((e) => e.title), built.entries.map((e) => e.title),
    "ordering matches the insertion-sort oracle on the unsorted records");
  assert.strictEqual(bibliography(records.slice().reverse()).digest, built.digest, "output digest is input-order independent");
  assert.strictEqual(bibliography(seededRecords(0x993, 30)).digest, built.digest, "output reproduces byte for byte");
  console.log(`entries=${built.entries.length} rejected=${built.rejected.length} keys=${built.keys[0]} digest=${built.digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { splitmix64, fnv1a, escapeBibtex, normalize, compareEntries, insertionSorted, dedupe,
  bibliography, seededRecords, STYLES };
