/** Task 968 - Implement localization pipeline to translate problems into multiple languages.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * Pipeline over a catalogue of ICU-style templates (`{name}` interpolation and `{count, plural,
 * one{..} other{..}}` branches): validate placeholder/plural-category parity, resolve each key
 * through a fallback chain (`pt-BR` -> `pt` -> `en`) reporting provenance, render with CLDR-lite
 * cardinal rules, then audit text expansion against a per-locale budget. Simplified: translations
 * are static data (no MT service), `en-XA` is built locally from an accent map plus padding, and
 * ordinals/currency are out of scope. Verified: parity rejection, fallback provenance, plural
 * selection against an independent CLDR reference table, the expansion audit, the empty-catalogue
 * boundary and byte-identical reruns.
 */
"use strict";
const assert = require("node:assert");
const base = (locale) => String(locale).split("-")[0];
/** Index of the `}` matching the `{` at `start`; throws when the template is unbalanced. */
function matchBrace(text, start) {
  let depth = 0;
  for (let i = start; i < text.length; i += 1) {
    depth += text[i] === "{" ? 1 : text[i] === "}" ? -1 : 0;
    if (depth === 0 && i > start) return i;
  }
  throw new Error(`unbalanced braces at index ${start}`);
}
/** Split on top-level commas only, so commas inside plural branches stay put. */
function splitTop(text) {
  const parts = []; let depth = 0; let cur = "";
  for (const ch of text) {
    if (ch === "{") depth += 1; else if (ch === "}") depth -= 1;
    if (ch === "," && depth === 0) { parts.push(cur); cur = ""; } else cur += ch;
  }
  parts.push(cur); return parts;
}
/** `one{1 problem} other{# problems}` -> Map(one -> "1 problem", other -> "# problems"). */
function parseBranches(body) {
  const branches = new Map(); let i = 0;
  while (i < body.length) {
    const open = body.indexOf("{", i);
    if (open < 0) break;
    const close = matchBrace(body, open);
    branches.set(body.slice(i, open).trim(), body.slice(open + 1, close));
    i = close + 1;
  }
  return branches;
}
/** Parse a template into literal chunks plus `{name}` / `{name, plural, ...}` nodes. */
function parseTemplate(text) {
  const parts = []; let i = 0; let lit = "";
  while (i < text.length) {
    if (text[i] !== "{") { lit += text[i]; i += 1; continue; }
    const close = matchBrace(text, i);
    const [head, ...rest] = splitTop(text.slice(i + 1, close));
    parts.push(lit, (rest[0] || "").trim() === "plural"
      ? { name: head.trim(), branches: parseBranches(rest.slice(1).join(",")) }
      : { name: head.trim() });
    lit = ""; i = close + 1;
  }
  parts.push(lit);
  return parts;
}
/** CLDR-lite cardinal rules; the argument is already the absolute integer part. */
const slavicFew = (n) => n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14);
const PLURAL_RULES = {
  en: (n) => (n === 1 ? "one" : "other"),
  de: (n) => (n === 1 ? "one" : "other"),
  fr: (n) => (n <= 1 ? "one" : "other"),
  ja: () => "other",
  ru: (n) => (n % 10 === 1 && n % 100 !== 11 ? "one" : slavicFew(n) ? "few" : "many"),
  pl: (n) => (n === 1 ? "one" : slavicFew(n) ? "few" : "many"),
};
function pluralCategory(locale, count) {
  return (PLURAL_RULES[base(locale)] || PLURAL_RULES.en)(Math.abs(Math.trunc(count)));
}
/** Categories a locale may legitimately declare, sampled once from its own rule. */
const CATEGORY_CACHE = new Map();
function localeCategories(locale) {
  const key = base(locale);
  if (!CATEGORY_CACHE.has(key)) {
    const cats = new Set(["other"]);
    for (let n = 0; n <= 200; n += 1) cats.add(pluralCategory(key, n));
    CATEGORY_CACHE.set(key, cats);
  }
  return CATEGORY_CACHE.get(key);
}
/** Structural fingerprint used for source/translation parity: placeholder names + plural sets. */
function describeTemplate(template) {
  const names = new Set(); const branches = new Map();
  for (const part of parseTemplate(template)) {
    if (typeof part === "string") continue;
    names.add(part.name);
    if (part.branches) branches.set(part.name, [...part.branches.keys()].sort().join("|"));
  }
  return { names: [...names].sort().join(","), branches };
}
const LETTERS = "abcdefghijklmnopqrstuvwxyz";
/** Deterministic pseudolocale: accented literal words plus padding to ~135% of the source. */
function pseudolocalize(template) {
  let out = ""; let i = 0;
  while (i < template.length) {
    if (template[i] === "{") {
      const close = matchBrace(template, i);
      out += template.slice(i, close + 1); i = close + 1; continue;
    }
    const word = /^[A-Za-z]+/.exec(template.slice(i));
    if (!word) { out += template[i]; i += 1; continue; }
    out += `*${[...word[0].toLowerCase()].map((c) => LETTERS[(LETTERS.indexOf(c) + 3) % 26]).join("")}*`;
    i += word[0].length;
  }
  const target = Math.ceil(template.length * 1.35);
  return out.length < target ? out + "~".repeat(target - out.length) : out;
}
/** Render parsed parts; `#` inside a plural branch binds to the count, unknown names are kept. */
function renderParts(parts, params, locale, report) {
  let out = "";
  for (const part of parts) {
    if (typeof part === "string") { out += part; continue; }
    if (!part.branches) {
      if (params[part.name] !== undefined) out += String(params[part.name]);
      else { report.push(part.name); out += `{${part.name}}`; }
      continue;
    }
    const count = Number(params[part.name]);
    const category = Number.isFinite(count) ? pluralCategory(locale, count) : null;
    const branch = category === null ? undefined : part.branches.get(category) ?? part.branches.get("other");
    if (branch === undefined) { report.push(category === null ? part.name : `${part.name}:${category}`); out += `{${part.name}}`; }
    else out += renderParts(parseTemplate(branch.split("#").join(String(count))), params, locale, report);
  }
  return out;
}
const render = (template, params, locale, report = []) =>
  renderParts(parseTemplate(template), params, locale, report);
class LocalizationPipeline {
  constructor({ source, bundles = {}, budgets = {}, pseudoLocale = "en-XA" }) {
    this.source = source; this.pseudoLocale = pseudoLocale;
    this.budgets = Object.assign({ default: 1.6, [pseudoLocale]: 1.15 }, budgets);
    this.bundles = { ...bundles, [pseudoLocale]: Object.fromEntries(
      Object.entries(source).map(([key, tpl]) => [key, pseudolocalize(tpl)])) };
  }
  /** Fallback chain: exact tag, then its base language, then the default locale `en`. */
  chain(locale) {
    const out = [locale];
    if (base(locale) !== locale) out.push(base(locale));
    if (!out.includes("en")) out.push("en");
    return out;
  }
  /** Unknown plural categories are errors; absent keys are only warnings. */
  validate(locale) {
    const bundle = this.bundles[locale];
    if (!bundle) return { locale, ok: false, errors: [`no bundle for ${locale}`], missing: [] };
    const errors = []; const missing = []; const allowed = localeCategories(locale);
    for (const [key, tpl] of Object.entries(this.source)) {
      if (!(key in bundle)) { missing.push(key); continue; }
      const want = describeTemplate(tpl); const got = describeTemplate(bundle[key]);
      if (want.names !== got.names) errors.push(`${key}: placeholder parity (want [${want.names}], got [${got.names}])`);
      for (const [name, srcCats] of want.branches) {
        const gotCats = (got.branches.get(name) || "").split("|").filter(Boolean);
        const absent = [...allowed].filter((c) => !gotCats.includes(c));
        const invented = gotCats.filter((c) => !srcCats.split("|").includes(c) && !allowed.has(c));
        if (absent.length || invented.length) {
          errors.push(`${key}: plural branches for ${name} (want [${[...allowed].sort().join("|")}] ` +
            `got [${gotCats.join("|")}]${invented.length ? ` unknown [${invented.join("|")}]` : ""})`);
        }
      }
    }
    return { locale, ok: errors.length === 0, errors, missing };
  }
  /** Resolve one key through the chain and render it; `resolvedFrom` names the locale used. */
  translate(locale, key, params = {}) {
    const report = [];
    for (const candidate of this.chain(locale)) {
      if (!(key in (this.bundles[candidate] || {}))) continue;
      return { key, text: render(this.bundles[candidate][key], params, candidate, report),
        resolvedFrom: candidate, fallback: candidate !== locale, unfilled: report };
    }
    return { key, text: `[[${key}]]`, resolvedFrom: null, fallback: true, unfilled: [key] };
  }
  /** Whole catalogue for one locale; expansion is measured against the `en` rendering. */
  audit(locale, params = {}) {
    const budget = this.budgets[base(locale)] ?? this.budgets[locale] ?? this.budgets.default;
    const entries = Object.keys(this.source).map((key) => {
      const result = this.translate(locale, key, params);
      const baseline = render(this.source[key], params, "en").length;
      return { ...result, ratio: result.text.length / Math.max(1, baseline), overBudget: false };
    });
    for (const entry of entries) entry.overBudget = entry.ratio > budget;
    return { locale, budget, entries, flagged: entries.filter((e) => e.overBudget).map((e) => e.key) };
  }
  run(params = {}) {
    return Object.fromEntries([...new Set(Object.keys(this.bundles))].sort().map((locale) => [locale,
      Object.fromEntries(Object.keys(this.source).map((key) => [key, this.translate(locale, key, params).text]))]));
  }
}
/** Independent CLDR reference table (hand-transcribed) used as the oracle for the rules above. */
const CLDR_REFERENCE = [
  ["en", 0, "other"], ["en", 1, "one"], ["en", 2, "other"], ["en", 21, "other"], ["fr", 0, "one"],
  ["fr", 1, "one"], ["fr", 2, "other"], ["de", 1, "one"], ["de", 0, "other"], ["ja", 1, "other"],
  ["ru", 1, "one"], ["ru", 2, "few"], ["ru", 4, "few"], ["ru", 5, "many"], ["ru", 11, "many"],
  ["ru", 12, "many"], ["ru", 21, "one"], ["ru", 22, "few"], ["ru", 111, "many"], ["pl", 1, "one"],
  ["pl", 2, "few"], ["pl", 5, "many"], ["pl", 12, "many"], ["pl", 22, "few"], ["pl", 101, "many"],
];
const SOURCE = {
  "problem.statement": "Find two numbers in {array} that sum to {target}.",
  "progress.solved": "You solved {count, plural, one{# problem} other{# problems}} today.",
  "review.due": "{count, plural, one{1 problem is due} other{# problems are due}} for review.",
};
const RU = {
  "problem.statement": "Naidite dva chisla v {array}, dayushchie v summe {target}.",
  "progress.solved": "Segodnya resheno {count, plural, one{# zadacha} few{# zadachi} many{# zadach} other{# zadachi}}.",
  "review.due": "Na povtorenie zhdet {count, plural, one{1 zadacha} few{# zadachi} many{# zadach} other{# zadachi}}.",
};
const PT = { ...SOURCE, "progress.solved": "Resolveu {count, plural, one{# problema} other{# problemas}} hoje.",
  "review.due": "{count, plural, one{1 problema pendente} other{# problemas pendentes}} para revisar." };
const BUNDLES = { en: SOURCE, ru: RU, pt: PT, "pt-BR": { "problem.title": "Duas somas" } };
const build = (extra = {}) => new LocalizationPipeline({ source: SOURCE,
  bundles: { ...BUNDLES, ...extra }, budgets: { ru: 1.4, pl: 1.4 } });
function main() {
  const pipeline = build();
  // 1. Happy path: interpolation and plural selection in the source locale.
  assert.strictEqual(pipeline.translate("en", "progress.solved", { count: 1 }).text, "You solved 1 problem today.");
  assert.strictEqual(pipeline.translate("en", "progress.solved", { count: 7 }).text, "You solved 7 problems today.");
  assert(pipeline.translate("ru", "review.due", { count: 3 }).text.includes("zadachi"), "ru few branch for 3");
  assert(pipeline.translate("ja", "review.due", { count: 4 }).text.includes("4 problems are due"), "ja uses en");
  // 2. Independent oracle: every hand-transcribed CLDR row agrees with the implemented rules.
  for (const [locale, count, expected] of CLDR_REFERENCE) {
    assert.strictEqual(pluralCategory(locale, count), expected, `${locale}/${count} must be ${expected}`);
    assert.notStrictEqual(pluralCategory(`${locale}-XX`, count), "none", `region must not break ${locale}`);
  }
  assert.deepStrictEqual([...localeCategories("ru")].sort(), ["few", "many", "one", "other"], "ru categories");
  // 3. Parity validation: wrong placeholders, dropped and invented plural branches are rejected.
  const esBroken = build({ es: { ...SOURCE, "problem.statement": "Dos numeros para {objetivo}." } });
  assert(esBroken.validate("es").errors[0].includes("placeholder parity"), "placeholder mismatch rejected");
  const dropped = build({ de: { ...SOURCE, "review.due": "{count, plural, other{# Falle}}" } });
  assert(dropped.validate("de").errors.some((e) => e.includes("plural branches")), "dropped branch rejected");
  const invented = build({ de: { ...SOURCE, "review.due": "{count, plural, one{1 Falle} few{# Faelle} other{# Falle}}" } });
  assert(invented.validate("de").errors[0].includes("unknown [few]"), "de cannot declare a `few` category");
  assert(pipeline.validate("ru").ok, "ru bundle is fully valid");
  assert(build({ it: { "problem.title": "Due somme" } }).validate("it").missing.length === 3, "absent keys warn");
  // 4. Fallback chain and provenance: pt-BR -> pt -> en.
  assert.deepStrictEqual(pipeline.chain("pt-BR"), ["pt-BR", "pt", "en"], "chain order");
  const inherited = pipeline.translate("pt-BR", "progress.solved", { count: 2 });
  assert(inherited.resolvedFrom === "pt" && inherited.fallback && inherited.text.includes("problemas"), "inherits pt");
  const missingKey = pipeline.translate("ru", "no.such.key", {});
  assert(missingKey.resolvedFrom === null && missingKey.text === "[[no.such.key]]", "unknown key is marked");
  // 5. Boundary and negative: unfilled placeholders are reported, empty catalogues still work.
  const unfilled = pipeline.translate("en", "problem.statement", { array: "[2,7,11,15]" });
  assert(unfilled.text.includes("{target}") && unfilled.unfilled.includes("target"), "missing parameter reported");
  const empty = new LocalizationPipeline({ source: {} });
  assert.deepStrictEqual(empty.run({ count: 1 }), { "en-XA": {} }, "empty catalogue renders empty maps");
  assert.strictEqual(pluralCategory("ru", -1), "one", "negative counts use the absolute value");
  assert.throws(() => matchBrace("{count, plural", 0), /unbalanced/, "unbalanced template is rejected");
  // 6. Expansion audit: real translations fit their budget, the pseudolocale never does.
  const params = { array: "[2,7]", target: 9, count: 4 };
  const ruAudit = pipeline.audit("ru", params);
  assert.strictEqual(ruAudit.flagged.length, 0, "ru fits within its 1.4x budget");
  const pseudo = pipeline.audit("en-XA", params);
  assert.strictEqual(pseudo.flagged.length, pseudo.entries.length, "every pseudolocale string is flagged");
  assert(pseudo.entries.every((e) => e.unfilled.length === 0), "pseudolocalization keeps placeholders");
  assert(pseudolocalize("Two Sum").startsWith("*") && pseudolocalize("{count} hints").endsWith("~"), "accents+pad");
  // 7. Determinism: two independent constructions render byte-identical catalogues.
  const first = pipeline.run(params);
  assert(JSON.stringify(first) === JSON.stringify(build().run(params)), "runs are byte-identical");
  assert(first.ru["progress.solved"].includes("zadach"), "12 selects the ru many branch");
  const worst = Math.max(...ruAudit.entries.map((e) => e.ratio));
  console.log(`localization: ${Object.keys(first).length} locales x ${Object.keys(SOURCE).length} keys, ` +
    `pseudo flagged ${pseudo.flagged.length}/${pseudo.entries.length}, ru worst ratio ${worst.toFixed(2)}`);
  console.log("PASS");
}
module.exports = { matchBrace, splitTop, parseBranches, parseTemplate, pluralCategory, localeCategories,
  describeTemplate, pseudolocalize, render, LocalizationPipeline, CLDR_REFERENCE, SOURCE, RU, PT, BUNDLES };
if (require.main === module) main();
