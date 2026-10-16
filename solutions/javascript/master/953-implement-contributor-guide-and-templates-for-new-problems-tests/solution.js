/** Task 953 - Implement contributor guide and templates for new problems, tests, and solutions.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * The scaffolding half of the repo: a slug rule, placeholder templates for the spec / README row /
 * per-language solution skeleton, a validator that a rendered skeleton actually satisfies the repo
 * contract, and a guide builder whose table of contents gets stable unique anchors with no dangling
 * internal link.
 *
 * Simplified: nothing is written to disk -- files are rendered in memory as strings and the layout is
 * checked with regexes against the same rules `solution-style.md` states (folder = NNN-slug, one file
 * per folder, `"use strict"` + module.exports + a `require.main === module` self-check block that
 * prints PASS). Verified: the slug rule incl. the 64-character truncation boundary against an
 * independent character-loop oracle, template rendering completeness, rejection of an unknown
 * placeholder, contract validation accept/reject, layout naming, anchor uniqueness plus dangling-link
 * detection, and byte-identical re-rendering.
 */
"use strict";
const assert = require("node:assert");

const PLACEHOLDER = /\{\{([A-Za-z][A-Za-z0-9_]*)\}\}/g;
const LANGUAGE_FILES = { python: "solution.py", java: "Solution.java", javascript: "solution.js", csharp: "Solution.cs" };
const MAX_SLUG = 64;

/** Repo slug rule: lower-case, runs of non-alphanumerics become one dash, trimmed, truncated to 64. */
function slugify(title, limit = MAX_SLUG) {
  return title.normalize("NFKD").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "")
    .slice(0, limit).replace(/-+$/, "");
}
const placeholderNames = (template) => [...new Set([...template.matchAll(PLACEHOLDER)].map((m) => m[1]))].sort();

/** Render a template; an unknown or unsubstituted placeholder is a hard error, never silent output. */
function renderTemplate(template, vars) {
  const rendered = template.replace(PLACEHOLDER, (_, key) => {
    if (!Object.hasOwn(vars, key)) throw new Error(`unknown placeholder {{${key}}}`);
    return String(vars[key]);
  });
  if (rendered.includes("{{")) throw new Error("unsubstituted placeholder left in output");
  return rendered;
}

const TEMPLATES = {
  spec: "# {{number}} - {{title}}.\n\n- **Number:** {{number}}\n- **Tier:** {{tierLabel}}\n"
    + "- **CTCI chapter:** {{chapter}}\n\n## Solutions\n\n{{solutionList}}\n",
  readmeRow: "- **{{number}}.** {{title}} - [spec](tasks/{{tier}}/{{folder}}.md), "
    + "[solution](solutions/javascript/{{tier}}/{{folder}}/solution.js)\n",
  solution: "/** Task {{number}} - {{title}}. (Tier: {{tierLabel}}, {{chapter}}) */\n\"use strict\";\n"
    + "const assert = require(\"node:assert\");\n\nfunction main() {\n  // {{checkCount}} checks\n  assert.ok(true);\n"
    + "  console.log(\"PASS\");\n}\n\nmodule.exports = { main };\nif (require.main === module) main();\n",
  test: "// Task {{number}} - {{title}}: randomized battery for {{folder}}\n// oracle: {{oracle}}\n",
};
const TIERS = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced", expert: "Expert", master: "Master / Challenge" };

/** Contract check for a JavaScript skeleton; returns every violation instead of the first one. */
function validateSolutionSource(source) {
  const problems = [];
  if (!source.startsWith("/** Task ")) problems.push("missing leading task comment block");
  if (!source.includes('"use strict";')) problems.push('missing "use strict";');
  if (!/(module\.exports\s*=)|(export\s)/.test(source)) problems.push("no module.exports");
  if (!source.includes("require.main === module")) problems.push("no self-check entry point");
  if (!source.includes('console.log("PASS")')) problems.push("does not print PASS");
  if (/\bTODO\b|\bFIXME\b/.test(source)) problems.push("contains a TODO/FIXME marker");
  if (!/^\/\*\* Task \d{3} - .+\. \(Tier: [^,]+, [^)]+\) \*\/$/m.test(source)) {
    problems.push("header comment does not match the house style");
  }
  return problems;
}

const folderName = (number, slug) => `${String(number).padStart(3, "0")}-${slug}`;
const solutionPath = (lang, tier, folder) => `solutions/${lang}/${tier}/${folder}/${LANGUAGE_FILES[lang]}`;
const LAYOUT = /^(tasks|solutions)\/(python|java|javascript|csharp|beginner|intermediate|advanced|expert|master)\//;

/** Render every artifact for one new problem and check the layout of each generated path. */
function buildContribution({ number, tier, title, chapter, oracle = "brute force", languages = Object.keys(LANGUAGE_FILES) }) {
  const slug = slugify(title);
  const folder = folderName(number, slug);
  const vars = { number, slug, folder, title, chapter, tier, tierLabel: TIERS[tier], oracle,
    checkCount: 8, solutionList: languages.map((lang) => `- \`${solutionPath(lang, tier, folder)}\``).join("\n") };
  const files = [
    { path: `tasks/${tier}/${folder}.md`, content: renderTemplate(TEMPLATES.spec, vars) },
    { path: `tasks/${tier}/${folder}.tests.md`, content: renderTemplate(TEMPLATES.test, vars) },
    { path: `README.md row`, content: renderTemplate(TEMPLATES.readmeRow, vars) },
    ...languages.map((lang) => ({ path: solutionPath(lang, tier, folder), content: renderTemplate(TEMPLATES.solution, vars) })),
  ];
  const problems = files.flatMap((file) => {
    const issues = [];
    if (file.path !== "README.md row" && !LAYOUT.test(file.path)) issues.push(`${file.path}: unexpected layout`);
    if (file.path !== "README.md row" && !file.path.includes(folder)) issues.push(`${file.path}: folder mismatch`);
    if (file.path.endsWith(".js")) issues.push(...validateSolutionSource(file.content).map((p) => `${file.path}: ${p}`));
    return issues;
  });
  return { slug, folder, vars, files, problems, paths: files.map((f) => f.path).sort() };
}

/** Guide builder: heading -> unique anchor, and every internal `#link` must resolve to an anchor. */
function buildGuide(sections) {
  const anchors = new Set();
  const lines = ["# Contributor guide", ""];
  for (const { title, level, body } of sections) {
    let anchor = slugify(title, 64);
    for (let n = 2; anchors.has(anchor); n += 1) anchor = `${slugify(title, 58)}-${n}`;
    anchors.add(anchor);
    lines.push(`<a id="${anchor}"></a>`, "", `${"#".repeat(level)} ${title}`, "", body, "");
  }
  const links = [...lines.join("\n").matchAll(/\]\(#([a-z0-9-]+)\)/g)].map((m) => m[1]);
  return { markdown: lines.join("\n"), anchors, links, dangling: links.filter((l) => !anchors.has(l)) };
}

function main() {
  // 1. Slug rule on hand-picked titles, including punctuation runs and the 64-character boundary.
  assert.strictEqual(slugify("Implement  end-to-end!!! harness"), "implement-end-to-end-harness", "punctuation collapses");
  assert.strictEqual(slugify("  Trailing  "), "trailing", "trim of leading/trailing dashes");
  const long = slugify(`${"a".repeat(40)} ${"b".repeat(40)}`);
  assert(long.length <= MAX_SLUG && !long.endsWith("-") && long.includes("-"), `truncated to ${long.length}`);
  // 2. Independent oracle: a character loop must agree with the regex slug rule on 60 seeded titles.
  const oracleSlug = (title, limit = MAX_SLUG) => {
    let out = "";
    for (const ch of title.toLowerCase()) out += /^[a-z0-9]$/.test(ch) ? ch : "-";
    return out.replace(/-+/g, "-").replace(/^-|-$/g, "").slice(0, limit).replace(/-+$/, "");
  };
  const rng = makeRng(20240953);
  const words = ["Implement", "fast", "O(n)", "graf", "traversal", "with", "hashmap", "edge", "cases"];
  let agreements = 0;
  for (let trial = 0; trial < 60; trial += 1) {
    const title = Array.from({ length: 1 + rng.int(8) }, () => words[rng.int(words.length)]).join(" ");
    if (slugify(title) === oracleSlug(title)) agreements += 1;
  }
  assert.strictEqual(agreements, 60, `slug oracle disagreements: ${60 - agreements}`);
  // 3. Templates substitute every placeholder and declare exactly the vars the builder supplies.
  const contribution = buildContribution({ number: 953, tier: "master", title: "Implement contributor guide", chapter: "Beyond" });
  assert.strictEqual(contribution.slug, "implement-contributor-guide", "derived slug");
  assert.strictEqual(contribution.folder, "953-implement-contributor-guide", "derived folder");
  assert(!contribution.files.some((f) => f.content.includes("{{")), "no placeholder survives rendering");
  const declared = Object.values(TEMPLATES).flatMap(placeholderNames);
  assert([...new Set(declared)].every((name) => Object.hasOwn(contribution.vars, name)),
    `every template var is supplied by the builder (${[...new Set(declared)].join(", ")})`);
  // 4. Negative: an unknown placeholder is rejected with the offending name.
  assert.throws(() => renderTemplate(TEMPLATES.test, { number: 1 }), /unknown placeholder \{\{title\}\}/,
    "unknown placeholder is a hard error");
  // 5. Contract validation accepts the rendered skeleton and rejects each single mutation of it.
  const skeleton = contribution.files.find((f) => f.path.endsWith(".js")).content;
  assert.deepStrictEqual(validateSolutionSource(skeleton), [], "the rendered skeleton satisfies the contract");
  const mutations = [
    skeleton.replace('"use strict";', "/* strict */"),
    skeleton.replace(/module\.exports[^\n]*/, "// exported elsewhere"),
    skeleton.replace('console.log("PASS")', "console.log(\"done\")"),
    skeleton.replace("assert.ok(true);", "// TODO implement"),
  ];
  assert(mutations.every((m) => validateSolutionSource(m).length === 1), "each mutation yields exactly one violation");
  // 6. Layout: four solution files plus spec/test/README row, all inside the problem's folder.
  assert.strictEqual(contribution.files.length, 7, `artifact count ${contribution.files.length}`);
  assert(contribution.problems.length === 0, `layout problems: ${contribution.problems.join("; ")}`);
  assert(contribution.paths.includes("solutions/javascript/master/953-implement-contributor-guide/solution.js"), "js path");
  // 7. Guide: duplicate headings still get unique anchors and no internal link dangles.
  const guide = buildGuide([
    { title: "Getting started", level: 2, body: "Read [the contract](#getting-started)." },
    { title: "Getting started", level: 2, body: "Duplicate heading." },
    { title: "Templates", level: 2, body: "See [the samples](#templates) and [the missing FAQ](#faqs)." },
    { title: "FAQ", level: 3, body: "Answered here." },
  ]);
  assert.strictEqual(guide.anchors.size, 4, `anchors ${[...guide.anchors].join(", ")}`);
  assert(guide.anchors.has("getting-started-2") && guide.markdown.includes('<a id="getting-started-2"></a>'),
    "the duplicate heading is disambiguated with an explicit anchor");
  assert(guide.links.length === 3 && guide.dangling.length === 1 && guide.dangling[0] === "faqs", "one dangling link found");
  // 8. Boundary + determinism: an all-symbol title has no slug, and rendering twice is byte-identical.
  assert.strictEqual(slugify("!!! ???"), "", "a title without alphanumerics has an empty slug");
  const again = buildContribution({ number: 953, tier: "master", title: "Implement contributor guide", chapter: "Beyond" });
  assert.deepStrictEqual(again.files, contribution.files, "rendering is deterministic");
  console.log(`contrib-guide: artifacts=${contribution.files.length} anchors=${guide.anchors.size} `
    + `dangling=${guide.dangling.length} slug=${contribution.slug}`);
  console.log("PASS");
}

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}

module.exports = { makeRng, slugify, placeholderNames, renderTemplate, TEMPLATES, TIERS, validateSolutionSource,
  folderName, solutionPath, LAYOUT, buildContribution, buildGuide };
if (require.main === module) main();
