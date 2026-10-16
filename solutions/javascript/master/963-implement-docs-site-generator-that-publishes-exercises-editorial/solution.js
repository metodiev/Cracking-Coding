/** Task 963 - Implement docs site generator that publishes exercises, editorial notes, and solutions.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A static site generator for a markdown subset: YAML-ish front matter, ATX headings, paragraphs, bullet
 * lists, fenced code blocks with a `language-*` class, and inline code / bold / links. Everything that
 * comes out of the content model is HTML-escaped before markup is re-introduced, so a lesson containing
 * `<script>` cannot inject markup. Headings get slugified anchors that are de-duplicated per page and
 * drive a table of contents, internal links are resolved against the page set (dangling ones are
 * reported rather than shipped), and each page carries a content hash so a rebuild can reuse unchanged
 * pages verbatim -- the "incremental" part is verifiable by identity of the emitted strings.
 *
 * Simplified: no YAML library, no syntax highlighting, no template inheritance, and only the markdown
 * constructs listed above. Verified: block rendering incl. the code-fence language class, escaping of
 * raw HTML in text and code, unique anchors for duplicate headings, dangling-link detection, reuse of
 * untouched pages on rebuild, sitemap ordering, the empty-site and no-front-matter boundaries, and
 * byte-identical full builds.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const escapeHtml = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const slugify = (text) => text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
function digest(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return (h >>> 0).toString(16).padStart(8, "0");
}
/** Front matter is a leading `---` block of `key: value` lines; the rest is the markdown body. */
function parseFrontMatter(source) {
  const lines = source.split("\n");
  if (lines[0] !== "---") return { meta: {}, body: source };
  const meta = {};
  let index = 1;
  for (; index < lines.length && lines[index] !== "---"; index += 1) {
    const match = /^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$/.exec(lines[index]);
    if (match) meta[match[1]] = match[2].trim();
  }
  return { meta, body: lines.slice(index + 1).join("\n") };
}
/** Escaping happens first; markup is then reintroduced, so content can never open a tag. */
function renderInline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2">$1</a>');
}
/** Render the supported subset; returns html plus the anchors and links it produced. */
function renderMarkdown(source) {
  const lines = source.split("\n");
  const out = [];
  const headings = [];
  const links = [];
  const used = new Map();
  let listOpen = false;
  const closeList = () => { if (listOpen) { out.push("</ul>"); listOpen = false; } };
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const fence = /^```([a-z0-9+#-]*)\s*$/.exec(line);
    if (fence) {
      closeList();
      const body = [];
      for (i += 1; i < lines.length && !/^```\s*$/.test(lines[i]); i += 1) body.push(lines[i]);
      const cls = fence[1] ? ` class="language-${fence[1]}"` : "";
      out.push(`<pre><code${cls}>${escapeHtml(body.join("\n"))}</code></pre>`);
      continue;
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      closeList();
      const level = heading[1].length;
      const text = heading[2].trim();
      const base = slugify(text);
      const seen = (used.get(base) || 0) + 1;
      used.set(base, seen);
      const anchor = seen === 1 ? base : `${base}-${seen}`;
      headings.push({ level, text, anchor });
      out.push(`<h${level} id="${anchor}">${renderInline(text)}</h${level}>`);
      continue;
    }
    const item = /^[-*]\s+(.*)$/.exec(line);
    if (item) {
      if (!listOpen) { out.push("<ul>"); listOpen = true; }
      out.push(`<li>${renderInline(item[1])}</li>`);
      continue;
    }
    if (line.trim() === "") { closeList(); continue; }
    closeList();
    out.push(`<p>${renderInline(line.trim())}</p>`);
  }
  closeList();
  for (const match of source.matchAll(/\]\(([^)\s]+)\)/g)) links.push(match[1]);
  return { html: out.join("\n"), headings, links };
}
/** Build a page: front matter defaults, rendered html, anchors and a content hash. */
function buildPage(path, source) {
  const { meta, body } = parseFrontMatter(source);
  const rendered = renderMarkdown(body);
  return { path, title: meta.title || path.replace(/\.md$/, ""), tags: (meta.tags || "").split(",").filter(Boolean),
    hash: digest(source), html: rendered.html, headings: rendered.headings, links: rendered.links };
}
/** Build the whole site, reusing the previous html verbatim for pages whose content hash is unchanged. */
function buildSite(contentModel, previous = null) {
  const previousByPath = new Map((previous ? previous.entries : []).map((entry) => [entry.path, entry]));
  const reused = [];
  const rendered = [];
  const entries = [];
  for (const path of Object.keys(contentModel).sort()) {
    const candidate = buildPage(path, contentModel[path]);
    const old = previousByPath.get(path);
    const entry = old && old.hash === candidate.hash ? old : candidate;
    (entry === old ? reused : rendered).push(path);
    entries.push(entry);
  }
  const known = new Set(entries.map((entry) => entry.path));
  const dangling = entries.flatMap((entry) => entry.links
    .filter((href) => href.endsWith(".md") && !known.has(href))
    .map((href) => ({ from: entry.path, href })));
  const sitemap = entries.map((entry) => `<url><loc>/${entry.path.replace(/\.md$/, ".html")}</loc></url>`);
  return { entries, reused, rendered, dangling, sitemap,
    index: entries.filter((entry) => entry.path !== "index.md").map((entry) => `<li><a href="/${entry.path.replace(/\.md$/, ".html")}">${escapeHtml(entry.title)}</a></li>`).join("\n") };
}

const CONTENT = {
  "index.md": "---\ntitle: Cracking Coding\ntags: home\n---\n\n# Welcome\n\nStart with [arrays](arrays.md) or the [editorial](editorial.md).\n",
  "arrays.md": "---\ntitle: Arrays\ntags: beginner,expert\n---\n\n## Two pointers\n\nUse two indices.\n\n```js\nconst xs = [1, 2, 3];\nif (a < b && c > d) { run(); }\n```\n\n## Two pointers\n\nA second section with the same title.\n",
  "editorial.md": "---\ntitle: Editorial\n---\n\n# Notes\n\n- Prefer invariants\n- Measure before optimising\n\nSee [arrays](arrays.md) and [missing](missing.md).\n",
  "sandbox.md": "---\ntitle: Sandbox\n---\n\n# Unsafe sample\n\n<script>alert(1)</script>\n\n```html\n<script>alert(2)</script>\n```\n",
};

function main() {
  const rng = makeRng(20240963);
  const site = buildSite(CONTENT);
  const byPath = new Map(site.entries.map((entry) => [entry.path, entry]));
  // 1. Block rendering: headings, list, paragraph and a code fence carrying its language class.
  const arrays = byPath.get("arrays.md");
  assert(arrays.html.includes("<h2 id=\"two-pointers\">Two pointers</h2>"), "headings render with anchors");
  assert(arrays.html.includes('<pre><code class="language-js">'), "the fence keeps its language class");
  const editorial = byPath.get("editorial.md");
  assert(editorial.html.includes("<ul>\n<li>Prefer invariants</li>\n<li>Measure before optimising</li>\n</ul>"), "list rendering");
  assert(editorial.html.includes("<p>See <a href=\"arrays.md\">arrays</a> and"), "paragraphs carry inline links");
  // 2. Escaping: raw HTML in prose and inside a code fence is neutralised, never emitted.
  const sandbox = byPath.get("sandbox.md");
  assert(!/<script>/.test(sandbox.html) && sandbox.html.includes("&lt;script&gt;alert(1)&lt;/script&gt;"), "prose html is escaped");
  assert(sandbox.html.includes("&lt;script&gt;alert(2)&lt;/script&gt;") && sandbox.html.split("&lt;script&gt;").length === 3,
    "code fences escape too");
  const injected = buildPage("evil.md", "# Title\n\n[<b>click</b>](https://example.com/x) and `</code>`\n");
  assert(injected.html.includes('<a href="https://example.com/x">&lt;b&gt;click&lt;/b&gt;</a>'),
    "link labels are escaped while the target is preserved");
  assert(injected.html.includes("<code>&lt;/code&gt;</code>"), "inline code escapes its own delimiters");
  // 3. Anchors: duplicate headings are disambiguated and every anchor is unique and lower-case dashed.
  const anchors = arrays.headings.map((heading) => heading.anchor);
  assert(anchors.join(",") === "two-pointers,two-pointers-2", `anchors ${anchors.join(",")}`);
  assert(new Set(site.entries.flatMap((entry) => entry.headings.map((h) => `${entry.path}#${h.anchor}`))).size
    === site.entries.reduce((a, entry) => a + entry.headings.length, 0), "anchors are unique per page");
  // 4. Cross references: the dangling link is reported, the resolvable ones are not.
  assert(site.dangling.length === 1 && site.dangling[0].from === "editorial.md" && site.dangling[0].href === "missing.md",
    `dangling ${JSON.stringify(site.dangling)}`);
  assert(!site.dangling.some((row) => row.href === "arrays.md"), "resolvable links are not flagged");
  // 5. Incremental rebuild: touching one page re-renders only that page and reuses the rest verbatim.
  const touched = { ...CONTENT, "arrays.md": `${CONTENT["arrays.md"]}\nA new closing line.\n` };
  const incremental = buildSite(touched, site);
  assert.deepStrictEqual(incremental.rendered, ["arrays.md"], `re-rendered ${incremental.rendered.join(",")}`);
  assert(incremental.reused.sort().join(",") === Object.keys(CONTENT).filter((p) => p !== "arrays.md").sort().join(","),
    `reused ${incremental.reused.join(",")}`);
  const reusedArrays = incremental.entries.find((entry) => entry.path === "index.md");
  assert.strictEqual(reusedArrays.html, byPath.get("index.md").html, "an untouched page is reused byte for byte");
  assert.strictEqual(incremental.entries.find((entry) => entry.path === "arrays.md").hash, digest(touched["arrays.md"]), "the touched page rehashes");
  assert(buildSite(CONTENT, site).rendered.length === 0, "a no-op rebuild re-renders nothing");
  // 6. Sitemap and index: one entry per page, sorted by path, and the index lists everything but itself.
  assert(site.sitemap.length === Object.keys(CONTENT).length && site.sitemap.join("").includes("/arrays.html"), "sitemap per page");
  const locs = site.sitemap.map((row) => /<loc>\/(.*?)<\/loc>/.exec(row)[1]);
  assert(locs.join(",") === "arrays.html,editorial.html,index.html,sandbox.html", `sitemap order ${locs.join(",")}`);
  assert(site.index.split("\n").length === Object.keys(CONTENT).length - 1 && site.index.includes(">Arrays<"), "index lists pages");
  // 7. Boundaries: an empty site, a page without front matter, and a page with no headings.
  const empty = buildSite({});
  assert(empty.entries.length === 0 && empty.sitemap.length === 0 && empty.dangling.length === 0 && empty.index === "", "empty site");
  const bare = buildPage("loose.md", "Just a paragraph with **bold** text.\n");
  assert(bare.title === "loose" && bare.tags.length === 0 && bare.headings.length === 0, "no front matter falls back to the filename");
  assert(bare.html.includes("<strong>bold</strong>"), "inline bold still renders without front matter");
  const headingOnly = buildPage("h.md", "#\n");
  assert(headingOnly.headings.length === 0 && headingOnly.html === "<p>#</p>", "a bare hash is prose, not a heading");
  // 8. Determinism: a full rebuild is byte-identical, and the seeded stream only shuffles the content order.
  assert.deepStrictEqual(buildSite(CONTENT), site, "full rebuild identical");
  const shuffled = {};
  const keys = Object.keys(CONTENT);
  for (let i = keys.length - 1; i > 0; i -= 1) { const j = rng.int(i + 1); [keys[i], keys[j]] = [keys[j], keys[i]]; }
  for (const key of keys) shuffled[key] = CONTENT[key];
  assert.deepStrictEqual(buildSite(shuffled).sitemap, site.sitemap, "content order does not change the sitemap");
  console.log(`docs-site: pages=${site.entries.length} anchors=${site.entries.reduce((a, e) => a + e.headings.length, 0)} `
    + `dangling=${site.dangling.length} rebuilt=${incremental.rendered.length} sitemap=${site.sitemap.length}`);
  console.log("PASS");
}

module.exports = { makeRng, escapeHtml, slugify, digest, parseFrontMatter, renderInline, renderMarkdown,
  buildPage, buildSite, CONTENT };
if (require.main === module) main();
