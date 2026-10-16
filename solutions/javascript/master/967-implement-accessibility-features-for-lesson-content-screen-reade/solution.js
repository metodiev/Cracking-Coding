/** Task 967 - Implement accessibility features for lesson content (screen-reader friendly).
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A lesson renderer plus a WCAG-oriented auditor. The renderer turns a block model into semantic HTML:
 * headings with stable ids, code inside `<code class="language-x" aria-label="...">`, images with their
 * alt text (or an explicit decorative marker), widgets with a role and an accessible name resolved
 * through `aria-labelledby`, and tables with a caption plus `<th scope>`. The auditor then checks the
 * screen-reader contract independently of the markup it just produced: heading levels may not skip (and
 * must start at h1), images need alt or `role="presentation"`, widgets need a name whose
 * `aria-labelledby` target actually exists, tables need a caption and scoped headers, code needs a
 * language for the spoken label, and text must clear the WCAG AA contrast ratio -- 4.5:1 normally and
 * 3:1 for large text, computed from hex colours with the published relative-luminance formula.
 *
 * Simplified: the block model stands in for rich lesson sources, only the listed a11y rules are
 * enforced, and "large text" is declared by the author instead of measured from CSS. Verified: contrast
 * against known reference pairs (21:1, 1:1, #767676/#777777 around AA), the rendered semantics, heading
 * skips, the decorative-image and large-text boundaries, dangling label references, table structure,
 * the empty lesson, and deterministic output.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const escapeHtml = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
/** WCAG relative luminance and contrast ratio. */
function luminance(hex) {
  const channel = (value) => (value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
  const parts = [1, 3, 5].map((offset) => channel(parseInt(hex.slice(offset, offset + 2), 16) / 255));
  return 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2];
}
function contrast(foreground, background) {
  const a = luminance(foreground), b = luminance(background);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}
/** Independent spelling of the same definition, used as the oracle in the checks. */
function contrastOracle(foreground, background) {
  const lin = (byte) => { const c = byte / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const relative = (hex) => 0.2126 * lin(parseInt(hex.slice(1, 3), 16)) + 0.7152 * lin(parseInt(hex.slice(3, 5), 16))
    + 0.0722 * lin(parseInt(hex.slice(5, 7), 16));
  const one = relative(foreground), two = relative(background);
  return (Math.max(one, two) + 0.05) / (Math.min(one, two) + 0.05);
}
const PREFIX = { heading: "h", code: "c", image: "i", widget: "w", table: "t" };
/** One stable id per block: the renderer and the auditor must agree on this naming. */
const blockId = (block, index) => `${PREFIX[block.type] || "b"}-${index}`;
const RATIOS = { normal: 4.5, large: 3 };
/** Render the block model to screen-reader friendly HTML and report the ids it emitted. */
function renderLesson(lesson) {
  const out = [];
  const ids = new Set();
  const widgets = [];
  lesson.blocks.forEach((block, index) => {
    const id = blockId(block, index);
    ids.add(id);
    switch (block.type) {
      case "heading":
        out.push(`<h${block.level} id="${id}">${escapeHtml(block.text)}</h${block.level}>`);
        break;
      case "paragraph":
        out.push(`<p style="color:${block.fg || "#000000"};background:${block.bg || "#ffffff"}">${escapeHtml(block.text)}</p>`);
        break;
      case "list":
        out.push(`<ul>${block.items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`);
        break;
      case "code":
        out.push(`<pre id="${id}"><code class="language-${block.language || "none"}" `
          + `aria-label="${escapeHtml(`${block.language || "unlabelled"} code sample`)}">${escapeHtml(block.code)}</code></pre>`);
        break;
      case "image":
        out.push(block.alt === undefined ? `<img src="${escapeHtml(block.src)}">`
          : `<img src="${escapeHtml(block.src)}" alt="${escapeHtml(block.alt)}"${block.decorative ? ' role="presentation"' : ""}>`);
        break;
      case "widget": {
        const markup = `<button id="${id}" role="${escapeHtml(block.role || "button")}" `
          + (block.labelledby ? `aria-labelledby="${escapeHtml(block.labelledby)}"` : `aria-label="${escapeHtml(block.label || "")}"`)
          + `>${escapeHtml(block.text || block.label || "")}</button>`;
        out.push(markup);
        widgets.push(markup);
        break;
      }
      case "table":
        out.push(`<table id="${id}">${block.caption ? `<caption>${escapeHtml(block.caption)}</caption>` : ""}`
          + `<thead><tr>${block.headers.map((header) => `<th scope="col">${escapeHtml(header)}</th>`).join("")}</tr></thead>`
          + `<tbody>${block.rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
        break;
      default:
        out.push(`<div role="group" id="${id}">${escapeHtml(block.text || "")}</div>`);
    }
  });
  return { html: out.join("\n"), ids, widgets: widgets.join("\n") };
}
/** Audit the lesson model against the accessibility rules; every finding names the rule and the block. */
function auditLesson(lesson, { ratios = RATIOS } = {}) {
  const ids = new Set(lesson.blocks.map((block, index) => blockId(block, index)));
  const issues = [];
  const report = (block, index, rule, detail) => issues.push({ rule, block: `${block.type}#${index}`, detail });
  let level = 0;
  lesson.blocks.forEach((block, index) => {
    if (block.type === "heading") {
      if (level === 0 && block.level !== 1) report(block, index, "heading-order", `starts at h${block.level}, expected h1`);
      else if (level > 0 && block.level > level + 1) report(block, index, "heading-order", `h${level} -> h${block.level} skips a level`);
      level = block.level;
    }
    if (block.type === "image" && block.alt === undefined) report(block, index, "image-alt", `missing alt for ${block.src}`);
    if (block.type === "image" && block.alt === "" && !block.decorative) report(block, index, "image-alt", "empty alt without role=presentation");
    if (block.type === "widget") {
      if (block.labelledby && !ids.has(block.labelledby)) report(block, index, "widget-label", `dangling aria-labelledby ${block.labelledby}`);
      if (!block.label && !block.labelledby) report(block, index, "widget-label", "no accessible name");
    }
    if (block.type === "table" && (!block.caption || !block.headers || block.headers.length === 0)) {
      report(block, index, "table-structure", block.caption ? "headers without scoped cells" : "missing caption");
    }
    if (block.type === "code" && !block.language) report(block, index, "code-language", block.code.slice(0, 12));
    if (block.type === "paragraph") {
      const ratio = contrast(block.fg || "#000000", block.bg || "#ffffff");
      const required = block.large ? ratios.large : ratios.normal;
      if (ratio < required) report(block, index, "contrast", `${ratio.toFixed(2)}:1 below ${required}:1`);
    }
  });
  return issues;
}
const summarize = (issues) => [...issues.reduce((map, issue) => map.set(issue.rule, (map.get(issue.rule) || 0) + 1), new Map())]
  .map(([rule, count]) => `${rule}:${count}`).sort();

const GOOD_LESSON = { title: "Two pointers", blocks: [
  { type: "heading", level: 1, text: "Two pointers" },
  { type: "paragraph", text: "Walk two indices towards each other.", fg: "#111111", bg: "#ffffff" },
  { type: "list", items: ["Start at both ends", "Move the smaller side"] },
  { type: "code", language: "js", code: "let i = 0, j = n - 1;" },
  { type: "image", src: "pointer.png", alt: "Two indices moving towards the middle" },
  { type: "widget", role: "button", label: "Run the two-pointer sample", text: "Run" },
  { type: "table", caption: "Step by step", headers: ["Step", "i", "j"], rows: [["1", "0", "4"], ["2", "1", "3"]] },
] };
const BAD_LESSON = { title: "Broken", blocks: [
  { type: "heading", level: 1, text: "Intro" },
  { type: "heading", level: 3, text: "Deep dive" },
  { type: "paragraph", text: "Grey on white.", fg: "#949494", bg: "#ffffff" },
  { type: "image", src: "diagram.png" },
  { type: "widget", role: "button", labelledby: "h-99" },
  { type: "table", headers: [], rows: [["a"]] },
  { type: "code", code: "run();" },
] };

function main() {
  const rng = makeRng(20240967);
  // 1. Contrast against known reference pairs, including the classic AA boundary colours.
  assert.strictEqual(contrast("#000000", "#ffffff"), 21, "black on white is exactly 21:1");
  assert.strictEqual(contrast("#ffffff", "#ffffff"), 1, "a colour against itself is 1:1");
  assert(contrast("#767676", "#ffffff") > RATIOS.normal && contrast("#777777", "#ffffff") < RATIOS.normal,
    `the AA boundary sits between #767676 and #777777 (${contrast("#777777", "#ffffff").toFixed(2)}:1)`);
  // 2. Independent oracle: the spelled-out formula agrees with the helper on seeded colour pairs.
  let oracleAgreements = 0;
  const hex = () => `#${Array.from({ length: 3 }, () => rng.int(256).toString(16).padStart(2, "0")).join("")}`;
  for (let trial = 0; trial < 40; trial += 1) {
    const foreground = hex(), background = hex();
    if (Math.abs(contrast(foreground, background) - contrastOracle(foreground, background)) < 1e-12) oracleAgreements += 1;
  }
  assert.strictEqual(oracleAgreements, 40, `contrast oracle disagreements ${40 - oracleAgreements}`);
  // 3. Rendering: the good lesson produces semantic, screen-reader friendly markup.
  const rendered = renderLesson(GOOD_LESSON);
  assert(rendered.html.includes('<h1 id="h-0">Two pointers</h1>') && rendered.ids.has("h-0"), "headings carry ids");
  assert(rendered.html.includes('<code class="language-js" aria-label="js code sample">'), "code is labelled for the reader");
  assert(rendered.html.includes('alt="Two indices moving towards the middle"'), "the image keeps its alt text");
  assert(rendered.html.includes('role="button" aria-label="Run the two-pointer sample"'), "the widget exposes an accessible name");
  assert(rendered.html.includes("<caption>Step by step</caption>") && rendered.html.includes('<th scope="col">Step</th>'), "the table is structured");
  assert(!/tabindex="[1-9]/.test(rendered.html), "no positive tabindex is emitted");
  assert(rendered.ids.has("w-5") && rendered.ids.has("t-6"), "every block id is registered");
  // 4. Happy path: the good lesson is clean; every rule has its own finding in the bad lesson.
  assert.deepStrictEqual(auditLesson(GOOD_LESSON), [], `good lesson issues ${JSON.stringify(auditLesson(GOOD_LESSON))}`);
  const issues = auditLesson(BAD_LESSON);
  assert(summarize(issues).join(",") === "code-language:1,contrast:1,heading-order:1,image-alt:1,table-structure:1,widget-label:1",
    `bad lesson summary ${summarize(issues).join(",")}`);
  assert(issues.every((issue) => issue.block && issue.detail), "each finding names its block and reason");
  // 5. Heading order: a skip is caught, a valid descent is not, and a document starting below h1 is flagged.
  const skip = { blocks: [{ type: "heading", level: 1, text: "a" }, { type: "heading", level: 4, text: "b" }] };
  assert(auditLesson(skip)[0].detail === "h1 -> h4 skips a level", "a skipped level is reported");
  const descent = { blocks: [{ type: "heading", level: 1, text: "a" }, { type: "heading", level: 2, text: "b" }, { type: "heading", level: 3, text: "c" }] };
  assert(auditLesson(descent).length === 0, "a proper descent is fine");
  assert(auditLesson({ blocks: [{ type: "heading", level: 2, text: "only h2" }] })[0].detail.startsWith("starts at h2"),
    "a lesson that starts below h1 is flagged");
  // 6. Image and widget boundaries: decorative images pass, dangling label references do not.
  assert(auditLesson({ blocks: [{ type: "image", src: "rule.png", alt: "", decorative: true }] }).length === 0, "decorative images pass");
  assert(auditLesson({ blocks: [{ type: "image", src: "rule.png", alt: "" }] })[0].rule === "image-alt", "empty alt without the marker is flagged");
  const linked = { blocks: [{ type: "heading", level: 1, text: "Playground" }, { type: "widget", role: "button", labelledby: "h-0" }] };
  assert(auditLesson(linked).length === 0, "a resolvable aria-labelledby is accepted");
  assert(auditLesson({ blocks: [{ type: "widget", role: "button", labelledby: "h-99" }] })[0].detail.includes("dangling"), "a dangling reference is caught");
  // 7. Large-text boundary: one colour pair passes at 3:1 and fails at 4.5:1.
  const grey = { type: "paragraph", text: "large heading text", fg: "#949494", bg: "#ffffff" };
  assert(contrast(grey.fg, grey.bg) > RATIOS.large && contrast(grey.fg, grey.bg) < RATIOS.normal, `grey ratio ${contrast(grey.fg, grey.bg)}`);
  assert(auditLesson({ blocks: [{ ...grey, large: true }] }).length === 0, "large text only needs 3:1");
  assert(auditLesson({ blocks: [grey] })[0].rule === "contrast", "the same colour fails normal text");
  // 8. Boundaries: an empty lesson renders and audits cleanly, and unstyled text defaults to black on white.
  const empty = renderLesson({ blocks: [] });
  assert(empty.html === "" && empty.ids.size === 0 && auditLesson({ blocks: [] }).length === 0, "an empty lesson is valid");
  const plain = { blocks: [{ type: "paragraph", text: "default colours" }] };
  assert(auditLesson(plain).length === 0 && renderLesson(plain).html.includes("color:#000000;background:#ffffff"), "defaults are AA-safe");
  // 9. Determinism: rendering and auditing replay byte for byte, and the summary counts add up.
  assert.deepStrictEqual(renderLesson(GOOD_LESSON), rendered, "rendering replays identically");
  assert.deepStrictEqual(auditLesson(BAD_LESSON), issues, "auditing replays identically");
  assert(issues.length === 6 && summarize(issues).every((entry) => Number(entry.split(":")[1]) >= 1), "summary partition");
  console.log(`a11y: blocks=${GOOD_LESSON.blocks.length} goodIssues=0 badIssues=${issues.length} `
    + `rules=${summarize(issues).join("/")} contrast=${contrast("#777777", "#ffffff").toFixed(2)}:1`);
  console.log("PASS");
}

module.exports = { makeRng, escapeHtml, luminance, contrast, contrastOracle, PREFIX, blockId, RATIOS,
  renderLesson, auditLesson, summarize, GOOD_LESSON, BAD_LESSON };
if (require.main === module) main();
