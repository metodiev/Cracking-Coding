#!/usr/bin/env node
/**
 * Parse every ```mermaid block in the tutorials with the real mermaid parser.
 *
 * `tools/check_mermaid.py` does the cheap structural checks (fences, keywords, tabs,
 * escaping, required sections); this script is the authoritative one: it runs the
 * actual mermaid parser in Node behind a jsdom shim, so a diagram that mermaid cannot
 * parse fails here.
 *
 * Usage:
 *     node tools/check_mermaid_render.mjs [path ...]        (default: tutorials)
 *
 * Dependencies are resolved from $MERMAID_MODULES (defaults to ./node_modules at the
 * repository root), so the repository itself stays dependency-free:
 *     npm install --prefix "$env:TEMP/mermaid" mermaid@11 jsdom
 *     $env:MERMAID_MODULES = "$env:TEMP/mermaid"; node tools/check_mermaid_render.mjs
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function collect(targets) {
  const files = [];
  for (const target of targets) {
    const st = statSync(target);
    if (st.isDirectory()) {
      for (const entry of readdirSync(target, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
        files.push(...collect([path.join(target, entry.name)]));
      }
    } else if (target.endsWith(".md")) {
      files.push(target);
    }
  }
  return files;
}

function mermaidBlocks(text) {
  const lines = text.split(/\r?\n/);
  const blocks = [];
  let current = null;
  let startLine = 0;
  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (current === null) {
      if (trimmed === "```mermaid") {
        current = [];
        startLine = index + 1;
      }
      return;
    }
    if (trimmed.startsWith("```")) {
      blocks.push({ startLine, text: current.join("\n") });
      current = null;
      return;
    }
    current.push(line);
  });
  return blocks;
}

async function loadMermaid() {
  const modulesDir = process.env.MERMAID_MODULES
    ? path.resolve(process.env.MERMAID_MODULES)
    : path.join(ROOT, "node_modules");
  const requireFrom = createRequire(path.join(modulesDir, "noop.js"));
  let mermaidPath;
  let jsdomPath;
  try {
    mermaidPath = requireFrom.resolve("mermaid");
    jsdomPath = requireFrom.resolve("jsdom");
  } catch (error) {
    console.error(`Cannot resolve mermaid/jsdom from ${modulesDir}.`);
    console.error("Install them (outside the repository) and point MERMAID_MODULES at the prefix:");
    console.error('  npm install --prefix "$env:TEMP/mermaid" mermaid@11 jsdom');
    console.error(`Details: ${error.message}`);
    process.exit(2);
  }

  const jsdomModule = await import(pathToFileURL(jsdomPath).href);
  const { JSDOM } = jsdomModule.default ?? jsdomModule;
  const dom = new JSDOM("<!DOCTYPE html><body></body>", { pretendToBeVisual: true });
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
  globalThis.DOMPurify = { sanitize: (value) => value, addHook: () => {} };

  const mermaidModule = await import(pathToFileURL(mermaidPath).href);
  const mermaid = mermaidModule.default ?? mermaidModule;
  mermaid.initialize({ startOnLoad: false, securityLevel: "loose", logLevel: "fatal" });
  return mermaid;
}

const targets = process.argv.slice(2).map((p) => path.resolve(p));
const files = collect(targets.length ? targets : [path.join(ROOT, "tutorials")]);
const mermaid = await loadMermaid();

let checked = 0;
const failures = [];
for (const file of files) {
  const text = readFileSync(file, "utf8");
  for (const block of mermaidBlocks(text)) {
    checked += 1;
    const body = block.text.trim();
    if (!body) {
      failures.push({ file, line: block.startLine, message: "empty mermaid block" });
      continue;
    }
    try {
      await mermaid.parse(body);
    } catch (error) {
      failures.push({
        file,
        line: block.startLine,
        message: String(error?.message ?? error).split("\n").slice(0, 3).join(" | ").slice(0, 220),
      });
    }
  }
}

for (const failure of failures.slice(0, 200)) {
  console.log(`RENDER-FAIL ${path.relative(ROOT, failure.file)}:${failure.line} ${failure.message}`);
}
if (failures.length > 200) console.log(`... and ${failures.length - 200} more`);
console.log(`parsed ${checked} mermaid diagrams in ${files.length} markdown files, ${failures.length} failures`);
process.exit(failures.length ? 1 : 0);
