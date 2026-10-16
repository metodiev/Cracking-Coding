/** Task 983 - Implement end-to-end security review checklist and automated scans for repository code.
 * (Tier: Master, CTCI chapter Beyond - Security & Privacy)
 *
 * A checklist of controls (OWASP-style ids) is wired to pattern rules that run over an in-memory
 * file set. A rule reports the 1-based line numbers it matched, so file-level controls - such as an
 * unprotected mutation route - can look at the whole file. Findings carry the control, the severity
 * and the snippet; the review then computes a risk score from the severity table and fails the gate
 * when any critical control fired, listing exactly which ones blocked it.
 * Simplified: the "repository" is a handful of source strings (no real AST, taint tracking or
 * dependency scanning) and the rules are regex/heuristic - the path rule, for instance, ignores
 * lines that already call a basename/sanitize helper. Verified: true positives with line numbers,
 * a false-positive-free hardened fixture, the gate and risk score, checklist coverage of every rule,
 * an independent line scan for one control, ordering, the empty-repository boundary and determinism.
 */
"use strict";
const assert = require("node:assert");

const SEVERITY_SCORE = { critical: 9.0, high: 7.0, medium: 4.0, low: 2.0 };
const CHECKS = [
  { control: "A01", title: "Broken access control" },
  { control: "A02", title: "Cryptographic failures and secrets" },
  { control: "A03", title: "Injection" },
  { control: "A05", title: "Security misconfiguration" },
];
const RULES = [
  { id: "SEC-SECRET-01", control: "A02", severity: "critical", title: "Hardcoded credential",
    find: (lines) => lines.flatMap((line, i) =>
      (/(api[_-]?key|secret|password|token)\s*[:=]\s*["'][^"']{8,}["']/i.test(line) ? [i + 1] : [])) },
  { id: "SEC-SQLI-01", control: "A03", severity: "critical", title: "SQL assembled by concatenation",
    find: (lines) => lines.flatMap((line, i) =>
      (/(SELECT|INSERT|UPDATE|DELETE)[^;]*["']\s*\+\s*\w/i.test(line) ? [i + 1] : [])) },
  { id: "SEC-EVAL-01", control: "A03", severity: "high", title: "Dynamic code execution",
    find: (lines) => lines.flatMap((line, i) => (/(^|[^.\w])eval\s*\(/.test(line) ? [i + 1] : [])) },
  { id: "SEC-HASH-01", control: "A02", severity: "medium", title: "Weak hash algorithm",
    find: (lines) => lines.flatMap((line, i) =>
      (/createHash\(\s*["'](md5|sha1)["']/i.test(line) ? [i + 1] : [])) },
  { id: "SEC-PATH-01", control: "A01", severity: "high", title: "Untrusted path in a file read",
    find: (lines) => lines.flatMap((line, i) =>
      (/(readFile|readFileSync|createReadStream)\s*\([^)]*req\.(params|query)/.test(line)
        && !/(basename|sanitize)\s*\(/i.test(line) ? [i + 1] : [])) },
  { id: "SEC-AUTH-01", control: "A01", severity: "critical", title: "Mutation route without auth middleware",
    find: (lines, context) => (context.code.includes("requireAuth") ? [] : lines.flatMap((line, i) =>
      (/app\.(post|put|patch|delete)\s*\(/.test(line) ? [i + 1] : []))) },
  { id: "SEC-CORS-01", control: "A05", severity: "medium", title: "Wildcard CORS origin",
    find: (lines) => lines.flatMap((line, i) => (/origin\s*:\s*["']\*["']/.test(line) ? [i + 1] : [])) },
];

const VULNERABLE = {
  "src/config.js": [
    "const API_KEY = \"sk_live_9f2b7c41d8\";",
    "const region = \"eu-west-1\";",
    "const timeout = 5000;",
  ].join("\n"),
  "src/db.js": [
    "function findUser(db, name) {",
    "  const sql = \"SELECT * FROM users WHERE name = '\" + name + \"'\";",
    "  return db.query(sql);",
    "}",
  ].join("\n"),
  "src/tools.js": [
    "function evaluate(expression) {",
    "  return eval(expression);",
    "}",
  ].join("\n"),
  "src/files.js": [
    "function read(req, res) {",
    "  const body = fs.readFileSync(req.params.name);",
    "  res.send(body);",
    "}",
  ].join("\n"),
  "src/routes.js": [
    "app.post(\"/users\", (req, res) => res.send(\"ok\"));",
    "app.get(\"/users\", (req, res) => res.send(\"ok\"));",
  ].join("\n"),
};
const HARDENED = {
  "src/config.js": [
    "const apiKey = process.env.API_KEY;",
    "const region = \"eu-west-1\";",
  ].join("\n"),
  "src/db.js": [
    "function findUser(db, name) {",
    "  return db.query(\"SELECT * FROM users WHERE name = $1\", [name]);",
    "}",
  ].join("\n"),
  "src/files.js": [
    "const BASENAME = (value) => path.basename(String(value));",
    "function read(req, res) {",
    "  const body = fs.readFileSync(path.join(\"public\", BASENAME(req.params.name)));",
    "  res.send(body);",
    "}",
  ].join("\n"),
  "src/routes.js": [
    "const requireAuth = (req, res, next) => (req.user ? next() : res.status(401).end());",
    "app.post(\"/users\", requireAuth, (req, res) => res.send(\"ok\"));",
  ].join("\n"),
  "src/server.js": [
    "app.use(cors({ origin: \"https://app.example.com\" }));",
    "const digest = crypto.createHash(\"sha256\").update(payload).digest(\"hex\");",
  ].join("\n"),
};

/** Scan every file with every rule; findings are ordered by severity then by location. */
function scan(files, rules = RULES) {
  const findings = [];
  for (const path of Object.keys(files).sort()) {
    const code = files[path];
    const lines = code.split("\n");
    for (const rule of rules) {
      for (const line of rule.find(lines, { path, code, files })) {
        findings.push({ rule: rule.id, control: rule.control, severity: rule.severity, title: rule.title,
          file: path, line, snippet: lines[line - 1].trim().slice(0, 72) });
      }
    }
  }
  const rank = { critical: 0, high: 1, medium: 2, low: 3 };
  return findings.sort((a, b) => (rank[a.severity] - rank[b.severity]) || (a.file < b.file ? -1 : a.file > b.file ? 1 : a.line - b.line));
}

/** Checklist review: gate on criticals, plus a risk score and a per-control tally. */
function review(findings, checks = CHECKS) {
  const blockedBy = findings.filter((finding) => finding.severity === "critical").map((finding) => finding.rule);
  const byControl = new Map(checks.map((check) => [check.control, 0]));
  for (const finding of findings) {
    if (!byControl.has(finding.control)) throw new Error(`finding ${finding.rule} has no checklist control`);
    byControl.set(finding.control, byControl.get(finding.control) + 1);
  }
  return {
    passed: blockedBy.length === 0,
    blockedBy: [...new Set(blockedBy)],
    warnings: [...new Set(findings.filter((f) => f.severity !== "critical").map((f) => f.rule))],
    risk: findings.length === 0 ? 0 : Math.max(...findings.map((finding) => SEVERITY_SCORE[finding.severity])),
    controls: [...byControl].filter(([, count]) => count > 0).map(([control]) => control),
    byControl: Object.fromEntries(byControl),
  };
}

function main() {
  const findings = scan(VULNERABLE);
  const report = review(findings);

  // 1. Happy path: every planted vulnerability is found, with its line.
  const ids = findings.map((finding) => `${finding.rule}@${finding.file}:${finding.line}`);
  assert.deepStrictEqual(ids, [
    "SEC-SECRET-01@src/config.js:1", "SEC-SQLI-01@src/db.js:2", "SEC-AUTH-01@src/routes.js:1",
    "SEC-PATH-01@src/files.js:2", "SEC-EVAL-01@src/tools.js:2",
  ], `findings: ${ids.join(", ")}`);
  assert.strictEqual(findings[0].snippet, "const API_KEY = \"sk_live_9f2b7c41d8\";", "the snippet quotes the offending line");
  assert(findings.every((finding) => finding.control && finding.title), "every finding is classified");

  // 2. Independent scan: a hand-written regex reproduces the secret rule line for line.
  const manual = Object.entries(VULNERABLE).flatMap(([file, code]) => code.split("\n")
    .flatMap((line, i) => (/(api[_-]?key|secret|password|token)\s*[:=]\s*["'][^"']{8,}["']/i.test(line) ? [`${file}:${i + 1}`] : [])));
  assert.deepStrictEqual(findings.filter((finding) => finding.rule === "SEC-SECRET-01")
    .map((finding) => `${finding.file}:${finding.line}`), manual, "the secret scan matches the manual scan");

  // 3. Gate: criticals block the review and name themselves; mediums alone only warn.
  assert.deepStrictEqual(report.blockedBy, ["SEC-SECRET-01", "SEC-SQLI-01", "SEC-AUTH-01"], "the blocking rules");
  assert.deepStrictEqual(report.warnings, ["SEC-PATH-01", "SEC-EVAL-01"], "non-critical findings warn");
  assert.strictEqual(report.risk, 9.0, "risk is driven by the worst severity");
  assert.deepStrictEqual(report.controls, ["A01", "A02", "A03"], "only the affected controls are listed");
  assert.strictEqual(report.byControl.A05, 0, "an untouched control reports zero");
  const warningsOnly = review(findings.filter((finding) => finding.severity !== "critical"));
  assert(warningsOnly.passed && warningsOnly.blockedBy.length === 0, "medium findings do not fail the gate");

  // 4. Hardened fixture: the reviewer finds no false positives.
  const clean = review(scan(HARDENED));
  assert.deepStrictEqual(scan(HARDENED), [], `the hardened repository is clean: ${JSON.stringify(scan(HARDENED))}`);
  assert(clean.passed && clean.risk === 0 && clean.controls.length === 0, "a clean report");

  // 5. Checklist coverage: every rule maps to a declared control and each control is covered.
  for (const rule of RULES) {
    assert(CHECKS.some((check) => check.control === rule.control), `${rule.id} maps to a checklist control`);
    assert(SEVERITY_SCORE[rule.severity] > 0, `${rule.id} has a severity score`);
  }
  for (const check of CHECKS) assert(RULES.some((rule) => rule.control === check.control), `control ${check.control} has rules`);
  assert.strictEqual(new Set(RULES.map((rule) => rule.id)).size, RULES.length, "rule ids are unique");

  // 6. Ordering: findings are sorted by severity, then location; severities are ranked correctly.
  const rank = { critical: 0, high: 1, medium: 2, low: 3 };
  for (let i = 1; i < findings.length; i += 1) {
    assert(rank[findings[i].severity] >= rank[findings[i - 1].severity], "severity order");
    if (findings[i].severity === findings[i - 1].severity) {
      assert(findings[i].file > findings[i - 1].file || findings[i].file === findings[i - 1].file, "file order");
    }
  }
  const mixed = scan({ "a.js": "const secret = \"abcdefghij\";\n", "b.js": "eval(1);" });
  assert.deepStrictEqual(mixed.map((finding) => finding.severity), ["critical", "high"], "critical first");

  // 7. Boundaries: an empty repository passes; prose about passwords is not a finding.
  const empty = review(scan({}));
  assert(empty.passed && empty.risk === 0, "an empty repository passes");
  const prose = scan({ "CHANGELOG.md": "password: rotate the token in the vault\nsecret = process.env.SECRET" });
  assert.deepStrictEqual(prose, [], "prose without a literal credential is not flagged");
  assert.strictEqual(scan({ "x.js": "// eval( is dangerous\n" }).length, 1, "a commented eval is still a hit");
  assert.throws(() => review([{ rule: "X", control: "Z99", severity: "low" }]), /no checklist control/,
    "an unmapped control is rejected");
  assert.strictEqual(scan({ "x.js": "app.post(\"/a\", h);" })[0].line, 1, "the route rule reports its line");

  // 8. Determinism: identical input, identical report.
  assert(JSON.stringify(scan(VULNERABLE)) === JSON.stringify(findings), "scans are reproducible");
  assert(JSON.stringify(review(scan(VULNERABLE))) === JSON.stringify(report), "reports are reproducible");
  const reordered = { ...VULNERABLE };
  delete reordered["src/tools.js"];
  reordered["src/tools.js"] = VULNERABLE["src/tools.js"];
  assert(JSON.stringify(scan(reordered)) === JSON.stringify(findings), "file insertion order does not matter");

  console.log(`security review: ${findings.length} findings over ${Object.keys(VULNERABLE).length} files, ` +
    `blocked by ${report.blockedBy.join(",")}, risk ${report.risk}, hardened fixture ${scan(HARDENED).length} findings`);
  console.log("PASS");
}
module.exports = { SEVERITY_SCORE, CHECKS, RULES, VULNERABLE, HARDENED, scan, review };
if (require.main === module) main();
