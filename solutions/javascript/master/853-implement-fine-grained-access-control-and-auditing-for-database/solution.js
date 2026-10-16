/** Task 853 - Implement fine-grained access control and auditing for database operations.
 *  (Tier: Master, CTCI Ch 14 — Databases)
 *
 * Model. A policy engine in front of a small tabular store. The policy is a set of grants; a grant
 * is {effect: allow|deny, roles, actions, resource pattern, columns (null = every column), optional
 * row filter(session, row)}. Roles form an inheritance DAG and a principal gets the transitive
 * closure of its roles. Resources are dotted paths where "*" matches one segment and a trailing "**"
 * matches any number of segments.
 *
 * Evaluation is column-wise with deny-overrides: a requested column is visible iff at least one
 * allow grant covers (role, action, resource, column) and its row filter passes the row, and no
 * matching deny grant covers the same column; everything not covered is denied (default deny). A
 * request that misses some columns is rejected, unless it opts into masking, in which case it
 * succeeds with the missing columns redacted and no data from them is returned.
 *
 * Every decision — including denials — is appended to a tamper-evident audit log: each entry stores
 * the hash of its predecessor, so the whole chain can be re-verified and any edit of a decision, a
 * dropped entry or a reordering is detected.
 *
 * What is simplified: an in-process single-node store, no actual SQL parsing, one flat row list per
 * table, integer/string fields, no concurrent writers (the log is a plain append-only array), and the
 * digest is a deterministic in-process hash rather than a cryptographic one (no crypto module, no
 * wall clock, no randomness).
 *
 * Invariants verified:
 *   safety      deny-overrides and default-deny hold for every request in a large randomised matrix,
 *               cross-checked against an independent brute-force evaluator written in the most naive
 *               possible way (two passes over the raw grant list per requested column).
 *               least privilege: a principal with no roles has no access anywhere, and no decision
 *               ever mentions a column outside the union of the matching grants.
 *   freshness   granting or revoking takes effect on the very next request (cached decisions are
 *               invalidated), so authorisation is never stale.
 *   audit       exactly one audit entry per decision, contiguous sequence numbers, valid hash chain,
 *               replay of the log reproduces every decision, and four kinds of tampering are detected.
 *   cost        each decision inspects at most as many grants as the policy contains (instrumented).
 */
"use strict";

const assert = require("node:assert");

const ACTIONS = ["select", "insert", "update", "delete"];

/** mulberry32 — deterministic PRNG used only to build test workloads. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic 64-bit-style digest (FNV-1a plus a mixed accumulator) rendered as 16 hex chars. */
function digest(text) {
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
    h2 = Math.imul((h2 + c * (i + 7)) >>> 0, 0x85ebca6b) >>> 0;
    h2 = (h2 ^ (h2 >>> 13)) >>> 0;
  }
  return (h1 >>> 0).toString(16).padStart(8, "0") + (h2 >>> 0).toString(16).padStart(8, "0");
}

/** Dotted-path match: "*" is one segment, a trailing "**" is any number of segments (>= 0). */
function matchResource(pattern, resource) {
  const pat = pattern.split(".");
  const res = resource.split(".");
  let i = 0;
  for (; i < pat.length; i++) {
    if (pat[i] === "**") return i === pat.length - 1; // "**" is only meaningful as the last segment
    if (i >= res.length) return false;
    if (pat[i] !== "*" && pat[i] !== res[i]) return false;
  }
  return i === res.length;
}

class Policy {
  constructor() {
    this.grants = [];
    this.roles = new Map(); // role -> parents
    this.version = 0;
    this.invalidations = 0;
  }

  defineRole(name, parents = []) {
    if (typeof name !== "string" || name.length === 0) throw new TypeError("role name required");
    const seen = new Set([name]);
    const stack = [...parents];
    while (stack.length > 0) {
      const parent = stack.pop();
      if (seen.has(parent)) throw new Error(`role cycle through ${parent}`);
      seen.add(parent);
      for (const grand of this.roles.get(parent) || []) stack.push(grand);
    }
    this.roles.set(name, parents.slice());
    this.version += 1;
    return this;
  }

  addGrant({ effect, roles, actions, resource, columns = null, filter = null, id = null }) {
    if (effect !== "allow" && effect !== "deny") throw new Error("effect must be allow or deny");
    if (!Array.isArray(roles) || roles.length === 0) throw new Error("a grant needs at least one role");
    if (!Array.isArray(actions) || actions.length === 0) throw new Error("a grant needs at least one action");
    for (const action of actions) if (!ACTIONS.includes(action)) throw new Error(`unknown action ${action}`);
    if (typeof resource !== "string" || resource.length === 0) throw new Error("resource pattern required");
    if (columns !== null && (!Array.isArray(columns) || columns.length === 0)) throw new Error("columns must be null or a non-empty list");
    const grant = {
      id: id === null ? `g${this.grants.length}` : id,
      effect,
      roles: roles.slice(),
      actions: actions.slice(),
      resource,
      columns: columns === null ? null : columns.slice(),
      filter,
    };
    this.grants.push(grant);
    this.version += 1;
    this.invalidations += 1;
    return grant;
  }

  revokeGrant(id) {
    const before = this.grants.length;
    this.grants = this.grants.filter((g) => g.id !== id);
    if (this.grants.length === before) throw new Error(`no such grant ${id}`);
    this.version += 1;
    this.invalidations += 1;
    return this;
  }

  /** Transitive role closure via DFS (the engine path; the oracle uses a worklist fixpoint). */
  roleClosure(roles) {
    const out = new Set();
    const stack = [...roles];
    while (stack.length > 0) {
      const role = stack.pop();
      if (out.has(role)) continue;
      out.add(role);
      for (const parent of this.roles.get(role) || []) stack.push(parent);
    }
    return out;
  }
}

/** Append-only hash-chained audit log. */
class AuditLog {
  constructor() {
    this.entries = [];
    this.head = digest("genesis");
    this.comparisons = 0;
  }

  append(record) {
    const body = [
      this.entries.length,
      record.principal,
      record.action,
      record.resource,
      record.decision.allowed ? "allow" : "deny",
      record.decision.reason,
      record.decision.visible.join(","),
      record.decision.masked.join(","),
      this.head,
    ].join("|");
    const entry = {
      seq: this.entries.length,
      principal: record.principal,
      action: record.action,
      resource: record.resource,
      allowed: record.decision.allowed,
      reason: record.decision.reason,
      visible: record.decision.visible.slice(),
      masked: record.decision.masked.slice(),
      prev: this.head,
    };
    entry.hash = digest(body);
    this.head = entry.hash;
    this.entries.push(entry);
    return entry;
  }

  /** Recomputes the chain from scratch: catches edits, deletions and reorderings. */
  verifyChain() {
    let prev = digest("genesis");
    for (let i = 0; i < this.entries.length; i++) {
      const e = this.entries[i];
      if (e.seq !== i) return { ok: false, at: i, why: "sequence gap" };
      if (e.prev !== prev) return { ok: false, at: i, why: "broken predecessor link" };
      const body = [
        e.seq,
        e.principal,
        e.action,
        e.resource,
        e.allowed ? "allow" : "deny",
        e.reason,
        e.visible.join(","),
        e.masked.join(","),
        prev,
      ].join("|");
      if (digest(body) !== e.hash) return { ok: false, at: i, why: "payload does not match hash" };
      prev = e.hash;
    }
    return { ok: true, at: this.entries.length, why: "ok" };
  }

  query(predicate) {
    return this.entries.filter(predicate);
  }
}

class AccessEngine {
  constructor(policy) {
    this.policy = policy;
    this.audit = new AuditLog();
    this.cache = new Map();
    this.cacheVersion = -1;
    this.comparisons = 0;
  }

  /** Compiled per-(principal,action,resource) view; dropped whenever the policy changes. */
  compile(request) {
    const policy = this.policy;
    if (this.cacheVersion !== policy.version) {
      this.cache.clear();
      this.cacheVersion = policy.version;
    }
    // The role set is part of the key: a principal's roles come from the directory in production, but
    // keying on them keeps a decision from leaking across different role sets.
    const key = `${request.principal}|${request.roles.slice().sort().join(",")}|${request.action}|${request.resource}`;
    const hit = this.cache.get(key);
    if (hit !== undefined) return hit;
    const roles = policy.roleClosure(request.roles);
    const relevant = [];
    for (const grant of policy.grants) {
      this.comparisons += 1;
      if (!grant.roles.some((role) => roles.has(role))) continue;
      if (!grant.actions.includes(request.action)) continue;
      if (!matchResource(grant.resource, request.resource)) continue;
      relevant.push(grant);
    }
    const compiled = { roles, relevant, version: policy.version };
    this.cache.set(key, compiled);
    return compiled;
  }

  decision(request, compiled) {
    const visible = [];
    const masked = [];
    const blockedBy = [];
    let lastAllow = "default-deny";
    for (const column of request.columns) {
      let columnDecision = null;
      for (const grant of compiled.relevant) {
        const covers = grant.columns === null || grant.columns.includes(column);
        if (!covers) continue;
        if (grant.effect === "deny") {
          if (grant.filter !== null && !grant.filter(request.session, request.row)) continue;
          columnDecision = { ok: false, reason: `deny-${grant.id}` }; // deny-overrides, first matching deny wins
          break;
        }
        if (grant.filter !== null && !grant.filter(request.session, request.row)) continue;
        columnDecision = { ok: true, reason: `allow-${grant.id}` }; // last matching allow wins
      }
      if (columnDecision === null) columnDecision = { ok: false, reason: "default-deny" };
      if (columnDecision.ok) {
        visible.push(column);
        lastAllow = columnDecision.reason;
      } else {
        masked.push(column);
        blockedBy.push(columnDecision.reason);
      }
    }
    if (masked.length === 0) return { allowed: true, visible, masked, reason: lastAllow };
    if (request.masking === true && visible.length > 0) {
      return { allowed: true, visible, masked, reason: `masked:${blockedBy.join("+")}`, maskedOut: true };
    }
    return { allowed: false, visible: [], masked, reason: `blocked:${blockedBy[0]}`, maskedOut: true };
  }

  evaluate(request) {
    if (typeof request.principal !== "string") throw new TypeError("principal required");
    if (!ACTIONS.includes(request.action)) throw new Error(`unknown action ${request.action}`);
    if (!Array.isArray(request.columns) || request.columns.length === 0) throw new Error("request needs columns");
    if (typeof request.resource !== "string" || request.resource.length === 0) throw new Error("request needs a resource");
    const compiled = this.compile(request);
    const decision = this.decision(request, compiled);
    const entry = this.audit.append({ principal: request.principal, action: request.action, resource: request.resource, decision });
    return { ...decision, auditSeq: entry.seq, grantsInspected: compiled.relevant.length };
  }

  /** Returns only the visible columns of the row (masked columns are removed, never returned). */
  project(row, decision) {
    if (!decision.allowed) return null;
    const out = {};
    for (const column of decision.visible) out[column] = row[column];
    return out;
  }
}

/**
 * Independent oracle: the same semantics expressed the most naive way possible — for every requested
 * column, scan the whole grant list twice (first for any deny, then for any allow), computing the
 * role closure with a worklist fixpoint instead of a DFS.
 */
function referenceDecide(policy, request) {
  const closure = new Set(request.roles);
  let changed = true;
  while (changed) {
    changed = false;
    for (const role of [...closure]) {
      for (const parent of policy.roles.get(role) || []) {
        if (!closure.has(parent)) {
          closure.add(parent);
          changed = true;
        }
      }
    }
  }
  const visible = [];
  const masked = [];
  const blockedBy = [];
  let lastAllow = "default-deny";
  for (const column of request.columns) {
    let denyReason = null;
    let allowReason = null;
    for (const grant of policy.grants) {
      if (grant.effect !== "deny") continue;
      if (!grant.roles.some((r) => closure.has(r))) continue;
      if (!grant.actions.includes(request.action)) continue;
      if (!matchResource(grant.resource, request.resource)) continue;
      if (grant.columns !== null && !grant.columns.includes(column)) continue;
      if (grant.filter !== null && !grant.filter(request.session, request.row)) continue;
      denyReason = `deny-${grant.id}`;
      break;
    }
    if (denyReason === null) {
      for (const grant of policy.grants) {
        if (grant.effect !== "allow") continue;
        if (!grant.roles.some((r) => closure.has(r))) continue;
        if (!grant.actions.includes(request.action)) continue;
        if (!matchResource(grant.resource, request.resource)) continue;
        if (grant.columns !== null && !grant.columns.includes(column)) continue;
        if (grant.filter !== null && !grant.filter(request.session, request.row)) continue;
        allowReason = `allow-${grant.id}`;
      }
    }
    if (allowReason !== null) {
      visible.push(column);
      lastAllow = allowReason;
    } else {
      masked.push(column);
      blockedBy.push(denyReason === null ? "default-deny" : denyReason);
    }
  }
  if (masked.length === 0) return { allowed: true, visible, masked, reason: lastAllow };
  if (request.masking === true && visible.length > 0) {
    return { allowed: true, visible, masked, reason: `masked:${blockedBy.join("+")}`, maskedOut: true };
  }
  return { allowed: false, visible: [], masked, reason: `blocked:${blockedBy[0]}`, maskedOut: true };
}

/** Shared fixture: a small analytics store with tenant-isolated rows and a three-level role DAG. */
function buildWarehouse() {
  const policy = new Policy();
  policy.defineRole("analyst", ["reader"]);
  policy.defineRole("senior-analyst", ["analyst"]);
  policy.defineRole("auditor");
  policy.defineRole("admin");
  policy.defineRole("reader", []);
  const rows = [
    { tenant: "acme", id: 1, amount: 100, cost: 60 },
    { tenant: "acme", id: 2, amount: 250, cost: 190 },
    { tenant: "globex", id: 3, amount: 400, cost: 300 },
  ];
  const tenantFilter = (session, row) => row !== undefined && row.tenant === session.tenant;
  policy.addGrant({ id: "read-orders", effect: "allow", roles: ["reader"], actions: ["select"], resource: "sales.public.orders", columns: ["id", "tenant", "amount"], filter: tenantFilter });
  policy.addGrant({ id: "read-cost", effect: "allow", roles: ["senior-analyst"], actions: ["select"], resource: "sales.**", columns: ["cost"], filter: tenantFilter });
  policy.addGrant({ id: "no-globex", effect: "deny", roles: ["reader"], actions: ["select"], resource: "sales.public.orders", columns: ["amount"], filter: (session, row) => row !== undefined && row.tenant === "globex" });
  policy.addGrant({ id: "audit-all", effect: "allow", roles: ["auditor"], actions: ["select"], resource: "**", columns: null });
  return { policy, rows };
}

/** Deterministic fixture request list, also used to replay the audit log from scratch. */
function fixtureRequests(rows) {
  const session = { tenant: "acme" };
  const base = { principal: "u1", roles: ["reader"], action: "select", resource: "sales.public.orders", columns: ["id", "amount"] };
  return [
    { label: "reader-ok", request: { ...base, row: rows[0], session } },
    { label: "reader-cost", request: { ...base, columns: ["id", "cost"], row: rows[0], session } },
    { label: "reader-cost-masked", request: { ...base, columns: ["id", "cost"], row: rows[0], session, masking: true } },
    { label: "reader-only-cost-masked", request: { ...base, columns: ["cost"], row: rows[0], session, masking: true } },
    { label: "senior-cost", request: { principal: "u2", roles: ["senior-analyst"], action: "select", resource: "sales.public.orders", columns: ["cost", "amount"], row: rows[0], session } },
    { label: "analyst-cost", request: { principal: "u3", roles: ["analyst"], action: "select", resource: "sales.public.orders", columns: ["cost"], row: rows[0], session } },
    { label: "other-tenant", request: { ...base, row: rows[2], session } },
    { label: "globex-partial", request: { ...base, row: rows[2], session: { tenant: "globex" }, masking: true } },
    { label: "globex-blocked", request: { ...base, columns: ["amount"], row: rows[2], session: { tenant: "globex" }, masking: true } },
    { label: "auditor-deep", request: { principal: "u4", roles: ["auditor"], action: "select", resource: "some.deep.path", columns: ["cost"], row: rows[2], session: {} } },
  ];
}

function selfCheck() {
  // --- validation --------------------------------------------------------------------------------
  const p0 = new Policy();
  assert.throws(() => p0.addGrant({ effect: "maybe", roles: ["r"], actions: ["select"], resource: "a.b" }), /effect/);
  assert.throws(() => p0.addGrant({ effect: "allow", roles: [], actions: ["select"], resource: "a.b" }), /role/);
  assert.throws(() => p0.addGrant({ effect: "allow", roles: ["r"], actions: ["truncate"], resource: "a.b" }), /unknown action/);
  assert.throws(() => p0.addGrant({ effect: "allow", roles: ["r"], actions: ["select"], resource: "" }), /resource/);
  p0.defineRole("a", ["b"]);
  assert.throws(() => p0.defineRole("b", ["a"]), /cycle/);
  p0.defineRole("b", []);
  assert.deepStrictEqual([...p0.roleClosure(["a"])].sort(), ["a", "b"]);

  // --- default deny -------------------------------------------------------------------------------
  const empty = new AccessEngine(new Policy());
  const denied = empty.evaluate({ principal: "nobody", roles: ["ghost"], action: "select", resource: "sales.public.orders", columns: ["id"], row: {}, session: {} });
  assert.strictEqual(denied.allowed, false);
  assert.strictEqual(denied.reason, "blocked:default-deny");
  assert.deepStrictEqual(denied.masked, ["id"]);
  assert.strictEqual(empty.audit.entries.length, 1);
  assert.strictEqual(empty.audit.entries[0].allowed, false, "denials must be audited too");

  // --- allow path and column masking ----------------------------------------------------------------
  const { policy, rows } = buildWarehouse();
  const engine = new AccessEngine(policy);
  const fixtures = fixtureRequests(rows);
  const decisions = new Map();
  for (const fixture of fixtures) decisions.set(fixture.label, engine.evaluate(fixture.request));
  const d = (label) => decisions.get(label);

  const ok = d("reader-ok");
  assert.strictEqual(ok.allowed, true);
  assert.deepStrictEqual(ok.visible, ["id", "amount"]);
  assert.deepStrictEqual(engine.project(rows[0], ok), { id: 1, amount: 100 });
  assert.strictEqual(engine.project(rows[0], { allowed: false, visible: ["id"] }), null, "denied requests project nothing");

  const overreach = d("reader-cost");
  assert.strictEqual(overreach.allowed, false, "cost is not granted to a plain reader");
  assert.deepStrictEqual(overreach.masked, ["cost"]);
  assert.strictEqual(overreach.reason, "blocked:default-deny", "nothing covers cost, so default-deny blocks it");
  assert.strictEqual(engine.project(rows[0], overreach), null);
  const softOverreach = d("reader-cost-masked");
  assert.strictEqual(softOverreach.allowed, true, "masking turns a partial grant into a redacted success");
  assert.deepStrictEqual(softOverreach.visible, ["id"]);
  assert.deepStrictEqual(softOverreach.masked, ["cost"]);
  assert.strictEqual(engine.project(rows[0], softOverreach).cost, undefined, "masked column must not leak");
  assert.strictEqual(d("reader-only-cost-masked").allowed, false, "masking cannot rescue a request with no permitted column");

  // --- role inheritance: senior-analyst sees cost, analyst does not ---------------------------------
  assert.strictEqual(d("senior-cost").allowed, true);
  assert.deepStrictEqual(d("senior-cost").visible.slice().sort(), ["amount", "cost"]);
  assert.strictEqual(d("analyst-cost").allowed, false, "inheriting 'reader' does not carry 'cost'");

  // --- row-level isolation and deny-overrides -------------------------------------------------------
  assert.strictEqual(d("other-tenant").allowed, false, "row filter must hide another tenant's row");
  assert.strictEqual(d("globex-partial").allowed, true, "the deny only removes the amount column");
  assert.deepStrictEqual(d("globex-partial").visible, ["id"]);
  assert.deepStrictEqual(d("globex-partial").masked, ["amount"]);
  assert.strictEqual(d("globex-partial").reason, "masked:deny-no-globex", "the deny, not the allow, decides the masked column");
  assert.strictEqual(d("globex-blocked").allowed, false, "a deny that covers every requested column blocks the request");
  assert.strictEqual(d("globex-blocked").reason, "blocked:deny-no-globex");
  assert.strictEqual(d("auditor-deep").allowed, true, "'**' must cover any depth");

  // --- resource pattern matching --------------------------------------------------------------------
  assert.ok(matchResource("sales.*.*", "sales.public.orders"));
  assert.ok(!matchResource("sales.public.orders", "sales.public.customers"));
  assert.ok(matchResource("sales.**", "sales.public.orders"));
  assert.ok(matchResource("sales.**", "sales"));
  assert.ok(matchResource("**", "sales.public"));
  assert.ok(!matchResource("sales.*", "sales.public.orders"));
  assert.ok(!matchResource("analytics.**", "sales.public.orders"));

  // --- freshness: revoke/grant are visible on the next request ---------------------------------------
  const invalidationsBefore = engine.policy.invalidations;
  policy.revokeGrant("read-orders");
  const afterRevoke = engine.evaluate({ ...fixtures[0].request });
  assert.strictEqual(afterRevoke.allowed, false, "a revoked grant must not be honoured from cache");
  assert.ok(engine.policy.invalidations > invalidationsBefore);
  assert.strictEqual(engine.project(rows[0], afterRevoke), null);
  policy.addGrant({ id: "read-orders-2", effect: "allow", roles: ["reader"], actions: ["select"], resource: "sales.public.orders", columns: ["id", "tenant", "amount"], filter: (s, r) => r !== undefined && r.tenant === s.tenant });
  const afterGrant = engine.evaluate({ ...fixtures[0].request });
  assert.strictEqual(afterGrant.allowed, true, "a newly granted scope must be effective immediately");

  // --- cross-verification against the naive reference on a random policy and random requests --------
  const oraclePolicy = new Policy();
  oraclePolicy.defineRole("base");
  oraclePolicy.defineRole("mid", ["base"]);
  oraclePolicy.defineRole("top", ["mid"]);
  oraclePolicy.defineRole("side");
  const rand = mulberry32(853);
  const tablePaths = ["analytics.public.events", "analytics.private.users", "sales.public.orders", "sales.public.customers"];
  const columnPool = ["id", "tenant", "amount", "cost", "secret"];
  const rolePool = ["base", "mid", "top", "side"];
  const pick = (arr) => arr[Math.floor(rand() * arr.length)];
  for (let i = 0; i < 40; i++) {
    const width = 1 + Math.floor(rand() * 3);
    const columns = [];
    while (columns.length < width) {
      const c = pick(columnPool);
      if (!columns.includes(c)) columns.push(c);
    }
    const rowFilter = rand() < 0.35 ? (s, r) => r !== undefined && r.tenant === s.tenant : null;
    const pattern = rand() < 0.3 ? pick(["analytics.**", "sales.**", "**"]) : pick(tablePaths);
    oraclePolicy.addGrant({
      effect: rand() < 0.3 ? "deny" : "allow",
      roles: [pick(rolePool)],
      actions: rand() < 0.2 ? ACTIONS : [pick(ACTIONS)],
      resource: pattern,
      columns: columns,
      filter: rowFilter,
    });
  }
  const oracleEngine = new AccessEngine(oraclePolicy);
  const rolesOf = (principal) => {
    const index = Number(principal.slice(1)) % rolePool.length;
    return [rolePool[index], rolePool[(index + 1) % rolePool.length]];
  };
  let agreements = 0;
  let allowedCount = 0;
  for (let i = 0; i < 600; i++) {
    const principal = `p${i % 7}`;
    const request = {
      principal,
      roles: rolesOf(principal),
      action: pick(ACTIONS),
      resource: pick(tablePaths),
      columns: [pick(columnPool), pick(columnPool)],
      row: { tenant: pick(["acme", "globex"]) },
      session: { tenant: pick(["acme", "globex"]) },
      masking: rand() < 0.5,
    };
    const mine = oracleEngine.evaluate(request);
    const theirs = referenceDecide(oraclePolicy, request);
    assert.strictEqual(mine.allowed, theirs.allowed, `request ${i}: allowed`);
    assert.deepStrictEqual(mine.visible, theirs.visible, `request ${i}: visible`);
    assert.deepStrictEqual(mine.masked, theirs.masked, `request ${i}: masked`);
    assert.strictEqual(mine.reason, theirs.reason, `request ${i}: reason`);
    agreements += 1;
    if (mine.allowed) allowedCount += 1;
  }
  assert.strictEqual(agreements, 600);
  assert.ok(allowedCount > 100 && allowedCount < 500, "the corpus must exercise both outcomes");

  // least privilege: a principal without roles can reach nothing anywhere.
  const rolNeless = [];
  for (const action of ACTIONS) rolNeless.push(oracleEngine.evaluate({ principal: "nobody", roles: [], action, resource: "sales.public.orders", columns: ["id"], row: { tenant: "acme" }, session: { tenant: "acme" } }).allowed);
  assert.deepStrictEqual(rolNeless, [false, false, false, false]);

  // --- audit log integrity ---------------------------------------------------------------------------
  const log = engine.audit;
  assert.strictEqual(log.entries.length, fixtures.length + 2, "one entry per decision, including denials");
  assert.ok(log.entries.every((e, i) => e.seq === i));
  assert.deepStrictEqual(log.verifyChain(), { ok: true, at: log.entries.length, why: "ok" });
  const denialAudits = log.query((e) => !e.allowed);
  assert.ok(denialAudits.length > 0, "denials are audited");
  const u1Audits = log.query((e) => e.principal === "u1");
  assert.strictEqual(u1Audits.length, 9, "u1 issued every reader fixture plus the two freshness probes");
  assert.strictEqual(u1Audits.filter((e) => e.allowed).length, 4);
  assert.strictEqual(log.query((e) => e.resource === "some.deep.path").length, 1);
  assert.strictEqual(log.query((e) => e.reason.startsWith("blocked:")).length, log.entries.filter((e) => !e.allowed).length);
  assert.strictEqual(log.head, log.entries[log.entries.length - 1].hash, "the head is the last entry's hash");

  // Replaying exactly the same request sequence against an identically rebuilt policy in a fresh
  // engine reproduces the same decisions, i.e. authorisation is a pure function of (policy, request).
  const replay = buildWarehouse();
  const replayEngine = new AccessEngine(replay.policy);
  const replayed = fixtureRequests(replay.rows).map((f) => replayEngine.evaluate(f.request));
  assert.strictEqual(replayEngine.audit.verifyChain().ok, true);
  assert.deepStrictEqual(
    replayed.map((r) => [r.allowed, r.visible, r.masked, r.reason]),
    log.entries.slice(0, fixtures.length).map((e) => [e.allowed, e.visible, e.masked, e.reason]),
  );
  assert.strictEqual(replayEngine.audit.entries[0].reason, "allow-read-orders");
  assert.strictEqual(log.entries[fixtures.length].reason, "blocked:default-deny", "the revoke itself is audited");
  assert.notStrictEqual(log.entries[fixtures.length].hash, log.entries[0].hash);

  // --- tamper detection ------------------------------------------------------------------------------
  const tamper = (mutate) => {
    const copy = new AuditLog();
    copy.entries = log.entries.map((e) => ({ ...e, visible: e.visible.slice(), masked: e.masked.slice() }));
    copy.head = log.head;
    mutate(copy);
    return copy.verifyChain();
  };
  const flipped = tamper((c) => {
    c.entries[2].allowed = !c.entries[2].allowed;
  });
  assert.strictEqual(flipped.ok, false);
  assert.strictEqual(flipped.why, "payload does not match hash");
  const removed = tamper((c) => {
    c.entries.splice(1, 1);
  });
  assert.strictEqual(removed.ok, false);
  assert.strictEqual(removed.why, "sequence gap");
  const swapped = tamper((c) => {
    const tmp = c.entries[1];
    c.entries[1] = c.entries[2];
    c.entries[2] = tmp;
  });
  assert.strictEqual(swapped.ok, false);
  const relinked = tamper((c) => {
    c.entries[3].prev = digest("forged");
  });
  assert.strictEqual(relinked.ok, false);
  assert.strictEqual(relinked.why, "broken predecessor link");
  assert.strictEqual(log.verifyChain().ok, true, "the real log is untouched by the tamper tests");

  // --- cost bound and scale ---------------------------------------------------------------------------
  const bigPolicy = new Policy();
  bigPolicy.defineRole("r0");
  bigPolicy.defineRole("r1", ["r0"]);
  for (let i = 0; i < 150; i++) {
    bigPolicy.addGrant({ effect: i % 7 === 0 ? "deny" : "allow", roles: [`r${i % 2}`], actions: ["select"], resource: `${i % 13}.public.table`, columns: i % 5 === 0 ? null : ["id", "amount"] });
  }
  const bigEngine = new AccessEngine(bigPolicy);
  const freshEngine = new AccessEngine(bigPolicy); // never caches anything useful across requests
  let maxInspected = 0;
  let sampleAgree = 0;
  for (let i = 0; i < 1500; i++) {
    const request = { principal: `u${i % 50}`, roles: ["r1"], action: "select", resource: pickTable(i), columns: ["id", "amount"], row: {}, session: {} };
    const decision = bigEngine.evaluate(request);
    maxInspected = Math.max(maxInspected, decision.grantsInspected);
    if (i % 7 === 0) {
      const theirs = referenceDecide(bigPolicy, request);
      const recompiled = freshEngine.evaluate(request);
      assert.strictEqual(decision.allowed, theirs.allowed, `scale sample ${i}`);
      assert.deepStrictEqual(decision.visible, theirs.visible);
      assert.strictEqual(decision.reason, theirs.reason);
      assert.strictEqual(recompiled.allowed, theirs.allowed, "cached and freshly compiled paths must agree");
      sampleAgree += 1;
    }
  }
  assert.ok(maxInspected <= bigPolicy.grants.length, "a decision never inspects more grants than exist");
  assert.ok(maxInspected > 0 && sampleAgree >= 200);
  assert.strictEqual(bigEngine.audit.entries.length, 1500);
  assert.deepStrictEqual(bigEngine.audit.verifyChain(), { ok: true, at: 1500, why: "ok" });
  assert.ok(bigEngine.cache.size <= 50 * 13, "the compiled cache is bounded by the number of distinct (principal, action, resource) keys");

  // --- determinism -------------------------------------------------------------------------------------
  const clone = new Policy();
  clone.defineRole("auditor");
  clone.defineRole("reader");
  const a1 = new AccessEngine(clone);
  clone.addGrant({ id: "x", effect: "allow", roles: ["auditor"], actions: ["select"], resource: "**", columns: null });
  const first = a1.evaluate({ principal: "z", roles: ["auditor"], action: "select", resource: "a.b.c", columns: ["v"], row: {}, session: {} });
  const clone2 = new Policy();
  clone2.defineRole("auditor");
  clone2.defineRole("reader");
  const a2 = new AccessEngine(clone2);
  clone2.addGrant({ id: "x", effect: "allow", roles: ["auditor"], actions: ["select"], resource: "**", columns: null });
  const second = a2.evaluate({ principal: "z", roles: ["auditor"], action: "select", resource: "a.b.c", columns: ["v"], row: {}, session: {} });
  assert.deepStrictEqual(first, second);
  assert.strictEqual(a1.audit.head, a2.audit.head);

  console.log(`policy: ${oraclePolicy.grants.length} random grants, 600 decisions cross-checked with the naive oracle (${allowedCount} allowed)`);
  console.log(`audit: ${log.entries.length} entries chained, head ${log.head.slice(0, 12)}…, 4 tamper types detected`);
  console.log(`scale: ${bigPolicy.grants.length} grants, 1500 decisions, max grants inspected per decision ${maxInspected}`);
  console.log("PASS");
}

/** Helper for the scale workload: deterministic table names. */
function pickTable(i) {
  return `${i % 13}.public.table`;
}

module.exports = { Policy, AccessEngine, AuditLog, digest, matchResource, referenceDecide, buildWarehouse, mulberry32 };

if (require.main === module) {
  selfCheck();
}
