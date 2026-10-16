/** Task 984 - Implement legal/privacy compliance checks for user data handling in platform.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * The platform keeps a processing inventory: for every system a purpose, a lawful basis, a retention
 * period, a transfer destination and the fields it stores, plus the actual rows. The auditor applies
 * a small rule set - lawful basis, special-category data without a DPIA, retention over the policy
 * maximum, missing consent for consent-based purposes, fields with no declared purpose, transfers
 * outside the adequacy list without SCCs - and reports a coded violation per issue. Erasure requests
 * are then executed across every system: rows are dropped, the residual systems are named, and the
 * audit log has to stay free of the erased personal data.
 * Simplified: the inventory is in memory, "today" is a fixed integer day and the rules are the
 * platform's own policy rather than a full legal review. Verified: the compliant fixture, one coded
 * violation per planted issue, retention recomputed independently with an exact-limit boundary,
 * purging, erasure with residuals, a PII-free audit log and determinism.
 */
"use strict";
const assert = require("node:assert");

const POLICY = {
  today: 1000,
  lawfulBases: ["consent", "contract", "legal-obligation", "legitimate-interest"],
  specialCategories: ["health", "biometric", "political", "religion"],
  adequacy: ["EU", "UK", "CH", "JP"],
  consentPurposes: ["marketing", "analytics"],
  maxRetentionDays: { analytics: 400, marketing: 730, support: 1825 },
};

const COMPLIANT = [
  { system: "tickets", purpose: "support", lawfulBasis: "contract", retentionDays: 1825, transfer: { country: "EU" },
    fields: [{ name: "email", pii: true, purpose: "reply to the customer" }],
    rows: [{ id: "t1", subject: "u1", createdAt: 900, values: { email: "u1@example.com" } }] },
  { system: "metrics", purpose: "analytics", lawfulBasis: "consent", retentionDays: 300,
    transfer: { country: "US", sccs: true },
    fields: [{ name: "eventCount", pii: false, purpose: "aggregate reporting" }],
    rows: [{ id: "m1", subject: "u1", createdAt: 800, values: { eventCount: "17" } }] },
];
const FLAWED = [
  { system: "tickets", purpose: "support", lawfulBasis: "vibes", retentionDays: 1825, transfer: { country: "EU" },
    fields: [{ name: "email", pii: true, purpose: "reply to the customer" }],
    rows: [{ id: "t1", subject: "u1", createdAt: 900, values: { email: "u1@example.com" } }] },
  { system: "profile", purpose: "support", lawfulBasis: "contract", retentionDays: 100, transfer: { country: "US" },
    fields: [{ name: "email", pii: true, purpose: "account management" },
      { name: "religion", pii: true, category: "religion", purpose: "personalization" }],
    rows: [{ id: "p1", subject: "u1", createdAt: 700, values: { email: "u1@example.com" } }] },
  { system: "campaigns", purpose: "marketing", lawfulBasis: "consent", retentionDays: 730, transfer: { country: "EU" },
    fields: [{ name: "email", pii: true, purpose: "send offers" }, { name: "deviceId", pii: true }],
    rows: [{ id: "c1", subject: "u2", createdAt: 950, values: { deviceId: "d-42" } }] },
];
const CONSENTS = { u1: { marketing: true, analytics: true, support: true }, u2: { marketing: false } };

/** Effective retention limit: the system may not exceed the policy maximum for its purpose. */
const retentionLimit = (system, policy = POLICY) =>
  Math.min(system.retentionDays ?? Infinity, policy.maxRetentionDays[system.purpose] ?? Infinity);
const ageOf = (row, policy = POLICY) => policy.today - row.createdAt;

/** Audit the inventory and return coded violations, ordered by code then system. */
function audit(inventory, { policy = POLICY, consents = CONSENTS } = {}) {
  const violations = [];
  const add = (code, system, detail) => violations.push({ code, system, detail });
  for (const system of inventory) {
    if (!policy.lawfulBases.includes(system.lawfulBasis)) {
      add("NO-LAWFUL-BASIS", system.system, `lawfulBasis "${system.lawfulBasis}" is not recognised`);
    }
    for (const field of system.fields ?? []) {
      if (field.purpose === undefined) add("PURPOSE-MISSING", system.system, `field ${field.name} has no declared purpose`);
      if (field.category && policy.specialCategories.includes(field.category) && system.dpia !== true) {
        add("SPECIAL-CATEGORY-NO-DPIA", system.system, `field ${field.name} (${field.category}) has no DPIA`);
      }
    }
    if (system.transfer && !policy.adequacy.includes(system.transfer.country) && system.transfer.sccs !== true) {
      add("TRANSFER-NO-SCC", system.system, `transfer to ${system.transfer.country} has no safeguards`);
    }
    const limit = retentionLimit(system, policy);
    for (const row of system.rows ?? []) {
      if (ageOf(row, policy) > limit) add("RETENTION-EXCEEDED", system.system, `row ${row.id} is ${ageOf(row, policy)} days old`);
      if (policy.consentPurposes.includes(system.purpose) && consents[row.subject]?.[system.purpose] !== true) {
        add("CONSENT-MISSING", system.system, `subject ${row.subject} never consented to ${system.purpose}`);
      }
    }
  }
  return { violations: violations.sort((a, b) => (a.code < b.code ? -1 : a.code > b.code ? 1
    : a.system < b.system ? -1 : 1)), summary: summarize(inventory) };
}

function summarize(inventory, policy = POLICY) {
  return {
    systems: inventory.length,
    rows: inventory.reduce((sum, system) => sum + (system.rows ?? []).length, 0),
    piiFields: inventory.reduce((sum, system) => sum + (system.fields ?? []).filter((field) => field.pii).length, 0),
    specialCategoryFields: inventory.reduce((sum, system) => sum + (system.fields ?? [])
      .filter((field) => policy.specialCategories.includes(field.category)).length, 0),
  };
}

/** Purge rows past the retention limit; the counts must conserve every row. */
function purgeExpired(inventory, { policy = POLICY } = {}) {
  const purged = [];
  const kept = inventory.map((system) => ({
    ...system,
    rows: (system.rows ?? []).filter((row) => {
      if (ageOf(row, policy) > retentionLimit(system, policy)) { purged.push(`${system.system}/${row.id}`); return false; }
      return true;
    }),
  }));
  return { kept, purged, totalBefore: summarize(inventory).rows, totalAfter: summarize(kept).rows };
}

/** Erase a subject everywhere; systems that keep the data are reported as residual. */
function eraseSubject(inventory, subject, { policy = POLICY, failing = [] } = {}) {
  const erased = [];
  const residual = [];
  const log = [];
  const kept = inventory.map((system) => {
    const doomed = (system.rows ?? []).filter((row) => row.subject === subject);
    if (doomed.length === 0) return system;
    if (failing.includes(system.system)) {
      residual.push(system.system);
      log.push({ action: "erasure-failed", system: system.system, rows: doomed.length });
      return system;
    }
    erased.push(system.system);
    log.push({ action: "erasure", system: system.system, rows: doomed.length, lawfulBasis: system.lawfulBasis,
      withinDays: policy.today - policy.today });
    return { ...system, rows: (system.rows ?? []).filter((row) => row.subject !== subject) };
  });
  return { inventory: kept, erased: erased.sort(), residual: residual.sort(), log,
    personalData: inventory.flatMap((system) => (system.rows ?? [])
      .filter((row) => row.subject === subject).flatMap((row) => Object.values(row.values ?? {}))) };
}

function main() {
  const clean = audit(COMPLIANT);
  const flawed = audit(FLAWED);

  // 1. Happy path: the compliant inventory produces no violations and an exact summary.
  assert.deepStrictEqual(clean.violations, [], `compliant inventory: ${JSON.stringify(clean.violations)}`);
  assert.deepStrictEqual(clean.summary, { systems: 2, rows: 2, piiFields: 1, specialCategoryFields: 0 }, "summary");
  assert.strictEqual(audit([]).violations.length, 0, "an empty inventory is trivially compliant");
  assert.deepStrictEqual(audit([]).summary, { systems: 0, rows: 0, piiFields: 0, specialCategoryFields: 0 }, "empty summary");

  // 2. One coded violation per planted issue, ordered deterministically.
  assert.deepStrictEqual(flawed.violations.map((violation) => `${violation.code}@${violation.system}`), [
    "CONSENT-MISSING@campaigns", "NO-LAWFUL-BASIS@tickets", "PURPOSE-MISSING@campaigns",
    "RETENTION-EXCEEDED@profile", "SPECIAL-CATEGORY-NO-DPIA@profile", "TRANSFER-NO-SCC@profile",
  ], `violations: ${flawed.violations.map((v) => v.code).join(",")}`);
  assert.deepStrictEqual(flawed.summary, { systems: 3, rows: 3, piiFields: 5, specialCategoryFields: 1 }, "flawed summary");
  assert(flawed.violations.find((v) => v.code === "RETENTION-EXCEEDED").detail.includes("300 days"), "the age is reported");

  // 3. Oracle and boundary: retention recomputed from raw ages, limit exactly at the edge.
  const manual = FLAWED.flatMap((system) => (system.rows ?? [])
    .filter((row) => POLICY.today - row.createdAt > Math.min(system.retentionDays, POLICY.maxRetentionDays[system.purpose]))
    .map((row) => `${system.system}/${row.id}`));
  assert.deepStrictEqual(flawed.violations.filter((v) => v.code === "RETENTION-EXCEEDED").map((v) => v.detail.split(" ")[1]),
    manual.map((entry) => entry.split("/")[1]), "the flagged rows match an independent age scan");
  const edge = [{ system: "edge", purpose: "analytics", lawfulBasis: "consent", retentionDays: 100, transfer: { country: "EU" },
    fields: [{ name: "n", pii: false, purpose: "counts" }],
    rows: [{ id: "e1", subject: "u1", createdAt: 900 }, { id: "e2", subject: "u1", createdAt: 899 }] }];
  assert.deepStrictEqual(audit(edge).violations.map((v) => v.detail),
    ["row e2 is 101 days old"], "exactly at the limit is kept, one day over is flagged");
  assert.strictEqual(retentionLimit(edge[0]), 100, "the tighter of policy and system wins");
  assert.strictEqual(retentionLimit({ purpose: "support", retentionDays: 9999 }), 1825, "the policy caps the system");

  // 4. Consent: purpose-limited systems need consent per subject, and other purposes do not.
  assert.strictEqual(audit([{ ...COMPLIANT[0], purpose: "support" }]).violations.length, 0,
    "support rows do not need marketing consent");
  assert(audit(FLAWED).violations.some((v) => v.detail.includes("u2")), "the non-consenting subject is named");
  const consented = FLAWED.map((system) => (system.system === "campaigns"
    ? { ...system, fields: system.fields.map((field) => ({ ...field, purpose: field.purpose ?? "send offers" })) } : system));
  assert.strictEqual(audit(consented, { consents: { u1: CONSENTS.u1, u2: { marketing: true } } })
    .violations.filter((v) => v.code === "CONSENT-MISSING").length, 0, "consent clears the violation");

  // 5. Purge: exactly the expired rows go, and nothing is lost or duplicated.
  const purged = purgeExpired(FLAWED);
  assert.deepStrictEqual(purged.purged, ["profile/p1"], "only the over-retention row is purged");
  assert.strictEqual(purged.totalAfter + purged.purged.length, purged.totalBefore, "rows are conserved");
  assert.strictEqual(purgeExpired(COMPLIANT).purged.length, 0, "a compliant inventory purges nothing");
  assert.deepStrictEqual(purgeExpired(edge).purged, ["edge/e2"], "the boundary row survives");

  // 6. Erasure: rows disappear everywhere, residuals are named, and the log carries no PII.
  const erased = eraseSubject(COMPLIANT, "u1");
  assert.deepStrictEqual(erased.erased, ["metrics", "tickets"], "every holding system erases");
  assert.deepStrictEqual(erased.residual, [], "no residual data");
  assert.strictEqual(summarize(erased.inventory).rows, 0, "no rows are left for that subject");
  assert.strictEqual(summarize(COMPLIANT).rows, 2, "the original inventory is untouched");
  const stuck = eraseSubject(COMPLIANT, "u1", { failing: ["tickets"] });
  assert.deepStrictEqual(stuck.residual, ["tickets"], "a stubborn system is reported");
  assert.deepStrictEqual(stuck.erased, ["metrics"], "the others still erase");
  assert(erased.personalData.includes("u1@example.com"), "the erased values are known to the caller");
  assert.strictEqual(erased.log.length, 2, "one log entry per system");
  for (const entry of erased.log) {
    const serialized = JSON.stringify(entry);
    assert(erased.personalData.every((value) => !serialized.includes(value)),
      `the audit log must stay PII-free: ${serialized}`);
  }
  assert.deepStrictEqual(eraseSubject(COMPLIANT, "nobody").log, [], "erasing an unknown subject is a no-op");

  // 7. Untouched systems and unknown bases are handled explicitly.
  assert.deepStrictEqual(eraseSubject(COMPLIANT, "u1").inventory.map((system) => system.system), ["tickets", "metrics"],
    "systems keep their order");
  assert(audit([{ system: "x", purpose: "marketing", lawfulBasis: "legitimate-interest", fields: [], rows: [] }])
    .violations.length === 0, "a recognised lawful basis passes");
  assert(audit([{ system: "x", purpose: "analytics", lawfulBasis: "consent", fields: [], rows: [] }]).violations.length === 0,
    "a system with no rows and no fields passes");

  // 8. Determinism: identical input, identical report.
  assert(JSON.stringify(audit(FLAWED)) === JSON.stringify(flawed), "audits are reproducible");
  assert(JSON.stringify(audit([...FLAWED].reverse())) === JSON.stringify(flawed), "the report is order-independent");
  assert.deepStrictEqual(audit([...FLAWED].reverse()).violations, flawed.violations, "the violation list is normalised");
  assert(JSON.stringify(purgeExpired(FLAWED).purged) === JSON.stringify(purged.purged), "purging is reproducible");

  console.log(`privacy audit: ${clean.violations.length} violations on the compliant fixture, ` +
    `${flawed.violations.length} on the flawed one (${[...new Set(flawed.violations.map((v) => v.code))].join(",")}), ` +
    `purged ${purged.purged.length} row(s), erasure log ${erased.log.length} entries`);
  console.log("PASS");
}
module.exports = { POLICY, COMPLIANT, FLAWED, CONSENTS, retentionLimit, ageOf, audit, summarize, purgeExpired,
  eraseSubject };
if (require.main === module) main();
