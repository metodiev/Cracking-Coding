/** Task 915 - Implement secure remote attestation flow integrated with orchestration for trust chain.
 *  (Tier: Master, CTCI Beyond — Confidential Computing & Trust Chains)
 *
 * A simulated measured boot: each component appends a digest to the event log and the platform extends
 * the PCR, pcr <- SHA-256(pcr || digest), starting from 32 zero bytes. A node answers an orchestration
 * challenge with a quote carrying {pcrDigest, counter, workloadId, nonce} signed by its endorsement key
 * (HMAC-SHA256 stands in for the TPM signature so runs stay deterministic and dependency-free).
 *
 * The orchestrator accepts a quote only when *all* of the following hold, each of which has its own test:
 * the signature recomputes; the nonce is the one it just issued and has not been spent (freshness/replay);
 * the PCR digest equals the golden value derived by independently replaying the approved event log (trust
 * chain, not a self-report); the monotonic counter never goes backwards (rollback); the node is not
 * revoked; and the quote names the workload that is being admitted (binding). Every decision lands in an
 * audit trail with a reason, so a denial is never silent.
 */
"use strict";
const assert = require("node:assert");
const crypto = require("node:crypto");
const sha256 = (buffer) => crypto.createHash("sha256").update(buffer).digest();
const hmac = (key, text) => crypto.createHmac("sha256", key).update(text, "utf8").digest("hex");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  hex(bytes) {
    let out = "";
    while (out.length < bytes * 2) out += this.next().toString(16).padStart(8, "0");
    return out.slice(0, bytes * 2);
  }
}
const ZERO_PCR = Buffer.alloc(32).toString("hex");
/** Extend one PCR with a measurement digest (the platform's only mutable state). */
const extend = (pcrHex, digestHex) => sha256(Buffer.concat([Buffer.from(pcrHex, "hex"), Buffer.from(digestHex, "hex")])).toString("hex");
/** Replay an event log into the PCR value it implies — the independent trust-chain oracle. */
function replayLog(log, initial = ZERO_PCR) {
  return log.reduce((pcr, event) => {
    assert.ok(Number.isInteger(event.pcr) && event.pcr >= 0, "a log entry needs a PCR index");
    return extend(pcr, sha256(Buffer.from(event.component, "utf8")).toString("hex"));
  }, initial);
}
const measurementOf = (component) => sha256(Buffer.from(component, "utf8")).toString("hex");
/** The approved boot: the golden PCR the orchestration policy expects. */
function approvedGolden(bootLog) {
  const pcr = replayLog(bootLog);
  return { digest: pcr, log: bootLog.map((event) => ({ ...event })) };
}
/** A node: endorsement secret, monotonic counter and its event log. */
class Node {
  constructor(name, serverKey, log, counter) {
    this.name = name;
    this.serverKey = serverKey;
    this.log = log;
    this.counter = counter;
    this.revoked = false;
  }
  quote(nonce, workloadId, override = {}) {
    const body = {
      node: this.name, pcr: override.pcr ?? replayLog(this.log),
      counter: override.counter ?? this.counter, workloadId: override.workloadId ?? workloadId,
      nonce, };
    const text = `${body.node}|${body.pcr}|${body.counter}|${body.workloadId}|${body.nonce}`;
    return { ...body, signature: override.signature ?? hmac(this.serverKey, text), text };
  }
}
/** The orchestrator: issues challenges, verifies quotes, keeps the audit trail. */
class Orchestrator {
  constructor(prng) {
    this.prng = prng;
    this.challenges = new Map();
    this.spentNonces = new Set();
    this.counterFloor = new Map();
    this.audit = [];
    this.issued = 0;
  }
  challenge(nodeName, workloadId) {
    this.issued += 1;
    const nonce = `${this.issued}:${this.prng.hex(16)}`;
    this.challenges.set(nodeName, { nonce, workloadId });
    return nonce;
  }
  verify(quote, node, golden) {
    const record = (verdict, reason) => {
      this.audit.push({ node: node.name, verdict, reason });
      return { trusted: verdict === "trusted", reason };
    };
    const challenge = this.challenges.get(node.name);
    if (!challenge) return record("denied", "no outstanding challenge");
    if (quote.nonce !== challenge.nonce) return record("denied", "stale or unknown nonce");
    if (this.spentNonces.has(quote.nonce)) return record("denied", "nonce already spent");
    const text = `${quote.node}|${quote.pcr}|${quote.counter}|${quote.workloadId}|${quote.nonce}`;
    if (hmac(node.serverKey, text) !== quote.signature) return record("denied", "signature mismatch");
    if (quote.pcr !== golden.digest) return record("denied", "measurement does not match the approved boot");
    if (quote.workloadId !== challenge.workloadId) return record("denied", "workload binding mismatch");
    if (node.revoked) return record("denied", "node revoked");
    const floor = this.counterFloor.get(node.name) ?? -Infinity;
    if (quote.counter < floor) return record("denied", "counter rollback");
    this.counterFloor.set(node.name, quote.counter);
    this.spentNonces.add(quote.nonce);
    return record("trusted", "quote verified against the approved boot");
  }
}

function main() {
  const prng = new Prng(0x5eed0915);
  const serverKey = "endorsement-key-915";
  const bootLog = [
    { pcr: 0, component: "firmware-1.2.3" }, { pcr: 0, component: "bootloader-4.5" },
    { pcr: 0, component: "kernel-6.1.0" }, { pcr: 1, component: "orchestrator-agent-2.0" },
  ];
  const golden = approvedGolden(bootLog);
  const workload = "workload-inference-7";
  const node = new Node("node-a", serverKey, bootLog, 12);
  const orchestrator = new Orchestrator(prng);
  // 1. the trust chain — the golden digest comes from replaying the log, not from the node's word.
  {
    assert.strictEqual(golden.digest, replayLog(bootLog), "the golden digest is the log's fixpoint");
    assert.notStrictEqual(golden.digest, ZERO_PCR, "the boot really measured something");
    const shorter = replayLog(bootLog.slice(0, 3));
    assert.notStrictEqual(shorter, golden.digest, "dropping a measurement changes the chain");
    assert.strictEqual(replayLog([]), ZERO_PCR, "an empty log leaves the PCR at zero");
    const single = [{ pcr: 0, component: "only" }];
    assert.strictEqual(replayLog(single), extend(ZERO_PCR, measurementOf("only")), "one extend step");
  }
  // 2. happy path — a fresh, correctly signed quote over the approved boot is trusted.
  {
    const nonce = orchestrator.challenge(node.name, workload);
    const verdict = orchestrator.verify(node.quote(nonce, workload), node, golden);
    assert.deepStrictEqual(verdict, { trusted: true, reason: "quote verified against the approved boot" });
    assert.strictEqual(orchestrator.audit.length, 1);
    node.counter += 1;
  }
  // 3. replay — the same quote, and a probe with an old nonce, are both denied.
  {
    const spent = [...orchestrator.spentNonces][0];
    orchestrator.challenge(node.name, workload);
    const replay = orchestrator.verify(node.quote(spent, workload), node, golden);
    assert.strictEqual(replay.trusted, false);
    assert.match(replay.reason, /stale or unknown nonce/, `got ${replay.reason}`);
    const nonce = orchestrator.challenge(node.name, workload);
    const honest = node.quote(nonce, workload);
    assert.strictEqual(orchestrator.verify(honest, node, golden).trusted, true, "a fresh nonce works");
    node.counter += 1;
    const second = orchestrator.verify(honest, node, golden);
    assert.strictEqual(second.trusted, false, "a nonce is single use");
    assert.match(second.reason, /already spent/, `got ${second.reason}`);
    node.counter += 1;
  }
  // 4. tampered measurement — a modified event log changes the PCR and is caught by the policy.
  {
    const tampered = new Node("node-b", serverKey, [...bootLog, { pcr: 0, component: "rootkit" }], 3);
    const verdict = orchestrator.verify(tampered.quote(orchestrator.challenge(tampered.name, workload), workload),
      tampered, golden);
    assert.strictEqual(verdict.trusted, false);
    assert.match(verdict.reason, /approved boot/, `got ${verdict.reason}`);
    assert.notStrictEqual(replayLog(tampered.log), golden.digest);
    const rewritten = new Node("node-c", serverKey, bootLog.map((event) =>
      event.component === "kernel-6.1.0" ? { ...event, component: "kernel-6.1.1" } : event), 1);
    assert.strictEqual(orchestrator.verify(rewritten.quote(orchestrator.challenge(rewritten.name, workload), workload),
      rewritten, golden).trusted, false, "a single swapped component is enough to fail");
  }
  // 5. forged signatures and a wrong key are denied before any policy check runs.
  {
    const nonce = orchestrator.challenge(node.name, workload);
    const forged = node.quote(nonce, workload, { signature: "deadbeef" });
    const verdict = orchestrator.verify(forged, node, golden);
    assert.strictEqual(verdict.trusted, false);
    assert.match(verdict.reason, /signature mismatch/);
    const impostor = new Node(node.name, "other-key", bootLog, 99);
    const stolen = node.quote(orchestrator.challenge(node.name, workload), workload);
    assert.strictEqual(orchestrator.verify(stolen, impostor, golden).trusted, false, "a stolen quote fails");
  }
  // 6. revocation — a node dropped from the fleet is denied even with a perfect quote.
  {
    const victim = new Node("node-d", serverKey, bootLog, 5);
    const nonce = orchestrator.challenge(victim.name, workload);
    assert.strictEqual(orchestrator.verify(victim.quote(nonce, workload), victim, golden).trusted, true);
    victim.revoked = true;
    victim.counter += 1;
    const afterRevocation = orchestrator.verify(
      victim.quote(orchestrator.challenge(victim.name, workload), workload), victim, golden);
    assert.strictEqual(afterRevocation.trusted, false);
    assert.match(afterRevocation.reason, /revoked/);
  }
  // 7. rollback and workload binding — the counter floor and the challenge context are both enforced.
  {
    const rollbackNode = new Node("node-e", serverKey, bootLog, 40);
    assert.strictEqual(orchestrator.verify(
      rollbackNode.quote(orchestrator.challenge(rollbackNode.name, workload), workload), rollbackNode, golden).trusted,
      true, "a high counter is accepted");
    const stale = rollbackNode.quote(orchestrator.challenge(rollbackNode.name, workload), workload, { counter: 39 });
    const verdict = orchestrator.verify(stale, rollbackNode, golden);
    assert.strictEqual(verdict.trusted, false);
    assert.match(verdict.reason, /rollback/);
    const other = new Node("node-f", serverKey, bootLog, 1);
    const wrongWorkload = other.quote(orchestrator.challenge(other.name, workload), "other-workload");
    const binding = orchestrator.verify(wrongWorkload, other, golden);
    assert.strictEqual(binding.trusted, false);
    assert.match(binding.reason, /binding mismatch/);
  }
  // 8. boundary — no challenge outstanding, an empty boot log, and an unknown node.
  {
    const stranger = new Node("node-g", serverKey, bootLog, 1);
    const noChallenge = orchestrator.verify(stranger.quote("made-up", workload), stranger, golden);
    assert.strictEqual(noChallenge.trusted, false);
    assert.match(noChallenge.reason, /no outstanding challenge/);
    const bareGolden = approvedGolden([]);
    assert.strictEqual(bareGolden.digest, ZERO_PCR);
    const bare = new Node("node-h", serverKey, [], 0);
    assert.strictEqual(orchestrator.verify(bare.quote(orchestrator.challenge(bare.name, workload), workload),
      bare, bareGolden).trusted, true, "an unmeasured platform matches an empty golden log");
    assert.throws(() => replayLog([{ pcr: "zero", component: "x" }]), /needs a PCR index/);
    assert.strictEqual(orchestrator.audit.filter((entry) => entry.verdict === "denied").length > 0, true);
  }
  // 9. determinism — challenges, quotes and verdicts replay from the same seed.
  {
    const run = () => {
      const local = new Orchestrator(new Prng(0x5eed0915));
      const localNode = new Node("node-a", serverKey, bootLog, 12);
      const results = [];
      for (let i = 0; i < 3; i++) {
        results.push(local.verify(localNode.quote(local.challenge("node-a", workload), workload), localNode, golden)
          .trusted);
        localNode.counter += 1;
      }
      return { results, audit: local.audit };
    };
    assert.deepStrictEqual(run(), run());
    assert.ok(orchestrator.spentNonces.size > 0, "spent nonces are remembered");
    assert.strictEqual(orchestrator.issued >= orchestrator.spentNonces.size, true, "every nonce was issued");
  }
  const trusted = orchestrator.audit.filter((entry) => entry.verdict === "trusted").length;
  const denied = orchestrator.audit.filter((entry) => entry.verdict === "denied").length;
  console.log(`attestation: golden PCR ${golden.digest.slice(0, 16)}... from a ${bootLog.length}-event log; ` +
    `${orchestrator.issued} challenges, ${trusted} trusted, ${denied} denied ` +
    `(replay, tampering, forgery, revocation, rollback and binding all caught)`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  sha256, hmac, ZERO_PCR, extend, replayLog, measurementOf, approvedGolden, Node, Orchestrator, Prng, };
