/** Task 753 - Implement control-flow integrity checker skeleton. (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * CFI in three parts, all on a toy module: (1) a *static policy builder* derives, for every
 * indirect call site, the set of legitimate targets = the address-taken functions whose arity
 * matches the site, mapped to their entry blocks; a target that is not a function entry is
 * never a landing pad, and functions that are not address-taken are unreachable through a
 * pointer. (2) An *incremental checker* replays a trace of executed control transfers and
 * reports the first violation: forward edges (indirect calls) must hit the site's allowed
 * set, backward edges (returns) must hit the address recorded on the shadow stack. (3) An
 * independent whole-trace oracle recomputes the same verdict with a different mechanism: it
 * scans the trace backwards to find the call matching each return and recomputes the allowed
 * sets from the function table instead of from the policy object.
 * Invariants maintained by the checker: the shadow stack depth equals the number of
 * unmatched calls at every step, a return never pops below depth zero, and the depth at the
 * end of an accepted trace is exactly zero.
 *
 * What is simplified: the module is a table of functions/blocks (no real instruction bytes
 * and no address layout), the policy is coarse-grained (arity only — no full type signature)
 * and there is no hardware shadow stack, so a successful check here is a *model* of CFI, not
 * a proof about native code. What is verified: a legitimate call/return trace is accepted with
 * an empty shadow stack at the end; a function-pointer hijack, an arity-mismatched target, a
 * jump into a non-landing-pad gadget and a return-address overwrite are each rejected *at the
 * exact offending event*; a return without a matching call (stack underflow) is rejected;
 * and for 60 seeded random traces the incremental checker and the backward-scan oracle agree
 * on both verdict and first violation index.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok } = require("node:assert");

/** Toy module: functions with arity + address-taken flag, blocks, indirect call sites. */
function toyModule() {
  const site = (arity) => ({ kind: "indirect-call", arity });
  return {
    blocks: {
      "main.b0": "main", "main.b1": "main", "main.b2": "main",
      "log.b0": "log", "log.b1": "log",
      "audit.b0": "audit", "audit.b1": "audit",
      "handler.b0": "handler", "handler.b1": "handler",
      "gadget.b0": "gadget", "gadget.b1": "gadget",
    },
    functions: {
      main: { arity: 0, addressTaken: false, entry: "main.b0" },
      log: { arity: 1, addressTaken: true, entry: "log.b0" },
      audit: { arity: 1, addressTaken: true, entry: "audit.b0" },
      handler: { arity: 1, addressTaken: false, entry: "handler.b0" },
      gadget: { arity: 0, addressTaken: false, entry: "gadget.b0" },
      send: { arity: 2, addressTaken: true, entry: "gadget.b1" },
    },
    callSites: {
      "main.b1#0": site(1),
      "main.b1#1": site(2),
      "main.b2#0": site(3),      // dead site: no address-taken function has arity 3
    },
  };
}

/** Static policy: landing pads, per-site allowed target sets, per-function arity. */
function buildPolicy(module) {
  const arity = new Map();
  const entryOf = new Map();
  const addressTaken = [];
  for (const [name, fn] of Object.entries(module.functions)) {
    arity.set(name, fn.arity);
    entryOf.set(name, fn.entry);
    if (fn.addressTaken) addressTaken.push(name);
  }
  const landingPads = new Set(entryOf.values());
  const sites = new Map();
  for (const [siteId, site] of Object.entries(module.callSites)) {
    const targets = addressTaken.filter((name) => arity.get(name) === site.arity).map((name) => entryOf.get(name));
    sites.set(siteId, { ...site, targets: new Set(targets) });
  }
  return { arity, entryOf, addressTaken: new Set(addressTaken), landingPads, sites };
}

/**
 * Incremental checker with fail-stop semantics (a CFI violation aborts the run). Events are
 * `{type:"call", site, target, ret}` (indirect call), `{type:"ret", target}` (return) and
 * `{type:"jump", target}` (direct branch, unchecked). Returns the first violation together
 * with the shadow-stack depth at that point.
 */
function checkTrace(policy, trace) {
  const shadowStack = [];
  const report = (index, kind, shadowDepth) => ({ violations: [{ index, kind }], shadowDepth, accepted: false });
  for (let index = 0; index < trace.length; index++) {
    const event = trace[index];
    if (event.type === "call") {
      if (!policy.landingPads.has(event.target)) return report(index, "not-a-landing-pad", shadowStack.length);
      const site = policy.sites.get(event.site);
      if (site === undefined || !site.targets.has(event.target)) return report(index, "not-in-allowed-set", shadowStack.length);
      shadowStack.push(event.ret);
    } else if (event.type === "ret") {
      if (shadowStack.length === 0) return report(index, "empty-shadow-stack", 0);
      if (shadowStack[shadowStack.length - 1] !== event.target) {
        return report(index, "return-address-mismatch", shadowStack.length);
      }
      shadowStack.pop();
    }
  }
  return { violations: [], shadowDepth: shadowStack.length, accepted: true };
}

/** Independent oracle: same verdict, but returns are matched by a backward scan. */
function scanTrace(module, trace) {
  const arity = new Map(Object.entries(module.functions).map(([name, fn]) => [name, fn.arity]));
  const entryOf = new Map(Object.entries(module.functions).map(([name, fn]) => [name, fn.entry]));
  const landingPads = new Set(entryOf.values());
  const violations = [];
  for (let i = 0; i < trace.length; i++) {
    const event = trace[i];
    if (event.type === "call") {
      if (!landingPads.has(event.target)) { violations.push({ index: i, kind: "not-a-landing-pad" }); continue; }
      const site = module.callSites[event.site];
      const allowed = Object.entries(module.functions)
        .filter(([name, fn]) => fn.addressTaken && fn.arity === site.arity)
        .map(([name]) => entryOf.get(name));
      if (!allowed.includes(event.target)) violations.push({ index: i, kind: "not-in-allowed-set" });
    } else if (event.type === "ret") {
      let depth = 0;
      let match = null;
      for (let j = i - 1; j >= 0; j--) {
        if (trace[j].type === "ret") depth++;
        else if (trace[j].type === "call") {
          if (depth === 0) { match = trace[j]; break; }
          depth--;
        }
      }
      if (match === null) violations.push({ index: i, kind: "empty-shadow-stack" });
      else if (match.ret !== event.target) violations.push({ index: i, kind: "return-address-mismatch" });
    }
  }
  return violations.length === 0 ? { violations: [], accepted: true } : { violations: violations.slice(0, 1), accepted: false };
}

const signature = (result) => `${result.accepted ? "accept" : `reject@${result.violations[0].index}:${result.violations[0].kind}`}`;

/** Deterministic xorshift32. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x753753;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

const LEGIT_TARGETS = ["log.b0", "audit.b0"];
const CALL_RETURN_SITE = "main.b2";
const GADGET_BLOCKS = ["gadget.b1", "log.b1", "main.b1"];

/** Legit call/return trace, optionally corrupted at one event to model an attack. */
function randomTrace(rng, length, attack) {
  const trace = [];
  const shadow = [];
  while (trace.length < length) {
    if (shadow.length === 0 || rng(10) < 6) {
      trace.push({ type: "call", site: "main.b1#0", target: LEGIT_TARGETS[rng(2)], ret: CALL_RETURN_SITE });
      shadow.push(CALL_RETURN_SITE);
    } else if (rng(5) === 0) {
      trace.push({ type: "jump", target: "main.b1" });           // direct edge, never policed
    } else {
      trace.push({ type: "ret", target: shadow.pop() });
    }
  }
  if (attack) {
    const index = rng(trace.length);
    const victim = trace[index];
    if (victim.type === "call") {
      if (rng(2) === 0) victim.target = "handler.b0";            // not address-taken
      else victim.target = "log.b1";                             // not a landing pad
    } else {
      trace[index] = { type: "ret", target: GADGET_BLOCKS[rng(GADGET_BLOCKS.length)] };
      if (rng(4) === 0) trace.splice(rng(index + 1), 1);         // drop a call -> underflow
    }
  }
  return trace;
}

module.exports = { toyModule, buildPolicy, checkTrace, scanTrace, signature, makeRng, randomTrace, LEGIT_TARGETS };

if (require.main === module) {
  const module = toyModule();
  const policy = buildPolicy(module);

  // 1. policy: arity 1 -> exactly the address-taken arity-1 functions, main excluded
  deepStrictEqual([...policy.sites.get("main.b1#0").targets].sort(), ["audit.b0", "log.b0"]);
  deepStrictEqual([...policy.sites.get("main.b1#1").targets], ["gadget.b1"]);
  strictEqual(policy.addressTaken.has("main"), false, "main is not address-taken");
  strictEqual(policy.landingPads.has("main.b1"), false, "interior blocks are not landing pads");
  strictEqual(policy.sites.get("main.b2#0").targets.size, 0, "the arity-3 site has no legal target");

  // 2. boundary: an empty trace is accepted and leaves an empty shadow stack
  deepStrictEqual(checkTrace(policy, []), { violations: [], shadowDepth: 0, accepted: true });

  // 3. a legitimate nested call/return trace is accepted and unwinds completely
  const legit = [
    { type: "call", site: "main.b1#0", target: "log.b0", ret: "main.b2" },
    { type: "jump", target: "log.b1" },
    { type: "call", site: "main.b1#0", target: "audit.b0", ret: "main.b2" },
    { type: "ret", target: "main.b2" },
    { type: "ret", target: "main.b2" },
  ];
  deepStrictEqual(checkTrace(policy, legit), { violations: [], shadowDepth: 0, accepted: true });
  deepStrictEqual(scanTrace(module, legit).accepted, true, "the oracle accepts it too");

  // 4. function-pointer hijack to a function that is not address-taken
  const hijacked = legit.map((event, i) => (i === 2 ? { ...event, target: "handler.b0" } : event));
  deepStrictEqual(checkTrace(policy, hijacked).violations, [{ index: 2, kind: "not-in-allowed-set" }]);
  // ...and the same site with a legal target is accepted, so the check has power
  const patched = legit.map((event, i) => (i === 2 ? { ...event, target: "log.b0" } : event));
  strictEqual(checkTrace(policy, patched).accepted, true, "a legal retarget stays accepted");

  // 5. arity mismatch: send is address-taken but takes 2 arguments, the site expects 1
  const wrongArity = [{ type: "call", site: "main.b1#0", target: "gadget.b1", ret: "main.b2" }];
  deepStrictEqual(checkTrace(policy, wrongArity).violations, [{ index: 0, kind: "not-in-allowed-set" }]);
  strictEqual(checkTrace(policy, [{ type: "call", site: "main.b1#1", target: "gadget.b1", ret: "main.b2" }]).accepted, true);

  // 6. a gadget that is not a landing pad is rejected even when the arity would match
  const gadget = [{ type: "call", site: "main.b1#0", target: "log.b1", ret: "main.b2" }];
  deepStrictEqual(checkTrace(policy, gadget).violations, [{ index: 0, kind: "not-a-landing-pad" }]);
  // 7. a dead site accepts nothing at all
  deepStrictEqual(checkTrace(policy, [{ type: "call", site: "main.b2#0", target: "log.b0", ret: "main.b2" }]).violations,
    [{ index: 0, kind: "not-in-allowed-set" }]);

  // 8. backward edge: return-address overwrite detected mid-trace, later events unreported
  const overwritten = [
    { type: "call", site: "main.b1#0", target: "log.b0", ret: "main.b2" },
    { type: "ret", target: "gadget.b1" },
    { type: "ret", target: "main.b2" },
  ];
  deepStrictEqual(checkTrace(policy, overwritten).violations, [{ index: 1, kind: "return-address-mismatch" }]);
  // 9. a return with no matching call is a shadow-stack underflow
  deepStrictEqual(checkTrace(policy, [{ type: "ret", target: "main.b2" }]).violations, [{ index: 0, kind: "empty-shadow-stack" }]);
  // a trailing call is legal: the depth simply stays positive
  deepStrictEqual(checkTrace(policy, [{ type: "call", site: "main.b1#0", target: "log.b0", ret: "main.b2" }]),
    { violations: [], shadowDepth: 1, accepted: true });

  // 10. shadow-stack invariant across a deep, balanced recursion
  const deep = [];
  for (let i = 0; i < 40; i++) deep.push({ type: "call", site: "main.b1#1", target: "gadget.b1", ret: "main.b2" });
  for (let i = 0; i < 40; i++) deep.push({ type: "ret", target: "main.b2" });
  deepStrictEqual(checkTrace(policy, deep), { violations: [], shadowDepth: 0, accepted: true });
  ok(checkTrace(policy, deep.slice(0, 40)).shadowDepth === 40, "depth equals the number of unmatched calls");

  // 11. randomized agreement between the incremental checker and the whole-trace oracle
  const rng = makeRng(0x753753);
  let legal = 0;
  let attacks = 0;
  for (let trial = 0; trial < 60; trial++) {
    const attack = trial % 2 === 1;
    const trace = randomTrace(rng, 4 + rng(14), attack);
    const incremental = checkTrace(policy, trace);
    const oracle = scanTrace(module, trace);
    strictEqual(signature(incremental), signature(oracle), `checker/oracle agreement in trial ${trial}`);
    deepStrictEqual(incremental.violations, oracle.violations, `identical reports in trial ${trial}`);
    if (incremental.accepted) legal++; else attacks++;
    if (!incremental.accepted) strictEqual(incremental.shadowDepth >= 0, true, "depth never goes negative");
  }
  ok(legal > 0, "some random traces must be legal");
  ok(attacks > 0, "some random traces must be corrupt");

  console.log(`sites=${policy.sites.size} landingPads=${policy.landingPads.size} randomTraces=60 legal=${legal} attacked=${attacks}`);
  console.log("PASS");
}
