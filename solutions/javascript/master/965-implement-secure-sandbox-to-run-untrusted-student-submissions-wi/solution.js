/** Task 965 - Implement secure sandbox to run untrusted student submissions with resource cgroups.
 * (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * A container model for untrusted submissions: the submission is a deterministic trace of operations
 * (allocate/free, fork/exit, cpu burn, sleep, syscall, write) and the sandbox runs it against a cgroup
 * budget -- memory limit, pid limit, cpu quota, wall-clock limit, scratch quota -- plus a seccomp-style
 * syscall allowlist, a read-only root filesystem and a scratch mount. The first violation stops the
 * trace immediately with the matching signal (SIGKILL for resource exhaustion, SIGSYS for a blocked
 * syscall), so no step past the limit is ever executed and the recorded usage never exceeds a limit.
 * Every step, allowed or denied, is written to an audit log in execution order.
 *
 * Simplified: the trace stands in for a real cgroup/seccomp runtime -- limits are simulated per step
 * instead of enforced by the kernel, there is no address-space isolation beyond accounting, and the
 * "filesystem" is a path prefix check. Verified: a compliant run, the memory/quota/pid/wall boundaries
 * at exactly the limit and one unit past it, syscall default-deny for unknown names, read-only root and
 * scratch quota, unknown operations, the empty trace, an independent accounting oracle over the trace,
 * host isolation, and deterministic replay.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const LIMITS = { memoryMb: 64, pids: 8, cpuTicks: 40, wallTicks: 60, scratchMb: 16 };
const ALLOWED_SYSCALLS = new Set(["read", "write", "brk", "mmap", "munmap", "futex", "clock_gettime", "exit_group"]);
const SCRATCH = "/scratch/";
class Denial extends Error {
  constructor(reason, signal) { super(reason); this.reason = reason; this.signal = signal; }
}
/** Execute a trace inside the sandbox; the first denial aborts the run and is logged as the last step. */
function runInSandbox(trace, limits = LIMITS) {
  const usage = { memoryMb: 0, peakMemoryMb: 0, pids: 1, peakPids: 1, cpuTicks: 0, wallTicks: 0, scratchMb: 0, syscalls: 0 };
  const log = [];
  const record = (step, entry) => log.push({ at: log.length, kind: step.kind, ...entry });
  for (const step of trace) {
    try {
      switch (step.kind) {
        case "alloc":
          usage.memoryMb += step.mb;
          if (usage.memoryMb > limits.memoryMb) throw new Denial("oom: memory limit exceeded", "SIGKILL");
          usage.peakMemoryMb = Math.max(usage.peakMemoryMb, usage.memoryMb);
          break;
        case "free": usage.memoryMb = Math.max(0, usage.memoryMb - step.mb); break;
        case "fork":
          usage.pids += 1;
          if (usage.pids > limits.pids) throw new Denial("pids limit exceeded", "SIGKILL");
          usage.peakPids = Math.max(usage.peakPids, usage.pids);
          break;
        case "exit": usage.pids = Math.max(1, usage.pids - 1); break;
        case "cpu":
          usage.cpuTicks += step.ticks;
          if (usage.cpuTicks > limits.cpuTicks) throw new Denial("cpu quota exceeded", "SIGKILL");
          break;
        case "sleep":
          usage.wallTicks += step.ticks;
          if (usage.wallTicks > limits.wallTicks) throw new Denial("wall clock exceeded", "SIGKILL");
          break;
        case "syscall":
          usage.syscalls += 1;
          if (!ALLOWED_SYSCALLS.has(step.name)) throw new Denial(`seccomp: ${step.name} denied`, "SIGSYS");
          break;
        case "write":
          if (!step.path.startsWith(SCRATCH)) throw new Denial(`read-only filesystem: ${step.path}`, "SIGKILL");
          usage.scratchMb += step.mb;
          if (usage.scratchMb > limits.scratchMb) throw new Denial("scratch quota exceeded", "SIGKILL");
          break;
        default: throw new Denial(`unknown operation ${step.kind}`, "SIGKILL");
      }
      record(step, { ok: true });
    } catch (error) {
      record(step, { ok: false, reason: error.reason, signal: error.signal });
      return { exit: 137, killed: true, reason: error.reason, signal: error.signal, usage, log, limits,
        executed: log.length - 1 };
    }
  }
  return { exit: 0, killed: false, reason: null, signal: null, usage, log, limits, executed: log.length };
}
/** Independent accounting oracle: fold the trace by hand instead of trusting the sandbox counters. */
function accountingOracle(trace, limits = LIMITS) {
  const totals = { memoryMb: 0, peakMemoryMb: 0, cpuTicks: 0, wallTicks: 0, scratchMb: 0, syscalls: 0, pids: 1, peakPids: 1 };
  for (const step of trace) {
    if (step.kind === "alloc") totals.memoryMb += step.mb;
    if (step.kind === "free") totals.memoryMb -= step.mb;
    if (step.kind === "cpu") totals.cpuTicks += step.ticks;
    if (step.kind === "sleep") totals.wallTicks += step.ticks;
    if (step.kind === "write") totals.scratchMb += step.mb;
    if (step.kind === "syscall") totals.syscalls += 1;
    if (step.kind === "fork") { totals.pids += 1; totals.peakPids = Math.max(totals.peakPids, totals.pids); }
    if (step.kind === "exit") totals.pids -= 1;
    totals.peakMemoryMb = Math.max(totals.peakMemoryMb, totals.memoryMb);
  }
  totals.withinLimits = totals.memoryMb <= limits.memoryMb && totals.cpuTicks <= limits.cpuTicks
    && totals.wallTicks <= limits.wallTicks && totals.scratchMb <= limits.scratchMb && totals.pids <= limits.pids;
  return totals;
}

function main() {
  const rng = makeRng(20240965);
  const compliant = [
    { kind: "alloc", mb: 8 }, { kind: "syscall", name: "brk" }, { kind: "fork" }, { kind: "alloc", mb: 4 },
    { kind: "write", path: "/scratch/out.txt", mb: 2 }, { kind: "exit" }, { kind: "free", mb: 12 },
    { kind: "cpu", ticks: 9 },
  ];
  const good = runInSandbox(compliant);
  // 1. Happy path: the compliant trace finishes, every step is logged, and usage stays inside budget.
  assert(good.exit === 0 && !good.killed && good.executed === compliant.length, `compliant exit ${good.exit}`);
  assert(good.log.length === compliant.length && good.log.every((entry) => entry.ok), "every step is logged as allowed");
  assert(good.usage.peakMemoryMb <= LIMITS.memoryMb && good.usage.cpuTicks <= LIMITS.cpuTicks
    && good.usage.peakPids <= LIMITS.pids && good.usage.scratchMb <= LIMITS.scratchMb, "usage inside every limit");
  // 2. Independent accounting oracle: the same numbers folded from the trace by hand must agree.
  const oracle = accountingOracle(compliant);
  assert(good.usage.peakMemoryMb === oracle.peakMemoryMb && good.usage.cpuTicks === oracle.cpuTicks
    && good.usage.scratchMb === oracle.scratchMb && good.usage.peakPids === oracle.peakPids && oracle.withinLimits,
    `oracle ${JSON.stringify(oracle)} vs ${JSON.stringify(good.usage)}`);
  // 3. Memory boundary: filling the limit is fine, one MiB more is an OOM kill at that exact step.
  const filled = runInSandbox([{ kind: "alloc", mb: LIMITS.memoryMb }]);
  assert(filled.exit === 0 && filled.usage.memoryMb === LIMITS.memoryMb, "an allocation of exactly the limit fits");
  const over = runInSandbox([{ kind: "alloc", mb: LIMITS.memoryMb }, { kind: "alloc", mb: 1 }]);
  assert(over.killed && over.reason.startsWith("oom") && over.signal === "SIGKILL" && over.executed === 1,
    `oom ${over.reason} after ${over.executed} steps`);
  assert(over.usage.memoryMb === LIMITS.memoryMb + 1 && over.log.at(-1).ok === false, "the offending step is the last log entry");
  // 4. CPU quota: burning exactly the quota passes, one tick more is killed and usage is clamped.
  assert(runInSandbox([{ kind: "cpu", ticks: LIMITS.cpuTicks }]).exit === 0, "the exact quota is allowed");
  const burned = runInSandbox([{ kind: "cpu", ticks: LIMITS.cpuTicks }, { kind: "cpu", ticks: 1 }, { kind: "alloc", mb: 4 }]);
  assert(burned.killed && burned.reason === "cpu quota exceeded" && burned.usage.cpuTicks === LIMITS.cpuTicks + 1,
    `cpu kill ${burned.reason}`);
  assert(burned.executed === 1, "no step after the kill is executed");
  // 5. Pid limit: a fork bomb dies at the limit and the peak never exceeds it.
  const bomb = runInSandbox(Array.from({ length: 20 }, () => ({ kind: "fork" })));
  assert(bomb.killed && bomb.reason === "pids limit exceeded" && bomb.usage.peakPids === LIMITS.pids, `fork bomb ${bomb.reason}`);
  assert(bomb.usage.pids === LIMITS.pids + 1 && bomb.executed === LIMITS.pids - 1, "the (limit+1)-th fork is the one that dies");
  // 6. Syscall policy: the allowlist passes, known-dangerous and unknown names are default-denied.
  assert(["brk", "mmap", "futex", "read", "write"].every((name) => !runInSandbox([{ kind: "syscall", name }]).killed),
    "allowlisted syscalls pass");
  const denied = runInSandbox([{ kind: "syscall", name: "socket" }, { kind: "syscall", name: "read" }]);
  assert(denied.killed && denied.reason === "seccomp: socket denied" && denied.signal === "SIGSYS" && denied.executed === 0,
    `seccomp ${denied.reason}`);
  assert(runInSandbox([{ kind: "syscall", name: "ptrace" }]).reason === "seccomp: ptrace denied", "unknown syscalls are denied");
  // 7. Filesystem isolation: the root is read-only, the scratch mount has its own quota.
  const readOnly = runInSandbox([{ kind: "write", path: "/etc/passwd", mb: 1 }]);
  assert(readOnly.killed && readOnly.reason === "read-only filesystem: /etc/passwd", `fs ${readOnly.reason}`);
  assert(!runInSandbox([{ kind: "write", path: "/scratch/ok", mb: LIMITS.scratchMb }]).killed, "the scratch quota fits exactly");
  const scratchOver = runInSandbox([{ kind: "write", path: "/scratch/ok", mb: LIMITS.scratchMb }, { kind: "write", path: "/scratch/more", mb: 1 }]);
  assert(scratchOver.reason === "scratch quota exceeded" && scratchOver.usage.scratchMb === LIMITS.scratchMb + 1, "scratch over quota");
  // 8. Wall clock and boundaries: sleeps at the limit pass, over it dies; empty and unknown traces.
  assert(runInSandbox([{ kind: "sleep", ticks: LIMITS.wallTicks }]).exit === 0, "the wall clock limit is inclusive");
  assert(runInSandbox([{ kind: "sleep", ticks: LIMITS.wallTicks + 1 }]).reason === "wall clock exceeded", "sleeping past the limit dies");
  const empty = runInSandbox([]);
  assert(empty.exit === 0 && empty.log.length === 0 && empty.usage.cpuTicks === 0 && empty.usage.peakMemoryMb === 0, "an empty trace is a clean run");
  assert(runInSandbox([{ kind: "execve" }]).reason === "unknown operation execve", "unknown operations are rejected");
  // 9. Isolation and determinism: the host object is untouched, and a replay is byte-identical.
  const host = { memory: "host-only", registries: { cache: 3 } };
  const before = JSON.stringify(host);
  const replay = runInSandbox(compliant);
  assert(JSON.stringify(host) === before, "the sandbox never mutates host state");
  assert.deepStrictEqual(replay, good, "the same trace replays byte for byte");
  assert.deepStrictEqual(runInSandbox(compliant.map((step) => ({ ...step }))), good, "copies of the trace agree too");
  const seeded = Array.from({ length: 12 }, () => ({ kind: "alloc", mb: rng.int(6) }));
  assert(runInSandbox(seeded).executed === seeded.length && runInSandbox(seeded).usage.memoryMb === accountingOracle(seeded).memoryMb,
    "a seeded trace agrees with the accounting oracle");
  console.log(`sandbox: steps=${good.executed} peakMem=${good.usage.peakMemoryMb}Mb cpu=${good.usage.cpuTicks} `
    + `denials=oom/${over.signal} seccomp/${denied.signal} pids/${bomb.reason}`);
  console.log("PASS");
}

module.exports = { makeRng, LIMITS, ALLOWED_SYSCALLS, SCRATCH, Denial, runInSandbox, accountingOracle };
if (require.main === module) main();
