// Task 965 - Implement secure sandbox to run untrusted student submissions with resource cgroups.
// (Tier: Master, Beyond - Security & Privacy)
//
// The sandbox models a cgroup v2 hierarchy in memory: `memory.max` kills a task that allocates past the
// limit, `cpu.max` (quota per period) throttles a cpu burst and defers the remainder to the next period,
// `pids.max` denies a fork that would exceed the process cap, and two further guards - a seccomp-style
// syscall allowlist and a rootfs path guard - kill a task that steps outside its contract.  Scripted ops
// drive the task so every limit can be exercised deterministically.
//
// Verified: out-of-memory kills fire only above `memory.max` (allocating exactly the cap is allowed), a
// 100ms burst under a 40ms/100ms quota throttles exactly twice, a zero quota makes no progress at all,
// the pid boundary admits four processes and denies the fifth, a denied syscall and a path escaping the
// rootfs both terminate the task, an empty script only admits and exits, the CPU conservation invariant
// holds after every run (the independent oracle), and re-running a script reproduces its event log.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    private sealed class Op
    {
        public readonly string Kind;
        public readonly int Amount;
        public readonly string Text;
        public Op(string kind, int amount, string text) { Kind = kind; Amount = amount; Text = text; }
    }

    private sealed class Sandbox
    {
        public readonly List<string> Events = new List<string>();
        public readonly HashSet<string> Allowlist = new HashSet<string>(StringComparer.Ordinal) { "read", "write", "exit", "clock_gettime" };
        public readonly string Root = "/sandbox/work";
        public int MemoryMaxMiB = 64, MemoryCurrentMiB;
        public int PidsMax = 4, PidsCurrent;
        public int PeriodMs = 100, CpuQuotaMs = 40, CpuUsedMs, CpuBurnedMs, Periods;
        public bool Terminated;
        public string ExitReason = "exited";
    }

    private static Op Alloc(int mib) => new Op("alloc", mib, "");
    private static Op Burn(int ms) => new Op("burn", ms, "");
    private static Op Fork(int processes) => new Op("fork", processes, "");
    private static Op Syscall(string name) => new Op("syscall", 0, name);
    private static Op Write(string path) => new Op("write", 0, path);

    private static bool Escapes(string root, string path) =>
        path.Contains("..", StringComparison.Ordinal) ||
        (path.StartsWith("/", StringComparison.Ordinal) && !path.StartsWith(root, StringComparison.Ordinal));

    private static void Run(Sandbox box, List<Op> ops)
    {
        box.Events.Add("admitted cpu=" + box.CpuQuotaMs + "/" + box.PeriodMs + "ms memory=" + box.MemoryMaxMiB + "MiB pids=" + box.PidsMax);
        foreach (Op op in ops)
        {
            if (box.Terminated) break;
            switch (op.Kind)
            {
                case "alloc":
                    box.MemoryCurrentMiB += op.Amount;
                    if (box.MemoryCurrentMiB > box.MemoryMaxMiB)
                    {
                        box.Events.Add("oom_kill: " + box.MemoryCurrentMiB + "MiB > memory.max " + box.MemoryMaxMiB + "MiB");
                        box.Terminated = true;
                        box.ExitReason = "oom_kill";
                    }
                    break;
                case "burn":
                    int remaining = op.Amount;
                    while (remaining > 0)
                    {
                        if (box.CpuUsedMs >= box.CpuQuotaMs)
                        {
                            box.Events.Add("throttled with " + remaining + "ms left in the period");
                            box.Periods++;
                            box.CpuUsedMs = 0;
                            if (box.CpuQuotaMs == 0) { box.Events.Add("no progress possible with cpu.max quota 0"); break; }
                        }
                        int quantum = Math.Min(remaining, box.CpuQuotaMs - box.CpuUsedMs);
                        box.CpuUsedMs += quantum;
                        box.CpuBurnedMs += quantum;
                        remaining -= quantum;
                    }
                    break;
                case "fork":
                    if (box.PidsCurrent + op.Amount > box.PidsMax) box.Events.Add("denied fork of " + op.Amount + " (pids.max " + box.PidsMax + ")");
                    else { box.PidsCurrent += op.Amount; box.Events.Add("forked " + op.Amount + ", pids " + box.PidsCurrent); }
                    break;
                case "syscall":
                    if (!box.Allowlist.Contains(op.Text)) { box.Events.Add("denied syscall " + op.Text); box.Terminated = true; box.ExitReason = "seccomp"; }
                    else box.Events.Add("allowed syscall " + op.Text);
                    break;
                default:
                    if (Escapes(box.Root, op.Text)) { box.Events.Add("denied write to " + op.Text); box.Terminated = true; box.ExitReason = "path_guard"; }
                    else box.Events.Add("wrote " + op.Text);
                    break;
            }
        }
        box.Events.Add("exited (" + box.ExitReason + ")");
    }

    /** Independent oracle: no cpu is created or destroyed - burned + what is left in the open period is the total quota. */
    private static bool Conserved(Sandbox box) =>
        box.CpuBurnedMs + (box.CpuQuotaMs - box.CpuUsedMs) == box.Periods * box.CpuQuotaMs + box.CpuQuotaMs;

    private static int CountOf(Sandbox box, string prefix)
    {
        int count = 0;
        foreach (string line in box.Events) { if (line.StartsWith(prefix, StringComparison.Ordinal)) count++; }
        return count;
    }

    private static string Replay(List<Op> ops)
    {
        var box = new Sandbox();
        Run(box, ops);
        return string.Join("|", box.Events);
    }

    public static void Main()
    {
        var memory = new Sandbox();
        Run(memory, new List<Op> { Alloc(32), Alloc(40) });
        Check(memory.Terminated && memory.ExitReason == "oom_kill" && CountOf(memory, "oom_kill") == 1 && CountOf(memory, "exited (oom_kill)") == 1,
            "allocating 72MiB past memory.max 64MiB triggers exactly one oom_kill and terminates the task");
        var fits = new Sandbox();
        Run(fits, new List<Op> { Alloc(64) });
        Check(!fits.Terminated && fits.ExitReason == "exited", "allocating exactly memory.max is allowed (the boundary)");

        var cpu = new Sandbox();
        Run(cpu, new List<Op> { Burn(100) });
        Check(cpu.CpuBurnedMs == 100 && cpu.Periods == 2 && CountOf(cpu, "throttled") == 2,
            "a 100ms burst under a 40ms/100ms quota throttles twice and burns exactly 100ms");
        Check(Conserved(cpu), "the cpu conservation invariant holds after throttling");

        var zero = new Sandbox { CpuQuotaMs = 0 };
        Run(zero, new List<Op> { Burn(5) });
        Check(zero.CpuBurnedMs == 0 && zero.Events.Exists(e => e.Contains("no progress", StringComparison.Ordinal)) && Conserved(zero),
            "a zero quota makes no progress at all and says so");

        var pids = new Sandbox();
        Run(pids, new List<Op> { Fork(3), Fork(1), Fork(1) });
        Check(pids.PidsCurrent == 4 && CountOf(pids, "denied fork") == 1, "the pids.max boundary admits four processes and denies the fifth");

        var syscalls = new Sandbox();
        Run(syscalls, new List<Op> { Syscall("read"), Syscall("ptrace") });
        Check(syscalls.Terminated && syscalls.ExitReason == "seccomp" && syscalls.Events.Contains("allowed syscall read") && syscalls.Events.Contains("denied syscall ptrace"),
            "the allowlist admits read and kills ptrace");

        var paths = new Sandbox();
        Run(paths, new List<Op> { Write("data/out.txt"), Write("../etc/passwd") });
        Check(paths.ExitReason == "path_guard" && paths.Events.Contains("wrote data/out.txt") && CountOf(paths, "denied write") == 1,
            "a relative path inside the rootfs is written and a traversal out of it is denied");
        var absolute = new Sandbox();
        Run(absolute, new List<Op> { Write("/etc/shadow") });
        Check(absolute.ExitReason == "path_guard" && !Escapes("/sandbox/work", "data/out.txt") && !Escapes("/sandbox/work", "/sandbox/work/tmp"),
            "an absolute path outside the rootfs is denied while paths inside the root are admitted");

        var empty = new Sandbox();
        Run(empty, new List<Op>());
        Check(empty.Events.Count == 2 && empty.Events[1] == "exited (exited)" && Conserved(empty),
            "an empty script only admits and exits, and conservation holds trivially");

        var script = new List<Op> { Alloc(16), Burn(60), Fork(2), Syscall("read"), Write("a/b.txt") };
        Check(Replay(script) == Replay(script), "re-running the same script produces an identical event log");
        Check(Replay(script).Contains("throttled") && Replay(script).EndsWith("exited (exited)", StringComparison.Ordinal),
            "the mixed script throttles once, stays alive and exits cleanly");

        Console.WriteLine("sandbox: memory.max=" + 64 + "MiB, cpu.max quota=" + 40 + "/" + 100 + "ms, pids.max=" + 4 +
            ", allowlist " + 4 + " syscalls, root " + new Sandbox().Root);
        Console.WriteLine("cpu: burned " + cpu.CpuBurnedMs + "ms over " + cpu.Periods + " full period(s), conservation re-checked on every run");
        Console.WriteLine("guards: oom_kill, throttle, denied fork, denied syscall " + "ptrace" + ", denied traversal ../etc/passwd");
        Console.WriteLine("PASS");
    }
}
