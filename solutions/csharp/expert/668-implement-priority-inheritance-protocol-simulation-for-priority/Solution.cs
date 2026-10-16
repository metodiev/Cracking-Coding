// Task 668 - Implement priority inheritance protocol simulation for priority inversion.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A tick-accurate uniprocessor fixed-priority scheduler with mutexes, simulated twice: once with the
// plain (no-protocol) semaphore semantics of a classic real-time kernel and once with the Priority
// Inheritance Protocol (PIP). Tasks are released at a start tick and hold a program of three op kinds
// (Acquire / Release / Compute(k), where Compute(k) burns exactly k ticks); every tick the scheduler
// picks the released, unfinished task with the largest *effective* priority, ties broken by task id.
// Effective priority is not stored per task but recomputed from scratch each tick as the least fixpoint of
//     eff(t) = max(base(t), max{ eff(b) : b blocked on a mutex currently held by t })
// which is the exact semantics of PIP (a waiter's priority is donated, transitively, to the holder of the
// lock it waits for) and is also the quantity the invariant checks below police. Nothing here is a
// wall-clock simulation: ticks are logical, the whole run is deterministic, and programs come from every
// task's own op list.
//
// What the file demonstrates:
//   * the inversion itself — L (low) holds the lock, M (medium, lock-free) preempts L, H (high) blocks on
//     the lock and is delayed for the whole of M's work;
//   * the fix — with PIP the holder's effective priority jumps to H's, so M can no longer preempt it and
//     H's blocking collapses to the holder's remaining critical-section work plus the single tick in which
//     the failed attempt was made; the remaining work is computed independently from the task spec (sum of
//     the ticks of L's ops from its program counter onward), not read back from the simulation;
//   * nested donation — a two-level chain (H waits for M2, M2 waits for L) must boost both holders, and
//     a released holder must drop straight back to its base priority;
//   * invariants, evaluated after *every* tick of *every* run: mutual exclusion (a mutex has at most one
//     holder, and a holder is never blocked on its own mutex), no waiter without a holder (a release
//     broadcasts to its waiters, which then race to re-acquire; the highest-priority one wins because the
//     scheduler picks it first, so the broadcast is a faithful stand-in for a priority wait queue),
//     eff(t) >= base(t), and eff(holder) >= eff(waiter) for every blocked waiter; plus termination within
//     the tick budget (no deadlock).
// The randomised cross-check generates 60 lock-disciplined workloads (each task has one non-nested
// critical section on one of two mutexes, so deadlock is structurally impossible), runs them both ways
// and asserts that the invariants hold throughout and that PIP never lengthens the blocking of the
// highest-priority task.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64: deterministic, well-mixed, no framework dependency.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal enum OpKind { Acquire, Release, Compute }

internal sealed class Op
{
    internal OpKind Kind;
    internal int Mutex = -1;
    internal int Ticks;

    internal static Op Acquire(int mutex) => new Op { Kind = OpKind.Acquire, Mutex = mutex };
    internal static Op Release(int mutex) => new Op { Kind = OpKind.Release, Mutex = mutex };
    internal static Op Compute(int ticks) => new Op { Kind = OpKind.Compute, Ticks = ticks };
}

internal sealed class TaskSpec
{
    internal string Name = "";
    internal int Priority;
    internal int Start;
    internal List<Op> Ops = new List<Op>();
}

internal sealed class Sim
{
    internal readonly List<TaskSpec> Tasks;
    internal readonly int MutexCount;
    internal readonly bool Inheritance;

    private readonly int[] pc;
    private readonly int[] remain;
    private readonly int[] eff;
    private readonly int[] heldBy;
    private readonly int[] blockedOn;
    private readonly int[] lastBlockStart;

    /// Holder's remaining programme length (ticks) at the moment the task blocked — captured from the
    /// task spec, so it is an independent yardstick for the PIP blocking bound.
    internal readonly int[] RemainingWorkAtBlock;

    internal readonly int[] Complete;
    internal readonly int[] BlockedTicks;
    internal readonly int[] BlockedEpisodes;
    internal readonly int[] MaxEpisodeLength;
    internal readonly int[] EpisodeBound;

    /// Episodes that outlasted the holder's remaining critical-section work (plus the tick in which the
    /// failed attempt was made) — counted only for waiters that hold the highest base priority, where the
    /// bound is exactly what the protocol guarantees. Under PIP this must stay 0; without inheritance it
    /// is exactly the inversion the file is about.
    internal int BoundViolations;
    internal int EpisodesCompleted;
    internal int Ticks;
    internal bool Deadlocked;
    internal int Violations;
    internal string FirstViolation = "";
    internal int MaxWaitersSeen;

    internal Sim(List<TaskSpec> tasks, int mutexCount, bool inheritance)
    {
        Tasks = tasks;
        MutexCount = mutexCount;
        Inheritance = inheritance;
        int n = tasks.Count;
        pc = new int[n];
        remain = new int[n];
        eff = new int[n];
        heldBy = new int[mutexCount];
        blockedOn = new int[n];
        lastBlockStart = new int[n];
        RemainingWorkAtBlock = new int[n];
        Complete = new int[n];
        BlockedTicks = new int[n];
        BlockedEpisodes = new int[n];
        MaxEpisodeLength = new int[n];
        EpisodeBound = new int[n];
        for (int m = 0; m < mutexCount; m++) heldBy[m] = -1;
        for (int i = 0; i < n; i++)
        {
            blockedOn[i] = -1;
            Complete[i] = -1;
            remain[i] = Tasks[i].Ops.Count > 0 && Tasks[i].Ops[0].Kind == OpKind.Compute ? Tasks[i].Ops[0].Ticks : 0;
            eff[i] = Tasks[i].Priority;
        }
    }

    private bool Finished(int i) => pc[i] >= Tasks[i].Ops.Count;

    /// Least fixpoint of eff(t) = max(base(t), eff of everything blocked on a lock t holds).
    private void ComputeEffectivePriorities()
    {
        for (int i = 0; i < Tasks.Count; i++) eff[i] = Tasks[i].Priority;
        if (!Inheritance) return;
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int m = 0; m < MutexCount; m++)
            {
                int holder = heldBy[m];
                if (holder < 0) continue;
                for (int b = 0; b < Tasks.Count; b++)
                {
                    if (blockedOn[b] != m) continue;
                    if (eff[b] > eff[holder])
                    {
                        eff[holder] = eff[b];
                        changed = true;
                    }
                }
            }
        }
    }

    /// Ticks of work still programmed for a task, counted from its program counter — used as the
    /// independent yardstick for how long a waiter may legitimately be delayed under inheritance.
    private int RemainingWork(int i)
    {
        int total = 0;
        for (int k = pc[i]; k < Tasks[i].Ops.Count; k++)
        {
            Op op = Tasks[i].Ops[k];
            if (op.Kind == OpKind.Compute) total += (k == pc[i] ? Math.Max(1, remain[i]) : op.Ticks);
            else total += 1;
        }
        return total;
    }

    /// Effective priority as the scheduler would see it right now (donation included).
    internal int EffectivePriority(int task)
    {
        ComputeEffectivePriorities();
        return eff[task];
    }

    /// Run stops as soon as this many ticks have elapsed (-1 disables it), which lets a check inspect
    /// the mid-flight donation state instead of only the end state.
    internal int StopAt = -1;

    internal bool IsBlocked(int task) => blockedOn[task] >= 0;

    internal int HolderOf(int mutex) => heldBy[mutex];

    /// How long the waiter of `mutex` may legitimately be delayed: the holder's remaining work.
    internal int RemainingWorkOfHolder(int mutex) => heldBy[mutex] < 0 ? 0 : RemainingWork(heldBy[mutex]);

    internal void Run(int maxTicks)
    {
        int n = Tasks.Count;
        while (true)
        {
            for (int i = 0; i < n; i++)
            {
                if (!Finished(i) && blockedOn[i] >= 0) BlockedTicks[i]++;
            }
            ComputeEffectivePriorities();
            if (Ticks >= maxTicks)
            {
                Deadlocked = true;
                return;
            }
            int ready = -1;
            for (int i = 0; i < n; i++)
            {
                if (Finished(i) || blockedOn[i] >= 0 || Ticks < Tasks[i].Start) continue;
                if (ready < 0 || eff[i] > eff[ready] || (eff[i] == eff[ready] && i < ready)) ready = i;
            }
            if (ready < 0)
            {
                bool anyLeft = false;
                bool runnableLater = false;
                for (int i = 0; i < n; i++)
                {
                    if (Finished(i)) continue;
                    anyLeft = true;
                    if (blockedOn[i] < 0) runnableLater = true;   // merely not released yet
                }
                if (!anyLeft) return;
                if (!runnableLater)
                {
                    Deadlocked = true;      // every unfinished task waits for a lock: circular wait
                    return;
                }
                Ticks++;                    // idle tick: the CPU has nothing to run yet
                continue;
            }
            Ticks++;
            Step(ready);
            Validate();
            if (StopAt >= 0 && Ticks >= StopAt) return;
            bool allDone = true;
            for (int i = 0; i < n; i++) if (!Finished(i)) allDone = false;
            if (allDone) return;
        }
    }

    private void Step(int i)
    {
        Op op = Tasks[i].Ops[pc[i]];
        switch (op.Kind)
        {
            case OpKind.Compute:
                remain[i]--;
                if (remain[i] <= 0)
                {
                    Advance(i);
                    if (pc[i] < Tasks[i].Ops.Count && Tasks[i].Ops[pc[i]].Kind == OpKind.Compute)
                        remain[i] = Tasks[i].Ops[pc[i]].Ticks;
                }
                break;
            case OpKind.Acquire:
                if (heldBy[op.Mutex] < 0)
                {
                    heldBy[op.Mutex] = i;
                    Advance(i);
                }
                else
                {
                    if (blockedOn[i] < 0)
                    {
                        blockedOn[i] = op.Mutex;
                        lastBlockStart[i] = Ticks;
                        RemainingWorkAtBlock[i] = RemainingWork(heldBy[op.Mutex]);
                        BlockedEpisodes[i]++;
                    }
                }
                break;
            default:
                heldBy[op.Mutex] = -1;
                WakeWaiters(op.Mutex);
                Advance(i);
                break;
        }
    }

    /// A release ends every waiter's blocking episode; the length is compared against the holder's
    /// remaining work captured when the episode began (the yardstick read off the task spec).
    private void WakeWaiters(int mutex)
    {
        int topPriority = 0;
        for (int i = 0; i < Tasks.Count; i++) if (Tasks[i].Priority > topPriority) topPriority = Tasks[i].Priority;
        for (int w = 0; w < Tasks.Count; w++)
        {
            if (blockedOn[w] != mutex) continue;
            int length = Ticks - lastBlockStart[w] + 1;
            EpisodesCompleted++;
            if (length > MaxEpisodeLength[w]) MaxEpisodeLength[w] = length;
            if (RemainingWorkAtBlock[w] + 1 > EpisodeBound[w]) EpisodeBound[w] = RemainingWorkAtBlock[w] + 1;
            if (Tasks[w].Priority == topPriority && length > RemainingWorkAtBlock[w] + 1) BoundViolations++;
            blockedOn[w] = -1;
        }
    }

    private void Advance(int i)
    {
        pc[i]++;
        if (pc[i] >= Tasks[i].Ops.Count) Complete[i] = Ticks;
        else remain[i] = Tasks[i].Ops[pc[i]].Kind == OpKind.Compute ? Tasks[i].Ops[pc[i]].Ticks : 0;
    }

    private void Validate()
    {
        int n = Tasks.Count;
        ComputeEffectivePriorities();
        for (int m = 0; m < MutexCount; m++)
        {
            int holder = heldBy[m];
            if (holder >= 0 && blockedOn[holder] == m) Fail("holder is blocked on its own mutex");
            int waiters = 0;
            for (int i = 0; i < n; i++)
            {
                if (blockedOn[i] != m) continue;
                waiters++;
                if (i == holder) Fail("the holder is queued on its own mutex");
            }
            if (holder < 0 && waiters > 0) Fail("waiter on an unheld mutex");
            if (waiters > MaxWaitersSeen) MaxWaitersSeen = waiters;
        }
        for (int i = 0; i < n; i++)
        {
            if (Finished(i)) continue;
            if (blockedOn[i] < 0) continue;
            if (eff[i] < Tasks[i].Priority) Fail("effective priority below base priority");
            if (!Inheritance) continue;
            if (eff[heldBy[blockedOn[i]]] < eff[i]) Fail("holder not boosted to its waiter's priority");
        }
    }

    private void Fail(string message)
    {
        Violations++;
        if (FirstViolation.Length == 0) FirstViolation = "tick " + Ticks + ": " + message;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static TaskSpec Make(string name, int priority, int start, params Op[] ops)
    {
        var spec = new TaskSpec { Name = name, Priority = priority, Start = start };
        spec.Ops.AddRange(ops);
        return spec;
    }

    /// L holds the lock, M is lock-free and priority-between, H is the highest and arrives while M runs.
    private static List<TaskSpec> ClassicScenario()
    {
        var tasks = new List<TaskSpec>();
        tasks.Add(Make("L", 1, 0, Op.Acquire(0), Op.Compute(12), Op.Release(0)));
        tasks.Add(Make("M", 2, 1, Op.Compute(20)));
        tasks.Add(Make("H", 3, 3, Op.Acquire(0), Op.Compute(2), Op.Release(0)));
        return tasks;
    }

    /// H -> m2 (held by M2), M2 -> m1 (held by L): one donation chain of length two.
    private static List<TaskSpec> NestedScenario()
    {
        var tasks = new List<TaskSpec>();
        tasks.Add(Make("L", 1, 0, Op.Acquire(0), Op.Compute(6), Op.Release(0)));
        tasks.Add(Make("M2", 2, 1, Op.Acquire(1), Op.Compute(2), Op.Acquire(0), Op.Compute(3), Op.Release(0), Op.Release(1)));
        tasks.Add(Make("H", 4, 2, Op.Acquire(1), Op.Compute(1), Op.Release(1)));
        return tasks;
    }

    private static List<TaskSpec> RandomScenario(ref ulong state, int taskCount)
    {
        var tasks = new List<TaskSpec>();
        var priorities = new int[taskCount];
        for (int i = 0; i < taskCount; i++) priorities[i] = i + 1;
        for (int i = taskCount - 1; i > 0; i--)
        {
            int j = Rng.NextBelow(ref state, i + 1);
            int tmp = priorities[i]; priorities[i] = priorities[j]; priorities[j] = tmp;
        }
        for (int i = 0; i < taskCount; i++)
        {
            int mutex = 0;   // one shared mutex: maximum contention, and non-nested sections make it safe
            var ops = new List<Op>();
            int lead = Rng.NextBelow(ref state, 4);
            if (lead > 0) ops.Add(Op.Compute(lead));
            ops.Add(Op.Acquire(mutex));
            ops.Add(Op.Compute(1 + Rng.NextBelow(ref state, 9)));
            ops.Add(Op.Release(mutex));
            // The highest-priority task is released late, so it usually arrives while a lower-priority
            // task is inside its critical section — exactly the situation the protocol is about.
            bool isTop = priorities[i] == taskCount;
            var spec = new TaskSpec
            {
                Name = "T" + i,
                Priority = priorities[i],
                Start = isTop ? 3 + Rng.NextBelow(ref state, 2) : Rng.NextBelow(ref state, 3)
            };
            spec.Ops.AddRange(ops);
            tasks.Add(spec);
        }
        return tasks;
    }

    public static void Main()
    {
        // ---- the inversion, and its removal --------------------------------------------------
        Sim plain = new Sim(ClassicScenario(), 1, false);
        plain.Run(5000);
        Sim pip = new Sim(ClassicScenario(), 1, true);
        pip.Run(5000);

        Check(!plain.Deadlocked && !pip.Deadlocked, "both classic runs terminate (no deadlock)");
        Check(plain.Violations == 0, "semaphore invariants hold every tick without inheritance: " + plain.FirstViolation);
        Check(pip.Violations == 0, "semaphore and inheritance invariants hold every tick with PIP: " + pip.FirstViolation);

        int high = 2;  // task index of H
        Check(plain.BlockedEpisodes[high] == 1, "H blocks exactly once on the held mutex in the plain run");
        Check(plain.BlockedTicks[high] >= 30,
              "the plain run shows real priority inversion: H waits " + plain.BlockedTicks[high] + " ticks for a 12-tick critical section");
        Check(pip.BlockedTicks[high] < plain.BlockedTicks[high] / 2,
              "PIP cuts H's blocking from " + plain.BlockedTicks[high] + " to " + pip.BlockedTicks[high] + " ticks");
        Check(pip.Complete[high] < plain.Complete[high],
              "H completes earlier under PIP (" + pip.Complete[high] + " vs " + plain.Complete[high] + ")");
        Check(pip.Complete[0] < plain.Complete[0],
              "the boosted holder L also finishes earlier (" + pip.Complete[0] + " vs " + plain.Complete[0] + ")");

        // Independent yardstick: the donor's remaining critical-section work, read from the task spec.
        int holderWorkAtBlock = pip.RemainingWorkAtBlock[high];
        Check(holderWorkAtBlock == 13, "the holder had exactly 12 compute ticks plus one release op left (got "
              + holderWorkAtBlock + ")");
        Check(pip.MaxEpisodeLength[high] <= holderWorkAtBlock + 1,
              "with PIP the wait never exceeds the holder's remaining critical section plus the failing tick ("
              + pip.MaxEpisodeLength[high] + " <= " + (holderWorkAtBlock + 1) + ")");
        Check(pip.MaxEpisodeLength[high] > 0, "the bound is not achieved vacuously (H really was blocked)");
        Check(plain.BlockedTicks[high] > holderWorkAtBlock + 1,
              "without inheritance the wait blows past that bound by the whole of M's work");
        Check(plain.BoundViolations == 1, "the bound is violated exactly once in the plain run");
        Check(pip.BoundViolations == 0, "the bound is never violated under inheritance");
        Check(pip.EpisodesCompleted == 1 && plain.EpisodesCompleted == 1, "both runs contain one completed blocking episode");

        // ---- nested (transitive) donation ----------------------------------------------------
        Sim nested = new Sim(NestedScenario(), 2, true);
        nested.Run(5000);
        Check(!nested.Deadlocked, "the nested-donation scenario terminates");
        Check(nested.Violations == 0, "invariants hold in the nested scenario: " + nested.FirstViolation);
        Sim nestedPlain = new Sim(NestedScenario(), 2, false);
        nestedPlain.Run(5000);
        Check(!nestedPlain.Deadlocked && nestedPlain.Violations == 0, "the same scenario runs plain as well");
        Check(nested.BlockedTicks[2] <= nestedPlain.BlockedTicks[2],
              "transitive donation never lengthens H's blocking (" + nested.BlockedTicks[2] + " vs "
              + nestedPlain.BlockedTicks[2] + ")");

        // Donation is transitive: sample the nested run at the moment H is blocked and check that the
        // waiter -> holder -> holder chain has pushed H's priority all the way down to L.
        Sim mid = new Sim(NestedScenario(), 2, true) { StopAt = 8 };
        mid.Run(5000);
        Check(mid.IsBlocked(2), "at the sample tick H is blocked on M2's mutex");
        Check(mid.HolderOf(1) == 1, "M2 holds the mutex H waits for");
        Check(mid.HolderOf(0) == 0, "L holds the mutex M2 waits for (M2 must already be blocked)");
        Check(mid.IsBlocked(1), "M2 is itself blocked, so the chain has two links");
        Check(mid.EffectivePriority(1) == 4 && mid.EffectivePriority(0) == 4,
              "both links of the chain are boosted to H's priority (got "
              + mid.EffectivePriority(1) + " and " + mid.EffectivePriority(0) + ")");
        Check(mid.EffectivePriority(2) == 4, "H itself stays at its own priority");

        // Donation is temporary: once every mutex is free the effective priorities are the base ones again.
        bool restored = true;
        for (int i = 0; i < 3; i++)
        {
            if (nested.EffectivePriority(i) != nested.Tasks[i].Priority) restored = false;
        }
        Check(restored, "after the last release every effective priority has fallen back to its base value");
        Check(!nested.IsBlocked(0) && !nested.IsBlocked(1) && !nested.IsBlocked(2),
              "the finished system has no blocked tasks left");

        // ---- randomised cross-check ----------------------------------------------------------
        ulong state = 668001UL;
        int improved = 0;
        int contended = 0;
        int plainViolating = 0;
        int totalEpisodes = 0;
        int scenarios = 60;
        for (int s = 0; s < scenarios; s++)
        {
            List<TaskSpec> spec = RandomScenario(ref state, 3 + (int)(Rng.NextBelow(ref state, 2)));
            Sim a = new Sim(spec, 1, false);
            a.Run(20000);
            Sim b = new Sim(spec, 1, true);
            b.Run(20000);
            Check(!a.Deadlocked && !b.Deadlocked, "random workload " + s + " terminates both ways (lock discipline forbids deadlock)");
            Check(a.Violations == 0 && b.Violations == 0, "random workload " + s + " keeps all invariants");
            Check(b.BoundViolations == 0, "random workload " + s + ": the top task's every inheritance episode fits"
                  + " inside the holder's remaining critical section (" + b.BoundViolations + " violations)");
            totalEpisodes += b.EpisodesCompleted;
            if (a.BoundViolations > 0) plainViolating++;
            int top = 0;
            for (int i = 1; i < spec.Count; i++) if (spec[i].Priority > spec[top].Priority) top = i;
            Check(b.BlockedTicks[top] <= a.BlockedTicks[top],
                  "random workload " + s + ": PIP does not lengthen the top task's blocking ("
                  + b.BlockedTicks[top] + " <= " + a.BlockedTicks[top] + ")");
            Check(b.Complete[top] <= a.Complete[top],
                  "random workload " + s + ": the top task never finishes later under PIP ("
                  + b.Complete[top] + " <= " + a.Complete[top] + ")");
            if (b.BlockedTicks[top] < a.BlockedTicks[top]) improved++;
            bool anyBlocked = false;
            for (int i = 0; i < spec.Count; i++) if (b.BlockedTicks[i] > 0) anyBlocked = true;
            if (anyBlocked) contended++;
        }
        Check(contended >= 25, "the workload generator produces real contention in " + contended + " of "
              + scenarios + " runs");
        Check(totalEpisodes >= 20, "the randomised runs complete " + totalEpisodes + " blocking episodes");
        Check(plainViolating >= 3, "the plain protocol breaks the blocking bound in " + plainViolating + " of "
              + scenarios + " randomised workloads");
        Check(improved >= 3, "inheritance shortens the top task's blocking in " + improved + " of "
              + scenarios + " randomised workloads");

        Console.WriteLine("plain-run blocking of H        : " + plain.BlockedTicks[high] + " ticks");
        Console.WriteLine("inheritance blocking of H      : " + pip.BlockedTicks[high] + " ticks (one critical section)");
        Console.WriteLine("random workloads: contended     : " + contended + " / " + scenarios);
        Console.WriteLine("random workloads: plain breaks bound: " + plainViolating + " / " + scenarios);
        Console.WriteLine("random workloads where PIP wins : " + improved + " / " + scenarios);
        Console.WriteLine("PASS");
    }
}
