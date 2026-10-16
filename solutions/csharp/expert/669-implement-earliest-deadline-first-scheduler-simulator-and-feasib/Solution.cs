// Task 669 - Implement earliest deadline first scheduler simulator and feasibility check.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A preemptive uniprocessor real-time scheduler with three cooperating pieces:
//   * a tick-level EDF simulator over a whole hyperperiod: jobs are released at multiples of each period,
//     the ready job with the earliest absolute deadline runs, and a job whose absolute deadline elapses
//     while it still has work left is reported as a miss (the deadline is then pushed to infinity so the
//     same job is not counted twice and the run can continue);
//   * the same simulator driven by rate-monotonic priority (shorter period wins) so the classic
//     U <= n(2^(1/n) - 1) bound can be contrasted with EDF on a set that RM cannot schedule;
//   * the exact processor-demand feasibility test, dbf(L) = sum_i max(0, floor((L - D_i)/T_i) + 1) * C_i
//     for every L > 0, which is necessary and sufficient for preemptive EDF with synchronous releases and
//     constrained deadlines D_i <= T_i. The test is evaluated at the classical candidate set
//     {D_k + j*T_k}, and the file *cross-checks that shortcut* against a brute-force evaluation of dbf at
//     every integer L in [0, hyperperiod] — the shortcut is only trusted because it agrees on all 200
//     randomised sets. Utilization is compared exactly as integer work-in-one-hyperperiod vs the
//     hyperperiod itself (no floating point in the accept/reject path).
// The documented invariant: for every randomised task set the simulator's verdict over one hyperperiod
// (no deadline misses) equals the demand test's verdict, EDF's busiest-tick accounting matches
// sum C_i * (H / T_i), nobody executes more than its C_i within a job, and a feasible EDF schedule has
// every observed response time bounded by its deadline.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic, no framework or wall-clock dependency.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    internal static int NextRange(ref ulong state, int low, int high) => low + NextBelow(ref state, high - low + 1);
}

internal sealed class TaskSet
{
    internal int[] C = Array.Empty<int>();
    internal int[] T = Array.Empty<int>();
    internal int[] D = Array.Empty<int>();

    internal int Count => C.Length;

    internal static TaskSet Of(int[] c, int[] t, int[] d) => new TaskSet { C = c, T = t, D = d };

    internal int Hyperperiod()
    {
        int h = 1;
        for (int i = 0; i < Count; i++) h = Rt.Lcm(h, T[i]);
        return h;
    }

    /// Total work released in one hyperperiod: sum C_i * (H / T_i). Integer arithmetic on purpose.
    internal int WorkPerHyperperiod()
    {
        int h = Hyperperiod();
        int total = 0;
        for (int i = 0; i < Count; i++) total += C[i] * (h / T[i]);
        return total;
    }
}

internal sealed class SimResult
{
    internal bool Feasible = true;
    internal int Misses;
    internal int FirstMissTick = -1;
    internal int Executed;
    internal int IdleTicks;
    internal int JobOverruns;                       // a job that ran longer than its C_i (must stay 0)
    internal int MaxResponse = -1;
    internal int[] MaxResponsePerTask = Array.Empty<int>();
    internal readonly List<int> Schedule = new List<int>();   // task index per tick, -1 for idle
}

internal static class Rt
{
    internal static int Gcd(int a, int b)
    {
        while (b != 0)
        {
            int t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    internal static int Lcm(int a, int b) => a / Gcd(a, b) * b;

    /// Exact test U <= 1: the work released in one hyperperiod fits the hyperperiod.
    internal static bool UtilizationAtMostOne(TaskSet set) =>
        set.WorkPerHyperperiod() <= set.Hyperperiod();

    /// dbf(L): processor demand of all jobs with deadline <= L, synchronous release.
    internal static int Demand(TaskSet set, int L)
    {
        int total = 0;
        for (int i = 0; i < set.Count; i++)
        {
            if (L < set.D[i]) continue;
            total += ((L - set.D[i]) / set.T[i] + 1) * set.C[i];
        }
        return total;
    }

    /// Exact test at the candidate points {D_k + j*T_k <= H}; overload (dbf > L) anywhere rejects.
    internal static bool DemandFeasible(TaskSet set)
    {
        int h = set.Hyperperiod();
        for (int k = 0; k < set.Count; k++)
        {
            for (int L = set.D[k]; L <= h; L += set.T[k])
            {
                if (Demand(set, L) > L) return false;
            }
        }
        return true;
    }

    /// Brute force: evaluate dbf at every integer L in [0, H]. Used only to validate the shortcut.
    internal static bool DemandFeasibleBruteForce(TaskSet set)
    {
        int h = set.Hyperperiod();
        for (int L = 0; L <= h; L++)
        {
            if (Demand(set, L) > L) return false;
        }
        return true;
    }

    /// First L with dbf(L) > L, brute force; -1 when the set is feasible.
    internal static int FirstOverloadTick(TaskSet set)
    {
        int h = set.Hyperperiod();
        for (int L = 0; L <= h; L++)
        {
            if (Demand(set, L) > L) return L;
        }
        return -1;
    }

    /// EDF: earliest absolute deadline first, ties broken by task index for determinism.
    internal static SimResult SimulateEdfOrdered(TaskSet set, int horizon) => Simulate(set, horizon, true);

    /// Rate-monotonic: shorter period first (the classical priority assignment EDF beats).
    internal static SimResult SimulateRateMonotonic(TaskSet set, int horizon) => Simulate(set, horizon, false);

    private static SimResult Simulate(TaskSet set, int horizon, bool earliestDeadlineFirst)
    {
        int n = set.Count;
        var result = new SimResult();
        result.MaxResponsePerTask = new int[n];
        var remaining = new int[n];
        var deadline = new int[n];
        var released = new int[n];
        var jobLeft = new int[n];
        for (int i = 0; i < n; i++) deadline[i] = int.MaxValue;

        for (int t = 0; t < horizon; t++)
        {
            for (int i = 0; i < n; i++)
            {
                if (t % set.T[i] != 0) continue;
                remaining[i] = set.C[i];
                jobLeft[i] = set.C[i];
                released[i] = t;
                deadline[i] = t + set.D[i];
            }

            int pick = -1;
            for (int i = 0; i < n; i++)
            {
                if (remaining[i] <= 0) continue;
                if (pick < 0) { pick = i; continue; }
                long iKey = earliestDeadlineFirst ? deadline[i] : set.T[i];
                long pickKey = earliestDeadlineFirst ? deadline[pick] : set.T[pick];
                if (iKey < pickKey || (iKey == pickKey && i < pick)) pick = i;
            }

            if (pick < 0)
            {
                result.IdleTicks++;
                result.Schedule.Add(-1);
                continue;
            }

            remaining[pick]--;
            result.Executed++;
            jobLeft[pick]--;
            if (jobLeft[pick] < 0) result.JobOverruns++;
            result.Schedule.Add(pick);
            if (remaining[pick] == 0)
            {
                int response = t + 1 - released[pick];
                if (response > result.MaxResponse) result.MaxResponse = response;
                if (response > result.MaxResponsePerTask[pick]) result.MaxResponsePerTask[pick] = response;
            }
            for (int i = 0; i < n; i++)
            {
                // A job that still has work after this tick cannot finish by a deadline that has now
                // arrived (the next unit of work would start at t+1 and end at t+2).
                if (remaining[i] > 0 && t + 1 >= deadline[i])
                {
                    result.Misses++;
                    result.Feasible = false;
                    if (result.FirstMissTick < 0) result.FirstMissTick = t + 1;
                    deadline[i] = int.MaxValue;   // report this job once, then let it drain
                }
            }
        }
        return result;
    }

    /// The reference implementation of the demand shortcut, kept separate from the simulator so a
    /// mismatch between the two is a test failure rather than a silent agreement.
    internal static int DemandAtCandidatePoints(TaskSet set)
    {
        int h = set.Hyperperiod();
        int worst = -1;
        for (int k = 0; k < set.Count; k++)
        {
            for (int L = set.D[k]; L <= h; L += set.T[k])
            {
                if (Demand(set, L) > L)
                {
                    if (worst < 0 || L < worst) worst = L;
                }
            }
        }
        return worst;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static TaskSet Classic() => TaskSet.Of(new[] { 1, 2, 2 }, new[] { 4, 5, 10 }, new[] { 4, 5, 10 });

    /// {2,5} and {4,7}: EDF schedules it (U = 34/35) but rate-monotonic does not — the textbook case.
    private static TaskSet RmBeaten() => TaskSet.Of(new[] { 2, 4 }, new[] { 5, 7 }, new[] { 5, 7 });

    private static TaskSet Overloaded() => TaskSet.Of(new[] { 3, 3, 3 }, new[] { 5, 5, 5 }, new[] { 5, 5, 5 });

    private static TaskSet TightInfeasible() => TaskSet.Of(new[] { 2, 2 }, new[] { 5, 5 }, new[] { 2, 2 });

    private static TaskSet TightFeasible() => TaskSet.Of(new[] { 1, 2 }, new[] { 6, 6 }, new[] { 2, 6 });

    private static TaskSet RandomSet(ref ulong state)
    {
        int n = Rng.NextRange(ref state, 2, 3);
        var c = new int[n];
        var t = new int[n];
        var d = new int[n];
        for (int i = 0; i < n; i++)
        {
            t[i] = Rng.NextRange(ref state, 2, 8);
            d[i] = Rng.NextRange(ref state, 1, t[i]);
            c[i] = Rng.NextRange(ref state, 1, d[i]);
        }
        return TaskSet.Of(c, t, d);
    }

    public static void Main()
    {
        // ---- arithmetic sanity ----------------------------------------------------------------
        Check(Rt.Gcd(12, 18) == 6 && Rt.Lcm(4, 10) == 20, "gcd/lcm building blocks are correct");
        Check(TaskSet.Of(new[] { 1 }, new[] { 2 }, new[] { 2 }).Hyperperiod() == 2, "a single task's hyperperiod is its period");
        var classic = Classic();
        Check(classic.Hyperperiod() == 20, "hyperperiod of {4,5,10} is 20 (got " + classic.Hyperperiod() + ")");
        int work = classic.WorkPerHyperperiod();
        Check(work == 17, "classic set releases 17 ticks of work per hyperperiod (got " + work + ")");
        Check(Rt.UtilizationAtMostOne(classic), "exact utilization of the classic set is 0.85 <= 1");

        // ---- EDF schedules the classic set, RM does not ----------------------------------------
        int horizon = classic.Hyperperiod();
        SimResult edf = Rt.SimulateEdfOrdered(classic, horizon);
        Check(edf.Feasible && edf.Misses == 0, "EDF meets every deadline on the classic set");
        Check(edf.Executed == work, "EDF executes exactly the released work (" + edf.Executed + ")");
        Check(edf.IdleTicks == horizon - work, "EDF is idle exactly when no work is pending");
        Check(edf.JobOverruns == 0, "no job exceeds its C_i in the EDF run");
        var beaten = RmBeaten();
        Check(beaten.Hyperperiod() == 35 && Rt.UtilizationAtMostOne(beaten),
              "the RM-beaten set has hyperperiod 35 and U = 34/35 <= 1");
        SimResult beatenEdf = Rt.SimulateEdfOrdered(beaten, beaten.Hyperperiod());
        Check(beatenEdf.Feasible && beatenEdf.Misses == 0, "EDF keeps every deadline on the RM-beaten set");
        SimResult rm = Rt.SimulateRateMonotonic(beaten, beaten.Hyperperiod());
        Check(!rm.Feasible && rm.Misses > 0,
              "rate-monotonic misses deadlines on the same set (U = 0.971 > n(2^(1/n)-1) = 0.828)");
        Check(rm.FirstMissTick == 7, "the first RM miss is reported at tick " + rm.FirstMissTick);

        // ---- the demand test, shortcut vs brute force -----------------------------------------
        Check(Rt.DemandFeasible(classic), "the exact demand test accepts the classic set");
        Check(Rt.DemandFeasibleBruteForce(classic), "the brute-force demand sweep agrees");
        Check(Rt.Demand(classic, 5) == 3, "dbf(5) = 1 (task 1) + 2 (task 2) = 3 (got " + Rt.Demand(classic, 5) + ")");

        // ---- overload ------------------------------------------------------------------------
        var over = Overloaded();
        Check(!Rt.UtilizationAtMostOne(over), "U = 1.8 is rejected by the exact utilization test");
        SimResult overSim = Rt.SimulateEdfOrdered(over, over.Hyperperiod());
        Check(!overSim.Feasible && overSim.Misses > 0, "EDF really misses on the overloaded set");

        // ---- constrained deadlines: U <= 1 is not enough --------------------------------------
        var tight = TightInfeasible();
        Check(Rt.UtilizationAtMostOne(tight), "the tight set still satisfies U = 0.8 <= 1");
        Check(!Rt.DemandFeasible(tight), "but the demand test rejects it (two jobs of 2 by time 2)");
        Check(!Rt.DemandFeasibleBruteForce(tight), "and so does the brute-force sweep");
        int overloadTick = Rt.FirstOverloadTick(tight);
        Check(overloadTick == 2, "the first overloaded demand interval is L = 2 (got " + overloadTick + ")");
        Check(Rt.DemandAtCandidatePoints(tight) == overloadTick,
              "the candidate-point shortcut finds the same first overload as the brute force");
        SimResult tightSim = Rt.SimulateEdfOrdered(tight, tight.Hyperperiod());
        Check(!tightSim.Feasible && tightSim.FirstMissTick == 2,
              "the simulator's first deadline miss is at tick " + tightSim.FirstMissTick + " (the overload tick)");
        var tightOk = TightFeasible();
        Check(Rt.DemandFeasible(tightOk), "the feasible tight set passes the demand test");
        SimResult tightOkSim = Rt.SimulateEdfOrdered(tightOk, tightOk.Hyperperiod());
        Check(tightOkSim.Feasible && tightOkSim.MaxResponse <= 6,
              "and EDF schedules it with a worst response of " + tightOkSim.MaxResponse);

        // ---- degenerate inputs --------------------------------------------------------------
        var single = TaskSet.Of(new[] { 2 }, new[] { 2 }, new[] { 2 });
        SimResult singleSim = Rt.SimulateEdfOrdered(single, 6);
        Check(singleSim.Feasible && singleSim.Executed == 6 && singleSim.IdleTicks == 0,
              "a single C=D=T task keeps the CPU busy for the whole horizon");
        var empty = TaskSet.Of(Array.Empty<int>(), Array.Empty<int>(), Array.Empty<int>());
        Check(Rt.DemandFeasible(empty) && Rt.DemandFeasibleBruteForce(empty), "the empty set is trivially feasible");
        SimResult emptySim = Rt.SimulateEdfOrdered(empty, 5);
        Check(emptySim.Feasible && emptySim.IdleTicks == 5 && emptySim.Executed == 0, "the empty set idles forever");

        // ---- randomised cross-check ---------------------------------------------------------
        ulong state = 669001UL;
        int sets = 200;
        int feasibleSeen = 0;
        for (int s = 0; s < sets; s++)
        {
            TaskSet set = RandomSet(ref state);
            bool demand = Rt.DemandFeasible(set);
            bool demandBrute = Rt.DemandFeasibleBruteForce(set);
            Check(demand == demandBrute, "set " + s + ": candidate-point and brute-force demand tests agree");
            Check(Rt.DemandAtCandidatePoints(set) == Rt.FirstOverloadTick(set),
                  "set " + s + ": both demand scans report the same first overload");
            SimResult sim = Rt.SimulateEdfOrdered(set, set.Hyperperiod());
            Check(sim.Feasible == demand,
                  "set " + s + ": EDF simulation and demand test agree (sim=" + sim.Feasible + " demand=" + demand + ")");
            Check(sim.JobOverruns == 0, "set " + s + ": no job overruns its execution time");
            Check(sim.Executed + sim.IdleTicks == set.Hyperperiod(), "set " + s + ": every tick is accounted for");
            if (demand)
            {
                feasibleSeen++;
                bool responsesOk = true;
                for (int i = 0; i < set.Count; i++)
                {
                    if (sim.MaxResponsePerTask[i] > set.D[i]) responsesOk = false;
                }
                Check(responsesOk, "set " + s + ": every completed job's response time fits its deadline");
                SimResult again = Rt.SimulateEdfOrdered(set, set.Hyperperiod());
                Check(again.Executed == sim.Executed && again.Misses == sim.Misses && again.IdleTicks == sim.IdleTicks,
                      "set " + s + ": the schedule is reproducible");
                bool sameSchedule = again.Schedule.Count == sim.Schedule.Count;
                for (int k = 0; sameSchedule && k < sim.Schedule.Count; k++)
                {
                    if (again.Schedule[k] != sim.Schedule[k]) sameSchedule = false;
                }
                Check(sameSchedule, "set " + s + ": the tick-by-tick schedule is identical on a re-run");
            }
        }
        Check(feasibleSeen >= 40, "the random sweep contains " + feasibleSeen + " schedulable sets, so both sides are exercised");
        Check(feasibleSeen <= sets - 20, "the sweep also contains " + (sets - feasibleSeen) + " unschedulable sets");

        Console.WriteLine("classic set: utilization 17/20, EDF misses 0, RM misses " + rm.Misses);        Console.WriteLine("tight set  : U = 0.8 but first overload at L = " + overloadTick);
        Console.WriteLine("random sets: " + feasibleSeen + " feasible / " + (sets - feasibleSeen) + " infeasible of " + sets);
        Console.WriteLine("PASS");
    }
}
