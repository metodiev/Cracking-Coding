// Task 667 - Implement schedulability analysis for real-time systems (rate-monotonic).
// (Tier: Expert, CTCI Beyond - Networking, OS & Real-Time)
//
// Rate-monotonic scheduling of independent, preemptive periodic tasks with implicit deadlines (D = T), where
// priority is inversely proportional to the period. Three analyses are implemented and cross-checked against
// each other and against an actual schedule:
//   * the Liu & Layland utilisation bound U = sum(C/T) <= n * (2^(1/n) - 1), sufficient but not necessary;
//   * exact response-time analysis, R_i = C_i + sum over higher-priority tasks j of ceil(R_i/T_j) * C_j
//     iterated to a fixed point, feasible iff R_i <= T_i for every task;
//   * a unit-time simulation of the hyperperiod, RM and EDF, which measures the response time of every job.
// The theory says the simulation's worst response time equals the response-time-analysis figure, because the
// synchronous release at time zero is the critical instant for every task under fixed priorities - so the
// simulation is the brute-force cross-check of the analysis rather than a second copy of it. The file also
// verifies the classical implications the theory promises: passing the utilisation bound implies passing the
// response-time test (the bound is sufficient); a task set that fails the bound can still pass the response-time
// test (it is not necessary, shown with a witness); rate-monotonic feasibility implies EDF feasibility (EDF is
// optimal for implicit deadlines); and a density of at most one is sufficient for EDF. What is simplified:
// integer periods and execution times, one processor, no blocking, no release jitter and no shared resources -
// the analysis itself (fixed-point iteration, critical instant, deadline misses) is genuine.
// Deterministic splitmix64 input, no wall clock, hyperperiods capped at a few thousand ticks.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Task
{
    internal int Index;
    internal int Period;
    internal int Cost;

    internal Task(int index, int period, int cost)
    {
        Index = index;
        Period = period;
        Cost = cost;
    }

    public override string ToString() => "T" + Index + "(C=" + Cost + ",T=" + Period + ")";
}

internal static class RmAnalysis
{
    internal static double Utilization(IReadOnlyList<Task> tasks)
    {
        double total = 0;
        foreach (Task task in tasks) total += (double)task.Cost / task.Period;
        return total;
    }

    internal static double Density(IReadOnlyList<Task> tasks)
    {
        double total = 0;
        foreach (Task task in tasks) total += (double)task.Cost / task.Period;   // D = T here
        return total;
    }

    internal static double LiuLaylandBound(int tasks) => tasks * (Math.Pow(2, 1.0 / tasks) - 1);

    internal static bool PassesUtilisationBound(IReadOnlyList<Task> tasks) =>
        tasks.Count > 0 && Utilization(tasks) <= LiuLaylandBound(tasks.Count) + 1e-12;

    internal static int HighestPriority(int task, IReadOnlyList<Task> tasks)
    {
        int higher = 0;
        foreach (Task other in tasks)
        {
            if (other.Index == task) continue;
            if (other.Period < tasks[task].Period) higher++;
            else if (other.Period == tasks[task].Period && other.Index < task) higher++;
        }
        return higher;
    }

    /// <summary>Exact response-time analysis: fixed-point iteration of the interference equation.</summary>
    internal static double ResponseTime(IReadOnlyList<Task> tasks, int index, out int iterations)
    {
        Task task = tasks[index];
        int higher = HighestPriority(index, tasks);
        double response = task.Cost;
        iterations = 0;
        while (true)
        {
            iterations++;
            double interference = 0;
            int seen = 0;
            foreach (Task other in tasks)
            {
                if (other.Index == task.Index) continue;
                bool isHigher = other.Period < task.Period || (other.Period == task.Period && other.Index < task.Index);
                if (!isHigher) continue;
                seen++;
                interference += Math.Ceiling(response / other.Period) * other.Cost;
            }
            if (seen != higher) throw new InvalidOperationException("higher-priority set is inconsistent");
            double next = task.Cost + interference;
            if (Math.Abs(next - response) < 1e-12) return response;
            if (next > task.Period + 1e-12 || iterations > 10000) return next;   // diverged: already infeasible
            response = next;
        }
    }

    internal static double[] AllResponseTimes(IReadOnlyList<Task> tasks, out bool feasible)
    {
        var responses = new double[tasks.Count];
        feasible = true;
        for (int i = 0; i < tasks.Count; i++)
        {
            responses[i] = ResponseTime(tasks, i, out _);
            if (responses[i] > tasks[i].Period + 1e-12) feasible = false;
        }
        return responses;
    }

    internal static int Hyperperiod(IReadOnlyList<Task> tasks)
    {
        long value = 1;
        foreach (Task task in tasks)
        {
            value = Lcm(value, task.Period);
            if (value > 100000) return int.MaxValue;
        }
        return (int)value;
    }

    private static long Lcm(long a, long b) => a / Gcd(a, b) * b;

    private static long Gcd(long a, long b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }

    internal sealed class Simulation
    {
        internal double[] WorstResponse = Array.Empty<double>();
        internal bool MissedDeadline;
        internal int JobsCompleted;
        internal int Preemptions;
    }

    /// <summary>
    /// Unit-time simulation over one hyperperiod. `edf` selects earliest-deadline-first instead of
    /// rate-monotonic priority. Tasks are released synchronously at time 0, which is the critical instant.
    /// </summary>
    internal static Simulation Run(IReadOnlyList<Task> tasks, bool edf, int hyperperiod)
    {
        var result = new Simulation { WorstResponse = new double[tasks.Count] };
        int n = tasks.Count;
        var remaining = new int[n];
        var releaseTime = new int[n];
        var deadline = new double[n];
        var active = new bool[n];
        int longestPeriod = 0;
        foreach (Task task in tasks) longestPeriod = Math.Max(longestPeriod, task.Period);
        // Jobs released at the end of the hyperperiod still need room to finish, so the run is extended.
        int horizon = hyperperiod + longestPeriod;

        for (int now = 0; now < horizon; now++)
        {
            if (now < hyperperiod)
            {
                for (int i = 0; i < n; i++)
                {
                    if (now % tasks[i].Period != 0) continue;
                    if (active[i])
                    {
                        // The previous job of this task has not finished: its deadline is already missed.
                        result.MissedDeadline = true;
                        continue;
                    }
                    active[i] = true;
                    remaining[i] = tasks[i].Cost;
                    releaseTime[i] = now;
                    deadline[i] = now + tasks[i].Period;
                }
            }

            int pick = -1;
            for (int i = 0; i < n; i++)
            {
                if (!active[i] || remaining[i] == 0) continue;
                if (pick < 0) { pick = i; continue; }
                bool better = edf
                    ? deadline[i] < deadline[pick]
                    : tasks[i].Period < tasks[pick].Period;
                if (edf && deadline[i] == deadline[pick] && tasks[i].Index < tasks[pick].Index) better = true;
                if (!edf && tasks[i].Period == tasks[pick].Period && tasks[i].Index < tasks[pick].Index) better = true;
                if (better) pick = i;
            }
            if (pick < 0) continue;

            for (int i = 0; i < n; i++)
                if (i != pick && active[i] && remaining[i] > 0) result.Preemptions++;

            remaining[pick]--;
            if (remaining[pick] == 0)
            {
                double response = now + 1 - releaseTime[pick];
                if (response > result.WorstResponse[pick]) result.WorstResponse[pick] = response;
                if (response > tasks[pick].Period + 1e-12) result.MissedDeadline = true;
                active[pick] = false;
                result.JobsCompleted++;
            }
        }
        // Anything still holding the processor at the end of the horizon never made its deadline.
        for (int i = 0; i < n; i++) if (active[i]) result.MissedDeadline = true;
        return result;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static bool Close(double a, double b) => Math.Abs(a - b) <= 1e-9;

    private static List<Task> Make(params (int Cost, int Period)[] entries)
    {
        var tasks = new List<Task>();
        for (int i = 0; i < entries.Length; i++) tasks.Add(new Task(i, entries[i].Period, entries[i].Cost));
        return tasks;
    }

    private static string Describe(IReadOnlyList<Task> tasks)
    {
        var parts = new List<string>();
        foreach (Task task in tasks) parts.Add(task.ToString());
        return string.Join(" ", parts);
    }

    public static void Main()
    {
        // ---- utilisation, the bound and the degenerate task sets --------------------------------
        {
            List<Task> classic = Make((1, 4), (1, 5), (1, 10));
            Check(Close(RmAnalysis.Utilization(classic), 0.25 + 0.2 + 0.1),
                "utilisation is the sum of C/T: " + RmAnalysis.Utilization(classic));
            Check(Close(RmAnalysis.Utilization(classic), DirectUtilisation(classic)),
                "the summed utilisation matches an independently accumulated total");
            Check(Close(RmAnalysis.LiuLaylandBound(1), 1.0) && Close(RmAnalysis.LiuLaylandBound(2), 2 * (Math.Sqrt(2) - 1))
                    && RmAnalysis.LiuLaylandBound(3) < 0.78 && RmAnalysis.LiuLaylandBound(10) > 0.71,
                "the Liu & Layland bound falls from 1.0 towards ln 2 as n grows");
            Check(RmAnalysis.PassesUtilisationBound(classic),
                "the classic task set passes the utilisation bound (" + RmAnalysis.Utilization(classic).ToString("F4")
                + " <= " + RmAnalysis.LiuLaylandBound(3).ToString("F4") + ")");

            var empty = new List<Task>();
            Check(Close(RmAnalysis.Utilization(empty), 0) && RmAnalysis.AllResponseTimes(empty, out bool emptyOk).Length == 0
                    && emptyOk, "an empty task set is trivially schedulable");
            Check(!RmAnalysis.PassesUtilisationBound(empty), "the utilisation bound is not defined for an empty set");

            var single = Make((3, 10));
            double[] singleResponse = RmAnalysis.AllResponseTimes(single, out bool singleOk);
            Check(singleOk && Close(singleResponse[0], 3), "a lone task's response time is its execution time");
            var overloaded = Make((12, 10));
            double[] overloadedResponse = RmAnalysis.AllResponseTimes(overloaded, out bool overloadedOk);
            Check(!overloadedOk && overloadedResponse[0] > 10,
                "a task whose execution exceeds its period is infeasible (" + overloadedResponse[0] + ")");
        }

        // ---- response-time analysis against the simulation --------------------------------------
        {
            ulong state = 0x0BADF00DCAFEBABEUL;
            int sets = 0, feasibleSets = 0, boundPasses = 0;
            bool responseAgrees = true, feasibilityAgrees = true, boundSufficient = true, edfDominates = true;
            string? failure = null;
            for (int trial = 0; trial < 120; trial++)
            {
                int n = 1 + Rng.NextInt(ref state, 4);
                var tasks = new List<Task>();
                for (int i = 0; i < n; i++)
                {
                    int period = 2 + Rng.NextInt(ref state, 11);
                    int cost = 1 + Rng.NextInt(ref state, Math.Min(period, 4));
                    tasks.Add(new Task(i, period, cost));
                }
                int hyperperiod = RmAnalysis.Hyperperiod(tasks);
                if (hyperperiod > 1200) continue;
                sets++;

                double[] responses = RmAnalysis.AllResponseTimes(tasks, out bool feasible);
                bool bound = RmAnalysis.PassesUtilisationBound(tasks);
                if (feasible) feasibleSets++;
                if (bound) boundPasses++;
                if (bound && !feasible)
                {
                    boundSufficient = false;
                    failure ??= "bound passed but RTA failed for " + Describe(tasks);
                }

                RmAnalysis.Simulation rm = RmAnalysis.Run(tasks, false, hyperperiod);
                if (feasible)
                {
                    for (int i = 0; i < n; i++)
                    {
                        if (!Close(rm.WorstResponse[i], responses[i]))
                        {
                            responseAgrees = false;
                            failure ??= "task " + i + " of " + Describe(tasks) + ": simulation "
                                + rm.WorstResponse[i] + " vs RTA " + responses[i];
                        }
                    }
                }
                if (rm.MissedDeadline != !feasible)
                {
                    feasibilityAgrees = false;
                    failure ??= "feasibility mismatch for " + Describe(tasks) + " (RTA=" + feasible
                        + ", simulation missed=" + rm.MissedDeadline + ")";
                }

                RmAnalysis.Simulation edfRun = RmAnalysis.Run(tasks, true, hyperperiod);
                if (feasible && edfRun.MissedDeadline) edfDominates = false;
                if (RmAnalysis.Density(tasks) <= 1 + 1e-12 && edfRun.MissedDeadline) edfDominates = false;
            }
            Check(sets > 60, "the random workload produced " + sets + " usable task sets");
            Check(feasibleSets > 5 && boundPasses > 5,
                "both the response-time test (" + feasibleSets + ") and the bound (" + boundPasses
                + ") had something to say");
            Check(boundSufficient, "passing the utilisation bound always implies passing response-time analysis: "
                + (failure ?? "ok"));
            Check(responseAgrees, "the simulation's worst response time equals the analysis for every task: "
                + (failure ?? "ok"));
            Check(feasibilityAgrees, "response-time feasibility predicts the simulation's deadline-miss verdict: "
                + (failure ?? "ok"));
            Check(edfDominates, "every set that is rate-monotonic feasible is also EDF feasible, and a density of "
                + "at most one is always EDF feasible: " + (failure ?? "ok"));
        }

        // ---- the utilisation bound is sufficient but not necessary ------------------------------
        {
            List<Task> witness = Make((2, 5), (6, 13));
            Check(!RmAnalysis.PassesUtilisationBound(witness),
                "the witness set fails the utilisation bound (" + RmAnalysis.Utilization(witness).ToString("F4")
                + " > " + RmAnalysis.LiuLaylandBound(2).ToString("F4") + ")");
            double[] responses = RmAnalysis.AllResponseTimes(witness, out bool feasible);
            Check(feasible && Close(responses[0], 2) && Close(responses[1], 10),
                "yet response-time analysis proves it schedulable: " + Describe(witness));
            var simulation = RmAnalysis.Run(witness, false, RmAnalysis.Hyperperiod(witness));
            Check(!simulation.MissedDeadline && Close(simulation.WorstResponse[1], 10),
                "and the simulation agrees: the second task finishes in 10 of its 13 ticks");

            List<Task> infeasible = Make((3, 5), (5, 7));
            double[] bad = RmAnalysis.AllResponseTimes(infeasible, out bool ok);
            Check(!ok && bad[1] > 7, "the second witness is provably infeasible: R2 = " + bad[1] + " > 7");
            var badSimulation = RmAnalysis.Run(infeasible, false, RmAnalysis.Hyperperiod(infeasible));
            Check(badSimulation.MissedDeadline,
                "and the simulation reports the deadline miss (" + badSimulation.JobsCompleted + " jobs completed)");
            var edfOnBad = RmAnalysis.Run(infeasible, true, RmAnalysis.Hyperperiod(infeasible));
            Check(edfOnBad.MissedDeadline && RmAnalysis.Density(infeasible) > 1,
                "even EDF cannot schedule it because the density exceeds one");
        }

        // ---- the fixed point is a fixed point ---------------------------------------------------
        {
            ulong state = 0x1234FEDC5678ABCDUL;
            int checkedSets = 0;
            bool stable = true, monotone = true, boundedIterations = true;
            for (int trial = 0; trial < 80; trial++)
            {
                int n = 1 + Rng.NextInt(ref state, 4);
                var tasks = new List<Task>();
                for (int i = 0; i < n; i++)
                    tasks.Add(new Task(i, 2 + Rng.NextInt(ref state, 10), 1 + Rng.NextInt(ref state, 3)));
                if (RmAnalysis.Hyperperiod(tasks) > 1200) continue;
                checkedSets++;
                for (int i = 0; i < n; i++)
                {
                    double first = RmAnalysis.ResponseTime(tasks, i, out int iterations);
                    double second = RmAnalysis.ResponseTime(tasks, i, out _);
                    if (!Close(first, second)) stable = false;
                    if (iterations > 1000 && first <= tasks[i].Period) boundedIterations = false;
                    if (first < tasks[i].Cost) monotone = false;
                    // At the fixed point the interference identity must hold exactly for a schedulable task.
                    if (first <= tasks[i].Period
                        && !Close(interferenceSum(tasks, i, first), Math.Max(0, first - tasks[i].Cost)))
                        monotone = false;
                }
            }
            Check(checkedSets > 40, "the fixed-point workload covered " + checkedSets + " task sets");
            Check(stable, "recomputing a response time returns the same fixed point");
            Check(monotone, "every response time is at least the execution time and equals C plus its interference");
            Check(boundedIterations, "schedulable tasks converge within the iteration cap");
        }

        // ---- the classic rate-monotonic-versus-EDF witness --------------------------------------
        {
            List<Task> two = Make((2, 5), (4, 7));
            double[] responses = RmAnalysis.AllResponseTimes(two, out bool feasible);
            Check(!feasible && Close(responses[1], 8),
                "R2 = 4 + ceil(6/5)*2 = 8 > 7, so this set is not rate-monotonic schedulable: " + responses[1]);
            var rmRun = RmAnalysis.Run(two, false, 35);
            Check(rmRun.MissedDeadline, "the rate-monotonic simulation reports the same deadline miss ("
                + rmRun.JobsCompleted + " jobs completed)");
            Check(RmAnalysis.Density(two) <= 1 + 1e-12,
                "its density is below one: " + RmAnalysis.Density(two).ToString("F4"));
            var edfRun = RmAnalysis.Run(two, true, 35);
            Check(!edfRun.MissedDeadline && edfRun.JobsCompleted == 12,
                "EDF schedules the very same set without a miss (" + edfRun.JobsCompleted + " jobs), which is "
                + "why rate-monotonic is a priority assignment rather than an optimal schedule");

            List<Task> harmonic = Make((1, 2), (1, 4), (2, 8));
            Check(Close(RmAnalysis.Utilization(harmonic), 1.0) && !RmAnalysis.PassesUtilisationBound(harmonic),
                "a harmonic set can use the whole processor while still failing the n*0.828 bound for n = 3");
            double[] harmonicResponses = RmAnalysis.AllResponseTimes(harmonic, out bool harmonicOk);
            Check(harmonicOk && Close(harmonicResponses[1], 2) && Close(harmonicResponses[2], 8),
                "response-time analysis proves the harmonic set schedulable, with the last task exactly on its "
                + "deadline: " + string.Join(",", harmonicResponses));
            var harmonicRun = RmAnalysis.Run(harmonic, false, 8);
            Check(!harmonicRun.MissedDeadline && harmonicRun.Preemptions > 0,
                "the schedule is fully busy yet meets every deadline (" + harmonicRun.Preemptions + " preemptions)");
            Console.WriteLine("harmonicResponses=" + string.Join(",", harmonicResponses)
                + " utilisation=" + RmAnalysis.Utilization(harmonic));
        }

        Console.WriteLine("PASS");
    }

    private static double DirectUtilisation(IReadOnlyList<Task> tasks)
    {
        double total = 0;
        for (int i = 0; i < tasks.Count; i++) total += (double)tasks[i].Cost / tasks[i].Period;
        return total;
    }

    /// <summary>Independently recomputed interference term for a task at a given response time.</summary>
    private static double interferenceSum(IReadOnlyList<Task> tasks, int index, double response)
    {
        Task task = tasks[index];
        double total = 0;
        foreach (Task other in tasks)
        {
            if (other.Index == task.Index) continue;
            bool isHigher = other.Period < task.Period || (other.Period == task.Period && other.Index < task.Index);
            if (isHigher) total += Math.Ceiling(response / other.Period) * other.Cost;
        }
        return total;
    }
}
