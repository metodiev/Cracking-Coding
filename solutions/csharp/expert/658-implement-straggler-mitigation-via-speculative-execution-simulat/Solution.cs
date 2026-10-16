// Task 658 - Implement straggler mitigation via speculative execution simulation.
// (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// A job of T tasks on W worker slots, simulated as an event-driven scheduler. A task is done when its first
// attempt finishes; the other attempts of that task are then killed and their elapsed time is booked as wasted
// work, which is the "only one attempt commits" rule that makes speculation safe for deterministic, idempotent
// tasks. Speculation uses the map-reduce rule: once a gate fraction of the tasks has committed, a *running*
// task whose estimated remaining time exceeds `factor` times the average remaining time of the running tasks
// gets a second attempt (Hadoop's progress-rate rule, expressed in remaining time), with a cap on how many
// attempts a task may ever hold.
// What is simplified: an attempt's duration is known up front rather than estimated from progress phases; the
// straggler's slowness is modelled as a per-attempt anomaly (its first attempt is inflated by a straggler
// factor) which a fresh attempt on an idle worker does not inherit - that assumption is precisely why
// speculation can help; the speculation decision is evaluated at attempt-completion boundaries rather than by
// a polling thread; there is no data locality, network, task failure or back-pressure.
// The invariant that survives all of that, and is the reason speculation is safe, is that a speculative
// attempt is only ever started on a worker that is otherwise idle, so it cannot delay a pending task. That is
// why the checks can assert that speculation never increases completion time when all tasks are equal.
// Verified: the event loop equals an independently written 0.25-tick simulation on 20 random instances, equals
// sum(durations) with one worker and max(durations) with enough workers; a single straggler is mitigated by
// more than 2x; completion is non-increasing as the rule gets more aggressive while wasted work stays inside
// the worker-time budget; every task commits exactly once; no attempt is launched before the gate, for a
// committed task, for a non-running task, or past the cap; a cap of zero disables speculation; the
// single-task job, the one-worker job and the empty job behave; and runs are reproducible.
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

internal sealed class Attempt
{
    internal int Task;
    internal double Start;
    internal double Duration;
    internal bool Speculative;
}

internal sealed class JobOutcome
{
    internal double Completion;
    internal double WastedWork;
    internal int Commits;
    internal int SpeculativeLaunches;
    internal int GateViolations;
    internal int PrematureLaunches;
    internal int OverCapLaunches;
    internal int[] AttemptsPerTask = Array.Empty<int>();
}

internal sealed class JobSpec
{
    internal double[] BaseDuration = Array.Empty<double>();
    internal int StragglerTask = -1;
    internal double StragglerFactor = 1.0;

    /// <summary>Duration of the first (non-speculative) attempt of a task.</summary>
    internal double FirstAttemptDuration(int task) =>
        task == StragglerTask ? BaseDuration[task] * StragglerFactor : BaseDuration[task];

    internal double[] FirstAttemptDurations()
    {
        var durations = new double[BaseDuration.Length];
        for (int task = 0; task < durations.Length; task++) durations[task] = FirstAttemptDuration(task);
        return durations;
    }
}

internal static class SpeculativeScheduler
{
    /// <summary>
    /// Event-driven run. A task may hold at most 1 + extraAttempts attempts over its lifetime, so
    /// extraAttempts == 0 disables speculation. `factor` is the remaining-time ratio that marks a straggler.
    /// </summary>
    internal static JobOutcome Run(JobSpec job, int workers, bool speculation, double gate, double factor,
                                  int extraAttempts)
    {
        if (workers <= 0) throw new ArgumentException("the worker count must be positive");
        int tasks = job.BaseDuration.Length;
        var outcome = new JobOutcome { AttemptsPerTask = new int[tasks] };
        var committed = new bool[tasks];
        var running = new List<Attempt>();
        double now = 0;
        int commits = 0;

        void Start(int task, bool speculative)
        {
            double duration = speculative ? job.BaseDuration[task] : job.FirstAttemptDuration(task);
            running.Add(new Attempt { Task = task, Start = now, Duration = duration, Speculative = speculative });
            outcome.AttemptsPerTask[task]++;
            if (!speculative) return;
            outcome.SpeculativeLaunches++;
            if (commits < gate * tasks - 1e-9) outcome.GateViolations++;
            if (committed[task]) outcome.PrematureLaunches++;
            if (outcome.AttemptsPerTask[task] > 1 + extraAttempts) outcome.OverCapLaunches++;
        }

        bool IsRunning(int task)
        {
            foreach (Attempt attempt in running) if (attempt.Task == task) return true;
            return false;
        }

        void StartPending()
        {
            for (int task = 0; task < tasks && running.Count < workers; task++)
                if (!committed[task] && !IsRunning(task)) Start(task, false);
        }

        while (commits < tasks)
        {
            StartPending();

            if (speculation && commits >= gate * tasks - 1e-9 && running.Count < workers)
            {
                double remainingSum = 0;
                int active = 0;
                for (int task = 0; task < tasks; task++)
                {
                    if (committed[task] || !IsRunning(task)) continue;
                    remainingSum += RemainingOf(task);
                    active++;
                }
                double average = active > 0 ? remainingSum / active : 0;
                int chosen = -1;
                double worst = 0;
                for (int task = 0; task < tasks; task++)
                {
                    if (committed[task] || !IsRunning(task)) continue;
                    if (outcome.AttemptsPerTask[task] > extraAttempts) continue;
                    double remaining = RemainingOf(task);
                    if (remaining <= average * factor) continue;
                    if (remaining > worst) { worst = remaining; chosen = task; }
                }
                if (chosen >= 0) Start(chosen, true);
            }

            if (running.Count == 0) break;
            Attempt? finisher = null;
            double next = double.MaxValue;
            foreach (Attempt attempt in running)
            {
                double finish = attempt.Start + attempt.Duration;
                if (finish < next) { next = finish; finisher = attempt; }
            }
            now = next;
            int done = finisher!.Task;
            foreach (Attempt attempt in running.ToArray())
            {
                if (attempt.Task != done) continue;
                if (!ReferenceEquals(attempt, finisher)) outcome.WastedWork += now - attempt.Start;
                running.Remove(attempt);
            }
            if (!committed[done]) { committed[done] = true; commits++; }
        }

        outcome.Completion = now;
        outcome.Commits = commits;
        return outcome;

        double RemainingOf(int task)
        {
            double remaining = 0;
            foreach (Attempt attempt in running)
                if (attempt.Task == task)
                    remaining = Math.Max(remaining, attempt.Start + attempt.Duration - now);
            return remaining;
        }
    }

    /// <summary>
    /// Independent implementation for the no-speculation case: fixed 0.25-time ticks, launch whenever a slot is
    /// free. A cross-check of the event loop rather than a second copy of it.
    /// </summary>
    internal static double TickSimulation(double[] firstAttemptDurations, int workers)
    {
        const double Step = 0.25;
        int tasks = firstAttemptDurations.Length;
        var finishesAt = new double[tasks];
        double now = 0;
        while (true)
        {
            int active = 0;
            int finished = 0;
            for (int task = 0; task < tasks; task++)
            {
                if (finishesAt[task] <= 0) continue;
                if (finishesAt[task] <= now) finished++;
                else active++;
            }
            if (finished == tasks) break;
            for (int task = 0; task < tasks && active < workers; task++)
                if (finishesAt[task] == 0) { finishesAt[task] = now + firstAttemptDurations[task]; active++; }
            now += Step;
        }
        double completion = 0;
        foreach (double finish in finishesAt) completion = Math.Max(completion, finish);
        return completion;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static bool Close(double a, double b) => Math.Abs(a - b) <= 1e-9;

    private static JobSpec MakeJob(int tasks, int baseDuration, int spread, ulong seed, int stragglerTask,
                                  double stragglerFactor)
    {
        ulong state = seed;
        var job = new JobSpec { StragglerTask = stragglerTask, StragglerFactor = stragglerFactor };
        var durations = new double[tasks];
        for (int i = 0; i < tasks; i++) durations[i] = baseDuration + Rng.NextInt(ref state, spread);
        job.BaseDuration = durations;
        return job;
    }

    private static double Sum(double[] values)
    {
        double total = 0;
        foreach (double value in values) total += value;
        return total;
    }

    private static double Max(double[] values)
    {
        double best = 0;
        foreach (double value in values) best = Math.Max(best, value);
        return best;
    }

    public static void Main()
    {
        // ---- the event loop against an independent tick simulation and two hand formulas --------
        {
            ulong state = 0x0BADC0FFEE0DDF00UL;
            bool tickAgrees = true;
            for (int t = 0; t < 20; t++)
            {
                JobSpec job = MakeJob(1 + Rng.NextInt(ref state, 9), 4, 7, Rng.Next(ref state), -1, 1.0);
                int workers = 1 + Rng.NextInt(ref state, 5);
                double eventDriven = SpeculativeScheduler.Run(job, workers, false, 0.5, 1.5, 1).Completion;
                double ticked = SpeculativeScheduler.TickSimulation(job.FirstAttemptDurations(), workers);
                if (!Close(eventDriven, ticked)) tickAgrees = false;
            }
            Check(tickAgrees, "the event-driven scheduler matches an independent tick simulation on 20 instances");

            JobSpec ladder = MakeJob(6, 5, 4, 0x1111222233334444UL, -1, 1.0);
            double[] ladderDurations = ladder.FirstAttemptDurations();
            Check(Close(SpeculativeScheduler.Run(ladder, 1, false, 0.5, 1.5, 1).Completion, Sum(ladderDurations)),
                "one worker serialises the tasks, so completion equals the sum of durations");
            Check(Close(SpeculativeScheduler.Run(ladder, ladderDurations.Length, false, 0.5, 1.5, 1).Completion,
                    Max(ladderDurations)),
                "with a worker per task completion equals the longest task");
            Check(Close(SpeculativeScheduler.Run(ladder, 2, true, 0.5, 1.5, 2).Completion,
                    SpeculativeScheduler.Run(ladder, 2, true, 0.5, 1.5, 2).Completion),
                "repeated runs of the same schedule are identical");
        }

        // ---- a straggler is actually mitigated -------------------------------------------------
        {
            JobSpec job = MakeJob(24, 3, 2, 0xABCDEF0192837465UL, 0, 40.0);
            double[] firstAttempts = job.FirstAttemptDurations();
            const int Workers = 8;
            var plain = SpeculativeScheduler.Run(job, Workers, false, 0.5, 1.5, 2);
            var mitigated = SpeculativeScheduler.Run(job, Workers, true, 0.5, 1.5, 2);

            Check(Close(plain.Completion, firstAttempts[0]),
                "without speculation the job waits for the straggler's inflated first attempt ("
                + plain.Completion + " vs " + firstAttempts[0] + ")");
            Check(mitigated.Completion < plain.Completion / 2,
                "speculative execution cuts completion by more than 2x (" + mitigated.Completion + " vs "
                + plain.Completion + ")");
            Check(mitigated.Commits == job.BaseDuration.Length && plain.Commits == job.BaseDuration.Length,
                "every task becomes committed exactly once");
            Check(mitigated.SpeculativeLaunches > 0, "the straggler really did trigger speculative attempts");
            Check(mitigated.GateViolations == 0 && mitigated.PrematureLaunches == 0 && mitigated.OverCapLaunches == 0,
                "no attempt was launched before the gate, for a committed task, or past the cap");
            Check(mitigated.WastedWork >= 0 && mitigated.WastedWork <= mitigated.Completion * Workers,
                "wasted work is bounded by the total worker time (" + mitigated.WastedWork + ")");

            bool capped = true;
            foreach (int attempts in mitigated.AttemptsPerTask) if (attempts > 3) capped = false;
            Check(capped, "no task ever exceeded 1 + extraAttempts attempts");

            var neverFires = SpeculativeScheduler.Run(job, Workers, true, 0.5, 1000.0, 2);
            Check(Close(neverFires.Completion, plain.Completion) && neverFires.SpeculativeLaunches == 0,
                "an unreachable straggler threshold leaves the schedule exactly as it was");
            var disabled = SpeculativeScheduler.Run(job, Workers, true, 0.5, 1.5, 0);
            Check(Close(disabled.Completion, plain.Completion) && disabled.SpeculativeLaunches == 0,
                "a cap of zero extra attempts disables speculation entirely");
            var lateGate = SpeculativeScheduler.Run(job, Workers, true, 1.0, 1.5, 2);
            Check(lateGate.GateViolations == 0,
                "a gate of 1.0 still never launches an attempt before the gate is passed");
            Console.WriteLine("stragglerFirstAttempt=" + firstAttempts[0] + " plainCompletion=" + plain.Completion
                + " mitigatedCompletion=" + mitigated.Completion + " launches=" + mitigated.SpeculativeLaunches
                + " wastedWork=" + mitigated.WastedWork);
        }

        // ---- speculation must never hurt -------------------------------------------------------
        {
            JobSpec even = MakeJob(16, 20, 1, 0x5555666677778888UL, -1, 1.0);
            var plain = SpeculativeScheduler.Run(even, 4, false, 0.5, 1.5, 2);
            var speculative = SpeculativeScheduler.Run(even, 4, true, 0.5, 1.5, 2);
            Check(Close(speculative.Completion, plain.Completion),
                "with equal durations speculation cannot finish earlier and never delays the job ("
                + speculative.Completion + " vs " + plain.Completion + ")");

            JobSpec skewed = MakeJob(30, 8, 6, 0x9999AAAABBBBCCCCUL, 0, 9.0);
            skewed.StragglerTask = 0;
            double previous = double.MaxValue;
            bool monotone = true, wastedSane = true, cappedEverywhere = true;
            foreach (double factor in new[] { 1000.0, 2.0, 1.25, 0.5 })
            {
                var outcome = SpeculativeScheduler.Run(skewed, 5, true, 0.4, factor, 2);
                if (outcome.Completion > previous + 1e-9) monotone = false;
                if (outcome.WastedWork < 0 || outcome.WastedWork > outcome.Completion * 5) wastedSane = false;
                foreach (int attempts in outcome.AttemptsPerTask) if (attempts > 3) cappedEverywhere = false;
                previous = outcome.Completion;
            }
            Check(monotone, "a more aggressive straggler rule never makes the job finish later");
            Check(wastedSane, "wasted work stays inside the worker-time budget at every aggressiveness level");
            Check(cappedEverywhere, "the attempt cap holds at every aggressiveness level");
            var mostAggressive = SpeculativeScheduler.Run(skewed, 5, true, 0.4, 0.5, 4);
            var cappedAggressive = SpeculativeScheduler.Run(skewed, 5, true, 0.4, 0.5, 2);
            Check(mostAggressive.Completion <= cappedAggressive.Completion + 1e-9,
                "allowing more attempts per task cannot make the job slower");
            Console.WriteLine("skewedPlain=" + SpeculativeScheduler.Run(skewed, 5, false, 0.4, 1.5, 2).Completion
                + " skewedSpeculative=" + cappedAggressive.Completion);
        }

        // ---- degenerate jobs -------------------------------------------------------------------
        {
            JobSpec single = MakeJob(1, 37, 1, 0x1234, -1, 1.0);
            var one = SpeculativeScheduler.Run(single, 3, true, 0.5, 1.5, 2);
            Check(Close(one.Completion, 37.0) && one.SpeculativeLaunches == 0 && one.Commits == 1,
                "a single-task job has nothing to speculate on and finishes with its only attempt");

            JobSpec pair = MakeJob(2, 5, 1, 0x5678, -1, 1.0);
            var serial = SpeculativeScheduler.Run(pair, 1, true, 0.0, 0.0, 2);
            Check(Close(serial.Completion, 10.0) && serial.SpeculativeLaunches == 0,
                "with one worker a speculative attempt never fits, so serial execution is preserved");

            var degenerate = SpeculativeScheduler.Run(MakeJob(0, 1, 1, 0x9ABC, -1, 1.0), 4, true, 0.5, 1.5, 2);
            Check(Close(degenerate.Completion, 0) && degenerate.Commits == 0,
                "an empty job completes instantly with no commits");

            bool rejected = false;
            try { SpeculativeScheduler.Run(single, 0, true, 0.5, 1.5, 1); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "a job with no workers is rejected");
        }

        Console.WriteLine("PASS");
    }
}
