// Task 937 - Implement deployment pipeline with canary analysis using online metrics and rollback.
// (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
//
// A canary controller walks a ramp plan (1%, 5%, 10%, 25%, 50%, 100%): each step sends a fixed
// request budget, routing every request to the canary or the baseline with a deterministic PRNG,
// and records the observed error/latency outcome. When a step's budget is spent the controller
// decides with two online guardrails - a pooled two-proportion z-test on the error rate and a
// mean-latency ratio - and either advances, holds the step for another budget, rolls back, or (if
// too little data ever arrives) rolls back conservatively rather than promoting on no evidence.
//
// Verified: a healthy canary walks the whole plan and promotes; a canary with 5x the error rate
// rolls back at the first step that carries enough samples; a canary with an unchanged error rate
// but 2.4x the latency rolls back with a latency reason, so the two guardrails are distinguished;
// after a rollback the canary receives no further traffic while the baseline keeps serving; and
// the recorded per-step decisions are cross-checked by an oracle that recomputes the pooled
// z-statistic from the raw request log rather than the controller's incremental counters.
// Simplified: one controller instance, no real traffic splitter (routing is a Bernoulli draw),
// errors are memoryless and latency is sampled from a small jitter range, so there is no serially
// correlated latency, no shared-fate correlation between the arms and no rollback of a stateful
// deployment - but the decision rule, its evidence thresholds and rollback semantics are real.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal readonly struct RequestRecord
{
    internal readonly bool Canary;
    internal readonly bool Error;
    internal readonly int LatencyMs;

    internal RequestRecord(bool canary, bool error, int latencyMs)
    {
        Canary = canary;
        Error = error;
        LatencyMs = latencyMs;
    }
}

internal readonly struct ArmStats
{
    internal readonly long Requests;
    internal readonly long Errors;
    internal readonly long LatencySum;

    internal ArmStats(long requests, long errors, long latencySum)
    {
        Requests = requests;
        Errors = errors;
        LatencySum = latencySum;
    }

    internal double ErrorRate => Requests == 0 ? 0.0 : (double)Errors / Requests;
    internal double MeanLatency => Requests == 0 ? 0.0 : (double)LatencySum / Requests;
}

internal sealed class CanaryController
{
    internal const double ZThreshold = 3.0;

    private readonly int[] ramp;
    private readonly int minSamples;    private readonly int requestsPerStep;
    private readonly int maxRetries;
    private readonly double baselineErrorRate;
    private readonly double canaryErrorRate;
    private readonly int baselineLatencyMs;
    private readonly int canaryLatencyMs;
    private readonly double latencyTolerance;
    private ulong state;

    internal readonly List<RequestRecord> Log = new List<RequestRecord>();
    internal readonly List<int> StepEnd = new List<int>();
    internal readonly List<string> Decisions = new List<string>();
    internal readonly List<int> Applied = new List<int>();
    internal readonly List<int> Served = new List<int>();
    internal string Outcome = "running";
    internal string Reason = "";
    internal long BaselineRequests, BaselineErrors, CanaryRequests, CanaryErrors;

    internal int MinSamples => minSamples;
    internal double LatencyTolerance => latencyTolerance;

    internal CanaryController(int[] ramp, int minSamples, int requestsPerStep, int maxRetries,
        double baselineErrorRate, double canaryErrorRate, int baselineLatencyMs, int canaryLatencyMs,
        double latencyTolerance, ulong seed)
    {
        this.ramp = ramp;
        this.minSamples = minSamples;
        this.requestsPerStep = requestsPerStep;
        this.maxRetries = maxRetries;
        this.baselineErrorRate = baselineErrorRate;
        this.canaryErrorRate = canaryErrorRate;
        this.baselineLatencyMs = baselineLatencyMs;
        this.canaryLatencyMs = canaryLatencyMs;
        this.latencyTolerance = latencyTolerance;
        state = seed;
    }

    private double Uniform() => (SplitMix64.Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    private void Serve(int percent, int count)
    {
        for (int i = 0; i < count; i++)
        {
            bool canary = Uniform() < percent / 100.0;
            bool error = Uniform() < (canary ? canaryErrorRate : baselineErrorRate);
            int latency = (canary ? canaryLatencyMs : baselineLatencyMs) + (int)(Uniform() * 10.0);
            Log.Add(new RequestRecord(canary, error, latency));
            if (canary)
            {
                CanaryRequests++;
                if (error) CanaryErrors++;
            }
            else
            {
                BaselineRequests++;
                if (error) BaselineErrors++;
            }
        }
    }

    /** Serves the tail after a rollback: baseline only, never the canary. */
    internal void ServeBaselineOnly(int count) => Serve(0, count);

    private ArmStats StatsOf(int from, int to, bool canary)
    {
        long requests = 0, errors = 0, latency = 0;
        for (int i = from; i < to; i++)
        {
            RequestRecord record = Log[i];
            if (record.Canary != canary) continue;
            requests++;
            latency += record.LatencyMs;
            if (record.Error) errors++;
        }
        return new ArmStats(requests, errors, latency);
    }

    internal static double ZScore(ArmStats baseline, ArmStats canary)
    {
        if (baseline.Requests == 0 || canary.Requests == 0) return 0.0;
        double pooled = (double)(baseline.Errors + canary.Errors) / (baseline.Requests + canary.Requests);
        double standardError = Math.Sqrt(pooled * (1 - pooled)
            * (1.0 / baseline.Requests + 1.0 / canary.Requests));
        return standardError == 0.0 ? 0.0 : (canary.ErrorRate - baseline.ErrorRate) / standardError;
    }

    private string Decide(int from, int to)
    {
        ArmStats canary = StatsOf(from, to, true);
        // The control arm is the cumulative baseline traffic seen so far, so the 100% step (which
        // carries no baseline requests of its own) is still judged against real evidence.
        ArmStats reference = StatsOf(0, to, false);
        if (canary.Requests < minSamples || reference.Requests < minSamples) return "hold";
        if (canary.MeanLatency > reference.MeanLatency * latencyTolerance) return "rollback-latency";
        return ZScore(reference, canary) > ZThreshold ? "rollback-error" : "advance";
    }

    internal void Run()
    {
        foreach (int percent in ramp)
        {
            int retries = 0;
            while (true)
            {
                int from = Log.Count;
                Serve(percent, requestsPerStep);
                Served.Add(percent);
                StepEnd.Add(Log.Count);
                string decision = Decide(from, Log.Count);
                Decisions.Add(decision);
                if (decision == "advance")
                {
                    Applied.Add(percent);
                    break;
                }
                if (decision == "rollback-error" || decision == "rollback-latency")
                {
                    Applied.Add(percent);
                    Outcome = "rolled-back";
                    Reason = decision;
                    return;
                }
                retries++;
                if (retries >= maxRetries)
                {
                    Applied.Add(percent);
                    Outcome = "rolled-back";
                    Reason = "insufficient-evidence";
                    return;
                }
            }
        }
        Outcome = "promoted";
        Reason = "evidence-passed";
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static readonly int[] Plan = { 1, 5, 10, 25, 50, 100 };

    private static CanaryController Healthy() => new CanaryController(Plan, 300, 60000, 3, 0.02, 0.02, 25, 25, 1.5, 0x9370A0UL);
    private static CanaryController BadErrors() => new CanaryController(Plan, 300, 60000, 3, 0.02, 0.10, 25, 25, 1.5, 0x9370B0UL);
    private static CanaryController BadLatency() => new CanaryController(Plan, 300, 60000, 3, 0.02, 0.02, 25, 60, 1.5, 0x9370C0UL);
    private static CanaryController NoEvidence() => new CanaryController(Plan, 5000, 60000, 3, 0.02, 0.10, 25, 25, 1.5, 0x9370D0UL);

    private static string[] ExpectedDecisions(CanaryController controller)
    {
        var expected = new List<string>();
        for (int s = 0; s < controller.StepEnd.Count; s++)
        {
            int from = s == 0 ? 0 : controller.StepEnd[s - 1];
            ArmStats canary = Slice(controller, from, controller.StepEnd[s], true);
            ArmStats baseline = Slice(controller, 0, controller.StepEnd[s], false);
            if (canary.Requests < controller.MinSamples || baseline.Requests < controller.MinSamples) { expected.Add("hold"); continue; }
            if (canary.MeanLatency > baseline.MeanLatency * controller.LatencyTolerance) { expected.Add("rollback-latency"); continue; }
            expected.Add(CanaryController.ZScore(baseline, canary) > CanaryController.ZThreshold ? "rollback-error" : "advance");
        }
        return expected.ToArray();
    }

    /** Oracle slice statistics recomputed from the raw request log, not from the counters. */
    private static ArmStats Slice(CanaryController controller, int from, int to, bool canary)
    {
        long requests = 0, errors = 0, latency = 0;
        for (int i = from; i < to; i++)
        {
            RequestRecord record = controller.Log[i];
            if (record.Canary != canary) continue;
            requests++;
            latency += record.LatencyMs;
            if (record.Error) errors++;
        }
        return new ArmStats(requests, errors, latency);
    }

    public static void Main()
    {
        // 1. Healthy canary: the whole plan is applied in order and the outcome is a promotion.
        var healthy = Healthy();
        healthy.Run();
        Check(healthy.Outcome == "promoted" && healthy.Applied.SequenceEqual(Plan),
            "a healthy canary walks the full ramp and promotes (" + healthy.Outcome + "/" + healthy.Reason
            + "; canary=" + healthy.CanaryRequests + " decisions=" + string.Join(",", healthy.Decisions) + ")");

        // 2. Error regression: rollback at the first step that carries enough samples.
        var badErrors = BadErrors();
        badErrors.Run();
        Check(badErrors.Outcome == "rolled-back" && badErrors.Reason == "rollback-error"
              && badErrors.Served.Distinct().Count() == 1 && badErrors.Served[0] == 1,
            "a 5x error rate rolls back at the 1% step (" + badErrors.Reason + " after " + badErrors.Served.Count + " budget(s))");
        Check(badErrors.Served.Max() <= 1, "the canary never carried more than 1% of the traffic");

        // 3. Latency regression: same error rate, but the latency guardrail fires.
        var badLatency = BadLatency();
        badLatency.Run();
        ArmStats canarySlice = Slice(badLatency, 0, badLatency.StepEnd[0], true);
        ArmStats baseSlice = Slice(badLatency, 0, badLatency.StepEnd[0], false);
        Check(badLatency.Reason == "rollback-latency"
              && Math.Abs(CanaryController.ZScore(baseSlice, canarySlice)) < CanaryController.ZThreshold,
            "a 2.4x latency regression rolls back while the error guardrail stays quiet (z="
            + CanaryController.ZScore(baseSlice, canarySlice).ToString("F2") + ")");

        // 4. Oracle: the incremental counters equal a recount of the raw request log.
        Check(healthy.Log.Count == healthy.BaselineRequests + healthy.CanaryRequests
              && healthy.Log.Count(r => r.Error) == healthy.BaselineErrors + healthy.CanaryErrors
              && healthy.Log.Count(r => r.Canary) == healthy.CanaryRequests
              && healthy.BaselineRequests > 0 && healthy.CanaryRequests > 0,
            "the running counters equal a recount of the " + healthy.Log.Count + " logged requests");

        // 5. Oracle: every recorded decision is reproduced from the log slices.
        string[] expected = ExpectedDecisions(healthy);
        Check(expected.SequenceEqual(healthy.Decisions),
            "the oracle reproduces all " + healthy.Decisions.Count + " per-step decisions from the log");
        var badExpected = ExpectedDecisions(badErrors);
        Check(badExpected.SequenceEqual(badErrors.Decisions), "the oracle also reproduces the rollback decisions");

        // 6. Safety after rollback: the tail is served by the baseline alone.
        long canaryBefore = badErrors.CanaryRequests;
        long baselineBefore = badErrors.BaselineRequests;
        badErrors.ServeBaselineOnly(20000);
        Check(badErrors.CanaryRequests == canaryBefore && badErrors.BaselineRequests == baselineBefore + 20000,
            "after the rollback the canary is frozen and the baseline absorbs all traffic");

        // 7. Boundary: promotion requires evidence - a bad canary with an unreachable sample size.
        var noEvidence = NoEvidence();
        noEvidence.Run();
        Check(noEvidence.Outcome == "rolled-back" && noEvidence.Reason == "insufficient-evidence"
              && noEvidence.Decisions.All(d => d == "hold"),
            "without enough samples the controller rolls back instead of promoting blind");
        var wiseHealthy = new CanaryController(Plan, 5000, 60000, 2, 0.02, 0.02, 25, 25, 1.5, 0x9370E0UL);
        wiseHealthy.Run();
        Check(wiseHealthy.Outcome == "rolled-back" && wiseHealthy.Applied.Count == 1,
            "even a healthy canary is not promoted on insufficient evidence");

        // 8. Boundary: an empty ramp promotes immediately without dividing by zero.
        var trivial = new CanaryController(new int[0], 300, 60000, 3, 0.02, 0.02, 25, 25, 1.5, 1UL);
        trivial.Run();
        Check(trivial.Outcome == "promoted" && trivial.Log.Count == 0 && trivial.Applied.Count == 0,
            "an empty ramp is a trivial promotion with no traffic and no arithmetic");

        // 9. Determinism: the same seed replays the same ramp, decisions and statistics.
        var repeatHealthy = Healthy();
        repeatHealthy.Run();
        var repeatBad = BadErrors();
        repeatBad.Run();
        Check(repeatHealthy.Decisions.SequenceEqual(healthy.Decisions) && repeatBad.Decisions.SequenceEqual(badErrors.Decisions)
              && repeatHealthy.CanaryErrors == healthy.CanaryErrors && repeatBad.CanaryErrors == badErrors.CanaryErrors,
            "two controllers with the same seed agree on every decision and error count");

        Console.WriteLine("  healthy: " + healthy.Log.Count + " requests, " + healthy.Decisions.Count
            + " steps -> " + healthy.Outcome + "; bad-errors: rollback at " + badErrors.Served[0] + "%"
            + "; bad-latency: " + badLatency.Reason);
        Console.WriteLine("PASS");
    }
}
