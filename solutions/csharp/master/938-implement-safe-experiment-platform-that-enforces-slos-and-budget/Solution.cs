// Task 938 - Implement safe experiment platform that enforces SLOs and budget on experiments.
// (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
//
// An A/B experiment platform with two hard guardrails. Every ramp step is priced before it runs:
// if the next step's cost would not fit inside the remaining budget the step is never started, so
// the spend can never overshoot. Each step then serves a fixed number of observations (routing
// every request to control or treatment with a deterministic PRNG) and is judged on the treatment
// arm against an error-rate SLO and a latency SLO; a breach halts the experiment and fails the
// ramp back to 0%, while a clean step advances. A halted experiment performs no further work.
//
// Verified: a clean experiment walks the whole ramp and lands exactly on its budget boundary; the
// spend invariant (spend <= budget and spend == observations x unit cost) is re-checked after
// every single step; an error regression halts at the first offending step after two clean steps
// (so there is no premature halt); a latency regression halts with the latency reason while the
// error SLO stays satisfied; a tight budget halts without ever paying for an unaffordable step;
// a zero budget halts before any observation; halted ramps read 0%; and an oracle recomputes the
// spend and the per-step guardrail decisions from the raw observation log.
// Simplified: one experiment, memoryless observations, an independent control arm, and the SLO is
// evaluated on the current step's treatment slice rather than a windowed statistic - but budget
// accounting, halt semantics and the fail-safe ramp reset are the real mechanisms.
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

internal readonly struct Observation
{
    internal readonly bool Treatment;
    internal readonly bool Error;
    internal readonly int LatencyMs;

    internal Observation(bool treatment, bool error, int latencyMs)
    {
        Treatment = treatment;
        Error = error;
        LatencyMs = latencyMs;
    }
}

internal sealed class ExperimentPlatform
{
    private readonly int[] ramp;
    private readonly double[] treatmentErrorSchedule;
    private readonly double controlErrorRate;
    private readonly int treatmentLatencyMs;
    private readonly int controlLatencyMs;
    private readonly int observationsPerStep;
    private readonly int costPerObservation;
    private readonly long budget;
    private ulong state;

    internal readonly double ErrorSlo;
    internal readonly double LatencySlo;
    internal readonly List<Observation> Log = new List<Observation>();
    internal readonly List<int> StepPercent = new List<int>();
    internal readonly List<int> StepEnd = new List<int>();
    internal readonly List<long> StepSpend = new List<long>();
    internal readonly List<double> StepErrorRate = new List<double>();
    internal readonly List<string> Decisions = new List<string>();
    internal long Spend;
    internal long InvariantViolations;
    internal string Outcome = "running";
    internal int CurrentPercent;

    internal ExperimentPlatform(int[] ramp, double[] treatmentErrorSchedule, double controlErrorRate,
        int treatmentLatencyMs, int controlLatencyMs, int observationsPerStep, int costPerObservation,
        long budget, double errorSlo, double latencySlo, ulong seed)
    {
        this.ramp = ramp;
        this.treatmentErrorSchedule = treatmentErrorSchedule;
        this.controlErrorRate = controlErrorRate;
        this.treatmentLatencyMs = treatmentLatencyMs;
        this.controlLatencyMs = controlLatencyMs;
        this.observationsPerStep = observationsPerStep;
        this.costPerObservation = costPerObservation;
        this.budget = budget;
        ErrorSlo = errorSlo;
        LatencySlo = latencySlo;
        state = seed;
    }

    private double Uniform() => (SplitMix64.Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    private void Serve(int percent, int count, double treatmentErrorRate)
    {
        for (int i = 0; i < count; i++)
        {
            bool treatment = Uniform() < percent / 100.0;
            bool error = Uniform() < (treatment ? treatmentErrorRate : controlErrorRate);
            int latency = (treatment ? treatmentLatencyMs : controlLatencyMs) + (int)(Uniform() * 10.0);
            Log.Add(new Observation(treatment, error, latency));
            Spend += costPerObservation;
        }
    }

    /** Treatment-arm guardrails for the slice [from, to); reports the observed error rate. */
    private string Guardrails(int from, int to, out double observedErrorRate)
    {
        long treatment = 0, errors = 0, latency = 0;
        for (int i = from; i < to; i++)
        {
            Observation observation = Log[i];
            if (!observation.Treatment) continue;
            treatment++;
            latency += observation.LatencyMs;
            if (observation.Error) errors++;
        }
        observedErrorRate = treatment == 0 ? 0.0 : (double)errors / treatment;
        if (treatment > 0 && (double)latency / treatment > LatencySlo) return "halted-slo-latency";
        return observedErrorRate > ErrorSlo ? "halted-slo-error" : "advance";
    }

    internal void Run()
    {
        if (Outcome != "running") return;  // a finished experiment is stopped, it cannot restart
        long stepCost = (long)observationsPerStep * costPerObservation;
        for (int step = 0; step < ramp.Length; step++)
        {
            if (Spend + stepCost > budget)
            {
                Outcome = "halted-budget";
                Decisions.Add("halt-budget");
                CurrentPercent = 0;
                return;
            }
            int percent = ramp[step];
            CurrentPercent = percent;
            double treatmentErrorRate = treatmentErrorSchedule[Math.Min(step, treatmentErrorSchedule.Length - 1)];
            int from = Log.Count;
            Serve(percent, observationsPerStep, treatmentErrorRate);
            string decision = Guardrails(from, Log.Count, out double observedError);
            StepPercent.Add(percent);
            StepEnd.Add(Log.Count);
            StepSpend.Add(Spend);
            StepErrorRate.Add(observedError);
            Decisions.Add(decision);
            if (Spend > budget || Spend != (long)Log.Count * costPerObservation) InvariantViolations++;
            if (decision != "advance")
            {
                Outcome = decision;
                CurrentPercent = 0;  // fail safe: a halted experiment serves nothing
                return;
            }
        }
        Outcome = "completed";
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static readonly int[] Plan = { 5, 10, 25, 50, 100 };
    private const int ObservationsPerStep = 20000;
    private const int UnitCost = 10;
    private const long StepCost = ObservationsPerStep * (long)UnitCost;

    private static ExperimentPlatform Healthy() => new ExperimentPlatform(Plan, new[] { 0.01 }, 0.01, 30, 30,
        ObservationsPerStep, UnitCost, 5 * StepCost, 0.03, 60, 0x9380A0UL);
    private static ExperimentPlatform ErrorBreach() => new ExperimentPlatform(Plan, new[] { 0.01, 0.01, 0.20 },
        0.01, 30, 30, ObservationsPerStep, UnitCost, 10 * StepCost, 0.03, 60, 0x9380B0UL);
    private static ExperimentPlatform LatencyBreach() => new ExperimentPlatform(Plan, new[] { 0.01 }, 0.01, 120, 30,
        ObservationsPerStep, UnitCost, 10 * StepCost, 0.03, 60, 0x9380C0UL);
    private static ExperimentPlatform TightBudget() => new ExperimentPlatform(Plan, new[] { 0.01 }, 0.01, 30, 30,
        ObservationsPerStep, UnitCost, StepCost + StepCost / 2, 0.03, 60, 0x9380D0UL);
    private static ExperimentPlatform NoBudget() => new ExperimentPlatform(Plan, new[] { 0.01 }, 0.01, 30, 30,
        ObservationsPerStep, UnitCost, 0, 0.03, 60, 0x9380E0UL);

    /** Oracle: recomputes the per-step guardrail verdict from the raw observation log. */
    private static string[] ExpectedDecisions(ExperimentPlatform platform)
    {
        var expected = new List<string>();
        int from = 0;
        for (int s = 0; s < platform.StepEnd.Count; s++)
        {
            long treatment = 0, errors = 0, latency = 0;
            for (int i = from; i < platform.StepEnd[s]; i++)
            {
                Observation observation = platform.Log[i];
                if (!observation.Treatment) continue;
                treatment++;
                latency += observation.LatencyMs;
                if (observation.Error) errors++;
            }
            from = platform.StepEnd[s];
            double rate = treatment == 0 ? 0.0 : (double)errors / treatment;
            if (Math.Abs(rate - platform.StepErrorRate[s]) > 1e-12) expected.Add("metric-mismatch");
            else if (treatment > 0 && (double)latency / treatment > platform.LatencySlo) expected.Add("halted-slo-latency");
            else expected.Add(rate > platform.ErrorSlo ? "halted-slo-error" : "advance");
        }
        return expected.ToArray();
    }

    public static void Main()
    {
        // 1. Clean experiment: the whole ramp runs and the budget is consumed exactly.
        var healthy = Healthy();
        healthy.Run();
        Check(healthy.Outcome == "completed" && healthy.StepPercent.SequenceEqual(Plan)
              && healthy.CurrentPercent == 100,
            "a clean experiment completes the ramp (" + healthy.Outcome + ", " + healthy.StepPercent.Count + " steps)");
        Check(healthy.Spend == 5 * StepCost && healthy.InvariantViolations == 0,
            "the clean experiment spends exactly the budget boundary " + healthy.Spend);

        // 2-3. Invariants after every step, and the oracle recomputation of spend and metrics.
        Check(healthy.InvariantViolations == 0 && healthy.Spend == (long)healthy.Log.Count * UnitCost,
            "spend never exceeded the budget and always equalled observations x unit cost");
        Check(ExpectedDecisions(healthy).SequenceEqual(healthy.Decisions),
            "the oracle reproduces every step decision from the log");
        long oracleSpend = healthy.Log.Count * (long)UnitCost;
        Check(oracleSpend == healthy.Spend && healthy.StepSpend.SequenceEqual(new[] { StepCost, 2 * StepCost, 3 * StepCost, 4 * StepCost, 5 * StepCost }),
            "the running spend matches the oracle's recount at every step");

        // 4. Error regression: two clean steps, then a halt at the step that breaches the SLO.
        var errorBreach = ErrorBreach();
        errorBreach.Run();
        Check(errorBreach.Outcome == "halted-slo-error" && errorBreach.StepPercent.SequenceEqual(new[] { 5, 10, 25 })
              && errorBreach.Decisions.Take(2).All(d => d == "advance"),
            "an error regression halts at the 25% step after two clean steps (no premature halt)");
        Check(errorBreach.Decisions[0] == "advance" && errorBreach.Decisions[1] == "advance"
              && errorBreach.StepErrorRate[2] > errorBreach.ErrorSlo
              && errorBreach.StepErrorRate[0] < errorBreach.ErrorSlo,
            "the halt comes from the treatment error rate crossing the SLO, not from noise");
        Check(ExpectedDecisions(errorBreach).SequenceEqual(errorBreach.Decisions.Take(errorBreach.StepEnd.Count)),
            "the oracle also reproduces the three decisions of the halting experiment");

        // 5. Latency regression: the other guardrail fires while the error SLO is satisfied.
        var latencyBreach = LatencyBreach();
        latencyBreach.Run();
        Check(latencyBreach.Outcome == "halted-slo-latency" && latencyBreach.StepErrorRate[0] <= latencyBreach.ErrorSlo
              && latencyBreach.StepPercent.SequenceEqual(new[] { 5 }),
            "a latency regression halts on the latency SLO while the error SLO is satisfied");

        // 6. Tight budget: the unaffordable step is never started and never paid for.
        var tight = TightBudget();
        tight.Run();
        Check(tight.Outcome == "halted-budget" && tight.Spend == StepCost && tight.StepPercent.SequenceEqual(new[] { 5 })
              && tight.Log.Count == ObservationsPerStep,
            "an unaffordable step is never started and never paid for (spend " + tight.Spend + " <= budget)");
        Check(tight.Decisions.Count == 2 && tight.Decisions[1] == "halt-budget",
            "the budget halt is recorded as a decision without a metric verdict");

        // 7. Boundary: zero budget halts before a single observation.
        var noBudget = NoBudget();
        noBudget.Run();
        Check(noBudget.Outcome == "halted-budget" && noBudget.Spend == 0 && noBudget.Log.Count == 0
              && noBudget.StepPercent.Count == 0,
            "a zero budget halts immediately with no observations and no spend");

        // 8. Safety: a halted experiment reverts the ramp to 0% and freezes its spend.
        Check(healthy.CurrentPercent == 100 && errorBreach.CurrentPercent == 0 && latencyBreach.CurrentPercent == 0
              && tight.CurrentPercent == 0,
            "halted ramps read 0% while the completed experiment stays at 100%");
        long frozen = errorBreach.Spend;
        errorBreach.Run();
        Check(errorBreach.Spend == frozen && errorBreach.StepPercent.Count == 3,
            "re-running a halted experiment changes nothing: spend and steps are frozen");

        // 9-10. Determinism of both the clean and the halting scenario.
        var repeatHealthy = Healthy();
        repeatHealthy.Run();
        var repeatBreach = ErrorBreach();
        repeatBreach.Run();
        Check(repeatHealthy.Spend == healthy.Spend && repeatHealthy.Outcome == healthy.Outcome
              && repeatBreach.Spend == errorBreach.Spend && repeatBreach.Outcome == errorBreach.Outcome
              && repeatBreach.StepErrorRate.SequenceEqual(errorBreach.StepErrorRate),
            "two runs with the same seed agree on outcome, spend and every step metric");

        Console.WriteLine("  clean: " + healthy.Log.Count + " observations, spend " + healthy.Spend + "/" + 5 * StepCost
            + "; error breach halted at " + errorBreach.StepPercent[errorBreach.StepPercent.Count - 1] + "%"
            + "; tight budget spent " + tight.Spend);
        Console.WriteLine("PASS");
    }
}
