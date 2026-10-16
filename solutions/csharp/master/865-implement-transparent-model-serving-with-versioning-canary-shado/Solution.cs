// Task 865 - Implement transparent model serving with versioning, canary, shadow tests, and rollback.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// A deterministic model of a model-serving router that keeps several versions live at once:
//
//   registry  every version carries an error probability and a latency, so a request's outcome and its
//             service time are a pure function of (version, request id, deterministic PRNG)
//   router    a request is bucketed by hash(requestId) % 100 and sent to the canary when its bucket is below
//             the configured canary percentage, otherwise to the stable version; the mapping is sticky, so a
//             fixed configuration always routes a given request to the same version, and pinned requests
//             bypass the split entirely
//   shadow    every request is *also* evaluated by the shadow version, but the shadow's answer and its errors
//             are recorded in the shadow's own counters: they never reach the client
//   guard     after every phase the canary's own requests in that phase are examined and the canary is rolled
//             back when its error rate or its p95 latency crosses the configured guard rails; a rolled-back
//             canary serves nothing until it is explicitly re-enabled, and promotion makes it the stable
//             version and resets its counters
//
// The split share, the per-version counters and the guard-rail verdict are all recomputed from the request
// log, and the shadow's failures are compared with a direct count of its own outcomes, so the checks test
// the router rather than restate it.
//
// What is simplified (documented, not behavioural): latency is a fixed per-version number instead of a
// measured wall clock, the deploy timeline is a hand-written sequence of phases, and the canary analysis
// window is one phase instead of a sliding time window. The registry, the sticky hashing, the shadow
// isolation, the windowed guard rails and the rollback/promotion state machine are the real content; the
// request counts and the guard-rail thresholds are typed parameters.
//
// Invariants verified: every request is served exactly once by exactly one primary version; the canary's
// share tracks the configuration within two points; routing is sticky and pinning is honoured; a fully
// failing shadow produces no client-visible failure and no shadow error reaches the primary's metrics; the
// error-rate guard rail fires on the unhealthy canary and not on the healthy one; a rolled-back canary
// serves nothing until re-enabled; promotion moves all traffic, resets counters and changes which version is
// stable; unknown versions, duplicate registrations and out-of-range percentages are rejected; the whole
// run reproduces from its seed.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng865
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class ModelVersion865
{
    internal string Name = "";
    internal double ErrorRate;
    internal double LatencyMs;

    internal bool Fails(int requestId)
    {
        // A per-request deterministic outcome, so a request always sees the same verdict for a version.
        ulong state = 0x865AUL ^ ((ulong)requestId * 0x9E3779B97F4A7C15UL) ^ (ulong)(Name.Length * 7919);
        return Rng865.Uniform(ref state) < ErrorRate;
    }
}

internal sealed class VersionMetrics865
{
    internal long Served;
    internal long Errors;
    internal long ShadowObserved;
    internal long ShadowErrors;

    internal double ErrorRate => Served == 0 ? 0.0 : Errors / (double)Served;
}

internal sealed class ServedRequest865
{
    internal int RequestId;
    internal string Version = "";
    internal bool Failed;
    internal double LatencyMs;
    internal string ShadowVersion = "";
}

internal sealed class Router865
{
    private readonly Dictionary<string, ModelVersion865> registry = new Dictionary<string, ModelVersion865>();
    internal readonly Dictionary<string, VersionMetrics865> Metrics = new Dictionary<string, VersionMetrics865>();
    internal readonly Dictionary<int, string> Pinned = new Dictionary<int, string>();
    internal string Stable = "";
    internal string Canary = "";
    internal string Shadow = "";
    internal int CanaryPercent;
    internal long Rollbacks;
    internal long Served;
    internal bool CanaryRolledBack;

    internal Router865(ModelVersion865 stable, ModelVersion865 canary, ModelVersion865 shadow, int canaryPercent)
    {
        if (stable == null || canary == null || shadow == null) throw new ArgumentNullException("version");
        if (canaryPercent < 0 || canaryPercent > 100) throw new ArgumentOutOfRangeException(nameof(canaryPercent));
        Register(stable);
        Register(canary);
        Register(shadow);
        Stable = stable.Name;
        Canary = canary.Name;
        Shadow = shadow.Name;
        CanaryPercent = canaryPercent;
    }

    internal void Register(ModelVersion865 version)
    {
        if (registry.ContainsKey(version.Name)) throw new ArgumentException("duplicate version " + version.Name);
        registry[version.Name] = version;
        Metrics[version.Name] = new VersionMetrics865();
    }

    internal ModelVersion865 Version(string name)
    {
        if (!registry.TryGetValue(name, out ModelVersion865 version))
            throw new ArgumentException("unknown version " + name);
        return version;
    }

    internal bool CanaryEnabled => Canary.Length > 0 && CanaryPercent > 0 && !CanaryRolledBack;
    internal bool HasCanary => Canary.Length > 0;

    internal int CanaryBucket(int requestId) => (int)((ulong)requestId * 2654435761UL % 100UL);

    internal string PrimaryFor(int requestId)
    {
        if (Pinned.TryGetValue(requestId, out string pinned)) return pinned;
        bool toCanary = CanaryEnabled && CanaryBucket(requestId) < CanaryPercent;
        return toCanary ? Canary : Stable;
    }

    internal ServedRequest865 Serve(int requestId)
    {
        string primary = PrimaryFor(requestId);
        ModelVersion865 chosen = Version(primary);
        bool failed = chosen.Fails(requestId);
        VersionMetrics865 metrics = Metrics[primary];
        metrics.Served++;
        if (failed) metrics.Errors++;
        Served++;

        // The shadow evaluates the same request; its verdict is never returned to the client.
        ModelVersion865 shadow = Version(Shadow);
        VersionMetrics865 shadowMetrics = Metrics[Shadow];
        shadowMetrics.ShadowObserved++;
        if (shadow.Fails(requestId)) shadowMetrics.ShadowErrors++;

        return new ServedRequest865
        {
            RequestId = requestId,
            Version = primary,
            Failed = failed,
            LatencyMs = chosen.LatencyMs,
            ShadowVersion = Shadow,
        };
    }

    internal void SetCanaryPercent(int percent)
    {
        if (percent < 0 || percent > 100) throw new ArgumentOutOfRangeException(nameof(percent));
        CanaryPercent = percent;
    }

    internal void EnableCanary(int percent)
    {
        SetCanaryPercent(percent);
        CanaryRolledBack = false;
    }

    internal void Rollback()
    {
        CanaryRolledBack = true;
        CanaryPercent = 0;
        Rollbacks++;
    }

    // The canary becomes the stable version and the canary slot is emptied: there is nothing left to canary
    // against until another version is registered in that slot.
    internal void Promote()
    {
        string promoted = Canary;
        CanaryRolledBack = false;
        Stable = promoted;
        Canary = "";
        CanaryPercent = 0;
        Metrics[promoted] = new VersionMetrics865();
    }
}

internal sealed class Guardrail865
{
    internal double MaximumErrorRate;
    internal double MaximumLatencyRatio;

    internal bool ShouldRollback(List<ServedRequest865> window, double stableP95)
    {
        if (window.Count == 0) return false;
        long errors = 0;
        var latencies = new List<double>();
        foreach (ServedRequest865 request in window)
        {
            if (request.Failed) errors++;
            latencies.Add(request.LatencyMs);
        }
        latencies.Sort();
        int index = (int)Math.Ceiling(0.95 * latencies.Count) - 1;
        double p95 = latencies[index < 0 ? 0 : index];
        if (errors / (double)window.Count > MaximumErrorRate) return true;
        return stableP95 > 0.0 && p95 > stableP95 * MaximumLatencyRatio;
    }
}

internal sealed class PhaseReport865
{
    internal string Phase = "";
    internal long CanaryServed;
    internal long CanaryErrors;
    internal double CanaryErrorRate;
    internal long PrimaryServed;
    internal bool RolledBack;
    internal bool GuardrailVerdict;
}

public static class Solution
{
    private const int RequestsPerPhase = 4000;
    private const int DarkLaunchRequests = 2000;

    public static void Main()
    {
        var stable = new ModelVersion865 { Name = "v1-stable", ErrorRate = 0.010, LatencyMs = 10.0 };
        var canary = new ModelVersion865 { Name = "v2-canary", ErrorRate = 0.030, LatencyMs = 12.0 };
        var shadow = new ModelVersion865 { Name = "v3-shadow", ErrorRate = 0.500, LatencyMs = 25.0 };
        var router = new Router865(stable, canary, shadow, 0);
        var guardrail = new Guardrail865 { MaximumErrorRate = 0.05, MaximumLatencyRatio = 1.5 };

        // 1-2. Registry and configuration validation.
        Check(router.Version("v1-stable").Name == "v1-stable" && Throws(() => router.Version("nope")),
            "an unknown model version is rejected by the registry");
        Check(Throws(() => new Router865(stable, canary, shadow, 101))
            && Throws(() => router.SetCanaryPercent(-1))
            && Throws(() => router.Register(stable)),
            "an out-of-range canary percentage and a duplicate registration are rejected");

        // 3-6. The deploy timeline: dark launch, healthy canary, unhealthy canary, re-enable, promotion.
        var reports = new List<PhaseReport865>();
        var log = new List<ServedRequest865>();
        int requestId = 0;

        PhaseReport865 dark = RunPhase("dark-launch", DarkLaunchRequests, router, guardrail, log, ref requestId);
        reports.Add(dark);
        Check(dark.CanaryServed == 0 && !dark.RolledBack, "the dark-launch phase sends no traffic to the canary");

        router.SetCanaryPercent(10);
        PhaseReport865 healthy = RunPhase("canary-10", RequestsPerPhase, router, guardrail, log, ref requestId);
        reports.Add(healthy);
        Check(healthy.CanaryServed > 0 && !healthy.RolledBack,
            "the healthy canary keeps serving: its error rate " + healthy.CanaryErrorRate.ToString("P2")
            + " is below the guard rail");

        canary.ErrorRate = 0.120;
        router.SetCanaryPercent(50);
        PhaseReport865 bad = RunPhase("canary-12", RequestsPerPhase, router, guardrail, log, ref requestId);
        reports.Add(bad);
        Check(bad.GuardrailVerdict && bad.RolledBack && router.CanaryRolledBack,
            "the unhealthy canary is rolled back by the error-rate guard rail (" + bad.CanaryErrorRate.ToString("P2") + ")");
        bool canaryQuiet = true;
        for (int id = 0; id < 200; id++) if (router.PrimaryFor(id) != stable.Name) canaryQuiet = false;
        Check(canaryQuiet, "a rolled-back canary serves no traffic at all until it is re-enabled");

        canary.ErrorRate = 0.030;
        router.EnableCanary(25);
        PhaseReport865 reEnabled = RunPhase("canary-25", RequestsPerPhase, router, guardrail, log, ref requestId);
        reports.Add(reEnabled);
        Check(reEnabled.CanaryServed > 0 && !reEnabled.RolledBack,
            "after the fix and a re-enable the canary serves the new share without a rollback");

        router.Promote();
        int postPromotionStart = log.Count;
        PhaseReport865 promoted = RunPhase("promoted", DarkLaunchRequests, router, guardrail, log, ref requestId);
        reports.Add(promoted);
        Check(router.Stable == canary.Name && promoted.PrimaryServed == DarkLaunchRequests && promoted.CanaryServed == 0,
            "promotion makes the canary the stable version and every request goes to it");

        // 7-9. Sticky routing, split accuracy and pinning, recomputed from scratch.
        var splitRouter = new Router865(new ModelVersion865 { Name = "s1", ErrorRate = 0.0, LatencyMs = 1.0 },
            new ModelVersion865 { Name = "s2", ErrorRate = 0.0, LatencyMs = 2.0 },
            new ModelVersion865 { Name = "s3", ErrorRate = 0.0, LatencyMs = 3.0 }, 25);
        bool sticky = true;
        long canaryShare = 0;
        for (int i = 0; i < 20000; i++)
        {
            string first = splitRouter.PrimaryFor(i);
            if (splitRouter.PrimaryFor(i) != first) sticky = false;
            if (first == "s2") canaryShare++;
        }
        double sharePercent = canaryShare * 100.0 / 20000;
        Console.WriteLine("sticky split at 25%: canary=" + canaryShare + "/20000 (" + sharePercent.ToString("F2") + "%)");
        Check(sticky, "routing is sticky: a request always maps to the same version for a fixed configuration");
        Check(Math.Abs(sharePercent - 25.0) < 2.0,
            "the canary's share tracks the configured percentage within two points (" + sharePercent.ToString("F2") + "%)");
        for (int id = 7000; id < 7010; id++) splitRouter.Pinned[id] = "s1";
        bool pinned = true;
        for (int id = 7000; id < 7010; id++) if (splitRouter.PrimaryFor(id) != "s1") pinned = false;
        Check(pinned, "pinned requests bypass the split and always reach their pinned version");

        // 10-12. Shadow isolation: every request, no client impact, no leakage into the primary.
        var shadowRouter = new Router865(new ModelVersion865 { Name = "a", ErrorRate = 0.0, LatencyMs = 5.0 },
            new ModelVersion865 { Name = "b", ErrorRate = 0.0, LatencyMs = 6.0 },
            new ModelVersion865 { Name = "c-shadow", ErrorRate = 1.0, LatencyMs = 9.0 }, 0);
        long clientFailures = 0;
        for (int i = 0; i < 500; i++)
        {
            ServedRequest865 served = shadowRouter.Serve(i);
            if (served.Failed) clientFailures++;
            if (served.ShadowVersion != "c-shadow") throw new Exception("the shadow was not consulted");
        }
        Check(clientFailures == 0, "a fully failing shadow never produces a client-visible failure");
        Check(shadowRouter.Metrics["c-shadow"].ShadowObserved == 500
            && shadowRouter.Metrics["c-shadow"].ShadowErrors == 500,
            "the shadow observed every request and recorded all of its own failures");
        Check(shadowRouter.Metrics["a"].Served == 500 && shadowRouter.Metrics["a"].Errors == 0,
            "the primary's metrics contain no shadow traffic and no shadow errors");

        // 13-14. Accounting: one log entry per request, counters equal an independent recount.
        Check(log.Count == router.Served && log.Count == 2 * DarkLaunchRequests + 3 * RequestsPerPhase,
            "the log holds exactly one entry per request (" + log.Count + ")");
        string promotedName = promoted.PrimaryServed > 0 ? router.Stable : "";
        bool metricsMatch = true;
        foreach (KeyValuePair<string, VersionMetrics865> pair in router.Metrics)
        {
            long expected = 0;
            for (int i = 0; i < log.Count; i++)
            {
                if (log[i].Version != pair.Key) continue;
                // Promotion resets the promoted version's counters, so only its suffix counts towards them.
                if (pair.Key == promotedName && i < postPromotionStart) continue;
                expected++;
            }
            if (pair.Value.Served != expected) metricsMatch = false;
        }
        Check(metricsMatch, "per-version served counters equal an independent recount of the request log");
        Check(router.Metrics[router.Shadow].ShadowObserved == log.Count,
            "the shadow observed every request of the whole timeline");

        // 15-17. Guard-rail semantics on hand-built windows and the rollback/promotion state machine.
        var window = new List<ServedRequest865>();
        for (int i = 0; i < 100; i++)
            window.Add(new ServedRequest865 { Failed = i < 6, LatencyMs = 12.0 });
        Check(guardrail.ShouldRollback(window, 10.0), "a 6% error rate trips a 5% guard rail");
        var clean = new List<ServedRequest865>();
        for (int i = 0; i < 100; i++) clean.Add(new ServedRequest865 { Failed = i < 3, LatencyMs = 12.0 });
        Check(!guardrail.ShouldRollback(clean, 10.0), "a 3% error rate and a 1.2x latency stay inside the guard rails");
        Check(guardrail.ShouldRollback(clean, 4.0), "a p95 latency far above the stable version trips the latency rail");
        Check(!guardrail.ShouldRollback(new List<ServedRequest865>(), 10.0), "an empty window never trips a rail");
        Check(reports[3].RolledBack == false && router.Rollbacks == 1,
            "exactly one rollback happened across the whole deploy timeline");

        // 18. Determinism.
        var repeat = new Router865(stable, canary, shadow, 0);
        bool identical = true;
        for (int i = 0; i < 2000; i++)
        {
            ServedRequest865 served = repeat.Serve(i);
            if (served.Failed != stable.Fails(i)) identical = false;
        }
        Check(identical && repeat.Metrics[stable.Name].Served == 2000,
            "the same request ids reproduce the same outcomes bit for bit");

        Console.WriteLine("PASS");
    }

    private static PhaseReport865 RunPhase(string name, int requests, Router865 router, Guardrail865 guardrail,
        List<ServedRequest865> log, ref int requestId)
    {
        var phaseLog = new List<ServedRequest865>();
        for (int i = 0; i < requests; i++)
        {
            ServedRequest865 served = router.Serve(requestId++);
            log.Add(served);
            phaseLog.Add(served);
        }
        var canaryWindow = new List<ServedRequest865>();
        double stableP95 = 0.0;
        foreach (ServedRequest865 served in phaseLog)
        {
            if (router.HasCanary && served.Version == router.Canary) canaryWindow.Add(served);
            if (served.Version == router.Stable && served.LatencyMs > stableP95) stableP95 = served.LatencyMs;
        }
        bool verdict = router.CanaryEnabled && guardrail.ShouldRollback(canaryWindow, stableP95);
        long errorsBefore = router.HasCanary ? router.Metrics[router.Canary].Errors : 0;
        long errorsInWindow = 0;
        foreach (ServedRequest865 served in canaryWindow) if (served.Failed) errorsInWindow++;
        if (verdict) router.Rollback();
        var report = new PhaseReport865
        {
            Phase = name,
            CanaryServed = canaryWindow.Count,
            CanaryErrors = errorsInWindow,
            CanaryErrorRate = canaryWindow.Count == 0 ? 0.0 : errorsInWindow / (double)canaryWindow.Count,
            PrimaryServed = requests - canaryWindow.Count,
            RolledBack = verdict,
            GuardrailVerdict = verdict,
        };
        Console.WriteLine("  phase " + name + ": served=" + requests + " canary=" + report.CanaryServed
            + " canary-error-rate=" + report.CanaryErrorRate.ToString("P2") + " rolled-back=" + verdict
            + " (cumulative canary errors " + errorsBefore + " -> "
            + (router.HasCanary ? router.Metrics[router.Canary].Errors : 0) + ")");
        return report;
    }

    private static bool Throws(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (ArgumentException)
        {
            return true;
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
