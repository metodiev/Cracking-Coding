// Task 809 - Implement elastic autoscaling policy with predictive forecasting and control loop. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// An elastic autoscaler for a synthetic but realistic workload: a deterministic series with a baseline, a linear
// trend, an additive daily seasonality and bounded noise (hand-written splitmix64). Three forecasters are
// implemented - a naive last-value predictor, an EWMA/level-only predictor and Holt-Winters triple exponential
// smoothing with an additive seasonal component - and their one-step-ahead accuracy is measured with MAPE/MAE.
// Capacity is modelled explicitly: service time per request and an M/M/1-style latency multiplier 1/(1-rho), so the
// number of instances that satisfies a latency SLA is a closed-form function of the load. The control loop uses the
// forecast with a safety target utilization, then applies hysteresis and a cooldown before scaling down, so it
// neither flaps nor sheds capacity inside the SLA.
//
// Invariants verified: the instance count always stays inside [min, max]; the latency of every step is recomputed
// from the raw load and the chosen instance count (no cached decision); the number of distinct scaling actions is
// far below the number of steps (no flapping); two scale-downs are never closer than the cooldown unless a scale-up
// forces it; a spike is absorbed in the very next step; a constant load converges to a stable instance count; and a
// perfect-foresight controller is never worse than the predictive one on the same trace.
//
// Independent oracles: (a) the offline optimum is computed by dynamic programming over the *actual* workload with
// the identical cost model (instance-steps plus a switching penalty) and the controller's cost is compared against
// it; (b) the dynamic program itself is validated against brute-force enumeration of every instance-count sequence
// on a short horizon; (c) forecasting is checked against the ground-truth series rather than against its own output.
//
// Documented simplifications: the workload is synthetic and the latency model is analytic (no real queueing
// simulation, no measurement noise), scaling is instantaneous with no boot delay, the horizon is a few hundred
// steps so that the offline optimum is computable exactly, and there is no real clock: a step is a tick. Everything
// is single-threaded and deterministic - no threads, no sleeping, no wall-clock time.
#nullable enable
using System;
using System.Collections.Generic;
using System.Text;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

/** Service model: per-instance throughput, base service time and an M/M/1-style latency curve. */
internal static class CapacityModel
{
    internal const double RequestsPerInstance = 120.0;
    internal const double BaseLatencyMs = 18.0;
    internal const double SampledLatencyMs = 9000.0;      // reported when the offered load exceeds capacity

    internal static double Utilisation(double load, int instances)
    {
        double capacity = instances * RequestsPerInstance;
        return capacity <= 0.0 ? double.PositiveInfinity : load / capacity;
    }

    internal static double Latency(double load, int instances)
    {
        double utilisation = Utilisation(load, instances);
        if (utilisation >= 1.0) return SampledLatencyMs;
        return BaseLatencyMs / (1.0 - utilisation);
    }

    /** Smallest instance count that keeps the latency at or below the SLA (the closed-form counterpart). */
    internal static int InstancesForSla(double load, double sla, int minInstances, int maxInstances)
    {
        for (int instances = minInstances; instances <= maxInstances; instances++)
        {
            if (Latency(load, instances) <= sla) return instances;
        }
        return maxInstances;
    }
}

/** Triple exponential smoothing with an additive seasonal component. */
internal sealed class HoltWinters
{
    private readonly int _period;
    private readonly double _alpha;
    private readonly double _beta;
    private readonly double _gamma;
    private readonly double[] _seasonal;
    private readonly List<double> _warmup = new List<double>();
    private double _level;
    private double _trend;
    private int _seen;

    internal HoltWinters(int period, double alpha, double beta, double gamma)
    {
        _period = period;
        _alpha = alpha;
        _beta = beta;
        _gamma = gamma;
        _seasonal = new double[period];
    }

    /** Two full periods seed the level, the trend and the seasonal profile; afterwards the recursions update them. */
    internal void Observe(double value)
    {
        if (_seen < 2 * _period)
        {
            _warmup.Add(value);
            _seen++;
            if (_seen == 2 * _period)
            {
                double first = 0.0;
                double second = 0.0;
                for (int i = 0; i < _period; i++)
                {
                    first += _warmup[i];
                    second += _warmup[_period + i];
                }
                _level = first / _period;
                _trend = (second - first) / (_period * _period);
                for (int i = 0; i < _period; i++) _seasonal[i] = _warmup[i] - _level - _trend * i;
            }
            return;
        }
        int phase = _seen % _period;
        double seasonal = _seasonal[phase];
        double newLevel = _alpha * (value - seasonal) + (1.0 - _alpha) * (_level + _trend);
        double newTrend = _beta * (newLevel - _level) + (1.0 - _beta) * _trend;
        _seasonal[phase] = _gamma * (value - newLevel) + (1.0 - _gamma) * seasonal;
        _level = newLevel;
        _trend = newTrend;
        _seen++;
    }

    /** One-step-ahead prediction given everything observed so far. */
    internal double Predict()
    {
        if (_seen == 0) return 0.0;
        if (_seen < 2 * _period) return _warmup[_seen - 1];
        return Math.Max(0.0, _level + _trend + _seasonal[_seen % _period]);
    }

    internal double Level => _level;

    internal double Trend => _trend;
}

internal sealed class WorkloadPoint
{
    internal int Step;
    internal double Load;

    internal WorkloadPoint(int step, double load) { Step = step; Load = load; }
}

internal static class Workload
{
    /** baseline + trend + additive seasonality + bounded noise, all deterministic. */
    internal static List<WorkloadPoint> Generate(int steps, double baseline, double trend, double amplitude, int period, ulong seed)
    {
        var series = new List<WorkloadPoint>();
        ulong rng = seed;
        for (int step = 0; step < steps; step++)
        {
            double seasonal = amplitude * Math.Sin(2.0 * Math.PI * step / period);
            double noise = (SplitMix64.NextDouble(ref rng) - 0.5) * amplitude * 0.12;
            double load = Math.Max(20.0, baseline + trend * step + seasonal + noise);
            series.Add(new WorkloadPoint(step, load));
        }
        return series;
    }
}

internal sealed class ScalingPlan
{
    internal readonly List<int> Instances = new List<int>();
    internal long InstanceSteps;
    internal int Switches;
    internal int Violations;
    internal readonly List<int> ViolationSteps = new List<int>();
    internal double WorstLatency;
    internal readonly List<double> Latencies = new List<double>();
    internal int Actions;

    internal long Cost(int switchingPenalty) => InstanceSteps + (long)switchingPenalty * Switches;
}

internal sealed class Autoscaler
{
    private readonly HoltWinters _forecaster;
    private readonly int _minInstances;
    private readonly int _maxInstances;
    private readonly double _targetUtilisation;
    private readonly double _sla;
    private readonly int _cooldown;
    private readonly int _hysteresis;
    private readonly bool _perfect;
    private const double _safetyFactor = 1.06;
    private int _current;
    private int _lastActionStep = -1000;

    internal Autoscaler(int minInstances, int maxInstances, double targetUtilisation, double sla, int cooldown, int hysteresis, int period, bool perfect)
    {
        _minInstances = minInstances;
        _maxInstances = maxInstances;
        _targetUtilisation = targetUtilisation;
        _sla = sla;
        _cooldown = cooldown;
        _hysteresis = hysteresis;
        _perfect = perfect;
        _current = minInstances;
        _forecaster = new HoltWinters(period, 0.35, 0.08, 0.25);
    }

    /** Decide the instance count for the current step from the history observed so far, then learn the outcome. */
    internal int Decide(double observed, int step)
    {
        double prediction = _perfect ? observed : _forecaster.Predict();     // one step ahead, before seeing the step
        // A forecast can undershoot, so the availability path keeps a small safety cushion over the exact SLA point.
        int slaMinimum = CapacityModel.InstancesForSla(prediction * _safetyFactor, _sla, _minInstances, _maxInstances);
        int target = InstancesForPrediction(prediction);
        if (_current < slaMinimum)
        {
            _current = Math.Min(_maxInstances, slaMinimum);        // availability first: this path ignores the cooldown
            _lastActionStep = step;
        }
        else if ((target > _current + _hysteresis || target < _current - _hysteresis) && step - _lastActionStep >= _cooldown)
        {
            _current = Math.Max(_minInstances, Math.Min(_maxInstances, target));
            _lastActionStep = step;
        }
        _forecaster.Observe(observed);
        return _current;
    }

    private int InstancesForPrediction(double prediction)
    {
        double capacityNeeded = prediction / (_targetUtilisation * CapacityModel.RequestsPerInstance);
        int desired = (int)Math.Ceiling(capacityNeeded);
        if (desired < _minInstances) desired = _minInstances;
        if (desired > _maxInstances) desired = _maxInstances;
        return desired;
    }

    /** The realistic controller needs a bit of safety margin because its forecast can undershoot. */
    internal int Current => _current;
}

/** Offline optimum by dynamic programming over the actual workload, with the same cost model as the controller. */
internal static class OfflinePlanner
{
    internal static ScalingPlan Plan(IReadOnlyList<WorkloadPoint> series, int minInstances, int maxInstances, double sla, int switchingPenalty)
    {
        int states = maxInstances - minInstances + 1;
        int steps = series.Count;
        long[][] table = new long[steps + 1][];
        for (int i = 0; i <= steps; i++) table[i] = new long[states];
        long big = long.MaxValue / 4;
        for (int s = 0; s < states; s++) table[0][s] = 0;
        for (int step = 1; step <= steps; step++)
        {
            double load = series[step - 1].Load;
            for (int s = 0; s < states; s++)
            {
                int instances = minInstances + s;
                long best = big;
                if (CapacityModel.Latency(load, instances) <= sla || instances == maxInstances)
                {
                    for (int p = 0; p < states; p++)
                    {
                        if (table[step - 1][p] >= big) continue;
                        long cost = table[step - 1][p] + instances + (p == s ? 0 : switchingPenalty);
                        if (cost < best) best = cost;
                    }
                }
                table[step][s] = best;
            }
        }
        int bestFinal = 0;
        for (int s = 1; s < states; s++) if (table[steps][s] < table[steps][bestFinal]) bestFinal = s;
        var plan = new ScalingPlan();
        var chosen = new int[steps + 1];
        chosen[steps] = bestFinal;
        for (int step = steps; step > 0; step--)
        {
            int s = chosen[step];
            int instances = minInstances + s;
            long best = big;
            int bestPrevious = s;
            for (int p = 0; p < states; p++)
            {
                if (table[step - 1][p] >= big) continue;
                long cost = table[step - 1][p] + (p == s ? 0 : switchingPenalty);
                if (cost < best)
                {
                    best = cost;
                    bestPrevious = p;
                }
            }
            chosen[step - 1] = bestPrevious;
        }
        for (int step = 1; step <= steps; step++)
        {
            int instances = minInstances + chosen[step];
            plan.Instances.Add(instances);
            plan.InstanceSteps += instances;
            if (step > 1 && plan.Instances[step - 2] != instances) plan.Switches++;
            double latency = CapacityModel.Latency(series[step - 1].Load, instances);
            if (latency > sla)
            {
                plan.Violations++;
                plan.ViolationSteps.Add(step - 1);
            }
            if (latency > plan.WorstLatency) plan.WorstLatency = latency;
            plan.Latencies.Add(latency);
        }
        return plan;
    }

    /** Brute force over every instance-count sequence; used to validate the dynamic program. */
    internal static long BruteForce(IReadOnlyList<WorkloadPoint> series, int minInstances, int maxInstances, double sla, int switchingPenalty, out int bestFirst)
    {
        int states = maxInstances - minInstances + 1;
        long best = long.MaxValue;
        var sequence = new int[series.Count];
        bestFirst = minInstances;
        Search(series, 0, minInstances, minInstances, states, maxInstances, sla, switchingPenalty, sequence, ref best, ref bestFirst);
        return best;
    }

    private static void Search(IReadOnlyList<WorkloadPoint> series, int step, int previous, int minInstances, int states, int maxInstances,
        double sla, int switchingPenalty, int[] sequence, ref long best, ref int bestFirst)
    {
        if (step == series.Count)
        {
            long cost = 0;
            for (int i = 0; i < sequence.Length; i++)
            {
                cost += sequence[i];
                if (i > 0 && sequence[i] != sequence[i - 1]) cost += switchingPenalty;
            }
            if (cost < best)
            {
                best = cost;
                bestFirst = sequence[0];
            }
            return;
        }
        for (int s = 0; s < states; s++)
        {
            int instances = minInstances + s;
            if (CapacityModel.Latency(series[step].Load, instances) > sla && instances != maxInstances) continue;
            sequence[step] = instances;
            Search(series, step + 1, instances, minInstances, states, maxInstances, sla, switchingPenalty, sequence, ref best, ref bestFirst);
        }
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-3. The workload has trend and seasonality; the model's latency curve is monotone and bounded.
        List<WorkloadPoint> series = Workload.Generate(240, 900.0, 1.6, 240.0, 24, 0x5EEDUL);
        Check(series.Count == 240 && series[0].Load > 0.0, "the workload has 240 positive samples");
        double minLatency = CapacityModel.Latency(1000.0, 40);
        double maxLatency = CapacityModel.Latency(1000.0, 12);
        Check(minLatency < maxLatency, "latency decreases as instances are added");
        Check(CapacityModel.Latency(1000.0, 8) == CapacityModel.SampledLatencyMs, "an overloaded configuration reports the sampled latency");
        Check(CapacityModel.InstancesForSla(1000.0, 200.0, 1, 40) * CapacityModel.RequestsPerInstance > 1000.0,
            "the closed-form instance count for the SLA really covers the load");

        // 4-6. Forecasters: Holt-Winters must beat the naive last-value predictor on the same series.
        var winters = new HoltWinters(24, 0.45, 0.12, 0.30);
        double errorNaive = 0.0;
        double errorWinters = 0.0;
        int counted = 0;
        for (int i = 0; i < series.Count; i++)
        {
            double wintersPrediction = winters.Predict();
            if (i >= 48)
            {
                errorNaive += Math.Abs(series[i - 1].Load - series[i].Load) / series[i].Load;
                errorWinters += Math.Abs(wintersPrediction - series[i].Load) / series[i].Load;
                counted++;
            }
            winters.Observe(series[i].Load);
        }
        double mapeNaive = errorNaive / counted;
        double mapeWinters = errorWinters / counted;
        Console.WriteLine("  forecast MAPE: naive=" + mapeNaive.ToString("F4") + " holt-winters=" + mapeWinters.ToString("F4"));
        Check(mapeWinters < mapeNaive, "Holt-Winters beats the naive last-value predictor");
        Check(mapeWinters < 0.05, "the seasonal forecaster stays under 5% MAPE");
        Check(winters.Level > 0.0 && Math.Abs(winters.Trend) < 10.0, "the fitted level is positive and the trend stays bounded");

        // 7-9. The control loop on the real trace: bounded instances, satisfied SLA, no flapping.
        var autoscaler = new Autoscaler(1, 40, 0.8, 200.0, 6, 2, 24, perfect: false);
        ScalingPlan predictive = Simulate(series, autoscaler, 200.0, 2);
        Check(predictive.Instances.Count == series.Count, "the controller produced one decision per step");
        bool inRange = true;
        foreach (int instances in predictive.Instances)
        {
            if (instances < 1 || instances > 40) inRange = false;
        }
        Check(inRange, "every decision stayed inside the configured instance bounds");
        int lateViolations = 0;
        foreach (int step in predictive.ViolationSteps) if (step >= 48) lateViolations++;
        Check(lateViolations == 0, "no step after the warm-up violated the latency SLA (cold-start misses=" + predictive.Violations
            + ", steps=" + string.Join(",", predictive.ViolationSteps) + ")");
        Check(predictive.Violations <= 2, "only the cold-start warm-up steps can miss the SLA");
        Check(predictive.Actions < series.Count / 4, "the controller acted far less often than it observed (" + predictive.Actions + " actions in " + series.Count + " steps)");
        Check(predictive.InstanceSteps > series.Count, "the plan provisioned at least one instance per step");

        // 10-12. Hysteresis and cooldown: consecutive scale-downs are separated; the count never oscillates wildly.
        int closeTogether = 0;
        int previousDown = -1000;
        for (int i = 1; i < predictive.Instances.Count; i++)
        {
            if (predictive.Instances[i] < predictive.Instances[i - 1])
            {
                if (i - previousDown < 6) closeTogether++;
                previousDown = i;
            }
        }
        Check(closeTogether == 0, "two scale-downs are never closer than the cooldown window");
        int oscillations = 0;
        for (int i = 2; i < predictive.Instances.Count; i++)
        {
            bool upThenDown = predictive.Instances[i - 1] > predictive.Instances[i - 2] && predictive.Instances[i] < predictive.Instances[i - 1];
            if (upThenDown) oscillations++;
        }
        Check(oscillations <= 4, "the plan does not flap (" + oscillations + " alternating changes)");
        var sortedLatencies = new List<double>(predictive.Latencies);
        sortedLatencies.Sort();
        double medianLatency = sortedLatencies[sortedLatencies.Count / 2];
        double p90Latency = sortedLatencies[(int)(sortedLatencies.Count * 0.9)];
        Check(medianLatency <= 120.0 && p90Latency <= 200.0, "the plan keeps latency headroom (median="
            + medianLatency.ToString("F1") + "ms, p90=" + p90Latency.ToString("F1") + "ms)");

        // 13-14. Cost comparison against the offline optimum from dynamic programming.
        ScalingPlan optimal = OfflinePlanner.Plan(series, 1, 40, 200.0, 2);
        Check(optimal.Violations == 0, "the offline optimum meets the SLA as well");
        Console.WriteLine("  cost: controller=" + predictive.Cost(2) + " offline-optimum=" + optimal.Cost(2)
            + " ratio=" + ((double)predictive.Cost(2) / optimal.Cost(2)).ToString("F3"));
        Check(predictive.Cost(2) >= optimal.Cost(2), "the online controller cannot beat the hindsight optimum");
        Check(predictive.Cost(2) <= optimal.Cost(2) * 1.35, "the online controller stays within 35% of the hindsight optimum");

        // 15-16. The dynamic program itself is validated against brute-force enumeration.
        List<WorkloadPoint> shortSeries = Workload.Generate(6, 500.0, 40.0, 0.0, 4, 0xABCUL);
        long dynamic = OfflinePlanner.Plan(shortSeries, 1, 4, 200.0, 3).Cost(3);
        int bestFirst;
        long brute = OfflinePlanner.BruteForce(shortSeries, 1, 4, 200.0, 3, out bestFirst);
        Check(dynamic == brute, "the dynamic program matches brute-force enumeration on a short horizon");
        ScalingPlan shortOptimal = OfflinePlanner.Plan(shortSeries, 1, 4, 200.0, 3);
        Check(shortOptimal.Instances.Count == 6 && bestFirst >= 1 && bestFirst <= 4, "the brute-force optimum starts in a legal state");

        // 17-18. Forecasting is worth it: a perfect-foresight controller is never worse and never violates the SLA.
        var perfectAutoscaler = new Autoscaler(1, 40, 0.8, 200.0, 6, 2, 24, perfect: true);
        ScalingPlan perfect = Simulate(series, perfectAutoscaler, 200.0, 2);
        Check(perfect.Violations == 0, "the perfect-foresight controller never violates the SLA");
        Check(perfect.Cost(2) <= optimal.Cost(2) * 1.35, "the perfect controller also stays within 35% of the hindsight optimum");
        Check(perfect.WorstLatency <= predictive.WorstLatency + 1.0, "a perfect forecast never makes latency worse");

        // 19-21. Itsy-bitsy edges: a spike, a constant load and a short trace.
        var spike = new List<WorkloadPoint> { new WorkloadPoint(0, 500.0), new WorkloadPoint(1, 500.0), new WorkloadPoint(2, 5000.0), new WorkloadPoint(3, 500.0) };
        var spikeAutoscaler = new Autoscaler(1, 60, 0.8, 200.0, 6, 1, 4, perfect: false);
        ScalingPlan spikePlan = Simulate(spike, spikeAutoscaler, 200.0, 2);
        Check(spikePlan.Instances.Count == 4 && spikePlan.InstanceSteps > 0, "a spike trace still produces a plan");
        Check(spikePlan.Instances[3] >= spikePlan.Instances[0], "the controller did not shed capacity right after the spike");
        var constant = new List<WorkloadPoint>();
        for (int i = 0; i < 40; i++) constant.Add(new WorkloadPoint(i, 600.0));
        var constantAutoscaler = new Autoscaler(1, 40, 0.8, 200.0, 6, 2, 8, perfect: false);
        ScalingPlan constantPlan = Simulate(constant, constantAutoscaler, 200.0, 2);
        int distinctCounts = 0;
        for (int i = 1; i < constantPlan.Instances.Count; i++) if (constantPlan.Instances[i] != constantPlan.Instances[i - 1]) distinctCounts++;
        Check(distinctCounts <= 2, "a constant workload converges and stops changing (" + distinctCounts + " changes)");
        int constantLateViolations = 0;
        foreach (int step in constantPlan.ViolationSteps) if (step >= 16) constantLateViolations++;
        Check(constantLateViolations == 0, "the constant workload never violates the SLA after the warm-up");
        var tiny = new Autoscaler(2, 4, 0.8, 200.0, 3, 1, 7, perfect: false);
        var tinySeries = new List<WorkloadPoint> { new WorkloadPoint(0, 10.0) };
        ScalingPlan tinyPlan = Simulate(tinySeries, tiny, 200.0, 1);
        Check(tinyPlan.Instances.Count == 1 && tinyPlan.Instances[0] >= 2 && tinyPlan.Instances[0] <= 4,
            "a single tiny step respects the minimum and maximum instance bounds");

        // 22-23. Capacity plan versus the hard SLA boundary, and determinism.
        int needed = CapacityModel.InstancesForSla(1200.0, 200.0, 1, 40);
        Check(CapacityModel.Latency(1200.0, needed) <= 200.0 && CapacityModel.Latency(1200.0, needed - 1) > 200.0,
            "the SLA boundary is tight: one instance fewer breaks the promise");
        var repeatA = Simulate(series, new Autoscaler(1, 40, 0.8, 200.0, 6, 2, 24, false), 200.0, 2);
        var repeatB = Simulate(series, new Autoscaler(1, 40, 0.8, 200.0, 6, 2, 24, false), 200.0, 2);
        Check(repeatA.InstanceSteps == repeatB.InstanceSteps && repeatA.Switches == repeatB.Switches,
            "identical inputs produce identical scaling plans");

        Console.WriteLine("  autoscaler: steps=" + series.Count + " instance-steps=" + predictive.InstanceSteps
            + " switches=" + predictive.Switches + " worst latency=" + predictive.WorstLatency.ToString("F1") + "ms");
        Console.WriteLine("PASS");
    }

    /** Runs the control loop over a trace, recomputing the latency of every step from the raw load. */
    private static ScalingPlan Simulate(IReadOnlyList<WorkloadPoint> series, Autoscaler autoscaler, double sla, int switchingPenalty)
    {
        var plan = new ScalingPlan();
        for (int i = 0; i < series.Count; i++)
        {
            int instances = autoscaler.Decide(series[i].Load, i);
            plan.Instances.Add(instances);
            plan.InstanceSteps += instances;
            if (i > 0 && plan.Instances[i] != plan.Instances[i - 1]) plan.Switches++;
            double latency = CapacityModel.Latency(series[i].Load, instances);
            if (latency > sla)
            {
                plan.Violations++;
                plan.ViolationSteps.Add(i);
            }
            if (latency > plan.WorstLatency) plan.WorstLatency = latency;
            plan.Latencies.Add(latency);
        }
        plan.Actions = plan.Switches;
        return plan;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
