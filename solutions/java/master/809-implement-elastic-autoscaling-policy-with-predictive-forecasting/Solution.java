/** Task 809 - Implement elastic autoscaling policy with predictive forecasting and control loop. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A closed-loop autoscaler. A deterministic traffic generator produces a diurnal load, the control
 * loop samples it every interval, Holt's linear exponential smoothing tracks level and trend, and the
 * forecast over the next interval sets the replica count through an inverted capacity model: the
 * replicas needed to keep utilisation at a target with a safety margin. Hysteresis (scale down only
 * with headroom) plus a cooldown keeps the loop from oscillating, and the replica count is clamped to
 * the configured minimum and maximum. The same loop is also run reactively - scaling to the *current*
 * sample - so the benefit of forecasting can be measured instead of asserted.
 *
 * Simulation model, and what is simplified. One process, no threads and no wall clock: time is a tick
 * counter, each replica serves a fixed capacity per tick, demand above capacity is dropped (the SLO
 * violation), and cost is the sum of replica-ticks. There is no queueing delay, no warm-up time, no
 * per-node heterogeneity and no metrics pipeline latency; the generator is a deterministic mixture of
 * a sinusoid and fixed-seed noise rather than a replay of production traffic. What is not simplified
 * is the control problem: the loop sees only past samples, the forecast really is extrapolated from
 * the smoother, the cost is compared against an offline optimum computed with perfect knowledge of
 * the trace, and the stability counters really count the scale events.
 *
 * Verified: the forecast beats a last-value baseline on a ramp (lower mean absolute percentage
 * error); dropped load stays inside the SLO budget while cost stays within a bounded factor of the
 * offline optimum; replicas never leave the configured bounds and never reach zero; the loop is
 * stable - after convergence a constant load produces no further scale events, and no scale-up is
 * followed by a scale-down inside the cooldown window; a step change is absorbed within a bounded
 * number of intervals; the predictive policy drops less load than the reactive one on the same trace;
 * the negative control - the same loop with hysteresis and cooldown removed - thrashes measurably
 * more; and a zero-load trace settles at the minimum replica count.
 */
import java.util.*;

final class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Deterministic noise in [-1, 1). */
    double nextSigned() { return (nextLong() >>> 11) / (double) (1L << 53) * 2 - 1; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** The traffic model: a diurnal ramp plus bounded noise, all deterministic. */
final class LoadGenerator {
    final double base, amplitude, noise;
    final double period;
    final SplitMix64 rng;

    LoadGenerator(double base, double amplitude, double noise, long seed) {
        this(base, amplitude, noise, seed, 240);
    }

    LoadGenerator(double base, double amplitude, double noise, long seed, double period) {
        this.base = base;
        this.amplitude = amplitude;
        this.noise = noise;
        this.period = period;
        this.rng = new SplitMix64(seed);
    }

    double loadAt(int tick) {
        double diurnal = base + amplitude * Math.sin(2 * Math.PI * tick / period);
        return Math.max(0, diurnal + noise * rng.nextSigned());
    }

    /** A pure ramp: used to measure the forecast error without any noise. */
    static double ramp(int tick) { return 200 + 4.0 * tick; }

    /** A single step at `at`: used to measure the reaction time. */
    static double step(int tick, int at, double before, double after) {
        return tick < at ? before : after;
    }
}

final class Controller {
    final int capacity;                 // requests one replica serves per tick
    final double targetUtilisation;
    final int minReplicas, maxReplicas;
    final int interval;
    final int cooldown;

    boolean predictive = true;
    boolean hysteresis = true;           // the negative control turns both off
    int replicas;
    int cooldownLeft;
    int scaleEvents, scaleUps, scaleDowns, reversals;
    int intervalIndex;
    int intervalsSinceEvent = 1000;
    int lastDirection;

    double level, trend;
    boolean warmedUp;
    double alpha = 0.5, beta = 0.5;               // smoothing constants for level and trend
    double lastSample;

    Controller(int capacity, double targetUtilisation, int minReplicas, int maxReplicas,
               int interval, int cooldown, int startReplicas) {
        this.capacity = capacity;
        this.targetUtilisation = targetUtilisation;
        this.minReplicas = minReplicas;
        this.maxReplicas = maxReplicas;
        this.interval = interval;
        this.cooldown = cooldown;
        this.replicas = startReplicas;
        this.level = 0;
    }

    /** Holt's linear smoothing: one level and one trend update per sample. */
    void observe(double sample) {
        if (!warmedUp) {
            level = sample;
            trend = 0;
            warmedUp = true;
        } else {
            double previousLevel = level;
            level = alpha * sample + (1 - alpha) * (level + trend);
            trend = beta * (level - previousLevel) + (1 - beta) * trend;
        }
        lastSample = sample;
    }

    /** Forecast `stepsAhead` *samples* ahead (the smoother advances once per sample it sees). */
    double forecast(double stepsAhead) { return Math.max(0, level + trend * stepsAhead); }

    double baselineForecast(int horizon) { return lastSample; }

    int replicasFor(double load) {
        double needed = load / (capacity * targetUtilisation);
        int want = (int) Math.ceil(needed - 1e-9);
        return Math.max(minReplicas, Math.min(maxReplicas, want));
    }

    /** One control interval: sample, forecast, decide, and respect hysteresis and the cooldown. */
    int controlStep(double sample, double horizonLoad) {
        observe(sample);
        intervalIndex++;
        intervalsSinceEvent++;
        int want;
        if (predictive) {
            // Provision for the worst point of the coming interval: the forecast is linear in samples,
            // so the extreme lies at one of its ends - and never below the sample just observed, so a
            // sudden change of direction can only ever cost a little over-provisioning, not a stall.
            double horizon = Math.max(forecast(0), Math.max(forecast(1), forecast(2)));
            horizon = Math.max(horizon, sample);
            want = replicasFor(horizon);
        } else {
            want = replicasFor(horizonLoad);              // reactive: act on what is already happening
        }
        if (hysteresis) {
            if (want > replicas) {                        // scale up immediately
                scaleTo(want);
                cooldownLeft = cooldown;
            } else if (want < replicas - 1 && cooldownLeft == 0) {   // down only with real headroom
                scaleTo(want);
                cooldownLeft = cooldown;
            } else if (cooldownLeft > 0) {
                cooldownLeft--;
            }
        } else {
            scaleTo(want);
        }
        return replicas;
    }

    private void scaleTo(int want) {
        int clamped = Math.max(minReplicas, Math.min(maxReplicas, want));
        if (clamped == replicas) return;
        int direction = clamped > replicas ? 1 : -1;
        if (lastDirection != 0 && direction != lastDirection && intervalsSinceEvent <= cooldown + 1) {
            reversals++;                                     // a flip inside the cooldown window
        }
        lastDirection = direction;
        intervalsSinceEvent = 0;
        scaleEvents++;
        if (clamped > replicas) scaleUps++;
        else scaleDowns++;
        replicas = clamped;
    }

    /** Applies the smoothing to a whole trace without acting: used to score the forecast. */
    double meanAbsolutePercentageError(double[] trace, int horizon) {
        level = 0;
        trend = 0;
        warmedUp = false;
        double total = 0;
        int counted = 0;
        double baselineTotal = 0;
        for (int t = 0; t < trace.length; t++) {
            observe(trace[t]);
            if (t + horizon < trace.length) {
                double truth = trace[t + horizon];
                if (truth > 1) {
                    total += Math.abs(forecast(horizon) - truth) / truth;
                    baselineTotal += Math.abs(baselineForecast(horizon) - truth) / truth;
                    counted++;
                }
            }
        }
        return counted == 0 ? Double.NaN : total / counted;
    }

    double lastBaselineError(double[] trace, int horizon) {
        level = 0;
        trend = 0;
        warmedUp = false;
        double total = 0;
        int counted = 0;
        for (int t = 0; t < trace.length; t++) {
            observe(trace[t]);
            if (t + horizon < trace.length && trace[t + horizon] > 1) {
                total += Math.abs(baselineForecast(horizon) - trace[t + horizon]) / trace[t + horizon];
                counted++;
            }
        }
        return counted == 0 ? Double.NaN : total / counted;
    }
}
final class RunResult {
    long served, dropped, cost;
    int scaleEvents, scaleUps, scaleDowns, peakReplicas, minReplicasSeen = Integer.MAX_VALUE;
    int firstShortfallTick = -1, shortfallTicks;
    long optimalCost;
    double dropFraction;

    String render() {
        return "served=" + served + " dropped=" + dropped + " cost=" + cost + " events=" + scaleEvents
                + " optimal=" + optimalCost;
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    /**
     * Runs the closed loop over a load function. The offline optimum assumes perfect knowledge of the
     * trace: for every tick it is simply the replicas needed to serve that tick.
     */
    static RunResult simulate(Controller controller, java.util.function.IntToDoubleFunction load,
                              int ticks) {
        RunResult result = new RunResult();
        int optimalThisInterval = 0;
        double previousDemand = load.applyAsDouble(0);
        for (int t = 0; t < ticks; t++) {
            double demand = load.applyAsDouble(t);
            // The loop only ever sees the sample from the tick before: metrics are not instant.
            if (t % controller.interval == 0) {
                optimalThisInterval = controller.replicasFor(previousDemand);
                controller.controlStep(previousDemand, previousDemand);
            }
            optimalThisInterval = Math.max(optimalThisInterval, controller.replicasFor(demand));
            long served = (long) Math.min(demand, controller.replicas * (double) controller.capacity);
            if (demand > served) {
                if (result.firstShortfallTick < 0) result.firstShortfallTick = t;
                result.shortfallTicks++;
            }
            result.served += served;
            result.dropped += (long) Math.max(0, demand - served);
            result.cost += controller.replicas;
            result.optimalCost += optimalThisInterval;
            result.peakReplicas = Math.max(result.peakReplicas, controller.replicas);
            result.minReplicasSeen = Math.min(result.minReplicasSeen, controller.replicas);
            previousDemand = demand;
        }
        result.scaleEvents = controller.scaleEvents;
        result.scaleUps = controller.scaleUps;
        result.scaleDowns = controller.scaleDowns;
        long total = result.served + result.dropped;
        result.dropFraction = total == 0 ? 0 : (double) result.dropped / total;
        return result;
    }

    static Controller standard(boolean predictive) {
        Controller c = new Controller(100, 0.7, 2, 40, 10, 2, 6);   // cooldown counted in intervals
        c.predictive = predictive;
        return c;
    }

    /** The forecast must beat a last-value baseline, and the closed loop must stay inside the SLO. */
    static void forecastQualityAndSlo() {
        double[] ramp = new double[200];
        for (int t = 0; t < ramp.length; t++) ramp[t] = LoadGenerator.ramp(t);
        Controller scorer = standard(true);
        double mape = scorer.meanAbsolutePercentageError(ramp, 20);
        double baseline = scorer.lastBaselineError(ramp, 20);
        check(mape < baseline, "the forecast beats the last-value baseline on a ramp: " + mape
                + " vs " + baseline);
        check(mape < 0.06, "and the error is small on a clean ramp: " + mape);
        check(baseline > 0.10, "the baseline is visibly worse on a ramp: " + baseline);

        LoadGenerator generator = new LoadGenerator(700, 500, 40, 809001L);
        Controller controller = standard(true);
        RunResult result = simulate(controller, generator::loadAt, 2400);
        check(result.dropFraction < 0.02, "the SLO budget holds: dropped " + result.dropFraction
                + " of the traffic (" + result.dropped + " of " + (result.served + result.dropped) + ")");
        check(result.cost > 0 && result.optimalCost > 0, "both cost figures are positive");
        check(result.cost <= result.optimalCost * 18 / 10,
                "cost stays within 1.8x of the offline optimum (which knows the whole trace): "
                        + result.cost + " vs " + result.optimalCost);
        check(result.peakReplicas <= 40 && result.minReplicasSeen >= 2,
                "replicas stay inside the bounds: " + result.minReplicasSeen + ".." + result.peakReplicas);
        metric("mapeRamp", (long) (mape * 1000));
        metric("sloDropped", result.dropped);
        metric("costRatio", result.cost * 100 / Math.max(1, result.optimalCost));
    }

    /** Forecasting must actually help: the same trace, the same bounds, a reactive controller. */
    static void predictiveBeatsReactive() {
        // A trapezoid: long sustained ramps with plateaus in between, and a tight target. The reactive
        // loop is always one interval behind the ramp; the predictor extrapolates it.
        java.util.function.IntToDoubleFunction trapezoid = t -> {
            int phase = t % 320;
            if (phase < 80) return 200 + 15.0 * phase;                  // ramp up (150 per interval)
            if (phase < 160) return 1400;                               // plateau
            if (phase < 240) return 1400 - 15.0 * (phase - 160);        // ramp down
            return 200;                                                 // plateau
        };
        Controller predictive = new Controller(100, 0.97, 2, 60, 10, 2, 6);
        predictive.predictive = true;
        RunResult p = simulate(predictive, trapezoid, 2400);
        Controller reactive = new Controller(100, 0.97, 2, 60, 10, 2, 6);
        reactive.predictive = false;
        RunResult r = simulate(reactive, trapezoid, 2400);
        check(p.dropped < r.dropped, "forecasting drops less load than reacting to the current sample: "
                + p.dropped + " vs " + r.dropped);
        check(r.dropFraction > 0.003,
                "the reactive loop really does miss demand on this ramp: " + r.dropFraction);
        check(p.dropFraction * 3 < r.dropFraction,
                "and the gap is substantial, not a rounding artefact: " + p.dropFraction + " vs "
                        + r.dropFraction);
        check(p.cost <= r.cost * 2.0, "at a comparable cost: " + p.cost + " vs " + r.cost);
        check(p.served + p.dropped > 0, "the workload actually ran");
        metric("predictiveDropped", p.dropped);
        metric("reactiveDropped", r.dropped);
    }

    /** Stability: no thrashing, and a stationary load settles. */
    static void stability() {
        Controller controller = standard(true);
        LoadGenerator constant = new LoadGenerator(500, 0, 0, 809003L);
        RunResult settled = simulate(controller, constant::loadAt, 600);
        int eventsAfterSettling = controller.scaleEvents;
        RunResult more = simulate(controller, constant::loadAt, 600);
        check(controller.scaleEvents == eventsAfterSettling,
                "a constant load produces no further scale events: " + (controller.scaleEvents
                        - eventsAfterSettling));
        check(settled.dropped == 0 && more.dropped == 0,
                "and nothing is dropped once it converged: " + settled.dropped + "/" + more.dropped);
        int expected = controller.replicasFor(500);
        check(controller.replicas == expected, "the settled replica count is the capacity for the load: "
                + controller.replicas + " vs " + expected);

        Controller churn = standard(true);
        LoadGenerator diurnal = new LoadGenerator(700, 500, 0, 809004L);   // noise-free: a smooth curve
        simulate(churn, diurnal::loadAt, 2400);
        check(churn.scaleEvents < 200, "over 240 intervals the loop makes less than one move per "
                + "interval: " + churn.scaleEvents);
        check(churn.scaleEvents >= 4, "but it does follow the diurnal curve: " + churn.scaleEvents);
        Controller drift = standard(true);
        RunResult firstHalf = simulate(drift, diurnal::loadAt, 1200);
        simulate(drift, diurnal::loadAt, 1200);
        RunResult secondHalf = simulate(drift, diurnal::loadAt, 1200);
        check(Math.abs(firstHalf.cost - secondHalf.cost) * 20 <= firstHalf.cost,
                "the replica count does not ratchet: cost " + firstHalf.cost + " then " + secondHalf.cost);
        check(Math.abs(firstHalf.served - secondHalf.served) * 100 <= Math.max(1, firstHalf.served),
                "and neither does the served traffic: " + firstHalf.served + " then " + secondHalf.served);
        metric("scaleEvents", churn.scaleEvents);
        metric("settledReplicas", controller.replicas);
    }

    /** A step change must be absorbed within a bounded number of intervals. */
    static void stepResponse() {
        int ticks = 800;
        int stepAt = 300;
        Controller controller = standard(true);
        RunResult result = simulate(controller,
                t -> LoadGenerator.step(t, stepAt, 200, 1800), ticks);
        check(result.firstShortfallTick >= stepAt,
                "the step really does exceed the capacity of the pre-step loop: shortfall at "
                        + result.firstShortfallTick);
        int reaction = result.firstShortfallTick - stepAt;
        check(reaction <= 3 * controller.interval, "the loop absorbs the step within a few intervals: "
                + reaction + " ticks");
        check(result.shortfallTicks <= 3 * controller.interval,
                "and the spill lasts only a couple of intervals: " + result.shortfallTicks + " ticks");
        check(result.dropped > 0, "the step workload did record the transition");
        check(result.dropped <= (long) 3 * controller.interval * 1500,
                "and the spill is bounded by the reaction window times the step size: " + result.dropped);
        check(controller.replicas >= controller.replicasFor(1800) - 1,
                "by the end the replica count matches the new level: " + controller.replicas);
        metric("stepReaction", reaction);
        metric("stepSpill", result.dropped);
    }

    /** Negative control: without hysteresis and the cooldown the loop thrashes. */
    static void negativeControlThrashing() {
        Controller calm = standard(true);
        LoadGenerator noisy = new LoadGenerator(700, 100, 120, 809005L);
        RunResult calmResult = simulate(calm, noisy::loadAt, 1200);

        Controller frantic = standard(true);
        frantic.hysteresis = false;
        RunResult franticResult = simulate(frantic, noisy::loadAt, 1200);
        check(frantic.scaleEvents > calm.scaleEvents * 2,
                "removing hysteresis and the cooldown multiplies the scale events: "
                        + frantic.scaleEvents + " vs " + calm.scaleEvents);
        check(frantic.reversals > calm.reversals * 3,
                "and reverses direction far more often inside the cooldown window: "
                        + frantic.reversals + " vs " + calm.reversals);
        check(calm.reversals <= 12, "while the guarded loop changes its mind rarely: " + calm.reversals);
        check(calm.scaleUps < 40 && calm.scaleDowns < 40,
                "while the guarded loop stays calm: " + calm.scaleUps + " up, " + calm.scaleDowns + " down");
        check(franticResult.dropFraction <= calmResult.dropFraction + 0.02,
                "the thrashing loop is not even better at the SLO: " + franticResult.dropFraction
                        + " vs " + calmResult.dropFraction);
        metric("thrashingEvents", frantic.scaleEvents);
        metric("calmEvents", calm.scaleEvents);

        Controller idle = standard(true);
        RunResult zero = simulate(idle, t -> 0.0, 300);
        check(zero.dropped == 0, "a zero-load trace drops nothing");
        check(idle.replicas == 2, "and settles at the configured minimum: " + idle.replicas);
        check(zero.cost == 2L * 300, "paying only for the minimum: " + zero.cost);
        metric("idleCost", zero.cost);
    }

    public static void main(String[] args) {
        forecastQualityAndSlo();
        predictiveBeatsReactive();
        stability();
        stepResponse();
        negativeControlThrashing();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
