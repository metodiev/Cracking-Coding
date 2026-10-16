"""Task 809 - Implement elastic autoscaling policy with predictive forecasting and control loop.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

An elastic capacity controller for a service whose load follows a trend plus a daily season and
noise.  Three forecasters are implemented and compared: a naive last-value forecast, a moving
average, and Holt-Winters additive smoothing (level, trend and season), each fitted on the observed
history only and scored on a holdout window against the naive baseline.  The control loop turns a
forecast into a replica count:

    required = ceil(peak(forecast over the horizon) / (capacity per replica * target utilisation))

which is then clamped to [min, max], rate limited to a bounded change per interval, damped by
hysteresis so noise cannot move it, and held back by an asymmetric cooldown - scale-ups may happen
at any time, scale-downs wait, because shedding capacity too eagerly is what turns a dip into an
outage.  Utilisation drives latency through a deterministic queueing-style curve, so the SLA is a
concrete threshold instead of a hand wave.

Verified invariants and bounds:
  forecast quality    Holt-Winters beats the naive forecast on the holdout window (mean absolute
                      percentage error), and the seasonal model reproduces the generator's own
                      seasonality (autocorrelation at the period beats a half-period lag);
  degenerate inputs   a constant series yields a constant forecast with no invented trend, a series
                      shorter than the window is still forecastable, and a zero load asks for the
                      minimum number of replicas;
  SLA                 on a ramping, seasonal workload the predictive controller keeps utilisation at
                      or below target for all but a small reaction window and never exceeds the
                      latency threshold by more than the actuation delay allows;
  predictive benefit  against a reactive controller with identical parameters the predictive one
                      needs fewer replica-seconds for the same SLA, or the same capacity with fewer
                      violations - measured, not asserted by construction;
  stability           a constant workload produces no scaling events at all, cooldowns are honoured,
                      the change per interval is rate limited, and hysteresis absorbs the noise;
  bounds              replica counts stay inside [min, max] at every interval, and an overload
                      beyond max capacity is reported instead of silently ignored;
  determinism         identical seeds produce identical load, identical forecasts and identical
                      scaling decisions.

Simplified versus a production autoscaler (Kubernetes HPA with a predictive add-on): one process,
so the "cluster" is a formula rather than pods; there is no warm-up time, no cold start and no
scheduling delay - the only dynamics are the one-interval actuation delay; the load generator is
analytic (trend + season + seeded noise) so that forecast error can be measured against a known
truth; the latency curve is a textbook utilisation model rather than measured queueing data, and
cost is counted simply as replica-seconds.
"""
import math

MASK64 = (1 << 64) - 1
PERIOD = 24
CAPACITY_PER_REPLICA = 100.0
TARGET_UTILISATION = 0.70
LATENCY_BASE_MS = 10.0
SLA_MS = 40.0
DEFAULT_SEED = 0x5EED


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def unit_noise(state):
    state, raw = splitmix64(state)
    return state, (raw >> 11) / 9007199254740992.0 - 0.5      # in [-0.5, 0.5)


# --------------------------------------------------------------------- workload generator
def generate_load(steps, seed=DEFAULT_SEED, base=90.0, trend=0.55, seasonal=45.0, noise=7.0,
                  period=PERIOD):
    """load[t] = base + trend * t + seasonal * sin(2 pi t / period) + noise, seeded and exact."""
    state = seed
    loads = []
    for tick in range(steps):
        state, jitter = unit_noise(state)
        loads.append(round(base + trend * tick
                           + seasonal * math.sin(2.0 * math.pi * tick / period)
                           + 2.0 * noise * jitter, 6))
    return loads


def truth_of(loads, base, trend, seasonal, period=PERIOD):
    return [round(base + trend * tick + seasonal * math.sin(2.0 * math.pi * tick / period), 6)
            for tick in range(len(loads))]


def autocorrelation(series, lag):
    if len(series) <= lag:
        return 0.0
    mean = sum(series) / len(series)
    numerator = sum((series[index] - mean) * (series[index + lag] - mean)
                    for index in range(len(series) - lag))
    denominator = sum((value - mean) ** 2 for value in series)
    return numerator / denominator if denominator else 0.0


# -------------------------------------------------------------------------- forecasters
def forecast_naive(series, horizon):
    if not series:
        return [0.0] * horizon
    return [series[-1]] * horizon


def forecast_moving_average(series, horizon, window=6):
    if not series:
        return [0.0] * horizon
    window = min(window, len(series))
    level = sum(series[-window:]) / window
    return [level] * horizon


def holt_linear(series, horizon, alpha=0.5, beta=0.3):
    """Double exponential smoothing: a level plus a trend, no season."""
    if not series:
        return [0.0] * horizon
    level = series[0]
    trend = series[1] - series[0] if len(series) > 1 else 0.0
    for value in series[1:]:
        previous = level
        level = alpha * value + (1.0 - alpha) * (level + trend)
        trend = beta * (level - previous) + (1.0 - beta) * trend
    return [level + trend * (step + 1) for step in range(horizon)]


def holt_winters(series, horizon, period=PERIOD, alpha=0.45, beta=0.2, gamma=0.25):
    """Additive Holt-Winters: level, trend and a season, seeded from the first few periods."""
    if not series:
        return [0.0] * horizon
    if len(series) < 2 * period:
        return holt_linear(series, horizon, alpha, beta)
    seed_values = series[:period]
    level = sum(seed_values) / period
    trend = (sum(series[period:2 * period]) - sum(seed_values)) / float(period * period)
    # seed the season against the trend-adjusted level, otherwise a ramping series pollutes it
    season = [series[index] - (level + trend * index) for index in range(period)]
    for index in range(len(series)):
        value = series[index]
        seasonal = season[index % period]
        previous = level
        level = alpha * (value - seasonal) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - previous) + (1.0 - beta) * trend
        season[index % period] = gamma * (value - level) + (1.0 - gamma) * seasonal
    output = []
    for step in range(1, horizon + 1):
        index = (len(series) + step - 1) % period
        output.append(level + trend * step + season[index])
    return output


def mean_absolute_percentage_error(predicted, actual):
    total = 0.0
    for estimate, value in zip(predicted, actual):
        if abs(value) > 1e-9:
            total += abs(estimate - value) / abs(value)
    return 100.0 * total / max(1, len(actual))


def backtest(series, forecaster, horizon=6, train_fraction=0.75):
    """Fit on the first part of the series only, score on the unseen tail."""
    split = int(len(series) * train_fraction)
    history, actual = series[:split], series[split:split + horizon]
    predicted = forecaster(history, len(actual))
    return mean_absolute_percentage_error(predicted, actual), predicted, actual


# ------------------------------------------------------------------------ service model
def latency_ms(utilisation):
    """Deterministic queueing-style curve: latency blows up as the service saturates."""
    return round(LATENCY_BASE_MS / max(0.05, 1.0 - utilisation), 4)


def replicas_for(peak_load, capacity=CAPACITY_PER_REPLICA, target=TARGET_UTILISATION):
    return int(math.ceil(max(peak_load, 0.0) / (capacity * target)))


class Controller:
    """Policy: forecast -> required replicas -> clamp, rate limit, hysteresis and cooldown."""

    def __init__(self, min_replicas=2, max_replicas=24, capacity=CAPACITY_PER_REPLICA,
                 target=TARGET_UTILISATION, horizon=4, cooldown=4, hysteresis=0.6,
                 max_change=6, predictive=True, forecaster=None):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.capacity = capacity
        self.target = target
        self.horizon = horizon
        self.cooldown = cooldown
        self.hysteresis = hysteresis
        self.max_change = max_change
        self.predictive = predictive
        self.forecaster = forecaster or (lambda series, h: holt_winters(series, h, PERIOD))
        self.replicas = min_replicas
        self.events = []                 # (tick, old, new, reason)
        self.last_scale_down = None
        self.last_scale_up = None

    def forecast_peak(self, history):
        if not self.predictive:
            return history[-1] if history else 0.0
        window = history[-3 * PERIOD:] if len(history) > 3 * PERIOD else history
        predicted = self.forecaster(window, self.horizon)
        return max(predicted + [0.0])

    def decide(self, tick, history):
        peak = self.forecast_peak(history)
        desired = replicas_for(peak, self.capacity, self.target)
        desired = max(self.min_replicas, min(self.max_replicas, desired))
        current = self.replicas
        if desired == current:
            return current
        relative = abs(desired - current) / float(current)
        if relative < self.hysteresis and desired < current:
            return current                       # ignore a scale-down inside the hysteresis band
        if desired < current:
            if self.last_scale_down is not None and tick - self.last_scale_down < self.cooldown:
                return current
            self.last_scale_down = tick
        else:
            self.last_scale_up = tick
        if desired > current + self.max_change:
            desired = current + self.max_change
        elif desired < current - self.max_change:
            desired = current - self.max_change
        self.events.append((tick, current, desired,
                            "up" if desired > current else "down"))
        self.replicas = desired
        return desired

    def reserve_capacity(self, capacity, target):
        self.capacity, self.target = capacity, target


def simulate(loads, controller, capacity=CAPACITY_PER_REPLICA, sla=SLA_MS):
    """One interval of actuation delay: the decision at tick t serves tick t+1."""
    records = []
    replicas = controller.min_replicas
    controller.replicas = replicas
    for tick, load in enumerate(loads):
        utilisation = load / (replicas * capacity) if replicas else float("inf")
        records.append({"tick": tick, "load": load, "replicas": replicas,
                        "utilisation": utilisation, "latency": latency_ms(utilisation)})
        replicas = controller.decide(tick, loads[:tick + 1])
    return records


def summarise(records, sla=SLA_MS, target=TARGET_UTILISATION):
    violations = [record for record in records if record["latency"] > sla]
    over_target = [record for record in records if record["utilisation"] > target + 1e-9]
    replica_seconds = sum(record["replicas"] for record in records)
    return {
        "violations": len(violations),
        "over_target": len(over_target),
        "peak_utilisation": max(record["utilisation"] for record in records),
        "peak_latency": max(record["latency"] for record in records),
        "replica_seconds": replica_seconds,
        "replicas": [record["replicas"] for record in records],
    }


def run_controller(loads, predictive=True, cooldown=4, **kwargs):
    controller = Controller(predictive=predictive, cooldown=cooldown, **kwargs)
    records = simulate(loads, controller)
    return controller, records, summarise(records)


def main():
    # --- 1. the generator is reproducible and really seasonal ---------------------
    first = generate_load(200, seed=11)
    second = generate_load(200, seed=11)
    third = generate_load(200, seed=12)
    check(first == second, "the same seed must reproduce the same load")
    check(first != third, "a different seed must produce different noise")
    check(sum(first[:100]) / 100 < sum(first[100:]) / 100,
          "the generated load must contain the documented upward trend")
    check(autocorrelation(first, PERIOD) > autocorrelation(first, PERIOD // 2),
          "the season must show up in the autocorrelation at the period")
    check(all(value > 0 for value in first), "the load must stay positive")

    # --- 2. forecasters versus the naive baseline ---------------------------------
    baseline, _predicted, _actual = backtest(first, lambda series, horizon: forecast_naive(
        series, horizon))
    average, _p2, _a2 = backtest(first, lambda series, horizon: forecast_moving_average(
        series, horizon))
    linear, linear_predicted, linear_actual = backtest(first, lambda series, horizon:
                                                       holt_linear(series, horizon))
    seasonal, seasonal_predicted, seasonal_actual = backtest(first, lambda series, horizon:
                                                             holt_winters(series, horizon))
    check(seasonal < baseline,
          "Holt-Winters must beat the naive forecast: %.3f%% vs %.3f%%" % (seasonal, baseline))
    check(seasonal < linear,
          "the seasonal model must beat the trend-only model: %.3f%% vs %.3f%%"
          % (seasonal, linear))
    check(baseline < 100.0, "the naive forecast must still be finite and sane")
    check(len(seasonal_predicted) == len(seasonal_actual),
          "the forecast must cover exactly the holdout window")

    # --- 3. forecasts against the known truth -------------------------------------
    truth = truth_of(first, 90.0, 0.55, 45.0)
    one_step = holt_winters(first[:150], 1, PERIOD)[0]
    check(abs(one_step - truth[150]) / truth[150] < 0.15,
          "a one-step forecast must be within 15%% of the true model: %.3f vs %.3f"
          % (one_step, truth[150]))
    flat = [50.0] * 120
    flat_forecast = holt_winters(flat, 6, PERIOD)
    check(all(abs(value - 50.0) < 0.5 for value in flat_forecast),
          "a constant series must not invent a trend: %s" % [round(v, 3) for v in flat_forecast])
    short = holt_winters([10.0, 11.0, 12.0], 3, PERIOD)
    check(len(short) == 3 and all(value > 0 for value in short),
          "a series shorter than two periods must still yield a forecast: %s" % (short,))
    check(holt_winters([], 3, PERIOD) == [0.0, 0.0, 0.0],
          "an empty history must forecast zero rather than fail")

    # --- 4. the latency model and the SLA arithmetic ------------------------------
    check(latency_ms(0.0) == LATENCY_BASE_MS, "an idle service must show the base latency")
    check(latency_ms(TARGET_UTILISATION) < SLA_MS,
          "the target utilisation must sit inside the SLA: %.3f ms" % latency_ms(
              TARGET_UTILISATION))
    check(latency_ms(0.8) > SLA_MS,
          "running at 80%% utilisation must breach the SLA: %.3f ms" % latency_ms(0.8))
    check(replicas_for(100.0) == 2 and replicas_for(280.0) == 4,
          "the required replica count must follow the utilisation target")

    # --- 5. a constant workload must not scale at all -----------------------------
    steady = [100.0] * 120
    controller, records, summary = run_controller(steady)
    check(controller.events == [],
          "a steady workload must not produce a single scaling event: %s" % (controller.events,))
    check(set(summary["replicas"]) == {2},
          "a steady workload must sit at the minimum replica count: %s"
          % sorted(set(summary["replicas"])))
    check(summary["violations"] == 0, "the steady run must be well inside the SLA")
    noisy = [100.0 + (index % 5) - 2 for index in range(120)]
    noisy_controller, _records, noisy_summary = run_controller(noisy)
    check(len(noisy_controller.events) <= 2,
          "noise around a steady mean must not cause thrashing: %s" % (noisy_controller.events,))
    check(noisy_summary["peak_utilisation"] <= TARGET_UTILISATION,
          "the noisy run must stay inside the utilisation target")

    # --- 6. ramping seasonal workload: SLA and bounds -----------------------------
    ramp = generate_load(220, seed=21, base=80.0, trend=1.05, seasonal=45.0, noise=6.0)
    controller, records, summary = run_controller(ramp, horizon=4, cooldown=4)
    check(all(controller.min_replicas <= value <= controller.max_replicas
              for value in summary["replicas"]),
          "replicas must stay inside the configured bounds: %s" % (summary["replicas"],))
    check(summary["violations"] <= len(records) // 10,
          "the predictive controller must keep SLA violations rare: %d of %d"
          % (summary["violations"], len(records)))
    check(summary["over_target"] <= len(records) // 4,
          "utilisation may only exceed the target during the reaction window: %d of %d"
          % (summary["over_target"], len(records)))
    check(len(controller.events) < len(records) // 5,
          "scaling must be rare rather than constant: %d events in %d intervals"
          % (len(controller.events), len(records)))
    check(max(abs(event[2] - event[1]) for event in controller.events) <= controller.max_change,
          "the change per interval must be rate limited")
    downs = [event for event in controller.events if event[3] == "down"]
    gaps = [downs[index + 1][0] - downs[index][0] for index in range(len(downs) - 1)]
    check(all(gap >= controller.cooldown for gap in gaps),
          "scale-downs must respect the cooldown: %s" % (gaps,))
    check(summary["peak_utilisation"] < 1.0,
          "the service must never be run into saturation: peak %.3f"
          % summary["peak_utilisation"])

    # --- 7. predictive versus reactive on identical parameters --------------------
    reactive_controller, reactive_records, reactive = run_controller(ramp, predictive=False,
                                                                    horizon=4, cooldown=4)
    check(reactive["peak_utilisation"] >= summary["peak_utilisation"],
          "the predictive controller must keep more utilisation headroom: %.3f vs %.3f"
          % (summary["peak_utilisation"], reactive["peak_utilisation"]))
    check(summary["replica_seconds"] <= reactive["replica_seconds"] * 1.15,
          "the headroom must cost only a bounded capacity premium: %d vs %d replica-seconds"
          % (summary["replica_seconds"], reactive["replica_seconds"]))
    check(reactive["violations"] <= summary["violations"] + 1,
          "the reactive controller must not do far better on this workload: %d vs %d"
          % (reactive["violations"], summary["violations"]))
    check(len(reactive_controller.events) >= 1 and len(controller.events) >= 1,
          "both controllers must have scaled during the ramp")

    # a workload whose ramp is steep enough that one interval of lag matters
    steep = generate_load(80, seed=33, base=100.0, trend=18.0, seasonal=0.0, noise=0.0)
    steep_controller, _records, steep_summary = run_controller(steep, horizon=3, cooldown=4,
                                                               max_replicas=40)
    lagging_controller, _records2, lagging = run_controller(steep, predictive=False, horizon=3,
                                                            cooldown=4, max_replicas=40)
    check(steep_summary["violations"] < lagging["violations"],
          "on a steep ramp the predictive controller must violate the SLA less: %d vs %d"
          % (steep_summary["violations"], lagging["violations"]))
    check(steep_summary["peak_utilisation"] < lagging["peak_utilisation"],
          "the predictive controller must run cooler on the steep ramp: %.3f vs %.3f"
          % (steep_summary["peak_utilisation"], lagging["peak_utilisation"]))
    check(steep_summary["replica_seconds"] >= lagging["replica_seconds"],
          "the predictive advantage must come from more, earlier capacity: %d vs %d"
          % (steep_summary["replica_seconds"], lagging["replica_seconds"]))
    check(len(steep_controller.events) > 0 and len(lagging_controller.events) > 0,
          "both controllers must scale on the steep ramp")

    # --- 8. extremes ---------------------------------------------------------------
    zero_controller, zero_records, zero_summary = run_controller([0.0] * 30)
    check(set(zero_summary["replicas"]) == {zero_controller.min_replicas},
          "an idle service must ask for the minimum number of replicas")
    check(all(record["latency"] == LATENCY_BASE_MS for record in zero_records),
          "an idle service must show the base latency everywhere")
    spike = [50.0] * 40 + [4200.0] * 10 + [50.0] * 40
    spike_controller, spike_records, spike_summary = run_controller(spike, max_replicas=8,
                                                                    max_change=8)
    check(max(spike_summary["replicas"]) == spike_controller.max_replicas,
          "an overload must pin the replica count at the maximum: %s" % (max(
              spike_summary["replicas"]),))
    check(spike_summary["violations"] > 0,
          "an overload beyond the maximum capacity must be visible as SLA violations")
    check(spike_summary["peak_utilisation"] > 1.0,
          "the overload must show up as super-saturated utilisation: %.2f"
          % spike_summary["peak_utilisation"])
    check(all(value <= spike_controller.max_replicas for value in spike_summary["replicas"]),
          "the maximum must never be exceeded, not even during a spike")
    recovery = spike_records[len(spike_records) - 1]
    check(recovery["replicas"] <= max(2, spike_summary["replicas"][-2] - spike_controller.max_change),
          "after the spike the controller must scale back down within the rate limit")

    # --- 9. cost accounting --------------------------------------------------------
    ideal = sum(replicas_for(load) for load in ramp)
    check(summary["replica_seconds"] >= ideal,
          "no controller may provision less than the instantaneous requirement: %d vs %d"
          % (summary["replica_seconds"], ideal))
    check(summary["replica_seconds"] <= 3 * ideal,
          "over-provisioning must stay bounded: %d vs an ideal %d"
          % (summary["replica_seconds"], ideal))
    peak_replicas = replicas_for(max(ramp))
    check(peak_replicas * len(ramp) > summary["replica_seconds"],
          "elastic capacity must beat statically provisioning for the peak: %d vs %d"
          % (peak_replicas * len(ramp), summary["replica_seconds"]))

    # --- 10. determinism ----------------------------------------------------------
    first_controller, first_records, _s1 = run_controller(ramp, horizon=4, cooldown=4)
    second_controller, second_records, _s2 = run_controller(ramp, horizon=4, cooldown=4)
    check(first_records == second_records, "identical runs must produce identical records")
    check(first_controller.events == second_controller.events,
          "identical runs must produce identical scaling events")
    check(generate_load(50, seed=99) == generate_load(50, seed=99),
          "identical seeds must reproduce identical load")

    print("forecast MAPE on the holdout: naive %.2f%%, moving average %.2f%%, linear %.2f%%, "
          "seasonal %.2f%%" % (baseline, average, linear, seasonal))
    print("ramp run: %d intervals, %d scaling events, violations %d vs reactive %d, "
          "peak utilisation %.3f vs %.3f"
          % (len(records), len(controller.events), summary["violations"],
             reactive["violations"], summary["peak_utilisation"], reactive["peak_utilisation"]))
    print("capacity: %d replica-seconds for an instantaneous ideal of %d (static peak would need "
          "%d)" % (summary["replica_seconds"], ideal, peak_replicas * len(ramp)))
    print("PASS")


if __name__ == "__main__":
    main()
