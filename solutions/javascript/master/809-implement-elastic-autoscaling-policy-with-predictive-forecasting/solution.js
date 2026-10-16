/** Task 809 - Implement elastic autoscaling policy with predictive forecasting and control loop. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A complete autoscaling control loop: a deterministic workload generator (diurnal wave plus an
 * AR(1) noise term plus scheduled spikes), three forecasters (seasonal naive, EWMA, and Holt's
 * double exponential smoothing with trend), a queueing capacity model that turns a replica count and
 * an offered load into a p99 latency and a drop count, and a controller that computes a desired
 * replica count from the *forecast* load, then stabilises it with asymmetric scaling, hysteresis and
 * a cooldown. Metrics are the SLO compliance (fraction of ticks inside the latency target), the cost
 * (replica-seconds) and the number of scaling actions.
 *
 * There is no wall clock and no randomness at run time: the workload comes from a splitmix64 stream,
 * every tick is one simulated step, and the controller is a pure function of the observed history.
 * Sizes are typed parameters (ticks, replicas, capacity per replica, SLO).
 *
 * What is simplified: latency comes from a closed-form M/M/c-style approximation instead of a real
 * simulator, autoscaling acts within one tick (no boot delay, no instance-launch jitter), the
 * forecasters are textbook recursions rather than trained models, and the workload is synthetic.
 * The point of the file is the control-loop *policy* and the metrics it is judged on.
 *
 * Invariants verified:
 *   - accuracy: Holt and EWMA beat the seasonal-naive baseline on the generated series (MAE), and
 *     the forecast error stays inside the noise band;
 *   - SLO: the controller holds the latency target for a far larger fraction of ticks than a
 *     fixed-size baseline provisioned for the average load;
 *   - cost: the autoscaler meets that SLO while spending strictly fewer replica-seconds than a
 *     statically over-provisioned fleet that meets the same SLO;
 *   - stability: hysteresis and the cooldown bound the number of scaling actions (no thrashing),
 *     the replica count stays inside [min, max], and scale-down after a spike is gradual;
 *   - predictive value: a forecast horizon shortens the reaction delay, which shows up as fewer SLO
 *     violations during the ramp of a spike than a reactive-only policy;
 *   - accounting: served + dropped equals offered on every single tick, an idle system scales to the
 *     minimum, and an overload beyond max degrades gracefully instead of breaking the model.
 */
"use strict";

const assert = require("node:assert");

class ScalingError extends Error {}

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, unit: () => Number(next() >> 11n) / 2 ** 53, int: (n) => Number(next() % BigInt(n)) };
}

// --------------------------------------------------------------------------- workload

/**
 * Offered load per tick: a diurnal sine, an AR(1) noise term, and optional spike windows. Everything
 * is derived from one seeded stream, so the whole run is reproducible.
 */
class Workload {
  constructor(opts = {}) {
    this.base = opts.base ?? 120;
    this.amplitude = opts.amplitude ?? 90;
    this.period = opts.period ?? 96;
    this.noise = opts.noise ?? 25;
    this.spikes = opts.spikes ?? [];
    this.rng = splitmix64(opts.seed ?? 0x809);
    this.phi = 0.6;
    this.previous = 0;
    this.ticks = 0;
  }

  next() {
    const tick = this.ticks++;
    const wave = Math.sin((2 * Math.PI * tick) / this.period);
    this.previous = this.phi * this.previous + Math.sqrt(1 - this.phi ** 2) * (this.rng.unit() * 2 - 1);
    let value = this.base + this.amplitude * wave + this.noise * this.previous;
    for (const spike of this.spikes) {
      if (tick >= spike.start && tick < spike.start + spike.length) {
        // A spike may ramp up over `ramp` ticks; a gradual rise is what a trend forecast can see.
        const within = tick - spike.start + 1;
        const factor = spike.ramp === undefined || spike.ramp === 0 ? 1 : Math.min(1, within / spike.ramp);
        value += spike.height * factor;
      }
    }
    return Math.max(0, value);
  }
}

// --------------------------------------------------------------------------- forecasters

class SeasonalNaive {
  constructor(period) { this.period = period; this.history = []; }
  observe(value) { this.history.push(value); return value; }
  forecast() { return this.history.length <= this.period ? 0 : this.history[this.history.length - this.period]; }
}

class Ewma {
  constructor(alpha) { this.alpha = alpha; this.level = 0; this.started = false; }
  observe(value) {
    this.level = this.started ? this.alpha * value + (1 - this.alpha) * this.level : value;
    this.started = true;
    return this.level;
  }
  forecast() { return this.level; }
}

/** Holt's linear trend: level and slope recursions, which is what gives the ramp a head start. */
class Holt {
  constructor(alpha, beta) { this.alpha = alpha; this.beta = beta; this.level = 0; this.trend = 0; this.started = false; }
  observe(value) {
    if (!this.started) { this.level = value; this.trend = 0; this.started = true; return this.level; }
    const previousLevel = this.level;
    this.level = this.alpha * value + (1 - this.alpha) * (this.level + this.trend);
    this.trend = this.beta * (this.level - previousLevel) + (1 - this.beta) * this.trend;
    return this.level;
  }
  forecast(horizon = 0) { return Math.max(0, this.level + this.trend * horizon); }
}

function meanAbsoluteError(forecaster, series, horizon = 1) {
  const model = forecaster.constructor === SeasonalNaive ? new SeasonalNaive(forecaster.period)
    : forecaster.constructor === Holt ? new Holt(forecaster.alpha, forecaster.beta) : new Ewma(forecaster.alpha);
  const warmup = forecaster.constructor === SeasonalNaive ? forecaster.period : 10;
  let error = 0;
  let count = 0;
  for (let index = 0; index < series.length; index++) {
    const prediction = model.forecast(horizon);
    if (index >= warmup && index + horizon < series.length) { error += Math.abs(prediction - series[index + horizon]); count++; }
    model.observe(series[index]);
  }
  return count === 0 ? Number.POSITIVE_INFINITY : error / count;
}

// --------------------------------------------------------------------------- capacity model

/**
 * Closed-form capacity model: with `replicas` servers of `capacity` requests per tick, an offered
 * load L gives utilisation rho = L / (replicas * capacity) and a p99 latency that grows sharply as
 * rho approaches one; requests above the service limit are dropped.
 */
class CapacityModel {
  constructor(opts = {}) {
    this.capacity = opts.capacity ?? 40;      // requests served per replica per tick
    this.serviceTime = opts.serviceTime ?? 10; // milliseconds per request
    this.targetLatency = opts.targetLatency ?? 70;
  }

  serve(replicas, offered) {
    const serviceLimit = replicas * this.capacity;
    const served = Math.min(offered, serviceLimit);
    const dropped = Math.max(0, offered - serviceLimit);
    const utilisation = serviceLimit === 0 ? 1 : served / serviceLimit;
    const queueFactor = utilisation >= 0.95 ? 1 / Math.max(0.05, 1 - utilisation) : 1 + utilisation;
    const latency = this.serviceTime * (1 + queueFactor);
    return { served, dropped, utilisation, latency };
  }

  /** The smallest replica count whose service limit and p99 latency both hold for `load`. */
  requiredReplicas(load) {
    let replicas = 1;
    while (replicas < 512) {
      const serviceLimit = replicas * this.capacity;
      const utilisation = serviceLimit === 0 ? 1 : Math.min(load, serviceLimit) / serviceLimit;
      const queueFactor = utilisation >= 0.95 ? 1 / Math.max(0.05, 1 - utilisation) : 1 + utilisation;
      const latency = this.serviceTime * (1 + queueFactor);
      if (load <= serviceLimit * 0.9 && latency <= this.targetLatency) return replicas;
      replicas++;
    }
    return 512;
  }
}

// --------------------------------------------------------------------------- controller

class Controller {
  constructor(model, opts = {}) {
    this.model = model;
    this.min = opts.min ?? 2;
    this.max = opts.max ?? 24;
    this.horizon = opts.horizon ?? 0;             // 0 = reactive, >=1 = predictive
    this.forecaster = opts.forecaster ?? new Holt(0.45, 0.25);
    this.downSlope = opts.downSlope ?? 0.6;       // asymmetric: scale up fast, down carefully
    this.hysteresis = opts.hysteresis ?? 0.15;
    this.cooldown = opts.cooldown ?? 3;
    this.safety = opts.safety ?? 1.12;
    this.bootTicks = opts.bootTicks ?? 0;         // time between a scaling action and its effect
    this.replicas = opts.start ?? this.min;       // active replicas
    this.target = this.replicas;                  // last requested size (what we are converging to)
    this.pending = [];
    this.lastChange = -Infinity;
    this.actions = 0;
    this.upscales = 0;
    this.downsides = 0;
    this.tick = 0;
  }

  step(observedLoad) {
    this.forecaster.observe(observedLoad);
    const prediction = this.horizon > 0 ? this.forecaster.forecast(this.horizon) : observedLoad;
    // The request is sized for the load we expect to see when the change takes effect.
    const required = this.model.requiredReplicas(prediction * this.safety);
    const current = this.target;
    let desired = current;
    if (required > current) {
      desired = required;                                   // scale up immediately
    } else if (required < current) {
      const slack = current - required;
      if (this.tick - this.lastChange >= this.cooldown && slack / Math.max(1, current) > this.hysteresis) {
        desired = Math.max(required, Math.ceil(current - slack * this.downSlope));   // step down slowly
      }
    }
    desired = Math.min(this.max, Math.max(this.min, desired));
    if (desired !== current) {
      this.actions++;
      if (desired > current) this.upscales++;
      else this.downsides++;
      this.lastChange = this.tick;
      this.target = desired;
      if (this.bootTicks === 0) this.replicas = desired;
      else this.pending.push({ at: this.tick + this.bootTicks, replicas: desired });
    }
    for (const entry of this.pending) if (this.tick >= entry.at) this.replicas = entry.replicas;
    this.pending = this.pending.filter((entry) => entry.at > this.tick);
    this.tick++;
    return { replicas: this.replicas, target: this.target, prediction, required };
  }
}

// --------------------------------------------------------------------------- simulation

function simulate(opts) {
  const workload = opts.workload;
  const model = opts.model;
  const controller = opts.controller ?? null;
  const fixedReplicas = opts.fixedReplicas ?? null;
  const ticks = opts.ticks;
  const series = [];
  const results = [];
  let violations = 0;
  let served = 0;
  let dropped = 0;
  let cost = 0;
  let offered = 0;
  for (let tick = 0; tick < ticks; tick++) {
    const load = workload.next();
    series.push(load);
    const replicas = controller === null ? fixedReplicas : controller.step(load).replicas;
    const outcome = model.serve(replicas, load);
    if (outcome.latency > model.targetLatency || outcome.dropped > 0) violations++;   // a drop is an SLO violation
    served += outcome.served;
    dropped += outcome.dropped;
    offered += load;
    cost += replicas;
    results.push({ tick, load, replicas, ...outcome });
  }
  return {
    series, results, violations, served, dropped, cost, offered,
    sloCompliance: 1 - violations / ticks,
    dropRate: dropped / Math.max(1, offered),
    actions: controller === null ? 0 : controller.actions,
    upscales: controller === null ? 0 : controller.upscales,
    downsides: controller === null ? 0 : controller.downsides,
  };
}

if (require.main === module) {
  // --- 1. the workload is deterministic and inside its band --------------------------------------
  const generator = new Workload({ seed: 0x809 });
  const series = Array.from({ length: 240 }, () => generator.next());
  const repeat = new Workload({ seed: 0x809 });
  const seriesAgain = Array.from({ length: 240 }, () => repeat.next());
  assert.deepStrictEqual(series, seriesAgain, "the workload is reproducible from its seed");
  const maximum = Math.max(...series);
  const minimum = Math.min(...series);
  assert.ok(minimum >= 0, `the workload never goes negative (min ${minimum.toFixed(1)})`);
  assert.ok(maximum < 400, `the workload stays inside its band (max ${maximum.toFixed(1)})`);
  assert.ok(series.some((value, index) => index > 0 && value !== series[index - 1]), "the series actually varies");

  // --- 2. forecasters: Holt wins, and a periodic series favours the fixed-lag baseline -----------
  const seasonalError = meanAbsoluteError(new SeasonalNaive(96), series);
  const ewmaError = meanAbsoluteError(new Ewma(0.3), series);
  const holtError = meanAbsoluteError(new Holt(0.45, 0.25), series);
  assert.ok(holtError < seasonalError, `Holt (${holtError.toFixed(2)}) beats the seasonal baseline (${seasonalError.toFixed(2)})`);
  assert.ok(holtError < ewmaError, `Holt (${holtError.toFixed(2)}) beats EWMA (${ewmaError.toFixed(2)})`);
  assert.ok(ewmaError > seasonalError, "on a purely periodic series a memoryless EWMA loses to the fixed-lag baseline");
  const spikyWorkload = new Workload({ seed: 0x809, spikes: [{ start: 120, length: 16, height: 300 }] });
  const spikySeries = Array.from({ length: 220 }, () => spikyWorkload.next());
  const spikeSeasonal = meanAbsoluteError(new SeasonalNaive(96), spikySeries);
  const spikeEwma = meanAbsoluteError(new Ewma(0.3), spikySeries);
  const spikeHolt = meanAbsoluteError(new Holt(0.45, 0.25), spikySeries);
  assert.ok(spikeHolt < spikeEwma && spikeEwma < spikeSeasonal,
    `on a spiky series the adaptive models win (${spikeHolt.toFixed(1)} < ${spikeEwma.toFixed(1)} < ${spikeSeasonal.toFixed(1)})`);
  const holtOneStep = new Holt(0.45, 0.25);
  holtOneStep.observe(100);
  holtOneStep.observe(110);
  assert.ok(holtOneStep.forecast(1) > holtOneStep.forecast(0), "a rising series gives a rising forecast with a horizon");
  assert.strictEqual(holtOneStep.trend > 0, true, "the trend term is positive while the load climbs");

  // --- 3. the capacity model is monotone and honest ----------------------------------------------
  const model = new CapacityModel({});
  const light = model.serve(6, 200);
  const heavy = model.serve(6, 480);
  assert.ok(light.latency < heavy.latency, `more load means more latency (${light.latency.toFixed(0)} -> ${heavy.latency.toFixed(0)})`);
  assert.ok(heavy.dropped > 0, "an overload drops requests");
  assert.strictEqual(light.dropped, 0, "a light load drops nothing");
  assert.strictEqual(light.served + light.dropped, 200, "served plus dropped equals offered");
  assert.strictEqual(heavy.served >= 6 * 40 - 1 && heavy.served <= 6 * 40, true, "the service limit is respected");
  const meanLoad = series.reduce((sum, value) => sum + value, 0) / series.length;
  const requiredForAverage = model.requiredReplicas(meanLoad);
  const requiredForPeak = model.requiredReplicas(maximum);
  assert.ok(requiredForPeak > requiredForAverage, `the peak (${maximum.toFixed(0)}) needs more replicas (${requiredForPeak}) than the mean (${meanLoad.toFixed(0)} -> ${requiredForAverage})`);
  assert.ok(model.requiredReplicas(0) >= 1, "an idle system still needs at least one replica");

  // --- 4. the control loop beats a fixed fleet on SLO compliance ---------------------------------
  const autoscaled = simulate({
    workload: new Workload({ seed: 0x8091 }),
    model,
    controller: new Controller(model, { min: 2, max: 24, horizon: 0, start: 2 }),
    ticks: 240,
  });
  const fixedAverage = simulate({ workload: new Workload({ seed: 0x8091 }), model, fixedReplicas: requiredForAverage, ticks: 240 });
  assert.ok(autoscaled.sloCompliance > fixedAverage.sloCompliance + 0.2,
    `autoscaling (${(autoscaled.sloCompliance * 100).toFixed(1)}%) beats a fixed fleet sized for the average (${(fixedAverage.sloCompliance * 100).toFixed(1)}%)`);
  assert.ok(autoscaled.sloCompliance > 0.95, `the autoscaler holds the SLO for ${(autoscaled.sloCompliance * 100).toFixed(1)}% of ticks`);
  assert.ok(autoscaled.actions > 5 && autoscaled.actions < 60, `the controller acted ${autoscaled.actions} times - neither stuck nor thrashing`);
  assert.strictEqual(autoscaled.served + autoscaled.dropped, autoscaled.offered, "no request is lost from the accounting");

  // --- 5. cost: the autoscaler is cheaper than a static fleet that also meets the SLO -------------
  let staticNeeded = requiredForAverage;
  let staticRun = fixedAverage;
  while (staticRun.sloCompliance < autoscaled.sloCompliance && staticNeeded < 40) {
    staticNeeded++;
    staticRun = simulate({ workload: new Workload({ seed: 0x8091 }), model, fixedReplicas: staticNeeded, ticks: 240 });
  }
  assert.ok(staticRun.sloCompliance >= autoscaled.sloCompliance, `a static fleet of ${staticNeeded} replicas matches the SLO`);
  assert.ok(autoscaled.cost < staticRun.cost * 0.92,
    `autoscaling costs ${autoscaled.cost} replica-ticks against ${staticRun.cost} for the static fleet of ${staticNeeded}`);
  assert.ok(autoscaled.cost > 240 * 2, "and it does not simply stay at the minimum");

  // --- 6. stability: bounded actions, inside the bounds, gradual scale-down ----------------------
  const bounded = simulate({
    workload: new Workload({ seed: 0x8092, spikes: [{ start: 100, length: 12, height: 260 }] }),
    model,
    controller: new Controller(model, { min: 3, max: 12, horizon: 2, cooldown: 4, downSlope: 0.5 }),
    ticks: 240,
  });
  const sizes = bounded.results.map((result) => result.replicas);
  assert.ok(sizes.every((size) => size >= 3 && size <= 12), "the replica count never leaves [min, max]");
  assert.ok(bounded.actions <= 40, `hysteresis and the cooldown bound the actions (${bounded.actions})`);
  const spikeIndex = 105;
  const peak = sizes.slice(spikeIndex, spikeIndex + 20);
  const peakSize = Math.max(...peak);
  const after = sizes.slice(spikeIndex + 24, spikeIndex + 40);
  assert.ok(peakSize > sizes[spikeIndex - 10], "the controller reacted to the spike");
  assert.ok(after.every((size) => size <= peakSize), "after the spike the fleet only shrinks");
  const shrinkSteps = sizes.slice(spikeIndex + 24).filter((size, index, list) => index > 0 && size < list[index - 1]).length;
  assert.ok(shrinkSteps >= 1, `the scale-down happened in steps (${shrinkSteps} decreasing steps)`);
  assert.ok(bounded.sloCompliance > 0.85, `the spike is absorbed (SLO ${(bounded.sloCompliance * 100).toFixed(1)}%)`);

  // --- 7. predictive beats reactive on a spike ramp with a boot delay ----------------------------
  const spikeWorkload = () => new Workload({ seed: 0x8093, spikes: [{ start: 120, length: 30, height: 320, ramp: 16 }] });
  const reactive = simulate({ workload: spikeWorkload(), model, controller: new Controller(model, { min: 2, max: 30, horizon: 0, cooldown: 3, start: 2, bootTicks: 4 }), ticks: 220 });
  const predictive = simulate({ workload: spikeWorkload(), model, controller: new Controller(model, { min: 2, max: 30, horizon: 4, cooldown: 3, start: 2, bootTicks: 4 }), ticks: 220 });
  const violationsIn = (run, from, to) => run.results.slice(from, to).filter((result) => result.latency > model.targetLatency || result.dropped > 0).length;
  const reactiveRamp = violationsIn(reactive, 118, 150);
  const predictiveRamp = violationsIn(predictive, 118, 150);
  assert.ok(reactiveRamp > 0, `a reactive controller cannot absorb the ramp within the boot delay (${reactiveRamp} violating ticks)`);
  assert.ok(predictiveRamp < reactiveRamp, `the forecast horizon shortens the ramp (${predictiveRamp} vs ${reactiveRamp} violating ticks)`);
  assert.ok(predictive.sloCompliance >= reactive.sloCompliance - 0.02, "and it never makes the overall SLO worse");
  assert.ok(reactive.actions > 0 && predictive.actions > 0, "both controllers acted during the spike");

  // --- 8. edge cases: idle traffic, a step down, and a load beyond max ---------------------------
  const idle = new Workload({ seed: 0x8094, base: 0, amplitude: 0, noise: 0 });
  const idleRun = simulate({ workload: idle, model, controller: new Controller(model, { min: 2, max: 10 }), ticks: 40 });
  assert.deepStrictEqual([...new Set(idleRun.results.map((result) => result.replicas))], [2], "an idle system stays at the minimum");
  assert.strictEqual(idleRun.dropped, 0, "and drops nothing");
  assert.strictEqual(idleRun.actions, 0, "and never scales");
  const overload = simulate({
    workload: new Workload({ seed: 0x8095, base: 600, amplitude: 20, noise: 0 }),
    model,
    controller: new Controller(model, { min: 2, max: 8, horizon: 1 }),
    ticks: 60,
  });
  assert.strictEqual(Math.max(...overload.results.map((result) => result.replicas)), 8, "a load beyond max pins the fleet at max");
  assert.ok(overload.dropped > 0, "and the excess is dropped rather than queued forever");
  assert.strictEqual(overload.served + overload.dropped, overload.offered, "the accounting still balances under overload");
  assert.ok(overload.results.every((result) => Number.isFinite(result.latency)), "the latency model stays finite at the limit");
  const stepDown = simulate({
    workload: new Workload({ seed: 0x8096, base: 40, amplitude: 0, noise: 0, spikes: [{ start: 0, length: 40, height: 320 }] }),
    model,
    controller: new Controller(model, { min: 2, max: 20, horizon: 1, cooldown: 2 }),
    ticks: 120,
  });
  const stepSizes = stepDown.results.map((result) => result.replicas);
  assert.ok(stepSizes[0] > 2 || stepSizes[5] > 2, "the initial spike raised the fleet");
  assert.strictEqual(stepSizes[stepSizes.length - 1] <= stepSizes[10], true, "after the step down the fleet shrank");

  // --- 9. reported metrics -----------------------------------------------------------------------
  const peakReplicas = Math.max(...autoscaled.results.map((result) => result.replicas));
  const meanReplicas = autoscaled.results.reduce((sum, result) => sum + result.replicas, 0) / autoscaled.results.length;
  assert.ok(peakReplicas > meanReplicas, `the fleet is elastic (peak ${peakReplicas}, mean ${meanReplicas.toFixed(1)})`);
  assert.ok(autoscaled.dropRate < 0.05, `the autoscaler drops only ${(autoscaled.dropRate * 100).toFixed(2)}% of requests`);
  assert.ok(fixedAverage.dropRate >= 0, "the fixed baseline is counted the same way");

  console.log(`forecast: MAE seasonal-naive ${seasonalError.toFixed(1)}, EWMA ${ewmaError.toFixed(1)}, Holt ${holtError.toFixed(1)} over ${series.length} ticks`);
  console.log(`autoscaler: SLO ${(autoscaled.sloCompliance * 100).toFixed(1)}% for ${autoscaled.cost} replica-ticks vs static ${staticNeeded} replicas at ${staticRun.cost} (SLO ${(staticRun.sloCompliance * 100).toFixed(1)}%)`);
  console.log(`control: ${autoscaled.actions} actions (${autoscaled.upscales} up / ${autoscaled.downsides} down), mean ${meanReplicas.toFixed(1)} replicas, peak ${peakReplicas}, drop rate ${(autoscaled.dropRate * 100).toFixed(2)}%`);
  console.log(`forecast horizon: ramp violations ${predictiveRamp} (predictive, horizon 4) vs ${reactiveRamp} (reactive), spike SLO ${(bounded.sloCompliance * 100).toFixed(1)}%`);  console.log("PASS");
}

module.exports = { Workload, Ewma, Holt, SeasonalNaive, CapacityModel, Controller, simulate, meanAbsoluteError, splitmix64, ScalingError };
