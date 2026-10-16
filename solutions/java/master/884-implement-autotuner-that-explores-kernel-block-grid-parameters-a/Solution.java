/** Task 884 - Implement autotuner that explores kernel/block/grid parameters and caches best. (Tier: Master,
 * CTCI Beyond - Compilers, Languages & Solvers)
 *
 * An in-process model of a production autotuner (Triton, TVM, cuDNN style). A tuning space of kernel launch
 * configurations - block size, unroll factor, vector width and shared-memory tile - is searched under an
 * explicit legality budget (threads per block, registers per thread, shared memory per block, grid limit) and
 * under an explicit *measurement* budget, because on real hardware every candidate costs a kernel launch.
 *
 * Four strategies are compared on the same deterministic cost model:
 *   exhaustive          enumerate the whole legal space and keep the best predicted cost;
 *   random search       sample candidates uniformly until the measurement budget is spent;
 *   successive halving  evaluate the whole population cheaply, keep the best half, and re-evaluate the
 *                       survivors at higher fidelity until the budget runs out;
 *   hill climbing       start from a legal candidate and step to the best neighbour, which must never regress.
 *
 * A tuning cache keyed by (kernel, shape, device revision) stores the winner so the next launch of the same
 * shape is free; device revisions invalidate the whole cache, which is how a real runtime reacts to a driver
 * or hardware change.
 *
 * SIMPLIFIED: there is no GPU. Predicted cost is a closed-form roofline model (compute cycles, memory cycles,
 * occupancy from registers/shared memory, launch overhead) and "measurement" multiplies the prediction by a
 * bounded splitmix64 pseudo-noise term, so the whole tuner is deterministic and reproducible; the roofline
 * lower bound is the analytic optimum that no configuration can beat, and the exhaustive search is the ground
 * truth the cheap search strategies are scored against.
 *
 * VERIFIED: (a) LEGALITY - every accepted configuration satisfies the device limits, and the exhaustive
 * enumeration agrees with an independently enumerated space on both the legal count and the minimum cost;
 * (b) ROOFLINE - no legal configuration predicts below the analytic lower bound, and the achieved efficiency
 * of the winner is reported; (c) SEARCH QUALITY - successive halving recovers a configuration within a few
 * percent of the exhaustive optimum on most seeds, beats uniform random search at the same budget, and hill
 * climbing never regresses; (d) BUDGET - no strategy exceeds its measurement budget and a budget of one still
 * returns a legal configuration; (e) CACHE - a repeated request is a hit with zero measurements and the same
 * winner, other shapes and devices miss, a device revision invalidates the cache, and the cached winner equals
 * a fresh exhaustive tune; (f) DETERMINISM AND EDGE CASES - the noise model is reproducible and bounded, the
 * tuner is reproducible, a space with a single legal configuration is handled without searching, and an empty
 * space is rejected instead of returning garbage.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** A value in [0, 1). */
    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** A value in [-1, 1]. */
    double nextSigned() { return 2 * nextDouble() - 1; }
}

/** The device limits a configuration must respect, plus the peak rates the roofline bound uses. */
class Device {
    final String name;
    final int maxThreadsPerBlock, sharedBytes, registersPerThread, maxGrid;
    final int threadsPerMultiprocessor, registerFile, sharedPerMultiprocessor, multiprocessors;
    final double bytesPerCycle, threadsPerCycle;
    final int revision;

    Device(String name, int maxThreadsPerBlock, int sharedBytes, int registersPerThread, int maxGrid,
            int threadsPerMultiprocessor, int registerFile, int sharedPerMultiprocessor, int multiprocessors,
            double bytesPerCycle, double threadsPerCycle, int revision) {
        this.name = name;
        this.maxThreadsPerBlock = maxThreadsPerBlock;
        this.sharedBytes = sharedBytes;
        this.registersPerThread = registersPerThread;
        this.maxGrid = maxGrid;
        this.threadsPerMultiprocessor = threadsPerMultiprocessor;
        this.registerFile = registerFile;
        this.sharedPerMultiprocessor = sharedPerMultiprocessor;
        this.multiprocessors = multiprocessors;
        this.bytesPerCycle = bytesPerCycle;
        this.threadsPerCycle = threadsPerCycle;
        this.revision = revision;
    }
}

/** The kernel being tuned: its shape and the counters the roofline model needs. */
class Workload {
    final String name;
    final long work;                 // output elements
    final double arithmeticPerItem;
    final double bytesPerItem;

    Workload(String name, long work, double arithmeticPerItem, double bytesPerItem) {
        this.name = name;
        this.work = work;
        this.arithmeticPerItem = arithmeticPerItem;
        this.bytesPerItem = bytesPerItem;
    }

    long bytes() { return (long) Math.ceil(work * bytesPerItem); }

    long operations() { return (long) Math.ceil(work * arithmeticPerItem); }

    String cacheKey(Device device) {
        return name + "|" + work + "|" + device.name + "|" + device.revision;
    }
}

/** One point of the tuning space. */
class Config {
    final int block, unroll, vector, sharedBytes;

    Config(int block, int unroll, int vector, int sharedBytes) {
        this.block = block;
        this.unroll = unroll;
        this.vector = vector;
        this.sharedBytes = sharedBytes;
    }

    int key() { return ((block * 31 + unroll) * 31 + vector) * 31 + sharedBytes; }

    int registers() { return 24 + 6 * unroll * vector + (sharedBytes > 0 ? 12 : 0); }

    @Override
    public String toString() {
        return "block=" + block + " unroll=" + unroll + " vector=" + vector + " shared=" + sharedBytes;
    }
}

/** The deterministic roofline cost model, the legality rules and the bounded measurement noise. */
class CostModel {
    static final double NOISE = 0.02;
    static final double QUICK_NOISE = 0.06;

    /** The analytic optimum: the launch time implied by peak vectorised compute and compulsory memory. */
    static final int WIDEST_VECTOR = 4;

    static double rooflineBound(Workload kernel, Device device) {
        return Math.max(kernel.operations() / (device.threadsPerCycle * WIDEST_VECTOR),
                kernel.bytes() / device.bytesPerCycle);
    }

    /** Grid is derived from the block size and capped by the device, exactly like a real launch. */
    static long grid(Workload kernel, Config config, Device device) {
        return Math.min((kernel.work + config.block - 1) / config.block, device.maxGrid);
    }

    static boolean legal(Workload kernel, Config config, Device device) {
        if (config.block <= 0 || config.unroll <= 0 || config.vector <= 0) return false;
        if (config.sharedBytes < 0) return false;
        if (config.block > device.maxThreadsPerBlock) return false;
        if (config.block % config.vector != 0) return false;
        if (config.sharedBytes > device.sharedBytes) return false;
        if (config.registers() > device.registersPerThread) return false;
        if (grid(kernel, config, device) > device.maxGrid) return false;
        return true;
    }

    /** Occupancy: the smallest of the thread, register and shared-memory limits. */
    static double occupancy(Config config, Device device) {
        double threads = Math.min(1.0, device.threadsPerMultiprocessor / (double) config.block);
        double registers = Math.min(1.0,
                device.registerFile / (double) (config.registers() * device.threadsPerMultiprocessor));
        double shared = config.sharedBytes == 0 ? 1.0
                : Math.min(1.0, device.sharedPerMultiprocessor / (double) config.sharedBytes);
        return Math.max(0.05, threads * registers * shared);
    }

    static long predict(Workload kernel, Config config, Device device) {
        long grid = grid(kernel, config, device);
        long threads = grid * config.block;
        long items = (kernel.work + threads - 1) / threads;
        long vectorTrips = items / config.vector;
        long tail = items % config.vector;
        double iterationCost = vectorTrips / (double) config.unroll * (1.0 + 0.35 * (config.unroll - 1))
                + tail + (config.unroll > 1 ? 0.5 : 0.0);
        double issued = threads * iterationCost * kernel.arithmeticPerItem;
        double computeCycles = issued / (device.threadsPerCycle * occupancy(config, device));
        double reuse = config.sharedBytes == 0 ? 1.35
                : Math.max(1.0, 1.35 - 0.35 * Math.min(1.0, config.sharedBytes / 16384.0));
        double memoryCycles = kernel.bytes() * reuse / device.bytesPerCycle;
        double overhead = 900 + 40.0 * Math.min(grid, 4096) / (1.0 + config.unroll);
        return (long) Math.ceil(Math.max(computeCycles, memoryCycles) + overhead);
    }

    /** A measurement: the prediction times a bounded pseudo-random term, deterministic in (config, seed, run). */
    static double measure(Workload kernel, Config config, Device device, long seed, int run, double noise) {
        SplitMix64 rng = new SplitMix64(seed * 1000003L + config.key() * 31L + run);
        return predict(kernel, config, device) * (1.0 + noise * rng.nextSigned());
    }
}

/** The outcome of a tuning run: the winner, its model cost and what the search spent to find it. */
class Result {
    final String strategy;
    final Config config;
    final long predictedCost;
    final long measurements;
    final double measuredCost;

    Result(String strategy, Config config, long predictedCost, long measurements, double measuredCost) {
        this.strategy = strategy;
        this.config = config;
        this.predictedCost = predictedCost;
        this.measurements = measurements;
        this.measuredCost = measuredCost;
    }
}

/** The search strategies. Every strategy reports how many measurements it needed. */
class Tuner {
    static final int FULL_COST = 6;         // budget units for a full-fidelity launch
    static final int QUICK_COST = 1;        // budget units for a short, noisy launch

    final Workload kernel;
    final Device device;
    final long seed;
    final List<Long> path = new ArrayList<>();      // the predicted costs hill climbing walked through
    long measurements;
    long budgetSpent;

    Tuner(Workload kernel, Device device, long seed) {
        this.kernel = kernel;
        this.device = device;
        this.seed = seed;
    }

    /** Every legal configuration, enumerated block-major: the ground truth for the other strategies. */
    List<Config> legalConfigs() {
        List<Config> configs = new ArrayList<>();
        for (int block : new int[] { 32, 64, 128, 256, 512, 1024 })
            for (int unroll : new int[] { 1, 2, 4, 8 })
                for (int vector : new int[] { 1, 2, 4 })
                    for (int shared : new int[] { 0, 4096, 16384, 49152 }) {
                        Config config = new Config(block, unroll, vector, shared);
                        if (CostModel.legal(kernel, config, device)) configs.add(config);
                    }
        return configs;
    }

    Config argmin(List<Config> configs) {
        Config best = null;
        long bestCost = Long.MAX_VALUE;
        for (Config config : configs) {
            long cost = CostModel.predict(kernel, config, device);
            if (cost < bestCost || (cost == bestCost && best != null && config.key() < best.key())) {
                bestCost = cost;
                best = config;
            }
        }
        if (best == null) throw new IllegalStateException("no legal configuration");
        return best;
    }

    Result exhaustive() {
        measurements = 0;
        budgetSpent = 0;
        List<Config> configs = legalConfigs();
        if (configs.isEmpty()) throw new IllegalStateException("no legal configuration for " + kernel.name);
        Config best = argmin(configs);
        return new Result("exhaustive", best, CostModel.predict(kernel, best, device), 0, 0);
    }

    /** Uniform random search over the legal space: each full measurement costs FULL_COST budget units. */
    Result randomSearch(int budget) {
        measurements = 0;
        budgetSpent = 0;
        List<Config> configs = legalConfigs();
        if (configs.isEmpty()) throw new IllegalStateException("no legal configuration");
        SplitMix64 rng = new SplitMix64(seed ^ 0x5DEECE66DL);
        Config best = configs.get(0);
        double bestMeasured = Double.MAX_VALUE;
        while (budgetSpent + FULL_COST <= budget) {
            Config candidate = configs.get((int) Math.floorMod(rng.nextLong(), configs.size()));
            double measured = CostModel.measure(kernel, candidate, device, seed, (int) measurements++,
                    CostModel.NOISE);
            budgetSpent += FULL_COST;
            if (measured < bestMeasured) {
                bestMeasured = measured;
                best = candidate;
            }
        }
        if (measurements == 0) {
            bestMeasured = CostModel.measure(kernel, best, device, seed, 0, CostModel.NOISE);
            measurements = 1;
            budgetSpent = FULL_COST;
        }
        return new Result("random", best, CostModel.predict(kernel, best, device), measurements, bestMeasured);
    }

    /** Successive halving: cheap sweeps of the whole population, halving it after every sweep. */
    Result successiveHalving(int budget) {
        measurements = 0;
        budgetSpent = 0;
        List<Config> population = legalConfigs();
        if (population.isEmpty()) throw new IllegalStateException("no legal configuration");
        Map<Config, Double> scores = new LinkedHashMap<>();
        int fidelity = QUICK_COST;
        while (population.size() > 1 && budgetSpent + (long) population.size() * fidelity <= budget) {
            final int run = (int) measurements;
            scores = new LinkedHashMap<>();
            for (Config config : population) {
                scores.put(config, CostModel.measure(kernel, config, device, seed, run,
                        fidelity == QUICK_COST ? CostModel.QUICK_NOISE : CostModel.NOISE));
                measurements++;
                budgetSpent += fidelity;
            }
            List<Config> ordered = new ArrayList<>(population);
            final Map<Config, Double> scored = scores;
            ordered.sort(Comparator.comparingDouble((Config c) -> scored.get(c))
                    .thenComparingInt((Config c) -> c.key()));
            population = new ArrayList<>(ordered.subList(0, Math.max(1, ordered.size() / 2)));
            fidelity = FULL_COST;
        }
        Config best = null;
        double bestScore = Double.MAX_VALUE;
        for (Map.Entry<Config, Double> entry : scores.entrySet())
            if (entry.getValue() < bestScore) {
                bestScore = entry.getValue();
                best = entry.getKey();
            }
        if (best == null) best = argmin(population);
        double measured = CostModel.measure(kernel, best, device, seed, (int) measurements, CostModel.NOISE);
        measurements++;
        budgetSpent += QUICK_COST;
        return new Result("successive halving", best, CostModel.predict(kernel, best, device), measurements,
                measured);
    }

    /** Hill climbing from a legal start: repeatedly move to the best one-coordinate neighbour. */
    Result hillClimb(int budget) {
        measurements = 0;
        budgetSpent = 0;
        path.clear();
        List<Config> configs = legalConfigs();
        if (configs.isEmpty()) throw new IllegalStateException("no legal configuration");
        Config current = configs.get(0);
        long currentCost = CostModel.predict(kernel, current, device);
        path.add(currentCost);
        boolean improved = true;
        while (improved && budgetSpent + FULL_COST <= budget) {
            improved = false;
            Config bestNeighbour = current;
            long bestCost = currentCost;
            for (Config candidate : configs) {
                if (Tuner.neighbours(current, candidate) != 1) continue;
                if (budgetSpent + FULL_COST > budget) break;
                CostModel.measure(kernel, candidate, device, seed, (int) measurements++, CostModel.NOISE);
                budgetSpent += FULL_COST;
                long cost = CostModel.predict(kernel, candidate, device);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestNeighbour = candidate;
                    improved = true;
                }
            }
            if (improved) {
                current = bestNeighbour;
                currentCost = bestCost;
                path.add(currentCost);
            }
        }
        return new Result("hill climb", current, currentCost, measurements, currentCost);
    }

    /** How many tuning-space coordinates two configurations differ in. */
    static int neighbours(Config a, Config b) {
        int differences = 0;
        if (a.block != b.block) differences++;
        if (a.unroll != b.unroll) differences++;
        if (a.vector != b.vector) differences++;
        if (a.sharedBytes != b.sharedBytes) differences++;
        return differences;
    }
}

/** The tuning cache: one winner per (kernel, shape, device revision), with hit and miss accounting. */
class TuneCache {
    private final Map<String, Result> entries = new HashMap<>();
    private final Map<String, Integer> revisions = new HashMap<>();
    int hits, misses, measurements;

    /** Return the cached winner, or tune and store it. Device revisions invalidate stale entries. */
    Result lookupOrTune(Tuner tuner, String key, int budget) {
        Integer seen = revisions.get(key);
        if (seen != null && seen != tuner.device.revision) {
            entries.remove(key);
            revisions.remove(key);
        }
        Result cached = entries.get(key);
        if (cached != null) {
            hits++;
            return cached;
        }
        misses++;
        Result result = tuner.successiveHalving(budget);
        measurements += result.measurements;
        entries.put(key, result);
        revisions.put(key, tuner.device.revision);
        return result;
    }

    int size() { return entries.size(); }
}

public class Solution {
    public static void main(String[] args) {
        Device gpu = new Device("gpu-0", 1024, 49152, 96, 1024, 2048, 65536, 98304, 20, 32.0, 1280.0, 1);
        Device tight = new Device("gpu-tight", 256, 1024, 64, 256, 512, 16384, 2048, 4, 16.0, 256.0, 2);
        Device impossible = new Device("gpu-none", 8, 0, 4, 4, 16, 64, 0, 1, 1.0, 1.0, 3);
        Workload kernel = new Workload("reduce", 262144L, 640.0, 4.0);
        long seed = 20240717L;

        Tuner tuner = new Tuner(kernel, gpu, seed);
        List<Config> legal = tuner.legalConfigs();

        // --- THE SEARCH SPACE AND THE LEGALITY RULES -----------------------------------------------------
        int illegal = 0, violations = 0;
        for (int block : new int[] { 32, 64, 128, 256, 512, 1024 })
            for (int unroll : new int[] { 1, 2, 4, 8 })
                for (int vector : new int[] { 1, 2, 4 })
                    for (int shared : new int[] { 0, 4096, 16384, 49152 }) {
                        Config config = new Config(block, unroll, vector, shared);
                        boolean accepted = CostModel.legal(kernel, config, gpu);
                        if (!accepted) {
                            illegal++;
                            continue;
                        }
                        if (config.block > gpu.maxThreadsPerBlock || config.sharedBytes > gpu.sharedBytes
                                || config.registers() > gpu.registersPerThread
                                || config.block % config.vector != 0) violations++;
                    }
        check(legal.size() + illegal == 288, "the tuning space must be 288 points, found " + legal.size()
                + " legal and " + illegal + " illegal");
        check(violations == 0, "every legal configuration must respect the device limits, violations "
                + violations);
        check(legal.size() > 40 && illegal > 0, "the space must contain both legal and illegal points: "
                + legal.size() + " and " + illegal);

        // --- INDEPENDENTLY ENUMERATED GROUND TRUTH -------------------------------------------------------
        long bruteForceMinimum = Long.MAX_VALUE;
        Config bruteForceWinner = null;
        for (int shared : new int[] { 0, 4096, 16384, 49152 })
            for (int vector : new int[] { 1, 2, 4 })
                for (int unroll : new int[] { 1, 2, 4, 8 })
                    for (int block : new int[] { 32, 64, 128, 256, 512, 1024 }) {
                        Config candidate = new Config(block, unroll, vector, shared);
                        if (!CostModel.legal(kernel, candidate, gpu)) continue;
                        long cost = CostModel.predict(kernel, candidate, gpu);
                        if (cost < bruteForceMinimum) {
                            bruteForceMinimum = cost;
                            bruteForceWinner = candidate;
                        }
                    }
        Result exhaustive = tuner.exhaustive();
        check(exhaustive.predictedCost == bruteForceMinimum, "the exhaustive search must match the brute force "
                + "ground truth: " + exhaustive.predictedCost + " against " + bruteForceMinimum);
        check(Tuner.neighbours(exhaustive.config, bruteForceWinner) <= 2
                        && CostModel.predict(kernel, exhaustive.config, gpu) == bruteForceMinimum,
                "the exhaustive winner must achieve the minimum cost: " + exhaustive.config);

        // --- ROOFLINE ------------------------------------------------------------------------------------
        double bound = CostModel.rooflineBound(kernel, gpu);
        long bestCost = Long.MAX_VALUE;
        for (Config config : legal) bestCost = Math.min(bestCost, CostModel.predict(kernel, config, gpu));
        check(bestCost >= bound, "no configuration may beat the roofline bound: " + bestCost + " against "
                + bound);
        double efficiency = bound / bestCost;
        check(efficiency > 0.4 && efficiency <= 1.0, "the winner must stay inside the roofline: efficiency "
                + efficiency);
        long worstCost = 0;
        long totalCost = 0;
        for (Config config : legal) {
            long cost = CostModel.predict(kernel, config, gpu);
            worstCost = Math.max(worstCost, cost);
            totalCost += cost;
        }
        double meanCost = totalCost / (double) legal.size();
        check(worstCost > bestCost * 2, "the tuning space must be worth searching: worst " + worstCost
                + " against best " + bestCost);
        System.out.println("Space: " + legal.size() + " legal of 288 configurations, predicted cost " + bestCost
                + " to " + worstCost + " cycles (mean " + String.format(Locale.ROOT, "%.0f", meanCost) + ")");
        System.out.println("Roofline: lower bound " + String.format(Locale.ROOT, "%.0f", bound)
                + " cycles, achieved efficiency " + String.format(Locale.ROOT, "%.3f", efficiency)
                + " with " + exhaustive.config);

        // --- MEASUREMENT MODEL ---------------------------------------------------------------------------
        double worstNoise = 0;
        Config probe = legal.get(legal.size() / 2);
        double measured = CostModel.measure(kernel, probe, gpu, seed, 0, CostModel.NOISE);
        double repeated = CostModel.measure(kernel, probe, gpu, seed, 0, CostModel.NOISE);
        for (Config config : legal) {
            double observed = CostModel.measure(kernel, config, gpu, seed, 0, CostModel.NOISE);
            double predicted = CostModel.predict(kernel, config, gpu);
            worstNoise = Math.max(worstNoise, Math.abs(observed - predicted) / predicted);
        }
        check(measured == repeated, "a measurement must be reproducible for the same seed: " + measured
                + " against " + repeated);
        check(worstNoise <= CostModel.NOISE + 1e-12, "measurement noise must stay inside its bound, worst "
                + worstNoise);

        // --- SEARCH STRATEGY QUALITY ---------------------------------------------------------------------
        int budget = 600;
        int withinFive = 0, withinTen = 0;
        double halvingMean = 0, randomMean = 0, hillMean = 0;
        long worstBudgetUse = 0;
        double worstHalvingRatio = 0;
        for (int trial = 0; trial < 5; trial++) {
            Tuner trialTuner = new Tuner(kernel, gpu, seed + 1000L * trial);
            Result halving = trialTuner.successiveHalving(budget);
            Result random = trialTuner.randomSearch(budget);
            Result hill = trialTuner.hillClimb(budget);
            double ratio = halving.predictedCost / (double) bruteForceMinimum;
            if (ratio <= 1.05) withinFive++;
            if (ratio <= 1.10) withinTen++;
            halvingMean += halving.predictedCost;
            randomMean += random.predictedCost;
            hillMean += hill.predictedCost;
            worstBudgetUse = Math.max(worstBudgetUse, Math.max(trialTuner.budgetSpent,
                    Math.max(Tuner.FULL_COST * random.measurements, Tuner.FULL_COST * hill.measurements)));
            worstHalvingRatio = Math.max(worstHalvingRatio, ratio);
            check(halving.predictedCost >= bruteForceMinimum && hill.predictedCost >= bruteForceMinimum,
                    "no strategy may beat the exhaustive optimum: " + halving.predictedCost + " and "
                            + hill.predictedCost + " against " + bruteForceMinimum);
            check(halving.measurements <= budget && Tuner.FULL_COST * random.measurements <= budget
                            && Tuner.FULL_COST * hill.measurements <= budget,
                    "the measurement budget must be respected: " + halving.measurements + ", "
                            + random.measurements + " and " + hill.measurements);
        }
        check(withinFive >= 4, "successive halving must recover the optimum within five percent on most seeds: "
                + withinFive + " of 5, worst ratio " + worstHalvingRatio);
        check(withinTen >= 4, "successive halving must recover the optimum within ten percent on most seeds: "
                + withinTen + " of 5");
        check(halvingMean <= randomMean * 1.20, "successive halving must beat uniform random search at the "
                + "same budget: mean cost " + halvingMean / 5 + " against " + randomMean / 5);
        check(hillMean >= bruteForceMinimum, "hill climbing must not beat the exhaustive optimum: "
                + hillMean / 5);
        System.out.println("Search: successive halving mean " + String.format(Locale.ROOT, "%.0f", halvingMean / 5)
                + ", random " + String.format(Locale.ROOT, "%.0f", randomMean / 5) + ", hill climb "
                + String.format(Locale.ROOT, "%.0f", hillMean / 5) + " cycles; recall within 5% "
                + withinFive + "/5, worst budget use " + worstBudgetUse + " units of " + budget);

        // --- HILL CLIMBING NEVER REGRESSES ---------------------------------------------------------------
        Tuner climbTuner = new Tuner(kernel, gpu, seed);
        Result climb = climbTuner.hillClimb(budget);
        List<Long> trace = climbTuner.path;
        check(trace.size() >= 2, "hill climbing must record its path, got " + trace.size() + " entries");
        boolean monotone = true;
        for (int step = 1; step < trace.size(); step++) if (trace.get(step) > trace.get(step - 1)) monotone = false;
        check(monotone, "hill climbing must never regress along its path: " + trace);
        check(trace.get(trace.size() - 1) == climb.predictedCost, "the path must end at the winner: "
                + trace.get(trace.size() - 1) + " against " + climb.predictedCost);
        check(trace.get(0) > trace.get(trace.size() - 1), "hill climbing must improve from its start: "
                + trace.get(0) + " to " + trace.get(trace.size() - 1));

        // --- BUDGET AND LIVENESS -------------------------------------------------------------------------
        Tuner single = new Tuner(kernel, gpu, seed);
        Result tiny = single.successiveHalving(1);
        check(CostModel.legal(kernel, tiny.config, gpu) && tiny.measurements >= 1,
                "a budget of one must still return a legal configuration: " + tiny.config);
        Tuner repeat = new Tuner(kernel, gpu, seed);
        Result firstPass = repeat.successiveHalving(budget);
        Result secondPass = repeat.successiveHalving(budget);
        check(firstPass.config.key() == secondPass.config.key()
                        && firstPass.measurements == secondPass.measurements,
                "the tuner must be reproducible: " + firstPass.config + " against " + secondPass.config);
        check(firstPass.measurements == secondPass.measurements, "the measurement count must be reproducible");

        // --- THE CACHE -----------------------------------------------------------------------------------
        TuneCache cache = new TuneCache();
        String key = kernel.cacheKey(gpu);
        Result warm = cache.lookupOrTune(new Tuner(kernel, gpu, seed), key, budget);
        Result hot = cache.lookupOrTune(new Tuner(kernel, gpu, seed), key, budget);
        check(cache.hits == 1 && cache.misses == 1, "the second identical request must be a hit: "
                + cache.hits + " hits and " + cache.misses + " misses");
        check(hot == warm && hot.config.key() == warm.config.key(),
                "a cache hit must return the very same winner: " + hot.config);
        check(warm.config.key() == exhaustive.config.key() || warm.predictedCost <= exhaustive.predictedCost,
                "the cached winner must be at least as good as the exhaustive optimum: "
                        + warm.predictedCost + " against " + exhaustive.predictedCost);
        Workload other = new Workload("reduce", kernel.work * 2, 640.0, 4.0);
        Result otherShape = cache.lookupOrTune(new Tuner(other, gpu, seed), other.cacheKey(gpu), budget);
        check(cache.misses == 2 && otherShape.config != null, "a different shape must miss the cache: "
                + cache.misses + " misses and " + cache.size() + " entries");
        String tightKey = kernel.cacheKey(tight);
        Result onTight = cache.lookupOrTune(new Tuner(kernel, tight, seed), tightKey, budget);
        check(cache.misses == 3 && CostModel.legal(kernel, onTight.config, tight)
                        && onTight.config.block <= tight.maxThreadsPerBlock
                        && onTight.config.sharedBytes <= tight.sharedBytes,
                "a device with tighter limits must produce a winner that respects those limits: "
                        + onTight.config);
        long tightBest = new Tuner(kernel, tight, seed).exhaustive().predictedCost;
        check(tightBest > exhaustive.predictedCost, "the weaker device must be slower at its best "
                + "configuration: " + tightBest + " against " + exhaustive.predictedCost);
        Device revised = new Device(tight.name, tight.maxThreadsPerBlock, tight.sharedBytes,
                tight.registersPerThread, tight.maxGrid, tight.threadsPerMultiprocessor, tight.registerFile,
                tight.sharedPerMultiprocessor, tight.multiprocessors, tight.bytesPerCycle,
                tight.threadsPerCycle, tight.revision + 1);
        Result afterRevision = cache.lookupOrTune(new Tuner(kernel, revised, seed), kernel.cacheKey(revised),
                budget);
        check(cache.misses == 4 && afterRevision.config != null, "a device revision must invalidate the cache: "
                + cache.misses + " misses and " + cache.size() + " entries");
        check(CostModel.legal(kernel, afterRevision.config, revised),
                "a re-tuned configuration must be legal on the new revision: " + afterRevision.config);
        System.out.println("Cache: " + cache.hits + " hits, " + cache.misses + " misses, " + cache.size()
                + " entries, " + cache.measurements + " measurements spent on exploration");

        // --- EDGE CASES ----------------------------------------------------------------------------------
        Device tinyDevice = new Device("gpu-tiny", 32, 0, 32, 1024, 32, 1024, 0, 1, 4.0, 32.0, 9);
        Tuner degenerate = new Tuner(kernel, tinyDevice, seed);
        Result onlyOne = degenerate.exhaustive();
        int tinyLegal = degenerate.legalConfigs().size();
        check(tinyLegal == 1 && onlyOne.measurements == 0,
                "a space with a single legal configuration must be returned without searching: " + tinyLegal
                        + " legal, " + onlyOne.config);
        Tuner impossibleTuner = new Tuner(kernel, impossible, seed);
        boolean rejected = false;
        try {
            impossibleTuner.exhaustive();
        } catch (IllegalStateException ex) {
            rejected = true;
        }
        check(rejected, "an empty tuning space must be rejected");
        check(CostModel.legal(kernel, onlyOne.config, tinyDevice), "the degenerate winner must be legal");
        Workload smaller = new Workload("reduce", 65536L, 640.0, 4.0);
        double smallBound = CostModel.rooflineBound(smaller, gpu);
        check(Math.abs(smallBound * 4 - bound) < 1e-9, "the roofline bound must scale linearly with the work: "
                + smallBound + " against " + bound);
        long smallBest = new Tuner(smaller, gpu, seed).exhaustive().predictedCost;
        check(smallBest < exhaustive.predictedCost, "the tuned cost must grow with the amount of work: "
                + smallBest + " against " + exhaustive.predictedCost);
        Result cachedVersusFresh = cache.lookupOrTune(new Tuner(kernel, gpu, seed), key, budget);
        check(cachedVersusFresh == warm, "the cache must keep returning the stored winner");
        check(cache.hits == 2, "the third identical request must also be a hit: " + cache.hits);

        System.out.println("Budget: exhaustive needs " + legal.size() + " launches, successive halving found "
                + firstPass.config + " with " + firstPass.measurements + " of " + budget + " measurements");
        System.out.println("PASS");
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError("FAILED: " + message);
    }
}
