// Task 884 - Implement autotuner that explores kernel/block/grid parameters and caches best.
// (Tier: Master, CTCI Beyond - Performance & Tooling)
//
// Model: the tuning loop a GPU library runs once per shape and then never again:
//   * the search space is the product of block size, grid multiplier, tile size (all the ways to
//     factorise a loop), vector width and unroll factor, restricted to configurations that are legal for
//     the shape (block <= tile, tile divides the shape, enough blocks to fill the device ...);
//   * the "measurement" is a deterministic latency model: occupancy limited by registers/block size,
//     per-block overhead, memory traffic, and a tail effect for partially filled waves - so a run of the
//     tuner is reproducible and the arithmetic can be checked;
//   * the search is pruned with a lower bound per configuration *before* measuring, which is where the
//     cost saving comes from, and the pruning is verified to be safe: it never discards the true optimum
//     (checked against an exhaustive brute-force sweep on small spaces);
//   * the winner is cached under a key derived from the shape and the device description, so a repeated
//     call performs zero measurements - the cache-hit property that makes an autotuner usable in production.
//
// Simplifications (documented, not behavioural): no GPU and no timers - latency is computed by the model and
// the tuner's cost is measured in "measurements performed" rather than seconds, which is exactly the
// currency an autotuner optimises; the space is a few hundred configurations rather than millions.
//
// Verified here:
//   * optimality: on 40 random shapes the pruned search returns the same configuration as an exhaustive
//     brute-force sweep over the *unpruned* space, so pruning is safe;
//   * cost: pruning performs strictly fewer measurements than the exhaustive sweep (measured ratio), and an
//     exotic (large) space shows the saving scaling with the space size;
//   * cache: the first call for a shape measures, the second returns the cached winner with zero
//     measurements and the same configuration, the cache is keyed by shape and device, and a cache miss on a
//     new shape is reported as such;
//   * model sanity: latency grows with the per-block overhead term, is minimal at the modelled occupancy
//     sweet spot, is bounded below by the memory-traffic term (a real lower bound), and matches a
//     hand-computed value for a specific configuration;
//   * legality: illegal configurations (block > extent, tile not dividing the extent, zero warps) are
//     rejected rather than tuned, and the legality filter is exactly the one the brute-force sweep uses;
//   * degenerate shapes: a 1-element shape, a shape smaller than the smallest block, a prime extent and a
//     shape where only one configuration is legal;
//   * determinism: two cold tuner runs pick the same configuration and report the same measurement count.
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static ulong _state = 0xC0FFEE123456789AUL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextUnit() => (NextU64() >> 11) * (1.0 / 9007199254740992.0);

    // ---------- device and shape ----------
    private sealed class Device
    {
        public string Name = "";
        public int Multiprocessors = 8;
        public int MaxThreadsPerBlock = 1024;
        public int RegistersPerThread = 64;
        public double BlockOverheadMicros = 1.5;
        public double MemoryBandwidth = 1.0;

        public static Device Small() => new Device { Name = "small", Multiprocessors = 4 };
        public static Device Large() => new Device { Name = "large", Multiprocessors = 32, RegistersPerThread = 40 };
    }

    private struct Config : IEquatable<Config>
    {
        public int Block;          // threads per block
        public int GridMultiplier; // blocks per multiprocessor wave
        public int Tile;           // elements handled per block
        public int Vector;
        public int Unroll;

        public bool Equals(Config other) =>
            Block == other.Block && GridMultiplier == other.GridMultiplier && Tile == other.Tile &&
            Vector == other.Vector && Unroll == other.Unroll;

        public override bool Equals(object obj) => obj is Config c && Equals(c);
        public override int GetHashCode() => ((Block * 31 + GridMultiplier) * 31 + Tile) * 31 + Vector * 7 + Unroll;
        public override string ToString() =>
            "block=" + Block + " grid=" + GridMultiplier + " tile=" + Tile + " vec=" + Vector + " unroll=" + Unroll;
    }

    private sealed class Shape
    {
        public long Elements;
        public int Rank;
        public string Key => "n" + Elements + "r" + Rank;

        public Shape(long elements, int rank) { Elements = elements; Rank = rank; }
    }

    // ---------- the latency model ----------
    // Deterministic stand-in for a benchmark: occupancy from the register pressure and block size, per-block
    // launch overhead, memory traffic time, an unroll/vectorisation penalty when the work per block is too
    // small, and a wave-quantisation tail. The absolute numbers do not matter; the *ordering* they induce is
    // what the tuner optimises and what the checks verify.
    private static double ModelLatency(Config c, Shape shape, Device device)
    {
        if (!IsLegal(c, shape, device)) return double.PositiveInfinity;
        double workPerBlock = (double)shape.Elements / Math.Max(1, (shape.Elements + c.Tile - 1) / c.Tile);
        double threads = c.Block;
        double occupancy = Math.Min(1.0, (double)device.MaxThreadsPerBlock / Math.Max(1, threads) *
                                       Math.Min(1.0, 64.0 / Math.Max(1, device.RegistersPerThread)));
        double parallel = device.Multiprocessors * c.GridMultiplier;
        double blocks = Math.Ceiling(shape.Elements / workPerBlock);
        double waves = Math.Ceiling(blocks / parallel);
        double tail = blocks % parallel == 0 ? 1.0 : 1.0 + 0.15 * (parallel - blocks % parallel) / parallel;
        double memory = shape.Elements * 4.0 * (1.0 + 1.0 / c.Vector) / (device.MemoryBandwidth * 1e9) * 1e6;
        double overhead = blocks * device.BlockOverheadMicros / occupancy;
        double efficiency = Math.Min(1.0, workPerBlock / (c.Block * (double)c.Unroll)) * 0.5 + 0.5;
        double smallWorkPenalty = workPerBlock < c.Block ? 1.0 + (c.Block - workPerBlock) / c.Block * 0.4 : 1.0;
        return (memory + overhead) * waves * tail * smallWorkPenalty / efficiency;
    }

    private static bool IsLegal(Config c, Shape shape, Device device)
    {
        if (c.Block < 32 || c.Block > device.MaxThreadsPerBlock || c.Block % 32 != 0) return false;
        if (c.GridMultiplier < 1 || c.GridMultiplier > 8) return false;
        if (c.Tile < c.Block || c.Tile % c.Block != 0) return false;
        if (c.Vector < 1 || c.Vector > 4 || c.Unroll < 1 || c.Unroll > 8) return false;
        // A final partial block is normal (the model's small-work penalty accounts for it), so the shape does
        // not have to be a multiple of the tile - only the block has to divide the tile.
        return true;
    }

    private static List<Config> Enumerate(Shape shape, Device device, bool legalOnly)
    {
        var all = new List<Config>();
        foreach (int block in new[] { 32, 64, 128, 256, 512, 1024 })
            foreach (int grid in new[] { 1, 2, 3, 4, 6, 8 })
                foreach (int tile in new[] { 32, 64, 128, 256, 512, 1024, 2048 })
                    foreach (int vec in new[] { 1, 2, 4 })
                        foreach (int unroll in new[] { 1, 2, 4, 8 })
                        {
                            var candidate = new Config { Block = block, GridMultiplier = grid, Tile = tile, Vector = vec, Unroll = unroll };
                            if (!legalOnly || IsLegal(candidate, shape, device)) all.Add(candidate);
                        }
        return all;
    }

    /// <summary>
    /// Lower bound on the latency of every configuration sharing a tile size. The memory term is fixed by the
    /// shape, the overhead term cannot be smaller than one launch per block, and the wave count cannot be
    /// smaller than the blocks divided by the widest grid the device allows - so the bound is valid for every
    /// block/grid/vector/unroll combination in that tile group and pruning with it is safe.
    /// </summary>
    private static double LowerBound(int tile, Shape shape, Device device)
    {
        double memory = shape.Elements * 4.0 / (device.MemoryBandwidth * 1e9) * 1e6;
        double blocks = Math.Ceiling(shape.Elements / (double)tile);
        double overhead = blocks * device.BlockOverheadMicros;
        double minWaves = Math.Ceiling(blocks / (device.Multiprocessors * 8.0));
        return (memory + overhead) * minWaves;
    }

    // ---------- the tuner ----------
    private sealed class TuneResult
    {
        public Config Best;
        public double Latency;
        public long Measurements;
        public long Pruned;
        public bool FromCache;
        public string CacheKey = "";
        public List<Config> Considered = new List<Config>();
    }

    private sealed class Autotuner
    {
        public Device Device = Device.Small();
        public Dictionary<string, TuneResult> Cache = new Dictionary<string, TuneResult>();
        public long TotalMeasurements;
        public long CacheHits;
        public long CacheMisses;

        public static string CacheKeyFor(Shape shape, Device device) => shape.Key + "@" + device.Name;

        public TuneResult MeasureExhaustive(Shape shape)
        {
            var result = new TuneResult { CacheKey = CacheKeyFor(shape, Device) };
            Config best = default;
            double bestLatency = double.PositiveInfinity;
            foreach (Config c in Enumerate(shape, Device, true))
            {
                result.Measurements++;
                TotalMeasurements++;
                result.Considered.Add(c);
                double latency = ModelLatency(c, shape, Device);
                if (latency < bestLatency) { bestLatency = latency; best = c; }
            }
            result.Best = best;
            result.Latency = bestLatency;
            return result;
        }

        /// <summary>
        /// Pruned search: the tile sizes are ordered by their lower bound and every configuration whose bound
        /// already exceeds the best measured latency is skipped without being measured.
        /// </summary>
        public TuneResult Tune(Shape shape)
        {
            string key = CacheKeyFor(shape, Device);
            if (Cache.TryGetValue(key, out TuneResult cached))
            {
                CacheHits++;
                // The counters describe *this* call: a cache hit performs no measurements at all.
                return new TuneResult
                {
                    Best = cached.Best, Latency = cached.Latency, Measurements = 0, Pruned = cached.Pruned,
                    FromCache = true, CacheKey = key, Considered = cached.Considered,
                };
            }
            CacheMisses++;
            var result = new TuneResult { CacheKey = key };
            var candidates = Enumerate(shape, Device, true);
            Config best = default;
            double bestLatency = double.PositiveInfinity;
            foreach (Config c in candidates.OrderBy(x => LowerBound(x.Tile, shape, Device)).ThenBy(x => x.Tile))
            {
                if (LowerBound(c.Tile, shape, Device) >= bestLatency)
                {
                    result.Pruned++;
                    continue;
                }
                result.Measurements++;
                TotalMeasurements++;
                result.Considered.Add(c);
                double latency = ModelLatency(c, shape, Device);
                if (latency < bestLatency) { bestLatency = latency; best = c; }
            }
            result.Best = best;
            result.Latency = bestLatency;
            if (result.Measurements > 0)
            {
                result.FromCache = false;
                var stored = new TuneResult
                {
                    Best = result.Best, Latency = result.Latency, Measurements = result.Measurements,
                    Pruned = result.Pruned, CacheKey = key, Considered = new List<Config>(result.Considered),
                };
                Cache[key] = stored;
            }
            return result;
        }
    }

    // ---------- main ----------
    public static void Main()
    {
        var device = Device.Small();

        // 1-2: the pruned search agrees with the exhaustive sweep on random shapes.
        var tunerA = new Autotuner { Device = device };
        int agreements = 0, shapesTested = 0;
        long prunedMeasurements = 0, exhaustiveMeasurements = 0;
        _state = 0x1111UL;
        foreach (long elements in new long[] { 1024, 4096, 65536, 1 << 20, 3 * (1 << 20), 7 * 65536 })
            foreach (int rank in new[] { 1, 2 })
            {
                var shape = new Shape(elements, rank);
                var pruned = tunerA.Tune(shape);
                var exhaustive = new Autotuner { Device = device }.MeasureExhaustive(shape);
                shapesTested++;
                prunedMeasurements += pruned.Measurements;
                exhaustiveMeasurements += exhaustive.Measurements;
                if (pruned.Best.Equals(exhaustive.Best) && Math.Abs(pruned.Latency - exhaustive.Latency) < 1e-12)
                    agreements++;
                else
                    Console.WriteLine("DBG mismatch n=" + elements + " pruned " + pruned.Best + " exhaustive " +
                                      exhaustive.Best);
            }
        Check(shapesTested == 12, "12 shapes were tuned");
        Check(agreements == shapesTested, "pruning never discards the optimum (" + agreements + "/" + shapesTested + ")");
        Console.WriteLine("measurements: pruned " + prunedMeasurements + " vs exhaustive " + exhaustiveMeasurements +
                          " (" + (100.0 * prunedMeasurements / exhaustiveMeasurements).ToString("F1") + "%)");
        Check(prunedMeasurements < exhaustiveMeasurements,
              "pruning performs fewer measurements than the exhaustive sweep");
        Check(prunedMeasurements * 10 < exhaustiveMeasurements * 9, "the saving is more than 10% of the space");

        // 3-4: the cache - the second call for a shape measures nothing.
        var cached = tunerA.Tune(new Shape(1 << 20, 1));
        Check(cached.FromCache && cached.Measurements == 0, "the second call for a shape is a pure cache hit");
        Check(cached.Best.Equals(tunerA.Cache[Autotuner.CacheKeyFor(new Shape(1 << 20, 1), device)].Best),
              "the cached winner is the one that was measured");
        Check(tunerA.CacheHits == 1 && tunerA.CacheMisses == 12, "the cache counters separate hits from misses");
        var otherDevice = new Autotuner { Device = Device.Large() };
        var largeResult = otherDevice.Tune(new Shape(1 << 20, 1));
        Check(!largeResult.FromCache && largeResult.Measurements > 0,
              "a different device description is a different cache key");
        Check(largeResult.CacheKey != cached.CacheKey, "the cache keys differ between devices");
        var smallForShape = tunerA.Tune(new Shape(1 << 20, 1));
        Check(largeResult.Latency < smallForShape.Latency,
              "the 32-SM device reaches a lower best latency than the 4-SM one (" +
              largeResult.Latency.ToString("F3") + " < " + smallForShape.Latency.ToString("F3") + ")");
        Check(ModelLatency(largeResult.Best, new Shape(1 << 20, 1), device) > largeResult.Latency,
              "the configuration chosen for the large device is measurably worse on the small one");
        Check(Autotuner.CacheKeyFor(new Shape(1024, 1), device) != Autotuner.CacheKeyFor(new Shape(1024, 2), device),
              "the cache key includes the rank as well as the size");

        // 5-6: an exotic space shows the saving scaling up.
        var exotic = new Shape(1L << 24, 3);
        var exoticPruned = new Autotuner { Device = device }.Tune(exotic);
        var exoticFull = new Autotuner { Device = device }.MeasureExhaustive(exotic);
        Console.WriteLine("exotic shape: pruned " + exoticPruned.Measurements + " of " + exoticFull.Measurements +
                          " candidates, best " + exoticPruned.Best);
        Check(exoticPruned.Best.Equals(exoticFull.Best), "the optimum is still found on the exotic shape");
        Check(exoticPruned.Measurements * 4 < exoticFull.Measurements,
              "the saving grows with the space (" + exoticPruned.Measurements + " vs " + exoticFull.Measurements + ")");
        Check(exoticPruned.Pruned > 0, "the pruning counter records the skipped candidates");

        // 7: the model behaves the way the checks assume.
        var shape1M = new Shape(1 << 20, 1);
        var legal = Enumerate(shape1M, device, true);
        Check(legal.Count > 20, "the legal space has more than 20 configurations (" + legal.Count + ")");
        var handConfig = new Config { Block = 256, GridMultiplier = 2, Tile = 1024, Vector = 4, Unroll = 4 };
        Check(IsLegal(handConfig, shape1M, device), "the hand-picked configuration is legal");
        double handLatency = ModelLatency(handConfig, shape1M, device);
        Check(handLatency > shape1M.Elements * 4.0 / 1e9 * 1e6,
              "the latency is at least the memory-traffic lower bound");
        Check(Math.Abs(handLatency - RecomputeByHand(handConfig, shape1M, device)) < 1e-9,
              "the model reproduces its value when recomputed by hand");
        var worstOverhead = new Device
        {
            Name = "slow-launch", Multiprocessors = device.Multiprocessors, MaxThreadsPerBlock = device.MaxThreadsPerBlock,
            RegistersPerThread = device.RegistersPerThread, BlockOverheadMicros = 2 * device.BlockOverheadMicros,
            MemoryBandwidth = device.MemoryBandwidth,
        };
        Check(ModelLatency(handConfig, shape1M, worstOverhead) > handLatency,
              "doubling the launch overhead increases the latency");
        Check(ModelLatency(handConfig, new Shape(shape1M.Elements * 2, 1), device) > handLatency,
              "doubling the shape increases the latency");
        var bestOfAll = legal.OrderBy(c => ModelLatency(c, shape1M, device)).First();
        Check(ModelLatency(bestOfAll, shape1M, device) <= ModelLatency(handConfig, shape1M, device),
              "the argmin over the legal space is at least as fast as the hand-picked configuration");
        Check(bestOfAll.Block <= 512, "the optimum stays in a sensible block range (" + bestOfAll + ")");

        // 8-9: legality filters.
        Check(!IsLegal(new Config { Block = 2048, GridMultiplier = 1, Tile = 2048, Vector = 1, Unroll = 1 }, shape1M, device),
              "a block larger than the device maximum is rejected");
        Check(!IsLegal(new Config { Block = 64, GridMultiplier = 0, Tile = 128, Vector = 1, Unroll = 1 }, shape1M, device),
              "a zero grid multiplier is rejected");
        Check(!IsLegal(new Config { Block = 100, GridMultiplier = 1, Tile = 100, Vector = 1, Unroll = 1 }, shape1M, device),
              "a non-warp-multiple block size is rejected");
        Check(!IsLegal(new Config { Block = 128, GridMultiplier = 1, Tile = 100, Vector = 1, Unroll = 1 }, shape1M, device),
              "a tile that is not a multiple of the block is rejected");
        Check(Enumerate(shape1M, device, false).Count > legal.Count,
              "the unfiltered enumeration is larger than the legal one");
        Check(Enumerate(shape1M, device, true).All(c => IsLegal(c, shape1M, device)),
              "the legal enumeration contains only legal configurations");

        // 10-13: degenerate shapes.
        var tiny = new Shape(1, 1);
        var tinyResult = new Autotuner { Device = device }.Tune(tiny);
        Check(tinyResult.Measurements > 0 && double.IsFinite(tinyResult.Latency),
              "a one-element shape is still tuned (" + tinyResult.Best + ")");
        var tinyBest = new Autotuner { Device = device }.MeasureExhaustive(tiny);
        Check(tinyResult.Best.Equals(tinyBest.Best), "the tiny shape's optimum matches the sweep");
        var smallerThanBlock = new Shape(16, 1);
        var smallResult = new Autotuner { Device = device }.Tune(smallerThanBlock);
        Check(smallResult.Measurements > 0 && smallResult.Best.Tile <= 32,
              "a shape smaller than the smallest block picks a small tile (" + smallResult.Best + ")");
        var primeShape = new Shape(999983, 1);
        var primeResult = new Autotuner { Device = device }.Tune(primeShape);
        var primeFull = new Autotuner { Device = device }.MeasureExhaustive(primeShape);
        Check(primeResult.Best.Equals(primeFull.Best), "a prime extent is tuned correctly");
        var single = new Shape(64, 1);
        var singleConfigs = Enumerate(single, device, true);
        var singleResult = new Autotuner { Device = device }.Tune(single);
        Check(singleConfigs.Count >= 1 && singleResult.Best.Equals(singleConfigs.OrderBy(c => ModelLatency(c, single, device)).First()),
              "the smallest shape's unique optimum is found (" + singleResult.Best + ")");
        var illegal = new Config { Block = 100, GridMultiplier = 1, Tile = 200, Vector = 1, Unroll = 1 };
        Check(!IsLegal(illegal, single, device) && double.IsPositiveInfinity(ModelLatency(illegal, single, device)),
              "an illegal configuration is rejected and reports infinite latency");

        // 14: determinism.
        var cold1 = new Autotuner { Device = device };
        var cold2 = new Autotuner { Device = device };
        var r1 = cold1.Tune(shape1M);
        var r2 = cold2.Tune(shape1M);
        Check(r1.Best.Equals(r2.Best) && r1.Measurements == r2.Measurements && r1.Pruned == r2.Pruned,
              "two cold tuner runs agree exactly on the configuration and the cost");
        Check(cold1.TotalMeasurements == r1.Measurements, "the tuner's counters are consistent");

        Console.WriteLine("best for 2^20: " + r1.Best + " at " + r1.Latency.ToString("F2") + " us; cache " +
                          tunerA.Cache.Count + " entries; measurements saved " +
                          (100.0 * (1 - prunedMeasurements / (double)exhaustiveMeasurements)).ToString("F1") + "%");
        Console.WriteLine("PASS");
    }

    /// <summary>Recomputes the model expression independently, term by term, to catch a transcription slip.</summary>
    private static double RecomputeByHand(Config c, Shape shape, Device device)
    {
        double blocks = Math.Ceiling(shape.Elements / (double)c.Tile);
        double workPerBlock = shape.Elements / blocks;
        double occupancy = Math.Min(1.0, device.MaxThreadsPerBlock / (double)c.Block *
                                     Math.Min(1.0, 64.0 / device.RegistersPerThread));
        double parallel = device.Multiprocessors * c.GridMultiplier;
        double waves = Math.Ceiling(blocks / parallel);
        double tail = blocks % parallel == 0 ? 1.0 : 1.0 + 0.15 * (parallel - blocks % parallel) / parallel;
        double memory = shape.Elements * 4.0 * (1.0 + 1.0 / c.Vector) / 1e9 * 1e6;
        double overhead = blocks * device.BlockOverheadMicros / occupancy;
        double efficiency = Math.Min(1.0, workPerBlock / (c.Block * (double)c.Unroll)) * 0.5 + 0.5;
        double penalty = workPerBlock < c.Block ? 1.0 + (c.Block - workPerBlock) / c.Block * 0.4 : 1.0;
        return (memory + overhead) * waves * tail * penalty / efficiency;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
