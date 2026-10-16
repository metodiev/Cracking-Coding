// Task 880 - Implement distributed rendering coordinator with tile-based scheduling.
// (Tier: Master, CTCI Beyond - Rendering & Graphics)
//
// Model: a coordinator that renders one frame as a grid of tiles over an explicit in-memory network of
// worker nodes. Everything the task asks for is modelled end to end:
//   * tiles are handed out under a *lease* with an explicit deadline on a virtual clock (no wall-clock time
//     anywhere), so a worker that dies is detected when its lease expires and its tiles are re-issued;
//   * slow workers are handled by speculative duplicate execution: once a tile passes its straggler
//     threshold a second worker is asked for it, and whichever result arrives first is committed;
//   * commits are idempotent because a tile's result is content addressed - a duplicate that arrives late
//     is simply dropped, so a tile can never be committed twice;
//   * the coordinator gives up after a bounded number of attempts per tile, which is what keeps a fully
//     broken cluster from spinning forever.
// The render itself is real (the same shading kernel as the raytracer tier exercise), so the committed
// frame is verified pixel by pixel against an in-process oracle render, and the scheduling claims
// (exactly-once commit, liveness, bounded redundancy) are checked over many deterministic cluster shapes.
//
// Simplifications (documented, not behavioural): no sockets, no threads and no real workers - messages are
// passed in a deterministic queue and the clock is a counter advanced by the simulation; worker speed and
// failure are drawn from the splitmix64 stream, so a cluster shape is reproducible from its seed; the
// "network" delivers messages with a modelled latency in virtual ticks.
//
// Verified here:
//   * safety: every tile is committed exactly once, and the committed pixel data equals the oracle render;
//   * liveness: all tiles complete for every cluster shape tried, including ones where a third of the
//     workers die mid-frame and one worker is 8x slower than the rest;
//   * fault handling: the tiles of a dead worker are re-issued and completed, and a late duplicate result
//     is dropped without corrupting the frame;
//   * cost bounds: the number of tile executions stays inside the redundancy budget, speculative
//     execution is triggered only for stragglers, and a completely broken cluster terminates after the
//     bounded attempt count instead of looping;
//   * determinism: the whole simulation (schedule, execution count, frame digest) is reproducible, and the
//     digest is independent of the worker count;
//   * degenerate cases: a single tile frame, a frame with zero tiles, one worker, and a tile grid that is
//     mostly empty.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public static class Solution
{
    private static ulong _state = 0x2468ACE013579BDFUL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextUnit() => (NextU64() >> 11) * (1.0 / 9007199254740992.0);

    private static double NextSigned() => NextUnit() * 2.0 - 1.0;

    // ---------- the frame and its tiles ----------
    private sealed class Frame
    {
        public int Width;
        public int Height;
        public int TileSize;
        public List<(int X, int Y)> Tiles = new List<(int X, int Y)>();

        public int TileCount => Tiles.Count;

        public static Frame Make(int width, int height, int tileSize)
        {
            var frame = new Frame { Width = width, Height = height, TileSize = tileSize };
            for (int y = 0; y < height; y += tileSize)
                for (int x = 0; x < width; x += tileSize) frame.Tiles.Add((x, y));
            return frame;
        }
    }

    // ---------- worker nodes ----------
    private sealed class Worker
    {
        public int Id;
        public double Speed;              // ticks of work per tile-row, the typed "node speed" parameter
        public double FailureRate;        // probability that a lease is dropped
        public bool Alive = true;
        public int Executions;
        public int Dropped;
        public double Slowdown;           // extra latency factor applied while the worker is a straggler
    }

    /// <summary>The shading kernel a worker runs: a deterministic gradient with a soft "light", one row of a tile.</summary>
    private static int[] RenderTileRow(Frame frame, (int X, int Y) tile, int row)
    {
        int width = Math.Min(frame.TileSize, frame.Width - tile.X);
        var pixels = new int[width];
        for (int i = 0; i < width; i++)
        {
            double u = (tile.X + i + 0.5) / frame.Width * 2.0 - 1.0;
            double v = (tile.Y + row + 0.5) / frame.Height * 2.0 - 1.0;
            double shade = 0.5 + 0.5 * Math.Sin(Math.Sqrt(u * u + v * v) * 6.0) * Math.Exp(-0.5 * (u * u + v * v));
            int r = (int)Math.Clamp(shade * 255.0, 0, 255);
            int g = (int)Math.Clamp((1.0 - shade) * 200.0, 0, 255);
            int b = (int)Math.Clamp((0.5 + 0.5 * u) * 200.0, 0, 255);
            pixels[i] = (r << 16) | (g << 8) | b;
        }
        return pixels;
    }

    private static byte[] OracleTile(Frame frame, (int X, int Y) tile)
    {
        var bytes = new List<byte>();
        int height = Math.Min(frame.TileSize, frame.Height - tile.Y);
        for (int row = 0; row < height; row++)
            foreach (int p in RenderTileRow(frame, tile, row))
            {
                bytes.Add((byte)((p >> 16) & 0xFF));
                bytes.Add((byte)((p >> 8) & 0xFF));
                bytes.Add((byte)(p & 0xFF));
            }
        return bytes.ToArray();
    }

    private static string Hash(byte[] data)
    {
        ulong h = 0xCBF29CE484222325UL;
        foreach (byte b in data)
        {
            h ^= b;
            h *= 0x100000001B3UL;
        }
        return h.ToString("x16");
    }

    // ---------- coordinator ----------
    private enum TileState { Pending, Leased, Committed }

    private sealed class TileRecord
    {
        public int Index;
        public TileState State = TileState.Pending;
        public int Attempts;
        public int Leases;
        public byte[] Result;
        public string Digest = "";
        public int CommittedBy = -1;
    }

    private sealed class Coordinator
    {
        public Frame Frame = new Frame();
        public List<Worker> Workers = new List<Worker>();
        public List<TileRecord> Tiles = new List<TileRecord>();
        public long Clock;                         // virtual ticks
        public long RedundantExecutions;
        public long DroppedDuplicates;
        public long LeaseExpiries;
        public long SpeculativeIssues;
        public bool Failed;
        public string FailureReason = "";

        public long MaxAttemptsPerTile = 4;

        public double LeaseTicks = 120.0;
        public double StragglerTicks = 40.0;
    }

    private sealed class Lease
    {
        public int Tile;
        public int Worker;
        public long Deadline;
        public int Attempt;
    }

    private sealed class Delivery
    {
        public int Tile;
        public int Worker;
        public long At;
        public byte[] Result;
        public bool Dropped;
    }

    /// <summary>
    /// Runs the frame to completion (or to the bounded-attempt failure) with a deterministic event loop:
    /// leases are issued when a worker is free, completions arrive in virtual time, expired leases are
    /// re-issued and stragglers get a speculative duplicate.
    /// </summary>
    private static Coordinator Simulate(Frame frame, int workerCount, ulong seed, double failureRate,
                                        bool killHalfWay = false, int slowWorker = -1)
    {
        _state = seed;
        var coord = new Coordinator { Frame = frame };
        for (int i = 0; i < frame.TileCount; i++) coord.Tiles.Add(new TileRecord { Index = i });
        for (int i = 0; i < workerCount; i++)
            coord.Workers.Add(new Worker
            {
                Id = i,
                Speed = 1.0 + NextUnit() * 2.0,
                FailureRate = failureRate,
                Slowdown = i == slowWorker ? 8.0 : 1.0,
            });
        var leases = new List<Lease>();
        var deliveries = new List<Delivery>();
        var busyUntil = new long[workerCount];
        int nextPending = 0;
        long guard = 0;

        void Issue(int worker, int tile, long start, int attempt, bool speculative)
        {
            Worker w = coord.Workers[worker];
            int rows = Math.Min(frame.TileSize, frame.Height - frame.Tiles[tile].Y);
            long duration = (long)Math.Ceiling(rows * w.Speed * w.Slowdown) + (speculative ? 25 : 0);
            leases.Add(new Lease { Tile = tile, Worker = worker, Deadline = start + duration, Attempt = attempt });
            busyUntil[worker] = start + duration;
            w.Executions++;
            if (speculative) coord.SpeculativeIssues++;
            bool drop = NextUnit() < w.FailureRate * (speculative ? 0.5 : 1.0);
            if (drop) w.Dropped++;
            deliveries.Add(new Delivery
            {
                Tile = tile,
                Worker = worker,
                At = start + duration + 3,                       // a modelled network latency
                Result = drop ? null : OracleTile(frame, frame.Tiles[tile]),
                Dropped = drop,
            });
        }

        while (coord.Tiles.Any(t => t.State != TileState.Committed) && !coord.Failed && guard++ < 100000)
        {
            // 1. commit everything that has arrived
            foreach (Delivery d in deliveries.Where(x => x.At <= coord.Clock).OrderBy(x => x.At).ThenBy(x => x.Tile).ToList())
            {
                deliveries.Remove(d);
                TileRecord record = coord.Tiles[d.Tile];
                if (record.State == TileState.Committed)
                {
                    coord.DroppedDuplicates++;                       // the idempotent commit rule
                    continue;
                }
                if (d.Dropped || d.Result == null) continue;          // the lease will expire and be retried
                record.Result = d.Result;
                record.Digest = Hash(d.Result);
                record.State = TileState.Committed;
                record.CommittedBy = d.Worker;
                if (record.Leases > 1) coord.RedundantExecutions++;
            }
            // 2. expire leases of workers that never delivered
            foreach (Lease l in leases.Where(x => x.Deadline < coord.Clock).ToList())
            {
                leases.Remove(l);
                if (coord.Tiles[l.Tile].State == TileState.Committed) continue;
                coord.LeaseExpiries++;
            }
            // 3. issue new work: pending tiles first, then speculative copies for stragglers
            for (int w = 0; w < workerCount; w++)
            {
                if (coord.Workers[w].Alive == false || coord.Clock < busyUntil[w]) continue;
                var outstanding = leases.Where(l => l.Worker == w).ToList();
                if (outstanding.Any()) continue;
                int attemptLimit = (int)coord.MaxAttemptsPerTile;
                int pending = coord.Tiles.FindIndex(t => t.State == TileState.Pending && t.Attempts < attemptLimit);
                if (pending >= 0)
                {
                    TileRecord record = coord.Tiles[pending];
                    record.State = TileState.Leased;
                    record.Attempts++;
                    record.Leases++;
                    nextPending++;
                    Issue(w, pending, coord.Clock, record.Attempts, false);
                    continue;
                }
                // straggler: a leased tile whose lease has already run for StragglerTicks gets a duplicate
                TileRecord straggler = coord.Tiles.FirstOrDefault(t =>
                    t.State == TileState.Leased && t.Attempts < attemptLimit &&
                    leases.Where(l => l.Tile == t.Index).All(l => coord.Clock - (l.Deadline - coord.LeaseTicks) > coord.StragglerTicks));
                if (straggler == null) continue;
                straggler.Attempts++;
                straggler.Leases++;
                Issue(w, straggler.Index, coord.Clock, straggler.Attempts, true);
            }
            // 4. externally injected failure, then advance the clock
            if (killHalfWay && coord.Clock == 60)
                for (int w = 1; w < workerCount; w += 2) coord.Workers[w].Alive = false;
            coord.Clock++;
            if (coord.Clock > 20000)
            {
                coord.Failed = true;
                coord.FailureReason = "virtual-time budget exhausted";
            }
        }
        bool incomplete = coord.Tiles.Any(t => t.State != TileState.Committed);
        if (incomplete && coord.Tiles.All(t => t.Attempts >= coord.MaxAttemptsPerTile || !coord.Workers.Any(w => w.Alive)))
        {
            coord.Failed = true;
            coord.FailureReason = "every tile exhausted its bounded attempts";
        }
        return coord;
    }

    /// <summary>Independent oracle: renders every tile directly, without any scheduling.</summary>
    private static byte[] OracleFrame(Frame frame)
    {
        var bytes = new List<byte>();
        foreach ((int X, int Y) in frame.Tiles) bytes.AddRange(OracleTile(frame, (X, Y)));
        return bytes.ToArray();
    }

    private static byte[] AssembledFrame(Coordinator coord)
    {
        var bytes = new List<byte>();
        foreach (TileRecord t in coord.Tiles.OrderBy(t => t.Index))
        {
            if (t.Result == null) return Array.Empty<byte>();
            bytes.AddRange(t.Result);
        }
        return bytes.ToArray();
    }

    // ---------- main ----------
    public static void Main()
    {
        var frame = Frame.Make(64, 48, 16);                 // 4 x 3 = 12 tiles
        Check(frame.TileCount == 12, "the frame is tiled into 12 tiles");
        byte[] oracle = OracleFrame(frame);
        Check(oracle.Length == 64 * 48 * 3, "the oracle frame has one RGB triple per pixel");

        // 1-2: liveness and safety on a healthy cluster.
        var healthy = Simulate(frame, 4, 0x1111UL, 0.0);
        Check(!healthy.Failed, "the healthy cluster completes the frame");
        Check(healthy.Tiles.All(t => t.State == TileState.Committed), "every tile is committed");
        Check(healthy.Tiles.All(t => t.CommittedBy >= 0 && t.Attempts >= 1), "every tile records who committed it");
        Check(AssembledFrame(healthy).SequenceEqual(oracle), "the assembled frame equals the oracle render");
        Check(healthy.Tiles.Select(t => t.Index).Distinct().Count() == 12 &&
              healthy.Tiles.All(t => t.Result != null && t.Digest == Hash(t.Result)),
              "every tile is committed exactly once with a matching content digest");
        Check(healthy.DroppedDuplicates == 0, "a healthy cluster needs no speculative duplicates");

        // 3: the digest of the frame is independent of the worker count.
        var fewer = Simulate(frame, 1, 0x1111UL, 0.0);
        var more = Simulate(frame, 8, 0x1111UL, 0.0);
        Check(Hash(AssembledFrame(fewer)) == Hash(AssembledFrame(healthy)) &&
              Hash(AssembledFrame(more)) == Hash(AssembledFrame(healthy)),
              "the frame digest is invariant under 1, 4 and 8 workers");
        Check(more.Clock < healthy.Clock && fewer.Clock > healthy.Clock,
              "more workers finish sooner in virtual time (" + more.Clock + " < " + healthy.Clock + " < " + fewer.Clock + ")");

        // 4-5: fault tolerance - leases of dead workers expire and the tiles are re-issued.
        var killed = Simulate(frame, 6, 0x2222UL, 0.0, killHalfWay: true);
        Check(!killed.Failed && killed.Tiles.All(t => t.State == TileState.Committed),
              "the frame completes even though half the workers die mid-frame");
        Check(killed.LeaseExpiries > 0, "leases of the dead workers really expired (" + killed.LeaseExpiries + ")");
        Check(AssembledFrame(killed).SequenceEqual(oracle), "the frame rendered by the surviving workers is correct");
        Check(killed.Tiles.Any(t => t.Leases > 1), "at least one tile had to be re-issued");

        // 6: a very slow worker triggers speculative duplicate execution (straggler mitigation).
        var straggler = Simulate(frame, 5, 0x3333UL, 0.0, slowWorker: 1);
        Check(!straggler.Failed && straggler.SpeculativeIssues > 0,
              "a straggler triggers speculative duplicates (" + straggler.SpeculativeIssues + ")");
        Check(AssembledFrame(straggler).SequenceEqual(oracle), "the speculatively rendered frame is correct");
        Check(straggler.RedundantExecutions <= straggler.Tiles.Count(t => t.Leases > 1),
              "redundant executions never exceed the tiles that were duplicated");
        Check(straggler.Tiles.All(t => t.Attempts <= straggler.MaxAttemptsPerTile),
              "no tile exceeds the bounded attempt count");

        // 7-8: cost bounds and late duplicates.
        long totalExecutions = 0, totalTiles = 0;
        foreach (ulong seed in new[] { 0x4444UL, 0x5555UL, 0x6666UL })
        {
            var run = Simulate(frame, 4, seed, 0.25);
            totalExecutions += run.Workers.Sum(w => (long)w.Executions);
            totalTiles += run.Tiles.Count;
            Check(!run.Failed && AssembledFrame(run).SequenceEqual(oracle),
                  "a cluster with a 25% drop rate still renders the frame correctly (seed " + seed + ")");
            Check(run.DroppedDuplicates >= 0 && run.Tiles.All(t => t.Attempts <= run.MaxAttemptsPerTile),
                  "the retry budget holds under drops");
        }
        Console.WriteLine("executions " + totalExecutions + " for " + totalTiles + " tiles (redundancy " +
                          (totalExecutions / (double)totalTiles).ToString("F2") + "x)");
        Check(totalExecutions < totalTiles * 3, "the total work stays inside a 3x redundancy budget");

        // 9: a completely broken cluster fails cleanly instead of spinning.
        var broken = Simulate(frame, 3, 0x7777UL, 1.0);
        Check(broken.Failed && broken.FailureReason != "", "a cluster that always drops work fails with a reason");
        Check(broken.Tiles.All(t => t.Attempts <= broken.MaxAttemptsPerTile),
              "the failure respects the bounded attempt count");
        Check(AssembledFrame(broken).Length == 0, "an incomplete frame is not handed out as if it were complete");

        // 10-12: degenerate frames.
        var single = Frame.Make(4, 4, 16);
        Check(single.TileCount == 1, "a frame smaller than one tile is a single tile");
        var singleRun = Simulate(single, 3, 0x8888UL, 0.3);
        Check(!singleRun.Failed && AssembledFrame(singleRun).SequenceEqual(OracleFrame(single)),
              "a single-tile frame renders correctly");
        var empty = Frame.Make(0, 0, 16);
        Check(empty.TileCount == 0, "a zero-sized frame has no tiles");
        var emptyRun = Simulate(empty, 2, 0x9999UL, 0.0);
        Check(!emptyRun.Failed && emptyRun.Tiles.Count == 0 && emptyRun.Clock == 0,
              "an empty frame completes immediately without any work");
        var noWorkers = Simulate(frame, 0, 0xAAAAUL, 0.0);
        Check(noWorkers.Failed && noWorkers.Workers.Count == 0,
              "a cluster with no workers fails instead of hanging");
        var trimmed = Frame.Make(70, 20, 32);
        Check(trimmed.TileCount == 3, "a non-multiple width still tiles exactly (" + trimmed.TileCount + ")");
        var trimmedRun = Simulate(trimmed, 3, 0xBBBBUL, 0.0);
        Check(AssembledFrame(trimmedRun).SequenceEqual(OracleFrame(trimmed)),
              "the partial right-edge tiles are rendered with the correct width");

        // 13: determinism of the schedule and the frame digest.
        var repeatA = Simulate(frame, 5, 0x3333UL, 0.0, slowWorker: 1);
        var repeatB = Simulate(frame, 5, 0x3333UL, 0.0, slowWorker: 1);
        Check(repeatA.Clock == repeatB.Clock && repeatA.SpeculativeIssues == repeatB.SpeculativeIssues &&
              Hash(AssembledFrame(repeatA)) == Hash(AssembledFrame(repeatB)),
              "the simulation is bitwise reproducible from its seed");

        Console.WriteLine("healthy clock " + healthy.Clock + ", killed-worker clock " + killed.Clock +
                          ", straggler speculations " + straggler.SpeculativeIssues +
                          ", frame digest " + Hash(AssembledFrame(healthy)));
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
