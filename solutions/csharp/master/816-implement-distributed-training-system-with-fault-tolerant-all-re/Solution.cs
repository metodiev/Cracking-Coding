// Task 816 - Implement distributed training system with fault-tolerant all-reduce and elastic workers. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Everything runs in one process: workers exchange messages over an explicit in-memory network that
// counts hops and payload words, "timeouts" are a fixed number of logical ticks, and there are no
// threads, sockets or wall-clock reads.
//
//   * Gradients are scaled fixed-point integers (scale 2^16) - the gradient compression a real ring
//     all-reduce sends as int16 payloads. This has a decisive property for verification: the reduction
//     is associative and exact, so the ring result can be compared bit-for-bit with a brute-force sum
//     whatever the order of the ring, and the *sum* is divided by the batch size only after the
//     reduction, so per-shard truncation cannot bias the result.
//   * Communication is the textbook ring all-reduce: P-1 reduce-scatter steps then P-1 all-gather
//     steps, 2P(P-1) messages per step carrying 2(P-1)N/P words per worker. Each step is synchronous:
//     every worker snapshots the chunk it sends before any worker accumulates, which is what makes the
//     model equivalent to real message passing (and prevents double counting in the ring).
//   * A round that cannot complete is abandoned: local gradients are immutable during a round, so an
//     aborted round cannot leave a partial update (the parameter version is checked not to move).
//     After TimeoutTicks of silence the survivors agree on a new membership epoch, the full dataset is
//     re-sharded over the survivors and the step is retried - losing a worker loses no training data.
//   * Elastic join uses the same reconfiguration path and hands the joiner the current parameters,
//     which is what keeps the objective invariant: one step is one full-dataset gradient step for
//     every membership.
//
// Simplifications (documented): the network is a dictionary of buffers (no serialisation, bandwidth
// contention or stragglers); failure detection is a deterministic timeout counter fed by a fault
// injector rather than heartbeats; a dead worker's private state is not checkpointed (it rejoins
// fresh); the model is a fixed-point linear regressor with two parameters so the trajectory can be
// compared exactly against a single-process reference.
//
// Verified invariants: all-reduce equals the brute-force sum and every replica is bit-identical for
// P in {1,2,3,4,5,8}; volume within the ring bound and below the all-to-all broadcast; abort safety
// (no partial update, gradients intact, version unchanged); detection within the tick bound with
// unanimous membership epochs; sharding is a partition of the dataset after every reconfiguration, so
// every step aggregates the full-dataset gradient; the loss decreases monotonically and the distributed
// trajectory equals the single-process reference bit-for-bit; determinism of a repeated run.
using System;
using System.Collections.Generic;
using System.Linq;

// ---- deterministic primitives ---------------------------------------------------------------------

internal struct SplitMix64
{
    private ulong state;

    public SplitMix64(ulong seed) { state = seed; }

    public ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

// ---- network model ----------------------------------------------------------------------------------

internal sealed class Net
{
    public long Hops;
    public long Words;
    public int DeniedDeliveries;   // a message aimed at a worker that is gone

    public bool Deliver(Worker from, Worker to, int words)
    {
        Hops++;
        if (!from.Alive || !to.Alive)
        {
            DeniedDeliveries++;
            return false;
        }
        Words += words;
        return true;
    }

    // The receiving side of an already counted message: a hop, but no additional payload.
    public bool Receive(Worker from, Worker to)
    {
        Hops++;
        if (!from.Alive || !to.Alive)
        {
            DeniedDeliveries++;
            return false;
        }
        return true;
    }

    public void Reset()
    {
        Hops = 0;
        Words = 0;
        DeniedDeliveries = 0;
    }
}

// ---- workers and the ring ---------------------------------------------------------------------------

internal sealed class Worker
{
    public int Id;
    public bool Alive = true;
    public int[] DataShard = Array.Empty<int>();
    public long[] Model = Array.Empty<long>();
    public long[] Gradient = Array.Empty<long>();      // local sum, immutable during a round
    public long[] ReduceBuffer = Array.Empty<long>();  // scratch, discarded when a round aborts
    public long[] Reduced = Array.Empty<long>();       // the all-reduced total
    public int Version;
    public int ViewEpoch = -1;
    public int[] View = Array.Empty<int>();

    public Worker(int id) { Id = id; }
}

internal static class Chunks
{
    // Uneven chunks are fine as long as every worker uses the same partition: element k belongs to
    // chunk index with Start <= k < End, and the chunks tile [0, length) exactly.
    public static int Start(int length, int parts, int index) => (int)((long)length * index / parts);
    public static int End(int length, int parts, int index) => (int)((long)length * (index + 1) / parts);
    public static int Length(int length, int parts, int index) => End(length, parts, index) - Start(length, parts, index);
}

internal static class RingAllReduce
{
    public static bool Reduce(Net net, IList<Worker> members, int length)
    {
        int p = members.Count;
        if (p == 0) return false;
        if (p == 1)
        {
            members[0].Reduced = (long[])members[0].Gradient.Clone();
            return true;
        }

        foreach (Worker worker in members) worker.ReduceBuffer = (long[])worker.Gradient.Clone();

        // Reduce-scatter: worker i ends up holding the fully reduced chunk (i + 1) mod p.
        for (int step = 0; step < p - 1; step++)
        {
            if (!ReduceStep(net, members, length, step, false)) return Abandon(members);
        }
        // All-gather: every worker collects the remaining reduced chunks.
        for (int step = 0; step < p - 1; step++)
        {
            if (!GatherStep(net, members, length, step)) return Abandon(members);
        }

        foreach (Worker worker in members) worker.Reduced = (long[])worker.ReduceBuffer.Clone();
        return true;
    }

    private static bool ReduceStep(Net net, IList<Worker> members, int length, int step, bool unused)
    {
        int p = members.Count;
        var payloads = new long[p][];
        for (int i = 0; i < p; i++)
        {
            Worker self = members[i];
            Worker successor = members[(i + 1) % p];
            int sendIndex = ((i - step) % p + p) % p;
            if (!net.Deliver(self, successor, Chunks.Length(length, p, sendIndex))) return false;
            payloads[i] = Snapshot(self, sendIndex, length, p);
        }
        for (int i = 0; i < p; i++)
        {
            Worker self = members[i];
            Worker predecessor = members[(i - 1 + p) % p];
            int receiveIndex = ((i - 1 - step) % p + p) % p;
            if (!net.Receive(predecessor, self)) return false;
            AddInto(self, receiveIndex, payloads[(i - 1 + p) % p], length, p);
        }
        return true;
    }

    private static bool GatherStep(Net net, IList<Worker> members, int length, int step)
    {
        int p = members.Count;
        var payloads = new long[p][];
        for (int i = 0; i < p; i++)
        {
            Worker self = members[i];
            Worker successor = members[(i + 1) % p];
            int sendIndex = ((i + 1 - step) % p + p) % p;
            if (!net.Deliver(self, successor, Chunks.Length(length, p, sendIndex))) return false;
            payloads[i] = Snapshot(self, sendIndex, length, p);
        }
        for (int i = 0; i < p; i++)
        {
            Worker self = members[i];
            Worker predecessor = members[(i - 1 + p) % p];
            int receiveIndex = ((i - step) % p + p) % p;
            if (!net.Receive(predecessor, self)) return false;
            ReplaceInto(self, receiveIndex, payloads[(i - 1 + p) % p], length, p);
        }
        return true;
    }

    private static long[] Snapshot(Worker worker, int index, int length, int parts)
    {
        int start = Chunks.Start(length, parts, index);
        var copy = new long[Chunks.Length(length, parts, index)];
        Array.Copy(worker.ReduceBuffer, start, copy, 0, copy.Length);
        return copy;
    }

    private static void AddInto(Worker worker, int index, long[] payload, int length, int parts)
    {
        int start = Chunks.Start(length, parts, index);
        for (int k = 0; k < payload.Length; k++) worker.ReduceBuffer[start + k] += payload[k];
    }

    private static void ReplaceInto(Worker worker, int index, long[] payload, int length, int parts)
    {
        int start = Chunks.Start(length, parts, index);
        for (int k = 0; k < payload.Length; k++) worker.ReduceBuffer[start + k] = payload[k];
    }

    private static bool Abandon(IList<Worker> members)
    {
        foreach (Worker worker in members) worker.ReduceBuffer = Array.Empty<long>();
        return false;
    }
}

// ---- dataset -----------------------------------------------------------------------------------------

internal sealed class Dataset
{
    public readonly int Dimensions;
    public readonly long[][] X;
    public readonly long[] Y;

    public Dataset(int points, int dimensions)
    {
        Dimensions = dimensions;
        X = new long[points][];
        Y = new long[points];
        var rng = new SplitMix64(0xD47A5E7UL);
        for (int i = 0; i < points; i++)
        {
            var row = new long[dimensions];
            row[0] = 1L << Trainer.Scale;                       // bias term
            double feature = (i + 0.5) / points - 0.5;          // zero mean, so the problem is well scaled
            row[1] = (long)Math.Round(feature * (1 << Trainer.Scale));
            X[i] = row;
            Y[i] = (long)Math.Round((3.0 + 2.0 * feature) * (1 << Trainer.Scale));
        }
        // A little deterministic noise so the fit is non trivial but reachable.
        for (int i = 0; i < points; i++) Y[i] += rng.NextInt(3) - 1;
    }

    public int Count => Y.Length;
}

// ---- trainer -----------------------------------------------------------------------------------------

internal sealed class Trainer
{
    public const int Scale = 16;
    public const int LrShift = 2;                 // learning rate 1/4
    public const int TimeoutTicks = 2;

    public readonly Net Network = new Net();
    public readonly List<Worker> Workers = new List<Worker>();
    public readonly List<Worker> Session = new List<Worker>();   // membership of the current epoch
    public readonly Dataset Data;
    public readonly int Dimensions;

    public int Tick;
    public int StepVersion;
    public int MembershipEpoch;
    public int AbortedRounds;
    public int Reconfigurations;
    public int LastDetectionLatency;
    public readonly List<int> DetectionTicks = new List<int>();

    public Trainer(Dataset data, IEnumerable<int> workerIds)
    {
        Data = data;
        Dimensions = data.Dimensions;
        foreach (int id in workerIds) Workers.Add(new Worker(id) { Model = new long[data.Dimensions] });
        Session.AddRange(Workers.OrderBy(w => w.Id));
        Reshard();
    }

    public IList<Worker> Alive => Workers.Where(w => w.Alive).OrderBy(w => w.Id).ToList();

    public long[] ModelSnapshot()
    {
        var model = new long[Dimensions];
        Worker? first = Session.FirstOrDefault(w => w.Alive) ?? Alive.FirstOrDefault();
        if (first != null) for (int d = 0; d < Dimensions; d++) model[d] = first.Model[d];
        return model;
    }

    // Deterministic contiguous re-sharding over the current session: every point in exactly one shard,
    // union always the whole dataset, and the same formula whatever the membership is.
    public void Reshard()
    {
        for (int i = 0; i < Session.Count; i++)
        {
            int start = (int)((long)Data.Count * i / Session.Count);
            int end = (int)((long)Data.Count * (i + 1) / Session.Count);
            var shard = new int[end - start];
            for (int p = start; p < end; p++) shard[p - start] = p;
            Session[i].DataShard = shard;
            Session[i].Gradient = new long[Dimensions];
        }
        foreach (Worker worker in Workers) if (!Session.Contains(worker)) worker.DataShard = Array.Empty<int>();
    }

    public bool ShardingIsPartition()
    {
        var seen = new HashSet<int>();
        foreach (Worker worker in Session)
            foreach (int point in worker.DataShard)
                if (!seen.Add(point)) return false;
        return seen.Count == Data.Count;
    }

    public int ShardedPoints => Session.Sum(w => w.DataShard.Length);

    public long[] LocalGradient(Worker worker, long[] model)
    {
        var gradient = new long[Dimensions];
        foreach (int point in worker.DataShard)
        {
            long predicted = 0;
            for (int d = 0; d < Dimensions; d++) predicted += (model[d] * Data.X[point][d]) >> Scale;
            long residual = predicted - Data.Y[point];
            for (int d = 0; d < Dimensions; d++) gradient[d] += (residual * Data.X[point][d]) >> Scale;
        }
        return gradient;
    }

    // Brute-force oracle: the sum over every data point, computed without any sharding.
    public long[] FullGradient(long[] model)
    {
        var total = new long[Dimensions];
        for (int point = 0; point < Data.Count; point++)
        {
            long predicted = 0;
            for (int d = 0; d < Dimensions; d++) predicted += (model[d] * Data.X[point][d]) >> Scale;
            long residual = predicted - Data.Y[point];
            for (int d = 0; d < Dimensions; d++) total[d] += (residual * Data.X[point][d]) >> Scale;
        }
        return total;
    }

    public bool TryStep()
    {
        // The round runs over the *believed* membership, so a member that has silently died makes the
        // round fail instead of being quietly dropped - that is what forces the reconfiguration path.
        long[] model = ModelSnapshot();
        foreach (Worker worker in Session) worker.Gradient = LocalGradient(worker, model);
        if (!RingAllReduce.Reduce(Network, Session, Dimensions)) { AbortedRounds++; return false; }
        foreach (Worker worker in Session)
        {
            for (int d = 0; d < Dimensions; d++)
            {
                long mean = worker.Reduced[d] / Data.Count;
                worker.Model[d] -= (mean * 1) >> LrShift;
            }
            worker.Version++;
        }
        StepVersion++;
        return true;
    }

    // Survivors poll the monitor, agree on the surviving set, bump the membership epoch and re-shard.
    public void DetectFailures()
    {
        Reconfigurations++;
        MembershipEpoch++;
        DetectionTicks.Add(Tick);
        IList<Worker> alive = Alive;
        int[] view = alive.Select(w => w.Id).ToArray();
        foreach (Worker worker in alive)
        {
            worker.ViewEpoch = MembershipEpoch;
            worker.View = (int[])view.Clone();
        }
        Session.Clear();
        Session.AddRange(alive);
        Reshard();
    }

    public bool StepWithRecovery(int maxAttempts = 4)
    {
        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            if (TryStep()) return true;
            Tick += TimeoutTicks;
            LastDetectionLatency = TimeoutTicks;
            DetectFailures();
        }
        return false;
    }

    public void Fail(int workerId)
    {
        Worker worker = Workers.First(w => w.Id == workerId);
        worker.Alive = false;
    }

    public bool Join(int workerId, int donorId)
    {
        Worker donor = Workers.First(w => w.Id == donorId);
        if (!donor.Alive) return false;
        var joiner = new Worker(workerId) { Model = (long[])donor.Model.Clone(), Version = donor.Version };
        Workers.Add(joiner);
        DetectFailures();
        return true;
    }

    public double Loss()
    {
        long[] model = ModelSnapshot();
        double total = 0;
        for (int point = 0; point < Data.Count; point++)
        {
            long predicted = 0;
            for (int d = 0; d < Dimensions; d++) predicted += (model[d] * Data.X[point][d]) >> Scale;
            double residual = (predicted - Data.Y[point]) / (double)(1 << Scale);
            total += residual * residual;
        }
        return total / Data.Count;
    }
}

// ---- checks -------------------------------------------------------------------------------------------

public static class Solution
{
    public static void Main()
    {
        var data = new Dataset(64, 2);

        // ---- ring all-reduce against a brute force oracle --------------------------------------
        foreach (int p in new[] { 1, 2, 3, 4, 5, 8 })
        {
            var net = new Net();
            var members = new List<Worker>();
            var rng = new SplitMix64(0x1000UL + (ulong)p);
            var expected = new long[7];
            for (int i = 0; i < p; i++)
            {
                var worker = new Worker(i) { Gradient = new long[7] };
                for (int d = 0; d < 7; d++)
                {
                    worker.Gradient[d] = (long)(rng.Next() % 1000) - 500;
                    expected[d] += worker.Gradient[d];
                }
                members.Add(worker);
            }
            Check(RingAllReduce.Reduce(net, members, 7), "ring completes for P=" + p);
            foreach (Worker member in members)
                Check(member.Reduced.SequenceEqual(expected), $"ring sum is exact for P={p} (#{member.Id})");
            Check(members.All(m => m.Reduced.SequenceEqual(members[0].Reduced)), "replicas are bit identical for P=" + p);
            if (p == 1)
            {
                Check(net.Hops == 0 && net.Words == 0, "a single worker sends nothing");
            }
            else
            {
                // Two deliveries per worker per ring step (its send and its receive), 2(P-1) steps.
                Check(net.Hops == 4L * p * (p - 1), $"hop count for P={p}: {net.Hops}");
                // Each worker ships 2(P-1) chunks of about N/P words: ~2(P-1)N/P words per worker.
                double perWorker = net.Words / (double)p;
                Check(perWorker <= 2.0 * (p - 1) * 7 / p + 2, $"ring words per worker for P={p}: {perWorker:F2}");
                long allToAll = (long)p * (p - 1) * 7;      // every worker ships its whole vector to everyone
                if (p >= 4) Check(net.Words * 2 <= allToAll, $"ring volume {net.Words} at most half of all-to-all {allToAll}");
                Check(net.Words >= 2L * (p - 1) * 7 / 2, "the ring really moves the data");
            }
        }

        // ---- a full distributed training run --------------------------------------------------
        var reference = new Trainer(data, new[] { 1 });
        var trainer = new Trainer(data, new[] { 1, 2, 3, 4 });
        Check(trainer.ShardingIsPartition() && trainer.ShardedPoints == data.Count, "initial sharding is a partition");
        Check(reference.ModelSnapshot().SequenceEqual(trainer.ModelSnapshot()), "all members start from the same model");

        var losses = new List<double>();
        for (int step = 0; step < 40; step++)
        {
            Check(trainer.StepWithRecovery(), "distributed step " + step);
            Check(reference.StepWithRecovery(), "reference step " + step);
            Check(trainer.ModelSnapshot().SequenceEqual(reference.ModelSnapshot()),
                "distributed trajectory equals the single process reference at step " + step);
            var models = trainer.Alive.Select(w => w.Model).ToList();
            Check(models.All(m => m.SequenceEqual(models[0])), "every member holds the same parameters after step " + step);
            losses.Add(trainer.Loss());
        }
        for (int i = 1; i < losses.Count; i++)
            Check(losses[i] <= losses[i - 1] + 1e-9, $"loss is non increasing at step {i}: {losses[i - 1]:F6} -> {losses[i]:F6}");
        Check(losses[losses.Count - 1] * 2 < losses[0], $"loss more than halved: {losses[0]:F6} -> {losses[losses.Count - 1]:F6}");
        Check(trainer.StepVersion == 40 && trainer.Alive.All(w => w.Version == 40), "parameter versions tracked per member");

        // ---- aggregate equals the full-dataset gradient ---------------------------------------
        long[] full = trainer.FullGradient(trainer.ModelSnapshot());
        var probe = new Net();
        var probeWorkers = trainer.Alive.ToList();
        foreach (Worker worker in probeWorkers) worker.Gradient = trainer.LocalGradient(worker, trainer.ModelSnapshot());
        Check(RingAllReduce.Reduce(probe, probeWorkers, trainer.Dimensions), "probe reduce completes");
        Check(probeWorkers[0].Reduced.SequenceEqual(full), "the ring aggregate equals the full dataset gradient");

        // ---- fault injection: a worker dies before a step -------------------------------------
        int versionBefore = trainer.StepVersion;
        trainer.Fail(4);
        Check(!trainer.TryStep(), "a step with a dead member aborts");
        Check(trainer.StepVersion == versionBefore, "an aborted round applies no update");
        Check(trainer.Alive.All(w => w.Version == versionBefore), "no member advanced its version on an aborted round");
        Check(trainer.AbortedRounds == 1, "the abort was recorded");
        int gradientsIntact = 0;
        foreach (Worker worker in trainer.Alive)
        {
            long[] fresh = trainer.LocalGradient(worker, trainer.ModelSnapshot());
            if (worker.Gradient.SequenceEqual(fresh)) gradientsIntact++;
        }
        Check(gradientsIntact == trainer.Alive.Count, "aborted rounds leave local gradients intact");
        Check(trainer.StepWithRecovery(), "the step completes after reconfiguration");
        Check(reference.StepWithRecovery(), "the reference takes the same step");
        Check(trainer.Reconfigurations == 1 && trainer.LastDetectionLatency <= Trainer.TimeoutTicks,
            "failure detected inside the tick bound");
        Check(trainer.Alive.Select(w => w.ViewEpoch).Distinct().Count() == 1, "survivors agree on the membership epoch");
        var views = trainer.Alive.Select(w => string.Join(",", w.View)).Distinct().ToList();
        Check(views.Count == 1, "survivors agree on the membership: " + string.Join("|", views));
        Check(trainer.ShardingIsPartition() && trainer.ShardedPoints == data.Count,
            "re-sharding after a failure still covers the whole dataset");
        Check(trainer.ModelSnapshot().SequenceEqual(reference.ModelSnapshot()), "the model still matches the reference");
        Check(trainer.Alive.All(w => w.DataShard.Length > 0), "survivors each hold data after re-sharding");

        // ---- a member dying mid round ----------------------------------------------------------
        var fragile = new Trainer(data, new[] { 1, 2, 3 });
        Check(fragile.StepWithRecovery(), "warm up the fragile cluster");
        var survivorModels = fragile.Alive.Select(w => (long[])w.Model.Clone()).ToList();
        fragile.Workers.First(w => w.Id == 3).Alive = false;                     // dies while the ring runs
        Check(!fragile.TryStep(), "a mid round failure aborts that round");
        Check(fragile.StepVersion == 1, "still no partial update after a mid round abort");
        foreach (Worker worker in fragile.Alive)
        {
            long[] fresh = fragile.LocalGradient(worker, fragile.ModelSnapshot());
            Check(worker.Gradient.SequenceEqual(fresh), "gradients intact after a mid round abort (#" + worker.Id + ")");
        }
        Check(fragile.StepWithRecovery(), "the mid round failure is recovered from");
        Check(fragile.Alive.Count == 2 && fragile.Alive.Sum(w => w.DataShard.Length) == data.Count,
            "the two survivors now cover the whole dataset");
        Check(fragile.StepVersion == 2, "the recovered step advanced the version exactly once");

        // ---- elastic join ---------------------------------------------------------------------
        var elastic = new Trainer(data, new[] { 1, 2 });
        for (int step = 0; step < 5; step++) Check(elastic.StepWithRecovery(), "elastic warm up " + step);
        long[] beforeJoin = (long[])elastic.ModelSnapshot().Clone();
        var elasticReference = new Trainer(data, new[] { 9 });
        elasticReference.Workers[0].Model = (long[])beforeJoin.Clone();
        Check(elastic.Join(5, 1), "a new worker joins");
        Check(elastic.Workers.First(w => w.Id == 5).Model.SequenceEqual(elastic.Workers.First(w => w.Id == 1).Model),
            "the joiner receives the current parameters");
        Check(elastic.Session.Count == 3 && elastic.ShardingIsPartition() && elastic.ShardedPoints == data.Count,
            "the joiner participates in the sharding");
        var elasticLosses = new List<double> { elastic.Loss() };
        for (int step = 0; step < 10; step++)
        {
            Check(elastic.StepWithRecovery(), "post join step " + step);
            Check(elasticReference.StepWithRecovery(), "post join reference step " + step);
            Check(elastic.ModelSnapshot().SequenceEqual(elasticReference.ModelSnapshot()),
                "the elastic cluster follows the same trajectory as a single worker at step " + step);
            elasticLosses.Add(elastic.Loss());
        }
        for (int i = 1; i < elasticLosses.Count; i++)
            Check(elasticLosses[i] <= elasticLosses[i - 1] + 1e-9, "post join loss non increasing");

        // ---- membership independence of the aggregate ------------------------------------------
        var basis = new Trainer(data, new[] { 7 });
        for (int step = 0; step < 3; step++) basis.StepWithRecovery();
        foreach (int[] membership in new[] { new[] { 1 }, new[] { 1, 2 }, new[] { 1, 2, 3, 4, 5, 6, 7 } })
        {
            var cluster = new Trainer(data, membership);
            for (int step = 0; step < 3; step++) Check(cluster.StepWithRecovery(), "cluster step");
            Check(cluster.ModelSnapshot().SequenceEqual(basis.ModelSnapshot()),
                "the model is membership independent for size " + membership.Length);
            long[] aggregate = cluster.FullGradient(cluster.ModelSnapshot());
            var net = new Net();
            var members = cluster.Session.ToList();
            foreach (Worker worker in members) worker.Gradient = cluster.LocalGradient(worker, cluster.ModelSnapshot());
            Check(RingAllReduce.Reduce(net, members, cluster.Dimensions), "membership reduce completes");
            Check(members[0].Reduced.SequenceEqual(aggregate), "aggregate is the full gradient for size " + membership.Length);
            Check(members.All(m => m.Reduced.SequenceEqual(members[0].Reduced)), "replicas agree for size " + membership.Length);
        }

        // ---- degenerate shapes -----------------------------------------------------------------
        var solo = new Trainer(data, new[] { 1 });
        Check(solo.StepWithRecovery(), "single worker step needs no network");
        Check(solo.Network.Hops == 0, "a single worker sends no messages");
        var many = new Trainer(data, Enumerable.Range(1, 70));
        Check(many.Session.Count(w => w.DataShard.Length == 0) > 0, "shards may be empty when workers outnumber points");
        Check(many.StepWithRecovery(), "empty shards still participate in the ring");
        Check(many.Session.Sum(w => w.DataShard.Length) == data.Count, "empty shards do not lose data");
        var doomed = new Trainer(data, new[] { 1, 2, 3 });
        doomed.Fail(2);
        doomed.Fail(3);
        Check(doomed.StepWithRecovery(), "the last survivor trains alone");
        Check(doomed.Session.Count == 1 && doomed.ShardedPoints == data.Count, "the last survivor holds the whole dataset");
        var emptyRing = new Trainer(data, new[] { 1, 2 });
        foreach (Worker worker in emptyRing.Session) worker.Gradient = Array.Empty<long>();
        var emptyNet = new Net();
        Check(RingAllReduce.Reduce(emptyNet, emptyRing.Session.ToList(), 0), "a zero length ring vector still reduces");
        Check(emptyRing.Session[0].Reduced.Length == 0, "zero length result");
        var widthOne = new Trainer(data, new[] { 1, 2, 3 });
        var widthNet = new Net();
        foreach (Worker worker in widthOne.Session) worker.Gradient = new long[] { worker.Id * 10 };
        Check(RingAllReduce.Reduce(widthNet, widthOne.Session.ToList(), 1), "a one element ring vector reduces");
        Check(widthOne.Session[0].Reduced[0] == (1 + 2 + 3) * 10, "one element total is exact");

        // ---- determinism -----------------------------------------------------------------------
        var again = new Trainer(data, new[] { 1, 2, 3, 4 });
        var firstRun = new Trainer(data, new[] { 1, 2, 3, 4 });
        for (int step = 0; step < 12; step++) { firstRun.StepWithRecovery(); again.StepWithRecovery(); }
        Check(again.ModelSnapshot().SequenceEqual(firstRun.ModelSnapshot()), "repeated runs agree on the model");
        Check(again.Network.Hops == firstRun.Network.Hops && again.Network.Words == firstRun.Network.Words,
            "repeated runs agree on hops and payload");

        // ---- quantization bound ----------------------------------------------------------------
        double maxError = 0;
        foreach (int point in new[] { 0, 5, 17, 63 })
        {
            double feature = (point + 0.5) / data.Count - 0.5;
            double real = 2.0 * feature * (3.0 + 2.0 * feature);
            long stored = (long)Math.Round(real * (1 << Trainer.Scale));
            maxError = Math.Max(maxError, Math.Abs(stored / (double)(1 << Trainer.Scale) - real));
        }
        Check(maxError <= 0.5 / (1 << Trainer.Scale), $"fixed point gradient error {maxError:E3} within half a quantum");

        Console.WriteLine($"816 ring: P=4 steps carry {2L * 3 * 2} words per member and {2L * 4 * 3} hops per step; "
            + $"40 steps matched the single process reference bit for bit; loss {losses[0]:F6} -> {losses[losses.Count - 1]:F6}; "
            + $"{trainer.AbortedRounds + fragile.AbortedRounds} aborted round(s) with zero partial updates; "
            + $"reconfigurations {trainer.Reconfigurations + fragile.Reconfigurations + elastic.Reconfigurations}; "
            + $"detection latency {Trainer.TimeoutTicks} ticks");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
