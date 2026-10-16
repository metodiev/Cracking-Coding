// Task 909 - Implement continuous learning with catastrophic forgetting mitigation (replay buffers).
// (Tier: Master, Beyond - Machine Learning & Simulation)
//
// A tiny linear classifier is trained on a stream of tasks whose decision boundaries rotate, which is the
// classic setting where naive fine-tuning overwrites the old boundary and the model forgets.  Two regimes
// run over identical data and identical seeds:
//
//  * naive - the model only ever sees the current task, so later tasks erase the earlier ones;
//  * replay - every gradient step on the new task is paired with a gradient step on a uniformly sampled
//    example from a fixed-capacity FIFO buffer of past examples, which keeps the old regions of the input
//    space alive.  The buffer is a ring: once it is full the oldest example is evicted, exactly matching an
//    independent list-based model of a ring buffer.
//
// Forgetting is measured as the drop of accuracy on task 0 from its peak (measured right after task 0 was
// learned) to the end of the stream, recomputed from the recorded accuracy matrix by a second code path.
//
// Simplifications: two-class linearly separable tasks with clean labels, a hinge-loss perceptron for the
// model, a handful of tasks and a few SGD epochs; the point is the retention contrast, not state-of-the-art
// continual learning.  Verified: the retention contrast, the forgetting metric against its recomputation,
// the buffer capacity/eviction oracle, uniform and reproducible sampling, and the empty/single-slot bounds.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static double NextUniform() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static int NextInt(int bound) => (int)(NextUInt64() % (ulong)bound);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    internal sealed class Sample
    {
        internal int Id;
        internal double X1;
        internal double X2;
        internal int Label;
    }

    internal sealed class Model
    {
        internal double W1;
        internal double W2;
        internal double Bias;

        internal double Score(double x1, double x2) => W1 * x1 + W2 * x2 + Bias;

        internal int Predict(double x1, double x2) => Score(x1, x2) > 0 ? 1 : -1;

        /** One hinge-loss step: only updates while the margin is violated. */
        internal void Step(Sample sample, double learningRate)
        {
            if (sample.Label * Score(sample.X1, sample.X2) >= 1.0) return;
            W1 += learningRate * sample.Label * sample.X1;
            W2 += learningRate * sample.Label * sample.X2;
            Bias += learningRate * sample.Label;
        }

        internal Model Copy() => new Model { W1 = W1, W2 = W2, Bias = Bias };
    }

    /** Uniform reservoir: fixed capacity, oldest-first eviction - the replay buffer. */
    internal sealed class ReplayBuffer
    {
        private readonly List<Sample> _items = new List<Sample>();
        internal int Capacity;
        internal int Added;
        internal int Evicted;

        internal ReplayBuffer(int capacity)
        {
            Capacity = capacity;
        }

        internal int Count => _items.Count;

        internal void Add(Sample sample)
        {
            Added++;
            if (Capacity <= 0) return;
            if (_items.Count == Capacity)
            {
                _items.RemoveAt(0);
                Evicted++;
            }
            _items.Add(sample);
        }

        internal List<Sample> Snapshot() => new List<Sample>(_items);

        internal bool TrySample(out Sample sample)
        {
            if (_items.Count == 0)
            {
                sample = new Sample();
                return false;
            }
            sample = _items[NextInt(_items.Count)];
            return true;
        }
    }

    private static List<List<Sample>> BuildTasks(int tasks, int perTask, int seed)
    {
        _state = (ulong)seed;
        List<List<Sample>> stream = new List<List<Sample>>();
        int id = 0;
        for (int task = 0; task < tasks; task++)
        {
            double angle = task * 0.95;
            double w1 = Math.Cos(angle);
            double w2 = Math.Sin(angle);
            List<Sample> data = new List<Sample>();
            for (int i = 0; i < perTask; i++)
            {
                double x1 = 2.0 * NextUniform() - 1.0;
                double x2 = 2.0 * NextUniform() - 1.0;
                data.Add(new Sample
                {
                    Id = id++,
                    X1 = x1,
                    X2 = x2,
                    Label = w1 * x1 + w2 * x2 >= 0 ? 1 : -1,
                });
            }
            stream.Add(data);
        }
        return stream;
    }

    private static double Accuracy(Model model, List<Sample> data)
    {
        int correct = 0;
        foreach (Sample sample in data)
        {
            if (model.Predict(sample.X1, sample.X2) == sample.Label) correct++;
        }
        return correct / (double)data.Count;
    }

    /** Trains through the stream; returns the accuracy matrix (rows = after task t, columns = tasks). */
    private static double[,] Train(List<List<Sample>> stream, bool useReplay, int capacity, int epochs, double learningRate)
    {
        _state = 0x5EED1234UL; // the replay sampling stream is fixed, so training is reproducible from the data alone
        Model model = new Model();
        ReplayBuffer buffer = new ReplayBuffer(capacity);
        double[,] matrix = new double[stream.Count, stream.Count];
        for (int task = 0; task < stream.Count; task++)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (Sample sample in stream[task])
                {
                    model.Step(sample, learningRate); // the new task always gets a full gradient step
                    // paired replay step: protects the old regions without replacing the new task's signal
                    if (useReplay && buffer.TrySample(out Sample replay)) model.Step(replay, learningRate);
                }
            }
            foreach (Sample sample in stream[task]) buffer.Add(sample);
            for (int seen = 0; seen <= task; seen++) matrix[task, seen] = Accuracy(model, stream[seen]);
            for (int unseen = task + 1; unseen < stream.Count; unseen++) matrix[task, unseen] = double.NaN;
        }
        return matrix;
    }

    /** Independent recomputation of the forgetting metric from the raw matrix. */
    private static double RecomputeForgetting(double[,] matrix, int task)
    {
        double peak = 0.0;
        for (int row = task; row < matrix.GetLength(0); row++)
        {
            if (!double.IsNaN(matrix[row, task])) peak = Math.Max(peak, matrix[row, task]);
        }
        double final = matrix[matrix.GetLength(0) - 1, task];
        return peak - final;
    }

    public static void Main()
    {
        const int tasks = 4;
        const int perTask = 120;
        List<List<Sample>> stream = BuildTasks(tasks, perTask, 2024);

        // ---- 1. the retention contrast ---------------------------------------------------------------
        double[,] naive = Train(stream, useReplay: false, capacity: 0, epochs: 4, learningRate: 0.05);
        double[,] replay = Train(stream, useReplay: true, capacity: 300, epochs: 4, learningRate: 0.05);
        double naiveRetention = naive[tasks - 1, 0];
        double replayRetention = replay[tasks - 1, 0];
        Check(naive[0, 0] > 0.9 && replay[0, 0] > 0.9, "both regimes learn the first task to above 90% accuracy");
        Check(naiveRetention < 0.6, "naive fine-tuning forgets the first task (retention " +
            naiveRetention.ToString("F4", CultureInfo.InvariantCulture) + ")");
        Check(replayRetention > naiveRetention + 0.08, "replay retains far more of the first task (" +
            replayRetention.ToString("F4", CultureInfo.InvariantCulture) + " vs " +
            naiveRetention.ToString("F4", CultureInfo.InvariantCulture) + ")");
        Check(replay[tasks - 1, tasks - 1] > 0.72, "the replay model still learns the newest task (" +
            replay[tasks - 1, tasks - 1].ToString("F4", CultureInfo.InvariantCulture) + ")");
        Check(naive[tasks - 1, tasks - 1] > 0.88, "the naive model learns the newest task even better");
        double naiveAverage = 0.0, replayAverage = 0.0;
        for (int task = 0; task < tasks; task++)
        {
            naiveAverage += naive[tasks - 1, task] / tasks;
            replayAverage += replay[tasks - 1, task] / tasks;
        }
        Check(replayAverage > naiveAverage + 0.04, "replay's final average accuracy over all seen tasks is higher (" +
            replayAverage.ToString("F4", CultureInfo.InvariantCulture) + " vs " +
            naiveAverage.ToString("F4", CultureInfo.InvariantCulture) + ")");

        // ---- 2. the forgetting metric against its recomputation ---------------------------------------
        double naiveForgetting = RecomputeForgetting(naive, 0);
        double replayForgetting = RecomputeForgetting(replay, 0);
        double drop = naive[0, 0] - naive[tasks - 1, 0];
        Check(Math.Abs(naiveForgetting - drop) < 1e-12, "peak-minus-final equals the direct first-task drop");
        Check(replayForgetting <= naiveForgetting * 0.85, "replay cuts the forgetting metric by at least 15% (" +
            replayForgetting.ToString("F4", CultureInfo.InvariantCulture) + " vs " +
            naiveForgetting.ToString("F4", CultureInfo.InvariantCulture) + ")");
        bool everyTaskLearned = true;
        for (int task = 0; task < tasks; task++) everyTaskLearned &= replay[task, task] > 0.72;
        Check(everyTaskLearned, "the replay model learns every task as it arrives");
        bool replayNeverWorse = true;
        for (int task = 0; task < tasks - 1; task++) replayNeverWorse &= replay[tasks - 1, task] >= naive[tasks - 1, task];
        Check(replayNeverWorse, "at the end of the stream replay is at least as accurate on every old task");

        // ---- 3. the ring buffer against an independent model of a ring --------------------------------
        ReplayBuffer buffer = new ReplayBuffer(250);
        List<Sample> reference = new List<Sample>();
        List<int> ids = new List<int>();
        List<int> referenceIds = new List<int>();
        foreach (List<Sample> task in stream)
        {
            foreach (Sample sample in task)
            {
                buffer.Add(sample);
                if (reference.Count == 250) reference.RemoveAt(0);
                reference.Add(sample);
                if (buffer.Count > 250) Check(false, "the replay buffer exceeded its capacity");
            }
        }
        foreach (Sample sample in reference) referenceIds.Add(sample.Id);
        List<Sample> snapshot = buffer.Snapshot();
        Check(buffer.Count == 250, "the buffer filled to exactly its capacity");
        Check(buffer.Added == tasks * perTask && buffer.Evicted == tasks * perTask - 250,
            "the buffer saw " + buffer.Added + " examples and evicted " + buffer.Evicted);
        bool sameContents = snapshot.Count == referenceIds.Count;
        for (int i = 0; i < snapshot.Count && sameContents; i++) sameContents = snapshot[i].Id == referenceIds[i];
        Check(sameContents, "the ring holds exactly the 250 most recent examples, matching the reference list");

        // ---- 4. uniform, reproducible sampling ----------------------------------------------------------
        _state = 99UL;
        int[] buckets = new int[10];
        int draws = 3000;
        for (int i = 0; i < draws; i++)
        {
            int index = NextInt(250);
            ids.Add(index);
            buckets[index % 10]++;
        }
        int expected = draws / 10;
        bool uniform = true;
        foreach (int count in buckets) uniform &= Math.Abs(count - expected) <= expected / 3;
        Check(uniform, "replay sampling spreads uniformly over the buffer's buckets");
        _state = 99UL;
        int[] again = new int[10];
        for (int i = 0; i < draws; i++) again[NextInt(250) % 10]++;
        bool reproducible = true;
        for (int i = 0; i < buckets.Length; i++) reproducible &= buckets[i] == again[i];
        Check(reproducible, "the replay stream is reproducible from its seed");

        // ---- 5. boundaries ------------------------------------------------------------------------------
        ReplayBuffer tiny = new ReplayBuffer(1);
        foreach (Sample sample in stream[0]) tiny.Add(sample);
        Check(tiny.Count == 1 && tiny.Evicted == perTask - 1, "a one-slot buffer keeps only the newest example");
        Check(tiny.Snapshot()[0].Id == stream[0][perTask - 1].Id, "and that example is the last one added");
        Check(!new ReplayBuffer(8).TrySample(out Sample _), "an empty buffer cannot sample");
        double[,] noReplay = Train(stream, useReplay: true, capacity: 0, epochs: 4, learningRate: 0.05);
        double[,] plain = Train(stream, useReplay: false, capacity: 0, epochs: 4, learningRate: 0.05);
        Check(Math.Abs(noReplay[tasks - 1, 0] - plain[tasks - 1, 0]) < 1e-12,
            "a zero-capacity buffer makes replay identical to naive training");
        double[,] oneSlot = Train(stream, useReplay: true, capacity: 1, epochs: 4, learningRate: 0.05);
        Check(oneSlot[0, 0] > 0.9 && oneSlot[tasks - 1, tasks - 1] > 0.85, "a single replay slot still learns both ends of the stream");
        Check(oneSlot[tasks - 1, 0] <= replay[tasks - 1, 0] + 1e-12,
            "a one-slot buffer retains no more than the full one, as expected");

        // ---- 6. reproducibility --------------------------------------------------------------------------
        double[,] againNaive = Train(stream, useReplay: false, capacity: 0, epochs: 4, learningRate: 0.05);
        double[,] againReplay = Train(stream, useReplay: true, capacity: 300, epochs: 4, learningRate: 0.05);
        bool identical = true;
        for (int row = 0; row < tasks; row++)
        {
            for (int column = 0; column <= row; column++)
            {
                identical &= againNaive[row, column] == naive[row, column];
                identical &= againReplay[row, column] == replay[row, column];
            }
        }
        Check(identical, "both regimes reproduce their accuracy matrices exactly");

        Console.WriteLine("stream: " + tasks + " rotating tasks x " + perTask + " samples, hinge-loss perceptron, 4 epochs");
        Console.WriteLine("retention of task 0: naive " + naiveRetention.ToString("F4", CultureInfo.InvariantCulture) +
            " (forgetting " + naiveForgetting.ToString("F4", CultureInfo.InvariantCulture) + "), replay " +
            replayRetention.ToString("F4", CultureInfo.InvariantCulture) + " (forgetting " +
            replayForgetting.ToString("F4", CultureInfo.InvariantCulture) + ")");
        Console.WriteLine("buffer: capacity 300 held " + buffer.Count + " examples after " + buffer.Added + " inserts and " +
            buffer.Evicted + " evictions, matching the independent ring model");
        Console.WriteLine("PASS");
    }
}
