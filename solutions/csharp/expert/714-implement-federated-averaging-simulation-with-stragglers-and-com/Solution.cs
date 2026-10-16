// Task 714 - Implement federated averaging simulation with stragglers and compression.
// (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// FedAvg alternates two steps: the server broadcasts w, every sampled client takes E local SGD steps
// on its own shard and reports the move w_local - w, and the server adds the average move. The model
// here is linear regression, so the exact optimum is available in closed form and the federated
// answer can be compared against it instead of being taken on trust.
//
//   * Stragglers. A client with delay `lag` computes its move from the *snapshot* of the global model
//     taken `lag` rounds ago (the trainer keeps the full history), and the move is applied only when
//     it arrives. The snapshot round of every computation is recorded, so staleness is observable.
//
//   * Compression. Moves are sparsified to their k largest coordinates and the unsent remainder is
//     kept in a per-client residual ("error feedback") that rides along with the next move. The
//     identity sum(sent) + final residual = sum(computed) therefore holds exactly after any number
//     of rounds, which is checked per client and coordinate, together with "every sent vector has at
//     most k nonzeros" and "k = d reproduces the uncompressed trajectory bit for bit".
//
// Verified below: the server step is exactly the mean of the local moves; training more than halves
// the loss and lands close to the closed-form least-squares optimum; a three-round delay and top-2
// compression each cost less than 50% extra loss; the accounting identity holds; and the degenerate
// configurations (zero rounds, one client, one sample, k = 0, k > d) behave sensibly.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static double Signed(ref ulong state) => 2.0 * Uniform(ref state) - 1.0;
}

internal sealed class Shard
{
    internal double[][] Inputs { get; }
    internal double[] Targets { get; }

    internal Shard(int samples, int dimension, ulong seed)
    {
        Inputs = new double[samples][];
        Targets = new double[samples];
        var state = seed;
        for (int i = 0; i < samples; i++)
        {
            Inputs[i] = new double[dimension];
            for (int d = 0; d < dimension; d++) Inputs[i][d] = Rng.Signed(ref state);
            Targets[i] = 0.5 * Inputs[i][0] - 1.5 * Inputs[i][1]
                + (dimension > 2 ? 0.25 * Inputs[i][2] : 0.0) + 0.1;
        }
    }

    internal int Count => Inputs.Length;
}

internal static class Regression
{
    internal static double Predict(double[] w, double[] x)
    {
        double sum = 0;
        for (int d = 0; d < w.Length; d++) sum += w[d] * x[d];
        return sum;
    }

    internal static double Loss(double[] w, IReadOnlyList<Shard> shards)
    {
        double total = 0;
        long count = 0;
        foreach (Shard shard in shards)
        {
            for (int i = 0; i < shard.Count; i++)
            {
                double error = Predict(w, shard.Inputs[i]) - shard.Targets[i];
                total += error * error;
                count++;
            }
        }
        return count == 0 ? 0 : total / count;
    }

    internal static double Loss(double[] w, Shard shard)
    {
        double total = 0;
        for (int i = 0; i < shard.Count; i++)
        {
            double error = Predict(w, shard.Inputs[i]) - shard.Targets[i];
            total += error * error;
        }
        return shard.Count == 0 ? 0 : total / shard.Count;
    }

    /// <summary>Full-batch SGD for a few steps on one shard, starting from the given weights.</summary>
    internal static double[] LocalUpdate(double[] w, Shard shard, int steps, double learningRate)
    {
        var local = (double[])w.Clone();
        int dimension = w.Length;
        var gradient = new double[dimension];
        for (int step = 0; step < steps; step++)
        {
            for (int d = 0; d < dimension; d++) gradient[d] = 0;
            for (int i = 0; i < shard.Count; i++)
            {
                double error = Predict(local, shard.Inputs[i]) - shard.Targets[i];
                for (int d = 0; d < dimension; d++) gradient[d] += 2.0 * error * shard.Inputs[i][d];
            }
            for (int d = 0; d < dimension; d++) local[d] -= learningRate * gradient[d] / shard.Count;
        }
        return local;
    }

    /// <summary>Exact least-squares weights through the normal equations.</summary>
    internal static double[] LeastSquares(IReadOnlyList<Shard> shards, int dimension)
    {
        int size = dimension + 1;                     // the last unknown is the bias
        var gram = new double[size, size];
        var rhs = new double[size];
        foreach (Shard shard in shards)
        {
            for (int i = 0; i < shard.Count; i++)
            {
                var row = new double[size];
                for (int d = 0; d < dimension; d++) row[d] = shard.Inputs[i][d];
                row[dimension] = 1.0;
                for (int a = 0; a < size; a++)
                {
                    for (int b = 0; b < size; b++) gram[a, b] += row[a] * row[b];
                    rhs[a] += row[a] * shard.Targets[i];
                }
            }
        }
        double[] solved = Solve(gram, rhs, size);
        var weights = new double[dimension];
        for (int d = 0; d < dimension; d++) weights[d] = solved[d];
        return weights;
    }

    private static double[] Solve(double[,] matrix, double[] rhs, int size)
    {
        var a = (double[,])matrix.Clone();
        var b = (double[])rhs.Clone();
        for (int col = 0; col < size; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < size; r++) if (Math.Abs(a[r, col]) > Math.Abs(a[pivot, col])) pivot = r;
            if (Math.Abs(a[pivot, col]) < 1e-14) throw new InvalidOperationException("singular normal equations");
            for (int c = 0; c < size; c++) (a[col, c], a[pivot, c]) = (a[pivot, c], a[col, c]);
            (b[col], b[pivot]) = (b[pivot], b[col]);
            for (int r = 0; r < size; r++)
            {
                if (r == col) continue;
                double factor = a[r, col] / a[col, col];
                for (int c = col; c < size; c++) a[r, c] -= factor * a[col, c];
                b[r] -= factor * b[col];
            }
        }
        var solution = new double[size];
        for (int i = 0; i < size; i++) solution[i] = b[i] / a[i, i];
        return solution;
    }

    /// <summary>Top-k sparsification keeping the k coordinates of largest magnitude.</summary>
    internal static double[] Sparsify(double[] vector, int k)
    {
        var sparse = new double[vector.Length];
        if (k <= 0) return sparse;
        if (k >= vector.Length) return (double[])vector.Clone();
        var order = new List<int>();
        for (int d = 0; d < vector.Length; d++) order.Add(d);
        order.Sort((a, b) =>
        {
            double diff = Math.Abs(vector[b]) - Math.Abs(vector[a]);
            return diff != 0 ? (diff < 0 ? -1 : 1) : a.CompareTo(b);
        });
        for (int i = 0; i < k; i++) sparse[order[i]] = vector[order[i]];
        return sparse;
    }
}

internal sealed class FederatedTrainer
{
    private readonly IReadOnlyList<Shard> _shards;
    private readonly int _dimension;

    internal FederatedTrainer(IReadOnlyList<Shard> shards, int dimension)
    {
        _shards = shards;
        _dimension = dimension;
    }

    internal double[][] ComputedTotal { get; private set; } = Array.Empty<double[]>();
    internal double[][] SentTotal { get; private set; } = Array.Empty<double[]>();
    internal double[][] ResidualPerClient { get; private set; } = Array.Empty<double[]>();
    internal double[] FinalResidual { get; private set; } = Array.Empty<double>();
    internal int[] SentNonZeroHistogram { get; private set; } = Array.Empty<int>();
    internal List<int> SnapshotRounds { get; private set; } = new List<int>();
    internal int ClientCount => _shards.Count;

    internal double[] Train(double[] start, int rounds, int localSteps, double learningRate, int lag, int topK)
    {
        int clients = _shards.Count;
        var weights = (double[])start.Clone();
        var history = new List<double[]>();
        var residuals = new double[clients][];
        ComputedTotal = new double[clients][];
        SentTotal = new double[clients][];
        for (int c = 0; c < clients; c++)
        {
            residuals[c] = new double[_dimension];
            ComputedTotal[c] = new double[_dimension];
            SentTotal[c] = new double[_dimension];
        }
        SentNonZeroHistogram = new int[_dimension + 1];
        SnapshotRounds = new List<int>();

        for (int round = 0; round < rounds; round++)
        {
            history.Add((double[])weights.Clone());
            int snapshotRound = Math.Max(0, round - lag);
            double[] basis = history[snapshotRound];
            var delta = new double[_dimension];

            for (int client = 0; client < clients; client++)
            {
                double[] local = Regression.LocalUpdate(basis, _shards[client], localSteps, learningRate);
                for (int d = 0; d < _dimension; d++)
                {
                    double move = local[d] - basis[d];
                    ComputedTotal[client][d] += move;
                    residuals[client][d] += move;
                }
                SnapshotRounds.Add(snapshotRound);

                double[] sent = Regression.Sparsify(residuals[client], topK);
                int nonZeros = 0;
                for (int d = 0; d < _dimension; d++)
                {
                    residuals[client][d] -= sent[d];
                    SentTotal[client][d] += sent[d];
                    delta[d] += sent[d];
                    if (sent[d] != 0) nonZeros++;
                }
                SentNonZeroHistogram[Math.Min(nonZeros, _dimension)]++;
            }
            for (int d = 0; d < _dimension; d++) weights[d] += delta[d] / clients;
        }

        FinalResidual = new double[_dimension];
        ResidualPerClient = new double[clients][];
        for (int client = 0; client < clients; client++)
        {
            ResidualPerClient[client] = residuals[client];
            for (int d = 0; d < _dimension; d++)
                FinalResidual[d] += residuals[client][d];
        }
        return weights;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Near(double a, double b, double tolerance) => Math.Abs(a - b) <= tolerance;

    public static void Main()
    {
        const int Dimension = 3, Clients = 8, SamplesPerClient = 40;
        var shards = new List<Shard>();
        for (int i = 0; i < Clients; i++) shards.Add(new Shard(SamplesPerClient, Dimension, 0x1000UL + (ulong)i * 7919UL));
        var start = new double[Dimension];
        double initialLoss = Regression.Loss(start, shards);
        double zeroModelLoss = 0;
        foreach (Shard shard in shards)
            for (int i = 0; i < shard.Count; i++) zeroModelLoss += shard.Targets[i] * shard.Targets[i];
        zeroModelLoss /= Clients * SamplesPerClient;
        Check(Near(initialLoss, zeroModelLoss, 1e-12), "the all-zero model scores the mean squared target");

        // ---- the server step really is the average of the local moves --------------------------
        {
            var trainer = new FederatedTrainer(shards, Dimension);
            double[] after = trainer.Train(start, 1, 5, 0.05, 0, int.MaxValue);
            var expected = new double[Dimension];
            for (int client = 0; client < Clients; client++)
            {
                double[] local = Regression.LocalUpdate(start, shards[client], 5, 0.05);
                for (int d = 0; d < Dimension; d++) expected[d] += (local[d] - start[d]) / Clients;
            }
            for (int d = 0; d < Dimension; d++) expected[d] += start[d];
            bool identical = true;
            for (int d = 0; d < Dimension; d++) if (!Near(after[d], expected[d], 1e-12)) identical = false;
            Check(identical, "one round of FedAvg is the mean of the local moves");
            Console.WriteLine("firstRoundWeights=[" + after[0].ToString("F6") + "," + after[1].ToString("F6")
                + "," + after[2].ToString("F6") + "]");
        }

        // ---- training reduces the loss and approaches the exact optimum ------------------------
        {
            var trainer = new FederatedTrainer(shards, Dimension);
            double[] trained = trainer.Train(start, 120, 4, 0.08, 0, int.MaxValue);
            double trainedLoss = Regression.Loss(trained, shards);
            Check(trainedLoss < initialLoss / 2,
                "FedAvg more than halves the loss (" + trainedLoss.ToString("E3") + " vs " + initialLoss.ToString("E3") + ")");

            double[] optimum = Regression.LeastSquares(shards, Dimension);
            double optimalLoss = Regression.Loss(optimum, shards);
            Check(trainedLoss <= optimalLoss * 2.0 + 1e-9,
                "the federated model is within 2x of the closed-form optimum (" + trainedLoss.ToString("E3")
                + " vs " + optimalLoss.ToString("E3") + ")");
            Check(trainer.SentNonZeroHistogram[Dimension] == 120 * Clients,
                "nothing is sparsified when k covers the model (" + trainer.SentNonZeroHistogram[Dimension] + ")");
            Console.WriteLine("initialLoss=" + initialLoss.ToString("E3") + " federatedLoss=" + trainedLoss.ToString("E3")
                + " optimalLoss=" + optimalLoss.ToString("E3"));
        }

        // ---- stragglers: genuinely stale moves still converge ----------------------------------
        {
            var prompt = new FederatedTrainer(shards, Dimension);
            var slow = new FederatedTrainer(shards, Dimension);
            double[] fast = prompt.Train(start, 120, 4, 0.08, 0, int.MaxValue);
            double[] delayed = slow.Train(start, 120, 4, 0.08, 3, int.MaxValue);
            double fastLoss = Regression.Loss(fast, shards), slowLoss = Regression.Loss(delayed, shards);
            Check(slowLoss < initialLoss / 2, "stale moves still halve the loss (" + slowLoss.ToString("E3") + ")");
            Check(slowLoss <= fastLoss * 1.5 + 1e-9,
                "a three-round delay costs at most 50% extra loss (" + slowLoss.ToString("E3")
                + " vs " + fastLoss.ToString("E3") + ")");
            Check(slow.SnapshotRounds.Count == 120 * Clients,
                "every client move records the snapshot it used (" + slow.SnapshotRounds.Count + ")");
            Check(slow.SnapshotRounds[0] == 0 && slow.SnapshotRounds[Clients * 50] == 47,
                "the recorded snapshot is exactly the configured three rounds behind");
            Console.WriteLine("staleLoss=" + slowLoss.ToString("E3") + " synchronousLoss=" + fastLoss.ToString("E3"));
        }

        // ---- compression: top-k with error feedback --------------------------------------------
        {
            var dense = new FederatedTrainer(shards, Dimension);
            double[] plain = dense.Train(start, 120, 4, 0.08, 0, int.MaxValue);
            double plainLoss = Regression.Loss(plain, shards);

            var compressed = new FederatedTrainer(shards, Dimension);
            double[] sparse = compressed.Train(start, 120, 4, 0.08, 0, 2);
            double sparseLoss = Regression.Loss(sparse, shards);
            Check(sparseLoss < initialLoss / 2, "top-2 compression still halves the loss (" + sparseLoss.ToString("E3") + ")");
            Check(sparseLoss <= plainLoss * 1.5 + 1e-9,
                "compression costs at most 50% extra loss (" + sparseLoss.ToString("E3")
                + " vs " + plainLoss.ToString("E3") + ")");

            int tolerated = 0;
            for (int nz = 0; nz <= Dimension; nz++) if (nz <= 2) tolerated += compressed.SentNonZeroHistogram[nz];
            Check(tolerated == 120 * Clients, "every transmitted vector had at most two nonzeros ("
                + tolerated + "/" + (120 * Clients) + ")");

            // k = d is lossless, so the trajectory matches the uncompressed run exactly
            var lossless = new FederatedTrainer(shards, Dimension);
            double[] full = lossless.Train(start, 30, 4, 0.08, 0, Dimension);
            var reference = new FederatedTrainer(shards, Dimension);
            double[] referenceWeights = reference.Train(start, 30, 4, 0.08, 0, int.MaxValue);
            bool sameTrajectory = true;
            for (int d = 0; d < Dimension; d++) if (!Near(full[d], referenceWeights[d], 1e-12)) sameTrajectory = false;
            Check(sameTrajectory, "top-d compression reproduces the uncompressed trajectory exactly");

            // error feedback accounting, per client and coordinate: sent + residual == computed
            var accounting = new FederatedTrainer(shards, Dimension);
            accounting.Train(start, 40, 3, 0.05, 0, 1);
            bool identity = true;
            string firstFailure = "none";
            double worstGap = 0;
            for (int client = 0; client < Clients; client++)
            {
                for (int d = 0; d < Dimension; d++)
                {
                    double reconstructed = accounting.SentTotal[client][d] + accounting.ResidualPerClient[client][d];
                    double gap = Math.Abs(reconstructed - accounting.ComputedTotal[client][d]);
                    if (gap > worstGap) worstGap = gap;
                    if (gap > 1e-9)
                    {
                        identity = false;
                        if (firstFailure == "none") firstFailure = "client " + client + " coordinate " + d;
                    }
                }
            }
            Check(identity, "sent + residual reproduces exactly what each client computed (" + firstFailure + ")");
            Check(worstGap < 1e-9, "the accounting identity holds to machine precision (gap " + worstGap.ToString("E2") + ")");
            Check(Math.Abs(accounting.FinalResidual[0]) + Math.Abs(accounting.FinalResidual[1]) + Math.Abs(accounting.FinalResidual[2]) > 0,
                "with top-1 some mass necessarily remains in the residual");
            Console.WriteLine("plainLoss=" + plainLoss.ToString("E3") + " top2Loss=" + sparseLoss.ToString("E3")
                + " residualAfter40Rounds=[" + accounting.FinalResidual[0].ToString("F5") + ","
                + accounting.FinalResidual[1].ToString("F5") + "," + accounting.FinalResidual[2].ToString("F5") + "]");
        }

        // ---- degenerate configurations ---------------------------------------------------------
        {
            var trainer = new FederatedTrainer(shards, Dimension);
            double[] none = trainer.Train(start, 0, 4, 0.08, 0, int.MaxValue);
            bool unchanged = true;
            for (int d = 0; d < Dimension; d++) if (none[d] != start[d]) unchanged = false;
            Check(unchanged, "zero rounds leave the model untouched");

            var single = new FederatedTrainer(new List<Shard> { shards[0] }, Dimension);
            double[] solo = single.Train(start, 60, 4, 0.08, 0, int.MaxValue);
            Check(Regression.Loss(solo, shards[0]) < Regression.Loss(start, shards[0]),
                "a single client trains on its own shard");

            var tiny = new FederatedTrainer(new List<Shard> { new Shard(1, Dimension, 0xABCUL) }, Dimension);
            double[] one = tiny.Train(start, 20, 3, 0.05, 0, 1);
            bool finite = true;
            for (int d = 0; d < Dimension; d++) if (double.IsNaN(one[d]) || double.IsInfinity(one[d])) finite = false;
            Check(finite, "a one-sample shard keeps the model finite");

            double[] source = { 1.0, -5.0, 0.5, 0.2 };
            double[] topOne = Regression.Sparsify(source, 1);
            int kept = 0;
            foreach (double v in topOne) if (v != 0) kept++;
            Check(kept == 1 && Near(topOne[1], -5.0, 1e-12), "top-1 keeps the largest magnitude entry");
            double[] topAll = Regression.Sparsify(source, 10);
            bool allKept = true;
            for (int d = 0; d < source.Length; d++) if (topAll[d] != source[d]) allKept = false;
            Check(allKept, "asking for more coordinates than exist keeps everything");
            int zeros = 0;
            foreach (double v in Regression.Sparsify(source, 0)) if (v == 0) zeros++;
            Check(zeros == source.Length, "top-0 sends nothing");

            var starving = new FederatedTrainer(shards, Dimension);
            double[] starved = starving.Train(start, 15, 4, 0.08, 0, 0);
            bool stillStart = true;
            for (int d = 0; d < Dimension; d++) if (!Near(starved[d], start[d], 1e-12)) stillStart = false;
            Check(stillStart, "a client that may send nothing cannot move the model");
            Console.WriteLine("starvedModel=[" + starved[0] + "," + starved[1] + "," + starved[2] + "]");
        }

        // ---- determinism ------------------------------------------------------------------------
        {
            var first = new FederatedTrainer(shards, Dimension);
            var second = new FederatedTrainer(shards, Dimension);
            double[] a = first.Train(start, 25, 4, 0.08, 2, 2);
            double[] b = second.Train(start, 25, 4, 0.08, 2, 2);
            bool same = true;
            for (int d = 0; d < Dimension; d++) if (a[d] != b[d]) same = false;
            Check(same, "the simulation is deterministic");
            Check(shards[0].Inputs[0][0] == new Shard(SamplesPerClient, Dimension, 0x1000UL).Inputs[0][0],
                "the shard generator is deterministic for a fixed seed");
        }

        Console.WriteLine("PASS");
    }
}
