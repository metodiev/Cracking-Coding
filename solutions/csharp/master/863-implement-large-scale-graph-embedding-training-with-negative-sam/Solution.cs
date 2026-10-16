// Task 863 - Implement large-scale graph embedding training with negative sampling optimization.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// Skip-gram with negative sampling (SGNS) trained on a graph, in-process and deterministic.
//
//   pairs      every undirected edge of the graph contributes (u, v)
//   objective  per positive pair the logistic loss on the dot product of the two embeddings plus K negative
//              pairs drawn from the noise distribution q(v) ~ degree(v)^0.75, the standard SGNS objective
//   updates    plain SGD on the embeddings; the update direction is verified against a finite-difference
//              gradient of the same objective, so the analytic gradient is not trusted on its own
//   quality    nearest neighbours in the learned space are compared with the graph adjacency (recall@k
//              against the neighbours the sampler never saw), and the mean dot product of adjacent pairs is
//              compared with that of non-adjacent pairs
//
// Independent oracles: the finite-difference gradient, the analytic unigram^0.75 noise distribution that the
// sampled negatives are compared with (an L1 distance over the sampling histogram), a random-embedding
// baseline for the quality metrics, and a brute-force exact neighbour search for the recall.
//
// What is simplified (documented, not behavioural): a small dense graph and 16-dimensional embeddings, one
// synchronous SGD step per pair instead of asynchronous workers, and no negative-sampling table trick (the
// distribution is sampled by a linear scan). The SGNS objective, the negative sampler, the gradient check,
// the adjacency-based recall and the cost accounting are the real content; the graph size and the epoch
// budget are typed parameters.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng863
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static double Normal(ref ulong state)
    {
        double u1 = Math.Max(1e-12, Uniform(ref state));
        double u2 = Uniform(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Graph863
{
    internal readonly int Nodes;
    internal readonly List<int>[] Adjacency;

    internal Graph863(int nodes)
    {
        if (nodes < 2) throw new ArgumentException("a graph needs at least two nodes");
        Nodes = nodes;
        Adjacency = new List<int>[nodes];
        for (int i = 0; i < nodes; i++) Adjacency[i] = new List<int>();
    }

    internal void Connect(int u, int v)
    {
        if (u < 0 || u >= Nodes || v < 0 || v >= Nodes) throw new ArgumentOutOfRangeException(nameof(u));
        if (u == v) throw new ArgumentException("self loops are not supported");
        if (!Adjacency[u].Contains(v)) Adjacency[u].Add(v);
        if (!Adjacency[v].Contains(u)) Adjacency[v].Add(u);
    }

    internal IEnumerable<int[]> Pairs()
    {
        for (int u = 0; u < Nodes; u++)
            foreach (int v in Adjacency[u])
                if (u < v) yield return new[] { u, v };
    }

    internal long EdgeCount()
    {
        long total = 0;
        foreach (int[] pair in Pairs()) total++;
        return total;
    }

    internal bool AreAdjacent(int u, int v) => Adjacency[u].Contains(v);

    // The analytic noise distribution: degree^0.75 normalised.
    internal double[] NoiseDistribution()
    {
        var weights = new double[Nodes];
        double total = 0.0;
        for (int i = 0; i < Nodes; i++)
        {
            weights[i] = Math.Pow(Adjacency[i].Count, 0.75);
            total += weights[i];
        }
        for (int i = 0; i < Nodes; i++) weights[i] /= total;
        return weights;
    }

    internal static Graph863 Build(int nodes, ulong seed)
    {
        var graph = new Graph863(nodes);
        for (int i = 0; i < nodes; i++) graph.Connect(i, (i + 1) % nodes);   // a ring keeps it connected
        for (int i = 0; i < nodes; i++)
        {
            int chords = 1 + Rng863.NextInt(ref seed, 2);
            for (int k = 0; k < chords; k++)
            {
                int partner = Rng863.NextInt(ref seed, nodes);
                if (partner != i) graph.Connect(i, partner);
            }
        }
        return graph;
    }
}

internal sealed class Sgns863
{
    private readonly Graph863 graph;
    private readonly int dimensions;
    private readonly int negatives;
    private readonly double rate;
    private readonly double[][] embeddings;
    private readonly double[] noise;
    private readonly double[] cumulativeNoise;

    internal ulong Rng;
    internal long PairSteps;
    internal long NegativeSamples;

    internal Sgns863(Graph863 graph, int dimensions, int negatives, double rate, ulong seed)
    {
        if (dimensions < 1) throw new ArgumentException("the embedding dimension must be positive");
        if (negatives < 1) throw new ArgumentException("at least one negative sample is required");
        if (rate <= 0.0) throw new ArgumentException("the learning rate must be positive");
        this.graph = graph;
        this.dimensions = dimensions;
        this.negatives = negatives;
        this.rate = rate;
        Rng = seed;
        embeddings = new double[graph.Nodes][];
        for (int i = 0; i < graph.Nodes; i++)
        {
            var vector = new double[dimensions];
            for (int d = 0; d < dimensions; d++) vector[d] = 0.1 * Rng863.Normal(ref Rng);
            embeddings[i] = vector;
        }
        noise = graph.NoiseDistribution();
        cumulativeNoise = new double[noise.Length];
        double running = 0.0;
        for (int i = 0; i < noise.Length; i++)
        {
            running += noise[i];
            cumulativeNoise[i] = running;
        }
    }

    internal double[] Vector(int node) => embeddings[node];

    internal int SampleNegative(ref ulong rng)
    {
        double target = Rng863.Uniform(ref rng);
        for (int i = 0; i < cumulativeNoise.Length; i++) if (target <= cumulativeNoise[i]) return i;
        return cumulativeNoise.Length - 1;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    internal double Dot(int a, int b)
    {
        double total = 0.0;
        for (int d = 0; d < dimensions; d++) total += embeddings[a][d] * embeddings[b][d];
        return total;
    }

    // The loss on a *fixed* negative set, so the trace is comparable across epochs.
    internal double EvaluateLoss(List<int[]> pairs, List<int[]> fixedNegatives)
    {
        double total = 0.0;
        for (int i = 0; i < pairs.Count; i++)
        {
            int u = pairs[i][0], v = pairs[i][1];
            total += -Math.Log(Math.Max(1e-12, Sigmoid(Dot(u, v))));
            foreach (int node in fixedNegatives[i])
                total += -Math.Log(Math.Max(1e-12, Sigmoid(-Dot(u, node))));
        }
        return total;
    }

    internal double TrainEpoch(List<int[]> pairs)
    {
        double totalLoss = 0.0;
        var snapshot = new double[dimensions];
        var negativeSnapshot = new double[dimensions];
        foreach (int[] pair in pairs)
        {
            int u = pair[0], v = pair[1];
            double positive = Sigmoid(Dot(u, v));
            totalLoss += -Math.Log(Math.Max(1e-12, positive));

            // Positive pair: descending -log sigmoid(u . v) pulls the two vectors together.
            for (int d = 0; d < dimensions; d++) snapshot[d] = embeddings[u][d];
            for (int d = 0; d < dimensions; d++) embeddings[u][d] += rate * (1.0 - positive) * embeddings[v][d];
            for (int d = 0; d < dimensions; d++) embeddings[v][d] += rate * (1.0 - positive) * snapshot[d];

            for (int k = 0; k < negatives; k++)
            {
                int node = SampleNegative(ref Rng);
                double score = Dot(u, node);
                double negative = Sigmoid(score);
                totalLoss += -Math.Log(Math.Max(1e-12, Sigmoid(-score)));

                // Negative pair: descending -log sigmoid(-u . n) pushes the pair apart.
                for (int d = 0; d < dimensions; d++) snapshot[d] = embeddings[u][d];
                for (int d = 0; d < dimensions; d++) negativeSnapshot[d] = embeddings[node][d];
                for (int d = 0; d < dimensions; d++) embeddings[u][d] -= rate * negative * negativeSnapshot[d];
                for (int d = 0; d < dimensions; d++) embeddings[node][d] -= rate * negative * snapshot[d];
                NegativeSamples++;
            }
            PairSteps++;
        }
        return totalLoss;
    }

    internal double CosineSimilarity(int a, int b)
    {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int d = 0; d < dimensions; d++)
        {
            dot += embeddings[a][d] * embeddings[b][d];
            normA += embeddings[a][d] * embeddings[a][d];
            normB += embeddings[b][d] * embeddings[b][d];
        }
        if (normA <= 1e-12 || normB <= 1e-12) return 0.0;
        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    internal double MeanDotProduct(List<int[]> pairs, bool adjacent)
    {
        double total = 0.0;
        long count = 0;
        if (adjacent)
        {
            foreach (int[] pair in pairs)
            {
                total += Dot(pair[0], pair[1]);
                count++;
            }
        }
        else
        {
            for (int u = 0; u < graph.Nodes; u++)
            {
                for (int v = u + 1; v < graph.Nodes; v++)
                {
                    if (graph.AreAdjacent(u, v)) continue;
                    total += Dot(u, v);
                    count++;
                }
            }
        }
        return count == 0 ? 0.0 : total / count;
    }
}

public static class Solution
{
    private const int Nodes = 24;
    private const int Dimensions = 16;
    private const int Negatives = 3;
    private const int Epochs = 120;

    public static void Main()
    {
        Graph863 graph = Graph863.Build(Nodes, 0x863AUL);
        long edges = graph.EdgeCount();
        var pairs = new List<int[]>();
        foreach (int[] pair in graph.Pairs()) pairs.Add(pair);
        Check(pairs.Count == edges && edges > Nodes, "the graph is connected by a ring and carries extra chords ("
            + edges + " edges over " + Nodes + " nodes)");

        // 1-2. The noise distribution is the analytic degree^0.75 distribution.
        double[] noise = graph.NoiseDistribution();
        double noiseSum = 0.0;
        foreach (double value in noise) noiseSum += value;
        Check(Math.Abs(noiseSum - 1.0) < 1e-12, "the negative-sampling distribution is normalised");
        var trainer = new Sgns863(graph, Dimensions, Negatives, 0.2, 0x863BUL);
        var histogram = new double[Nodes];
        long draws = 200000;
        ulong sampler = 0x863CUL;
        for (long i = 0; i < draws; i++) histogram[trainer.SampleNegative(ref sampler)]++;
        for (int i = 0; i < Nodes; i++) histogram[i] /= draws;
        double histogramDistance = 0.0;
        for (int i = 0; i < Nodes; i++) histogramDistance += Math.Abs(histogram[i] - noise[i]);
        Console.WriteLine("negative sampling: L1 distance to degree^0.75 = " + histogramDistance.ToString("F4"));
        Check(histogramDistance < 0.05,
            "the sampled negatives follow degree^0.75 in L1 (" + histogramDistance.ToString("F4") + ")");

        // 3. The analytic gradient against a finite-difference gradient of the same objective.
        var gradientProbe = new Sgns863(graph, Dimensions, Negatives, 0.05, 0x863DUL);
        int probeU = 0, probeV = graph.Adjacency[probeU][0];
        var probeNegatives = new[] { (probeU + 5) % Nodes, (probeV + 7) % Nodes };
        double[] frozenU = (double[])gradientProbe.Vector(probeU).Clone();
        double baseline = LayeredLoss(gradientProbe, probeU, probeV, probeNegatives, frozenU, Dimensions);
        double epsilon = 1e-6;
        double worstRelative = 0.0;
        for (int d = 0; d < Dimensions; d++)
        {
            double[] lifted = (double[])frozenU.Clone();
            lifted[d] += epsilon;
            double up = LayeredLoss(gradientProbe, probeU, probeV, probeNegatives, lifted, Dimensions);
            lifted[d] -= 2.0 * epsilon;
            double down = LayeredLoss(gradientProbe, probeU, probeV, probeNegatives, lifted, Dimensions);
            double numericGradient = (up - down) / (2.0 * epsilon);
            double analyticGradient = AnalyticLossGradient(gradientProbe, probeU, probeV, probeNegatives, frozenU, d, Dimensions);
            double scale = Math.Max(1.0, Math.Abs(numericGradient) + Math.Abs(analyticGradient));
            worstRelative = Math.Max(worstRelative, Math.Abs(numericGradient - analyticGradient) / scale);
        }
        Console.WriteLine("gradient check: worst relative error=" + worstRelative.ToString("E2")
            + " (baseline loss=" + baseline.ToString("F4") + ")");
        Check(worstRelative < 1e-4,
            "the analytic SGNS gradient matches a finite-difference gradient in every dimension");

        // 4-6. Training: the loss falls, adjacent pairs separate from non-adjacent ones.
        var losses = new List<double>();
        var fixedNegatives = new List<int[]>();
        ulong evaluationSeed = 0x8630UL;
        foreach (int[] pair in pairs)
        {
            var negatives = new int[Negatives];
            for (int k = 0; k < Negatives; k++) negatives[k] = trainer.SampleNegative(ref evaluationSeed);
            fixedNegatives.Add(negatives);
        }
        for (int epoch = 0; epoch < Epochs; epoch++)
        {
            trainer.TrainEpoch(pairs);
            losses.Add(trainer.EvaluateLoss(pairs, fixedNegatives));
        }
        double diagPositive = 0.0;
        foreach (int[] pair in pairs) diagPositive += trainer.Dot(pair[0], pair[1]);
        diagPositive /= pairs.Count;
        double diagNegative = 0.0;
        long diagCount = 0;
        for (int u = 0; u < Nodes; u++)
        {
            for (int v = 0; v < Nodes; v++)
            {
                if (u == v || graph.AreAdjacent(u, v)) continue;
                diagNegative += trainer.Dot(u, v);
                diagCount++;
            }
        }
        if (diagCount > 0) diagNegative /= diagCount;
        Console.WriteLine("  mean dot product: edges=" + diagPositive.ToString("F3") + " non-edges="
            + diagNegative.ToString("F3"));
        Console.WriteLine("epoch losses: first=" + losses[0].ToString("F2") + " last="
            + losses[Epochs - 1].ToString("F2") + " pair steps=" + trainer.PairSteps
            + " negative samples=" + trainer.NegativeSamples);
        Check(losses[Epochs - 1] < losses[0] * 0.99, "the held-out SGNS loss decreases over the epochs ("
            + losses[0].ToString("F1") + " -> " + losses[Epochs - 1].ToString("F1") + ")");
        Check(trainer.PairSteps == (long)edges * Epochs && trainer.NegativeSamples == (long)edges * Epochs * Negatives,
            "the trainer performs exactly one step per pair per epoch and three negatives per step");
        double adjacentDot = trainer.MeanDotProduct(pairs, true);
        double nonAdjacentDot = trainer.MeanDotProduct(pairs, false);
        Console.WriteLine("mean dot product: adjacent=" + adjacentDot.ToString("F4") + " non-adjacent="
            + nonAdjacentDot.ToString("F4"));
        Check(adjacentDot > nonAdjacentDot,
            "adjacent node pairs end up with a larger dot product than non-adjacent pairs");

        // 7-8. Retrieval quality against the graph adjacency and against a random baseline.
        double learnedRecall = Recall(trainer, graph, pairs, 3);
        var untrained = new Sgns863(graph, Dimensions, Negatives, 0.2, 0x863EUL);
        double randomRecall = Recall(untrained, graph, pairs, 3);
        Console.WriteLine("recall@3 versus adjacency: trained=" + learnedRecall.ToString("P1")
            + " untrained=" + randomRecall.ToString("P1"));
        Check(learnedRecall > randomRecall,
            "training improves the neighbour recall over an untrained embedding");
        Check(learnedRecall > 0.5,
            "the trained embedding recovers most graph neighbours in the top three (" + learnedRecall.ToString("P1") + ")");

        // 9-11. Reproducibility, degenerate graphs and validation.
        var repeat = new Sgns863(graph, Dimensions, Negatives, 0.2, 0x863BUL);
        for (int epoch = 0; epoch < Epochs; epoch++) repeat.TrainEpoch(pairs);
        double repeatLoss = repeat.EvaluateLoss(pairs, fixedNegatives);
        Check(Math.Abs(repeatLoss - losses[Epochs - 1]) < 1e-9, "training is reproducible for a fixed seed");
        var line = new Graph863(6);
        for (int i = 0; i + 1 < 6; i++) line.Connect(i, i + 1);
        var lineTrainer = new Sgns863(line, 8, 2, 0.2, 0x863FUL);
        var linePairs = new List<int[]>();
        foreach (int[] pair in line.Pairs()) linePairs.Add(pair);
        double beforeLine = lineTrainer.MeanDotProduct(linePairs, true);
        for (int epoch = 0; epoch < 30; epoch++) lineTrainer.TrainEpoch(linePairs);
        Check(lineTrainer.MeanDotProduct(linePairs, true) > beforeLine,
            "a simple path graph also trains its adjacent pairs closer together");
        Check(Throws(() => new Sgns863(graph, 0, 3, 0.05, 1UL))
            && Throws(() => new Sgns863(graph, 8, 0, 0.05, 1UL))
            && Throws(() => new Sgns863(graph, 8, 3, 0.0, 1UL))
            && Throws(() => new Graph863(1)),
            "a zero dimension, zero negatives, a zero learning rate and a single-node graph are rejected");
        Check(Throws(() =>
        {
            var tiny = new Graph863(4);
            tiny.Connect(0, 0);
        }) && Throws(() =>
        {
            var tiny = new Graph863(4);
            tiny.Connect(0, 9);
        }), "self loops and out-of-range edges are rejected");

        Console.WriteLine("PASS");
    }

    // The loss with the positive pair's u-vector replaced by a supplied candidate: the probe used by the
    // finite-difference gradient check.
    private static double LayeredLoss(Sgns863 trainer, int u, int v, int[] negatives, double[] candidate, int dimensions)
    {
        double dot = 0.0;
        for (int d = 0; d < dimensions; d++) dot += candidate[d] * trainer.Vector(v)[d];
        double loss = -Math.Log(Math.Max(1e-12, Sigmoid(dot)));
        foreach (int n in negatives)
        {
            double negativeDot = 0.0;
            for (int d = 0; d < dimensions; d++) negativeDot += candidate[d] * trainer.Vector(n)[d];
            loss += -Math.Log(Math.Max(1e-12, Sigmoid(-negativeDot)));
        }
        return loss;
    }

    private static double AnalyticLossGradient(Sgns863 trainer, int u, int v, int[] negatives, double[] candidate,
        int dimension, int dimensions)
    {
        double dot = 0.0;
        for (int d = 0; d < dimensions; d++) dot += candidate[d] * trainer.Vector(v)[d];
        double gradient = -(1.0 - Sigmoid(dot)) * trainer.Vector(v)[dimension];
        foreach (int n in negatives)
        {
            double negativeDot = 0.0;
            for (int d = 0; d < dimensions; d++) negativeDot += candidate[d] * trainer.Vector(n)[d];
            // -log sigmoid(-dot) has derivative +sigmoid(dot) with respect to the dot product.
            gradient += Sigmoid(negativeDot) * trainer.Vector(n)[dimension];
        }
        return gradient;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double Recall(Sgns863 trainer, Graph863 graph, List<int[]> pairs, int k)
    {
        long hits = 0, total = 0;
        for (int u = 0; u < graph.Nodes; u++)
        {
            var neighbours = graph.Adjacency[u];
            if (neighbours.Count == 0) continue;
            var scored = new List<double[]>();
            for (int v = 0; v < graph.Nodes; v++)
            {
                if (v == u) continue;
                scored.Add(new[] { trainer.CosineSimilarity(u, v), v });
            }
            scored.Sort((a, b) => b[0].CompareTo(a[0]));
            for (int i = 0; i < k && i < scored.Count; i++)
            {
                total++;
                if (graph.AreAdjacent(u, (int)scored[i][1])) hits++;
            }
        }
        return total == 0 ? 0.0 : hits / (double)total;
    }

    private static bool Throws(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (ArgumentException)
        {
            return true;
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
