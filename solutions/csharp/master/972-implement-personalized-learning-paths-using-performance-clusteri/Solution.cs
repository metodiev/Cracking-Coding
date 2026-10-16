// Task 972 - Implement personalized learning paths using performance clustering.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Each learner is a three dimensional skill vector (accuracy, speed, persistence). Lloyd's k-means with
// deterministic farthest-first seeding - first seed is the first learner, each next seed is the learner
// farthest from the chosen set, ties to the lowest index - groups them, and the clusters are ranked by
// mean accuracy so every learner is routed to a track: rank 1 is the remedial path, the top rank is the
// advanced path.
//
// Verified: recovery of three injected skill groups (pairwise partition oracle), nearest-centroid
// optimality of every assignment, non-increasing Lloyd inertia, the k=1 / k>n / empty boundaries,
// bitwise determinism across two runs, and track ranks that agree with an independent mean ordering.
// Simplification: 18 in-memory learners in three dimensions and three clusters with jittered blobs; no
// feature scaling, no model persistence and no convergence proof beyond the observed stable assignment.
using System;
using System.Collections.Generic;

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

    private static double NextUnit() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Learner
    {
        public readonly double[] Skills;
        public Learner(double[] skills) { Skills = skills; }
    }

    private sealed class Clustering
    {
        public int[] Labels = new int[0]; public double[][] Centroids = new double[0][];
        public readonly List<double> Inertia = new List<double>();
        public bool Valid;
    }

    private static double SquaredDistance(double[] a, double[] b)
    {
        double total = 0.0;
        for (int i = 0; i < a.Length; i++) { double difference = a[i] - b[i]; total += difference * difference; }
        return total;
    }

    private static int Nearest(double[][] centroids, double[] point)
    {
        int best = 0;
        double bestDistance = double.MaxValue;
        for (int c = 0; c < centroids.Length; c++)
        {
            double distance = SquaredDistance(centroids[c], point);
            if (distance < bestDistance - 1e-15) { bestDistance = distance; best = c; }
        }
        return best;
    }

    /** Farthest-first seeding: deterministic, and it spreads the seeds over the observed skills. */
    private static double[][] Seed(List<Learner> data, int k)
    {
        var chosen = new List<double[]> { data[0].Skills };
        while (chosen.Count < k)
        {
            int pick = -1;
            double pickDistance = -1.0;
            for (int i = 0; i < data.Count; i++)
            {
                double nearest = double.MaxValue;
                foreach (double[] centre in chosen) nearest = Math.Min(nearest, SquaredDistance(centre, data[i].Skills));
                if (nearest > pickDistance + 1e-15) { pickDistance = nearest; pick = i; }
            }
            chosen.Add(data[pick].Skills);
        }
        var centroids = new double[k][];
        for (int c = 0; c < k; c++) centroids[c] = (double[])chosen[c].Clone();
        return centroids;
    }

    private static double[][] Recentre(List<Learner> data, int[] labels, double[][] previous, int k)
    {
        var sums = new double[k][]; var counts = new int[k];
        for (int c = 0; c < k; c++) sums[c] = new double[data[0].Skills.Length];
        for (int i = 0; i < data.Count; i++)
        {
            counts[labels[i]]++;
            for (int d = 0; d < data[i].Skills.Length; d++) sums[labels[i]][d] += data[i].Skills[d];
        }
        var centroids = new double[k][];
        for (int c = 0; c < k; c++)
        {
            if (counts[c] == 0) { centroids[c] = (double[])previous[c].Clone(); continue; }
            var centre = new double[data[0].Skills.Length];
            for (int d = 0; d < centre.Length; d++) centre[d] = sums[c][d] / counts[c];
            centroids[c] = centre;
        }
        return centroids;
    }

    private static double Inertia(List<Learner> data, int[] labels, double[][] centroids)
    {
        double total = 0.0;
        for (int i = 0; i < data.Count; i++) total += SquaredDistance(data[i].Skills, centroids[labels[i]]);
        return total;
    }

    private static Clustering KMeans(List<Learner> data, int k, int maxIterations)
    {
        var result = new Clustering();
        if (k < 1 || data.Count == 0 || k > data.Count) return result;
        double[][] centroids = Seed(data, k);
        int[] labels = new int[data.Count];
        for (int i = 0; i < labels.Length; i++) labels[i] = -1;
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            bool changed = false;
            for (int i = 0; i < data.Count; i++)
            {
                int label = Nearest(centroids, data[i].Skills);
                if (label != labels[i]) { labels[i] = label; changed = true; }
            }
            centroids = Recentre(data, labels, centroids, k);
            result.Inertia.Add(Inertia(data, labels, centroids));
            if (!changed) break;
        }
        result.Labels = labels; result.Centroids = centroids; result.Valid = true;
        return result;
    }

    private static double Jitter(double centre) => centre + (NextUnit() - 0.5) * 0.1;

    private static List<Learner> Blobs(ulong seed, int perGroup)
    {
        _state = seed;
        var data = new List<Learner>();
        foreach (double centre in new[] { 0.3, 0.6, 0.9 })
            for (int i = 0; i < perGroup; i++)
                data.Add(new Learner(new[] { Jitter(centre), Jitter(centre), Jitter(centre) }));
        return data;
    }

    private static double MeanSkill(List<Learner> data, int[] labels, int cluster)
    {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < data.Count; i++) if (labels[i] == cluster) { sum += data[i].Skills[0]; count++; }
        return count == 0 ? double.NaN : sum / count;
    }

    /** Track rank 1 is the weakest cluster (remedial path); the top rank is the advanced track. */
    private static Dictionary<int, int> TrackRanks(List<Learner> data, int[] labels, int k)
    {
        var order = new List<int>();
        for (int c = 0; c < k; c++) order.Add(c);
        order.Sort((a, b) =>
        {
            double meanA = MeanSkill(data, labels, a), meanB = MeanSkill(data, labels, b);
            return Math.Abs(meanA - meanB) > 1e-12 ? meanA.CompareTo(meanB) : a.CompareTo(b);
        });
        var ranks = new Dictionary<int, int>();
        for (int r = 0; r < order.Count; r++) ranks[order[r]] = r + 1;
        return ranks;
    }

    public static void Main()
    {
        const int perGroup = 6;
        List<Learner> data = Blobs(20240972UL, perGroup);
        Clustering result = KMeans(data, 3, 100);
        int outOfRange = 0;
        foreach (int label in result.Labels) if (label < 0 || label > 2) outOfRange++;
        Check(result.Valid && result.Labels.Length == data.Count && outOfRange == 0,
            "three clusters are produced for 18 learners and every label is a valid cluster index");

        // Oracle 1: pairwise partition comparison against the injected skill groups.
        int disagreements = 0;
        for (int i = 0; i < data.Count; i++)
            for (int j = i + 1; j < data.Count; j++)
                if ((i / perGroup == j / perGroup) != (result.Labels[i] == result.Labels[j])) disagreements++;
        Check(disagreements == 0, "the clustering reproduces the three injected skill groups pairwise");

        // Oracle 2 and the convergence property: with the final centroids no learner prefers another one.
        int reassigned = 0;
        for (int i = 0; i < data.Count; i++)
            if (Nearest(result.Centroids, data[i].Skills) != result.Labels[i]) reassigned++;
        Check(reassigned == 0, "every learner sits in the cluster whose centroid is nearest");

        bool monotone = true;
        for (int i = 1; i < result.Inertia.Count; i++) monotone &= result.Inertia[i] <= result.Inertia[i - 1] + 1e-9;
        Check(monotone, "Lloyd inertia never increases over the " + result.Inertia.Count + " iterations");

        Clustering again = KMeans(data, 3, 100);
        bool identical = true;
        for (int i = 0; i < data.Count; i++) identical &= again.Labels[i] == result.Labels[i];
        for (int c = 0; c < 3; c++)
            for (int d = 0; d < 3; d++) identical &= again.Centroids[c][d] == result.Centroids[c][d];
        Check(identical, "two runs produce bitwise identical labels and centroids");

        // Boundary: k=1 collapses everything, k>n and an empty cohort are refused.
        Clustering single = KMeans(data, 1, 50);
        int zeros = 0;
        foreach (int label in single.Labels) if (label == 0) zeros++;
        Check(single.Valid && zeros == data.Count, "k=1 assigns every learner to a single cluster");
        Check(!KMeans(data, data.Count + 1, 50).Valid && !KMeans(new List<Learner>(), 3, 50).Valid,
            "asking for more clusters than learners, or clustering an empty cohort, is refused");

        Dictionary<int, int> ranks = TrackRanks(data, result.Labels, 3);
        int strongest = 0;
        for (int i = 1; i < data.Count; i++) if (data[i].Skills[0] > data[strongest].Skills[0]) strongest = i;
        Check(ranks[result.Labels[strongest]] == 3, "the strongest learner is routed to track 3, the advanced path");
        int inversions = 0;
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                if (MeanSkill(data, result.Labels, a) < MeanSkill(data, result.Labels, b) - 1e-12 && ranks[a] > ranks[b]) inversions++;
        Check(inversions == 0, "track ranks follow the independent per-cluster mean skill ordering");

        Console.WriteLine("clusters: 3 over " + data.Count + " learners, inertia "
            + result.Inertia[0].ToString("F4") + " -> " + result.Inertia[result.Inertia.Count - 1].ToString("F4")
            + " in " + result.Inertia.Count + " iterations");
        Console.WriteLine("tracks: strongest learner on rank " + ranks[result.Labels[strongest]]
            + ", rank inversions " + inversions);
        Console.WriteLine("PASS");
    }
}
