// Task 518 - Implement k-nearest neighbor classification fast with kd-tree. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// A kd-tree over n labelled samples. Every node splits on axis (depth % d) at the median of the
// coordinates of the samples it owns, ties broken by sample index, so the tree shape - and hence the
// search order and every counter - is a pure function of the data (no randomness anywhere). A query
// keeps a best-list of the k closest samples under the total order (squared distance, sample index);
// it descends first into the child on the query's side of the splitting plane and visits the far
// child only when the squared distance to that plane is <= the current k-th best distance. The plane
// distance lower-bounds every distance in the far halfspace, which makes a strict ">" test safe,
// while the "==" case must still be visited so that distance ties resolve exactly like the
// brute-force scan (a point at the current k-th distance but with a smaller index could hide there).
// The class exposes DistanceChecks / NodesVisited so the speed-up can be measured: on the 300-point
// 2-D test set the tree evaluates at least 2x fewer distances (in practice an order of magnitude
// fewer) than the naive scan while returning element-wise identical neighbour lists for k = 1, 3, 7,
// 15 and k = n.
//
// Documented checks. (1) Structure: the tree holds every index exactly once, its depth equals
// floor(log2 n) + 1 (the median-split depth recurrence), and the per-node split invariant - the
// lexicographic (coordinate, index) bounds implied by all ancestor splits on each axis - is verified
// recursively against the actual point coordinates. (2) Correctness: for four 2-D and one 5-D LCG
// data sets and several k the kd-tree neighbour list equals the brute-force list, including a
// duplicate-heavy set and a hand-made four-way distance tie (which must resolve by index), and the
// majority vote over the two lists is identical. (3) Counting: the kd-tree distance evaluations are
// at most half the naive ones over the same queries. (4) Boundaries: self queries return the probe
// point first, k = n returns the whole set in distance order with the global-majority vote, a
// one-sample tree answers, k = 0, k > n and a wrong query dimension each throw ArgumentException, and
// a rebuilt tree reproduces identical neighbour lists. Classification is the k-neighbour majority
// vote with the smallest label breaking ties; a jittered two-blob set must classify with zero errors
// at k = 3.
using System;
using System.Collections.Generic;

internal sealed class Sample
{
    internal readonly double[] X;
    internal readonly int Label;

    internal Sample(double[] x, int label)
    {
        X = x;
        Label = label;
    }
}

internal sealed class KdTree
{
    private sealed class Node
    {
        internal readonly int Axis;
        internal readonly int PointIndex;
        internal readonly Node? Left;
        internal readonly Node? Right;

        internal Node(int axis, int pointIndex, Node? left, Node? right)
        {
            Axis = axis;
            PointIndex = pointIndex;
            Left = left;
            Right = right;
        }
    }

    private readonly Sample[] samples;
    private readonly int dim;
    private readonly Node? root;

    internal long DistanceChecks;
    internal long NodesVisited;

    internal int Count => samples.Length;
    internal int Depth => DepthOf(root);

    internal KdTree(Sample[] samples, int dim)
    {
        if (dim < 1) throw new ArgumentOutOfRangeException(nameof(dim));
        this.samples = samples;
        this.dim = dim;
        var order = new int[samples.Length];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        root = Build(order, 0, order.Length, 0);
    }

    private Node? Build(int[] indices, int lo, int hi, int depth)
    {
        if (lo >= hi) return null;
        int axis = depth % dim;
        int count = hi - lo;
        var keys = new int[count];
        Array.Copy(indices, lo, keys, 0, count);
        Array.Sort(keys, (a, b) =>
        {
            int cmp = samples[a].X[axis].CompareTo(samples[b].X[axis]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        int mid = count / 2;
        return new Node(axis, keys[mid],
                Build(keys, 0, mid, depth + 1),
                Build(keys, mid + 1, count, depth + 1));
    }

    private static int DepthOf(Node? node)
    {
        return node == null ? 0 : 1 + Math.Max(DepthOf(node.Left), DepthOf(node.Right));
    }

    internal int[] Query(double[] query, int k)
    {
        if (query.Length != dim)
        {
            throw new ArgumentException("query dimension " + query.Length + " != " + dim, nameof(query));
        }
        if (k < 1) throw new ArgumentOutOfRangeException(nameof(k));
        if (k > samples.Length)
        {
            throw new ArgumentException("k = " + k + " exceeds the " + samples.Length + " samples", nameof(k));
        }
        var best = new List<int>(k);
        var dist = new List<double>(k);
        Search(root, query, k, best, dist);
        return best.ToArray();
    }

    private void Search(Node? node, double[] query, int k, List<int> best, List<double> dist)
    {
        if (node == null) return;
        NodesVisited++;
        double d2 = SquaredDistance(query, node.PointIndex);
        int pos = 0;
        while (pos < best.Count && (dist[pos] < d2 || (dist[pos] == d2 && best[pos] < node.PointIndex)))
        {
            pos++;
        }
        if (pos < k)
        {
            best.Insert(pos, node.PointIndex);
            dist.Insert(pos, d2);
            if (best.Count > k)
            {
                best.RemoveAt(k);
                dist.RemoveAt(k);
            }
        }
        double delta = query[node.Axis] - samples[node.PointIndex].X[node.Axis];
        bool leftFirst = delta <= 0.0;
        Search(leftFirst ? node.Left : node.Right, query, k, best, dist);
        double worst = dist.Count == k ? dist[k - 1] : double.PositiveInfinity;
        if (delta * delta <= worst) Search(leftFirst ? node.Right : node.Left, query, k, best, dist);
    }

    private double SquaredDistance(double[] query, int index)
    {
        DistanceChecks++;
        double[] x = samples[index].X;
        double sum = 0.0;
        for (int d = 0; d < dim; d++)
        {
            double diff = query[d] - x[d];
            sum += diff * diff;
        }
        return sum;
    }

    internal int[] PointIndices()
    {
        var collected = new List<int>();
        Collect(root, collected);
        int[] result = collected.ToArray();
        Array.Sort(result);
        return result;
    }

    private static void Collect(Node? node, List<int> into)
    {
        if (node == null) return;
        into.Add(node.PointIndex);
        Collect(node.Left, into);
        Collect(node.Right, into);
    }

    /** Recursively verifies that every point in a subtree respects all ancestor splits. */
    internal bool VerifyInvariant()
    {
        var lowX = new double[dim];
        var lowI = new int[dim];
        var highX = new double[dim];
        var highI = new int[dim];
        for (int d = 0; d < dim; d++)
        {
            lowX[d] = double.NegativeInfinity;
            lowI[d] = int.MinValue;
            highX[d] = double.PositiveInfinity;
            highI[d] = int.MaxValue;
        }
        return Verify(root, 0, lowX, lowI, highX, highI);
    }

    private bool Verify(Node? node, int depth, double[] lowX, int[] lowI, double[] highX, int[] highI)
    {
        if (node == null) return true;
        int axis = depth % dim;
        int index = node.PointIndex;
        double coord = samples[index].X[axis];
        if (Compare(coord, index, lowX[axis], lowI[axis]) <= 0) return false;
        if (Compare(coord, index, highX[axis], highI[axis]) >= 0) return false;
        var leftHighX = (double[])highX.Clone();
        var leftHighI = (int[])highI.Clone();
        leftHighX[axis] = coord;
        leftHighI[axis] = index;
        var rightLowX = (double[])lowX.Clone();
        var rightLowI = (int[])lowI.Clone();
        rightLowX[axis] = coord;
        rightLowI[axis] = index;
        return Verify(node.Left, depth + 1, lowX, lowI, leftHighX, leftHighI)
            && Verify(node.Right, depth + 1, rightLowX, rightLowI, highX, highI);
    }

    private static int Compare(double coordA, int indexA, double coordB, int indexB)
    {
        int cmp = coordA.CompareTo(coordB);
        return cmp != 0 ? cmp : indexA.CompareTo(indexB);
    }

    /** Naive k-nearest-neighbour scan used as the reference implementation. */
    internal static int[] BruteForce(Sample[] samples, double[] query, int k, ref long checks)
    {
        int dim = query.Length;
        var dist = new double[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            checks++;
            double sum = 0.0;
            for (int d = 0; d < dim; d++)
            {
                double diff = query[d] - samples[i].X[d];
                sum += diff * diff;
            }
            dist[i] = sum;
        }
        var order = new int[samples.Length];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int cmp = dist[a].CompareTo(dist[b]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        var result = new int[k];
        Array.Copy(order, result, k);
        return result;
    }

    /** Majority vote of the labels of the given neighbours; the smallest label breaks ties. */
    internal static int Classify(Sample[] samples, int[] neighbours)
    {
        var counts = new Dictionary<int, int>();
        foreach (int index in neighbours)
        {
            int label = samples[index].Label;
            counts.TryGetValue(label, out int current);
            counts[label] = current + 1;
        }
        int bestLabel = int.MaxValue;
        int bestCount = -1;
        foreach (KeyValuePair<int, int> entry in counts)
        {
            if (entry.Value > bestCount || (entry.Value == bestCount && entry.Key < bestLabel))
            {
                bestCount = entry.Value;
                bestLabel = entry.Key;
            }
        }
        return bestLabel;
    }
}

public static class Solution
{
    public static void Main()
    {
        Sample[] set = LcgSamples(300, 91, 2, 40, 3);
        var tree = new KdTree(set, 2);
        Check(tree.Count == set.Length, "the tree holds every sample");
        Check(tree.Depth == FloorLog2(set.Length) + 1, "median splits give depth floor(log2 n) + 1");
        Check(tree.VerifyInvariant(), "kd-tree split invariant holds");
        int[] indices = tree.PointIndices();
        Check(indices.Length == set.Length, "one node per sample");
        bool permutation = true;
        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] != i) permutation = false;
        }
        Check(permutation, "every sample index appears exactly once");

        double[][] queries = LcgQueries(200, 555, 2, 40);
        int[] ks = { 1, 3, 7, 15 };
        long naiveChecks = 0;
        foreach (int k in ks)
        {
            for (int q = 0; q < queries.Length; q++)
            {
                int[] fast = tree.Query(queries[q], k);
                long perQuery = 0;
                int[] slow = KdTree.BruteForce(set, queries[q], k, ref perQuery);
                naiveChecks += perQuery;
                SameList(fast, slow, "kd-tree neighbours, k = " + k + ", query " + q);
                Check(KdTree.Classify(set, fast) == KdTree.Classify(set, slow),
                        "majority vote, k = " + k + ", query " + q);
            }
        }
        Check(tree.DistanceChecks * 2 <= naiveChecks,
                "kd-tree evaluates at most half the distances: " + tree.DistanceChecks + " vs " + naiveChecks);
        Check(tree.NodesVisited > 0, "node visits were counted");

        var rebuilt = new KdTree(set, 2);
        for (int q = 0; q < 40; q++)
        {
            SameList(tree.Query(queries[q], 7), rebuilt.Query(queries[q], 7),
                    "rebuilt tree gives the same neighbours, query " + q);
        }

        Sample[] grid = GridSamples(64, 3);
        var gridTree = new KdTree(grid, 2);
        Check(gridTree.Depth == FloorLog2(grid.Length) + 1, "grid tree depth");
        for (int i = 0; i < grid.Length; i++)
        {
            Check(gridTree.Query(grid[i].X, 1)[0] == i, "self query returns the probe point: " + i);
        }

        Sample[] tied = {
            new Sample(new[] { -1.0, 0.0 }, 0),
            new Sample(new[] { 1.0, 0.0 }, 1),
            new Sample(new[] { 0.0, 1.0 }, 2),
            new Sample(new[] { 0.0, -1.0 }, 3),
        };
        var tiedTree = new KdTree(tied, 2);
        var origin = new[] { 0.0, 0.0 };
        long tiedChecks = 0;
        int[] tiedTop = tiedTree.Query(origin, 3);
        SameList(tiedTop, KdTree.BruteForce(tied, origin, 3, ref tiedChecks), "four-way distance tie");
        Check(tiedTop[0] == 0 && tiedTop[1] == 1 && tiedTop[2] == 2, "ties resolve by sample index");

        Sample[] dup = LcgSamples(40, 7, 2, 4, 2);
        var dupTree = new KdTree(dup, 2);
        Check(HasDuplicatePoint(dup), "the duplicate-heavy set really repeats coordinates");
        double[][] dupQueries = LcgQueries(30, 999, 2, 4);
        foreach (int k in new[] { 2, 6, 11 })
        {
            for (int q = 0; q < dupQueries.Length; q++)
            {
                long checks = 0;
                SameList(dupTree.Query(dupQueries[q], k),
                        KdTree.BruteForce(dup, dupQueries[q], k, ref checks),
                        "duplicate-heavy set, k = " + k + ", query " + q);
            }
        }

        Sample[] five = LcgSamples(120, 31, 5, 25, 4);
        var fiveTree = new KdTree(five, 5);
        Check(fiveTree.VerifyInvariant(), "5-D split invariant holds");
        double[][] fiveQueries = LcgQueries(40, 4242, 5, 25);
        for (int q = 0; q < fiveQueries.Length; q++)
        {
            long checks = 0;
            SameList(fiveTree.Query(fiveQueries[q], 4),
                    KdTree.BruteForce(five, fiveQueries[q], 4, ref checks),
                    "5-D neighbours, query " + q);
        }

        var probe = new[] { 5.0, 5.0 };
        long allChecks = 0;
        int[] everything = tree.Query(probe, set.Length);
        SameList(everything, KdTree.BruteForce(set, probe, set.Length, ref allChecks),
                "k = n returns the whole set in distance order");
        Check(KdTree.Classify(set, everything) == MajorityLabel(set), "k = n vote is the global majority");
        Check(everything[0] == NearestIndex(set, probe), "k = n starts with the nearest sample");

        var single = new[] { new Sample(new[] { 3.0, 4.0 }, 7) };
        var singleTree = new KdTree(single, 2);
        Check(singleTree.Depth == 1, "one-sample tree has a single node");
        Check(singleTree.Query(origin, 1)[0] == 0, "one-sample tree answers");
        Check(KdTree.Classify(single, singleTree.Query(origin, 1)) == 7, "one-sample vote");

        CheckThrows(() => tree.Query(new[] { 1.0, 2.0, 3.0 }, 1), "wrong query dimension throws");
        CheckThrows(() => tree.Query(new[] { 1.0, 2.0 }, set.Length + 1), "k > n throws");
        CheckThrows(() => tree.Query(new[] { 1.0, 2.0 }, 0), "k = 0 throws");
        CheckThrows(() => new KdTree(set, 0), "dimension 0 throws");

        Sample[] blobs = TwoBlobs(60, 2, 777);
        var blobTree = new KdTree(blobs, 2);
        int correct = 0;
        long blobNaive = 0;
        for (int i = 0; i < blobs.Length; i++)
        {
            int[] fast = blobTree.Query(blobs[i].X, 3);
            SameList(fast, KdTree.BruteForce(blobs, blobs[i].X, 3, ref blobNaive),
                    "blob neighbours, sample " + i);
            if (KdTree.Classify(blobs, fast) == blobs[i].Label) correct++;
        }
        Check(correct == blobs.Length, "k = 3 classifies both blobs perfectly");

        Console.WriteLine("PASS");
    }

    private static void SameList(int[] a, int[] b, string label)
    {
        Check(a.Length == b.Length, label + ": length");
        for (int i = 0; i < a.Length; i++)
        {
            Check(a[i] == b[i], label + ": element " + i + " (" + a[i] + " != " + b[i] + ")");
        }
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Sample[] LcgSamples(int n, int seed, int dim, int span, int labels)
    {
        var samples = new Sample[n];
        for (int i = 0; i < n; i++)
        {
            var x = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                seed = (seed * 75 + 74) % 65537;
                x[d] = seed % span;
            }
            seed = (seed * 75 + 74) % 65537;
            samples[i] = new Sample(x, seed % labels);
        }
        return samples;
    }

    private static double[][] LcgQueries(int n, int seed, int dim, int span)
    {
        var queries = new double[n][];
        for (int i = 0; i < n; i++)
        {
            var x = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                seed = (seed * 75 + 74) % 65537;
                x[d] = 0.5 * span + seed % (2 * span) - span;
            }
            queries[i] = x;
        }
        return queries;
    }

    private static Sample[] GridSamples(int side, int spacing)
    {
        var samples = new Sample[side * side];
        int index = 0;
        for (int i = 0; i < side; i++)
        {
            for (int j = 0; j < side; j++)
            {
                samples[index] = new Sample(new[] { (double)(i * spacing), (double)(j * spacing) },
                        (i + j) % 2);
                index++;
            }
        }
        return samples;
    }

    private static Sample[] TwoBlobs(int perBlob, int jitter, int seed)
    {
        var samples = new Sample[2 * perBlob];
        int index = 0;
        for (int blob = 0; blob < 2; blob++)
        {
            double center = blob == 0 ? 0.0 : 12.0;
            for (int i = 0; i < perBlob; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                double x = center + seed % (2 * jitter + 1) - jitter;
                seed = (seed * 75 + 74) % 65537;
                double y = seed % (2 * jitter + 1) - jitter;
                samples[index++] = new Sample(new[] { x, y }, blob);
            }
        }
        return samples;
    }

    private static bool HasDuplicatePoint(Sample[] samples)
    {
        var seen = new HashSet<string>();
        foreach (Sample sample in samples)
        {
            var key = new System.Text.StringBuilder();
            foreach (double value in sample.X) key.Append(value).Append('|');
            if (!seen.Add(key.ToString())) return true;
        }
        return false;
    }

    private static int MajorityLabel(Sample[] samples)
    {
        var counts = new Dictionary<int, int>();
        foreach (Sample sample in samples)
        {
            counts.TryGetValue(sample.Label, out int current);
            counts[sample.Label] = current + 1;
        }
        int bestLabel = int.MaxValue;
        int bestCount = -1;
        foreach (KeyValuePair<int, int> entry in counts)
        {
            if (entry.Value > bestCount || (entry.Value == bestCount && entry.Key < bestLabel))
            {
                bestCount = entry.Value;
                bestLabel = entry.Key;
            }
        }
        return bestLabel;
    }

    private static int NearestIndex(Sample[] samples, double[] query)
    {
        int best = 0;
        double bestDist = double.PositiveInfinity;
        for (int i = 0; i < samples.Length; i++)
        {
            double sum = 0.0;
            for (int d = 0; d < query.Length; d++)
            {
                double diff = query[d] - samples[i].X[d];
                sum += diff * diff;
            }
            if (sum < bestDist)
            {
                bestDist = sum;
                best = i;
            }
        }
        return best;
    }

    private static int FloorLog2(int value)
    {
        int result = 0;
        while (value > 1)
        {
            value >>= 1;
            result++;
        }
        return result;
    }
}
