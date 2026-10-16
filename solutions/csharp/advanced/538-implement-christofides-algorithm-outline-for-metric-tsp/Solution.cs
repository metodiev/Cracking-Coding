// Task 538 - Implement Christofides algorithm outline for metric TSP. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
//
// A complete (not merely outlined) Christofides implementation for metric instances: distances are
// Euclidean and the file first verifies the triangle inequality numerically, so the 3/2 guarantee
// applies. The pipeline is (1) a minimum spanning tree by dense Prim with index tie-breaking,
// independently verified against Kruskal with a union-find, (2) the set of odd-degree vertices of
// that tree - always even and non-empty for n > 1 - (3) a minimum-weight perfect matching on the
// odd set, solved exactly by a bitmask dynamic program over the odd vertices and verified against
// brute-force enumeration of every perfect matching, (4) the Eulerian multigraph (tree + matching,
// every degree even by construction, which the file asserts) and its Euler circuit by Hierholzer's
// stack algorithm, and (5) shortcutting the circuit to first visits, which yields a Hamiltonian
// cycle no longer than the circuit by the triangle inequality. The Euler traversal consumes every
// multigraph edge exactly once (edge ids plus a used-flag, not the directed next-pointer shortcut
// that silently re-traverses undirected edges), so the circuit has |E| + 1 vertices and its length
// is exactly MST + matching. Exact Held-Karp optima (n <= 12) and the Christofides bounds
// (matching <= OPT/2, tour <= 3/2 * OPT) are the yardsticks.
//
// Documented checks: the matching found by the DP equals the brute-force minimum for a 10-node LCG
// instance and for a 6-node instance; the odd set has even size and every multigraph degree is
// even; the tree has n - 1 edges and (Prim) matches Kruskal's weight; the Euler circuit has |E| + 1
// vertices and length exactly MST + matching, and shortcutting never lengthens it; the produced
// tour is a permutation, lies between the Held-Karp optimum and 1.5 * OPT, satisfies
// matching <= OPT / 2, and the Held-Karp witness tour has the value the DP reports; the result is
// reproducible on a second run; the triangle inequality holds for all triples of the 10-node
// instance within 1e-9; and on three points the algorithm returns the exact optimum (the triangle
// perimeter) with 2 odd vertices and a one-edge matching.
using System;
using System.Collections.Generic;

internal sealed class Christofides
{
    internal readonly int N;
    internal readonly double[][] D;

    internal Christofides(double[] x, double[] y)
    {
        N = x.Length;
        D = new double[N][];
        for (int i = 0; i < N; i++)
        {
            D[i] = new double[N];
            for (int j = 0; j < N; j++)
            {
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                D[i][j] = Math.Sqrt(dx * dx + dy * dy);
            }
        }
    }

    internal bool TriangleInequality()
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < N; k++)
                {
                    if (D[i][k] > D[i][j] + D[j][k] + 1e-9) return false;
                }
            }
        }
        return true;
    }

    /** Dense Prim; ties resolved by the smaller vertex index. */
    internal List<int[]> PrimMst(out double weight)
    {
        var inTree = new bool[N];
        var bestEdge = new double[N];
        var parent = new int[N];
        for (int v = 0; v < N; v++)
        {
            bestEdge[v] = double.PositiveInfinity;
            parent[v] = -1;
        }
        bestEdge[0] = 0.0;
        var edges = new List<int[]>();
        weight = 0.0;
        for (int step = 0; step < N; step++)
        {
            int pick = -1;
            for (int v = 0; v < N; v++)
            {
                if (!inTree[v] && (pick < 0 || bestEdge[v] < bestEdge[pick] - 1e-15))
                {
                    pick = v;
                }
            }
            if (pick < 0) break;
            inTree[pick] = true;
            if (parent[pick] >= 0)
            {
                edges.Add(new[] { parent[pick], pick });
                weight += D[parent[pick]][pick];
            }
            for (int v = 0; v < N; v++)
            {
                if (inTree[v]) continue;
                double dist = D[pick][v];
                if (dist < bestEdge[v] - 1e-15)
                {
                    bestEdge[v] = dist;
                    parent[v] = pick;
                }
            }
        }
        return edges;
    }

    /** Independent MST weight: Kruskal over all pairs with a union-find. */
    internal double KruskalWeight()
    {
        var edges = new List<double[]>();
        for (int i = 0; i < N; i++)
        {
            for (int j = i + 1; j < N; j++) edges.Add(new[] { D[i][j], i, j });
        }
        edges.Sort((a, b) =>
        {
            int cmp = a[0].CompareTo(b[0]);
            if (cmp != 0) return cmp;
            cmp = a[1].CompareTo(b[1]);
            if (cmp != 0) return cmp;
            return a[2].CompareTo(b[2]);
        });
        var parent = new int[N];
        for (int v = 0; v < N; v++) parent[v] = v;
        double total = 0.0;
        int used = 0;
        foreach (double[] e in edges)
        {
            int a = Find(parent, (int)e[1]);
            int b = Find(parent, (int)e[2]);
            if (a == b) continue;
            parent[a] = b;
            total += e[0];
            used++;
            if (used == N - 1) break;
        }
        return used == N - 1 ? total : double.PositiveInfinity;
    }

    private static int Find(int[] parent, int v)
    {
        while (parent[v] != v)
        {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        return v;
    }

    internal static int[] OddVertices(IReadOnlyList<int[]> edges, int n)
    {
        var degree = new int[n];
        foreach (int[] e in edges)
        {
            degree[e[0]]++;
            degree[e[1]]++;
        }
        var odd = new List<int>();
        for (int v = 0; v < n; v++)
        {
            if (degree[v] % 2 == 1) odd.Add(v);
        }
        return odd.ToArray();
    }

    /** Exact minimum perfect matching over the odd set: bitmask DP, returns pairs and weight. */
    internal List<int[]> MinMatching(int[] odd, out double weight)
    {
        int m = odd.Length;
        if (m % 2 != 0) throw new InvalidOperationException("perfect matching needs an even odd-set");
        if (m == 0)
        {
            weight = 0.0;
            return new List<int[]>();
        }
        int full = 1 << m;
        var dp = new double[full];
        var choice = new int[full];
        for (int mask = 0; mask < full; mask++)
        {
            dp[mask] = double.PositiveInfinity;
            choice[mask] = -1;
        }
        dp[0] = 0.0;
        for (int mask = 1; mask < full; mask++)
        {
            int first = -1;
            for (int b = 0; b < m; b++)
            {
                if ((mask & (1 << b)) != 0)
                {
                    first = b;
                    break;
                }
            }
            if (first < 0) continue;
            for (int other = first + 1; other < m; other++)
            {
                if ((mask & (1 << other)) == 0) continue;
                int rest = mask ^ (1 << first) ^ (1 << other);
                if (double.IsPositiveInfinity(dp[rest])) continue;
                double cand = dp[rest] + D[odd[first]][odd[other]];
                if (cand < dp[mask])
                {
                    dp[mask] = cand;
                    choice[mask] = other;
                }
            }
        }
        weight = dp[full - 1];
        var pairs = new List<int[]>();
        int remaining = full - 1;
        while (remaining != 0)
        {
            int first = -1;
            for (int b = 0; b < m; b++)
            {
                if ((remaining & (1 << b)) != 0)
                {
                    first = b;
                    break;
                }
            }
            int other = choice[remaining];
            pairs.Add(new[] { odd[first], odd[other] });
            remaining ^= (1 << first) ^ (1 << other);
        }
        return pairs;
    }

    /** Brute force over every perfect matching of the odd set, for cross-checking. */
    internal double MinMatchingBrute(int[] odd)
    {
        int m = odd.Length;
        var used = new bool[m];
        return MatchFrom(odd, used, 0);
    }

    private double MatchFrom(int[] odd, bool[] used, double cost)
    {
        int first = -1;
        for (int i = 0; i < odd.Length; i++)
        {
            if (!used[i])
            {
                first = i;
                break;
            }
        }
        if (first < 0) return cost;
        double best = double.PositiveInfinity;
        used[first] = true;
        for (int j = first + 1; j < odd.Length; j++)
        {
            if (used[j]) continue;
            used[j] = true;
            double cand = MatchFrom(odd, used, cost + D[odd[first]][odd[j]]);
            if (cand < best) best = cand;
            used[j] = false;
        }
        used[first] = false;
        return best;
    }

    /** Euler circuit (vertex sequence, |E| + 1 entries) over the multigraph tree + matching. */
    internal int[] EulerCircuit(IReadOnlyList<int[]> tree, IReadOnlyList<int[]> matching)
    {
        var edgeA = new List<int>();
        var edgeB = new List<int>();
        foreach (int[] e in tree)
        {
            edgeA.Add(e[0]);
            edgeB.Add(e[1]);
        }
        foreach (int[] e in matching)
        {
            edgeA.Add(e[0]);
            edgeB.Add(e[1]);
        }

        var degree = new int[N];
        var incident = new List<int>[N];
        for (int v = 0; v < N; v++) incident[v] = new List<int>();
        for (int id = 0; id < edgeA.Count; id++)
        {
            incident[edgeA[id]].Add(id);
            incident[edgeB[id]].Add(id);
            degree[edgeA[id]]++;
            degree[edgeB[id]]++;
        }
        for (int v = 0; v < N; v++)
        {
            if (degree[v] % 2 != 0) throw new InvalidOperationException("multigraph has an odd degree at " + v);
        }

        var usedEdge = new bool[edgeA.Count];
        var next = new int[N];
        var stack = new List<int> { 0 };
        var circuit = new List<int>();
        while (stack.Count > 0)
        {
            int v = stack[stack.Count - 1];
            bool advanced = false;
            while (next[v] < incident[v].Count)
            {
                int id = incident[v][next[v]++];
                if (usedEdge[id]) continue;
                usedEdge[id] = true;
                stack.Add(edgeA[id] == v ? edgeB[id] : edgeA[id]);
                advanced = true;
                break;
            }
            if (!advanced)
            {
                circuit.Add(v);
                stack.RemoveAt(stack.Count - 1);
            }
        }
        circuit.Reverse();
        return circuit.ToArray();
    }

    /** Euler circuit over the multigraph, then shortcut to first visits. */
    internal int[] ShortcutTour(IReadOnlyList<int[]> tree, IReadOnlyList<int[]> matching)
    {
        int[] circuit = EulerCircuit(tree, matching);
        var seen = new bool[N];
        var tour = new List<int>();
        foreach (int v in circuit)
        {
            if (seen[v]) continue;
            seen[v] = true;
            tour.Add(v);
        }
        return tour.ToArray();
    }

    internal double TourLength(int[] tour)
    {
        double total = 0.0;
        for (int i = 0; i < tour.Length; i++) total += D[tour[i]][tour[(i + 1) % tour.Length]];
        return total;
    }

    internal bool AllDegreesEven(IReadOnlyList<int[]> tree, IReadOnlyList<int[]> matching)
    {
        var degree = new int[N];
        foreach (int[] e in tree)
        {
            degree[e[0]]++;
            degree[e[1]]++;
        }
        foreach (int[] e in matching)
        {
            degree[e[0]]++;
            degree[e[1]]++;
        }
        for (int v = 0; v < N; v++)
        {
            if (degree[v] % 2 != 0) return false;
        }
        return true;
    }

    /** Exact optimum (n <= 12) as a value plus a witness tour, by Held-Karp over subsets. */
    internal int[] HeldKarpTour(out double bestLength)
    {
        int n = N;
        if (n > 12) throw new InvalidOperationException("Held-Karp limited to 12 nodes");
        int full = 1 << (n - 1);
        var dp = new double[full][];
        for (int s = 0; s < full; s++)
        {
            dp[s] = new double[n];
            for (int v = 0; v < n; v++) dp[s][v] = double.PositiveInfinity;
        }
        for (int v = 1; v < n; v++) dp[1 << (v - 1)][v] = D[0][v];
        for (int s = 1; s < full; s++)
        {
            for (int v = 1; v < n; v++)
            {
                if (double.IsPositiveInfinity(dp[s][v])) continue;
                for (int w = 1; w < n; w++)
                {
                    if ((s & (1 << (w - 1))) != 0) continue;
                    int next = s | (1 << (w - 1));
                    double cand = dp[s][v] + D[v][w];
                    if (cand < dp[next][w]) dp[next][w] = cand;
                }
            }
        }

        int mask = full - 1;
        int current = -1;
        bestLength = double.PositiveInfinity;
        for (int v = 1; v < n; v++)
        {
            double cand = dp[full - 1][v] + D[v][0];
            if (cand < bestLength)
            {
                bestLength = cand;
                current = v;
            }
        }

        var tour = new int[n];
        tour[0] = 0;
        for (int position = n - 1; position >= 1; position--)
        {
            tour[position] = current;
            int prevMask = mask ^ (1 << (current - 1));
            if (prevMask == 0) break;
            int prev = -1;
            double bestPrev = double.PositiveInfinity;
            for (int v = 1; v < n; v++)
            {
                if ((prevMask & (1 << (v - 1))) == 0) continue;
                double cand = dp[prevMask][v] + D[v][current];
                if (cand < bestPrev - 1e-12)
                {
                    bestPrev = cand;
                    prev = v;
                }
            }
            mask = prevMask;
            current = prev;
        }
        return tour;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int state = 538001;

    private static int NextInt(int mod)
    {
        state = (state * 75 + 74) % 65537;
        return state % mod;
    }

    private static Christofides Instance(int n, int limit, int seed)
    {
        state = seed;
        var x = new double[n];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = NextInt(limit);
            y[i] = NextInt(limit);
        }
        return new Christofides(x, y);
    }

    private static string Render(int[] tour)
    {
        return string.Join(",", tour);
    }

    internal static void Main()
    {
        var ten = Instance(10, 1000, 538001);
        Check(ten.TriangleInequality(), "the 10-node instance satisfies the triangle inequality");
        Check(ten.PrimMst(out double primWeight).Count == 9, "the tree has n - 1 edges");
        Check(Math.Abs(primWeight - ten.KruskalWeight()) <= 1e-9, "Prim and Kruskal agree on the MST weight");

        int[] odd = Christofides.OddVertices(ten.PrimMst(out _), ten.N);
        Check(odd.Length % 2 == 0, "the odd-degree set has even size");
        Check(odd.Length >= 2, "a tree always has at least two odd-degree vertices");
        List<int[]> matching = ten.MinMatching(odd, out double matchWeight);
        Check(matching.Count == odd.Length / 2, "the matching is perfect");
        Check(Math.Abs(matchWeight - ten.MinMatchingBrute(odd)) <= 1e-9, "the DP matching is minimum");

        var tree = ten.PrimMst(out primWeight);
        var checkTree = tree;
        int[] tour = ten.ShortcutTour(checkTree, matching);
        double tourLength = ten.TourLength(tour);
        Check(tour.Length == 10, "the shortcut tour visits every vertex exactly once");
        var seen = new bool[10];
        foreach (int v in tour)
        {
            Check(!seen[v], "the shortcut tour is a permutation");
            seen[v] = true;
        }
        int[] circuit = ten.EulerCircuit(checkTree, matching);
        double circuitLength = ten.TourLength(circuit);
        Check(circuitLength >= tourLength - 1e-9, "shortcutting cannot lengthen the Euler circuit");
        Check(circuit.Length == checkTree.Count + matching.Count + 1, "Euler circuit has |E| + 1 vertices");
        Check(Math.Abs(circuitLength - (primWeight + matchWeight)) <= 1e-9, "Euler circuit length is tree + matching");

        int[] optimum = ten.HeldKarpTour(out double optimumLength);
        Check(Math.Abs(ten.TourLength(optimum) - optimumLength) <= 1e-9, "the Held-Karp witness matches its value");
        Check(ten.AllDegreesEven(checkTree, matching), "every multigraph degree is even");
        Check(optimumLength <= tourLength + 1e-9, "the Christofides tour is at least the true optimum");
        Check(tourLength <= 1.5 * optimumLength + 1e-9, "tour <= 3/2 * OPT");
        Check(matchWeight <= 0.5 * optimumLength + 1e-9, "matching <= OPT / 2 (the Christofides lemma)");
        Check(tourLength <= primWeight + matchWeight + 1e-9, "shortcut tour <= MST + matching");
        Check(tourLength >= optimumLength - 1e-9, "tour cannot beat the optimum");

        var repeat = Instance(10, 1000, 538001);
        var repeatTree = repeat.PrimMst(out _);
        int[] repeatOdd = Christofides.OddVertices(repeatTree, 10);
        var repeatMatching = repeat.MinMatching(repeatOdd, out _);
        Check(Render(repeat.ShortcutTour(repeatTree, repeatMatching)) == Render(tour),
                "the Christofides tour is reproducible");

        var six = Instance(6, 500, 538007);
        var sixTree = six.PrimMst(out double sixWeight);
        int[] sixOdd = Christofides.OddVertices(sixTree, 6);
        var sixMatching = six.MinMatching(sixOdd, out double sixMatchWeight);
        Check(Math.Abs(sixMatchWeight - six.MinMatchingBrute(sixOdd)) <= 1e-9, "6-node matching is minimum");
        int[] sixTour = six.ShortcutTour(sixTree, sixMatching);
        double sixTourLength = six.TourLength(sixTour);
        int[] sixOptimumTour = six.HeldKarpTour(out double sixOptimum);
        Check(Math.Abs(six.TourLength(sixOptimumTour) - sixOptimum) <= 1e-9, "6-node Held-Karp witness is consistent");
        Check(sixOptimum <= sixTourLength + 1e-9, "6-node tour is at least the optimum");
        Check(sixWeight <= sixOptimum + 1e-9, "6-node MST is a lower bound on OPT");
        Check(sixTourLength <= 1.5 * sixOptimum + 1e-9, "6-node tour <= 3/2 * OPT");
        Check(sixMatchWeight <= 0.5 * sixOptimum + 1e-9, "6-node matching <= OPT / 2");
        Check(six.AllDegreesEven(sixTree, sixMatching), "6-node multigraph degrees are even");
        Check(sixTree.Count == 5 && sixMatching.Count == sixOdd.Length / 2, "6-node tree and matching sizes");

        // Three points: Christofides returns the exact optimum, the perimeter.
        var three = new Christofides(new[] { 0.0, 3.0, 0.0 }, new[] { 0.0, 0.0, 4.0 });
        var threeTree = three.PrimMst(out double threeWeight);
        int[] threeOdd = Christofides.OddVertices(threeTree, 3);
        Check(threeOdd.Length == 2, "three points give a two-vertex odd set");
        var threeMatching = three.MinMatching(threeOdd, out _);
        Check(threeMatching.Count == 1, "the three-point matching has a single edge");
        int[] threeTour = three.ShortcutTour(threeTree, threeMatching);
        Check(Math.Abs(three.TourLength(threeTour) - 12.0) <= 1e-9, "the 3-4-5 triangle tour is the perimeter 12");
        three.HeldKarpTour(out double threeOptimum);
        Check(Math.Abs(threeOptimum - 12.0) <= 1e-9, "Held-Karp confirms 12");
        Check(three.TriangleInequality(), "the triangle instance is metric");
        Check(threeWeight > 0.0, "the three-point MST is non-degenerate");

        // A four-point square exercises an even-degree matching-free multigraph path.
        var square = new Christofides(new[] { 0.0, 1.0, 1.0, 0.0 }, new[] { 0.0, 0.0, 1.0, 1.0 });
        var squareTree = square.PrimMst(out double squareWeight);
        int[] squareOdd = Christofides.OddVertices(squareTree, 4);
        var squareMatching = square.MinMatching(squareOdd, out double squareMatch);
        int[] squareTour = square.ShortcutTour(squareTree, squareMatching);
        double squareLength = square.TourLength(squareTour);
        Check(squareTour.Length == 4, "the square tour visits all four points");
        square.HeldKarpTour(out double squareOptimum);
        Check(squareLength >= squareOptimum - 1e-9, "the square tour respects the optimum");
        Check(squareLength <= 1.5 * squareOptimum + 1e-9, "the square tour is within 3/2 * OPT");
        Check(squareMatch <= 0.5 * squareOptimum + 1e-9, "the square matching respects OPT / 2");
        Check(squareMatch >= 0.0, "the square matching weight is non-negative");
        Check(squareWeight > 0.0, "the square MST is non-degenerate");

        Console.WriteLine("PASS");
    }
}
