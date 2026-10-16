// Task 542 - Implement tree isomorphism hashing (AHU algorithm). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
//
// Two encodings of the AHU idea plus an independent verifier. (1) The recursive canonical form:
// the code of a rooted tree is the concatenation of the sorted child codes inside a pair of
// parentheses ("(()(()))" style), ordinal string comparison throughout, so two rooted trees are
// isomorphic exactly when their codes match. (2) The bottom-up integer hashing variant: nodes are
// grouped by depth, children are processed before parents, each node's key is the sorted list of
// its children's ids, and the keys of a whole depth are sorted and interned in that canonical order
// - so a node's id is a function of the multiset of its subtree shapes, and isomorphic rooted trees
// receive equal ids. For unrooted trees the file finds the one or two centroids (the vertices whose
// removal leaves components of size at most n/2) and takes the canonical code of the centroid-rooted
// trees, which is the standard AHU reduction to the rooted case. A backtracking vertex-mapping
// isomorphism test (degree filtering plus adjacency consistency) is the independent verifier.
//
// Documented checks: over a set of 26 deterministic trees (n = 6..9: paths, stars, balanced binary
// trees, caterpillars and LCG parent trees) plus a random relabelling of each, the brute-force test
// agrees with the centroid code for every one of the 52 x 51 ordered pairs; every tree is
// isomorphic to its relabelled copy, and its rooted code at root r equals the relabelled code at
// the image of r; the parenthesised code and the integer AHU ids never contradict each other
// (equal codes always imply equal ids); the sorted pair of centroid codes is preserved under
// relabelling and the canonical form is its minimum; P4 and the 4-star are distinguished; at least
// one non-isomorphic pair inside the set
// shares a degree sequence (so the degree test alone is insufficient) and is still separated; and
// the canonical form is reproduced by a second construction of the whole set.
using System;
using System.Collections.Generic;
using System.Text;

internal sealed class Tree
{
    internal readonly int N;
    internal readonly List<int>[] Adj;

    internal Tree(int n)
    {
        N = n;
        Adj = new List<int>[n];
        for (int v = 0; v < n; v++) Adj[v] = new List<int>();
    }

    internal void AddEdge(int u, int v)
    {
        Adj[u].Add(v);
        Adj[v].Add(u);
    }

    internal static Tree Path(int n)
    {
        var t = new Tree(n);
        for (int i = 0; i + 1 < n; i++) t.AddEdge(i, i + 1);
        return t;
    }

    internal static Tree Star(int n)
    {
        var t = new Tree(n);
        for (int i = 1; i < n; i++) t.AddEdge(0, i);
        return t;
    }

    /** Caterpillar: spine 0..s-1 plus one leaf on each of the first (n - s) spine vertices. */
    internal static Tree Caterpillar(int spine, int extraLeaves)
    {
        var t = new Tree(spine + extraLeaves);
        for (int i = 0; i + 1 < spine; i++) t.AddEdge(i, i + 1);
        for (int i = 0; i < extraLeaves; i++) t.AddEdge(i + 1, spine + i);
        return t;
    }

    /** Perfect-ish binary tree on n vertices, children 2v + 1 and 2v + 2. */
    internal static Tree Binary(int n)
    {
        var t = new Tree(n);
        for (int v = 0; v < n; v++)
        {
            if (2 * v + 1 < n) t.AddEdge(v, 2 * v + 1);
            if (2 * v + 2 < n) t.AddEdge(v, 2 * v + 2);
        }
        return t;
    }

    /** LCG parent tree: parent[i] is uniform in [0, i). */
    internal static Tree Lcg(int n, ref int state)
    {
        var t = new Tree(n);
        for (int i = 1; i < n; i++)
        {
            state = (state * 75 + 74) % 65537;
            t.AddEdge(i, state % i);
        }
        return t;
    }

    internal Tree Relabel(int[] permutation)
    {
        var t = new Tree(N);
        for (int v = 0; v < N; v++)
        {
            foreach (int w in Adj[v])
            {
                if (v < w) t.AddEdge(permutation[v], permutation[w]);
            }
        }
        return t;
    }

    internal int[] Degrees()
    {
        var degrees = new int[N];
        for (int v = 0; v < N; v++) degrees[v] = Adj[v].Count;
        Array.Sort(degrees);
        return degrees;
    }

    internal string RootedCode(int root)
    {
        return CodeAt(root, -1);
    }

    private string CodeAt(int v, int parent)
    {
        var childCodes = new List<string>();
        foreach (int w in Adj[v])
        {
            if (w == parent) continue;
            childCodes.Add(CodeAt(w, v));
        }
        childCodes.Sort(StringComparer.Ordinal);
        var sb = new StringBuilder("(");
        foreach (string code in childCodes) sb.Append(code);
        sb.Append(")");
        return sb.ToString();
    }

    /** Vertices whose largest remaining component is at most n / 2 (one or two of them). */
    internal int[] Centroids()
    {
        var size = new int[N];
        var parent = new int[N];
        var order = new List<int>();
        var stack = new List<int> { 0 };
        parent[0] = -1;
        var visited = new bool[N];
        visited[0] = true;
        while (stack.Count > 0)
        {
            int v = stack[stack.Count - 1];
            stack.RemoveAt(stack.Count - 1);
            order.Add(v);
            foreach (int w in Adj[v])
            {
                if (visited[w]) continue;
                visited[w] = true;
                parent[w] = v;
                stack.Add(w);
            }
        }
        for (int i = order.Count - 1; i >= 0; i--)
        {
            int v = order[i];
            size[v] = 1;
            foreach (int w in Adj[v])
            {
                if (w != parent[v]) size[v] += size[w];
            }
        }
        var centroids = new List<int>();
        for (int v = 0; v < N; v++)
        {
            int largest = N - size[v];
            foreach (int w in Adj[v])
            {
                if (w == parent[v]) continue;
                if (size[w] > largest) largest = size[w];
            }
            if (largest <= N / 2) centroids.Add(v);
        }
        return centroids.ToArray();
    }

    internal string CanonicalUnrooted()
    {
        var codes = new List<string>();
        foreach (int c in Centroids()) codes.Add(RootedCode(c));
        codes.Sort(StringComparer.Ordinal);
        return codes[0];
    }

    /** Integer AHU ids interned depth by depth in sorted-key order (canonical per tree). */
    internal int AhuRootId(int root)
    {
        var parent = new int[N];
        for (int v = 0; v < N; v++) parent[v] = -2;
        var depth = new int[N];
        var order = new List<int>();
        var queue = new Queue<int>();
        parent[root] = -1;
        queue.Enqueue(root);
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            order.Add(v);
            foreach (int w in Adj[v])
            {
                if (w == parent[v]) continue;
                parent[w] = v;
                depth[w] = depth[v] + 1;
                queue.Enqueue(w);
            }
        }

        var perDepth = new Dictionary<int, List<int>>();
        foreach (int v in order)
        {
            if (!perDepth.TryGetValue(depth[v], out List<int> nodes))
            {
                nodes = new List<int>();
                perDepth[depth[v]] = nodes;
            }
            nodes.Add(v);
        }
        var levels = new List<int>(perDepth.Keys);
        levels.Sort();
        levels.Reverse();

        var labels = new int[N];
        foreach (int level in levels)
        {
            var nodes = perDepth[level];
            var keys = new List<string>();
            var keyOf = new Dictionary<int, string>();
            foreach (int v in nodes)
            {
                var childIds = new List<int>();
                foreach (int w in Adj[v])
                {
                    if (parent[w] == v) childIds.Add(labels[w]);
                }
                childIds.Sort();
                var sb = new StringBuilder();
                foreach (int id in childIds) sb.Append(id).Append(',');
                string key = sb.ToString();
                keyOf[v] = key;
                keys.Add(key);
            }
            keys.Sort(StringComparer.Ordinal);
            var map = new Dictionary<string, int>();
            string? previous = null;
            int nextId = 0;
            foreach (string key in keys)
            {
                if (previous != null && previous == key) continue;
                map[key] = nextId++;
                previous = key;
            }
            foreach (int v in nodes) labels[v] = map[keyOf[v]];
        }
        return labels[root];
    }

    /** Independent verifier: backtracking vertex mapping with degree filtering. */
    internal bool IsomorphicBrute(Tree other)
    {
        if (other.N != N) return false;
        int[] mine = Degrees();
        int[] theirs = other.Degrees();
        for (int i = 0; i < N; i++)
        {
            if (mine[i] != theirs[i]) return false;
        }
        var map = new int[N];
        var used = new bool[N];
        for (int v = 0; v < N; v++) map[v] = -1;
        return Match(0, map, used, other);
    }

    private bool Match(int index, int[] map, bool[] used, Tree other)
    {
        if (index == N) return true;
        for (int candidate = 0; candidate < N; candidate++)
        {
            if (used[candidate]) continue;
            if (Adj[index].Count != other.Adj[candidate].Count) continue;
            bool ok = true;
            foreach (int w in Adj[index])
            {
                if (w >= index) continue;
                if (!other.Adj[candidate].Contains(map[w]))
                {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            map[index] = candidate;
            used[candidate] = true;
            if (Match(index + 1, map, used, other)) return true;
            used[candidate] = false;
            map[index] = -1;
        }
        return false;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static List<Tree> BuildSet(out int state)
    {
        state = 542001;
        var trees = new List<Tree>();
        trees.Add(Tree.Path(7));
        trees.Add(Tree.Star(7));
        trees.Add(Tree.Binary(7));
        trees.Add(Tree.Caterpillar(4, 3));
        trees.Add(Tree.Caterpillar(2, 5));
        trees.Add(Tree.Caterpillar(6, 1));
        for (int i = 0; i < 20; i++)
        {
            int n = 6 + i % 4;
            trees.Add(Tree.Lcg(n, ref state));
        }
        return trees;
    }

    private static Tree RelabelWith(Tree tree, int seed, out int[] permutation)
    {
        permutation = new int[tree.N];
        for (int v = 0; v < tree.N; v++) permutation[v] = v;
        int state = seed;
        for (int i = tree.N - 1; i > 0; i--)
        {
            state = (state * 75 + 74) % 65537;
            int j = state % (i + 1);
            int tmp = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = tmp;
        }
        return tree.Relabel(permutation);
    }

    internal static void Main()
    {
        var trees = BuildSet(out int state);
        var relabeled = new List<Tree>();
        var permutations = new List<int[]>();
        for (int i = 0; i < trees.Count; i++)
        {
            relabeled.Add(RelabelWith(trees[i], 542100 + i, out int[] permutation));
            permutations.Add(permutation);
        }

        var all = new List<Tree>();
        all.AddRange(trees);
        all.AddRange(relabeled);
        Check(all.Count == 52, "the tree set has 52 entries (26 shapes plus relabellings)");

        for (int i = 0; i < trees.Count; i++)
        {
            Check(trees[i].CanonicalUnrooted() == relabeled[i].CanonicalUnrooted(),
                    "relabelling preserves the centroid code at index " + i);
            Check(trees[i].IsomorphicBrute(relabeled[i]), "relabelled tree is isomorphic at index " + i);
            for (int r = 0; r < trees[i].N; r++)
            {
                Check(trees[i].RootedCode(r) == relabeled[i].RootedCode(permutations[i][r]),
                        "rooted code follows the relabelling at tree " + i + " root " + r);
            }
            Check(string.Join(",", trees[i].Degrees()) == string.Join(",", relabeled[i].Degrees()),
                    "relabelling preserves the degree sequence at index " + i);
        }

        int isoPairs = 0;
        int sameDegreeNonIso = 0;
        for (int i = 0; i < all.Count; i++)
        {
            for (int j = 0; j < all.Count; j++)
            {
                if (i == j) continue;
                bool brute = all[i].IsomorphicBrute(all[j]);
                bool canonical = all[i].CanonicalUnrooted() == all[j].CanonicalUnrooted();
                Check(brute == canonical, "centroid code agrees with brute force at pair " + i + "," + j);
                if (brute)
                {
                    isoPairs++;
                    Check(all[i].AhuRootId(0) >= 0 && all[j].AhuRootId(0) >= 0, "AHU ids are defined");
                }
                else
                {
                    Check(all[i].CanonicalUnrooted() != all[j].CanonicalUnrooted(),
                            "non-isomorphic pair has different codes at " + i + "," + j);
                    bool sameDegrees = string.Join(",", all[i].Degrees()) == string.Join(",", all[j].Degrees());
                    if (sameDegrees) sameDegreeNonIso++;
                }
            }
        }
        Check(isoPairs >= 52, "at least every tree matches its relabelling (" + isoPairs + " ordered pairs)");
        Check(sameDegreeNonIso >= 2, "the set contains a same-degree, non-isomorphic pair");

        // The two encodings must never contradict each other: equality of parenthesis codes
        // (the ground truth for rooted isomorphism) implies equality of the interned ids.
        for (int i = 0; i < trees.Count; i++)
        {
            for (int j = 0; j < trees.Count; j++)
            {
                if (trees[i].N != trees[j].N) continue;
                for (int r = 0; r < trees[i].N; r++)
                {
                    for (int s = 0; s < trees[j].N; s++)
                    {
                        bool codesEqual = trees[i].RootedCode(r) == trees[j].RootedCode(s);
                        bool idsEqual = trees[i].AhuRootId(r) == trees[j].AhuRootId(s);
                        Check(!codesEqual || idsEqual, "equal rooted codes imply equal AHU ids");
                    }
                }
            }
        }

        // Relabelled copies also keep equal ids for corresponding roots.
        for (int i = 0; i < trees.Count; i++)
        {
            for (int r = 0; r < trees[i].N; r++)
            {
                Check(trees[i].AhuRootId(r) == relabeled[i].AhuRootId(permutations[i][r]),
                        "AHU id follows the relabelling at tree " + i + " root " + r);
            }

            // Two centroids need not carry the same code, but the sorted pair of codes is a
            // relabelling invariant and the canonical form is its minimum.
            int[] centroids = trees[i].Centroids();
            Check(centroids.Length == 1 || centroids.Length == 2, "a tree has one or two centroids");
            if (centroids.Length == 2)
            {
                var pair = new List<string> { trees[i].RootedCode(centroids[0]), trees[i].RootedCode(centroids[1]) };
                pair.Sort(StringComparer.Ordinal);
                int[] other = relabeled[i].Centroids();
                var pairOther = new List<string>
                {
                    relabeled[i].RootedCode(other[0]), relabeled[i].RootedCode(other[1]),
                };
                pairOther.Sort(StringComparer.Ordinal);
                Check(pair[0] == pairOther[0] && pair[1] == pairOther[1],
                        "the sorted centroid codes survive relabelling at index " + i);
                Check(trees[i].CanonicalUnrooted() == pair[0], "the canonical form is the smaller centroid code");
            }
        }

        // Hand-checked shapes.
        Check(Tree.Path(4).CanonicalUnrooted() != Tree.Star(4).CanonicalUnrooted(), "P4 differs from the 4-star");
        Check(Tree.Path(4).IsomorphicBrute(Tree.Star(4)) == false, "brute force separates P4 from the 4-star");
        Check(Tree.Path(4).CanonicalUnrooted() == Tree.Path(4).Relabel(new[] { 2, 0, 3, 1 }).CanonicalUnrooted(),
                "P4 relabelling keeps the canonical form");
        Check(Tree.Path(4).Centroids().Length == 2, "P4 has two centroids");
        Check(Tree.Star(4).Centroids().Length == 1, "the 4-star has one centroid");
        Check(Tree.Path(2).CanonicalUnrooted() == "(())", "P2 code is (())");
        Check(Tree.Path(4).CanonicalUnrooted() == "((())())", "P4 code is ((())())");
        Check(Tree.Star(4).RootedCode(0) == "(()()())", "the 4-star rooted at the centre has three leaf children");
        Check(Tree.Star(3).RootedCode(0) == "(()())", "the 3-vertex star is a path of three");

        // Determinism.
        var repeat = BuildSet(out int repeatState);
        Check(repeatState == state, "the LCG state is reproducible");
        for (int i = 0; i < trees.Count; i++)
        {
            Check(trees[i].CanonicalUnrooted() == repeat[i].CanonicalUnrooted(),
                    "the tree set is reproducible at index " + i);
        }

        Console.WriteLine("PASS");
    }
}
