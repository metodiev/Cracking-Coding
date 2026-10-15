/**
 * Task 343 - Graph isomorphism: simple checks with degree-sequence invariants. (Tier: Intermediate, CTCI Ch 4)
 *
 * Necessary conditions, deliberately not sufficient: equal sorted degree sequences, equal
 * triangle counts and equal connected-component size multisets.  C6 vs C3+C3 fails on
 * components, K3,3 vs the triangular prism fails on triangles, and the documented
 * counterexample (cube Q3 vs Moebius ladder V8) passes all three checks yet the graphs are
 * not isomorphic - Q3 is bipartite and V8 is not.
 */
import java.util.*;

public class Solution {
    static int[] degrees(int[][] adj) {
        int[] out = new int[adj.length];
        for (int i = 0; i < adj.length; i++) {
            Set<Integer> unique = new HashSet<>();
            for (int v : adj[i]) unique.add(v);
            out[i] = unique.size();
        }
        Arrays.sort(out);
        return out;
    }

    static int triangleCount(int[][] adj) {
        List<Set<Integer>> sets = new ArrayList<>();
        for (int[] neighbours : adj) {
            Set<Integer> set = new HashSet<>();
            for (int v : neighbours) set.add(v);
            sets.add(set);
        }
        int count = 0;
        for (int u = 0; u < adj.length; u++) {
            for (int v : sets.get(u)) {
                if (v <= u) continue;
                for (int w : sets.get(u)) {
                    if (w > v && sets.get(v).contains(w)) count++;
                }
            }
        }
        return count;
    }

    static int[] componentSizes(int[][] adj) {
        int n = adj.length;
        boolean[] seen = new boolean[n];
        List<Integer> sizes = new ArrayList<>();
        for (int start = 0; start < n; start++) {
            if (seen[start]) continue;
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(start);
            seen[start] = true;
            int size = 0;
            while (!stack.isEmpty()) {
                int u = stack.pop();
                size++;
                for (int v : adj[u]) {
                    if (!seen[v]) {
                        seen[v] = true;
                        stack.push(v);
                    }
                }
            }
            sizes.add(size);
        }
        int[] out = new int[sizes.size()];
        for (int i = 0; i < out.length; i++) out[i] = sizes.get(i);
        Arrays.sort(out);
        return out;
    }

    /** Independent oracle for the component multiset. */
    static int[] componentSizesUnionFind(int[][] adj) {
        int n = adj.length;
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        for (int u = 0; u < n; u++) {
            for (int v : adj[u]) {
                int rootU = find(parent, u);
                int rootV = find(parent, v);
                if (rootU != rootV) parent[rootU] = rootV;
            }
        }
        Map<Integer, Integer> counts = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int root = find(parent, i);
            counts.merge(root, 1, Integer::sum);
        }
        int[] out = new int[counts.size()];
        int k = 0;
        for (int size : counts.values()) out[k++] = size;
        Arrays.sort(out);
        return out;
    }

    static int find(int[] parent, int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    static boolean checkIsomorphic(int[][] a, int[][] b) {
        if (a.length != b.length) return false;
        if (!Arrays.equals(degrees(a), degrees(b))) return false;
        if (triangleCount(a) != triangleCount(b)) return false;
        if (!Arrays.equals(componentSizes(a), componentSizes(b))) return false;
        return true;
    }

    static int diameter(int[][] adj) {
        int n = adj.length;
        int best = 0;
        for (int start = 0; start < n; start++) {
            int[] dist = new int[n];
            Arrays.fill(dist, -1);
            dist[start] = 0;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(start);
            while (!queue.isEmpty()) {
                int u = queue.poll();
                for (int v : adj[u]) {
                    if (dist[v] < 0) {
                        dist[v] = dist[u] + 1;
                        queue.add(v);
                    }
                }
            }
            for (int d : dist) {
                if (d < 0) return -1;
                best = Math.max(best, d);
            }
        }
        return best;
    }

    /** perm maps old vertex -> new vertex. */
    static int[][] relabel(int[][] adj, int[] perm) {
        int n = adj.length;
        int[][] out = new int[n][];
        for (int u = 0; u < n; u++) {
            Set<Integer> mapped = new TreeSet<>();
            for (int v : adj[u]) mapped.add(perm[v]);
            int[] row = new int[mapped.size()];
            int i = 0;
            for (int value : mapped) row[i++] = value;
            out[perm[u]] = row;
        }
        return out;
    }

    static boolean sameAdjacency(int[][] a, int[][] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (!Arrays.equals(a[i], b[i])) return false;
        }
        return true;
    }

    static int[] inverse(int[] perm) {
        int[] out = new int[perm.length];
        for (int old = 0; old < perm.length; old++) out[perm[old]] = old;
        return out;
    }

    static boolean isBipartite(int[][] adj) {
        int[] color = new int[adj.length];
        Arrays.fill(color, -1);
        for (int start = 0; start < adj.length; start++) {
            if (color[start] >= 0) continue;
            color[start] = 0;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(start);
            while (!queue.isEmpty()) {
                int u = queue.poll();
                for (int v : adj[u]) {
                    if (color[v] < 0) {
                        color[v] = 1 - color[u];
                        queue.add(v);
                    } else if (color[v] == color[u]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    static int[][] normalize(int[][] adj) {
        int[][] out = new int[adj.length][];
        for (int i = 0; i < adj.length; i++) {
            Set<Integer> unique = new TreeSet<>();
            for (int v : adj[i]) unique.add(v);
            out[i] = new int[unique.size()];
            int k = 0;
            for (int value : unique) out[i][k++] = value;
        }
        return out;
    }

    static int[][] cycleGraph(int n) {
        int[][] adj = new int[n][2];
        for (int i = 0; i < n; i++) {
            adj[i][0] = (i - 1 + n) % n;
            adj[i][1] = (i + 1) % n;
            Arrays.sort(adj[i]);
        }
        return adj;
    }

    static int[][] twoTriangles() {
        return normalize(new int[][]{{1, 2}, {0, 2}, {0, 1}, {4, 5}, {3, 5}, {3, 4}});
    }

    static int[][] k33() {
        return normalize(new int[][]{{3, 4, 5}, {3, 4, 5}, {3, 4, 5}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}});
    }

    static int[][] prism() {
        return normalize(new int[][]{{1, 2, 3}, {0, 2, 4}, {0, 1, 5}, {4, 5, 0}, {3, 5, 1}, {3, 4, 2}});
    }

    static int[][] cubeQ3() {
        int[][] adj = new int[8][3];
        for (int i = 0; i < 8; i++) {
            for (int bit = 0; bit < 3; bit++) adj[i][bit] = i ^ (1 << bit);
            Arrays.sort(adj[i]);
        }
        return adj;
    }

    static int[][] mobiusV8() {
        List<Set<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < 8; i++) adj.add(new TreeSet<>());
        for (int i = 0; i < 8; i++) {
            adj.get(i).add((i + 1) % 8);
            adj.get((i + 1) % 8).add(i);
        }
        for (int i = 0; i < 4; i++) {
            adj.get(i).add(i + 4);
            adj.get(i + 4).add(i);
        }
        return toArray(adj);
    }

    static int[][] toArray(List<Set<Integer>> rows) {
        int[][] out = new int[rows.size()][];
        for (int i = 0; i < rows.size(); i++) {
            Set<Integer> row = rows.get(i);
            out[i] = new int[row.size()];
            int k = 0;
            for (int value : row) out[i][k++] = value;
        }
        return out;
    }

    private static int seed = 20261016;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    static int[][] lcgGraph(int n) {
        List<Set<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new TreeSet<>());
        for (int u = 0; u < n; u++) {
            for (int v = u + 1; v < n; v++) {
                if (nextSeed() % 2 == 0) {
                    adj.get(u).add(v);
                    adj.get(v).add(u);
                }
            }
        }
        return toArray(adj);
    }

    static int[] lcgPermutation(int n) {
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = nextSeed() % (i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        return perm;
    }

    public static void main(String[] args) {
        int[][] c5 = cycleGraph(5);
        int[][] c6 = cycleGraph(6);
        int[][] path4 = normalize(new int[][]{{1}, {0, 2}, {1, 3}, {2}});
        int[][] star4 = normalize(new int[][]{{1, 2, 3, 4}, {0}, {0}, {0}, {0}});
        int[][] bowtie = normalize(new int[][]{{1, 2, 3, 4}, {0, 2}, {0, 1}, {0, 4}, {0, 3}});

        check(checkIsomorphic(c5, c5), "C5 with itself");
        int[][] pentagram = relabel(c5, new int[]{0, 2, 4, 1, 3});
        check(Arrays.deepEquals(pentagram, new int[][]{{2, 3}, {3, 4}, {0, 4}, {0, 1}, {1, 2}}), "pentagram adjacency");
        check(!Arrays.deepEquals(pentagram, c5), "pentagram is a different labelling");
        check(checkIsomorphic(c5, pentagram), "C5 vs pentagram");
        check(checkIsomorphic(path4, relabel(path4, new int[]{2, 0, 3, 1})), "P4 relabel");
        check(checkIsomorphic(star4, relabel(star4, new int[]{4, 0, 1, 2, 3})), "star relabel");
        check(checkIsomorphic(bowtie, relabel(bowtie, new int[]{3, 1, 4, 0, 2})), "bowtie relabel");
        check(checkIsomorphic(new int[0][], new int[0][]), "empty graphs");
        check(checkIsomorphic(new int[][]{{}}, new int[][]{{}}), "single vertex");

        int[][] star3 = normalize(new int[][]{{1, 2, 3}, {0}, {0}, {0}});
        check(!checkIsomorphic(star3, path4), "star K1,3 vs P4");
        check(!checkIsomorphic(bowtie, c5), "bowtie vs C5");

        int[][] twoTriangles = twoTriangles();
        check(Arrays.equals(degrees(c6), degrees(twoTriangles)), "C6 vs 2 triangles same degrees");
        check(Arrays.equals(degrees(c6), new int[]{2, 2, 2, 2, 2, 2}), "degrees [2]*6");
        check(triangleCount(c6) == 0 && triangleCount(twoTriangles) == 2, "triangles 0 vs 2");
        check(Arrays.equals(componentSizes(c6), new int[]{6}), "comps [6]");
        check(Arrays.equals(componentSizes(twoTriangles), new int[]{3, 3}), "comps [3,3]");
        check(!checkIsomorphic(c6, twoTriangles), "C6 not isomorphic to 2 triangles");

        int[][] k33 = k33();
        int[][] prism = prism();
        check(Arrays.equals(degrees(k33), degrees(prism)), "K33 vs prism same degrees");
        check(Arrays.equals(degrees(k33), new int[]{3, 3, 3, 3, 3, 3}), "degrees [3]*6");
        check(Arrays.equals(componentSizes(k33), componentSizes(prism)), "K33 vs prism same components");
        check(triangleCount(k33) == 0 && triangleCount(prism) == 2, "K33 vs prism triangles");
        check(!checkIsomorphic(k33, prism), "K33 not isomorphic to prism");

        int[][] cube = cubeQ3();
        int[][] v8 = mobiusV8();
        check(checkIsomorphic(cube, v8), "cube vs V8 passes all three checks");
        check(Arrays.equals(degrees(cube), degrees(v8)), "cube vs V8 same degrees");
        check(triangleCount(cube) == 0 && triangleCount(v8) == 0, "cube vs V8 triangles");
        check(Arrays.equals(componentSizes(cube), componentSizes(v8)), "cube vs V8 same components");
        check(isBipartite(cube) && !isBipartite(v8), "bipartiteness separates the counterexample");
        check(diameter(cube) == 3 && diameter(v8) == 2, "diameters 3 vs 2");

        for (int trial = 0; trial < 40; trial++) {
            int[][] graph = lcgGraph(7);
            int[] perm = lcgPermutation(7);
            int[][] relabelled = relabel(graph, perm);
            check(checkIsomorphic(graph, relabelled), "random relabel " + trial);
            check(Arrays.equals(degrees(graph), degrees(relabelled)), "random degrees " + trial);
            check(triangleCount(graph) == triangleCount(relabelled), "random triangles " + trial);
            check(Arrays.equals(componentSizes(graph), componentSizesUnionFind(graph)), "union-find oracle " + trial);
            check(Arrays.equals(componentSizes(relabelled), componentSizesUnionFind(relabelled)), "union-find relabelled " + trial);
            check(sameAdjacency(relabel(relabelled, inverse(perm)), graph), "double relabel " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
