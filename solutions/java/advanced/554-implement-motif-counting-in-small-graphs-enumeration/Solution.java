/** Task 554 - Implement motif counting in small graphs (enumeration). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * Motif counting by exhaustive enumeration of vertex subsets: for k = 2..5 every C(n,k) subset of the
 * host is extracted, its induced adjacency is reduced to a canonical form — the minimum over all k!
 * relabelings of the k(k-1)/2-bit upper-triangle mask — and the canonical key is tallied, so the
 * histogram is the exact induced-subgraph motif distribution and its counts sum to C(n,k).
 *
 * The self-test also settles how far the cheap signature (edge count plus sorted degree sequence) can
 * replace that canonical form. Enumerating *all* labeled graphs on k vertices shows the signature is
 * complete for k <= 4 (every one of the 4 and 11 isomorphism classes gets its own signature) but not
 * from k = 5 on: exactly 3 of the 31 signatures of 5-vertex graphs merge two isomorphism classes each,
 * so the signature sees 31 buckets where there are 34 motifs. The witness used here is the simplest of
 * the three — the path P5 and the disjoint union of a triangle with an edge K3 + K2 both have 4 edges
 * and degree sequence (1,1,2,2,2), yet P5 is connected and K3 + K2 is not, so their canonical keys must
 * differ; the other two collisions are 5 edges with degrees (1,2,2,2,3) (a 4-cycle with a pendant edge
 * versus a triangle with a 2-vertex path attached) and 6 edges with degrees (2,2,2,3,3). This is why the
 * implementation pays for the permutation scan instead of trusting the signature.
 *
 * Other documented checks, all independent of the enumeration itself: the edge count comes from the
 * degree sum, the triangle count from trace(A^3)/6 computed with integer matrix multiplication and also
 * from sum over edges of |N(u) n N(v)| / 3, and the 4-clique count from sum over edges of the number of
 * adjacent pairs inside N(u) n N(v) divided by 6 — all four agree with the enumerated motifs. On K5 the
 * only motifs are the complete graphs (all C(5,k) subsets for every k), on C5 there is no triangle, the
 * triples split 5/5 between the 3-vertex path and the single-edge motif, all five 4-subsets are P4 and
 * the sole 5-subset is the cycle itself; on the empty graph only the empty motif occurs; the 9-vertex
 * seeded test graph realises all 4 unlabeled graphs on 3 vertices and all 11 on 4 vertices; histograms
 * are invariant under a seeded relabeling; and a second run reproduces every count.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) {
            this.state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        double nextDouble() {
            return (next() >>> 11) * 0x1.0p-53;
        }

        int nextInt(int bound) {
            return (int) ((next() >>> 1) % bound);
        }
    }

    static boolean[][] graph(int n, int[][] edges) {
        boolean[][] adj = new boolean[n][n];
        for (int[] e : edges) {
            adj[e[0]][e[1]] = true;
            adj[e[1]][e[0]] = true;
        }
        return adj;
    }

    static boolean[][] complete(int n) {
        boolean[][] adj = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
        return adj;
    }

    static List<int[]> permutations(int k) {
        List<int[]> out = new ArrayList<int[]>();
        buildPermutations(k, 0, new int[k], new boolean[k], out);
        return out;
    }

    private static void buildPermutations(int k, int depth, int[] current, boolean[] used,
                                          List<int[]> out) {
        if (depth == k) {
            out.add(current.clone());
            return;
        }
        for (int i = 0; i < k; i++) {
            if (used[i]) continue;
            used[i] = true;
            current[depth] = i;
            buildPermutations(k, depth + 1, current, used, out);
            used[i] = false;
        }
    }

    /** Minimum upper-triangle bitmask over all relabelings: a complete isomorphism invariant. */
    static int canonicalKey(boolean[][] host, int[] subset, List<int[]> perms) {
        int k = subset.length;
        boolean[][] induced = new boolean[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                induced[i][j] = host[subset[i]][subset[j]];
                induced[j][i] = induced[i][j];
            }
        }
        int best = Integer.MAX_VALUE;
        for (int[] perm : perms) {
            int mask = 0;
            int bit = 0;
            for (int i = 0; i < k; i++) {
                for (int j = i + 1; j < k; j++) {
                    if (induced[perm[i]][perm[j]]) mask |= 1 << bit;
                    bit++;
                }
            }
            best = Math.min(best, mask);
        }
        return best;
    }

    static int inducedEdges(boolean[][] host, int[] subset) {
        int edges = 0;
        for (int i = 0; i < subset.length; i++) {
            for (int j = i + 1; j < subset.length; j++) {
                if (host[subset[i]][subset[j]]) edges++;
            }
        }
        return edges;
    }

    /** Signature (edge count, sorted degree sequence) — provably incomplete from k = 5 on. */
    static String signature(boolean[][] host, int[] subset) {
        int k = subset.length;
        int[] degree = new int[k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (i != j && host[subset[i]][subset[j]]) degree[i]++;
            }
        }
        int[] sorted = degree.clone();
        Arrays.sort(sorted);
        StringBuilder sb = new StringBuilder();
        sb.append(inducedEdges(host, subset)).append(':');
        for (int d : sorted) sb.append(d).append(',');
        return sb.toString();
    }

    static TreeMap<Integer, Long> motifHistogram(boolean[][] host, int k, List<int[]> perms) {
        TreeMap<Integer, Long> histogram = new TreeMap<Integer, Long>();
        enumerate(host, k, 0, 0, new int[k], perms, histogram, null);
        return histogram;
    }

    private static void enumerate(boolean[][] host, int k, int start, int depth, int[] subset,
                                  List<int[]> perms, TreeMap<Integer, Long> histogram, long[] visited) {
        int n = host.length;
        if (depth == k) {
            if (visited != null) visited[0]++;
            int key = canonicalKey(host, subset, perms);
            Long old = histogram.get(key);
            histogram.put(key, old == null ? 1L : old + 1L);
            return;
        }
        for (int v = start; v <= n - (k - depth); v++) {
            subset[depth] = v;
            enumerate(host, k, v + 1, depth + 1, subset, perms, histogram, visited);
        }
    }

    static long binomial(int n, int k) {
        long result = 1;
        for (int i = 0; i < k; i++) result = result * (n - i) / (i + 1);
        return result;
    }

    static long histogramTotal(Map<Integer, Long> histogram) {
        long total = 0;
        for (long value : histogram.values()) total += value;
        return total;
    }

    static long[][] multiply(long[][] a, long[][] b) {
        int n = a.length;
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                if (a[i][k] == 0) continue;
                for (int j = 0; j < n; j++) out[i][j] += a[i][k] * b[k][j];
            }
        }
        return out;
    }

    /** triangles = trace(A^3) / 6. */
    static long trianglesFromMatrix(boolean[][] adj) {
        int n = adj.length;
        long[][] a = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) if (adj[i][j]) a[i][j] = 1;
        }
        long[][] cube = multiply(multiply(a, a), a);
        long trace = 0;
        for (int i = 0; i < n; i++) trace += cube[i][i];
        return trace / 6;
    }

    /** Every triangle is seen by each of its three edges: sum |N(u) n N(v)| / 3. */
    static long trianglesFromCommonNeighbours(boolean[][] adj) {
        int n = adj.length;
        long total = 0;
        for (int u = 0; u < n; u++) {
            for (int v = u + 1; v < n; v++) {
                if (!adj[u][v]) continue;
                for (int w = 0; w < n; w++) if (adj[u][w] && adj[v][w]) total++;
            }
        }
        return total / 3;
    }

    /** Every K4 is seen by each of its six edges: adjacent pairs inside N(u) n N(v), divided by 6. */
    static long fourCliquesFromCommonNeighbours(boolean[][] adj) {
        int n = adj.length;
        long total = 0;
        for (int u = 0; u < n; u++) {
            for (int v = u + 1; v < n; v++) {
                if (!adj[u][v]) continue;
                List<Integer> common = new ArrayList<Integer>();
                for (int w = 0; w < n; w++) if (adj[u][w] && adj[v][w]) common.add(w);
                for (int i = 0; i < common.size(); i++) {
                    for (int j = i + 1; j < common.size(); j++) {
                        if (adj[common.get(i)][common.get(j)]) total++;
                    }
                }
            }
        }
        return total / 6;
    }

    static long edgeCount(boolean[][] adj) {
        long edges = 0;
        for (int i = 0; i < adj.length; i++) {
            for (int j = 0; j < adj.length; j++) if (adj[i][j]) edges++;
        }
        return edges / 2;
    }

    static int[] degrees(boolean[][] adj) {
        int n = adj.length;
        int[] d = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) if (adj[i][j]) d[i]++;
        }
        return d;
    }

    static boolean connected(boolean[][] adj, int[] subset) {
        boolean[] seen = new boolean[subset.length];
        int[] stack = new int[subset.length];
        int top = 0;
        stack[top++] = 0;
        seen[0] = true;
        int reach = 1;
        while (top > 0) {
            int i = stack[--top];
            for (int j = 0; j < subset.length; j++) {
                if (!seen[j] && adj[subset[i]][subset[j]]) {
                    seen[j] = true;
                    stack[top++] = j;
                    reach++;
                }
            }
        }
        return reach == subset.length;
    }

    static boolean[][] relabel(boolean[][] adj, Rng rng) {
        int n = adj.length;
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        boolean[][] out = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) out[i][j] = adj[perm[i]][perm[j]];
        }
        return out;
    }

    static boolean[][] fromMask(int k, int mask) {
        boolean[][] adj = new boolean[k][k];
        int bit = 0;
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                if (((mask >> bit) & 1) != 0) {
                    adj[i][j] = true;
                    adj[j][i] = true;
                }
                bit++;
            }
        }
        return adj;
    }

    /** Signature -> distinct canonical keys, over every labeled graph on k vertices. */
    static TreeMap<String, TreeSet<Integer>> signatureAudit(int k) {
        List<int[]> perms = permutations(k);
        int[] subset = new int[k];
        for (int i = 0; i < k; i++) subset[i] = i;
        TreeMap<String, TreeSet<Integer>> bySignature = new TreeMap<String, TreeSet<Integer>>();
        for (int mask = 0; mask < (1 << (k * (k - 1) / 2)); mask++) {
            boolean[][] adj = fromMask(k, mask);
            String sig = signature(adj, subset);
            TreeSet<Integer> keys = bySignature.get(sig);
            if (keys == null) {
                keys = new TreeSet<Integer>();
                bySignature.put(sig, keys);
            }
            keys.add(canonicalKey(adj, subset, perms));
        }
        return bySignature;
    }

    static int ambiguousSignatures(TreeMap<String, TreeSet<Integer>> audit) {
        int ambiguous = 0;
        for (TreeSet<Integer> keys : audit.values()) if (keys.size() > 1) ambiguous++;
        return ambiguous;
    }

    public static void main(String[] args) {
        // --- K5: the only induced motifs are the complete graphs.
        boolean[][] k5 = complete(5);
        for (int k = 2; k <= 5; k++) {
            TreeMap<Integer, Long> histogram = motifHistogram(k5, k, permutations(k));
            check(histogramTotal(histogram) == binomial(5, k), "K5, k = " + k + ": C(5,k) subsets");
            check(histogram.size() == 1 && histogram.containsKey(completeKey(k)),
                    "K5, k = " + k + ": the complete graph is the only motif");
            check(histogram.get(completeKey(k)) == binomial(5, k), "K5, k = " + k + ": all subsets");
        }

        // --- C5.
        boolean[][] c5 = graph(5, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}});
        TreeMap<Integer, Long> c5Tri = motifHistogram(c5, 3, permutations(3));
        check(histogramTotal(c5Tri) == 10 && c5Tri.size() == 2, "C5 triples: two motifs, ten subsets");
        check(c5Tri.get(pathKey(3)) == 5, "five triples of C5 induce a 3-vertex path");
        check(!c5Tri.containsKey(completeKey(3)), "no triangle in C5");
        check(trianglesFromMatrix(c5) == 0 && trianglesFromCommonNeighbours(c5) == 0,
                "no triangle in C5, by both independent methods");
        TreeMap<Integer, Long> c5Four = motifHistogram(c5, 4, permutations(4));
        check(histogramTotal(c5Four) == 5 && c5Four.size() == 1
                && c5Four.get(pathKey(4)) == 5, "all five 4-subsets of C5 induce P4");
        TreeMap<Integer, Long> c5Five = motifHistogram(c5, 5, permutations(5));
        check(histogramTotal(c5Five) == 1 && c5Five.containsKey(canonicalKey(c5, new int[]{0, 1, 2, 3, 4},
                permutations(5))), "C5, k = 5: the only motif is the 5-cycle itself");
        check(edgeCount(c5) == 5, "C5 has 5 edges");

        // --- Empty graph.
        boolean[][] empty6 = new boolean[6][6];
        for (int k = 2; k <= 5; k++) {
            TreeMap<Integer, Long> histogram = motifHistogram(empty6, k, permutations(k));
            check(histogram.size() == 1 && histogram.containsKey(0) && histogram.get(0) == binomial(6, k),
                    "the empty graph has the empty motif only, k = " + k);
        }

        // --- 9-vertex seeded test graph carrying a planted triangle and a planted pendant edge.
        Rng rng = new Rng(20240554L);
        boolean[][] test = new boolean[9][9];
        for (int i = 0; i < 9; i++) {
            for (int j = i + 1; j < 9; j++) {
                if (rng.nextDouble() < 0.25) {
                    test[i][j] = true;
                    test[j][i] = true;
                }
            }
        }
        test[0][1] = test[1][0] = test[0][2] = test[2][0] = test[1][2] = test[2][1] = true;
        test[0][3] = test[3][0] = true;

        long visitCount = 0;
        for (int k = 2; k <= 5; k++) {
            long[] visited = new long[1];
            TreeMap<Integer, Long> histogram = new TreeMap<Integer, Long>();
            enumerate(test, k, 0, 0, new int[k], permutations(k), histogram, visited);
            visitCount += visited[0];
            check(visited[0] == binomial(9, k), "test graph, k = " + k + ": C(9," + k + ") subsets");
            check(histogramTotal(histogram) == binomial(9, k),
                    "test graph, k = " + k + ": the counts sum to C(9," + k + ")");
            check(histogram.size() <= unlabeledCount(k),
                    "test graph, k = " + k + ": at most " + unlabeledCount(k) + " motifs");
        }
        TreeMap<Integer, Long> testTri = motifHistogram(test, 3, permutations(3));
        TreeMap<Integer, Long> testFour = motifHistogram(test, 4, permutations(4));
        TreeMap<Integer, Long> testFive = motifHistogram(test, 5, permutations(5));
        check(testTri.size() == 4, "the test graph realises all 4 motifs on 3 vertices");
        check(testFour.size() == 11, "and all 11 motifs on 4 vertices");
        check(testFive.size() == 24, "and 24 of the 34 motifs on 5 vertices");
        check(visitCount == 372 && visitCount == binomial(9, 2) + binomial(9, 3) + binomial(9, 4)
                + binomial(9, 5),
                "the k = 2..5 sweeps visit exactly 372 subsets");

        long triangleKeyCount = testTri.get(completeKey(3));
        check(triangleKeyCount == trianglesFromMatrix(test), "triangles via trace(A^3)/6 ("
                + trianglesFromMatrix(test) + ") match the enumeration (" + triangleKeyCount + ")");
        check(trianglesFromMatrix(test) == trianglesFromCommonNeighbours(test),
                "trace and common-neighbour triangle counts agree");
        check(testFour.get(completeKey(4)) == fourCliquesFromCommonNeighbours(test),
                "4-cliques via common neighbours (" + fourCliquesFromCommonNeighbours(test)
                        + ") match (" + testFour.get(completeKey(4)) + ")");
        check(testFour.get(completeKey(4)) >= 1, "the planted triangle extends to a K4");

        long degreeSum = 0;
        for (int d : degrees(test)) degreeSum += d;
        check(degreeSum / 2 == edgeCount(test), "the degree sum reproduces the edge count");
        TreeMap<Integer, Long> testPairs = motifHistogram(test, 2, permutations(2));
        check(histogramTotal(testPairs) == binomial(9, 2) && testPairs.size() == 2
                && testPairs.get(completeKey(2)) == edgeCount(test),
                "k = 2: edges (" + edgeCount(test) + ") and non-edges");

        // --- Signature versus canonical form, decided by exhausting all graphs on k vertices.
        TreeMap<String, TreeSet<Integer>> audit3 = signatureAudit(3);
        TreeMap<String, TreeSet<Integer>> audit4 = signatureAudit(4);
        TreeMap<String, TreeSet<Integer>> audit5 = signatureAudit(5);
        check(audit3.size() == 4 && ambiguousSignatures(audit3) == 0,
                "k = 3: 4 signatures for 4 classes");
        check(audit4.size() == 11 && ambiguousSignatures(audit4) == 0,
                "k = 4: the signature is complete (11 signatures, 11 classes)");
        check(audit5.size() == 31 && ambiguousSignatures(audit5) == 3,
                "k = 5: 31 signatures for 34 classes, " + ambiguousSignatures(audit5) + " ambiguous");

        boolean[][] p5 = graph(5, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}});
        boolean[][] k3PlusK2 = graph(5, new int[][]{{0, 1}, {1, 2}, {2, 0}, {3, 4}});
        int[] all5 = {0, 1, 2, 3, 4};
        List<int[]> perms5 = permutations(5);
        check(signature(p5, all5).equals(signature(k3PlusK2, all5)),
                "P5 and K3 + K2 share the signature " + signature(p5, all5));
        check(canonicalKey(p5, all5, perms5) != canonicalKey(k3PlusK2, all5, perms5),
                "but their canonical keys differ, so they are not isomorphic");
        check(connected(p5, all5) != connected(k3PlusK2, all5),
                "P5 is connected and K3 + K2 is not");
        check(audit5.get(signature(p5, all5)).size() == 2,
                "the P5 signature is the merging bucket: " + audit5.get(signature(p5, all5)));
        check(signature(p5, all5).equals("4:1,1,2,2,2,"),
                "the shared signature reads " + signature(p5, all5));

        // --- Relabeling invariance and determinism.
        Rng relabelRng = new Rng(554001L);
        boolean[][] shuffled = relabel(test, relabelRng);
        for (int k = 3; k <= 5; k++) {
            TreeMap<Integer, Long> a = motifHistogram(test, k, permutations(k));
            TreeMap<Integer, Long> b = motifHistogram(shuffled, k, permutations(k));
            check(a.equals(b), "relabeling preserves the k = " + k + " histogram");
        }
        check(motifHistogram(test, 4, permutations(4)).equals(testFour), "deterministic re-run");

        System.out.println("PASS");
    }

    /** All k(k-1)/2 pairs present. */
    static int completeKey(int k) {
        return (1 << (k * (k - 1) / 2)) - 1;
    }

    /** Canonical key of the k-vertex path. */
    static int pathKey(int k) {
        int[][] edges = new int[k - 1][2];
        for (int i = 0; i + 1 < k; i++) edges[i] = new int[]{i, i + 1};
        boolean[][] path = graph(k, edges);
        int[] subset = new int[k];
        for (int i = 0; i < k; i++) subset[i] = i;
        return canonicalKey(path, subset, permutations(k));
    }

    static int unlabeledCount(int k) {
        int[] counts = {1, 1, 2, 4, 11, 34};
        return counts[k];
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
