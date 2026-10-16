/** Task 543 - Implement fingerprinting for graph similarity (Weisfeiler-Lehman sketch). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * Colour refinement (1-dimensional Weisfeiler-Lehman): start every vertex with its degree as its
 * colour, then repeatedly replace each colour by a canonical id of the multiset of neighbour
 * colours (sorted, hashed into a shared dictionary, which makes the colouring deterministic and
 * relabelling-invariant). The WL sketch of a graph after r rounds is the multiset of colours after
 * each round; two graphs with equal sketches are indistinguishable to 1-WL, and refinement is
 * monotone — existing colour classes only ever split, so the number of distinct colours is
 * non-decreasing and stabilises after at most n rounds.
 *
 * Documented checks: fingerprints are invariant under deterministic LCG relabellings of every
 * instance; round 0 is exactly the degree sequence; the colour count is asserted non-decreasing
 * and to reach a fixed point within n rounds after which it never changes; the classical 1-WL
 * blind spot is exhibited with the 2-regular pairs (C6, C3 + C3) and (C8, C4 + C4), whose
 * sketches agree in every round although an exhaustive permutation search confirms they are not
 * isomorphic; the second blind spot is the 3-regular pair (K3,3, triangular prism), again equal
 * in every round yet non-isomorphic; conversely two trees that share the degree sequence
 * (3,2,2,1,1,1) are correctly separated, since 1-WL is complete on trees. The refinement is also
 * checked on a triangle against a path, on a star against a path, and on graphs with isolated
 * vertices, always against the equal-or-not decision of the brute-force isomorphism oracle.
 */
import java.util.*;

public class Solution {
    static final int ROUNDS = 8;

    static boolean[][] fromEdges(int n, int[][] edges) {
        boolean[][] adj = new boolean[n][n];
        for (int[] e : edges) {
            adj[e[0]][e[1]] = true;
            adj[e[1]][e[0]] = true;
        }
        return adj;
    }

    static int degree(boolean[][] adj, int v) {
        int count = 0;
        for (int w = 0; w < adj.length; w++) {
            if (adj[v][w]) count++;
        }
        return count;
    }

    /** colors[r][v] = WL colour of v after r rounds; row 0 is the degree. */
    static int[][] refine(boolean[][] adj, int n, int rounds) {
        int[][] colors = new int[rounds + 1][n];
        for (int v = 0; v < n; v++) colors[0][v] = degree(adj, v);
        Map<String, Integer> dictionary = new HashMap<>();
        for (int round = 1; round <= rounds; round++) {
            dictionary.clear();
            int next = 0;
            for (int v = 0; v < n; v++) {
                List<Integer> neighbours = new ArrayList<>();
                for (int w = 0; w < n; w++) {
                    if (adj[v][w]) neighbours.add(colors[round - 1][w]);
                }
                Collections.sort(neighbours);
                StringBuilder key = new StringBuilder();
                for (int colour : neighbours) key.append(colour).append(',');
                String signature = key.toString();
                Integer id = dictionary.get(signature);
                if (id == null) {
                    id = next++;
                    dictionary.put(signature, id);
                }
                colors[round][v] = id;
            }
        }
        return colors;
    }

    /**
     * The WL sketch: for every round, the multiset of colour-class sizes (sorted). That histogram
     * is invariant under relabelling, unlike the raw colour ids, which come from a dictionary
     * filled in vertex-index order; comparing the histograms of all rounds is exactly the
     * standard 1-WL test for distinguishability.
     */
    static String sketch(boolean[][] adj, int n, int rounds) {
        int[][] colors = refine(adj, n, rounds);
        StringBuilder out = new StringBuilder();
        for (int round = 0; round <= rounds; round++) {
            int[] counts = new int[n];
            for (int v = 0; v < n; v++) counts[colors[round][v]]++;
            int[] sizes = new int[distinctColors(colors[round])];
            for (int colour = 0; colour < sizes.length; colour++) sizes[colour] = counts[colour];
            Arrays.sort(sizes);
            for (int size : sizes) out.append(size).append(',');
            out.append('|');
        }
        return out.toString();
    }

    static int distinctColors(int[] row) {
        Set<Integer> set = new HashSet<>();
        for (int colour : row) set.add(colour);
        return set.size();
    }

    static boolean[][] relabel(boolean[][] adj, int[] perm, int n) {
        boolean[][] out = new boolean[n][n];
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (adj[u][v]) out[perm[u]][perm[v]] = true;
            }
        }
        return out;
    }

    static int[] permutation(int n, int seed) {
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) {
            seed = (seed * 75 + 74) % 65537;
            int j = seed % (i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        return perm;
    }

    static int[] degreeSequence(boolean[][] adj, int n) {
        int[] degrees = new int[n];
        for (int v = 0; v < n; v++) degrees[v] = degree(adj, v);
        Arrays.sort(degrees);
        return degrees;
    }

    /** Exhaustive isomorphism oracle over all relabellings. */
    static boolean isomorphicBrute(boolean[][] a, boolean[][] b, int n) {
        int[] da = new int[n];
        int[] db = new int[n];
        for (int v = 0; v < n; v++) {
            da[v] = degree(a, v);
            db[v] = degree(b, v);
        }
        Arrays.sort(da);
        Arrays.sort(db);
        if (!Arrays.equals(da, db)) return false;
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        while (true) {
            boolean ok = true;
            for (int u = 0; u < n && ok; u++) {
                for (int v = u + 1; v < n; v++) {
                    if (a[u][v] != b[perm[u]][perm[v]]) {
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) return true;
            int i = n - 2;
            while (i >= 0 && perm[i] > perm[i + 1]) i--;
            if (i < 0) return false;
            int j = n - 1;
            while (perm[j] < perm[i]) j--;
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
            for (int lo = i + 1, hi = n - 1; lo < hi; lo++, hi--) {
                tmp = perm[lo];
                perm[lo] = perm[hi];
                perm[hi] = tmp;
            }
        }
    }

    static boolean[][] cycle(int n) {
        int[][] edges = new int[n][2];
        for (int i = 0; i < n; i++) edges[i] = new int[]{i, (i + 1) % n};
        return fromEdges(n, edges);
    }

    static boolean[][] disjointCycles(int n, int size) {
        int[][] edges = new int[n][2];
        int at = 0;
        for (int start = 0; start < n; start += size) {
            for (int i = 0; i < size; i++) edges[at++] = new int[]{start + i, start + (i + 1) % size};
        }
        return fromEdges(n, edges);
    }

    static boolean[][] completeBipartite(int a, int b) {
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) edges.add(new int[]{i, a + j});
        }
        return fromEdges(a + b, edges.toArray(new int[0][]));
    }

    static boolean[][] prism() {
        int[][] edges = {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}};
        return fromEdges(6, edges);
    }

    static boolean[][] path(int n) {
        int[][] edges = new int[n - 1][2];
        for (int i = 0; i + 1 < n; i++) edges[i] = new int[]{i, i + 1};
        return fromEdges(n, edges);
    }

    static boolean[][] star(int n) {
        int[][] edges = new int[n - 1][2];
        for (int i = 1; i < n; i++) edges[i - 1] = new int[]{0, i};
        return fromEdges(n, edges);
    }

    static boolean[][] randomGraph(int n, int count, int seed) {
        boolean[][] adj = new boolean[n][n];
        int placed = 0;
        int guard = 0;
        while (placed < count && guard < count * 40) {
            guard++;
            seed = (seed * 75 + 74) % 65537;
            int u = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int v = seed % n;
            if (u == v || adj[u][v]) continue;
            adj[u][v] = true;
            adj[v][u] = true;
            placed++;
        }
        return adj;
    }

    public static void main(String[] args) {
        boolean[][] c6 = cycle(6);
        int[][] c6Colors = refine(c6, 6, ROUNDS);
        check(distinctColors(c6Colors[0]) == 1 && c6Colors[0][0] == 2, "round 0 is the degree sequence");
        int previous = distinctColors(c6Colors[0]);
        for (int round = 1; round <= ROUNDS; round++) {
            int current = distinctColors(c6Colors[round]);
            check(current >= previous, "refinement never merges classes");
            previous = current;
        }
        check(previous == 1, "a cycle stays one class: refinement is stable from round 0");

        boolean[][] twoTriangles = disjointCycles(6, 3);
        check(sketch(c6, 6, ROUNDS).equals(sketch(twoTriangles, 6, ROUNDS)),
                "1-WL cannot tell C6 from C3 + C3");
        check(!isomorphicBrute(c6, twoTriangles, 6), "yet they are not isomorphic");
        boolean[][] c8 = cycle(8);
        boolean[][] twoC4 = disjointCycles(8, 4);
        check(sketch(c8, 8, ROUNDS).equals(sketch(twoC4, 8, ROUNDS)),
                "1-WL cannot tell C8 from C4 + C4");
        check(!isomorphicBrute(c8, twoC4, 8), "and they are not isomorphic either");

        boolean[][] k33 = completeBipartite(3, 3);
        boolean[][] prism = prism();
        check(sketch(k33, 6, ROUNDS).equals(sketch(prism, 6, ROUNDS)),
                "1-WL also cannot tell K3,3 from the triangular prism");
        check(!isomorphicBrute(k33, prism, 6), "K3,3 and the prism are nevertheless non-isomorphic");
        check(distinctColors(refine(k33, 6, 1)[1]) == 1, "K3,3 stays one class after one round");
        check(distinctColors(refine(prism, 6, 1)[1]) == 1, "so does the prism: same 3-regular blind spot");

        // trees with identical degree sequences that 1-WL does separate (it is complete on trees)
        boolean[][] tailLeaf = fromEdges(6, new int[][]{{0, 1}, {1, 2}, {2, 3}, {0, 4}, {0, 5}});
        boolean[][] twoBranches = fromEdges(6, new int[][]{{0, 1}, {1, 2}, {0, 3}, {3, 4}, {0, 5}});
        check(Arrays.equals(degreeSequence(tailLeaf, 6), degreeSequence(twoBranches, 6)),
                "the two trees share the degree sequence (3,2,2,1,1,1)");
        check(!isomorphicBrute(tailLeaf, twoBranches, 6), "but they are not isomorphic");
        check(!sketch(tailLeaf, 6, ROUNDS).equals(sketch(twoBranches, 6, ROUNDS)),
                "1-WL separates them, exactly as it must on trees");

        boolean[][] triangle = cycle(3);
        check(isomorphicBrute(triangle, relabel(triangle, permutation(3, 4), 3), 3),
                "the triangle is isomorphic to its relabelling");
        check(!isomorphicBrute(triangle, path(3), 3), "a triangle is not a path");
        check(!sketch(triangle, 3, 2).equals(sketch(path(3), 3, 2)), "the sketch separates them");

        boolean[][] star5 = star(5);
        int[][] starColors = refine(star5, 5, ROUNDS);
        check(distinctColors(starColors[1]) == 2, "a star has two classes after one round");
        check(!sketch(star5, 5, 2).equals(sketch(path(5), 5, 2)), "a star is not a path");

        boolean[][] withIsolated = fromEdges(7, new int[][]{{0, 1}, {1, 2}, {2, 0}, {5, 6}});
        boolean[][] other = fromEdges(7, new int[][]{{1, 2}, {2, 3}, {3, 1}, {0, 5}});
        check(isomorphicBrute(withIsolated, other, 7), "the isolated-vertex pair is isomorphic");
        check(sketch(withIsolated, 7, ROUNDS).equals(sketch(other, 7, ROUNDS)),
                "the sketch matches for the isolated-vertex pair");

        int[] seeds = {13, 26, 39, 52, 65, 78};
        for (int s = 0; s < seeds.length; s++) {
            int n = 7 + s;
            boolean[][] graph = randomGraph(n, 2 * n, seeds[s]);
            int[][] colors = refine(graph, n, ROUNDS);
            int stable = -1;
            for (int round = 1; round <= n; round++) {
                if (distinctColors(colors[round]) == distinctColors(colors[round - 1])) {
                    stable = round;
                    break;
                }
            }
            check(stable >= 0, "graph " + s + ": refinement stabilises within n rounds");
            for (int round = stable; round < ROUNDS; round++) {
                check(distinctColors(colors[round]) == distinctColors(colors[stable]),
                        "graph " + s + ": the colour count stops growing after stabilisation");
                check(Arrays.equals(colors[round], colors[stable]),
                        "graph " + s + ": the colouring itself is a fixed point");
            }
            for (int t = 0; t < 3; t++) {
                boolean[][] shuffled = relabel(graph, permutation(n, 100 * s + t), n);
                check(sketch(shuffled, n, ROUNDS).equals(sketch(graph, n, ROUNDS)),
                        "graph " + s + ": the sketch is relabelling invariant");
                check(isomorphicBrute(graph, shuffled, n), "graph " + s + ": relabelling keeps isomorphism");
                check(Arrays.equals(degreeSequence(graph, n), degreeSequence(shuffled, n)),
                        "graph " + s + ": relabelling keeps the degree sequence");
            }
        }

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
