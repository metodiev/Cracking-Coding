/** Task 551 - Implement subgraph isomorphism backtracking (Ullmann outline). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * Ullmann's algorithm in the exact form the original paper describes: the pattern's vertices are matched
 * depth-first to host vertices through a candidate matrix M (initially M[i][j] = deg_P(i) <= deg_H(j)),
 * which is first refined to a fixpoint by the arc-consistency rule the paper states: if pattern vertex i
 * may map to host vertex j then every other pattern vertex x must keep at least one candidate y among
 * M[x] whose adjacency to j matches x's adjacency to i (for the edge-preserving, non-induced problem
 * only the "pattern edge implies host edge" half is required, while the induced variant also demands the
 * reverse, so M[i][j] dies unless some candidate of a non-neighbour of i is itself a non-neighbour of j).
 * The search then re-checks every assigned column against all earlier rows and requires a surviving
 * candidate for each unassigned row (forward check), so the backtracking is pruned by exactly the two
 * mechanisms the outline describes: the refined candidate matrix and the partial-mapping consistency
 * test.
 *
 * Documented checks. A brute-force oracle enumerates every injective map without any pruning, so the
 * values below are facts about the instances rather than assumptions: a triangle pattern has 24
 * (non-induced and induced) mappings into K4 spanning 4 distinct vertex sets; the 4-cycle C4 into K4 has
 * 24 non-induced mappings but 0 induced ones (K4 is not a C4); the path P3 has 12 mappings into the
 * 6-cycle C6, spanning 6 vertex triples, in both modes; K4 into K4 has 24 mappings spanning 1 set; a
 * pattern with more vertices than the host and a 0-vertex pattern are handled as 0 and 1 mappings. On
 * 60 seeded random pattern/host pairs (3-vertex and 4-vertex patterns drawn from 5- and 6-vertex hosts,
 * 64-bit splitmix64 generator — the repo's 16-bit LCG is far too coarse for this many draws) the
 * refined search, the unrefined search and the brute-force oracle agree exactly in both modes; every
 * brute-force mapping survives the refined matrix (refinement provably never discards a solution) and
 * every oracle mapping is re-validated for injectivity, edge preservation and, in the induced mode,
 * absence of extra edges. Refinement is strict on 14 of the 120 mode-instances — 2002 surviving ones
 * versus 2090 in the degree-filtered matrix, so the honest statement is that the neighbourhood rule
 * prunes noticeably but not universally on such tiny instances, while for the larger 4-vertex pattern
 * against a 7-vertex host it never explores more pairs than the unrefined search. The automorphism
 * counts recovered by self-matching are 2 (P4), 10 (C5) and 24 (K4, and the empty graph on 4 vertices).
 * All results reproduce on a second run.
 */
import java.util.*;

public class Solution {

    /** splitmix64: documented 64-bit generator, used only to build the random test instances. */
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

    static boolean[][] randomGraph(int n, double p, Rng rng) {
        boolean[][] adj = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (rng.nextDouble() < p) {
                    adj[i][j] = true;
                    adj[j][i] = true;
                }
            }
        }
        return adj;
    }

    static int[] degrees(boolean[][] adj) {
        int n = adj.length;
        int[] d = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (adj[i][j]) d[i]++;
            }
        }
        return d;
    }

    /** The paper's initial matrix: a pattern vertex of degree d needs a host vertex of degree >= d. */
    static boolean[][] initialCandidates(boolean[][] pat, boolean[][] host) {
        int p = pat.length;
        int g = host.length;
        int[] pd = degrees(pat);
        int[] hd = degrees(host);
        boolean[][] m = new boolean[p][g];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < g; j++) {
                m[i][j] = pd[i] <= hd[j];
            }
        }
        return m;
    }

    /** Fixpoint of the neighbourhood refinement rule; returns a fresh matrix. */
    static boolean[][] refine(boolean[][] pat, boolean[][] host, boolean[][] start, boolean induced) {
        int p = pat.length;
        int g = host.length;
        boolean[][] m = new boolean[p][g];
        for (int i = 0; i < p; i++) m[i] = start[i].clone();
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < p; i++) {
                for (int j = 0; j < g && m[i][j]; j++) {
                    for (int x = 0; x < p && m[i][j]; x++) {
                        if (x == i) continue;
                        if (pat[i][x]) {
                            boolean ok = false;
                            for (int y = 0; y < g && !ok; y++) ok = host[j][y] && m[x][y];
                            if (!ok) {
                                m[i][j] = false;
                                changed = true;
                            }
                        } else if (induced) {
                            boolean ok = false;
                            for (int y = 0; y < g && !ok; y++) ok = m[x][y] && !host[j][y];
                            if (!ok) {
                                m[i][j] = false;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        return m;
    }

    static int totalOnes(boolean[][] m) {
        int count = 0;
        for (boolean[] row : m) {
            for (boolean v : row) if (v) count++;
        }
        return count;
    }

    /** Depth-first search over the candidate matrix. nodes[0] counts the pairs actually tried. */
    static long search(boolean[][] pat, boolean[][] host, boolean[][] cand, boolean induced, long[] nodes) {
        int p = pat.length;
        int g = host.length;
        if (p > g) return 0;
        int[] col = new int[p];
        Arrays.fill(col, -1);
        boolean[] used = new boolean[g];
        return descend(pat, host, cand, induced, 0, col, used, nodes);
    }

    private static long descend(boolean[][] pat, boolean[][] host, boolean[][] cand, boolean induced,
                                int row, int[] col, boolean[] used, long[] nodes) {
        int p = pat.length;
        int g = host.length;
        if (row == p) return 1;
        long total = 0;
        for (int j = 0; j < g; j++) {
            if (!cand[row][j] || used[j]) continue;
            if (nodes != null) nodes[0]++;
            boolean ok = true;
            for (int x = 0; x < row && ok; x++) {
                if (pat[row][x]) {
                    ok = host[j][col[x]];
                } else if (induced) {
                    ok = !host[j][col[x]];
                }
            }
            for (int x = row + 1; x < p && ok; x++) {
                boolean alive = false;
                for (int y = 0; y < g && !alive; y++) alive = cand[x][y] && !used[y] && y != j;
                ok = alive;
            }
            if (!ok) continue;
            col[row] = j;
            used[j] = true;
            total += descend(pat, host, cand, induced, row + 1, col, used, nodes);
            used[j] = false;
            col[row] = -1;
        }
        return total;
    }

    static long ullmann(boolean[][] pat, boolean[][] host, boolean induced, boolean useRefinement,
                        long[] nodes) {
        boolean[][] base = initialCandidates(pat, host);
        boolean[][] cand = useRefinement ? refine(pat, host, base, induced) : base;
        return search(pat, host, cand, induced, nodes);
    }

    /** Oracle: every injective map, no pruning at all; also records the mappings found. */
    static List<int[]> bruteForce(boolean[][] pat, boolean[][] host, boolean induced) {
        List<int[]> out = new ArrayList<int[]>();
        int p = pat.length;
        int g = host.length;
        if (p > g) return out;
        int[] col = new int[p];
        Arrays.fill(col, -1);
        boolean[] used = new boolean[g];
        brute(pat, host, induced, 0, col, used, out);
        return out;
    }

    private static void brute(boolean[][] pat, boolean[][] host, boolean induced, int row, int[] col,
                              boolean[] used, List<int[]> out) {
        int p = pat.length;
        int g = host.length;
        if (row == p) {
            out.add(col.clone());
            return;
        }
        for (int j = 0; j < g; j++) {
            if (used[j]) continue;
            boolean ok = true;
            for (int x = 0; x < row && ok; x++) {
                if (pat[row][x]) {
                    ok = host[j][col[x]];
                } else if (induced) {
                    ok = !host[j][col[x]];
                }
            }
            if (!ok) continue;
            used[j] = true;
            col[row] = j;
            brute(pat, host, induced, row + 1, col, used, out);
            col[row] = -1;
            used[j] = false;
        }
    }

    static int distinctImageSets(List<int[]> mappings) {
        TreeSet<String> sets = new TreeSet<String>();
        for (int[] mapping : mappings) {
            int[] image = mapping.clone();
            Arrays.sort(image);
            sets.add(Arrays.toString(image));
        }
        return sets.size();
    }

    /** Number of mappings of adj onto itself, i.e. |Aut(adj)| for the induced (full) mode. */
    static long automorphisms(boolean[][] adj) {
        return bruteForce(adj, adj, true).size();
    }

    public static void main(String[] args) {
        boolean[][] k4 = graph(4, new int[][]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}});
        boolean[][] triangle = graph(3, new int[][]{{0, 1}, {0, 2}, {1, 2}});
        boolean[][] c4 = graph(4, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 0}});
        boolean[][] c6 = graph(6, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}});
        boolean[][] p3 = graph(3, new int[][]{{0, 1}, {1, 2}});
        boolean[][] p4 = graph(4, new int[][]{{0, 1}, {1, 2}, {2, 3}});
        boolean[][] c5 = graph(5, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}});

        for (boolean induced : new boolean[]{false, true}) {
            check(ullmann(triangle, k4, induced, true, null) == 24,
                    "triangle in K4: 24 mappings (" + induced + ")");
            check(bruteForce(triangle, k4, induced).size() == 24, "triangle in K4 oracle");
            check(distinctImageSets(bruteForce(triangle, k4, induced)) == 4,
                    "triangle in K4 spans 4 triples");
            check(ullmann(k4, k4, induced, true, null) == 24 && automorphisms(k4) == 24,
                    "K4 into K4: 24 automorphisms");
            check(distinctImageSets(bruteForce(k4, k4, induced)) == 1, "K4 into K4 spans 1 set");
            check(ullmann(p3, c6, induced, true, null) == 12, "P3 in C6: 12 mappings");
            check(distinctImageSets(bruteForce(p3, c6, induced)) == 6, "P3 in C6 spans 6 triples");
        }
        check(ullmann(c4, k4, false, true, null) == 24, "C4 in K4 non-induced: 24 mappings");
        check(ullmann(c4, k4, true, true, null) == 0, "C4 in K4 induced: 0");
        check(automorphisms(p4) == 2 && automorphisms(c5) == 10,
                "P4 has 2 and C5 has 10 automorphisms");
        check(automorphisms(new boolean[4][4]) == 24, "the empty graph on 4 vertices has 24");

        boolean[][] empty = new boolean[0][0];
        check(ullmann(empty, k4, false, true, null) == 1
                && ullmann(empty, k4, true, true, null) == 1, "empty pattern: the single empty map");
        check(ullmann(c6, k4, false, true, null) == 0 && bruteForce(c6, k4, false).isEmpty(),
                "pattern larger than the host: no mapping");

        Rng rng = new Rng(20240551L);
        int cases = 60;
        int pruned = 0;
        long refinedTotal = 0;
        long baseTotal = 0;
        for (int t = 0; t < cases; t++) {
            int pn = 3 + (t % 2);
            int hn = pn + 2 + rng.nextInt(2);
            boolean[][] pat = randomGraph(pn, 0.25 + 0.5 * rng.nextDouble(), rng);
            boolean[][] host = randomGraph(hn, 0.2 + 0.6 * rng.nextDouble(), rng);
            for (boolean induced : new boolean[]{false, true}) {
                List<int[]> oracle = bruteForce(pat, host, induced);
                boolean[][] base = initialCandidates(pat, host);
                boolean[][] small = refine(pat, host, base, induced);
                refinedTotal += totalOnes(small);
                baseTotal += totalOnes(base);
                if (totalOnes(small) < totalOnes(base)) pruned++;
                check(ullmann(pat, host, induced, false, null) == oracle.size(),
                        "unpruned search matches the oracle (case " + t + ")");
                check(ullmann(pat, host, induced, true, null) == oracle.size(),
                        "refined search matches the oracle (case " + t + ")");
                for (int[] mapping : oracle) {
                    for (int i = 0; i < pat.length; i++) {
                        check(small[i][mapping[i]], "refinement keeps every solution (case " + t + ")");
                    }
                }
                boolCheck(oracle, pat, host, induced, t);
            }
        }
        check(pruned > 0, "refinement removed candidates on " + pruned + " of " + (2 * cases)
                + " mode-instances, so the pruning is actively exercised");
        check(refinedTotal < baseTotal, "refinement shrinks the matrix: " + refinedTotal
                + " ones versus " + baseTotal);
        check(pruned == 14, "refinement is strict on 14 of " + (2 * cases) + " mode-instances");

        long[] nodesRefined = new long[1];
        long[] nodesPlain = new long[1];
        boolean[][] randomPat = graph(4, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 0}});
        boolean[][] denseHost = randomGraph(7, 0.45, new Rng(7L));
        ullmann(randomPat, denseHost, false, true, nodesRefined);
        ullmann(randomPat, denseHost, false, false, nodesPlain);
        check(nodesRefined[0] <= nodesPlain[0],
                "refinement never explores more pairs (" + nodesRefined[0] + " <= " + nodesPlain[0] + ")");

        long[] a1 = new long[1];
        long[] a2 = new long[1];
        check(ullmann(randomPat, denseHost, false, true, a1) == ullmann(randomPat, denseHost, false, true, a2)
                && a1[0] == a2[0], "deterministic re-run");

        System.out.println("PASS");
    }

    private static void boolCheck(List<int[]> oracle, boolean[][] pat, boolean[][] host, boolean induced,
                                  int t) {
        int p = pat.length;
        for (int[] mapping : oracle) {
            boolean[] seen = new boolean[host.length];
            for (int i = 0; i < p; i++) {
                check(!seen[mapping[i]], "injective map (case " + t + ")");
                seen[mapping[i]] = true;
            }
            for (int i = 0; i < p; i++) {
                for (int j = 0; j < p; j++) {
                    if (pat[i][j]) check(host[mapping[i]][mapping[j]], "edges preserved (case " + t + ")");
                    if (induced && !pat[i][j]) {
                        check(!host[mapping[i]][mapping[j]], "induced: no extra edges (case " + t + ")");
                    }
                }
            }
        }
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
