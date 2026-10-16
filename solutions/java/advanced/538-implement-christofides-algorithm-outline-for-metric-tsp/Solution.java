/** Task 538 - Implement Christofides algorithm outline for metric TSP. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The Christofides 1.5-approximation for metric TSP, implemented in full rather than sketched:
 * (1) a minimum spanning tree — built twice, by Prim with an O(n^2) scan and by Kruskal with a
 * weighted union-find over the sorted edge list, the two weights being asserted equal; (2) the
 * odd-degree vertices of that tree; (3) a minimum-weight perfect matching on them, solved by a
 * bitmask DP over subsets of the odd set with parent pointers for reconstruction (and checked
 * against an exhaustive enumeration of every perfect matching); (4) an Eulerian circuit of the
 * multigraph formed by the tree plus the matching, produced by Hierholzer's algorithm; and
 * (5) the shortcutting step that walks the circuit and keeps the first occurrence of each
 * vertex, which by the triangle inequality never lengthens the walk.
 *
 * Documented checks: instances are Euclidean point sets (the triangle inequality is verified on
 * every triple), the returned object is a Hamiltonian cycle (a permutation of 0..n-1) whose cost
 * is recomputed from the distance matrix; Prim and Kruskal agree and the tree has n-1 edges; the
 * matching pairs are disjoint, cover every odd-degree vertex, and its weight equals the brute
 * force optimum; the multigraph has only even degrees, Hierholzer's circuit uses every edge
 * exactly once and has E+1 vertices; the shortcut tour never costs more than the Eulerian walk
 * and never less than the brute-force optimum (all (n-1)!/2 tours, n <= 9), while staying within
 * documented 1.5 * optimum bound (equivalently MST + matching <= MST + OPT/2, the matching
 * being at most half the optimum by the half-tour shortcut argument); the MST weight is a lower
 * bound on the optimum; and the hand-checked 3x4 rectangle gives cost 14 = 3 + 4 + 3 + 4,
 * exactly optimal.
 */
import java.util.*;

public class Solution {
    static final double EPS = 1e-9;

    static double[][] distances(double[][] pts) {
        int n = pts.length;
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dx = pts[i][0] - pts[j][0];
                double dy = pts[i][1] - pts[j][1];
                d[i][j] = Math.sqrt(dx * dx + dy * dy);
            }
        }
        return d;
    }

    /** Prim's O(n^2) variant; parent[root] stays -1; ties resolved by the smallest index. */
    static int[] primParent(double[][] d) {
        int n = d.length;
        boolean[] inTree = new boolean[n];
        double[] key = new double[n];
        int[] parent = new int[n];
        Arrays.fill(key, Double.POSITIVE_INFINITY);
        Arrays.fill(parent, -1);
        key[0] = 0.0;
        for (int iter = 0; iter < n; iter++) {
            int best = -1;
            for (int v = 0; v < n; v++) {
                if (!inTree[v] && (best < 0 || key[v] < key[best])) best = v;
            }
            inTree[best] = true;
            for (int v = 0; v < n; v++) {
                if (!inTree[v] && d[best][v] < key[v]) {
                    key[v] = d[best][v];
                    parent[v] = best;
                }
            }
        }
        return parent;
    }

    static double weight(int[] parent, double[][] d) {
        double total = 0.0;
        for (int v = 0; v < parent.length; v++) {
            if (parent[v] >= 0) total += d[v][parent[v]];
        }
        return total;
    }

    /** Kruskal with union-find over the deterministic weight-sorted edge list. */
    static double kruskalWeight(double[][] d) {
        int n = d.length;
        List<double[]> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) edges.add(new double[]{d[i][j], i, j});
        }
        edges.sort((x, y) -> x[0] != y[0] ? Double.compare(x[0], y[0])
                : (x[1] != y[1] ? Double.compare(x[1], y[1]) : Double.compare(x[2], y[2])));
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        double total = 0.0;
        int used = 0;
        for (double[] e : edges) {
            int a = find(parent, (int) e[1]);
            int b = find(parent, (int) e[2]);
            if (a == b) continue;
            parent[a] = b;
            total += e[0];
            used++;
        }
        return used == n - 1 ? total : Double.POSITIVE_INFINITY;
    }

    static int find(int[] parent, int v) {
        while (parent[v] != v) {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        return v;
    }

    static int[] degreesFromMST(int[] treeParent) {
        int n = treeParent.length;
        int[] degree = new int[n];
        for (int v = 0; v < n; v++) {
            if (treeParent[v] >= 0) {
                degree[v]++;
                degree[treeParent[v]]++;
            }
        }
        return degree;
    }

    static int[] oddVertices(int[] treeParent) {
        int[] degree = degreesFromMST(treeParent);
        List<Integer> odds = new ArrayList<>();
        for (int v = 0; v < degree.length; v++) {
            if (degree[v] % 2 == 1) odds.add(v);
        }
        int[] out = new int[odds.size()];
        for (int i = 0; i < out.length; i++) out[i] = odds.get(i);
        return out;
    }

    /** Bitmask DP over the odd set: pair the lowest remaining index with every other index. */
    static int[][] minWeightMatching(double[][] d, int[] odds) {
        int k = odds.length;
        double[] dp = new double[1 << k];
        int[] choice = new int[1 << k];
        Arrays.fill(dp, Double.POSITIVE_INFINITY);
        Arrays.fill(choice, -1);
        dp[0] = 0.0;
        for (int mask = 1; mask < (1 << k); mask++) {
            int low = Integer.numberOfTrailingZeros(mask);
            int rest = mask ^ (1 << low);
            for (int j = low + 1; j < k; j++) {
                if ((rest & (1 << j)) == 0) continue;
                int previous = rest ^ (1 << j);
                double candidate = dp[previous] + d[odds[low]][odds[j]];
                if (candidate < dp[mask]) {
                    dp[mask] = candidate;
                    choice[mask] = j;
                }
            }
        }
        List<int[]> pairs = new ArrayList<>();
        int mask = (1 << k) - 1;
        while (mask != 0) {
            int low = Integer.numberOfTrailingZeros(mask);
            int j = choice[mask];
            pairs.add(new int[]{low, j});
            mask ^= (1 << low) | (1 << j);
        }
        int[][] out = new int[pairs.size()][2];
        for (int i = 0; i < out.length; i++) {
            out[i][0] = odds[pairs.get(i)[0]];
            out[i][1] = odds[pairs.get(i)[1]];
        }
        return out;
    }

    /** Exhaustive minimum-weight perfect matching, used as the independent oracle. */
    static double bruteMatchingCost(double[][] d, int[] odds, boolean[] taken, double acc) {
        int k = odds.length;
        int first = -1;
        for (int i = 0; i < k; i++) {
            if (!taken[i]) {
                first = i;
                break;
            }
        }
        if (first < 0) return acc;
        taken[first] = true;
        double best = Double.POSITIVE_INFINITY;
        for (int j = first + 1; j < k; j++) {
            if (taken[j]) continue;
            taken[j] = true;
            best = Math.min(best, bruteMatchingCost(d, odds, taken, acc + d[odds[first]][odds[j]]));
            taken[j] = false;
        }
        taken[first] = false;
        return best;
    }

    static double matchingCost(double[][] d, int[][] pairs) {
        double total = 0.0;
        for (int[] pair : pairs) total += d[pair[0]][pair[1]];
        return total;
    }

    /** Hierholzer's algorithm on the multigraph given as an edge list. */
    static List<Integer> eulerCircuit(int n, List<int[]> edges) {
        List<List<Integer>> incident = new ArrayList<>();
        for (int i = 0; i < n; i++) incident.add(new ArrayList<>());
        for (int id = 0; id < edges.size(); id++) {
            incident.get(edges.get(id)[0]).add(id);
            incident.get(edges.get(id)[1]).add(id);
        }
        boolean[] used = new boolean[edges.size()];
        int[] pointer = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        List<Integer> circuit = new ArrayList<>();
        stack.push(0);
        while (!stack.isEmpty()) {
            int v = stack.peek();
            while (pointer[v] < incident.get(v).size() && used[incident.get(v).get(pointer[v])]) pointer[v]++;
            if (pointer[v] < incident.get(v).size()) {
                int id = incident.get(v).get(pointer[v]++);
                used[id] = true;
                int[] e = edges.get(id);
                stack.push(e[0] == v ? e[1] : e[0]);
            } else {
                circuit.add(stack.pop());
            }
        }
        Collections.reverse(circuit);
        for (boolean flag : used) {
            if (!flag) throw new RuntimeException("FAILED: Hierholzer did not use every edge");
        }
        return circuit;
    }

    /** Keep the first occurrence of every vertex; the triangle inequality keeps the cost. */
    static int[] shortcut(List<Integer> circuit, int n) {
        boolean[] seen = new boolean[n];
        List<Integer> tour = new ArrayList<>();
        for (int v : circuit) {
            if (!seen[v]) {
                seen[v] = true;
                tour.add(v);
            }
        }
        if (tour.size() != n) throw new RuntimeException("FAILED: shortcut is not Hamiltonian");
        int[] out = new int[n];
        int at = 0;
        for (int i = 0; i < n; i++) {
            if (tour.get(i) == 0) at = i;
        }
        for (int i = 0; i < n; i++) out[i] = tour.get((at + i) % n);
        return out;
    }

    static double tourCost(double[][] d, int[] tour) {
        double total = 0.0;
        for (int i = 0; i < tour.length; i++) total += d[tour[i]][tour[(i + 1) % tour.length]];
        return total;
    }

    static class Christofides {
        final int[] tour;
        final double cost;
        final double mstWeight;
        final int[] odds;
        final int[][] matching;
        final double eulerCost;
        final int eulerEdges;

        Christofides(int[] tour, double cost, double mstWeight, int[] odds, int[][] matching,
                     double eulerCost, int eulerEdges) {
            this.tour = tour;
            this.cost = cost;
            this.mstWeight = mstWeight;
            this.odds = odds;
            this.matching = matching;
            this.eulerCost = eulerCost;
            this.eulerEdges = eulerEdges;
        }
    }

    static Christofides christofides(double[][] d) {
        int n = d.length;
        int[] tree = primParent(d);
        double mst = weight(tree, d);
        int[] odds = oddVertices(tree);
        int[][] matching = odds.length == 0 ? new int[0][2] : minWeightMatching(d, odds);
        List<int[]> multigraph = new ArrayList<>();
        for (int v = 0; v < n; v++) {
            if (tree[v] >= 0) multigraph.add(new int[]{v, tree[v]});
        }
        for (int[] pair : matching) multigraph.add(new int[]{pair[0], pair[1]});
        List<Integer> circuit = eulerCircuit(n, multigraph);
        int[] tour = shortcut(circuit, n);
        double eulerCost = 0.0;
        for (int i = 0; i + 1 < circuit.size(); i++) eulerCost += d[circuit.get(i)][circuit.get(i + 1)];
        return new Christofides(tour, tourCost(d, tour), mst, odds, matching, eulerCost, multigraph.size());
    }

    static double optimum(double[][] d) {
        int n = d.length;
        if (n == 1) return 0.0;
        int[] rest = new int[n - 1];
        for (int i = 0; i < n - 1; i++) rest[i] = i + 1;
        double best = Double.POSITIVE_INFINITY;
        while (true) {
            int[] tour = new int[n];
            tour[0] = 0;
            System.arraycopy(rest, 0, tour, 1, n - 1);
            if (tour[1] < tour[n - 1]) best = Math.min(best, tourCost(d, tour));
            int i = n - 3;
            while (i >= 0 && rest[i] > rest[i + 1]) i--;
            if (i < 0) break;
            int j = n - 2;
            while (rest[j] < rest[i]) j--;
            int tmp = rest[i];
            rest[i] = rest[j];
            rest[j] = tmp;
            for (int lo = i + 1, hi = n - 2; lo < hi; lo++, hi--) {
                tmp = rest[lo];
                rest[lo] = rest[hi];
                rest[hi] = tmp;
            }
        }
        return best;
    }

    static boolean isPermutation(int[] tour, int n) {
        boolean[] seen = new boolean[n];
        for (int value : tour) {
            if (value < 0 || value >= n || seen[value]) return false;
            seen[value] = true;
        }
        return true;
    }

    static double[][] lcgPoints(int n, int seed, int limit) {
        double[][] pts = new double[n][2];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            pts[i][0] = seed % limit;
            seed = (seed * 75 + 74) % 65537;
            pts[i][1] = seed % limit;
        }
        return pts;
    }

    static void checkMetric(double[][] d) {
        int n = d.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    check(d[i][k] <= d[i][j] + d[j][k] + EPS, "triangle inequality");
                }
            }
        }
    }

    public static void main(String[] args) {
        double[][] rectangle = {{0.0, 0.0}, {3.0, 0.0}, {3.0, 4.0}, {0.0, 4.0}};
        double[][] rd = distances(rectangle);
        Christofides hand = christofides(rd);
        check(isPermutation(hand.tour, 4), "rectangle tour is Hamiltonian");
        check(Math.abs(hand.mstWeight - 10.0) <= EPS, "rectangle MST weight is 3 + 4 + 3");
        check(hand.odds.length == 2, "rectangle MST has exactly two odd vertices");
        check(Math.abs(hand.cost - 14.0) <= EPS, "rectangle Christofides cost is 14");
        check(Math.abs(optimum(rd) - 14.0) <= EPS, "rectangle optimum is 14");
        check(hand.eulerEdges == 4, "multigraph has n-1 + |matching| = 4 edges");

        int[] seeds = {23, 31, 39, 47, 55, 63, 71};
        for (int s = 0; s < seeds.length; s++) {
            int n = 6 + s % 4;
            double[][] d = distances(lcgPoints(n, seeds[s], 100));
            checkMetric(d);
            Christofides result = christofides(d);
            double opt = optimum(d);
            check(isPermutation(result.tour, n), "tour " + s + " is Hamiltonian");
            check(Math.abs(result.cost - tourCost(d, result.tour)) <= EPS, "cost is recomputed exactly");
            check(Math.abs(weight(primParent(d), d) - kruskalWeight(d)) <= EPS,
                    "Prim and Kruskal agree on " + s);
            check(Math.abs(result.mstWeight - kruskalWeight(d)) <= EPS, "reported MST weight matches Kruskal");
            check(result.mstWeight <= opt + EPS, "MST is a lower bound on the optimum");
            check(result.cost >= opt - EPS, "no heuristic beats the optimum");
            check(result.cost <= 1.5 * opt + EPS, "within 1.5 x optimum");
            check(matchingCost(d, result.matching) <= opt / 2 + EPS,
                    "matching weight is at most half the optimum (half-tour shortcut argument)");
            check(result.cost <= result.mstWeight + opt / 2 + EPS, "the MST + matching bound");
            check(result.odds.length % 2 == 0, "an even number of odd-degree vertices");
            boolean[] covered = new boolean[n];
            for (int[] pair : result.matching) {
                check(pair[0] != pair[1] && !covered[pair[0]] && !covered[pair[1]],
                        "matching pairs are disjoint and distinct");
                covered[pair[0]] = true;
                covered[pair[1]] = true;
            }
            for (int v : result.odds) check(covered[v], "every odd vertex is matched");
            int matched = 0;
            for (boolean flag : covered) {
                if (flag) matched++;
            }
            check(matched == result.odds.length, "the matching covers exactly the odd vertices");
            double brute = bruteMatchingCost(d, result.odds, new boolean[result.odds.length], 0.0);
            check(Math.abs(matchingCost(d, result.matching) - brute) <= EPS,
                    "matching is minimum weight (DP equals brute force)");
            check(Math.abs(result.eulerCost - (result.mstWeight + brute)) <= EPS,
                    "the Eulerian walk weighs MST + matching");
            check(result.cost <= result.eulerCost + EPS, "shortcutting never lengthens the walk");
            check(result.eulerEdges == (n - 1) + result.odds.length / 2, "multigraph edge count");
        }

        double[][] single = distances(new double[][]{{2.0, 3.0}});
        Christofides one = christofides(single);
        check(one.tour.length == 1 && Math.abs(one.cost) <= EPS && one.odds.length == 0,
                "single point: no tree, no odd vertices, cost 0");

        double[][] pairPts = {{0.0, 0.0}, {3.0, 4.0}};
        double[][] pd = distances(pairPts);
        Christofides two = christofides(pd);
        check(isPermutation(two.tour, 2) && Math.abs(two.cost - 10.0) <= EPS
                        && two.odds.length == 2, "two points: the matching doubles the tree edge");

        double[][] triPts = {{0.0, 0.0}, {3.0, 0.0}, {0.0, 4.0}};
        double[][] td = distances(triPts);
        Christofides three = christofides(td);
        check(Math.abs(three.cost - 12.0) <= EPS && Math.abs(optimum(td) - 12.0) <= EPS,
                "triangle instance is solved optimally (3 + 4 + 5)");
        check(three.odds.length == 2, "the two tree leaves are the odd vertices");

        double[][] collinear = distances(new double[][]{{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}});
        Christofides line = christofides(collinear);
        check(Math.abs(line.cost - 6.0) <= EPS, "collinear points cost twice the span");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
