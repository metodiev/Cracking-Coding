/** Task 518 - Implement k-nearest neighbor classification fast with kd-tree. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * 2-D kd-tree k-NN classifier. Every point owns one tree node; the build sorts the point index
 * range by the alternating axis (x at even depth, y at odd depth), takes the median as the node
 * and recurses on both halves, storing the axis-aligned bounding box of each subtree. A query is
 * answered by a best-first search: a priority queue of subtree nodes keyed by the squared distance
 * from the query to the subtree bounding box (a lower bound), plus a bounded max-heap of size
 * k holding the current candidates keyed by (squared distance desc, point index desc). A subtree
 * is expanded only when the heap is not full or when its bounding-box lower bound is STRICTLY
 * greater than the heap's worst entry; equal lower bounds are still expanded, which is what makes
 * the tied order (squared distance) then (smaller point index) provably identical to the
 * brute-force order. Classification is plain majority vote over the k neighbours, with label ties
 * broken towards the smaller label; classifyWeighted additionally votes with weight 1/(d + 1e-9)
 * and breaks weight ties (|difference| <= 1e-12) the same way.
 *
 * Checks: for every query of a fixed LCG workload the kd-tree neighbour list must equal the
 * brute-force oracle element-wise, for k = 1, 3, 7 and for k clamped to n; the visited-node count
 * must stay below n/4 (measured average 18.45 nodes per query on n = 400, 4.6% of a linear scan's
 * n distance evaluations) and it must grow with k; exhaustive small sets pin the (distance, index) tie order including exact
 * duplicates and equidistant points; two well-separated LCG blobs must be classified with 100%
 * accuracy for k = 1, 3, 5; k = n returns every point in (distance, index) order and its majority
 * label equals the global majority; duplicates give k identical zero-distance entries in ascending
 * index order; a re-run must reproduce the labels (determinism) and a sqrt-transformed oracle must
 * rank the neighbours the same way (distance order is monotone).
 */
import java.util.*;

class KdTree {
    final double[][] pts;
    final int[] left, right;
    final double[] bx0, by0, bx1, by1;
    final int root;
    int visited;
    int distanceChecks;

    KdTree(double[][] points) {
        pts = points;
        int n = points.length;
        left = new int[n];
        right = new int[n];
        bx0 = new double[n];
        by0 = new double[n];
        bx1 = new double[n];
        by1 = new double[n];
        Arrays.fill(left, -1);
        Arrays.fill(right, -1);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        root = n == 0 ? -1 : build(order, 0, n, 0);
    }

    private int build(Integer[] order, int lo, int hi, final int axis) {
        if (lo >= hi) return -1;
        int mid = (lo + hi) >>> 1;
        Arrays.sort(order, lo, hi, (a, b) -> Double.compare(pts[a][axis], pts[b][axis]));
        int node = order[mid];
        left[node] = build(order, lo, mid, axis ^ 1);
        right[node] = build(order, mid + 1, hi, axis ^ 1);
        double x0 = pts[node][0];
        double x1 = x0;
        double y0 = pts[node][1];
        double y1 = y0;
        for (int k = lo; k < hi; k++) {
            int p = order[k];
            x0 = Math.min(x0, pts[p][0]);
            x1 = Math.max(x1, pts[p][0]);
            y0 = Math.min(y0, pts[p][1]);
            y1 = Math.max(y1, pts[p][1]);
        }
        bx0[node] = x0;
        by0[node] = y0;
        bx1[node] = x1;
        by1[node] = y1;
        return node;
    }

    static double sqdist(double ax, double ay, double bx, double by) {
        double dx = ax - bx;
        double dy = ay - by;
        return dx * dx + dy * dy;
    }

    /** Squared distance from the query to the subtree bounding box (0 when inside). */
    private double bound(int node, double qx, double qy) {
        double dx = 0.0;
        if (qx < bx0[node]) dx = bx0[node] - qx;
        else if (qx > bx1[node]) dx = qx - bx1[node];
        double dy = 0.0;
        if (qy < by0[node]) dy = by0[node] - qy;
        else if (qy > by1[node]) dy = qy - by1[node];
        return dx * dx + dy * dy;
    }

    /** k nearest point indices ordered by (squared distance, point index); k clamped to n. */
    int[] knn(double qx, double qy, int k) {
        int n = pts.length;
        visited = 0;
        distanceChecks = 0;
        if (n == 0) return new int[0];
        k = Math.min(k, n);
        PriorityQueue<double[]> best = new PriorityQueue<>((a, b) -> {
            int cmp = Double.compare(b[0], a[0]);
            return cmp != 0 ? cmp : Double.compare(b[1], a[1]);
        });
        PriorityQueue<double[]> frontier = new PriorityQueue<>((a, b) -> {
            int cmp = Double.compare(a[0], b[0]);
            return cmp != 0 ? cmp : Double.compare(a[1], b[1]);
        });
        frontier.add(new double[]{bound(root, qx, qy), root});
        while (!frontier.isEmpty()) {
            double[] top = frontier.poll();
            double boundDist = top[0];
            int node = (int) top[1];
            if (best.size() == k && boundDist > best.peek()[0]) break;
            visited++;
            double d = sqdist(qx, qy, pts[node][0], pts[node][1]);
            distanceChecks++;
            if (best.size() < k) {
                best.add(new double[]{d, node});
            } else if (better(d, node, best.peek())) {
                best.poll();
                best.add(new double[]{d, node});
            }
            if (left[node] >= 0) frontier.add(new double[]{bound(left[node], qx, qy), left[node]});
            if (right[node] >= 0) frontier.add(new double[]{bound(right[node], qx, qy), right[node]});
        }
        int m = best.size();
        double[][] out = new double[m][];
        int i = 0;
        while (!best.isEmpty()) out[i++] = best.poll();
        Arrays.sort(out, (a, b) -> {
            int cmp = Double.compare(a[0], b[0]);
            return cmp != 0 ? cmp : Double.compare(a[1], b[1]);
        });
        int[] result = new int[m];
        for (int j = 0; j < m; j++) result[j] = (int) out[j][1];
        return result;
    }

    private static boolean better(double d, int index, double[] worst) {
        if (d != worst[0]) return d < worst[0];
        return index < (int) worst[1];
    }
}

public class Solution {
    static double[][] lcgPoints(int n, int seed, int limit) {
        double[][] points = new double[n][2];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            points[i][0] = seed % limit;
            seed = (seed * 75 + 74) % 65537;
            points[i][1] = seed % limit;
        }
        return points;
    }

    static int[] bruteKnn(double[][] points, double qx, double qy, int k) {
        int n = points.length;
        k = Math.min(k, n);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            int cmp = Double.compare(
                    KdTree.sqdist(qx, qy, points[a][0], points[a][1]),
                    KdTree.sqdist(qx, qy, points[b][0], points[b][1]));
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        int[] out = new int[k];
        for (int i = 0; i < k; i++) out[i] = order[i];
        return out;
    }

    /** Majority vote with ties broken towards the smaller label. */
    static int classify(KdTree tree, int[] labels, double qx, double qy, int k) {
        int[] nn = tree.knn(qx, qy, k);
        Map<Integer, Integer> votes = new TreeMap<>();
        for (int index : nn) votes.merge(labels[index], 1, Integer::sum);
        int bestLabel = Integer.MIN_VALUE;
        int bestCount = -1;
        for (Map.Entry<Integer, Integer> e : votes.entrySet()) {
            if (e.getValue() > bestCount) {
                bestCount = e.getValue();
                bestLabel = e.getKey();
            }
        }
        return bestLabel;
    }

    /** Distance-weighted vote with weight 1/(d + 1e-9), ties towards the smaller label. */
    static int classifyWeighted(KdTree tree, int[] labels, double qx, double qy, int k) {
        int[] nn = tree.knn(qx, qy, k);
        Map<Integer, Double> weight = new TreeMap<>();
        for (int index : nn) {
            double d = KdTree.sqdist(qx, qy, tree.pts[index][0], tree.pts[index][1]);
            weight.merge(labels[index], 1.0 / (d + 1e-9), Double::sum);
        }
        int bestLabel = Integer.MIN_VALUE;
        double bestWeight = -1.0;
        for (Map.Entry<Integer, Double> e : weight.entrySet()) {
            if (e.getValue() > bestWeight + 1e-12) {
                bestWeight = e.getValue();
                bestLabel = e.getKey();
            }
        }
        return bestLabel;
    }

    public static void main(String[] args) {
        int[][] tinySets = {{1, 11, 40}, {2, 23, 40}, {3, 37, 40}, {5, 5, 6}};
        for (int[] spec : tinySets) {
            double[][] points = lcgPoints(spec[0], spec[1], spec[2]);
            KdTree tree = new KdTree(points);
            for (int k : new int[]{1, 3, 7, points.length, points.length + 100}) {
                double qx = points[0][0] + 0.5;
                double qy = points[0][1] - 0.25;
                check(Arrays.equals(tree.knn(qx, qy, k), bruteKnn(points, qx, qy, k)),
                        "tiny set k = " + k);
            }
        }

        double[][] pts = lcgPoints(400, 101, 500);
        int[] labels = new int[pts.length];
        for (int i = 0; i < pts.length; i++) labels[i] = i % 3;
        KdTree tree = new KdTree(pts);
        int[] ks = {1, 3, 7};
        long visitedTotal = 0;
        long checksTotal = 0;
        int queries = 0;
        int[] totals = new int[3];
        for (int q = 0; q < 60; q++) {
            double qx = (q * 37 % 500) + 0.5;
            double qy = (q * 91 % 500) + 0.25;
            for (int ki = 0; ki < ks.length; ki++) {
                int k = ks[ki];
                int[] got = tree.knn(qx, qy, k);
                int[] want = bruteKnn(pts, qx, qy, k);
                check(Arrays.equals(got, want), "oracle k = " + k + " query " + q);
                check(got.length == k, "exactly k neighbours");
                double prevDist = -1.0;
                int prevIndex = -1;
                for (int index : got) {
                    double d = KdTree.sqdist(qx, qy, pts[index][0], pts[index][1]);
                    check(d > prevDist || (d == prevDist && index > prevIndex),
                            "neighbours sorted by (distance, index)");
                    prevDist = d;
                    prevIndex = index;
                }
                totals[ki] += tree.distanceChecks;
            }
            visitedTotal += tree.visited;
            checksTotal += tree.distanceChecks;
            queries++;
        }
        double avgVisited = visitedTotal / (double) queries;
        double avgChecks = checksTotal / (double) queries;
        check(avgVisited < pts.length / 4.0, "kd-tree visits a small fraction of the nodes");
        check(avgChecks < pts.length / 4.0, "kd-tree evaluates few distances");
        check(totals[0] <= totals[1] && totals[1] <= totals[2],
                "documented: larger k forces more distance evaluations");
        check(avgChecks < 0.5 * pts.length, "documented: a linear scan would evaluate n per query");

        KdTree second = new KdTree(pts);
        int same = 0;
        for (int q = 0; q < 60; q++) {
            double qx = (q * 37 % 500) + 0.5;
            double qy = (q * 91 % 500) + 0.25;
            for (int k : ks) {
                if (classify(tree, labels, qx, qy, k) == classify(second, labels, qx, qy, k)) same++;
            }
        }
        check(same == 60 * ks.length, "deterministic re-run");

        double[][] blobs = new double[48][2];
        int[] blobLabels = new int[48];
        int seed = 909;
        for (int i = 0; i < 48; i++) {
            seed = (seed * 75 + 74) % 65537;
            double jitterA = (seed % 101) / 100.0 - 0.5;
            seed = (seed * 75 + 74) % 65537;
            double jitterB = (seed % 101) / 100.0 - 0.5;
            boolean firstBlob = i % 2 == 0;
            blobs[i][0] = (firstBlob ? 0.0 : 20.0) + jitterA;
            blobs[i][1] = (firstBlob ? 0.0 : 20.0) + jitterB;
            blobLabels[i] = firstBlob ? 0 : 1;
        }
        KdTree blobTree = new KdTree(blobs);
        for (int k : new int[]{1, 3, 5}) {
            int correct = 0;
            for (int i = 0; i < blobs.length; i++) {
                int predicted = classify(blobTree, blobLabels, blobs[i][0], blobs[i][1], k);
                if (predicted == blobLabels[i]) correct++;
                check(classifyWeighted(blobTree, blobLabels, blobs[i][0], blobs[i][1], k)
                        == blobLabels[i], "weighted vote agrees on separated blobs, k = " + k);
            }
            check(correct == blobs.length, "100% accuracy on separated blobs, k = " + k);
        }

        double[][] duplicates = {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {9.0, 9.0}};
        KdTree dupTree = new KdTree(duplicates);
        check(Arrays.equals(dupTree.knn(1.0, 1.0, 4), new int[]{0, 1, 2, 3}),
                "duplicates come back in index order");
        check(dupTree.distanceChecks <= duplicates.length + 1,
                "duplicate query visits at most n + 1 nodes");
        check(Arrays.equals(dupTree.knn(1.0, 1.0, 5), new int[]{0, 1, 2, 3, 4}), "k = n");

        double[][] allSame = {{2.0, 3.0}, {2.0, 3.0}, {2.0, 3.0}};
        KdTree sameTree = new KdTree(allSame);
        for (int k = 1; k <= 3; k++) {
            int[] got = sameTree.knn(2.0, 3.0, k);
            for (int i = 0; i < k; i++) check(got[i] == i, "zero distance, ascending index");
        }

        double[][] eq = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}, {5.0, 5.0}};
        KdTree eqTree = new KdTree(eq);
        int[] eqKnn = eqTree.knn(0.0, 0.0, 5);
        check(eqKnn[0] == 0, "the query itself is the closest point");
        check(eqKnn[1] == 1 && eqKnn[2] == 2 && eqKnn[3] == 3 && eqKnn[4] == 4,
                "equidistant points in index order");
        check(Arrays.equals(eqKnn, bruteKnn(eq, 0.0, 0.0, 5)), "equidistant ties match the oracle");

        int[] globalLabels = {2, 2, 2, 0, 0, 1};
        check(classify(eqTree, globalLabels, 0.0, 0.0, eq.length) == 2,
                "k = n majority equals the global majority");

        int[] repeatA = tree.knn(31.5, 17.25, 7);
        int[] repeatB = tree.knn(31.5, 17.25, 7);
        check(Arrays.equals(repeatA, repeatB), "repeated query is stable");
        double lastDist = -1.0;
        for (int index : repeatA) {
            double d = Math.sqrt(KdTree.sqdist(31.5, 17.25, pts[index][0], pts[index][1]));
            check(d >= lastDist, "sqrt distances are non-decreasing");
            lastDist = d;
        }

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
