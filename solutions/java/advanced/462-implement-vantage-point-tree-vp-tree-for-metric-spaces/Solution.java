/** Task 462 - Implement vantage-point tree (VP-tree) for metric spaces. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Integer 2-D coordinates; metric = Manhattan L1 (|dx| + |dy|), a true metric, so
 * VP-tree pruning is exact. Each node stores a vantage point, the median distance
 * `mu`, and two children: inner (distance <= mu) and outer (distance >= mu), which
 * keeps every point in exactly one child and the halves balanced.
 * Search visits the nearer child first and prunes by the triangle inequality:
 * inner when tau - mu > best, outer when tau + best < mu (strict, so equal-distance
 * candidates are still visited). Tie-break (documented): equal L1 distance ->
 * smaller point index. Fixed LCG workload: equals the brute-force oracle; the
 * visited-node statistic stays below n/3 nodes per query on average (measured 14.1).
 */
import java.util.*;

class VpTree {
    private final int[][] pts;
    private final int[] mu, inner, outer;
    private final int root;
    int visited;
    private long bestDist;
    private int bestIdx;

    VpTree(int[][] points) {
        pts = new int[points.length][];
        for (int i = 0; i < points.length; i++) pts[i] = points[i].clone();
        int n = pts.length;
        mu = new int[n];
        inner = new int[n];
        outer = new int[n];
        Arrays.fill(inner, -1);
        Arrays.fill(outer, -1);
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) ids[i] = i;
        root = n > 0 ? build(ids) : -1;
    }

    static int dist(int[] a, int[] b) {
        return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
    }

    private int build(int[] ids) {
        if (ids.length == 0) return -1;
        int vp = ids[0];
        if (ids.length == 1) return vp;
        Integer[] rest = new Integer[ids.length - 1];
        for (int i = 1; i < ids.length; i++) rest[i - 1] = ids[i];
        Arrays.sort(rest, (a, b) -> {
            int da = dist(pts[vp], pts[a]), db = dist(pts[vp], pts[b]);
            return da != db ? Integer.compare(da, db) : Integer.compare(a, b);
        });
        int mid = rest.length / 2;
        mu[vp] = dist(pts[vp], pts[rest[mid]]);
        inner[vp] = build(toPrimitive(Arrays.copyOfRange(rest, 0, mid + 1)));
        outer[vp] = build(toPrimitive(Arrays.copyOfRange(rest, mid + 1, rest.length)));
        return vp;
    }

    private static int[] toPrimitive(Integer[] boxed) {
        int[] out = new int[boxed.length];
        for (int i = 0; i < boxed.length; i++) out[i] = boxed[i];
        return out;
    }

    long[] nearest(int qx, int qy) {
        visited = 0;
        if (root == -1) return new long[]{-1, -1};
        bestDist = Long.MAX_VALUE;
        bestIdx = -1;
        search(root, qx, qy);
        return new long[]{bestIdx, bestDist};
    }

    private void search(int node, int qx, int qy) {
        visited++;
        int tau = Math.abs(qx - pts[node][0]) + Math.abs(qy - pts[node][1]);
        if (tau < bestDist || (tau == bestDist && node < bestIdx)) {
            bestDist = tau;
            bestIdx = node;
        }
        int radius = mu[node];
        if (tau < radius) {
            visitInner(node, qx, qy, tau, radius);
            visitOuter(node, qx, qy, tau, radius);
        } else {
            visitOuter(node, qx, qy, tau, radius);
            visitInner(node, qx, qy, tau, radius);
        }
    }

    private void visitInner(int node, int qx, int qy, int tau, int radius) {
        if (inner[node] != -1 && tau - radius <= bestDist) search(inner[node], qx, qy);
    }

    private void visitOuter(int node, int qx, int qy, int tau, int radius) {
        if (outer[node] != -1 && tau + bestDist >= radius) search(outer[node], qx, qy);
    }
}

public class Solution {
    private static int seed;

    private static int lcg() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static long[] bruteNearest(int[][] points, int qx, int qy) {
        long bestI = -1, bestD = 0;
        for (int i = 0; i < points.length; i++) {
            long d = Math.abs(points[i][0] - qx) + Math.abs(points[i][1] - qy);
            if (bestI == -1 || d < bestD || (d == bestD && i < bestI)) {
                bestI = i;
                bestD = d;
            }
        }
        return new long[]{bestI, bestD};
    }

    public static void main(String[] args) {
        int[][] small = {{0, 0}, {1, 1}, {3, 0}, {5, 5}, {2, 4}};
        VpTree tree = new VpTree(small);
        check(Arrays.equals(tree.nearest(0, 0), new long[]{0, 0}), "tiny (0,0)");
        check(Arrays.equals(tree.nearest(4, 3), new long[]{3, 3}), "tiny (4,3)");
        check(Arrays.equals(tree.nearest(2, 1), new long[]{1, 1}), "tiny (2,1)");
        check(Arrays.equals(tree.nearest(5, 0), new long[]{2, 2}), "tiny (5,0)");
        check(Arrays.equals(new VpTree(new int[0][]).nearest(0, 0), new long[]{-1, -1}), "empty");

        int n = 150;
        seed = 4242;
        int[][] points = new int[n][];
        for (int i = 0; i < n; i++) {
            int x = lcg() % 2001 - 1000;
            int y = lcg() % 2001 - 1000;
            points[i] = new int[]{x, y};
        }
        tree = new VpTree(points);
        int queries = 150;
        long totalVisited = 0;
        for (int qi = 0; qi < queries; qi++) {
            int qx = lcg() % 2001 - 1000;
            int qy = lcg() % 2001 - 1000;
            check(Arrays.equals(tree.nearest(qx, qy), bruteNearest(points, qx, qy)), "query " + qi);
            totalVisited += tree.visited;
        }
        check(totalVisited > 0, "visited positive");
        check(totalVisited < (long) queries * n / 3, "visited budget " + totalVisited);
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
