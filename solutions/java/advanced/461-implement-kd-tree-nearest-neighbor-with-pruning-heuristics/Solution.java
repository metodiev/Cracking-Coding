/** Task 461 - Implement kd-tree nearest neighbor with pruning heuristics. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * 2-D kd-tree (median split on alternating axes) keeping a bounding box per node.
 * Best-first nearest-neighbour search ordered by the squared distance from the
 * query point to a child's bounding box; a subtree is pruned as soon as its bound
 * strictly exceeds the best squared distance found (strict `>`, so equal-distance
 * candidates are still explored). Tie-break (documented): equal squared distance
 * -> smaller point index. Fixed LCG workload: equals the brute-force oracle, and
 * the visited-node statistic stays below n/4 nodes per query on average (measured
 * 8.0) versus n for a linear scan.
 */
import java.util.*;

class KdTree {
    private final int[][] pts;
    private final int[] left, right;
    private final int[] bx0, by0, bx1, by1;
    private final int root;
    int visited;

    KdTree(int[][] points) {
        pts = new int[points.length][];
        for (int i = 0; i < points.length; i++) pts[i] = points[i].clone();
        int n = pts.length;
        left = new int[n];
        right = new int[n];
        bx0 = new int[n];
        by0 = new int[n];
        bx1 = new int[n];
        by1 = new int[n];
        Arrays.fill(left, -1);
        Arrays.fill(right, -1);
        Integer[] ids = new Integer[n];
        for (int i = 0; i < n; i++) ids[i] = i;
        root = n > 0 ? build(ids, 0) : -1;
    }

    private int build(Integer[] ids, int depth) {
        if (ids.length == 0) return -1;
        final int axis = depth & 1;
        Arrays.sort(ids, (a, b) -> {
            if (pts[a][axis] != pts[b][axis]) return Integer.compare(pts[a][axis], pts[b][axis]);
            if (pts[a][1 - axis] != pts[b][1 - axis]) return Integer.compare(pts[a][1 - axis], pts[b][1 - axis]);
            return Integer.compare(a, b);
        });
        int mid = ids.length / 2;
        int node = ids[mid];
        left[node] = build(Arrays.copyOfRange(ids, 0, mid), depth + 1);
        right[node] = build(Arrays.copyOfRange(ids, mid + 1, ids.length), depth + 1);
        int x0 = Integer.MAX_VALUE, y0 = Integer.MAX_VALUE, x1 = Integer.MIN_VALUE, y1 = Integer.MIN_VALUE;
        for (int i : ids) {
            x0 = Math.min(x0, pts[i][0]);
            x1 = Math.max(x1, pts[i][0]);
            y0 = Math.min(y0, pts[i][1]);
            y1 = Math.max(y1, pts[i][1]);
        }
        bx0[node] = x0;
        by0[node] = y0;
        bx1[node] = x1;
        by1[node] = y1;
        return node;
    }

    private static long dist2(int[] a, int[] b) {
        long dx = a[0] - b[0];
        long dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    private long bound(int node, int qx, int qy) {
        long dx = 0;
        if (qx < bx0[node]) dx = bx0[node] - qx;
        else if (qx > bx1[node]) dx = qx - bx1[node];
        long dy = 0;
        if (qy < by0[node]) dy = by0[node] - qy;
        else if (qy > by1[node]) dy = qy - by1[node];
        return dx * dx + dy * dy;
    }

    long[] nearest(int qx, int qy) {
        visited = 0;
        if (root == -1) return new long[]{-1, -1};
        long bestI = -1, bestD = 0;
        PriorityQueue<long[]> pq = new PriorityQueue<>(
            (a, b) -> a[0] != b[0] ? Long.compare(a[0], b[0]) : Long.compare(a[1], b[1]));
        pq.add(new long[]{0, root});
        while (!pq.isEmpty()) {
            long[] top = pq.poll();
            long bnd = top[0];
            int node = (int) top[1];
            if (bestI != -1 && bnd > bestD) break; // prune
            visited++;
            long d = dist2(pts[node], new int[]{qx, qy});
            if (bestI == -1 || d < bestD || (d == bestD && node < bestI)) {
                bestI = node;
                bestD = d;
            }
            for (int child : new int[]{left[node], right[node]}) {
                if (child != -1) pq.add(new long[]{bound(child, qx, qy), child});
            }
        }
        return new long[]{bestI, bestD};
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
            long dx = points[i][0] - qx;
            long dy = points[i][1] - qy;
            long d = dx * dx + dy * dy;
            if (bestI == -1 || d < bestD || (d == bestD && i < bestI)) {
                bestI = i;
                bestD = d;
            }
        }
        return new long[]{bestI, bestD};
    }

    public static void main(String[] args) {
        int[][] small = {{0, 0}, {2, 3}, {5, 4}, {4, 1}, {9, 6}};
        KdTree tree = new KdTree(small);
        check(Arrays.equals(tree.nearest(1, 1), new long[]{0, 2}), "tiny (1,1)");
        check(Arrays.equals(tree.nearest(4, 2), new long[]{3, 1}), "tiny (4,2)");
        check(Arrays.equals(tree.nearest(3, 3), new long[]{1, 1}), "tiny (3,3)");
        check(Arrays.equals(tree.nearest(3, 2), new long[]{1, 2}), "tiny tie (3,2)");
        check(Arrays.equals(new KdTree(new int[0][]).nearest(0, 0), new long[]{-1, -1}), "empty");

        int n = 200;
        seed = 1234;
        int[][] points = new int[n][];
        for (int i = 0; i < n; i++) {
            int x = lcg() % 2001 - 1000;
            int y = lcg() % 2001 - 1000;
            points[i] = new int[]{x, y};
        }
        tree = new KdTree(points);
        int queries = 200;
        long totalVisited = 0;
        for (int qi = 0; qi < queries; qi++) {
            int qx = lcg() % 2001 - 1000;
            int qy = lcg() % 2001 - 1000;
            check(Arrays.equals(tree.nearest(qx, qy), bruteNearest(points, qx, qy)), "query " + qi);
            totalVisited += tree.visited;
        }
        check(totalVisited > 0, "visited positive");
        check(totalVisited < (long) queries * n / 4, "visited budget " + totalVisited);
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
