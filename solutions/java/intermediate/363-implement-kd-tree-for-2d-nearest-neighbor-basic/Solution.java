/** Task 363 - Implement kd-tree for 2D nearest neighbor (basic). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class KdNode {
    final int[] point;
    final int index;
    final int axis;
    final KdNode left;
    final KdNode right;

    KdNode(int[] point, int index, int axis, KdNode left, KdNode right) {
        this.point = point;
        this.index = index;
        this.axis = axis;
        this.left = left;
        this.right = right;
    }
}

class KdTree {
    // Axis alternates with depth and each node takes the median of its points sorted by
    // that axis (input index breaks ties), so the tree is balanced and deterministic.
    private final int[][] points;
    private final KdNode root;

    KdTree(int[][] rawPoints) {
        points = rawPoints.clone();
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < points.length; i++) ids.add(i);
        root = build(ids, 0);
    }

    private KdNode build(List<Integer> ids, int depth) {
        if (ids.isEmpty()) return null;
        int axis = depth % 2;
        ids.sort((i, j) -> points[i][axis] != points[j][axis]
                ? Integer.compare(points[i][axis], points[j][axis])
                : Integer.compare(i, j));
        int mid = ids.size() / 2;
        int index = ids.get(mid);
        KdNode left = build(new ArrayList<>(ids.subList(0, mid)), depth + 1);
        KdNode right = build(new ArrayList<>(ids.subList(mid + 1, ids.size())), depth + 1);
        return new KdNode(points[index].clone(), index, axis, left, right);
    }

    /** Returns {squared distance, input index} of the nearest stored point. */
    long[] nearest(int[] query) {
        if (root == null) throw new IllegalStateException("empty kd-tree");
        return search(root, query, null);
    }

    private long[] search(KdNode node, int[] query, long[] best) {
        if (node == null) return best;
        long dx = query[0] - node.point[0];
        long dy = query[1] - node.point[1];
        long distance = dx * dx + dy * dy;
        if (best == null || distance < best[0] ||
                (distance == best[0] && node.index < best[1])) {
            best = new long[]{distance, node.index};
        }
        long offset = query[node.axis] - node.point[node.axis];
        KdNode near = offset <= 0 ? node.left : node.right;
        KdNode far = offset <= 0 ? node.right : node.left;
        best = search(near, query, best);
        // `<=` also explores the far side on ties so an equal-distance point with a
        // smaller input index can still win, exactly matching brute force.
        if (offset * offset <= best[0]) best = search(far, query, best);
        return best;
    }
}

public class Solution {
    private static int seed = 60601;

    private static long[] bruteNearest(int[][] points, int[] query) {
        long[] best = null;
        for (int index = 0; index < points.length; index++) {
            long dx = query[0] - points[index][0];
            long dy = query[1] - points[index][1];
            long distance = dx * dx + dy * dy;
            if (best == null || distance < best[0] ||
                    (distance == best[0] && index < best[1])) {
                best = new long[]{distance, index};
            }
        }
        return best;
    }

    private static void sameAsBrute(KdTree tree, int[][] points, int[] query, String label) {
        long[] got = tree.nearest(query);
        long[] want = bruteNearest(points, query);
        check(got[0] == want[0] && got[1] == want[1],
                label + " got=(" + got[0] + "," + got[1] + ") want=(" + want[0] + "," + want[1] + ")");
    }

    public static void main(String[] args) {
        int[][] hand = {{0, 0}, {3, 4}, {1, 1}, {-5, -5}};
        KdTree tree = new KdTree(hand);
        long[] result = tree.nearest(new int[]{3, 3});
        check(result[0] == 1 && result[1] == 1, "hand (3,3) -> (3,4)");
        result = tree.nearest(new int[]{1, 1});
        check(result[0] == 0 && result[1] == 2, "query is a stored point");
        result = tree.nearest(new int[]{0, 0});
        check(result[0] == 0 && result[1] == 0, "query is the first point");
        result = tree.nearest(new int[]{-4, -6});
        check(result[0] == 2 && result[1] == 3, "hand (-4,-6) -> (-5,-5)");
        result = tree.nearest(new int[]{7, 7});
        check(result[0] == 25 && result[1] == 1, "hand (7,7) -> (3,4)");
        for (int[] query : new int[][]{new int[]{3, 3}, new int[]{1, 1}, new int[]{-4, -6},
                new int[]{7, 7}, new int[]{-9, 4}}) {
            sameAsBrute(tree, hand, query, "hand");
        }

        // Deterministic tie-break: four points at squared distance 2 from (1,1); index 0 wins.
        int[][] tie = {{0, 0}, {2, 0}, {0, 2}, {2, 2}};
        KdTree tieTree = new KdTree(tie);
        result = tieTree.nearest(new int[]{1, 1});
        check(result[0] == 2 && result[1] == 0, "tie-break picks smallest index");
        sameAsBrute(tieTree, tie, new int[]{1, 1}, "tie set");

        // Duplicate points: the earliest input index among equal points must win.
        int[][] dup = {{5, 5}, {5, 5}, {1, 1}};
        KdTree dupTree = new KdTree(dup);
        result = dupTree.nearest(new int[]{4, 4});
        check(result[0] == 2 && result[1] == 0, "duplicate tie-break");
        result = dupTree.nearest(new int[]{1, 1});
        check(result[0] == 0 && result[1] == 2, "duplicate query");

        KdTree single = new KdTree(new int[][]{{-2, 7}});
        result = single.nearest(new int[]{0, 0});
        check(result[0] == 53 && result[1] == 0, "single point 4 + 49");

        // LCG cross-check: many sets, queries both random and drawn from the set itself.
        for (int trial = 0; trial < 40; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int size = 1 + seed % 40;
            int[][] points = new int[size][];
            for (int i = 0; i < size; i++) {
                seed = (seed * 75 + 74) % 65537;
                int x = seed % 101 - 50;
                seed = (seed * 75 + 74) % 65537;
                int y = seed % 101 - 50;
                points[i] = new int[]{x, y};
            }
            KdTree randomTree = new KdTree(points);
            for (int[] point : points) {
                sameAsBrute(randomTree, points, point, "in-set query trial " + trial);
            }
            for (int q = 0; q < 12; q++) {
                seed = (seed * 75 + 74) % 65537;
                int x = seed % 121 - 60;
                seed = (seed * 75 + 74) % 65537;
                int y = seed % 121 - 60;
                sameAsBrute(randomTree, points, new int[]{x, y}, "random query trial " + trial);
            }
        }

        // Duplicate-heavy sets (coordinates in a tiny window) agree with brute force too.
        for (int trial = 0; trial < 10; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int size = 2 + seed % 30;
            int[][] points = new int[size][];
            for (int i = 0; i < size; i++) {
                seed = (seed * 75 + 74) % 65537;
                int x = seed % 4;
                seed = (seed * 75 + 74) % 65537;
                int y = seed % 4;
                points[i] = new int[]{x, y};
            }
            KdTree randomTree = new KdTree(points);
            int[][] queries = {{0, 0}, {1, 1}, {3, 3}, {2, 1}, {1, 2}};
            for (int[] query : queries) {
                sameAsBrute(randomTree, points, query, "dense trial " + trial);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
