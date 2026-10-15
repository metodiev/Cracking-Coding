/** Task 329 - Range intersection queries (interval trees basics). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
    private int seed;

    Lcg(int seed) {
        this.seed = seed;
    }

    int next(int modulus) {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

/** Augmented BST over interval starts with subtree max-end values. */
class IntervalTree {
    static class Node {
        int start, end, maxEnd;
        Node left, right;

        Node(int start, int end) {
            this.start = start;
            this.end = end;
            this.maxEnd = end;
        }
    }

    private final Node root;

    IntervalTree(int[][] intervals) {
        int[][] ordered = new int[intervals.length][];
        for (int i = 0; i < intervals.length; i++) ordered[i] = intervals[i].clone();
        Arrays.sort(ordered, (u, v) -> u[0] != v[0] ? Integer.compare(u[0], v[0]) : Integer.compare(u[1], v[1]));
        root = build(ordered, 0, ordered.length);
    }

    private static Node build(int[][] ordered, int from, int to) {
        if (from >= to) return null;
        int mid = (from + to) / 2;
        Node node = new Node(ordered[mid][0], ordered[mid][1]);
        node.left = build(ordered, from, mid);
        node.right = build(ordered, mid + 1, to);
        node.maxEnd = node.end;
        if (node.left != null) node.maxEnd = Math.max(node.maxEnd, node.left.maxEnd);
        if (node.right != null) node.maxEnd = Math.max(node.maxEnd, node.right.maxEnd);
        return node;
    }

    /** All stored intervals intersecting the closed range [lo, hi], in start order. */
    int[][] query(int lo, int hi) {
        List<int[]> found = new ArrayList<>();
        query(root, lo, hi, found);
        return found.toArray(new int[0][]);
    }

    private static void query(Node node, int lo, int hi, List<int[]> found) {
        if (node == null) return;
        if (node.left != null && node.left.maxEnd >= lo) query(node.left, lo, hi, found);
        if (node.start <= hi && node.end >= lo) found.add(new int[]{node.start, node.end});
        if (node.start <= hi) query(node.right, lo, hi, found);
    }
}

public class Solution {
    /** Independent oracle: filter every interval. */
    static int[][] naiveQuery(int[][] intervals, int lo, int hi) {
        List<int[]> found = new ArrayList<>();
        for (int[] interval : intervals) {
            if (interval[0] <= hi && interval[1] >= lo) found.add(interval.clone());
        }
        return sortedPoints(found.toArray(new int[0][]));
    }

    static int[][] sortedPoints(int[][] points) {
        int[][] out = new int[points.length][];
        for (int i = 0; i < points.length; i++) out[i] = points[i].clone();
        Arrays.sort(out, (u, v) -> u[0] != v[0] ? Integer.compare(u[0], v[0]) : Integer.compare(u[1], v[1]));
        return out;
    }

    static int[][] lcgIntervals(int count, Lcg lcg) {
        int[][] intervals = new int[count][];
        for (int i = 0; i < count; i++) {
            int start = lcg.next(60);
            intervals[i] = new int[]{start, start + 1 + lcg.next(12)};
        }
        return intervals;
    }

    static final int[][] FIXED = {{15, 20}, {10, 30}, {17, 19}, {5, 20}, {12, 15}, {30, 40}};

    public static void main(String[] args) {
        IntervalTree tree = new IntervalTree(FIXED);
        check(Arrays.deepEquals(tree.query(6, 7), new int[][]{{5, 20}}), "query [6,7]");
        check(Arrays.deepEquals(tree.query(14, 16),
                new int[][]{{5, 20}, {10, 30}, {12, 15}, {15, 20}}), "query [14,16]");
        check(Arrays.deepEquals(tree.query(30, 35), new int[][]{{10, 30}, {30, 40}}), "query [30,35]");
        check(Arrays.deepEquals(tree.query(19, 19),
                new int[][]{{5, 20}, {10, 30}, {15, 20}, {17, 19}}), "query [19,19]");
        check(tree.query(0, 4).length == 0, "query below everything");
        check(tree.query(41, 50).length == 0, "query above everything");

        check(new IntervalTree(new int[0][]).query(0, 10).length == 0, "empty tree");
        check(Arrays.deepEquals(new IntervalTree(new int[][]{{4, 4}}).query(4, 4),
                new int[][]{{4, 4}}), "point interval");

        Lcg lcg = new Lcg(7071);
        for (int trial = 0; trial < 40; trial++) {
            int[][] intervals = lcgIntervals(30, lcg);
            IntervalTree fromBuild = new IntervalTree(intervals);
            for (int q = 0; q < 30; q++) {
                int lo = lcg.next(70);
                int hi = lo + lcg.next(25);
                check(Arrays.deepEquals(sortedPoints(fromBuild.query(lo, hi)), naiveQuery(intervals, lo, hi)),
                        "random query cross-check " + trial + "/" + q);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
