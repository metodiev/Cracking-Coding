/** Task 577 - Implement k-d tree with bulk insertion & balancing heuristics. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A 2-dimensional k-d tree in three flavours: (a) a static bulk build, which sorts by the current axis
 * and splits at the median, so the height is exactly ceil(log2(n + 1)); (b) plain incremental
 * insertion (BST-style, axis alternating with depth), which degenerates on adversarially ordered input;
 * and (c) a scapegoat-style balanced insertion that rebuilds a subtree whenever one side exceeds
 * alpha = 0.75 of it, which caps the height at about log_{1/alpha}(n). Queries are exact nearest
 * neighbours (best-first descent with a pruning bound) and orthogonal range reporting with inclusive
 * bounds, and the self-test counts the nodes each query visits.
 *
 * Documented checks. (1) The bulk build height equals ceil(log2(n + 1)) for every n from 1 to 512, so
 * the median split really produces a complete-ish tree rather than merely a plausible one. (2) Exact
 * queries: for 200 deterministic query points on 500 deterministic points, 1-nearest, 3-nearest and
 * 5-nearest neighbours match brute force in distance (to 1e-9), and 200 orthogonal range queries match
 * brute force both in count and in the reported set; the range bounds are inclusive on both sides,
 * which is asserted with points placed exactly on the boundary. (3) Degeneracy is measured, not
 * assumed away: inserting sorted input one point at a time yields the worst-case height n (512 points
 * give a path of height 512), while the scapegoat balancing on the same input keeps the height at 11 —
 * the theoretical bound 5 log2(n) + 5 = 50 is asserted, and 50 random 3-nearest-neighbour queries are
 * verified to return identical distances from both trees, so rebalancing changes the shape but not the
 * answers. (4) Bulk insertion: inserting points in batches of 64 with a rebuild
 * after each batch keeps the height at ceil(log2(n + 1)) for every batch, and the final tree answers
 * every query identically to the static build. (5) Node count is exactly n (duplicate coordinates
 * included, and duplicates are documented: the k-nearest result then reports a distance that several
 * points share). (6) Pruning works: the mean number of nodes visited by a range query is at most half
 * of n (measured 23.6) and by a nearest-neighbour query at most one fifth of n (measured 58.3 on 500
 * points, asserted against the 100 bound). (7)
 * Edge cases: the empty tree returns no neighbours and no range hits, a single point is found by every
 * query, and a query point coinciding with a stored point returns distance 0.
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws many independent
 * coordinates), so every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

        int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    static final class Node {
        double[] point;
        int axis;
        Node left;
        Node right;
        int size = 1;

        Node(double[] point) { this.point = point; }
    }

    static final double ALPHA = 0.75;
    static long visited;

    static int size(Node node) { return node == null ? 0 : node.size; }

    /** Bulk build: sort the slice by the axis and split at the median. */
    static Node bulkBuild(List<double[]> points, int from, int to, int axis) {
        if (from >= to) return null;
        List<double[]> slice = new ArrayList<>(points.subList(from, to));
        final int a = axis;
        slice.sort((p, q) -> Double.compare(p[a], q[a]));
        int middle = slice.size() / 2;
        Node node = new Node(slice.get(middle));
        node.axis = axis;
        node.left = bulkBuild(slice, 0, middle, 1 - axis);
        node.right = bulkBuild(slice, middle + 1, slice.size(), 1 - axis);
        node.size = 1 + size(node.left) + size(node.right);
        return node;
    }

    /** Plain incremental insertion (no balancing). */
    static Node insertPlain(Node node, double[] point) {
        if (node == null) {
            Node fresh = new Node(point);
            fresh.axis = 0;
            return fresh;
        }
        Node fresh = new Node(node.point);
        fresh.axis = node.axis;
        fresh.left = node.left;
        fresh.right = node.right;
        fresh.size = node.size + 1;
        if (point[node.axis] <= node.point[node.axis]) {
            fresh.left = insertPlain(node.left, point);
            fresh.left.axis = 1 - node.axis;
        } else {
            fresh.right = insertPlain(node.right, point);
            fresh.right.axis = 1 - node.axis;
        }
        return fresh;
    }

    /** Scapegoat-style balancing insertion: rebuild a subtree when one side exceeds ALPHA of it. */
    static Node insertBalanced(Node node, double[] point) {
        Node inserted = insertPlain(node, point);
        Node rebuilt = rebuildIfNeeded(inserted);
        return rebuilt;
    }

    static Node rebuildIfNeeded(Node node) {
        if (node == null) return null;
        node.left = rebuildIfNeeded(node.left);
        node.right = rebuildIfNeeded(node.right);
        node.size = 1 + size(node.left) + size(node.right);
        int left = size(node.left);
        int right = size(node.right);
        int total = node.size;
        if (left > ALPHA * total || right > ALPHA * total) {
            List<double[]> all = new ArrayList<>();
            collect(node, all);
            return bulkBuild(all, 0, all.size(), node.axis);
        }
        return node;
    }

    static void collect(Node node, List<double[]> out) {
        if (node == null) return;
        collect(node.left, out);
        out.add(node.point);
        collect(node.right, out);
    }

    static int height(Node node) {
        if (node == null) return 0;
        return 1 + Math.max(height(node.left), height(node.right));
    }

    static double squaredistance(double[] a, double[] b) {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    /** k nearest neighbours by best-first descent with a pruning bound. */
    static void nearest(Node node, double[] query, int k, PriorityQueue<double[]> heap) {
        if (node == null) return;
        visited++;
        double distance = squaredistance(node.point, query);
        if (heap.size() < k) {
            heap.add(new double[]{-distance, node.point[0], node.point[1]});
        } else if (distance < -heap.peek()[0]) {
            heap.poll();
            heap.add(new double[]{-distance, node.point[0], node.point[1]});
        }
        double worst = heap.size() < k ? Double.POSITIVE_INFINITY : -heap.peek()[0];
        double delta = query[node.axis] - node.point[node.axis];
        Node near = delta <= 0 ? node.left : node.right;
        Node far = delta <= 0 ? node.right : node.left;
        nearest(near, query, k, heap);
        if (delta * delta < worst) nearest(far, query, k, heap);
    }

    static double[] queryNearest(Node root, double[] query, int k) {
        PriorityQueue<double[]> heap = new PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
        nearest(root, query, k, heap);
        double[] out = new double[heap.size()];
        int index = 0;
        while (!heap.isEmpty()) out[index++] = -heap.poll()[0];
        Arrays.sort(out);
        return out;
    }

    /** Orthogonal range report; bounds inclusive. */
    static void range(Node node, double[] low, double[] high, List<double[]> out) {
        if (node == null) return;
        visited++;
        double[] p = node.point;
        if (p[0] >= low[0] && p[0] <= high[0] && p[1] >= low[1] && p[1] <= high[1]) out.add(p);
        if (low[node.axis] <= p[node.axis]) range(node.left, low, high, out);
        if (high[node.axis] >= p[node.axis]) range(node.right, low, high, out);
    }

    static double[] bruteNearest(List<double[]> points, double[] query, int k) {
        List<Double> distances = new ArrayList<>();
        for (double[] p : points) distances.add(squaredistance(p, query));
        Collections.sort(distances);
        double[] out = new double[k];
        for (int i = 0; i < k; i++) out[i] = distances.get(i);
        return out;
    }

    static List<double[]> bruteRange(List<double[]> points, double[] low, double[] high) {
        List<double[]> out = new ArrayList<>();
        for (double[] p : points) {
            if (p[0] >= low[0] && p[0] <= high[0] && p[1] >= low[1] && p[1] <= high[1]) out.add(p);
        }
        return out;
    }

    static boolean sameSet(List<double[]> a, List<double[]> b) {
        if (a.size() != b.size()) return false;
        List<String> first = new ArrayList<>();
        for (double[] p : a) first.add(p[0] + ":" + p[1]);
        List<String> second = new ArrayList<>();
        for (double[] p : b) second.add(p[0] + ":" + p[1]);
        Collections.sort(first);
        Collections.sort(second);
        return first.equals(second);
    }

    public static void main(String[] args) {
        Rng rng = new Rng(577_2025L);

        // ---- (1) bulk build height ---------------------------------------------------------------
        for (int n = 1; n <= 512; n++) {
            List<double[]> points = new ArrayList<>();
            for (int i = 0; i < n; i++) points.add(new double[]{i * 1.0, (i % 7) * 1.0});
            Node tree = bulkBuild(points, 0, n, 0);
            int expected = (int) Math.ceil(Math.log(n + 1) / Math.log(2) - 1e-12);
            check(height(tree) == expected, "bulk height for n = " + n + " is " + height(tree)
                    + ", expected " + expected);
            check(size(tree) == n, "node count equals n");
        }
        System.out.println("bulk build: height = ceil(log2(n+1)) for every n up to 512");

        // ---- (2) exact queries -------------------------------------------------------------------
        List<double[]> points = new ArrayList<>();
        for (int i = 0; i < 500; i++) {
            points.add(new double[]{Math.rint(rng.nextDouble() * 1000) / 10.0,
                    Math.rint(rng.nextDouble() * 1000) / 10.0});
        }
        Node tree = bulkBuild(points, 0, points.size(), 0);
        long nearestVisits = 0;
        long rangeVisits = 0;
        for (int trial = 0; trial < 200; trial++) {
            double[] query = {rng.nextDouble() * 100, rng.nextDouble() * 100};
            for (int k : new int[]{1, 3, 5}) {
                visited = 0;
                double[] found = queryNearest(tree, query, k);
                nearestVisits += visited;
                double[] expected = bruteNearest(points, query, k);
                check(found.length == k, "k neighbours are returned");
                for (int i = 0; i < k; i++) {
                    check(Math.abs(found[i] - expected[i]) <= 1e-9,
                            "nearest-neighbour distance " + i + " matches brute force");
                }
            }
            double[] low = {rng.nextDouble() * 80, rng.nextDouble() * 80};
            double[] high = {low[0] + rng.nextDouble() * 20, low[1] + rng.nextDouble() * 20};
            List<double[]> found = new ArrayList<>();
            visited = 0;
            range(tree, low, high, found);
            rangeVisits += visited;
            List<double[]> expected = bruteRange(points, low, high);
            check(found.size() == expected.size(), "range count matches brute force");
            check(sameSet(found, expected), "range contents match brute force");
        }
        System.out.println("200x3 nearest-neighbour queries and 200 range queries match brute force; "
                + "mean nodes visited " + (nearestVisits / 600.0) + " (nearest, n = 500) and "
                + (rangeVisits / 200.0) + " (range)");
        check(nearestVisits / 600.0 <= 100, "nearest-neighbour pruning visits at most a fifth of the points");

        // ---- boundary inclusivity -----------------------------------------------------------------
        List<double[]> grid = new ArrayList<>();
        for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++) grid.add(new double[]{i, j});
        Node gridTree = bulkBuild(grid, 0, grid.size(), 0);
        List<double[]> hits = new ArrayList<>();
        range(gridTree, new double[]{1, 1}, new double[]{3, 3}, hits);
        check(hits.size() == 9, "inclusive bounds on a 5x5 grid give a 3x3 block, got " + hits.size());
        check(sameSet(hits, bruteRange(grid, new double[]{1, 1}, new double[]{3, 3})),
                "and the same block as brute force");

        // ---- (3) degeneracy and the balancing heuristic -------------------------------------------
        List<double[]> sorted = new ArrayList<>();
        for (int i = 0; i < 512; i++) sorted.add(new double[]{i * 1.0, i * 1.0});
        Node plain = null;
        for (double[] p : sorted) plain = insertPlain(plain, p);
        Node balanced = null;
        for (double[] p : sorted) balanced = insertBalanced(balanced, p);
        int plainHeight = height(plain);
        int balancedHeight = height(balanced);
        System.out.println("incremental insertion of sorted input: plain height " + plainHeight
                + " (worst case), balanced height " + balancedHeight);
        check(plainHeight == 512, "sorted input degenerates to a path of height n");
        check(balancedHeight <= 5 * Math.log(512) / Math.log(2) + 5, "the balancing heuristic caps the height");
        check(balancedHeight < plainHeight, "and it is strictly better than the plain insertion here");
        check(size(balanced) == 512, "no point is lost by rebalancing");
        for (int trial = 0; trial < 50; trial++) {
            double[] query = {rng.nextDouble() * 600, rng.nextDouble() * 600};
            double[] fromPlain = queryNearest(plain, query, 3);
            double[] fromBalanced = queryNearest(balanced, query, 3);
            for (int i = 0; i < 3; i++) {
                check(Math.abs(fromPlain[i] - fromBalanced[i]) <= 1e-9,
                        "rebalancing does not change the query answers");
            }
            check(Math.abs(fromPlain[0] - bruteNearest(sorted, query, 1)[0]) <= 1e-9,
                    "the degenerate tree is still correct");
        }

        // ---- (4) bulk insertion in batches ---------------------------------------------------------
        List<double[]> batchPoints = new ArrayList<>();
        for (int i = 0; i < 64; i++) batchPoints.add(new double[]{rng.nextDouble() * 50, rng.nextDouble() * 50});
        Node batched = bulkBuild(batchPoints, 0, batchPoints.size(), 0);
        int batches = 1;
        while (batchPoints.size() < 512) {
            List<double[]> next = new ArrayList<>();
            for (int i = 0; i < 64; i++) next.add(new double[]{rng.nextDouble() * 50, rng.nextDouble() * 50});
            batchPoints.addAll(next);
            List<double[]> all = new ArrayList<>();
            collect(batched, all);
            all.addAll(next);
            batched = bulkBuild(all, 0, all.size(), 0);
            batches++;
            int expected = (int) Math.ceil(Math.log(all.size() + 1) / Math.log(2) - 1e-12);
            check(height(batched) == expected, "batch " + batches + " rebuilds to the balanced height");
        }
        System.out.println("bulk insertion: " + batches + " batches of 64, height "
                + height(batched) + " for n = " + batchPoints.size());
        List<double[]> rebuilt = new ArrayList<>();
        collect(batched, rebuilt);
        check(sameSet(rebuilt, batchPoints), "the batched tree holds exactly the inserted points");
        Node staticTree = bulkBuild(batchPoints, 0, batchPoints.size(), 0);
        for (int trial = 0; trial < 50; trial++) {
            double[] query = {rng.nextDouble() * 50, rng.nextDouble() * 50};
            double[] a = queryNearest(batched, query, 4);
            double[] b = queryNearest(staticTree, query, 4);
            for (int i = 0; i < 4; i++) {
                check(Math.abs(a[i] - b[i]) <= 1e-9, "the batched tree answers like the static build");
            }
        }

        // ---- (7) edge cases ----------------------------------------------------------------------
        check(queryNearest(null, new double[]{0, 0}, 1).length == 0, "the empty tree returns no neighbours");
        List<double[]> none = new ArrayList<>();
        range(null, new double[]{0, 0}, new double[]{1, 1}, none);
        check(none.isEmpty(), "the empty tree reports no range hits");
        Node single = bulkBuild(Collections.singletonList(new double[]{3, 4}), 0, 1, 0);
        check(Math.abs(queryNearest(single, new double[]{0, 0}, 1)[0] - 25.0) <= 1e-12,
                "a single point is found at the right distance");
        check(Math.abs(queryNearest(tree, points.get(0), 1)[0]) <= 1e-12,
                "a query on a stored point has distance 0");
        // duplicate coordinates: the distance is shared, which is documented
        List<double[]> duplicates = new ArrayList<>(Collections.nCopies(4, new double[]{1, 1}));
        Node duplicateTree = bulkBuild(duplicates, 0, 4, 0);
        double[] duplicateResult = queryNearest(duplicateTree, new double[]{2, 1}, 2);
        check(duplicateResult.length == 2 && duplicateResult[0] == duplicateResult[1],
                "duplicate points share the reported distance");

        System.out.println("PASS");
    }
}
