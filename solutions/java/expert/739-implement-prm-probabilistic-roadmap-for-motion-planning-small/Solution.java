/** Task 739 - Implement PRM (probabilistic roadmap) for motion planning (small).
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The configuration space is the integer lattice [0,w] x [0,h]; obstacles are closed integer
 * rectangles, so point-blocked and segment-blocked tests are exact integer predicates. The roadmap
 * is built the classic way:
 *   1. sample N free lattice points uniformly from a fixed-seed splitmix64 stream (duplicates and
 *      blocked samples are discarded, and they are counted so the sampling is auditable);
 *   2. connect every node - samples plus the start and the goal - to its k nearest neighbours whose
 *      connecting segment is collision free, giving an undirected graph with Euclidean edge weights;
 *   3. answer the query with Dijkstra on that graph and read the waypoint chain back from the
 *      parent array.
 *
 * Simplification (documented). "Small" is taken literally: N is O(10^2) and k is 6-8, so the
 * nearest-neighbour step is a brute-force sort instead of a kd-tree, the graph is stored as an edge
 * list with a dense distance matrix for the oracle, and there is no cycle-closing / obstacle-based
 * sampling, no roadmap reuse across queries and no path smoothing. Collision checks are exact
 * because the workspace is a lattice, and the planner is fully deterministic.
 *
 * Verified. The roadmap invariants are re-derived from scratch: every stored edge is collision free,
 * no duplicate edges, no self loops; the returned path is continuous, starts at the start, ends at
 * the goal, and consists only of roadmap edges. Dijkstra's answer is cross-checked against an
 * independent Floyd-Warshall closure of the same graph (the path length must equal both), and the
 * path length is bounded below by the straight-line distance and above by the exact 8-connected
 * lattice optimum computed by an integer Dijkstra. Unreachable instances (sealed room, blocked
 * start) must return null, which is cross-checked with an independent 4-connected BFS.
 */
import java.util.*;

/** Probabilistic roadmap over an integer lattice inside closed rectangular obstacles. */
final class Prm {
    final int width;
    final int height;
    final int[][] obstacles;
    final List<int[]> nodes = new ArrayList<>();       // samples, then start, then goal
    final List<int[]> edges = new ArrayList<>();       // {i, j}, i < j
    final List<Double> lengths = new ArrayList<>();    // parallel to edges
    int startIndex = -1;
    int goalIndex = -1;
    int discardedSamples;
    List<int[]> path;                                  // filled by solve()

    Prm(int width, int height, int[][] obstacles) {
        this.width = width;
        this.height = height;
        this.obstacles = obstacles;
    }

    boolean blocked(int x, int y) {
        if (x < 0 || y < 0 || x > width || y > height) return true;
        for (int[] r : obstacles) {
            if (x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3]) return true;
        }
        return false;
    }

    boolean segmentFree(int ax, int ay, int bx, int by) {
        for (int[] r : obstacles) {
            if (segmentHitsRect(ax, ay, bx, by, r)) return false;
        }
        return true;
    }

    static boolean segmentHitsRect(int ax, int ay, int bx, int by, int[] r) {
        if (contains(ax, ay, r) || contains(bx, by, r)) return true;
        int[] cx = {r[0], r[2], r[2], r[0]};
        int[] cy = {r[1], r[1], r[3], r[3]};
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            if (segmentsIntersect(ax, ay, bx, by, cx[i], cy[i], cx[j], cy[j])) return true;
        }
        return false;
    }

    static boolean contains(int x, int y, int[] r) {
        return x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3];
    }

    static long orient(long ax, long ay, long bx, long by, long cx, long cy) {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    }

    static boolean segmentsIntersect(int ax, int ay, int bx, int by, int cx, int cy, int dx, int dy) {
        long d1 = orient(cx, cy, dx, dy, ax, ay);
        long d2 = orient(cx, cy, dx, dy, bx, by);
        long d3 = orient(ax, ay, bx, by, cx, cy);
        long d4 = orient(ax, ay, bx, by, dx, dy);
        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
                && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
            return true;
        }
        return (d1 == 0 && onSegment(cx, cy, dx, dy, ax, ay))
                || (d2 == 0 && onSegment(cx, cy, dx, dy, bx, by))
                || (d3 == 0 && onSegment(ax, ay, bx, by, cx, cy))
                || (d4 == 0 && onSegment(ax, ay, bx, by, dx, dy));
    }

    private static boolean onSegment(int ax, int ay, int bx, int by, int px, int py) {
        return Math.min(ax, bx) <= px && px <= Math.max(ax, bx)
                && Math.min(ay, by) <= py && py <= Math.max(ay, by);
    }

    static double distance(int[] a, int[] b) {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        return Math.sqrt(dx * dx + dy * dy);
    }

    /** Samples free lattice points and links each node to its k nearest connectable neighbours. */
    void buildRoadmap(int sampleCount, int k, int sx, int sy, int gx, int gy, long seed) {
        long state = seed;
        Set<Long> seen = new HashSet<>();
        for (int i = 0; i < sampleCount; i++) {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            long mixed = z ^ (z >>> 31);
            int x = (int) Math.floorMod(mixed, (long) (width + 1));
            state += 0x9E3779B97F4A7C15L;
            z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            long mixed2 = z ^ (z >>> 31);
            int y = (int) Math.floorMod(mixed2, (long) (height + 1));
            long key = (long) y * (width + 1) + x;
            if (blocked(x, y) || !seen.add(key)) {
                discardedSamples++;
                continue;
            }
            nodes.add(new int[]{x, y});
        }
        startIndex = nodes.size();
        nodes.add(new int[]{sx, sy});
        goalIndex = nodes.size();
        nodes.add(new int[]{gx, gy});
        Set<Long> edgeKeys = new HashSet<>();
        for (int i = 0; i < nodes.size(); i++) {
            for (int j : nearestCandidates(i, k)) {
                int a = Math.min(i, j);
                int b = Math.max(i, j);
                long key = (long) a * nodes.size() + b;
                if (edgeKeys.contains(key)) continue;
                if (!segmentFree(nodes.get(a)[0], nodes.get(a)[1], nodes.get(b)[0], nodes.get(b)[1])) {
                    continue;
                }
                edgeKeys.add(key);
                edges.add(new int[]{a, b});
                lengths.add(distance(nodes.get(a), nodes.get(b)));
            }
        }
    }

    /** Indices of the k nearest other nodes, ties broken by index so the result is reproducible. */
    private List<Integer> nearestCandidates(int i, int k) {
        final int[] self = nodes.get(i);
        Integer[] others = new Integer[nodes.size() - 1];
        int cursor = 0;
        for (int j = 0; j < nodes.size(); j++) if (j != i) others[cursor++] = j;
        Arrays.sort(others, (a, b) -> {
            int c = Double.compare(distance(self, nodes.get(a)), distance(self, nodes.get(b)));
            return c != 0 ? c : Integer.compare(a, b);
        });
        List<Integer> out = new ArrayList<>();
        for (int t = 0; t < Math.min(k, others.length); t++) out.add(others[t]);
        return out;
    }

    /** Dijkstra from start to goal; fills `path` and returns its length, or null when unreachable. */
    Double solve() {
        int[] startNode = nodes.get(startIndex);
        int[] goalNode = nodes.get(goalIndex);
        if (startNode[0] == goalNode[0] && startNode[1] == goalNode[1]) {
            path = List.of(startNode);
            return 0.0;
        }
        int n = nodes.size();
        double[] dist = new double[n];
        int[] parent = new int[n];
        Arrays.fill(dist, Double.POSITIVE_INFINITY);
        Arrays.fill(parent, -1);
        dist[startIndex] = 0;
        PriorityQueue<double[]> queue = new PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
        queue.add(new double[]{0, startIndex});
        while (!queue.isEmpty()) {
            double[] top = queue.poll();
            int u = (int) top[1];
            if (top[0] > dist[u]) continue;
            if (u == goalIndex) break;
            for (int e = 0; e < edges.size(); e++) {
                int[] edge = edges.get(e);
                int v = edge[0] == u ? edge[1] : edge[1] == u ? edge[0] : -1;
                if (v < 0) continue;
                double candidate = dist[u] + lengths.get(e);
                if (candidate < dist[v] - 1e-12) {
                    dist[v] = candidate;
                    parent[v] = u;
                    queue.add(new double[]{candidate, v});
                }
            }
        }
        if (!Double.isFinite(dist[goalIndex])) {
            path = null;
            return null;
        }
        LinkedList<int[]> chain = new LinkedList<>();
        for (int v = goalIndex; v >= 0; v = parent[v]) chain.addFirst(nodes.get(v));
        path = chain;
        return dist[goalIndex];
    }

    /** Independent oracle: dense Floyd-Warshall over the same edge list. */
    double[][] floydWarshall() {
        int n = nodes.size();
        double[][] d = new double[n][n];
        for (double[] row : d) Arrays.fill(row, Double.POSITIVE_INFINITY);
        for (int i = 0; i < n; i++) d[i][i] = 0;
        for (int e = 0; e < edges.size(); e++) {
            int a = edges.get(e)[0];
            int b = edges.get(e)[1];
            d[a][b] = Math.min(d[a][b], lengths.get(e));
            d[b][a] = d[a][b];
        }
        for (int m = 0; m < n; m++) {
            for (int i = 0; i < n; i++) {
                if (!Double.isFinite(d[i][m])) continue;
                for (int j = 0; j < n; j++) {
                    if (!Double.isFinite(d[m][j])) continue;
                    if (d[i][m] + d[m][j] < d[i][j]) d[i][j] = d[i][m] + d[m][j];
                }
            }
        }
        return d;
    }

    static double pathLength(List<int[]> chain) {
        double total = 0;
        for (int i = 1; i < chain.size(); i++) {
            total += Math.sqrt(Math.pow(chain.get(i)[0] - chain.get(i - 1)[0], 2)
                    + Math.pow(chain.get(i)[1] - chain.get(i - 1)[1], 2));
        }
        return total;
    }

    /** Independent feasibility oracle: 4-connected lattice BFS with the exact blocked predicate. */
    static boolean bfsReachable(int width, int height, int[][] obstacles,
                                int sx, int sy, int gx, int gy) {
        Prm probe = new Prm(width, height, obstacles);
        if (probe.blocked(sx, sy) || probe.blocked(gx, gy)) return false;
        int cols = width + 1;
        boolean[] seen = new boolean[cols * (height + 1)];
        Deque<Integer> queue = new ArrayDeque<>();
        int startId = sy * cols + sx;
        int goalId = gy * cols + gx;
        seen[startId] = true;
        queue.add(startId);
        while (!queue.isEmpty()) {
            int id = queue.poll();
            if (id == goalId) return true;
            int x = id % cols;
            int y = id / cols;
            for (int[] step : new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
                int nx = x + step[0];
                int ny = y + step[1];
                if (nx < 0 || ny < 0 || nx >= cols || ny >= height + 1) continue;
                int nid = ny * cols + nx;
                if (seen[nid] || probe.blocked(nx, ny)) continue;
                seen[nid] = true;
                queue.add(nid);
            }
        }
        return false;
    }

    /**
     * Exact 8-connected lattice optimum (axis move = 10, diagonal = 14 tenths), used as the quality
     * reference for the roadmap path. -1 when unreachable.
     */
    static int latticeOptimum(int width, int height, int[][] obstacles,
                              int sx, int sy, int gx, int gy) {
        Prm probe = new Prm(width, height, obstacles);
        if (probe.blocked(sx, sy) || probe.blocked(gx, gy)) return -1;
        int cols = width + 1;
        int size = cols * (height + 1);
        int[] dist = new int[size];
        Arrays.fill(dist, Integer.MAX_VALUE);
        int startId = sy * cols + sx;
        int goalId = gy * cols + gx;
        dist[startId] = 0;
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> Integer.compare(a[0], b[0]));
        queue.add(new int[]{0, startId});
        while (!queue.isEmpty()) {
            int[] top = queue.poll();
            int cost = top[0];
            int id = top[1];
            if (cost > dist[id]) continue;
            if (id == goalId) break;
            int x = id % cols;
            int y = id / cols;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx < 0 || ny < 0 || nx > width || ny > height) continue;
                    if (probe.blocked(nx, ny)) continue;
                    int nid = ny * cols + nx;
                    int candidate = cost + (dx != 0 && dy != 0 ? 14 : 10);
                    if (candidate < dist[nid]) {
                        dist[nid] = candidate;
                        queue.add(new int[]{candidate, nid});
                    }
                }
            }
        }
        return dist[goalId] == Integer.MAX_VALUE ? -1 : dist[goalId];
    }
}

public class Solution {
    static int checksRun;

    public static void main(String[] args) {
        int[][] openSpace = new int[0][];
        int[][] gapWall = {{40, 0, 45, 60}};

        // --- boundary: start equal to goal --------------------------------------
        Prm trivial = new Prm(100, 100, openSpace);
        trivial.buildRoadmap(0, 6, 20, 20, 20, 20, 1L);
        trivial.solve();
        check(trivial.path.size() == 1, "start == goal gives one waypoint");
        check(trivial.pathLength(trivial.path) == 0, "start == goal path has zero length");

        // --- rejected input: blocked start or goal -------------------------------
        Prm blockedStart = new Prm(100, 100, gapWall);
        blockedStart.buildRoadmap(40, 6, 42, 10, 90, 90, 2L);
        check(blockedStart.solve() == null, "start inside an obstacle is unreachable");
        check(!Prm.bfsReachable(100, 100, gapWall, 42, 10, 90, 90), "oracle agrees");

        // --- zero samples: only the direct start-goal edge can help -------------
        Prm noSamples = new Prm(100, 100, openSpace);
        noSamples.buildRoadmap(0, 6, 0, 0, 30, 40, 3L);
        Double direct = noSamples.solve();
        check(direct != null && Math.abs(direct - 50.0) < 1e-9,
                "with no samples the direct visible edge gives exactly 50.0, got " + direct);
        check(noSamples.nodes.size() == 2 && noSamples.edges.size() == 1,
                "zero-sample roadmap has exactly one edge");

        Prm blockedDirect = new Prm(100, 100, gapWall);
        blockedDirect.buildRoadmap(0, 6, 0, 30, 90, 30, 4L);
        check(blockedDirect.solve() == null, "with no samples a blocked direct edge is unreachable");
        check(Prm.bfsReachable(100, 100, gapWall, 0, 30, 90, 30),
                "a detour exists, but the empty roadmap cannot discover it");

        // --- roadmap invariants -------------------------------------------------
        Prm wall = new Prm(100, 100, gapWall);
        wall.buildRoadmap(180, 8, 0, 0, 100, 100, SEED_WALL);
        Double wallLength = wall.solve();
        check(wallLength != null, "the roadmap connects start to goal around the wall");
        check(wall.nodes.size() >= 100, "most samples survive, got " + wall.nodes.size());
        Set<Long> seenPairs = new HashSet<>();
        for (int e = 0; e < wall.edges.size(); e++) {
            int[] edge = wall.edges.get(e);
            check(edge[0] < edge[1], "edges are stored with i < j");
            check(edge[0] != edge[1], "no self loops");
            long key = (long) edge[0] * wall.nodes.size() + edge[1];
            check(seenPairs.add(key), "duplicate edge in the roadmap");
            int[] a = wall.nodes.get(edge[0]);
            int[] b = wall.nodes.get(edge[1]);
            check(!wall.blocked(a[0], a[1]) && !wall.blocked(b[0], b[1]),
                    "roadmap node inside an obstacle");
            for (int[] r : gapWall) {
                check(!Prm.segmentHitsRect(a[0], a[1], b[0], b[1], r),
                        "roadmap edge e=" + e + " clips an obstacle");
            }
            check(Math.abs(wall.lengths.get(e) - Prm.distance(a, b)) < 1e-9,
                    "edge length does not match its endpoints");
        }
        check(wall.edges.size() > wall.nodes.size(), "the roadmap is not a forest");

        // --- Dijkstra against Floyd-Warshall on the same graph -------------------
        double[][] closure = wall.floydWarshall();
        double reference = closure[wall.startIndex][wall.goalIndex];
        check(Math.abs(wallLength - reference) < 1e-9,
                "Dijkstra " + wallLength + " differs from Floyd-Warshall " + reference);
        check(Math.abs(wallLength - Prm.pathLength(wall.path)) < 1e-9,
                "path waypoints do not add up to the reported distance");
        check(wall.path.get(0)[0] == 0 && wall.path.get(0)[1] == 0, "path starts at the start");
        check(wall.path.get(wall.path.size() - 1)[0] == 100
                        && wall.path.get(wall.path.size() - 1)[1] == 100,
                "path ends at the goal");
        boolean onRoadmap = true;
        for (int i = 1; i < wall.path.size(); i++) {
            boolean matched = false;
            for (int e = 0; e < wall.edges.size(); e++) {
                int[] edge = wall.edges.get(e);
                int[] a = wall.nodes.get(edge[0]);
                int[] b = wall.nodes.get(edge[1]);
                boolean forward = same(a, wall.path.get(i - 1)) && same(b, wall.path.get(i));
                boolean backward = same(b, wall.path.get(i - 1)) && same(a, wall.path.get(i));
                if (forward || backward) matched = true;
            }
            if (!matched) onRoadmap = false;
        }
        check(onRoadmap, "every hop of the path is a real roadmap edge");

        // --- quality against the lattice optimum --------------------------------
        double straight = Math.sqrt(2 * 100.0 * 100.0);
        check(wallLength >= straight - 1e-9, "path cannot beat the straight line");
        int optimum = Prm.latticeOptimum(100, 100, gapWall, 0, 0, 100, 100);
        check(optimum > 0, "lattice optimum exists for the wall instance");
        check(wallLength * 10 <= optimum * 1.6,
                "roadmap path " + wallLength + " too far above the lattice optimum "
                        + optimum / 10.0);

        // --- determinism --------------------------------------------------------
        Prm wallAgain = new Prm(100, 100, gapWall);
        wallAgain.buildRoadmap(180, 8, 0, 0, 100, 100, SEED_WALL);
        Double againLength = wallAgain.solve();
        check(wallAgain.nodes.size() == wall.nodes.size(), "same seed, same node count");
        check(wallAgain.edges.size() == wall.edges.size(), "same seed, same edge count");
        check(Math.abs(againLength - wallLength) < 1e-12, "same seed, same path length");

        // --- unreachable instance ------------------------------------------------
        int[][] sealed = {{0, 0, 20, 3}, {0, 17, 20, 20}, {0, 0, 3, 20}, {17, 0, 20, 20}};
        Prm trapped = new Prm(100, 100, sealed);
        trapped.buildRoadmap(120, 6, 10, 10, 90, 90, 5L);
        check(trapped.solve() == null, "sealed room is unreachable");
        check(!Prm.bfsReachable(100, 100, sealed, 10, 10, 90, 90), "oracle agrees on the sealed room");
        Prm inside = new Prm(100, 100, sealed);
        inside.buildRoadmap(40, 6, 10, 10, 10, 16, 6L);
        Double insideLength = inside.solve();
        check(insideLength != null && insideLength >= 6.0 - 1e-9,
                "inside the room nearby free cells are still reachable");

        // --- sample-count sweep: more samples never make the path worse in this run
        Prm coarse = new Prm(100, 100, openSpace);
        coarse.buildRoadmap(12, 6, 0, 0, 100, 100, SEED_OPEN);
        Double coarseLength = coarse.solve();
        check(coarseLength != null, "the coarse roadmap still solves the open field");
        check(coarseLength >= straight - 1e-9 && coarseLength <= straight * 1.6,
                "coarse roadmap path stays plausible: " + coarseLength);
        Prm fine = new Prm(100, 100, openSpace);
        fine.buildRoadmap(240, 8, 0, 0, 100, 100, SEED_OPEN);
        Double fineLength = fine.solve();
        check(fineLength != null, "the fine roadmap solves the open field");
        check(fine.nodes.size() > coarse.nodes.size(), "the fine roadmap has more nodes");
        check(fineLength <= straight * 1.2,
                "fine roadmap path should be near straight, got " + fineLength);
        check(coarseLength >= straight - 1e-9, "coarse path cannot beat the straight line");

        System.out.println("wall instance: nodes = " + wall.nodes.size() + " (discarded "
                + wall.discardedSamples + "), edges = " + wall.edges.size() + ", path length = "
                + round2(wallLength) + " vs lattice optimum " + round2(optimum / 10.0));
        System.out.println("open field: coarse " + round2(coarseLength) + " (" + coarse.nodes.size()
                + " nodes) vs fine " + round2(fineLength) + " (" + fine.nodes.size() + " nodes), "
                + checksRun + " checks");
        System.out.println("PASS");
    }

    /** Fixed seeds for the roadmap experiments (kept as named constants, not magic numbers). */
    static final long SEED_WALL = 0x9E3779B97F4A7C15L;

    static final long SEED_OPEN = 0x5EED5EED5EED5EEDL;

    static boolean same(int[] a, int[] b) { return a[0] == b[0] && a[1] == b[1]; }

    static double round2(double value) { return Math.round(value * 100) / 100.0; }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
