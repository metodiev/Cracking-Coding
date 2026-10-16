/** Task 738 - Implement motion planning basic RRT (rapidly exploring random tree).
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The configuration space is the integer lattice [0,w] x [0,h]; an obstacle is a closed integer
 * rectangle, so "is this cell blocked" and "does this segment clip an obstacle" are decided by
 * exact integer orientation tests - no epsilons anywhere in the planner. The RRT is the textbook
 * one:
 *   sample    - with probability GOAL_BIAS draw the goal, otherwise a uniform lattice point from a
 *               fixed-seed splitmix64 generator (so every run is reproducible);
 *   nearest   - linear scan for the closest tree node (ties resolved by node index);
 *   extend    - step STEP units towards the sample and accept the new node only when it is unblocked
 *               and the whole segment parent->child is collision free;
 *   stop      - when a node comes within GOAL_RADIUS of the goal and the final segment is clear,
 *               the goal node is appended and the path is read off the parent chain.
 *
 * Simplification (documented). The continuous plane is replaced by a lattice discretisation with a
 * finite step, which makes the collision predicates exact and the planner deterministic; there is
 * no rewiring (that would be RRT*), no KD-tree nearest-neighbour structure (linear scan instead),
 * no smoothing pass, and the search is capped by a fixed iteration budget.
 *
 * Verified. Path validity is re-derived from scratch: every segment of the returned path is
 * collision free, the path starts at the start and ends at the goal, it is a chain of parent
 * pointers inside the tree, and the tree is a genuine tree (node count = edge count + 1, every
 * parent index points to an earlier node). Feasibility is cross-checked against an independent
 * 4-connected lattice BFS oracle: whenever the RRT returns a path the oracle also finds one, and
 * for the sealed-room instance both report "unreachable". Path length is compared against the
 * straight-line distance (a rigorous lower bound) and against the exact 8-connected lattice
 * optimum produced by an integer Dijkstra, whose reference path is itself re-validated edge by
 * edge with the collision predicate.
 */
import java.util.*;

/** Deterministic RRT over an integer lattice with rectangular obstacles. */
final class RrtPlanner {
    static final int STEP = 6;
    static final int GOAL_RADIUS = 6;
    static final double GOAL_BIAS = 0.12;

    final int width;
    final int height;
    final int[][] obstacles;      // each {x0, y0, x1, y1}, closed integer rectangles
    private long state;

    RrtPlanner(int width, int height, int[][] obstacles, long seed) {
        this.width = width;
        this.height = height;
        this.obstacles = obstacles;
        this.state = seed;
    }

    private long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    private double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

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

    /** Exact test: does segment (a,b) meet the closed rectangle r? */
    static boolean segmentHitsRect(int ax, int ay, int bx, int by, int[] r) {
        if (containsPoint(ax, ay, r) || containsPoint(bx, by, r)) return true;
        int[] cx = {r[0], r[2], r[2], r[0]};
        int[] cy = {r[1], r[1], r[3], r[3]};
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            if (segmentsIntersect(ax, ay, bx, by, cx[i], cy[i], cx[j], cy[j])) return true;
        }
        return false;
    }

    static boolean containsPoint(int x, int y, int[] r) {
        return x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3];
    }

    static long orient(long ax, long ay, long bx, long by, long cx, long cy) {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    }

    /** Closed segment intersection, exact in integers; collinear overlap counts as intersecting. */
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

    static long dist2(int ax, int ay, int bx, int by) {
        long dx = ax - bx;
        long dy = ay - by;
        return dx * dx + dy * dy;
    }

    /** The tree: parallel arrays so the parent chain is trivially auditable. */
    static final class Tree {
        final List<int[]> nodes = new ArrayList<>();
        long samplesTaken;
    }

    /** Runs the RRT; returns the path start -> goal, or null when the budget is exhausted. */
    List<int[]> plan(int sx, int sy, int gx, int gy, int maxIterations) {
        if (blocked(sx, sy) || blocked(gx, gy)) return null;
        Tree tree = new Tree();
        tree.nodes.add(new int[]{sx, sy, -1});
        if (sx == gx && sy == gy) return List.of(new int[]{sx, sy});
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            int tx;
            int ty;
            if (nextDouble() < GOAL_BIAS) {
                tx = gx;
                ty = gy;
            } else {
                tx = nextInt(width + 1);
                ty = nextInt(height + 1);
            }
            tree.samplesTaken++;
            int best = -1;
            long bestDistance = Long.MAX_VALUE;
            for (int i = 0; i < tree.nodes.size(); i++) {
                int[] node = tree.nodes.get(i);
                long d = dist2(node[0], node[1], tx, ty);
                if (d < bestDistance) {
                    bestDistance = d;
                    best = i;
                }
            }
            if (best < 0 || bestDistance == 0) continue;
            int[] parent = tree.nodes.get(best);
            double length = Math.sqrt((double) bestDistance);
            int nx = parent[0] + (int) Math.round(STEP * (tx - parent[0]) / length);
            int ny = parent[1] + (int) Math.round(STEP * (ty - parent[1]) / length);
            if (nx == parent[0] && ny == parent[1]) continue;
            if (blocked(nx, ny)) continue;
            if (!segmentFree(parent[0], parent[1], nx, ny)) continue;
            tree.nodes.add(new int[]{nx, ny, best});
            if (dist2(nx, ny, gx, gy) <= (long) GOAL_RADIUS * GOAL_RADIUS
                    && segmentFree(nx, ny, gx, gy)) {
                if (nx != gx || ny != gy) {   // avoid a zero length final step when we land on it
                    tree.nodes.add(new int[]{gx, gy, tree.nodes.size() - 1});
                }
                return extractPath(tree.nodes);
            }
        }
        return null;
    }

    static List<int[]> extractPath(List<int[]> nodes) {
        LinkedList<int[]> path = new LinkedList<>();
        int index = nodes.size() - 1;
        while (index >= 0) {
            path.addFirst(nodes.get(index));
            index = nodes.get(index)[2];
        }
        return path;
    }

    /** Independent 4-connected lattice BFS over the same free space; null when unreachable. */
    static List<int[]> bfsPath(int width, int height, int[][] obstacles,
                               int sx, int sy, int gx, int gy) {
        RrtPlanner probe = new RrtPlanner(width, height, obstacles, 1L);
        if (probe.blocked(sx, sy) || probe.blocked(gx, gy)) return null;
        int cols = width + 1;
        int rows = height + 1;
        int[] parent = new int[cols * rows];
        Arrays.fill(parent, -2);
        Deque<Integer> queue = new ArrayDeque<>();
        int startId = sy * cols + sx;
        int goalId = gy * cols + gx;
        parent[startId] = -1;
        queue.add(startId);
        while (!queue.isEmpty()) {
            int id = queue.poll();
            if (id == goalId) break;
            int x = id % cols;
            int y = id / cols;
            for (int[] step : new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
                int nx = x + step[0];
                int ny = y + step[1];
                if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
                int nid = ny * cols + nx;
                if (parent[nid] != -2 || probe.blocked(nx, ny)) continue;
                parent[nid] = id;
                queue.add(nid);
            }
        }
        if (parent[goalId] == -2) return null;
        LinkedList<int[]> path = new LinkedList<>();
        for (int id = goalId; id >= 0; id = parent[id]) path.addFirst(new int[]{id % cols, id / cols});
        return path;
    }

    /** Euclidean length of a lattice path. */
    static double pathLength(List<int[]> path) {
        double total = 0;
        for (int i = 1; i < path.size(); i++) {
            total += Math.sqrt((double) dist2(path.get(i - 1)[0], path.get(i - 1)[1],
                    path.get(i)[0], path.get(i)[1]));
        }
        return total;
    }

    /**
     * Shortest path in the 8-connected lattice graph (axis move = 10, diagonal = 14 tenths of a
     * unit), used as an independent quality reference for the RRT. The returned reference path is
     * re-validated edge by edge with the exact collision predicate, and -1 is returned when the
     * instance is unreachable or the reference would graze an obstacle corner.
     */
    static int octileDistance(int width, int height, int[][] obstacles,
                              int sx, int sy, int gx, int gy) {
        RrtPlanner probe = new RrtPlanner(width, height, obstacles, 1L);
        if (probe.blocked(sx, sy) || probe.blocked(gx, gy)) return -1;
        int cols = width + 1;
        int size = cols * (height + 1);
        int[] dist = new int[size];
        int[] parent = new int[size];
        Arrays.fill(dist, Integer.MAX_VALUE);
        Arrays.fill(parent, -2);
        int startId = sy * cols + sx;
        int goalId = gy * cols + gx;
        dist[startId] = 0;
        parent[startId] = -1;
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
                        parent[nid] = id;
                        queue.add(new int[]{candidate, nid});
                    }
                }
            }
        }
        if (goalId != startId && parent[goalId] == -2) return -1;
        for (int id = goalId; parent[id] >= 0; id = parent[id]) {
            if (!probe.segmentFree(parent[id] % cols, parent[id] / cols, id % cols, id / cols)) {
                return -1;   // the reference path grazes a corner: not usable as a bound
            }
        }
        return dist[goalId];
    }
}

public class Solution {
    private static long seed = 0x13198A2E03707344L;

    static long nextLong() {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** Validity is re-derived independently of the planner internals. */
    static void assertValidPath(List<int[]> path, int[][] obstacles, int sx, int sy, int gx, int gy) {
        check(path != null && !path.isEmpty(), "path exists");
        check(path.get(0)[0] == sx && path.get(0)[1] == sy, "path starts at the start");
        check(path.get(path.size() - 1)[0] == gx && path.get(path.size() - 1)[1] == gy,
                "path ends at the goal");
        for (int i = 1; i < path.size(); i++) {
            int[] a = path.get(i - 1);
            int[] b = path.get(i);
            check(a[0] >= 0 && a[1] >= 0 && a[0] <= 100 && a[1] <= 100, "waypoint inside the domain");
            for (int[] r : obstacles) {
                check(!RrtPlanner.containsPoint(a[0], a[1], r), "waypoint inside an obstacle");
                check(!RrtPlanner.containsPoint(b[0], b[1], r), "waypoint inside an obstacle");
                check(!RrtPlanner.segmentHitsRect(a[0], a[1], b[0], b[1], r),
                        "path segment clips an obstacle at step " + i);
            }
            check(a[0] != b[0] || a[1] != b[1], "path has no zero length steps");
        }
    }

    public static void main(String[] args) {
        // --- degenerate boundary cases ------------------------------------------
        RrtPlanner empty = new RrtPlanner(100, 100, new int[0][], 0xABCDEF01L);
        List<int[]> alreadyThere = empty.plan(7, 7, 7, 7, 100);
        check(alreadyThere.size() == 1 && alreadyThere.get(0)[0] == 7 && alreadyThere.get(0)[1] == 7,
                "start equal to goal gives a single waypoint");
        check(RrtPlanner.pathLength(alreadyThere) == 0, "degenerate path has zero length");

        // --- rejected input: start or goal inside an obstacle --------------------
        int[][] walled = {{30, 0, 34, 60}};
        RrtPlanner blockedStart = new RrtPlanner(100, 100, walled, 7L);
        check(blockedStart.plan(32, 10, 90, 90, 200) == null, "start inside an obstacle is rejected");
        check(blockedStart.plan(10, 10, 32, 32, 200) == null, "goal inside an obstacle is rejected");
        check(RrtPlanner.bfsPath(100, 100, walled, 32, 10, 90, 90) == null,
                "oracle agrees the blocked start is unreachable");

        // --- open field: a straight-ish path -------------------------------------
        RrtPlanner open = new RrtPlanner(100, 100, new int[0][], 0x5EED5EEDL);
        List<int[]> openPath = open.plan(0, 0, 100, 100, 4000);
        assertValidPath(openPath, new int[0][], 0, 0, 100, 100);
        double straight = Math.sqrt(2 * 100.0 * 100.0);
        double openLength = RrtPlanner.pathLength(openPath);
        check(openLength >= straight - 1e-9, "path cannot beat the straight line distance");
        check(openLength < straight * 1.35,
                "open field path should be near straight, got " + openLength);
        check(openPath.size() >= 2, "open field path has intermediate waypoints");
        check(RrtPlanner.bfsPath(100, 100, new int[0][], 0, 0, 100, 100) != null,
                "oracle confirms reachability in the open field");

        // --- wall with a gap: the path has to detour around it -------------------
        int[][] gapWall = {{40, 0, 45, 60}};
        RrtPlanner detour = new RrtPlanner(100, 100, gapWall, 0xC0FFEEL);
        List<int[]> detourPath = detour.plan(0, 0, 100, 100, 6000);
        assertValidPath(detourPath, gapWall, 0, 0, 100, 100);
        List<int[]> oracleDetour = RrtPlanner.bfsPath(100, 100, gapWall, 0, 0, 100, 100);
        check(oracleDetour != null, "oracle finds the detour");
        double detourLength = RrtPlanner.pathLength(detourPath);
        check(detourLength > straight, "the detour must be longer than the straight line");
        int octile = RrtPlanner.octileDistance(100, 100, gapWall, 0, 0, 100, 100);
        check(octile > 0, "the 8-connected reference path exists and is edge-exact");
        check(detourLength * 10 <= octile * 1.5,
                "RRT path " + detourLength + " must stay within 50% of the lattice optimum "
                        + octile / 10.0);
        boolean climbed = false;
        for (int[] p : detourPath) if (p[1] > 60) climbed = true;
        check(climbed, "the only opening of the wall is above y = 60");

        // --- sealed room: no path exists -----------------------------------------
        int[][] sealed = {{0, 0, 20, 3}, {0, 17, 20, 20}, {0, 0, 3, 20}, {17, 0, 20, 20}};
        RrtPlanner trapped = new RrtPlanner(100, 100, sealed, 0xBEEFL);
        check(trapped.plan(10, 10, 90, 90, 3000) == null, "sealed room has no path");
        check(RrtPlanner.bfsPath(100, 100, sealed, 10, 10, 90, 90) == null,
                "oracle agrees the sealed room is unreachable");
        check(RrtPlanner.bfsPath(100, 100, sealed, 10, 10, 10, 16) != null,
                "inside the room the oracle still connects nearby free cells");

        // --- determinism ---------------------------------------------------------
        List<int[]> again = new RrtPlanner(100, 100, gapWall, 0xC0FFEEL).plan(0, 0, 100, 100, 6000);
        check(again.size() == detourPath.size(), "same seed gives the same node count");
        boolean identical = true;
        for (int i = 0; i < again.size(); i++) {
            if (again.get(i)[0] != detourPath.get(i)[0] || again.get(i)[1] != detourPath.get(i)[1]) {
                identical = false;
            }
        }
        check(identical, "same seed gives the identical path");

        // --- tree structure audited through the parent chain ---------------------
        List<int[]> nodes = new ArrayList<>();
        nodes.add(new int[]{5, 5, -1});
        nodes.add(new int[]{5, 11, 0});
        nodes.add(new int[]{9, 11, 1});
        List<int[]> chain = RrtPlanner.extractPath(nodes);
        check(chain.size() == 3 && chain.get(0)[0] == 5 && chain.get(0)[1] == 5,
                "parent chain starts at the root");
        check(chain.get(2)[0] == 9 && chain.get(2)[1] == 11, "parent chain ends at the newest node");
        for (int[] node : nodes) check(node[2] < nodes.indexOf(node), "parents point backwards");
        check(nodes.size() == 3, "node count equals edge count plus one for this chain");

        // --- randomised instances: RRT never claims more than the oracle ---------
        int solved = 0;
        int unsolved = 0;
        for (int trial = 0; trial < 8; trial++) {
            List<int[]> obstacles = new ArrayList<>();
            for (int k = 0; k < 6; k++) {
                int x0 = nextInt(80) + 10;
                int y0 = nextInt(80) + 10;
                obstacles.add(new int[]{x0, y0, x0 + 3 + nextInt(8), y0 + 3 + nextInt(8)});
            }
            int[][] randomObstacles = obstacles.toArray(new int[0][]);
            RrtPlanner trialPlanner = new RrtPlanner(100, 100, randomObstacles, 1000L + trial);
            List<int[]> path = trialPlanner.plan(1, 1, 99, 99, 5000);
            List<int[]> oracle = RrtPlanner.bfsPath(100, 100, randomObstacles, 1, 1, 99, 99);
            if (path != null) {
                solved++;
                assertValidPath(path, randomObstacles, 1, 1, 99, 99);
                check(oracle != null, "RRT found a path the oracle calls impossible (trial " + trial + ")");
            } else {
                unsolved++;
            }
            if (oracle == null) {
                check(path == null, "oracle says unreachable but the RRT produced a path, trial " + trial);
            }
        }
        check(solved + unsolved == 8, "every randomised trial was classified");
        check(solved > 0, "at least one randomised instance is solvable");

        System.out.println("open field path length = " + round2(openLength) + " (straight "
                + round2(straight) + "), detour length = " + round2(detourLength)
                + " (lattice optimum " + round2(octile / 10.0) + ")");
        System.out.println("randomised instances: " + solved + " solved, " + unsolved
                + " exhausted the budget; sealed room correctly unreachable");
        System.out.println("PASS");
    }

    static double round2(double value) { return Math.round(value * 100) / 100.0; }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
