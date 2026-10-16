/** Task 740 - Implement visibility graph path planning for polygons.
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The workspace holds convex polygon obstacles with integer vertices (CCW). The planner builds the
 * classic visibility graph:
 *   nodes  - every obstacle vertex plus the start and the goal;
 *   edges  - a pair (a, b) is joined with weight |ab| iff the *open* segment ab contains no point
 *            strictly inside any obstacle (exact linear feasibility test per polygon edge, see
 *            segmentEntersInterior). Touching an obstacle at a vertex or running along an edge is
 *            therefore permitted, which is the standard closed-workspace convention;
 *   query  - Dijkstra over that graph; the shortest path in a polygonal domain bends only at
 *            obstacle vertices, so the graph optimum is the true euclidean optimum.
 * All predicates are exact integer arithmetic: orientation uses longs, and interior tests for
 * rational sample points are scaled to integers, which is exact because orientation is linear in
 * its third argument.
 *
 * Simplification (documented). Only convex polygonal obstacles are supported (one polygon per
 * obstacle, given CCW), the graph is built in O(V^2 * E) without rotational sweeping or a
 * visibility-query structure, and no path smoothing or clearance margin is applied - the planner
 * returns the exact shortest polyline through visible vertices.
 *
 * Verified. Every graph edge is re-tested with the visibility predicate and every node is outside
 * every polygon; the returned path is continuous, anchored at start and goal, and each of its
 * segments is additionally sampled at 200 rational points to confirm no sample lies strictly inside
 * an obstacle (an oracle independent of the visibility predicate). The length equals Dijkstra's
 * distance and the Floyd-Warshall closure of the same graph, is bounded below by the straight-line
 * distance, and is compared against an independent 8-connected lattice Dijkstra (a valid upper
 * bound on the optimum) which it must not exceed by more than the discretisation slack.
 */
import java.util.*;

/** Visibility graph over convex polygon obstacles. */
final class VisibilityPlanner {
    final List<long[][]> polygons = new ArrayList<>();   // each polygon: CCW vertices {x, y}
    final List<long[]> nodes = new ArrayList<>();        // obstacle vertices, then start, goal
    final List<int[]> edges = new ArrayList<>();         // {i, j} with i < j
    final List<Double> weights = new ArrayList<>();
    int startIndex = -1;
    int goalIndex = -1;
    List<long[]> path;

    static long orient(long ax, long ay, long bx, long by, long cx, long cy) {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    }

    void addPolygon(long[]... vertices) {
        polygons.add(vertices);
        for (long[] v : vertices) nodes.add(v);
    }

    /** True when the open segment (ax,ay)-(bx,by) crosses the open segment (cx,cy)-(dx,dy). */
    static boolean properCross(long ax, long ay, long bx, long by,
                               long cx, long cy, long dx, long dy) {
        long d1 = orient(cx, cy, dx, dy, ax, ay);
        long d2 = orient(cx, cy, dx, dy, bx, by);
        long d3 = orient(ax, ay, bx, by, cx, cy);
        long d4 = orient(ax, ay, bx, by, dx, dy);
        return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
                && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
    }

    /**
     * Strict interior test for a point given in scaled coordinates (px, py); scaling by a positive
     * integer preserves the orientation signs, so this works for mid points and sampled points.
     */
    static boolean strictlyInside(long[][] polygon, long px, long py) {
        for (int i = 0; i < polygon.length; i++) {
            long[] a = polygon[i];
            long[] b = polygon[(i + 1) % polygon.length];
            if (orient(a[0], a[1], b[0], b[1], px, py) <= 0) return false;
        }
        return true;
    }

    /** Interior test for a lattice point: identical semantics, no scaling needed. */
    boolean insideAny(long x, long y) {
        for (long[][] polygon : polygons) if (strictlyInside(polygon, x, y)) return true;
        return false;
    }

    /**
     * Exact test: does the open segment (ax,ay)-(bx,by) contain a point strictly inside the CCW
     * convex polygon? For each edge the strict-inside predicate is
     *     f_i(t) = orient(v_i, v_i+1, a + t*(b - a)) = f_i(0) + t * slope_i > 0,
     * a linear condition in t, so each edge contributes an (possibly empty or unbounded) open
     * interval and the segment enters the interior iff the intersection of those intervals with
     * (0,1) is non empty. Touching a vertex, running along an edge, or crossing the boundary at a
     * single point therefore never counts as an interior hit, while a segment that slips into the
     * polygon through a corner does - which the mid point test alone would miss.
     */
    static boolean segmentEntersInterior(long[] a, long[] b, long[][] polygon) {
        long loNum = 0;
        long loDen = 1;      // t > lo
        long hiNum = 1;
        long hiDen = 1;      // t < hi
        for (int i = 0; i < polygon.length; i++) {
            long[] v1 = polygon[i];
            long[] v2 = polygon[(i + 1) % polygon.length];
            long f0 = orient(v1[0], v1[1], v2[0], v2[1], a[0], a[1]);
            long f1 = orient(v1[0], v1[1], v2[0], v2[1], b[0], b[1]);
            long slope = f1 - f0;
            if (slope == 0) {
                if (f0 <= 0) return false;      // this edge is never strictly satisfied
                continue;
            }
            long num = -f0;
            long den = slope;
            if (den < 0) {
                num = -num;
                den = -den;
            }
            if (slope > 0) {                    // f(t) > 0 for t > num/den
                if (num * loDen > loNum * den) {
                    loNum = num;
                    loDen = den;
                }
            } else if (num * hiDen < hiNum * den) {   // f(t) > 0 for t < num/den
                hiNum = num;
                hiDen = den;
            }
        }
        return loNum * hiDen < hiNum * loDen;
    }

    boolean visible(long[] a, long[] b) {
        return visible(a, b, polygons);
    }

    static boolean visible(long[] a, long[] b, List<long[][]> polygons) {
        for (long[][] polygon : polygons) {
            if (segmentEntersInterior(a, b, polygon)) return false;
        }
        return true;
    }

    /** Adds the start and the goal, then all visible pairs; returns false for rejected input. */
    boolean prepare(long[] start, long[] goal) {
        if (insideAny(start[0], start[1]) || insideAny(goal[0], goal[1])) return false;
        startIndex = nodes.size();
        nodes.add(start);
        goalIndex = nodes.size();
        nodes.add(goal);
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {
                if (!visible(nodes.get(i), nodes.get(j))) continue;
                edges.add(new int[]{i, j});
                weights.add(distance(nodes.get(i), nodes.get(j)));
            }
        }
        return true;
    }

    static double distance(long[] a, long[] b) {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        return Math.sqrt(dx * dx + dy * dy);
    }

    /** Dijkstra over the visibility graph; fills `path`, returns the length or null. */
    Double solve() {
        long[] start = nodes.get(startIndex);
        long[] goal = nodes.get(goalIndex);
        if (start[0] == goal[0] && start[1] == goal[1]) {
            path = List.of(start);
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
                double candidate = dist[u] + weights.get(e);
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
        LinkedList<long[]> chain = new LinkedList<>();
        for (int v = goalIndex; v >= 0; v = parent[v]) chain.addFirst(nodes.get(v));
        path = chain;
        return dist[goalIndex];
    }

    /** Independent oracle: dense Floyd-Warshall over the same graph. */
    double[][] floydWarshall() {
        int n = nodes.size();
        double[][] d = new double[n][n];
        for (double[] row : d) Arrays.fill(row, Double.POSITIVE_INFINITY);
        for (int i = 0; i < n; i++) d[i][i] = 0;
        for (int e = 0; e < edges.size(); e++) {
            int a = edges.get(e)[0];
            int b = edges.get(e)[1];
            d[a][b] = Math.min(d[a][b], weights.get(e));
            d[b][a] = d[a][b];
        }
        for (int m = 0; m < n; m++) {
            for (int i = 0; i < n; i++) {
                if (!Double.isFinite(d[i][m])) continue;
                for (int j = 0; j < n; j++) {
                    if (!Double.isFinite(d[m][j])) continue;
                    double candidate = d[i][m] + d[m][j];
                    if (candidate < d[i][j]) d[i][j] = candidate;
                }
            }
        }
        return d;
    }

    /**
     * Independent lattice oracle: 8-connected Dijkstra with weights 10 (axis) and 14 (diagonal)
     * over lattice points that are outside or on the obstacles; diagonal moves are rejected when
     * they would cut into an obstacle interior. An upper bound on the euclidean optimum.
     */
    static long latticeUpperBound(int size, List<long[][]> polygons,
                                  long[] start, long[] goal) {
        VisibilityPlanner probe = new VisibilityPlanner();
        for (long[][] polygon : polygons) probe.polygons.add(polygon);
        int cols = size + 1;
        int total = cols * cols;
        int[] dist = new int[total];
        Arrays.fill(dist, Integer.MAX_VALUE);
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> Integer.compare(a[0], b[0]));
        int startId = (int) (start[1] * cols + start[0]);
        int goalId = (int) (goal[1] * cols + goal[0]);
        dist[startId] = 0;
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
                    if (nx < 0 || ny < 0 || nx >= cols || ny >= cols) continue;
                    if (probe.insideAny(nx, ny)) continue;
                    if (dx != 0 && dy != 0) {
                        // a diagonal step may not squeeze through an obstacle corner
                        if (probe.insideAny(nx, y) || probe.insideAny(x, ny)) continue;
                        long sumX = (long) nx + x;
                        long sumY = (long) ny + y;
                        boolean cut = false;
                        for (long[][] polygon : polygons) {
                            if (strictlyInside(polygon, sumX, sumY)) cut = true;
                        }
                        if (cut) continue;
                    }
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

    static long[] p(long x, long y) { return new long[]{x, y}; }

    /** Samples the open segment at `steps` rational points and rejects interior hits. */
    static void assertSegmentOutsideObstacles(long[] a, long[] b, List<long[][]> polygons,
                                              int steps, String label) {
        for (int i = 1; i < steps; i++) {
            long px = a[0] * steps + (long) i * (b[0] - a[0]);
            long py = a[1] * steps + (long) i * (b[1] - a[1]);
            for (long[][] polygon : polygons) {
                check(!VisibilityPlanner.strictlyInside(polygon, px, py),
                        label + ": sample " + i + " of " + steps + " is inside an obstacle");
            }
        }
    }

    static void assertValidPath(List<long[]> path, List<long[][]> polygons, long[] start, long[] goal) {
        check(path != null && !path.isEmpty(), "path exists");
        check(path.get(0)[0] == start[0] && path.get(0)[1] == start[1], "path starts at the start");
        check(path.get(path.size() - 1)[0] == goal[0] && path.get(path.size() - 1)[1] == goal[1],
                "path ends at the goal");
        for (int i = 1; i < path.size(); i++) {
            long[] a = path.get(i - 1);
            long[] b = path.get(i);
            check(VisibilityPlanner.visible(a, b, polygons), "hop " + i + " is not a visible pair");
            assertSegmentOutsideObstacles(a, b, polygons, 200, "hop " + i);
        }
    }

    public static void main(String[] args) {
        // --- obstacle set --------------------------------------------------------
        List<long[][]> polygons = new ArrayList<>();
        long[][] square = {p(30, 30), p(50, 30), p(50, 50), p(30, 50)};
        long[][] hexagon = {p(70, 15), p(80, 20), p(80, 32), p(70, 37), p(62, 32), p(62, 20)};
        long[][] triangle = {p(15, 60), p(35, 60), p(25, 82)};
        polygons.add(square);
        polygons.add(hexagon);
        polygons.add(triangle);

        VisibilityPlanner planner = new VisibilityPlanner();
        planner.addPolygon(square);
        planner.addPolygon(hexagon);
        planner.addPolygon(triangle);
        check(planner.nodes.size() == 4 + 6 + 3, "graph holds one node per obstacle vertex");

        // --- rejected input: start or goal inside an obstacle --------------------
        check(!planner.prepare(p(40, 40), p(100, 100)), "start inside a polygon is rejected");
        check(!planner.prepare(p(0, 0), p(75, 25)), "goal inside a polygon is rejected");

        // --- trivial query: start equal to the goal ------------------------------
        VisibilityPlanner trivial = new VisibilityPlanner();
        trivial.addPolygon(square);
        check(trivial.prepare(p(5, 5), p(5, 5)), "trivial query accepted");
        check(trivial.solve() == 0.0, "start == goal gives length 0");
        check(trivial.path.size() == 1, "start == goal gives a single waypoint");

        // --- unblocked query: the direct segment is optimal ----------------------
        VisibilityPlanner direct = new VisibilityPlanner();
        direct.addPolygon(square);
        check(direct.prepare(p(0, 0), p(10, 10)), "direct query accepted");
        Double directLength = direct.solve();
        check(Math.abs(directLength - Math.sqrt(200)) < 1e-9,
                "direct query is the straight line, got " + directLength);
        check(direct.path.size() == 2, "direct query has no detour waypoints");

        // --- blocked direct line: the path must bend around the square -----------
        VisibilityPlanner around = new VisibilityPlanner();
        around.addPolygon(square);
        check(around.prepare(p(0, 0), p(100, 100)), "corner query accepted");
        Double aroundLength = around.solve();
        assertValidPath(around.path, polygons.subList(0, 1), p(0, 0), p(100, 100));
        double straight = Math.sqrt(2 * 100.0 * 100.0);
        check(aroundLength > straight, "going around the square must be longer than the diagonal");
        check(aroundLength < straight * 1.2, "the detour should be small, got " + aroundLength);
        boolean touchesSquareVertex = false;
        for (long[] waypoint : around.path) {
            if (waypoint[0] >= 30 && waypoint[0] <= 50 && waypoint[1] >= 30 && waypoint[1] <= 50) {
                touchesSquareVertex = true;
            }
        }
        check(touchesSquareVertex, "the detour bends at square vertices");

        // --- full obstacle course ------------------------------------------------
        VisibilityPlanner course = new VisibilityPlanner();
        course.addPolygon(square);
        course.addPolygon(hexagon);
        course.addPolygon(triangle);
        check(course.prepare(p(0, 0), p(100, 100)), "course query accepted");
        Double courseLength = course.solve();
        assertValidPath(course.path, polygons, p(0, 0), p(100, 100));
        check(courseLength >= straight - 1e-9, "no path can beat the straight line");
        check(course.nodes.size() == 15, "course graph has 13 vertices plus start and goal");

        // --- graph invariants ----------------------------------------------------
        int blockedPairs = 0;
        for (int i = 0; i < course.nodes.size(); i++) {
            check(!course.insideAny(course.nodes.get(i)[0], course.nodes.get(i)[1]),
                    "graph node inside an obstacle");
            for (int j = i + 1; j < course.nodes.size(); j++) {
                boolean visible = VisibilityPlanner.visible(course.nodes.get(i), course.nodes.get(j),
                        polygons);
                if (!visible) blockedPairs++;
                boolean inGraph = false;
                for (int e = 0; e < course.edges.size(); e++) {
                    int[] edge = course.edges.get(e);
                    if (edge[0] == i && edge[1] == j) inGraph = true;
                }
                check(inGraph == visible, "edge (" + i + "," + j + ") disagrees with the predicate");
            }
        }
        check(blockedPairs > 0, "the obstacle course really does block some vertex pairs");
        check(course.edges.size() < course.nodes.size() * (course.nodes.size() - 1) / 2,
                "the visibility graph is not complete");

        // --- the two formulations agree: a proper edge crossing always blocks ---------
        int properCrossCases = 0;
        for (int i = 0; i < course.nodes.size(); i++) {
            for (int j = i + 1; j < course.nodes.size(); j++) {
                long[] a = course.nodes.get(i);
                long[] b = course.nodes.get(j);
                boolean crosses = false;
                for (long[][] polygon : polygons) {
                    for (int e = 0; e < polygon.length; e++) {
                        long[] c = polygon[e];
                        long[] d = polygon[(e + 1) % polygon.length];
                        if (VisibilityPlanner.properCross(a[0], a[1], b[0], b[1],
                                c[0], c[1], d[0], d[1])) {
                            crosses = true;
                        }
                    }
                }
                if (crosses) {
                    properCrossCases++;
                    check(!VisibilityPlanner.visible(a, b, polygons),
                            "a proper edge crossing must block visibility");
                }
            }
        }
        check(properCrossCases > 0, "the course produces proper crossings to test");
        for (int i = 0; i < 4; i++) {   // polygon edges are grazing-visible by convention
            long[] a = square[i];
            long[] b = square[(i + 1) % 4];
            check(VisibilityPlanner.visible(a, b, polygons), "adjacent square vertices are visible");
        }

        // --- optimality: Dijkstra == Floyd-Warshall, and the lattice bound holds --
        double[][] closure = course.floydWarshall();
        double reference = closure[course.startIndex][course.goalIndex];
        check(Math.abs(courseLength - reference) < 1e-9,
                "Dijkstra " + courseLength + " differs from Floyd-Warshall " + reference);
        double waypointSum = 0;
        for (int i = 1; i < course.path.size(); i++) {
            waypointSum += VisibilityPlanner.distance(course.path.get(i - 1), course.path.get(i));
        }
        check(Math.abs(waypointSum - courseLength) < 1e-9, "waypoints do not add up to the length");
        long latticeCost = VisibilityPlanner.latticeUpperBound(100, polygons, p(0, 0), p(100, 100));
        check(latticeCost > 0, "the lattice oracle finds a route through the course");
        check(courseLength * 10 <= latticeCost * 1.03 + 1,
                "visibility path " + courseLength + " exceeds the lattice bound "
                        + latticeCost / 10.0 + " by more than the discretisation slack");

        // --- determinism ---------------------------------------------------------
        VisibilityPlanner repeat = new VisibilityPlanner();
        repeat.addPolygon(square);
        repeat.addPolygon(hexagon);
        repeat.addPolygon(triangle);
        repeat.prepare(p(0, 0), p(100, 100));
        Double repeatLength = repeat.solve();
        check(Math.abs(repeatLength - courseLength) < 1e-12, "same query, same length");
        check(repeat.edges.size() == course.edges.size(), "same query, same edge count");

        // --- completely walled off query -----------------------------------------
        // A square frame (four overlapping wall polygons) with a 12x12 hole: the start sits in the
        // hole, the goal outside, and no grazing path can slip through the overlapping corners.
        long[][] ringBottom = {p(0, 0), p(20, 0), p(20, 4), p(0, 4)};
        long[][] ringTop = {p(0, 16), p(20, 16), p(20, 20), p(0, 20)};
        long[][] ringLeft = {p(0, 0), p(4, 0), p(4, 20), p(0, 20)};
        long[][] ringRight = {p(16, 0), p(20, 0), p(20, 20), p(16, 20)};
        List<long[][]> ring = List.of(ringBottom, ringTop, ringLeft, ringRight);
        VisibilityPlanner sealed = new VisibilityPlanner();
        for (long[][] side : ring) sealed.addPolygon(side);
        check(sealed.prepare(p(10, 10), p(90, 90)), "sealed query accepted");
        check(sealed.solve() == null, "the frame separates the start from the goal");
        check(sealed.path == null, "no path stored for the sealed query");
        check(VisibilityPlanner.latticeUpperBound(100, ring, p(10, 10), p(90, 90)) == -1,
                "lattice oracle agrees the frame is impassable");
        check(VisibilityPlanner.latticeUpperBound(100, ring, p(6, 6), p(14, 14)) > 0,
                "inside the hole the lattice oracle still connects free cells");

        System.out.println("course: nodes = " + course.nodes.size() + ", edges = " + course.edges.size()
                + " (" + blockedPairs + " vertex pairs blocked), path = " + round2(courseLength)
                + ", lattice bound = " + round2(latticeCost / 10.0));
        System.out.println("square detour = " + round2(aroundLength) + " vs straight "
                + round2(straight) + "; " + checksRun + " checks");
        System.out.println("PASS");
    }

    static double round2(double value) { return Math.round(value * 100) / 100.0; }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
