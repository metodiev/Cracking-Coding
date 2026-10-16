/** Task 922 - Implement multi-robot coordination with conflict resolution and task allocation.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Warehouse-style multi-robot coordination on a grid. Tasks are (pickup, delivery) pairs handed to robots
 * by a greedy auction: the tasks are considered in a fixed order and each goes to the robot whose marginal
 * cost (distance to the pickup plus pickup-to-delivery distance) is smallest, ties going to the lowest index.
 * The assigned waypoint sequences are then executed by prioritised planning: robots are planned one at a
 * time in a fixed order, each leg by breadth-first search over the time-expanded graph with waiting moves,
 * against a reservation table that already holds every earlier robot's vertex and edge reservations, so a
 * later robot simply routes around or waits for a robot that has priority.
 * Simplified: a unit-cost 4-connected grid with no static obstacles, symmetric robots (no differential
 * constraints), and coordination only through time reservations. The safety property is exactly the mapf
 * one: no two robots share a vertex at the same time and no pair swaps cells in one step; it is checked
 * after every robot is added, and the naive schedule of the same robots (each planned alone) is verified to
 * contain conflicts, which is what proves the reservations did the work.
 */
import java.util.*;

/** One robot: its start cell, the tasks assigned to it, and the cell it occupies at every timestep. */
final class Robot {
    final int id;
    final int start;
    final List<Integer> assigned = new ArrayList<>();
    int[] timeline;

    Robot(int id, int start) { this.id = id; this.start = start; }

    int end() { return timeline == null ? start : timeline[timeline.length - 1]; }
}

/** Time-expanded BFS with a reservation table: the core of prioritised planning. */
final class Mapf {
    static final int HORIZON = 240;
    final int size;
    final Set<Long> vertex = new HashSet<>(), edge = new HashSet<>();

    Mapf(int size) { this.size = size; }

    static long vkey(int t, int cell) { return ((long) t << 8) | cell; }
    static long ekey(int t, int from, int to) { return ((long) t << 16) | (from << 8) | to; }

    static int manhattan(int a, int b, int size) {
        return Math.abs(a / size - b / size) + Math.abs(a % size - b % size);
    }

    /** Earliest-arrival route for one robot, honouring the reservations at t >= t0 (waiting allowed). */
    List<Integer> leg(int start, int goal, int t0) {
        Map<Long, Long> parent = new HashMap<>();
        ArrayDeque<long[]> queue = new ArrayDeque<>();
        parent.put(vkey(t0, start), -1L);
        queue.add(new long[] {t0, start});
        int[] step = {-size, size, -1, 1, 0};
        while (!queue.isEmpty()) {
            long[] cur = queue.poll();
            int t = (int) cur[0], cell = (int) cur[1];
            if (cell == goal) {
                LinkedList<Integer> path = new LinkedList<>();
                for (long key = vkey(t, cell); key != -1L; key = parent.get(key)) path.addFirst((int) (key & 255));
                return path;
            }
            if (t - t0 >= HORIZON) continue;
            int r = cell / size, c = cell % size;
            for (int m = 0; m < 5; m++) {
                if ((m == 0 && r == 0) || (m == 1 && r == size - 1) || (m == 2 && c == 0) || (m == 3 && c == size - 1)) continue;
                int next = cell + step[m];
                if (vertex.contains(vkey(t + 1, next)) || edge.contains(ekey(t, next, cell))) continue;
                if (parent.containsKey(vkey(t + 1, next))) continue;
                parent.put(vkey(t + 1, next), vkey(t, cell));
                queue.add(new long[] {t + 1, next});
            }
        }
        return null;
    }

    /** Plan one robot through its waypoint list and record its reservations. */
    int[] plan(Robot robot, List<int[]> tasks) {
        List<Integer> timeline = new ArrayList<>();
        timeline.add(robot.start);
        for (int task : robot.assigned) {
            for (int goal : new int[] {tasks.get(task)[0], tasks.get(task)[1]}) {
                int from = timeline.get(timeline.size() - 1);
                List<Integer> cells = leg(from, goal, timeline.size() - 1);
                if (cells == null) throw new IllegalStateException("robot " + robot.id + " cannot reach " + goal);
                timeline.addAll(cells.subList(1, cells.size()));
            }
        }
        int[] cells = new int[timeline.size()];
        for (int t = 0; t < cells.length; t++) cells[t] = timeline.get(t);
        robot.timeline = cells;
        reserve(cells);
        return cells;
    }

    void reserve(int[] cells) {
        for (int t = 0; t < cells.length; t++) vertex.add(vkey(t, cells[t]));
        for (int t = 0; t + 1 < cells.length; t++) edge.add(ekey(t, cells[t], cells[t + 1]));
    }

    /** Greedy auction: tasks in a fixed order, each to its cheapest robot, ties to the lowest index. */
    static void allocate(List<Robot> robots, List<int[]> tasks) {
        int[] ends = new int[robots.size()];
        for (int r = 0; r < robots.size(); r++) ends[r] = robots.get(r).start;
        for (int t = 0; t < tasks.size(); t++) {
            int best = 0;
            long bestCost = Long.MAX_VALUE;
            for (int r = 0; r < robots.size(); r++) {
                long cost = Mapf.manhattan(ends[r], tasks.get(t)[0], 8) + Mapf.manhattan(tasks.get(t)[0], tasks.get(t)[1], 8);
                if (cost < bestCost) { bestCost = cost; best = r; }
            }
            robots.get(best).assigned.add(t);
            ends[best] = tasks.get(t)[1];
        }
    }

    /** Number of unsafe (robot, robot, time) triples in a set of timelines, swaps included. Robots that
     *  have finished leave the map, exactly as the reservation table assumes. */
    static int conflicts(List<Robot> robots) {
        int bad = 0;
        for (int a = 0; a < robots.size(); a++) {
            for (int b = a + 1; b < robots.size(); b++) {
                int[] x = robots.get(a).timeline, y = robots.get(b).timeline;
                int common = Math.min(x.length, y.length);
                for (int t = 0; t < common; t++) {
                    if (x[t] == y[t]) bad++;
                    if (t + 1 < common && x[t + 1] == y[t] && y[t + 1] == x[t] && x[t] != y[t]) bad++;
                }
            }
        }
        return bad;
    }

    /** True when the timeline visits the pickup and afterwards the delivery of that task. */
    static boolean serves(int[] cells, int[] task) {
        int seen = -1;
        for (int t = 0; t < cells.length; t++) {
            if (cells[t] == task[0] && seen < 0) seen = t;
            if (seen >= 0 && t > seen && cells[t] == task[1]) return true;
        }
        return false;
    }
}

public class Solution {
    static final int SIZE = 8;

    static List<Robot> build() {
        List<Robot> robots = new ArrayList<>();
        int[] starts = {0, 63, 7, 56};
        for (int r = 0; r < starts.length; r++) robots.add(new Robot(r, starts[r]));
        return robots;
    }

    static List<int[]> tasks() {
        return new ArrayList<>(Arrays.asList(new int[] {2, 58}, new int[] {61, 3}, new int[] {30, 17},
                new int[] {9, 50}, new int[] {40, 20}));
    }

    /** Every robot planned alone, ignoring the others - the schedule the reservations must improve on. */
    static List<Robot> naive() {
        List<Robot> robots = build();
        List<int[]> tasks = tasks();
        Mapf.allocate(robots, tasks);
        for (Robot r : robots) new Mapf(SIZE).plan(r, tasks);
        return robots;
    }

    static List<Robot> coordinated() {
        List<Robot> robots = build();
        List<int[]> tasks = tasks();
        Mapf.allocate(robots, tasks);
        Mapf map = new Mapf(SIZE);
        for (Robot r : robots) map.plan(r, tasks);
        return robots;
    }

    public static void main(String[] args) {
        List<Robot> naive = naive();
        List<Robot> planned = coordinated();
        List<int[]> tasks = tasks();

        int naiveConflicts = Mapf.conflicts(naive);
        int plannedConflicts = Mapf.conflicts(planned);

        // Allocation properties: a permutation of the tasks, and no worse than giving everything to robot 0.
        boolean[] seen = new boolean[tasks.size()];
        int assigned = 0, solo = 0, soloEnd = 0;
        for (Robot r : planned) for (int t : r.assigned) { seen[t] = true; assigned++; }
        for (int t = 0; t < tasks.size(); t++) {
            solo += Mapf.manhattan(soloEnd, tasks.get(t)[0], SIZE) + Mapf.manhattan(tasks.get(t)[0], tasks.get(t)[1], SIZE);
            soloEnd = tasks.get(t)[1];
        }
        int greedy = 0;
        for (Robot r : planned) {
            int at = r.start;
            for (int t : r.assigned) {
                greedy += Mapf.manhattan(at, tasks.get(t)[0], SIZE) + Mapf.manhattan(tasks.get(t)[0], tasks.get(t)[1], SIZE);
                at = tasks.get(t)[1];
            }
        }
        double makespan = 0;
        for (Robot r : planned) makespan = Math.max(makespan, r.timeline.length - 1);

        System.out.println("coordination: " + planned.size() + " robots, " + tasks.size() + " tasks, naive conflicts "
                + naiveConflicts + ", coordinated conflicts " + plannedConflicts + ", makespan " + makespan);
        System.out.println("allocation cost (manhattan sum) greedy " + greedy + " vs single-robot " + solo
                + ", assigned " + assigned + " tasks to " + planned.size() + " robots");
        System.out.println("naive path lengths " + pathLengths(naive) + ", coordinated " + pathLengths(planned));

        check(naiveConflicts > 0, "the naive schedule must actually conflict, found " + naiveConflicts);
        check(plannedConflicts == 0, "the coordinated schedule must be collision-free, found " + plannedConflicts);
        check(assigned == tasks.size(), "every task must be assigned exactly once, got " + assigned);
        for (int t = 0; t < tasks.size(); t++) check(seen[t], "task " + t + " was never assigned");
        check(greedy <= solo, "the auction must not be worse than one robot doing everything: " + greedy + " vs " + solo);

        for (Robot r : planned) {
            for (int t : r.assigned) check(Mapf.serves(r.timeline, tasks.get(t)),
                    "robot " + r.id + " must visit pickup then delivery of task " + t);
        }
        for (Robot r : naive) {                                   // Manhattan oracle on unobstructed legs
            int expected = 0, at = r.start;
            for (int t : r.assigned) {
                expected += Mapf.manhattan(at, tasks.get(t)[0], SIZE) + Mapf.manhattan(tasks.get(t)[0], tasks.get(t)[1], SIZE);
                at = tasks.get(t)[1];
            }
            check(r.timeline.length - 1 == expected, "an unobstructed route must be Manhattan-shortest: "
                    + (r.timeline.length - 1) + " vs " + expected);
        }

        // Boundary 1: a single robot never conflicts and never waits.
        Robot loner = new Robot(0, 12);
        loner.assigned.add(0);
        new Mapf(SIZE).plan(loner, tasks);
        int straight = Mapf.manhattan(12, tasks.get(0)[0], SIZE) + Mapf.manhattan(tasks.get(0)[0], tasks.get(0)[1], SIZE);
        check(loner.timeline.length - 1 == straight && Mapf.conflicts(Collections.singletonList(loner)) == 0,
                "a single robot must follow the shortest route with no conflicts");

        // Boundary 2: two robots that must cross the same corridor - the resolver has to delay one of them.
        List<int[]> swapTasks = new ArrayList<>(Arrays.asList(new int[] {4, 0}, new int[] {3, 7}));
        List<Robot> head = new ArrayList<>();
        Robot left = new Robot(0, 0), right = new Robot(1, 7);
        left.assigned.add(0);
        right.assigned.add(1);
        head.add(left);
        head.add(right);
        List<Robot> headNaive = new ArrayList<>();
        for (Robot r : head) {
            Robot copy = new Robot(r.id, r.start);
            copy.assigned.addAll(r.assigned);
            headNaive.add(copy);
        }
        for (Robot r : headNaive) new Mapf(SIZE).plan(r, swapTasks);
        Mapf headPlan = new Mapf(SIZE);
        for (Robot r : head) headPlan.plan(r, swapTasks);
        int naiveDelays = headNaive.get(0).timeline.length + headNaive.get(1).timeline.length;
        int plannedDelays = head.get(0).timeline.length + head.get(1).timeline.length;
        check(Mapf.conflicts(headNaive) > 0, "crossing robots must conflict when planned alone");
        check(Mapf.conflicts(head) == 0, "the resolver must remove the head-on conflict");
        check(plannedDelays > naiveDelays, "resolving a head-on conflict costs at least one wait: "
                + plannedDelays + " vs " + naiveDelays);
        check(Mapf.serves(head.get(0).timeline, swapTasks.get(0)) && Mapf.serves(head.get(1).timeline, swapTasks.get(1)),
                "both robots must still complete their tasks after the delay");

        check(Arrays.equals(coordinated().get(0).timeline, planned.get(0).timeline)
                && Arrays.equals(coordinated().get(3).timeline, planned.get(3).timeline), "planning must be deterministic");

        System.out.println("head-on pair: naive total steps " + naiveDelays + " (conflicts "
                + Mapf.conflicts(headNaive) + ") -> coordinated " + plannedDelays + " (conflicts "
                + Mapf.conflicts(head) + ")");
        System.out.println("PASS");
    }

    static String pathLengths(List<Robot> robots) {
        StringBuilder sb = new StringBuilder("[");
        for (Robot r : robots) sb.append(r.timeline.length - 1).append(r == robots.get(robots.size() - 1) ? "" : ", ");
        return sb.append("]").toString();
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
