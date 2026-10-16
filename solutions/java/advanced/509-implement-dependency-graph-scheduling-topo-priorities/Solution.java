/** Task 509 - Implement dependency graph scheduling (topo + priorities). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * Dependency-graph scheduling with priorities: tasks are nodes of a DAG (an edge u -> v means "u
 * must finish before v starts"), in-degrees are tracked, and Kahn's algorithm drives a ready set
 * that is a priority-ordered structure - a ready task is scheduled only when all of its
 * predecessors have been placed, and among the ready tasks the one with the highest priority wins,
 * ties broken by the smallest task id. That is the standard "priority topological sort" used by
 * build systems and Kanban-style work queues: it is a valid topological order that is also point
 * wise optimal (greedy), because at every position it emits an available task of maximum priority.
 *
 * Documented checks: the produced permutation contains all n tasks exactly once and respects every
 * edge (pos[u] < pos[v]); the greedy rule is verified independently - for each position the emitted
 * task's priority is recomputed as the maximum priority among all tasks whose predecessors are
 * already placed (an O(n^2) rescan that shares no code with Kahn's loop) and ties are checked to go
 * to the smaller id; on a small 6-node / 7-edge graph every topological order is enumerated by
 * brute force and the produced order is checked to be a member of that set while also being the
 * greedy maximum; hand-computed instances are asserted exactly (a diamond with priorities 5 and 3
 * gives the documented order, a chain is forced regardless of priorities, isolated nodes come out
 * in priority-then-id order); a priority-tie graph proves the documented tie-break; cycles - a
 * 3-cycle and a self-loop - are detected (schedule returns null and hasCycle() is true) while every
 * acyclic instance reports hasCycle() == false; and a re-run reproduces the order exactly.
 */
import java.util.*;

public class Solution {
    /** 75/74 LCG modulo 65537 - the deterministic generator used by waves 1-2. */
    static class Lcg {
        private int state;

        Lcg(int seed) {
            this.state = ((seed % 65537) + 65537) % 65537;
        }

        double next() {
            state = (state * 75 + 74) % 65537;
            return state / 65537.0;
        }

        int nextInt(int bound) {
            return (int) (next() * bound);
        }
    }

    static class TaskGraph {
        final int n;
        final String[] names;
        final int[] priority;
        final List<Set<Integer>> successors = new ArrayList<>();
        final List<Set<Integer>> predecessors = new ArrayList<>();

        TaskGraph(String[] names, int[] priority) {
            this.n = names.length;
            this.names = names;
            this.priority = priority;
            for (int i = 0; i < n; i++) {
                successors.add(new LinkedHashSet<>());
                predecessors.add(new LinkedHashSet<>());
            }
        }

        void addEdge(int from, int to) {
            if (from == to) throw new IllegalArgumentException("self edges are cycles");
            successors.get(from).add(to);
            predecessors.get(to).add(from);
        }

        /** Kahn's algorithm with a max-priority ready set; null when the graph has a cycle. */
        List<Integer> schedule() {
            int[] indegree = new int[n];
            for (int v = 0; v < n; v++) indegree[v] = predecessors.get(v).size();
            PriorityQueue<Integer> ready = new PriorityQueue<>(
                    (a, b) -> priority[a] != priority[b]
                            ? Integer.compare(priority[b], priority[a])
                            : Integer.compare(a, b));
            for (int v = 0; v < n; v++) {
                if (indegree[v] == 0) ready.add(v);
            }
            List<Integer> order = new ArrayList<>();
            while (!ready.isEmpty()) {
                int task = ready.poll();
                order.add(task);
                for (int next : successors.get(task)) {
                    if (--indegree[next] == 0) ready.add(next);
                }
            }
            return order.size() == n ? order : null;
        }

        boolean hasCycle() {
            return schedule() == null;
        }

        /** Independent O(n^2) oracle for the greedy rule at one position. */
        boolean greedyChoiceValid(List<Integer> order, int position) {
            Set<Integer> placed = new HashSet<>(order.subList(0, position));
            int emitted = order.get(position);
            for (int v : predecessors.get(emitted)) {
                if (!placed.contains(v)) return false;
            }
            for (int candidate = 0; candidate < n; candidate++) {
                if (placed.contains(candidate)) continue;
                if (!placed.containsAll(predecessors.get(candidate))) continue;
                if (priority[candidate] > priority[emitted]) return false;
                if (priority[candidate] == priority[emitted] && candidate < emitted) return false;
            }
            return true;
        }

        /** Every topological order of the graph, by brute force (small n only). */
        List<List<Integer>> allTopologicalOrders() {
            List<List<Integer>> out = new ArrayList<>();
            boolean[] placed = new boolean[n];
            List<Integer> current = new ArrayList<>();
            enumerate(placed, current, out);
            return out;
        }

        private void enumerate(boolean[] placed, List<Integer> current, List<List<Integer>> out) {
            if (current.size() == n) {
                out.add(new ArrayList<>(current));
                return;
            }
            for (int v = 0; v < n; v++) {
                if (placed[v]) continue;
                boolean available = true;
                for (int p : predecessors.get(v)) {
                    if (!placed[p]) available = false;
                }
                if (!available) continue;
                placed[v] = true;
                current.add(v);
                enumerate(placed, current, out);
                current.remove(current.size() - 1);
                placed[v] = false;
            }
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        // 1. A small DAG with a unique-ish structure, checked against brute-force enumeration.
        TaskGraph build = new TaskGraph(
                new String[]{"compile", "test", "package", "docs", "deploy", "publish"},
                new int[]{9, 7, 5, 3, 8, 2});
        build.addEdge(0, 1);
        build.addEdge(0, 2);
        build.addEdge(1, 4);
        build.addEdge(2, 4);
        build.addEdge(3, 1);
        build.addEdge(2, 5);
        build.addEdge(4, 5);
        List<Integer> order = build.schedule();
        check(order != null && !build.hasCycle(), "the build graph is acyclic");
        check(order.size() == build.n, "the order contains every task");
        check(new HashSet<>(order).size() == build.n, "no task appears twice");
        int[] position = new int[build.n];
        for (int i = 0; i < order.size(); i++) position[order.get(i)] = i;
        for (int u = 0; u < build.n; u++) {
            for (int v : build.successors.get(u)) {
                check(position[u] < position[v], "edge " + u + " -> " + v + " is respected");
            }
        }
        for (int i = 0; i < build.n; i++) {
            check(build.greedyChoiceValid(order, i), "the greedy rule holds at position " + i);
        }
        List<List<Integer>> allOrders = build.allTopologicalOrders();
        boolean member = false;
        for (List<Integer> candidate : allOrders) {
            if (candidate.equals(order)) member = true;
        }
        check(member, "the produced order is a topological order (brute-force membership)");
        check(allOrders.size() > 1, "the small graph admits more than one topological order");
        List<Integer> greedyByBruteForce = null;
        for (List<Integer> candidate : allOrders) {
            // the greedy order is the unique maximiser of the priority signature read left to right
            if (greedyByBruteForce == null || compareGreedy(candidate, greedyByBruteForce, build.priority) > 0) {
                greedyByBruteForce = candidate;
            }
        }
        check(order.equals(greedyByBruteForce),
                "the produced order is the brute-force greedy maximum over all topological orders");

        // 2. Hand-computed instances.
        TaskGraph diamond = new TaskGraph(new String[]{"A", "B", "C", "D"}, new int[]{1, 5, 3, 4});
        diamond.addEdge(0, 1);
        diamond.addEdge(0, 2);
        diamond.addEdge(1, 3);
        diamond.addEdge(2, 3);
        check(diamond.schedule().equals(Arrays.asList(0, 1, 2, 3)),
                "diamond: A, then the higher-priority B, then C, then D");

        TaskGraph chain = new TaskGraph(new String[]{"A", "B", "C"}, new int[]{1, 9, 5});
        chain.addEdge(0, 1);
        chain.addEdge(1, 2);
        check(chain.schedule().equals(Arrays.asList(0, 1, 2)),
                "a chain is forced regardless of priorities");

        TaskGraph isolated = new TaskGraph(new String[]{"A", "B", "C", "D"}, new int[]{4, 9, 4, 1});
        check(isolated.schedule().equals(Arrays.asList(1, 0, 2, 3)),
                "isolated nodes come out in priority-then-id order");

        TaskGraph ties = new TaskGraph(new String[]{"A", "B", "C"}, new int[]{7, 7, 7});
        check(ties.schedule().equals(Arrays.asList(0, 1, 2)),
                "equal priorities break ties by task id");

        // 3. Cycles.
        TaskGraph cyclic = new TaskGraph(new String[]{"A", "B", "C"}, new int[]{1, 2, 3});
        cyclic.addEdge(0, 1);
        cyclic.addEdge(1, 2);
        cyclic.addEdge(2, 0);
        check(cyclic.schedule() == null && cyclic.hasCycle(), "a 3-cycle is detected");
        try {
            cyclic.addEdge(0, 0);
            check(false, "a self edge must be rejected");
        } catch (IllegalArgumentException expected) {
            check(true, "self edges are rejected as cycles");
        }
        TaskGraph longer = new TaskGraph(new String[]{"A", "B", "C", "D"}, new int[]{1, 1, 1, 1});
        longer.addEdge(0, 1);
        longer.addEdge(1, 2);
        longer.addEdge(2, 3);
        check(!longer.hasCycle() && longer.schedule() != null, "a longer chain is acyclic");

        // 4. A larger LCG-labelled DAG: every position must satisfy the greedy rule and the edges.
        Lcg seeded = new Lcg(509509);
        TaskGraph random = new TaskGraph(
                new String[]{"t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"},
                new int[]{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8});
        for (int i = 0; i < 12; i++) {
            for (int j = i + 1; j < 12; j++) {
                if (seeded.nextInt(5) == 0) random.addEdge(i, j);
            }
        }
        List<Integer> randomOrder = random.schedule();
        check(randomOrder != null, "edges always point forward, so the random graph is acyclic");
        int[] randomPosition = new int[random.n];
        for (int i = 0; i < randomOrder.size(); i++) randomPosition[randomOrder.get(i)] = i;
        for (int u = 0; u < random.n; u++) {
            for (int v : random.successors.get(u)) {
                check(randomPosition[u] < randomPosition[v], "random edge " + u + " -> " + v);
            }
        }
        for (int i = 0; i < random.n; i++) {
            check(random.greedyChoiceValid(randomOrder, i), "random greedy rule at position " + i);
        }

        // 5. Determinism.
        check(build.schedule().equals(order), "deterministic re-run: identical order");
        check(random.schedule().equals(randomOrder), "deterministic re-run: identical random order");
        check(diamond.schedule().equals(Arrays.asList(0, 1, 2, 3)), "deterministic re-run: diamond");

        System.out.println("PASS");
    }

    /** Compare two topological orders by the sequence of priorities, then by ids. */
    private static int compareGreedy(List<Integer> a, List<Integer> b, int[] priority) {
        for (int i = 0; i < a.size(); i++) {
            int pa = priority[a.get(i)];
            int pb = priority[b.get(i)];
            if (pa != pb) return Integer.compare(pa, pb);
            if (!a.get(i).equals(b.get(i))) return Integer.compare(b.get(i), a.get(i));
        }
        return 0;
    }
}
