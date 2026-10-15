/** Task 357 - Deadlock detection for resource allocation graph (algorithm). (Tier: Intermediate, CTCI Ch 15) */
import java.util.*;

public class Solution {
    static final int SCRIPT_TRIALS = 300;

    static List<Integer> canonicalCycle(List<Integer> cycle) {
        int minIndex = 0;
        for (int i = 1; i < cycle.size(); i++) if (cycle.get(i) < cycle.get(minIndex)) minIndex = i;
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < cycle.size(); i++) result.add(cycle.get((minIndex + i) % cycle.size()));
        return result;
    }

    /**
     * Collapse the resource allocation graph into the wait-for graph: a request (p, r)
     * with r held by q becomes the edge p -> q; a request for a free resource adds nothing.
     */
    static List<List<Integer>> buildWaitFor(int processes, int resources, int[][] holds, int[][] requests) {
        int[] holder = new int[resources];
        Arrays.fill(holder, -1);
        for (int[] pair : holds) {
            if (holder[pair[1]] != -1 && holder[pair[1]] != pair[0]) {
                throw new IllegalArgumentException("resource held by more than one process: " + pair[1]);
            }
            holder[pair[1]] = pair[0];
        }
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < processes; i++) adjacency.add(new ArrayList<>());
        for (int[] pair : requests) {
            int currentHolder = holder[pair[1]];
            if (currentHolder != -1) adjacency.get(pair[0]).add(currentHolder);
        }
        return adjacency;
    }

    /** Returns a canonical cycle or null. 0 = white, 1 = gray (on the DFS path), 2 = black. */
    static List<Integer> findCycle(int processes, List<List<Integer>> adjacency) {
        int[] color = new int[processes];
        List<Integer> path = new ArrayList<>();
        for (int start = 0; start < processes; start++) {
            if (color[start] == 0) {
                List<Integer> cycle = dfsCycle(start, adjacency, color, path);
                if (cycle != null) return canonicalCycle(cycle);
            }
        }
        return null;
    }

    static List<Integer> dfsCycle(int node, List<List<Integer>> adjacency, int[] color, List<Integer> path) {
        color[node] = 1;
        path.add(node);
        for (int next : adjacency.get(node)) {
            if (color[next] == 1) {                       // back edge: the path slice is a cycle
                int start = path.indexOf(next);
                return new ArrayList<>(path.subList(start, path.size()));
            }
            if (color[next] == 0) {
                List<Integer> cycle = dfsCycle(next, adjacency, color, path);
                if (cycle != null) return cycle;
            }
        }
        path.remove(path.size() - 1);
        color[node] = 2;
        return null;
    }

    /** Independent check: a cycle exists iff some Kosaraju SCC has size > 1 or a self-loop exists. */
    static boolean hasCycleViaScc(int processes, List<List<Integer>> adjacency) {
        List<List<Integer>> reverse = new ArrayList<>();
        for (int i = 0; i < processes; i++) reverse.add(new ArrayList<>());
        for (int u = 0; u < processes; u++) {
            for (int v : adjacency.get(u)) reverse.get(v).add(u);
        }

        boolean[] visited = new boolean[processes];
        List<Integer> order = new ArrayList<>();
        for (int u = 0; u < processes; u++) if (!visited[u]) dfsFinish(u, adjacency, visited, order);

        int[] component = new int[processes];
        Arrays.fill(component, -1);
        List<Integer> sizes = new ArrayList<>();
        for (int i = order.size() - 1; i >= 0; i--) {
            int u = order.get(i);
            if (component[u] != -1) continue;
            int groupIndex = sizes.size();
            component[u] = groupIndex;
            int size = 0;
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(u);
            while (!stack.isEmpty()) {
                int x = stack.pop();
                size++;
                for (int y : reverse.get(x)) {
                    if (component[y] == -1) {
                        component[y] = groupIndex;
                        stack.push(y);
                    }
                }
            }
            sizes.add(size);
        }

        for (int size : sizes) if (size > 1) return true;
        for (int u = 0; u < processes; u++) if (adjacency.get(u).contains(u)) return true;
        return false;
    }

    static void dfsFinish(int u, List<List<Integer>> adjacency, boolean[] visited, List<Integer> order) {
        visited[u] = true;
        for (int v : adjacency.get(u)) if (!visited[v]) dfsFinish(v, adjacency, visited, order);
        order.add(u);
    }

    static void assertValidCycle(List<List<Integer>> adjacency, List<Integer> cycle) {
        check(canonicalCycle(cycle).equals(cycle), "cycle returned canonicalised");
        check(new HashSet<>(cycle).size() == cycle.size(), "cycle vertices are distinct");
        for (int i = 0; i < cycle.size(); i++) {
            int next = cycle.get((i + 1) % cycle.size());
            check(adjacency.get(cycle.get(i)).contains(next), "cycle edges exist");
        }
    }

    static class Rag {
        final int[][] holds;
        final int[][] requests;
        Rag(int[][] holds, int[][] requests) { this.holds = holds; this.requests = requests; }
    }

    static long seed = 13579;

    /** LCG RAG: every resource gets one holder or stays free; processes issue 0..2 requests. */
    static Rag randomRag(int processes, int resources) {
        int[] holder = new int[resources];
        Arrays.fill(holder, -1);
        List<int[]> holds = new ArrayList<>();
        for (int resource = 0; resource < resources; resource++) {
            seed = (seed * 75 + 74) % 65537;
            if (seed % 3 != 0) {
                int process = (int) (seed % processes);
                holder[resource] = process;
                holds.add(new int[]{process, resource});
            }
        }
        List<int[]> requests = new ArrayList<>();
        for (int process = 0; process < processes; process++) {
            seed = (seed * 75 + 74) % 65537;
            int count = (int) (seed % 3);
            for (int i = 0; i < count; i++) {
                seed = (seed * 75 + 74) % 65537;
                requests.add(new int[]{process, (int) (seed % resources)});
            }
        }
        return new Rag(holds.toArray(new int[0][]), requests.toArray(new int[0][]));
    }

    public static void main(String[] args) {
        // Hand-built deadlock: two processes wait for each other; canonical cycle [0, 1].
        List<List<Integer>> adjacency = buildWaitFor(2, 2,
                new int[][]{{0, 0}, {1, 1}}, new int[][]{{0, 1}, {1, 0}});
        check(adjacency.equals(List.of(List.of(1), List.of(0))), "wait-for edges");
        check(findCycle(2, adjacency).equals(List.of(0, 1)), "classic two-process deadlock");

        // Acyclic: process 0 waits for process 1, which waits for nothing.
        adjacency = buildWaitFor(2, 2, new int[][]{{0, 0}, {1, 1}}, new int[][]{{0, 1}});
        check(adjacency.equals(List.of(List.of(1), List.of())), "one-way wait edge");
        check(findCycle(2, adjacency) == null, "no cycle without a wait cycle");

        // Self-loop: a process requests a resource it already holds.
        adjacency = buildWaitFor(1, 1, new int[][]{{0, 0}}, new int[][]{{0, 0}});
        check(adjacency.equals(List.of(List.of(0))), "self-loop edge");
        check(findCycle(1, adjacency).equals(List.of(0)), "self-loop is detected");

        // Three-process cycle; the returned cycle is the hand-checked canonical rotation.
        adjacency = buildWaitFor(3, 3, new int[][]{{1, 0}, {2, 1}, {0, 2}},
                new int[][]{{0, 0}, {1, 1}, {2, 2}});
        check(adjacency.equals(List.of(List.of(1), List.of(2), List.of(0))), "three-cycle edges");
        check(findCycle(3, adjacency).equals(List.of(0, 1, 2)), "three-process deadlock");

        // Requests for free resources create no edges.
        adjacency = buildWaitFor(2, 2, new int[][]{{0, 0}}, new int[][]{{1, 1}});
        check(adjacency.equals(List.of(List.of(), List.of())), "free resource: no edge");
        check(findCycle(2, adjacency) == null, "waiting for a free resource is not a deadlock");

        // All processes idle, and an extra process outside the cycle.
        adjacency = buildWaitFor(3, 3, new int[][]{{0, 0}, {1, 1}, {2, 2}}, new int[0][]);
        check(findCycle(3, adjacency) == null, "idle processes never deadlock");
        adjacency = buildWaitFor(3, 3, new int[][]{{0, 0}, {1, 1}, {2, 2}},
                new int[][]{{0, 1}, {1, 0}});
        check(findCycle(3, adjacency).equals(List.of(0, 1)), "cycle independent of an idle process");

        // A resource may have only one holder.
        try {
            buildWaitFor(2, 1, new int[][]{{0, 0}, {1, 0}}, new int[0][]);
            check(false, "a double-held resource must be rejected");
        } catch (IllegalArgumentException expected) {
            // documented rejection
        }

        // Cross-check cycle existence against the independent SCC check on LCG RAGs.
        seed = 13579;
        int cyclesSeen = 0;
        for (int trial = 0; trial < SCRIPT_TRIALS; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int processes = 2 + (int) (seed % 6);
            seed = (seed * 75 + 74) % 65537;
            int resources = 1 + (int) (seed % 6);
            Rag rag = randomRag(processes, resources);
            List<List<Integer>> waitFor = buildWaitFor(processes, resources, rag.holds, rag.requests);
            List<Integer> cycle = findCycle(processes, waitFor);
            boolean sccSaysCycle = hasCycleViaScc(processes, waitFor);
            check((cycle != null) == sccSaysCycle, "SCC cross-check at trial " + trial);
            if (cycle != null) {
                cyclesSeen++;
                assertValidCycle(waitFor, cycle);
            }
        }
        check(cyclesSeen >= 20, "LCG graphs should contain deadlocks (saw " + cyclesSeen + ")");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
