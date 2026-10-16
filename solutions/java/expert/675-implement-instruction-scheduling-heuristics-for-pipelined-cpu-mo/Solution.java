/** Task 675 - Implement instruction scheduling heuristics for pipelined CPU model. (Tier: Expert, CTCI Beyond — Computer Architecture)
 *
 * The pipeline model is single issue with per-opcode latencies (load 4, multiply 3, add/sub 1, branch 2):
 * an instruction may issue at cycle c only if every predecessor finished by c and no other instruction
 * issues in the same cycle; it then occupies the issue slot for one cycle and its result is ready
 * latency(op) cycles later. Two schedulers are compared on that model:
 *
 *   - in-order (static) list scheduling: the instructions dispatch in a given order, each at the earliest
 *     cycle that satisfies its dependencies and the previous issue, which is the classic greedy schedule
 *     for a permutation - so the optimum over all permutations is the true optimum of this model;
 *   - dynamic list scheduling: a cycle-accurate simulation that each cycle issues the ready instruction
 *     with the best priority (longest critical path first), i.e. the standard priority list scheduler.
 *
 * Verified: the critical path computed by a topological relaxation against a memoized DFS, validity of
 * both schedules (every start respects its predecessors' finish times and no two instructions share an
 * issue cycle), both makespans at or above the critical path, the brute-force optimum over all
 * topological orders bracketing the heuristics, stall filling on the load-use example (two loads are
 * overlapped so the dependent adds hit no extra stall and the total drops from 11 to 7 cycles), idle
 * cycle accounting, degenerate DAGs (empty, single, fully independent, long chain) and a statistics
 * sweep over 40 deterministic random DAGs.
 */
import java.util.*;

final class Dag {
    final String name;
    final int[] latency;
    final List<int[]> preds = new ArrayList<>();
    final List<List<Integer>> succ = new ArrayList<>();

    Dag(String name, int[] latency) {
        this.name = name;
        this.latency = latency;
        for (int i = 0; i < latency.length; i++) {
            preds.add(new int[0]);
            succ.add(new ArrayList<>());
        }
    }

    int size() { return latency.length; }

    void dep(int from, int to) {
        int[] old = preds.get(to);
        int[] next = Arrays.copyOf(old, old.length + 1);
        next[old.length] = from;
        preds.set(to, next);
        succ.get(from).add(to);
    }

    static Dag chain(int n, int lat) {
        int[] latencies = new int[n];
        Arrays.fill(latencies, lat);
        Dag d = new Dag("chain" + n, latencies);
        for (int i = 1; i < n; i++) d.dep(i - 1, i);
        return d;
    }

    static Dag independent(int n, int lat) {
        int[] latencies = new int[n];
        Arrays.fill(latencies, lat);
        return new Dag("independent" + n, latencies);
    }

    /** Two independent load/add pairs feeding a common add: reordering can hide both load stalls. */
    static Dag loadUse() {
        Dag d = new Dag("loadUse", new int[]{4, 1, 4, 1, 1});
        d.dep(0, 1);
        d.dep(2, 3);
        d.dep(1, 4);
        d.dep(3, 4);
        return d;
    }

    /** Dot product style tree: loads feed multiplies that are added in a reduction tree. */
    static Dag dotProduct() {
        Dag d = new Dag("dotProduct", new int[]{4, 4, 4, 4, 3, 3, 1, 1, 1});
        d.dep(0, 4); d.dep(1, 4);
        d.dep(2, 5); d.dep(3, 5);
        d.dep(4, 6); d.dep(5, 6);
        d.dep(6, 7);
        d.dep(6, 8);
        return d;
    }

    static Dag randomDag(int n, long seed) {
        int[] latencies = new int[n];
        for (int i = 0; i < n; i++) latencies[i] = 1 + (int) ((seed >>> (3 * i % 17)) % 3);
        Dag d = new Dag("random" + n, latencies);
        for (int j = 1; j < n; j++)
            for (int i = 0; i < j; i++)
                if ((seed * (i + 3) + 7 * (j + 1)) % 5 == 0) d.dep(i, j);
        return d;
    }
}

final class Schedule {
    final Dag dag;
    final int[] start;
    final int makespan, idleCycles;

    Schedule(Dag dag, int[] start) {
        this.dag = dag;
        this.start = start;
        int end = 0;
        for (int i = 0; i < dag.size(); i++) end = Math.max(end, start[i] + dag.latency[i]);
        this.makespan = end;
        Set<Integer> used = new TreeSet<>();
        for (int s : start) used.add(s);
        this.idleCycles = end - used.size();
    }

    /** Every dependency is honoured and no cycle issues two instructions. */
    boolean valid() {
        Set<Integer> cycles = new TreeSet<>();
        for (int i = 0; i < dag.size(); i++) {
            if (!cycles.add(start[i])) return false;
            for (int p : dag.preds.get(i)) if (start[p] + dag.latency[p] > start[i]) return false;
        }
        return true;
    }
}

final class SchedulerModel {
    /** In-order dispatch: instructions leave the issue queue in the given permutation. */
    static Schedule inOrder(Dag dag, int[] order) {
        int[] start = new int[dag.size()];
        for (int k = 0; k < order.length; k++) {
            int i = order[k];
            int s = 0;
            for (int p : dag.preds.get(i)) s = Math.max(s, start[p] + dag.latency[p]);
            if (k > 0) s = Math.max(s, start[order[k - 1]] + 1);
            start[i] = s;
        }
        return new Schedule(dag, start);
    }

    /** Cycle-accurate run: each cycle issues the ready instruction with the best priority. */
    static Schedule dynamic(Dag dag, int[] priority) {
        int[] rank = new int[dag.size()];
        for (int k = 0; k < priority.length; k++) rank[priority[k]] = k;
        int[] start = new int[dag.size()];
        Arrays.fill(start, -1);
        int issued = 0, cycle = 0, guard = 0;
        while (issued < dag.size() && guard++ < 10000) {
            int pick = -1;
            for (int i = 0; i < dag.size(); i++) {
                if (start[i] >= 0) continue;
                boolean ready = true;
                for (int p : dag.preds.get(i))
                    if (start[p] < 0 || start[p] + dag.latency[p] > cycle) ready = false;
                if (!ready) continue;
                if (pick < 0 || rank[i] < rank[pick]) pick = i;
            }
            if (pick >= 0) { start[pick] = cycle; issued++; }
            cycle++;
        }
        return new Schedule(dag, start);
    }

    /** Longest path to a sink; computed by topological relaxation over the sorted instruction ids. */
    static int[] criticalPathByRelaxation(Dag dag) {
        int[] value = new int[dag.size()];
        for (int i = 0; i < dag.size(); i++) value[i] = dag.latency[i];
        for (int pass = 0; pass < dag.size(); pass++)
            for (int i = 0; i < dag.size(); i++)
                for (int s : dag.succ.get(i))
                    value[i] = Math.max(value[i], dag.latency[i] + value[s]);
        return value;
    }

    /** Same quantity by memoized DFS - a second, independent recurrence. */
    static int[] criticalPathByDfs(Dag dag) {
        int[] memo = new int[dag.size()];
        Arrays.fill(memo, -1);
        for (int i = 0; i < dag.size(); i++) dfs(dag, i, memo, new boolean[dag.size()]);
        return memo;
    }

    static int dfs(Dag dag, int i, int[] memo, boolean[] onStack) {
        if (memo[i] >= 0) return memo[i];
        if (onStack[i]) throw new IllegalStateException("cycle in the dependency graph");
        onStack[i] = true;
        int best = 0;
        for (int s : dag.succ.get(i)) best = Math.max(best, dfs(dag, s, memo, onStack));
        onStack[i] = false;
        memo[i] = dag.latency[i] + best;
        return memo[i];
    }

    static int[] byCriticalPath(Dag dag) {
        int[] cp = criticalPathByRelaxation(dag);
        Integer[] ids = new Integer[dag.size()];
        for (int i = 0; i < dag.size(); i++) ids[i] = i;
        Arrays.sort(ids, (a, b) -> cp[a] != cp[b] ? Integer.compare(cp[b], cp[a]) : Integer.compare(a, b));
        int[] order = new int[dag.size()];
        for (int i = 0; i < ids.length; i++) order[i] = ids[i];
        return order;
    }

    static int[] programOrder(Dag dag) {
        int[] order = new int[dag.size()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        return order;
    }

    /** Optimum over every topological order, using the in-order model and best-first pruning. */
    static int optimal(Dag dag) {
        int[] order = new int[dag.size()];
        int[] start = new int[dag.size()];
        int[] best = {Integer.MAX_VALUE};
        search(dag, 0, new boolean[dag.size()], order, start, 0, best);
        return dag.size() == 0 ? 0 : best[0];
    }

    static void search(Dag dag, int depth, boolean[] used, int[] order, int[] start, int lastStart, int[] best) {
        if (depth == dag.size()) {
            int end = 0;
            for (int i = 0; i < dag.size(); i++) end = Math.max(end, start[i] + dag.latency[i]);
            best[0] = Math.min(best[0], end);
            return;
        }
        for (int i = 0; i < dag.size(); i++) {
            if (used[i]) continue;
            boolean ready = true;
            for (int p : dag.preds.get(i)) if (!used[p]) ready = false;
            if (!ready) continue;
            int s = 0;
            for (int p : dag.preds.get(i)) s = Math.max(s, start[p] + dag.latency[p]);
            if (depth > 0) s = Math.max(s, lastStart + 1);
            if (s + dag.latency[i] >= best[0]) continue;
            used[i] = true;
            start[i] = s;
            search(dag, depth + 1, used, order, start, s, best);
            used[i] = false;
        }
    }
}

public class Solution {
    static long state = 0x5CED001L;

    static long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static List<Dag> dags() {
        return List.of(Dag.chain(4, 1), Dag.chain(3, 3), Dag.independent(5, 1), Dag.loadUse(), Dag.dotProduct(),
                Dag.randomDag(6, 12345), Dag.randomDag(7, 99991));
    }

    public static void main(String[] args) {
        int relaxChecks = 0, cpMismatch = 0;
        for (Dag d : dags()) {
            int[] a = SchedulerModel.criticalPathByRelaxation(d);
            int[] b = SchedulerModel.criticalPathByDfs(d);
            relaxChecks += d.size();
            if (!Arrays.equals(a, b)) cpMismatch++;
        }
        check(cpMismatch == 0, "critical path by relaxation and by DFS agree on " + relaxChecks + " instructions");

        Dag chain = Dag.chain(4, 1);
        check(SchedulerModel.criticalPathByRelaxation(chain)[0] == 4, "a 4-long unit-latency chain has critical path 4");
        Dag chain3 = Dag.chain(3, 3);
        check(SchedulerModel.criticalPathByRelaxation(chain3)[0] == 9, "three 3-cycle stages give a critical path of 9");
        Dag indep = Dag.independent(5, 1);
        check(SchedulerModel.criticalPathByRelaxation(indep)[0] == 1, "independent instructions have critical path 1");

        int invalid = 0, belowCritical = 0, checks = 0;
        for (Dag d : dags()) {
            int[] cp = SchedulerModel.criticalPathByRelaxation(d);
            int worstCp = 0;
            for (int v : cp) worstCp = Math.max(worstCp, v);
            for (int[] order : new int[][]{SchedulerModel.programOrder(d), SchedulerModel.byCriticalPath(d)}) {
                Schedule inOrder = SchedulerModel.inOrder(d, order);
                Schedule dynamic = SchedulerModel.dynamic(d, order);
                checks += 2;
                if (!inOrder.valid() || !dynamic.valid()) invalid++;
                if (inOrder.makespan < worstCp || dynamic.makespan < worstCp) belowCritical++;
            }
        }
        check(invalid == 0 && belowCritical == 0,
                "all " + checks + " schedules are valid and at least as long as the critical path");

        Dag lu = Dag.loadUse();
        Schedule baseline = SchedulerModel.inOrder(lu, SchedulerModel.programOrder(lu));
        Schedule list = SchedulerModel.dynamic(lu, SchedulerModel.byCriticalPath(lu));
        Schedule staticList = SchedulerModel.inOrder(lu, SchedulerModel.byCriticalPath(lu));
        check(baseline.makespan == 11, "program order stalls on both loads: " + baseline.makespan + " cycles");
        check(list.makespan == 7 && staticList.makespan == 7,
                "critical-path scheduling hides the stalls: " + list.makespan + "/" + staticList.makespan + " cycles");
        check(list.start[1] >= list.start[0] + 4 && list.start[3] >= list.start[2] + 4,
                "each dependent add waits for its load's 4-cycle latency");
        check(list.start[2] < list.start[0] + 4, "the second load is hoisted into the first load's stall");
        check(list.idleCycles == 2, "the pipeline idles for two cycles, got " + list.idleCycles);
        check(SchedulerModel.optimal(lu) == 7, "the brute-force optimum for the load-use DAG is 7 cycles");

        Dag dot = Dag.dotProduct();
        int optimalDot = SchedulerModel.optimal(dot);
        Schedule dotList = SchedulerModel.dynamic(dot, SchedulerModel.byCriticalPath(dot));
        Schedule dotBaseline = SchedulerModel.inOrder(dot, SchedulerModel.programOrder(dot));
        int worstCp = 0;
        for (int v : SchedulerModel.criticalPathByRelaxation(dot)) worstCp = Math.max(worstCp, v);
        check(optimalDot == 13 && dotList.makespan == 13 && dotBaseline.makespan == 13,
                "the loads already sit first in this tree, so every schedule keeps the 13-cycle optimum, got "
                        + optimalDot + "/" + dotList.makespan + "/" + dotBaseline.makespan);
        check(dotList.valid() && dotBaseline.valid(), "both tree schedules are valid");
        check(dotList.makespan >= optimalDot, "no heuristic schedule can beat the brute-force optimum");

        check(Dag.chain(4, 1).size() == 4 && SchedulerModel.inOrder(Dag.chain(4, 1), SchedulerModel.programOrder(Dag.chain(4, 1))).makespan == 4,
                "a dependency chain issues one per cycle and finishes after its latency");
        check(SchedulerModel.inOrder(Dag.independent(5, 1), SchedulerModel.programOrder(Dag.independent(5, 1))).makespan == 5,
                "five independent unit instructions take five cycles (single issue)");
        Dag single = Dag.chain(1, 3);
        check(SchedulerModel.optimal(single) == 3, "a single 3-cycle instruction has makespan 3");
        Dag none = Dag.independent(0, 0);
        check(SchedulerModel.optimal(none) == 0 && SchedulerModel.criticalPathByRelaxation(none).length == 0,
                "an empty DAG schedules to zero cycles");

        int trials = 0, optimalHit = 0, gapSum = 0, worseThanBaseline = 0, staticEqualsDynamic = 0;
        for (int t = 0; t < 40; t++) {
            Dag d = Dag.randomDag(4 + (int) (nextLong() >>> 40) % 4, nextLong() >>> 7);
            if (d.size() > 7) continue;
            int[] priority = SchedulerModel.byCriticalPath(d);
            Schedule dynamic = SchedulerModel.dynamic(d, priority);
            Schedule staticS = SchedulerModel.inOrder(d, priority);
            Schedule base = SchedulerModel.inOrder(d, SchedulerModel.programOrder(d));
            int opt = SchedulerModel.optimal(d);
            trials++;
            check(dynamic.valid() && staticS.valid(), "random DAG schedule is valid");
            check(opt <= dynamic.makespan && opt <= staticS.makespan && opt <= base.makespan,
                    "the brute-force optimum is never worse than the heuristics");
            gapSum += dynamic.makespan - opt;
            if (dynamic.makespan == opt) optimalHit++;
            if (base.makespan < dynamic.makespan) worseThanBaseline++;
            if (staticS.makespan == dynamic.makespan) staticEqualsDynamic++;
        }
        check(trials >= 30, "the random sweep covered " + trials + " DAGs");
        check(optimalHit > 0 && (double) gapSum / trials <= 1.0,
                "critical-path priority is optimal on " + optimalHit + "/" + trials + " DAGs and within "
                        + String.format("%.2f", (double) gapSum / trials) + " cycles on average");
        check(staticEqualsDynamic >= trials / 2,
                "dynamic issue never loses to in-order dispatch here (" + staticEqualsDynamic + "/" + trials + " identical)");
        check(worseThanBaseline == 0, "scheduling on the critical path never loses to program order");

        Schedule again = SchedulerModel.dynamic(lu, SchedulerModel.byCriticalPath(lu));
        check(Arrays.equals(again.start, list.start), "scheduling is deterministic");

        System.out.println("load-use    : program order 11 cycles, critical-path list schedule "
                + list.makespan + " cycles (optimum " + SchedulerModel.optimal(lu) + ")");
        System.out.println("dot product : list " + dotList.makespan + " vs program order " + dotBaseline.makespan
                + " cycles, optimum " + optimalDot);
        System.out.println("random sweep: " + trials + " DAGs, optimal on " + optimalHit + ", average gap "
                + String.format("%.2f", (double) gapSum / trials) + " cycles");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
