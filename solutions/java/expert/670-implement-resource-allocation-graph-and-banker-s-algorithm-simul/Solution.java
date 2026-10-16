/** Task 670 - Implement resource allocation graph and Banker's algorithm simulator. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Bankers - the classic deadlock-avoidance state (Available, Max, Allocation, Need) with the safety
 * algorithm. A request is granted only when it stays inside Need and Available *and* the resulting
 * state is safe, i.e. some order of the processes lets every one of them run to completion with the
 * resources currently free plus those released by the earlier finishers. Two independent oracles back
 * the greedy safety check: a backtracking search over process orders (the greedy commits to the first
 * viable candidate, the search backtracks, so agreement shows the arbitrary choice does not matter)
 * and a replay of any reported order which re-checks Need <= Work at every step.
 *
 * The resource allocation graph is the single-instance special case of the same state: process p holds
 * resource r when allocation[p][r] == 1 and wants it when need[p][r] == 1, so a wait-for edge p -> q
 * exists whenever p wants a resource that q holds. A cycle in that graph is a deadlock, and for
 * single-instance resources a deadlocked (unsafe) state is exactly a state with a wait-for cycle - that
 * equivalence is checked, not assumed.
 *
 * Verified: the textbook state and its safe sequence P1,P3,P4,P0,P2 (replayed), request rejection for
 * Need/Available violations and state-unchanged rollback, resource conservation (Available +
 * Allocation is invariant), the cyclic-graph deadlock example, degenerate empty states, 200 deterministic
 * random states where greedy safety, the backtracking oracle and the replay all agree, and a random
 * request stream that is granted only into safe states and still lets every process finish.
 */
import java.util.*;

final class Bankers {
    static final int OK = 0, EXCEEDS_NEED = 1, EXCEEDS_AVAILABLE = 2, UNSAFE = 3, INVALID = 4;

    final int n, m;                 // processes, resource types
    final int[] available;
    final int[][] max, allocation, need;
    final int[] total;              // total instances of each resource (conserved by grants)

    Bankers(int[] available, int[][] max, int[][] allocation, int[] total) {
        this.n = allocation.length;
        this.m = available.length;
        this.available = available.clone();
        this.max = deepCopy(max);
        this.allocation = deepCopy(allocation);
        this.need = new int[n][m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) need[i][j] = max[i][j] - allocation[i][j];
        this.total = total.clone();
    }

    static int[][] deepCopy(int[][] src) {
        int[][] copy = new int[src.length][];
        for (int i = 0; i < src.length; i++) copy[i] = src[i].clone();
        return copy;
    }

    /** Flattened copy of the whole state, used to prove a rejected request changes nothing. */
    int[] flat() {
        int[] out = new int[m + n * m * 2];
        System.arraycopy(available, 0, out, 0, m);
        for (int i = 0; i < n; i++) System.arraycopy(allocation[i], 0, out, m + i * m, m);
        for (int i = 0; i < n; i++) System.arraycopy(need[i], 0, out, m + n * m + i * m, m);
        return out;
    }

    boolean stateEquals(int[] snapshot) { return Arrays.equals(flat(), snapshot); }

    static boolean leq(int[] a, int[] b) {
        for (int j = 0; j < a.length; j++) if (a[j] > b[j]) return false;
        return true;
    }

    /** Resource conservation: what is free plus what is allocated never changes. */
    boolean conserves() {
        int[] used = available.clone();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) used[j] += allocation[i][j];
        return Arrays.equals(used, total);
    }

    /** Greedy safety algorithm; returns a safe order or null when the state is unsafe. */
    List<Integer> safeSequence() {
        int[] work = available.clone();
        boolean[] done = new boolean[n];
        List<Integer> order = new ArrayList<>();
        for (int step = 0; step < n; step++) {
            int pick = -1;
            for (int i = 0; i < n && pick < 0; i++)
                if (!done[i] && leq(need[i], work)) pick = i;
            if (pick < 0) return null;
            done[pick] = true;
            order.add(pick);
            for (int j = 0; j < m; j++) work[j] += allocation[pick][j];
        }
        return order;
    }

    /** Backtracking oracle: does *some* process order complete? (greedy never backtracks) */
    List<Integer> searchSequence() {
        int[] order = new int[n];
        boolean[] used = new boolean[n];
        return search(0, available.clone(), used, order) ? box(order) : null;
    }

    private boolean search(int depth, int[] work, boolean[] used, int[] order) {
        if (depth == n) return true;
        for (int i = 0; i < n; i++) {
            if (used[i] || !leq(need[i], work)) continue;
            used[i] = true;
            order[depth] = i;
            int[] next = work.clone();
            for (int j = 0; j < m; j++) next[j] += allocation[i][j];
            if (search(depth + 1, next, used, order)) return true;
            used[i] = false;
        }
        return false;
    }

    private static List<Integer> box(int[] order) {
        List<Integer> list = new ArrayList<>();
        for (int v : order) list.add(v);
        return list;
    }

    /** Replays a claimed order from scratch, re-checking the Need <= Work condition at every step. */
    boolean replayValid(List<Integer> order) {
        if (order.size() != n) return false;
        boolean[] seen = new boolean[n];
        int[] work = available.clone();
        for (int p : order) {
            if (p < 0 || p >= n || seen[p] || !leq(need[p], work)) return false;
            seen[p] = true;
            for (int j = 0; j < m; j++) work[j] += allocation[p][j];
        }
        return true;
    }

    boolean needSatisfied(int p) {
        for (int j = 0; j < m; j++) if (need[p][j] > 0) return false;
        return true;
    }

    /** The process finishes and gives everything back; releasing can only raise Available, so the
     *  state stays safe (the safety argument is monotone in the work vector). */
    boolean complete(int p) {
        if (p < 0 || p >= n || !needSatisfied(p)) return false;
        for (int j = 0; j < m; j++) { available[j] += allocation[p][j]; allocation[p][j] = 0; }
        return true;
    }

    /** Request algorithm: the state changes only when the request is granted. */
    int request(int p, int[] req) {
        if (p < 0 || p >= n || req.length != m) return INVALID;
        for (int v : req) if (v < 0) return INVALID;
        if (!leq(req, need[p])) return EXCEEDS_NEED;
        if (!leq(req, available)) return EXCEEDS_AVAILABLE;
        for (int j = 0; j < m; j++) {
            available[j] -= req[j];
            allocation[p][j] += req[j];
            need[p][j] -= req[j];
        }
        if (safeSequence() == null) {                       // roll back an unsafe grant
            for (int j = 0; j < m; j++) {
                available[j] += req[j];
                allocation[p][j] -= req[j];
                need[p][j] += req[j];
            }
            return UNSAFE;
        }
        return OK;
    }

    /** Single-instance wait-for graph: p -> q when p needs a resource that q holds. */
    int[] waitFor() {
        int[] edge = new int[n];
        Arrays.fill(edge, -1);
        for (int p = 0; p < n; p++)
            for (int q = 0; q < n; q++) {
                if (p == q) continue;
                for (int j = 0; j < m; j++)
                    if (need[p][j] == 1 && allocation[q][j] == 1 && edge[p] < 0) edge[p] = q;
            }
        return edge;
    }

    boolean hasWaitCycle() {
        int[] edge = waitFor();
        for (int start = 0; start < n; start++) {
            Set<Integer> path = new LinkedHashSet<>();
            int cur = start;
            while (cur >= 0 && path.add(cur)) cur = edge[cur];
            if (cur >= 0) return true;
        }
        return false;
    }

    static Bankers textbook() {
        return new Bankers(
                new int[]{3, 3, 2},
                new int[][]{{7, 5, 3}, {3, 2, 2}, {9, 0, 2}, {2, 2, 2}, {4, 3, 3}},
                new int[][]{{0, 1, 0}, {2, 0, 0}, {3, 0, 2}, {2, 1, 1}, {0, 0, 2}},
                new int[]{10, 5, 7});
    }

    static Bankers cyclic() {
        return new Bankers(new int[]{0, 0},
                new int[][]{{1, 1}, {1, 1}},
                new int[][]{{1, 0}, {0, 1}},
                new int[]{1, 1});
    }

    /** Acyclic wait-for chain P0 -> P1 -> P2 -> nothing: safe even though nothing is free right now. */
    static Bankers chained() {
        return new Bankers(new int[]{0, 0, 0},
                new int[][]{{1, 1, 0}, {0, 1, 1}, {0, 0, 1}},
                new int[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                new int[]{1, 1, 1});
    }
}

public class Solution {
    static long state = 0xC0FFEE11L;

    static long nextLong() {                        // splitmix64, fixed seed
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Random but reproducible state: allocation <= max, and Available is whatever is left over. */
    static Bankers randomState(int n, int m, int pool) {
        int[] available = new int[m];
        int[][] max = new int[n][m], allocation = new int[n][m];
        int[] total = new int[m];
        for (int j = 0; j < m; j++) total[j] = 1 + (int) (nextLong() >>> 40) % Math.max(1, pool);
        for (int j = 0; j < m; j++) available[j] = total[j];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) {
                int alloc = (int) (nextLong() >>> 41) % (available[j] + 1);
                allocation[i][j] = alloc;
                available[j] -= alloc;
                max[i][j] = alloc + (int) (nextLong() >>> 42) % 3;
            }
        return new Bankers(available, max, allocation, total);
    }

    /** Realistic starting point: nothing is allocated yet and every process declares a claim that
     *  fits inside the system, so the state is safe before the first request arrives. */
    static Bankers initialSafeState(int n, int m, int pool) {
        int[] available = new int[m], total = new int[m];
        int[][] max = new int[n][m], allocation = new int[n][m];
        for (int j = 0; j < m; j++) { total[j] = pool; available[j] = pool; }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) max[i][j] = (int) (nextLong() >>> 41) % (pool + 1);
        return new Bankers(available, max, allocation, total);
    }

    public static void main(String[] args) {
        Bankers b = Bankers.textbook();
        check(b.n == 5 && b.m == 3 && b.conserves(), "textbook state is consistent and conserves resources");
        List<Integer> safe = b.safeSequence();
        check(safe != null && b.replayValid(safe), "textbook state is safe, order " + safe);
        check(b.replayValid(List.of(1, 3, 4, 0, 2)), "the textbook order P1,P3,P4,P0,P2 replays cleanly");
        List<Integer> searched = b.searchSequence();
        check(searched != null && b.replayValid(searched), "backtracking oracle confirms the state is safe");
        check(Arrays.equals(b.available, new int[]{3, 3, 2}), "safety checks never mutate the state");

        int[] before = b.flat();
        check(b.request(4, new int[]{3, 3, 0}) == Bankers.UNSAFE,
                "granting P4's (3,3,0) would leave no order that finishes every process");
        check(b.stateEquals(before), "the unsafe request is rolled back exactly");
        check(b.request(1, new int[]{1, 0, 2}) == Bankers.OK, "P1 is granted (1,0,2)");
        check(Arrays.equals(b.available, new int[]{2, 3, 0}), "Available drops to (2,3,0), got "
                + Arrays.toString(b.available));
        check(b.conserves() && b.safeSequence() != null, "the state stays safe and conserves resources");

        before = b.flat();
        check(b.request(3, new int[]{0, 4, 0}) == Bankers.EXCEEDS_NEED, "P3's (0,4,0) exceeds its remaining need");
        check(b.request(4, new int[]{3, 3, 0}) == Bankers.EXCEEDS_AVAILABLE, "P4's (3,3,0) exceeds Available");
        check(b.stateEquals(before), "a rejected request leaves the state byte-identical");
        check(b.request(1, new int[]{9, 9, 9}) == Bankers.EXCEEDS_NEED
                        && b.request(2, new int[]{-1, 0, 0}) == Bankers.INVALID
                        && b.request(9, new int[]{0, 0, 0}) == Bankers.INVALID,
                "over-large, negative and out-of-range requests are all rejected");

        Bankers cyc = Bankers.cyclic();
        check(cyc.hasWaitCycle() && cyc.safeSequence() == null && cyc.searchSequence() == null,
                "the two-process cycle is deadlocked, unsafe and has no completion order");
        Bankers chain = Bankers.chained();
        check(!chain.hasWaitCycle() && chain.safeSequence() != null,
                "an acyclic wait-for graph is safe (no cycle means no deadlock)");
        check(Arrays.equals(chain.waitFor(), new int[]{1, 2, -1}), "wait-for edges follow the hold/need sets");

        Bankers none = new Bankers(new int[]{0, 0}, new int[0][2], new int[0][2], new int[]{0, 0});
        check(none.safeSequence().isEmpty() && none.searchSequence().isEmpty() && !none.hasWaitCycle(),
                "zero processes are trivially safe");
        Bankers nothing = new Bankers(new int[0], new int[][]{{}, {}, {}}, new int[][]{{}, {}, {}}, new int[0]);
        check(nothing.safeSequence().size() == 3 && nothing.conserves(), "zero resource types are trivially safe");

        int trials = 200, agree = 0, unsafeStates = 0, safeStates = 0;
        for (int trial = 0; trial < trials; trial++) {
            Bankers r = randomState(1 + (int) (nextLong() >>> 43) % 4, 1 + (int) (nextLong() >>> 44) % 3, 4);
            check(r.conserves(), "generated state conserves resources");
            List<Integer> greedy = r.safeSequence();
            List<Integer> oracle = r.searchSequence();
            if ((greedy == null) == (oracle == null)) agree++;
            if (greedy != null) {
                safeStates++;
                check(r.replayValid(greedy), "greedy order replays at trial " + trial);
            } else {
                unsafeStates++;
            }
        }
        check(agree == trials, "greedy safety and the backtracking oracle agree on all " + trials + " states");
        check(safeStates > 0 && unsafeStates > 0, "the random sample covers safe and unsafe states");

        int singleInstance = 0, cycleStates = 0, equivalenceErrors = 0;
        for (int trial = 0; trial < 60; trial++) {
            int n = 2 + (int) (nextLong() >>> 47) % 4;
            int[] total = new int[n], available = new int[n];
            int[][] alloc = new int[n][n], max = new int[n][n];
            int[] perm = permutation(n);
            for (int i = 0; i < n; i++) {                 // resource perm[i] has exactly one holder
                alloc[i][perm[i]] = 1;
                max[i][perm[i]] = 1;
                total[perm[i]] = 1;
            }
            for (int i = 0; i < n; i++) max[i][(int) (nextLong() >>> 48) % n] = 1;   // claim one more
            Bankers s = new Bankers(available, max, alloc, total);
            boolean cyclic = s.hasWaitCycle(), safeState = s.safeSequence() != null;
            singleInstance++;
            if (cyclic) cycleStates++;
            if (cyclic == safeState) equivalenceErrors++;     // a cycle must mean an unsafe state
        }
        check(equivalenceErrors == 0 && cycleStates > 0 && cycleStates < singleInstance,
                "on single-instance graphs cycle == unsafe in all " + singleInstance + " states ("
                        + cycleStates + " cyclic)");

        boolean deadlockFree = true;
        int grants = 0, denials = 0, nonTrivialGrants = 0, completions = 0;
        Bankers run = initialSafeState(12, 3, 10);
        boolean[] done = new boolean[run.n];
        for (int step = 0; step < 4000; step++) {
            List<Integer> alive = new ArrayList<>();
            for (int i = 0; i < run.n; i++) if (!done[i]) alive.add(i);
            if (alive.isEmpty()) break;
            int p = alive.get((int) (nextLong() >>> 45) % alive.size());
            int[] req = new int[run.m];
            for (int j = 0; j < run.m; j++) {
                int cap = Math.min(run.need[p][j], run.available[j]);  // never over-ask or over-borrow
                if (cap > 0) req[j] = 1 + (int) ((nextLong() >>> 46) % cap);
            }
            int code = run.request(p, req);
            if (code == Bankers.OK) {
                grants++;
                int asked = 0;
                for (int v : req) asked += v;
                if (asked > 0) nonTrivialGrants++;
                if (run.safeSequence() == null || !run.conserves()) deadlockFree = false;
                if (run.needSatisfied(p)) { run.complete(p); done[p] = true; completions++; }
            } else {
                denials++;
            }
        }
        check(deadlockFree && grants > 20 && denials > 0 && nonTrivialGrants > 20 && completions > 0,
                "every granted request kept the state safe (" + grants + " grants, " + denials
                        + " rejections, " + nonTrivialGrants + " non-empty, " + completions + " completions)");
        List<Integer> finalOrder = run.safeSequence();
        check(finalOrder != null && run.replayValid(finalOrder),
                "after the random stream the system can still finish every process");

        System.out.println("textbook state     : safe sequence " + safe + ", grants (1,0,2) to P1 then Available "
                + Arrays.toString(b.available));
        System.out.println("random cross-check : " + trials + " states, greedy == backtracking oracle, "
                + safeStates + " safe / " + unsafeStates + " unsafe");
        System.out.println("single-instance    : " + singleInstance + " wait-for graphs, " + cycleStates
                + " cyclic, all cyclic graphs unsafe");
        System.out.println("random request run : " + grants + " grants, " + denials + " rejections, always safe");
        System.out.println("PASS");
    }

    static int[] permutation(int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) p[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = (int) ((nextLong() >>> 50) % (i + 1));
            int t = p[i]; p[i] = p[j]; p[j] = t;
        }
        return p;
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
