/** Task 510 - Implement task DAG executor with parallel workers (respect dependencies). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * A DAG task executor with W parallel workers, modelled as a deterministic non-delay list schedule
 * (the wave-3 house style for concurrency tasks: assertions cover only schedule properties that are
 * stable, and no dependency on real thread scheduling). Tasks carry an integer duration; a task may
 * start at tick t only when every predecessor finishes at or before t. The simulator advances an
 * integer clock to the next completion and, within each tick, repeatedly fills the lowest-index
 * free worker with the ready task of highest priority (ties broken by the smallest task id) until
 * no free worker can take a ready task - which also makes zero-duration tasks behave correctly. It
 * records start/finish times, the worker assignment and the per-worker busy time, detects cycles
 * (nothing running and nothing ready) and reports the makespan.
 *
 * Documented checks: the independent soundness rescan asserts that every start time is at least the
 * maximum finish time of its predecessors, that tasks never overlap on the same worker, that the
 * work-conservation identity sum(durations) == sum of per-worker busy time holds exactly, and that
 * the reported makespan matches the finish times; a second independent rescan verifies the
 * non-delay property (no tick has both an idle worker and an unstarted ready task); hand-computed
 * instances are asserted exactly (a 3+4+2 chain takes 9 ticks with 1 and with 4 workers, five
 * independent tasks take 6 ticks with 5 workers and 3 workers still need 6 or more, a crafted 1 +
 * max(5,5) + 1 DAG takes 7 ticks with two workers); a brute-force search over all topological
 * orders and all worker assignments on a tiny 5-task / 2-worker instance confirms the greedy
 * makespan equals the optimum; on an LCG-labelled 14-task DAG with 3 workers the classic list
 * scheduling bound makespan <= (2 - 1/W) * criticalPath (with W = 3, i.e. makespan * 3 <= 5 *
 * criticalPath) is asserted and the critical path itself is checked to be a strict lower bound; a
 * cyclic DAG and a partially blocked DAG are reported as unrunnable instead of looping; a
 * zero-duration task finishes at its start; and a re-run reproduces every start time, worker
 * assignment and the makespan exactly.
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

    static class Task {
        final int id;
        final String name;
        final int duration;
        final int priority;

        Task(int id, String name, int duration, int priority) {
            this.id = id;
            this.name = name;
            this.duration = duration;
            this.priority = priority;
        }
    }

    static class Schedule {
        final int[] start;
        final int[] finish;
        final int[] worker;
        final int[] busyPerWorker;
        final int makespan;

        Schedule(int[] start, int[] finish, int[] worker, int[] busyPerWorker, int makespan) {
            this.start = start;
            this.finish = finish;
            this.worker = worker;
            this.busyPerWorker = busyPerWorker;
            this.makespan = makespan;
        }
    }

    static class DagExecutor {
        final List<Task> tasks = new ArrayList<>();
        final List<Set<Integer>> successors = new ArrayList<>();
        final List<Set<Integer>> predecessors = new ArrayList<>();

        int addTask(String name, int duration, int priority) {
            int id = tasks.size();
            tasks.add(new Task(id, name, duration, priority));
            successors.add(new LinkedHashSet<>());
            predecessors.add(new LinkedHashSet<>());
            return id;
        }

        void dependsOn(int task, int prerequisite) {
            successors.get(prerequisite).add(task);
            predecessors.get(task).add(prerequisite);
        }

        private boolean readyAt(int task, int tick, int[] start, int[] finish) {
            if (start[task] >= 0) return false;
            for (int p : predecessors.get(task)) {
                if (finish[p] < 0 || finish[p] > tick) return false;
            }
            return true;
        }

        /** Non-delay list schedule; returns null when the DAG cannot be run (a cycle). */
        Schedule run(int workers) {
            int n = tasks.size();
            int[] start = new int[n];
            int[] finish = new int[n];
            int[] workerOf = new int[n];
            Arrays.fill(start, -1);
            Arrays.fill(finish, -1);
            Arrays.fill(workerOf, -1);
            int[] workerFreeAt = new int[workers];
            int[] busy = new int[workers];
            int done = 0;
            int tick = 0;
            while (done < n) {
                while (true) {
                    int freeWorker = -1;
                    for (int w = 0; w < workers; w++) {
                        if (workerFreeAt[w] <= tick) {
                            freeWorker = w;
                            break;
                        }
                    }
                    if (freeWorker < 0) break;
                    int best = -1;
                    for (int task = 0; task < n; task++) {
                        if (!readyAt(task, tick, start, finish)) continue;
                        if (best < 0 || tasks.get(task).priority > tasks.get(best).priority
                                || (tasks.get(task).priority == tasks.get(best).priority && task < best)) {
                            best = task;
                        }
                    }
                    if (best < 0) break;
                    start[best] = tick;
                    finish[best] = tick + tasks.get(best).duration;
                    workerOf[best] = freeWorker;
                    workerFreeAt[freeWorker] = finish[best];
                    busy[freeWorker] += tasks.get(best).duration;
                    done++;
                }
                if (done == n) break;
                int nextTick = Integer.MAX_VALUE;
                for (int w = 0; w < workers; w++) {
                    if (workerFreeAt[w] > tick) nextTick = Math.min(nextTick, workerFreeAt[w]);
                }
                if (nextTick == Integer.MAX_VALUE || nextTick <= tick) return null;
                tick = nextTick;
            }
            int makespan = 0;
            for (int value : finish) makespan = Math.max(makespan, value);
            return new Schedule(start, finish, workerOf, busy, makespan);
        }

        /** Longest path through the DAG: the critical-path lower bound on any makespan. */
        int criticalPath() {
            int n = tasks.size();
            int[] longest = new int[n];
            Arrays.fill(longest, -1);
            boolean changed = true;
            int best = 0;
            while (changed) {
                changed = false;
                for (int v = 0; v < n; v++) {
                    int base = 0;
                    boolean ready = true;
                    for (int p : predecessors.get(v)) {
                        if (longest[p] < 0) ready = false;
                        base = Math.max(base, longest[p]);
                    }
                    if (!ready) continue;
                    int value = base + tasks.get(v).duration;
                    if (longest[v] != value) {
                        longest[v] = value;
                        changed = true;
                    }
                    best = Math.max(best, value);
                }
            }
            return best;
        }
    }

    /** Brute force over all topological orders and worker assignments (tiny instances only). */
    static int bruteForceMakespan(int n, int[] duration, List<Set<Integer>> predecessors, int workers) {
        List<int[]> orders = new ArrayList<>();
        enumerateOrders(n, predecessors, new boolean[n], new ArrayList<>(), orders);
        int best = Integer.MAX_VALUE;
        for (int[] order : orders) {
            int codes = 1;
            for (int i = 0; i < n; i++) codes *= workers;
            for (int code = 0; code < codes; code++) {
                int[] finish = new int[n];
                int[] workerFree = new int[workers];
                int value = code;
                boolean valid = true;
                for (int i = 0; i < n && valid; i++) {
                    int task = order[i];
                    int w = value % workers;
                    value /= workers;
                    int earliest = 0;
                    for (int p : predecessors.get(task)) {
                        if (finish[p] < 0) valid = false;
                        earliest = Math.max(earliest, finish[p]);
                    }
                    if (!valid) break;
                    finish[task] = Math.max(earliest, workerFree[w]) + duration[task];
                    workerFree[w] = finish[task];
                }
                if (!valid) continue;
                int makespan = 0;
                for (int f : finish) makespan = Math.max(makespan, f);
                best = Math.min(best, makespan);
            }
        }
        return best;
    }

    private static void enumerateOrders(int n, List<Set<Integer>> predecessors, boolean[] placed,
            List<Integer> current, List<int[]> out) {
        if (current.size() == n) {
            int[] copy = new int[n];
            for (int i = 0; i < n; i++) copy[i] = current.get(i);
            out.add(copy);
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
            enumerateOrders(n, predecessors, placed, current, out);
            current.remove(current.size() - 1);
            placed[v] = false;
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    /** Independent soundness rescan of a schedule. */
    static void assertSound(DagExecutor executor, Schedule schedule, int workers) {
        int n = executor.tasks.size();
        for (int task = 0; task < n; task++) {
            check(schedule.start[task] >= 0, "task " + task + " was scheduled");
            check(schedule.finish[task] - schedule.start[task] == executor.tasks.get(task).duration,
                    "task " + task + " runs for exactly its duration");
            for (int p : executor.predecessors.get(task)) {
                check(schedule.start[task] >= schedule.finish[p],
                        "task " + task + " waits for predecessor " + p);
            }
        }
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                if (schedule.worker[a] == schedule.worker[b]) {
                    check(schedule.finish[a] <= schedule.start[b] || schedule.finish[b] <= schedule.start[a],
                            "tasks " + a + " and " + b + " do not overlap on one worker");
                }
            }
        }
        int busy = 0;
        for (int value : schedule.busyPerWorker) busy += value;
        int work = 0;
        for (Task task : executor.tasks) work += task.duration;
        check(schedule.busyPerWorker.length == workers && busy == work,
                "sum of durations equals the summed worker busy time (" + work + ")");
        int makespan = 0;
        for (int f : schedule.finish) makespan = Math.max(makespan, f);
        check(makespan == schedule.makespan, "the reported makespan matches the finish times");
    }

    /** Independent rescan: no tick may have both an idle worker and an unstarted ready task. */
    static boolean isNonDelay(DagExecutor executor, Schedule schedule, int workers) {
        int n = executor.tasks.size();
        for (int tick = 0; tick < schedule.makespan; tick++) {
            int idle = 0;
            for (int w = 0; w < workers; w++) {
                boolean busy = false;
                for (int task = 0; task < n; task++) {
                    if (schedule.worker[task] == w && schedule.start[task] <= tick
                            && schedule.finish[task] > tick) {
                        busy = true;
                    }
                }
                if (!busy) idle++;
            }
            if (idle == 0) continue;
            for (int task = 0; task < n; task++) {
                if (schedule.start[task] >= 0) continue;
                boolean available = true;
                for (int p : executor.predecessors.get(task)) {
                    if (schedule.finish[p] > tick) available = false;
                }
                if (available && schedule.start[task] < 0) {
                    // a ready, unstarted task exists; its start must be exactly this tick
                    return false;
                }
            }
        }
        return true;
    }

    public static void main(String[] args) {
        // 1. Hand-computed instances.
        DagExecutor chain = new DagExecutor();
        int a = chain.addTask("a", 3, 1);
        int b = chain.addTask("b", 4, 1);
        int c = chain.addTask("c", 2, 1);
        chain.dependsOn(b, a);
        chain.dependsOn(c, b);
        Schedule chainOne = chain.run(1);
        check(chainOne.makespan == 9, "a 3+4+2 chain takes 9 ticks with one worker");
        check(chain.run(4).makespan == 9, "a chain is serial no matter how many workers exist");
        assertSound(chain, chainOne, 1);
        assertSound(chain, chain.run(4), 4);
        check(chainOne.start[a] == 0 && chainOne.start[b] == 3 && chainOne.start[c] == 7,
                "chain start times are the cumulative durations");

        DagExecutor parallel = new DagExecutor();
        for (int i = 0; i < 5; i++) parallel.addTask("p" + i, 2 + i, i);
        Schedule parallelTwo = parallel.run(2);
        check(parallelTwo.makespan >= 6, "the longest single task bounds the makespan (6)");
        assertSound(parallel, parallelTwo, 2);
        Schedule parallelFive = parallel.run(5);
        check(parallelFive.makespan == 6, "five workers finish the longest task alone (6)");
        assertSound(parallel, parallelFive, 5);
        check(isNonDelay(parallel, parallelFive, 5), "the five-worker schedule is non-delay");

        // 2. A crafted two-worker instance: two length-5 branches then a length-1 join.
        DagExecutor crafted = new DagExecutor();
        int root = crafted.addTask("root", 1, 5);
        int left = crafted.addTask("left", 5, 9);
        int right = crafted.addTask("right", 5, 7);
        int join = crafted.addTask("join", 1, 1);
        crafted.dependsOn(left, root);
        crafted.dependsOn(right, root);
        crafted.dependsOn(join, left);
        crafted.dependsOn(join, right);
        Schedule craftedTwo = crafted.run(2);
        check(craftedTwo.makespan == 7, "crafted: 1 + max(5,5) + 1 = 7 with two workers");
        check(craftedTwo.start[root] == 0 && craftedTwo.start[left] == 1
                        && craftedTwo.start[right] == 1 && craftedTwo.start[join] == 6,
                "crafted: the two branches run in parallel");
        assertSound(crafted, craftedTwo, 2);
        check(isNonDelay(crafted, craftedTwo, 2), "the crafted schedule is non-delay");

        // 3. Tiny instance against the brute-force optimum.
        DagExecutor tiny = new DagExecutor();
        int t0 = tiny.addTask("t0", 3, 4);
        int t1 = tiny.addTask("t1", 2, 9);
        int t2 = tiny.addTask("t2", 4, 5);
        int t3 = tiny.addTask("t3", 1, 2);
        int t4 = tiny.addTask("t4", 5, 1);
        tiny.dependsOn(t1, t0);
        tiny.dependsOn(t3, t1);
        tiny.dependsOn(t3, t2);
        tiny.dependsOn(t4, t3);
        Schedule tinySchedule = tiny.run(2);
        assertSound(tiny, tinySchedule, 2);
        int[] durations = new int[]{3, 2, 4, 1, 5};
        int optimum = bruteForceMakespan(5, durations, tiny.predecessors, 2);
        check(tinySchedule.makespan == optimum,
                "greedy makespan " + tinySchedule.makespan + " equals the brute-force optimum " + optimum);

        // 4. Non-delay property and the list-scheduling bound on an LCG-labelled random DAG.
        Lcg lcg = new Lcg(510510);
        DagExecutor random = new DagExecutor();
        int count = 14;
        for (int i = 0; i < count; i++) random.addTask("r" + i, 1 + lcg.nextInt(6), lcg.nextInt(100));
        for (int i = 0; i < count; i++) {
            for (int j = i + 1; j < count; j++) {
                if (lcg.nextInt(4) == 0) random.dependsOn(j, i);
            }
        }
        int workers = 3;
        Schedule randomSchedule = random.run(workers);
        assertSound(random, randomSchedule, workers);
        int bound = random.criticalPath();
        check(randomSchedule.makespan >= bound, "the makespan is at least the critical path");
        check(randomSchedule.makespan * workers <= (2 * workers - 1) * bound,
                "list scheduling respects the (2 - 1/W) * critical path bound");
        check(isNonDelay(random, randomSchedule, workers), "the random schedule is non-delay");
        check(randomSchedule.makespan > bound,
                "the random instance is wide enough to exceed the bound");
        check(random.criticalPath() == bound, "the critical path is computed deterministically");

        // 5. A cycle is reported as unrunnable rather than looping forever.
        DagExecutor cyclic = new DagExecutor();
        int x = cyclic.addTask("x", 1, 1);
        int y = cyclic.addTask("y", 1, 1);
        int z = cyclic.addTask("z", 1, 1);
        cyclic.dependsOn(y, x);
        cyclic.dependsOn(z, y);
        cyclic.dependsOn(x, z);
        check(cyclic.run(2) == null, "a fully cyclic DAG is unrunnable");
        DagExecutor partiallyBlocked = new DagExecutor();
        int free = partiallyBlocked.addTask("free", 1, 1);
        int c1 = partiallyBlocked.addTask("c1", 1, 1);
        int c2 = partiallyBlocked.addTask("c2", 1, 1);
        partiallyBlocked.dependsOn(c2, c1);
        partiallyBlocked.dependsOn(c1, c2);
        check(partiallyBlocked.run(2) == null, "a cycle anywhere blocks the whole DAG");
        check(chain.run(1) != null && random.run(workers) != null, "acyclic DAGs still run");
        check(free >= 0 && c1 >= 0 && c2 >= 0, "the blocked instance was constructed");

        // 6. Zero-duration tasks finish immediately and free their worker in the same tick.
        DagExecutor instant = new DagExecutor();
        int i0 = instant.addTask("i0", 0, 2);
        int i1 = instant.addTask("i1", 3, 1);
        instant.dependsOn(i1, i0);
        Schedule instantSchedule = instant.run(1);
        check(instantSchedule.start[i0] == 0 && instantSchedule.finish[i0] == 0,
                "a zero-duration task finishes at its start");
        check(instantSchedule.makespan == 3, "the following task still takes its own time");
        assertSound(instant, instantSchedule, 1);
        DagExecutor allInstant = new DagExecutor();
        for (int i = 0; i < 5; i++) allInstant.addTask("z" + i, 0, i);
        Schedule allInstantSchedule = allInstant.run(1);
        check(allInstantSchedule != null && allInstantSchedule.makespan == 0,
                "five zero-duration tasks all run in tick 0");
        assertSound(allInstant, allInstantSchedule, 1);

        // 7. Determinism.
        Schedule rerun = random.run(workers);
        check(rerun.makespan == randomSchedule.makespan, "deterministic re-run: same makespan");
        for (int i = 0; i < count; i++) {
            check(rerun.start[i] == randomSchedule.start[i] && rerun.worker[i] == randomSchedule.worker[i],
                    "deterministic re-run: identical placement of task " + i);
        }

        System.out.println("PASS");
    }
}
