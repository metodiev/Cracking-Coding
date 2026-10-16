/** Task 954 - Implement CI pipeline that runs on multiple JDKs and OSes with resource limits.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory CI scheduler. The matrix is {jdk 17, 21} x {ubuntu, windows, macos} plus the shared lint,
 *  docs, report and integrity jobs; every job declares a CPU-slot and memory footprint, a duration, and
 *  a dependency list. A tick-driven scheduler admits a job only while the executor slots and the memory
 *  pool allow it, kills jobs that exceed the time limit, retries flaky jobs once, fails broken jobs and
 *  skips the dependents of a failed required job (fail-fast) while unrelated jobs keep running.
 *  Simplified: resources are integer slots in one simulated pool, durations are ticks, and no real
 *  container or process is spawned - the invariants checked are those a real CI configuration must keep.
 *  Verified: every matrix cell is scheduled exactly once; recorded intervals never exceed the slot or
 *  memory budget and are replayed independently from the trace; dependents start after their
 *  prerequisites succeed; a broken job skips exactly its own dependents; a hung job is killed at the time
 *  limit without stalling the pipeline; the flaky job passes on its retry; two runs produce one digest.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final long FNV_OFFSET = 0xcbf29ce484222325L, FNV_PRIME = 0x100000001b3L;

    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }

    static String hex(long v) { return String.format("%016x", v); }

    record Job(String name, int slots, int memory, int duration, boolean required, List<String> dependsOn) {}

    static final class Run {
        final Job job;
        final List<int[]> intervals = new ArrayList<>();   // [startTick, endTick) per attempt
        int attempts = 1, currentStart = -1, startedAt = -1, finishedAt = -1;
        String status = "pending";
        Run(Job job) { this.job = job; }
        boolean running() { return status.equals("running"); }
        boolean terminal() { return status.equals("passed") || status.equals("failed") || status.equals("skipped"); }
        String describe() { return job.name() + "=" + status + (attempts > 1 ? " attempts=" + attempts : ""); }
    }

    static final class Pipeline {
        static final int SLOTS = 4, MEMORY = 10, TIME_LIMIT = 12, MAX_TICKS = 600;
        final List<Run> runs = new ArrayList<>();
        final Map<String, Run> byName = new LinkedHashMap<>();
        final Set<String> flaky, broken, hangs;
        final List<String> trace = new ArrayList<>();
        int violations, retries, timeouts, busySlotTicks, ticks;

        Pipeline(List<Job> jobs, Collection<String> flaky, Collection<String> broken, Collection<String> hangs) {
            for (Job job : jobs) { Run run = new Run(job); runs.add(run); byName.put(job.name(), run); }
            this.flaky = new LinkedHashSet<>(flaky);
            this.broken = new LinkedHashSet<>(broken);
            this.hangs = new LinkedHashSet<>(hangs);
        }

        boolean depsFailed(Run run) {
            for (String name : run.job.dependsOn()) {
                Run dep = byName.get(name);
                if (dep == null || dep.status.equals("failed") || dep.status.equals("skipped")) return true;
            }
            return false;
        }

        boolean depsPassed(Run run) {
            for (String name : run.job.dependsOn()) {
                Run dep = byName.get(name);
                if (dep == null || !dep.status.equals("passed")) return false;
            }
            return true;
        }

        void propagateSkips() {
            boolean changed = true;
            while (changed) {
                changed = false;
                for (Run run : runs)
                    if (run.status.equals("pending") && depsFailed(run)) {
                        run.status = "skipped";
                        run.finishedAt = ticks;
                        trace.add("t" + ticks + " skip " + run.job.name());
                        changed = true;
                    }
            }
        }

        void close(Run run, String status) {
            run.intervals.add(new int[] { run.currentStart, ticks });
            run.status = status;
            run.finishedAt = ticks;
            run.currentStart = -1;
        }

        void run() {
            for (ticks = 1; ticks <= MAX_TICKS && !allTerminal(); ticks++) {
                propagateSkips();
                for (Run run : runs) {
                    if (!run.running()) continue;
                    if (hangs.contains(run.job.name())) {
                        if (ticks - run.currentStart >= TIME_LIMIT) {
                            close(run, "failed");
                            timeouts++;
                            trace.add("t" + ticks + " timeout " + run.job.name());
                        }
                        continue;
                    }
                    if (ticks - run.currentStart >= run.job.duration()) {
                        if (broken.contains(run.job.name())) {
                            close(run, "failed");
                            trace.add("t" + ticks + " fail " + run.job.name());
                        } else if (flaky.contains(run.job.name()) && run.attempts == 1) {
                            close(run, "pending");
                            run.attempts = 2;
                            retries++;
                            trace.add("t" + ticks + " retry " + run.job.name());
                        } else {
                            close(run, "passed");
                            trace.add("t" + ticks + " pass " + run.job.name());
                        }
                    }
                }
                propagateSkips();
                int slotsUsed = 0, memoryUsed = 0;
                for (Run run : runs) if (run.running()) { slotsUsed += run.job.slots(); memoryUsed += run.job.memory(); }
                for (Run run : runs) {
                    if (!run.status.equals("pending") || !depsPassed(run)) continue;
                    if (slotsUsed + run.job.slots() > SLOTS || memoryUsed + run.job.memory() > MEMORY) continue;
                    run.status = "running";
                    run.currentStart = ticks;
                    run.startedAt = run.attempts == 1 ? ticks : run.startedAt;
                    slotsUsed += run.job.slots();
                    memoryUsed += run.job.memory();
                    trace.add("t" + ticks + " start " + run.job.name() + " attempt" + run.attempts);
                }
                slotsUsed = 0;
                memoryUsed = 0;
                for (Run run : runs) if (run.running()) { slotsUsed += run.job.slots(); memoryUsed += run.job.memory(); }
                if (slotsUsed > SLOTS || memoryUsed > MEMORY) violations++;
                busySlotTicks += slotsUsed;
            }
            for (Run run : runs) if (!run.terminal()) { run.status = "blocked"; run.finishedAt = ticks; }
        }

        boolean allTerminal() {
            for (Run run : runs) if (!run.terminal()) return false;
            return true;
        }

        Run get(String name) { return byName.get(name); }

        String digest() {
            long h = fnvText(FNV_OFFSET, "ci");
            for (String line : trace) h = fnvText(h, line);
            return hex(h);
        }

        List<String> statuses(String status) {
            List<String> names = new ArrayList<>();
            for (Run run : runs) if (run.status.equals(status)) names.add(run.job.name());
            return names;
        }
    }

    /** The matrix; `extraColumn` adds a seventh cell, `brokenCells` forces cells to always fail. */
    static List<Job> matrix(boolean extraColumn) {
        List<Job> jobs = new ArrayList<>();
        jobs.add(new Job("lint", 1, 2, 3, true, List.of()));
        jobs.add(new Job("docs", 1, 2, 4, true, List.of()));
        for (String os : new String[] { "ubuntu", "windows", "macos" })
            for (String jdk : new String[] { "17", "21" }) {
                String build = "build-" + os + "-jdk" + jdk;
                String test = "test-" + os + "-jdk" + jdk;
                jobs.add(new Job(build, 2, 4, 4, true, List.of()));
                jobs.add(new Job(test, 2, 4, 5, true, List.of(build)));
                jobs.add(new Job("package-" + os + "-jdk" + jdk, 1, 2, 3, true, List.of(test)));
            }
        if (extraColumn) jobs.add(new Job("test-alpine-jdk21", 2, 4, 5, true, List.of()));
        jobs.add(new Job("flaky-report", 1, 2, 3, false, List.of("lint")));
        jobs.add(new Job("hang-integrity", 1, 2, 3, false, List.of("docs")));
        return jobs;
    }

    static Pipeline pipeline(boolean extraColumn, String brokenCell) {
        return new Pipeline(matrix(extraColumn), List.of("flaky-report"),
                brokenCell == null ? List.of() : List.of(brokenCell), List.of("hang-integrity"));
    }

    public static void main(String[] args) {
        Pipeline first = pipeline(false, null);
        first.run();

        // 1. every job reaches a terminal state and the matrix cells were actually scheduled
        boolean cellsScheduled = true;
        for (Run run : first.runs)
            cellsScheduled &= run.terminal() && run.startedAt > 0 && !run.status.equals("blocked");
        check(cellsScheduled && first.runs.size() == matrix(false).size()
                        && first.statuses("passed").size() == matrix(false).size() - 1
                        && first.statuses("failed").equals(List.of("hang-integrity")),
                "all " + matrix(false).size() + " jobs terminal; only the hung job fails");

        // 2. the scheduler never over-commits its own budget
        check(first.violations == 0 && first.busySlotTicks > 0,
                "no tick exceeds " + Pipeline.SLOTS + " slots or " + Pipeline.MEMORY + " memory units");

        // 3. dependency order holds for every edge
        boolean orderOk = true;
        for (Run run : first.runs)
            for (String name : run.job.dependsOn()) {
                Run dep = first.get(name);
                if (run.startedAt > 0 && dep.finishedAt > 0 && dep.status.equals("passed"))
                    orderOk &= run.startedAt >= dep.finishedAt;
            }
        check(orderOk, "each job starts no earlier than the finish of its successful prerequisites");

        // 4. fail-fast: a broken matrix cell skips exactly its dependents, unrelated jobs still pass
        Pipeline failing = pipeline(false, "test-windows-jdk17");
        failing.run();
        check(failing.statuses("skipped").equals(List.of("package-windows-jdk17"))
                        && failing.statuses("failed").contains("test-windows-jdk17")
                        && failing.get("package-macos-jdk21").status.equals("passed")
                        && failing.get("test-macos-jdk21").status.equals("passed"),
                "broken cell fails and skips exactly its dependent; other platforms keep running");

        // 5. a hung job is killed at the time limit and does not stall the pipeline
        Run hung = first.get("hang-integrity");
        check(hung.status.equals("failed") && first.timeouts == 1
                        && hung.finishedAt - hung.startedAt == Pipeline.TIME_LIMIT
                        && first.ticks <= Pipeline.MAX_TICKS
                        && first.statuses("passed").size() + first.statuses("failed").size() == first.runs.size(),
                "hung job is killed at its limit, its resources return and every other job completes");

        // 6. the flaky job passes on its second attempt
        Run flaky = first.get("flaky-report");
        check(flaky.status.equals("passed") && flaky.attempts == 2 && first.retries == 1 && flaky.intervals.size() == 2,
                "flaky job is retried once and then passes");

        // 7. determinism, plus the empty-matrix boundary
        Pipeline repeat = pipeline(false, null);
        repeat.run();
        Pipeline wider = pipeline(true, null);
        wider.run();
        check(first.digest().equals(repeat.digest()) && !first.digest().equals(wider.digest()),
                "same matrix reproduces the trace digest, an added cell changes it");
        Pipeline empty = new Pipeline(new ArrayList<>(), List.of(), List.of(), List.of());
        empty.run();
        check(empty.runs.isEmpty() && empty.busySlotTicks == 0 && empty.violations == 0 && empty.digest().equals(hex(fnvText(FNV_OFFSET, "ci"))),
                "empty matrix runs zero jobs and keeps the empty trace digest");

        // 8. independent oracle: replay the recorded intervals tick by tick and re-derive the accounting
        int replayBusy = 0, replayPeakSlots = 0, replayViolations = 0;
        for (int tick = 1; tick <= Pipeline.MAX_TICKS; tick++) {
            int slots = 0, memory = 0;
            for (Run run : first.runs)
                for (int[] interval : run.intervals)
                    if (tick >= interval[0] && tick < interval[1]) { slots += run.job.slots(); memory += run.job.memory(); }
            if (slots > Pipeline.SLOTS || memory > Pipeline.MEMORY) replayViolations++;
            replayPeakSlots = Math.max(replayPeakSlots, slots);
            replayBusy += slots;
        }
        check(replayViolations == 0 && replayBusy == first.busySlotTicks && replayPeakSlots <= Pipeline.SLOTS,
                "tick-by-tick replay of the intervals reproduces the busy-slot accounting");

        System.out.println("jobs in matrix: " + first.runs.size() + ", ticks: " + first.ticks);
        System.out.println("passed: " + first.statuses("passed").size() + ", failed: " + first.statuses("failed")
                + ", retries: " + first.retries + ", timeouts: " + first.timeouts);
        System.out.println("busy slot-ticks: " + first.busySlotTicks + ", peak concurrency: " + replayPeakSlots
                + " of " + Pipeline.SLOTS);
        System.out.println("trace digest: " + first.digest() + ", fail-fast digest: " + failing.digest());
        System.out.println("PASS");
    }
}
