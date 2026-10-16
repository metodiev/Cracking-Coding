/** Task 965 - Implement secure sandbox to run untrusted student submissions with resource cgroups.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory cgroup-based sandbox. Every submission is a scripted schedule of resource requests (memory
 *  units, CPU units, process spawns and syscalls) that a runner executes against a fixed cgroup budget.
 *  The runner charges usage until a request would cross a limit, then kills the submission at exactly the
 *  limit with the matching status (OOM, TIMEOUT, PID_LIMIT), or denies a syscall that is not on the
 *  allow list. Usage is capped at the limit, so no report can ever exceed the budget, and the aggregate
 *  charged units are reconciled against an independent recomputation from the raw schedules.
 *  Simplified: schedules stand in for real payloads and there is no container runtime or seccomp filter;
 *  the enforcement order, capping, status taxonomy and accounting are the real mechanics.
 *  Verified: polite submissions finish OK inside the budget; memory, CPU and process hogs are killed at
 *  exactly their limit with the right status; a denied syscall stops the run where it is; usage never
 *  exceeds the limits for any submission; a schedule exactly at every limit passes while one unit more
 *  dies; the batch is deterministic; independent recomputation reproduces every charge.
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

    record Step(String kind, int amount, String name) {
        static Step memory(int units) { return new Step("memory", units, ""); }
        static Step cpu(int units) { return new Step("cpu", units, ""); }
        static Step spawn(int count) { return new Step("spawn", count, ""); }
        static Step syscall(String name) { return new Step("syscall", 0, name); }
    }

    record Submission(String student, List<Step> schedule) {}

    record Report(String student, String status, int memoryUsed, int cpuUsed, int pidsUsed) {
        String describe() { return student + "=" + status + " mem=" + memoryUsed + " cpu=" + cpuUsed + " pids=" + pidsUsed; }
    }

    static final class Sandbox {
        final int memoryLimit, cpuLimit, pidsLimit;
        final Set<String> allowedSyscalls;
        int charges;                                  // aggregate accounting across the whole batch

        Sandbox(int memoryLimit, int cpuLimit, int pidsLimit, Collection<String> allowedSyscalls) {
            this.memoryLimit = memoryLimit;
            this.cpuLimit = cpuLimit;
            this.pidsLimit = pidsLimit;
            this.allowedSyscalls = new TreeSet<>(allowedSyscalls);
        }

        Report run(Submission submission) {
            int memory = 0, cpu = 0, pids = 0;
            String status = "OK";
            for (Step step : submission.schedule()) {
                switch (step.kind()) {
                    case "memory":
                        if (memory + step.amount() > memoryLimit) { memory = memoryLimit; status = "OOM"; }
                        else memory += step.amount();
                        break;
                    case "cpu":
                        if (cpu + step.amount() > cpuLimit) { cpu = cpuLimit; status = "TIMEOUT"; }
                        else cpu += step.amount();
                        break;
                    case "spawn":
                        if (pids + step.amount() > pidsLimit) { pids = pidsLimit; status = "PID_LIMIT"; }
                        else pids += step.amount();
                        break;
                    default:
                        if (!allowedSyscalls.contains(step.name())) status = "DENIED";
                }
                if (!status.equals("OK")) break;
            }
            return new Report(submission.student(), status, memory, cpu, pids);
        }

        List<Report> runAll(List<Submission> submissions) {
            List<Report> reports = new ArrayList<>();
            charges = 0;
            for (Submission submission : submissions) {
                Report report = run(submission);
                charges += report.memoryUsed() + report.cpuUsed() + report.pidsUsed();
                reports.add(report);
            }
            return reports;
        }

        String digest(List<Report> reports) {
            long h = fnvText(FNV_OFFSET, "sandbox:" + memoryLimit + "/" + cpuLimit + "/" + pidsLimit);
            int total = 0;
            for (Report report : reports) total += report.memoryUsed() + report.cpuUsed() + report.pidsUsed();
            h = fnvText(h, "charges=" + total);
            for (Report report : reports) h = fnvText(h, report.describe() + "|");
            return hex(h);
        }
    }

    /** Independent recomputation of what a schedule should charge, straight from the raw steps. */
    static int[] expectedUsage(Submission submission, Sandbox sandbox) {
        int memory = 0, cpu = 0, pids = 0;
        for (Step step : submission.schedule()) {
            boolean over;
            if (step.kind().equals("memory")) {
                over = memory + step.amount() > sandbox.memoryLimit;
                memory = over ? sandbox.memoryLimit : memory + step.amount();
            } else if (step.kind().equals("cpu")) {
                over = cpu + step.amount() > sandbox.cpuLimit;
                cpu = over ? sandbox.cpuLimit : cpu + step.amount();
            } else if (step.kind().equals("spawn")) {
                over = pids + step.amount() > sandbox.pidsLimit;
                pids = over ? sandbox.pidsLimit : pids + step.amount();
            } else {
                over = !sandbox.allowedSyscalls.contains(step.name());
            }
            if (over) break;
        }
        return new int[] { memory, cpu, pids };
    }

    static List<Submission> batch() {
        List<Submission> submissions = new ArrayList<>();
        submissions.add(new Submission("polite-reader", List.of(Step.syscall("read"), Step.memory(8), Step.cpu(5),
                Step.syscall("write"), Step.spawn(1), Step.cpu(3))));
        submissions.add(new Submission("memory-hog", List.of(Step.memory(48), Step.memory(48), Step.cpu(2))));
        submissions.add(new Submission("cpu-hog", List.of(Step.cpu(20), Step.cpu(20))));
        submissions.add(new Submission("fork-bomb", List.of(Step.spawn(3), Step.spawn(3))));
        submissions.add(new Submission("socket-caller", List.of(Step.memory(4), Step.cpu(2), Step.syscall("connect"),
                Step.memory(60))));
        submissions.add(new Submission("empty-submission", List.of()));
        submissions.add(new Submission("exactly-at-limits", List.of(Step.memory(64), Step.cpu(32), Step.spawn(4))));
        submissions.add(new Submission("one-over", List.of(Step.memory(64), Step.cpu(32), Step.spawn(5))));
        return submissions;
    }

    static Sandbox sandbox() { return new Sandbox(64, 32, 4, List.of("read", "write", "exit")); }

    static Report find(List<Report> reports, String student) {
        for (Report report : reports) if (report.student().equals(student)) return report;
        return null;
    }

    public static void main(String[] args) {
        Sandbox sandbox = sandbox();
        List<Report> reports = sandbox.runAll(batch());

        // 1. polite submissions finish OK well inside every limit
        Report polite = find(reports, "polite-reader");
        check(polite.status().equals("OK") && polite.memoryUsed() == 8 && polite.cpuUsed() == 8
                        && polite.pidsUsed() == 1,
                "polite submission completes with its recorded usage");

        // 2. memory hog is killed at exactly the memory limit
        Report memoryHog = find(reports, "memory-hog");
        check(memoryHog.status().equals("OOM") && memoryHog.memoryUsed() == 64 && memoryHog.cpuUsed() == 0,
                "memory hog is killed at the memory limit before touching the CPU");

        // 3. cpu hog is killed on timeout at exactly the cpu limit
        Report cpuHog = find(reports, "cpu-hog");
        check(cpuHog.status().equals("TIMEOUT") && cpuHog.cpuUsed() == 32 && cpuHog.memoryUsed() == 0,
                "cpu hog is killed at the cpu limit without allocating memory");

        // 4. fork bomb stops at the process limit
        Report forkBomb = find(reports, "fork-bomb");
        check(forkBomb.status().equals("PID_LIMIT") && forkBomb.pidsUsed() == 4,
                "fork bomb is capped by the pid limit");

        // 5. a syscall outside the allow list denies the run where it stands
        Report socketCaller = find(reports, "socket-caller");
        check(socketCaller.status().equals("DENIED") && socketCaller.memoryUsed() == 4 && socketCaller.cpuUsed() == 2
                        && socketCaller.pidsUsed() == 0,
                "denied syscall aborts the submission and nothing after it is charged");

        // 6. no report can exceed the limits, whatever the schedule asked for
        boolean capped = true;
        int budget = sandbox.memoryLimit + sandbox.cpuLimit + sandbox.pidsLimit;
        for (Report report : reports)
            capped &= report.memoryUsed() <= sandbox.memoryLimit && report.cpuUsed() <= sandbox.cpuLimit
                    && report.pidsUsed() <= sandbox.pidsLimit;
        check(capped && sandbox.charges <= reports.size() * budget,
                "every report stays within the cgroup budget (" + sandbox.charges + " charged of "
                        + reports.size() * budget + ")");

        // 7. boundaries: empty schedule, exactly at the limits, one unit over
        Report empty = find(reports, "empty-submission");
        Report exact = find(reports, "exactly-at-limits");
        Report over = find(reports, "one-over");
        check(empty.status().equals("OK") && empty.memoryUsed() == 0 && empty.cpuUsed() == 0 && empty.pidsUsed() == 0
                        && exact.status().equals("OK") && exact.memoryUsed() == 64 && exact.cpuUsed() == 32
                        && exact.pidsUsed() == 4 && over.status().equals("PID_LIMIT") && over.pidsUsed() == 4,
                "empty schedule and an exactly-at-limit schedule pass, one process over is killed");

        // 8. determinism and an independent recomputation of every charge
        List<Report> again = sandbox().runAll(batch());
        boolean deterministic = sandbox.digest(reports).equals(sandbox().digest(again)) && reports.equals(again);
        boolean accounting = true;
        for (Submission submission : batch()) {
            int[] expected = expectedUsage(submission, sandbox);
            Report report = find(reports, submission.student());
            accounting &= report.memoryUsed() == expected[0] && report.cpuUsed() == expected[1] && report.pidsUsed() == expected[2];
        }
        check(deterministic && accounting,
                "a second batch reproduces every report and an independent walk reproduces every charge");

        System.out.println("cgroup limits: memory " + sandbox.memoryLimit + ", cpu " + sandbox.cpuLimit
                + ", pids " + sandbox.pidsLimit + ", allowed " + sandbox.allowedSyscalls);
        for (Report report : reports) System.out.println("  " + report.describe());
        System.out.println("batch digest: " + sandbox.digest(reports));
        System.out.println("PASS");
    }
}
