/** Task 658 - Implement straggler mitigation via speculative execution simulation. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A discrete event simulator of a map-reduce style task scheduler with progress based speculative
 * execution: N tasks of known work meet W workers of known speed, the scheduler fills idle workers
 * FIFO, and periodically asks whether a running attempt is a straggler, i.e. whether its observed
 * rate lags the median rate of the running attempts by more than `lagPermille`. A straggler gets one
 * speculative copy on an idle worker when that copy could finish before the running attempt; the
 * first attempt to finish commits the task and the redundant one is cancelled.
 *
 * Simplifications (documented, as required): progress reports are exact (no measurement noise), a
 * worker's rate is constant so the observed rate equals its speed, there is no network or scheduling
 * overhead, a task has at most one speculative copy, and a task's output is a pure function of the
 * task id so both attempts produce the same value.
 *
 * Verified: the designed straggler scenario improves the makespan (30 vs 50) while committing every
 * task exactly once; uniform work never triggers speculation; a single worker cannot speculate (no
 * idle worker) and degenerates to the sequential sum; speculative copies never share a worker with
 * the primary and no worker ever runs two attempts at once; attempts per task are <= 2; the makespan
 * is between the critical path bound and the non-speculative makespan; the discard counter equals an
 * independently recomputed count of attempts that finished after their task was already committed;
 * the speculative attempt count is non-increasing in the lag parameter; and invalid inputs are
 * rejected.
 */
import java.util.*;

final class Attempt {
    final int task;
    final int worker;
    final boolean speculative;
    final long start;
    final int output;
    long finish = -1;
    long cancelledAt = -1;

    Attempt(int task, int worker, boolean speculative, long start, int output) {
        this.task = task;
        this.worker = worker;
        this.speculative = speculative;
        this.start = start;
        this.output = output;
    }

    boolean running() { return finish < 0 && cancelledAt < 0; }

    long effectiveEnd() { return finish >= 0 ? finish : cancelledAt; }
}

final class SimulationResult {
    long makespan;
    int commits;
    int speculativeAttempts;
    int discarded;
    long consumedWork;
    long[] commitTime = new long[0];
    int[] commitOutput = new int[0];
    List<Attempt> attempts = new ArrayList<>();

    int attemptsOfTask(int task) {
        int count = 0;
        for (Attempt attempt : attempts) if (attempt.task == task) count++;
        return count;
    }

    List<Attempt> attemptsOfWorker(int worker) {
        List<Attempt> out = new ArrayList<>();
        for (Attempt attempt : attempts) if (attempt.worker == worker) out.add(attempt);
        return out;
    }
}

final class Event implements Comparable<Event> {
    static final int FINISH = 0, HEARTBEAT = 1;
    final long time;
    final int type;
    final int attemptIndex;
    final long sequence;

    Event(long time, int type, int attemptIndex, long sequence) {
        this.time = time;
        this.type = type;
        this.attemptIndex = attemptIndex;
        this.sequence = sequence;
    }

    @Override
    public int compareTo(Event other) {                 // finishes are handled before heartbeats
        if (time != other.time) return Long.compare(time, other.time);
        if (type != other.type) return Integer.compare(type, other.type);
        return Long.compare(sequence, other.sequence);
    }
}

final class Simulator {
    private final int[] work;
    private final int[] speed;
    private final boolean speculate;
    private final int lagPermille;
    private final int heartbeat;

    private final List<Attempt> attempts = new ArrayList<>();
    private final int[] workerAttempt;
    private final long[] commitTime;
    private final int[] commitOutput;
    private final boolean[] speculated;
    private final ArrayDeque<Integer> pending = new ArrayDeque<>();
    private final PriorityQueue<Event> queue = new PriorityQueue<>();

    private long now;
    private long sequence;
    private long events;
    private int commits;
    private int speculativeCount;
    private int discarded;

    Simulator(int[] work, int[] speed, boolean speculate, int lagPermille, int heartbeat) {
        if (speed.length < 1) throw new IllegalArgumentException("need at least one worker");
        for (int units : work) if (units <= 0) throw new IllegalArgumentException("task work must be positive");
        for (int rate : speed) if (rate <= 0) throw new IllegalArgumentException("worker speed must be positive");
        if (lagPermille < 0) throw new IllegalArgumentException("negative lag");
        if (heartbeat <= 0) throw new IllegalArgumentException("heartbeat must be positive");
        this.work = work;
        this.speed = speed;
        this.speculate = speculate;
        this.lagPermille = lagPermille;
        this.heartbeat = heartbeat;
        this.workerAttempt = new int[speed.length];
        Arrays.fill(this.workerAttempt, -1);
        this.commitTime = new long[work.length];
        Arrays.fill(this.commitTime, -1);
        this.commitOutput = new int[work.length];
        Arrays.fill(this.commitOutput, -1);
        this.speculated = new boolean[work.length];
    }

    private int outputOf(int task) {
        // pure function of the task: both attempts of a task must agree
        return (task * 31 + work[task] * 7) % 101;
    }

    SimulationResult run() {
        SimulationResult result = new SimulationResult();
        result.commitTime = commitTime;
        result.commitOutput = commitOutput;
        if (work.length == 0) {
            result.attempts = attempts;
            return result;
        }
        for (int task = 0; task < work.length; task++) pending.add(task);
        if (speculate) queue.add(new Event(heartbeat, Event.HEARTBEAT, -1, sequence++));
        launchPending();
        while (commits < work.length) {
            if (queue.isEmpty() || ++events > 4_000_000) throw new IllegalStateException("simulation stalled");
            Event event = queue.poll();
            now = event.time;
            if (event.type == Event.FINISH) handleFinish(event.attemptIndex); else handleHeartbeat();
        }
        for (Attempt attempt : attempts) if (attempt.running()) attempt.cancelledAt = now;
        long makespan = 0;
        long consumed = 0;
        for (int task = 0; task < work.length; task++) makespan = Math.max(makespan, commitTime[task]);
        for (Attempt attempt : attempts) {
            long span = attempt.effectiveEnd() - attempt.start;
            consumed += Math.min(work[attempt.task], span * speed[attempt.worker]);
        }
        result.makespan = makespan;
        result.commits = commits;
        result.speculativeAttempts = speculativeCount;
        result.discarded = discarded;
        result.consumedWork = consumed;
        result.attempts = attempts;
        return result;
    }

    private void launch(int task, int worker, boolean speculative) {
        if (workerAttempt[worker] != -1) throw new IllegalStateException("worker already busy");
        Attempt attempt = new Attempt(task, worker, speculative, now, outputOf(task));
        attempts.add(attempt);
        workerAttempt[worker] = attempts.size() - 1;
        if (speculative) {
            speculativeCount++;
            speculated[task] = true;
        }
        long duration = (work[task] + speed[worker] - 1) / speed[worker];
        queue.add(new Event(now + duration, Event.FINISH, attempts.size() - 1, sequence++));
    }

    private void launchPending() {
        for (int worker = 0; worker < speed.length && !pending.isEmpty(); worker++) {
            if (workerAttempt[worker] == -1) launch(pending.poll(), worker, false);
        }
    }

    private void handleFinish(int index) {
        Attempt attempt = attempts.get(index);
        attempt.finish = now;
        workerAttempt[attempt.worker] = -1;
        if (commitTime[attempt.task] < 0) {
            commitTime[attempt.task] = now;
            commitOutput[attempt.task] = attempt.output;
            commits++;
            for (Attempt other : attempts) {
                if (other.task == attempt.task && other != attempt && other.running()) other.cancelledAt = now;
            }
        } else {
            discarded++;                       // the task was already committed by the other attempt
        }
        launchPending();
    }

    private void handleHeartbeat() {
        if (speculate) {
            List<Attempt> running = new ArrayList<>();
            for (Attempt attempt : attempts) {
                // every running attempt has a rate: with exact progress reports it is the worker speed
                if (attempt.running()) running.add(attempt);
            }
            if (!running.isEmpty()) {
                int[] rates = new int[running.size()];
                for (int i = 0; i < running.size(); i++) rates[i] = speed[running.get(i).worker];
                Arrays.sort(rates);
                int medianRate = rates[rates.length / 2];
                for (int worker = 0; worker < speed.length; worker++) {
                    if (workerAttempt[worker] != -1) continue;
                    int best = -1;
                    long bestFinish = Long.MAX_VALUE;
                    for (int i = 0; i < running.size(); i++) {
                        Attempt attempt = running.get(i);
                        if (speculated[attempt.task]) continue;
                        long rate = speed[attempt.worker];
                        if (rate * 1000L >= (long) medianRate * (1000 - lagPermille)) continue;   // not a straggler
                        long done = Math.min(work[attempt.task], (now - attempt.start) * speed[attempt.worker]);
                        long remaining = work[attempt.task] - done;
                        long currentFinish = now + (remaining + speed[attempt.worker] - 1) / speed[attempt.worker];
                        long speculativeFinish = now + (work[attempt.task] + speed[worker] - 1) / speed[worker];
                        if (speculativeFinish >= currentFinish) continue;   // the copy could not win
                        if (speculativeFinish < bestFinish) {
                            bestFinish = speculativeFinish;
                            best = i;
                        }
                    }
                    if (best >= 0) launch(running.get(best).task, worker, true);
                }
            }
        }
        queue.add(new Event(now + heartbeat, Event.HEARTBEAT, -1, sequence++));
    }
}

public class Solution {
    private static long state = 0x243F6A8885A308D3L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    static int[] filled(int size, int value) {
        int[] out = new int[size];
        Arrays.fill(out, value);
        return out;
    }

    static SimulationResult run(int[] work, int[] speed, boolean speculate, int lagPermille, int heartbeat) {
        return new Simulator(work, speed, speculate, lagPermille, heartbeat).run();
    }

    /** Independent recomputation of the discard counter from the attempt list. */
    static int recomputeDiscarded(SimulationResult result) {
        int count = 0;
        for (Attempt attempt : result.attempts) {
            if (attempt.finish < 0) continue;
            if (result.commitTime[attempt.task] >= 0 && result.commitTime[attempt.task] < attempt.finish) count++;
        }
        return count;
    }

    /** Recomputes consumed work from the attempt list. */
    static long recomputeConsumed(int[] work, int[] speed, SimulationResult result) {
        long consumed = 0;
        for (Attempt attempt : result.attempts) {
            long span = attempt.effectiveEnd() - attempt.start;
            consumed += Math.min(work[attempt.task], span * speed[attempt.worker]);
        }
        return consumed;
    }

    static void checkInvariants(int[] work, int[] speed, SimulationResult result, String label) {
        check(result.commits == work.length, label + ": every task commits");
        for (int task = 0; task < work.length; task++) {
            check(result.commitTime[task] >= 0, label + ": task " + task + " has a commit time");
            check(result.attemptsOfTask(task) >= 1 && result.attemptsOfTask(task) <= 2,
                    label + ": task " + task + " has one or two attempts");
        }
        long latest = 0;
        for (long time : result.commitTime) latest = Math.max(latest, time);
        check(result.makespan == latest, label + ": makespan is the last commit");
        for (Attempt attempt : result.attempts) {
            check(attempt.effectiveEnd() > attempt.start, label + ": attempts take positive time");
            check(attempt.output == (attempt.task * 31 + work[attempt.task] * 7) % 101,
                    label + ": attempt output is the deterministic task output");
        }
        // a worker never runs two attempts at the same time, and a copy never shares the primary's worker
        for (int worker = 0; worker < speed.length; worker++) {
            List<Attempt> mine = result.attemptsOfWorker(worker);
            for (int i = 0; i < mine.size(); i++) {
                for (int j = i + 1; j < mine.size(); j++) {
                    Attempt a = mine.get(i), b = mine.get(j);
                    boolean disjoint = a.effectiveEnd() <= b.start || b.effectiveEnd() <= a.start;
                    check(disjoint, label + ": worker " + worker + " double booked");
                }
            }
        }
        for (int task = 0; task < work.length; task++) {
            Set<Integer> workers = new HashSet<>();
            for (Attempt attempt : result.attempts) if (attempt.task == task) workers.add(attempt.worker);
            check(workers.size() == result.attemptsOfTask(task), label + ": speculations use a fresh worker");
        }
        check(result.discarded == recomputeDiscarded(result), label + ": discard counter accounts for every duplicate");
        check(result.consumedWork == recomputeConsumed(work, speed, result), label + ": consumed work accounting");
        for (int task = 0; task < work.length; task++) {
            check(result.commitOutput[task] == (task * 31 + work[task] * 7) % 101,
                    label + ": committed output for task " + task);
        }
    }

    public static void main(String[] args) {
        int heartbeat = 5;

        // --- zero tasks ---------------------------------------------------------
        SimulationResult none = run(new int[0], new int[]{4, 4}, true, 250, heartbeat);
        check(none.makespan == 0 && none.commits == 0 && none.attempts.isEmpty(), "no tasks -> no work");

        // --- designed straggler scenario ---------------------------------------
        int[] work = filled(6, 100);
        int[] speed = {10, 10, 2};
        SimulationResult plain = run(work, speed, false, 250, heartbeat);
        check(plain.makespan == 50, "without speculation the slow worker sets the makespan, got " + plain.makespan);
        check(plain.speculativeAttempts == 0, "the baseline never speculates");
        check(plain.commits == 6 && plain.attempts.size() == 6, "the baseline runs each task once");
        checkInvariants(work, speed, plain, "baseline");

        SimulationResult speculative = run(work, speed, true, 250, heartbeat);
        check(speculative.makespan == 30, "speculation cuts the makespan to 30, got " + speculative.makespan);
        check(speculative.makespan < plain.makespan, "speculation must not be slower");
        check(speculative.speculativeAttempts == 1, "exactly one straggler is copied, got "
                + speculative.speculativeAttempts);
        check(speculative.attempts.size() == 7, "seven attempts in total, got " + speculative.attempts.size());
        check(speculative.consumedWork > plain.consumedWork,
                "speculation buys latency with extra work: " + speculative.consumedWork + " > " + plain.consumedWork);
        checkInvariants(work, speed, speculative, "straggler");

        boolean primaryCancelled = false;
        for (Attempt attempt : speculative.attempts) {
            if (attempt.task == 2 && !attempt.speculative && attempt.cancelledAt >= 0) primaryCancelled = true;
        }
        check(primaryCancelled, "the losing attempt of task 2 is cancelled when the copy commits");

        // --- uniform work: nothing looks like a straggler ----------------------
        int[] equalSpeed = {10, 10, 10};
        SimulationResult uniformPlain = run(work, equalSpeed, false, 250, heartbeat);
        SimulationResult uniformSpeculative = run(work, equalSpeed, true, 250, heartbeat);
        check(uniformSpeculative.speculativeAttempts == 0, "uniform work never triggers speculation");
        check(uniformSpeculative.makespan == uniformPlain.makespan,
                "and therefore does not change the makespan");
        checkInvariants(work, equalSpeed, uniformSpeculative, "uniform");

        // --- one worker: speculation has nowhere to run ------------------------
        int[] singleSpeed = {4};
        SimulationResult singlePlain = run(new int[]{40, 20, 60}, singleSpeed, false, 250, 1);
        SimulationResult singleSpeculative = run(new int[]{40, 20, 60}, singleSpeed, true, 250, 1);
        check(singleSpeculative.speculativeAttempts == 0, "a single worker cannot host a copy");
        check(singlePlain.makespan == (40 + 20 + 60) / 4, "one worker degenerates to the sequential sum, got "
                + singlePlain.makespan);
        check(singleSpeculative.makespan == singlePlain.makespan, "makespan unchanged with one worker");

        // --- lag parameter: stricter lag means fewer copies --------------------
        int[] manyWork = filled(10, 100);
        int[] manySpeed = {10, 10, 10, 10, 2};
        int previous = Integer.MAX_VALUE;
        for (int lag : new int[]{50, 150, 250, 400, 600, 900}) {
            SimulationResult result = run(manyWork, manySpeed, true, lag, heartbeat);
            check(result.speculativeAttempts <= previous, "speculative attempts grow with the lag at " + lag);
            previous = result.speculativeAttempts;
            checkInvariants(manyWork, manySpeed, result, "lag " + lag);
        }
        check(previous == 0, "a very strict lag suppresses speculation entirely");

        // --- random sweep against the properties -------------------------------
        int improved = 0, triggered = 0, ties = 0;
        for (int trial = 0; trial < 80; trial++) {
            int tasks = 1 + nextInt(9);
            int[] sweepWork = new int[tasks];
            for (int i = 0; i < tasks; i++) sweepWork[i] = 1 + nextInt(120);
            int workers = 1 + nextInt(5);
            int[] sweepSpeed = new int[workers];
            for (int i = 0; i < workers; i++) sweepSpeed[i] = 1 + nextInt(9);
            int hb = 1 + nextInt(6);
            SimulationResult base = run(sweepWork, sweepSpeed, false, 250, hb);
            SimulationResult spec = run(sweepWork, sweepSpeed, true, 200 + nextInt(300), hb);
            checkInvariants(sweepWork, sweepSpeed, base, "sweep base " + trial);
            checkInvariants(sweepWork, sweepSpeed, spec, "sweep spec " + trial);
            check(spec.makespan <= base.makespan, "sweep " + trial + ": speculation never hurts");
            long fastest = 1;
            for (int rate : sweepSpeed) fastest = Math.max(fastest, rate);
            long criticalPath = 0;
            for (int units : sweepWork) criticalPath = Math.max(criticalPath, (units + fastest - 1) / fastest);
            check(spec.makespan >= criticalPath, "sweep " + trial + ": makespan beats the critical path bound");
            if (spec.makespan < base.makespan) improved++;
            if (spec.speculativeAttempts > 0) triggered++;
            if (spec.discarded > 0) ties++;
        }
        check(improved > 0, "some sweep instance must benefit from speculation");
        check(triggered > 0, "some sweep instance must launch a copy");
        check(speculative.discarded == 0, "the straggler scenario has no duplicate completion");

        // --- determinism --------------------------------------------------------
        SimulationResult repeat = run(work, speed, true, 250, heartbeat);
        check(repeat.makespan == speculative.makespan && repeat.attempts.size() == speculative.attempts.size(),
                "the simulation is deterministic in shape");
        check(Arrays.equals(repeat.commitTime, speculative.commitTime), "commit times are deterministic");
        check(repeat.speculativeAttempts == speculative.speculativeAttempts, "copy count is deterministic");

        // --- rejected input -----------------------------------------------------
        boolean rejected = false;
        try {
            run(new int[]{0}, new int[]{5}, true, 250, 5);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "zero work must be rejected");
        rejected = false;
        try {
            run(new int[]{5}, new int[]{0}, true, 250, 5);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "zero speed must be rejected");
        rejected = false;
        try {
            run(new int[]{5}, new int[]{5}, true, -1, 5);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "negative lag must be rejected");
        rejected = false;
        try {
            run(new int[]{5}, new int[]{5}, true, 250, 0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "zero heartbeat must be rejected");
        rejected = false;
        try {
            run(new int[]{5}, new int[0], true, 250, 5);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a cluster without workers must be rejected");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
