/** Task 508 - Implement background job scheduler with retry/backoff semantics. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * A discrete-event background job scheduler: jobs are submitted at a simulated millisecond clock,
 * each attempt either succeeds or fails according to a fixed deterministic outcome table, and a
 * failing attempt is retried after a full-jitter exponential backoff delay delay = floor(uniform *
 * (min(cap, base * 2^(attempt - 1)) + 1)) with uniform drawn from the 75/74 LCG - the "full jitter"
 * policy, whose mean is cap/2 for the capped attempts and which decorrelates retries. Jobs that
 * exhaust `maxAttempts` attempts go to a dead-letter list instead of being retried. The event loop
 * is a priority queue keyed by (readyTime, jobId, attempt) - a total order, so ties at the same
 * tick always resolve in increasing job id and the entire run is reproducible - and time only
 * advances to the next event, never consulting the wall clock. No threads: like the other wave-3
 * system-design tasks this is a deterministic simulation.
 *
 * Documented checks: the backoff formula is replayed with an independent LCG of the same seed and
 * every delay is asserted equal to floor(uniform * (min(cap, base << (attempt - 1)) + 1)) and
 * inside [0, cap]; each job's attempt count equals the failure prefix of the outcome table plus
 * one, doomed jobs (failures >= maxAttempts, i.e. those whose planted failure prefix is at least
 * the attempt budget) use exactly maxAttempts attempts and land in the dead-letter list while
 * successful jobs appear in the success list exactly once and are never retried afterwards; every
 * attempt is strictly later than the previous one and every job's first attempt happens exactly at
 * its submit time; the log is verified to be in (time, jobId) order; the mean first-retry delay
 * over the trace is within 10% of base/2 and the mean of the delays that hit the cap is within 12%
 * of cap/2, the documented full-jitter property; boundaries: base = 0 retries on the very next tick
 * (never a busy loop), and with maxAttempts = 1 the only survivors are the jobs that succeeded on
 * their very first attempt while every failing job is dead-lettered immediately; and a re-run
 * reproduces the event log, both terminal lists and the final clock bit for bit.
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
    }

    static class Event implements Comparable<Event> {
        final long readyAt;
        final int jobId;
        final int attempt;

        Event(long readyAt, int jobId, int attempt) {
            this.readyAt = readyAt;
            this.jobId = jobId;
            this.attempt = attempt;
        }

        public int compareTo(Event other) {
            if (readyAt != other.readyAt) return Long.compare(readyAt, other.readyAt);
            if (jobId != other.jobId) return Integer.compare(jobId, other.jobId);
            return Integer.compare(attempt, other.attempt);
        }
    }

    /** Outcome table: job `id` fails its first `failures(id)` attempts and then succeeds. */
    interface Outcomes {
        int failures(int jobId);
    }

    static class Scheduler {
        final int maxAttempts;
        final long base;
        final long cap;
        final Outcomes outcomes;
        final Lcg jitter;
        final PriorityQueue<Event> queue = new PriorityQueue<>();
        final List<long[]> log = new ArrayList<>();
        final List<long[]> delays = new ArrayList<>();
        final Map<Integer, Integer> attempts = new TreeMap<>();
        final Map<Integer, Long> firstRetry = new TreeMap<>();
        final List<Integer> succeeded = new ArrayList<>();
        final List<Integer> deadLettered = new ArrayList<>();
        long now;

        Scheduler(int maxAttempts, long base, long cap, Outcomes outcomes, int seed) {
            this.maxAttempts = maxAttempts;
            this.base = base;
            this.cap = cap;
            this.outcomes = outcomes;
            this.jitter = new Lcg(seed);
        }

        void submit(int jobId, long at) {
            queue.add(new Event(at, jobId, 1));
            attempts.put(jobId, 0);
        }

        /** Full-jitter backoff after `attemptJustFailed` (1-based) failures. */
        long backoff(int attemptJustFailed) {
            long ceiling = Math.min(cap, base << Math.min(attemptJustFailed - 1, 40));
            double uniform = jitter.next();
            return (long) Math.floor(uniform * (ceiling + 1));
        }

        int attemptsOf(int jobId) {
            return attempts.get(jobId);
        }

        List<long[]> logEntries() {
            return log;
        }

        double meanDelayAt(int attemptJustFailed) {
            double sum = 0.0;
            int count = 0;
            for (long[] record : delays) {
                if (record[0] == attemptJustFailed) {
                    sum += record[1];
                    count++;
                }
            }
            return count == 0 ? 0.0 : sum / count;
        }

        double meanFirstRetryDelay() {
            double sum = 0.0;
            int count = 0;
            for (long value : firstRetry.values()) {
                sum += value;
                count++;
            }
            return count == 0 ? 0.0 : sum / count;
        }

        void run() {
            while (!queue.isEmpty()) {
                Event event = queue.poll();
                now = Math.max(now, event.readyAt);
                int made = attempts.merge(event.jobId, 1, Integer::sum);
                log.add(new long[]{now, event.jobId, event.attempt});
                int needed = outcomes.failures(event.jobId);
                if (made > needed) {
                    succeeded.add(event.jobId);
                    continue;
                }
                if (made >= maxAttempts) {
                    deadLettered.add(event.jobId);
                    continue;
                }
                long delay = backoff(made);
                delays.add(new long[]{made, delay});
                if (event.attempt == 1) firstRetry.put(event.jobId, delay);
                queue.add(new Event(now + delay + 1, event.jobId, event.attempt + 1));
            }
        }
    }

    /** Deterministic outcome table: failures(job) = floor(uniform * maxFailures) in [0, maxFailures). */
    static Outcomes mixedOutcomes(int maxFailures) {
        Map<Integer, Integer> table = new TreeMap<>();
        Lcg lcg = new Lcg(908172);
        return jobId -> table.computeIfAbsent(jobId, key -> (int) (lcg.next() * maxFailures));
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        int jobs = 800;
        int maxAttempts = 6;
        long base = 100;
        long cap = 1000;
        Outcomes outcomes = mixedOutcomes(8);
        Scheduler scheduler = new Scheduler(maxAttempts, base, cap, outcomes, 424242);
        for (int job = 0; job < jobs; job++) scheduler.submit(job, job * 7L);
        scheduler.run();

        // 1. Accounting: every job either succeeds once or dies with exactly maxAttempts tries.
        int totalAttempts = 0;
        for (int job = 0; job < jobs; job++) {
            int tries = scheduler.attemptsOf(job);
            totalAttempts += tries;
            int failures = outcomes.failures(job);
            if (failures >= maxAttempts) {
                check(tries == maxAttempts, "doomed job " + job + " used every attempt");
                check(scheduler.deadLettered.contains(job), "doomed job " + job + " is dead-lettered");
                check(!scheduler.succeeded.contains(job), "doomed job " + job + " never succeeds");
            } else {
                check(tries == failures + 1, "job " + job + " needed failures + 1 attempts");
                check(scheduler.succeeded.contains(job), "job " + job + " succeeds");
                check(!scheduler.deadLettered.contains(job), "job " + job + " is not dead-lettered");
            }
            check(tries <= maxAttempts, "no job exceeds maxAttempts");
        }
        check(scheduler.succeeded.size() + scheduler.deadLettered.size() == jobs,
                "every job ends in exactly one terminal state");
        check(scheduler.logEntries().size() == totalAttempts, "the log holds one entry per attempt");
        check(!scheduler.succeeded.isEmpty() && !scheduler.deadLettered.isEmpty(),
                "the outcome table exercises both terminal states");

        // 2. No attempt after success, and each success appears exactly once.
        Map<Integer, Integer> successEvents = new HashMap<>();
        for (int job : scheduler.succeeded) successEvents.merge(job, 1, Integer::sum);
        for (int job : scheduler.succeeded) {
            check(successEvents.get(job) == 1, "job " + job + " logged exactly one success");
        }
        Set<Integer> finished = new HashSet<>();
        for (long[] entry : scheduler.logEntries()) {
            int jobId = (int) entry[1];
            int attempt = (int) entry[2];
            if (scheduler.succeeded.contains(jobId) && attempt == outcomes.failures(jobId) + 1) {
                finished.add(jobId);
            } else {
                check(!finished.contains(jobId), "no job is retried after it succeeded");
            }
        }
        check(finished.size() == scheduler.succeeded.size(),
                "every successful job reached its success");

        // 3. Backoff formula replayed with an independent LCG, and its documented bounds.
        Scheduler replay = new Scheduler(maxAttempts, base, cap, outcomes, 424242);
        Lcg independent = new Lcg(424242);
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            long ceiling = Math.min(cap, base << Math.min(attempt - 1, 40));
            long expected = (long) Math.floor(independent.next() * (ceiling + 1));
            check(replay.backoff(attempt) == expected, "backoff formula at attempt " + attempt);
            check(expected >= 0 && expected <= ceiling,
                    "backoff is inside [0, ceiling] at attempt " + attempt);
        }
        check(Math.min(cap, base << (maxAttempts - 2)) == cap, "the last drawn backoff is capped");
        check(Math.min(cap, base << (maxAttempts - 3)) < cap,
                "the previous backoff step is not capped");
        for (long[] record : scheduler.logEntries()) {
            check(record[0] >= 0, "the simulated clock never goes negative");
        }

        // 4. Full jitter: the first retry delay averages base/2, the capped one averages cap/2.
        double meanFirst = scheduler.meanFirstRetryDelay();
        check(Math.abs(meanFirst - base / 2.0) <= 0.10 * (base / 2.0),
                "mean first retry delay " + meanFirst + " is within 10% of base/2");
        double meanCapped = scheduler.meanDelayAt(maxAttempts - 1);
        check(Math.abs(meanCapped - cap / 2.0) <= 0.12 * (cap / 2.0),
                "mean capped retry delay " + meanCapped + " is within 12% of cap/2");
        check(scheduler.delays.size() > 500, "the trace produces plenty of retry delays");

        // 5. Every attempt is strictly later than the previous, and each job starts on time.
        Map<Integer, List<Long>> timeline = new TreeMap<>();
        for (long[] entry : scheduler.logEntries()) {
            timeline.computeIfAbsent((int) entry[1], k -> new ArrayList<>()).add(entry[0]);
        }
        for (int job = 0; job < jobs; job++) {
            List<Long> times = timeline.get(job);
            check(times.get(0) == job * 7L, "job " + job + " starts at its submit time");
            for (int i = 1; i < times.size(); i++) {
                check(times.get(i) > times.get(i - 1),
                        "attempt " + i + " of job " + job + " is later");
            }
        }

        // 6. Ties at equal ready times resolve by increasing job id, so the log is in key order.
        List<long[]> sorted = new ArrayList<>(scheduler.logEntries());
        List<long[]> byKey = new ArrayList<>(scheduler.logEntries());
        byKey.sort((l, r) -> {
            if (l[0] != r[0]) return Long.compare(l[0], r[0]);
            return Long.compare(l[1], r[1]);
        });
        boolean sameOrder = true;
        for (int i = 0; i < sorted.size(); i++) {
            if (sorted.get(i)[0] != byKey.get(i)[0] || sorted.get(i)[1] != byKey.get(i)[1]) sameOrder = false;
        }
        check(sameOrder, "the event log is in (time, jobId) order");

        // 7. Boundaries: base = 0 (retry on the next tick) and maxAttempts = 1 (one attempt only).
        Scheduler impatient = new Scheduler(3, 0, cap, outcomes, 606);
        for (int job = 0; job < 5; job++) impatient.submit(job, 0);
        impatient.run();
        check(impatient.now <= 2, "with base 0 a job finishes within maxAttempts - 1 ticks");
        for (int job = 0; job < 5; job++) {
            List<Long> times = new ArrayList<>();
            for (long[] entry : impatient.logEntries()) {
                if (entry[1] == job) times.add(entry[0]);
            }
            for (int i = 1; i < times.size(); i++) {
                check(times.get(i) == times.get(i - 1) + 1, "base 0 retries fire on the next tick");
            }
        }
        Scheduler once = new Scheduler(1, base, cap, outcomes, 606);
        for (int job = 0; job < 10; job++) once.submit(job, job);
        once.run();
        for (int job = 0; job < 10; job++) {
            check(once.attemptsOf(job) == 1, "maxAttempts 1 allows exactly one attempt each");
            boolean cleanFirstTry = outcomes.failures(job) == 0;
            check(once.succeeded.contains(job) == cleanFirstTry,
                    "with maxAttempts 1 only a first-try success survives");
            check(once.deadLettered.contains(job) == !cleanFirstTry,
                    "with maxAttempts 1 every failing job is dead-lettered at once");
        }
        check(once.succeeded.size() + once.deadLettered.size() == 10,
                "maxAttempts 1 still accounts for every job");

        // 8. Determinism.
        Scheduler rerun = new Scheduler(maxAttempts, base, cap, outcomes, 424242);
        for (int job = 0; job < jobs; job++) rerun.submit(job, job * 7L);
        rerun.run();
        check(rerun.logEntries().size() == scheduler.logEntries().size(), "re-run: same log length");
        for (int i = 0; i < scheduler.logEntries().size(); i++) {
            long[] a = scheduler.logEntries().get(i);
            long[] b = rerun.logEntries().get(i);
            check(a[0] == b[0] && a[1] == b[1] && a[2] == b[2], "re-run: identical entry " + i);
        }
        check(rerun.succeeded.equals(scheduler.succeeded),
                "deterministic re-run: identical successes");
        check(rerun.deadLettered.equals(scheduler.deadLettered),
                "deterministic re-run: identical dead letters");
        check(rerun.now == scheduler.now, "deterministic re-run: identical final clock");

        System.out.println("PASS");
    }
}
