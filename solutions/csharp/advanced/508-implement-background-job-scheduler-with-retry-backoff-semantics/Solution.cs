// Task 508 - Implement background job scheduler with retry/backoff semantics. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
//
// An event-driven, single-worker job scheduler on a virtual clock, plus the retry policy that
// makes it interesting. Jobs carry a priority, a ready tick, a duration and a retry budget; the
// worker always takes the pending job with the smallest (readyTick, priority, submission sequence)
// key, so ties are broken FIFO and a lower priority number means a more urgent job. Failures are
// deterministic: job j fails its first F_j attempts and succeeds on attempt F_j + 1, where
// F_j = j % 3 for the bulk workload. After a failed attempt the job becomes ready again at
// finish + delay, where delay = min(cap, base * 2^(attempt-1)) + jitter and the jitter is a
// deterministic value in [0, jitterFraction * base] derived from a counter-indexed splitmix64
// stream keyed by the job id - no wall clock, no real threads, no unseeded randomness.
// Documented checks:
//   * a hand-checked single job (base 10, cap 40, jitter fraction 0.5, 4 allowed attempts) shows
//     the exponential-then-flat schedule min(cap, 10 * 2^(a-1)) + jitter with the jitter inside
//     [0, 5] (the cap bounds the exponential part only, so a delay may exceed 40 slightly), and
//     every retry start tick equals finish + delay;
//   * a job that always fails ends in the dead-letter list after exactly maxAttempts attempts and
//     is never scheduled again, while a job failing F < maxAttempts times ends with exactly F + 1
//     attempts and a success;
//   * for the whole 30-job workload every job's attempt count equals min(F_j + 1, maxAttempts), the
//     terminal outcome is exactly one of success/dead-letter per job, no attempt ever starts before
//     its ready tick, attempts never overlap (single worker), and the exact identity
//     busy + idle == makespan holds (32 retries and a single dead letter for the crafted workload);
//   * the event log is bit-identical to an independent linear-scan scheduler implementation
//     (heap vs list) on the same workload;
//   * crafted ordering cases pin the priority policy: three jobs ready together run in priority
//     order, equal priorities keep submission order, and a high-priority job that becomes ready
//     while a low-priority job runs starts immediately afterwards;
//   * the jitter stays inside its documented band, is not constant, and two runs of the same
//     workload produce identical event logs.
using System;
using System.Collections.Generic;

internal static class Deterministic
{
    internal static ulong Mix64(ulong x)
    {
        x += 0x9E3779B97F4A7C15UL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        return x ^ (x >> 31);
    }

    /** Uniform in [0, 1) from the key (stream, a, b). */
    internal static double Fraction(long stream, long a, long b)
    {
        ulong h = Mix64((ulong)stream * 0x9E3779B97F4A7C15UL + (ulong)a * 0xD1B54A32D192ED03UL
                + (ulong)b * 0x165667B19E3779F9UL);
        return (h >> 11) * (1.0 / 9007199254740992.0);
    }
}

internal class JobSpec
{
    internal long Id;
    internal long Priority;
    internal long ReadyTick;
    internal long Duration;
    internal int MaxAttempts;
    internal int FailuresBeforeSuccess;

    internal JobSpec Clone() => (JobSpec)MemberwiseClone();
}

internal readonly struct JobEvent
{
    internal JobEvent(long jobId, int attempt, long start, long finish, string outcome)
    {
        JobId = jobId;
        Attempt = attempt;
        Start = start;
        Finish = finish;
        Outcome = outcome;
    }

    internal long JobId { get; }

    internal int Attempt { get; }

    internal long Start { get; }

    internal long Finish { get; }

    internal string Outcome { get; }
}

internal static class Backoff
{
    internal const long BaseDelay = 10;
    internal const long Cap = 40;
    internal const double JitterFraction = 0.5;
    internal const long Stream = 20240608;

    internal static long Exponential(int failedAttempt)
    {
        long delay = BaseDelay;
        for (int i = 1; i < failedAttempt; i++)
        {
            if (delay >= Cap) return Cap;
            delay *= 2;
        }
        return Math.Min(Cap, delay);
    }

    internal static long Jitter(long jobId, int failedAttempt)
    {
        double fraction = Deterministic.Fraction(Stream, jobId, failedAttempt);
        return (long)(fraction * JitterFraction * BaseDelay);
    }

    internal static long Delay(long jobId, int failedAttempt) => Exponential(failedAttempt) + Jitter(jobId, failedAttempt);
}

/** Binary-heap scheduler. */
internal class HeapScheduler
{
    private readonly List<(long Ready, long Priority, long Seq, JobSpec Job, int Attempt)> heap =
            new List<(long, long, long, JobSpec, int)>();

    internal List<JobEvent> Run(List<JobSpec> specs, long horizon)
    {
        var events = new List<JobEvent>();
        var order = new Dictionary<long, long>();
        for (int i = 0; i < specs.Count; i++)
        {
            order[specs[i].Id] = i;
            Push((specs[i].ReadyTick, specs[i].Priority, i, specs[i].Clone(), 1));
        }
        long clock = 0;
        while (heap.Count > 0 && clock <= horizon)
        {
            (long ready, long Priority, long Seq, JobSpec job, int attempt) = Pop();
            if (ready > horizon) break;
            long start = Math.Max(clock, ready);
            long finish = start + job.Duration;
            bool success = attempt > job.FailuresBeforeSuccess;
            bool exhausted = !success && attempt >= job.MaxAttempts;
            string outcome = success ? "success" : exhausted ? "dead-letter" : "retry";
            events.Add(new JobEvent(job.Id, attempt, start, finish, outcome));
            clock = finish;
            if (!success && !exhausted)
            {
                long nextReady = finish + Backoff.Delay(job.Id, attempt);
                Push((nextReady, job.Priority, Seq, job, attempt + 1));
            }
        }
        return events;
    }

    private void Push((long Ready, long Priority, long Seq, JobSpec Job, int Attempt) item)
    {
        heap.Add(item);
        int index = heap.Count - 1;
        while (index > 0)
        {
            int parent = (index - 1) / 2;
            if (Compare(heap[parent], heap[index]) <= 0) break;
            (heap[parent], heap[index]) = (heap[index], heap[parent]);
            index = parent;
        }
    }

    private (long, long, long, JobSpec, int) Pop()
    {
        (long, long, long, JobSpec, int) top = heap[0];
        heap[0] = heap[heap.Count - 1];
        heap.RemoveAt(heap.Count - 1);
        int index = 0;
        while (true)
        {
            int left = 2 * index + 1;
            int right = left + 1;
            int best = index;
            if (left < heap.Count && Compare(heap[left], heap[best]) < 0) best = left;
            if (right < heap.Count && Compare(heap[right], heap[best]) < 0) best = right;
            if (best == index) break;
            (heap[best], heap[index]) = (heap[index], heap[best]);
            index = best;
        }
        return top;
    }

    private static int Compare((long Ready, long Priority, long Seq, JobSpec Job, int Attempt) a,
            (long Ready, long Priority, long Seq, JobSpec Job, int Attempt) b)
    {
        if (a.Ready != b.Ready) return a.Ready.CompareTo(b.Ready);
        if (a.Priority != b.Priority) return a.Priority.CompareTo(b.Priority);
        return a.Seq.CompareTo(b.Seq);
    }
}

/** Independent reference: linear scan for the minimum key instead of a heap. */
internal class ScanScheduler
{
    internal List<JobEvent> Run(List<JobSpec> specs, long horizon)
    {
        var events = new List<JobEvent>();
        var pending = new List<(long Ready, long Priority, long Seq, JobSpec Job, int Attempt)>();
        for (int i = 0; i < specs.Count; i++)
        {
            pending.Add((specs[i].ReadyTick, specs[i].Priority, i, specs[i].Clone(), 1));
        }
        long clock = 0;
        while (pending.Count > 0 && clock <= horizon)
        {
            int best = -1;
            for (int i = 0; i < pending.Count; i++)
            {
                if (pending[i].Ready > horizon) continue;
                if (best < 0 || Compare(pending[i], pending[best]) < 0) best = i;
            }
            if (best < 0) break;
            (long ready, long priority, long seq, JobSpec job, int attempt) = pending[best];
            pending.RemoveAt(best);
            long start = Math.Max(clock, ready);
            long finish = start + job.Duration;
            bool success = attempt > job.FailuresBeforeSuccess;
            bool exhausted = !success && attempt >= job.MaxAttempts;
            events.Add(new JobEvent(job.Id, attempt, start, finish,
                    success ? "success" : exhausted ? "dead-letter" : "retry"));
            clock = finish;
            if (!success && !exhausted)
            {
                pending.Add((finish + Backoff.Delay(job.Id, attempt), priority, seq, job, attempt + 1));
            }
        }
        return events;
    }

    private static int Compare((long Ready, long Priority, long Seq, JobSpec Job, int Attempt) a,
            (long Ready, long Priority, long Seq, JobSpec Job, int Attempt) b)
    {
        if (a.Ready != b.Ready) return a.Ready.CompareTo(b.Ready);
        if (a.Priority != b.Priority) return a.Priority.CompareTo(b.Priority);
        return a.Seq.CompareTo(b.Seq);
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- one hand-checked job ---------------------------------------------------------------------
        var lonely = new JobSpec
        {
            Id = 1, Priority = 0, ReadyTick = 0, Duration = 5, MaxAttempts = 4,
            FailuresBeforeSuccess = 9,
        };
        List<JobEvent> lonelyLog = new HeapScheduler().Run(new List<JobSpec> { lonely }, 10000);
        Check(lonelyLog.Count == 4, "the always-failing job ran exactly maxAttempts times");
        Check(Backoff.Delay(1, 1) == 10 + Backoff.Jitter(1, 1), "first delay is base + jitter");
        Check(Backoff.Delay(1, 2) == 20 + Backoff.Jitter(1, 2), "second delay doubles the base");
        Check(Backoff.Delay(1, 3) == 40 + Backoff.Jitter(1, 3), "third delay hits the cap");
        Check(Backoff.Delay(1, 4) == 40 + Backoff.Jitter(1, 4), "later delays stay at the cap");
        long expectedStart = 0;
        for (int i = 0; i < lonelyLog.Count; i++)
        {
            Check(lonelyLog[i].Attempt == i + 1, "attempt numbering is sequential");
            Check(lonelyLog[i].Start == expectedStart, "start tick " + i + " matches the schedule");
            Check(lonelyLog[i].Finish == lonelyLog[i].Start + 5, "the duration is honoured");
            Check(lonelyLog[i].Outcome == (i == 3 ? "dead-letter" : "retry"), "outcome at attempt " + (i + 1));
            if (i < 3)
            {
                expectedStart = lonelyLog[i].Finish + Backoff.Delay(1, i + 1);
            }
        }
        Check(lonelyLog[3].Outcome == "dead-letter", "the exhausted job is dead-lettered");
        for (int attempt = 1; attempt <= 8; attempt++)
        {
            long jitter = Backoff.Jitter(1, attempt);
            long baseDelay = Math.Min(40, 10L << Math.Min(attempt - 1, 3));
            Check(jitter >= 0 && jitter <= 5, "the jitter stays inside [0, jitterFraction * base] = [0, 5]");
            Check(Backoff.Delay(1, attempt) == baseDelay + jitter, "delay = min(cap, base * 2^(a-1)) + jitter");
        }

        // --- a job that recovers ------------------------------------------------------------------------
        var recovering = new JobSpec
        {
            Id = 2, Priority = 0, ReadyTick = 0, Duration = 3, MaxAttempts = 4,
            FailuresBeforeSuccess = 2,
        };
        var recoveringLog = new HeapScheduler().Run(new List<JobSpec> { recovering }, 10000);
        Check(recoveringLog.Count == 3, "the recovering job ran F + 1 = 3 times");
        Check(recoveringLog[2].Outcome == "success", "it finally succeeded");
        Check(recoveringLog[recoveringLog.Count - 1].Start >= 10 + 20, "after both backoff delays");
        var deadLetterOnly = new HeapScheduler().Run(new List<JobSpec> { lonely }, 10000);
        Check(deadLetterOnly.Count == 4, "no rescheduling after the dead letter");
        var oneShot = new JobSpec
        {
            Id = 3, Priority = 0, ReadyTick = 0, Duration = 1, MaxAttempts = 1,
            FailuresBeforeSuccess = 0,
        };
        var oneShotLog = new HeapScheduler().Run(new List<JobSpec> { oneShot }, 100);
        Check(oneShotLog.Count == 1 && oneShotLog[0].Outcome == "success", "maxAttempts = 1 still succeeds");

        // --- the priority policy --------------------------------------------------------------------------
        var ordered = new List<JobSpec>
        {
            new JobSpec { Id = 10, Priority = 2, ReadyTick = 0, Duration = 2, MaxAttempts = 1 },
            new JobSpec { Id = 11, Priority = 0, ReadyTick = 0, Duration = 2, MaxAttempts = 1 },
            new JobSpec { Id = 12, Priority = 1, ReadyTick = 0, Duration = 2, MaxAttempts = 1 },
        };
        List<JobEvent> orderedLog = new HeapScheduler().Run(ordered, 100);
        Check(orderedLog[0].JobId == 11 && orderedLog[1].JobId == 12 && orderedLog[2].JobId == 10,
                "lower priority numbers run first");
        Check(orderedLog[0].Start == 0 && orderedLog[1].Start == 2 && orderedLog[2].Start == 4,
                "a single worker leaves no gaps");

        var ties = new List<JobSpec>
        {
            new JobSpec { Id = 20, Priority = 1, ReadyTick = 0, Duration = 1, MaxAttempts = 1 },
            new JobSpec { Id = 21, Priority = 1, ReadyTick = 0, Duration = 1, MaxAttempts = 1 },
            new JobSpec { Id = 22, Priority = 1, ReadyTick = 0, Duration = 1, MaxAttempts = 1 },
        };
        List<JobEvent> tieLog = new HeapScheduler().Run(ties, 100);
        Check(tieLog[0].JobId == 20 && tieLog[1].JobId == 21 && tieLog[2].JobId == 22,
                "equal priorities keep submission order");

        var later = new List<JobSpec>
        {
            new JobSpec { Id = 30, Priority = 5, ReadyTick = 0, Duration = 6, MaxAttempts = 1 },
            new JobSpec { Id = 31, Priority = 0, ReadyTick = 3, Duration = 2, MaxAttempts = 1 },
        };
        List<JobEvent> laterLog = new HeapScheduler().Run(later, 100);
        Check(laterLog[0].JobId == 30 && laterLog[0].Finish == 6, "the running job is not preempted");
        Check(laterLog[1].JobId == 31 && laterLog[1].Start == 6,
                "the urgent job starts immediately after the busy worker frees up");

        // --- the bulk workload -----------------------------------------------------------------------------
        var workload = new List<JobSpec>();
        for (long i = 0; i < 30; i++)
        {
            workload.Add(new JobSpec
            {
                Id = 100 + i,
                Priority = i % 3,
                ReadyTick = i,
                Duration = 1 + i % 5,
                MaxAttempts = 4,
                FailuresBeforeSuccess = (int)(i % 3),
            });
        }
        workload.Add(new JobSpec
        {
            Id = 900, Priority = 0, ReadyTick = 0, Duration = 4, MaxAttempts = 3, FailuresBeforeSuccess = 99,
        });
        List<JobEvent> log = new HeapScheduler().Run(workload, 100000);
        List<JobEvent> reference = new ScanScheduler().Run(workload, 100000);
        Check(log.Count == reference.Count, "heap and scan schedulers agree on the event count");
        for (int i = 0; i < log.Count; i++)
        {
            Check(log[i].JobId == reference[i].JobId && log[i].Attempt == reference[i].Attempt
                    && log[i].Start == reference[i].Start && log[i].Finish == reference[i].Finish
                    && log[i].Outcome == reference[i].Outcome,
                    "heap and scan schedulers agree at event " + i);
        }

        var attempts = new Dictionary<long, int>();
        var terminal = new Dictionary<long, string>();
        long busy = 0;
        long idle = 0;
        long previousFinish = 0;
        long retries = 0;
        long deadLetters = 0;
        for (int i = 0; i < log.Count; i++)
        {
            JobEvent entry = log[i];
            attempts.TryGetValue(entry.JobId, out int count);
            attempts[entry.JobId] = count + 1;
            busy += entry.Finish - entry.Start;
            Check(entry.Finish > entry.Start, "attempts take positive time");
            if (i > 0)
            {
                Check(entry.Start >= previousFinish, "one worker only: attempts never overlap");
                idle += entry.Start - previousFinish;
            }
            previousFinish = entry.Finish;
            if (entry.Outcome == "retry") retries += 1;
            if (entry.Outcome == "dead-letter")
            {
                deadLetters += 1;
                terminal[entry.JobId] = "dead-letter";
            }
            if (entry.Outcome == "success") terminal[entry.JobId] = "success";
        }
        Check(terminal.Count == workload.Count, "every job reached a terminal state exactly once");
        Check(deadLetters == 1, "exactly one job was dead-lettered");
        Check(terminal[900] == "dead-letter", "the always-failing job was the one");
        Check(retries == 32, "retries = sum(F) over the bulk + 2 for the dead-lettered job (got " + retries + ")");
        foreach (JobSpec spec in workload)
        {
            attempts.TryGetValue(spec.Id, out int count);
            int expected = Math.Min(spec.FailuresBeforeSuccess + 1, spec.MaxAttempts);
            Check(count == expected, "job " + spec.Id + " ran " + expected + " attempts");
            Check(spec.Id == 900 ? terminal[spec.Id] == "dead-letter" : terminal[spec.Id] == "success",
                    "terminal state for job " + spec.Id);
        }
        long makespan = previousFinish;
        Check(busy + idle == makespan, "busy + idle == makespan");
        Check(busy == SumDurations(log), "busy time equals the sum of attempt durations");
        Check(idle >= 0 && busy > 0.5 * makespan, "the worker is kept busy");
        Check(makespan >= busy && makespan <= 100000, "the makespan fits inside the horizon");

        // --- jitter and determinism -------------------------------------------------------------------------
        var jitters = new HashSet<long>();
        for (int i = 1; i <= 5; i++) jitters.Add(Backoff.Jitter(900, i));
        var allJitters = new HashSet<long>();
        for (long jobId = 100; jobId < 130; jobId++)
        {
            for (int attempt = 1; attempt <= 4; attempt++) allJitters.Add(Backoff.Jitter(jobId, attempt));
        }
        Check(allJitters.Count >= 5, "the jitter takes several distinct values");
        foreach (long value in allJitters)
        {
            Check(value >= 0 && value <= Backoff.JitterFraction * Backoff.BaseDelay,
                    "jitter band (saw " + value + ")");
        }
        Check(jitters.Count >= 1, "the jitter is defined for the dead-lettered job too");
        List<JobEvent> repeatWorkload = new HeapScheduler().Run(workload, 100000);
        for (int i = 0; i < log.Count; i++)
        {
            Check(log[i].Start == repeatWorkload[i].Start && log[i].Attempt == repeatWorkload[i].Attempt
                    && log[i].Outcome == repeatWorkload[i].Outcome,
                    "the scheduler is deterministic at event " + i);
        }
        long repeatMakespan = repeatWorkload[repeatWorkload.Count - 1].Finish;
        Check(repeatMakespan == makespan, "identical makespan on the repeat run");

        Console.WriteLine("PASS");
    }

    private static long SumDurations(List<JobEvent> log)
    {
        long total = 0;
        foreach (JobEvent entry in log) total += entry.Finish - entry.Start;
        return total;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
