// Task 954 - Implement CI pipeline that runs on multiple JDKs and OSes with resource limits.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The pipeline is an explicit in-memory model: a (JDK x OS) matrix of jobs, each with CPU/memory
// requirements, a declared pipeline resource limit, a fixed pool of runners of known capacity, and a
// deterministic list scheduler that places every admitted job on the least-loaded runner able to host
// it.  Outcomes are simulated: flaky jobs fail their first attempt and pass the retry, forced failures
// are injected per job id, and the gate fails whenever a *required* job is rejected or fails.
//
// Verified: matrix cardinality and ordering, resource fit of every admission, rejection of a job above
// the declared limit and of a job no runner can host, the makespan lower bound ceil(total/runners) plus
// a hand-checked tiny schedule, flaky retry accounting, gate propagation for required vs optional jobs,
// and the empty-matrix / single-runner boundaries.  Simplification: durations come from a formula and
// nothing is executed - the modelled surface is scheduling, limits and gating.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Job
    {
        public readonly string Id;
        public readonly int Cpu;
        public readonly int MemMiB;
        public readonly int DurationMs;
        public readonly bool Required;
        public readonly bool Flaky;

        public Job(string id, int cpu, int memMiB, int durationMs, bool required, bool flaky)
        {
            Id = id;
            Cpu = cpu;
            MemMiB = memMiB;
            DurationMs = durationMs;
            Required = required;
            Flaky = flaky;
        }
    }

    private sealed class Runner
    {
        public readonly int Cpu;
        public readonly int MemMiB;
        public int LoadMs;

        public Runner(int cpu, int memMiB)
        {
            Cpu = cpu;
            MemMiB = memMiB;
        }
    }

    private sealed class Outcome
    {
        public readonly List<string> Scheduled = new List<string>();
        public readonly List<string> Rejected = new List<string>();
        public readonly List<string> Retried = new List<string>();
        public readonly List<string> Failed = new List<string>();
        public readonly List<(string JobId, int Runner, int JobCpu, int JobMem)> Assignments = new List<(string, int, int, int)>();
        public int MakespanMs;
        public bool GateFailed;
    }

    /** Deterministic matrix expansion; job ids are stable because the loops are ordered. */
    private static List<Job> BuildMatrix(string[] jdks, string[] oses)
    {
        var jobs = new List<Job>();
        for (int j = 0; j < jdks.Length; j++)
        {
            for (int o = 0; o < oses.Length; o++)
            {
                string os = oses[o];
                int cpu = os == "windows" ? 4 : 2;
                int mem = os == "windows" ? 4096 : os == "macos" ? 3072 : 2048;
                int duration = 1000 + 250 * j + 500 * o;
                jobs.Add(new Job("jdk" + jdks[j] + "-" + os, cpu, mem, duration, j == 0 && o == 0, os == "macos" && jdks[j] == "23"));
            }
        }
        return jobs;
    }

    private static Outcome RunPipeline(List<Job> jobs, Runner[] runners, int maxCpu, int maxMemMiB, List<string> alwaysFail)
    {
        var outcome = new Outcome();
        foreach (Job job in jobs)
        {
            if (job.Cpu > maxCpu || job.MemMiB > maxMemMiB)
            {
                outcome.Rejected.Add(job.Id + " (exceeds pipeline limit cpu<=" + maxCpu +
                    " mem<=" + maxMemMiB.ToString(CultureInfo.InvariantCulture) + ")");
                if (job.Required) outcome.GateFailed = true;
                continue;
            }
            int chosen = -1;
            for (int r = 0; r < runners.Length; r++)
            {
                Runner candidate = runners[r];
                if (candidate.Cpu < job.Cpu || candidate.MemMiB < job.MemMiB) continue;
                if (chosen < 0 || candidate.LoadMs < runners[chosen].LoadMs) chosen = r;
            }
            if (chosen < 0)
            {
                outcome.Rejected.Add(job.Id + " (no runner offers cpu=" + job.Cpu + " mem=" + job.MemMiB + ")");
                if (job.Required) outcome.GateFailed = true;
                continue;
            }
            runners[chosen].LoadMs += job.DurationMs;
            outcome.Assignments.Add((job.Id, chosen, job.Cpu, job.MemMiB));
            outcome.Scheduled.Add(job.Id);
            if (alwaysFail.Contains(job.Id))
            {
                outcome.Failed.Add(job.Id);
                if (job.Required) outcome.GateFailed = true;
            }
            else if (job.Flaky)
            {
                outcome.Retried.Add(job.Id); // first attempt fails, the retry succeeds
            }
        }
        foreach (Runner runner in runners) outcome.MakespanMs = Math.Max(outcome.MakespanMs, runner.LoadMs);
        return outcome;
    }

    private static ulong Fingerprint(Outcome outcome)
    {
        var sb = new StringBuilder();
        sb.Append(outcome.MakespanMs).Append('|').Append(outcome.GateFailed ? 'F' : 'O').Append('|');
        foreach (string id in outcome.Scheduled) sb.Append(id).Append(',');
        sb.Append('|');
        foreach ((string jobId, int runner, int cpu, int mem) in outcome.Assignments)
            sb.Append(jobId).Append('@').Append(runner).Append(':').Append(cpu).Append('/').Append(mem).Append(';');
        unchecked
        {
            ulong hash = 14695981039346656037UL;
            foreach (char ch in sb.ToString())
            {
                hash ^= ch;
                hash *= 1099511628211UL;
            }
            return hash;
        }
    }

    private static long TotalWork(List<Job> jobs)
    {
        long total = 0;
        foreach (Job job in jobs) total += job.DurationMs;
        return total;
    }

    public static void Main()
    {
        string[] jdks = { "17", "21", "23" };
        string[] oses = { "linux", "windows", "macos" };
        List<Job> matrix = BuildMatrix(jdks, oses);
        List<Job> matrixAgain = BuildMatrix(jdks, oses);

        Check(matrix.Count == jdks.Length * oses.Length, "the matrix has |JDK| x |OS| = " + matrix.Count + " jobs");
        bool orderStable = matrix.Count == matrixAgain.Count;
        for (int i = 0; orderStable && i < matrix.Count; i++) orderStable &= matrix[i].Id == matrixAgain[i].Id;
        Check(orderStable, "expanding the same matrix twice produces the same job order");
        Check(matrix[0].Required, "the first job (jdk17-linux) is the required gate job");

        var runners = new[] { new Runner(4, 8192), new Runner(4, 8192), new Runner(4, 8192) };
        Outcome full = RunPipeline(matrix, runners, 4, 8192, new List<string>());
        Check(full.Scheduled.Count == matrix.Count && full.Rejected.Count == 0, "every job is admitted when the pool is adequate");
        long lowerBound = (TotalWork(matrix) + runners.Length - 1) / runners.Length;
        Check(full.MakespanMs >= lowerBound, "makespan " + full.MakespanMs + "ms >= ceil(total/runners) = " + lowerBound + "ms");

        bool fit = full.Assignments.Count == matrix.Count;
        foreach ((string jobId, int runner, int cpu, int mem) in full.Assignments)
        {
            Job job = matrix.Find(x => x.Id == jobId)!;
            fit &= job.Cpu <= runners[runner].Cpu && job.MemMiB <= runners[runner].MemMiB && cpu == job.Cpu && mem == job.MemMiB;
        }
        Check(fit, "every assignment satisfies the runner's cpu and memory capacity");

        bool retriedOnce = full.Retried.Count > 0 && full.Failed.Count == 0;
        foreach (string id in full.Retried)
        {
            Job job = matrix.Find(x => x.Id == id)!;
            retriedOnce &= job.Flaky && id == "jdk23-macos";
        }
        Check(retriedOnce, "the flaky macos/jdk23 job is retried exactly once and ends green");

        var tiny = new List<Job>
        {
            new Job("a", 2, 1024, 1000, false, false),
            new Job("b", 2, 1024, 2000, false, false),
            new Job("c", 2, 1024, 3000, false, false),
        };
        Runner[] two = { new Runner(2, 2048), new Runner(2, 2048) };
        Check(RunPipeline(tiny, two, 2, 2048, new List<string>()).MakespanMs == 4000,
            "the hand-checked three-job/two-runner schedule finishes at 4000ms");

        var oversized = new List<Job> { new Job("stress", 4, 16384, 1000, true, false) };
        Outcome rejected = RunPipeline(oversized, new[] { new Runner(4, 8192) }, 4, 8192, new List<string>());
        Check(rejected.Rejected.Count == 1 && rejected.Rejected[0].Contains("exceeds pipeline limit"),
            "a job above the declared pipeline limit is rejected with a reason");
        Check(rejected.GateFailed && rejected.Scheduled.Count == 0, "rejecting a required job fails the gate");

        var smallRunner = new List<Job> { new Job("wide", 4, 4096, 1000, false, false) };
        Check(RunPipeline(smallRunner, new[] { new Runner(2, 8192) }, 8, 8192, new List<string>())
            .Rejected[0].Contains("no runner offers cpu=4"), "a job no runner can host is rejected for capacity, not for the limit");

        Outcome requiredFails = RunPipeline(matrix, new[] { new Runner(4, 8192) }, 4, 8192, new List<string> { "jdk17-linux" });
        Check(requiredFails.GateFailed && requiredFails.Failed.Count == 1, "a failing required job fails the pipeline gate");
        Outcome optionalFails = RunPipeline(matrix, new[] { new Runner(4, 8192) }, 4, 8192, new List<string> { "jdk21-linux" });
        Check(!optionalFails.GateFailed && optionalFails.Failed.Count == 1, "a failing non-required job does not fail the gate");

        Outcome empty = RunPipeline(new List<Job>(), new[] { new Runner(2, 1024) }, 2, 1024, new List<string>());
        Check(empty.Scheduled.Count == 0 && empty.MakespanMs == 0 && !empty.GateFailed,
            "the empty matrix schedules nothing, spends 0ms and passes the gate");

        Outcome singleRunner = RunPipeline(matrix, new[] { new Runner(4, 8192) }, 4, 8192, new List<string>());
        Check(singleRunner.Scheduled.Count == matrix.Count && singleRunner.MakespanMs == TotalWork(matrix),
            "with a single runner the makespan equals the total work");

        Outcome repeat = RunPipeline(matrix, new[] { new Runner(4, 8192), new Runner(4, 8192), new Runner(4, 8192) }, 4, 8192, new List<string>());
        Check(Fingerprint(full) == Fingerprint(repeat), "the schedule fingerprint is identical across two runs");

        Console.WriteLine("matrix: " + jdks.Length + " JDKs x " + oses.Length + " OSes = " + matrix.Count +
            " jobs, total work " + TotalWork(matrix) + "ms, makespan " + full.MakespanMs + "ms (bound " + lowerBound + "ms)");
        Console.WriteLine("limits: cpu<=" + 4 + " mem<=8192 enforced; oversized and unrunnable jobs rejected with reasons");
        Console.WriteLine("flaky: " + full.Retried.Count + " retried job(s); gate requires " +
            (requiredFails.GateFailed ? "required jobs to pass" : "nothing"));
        Console.WriteLine("PASS");
    }
}
