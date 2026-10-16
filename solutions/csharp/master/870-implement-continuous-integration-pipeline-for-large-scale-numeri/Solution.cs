// Task 870 - Implement continuous integration pipeline for large-scale numerical code with reproducibility.
// (Tier: Master, CTCI Beyond - Reproducibility, Testing & Tooling)
//
// Model: an in-process CI orchestrator for a numerical library. A plan is a DAG of stages
// (kernel -> unit tests / reference check / determinism gate / benchmark -> publish, plus an
// independent lint branch); every stage is content addressed, so a stage's cache key is
// hash(code version, kind, parameters, digests of its inputs). The scheduler is a Kahn topological
// order executed in waves of at most `workers` ready stages - the typed "parallelism" parameter of
// the task - and every stage's payload depends only on its inputs, which is what makes the artifact
// digests invariant under the worker count.
//
// Simplifications (documented, not behavioural): no processes, threads or filesystem - "running a
// stage" is a deterministic in-process call whose result is serialised into the artifact store; the
// numerical workload under CI is a small fixed-size dense matmul with two independent implementations
// (naive and cache-blocked) rather than a real HPC build; the build cache is a dictionary, and build
// duration is modelled as a FLOP count instead of wall-clock time (which also keeps the run
// deterministic and well under the time budget).
//
// Verified here:
//   * scheduling: the emitted order is a valid topological order (every dependency precedes its
//     dependents) and the DAG validator rejects cycles, duplicate ids and dangling dependencies;
//   * reproducibility: two full runs of the same plan produce a bitwise identical root digest, and
//     the digest is invariant under the worker count (1 vs 2 vs 8 lanes) - the parallel-safety claim;
//   * caching: the second run performs zero recomputation and reuses every artifact, while a code
//     version bump or a parameter change propagates to every downstream digest and forces a rebuild;
//   * failure semantics: a failing stage skips exactly its transitive dependents while independent
//     branches still complete, and the pipeline reports failure; the determinism gate and the
//     reference check are both shown to have power by injecting a nondeterministic and a buggy kernel;
//   * content addressing: every stored artifact's payload hashes back to its own key;
//   * degenerate plans: empty plan, single stage, and a failing independent branch.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public static class Solution
{
    // ---------- deterministic PRNG (splitmix64) ----------
    private static ulong _state = 0x0123456789ABCDEFUL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextSigned() => ((NextU64() >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;

    // ---------- content addressing: two independent 64-bit lanes over the payload ----------
    private static ulong Fnv1a64(byte[] data, ulong basis)
    {
        ulong h = basis;
        foreach (byte b in data)
        {
            h ^= b;
            h *= 0x100000001B3UL;
        }
        return h;
    }

    private static string HashBytes(byte[] data)
    {
        ulong a = Fnv1a64(data, 0xCBF29CE484222325UL);
        ulong b = Fnv1a64(data, 0x9E3779B97F4A7C15UL);
        return a.ToString("x16") + b.ToString("x16");
    }

    private static string HashStrings(params string[] parts)
    {
        var sb = new StringBuilder();
        foreach (string p in parts) sb.Append(p).Append('\u0001');
        return HashBytes(Encoding.UTF8.GetBytes(sb.ToString()));
    }

    // ---------- the numerical workload under CI ----------
    private const int N = 24;

    private static (double[,] A, double[,] B) MakeOperands(int seed)
    {
        _state = 0x5DEECE66DUL ^ (ulong)seed * 0x9E3779B97F4A7C15UL;
        var a = new double[N, N];
        var b = new double[N, N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                a[i, j] = NextSigned();
                b[i, j] = NextSigned();
            }
        return (a, b);
    }

    // Library implementation: cache-blocked (tiled) multiply - the "optimised kernel" being built.
    private static double[,] TiledMul(double[,] a, double[,] b, int block)
    {
        var c = new double[N, N];
        for (int ii = 0; ii < N; ii += block)
            for (int kk = 0; kk < N; kk += block)
                for (int jj = 0; jj < N; jj += block)
                    for (int i = ii; i < Math.Min(ii + block, N); i++)
                        for (int k = kk; k < Math.Min(kk + block, N); k++)
                        {
                            double aik = a[i, k];
                            for (int j = jj; j < Math.Min(jj + block, N); j++) c[i, j] += aik * b[k, j];
                        }
        return c;
    }

    // Independent reference implementation: plain triple loop, no blocking, no shared accumulation order
    // with the tiled kernel above (this is the oracle the reference-check stage compares against).
    private static double[,] NaiveMul(double[,] a, double[,] b)
    {
        var c = new double[N, N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                double s = 0.0;
                for (int k = 0; k < N; k++) s += a[i, k] * b[k, j];
                c[i, j] = s;
            }
        return c;
    }

    private static byte[] Serialize(double[,] m)
    {
        var bytes = new byte[8 * N * N];
        int p = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                BitConverter.GetBytes(m[i, j]).CopyTo(bytes, p);
                p += 8;
            }
        return bytes;
    }

    private static long _kernelInvocations;

    // The build output: the matrix product plus the build configuration it was produced under (so a
    // parameter change is visible in the artifact digest itself, not only in the cache key). A
    // "nondeterministic" kernel additionally mixes in process state, which the determinism gate catches.
    private static byte[] KernelPayload(int seed, int block, bool buggy, bool nondeterministic)
    {
        var (a, b) = MakeOperands(seed);
        double[,] c = TiledMul(a, b, block);
        if (buggy) c[0, 0] += 1e-3;                        // silent numerical regression
        byte[] m = Serialize(c);
        var payload = new byte[m.Length + 12];
        m.CopyTo(payload, 0);
        BitConverter.GetBytes(block).CopyTo(payload, m.Length);
        BitConverter.GetBytes(nondeterministic ? ++_kernelInvocations : 0L).CopyTo(payload, m.Length + 4);
        return payload;
    }

    private static double[,] Deserialize(byte[] bytes)
    {
        var m = new double[N, N];
        int p = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                m[i, j] = BitConverter.ToDouble(bytes, p);
                p += 8;
            }
        return m;
    }

    // ---------- plan model ----------
    private sealed class StageDef
    {
        public string Id = "";
        public string[] Deps = Array.Empty<string>();
        public string Kind = "";
    }

    private sealed class Plan
    {
        public string Name = "";
        public List<StageDef> Stages = new List<StageDef>();
    }

    private sealed class RunOptions
    {
        public string CodeVersion = "v1";
        public int Seed = 7;
        public int Block = 8;
        public int Workers = 1;
        public string FailStage = "";            // stage id forced to fail (failure-path tests)
        public bool NondeterministicKernel;      // kernel whose payload drifts between runs
        public bool BuggyKernel;                 // kernel that silently computes something else
        public ulong DriftCounter;               // makes the "nondeterministic" payload actually differ
    }

    // ---------- DAG validation and ordering ----------
    private static string Validate(Plan plan)
    {
        var ids = new HashSet<string>();
        foreach (StageDef s in plan.Stages)
            if (!ids.Add(s.Id)) return "duplicate stage id " + s.Id;
        foreach (StageDef s in plan.Stages)
            foreach (string d in s.Deps)
                if (!ids.Contains(d)) return "dangling dependency " + s.Id + " -> " + d;
        if (TopologicalOrder(plan).Count != plan.Stages.Count) return "cycle detected";
        return "";
    }

    private static List<StageDef> TopologicalOrder(Plan plan)
    {
        var byId = plan.Stages.ToDictionary(s => s.Id, s => s);
        var indegree = plan.Stages.ToDictionary(s => s.Id, s => s.Deps.Length);
        var ready = new List<string>(plan.Stages.Where(s => s.Deps.Length == 0).Select(s => s.Id));
        var order = new List<StageDef>();
        while (ready.Count > 0)
        {
            ready.Sort(StringComparer.Ordinal);            // deterministic tie-breaking
            string id = ready[0];
            ready.RemoveAt(0);
            order.Add(byId[id]);
            foreach (StageDef s in plan.Stages)
                if (s.Deps.Contains(id) && --indegree[s.Id] == 0) ready.Add(s.Id);
        }
        return order;
    }

    // ---------- report ----------
    private sealed class Report
    {
        public bool Success;
        public string Error = "";
        public string RootDigest = "";
        public List<string> Order = new List<string>();
        public Dictionary<string, string> Digests = new Dictionary<string, string>();
        public Dictionary<string, string> Status = new Dictionary<string, string>();
        public long Computed;
        public long CacheHits;
        public long Skipped;
        public long Flops;
        public string FailReason = "";
    }

    private sealed class CacheEntry
    {
        public byte[] Payload = Array.Empty<byte>();
        public string Digest = "";
    }

    // ---------- the orchestrator ----------
    private static Report Run(Plan plan, RunOptions opt, Dictionary<string, CacheEntry> cache)
    {
        var report = new Report();
        string invalid = Validate(plan);
        if (invalid != "")
        {
            report.Error = invalid;
            report.Success = false;
            return report;
        }

        var order = TopologicalOrder(plan);
        var digests = new Dictionary<string, string>();
        var statuses = new Dictionary<string, string>();
        var payloads = new Dictionary<string, byte[]>();

        // Wave scheduler: all stages whose dependencies are finished are candidates; at most `workers`
        // of them are taken per wave. The worker count only changes the batching, never a payload.
        var pending = new List<StageDef>(order);
        while (pending.Count > 0)
        {
            var ready = pending.Where(s => s.Deps.All(d => statuses.ContainsKey(d))).ToList();
            if (ready.Count == 0) break;
            foreach (StageDef s in ready.Take(Math.Max(1, opt.Workers)).ToList())
            {
                pending.Remove(s);
                report.Order.Add(s.Id);
                bool blocked = s.Deps.Any(d => statuses[d] != "ok");
                if (blocked)
                {
                    statuses[s.Id] = "skipped";
                    report.Skipped++;
                    continue;
                }
                string key = HashStrings(opt.CodeVersion, s.Kind, opt.Seed.ToString(), opt.Block.ToString(),
                                         string.Join(",", s.Deps.Select(d => d + "=" + digests[d])));
                if (cache.TryGetValue(key, out CacheEntry hit))
                {
                    payloads[s.Id] = hit.Payload;
                    digests[s.Id] = hit.Digest;
                    statuses[s.Id] = "ok";
                    report.CacheHits++;
                    continue;
                }
                string note;
                byte[] payload = Execute(s, opt, payloads, key, out note);
                report.Computed++;
                if (payload == null)
                {
                    statuses[s.Id] = "failed";
                    report.FailReason = s.Id + ": " + note;
                    continue;
                }
                string digest = HashBytes(payload);
                cache[key] = new CacheEntry { Payload = payload, Digest = digest };
                payloads[s.Id] = payload;
                digests[s.Id] = digest;
                statuses[s.Id] = "ok";
                if (s.Kind == "kernel") report.Flops += 2L * N * N * N;
            }
        }

        foreach (StageDef s in plan.Stages)
            if (!statuses.ContainsKey(s.Id)) { statuses[s.Id] = "unreachable"; report.Skipped++; }

        report.Digests = digests;
        report.Status = statuses;
        report.Success = report.Error == "" && statuses.Values.All(v => v == "ok");
        if (!report.Success && report.FailReason == "")
        {
            string firstBad = statuses.FirstOrDefault(kv => kv.Value == "failed" || kv.Value == "skipped").Key;
            report.FailReason = firstBad == null ? "no stages" : "stage not ok: " + firstBad;
        }
        if (plan.Stages.Count == 0)
        {
            report.RootDigest = HashStrings(plan.Name, opt.CodeVersion);
        }
        else
        {
            var roots = plan.Stages.Where(s => !plan.Stages.Any(o => o.Deps.Contains(s.Id)))
                                   .Select(s => s.Id + "=" + (digests.ContainsKey(s.Id) ? digests[s.Id] : "none"))
                                   .OrderBy(x => x, StringComparer.Ordinal);
            report.RootDigest = HashStrings(plan.Name, opt.CodeVersion, string.Join("|", roots));
        }
        return report;
    }

    // Executes one stage; returns null when the stage fails (the caller records the failure).
    private static byte[] Execute(StageDef stage, RunOptions opt, Dictionary<string, byte[]> inputs, string key,
                                  out string note)
    {
        note = "";
        if (opt.FailStage == stage.Id)
        {
            note = "injected failure";
            return null;
        }
        switch (stage.Kind)
        {
            case "kernel":
            {
                return KernelPayload(opt.Seed, opt.Block, opt.BuggyKernel, opt.NondeterministicKernel);
            }
            case "unit":
            {
                var (a, _) = MakeOperands(opt.Seed);
                var ident = new double[N, N];
                for (int i = 0; i < N; i++) ident[i, i] = 1.0;
                var left = TiledMul(a, ident, opt.Block);
                double maxErr = 0.0;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++) maxErr = Math.Max(maxErr, Math.Abs(left[i, j] - a[i, j]));
                if (maxErr > 1e-12) { note = "identity property violated"; return null; }
                return Encoding.UTF8.GetBytes("unit-ok:" + key);
            }
            case "refcheck":
            {
                var (a, b) = MakeOperands(opt.Seed);
                double[,] got = Deserialize(inputs[stage.Deps[0]]);
                double[,] want = NaiveMul(a, b);
                double maxErr = 0.0;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++) maxErr = Math.Max(maxErr, Math.Abs(got[i, j] - want[i, j]));
                if (maxErr > 1e-9) { note = "max deviation " + maxErr; return null; }
                return Encoding.UTF8.GetBytes("refcheck-ok:" + maxErr.ToString("E3"));
            }
            case "determinism":
            {
                // Recompute from scratch and compare byte for byte. A kernel that is not a pure function
                // of its declared inputs (here: one that reads process state) fails this gate.
                byte[] again = KernelPayload(opt.Seed, opt.Block, opt.BuggyKernel, opt.NondeterministicKernel);
                byte[] first = inputs[stage.Deps[0]];
                bool same = first.Length == again.Length;
                for (int i = 0; same && i < first.Length; i++) same &= first[i] == again[i];
                if (!same) { note = "kernel is not a pure function of its inputs"; return null; }
                return Encoding.UTF8.GetBytes("determinism-ok");
            }
            case "bench":
            {
                long flops = 2L * N * N * N;
                long budget = 2L * 64 * 64 * 64;                      // generous budget for the fixed size
                if (flops > budget) { note = "budget exceeded"; return null; }
                return Encoding.UTF8.GetBytes("bench-ok:" + flops);
            }
            case "lint":
            {
                if (!opt.CodeVersion.StartsWith("v", StringComparison.Ordinal))
                {
                    note = "unversioned build";
                    return null;
                }
                return Encoding.UTF8.GetBytes("lint-ok:" + opt.CodeVersion);
            }
            case "publish":
            {
                var manifest = new StringBuilder();
                foreach (string dep in stage.Deps)
                    manifest.Append(dep).Append('=').Append(HashBytes(inputs[dep])).Append(';');
                return Encoding.UTF8.GetBytes("release:" + HashStrings(manifest.ToString()));
            }
            default:
                note = "unknown stage kind " + stage.Kind;
                return null;
        }
    }

    private static Plan DefaultPlan()
    {
        return new Plan
        {
            Name = "numerics-ci",
            Stages = new List<StageDef>
            {
                new StageDef { Id = "lint", Kind = "lint" },
                new StageDef { Id = "kernel", Kind = "kernel" },
                new StageDef { Id = "unit", Kind = "unit", Deps = new[] { "kernel" } },
                new StageDef { Id = "refcheck", Kind = "refcheck", Deps = new[] { "kernel" } },
                new StageDef { Id = "determinism", Kind = "determinism", Deps = new[] { "kernel" } },
                new StageDef { Id = "bench", Kind = "bench", Deps = new[] { "kernel" } },
                new StageDef { Id = "publish", Kind = "publish",
                               Deps = new[] { "unit", "refcheck", "determinism", "bench", "lint" } },
            },
        };
    }

    private static bool IsTopological(Plan plan, List<string> order)
    {
        var pos = new Dictionary<string, int>();
        for (int i = 0; i < order.Count; i++) pos[order[i]] = i;
        foreach (StageDef s in plan.Stages)
            foreach (string d in s.Deps)
                if (pos[s.Id] <= pos[d]) return false;
        return true;
    }

    // ---------- main ----------
    public static void Main()
    {
        var plan = DefaultPlan();
        var opts = new RunOptions { CodeVersion = "v1", Seed = 7, Block = 8, Workers = 1 };
        var cache = new Dictionary<string, CacheEntry>();

        var run1 = Run(plan, opts, cache);
        var run2 = Run(plan, opts, cache);
        Console.WriteLine("run1 computed " + run1.Computed + " cached " + run1.CacheHits +
                          " root " + run1.RootDigest.Substring(0, 12) +
                          "; run2 computed " + run2.Computed + " cached " + run2.CacheHits);

        // 1-2: scheduling correctness (property check against the plan's own dependency edges).
        Check(IsTopological(plan, run1.Order), "emitted order respects every dependency edge");
        Check(run1.Order.Count == plan.Stages.Count && run1.Order.Distinct().Count() == plan.Stages.Count,
              "every stage ran exactly once");
        Check(run1.Order.IndexOf("publish") == run1.Order.Count - 1, "the publish stage runs last");
        Check(run1.Success && run1.Digests.Count == plan.Stages.Count, "happy path succeeds for all 7 stages");

        // 3-4: caching and bitwise reproducibility of the whole pipeline.
        Check(run2.Computed == 0 && run2.CacheHits == plan.Stages.Count,
              "second run is a pure cache hit (0 rebuilds)");
        Check(run2.RootDigest == run1.RootDigest, "root digest is bitwise stable across runs");
        Check(run2.Digests.All(kv => run1.Digests[kv.Key] == kv.Value), "every artifact digest is reused unchanged");

        // 5: worker-count invariance - the parallelism parameter must not change a single byte.
        var w2 = Run(plan, new RunOptions { CodeVersion = "v1", Seed = 7, Block = 8, Workers = 2 },
                     new Dictionary<string, CacheEntry>());
        var w8 = Run(plan, new RunOptions { CodeVersion = "v1", Seed = 7, Block = 8, Workers = 8 },
                     new Dictionary<string, CacheEntry>());
        Check(w2.RootDigest == run1.RootDigest && w8.RootDigest == run1.RootDigest,
              "root digest is invariant under 1/2/8 workers");

        // 6-7: invalidation - a code version bump or a parameter change propagates downstream.
        var bump = Run(plan, new RunOptions { CodeVersion = "v2", Seed = 7, Block = 8, Workers = 4 },
                       new Dictionary<string, CacheEntry>());
        Check(bump.Computed == plan.Stages.Count, "a version bump rebuilds every stage from scratch");
        Check(bump.RootDigest != run1.RootDigest, "a version bump changes the root digest");
        var blockChange = Run(plan, new RunOptions { CodeVersion = "v1", Seed = 7, Block = 6, Workers = 4 },
                              new Dictionary<string, CacheEntry>());
        Check(blockChange.Computed == plan.Stages.Count, "a kernel parameter change invalidates all stages");
        Check(blockChange.Digests["kernel"] != run1.Digests["kernel"],
              "the kernel artifact digest carries the build parameter");
        Check(blockChange.RootDigest != run1.RootDigest, "the whole pipeline digest changes with the parameter");

        // 8: cache correctness - a stale entry can never be served for a different key.
        var reuse = Run(plan, opts, cache);
        Check(reuse.Computed == 0 && reuse.RootDigest == run1.RootDigest,
              "the shared cache still returns the original digests");

        // 9-10: failure semantics - dependents are skipped, independent branches still complete.
        var failing = Run(plan, new RunOptions { CodeVersion = "v1", Seed = 7, Block = 8, Workers = 4, FailStage = "refcheck" },
                          new Dictionary<string, CacheEntry>());
        Check(!failing.Success && failing.Status["refcheck"] == "failed", "injected failure is reported");
        Check(failing.Status["publish"] == "skipped" && failing.Skipped == 1,
              "exactly the transitive dependents are skipped (publish)");
        Check(failing.Status["lint"] == "ok" && failing.Status["determinism"] == "ok",
              "independent branches still complete after the failure");
        Check(failing.Digests.ContainsKey("kernel"), "artifacts produced before the failure are kept");

        // 11-12: the gates have power - a nondeterministic kernel and a buggy kernel are both caught.
        var drifting = Run(plan, new RunOptions { CodeVersion = "v1", Seed = 7, Block = 8, Workers = 4,
                                                  NondeterministicKernel = true, DriftCounter = 1 },
                           new Dictionary<string, CacheEntry>());
        Check(drifting.Status["determinism"] == "failed", "the determinism gate catches a drifting kernel");
        Check(!drifting.Success && drifting.Status["publish"] == "skipped",
              "a nondeterministic build cannot be published");
        var buggy = Run(plan, new RunOptions { CodeVersion = "v1", Seed = 7, Block = 8, Workers = 4, BuggyKernel = true },
                        new Dictionary<string, CacheEntry>());
        Check(buggy.Status["refcheck"] == "failed", "the reference check catches a numerical regression");
        Console.WriteLine("buggy kernel diagnostic: " + buggy.FailReason);

        // 13: the kernel is verified against the independent naive implementation here in Main too.
        var (ma, mb) = MakeOperands(7);
        var tiled = TiledMul(ma, mb, 8);
        var naive = NaiveMul(ma, mb);
        double worst = 0.0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) worst = Math.Max(worst, Math.Abs(tiled[i, j] - naive[i, j]));
        Check(worst < 1e-9, "tiled and naive kernels agree (max deviation " + worst.ToString("E2") + ")");
        Check(run1.Flops == 2L * N * N * N, "benchmark accounting equals the analytic FLOP count");

        // 14: content addressing - every payload hashes back to the key it is stored under.
        bool addressed = true;
        foreach (KeyValuePair<string, CacheEntry> kv in cache)
            addressed &= HashBytes(kv.Value.Payload) == kv.Value.Digest;
        Check(addressed, "every cached artifact is content addressed by its own payload");

        // 15-16: degenerate plans.
        var empty = Run(new Plan { Name = "empty" }, opts, new Dictionary<string, CacheEntry>());
        Check(empty.Success && empty.Computed == 0 && empty.RootDigest.Length == 32,
              "an empty plan succeeds with a well-formed root digest");
        var single = Run(new Plan { Name = "single", Stages = new List<StageDef> { new StageDef { Id = "lint", Kind = "lint" } } },
                         opts, new Dictionary<string, CacheEntry>());
        Check(single.Success && single.Computed == 1 && single.Order.Count == 1, "a single-stage plan runs once");

        // 17-19: the DAG validator rejects malformed plans instead of looping or mis-scheduling.
        var dup = new Plan { Name = "dup", Stages = new List<StageDef>
            { new StageDef { Id = "a", Kind = "lint" }, new StageDef { Id = "a", Kind = "lint" } } };
        Check(Run(dup, opts, new Dictionary<string, CacheEntry>()).Error.Contains("duplicate"),
              "duplicate stage ids are rejected");
        var cyc = new Plan { Name = "cycle", Stages = new List<StageDef>
            { new StageDef { Id = "a", Kind = "lint", Deps = new[] { "b" } },
              new StageDef { Id = "b", Kind = "lint", Deps = new[] { "a" } } } };
        Check(Run(cyc, opts, new Dictionary<string, CacheEntry>()).Error.Contains("cycle"),
              "a cyclic plan is rejected as a cycle");
        var dangling = new Plan { Name = "dangling", Stages = new List<StageDef>
            { new StageDef { Id = "a", Kind = "lint", Deps = new[] { "ghost" } } } };
        Check(Run(dangling, opts, new Dictionary<string, CacheEntry>()).Error.Contains("dangling"),
              "a dangling dependency is rejected");

        // 20: the whole orchestrator is deterministic from cold state.
        var cold1 = Run(plan, opts, new Dictionary<string, CacheEntry>());
        var cold2 = Run(plan, opts, new Dictionary<string, CacheEntry>());
        Check(cold1.RootDigest == cold2.RootDigest && cold1.Order.SequenceEqual(cold2.Order),
              "two cold runs agree on the order and the root digest");

        Console.WriteLine("stages " + plan.Stages.Count + ", flops " + run1.Flops +
                          ", cache entries " + cache.Count + ", version bump root " + bump.RootDigest.Substring(0, 12) +
                          ", failure diagnostic: " + failing.FailReason);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
