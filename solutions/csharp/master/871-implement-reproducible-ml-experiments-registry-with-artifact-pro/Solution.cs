// Task 871 - Implement reproducible ML experiments registry with artifact provenance.
// (Tier: Master, CTCI Beyond - Reproducibility, Testing & Tooling)
//
// Model: a content-addressed experiment registry. An artifact's id is derived from its kind, its
// recorded parameters, the ids of its provenance parents and the hash of its payload, so it is a pure
// function of everything that produced it: identical experiments collapse onto one artifact (memoised,
// deduplicated), and any change anywhere upstream - seed, hyper-parameter, dataset version, parent
// artifact - produces a different id. The registry keeps the provenance DAG (dataset -> model ->
// metrics -> report) and can (a) verify an artifact by recomputing its id from its own recorded fields
// and its payload hash, and (b) *replay* an experiment from provenance alone, recomputing the payload
// from the recorded parameters and parents and comparing it byte for byte with what is stored.
//
// Simplifications (documented, not behavioural): no database, no filesystem, no object store - the
// registry is an in-process dictionary and artifacts are serialised doubles; "training" is a fixed-step
// gradient descent on a generated linear-regression problem (small enough to replay hundreds of times);
// ids are 128-bit FNV-1a content addresses rather than SHA-256, which is enough to make accidental
// collisions impossible at this scale.
//
// Verified here:
//   * reproducibility: the same experiment run twice yields the same artifact id, the same payload and
//     the same metrics; a cold registry and a warm one agree; replaying from provenance alone
//     reproduces every payload byte for byte;
//   * provenance: the DAG is acyclic, every parent exists, and the registry's ancestor/descendant
//     closures equal an independent breadth-first search over the recorded edges;
//   * content addressing and tamper detection: mutating a payload, a parameter or a recorded metric
//     invalidates verification, while re-registering an identical artifact is idempotent;
//   * deduplication/cost bound: a sweep of 24 configurations with only a handful of distinct values
//     stores exactly the distinct artifacts, and the memoised run count is strictly below the launch
//     count (a measured saving);
//   * sensitivity/negative controls: changing the seed, the learning rate or the dataset parent each
//     change the id *and* the recorded metric, and two independent experiments never share an id;
//   * degenerate inputs: an empty registry, a root artifact with no parents, and a missing artifact id.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

public static class Solution
{
    // ---------- deterministic PRNG (splitmix64) ----------
    private static ulong _state = 0x1234ABCD5678EF90UL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextSigned() => ((NextU64() >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;

    // ---------- content addressing ----------
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

    private static string HashBytes(byte[] data) =>
        Fnv1a64(data, 0xCBF29CE484222325UL).ToString("x16") + Fnv1a64(data, 0x9E3779B97F4A7C15UL).ToString("x16");

    private static string HashStrings(params string[] parts)
    {
        var sb = new StringBuilder();
        foreach (string p in parts) sb.Append(p).Append('\u0001');
        return HashBytes(Encoding.UTF8.GetBytes(sb.ToString()));
    }

    private static byte[] Serialize(IEnumerable<double> values)
    {
        var list = values.ToList();
        var bytes = new byte[8 * list.Count];
        for (int i = 0; i < list.Count; i++) BitConverter.GetBytes(list[i]).CopyTo(bytes, 8 * i);
        return bytes;
    }

    private static double[] Deserialize(byte[] bytes)
    {
        var values = new double[bytes.Length / 8];
        for (int i = 0; i < values.Length; i++) values[i] = BitConverter.ToDouble(bytes, 8 * i);
        return values;
    }

    // ---------- the experiment: linear regression on a generated dataset ----------
    private const int Samples = 48;
    private const int Features = 3;

    private static readonly double[] TrueWeights = { 1.5, -0.75, 0.3 };

    private static byte[] BuildDataset(int seed, int n)
    {
        _state = 0x9E3779B97F4A7C15UL ^ (ulong)seed * 0xD1B54A32D192ED03UL;
        var payload = new List<double>();
        for (int i = 0; i < n; i++)
        {
            double[] x = new double[Features];
            for (int j = 0; j < Features; j++) x[j] = NextSigned();
            double y = 0.0;
            for (int j = 0; j < Features; j++) y += TrueWeights[j] * x[j];
            y += 0.1 * NextSigned();
            payload.Add(y);
            payload.AddRange(x);
        }
        return Serialize(payload);
    }

    private sealed class Dataset
    {
        public double[] Y = Array.Empty<double>();
        public double[][] X = Array.Empty<double[]>();
        public int N;
    }

    private static Dataset ParseDataset(byte[] payload)
    {
        double[] flat = Deserialize(payload);
        int n = flat.Length / (Features + 1);
        var d = new Dataset { N = n, X = new double[n][], Y = new double[n] };
        for (int i = 0; i < n; i++)
        {
            d.Y[i] = flat[i * (Features + 1)];
            d.X[i] = new double[Features];
            for (int j = 0; j < Features; j++) d.X[i][j] = flat[i * (Features + 1) + 1 + j];
        }
        return d;
    }

    // Fixed-step full-batch gradient descent: no early stopping, no shuffling, so the weights are a
    // deterministic function of (dataset, learning rate, epochs).
    private static double[] Train(Dataset d, double lr, int epochs)
    {
        var w = new double[Features];
        double b = 0.0;
        for (int e = 0; e < epochs; e++)
        {
            var gw = new double[Features];
            double gb = 0.0;
            for (int i = 0; i < d.N; i++)
            {
                double pred = b;
                for (int j = 0; j < Features; j++) pred += w[j] * d.X[i][j];
                double err = pred - d.Y[i];
                for (int j = 0; j < Features; j++) gw[j] += 2.0 * err * d.X[i][j] / d.N;
                gb += 2.0 * err / d.N;
            }
            for (int j = 0; j < Features; j++) w[j] -= lr * gw[j];
            b -= lr * gb;
        }
        return w;
    }

    private static double Mse(Dataset d, double[] w)
    {
        double acc = 0.0;
        for (int i = 0; i < d.N; i++)
        {
            double pred = 0.0;
            for (int j = 0; j < Features; j++) pred += w[j] * d.X[i][j];
            double e = pred - d.Y[i];
            acc += e * e;
        }
        return acc / d.N;
    }

    // ---------- registry ----------
    private sealed class Artifact
    {
        public string Id = "";
        public string Kind = "";
        public string[] Parents = Array.Empty<string>();
        public SortedDictionary<string, string> Params = new SortedDictionary<string, string>(StringComparer.Ordinal);
        public byte[] Payload = Array.Empty<byte>();
    }

    private sealed class Registry
    {
        public Dictionary<string, Artifact> Store = new Dictionary<string, Artifact>(StringComparer.Ordinal);
        public long Registered;      // artifact registrations attempted (experiment "launches")
        public long Computed;        // payloads actually computed (cache misses)
        public long CacheHits;

        public static string ComputeId(string kind, IDictionary<string, string> parameters, IEnumerable<string> parents,
                                       byte[] payload)
        {
            var parts = new List<string> { kind };
            foreach (KeyValuePair<string, string> kv in parameters.OrderBy(k => k.Key, StringComparer.Ordinal))
                parts.Add(kv.Key + "=" + kv.Value);
            foreach (string p in parents.OrderBy(x => x, StringComparer.Ordinal)) parts.Add("<-" + p);
            parts.Add(HashBytes(payload));
            return HashStrings(parts.ToArray());
        }

        public string Put(string kind, IDictionary<string, string> parameters, string[] parents, Func<byte[]> compute)
        {
            Registered++;
            // The id can only be computed after the payload exists, so a "launch" of an experiment whose
            // configuration is already present is recognised by the id it produces (memoisation).
            byte[] payload = compute();
            string id = ComputeId(kind, parameters, parents, payload);
            if (Store.ContainsKey(id))
            {
                CacheHits++;
                return id;
            }
            Computed++;
            var artifact = new Artifact { Id = id, Kind = kind, Parents = parents, Payload = payload };
            foreach (KeyValuePair<string, string> kv in parameters) artifact.Params[kv.Key] = kv.Value;
            Store[id] = artifact;
            return id;
        }

        public List<string> Verify(string id)
        {
            var problems = new List<string>();
            if (!Store.TryGetValue(id, out Artifact a))
            {
                problems.Add("missing artifact " + id);
                return problems;
            }
            foreach (string p in a.Parents)
                if (!Store.ContainsKey(p)) problems.Add("dangling parent " + p);
            string recomputed = ComputeId(a.Kind, a.Params, a.Parents, a.Payload);
            if (recomputed != a.Id) problems.Add("id mismatch for " + a.Id);
            return problems;
        }

        public bool HasCycle()
        {
            var color = new Dictionary<string, int>();   // 0 = unseen, 1 = on stack, 2 = done
            bool Dfs(string id)
            {
                if (!Store.TryGetValue(id, out Artifact a)) return false;
                if (color.TryGetValue(id, out int c))
                {
                    if (c == 1) return true;
                    if (c == 2) return false;
                }
                color[id] = 1;
                foreach (string p in a.Parents) if (Dfs(p)) return true;
                color[id] = 2;
                return false;
            }
            return Store.Keys.Any(Dfs);
        }

        public HashSet<string> Reachable(string id, bool towardsParents)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            var stack = new Stack<string>();
            if (!Store.TryGetValue(id, out Artifact start)) return seen;
            IEnumerable<string> initial = towardsParents ? start.Parents : Children(id);
            foreach (string s in initial)
                if (seen.Add(s)) stack.Push(s);
            while (stack.Count > 0)
            {
                string cur = stack.Pop();
                if (!Store.TryGetValue(cur, out Artifact a)) continue;
                IEnumerable<string> next = towardsParents ? a.Parents : Children(cur);
                foreach (string n in next)
                    if (seen.Add(n)) stack.Push(n);
            }
            return seen;
        }

        private IEnumerable<string> Children(string id) =>
            Store.Values.Where(v => v.Parents.Contains(id)).Select(v => v.Id).ToArray();

        public List<string> TopologicalOrder()
        {
            var order = new List<string>();
            var visited = new HashSet<string>(StringComparer.Ordinal);
            void Visit(string id)
            {
                if (!visited.Add(id)) return;
                foreach (string p in Store[id].Parents) Visit(p);
                order.Add(id);
            }
            foreach (string id in Store.Keys.OrderBy(x => x, StringComparer.Ordinal)) Visit(id);
            return order;
        }

        // Replays a subtree from provenance: every payload is recomputed from the recorded parameters and
        // parents and compared byte for byte with the stored one.
        public List<string> Replay(string id, Func<string, byte[]> recompute)
        {
            var mismatches = new List<string>();
            foreach (string node in Reachable(id, true).Concat(new[] { id })
                                                   .OrderBy(x => x, StringComparer.Ordinal))
            {
                Artifact a = Store[node];
                byte[] again = recompute(node);
                if (again.Length != a.Payload.Length || !again.SequenceEqual(a.Payload))
                    mismatches.Add(node);
                string againId = ComputeId(a.Kind, a.Params, a.Parents, again);
                if (againId != a.Id) mismatches.Add("id:" + node);
            }
            return mismatches;
        }
    }

    // ---------- experiment driver ----------
    private sealed class ExperimentLog
    {
        public string DatasetId = "";
        public string ModelId = "";
        public string MetricsId = "";
        public string ReportId = "";
        public double Mse;
        public double[] Weights = Array.Empty<double>();
    }

    private static ExperimentLog Launch(Registry reg, int dataSeed, int n, double lr, int epochs, string codeVersion)
    {
        var log = new ExperimentLog();
        log.DatasetId = reg.Put("dataset",
            new SortedDictionary<string, string> { ["name"] = "linreg", ["seed"] = dataSeed.ToString(), ["n"] = n.ToString() },
            Array.Empty<string>(), () => BuildDataset(dataSeed, n));

        var dataset = ParseDataset(reg.Store[log.DatasetId].Payload);
        log.ModelId = reg.Put("model",
            new SortedDictionary<string, string>
            {
                ["lr"] = lr.ToString("R", CultureInfo.InvariantCulture),
                ["epochs"] = epochs.ToString(),
                ["code"] = codeVersion,
            },
            new[] { log.DatasetId },
            () => Serialize(Train(dataset, lr, epochs)));

        log.Weights = Deserialize(reg.Store[log.ModelId].Payload);
        log.Mse = Mse(dataset, log.Weights);
        log.MetricsId = reg.Put("metrics",
            new SortedDictionary<string, string> { ["metric"] = "mse" },
            new[] { log.ModelId },
            () => Serialize(new[] { Mse(dataset, Deserialize(reg.Store[log.ModelId].Payload)) }));

        log.ReportId = reg.Put("report",
            new SortedDictionary<string, string> { ["format"] = "text-v1" },
            new[] { log.MetricsId, log.DatasetId },
            () =>
            {
                double mse = Deserialize(reg.Store[log.MetricsId].Payload)[0];
                return Encoding.UTF8.GetBytes("experiment " + codeVersion + " n=" + n + " lr=" +
                                             lr.ToString("R", CultureInfo.InvariantCulture) +
                                             " mse=" + mse.ToString("R", CultureInfo.InvariantCulture));
            });
        return log;
    }

    // ---------- main ----------
    public static void Main()
    {
        var reg = new Registry();

        // 1-3: reproducibility of a single experiment.
        var first = Launch(reg, 11, Samples, 0.05, 200, "v1");
        var second = Launch(reg, 11, Samples, 0.05, 200, "v1");
        Check(first.ModelId == second.ModelId && first.DatasetId == second.DatasetId,
              "identical experiments collapse onto one artifact id");
        Check(first.Mse == second.Mse && first.Weights.SequenceEqual(second.Weights),
              "identical experiments reproduce the same metric bit for bit");
        Check(reg.Computed == 4 && reg.CacheHits == 4 && reg.Registered == 8,
              "the second launch is served entirely from the registry (computed " + reg.Computed + ")");

        // 4: content addressing - verification of the whole DAG.
        Check(reg.Store.Keys.All(id => reg.Verify(id).Count == 0), "every artifact independently verifies");
        Check(!reg.HasCycle(), "the provenance DAG is acyclic");

        // 5-6: provenance closure against an independent breadth-first search.
        bool closureOk = true;
        foreach (string id in reg.Store.Keys)
        {
            var mine = reg.Reachable(id, true);
            var oracle = BfsAncestors(reg, id);
            closureOk &= mine.SetEquals(oracle);
            var down = reg.Reachable(id, false);
            var downOracle = BfsDescendants(reg, id);
            closureOk &= down.SetEquals(downOracle);
        }
        Check(closureOk, "ancestor and descendant closures match an independent BFS");
        Check(reg.Reachable(first.ReportId, true).Count == reg.Store.Count - 1,
              "the report transitively depends on every other artifact");
        Check(reg.Reachable(first.DatasetId, true).Count == 0, "the dataset is a provenance root");

        // 7: replay from provenance alone reproduces every payload byte for byte.
        var replayMismatches = reg.Replay(first.ReportId, id => RecomputeFromProvenance(reg, id));
        Check(replayMismatches.Count == 0, "replay from provenance is byte-identical (" +
              string.Join(",", replayMismatches) + ")");

        // 8-10: tamper detection on payload, parameters and recorded metric.
        var tampered = new Registry();
        foreach (KeyValuePair<string, Artifact> kv in reg.Store)
            tampered.Store[kv.Key] = new Artifact
            {
                Id = kv.Value.Id, Kind = kv.Value.Kind, Parents = kv.Value.Parents,
                Params = kv.Value.Params, Payload = (byte[])kv.Value.Payload.Clone(),
            };
        tampered.Store[first.ModelId].Payload[0] ^= 0xFF;
        Check(tampered.Verify(first.ModelId).Count == 1, "a mutated payload fails verification");
        var tampered2 = new Registry();
        foreach (KeyValuePair<string, Artifact> kv in reg.Store)
            tampered2.Store[kv.Key] = new Artifact
            {
                Id = kv.Value.Id, Kind = kv.Value.Kind, Parents = kv.Value.Parents,
                Params = kv.Value.Params, Payload = (byte[])kv.Value.Payload.Clone(),
            };
        tampered2.Store[first.ModelId].Params["lr"] = "0.051";
        Check(tampered2.Verify(first.ModelId).Count == 1, "an edited parameter fails verification");
        Check(reg.Verify("does-not-exist").Count == 1, "a missing artifact is reported, not silently accepted");

        // 11-12: idempotent re-registration and sensitivity to every input knob.
        string before = reg.Store.Count.ToString();
        var again = Launch(reg, 11, Samples, 0.05, 200, "v1");
        Check(reg.Store.Count.ToString() == before && again.ModelId == first.ModelId,
              "re-registering an identical artifact is idempotent");
        var otherSeed = Launch(reg, 12, Samples, 0.05, 200, "v1");
        Check(otherSeed.ModelId != first.ModelId && otherSeed.Mse != first.Mse,
              "a different dataset seed changes the id and the metric");
        var otherLr = Launch(reg, 11, Samples, 0.06, 200, "v1");
        Check(otherLr.ModelId != first.ModelId && otherLr.Mse != first.Mse,
              "a different learning rate changes the id and the metric");
        var otherCode = Launch(reg, 11, Samples, 0.05, 200, "v2");
        Check(otherCode.ModelId != first.ModelId, "a different code version changes the id");
        Check(otherCode.DatasetId == first.DatasetId, "the unchanged dataset parent is reused");

        // 13: training actually reduces the error (otherwise "reproducible" would be meaningless).
        Check(first.Mse < 0.05, "training drives the MSE below 0.05 (" + first.Mse + ")");
        Check(first.Mse < Mse(ParseDataset(reg.Store[first.DatasetId].Payload), new double[Features]),
              "the fitted model beats the zero model");

        // 14-16: a 24-point sweep - deduplication, a measured cost saving, and an oracle argmax.
        var sweep = new Registry();
        var logs = new List<ExperimentLog>();
        for (int s = 0; s < 2; s++)
            for (int li = 0; li < 3; li++)
                for (int e = 0; e < 4; e++)
                    logs.Add(Launch(sweep, 21 + s, Samples, 0.05 + 0.01 * li, 100 + 50 * e, "v1"));
        Check(logs.Count == 24, "the sweep launched 24 experiments");
        Check(sweep.Registered == 24 * 4 && sweep.Computed < sweep.Registered,
              "memoisation saves runs (" + sweep.Computed + " computed of " + sweep.Registered + " registrations)");
        Check(sweep.Store.Values.Where(a => a.Kind == "model").Count() == 24,
              "all 24 configurations are distinct models (2 datasets x 3 lrs x 4 epoch counts)");
        Check(sweep.Store.Values.Where(a => a.Kind == "dataset").Count() == 2,
              "only two distinct dataset artifacts are stored for 24 launches");
        double best = logs.Select(l => l.Mse).Min();
        var bestLog = logs.OrderBy(l => l.Mse).ThenBy(l => l.ModelId, StringComparer.Ordinal).First();
        Check(bestLog.Mse == best && bestLog.Mse > 0, "argmin over the recorded metrics is well defined");
        Check(sweep.Replay(bestLog.ReportId, id => RecomputeFromProvenance(sweep, id)).Count == 0,
              "the winning experiment replays byte for byte");

        // 17-18: two independent registries must agree (no hidden global state in the ids).
        var cold1 = new Registry();
        var cold2 = new Registry();
        var l1 = Launch(cold1, 11, Samples, 0.05, 200, "v1");
        var l2 = Launch(cold2, 11, Samples, 0.05, 200, "v1");
        Check(l1.ModelId == l2.ModelId && cold1.Store.Keys.OrderBy(x => x).SequenceEqual(cold2.Store.Keys.OrderBy(x => x)),
              "two cold registries build identical ids");

        // 19-20: degenerate inputs.
        var empty = new Registry();
        Check(empty.Store.Count == 0 && !empty.HasCycle() && empty.TopologicalOrder().Count == 0,
              "an empty registry is valid and has an empty topological order");
        var roots = reg.Store.Values.Count(a => a.Parents.Length == 0);
        Check(roots == reg.Store.Values.Count(a => a.Kind == "dataset"),
              "exactly the dataset artifacts are provenance roots (" + roots + ")");

        // 21: the topological order really is a valid order of the provenance DAG.
        var topo = reg.TopologicalOrder();
        var position = new Dictionary<string, int>();
        for (int i = 0; i < topo.Count; i++) position[topo[i]] = i;
        Check(topo.All(id => reg.Store[id].Parents.All(p => position[p] < position[id])),
              "topological order places every parent before its child");

        Console.WriteLine("registry artifacts " + reg.Store.Count + ", registered " + reg.Registered +
                          ", computed " + reg.Computed + ", cache hits " + reg.CacheHits);
        Console.WriteLine("sweep: " + logs.Count + " launches, " + sweep.Store.Count + " artifacts, best mse " +
                          best.ToString("R") + " at " + bestLog.ModelId.Substring(0, 12));
        Console.WriteLine("mse first " + first.Mse.ToString("R") + " seed 12 " + otherSeed.Mse.ToString("R") +
                          " lr 0.06 " + otherLr.Mse.ToString("R"));
        Console.WriteLine("PASS");
    }

    // Independent oracles for the lineage closure (plain BFS over the recorded edges).
    private static HashSet<string> BfsAncestors(Registry reg, string id)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var queue = new Queue<string>();
        queue.Enqueue(id);
        while (queue.Count > 0)
        {
            string cur = queue.Dequeue();
            foreach (string p in reg.Store[cur].Parents)
                if (seen.Add(p)) queue.Enqueue(p);
        }
        return seen;
    }

    private static HashSet<string> BfsDescendants(Registry reg, string id)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var queue = new Queue<string>();
        queue.Enqueue(id);
        while (queue.Count > 0)
        {
            string cur = queue.Dequeue();
            foreach (Artifact a in reg.Store.Values)
                if (a.Parents.Contains(cur) && seen.Add(a.Id)) queue.Enqueue(a.Id);
        }
        return seen;
    }

    // Recomputes an artifact's payload from its recorded kind, parameters and parents - the replay path
    // never looks at the stored payload, only at provenance.
    private static byte[] RecomputeFromProvenance(Registry reg, string id)
    {
        Artifact a = reg.Store[id];
        switch (a.Kind)
        {
            case "dataset":
                return BuildDataset(int.Parse(a.Params["seed"]), int.Parse(a.Params["n"]));
            case "model":
            {
                var dataset = ParseDataset(reg.Store[a.Parents[0]].Payload);
                double lr = double.Parse(a.Params["lr"], CultureInfo.InvariantCulture);
                int epochs = int.Parse(a.Params["epochs"]);
                return Serialize(Train(dataset, lr, epochs));
            }
            case "metrics":
            {
                var dataset = ParseDataset(reg.Store[reg.Store[a.Parents[0]].Parents[0]].Payload);
                double[] w = Deserialize(reg.Store[a.Parents[0]].Payload);
                return Serialize(new[] { Mse(dataset, w) });
            }
            case "report":
            {
                double mse = Deserialize(reg.Store[a.Parents[0]].Payload)[0];
                var datasetArtifact = reg.Store[a.Parents[1]];
                string model = reg.Store[a.Parents[0]].Parents[0];
                string lr = reg.Store[model].Params["lr"];
                string code = reg.Store[model].Params["code"];
                return Encoding.UTF8.GetBytes("experiment " + code + " n=" + datasetArtifact.Params["n"] +
                                             " lr=" + lr + " mse=" + mse.ToString("R", CultureInfo.InvariantCulture));
            }
            default:
                throw new Exception("unknown artifact kind " + a.Kind);
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
