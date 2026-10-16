// Task 988 - Implement cross-repo synchronization and dependency management tool for exercises.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Repositories publish semver candidates and declare (repo, range) dependencies (`^`, `~`, `=`, `*`). The tool
// picks for each repo the smallest candidate satisfying every dependent's range, refuses unknown references,
// unsatisfiable ranges and cycles, then emits an explicit schedule of Kahn layers, each layer fetched from a
// simulated network only after every dependency is verified. A payload whose hash differs from the pinned
// expectation is drift: that repo and everything downstream is blocked, never synced blind. Verified: the
// range table, selection against a brute-force oracle, exact refusal strings, layer safety with a
// deterministic tie-break, one fetch per repo with rounds equal to the hand-checked longest chain, drift plus
// downstream blocking, and the single-repo / empty-set boundaries.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }
    private static string Error(Action action)
    {
        try { action(); return ""; }
        catch (Exception ex) { return ex.Message; }
    }

    private static int[] Parts(string version)
    {
        string[] fields = version.Split('.');
        if (fields.Length != 3) throw new Exception("bad-version:" + version);
        var numbers = new int[3];
        for (int i = 0; i < 3; i++)
            if (!int.TryParse(fields[i], NumberStyles.Integer, CultureInfo.InvariantCulture, out numbers[i]))
                throw new Exception("bad-version-part:" + fields[i]);
        return numbers;
    }

    private static int Compare(string left, string right)
    {
        int[] a = Parts(left), b = Parts(right);
        for (int i = 0; i < 3; i++) if (a[i] != b[i]) return a[i].CompareTo(b[i]);
        return 0;
    }

    private static bool Satisfies(string version, string range)
    {
        if (range == "*") return true;
        if (range.Length < 2) throw new Exception("bad-range:" + range);
        int[] candidate = Parts(version), bound = Parts(range.Substring(1));
        int compare = Compare(version, range.Substring(1));
        if (range[0] == '=') return compare == 0;
        if (range[0] == '^') return compare >= 0 && candidate[0] == bound[0];
        if (range[0] == '~') return compare >= 0 && candidate[0] == bound[0] && candidate[1] == bound[1];
        throw new Exception("bad-range:" + range);
    }

    private sealed class Repo
    {
        public readonly string Name;
        public readonly List<string> Candidates = new List<string>();
        public readonly List<(string Repo, string Range)> Deps = new List<(string, string)>();
        public Repo(string name) { Name = name; }
        public Repo With(params string[] versions) { Candidates.AddRange(versions); return this; }
        public Repo Needs(string repo, string range) { Deps.Add((repo, range)); return this; }
    }

    /** Simulated network: payload identity is a pure function of (repo, version); drift flips one bit. */
    private sealed class Network
    {
        private readonly string _drifted;
        public int Fetches;
        public Network(string drifted = null) { _drifted = drifted; }
        public static ulong Pinned(string repo, string version)
        {
            unchecked
            {
                ulong h = 14695981039346656037UL;
                foreach (byte b in Encoding.UTF8.GetBytes(repo + "@" + version)) { h ^= b; h *= 1099511628211UL; }
                return (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9UL ^ (h >> 27);
            }
        }
        public ulong Fetch(string repo, string version)
        {
            Fetches++;
            ulong clean = Pinned(repo, version);
            return repo == _drifted ? clean ^ 1UL : clean;
        }
    }

    private sealed class Plan
    {
        public readonly List<List<string>> Layers = new List<List<string>>();
        public readonly Dictionary<string, string> Versions = new Dictionary<string, string>(StringComparer.Ordinal);
        public readonly List<string> Errors = new List<string>(), Synced = new List<string>(), Blocked = new List<string>();
        public string ErrorText() => string.Join(" / ", Errors);
    }

    private static List<string> Sorted(IEnumerable<string> values)
    {
        var list = new List<string>(values); list.Sort(StringComparer.Ordinal); return list;
    }
    private static Dictionary<string, Repo> Index(List<Repo> repos)
    {
        var index = new Dictionary<string, Repo>(StringComparer.Ordinal);
        foreach (Repo repo in repos) index[repo.Name] = repo;
        return index;
    }

    private static bool SelectVersions(Dictionary<string, Repo> index, Plan plan)
    {
        var ranges = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (string name in index.Keys) ranges[name] = new List<string>();
        foreach (string dependent in Sorted(index.Keys))
            foreach ((string repo, string range) in index[dependent].Deps)
                if (!index.ContainsKey(repo)) plan.Errors.Add("unknown-repo:" + dependent + "->" + repo);
                else ranges[repo].Add(range);
        bool ok = true;
        foreach (string name in Sorted(index.Keys))
        {
            var candidates = new List<string>(index[name].Candidates);
            candidates.Sort(Compare);
            string best = null;
            foreach (string candidate in candidates)
            {
                bool all = true;
                foreach (string range in ranges[name]) if (!Satisfies(candidate, range)) { all = false; break; }
                if (all) { best = candidate; break; }
            }
            if (best == null)
            {
                plan.Errors.Add("unsatisfiable:" + name + ":" + (ranges[name].Count > 0 ? ranges[name][0] : "*"));
                ok = false;
            }
            else plan.Versions[name] = best;
        }
        return ok;
    }

    /** Kahn layers; a stalled round means the still-pending repositories form or feed a cycle. */
    private static void BuildLayers(Dictionary<string, Repo> index, Plan plan)
    {
        var remaining = new HashSet<string>(index.Keys, StringComparer.Ordinal);
        while (remaining.Count > 0)
        {
            var layer = new List<string>();
            foreach (string name in Sorted(remaining))
            {
                bool ready = true;
                foreach ((string repo, string _) in index[name].Deps) if (remaining.Contains(repo)) ready = false;
                if (ready) layer.Add(name);
            }
            if (layer.Count == 0) { plan.Errors.Add("cycle:" + string.Join(",", Sorted(remaining))); return; }
            plan.Layers.Add(layer);
            foreach (string name in layer) remaining.Remove(name);
        }
    }

    private static Plan BuildPlan(List<Repo> repos, Network network)
    {
        var plan = new Plan();
        Dictionary<string, Repo> index = Index(repos);
        if (!SelectVersions(index, plan) || plan.Errors.Count != 0) return plan;
        BuildLayers(index, plan);
        var synced = new HashSet<string>(StringComparer.Ordinal);
        var blocked = new HashSet<string>(StringComparer.Ordinal);
        foreach (List<string> layer in plan.Layers)
            foreach (string name in layer)
            {
                bool ready = true;
                foreach ((string repo, string _) in index[name].Deps)
                    if (index.ContainsKey(repo) && !synced.Contains(repo)) ready = false;
                if (!ready) { blocked.Add(name); plan.Errors.Add("blocked:" + name); continue; }
                if (Network.Pinned(name, plan.Versions[name]) != network.Fetch(name, plan.Versions[name]))
                {
                    blocked.Add(name);
                    plan.Errors.Add("drift:" + name + "@" + plan.Versions[name]);
                }
                else synced.Add(name);
            }
        plan.Synced.AddRange(Sorted(synced));
        plan.Blocked.AddRange(Sorted(blocked));
        return plan;
    }
    private static List<Repo> Fixture() => new List<Repo>
    {
        new Repo("problem-core").With("1.1.0", "1.0.0"),
        new Repo("tests").With("1.0.0", "1.1.0").Needs("problem-core", "^1.0.0"),
        new Repo("harness").With("1.0.0").Needs("tests", "~1.0.0"), new Repo("docs").With("2.0.0"),
        new Repo("site").With("1.0.0").Needs("docs", "*").Needs("problem-core", "^1.0.0"),
    };

    public static void Main()
    {
        Check(Satisfies("1.2.0", "^1.2.0") && !Satisfies("2.0.0", "^1.2.0") && Satisfies("1.2.9", "~1.2.0")
              && !Satisfies("1.3.0", "~1.2.0") && Satisfies("1.0.0", "=1.0.0") && !Satisfies("1.0.1", "=1.0.0")
              && Satisfies("9.9.9", "*") && Compare("1.2.0", "1.10.0") < 0 && Error(() => Satisfies("1.0.0", ">1.0.0")) == "bad-range:>1.0.0"
              && Error(() => Parts("1.2")) == "bad-version:1.2", "range semantics, numeric ordering and malformed input");
        List<Repo> fixture = Fixture();
        Plan plan = BuildPlan(fixture, new Network());
        Check(plan.Errors.Count == 0 && plan.Versions["problem-core"] == "1.0.0" && plan.Versions["tests"] == "1.0.0"
              && plan.Versions["harness"] == "1.0.0" && plan.Versions["docs"] == "2.0.0" && plan.Versions["site"] == "1.0.0",
              "smallest candidate satisfying every dependent range");
        // Oracle: a naive rescan of every (candidate, range) pair must pick the same minimum.
        string bruteForce = "";
        foreach (string candidate in new[] { "1.0.0", "1.1.0", "1.2.0" })
        {
            int satisfied = 0;
            foreach (string range in new[] { "~1.1.0", "^1.0.0" }) if (Satisfies(candidate, range)) satisfied++;
            if (satisfied == 2 && bruteForce.Length == 0) bruteForce = candidate;
        }
        Plan narrow = BuildPlan(new List<Repo> { new Repo("leaf").With("1.0.0", "1.1.0", "1.2.0"),
            new Repo("top").With("1.0.0").Needs("leaf", "~1.1.0").Needs("leaf", "^1.0.0") }, new Network());
        Check(bruteForce == "1.1.0" && narrow.Versions["leaf"] == bruteForce, "selection matches the brute-force oracle");
        Plan ghost = BuildPlan(new List<Repo> { new Repo("a").With("1.0.0").Needs("ghost", "^1.0.0") }, new Network());
        Plan stuck = BuildPlan(new List<Repo> { new Repo("a").With("1.0.0"),
            new Repo("b").With("2.0.0").Needs("a", "~2.0.0") }, new Network());
        Plan cyclic = BuildPlan(new List<Repo> { new Repo("a").With("1.0.0").Needs("b", "^1.0.0"),
            new Repo("b").With("1.0.0").Needs("a", "^1.0.0") }, new Network());
        Plan solo = BuildPlan(new List<Repo> { new Repo("solo").With("3.1.4") }, new Network());
        Plan none = BuildPlan(new List<Repo>(), new Network());
        Check(ghost.ErrorText() == "unknown-repo:a->ghost" && ghost.Layers.Count == 0 && stuck.ErrorText() == "unsatisfiable:a:~2.0.0"
              && stuck.Synced.Count == 0 && cyclic.ErrorText() == "cycle:a,b" && cyclic.Synced.Count == 0
              && solo.Layers.Count == 1 && solo.Synced.Count == 1 && solo.Versions["solo"] == "3.1.4"
              && none.Layers.Count == 0 && none.Synced.Count == 0 && none.Errors.Count == 0,
              "unknown dependency, unsatisfiable range and cycle refused before any sync, plus the single-repository"
              + " and empty-set boundaries");
        var positions = new Dictionary<string, int>(StringComparer.Ordinal);
        int at = 0;
        foreach (List<string> layer in plan.Layers) foreach (string name in layer) positions[name] = at++;
        bool dependencyOrder = true;
        foreach (Repo repo in fixture) foreach ((string repoName, string _) in repo.Deps)
            if (positions[repoName] >= positions[repo.Name]) dependencyOrder = false;
        var network = new Network();
        Plan live = BuildPlan(fixture, network);
        Check(plan.Layers.Count == 3 && string.Join(",", plan.Layers[0]) == "docs,problem-core"
              && string.Join(",", plan.Layers[1]) == "site,tests" && string.Join(",", plan.Layers[2]) == "harness"
              && dependencyOrder && live.Synced.Count == 5 && live.Blocked.Count == 0 && network.Fetches == 5,
              "layered safe schedule with a deterministic tie-break, one fetch per repository and rounds equal to the"
              + " hand-checked longest chain (core -> tests -> harness)");
        Plan drifted = BuildPlan(fixture, new Network("tests"));
        bool stillSafe = true;
        foreach (string name in drifted.Synced) foreach (Repo repo in fixture) if (repo.Name == name)
            foreach ((string dependency, string _) in repo.Deps) if (!drifted.Synced.Contains(dependency)) stillSafe = false;
        Check(drifted.ErrorText() == "drift:tests@1.0.0 / blocked:harness" && string.Join(",", drifted.Blocked) == "harness,tests"
              && string.Join(",", drifted.Synced) == "docs,problem-core,site" && stillSafe,
              "drift blocks the repo and everything downstream, nothing syncs before its dependencies");
        Console.WriteLine("rounds: " + plan.Layers.Count + ", fetches: " + network.Fetches
                          + ", synced after drift: " + drifted.Synced.Count);
        Console.WriteLine("PASS");
    }
}
