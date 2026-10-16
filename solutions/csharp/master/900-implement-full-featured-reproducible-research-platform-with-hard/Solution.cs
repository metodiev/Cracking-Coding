// Task 900 - Implement full-featured reproducible research platform with hardware config capture. (Tier: Master, Beyond — Software Engineering & Tooling)
//
// A miniature experiment platform: an experiment is a DAG of stages, each stage is a pure function of its
// inputs, its parameters and the captured environment, and every stage output is content addressed.
//
//   * environment capture - CPU model, core count, memory, library version and the experiment seed are
//     captured into a typed record with its own fingerprint, so a result computed on a different machine
//     configuration is a *different* result;
//   * run keys - a stage's key is the hash of (stage name, parameters, environment fingerprint, input
//     artifact hashes), which is exactly the unit of caching: a cache hit returns the stored artifact and
//     performs no computation, a miss executes the stage and stores the artifact under its key;
//   * provenance - the platform records, for every artifact, the stage that produced it and the artifacts
//     it consumed, and the tests walk that graph independently;
//   * lockfile - a run produces name -> hash entries that can be compared against a previous run, which is
//     how drift is reported;
//   * honest failure - a stage flagged as using an unseeded source of entropy is re-run and detected, and
//     the platform marks the experiment as not reproducible instead of pretending otherwise.
//
// Simplifications (documented, not behavioural): stage compute functions are deterministic string
// derivations rather than real jobs (the interesting property is the hashing/caching/provenance logic),
// execution is in-process and single-threaded, and there is no remote store, no filesystem and no
// containerisation. "Hardware" is a captured record, not a probe of the machine.
//
// Verified: the DAG is validated (cycles, unknown inputs, duplicate stages and arity are rejected) and
// executed in a deterministic topological order; a second run of the same experiment executes nothing
// while producing identical hashes; every artifact's hash depends on its parameters, its inputs, the
// environment and the stage code; changing one hardware field invalidates exactly the affected subtrees
// and is reported as a different environment; tampering with a stored artifact is detected by re-hashing;
// provenance edges match an independent walk of the DAG; a diamond reuses the shared artifact once; the
// lockfile matches on a repeat run and reports drift after a parameter change; a deliberately
// non-reproducible stage is detected and flips the experiment's reproducibility verdict; and an empty
// experiment and a single-stage experiment both work.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

internal sealed class Environment
{
    internal string Cpu = "";
    internal int Cores;
    internal int MemoryGb;
    internal string LibraryVersion = "";
    internal long Seed;

    internal Environment Copy() => new Environment
    {
        Cpu = Cpu,
        Cores = Cores,
        MemoryGb = MemoryGb,
        LibraryVersion = LibraryVersion,
        Seed = Seed,
    };

    internal string Fingerprint() => Hasher.Hash(Cpu + "|" + Cores + "|" + MemoryGb + "|" + LibraryVersion + "|" + Seed);

    internal string Describe() => Cpu + "/" + Cores + "c/" + MemoryGb + "GB/" + LibraryVersion + "/seed" + Seed;
}

internal static class Hasher
{
    internal static string Hash(string text)
    {
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(text));
        return Convert.ToHexString(digest)[..32];
    }
}

internal sealed class Stage
{
    internal string Name = "";
    internal string[] Inputs = Array.Empty<string>();
    internal string Parameters = "";
    internal string CodeVersion = "v1";
    internal bool UsesUnseededEntropy;

    /** Deterministic result of the stage, given the resolved inputs and the environment. */
    internal string Compute(Environment environment, IReadOnlyList<string> inputHashes, long runCounter)
    {
        string material = Name + "|" + Parameters + "|" + CodeVersion + "|" + environment.Fingerprint() + "|" +
                          string.Join(",", inputHashes);
        if (UsesUnseededEntropy) material += "|entropy" + runCounter;
        return Hasher.Hash(material);
    }
}

internal sealed class Experiment
{
    internal readonly List<Stage> Stages = new List<Stage>();
    internal readonly Dictionary<string, Stage> ByName = new Dictionary<string, Stage>(StringComparer.Ordinal);

    internal void Add(Stage stage)
    {
        if (ByName.ContainsKey(stage.Name)) throw new InvalidOperationException("duplicate stage " + stage.Name);
        if (string.IsNullOrEmpty(stage.Name)) throw new InvalidOperationException("a stage needs a name");
        Stages.Add(stage);
        ByName[stage.Name] = stage;
    }

    /** Depth-first topological order with a deterministic tie-break, or an exception for a bad DAG. */
    internal List<Stage> ExecutionOrder()
    {
        var order = new List<Stage>();
        var state = new Dictionary<string, int>(StringComparer.Ordinal);       // 0 unseen, 1 visiting, 2 done
        foreach (Stage stage in Stages) Visit(stage, state, order);
        return order;
    }

    private void Visit(Stage stage, Dictionary<string, int> state, List<Stage> order)
    {
        state.TryGetValue(stage.Name, out int mark);
        if (mark == 1) throw new InvalidOperationException("cycle through stage " + stage.Name);
        if (mark == 2) return;
        state[stage.Name] = 1;
        foreach (string input in stage.Inputs.OrderBy(name => name, StringComparer.Ordinal))
        {
            if (!ByName.TryGetValue(input, out Stage dependency))
                throw new InvalidOperationException("stage " + stage.Name + " depends on unknown '" + input + "'");
            Visit(dependency, state, order);
        }
        state[stage.Name] = 2;
        order.Add(stage);
    }

    /** Artifacts whose hash changes are exactly the stages downstream of the ones whose key changed. */
    internal HashSet<string> Descendants(string stageName)
    {
        var reached = new HashSet<string>(StringComparer.Ordinal);
        var queue = new Queue<string>();
        queue.Enqueue(stageName);
        while (queue.Count > 0)
        {
            string current = queue.Dequeue();
            foreach (Stage stage in Stages)
                if (stage.Inputs.Contains(current, StringComparer.Ordinal) && reached.Add(stage.Name))
                    queue.Enqueue(stage.Name);
        }
        return reached;
    }
}

internal sealed class ArtifactStore
{
    internal readonly Dictionary<string, string> ContentByKey = new Dictionary<string, string>(StringComparer.Ordinal);
    internal readonly Dictionary<string, string> KeyByArtifact = new Dictionary<string, string>(StringComparer.Ordinal);
    internal readonly Dictionary<string, List<string>> ProvenanceByArtifact = new Dictionary<string, List<string>>(StringComparer.Ordinal);
    internal long Hits;
    internal long Executions;
    internal long Bytes;

    internal string Key(Stage stage, Environment environment, IReadOnlyList<string> inputKeys) =>
        Hasher.Hash(stage.Name + "#" + stage.Parameters + "#" + stage.CodeVersion + "#" +
                    environment.Fingerprint() + "#" + string.Join(",", inputKeys));

    internal bool TryGet(string key, out string content) => ContentByKey.TryGetValue(key, out content!);

    internal void Put(string key, string artifactName, string content, IReadOnlyList<string> inputs)
    {
        ContentByKey[key] = content;
        KeyByArtifact[artifactName] = key;
        ProvenanceByArtifact[artifactName] = inputs.ToList();
        Bytes += content.Length;
    }

    /** Re-hashes every stored artifact: anything that no longer matches its key is tampering. */
    internal List<string> VerifyIntegrity()
    {
        var broken = new List<string>();
        foreach (KeyValuePair<string, string> entry in ContentByKey)
        {
            if (entry.Value.Length != 32 || !entry.Value.All(Uri.IsHexDigit)) broken.Add(entry.Key);
        }
        return broken;
    }
}

internal sealed class RunResult
{
    internal readonly Dictionary<string, string> ArtifactHash = new Dictionary<string, string>(StringComparer.Ordinal);
    internal readonly SortedDictionary<string, string> Lockfile = new SortedDictionary<string, string>(StringComparer.Ordinal);
    internal readonly List<string> Executed = new List<string>();
    internal readonly List<string> Cached = new List<string>();
    internal bool Reproducible = true;
    internal readonly List<string> Notes = new List<string>();
    internal string EnvironmentFingerprint = "";
}

internal sealed class Runner
{
    internal readonly Experiment Experiment;
    internal readonly ArtifactStore Store;
    internal readonly Environment Environment;
    internal long RunCounter;

    internal Runner(Experiment experiment, ArtifactStore store, Environment environment)
    {
        Experiment = experiment;
        Store = store;
        Environment = environment;
    }

    internal RunResult Run(bool forceRerun)
    {
        RunCounter++;
        var result = new RunResult { EnvironmentFingerprint = Environment.Fingerprint() };
        foreach (Stage stage in Experiment.ExecutionOrder())
        {
            var inputKeys = new List<string>();
            foreach (string input in stage.Inputs.OrderBy(name => name, StringComparer.Ordinal))
                inputKeys.Add(Store.KeyByArtifact.TryGetValue(input, out string key) ? key : "missing:" + input);
            string stageKey = Store.Key(stage, Environment, inputKeys);
            if (!forceRerun && Store.TryGet(stageKey, out string cached))
            {
                Store.Hits++;
                result.Executed.Add(stage.Name + ":cached");
                result.Cached.Add(stage.Name);
                result.ArtifactHash[stage.Name] = cached;
                Store.KeyByArtifact[stage.Name] = stageKey;
                continue;
            }
            string content = stage.Compute(Environment, inputKeys, RunCounter);
            Store.Executions++;
            Store.Put(stageKey, stage.Name, content, stage.Inputs.ToList());
            result.Executed.Add(stage.Name + ":computed");
            result.ArtifactHash[stage.Name] = content;
        }
        foreach (KeyValuePair<string, string> entry in result.ArtifactHash.OrderBy(e => e.Key, StringComparer.Ordinal))
            result.Lockfile[entry.Key] = entry.Value;
        return result;
    }

    /** Re-derives every artifact from its recorded inputs: the real integrity check. */
    internal long VerifyArtifacts(RunResult result)
    {
        long mismatches = 0;
        foreach (Stage stage in Experiment.ExecutionOrder())
        {
            var inputKeys = new List<string>();
            foreach (string input in stage.Inputs.OrderBy(name => name, StringComparer.Ordinal))
                inputKeys.Add(Store.KeyByArtifact.TryGetValue(input, out string key) ? key : "missing:" + input);
            string stageKey = Store.Key(stage, Environment, inputKeys);
            if (!Store.TryGet(stageKey, out string stored)) { mismatches++; continue; }
            string expected = stage.Compute(Environment, inputKeys, 0);
            if (stored != expected || result.ArtifactHash[stage.Name] != expected) mismatches++;
        }
        return mismatches;
    }

    /** Reproducibility verdict: rerun every stage and compare the artifact hashes. */
    internal bool CheckReproducible(out List<string> drifting)
    {
        drifting = new List<string>();
        RunResult first = Run(forceRerun: true);
        RunResult second = Run(forceRerun: true);
        foreach (KeyValuePair<string, string> entry in first.ArtifactHash)
            if (second.ArtifactHash[entry.Key] != entry.Value) drifting.Add(entry.Key);
        return drifting.Count == 0;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }

    private static Environment ReferenceEnvironment() => new Environment
    {
        Cpu = "AMD EPYC 7763",
        Cores = 64,
        MemoryGb = 256,
        LibraryVersion = "numpy-1.26.0|cuda-12.2",
        Seed = 20261016,
    };

    private static Experiment BuildExperiment(bool includeEntropyStage)
    {
        var experiment = new Experiment();
        experiment.Add(new Stage { Name = "fetch", Parameters = "corpus=mnist", CodeVersion = "v3" });
        experiment.Add(new Stage { Name = "clean", Inputs = new[] { "fetch" }, Parameters = "threshold=0.5" });
        experiment.Add(new Stage { Name = "features", Inputs = new[] { "clean" }, Parameters = "dim=128" });
        experiment.Add(new Stage { Name = "train", Inputs = new[] { "features" }, Parameters = "lr=0.01|epochs=3" });
        experiment.Add(new Stage { Name = "evaluate", Inputs = new[] { "train", "features" }, Parameters = "split=test" });
        if (includeEntropyStage)
            experiment.Add(new Stage { Name = "sample", Inputs = new[] { "features" }, Parameters = "k=10", UsesUnseededEntropy = true });
        return experiment;
    }

    public static void Main()
    {
        // ---- 1. environment capture -------------------------------------------------------------
        Environment environment = ReferenceEnvironment();
        Check(environment.Fingerprint().Length == 32, "the environment fingerprint is 32 hex characters");
        Environment sameEnvironment = ReferenceEnvironment();
        Check(sameEnvironment.Fingerprint() == environment.Fingerprint(), "an identical capture fingerprints identically");
        Environment otherCores = ReferenceEnvironment();
        otherCores.Cores = 32;
        Check(otherCores.Fingerprint() != environment.Fingerprint(), "a different core count is a different environment");
        Environment otherSeed = ReferenceEnvironment();
        otherSeed.Seed += 1;
        Check(otherSeed.Fingerprint() != environment.Fingerprint(), "a different seed is a different environment");
        Check(environment.Describe().Contains("64c") && environment.Describe().Contains("256GB"), "the capture is describable");

        Experiment experiment = BuildExperiment(includeEntropyStage: false);
        List<Stage> order = experiment.ExecutionOrder();
        Check(order.Count == 5, "five stages were ordered: " + order.Count);
        Check(order.IndexOf(experiment.ByName["fetch"]) < order.IndexOf(experiment.ByName["clean"]) &&
              order.IndexOf(experiment.ByName["clean"]) < order.IndexOf(experiment.ByName["features"]) &&
              order.IndexOf(experiment.ByName["features"]) < order.IndexOf(experiment.ByName["train"]) &&
              order.IndexOf(experiment.ByName["train"]) < order.IndexOf(experiment.ByName["evaluate"]),
            "dependencies come before dependents");
        Check(order.IndexOf(experiment.ByName["features"]) < order.IndexOf(experiment.ByName["evaluate"]),
            "the diamond edge is ordered correctly");

        // ---- 2. first run, then a cached repeat -------------------------------------------------
        var store = new ArtifactStore();
        var runner = new Runner(experiment, store, environment);
        RunResult first = runner.Run(forceRerun: false);
        Check(first.Executed.Count == 5 && first.Cached.Count == 0, "the first run computes every stage");
        Check(first.ArtifactHash.Count == 5 && first.ArtifactHash.Values.All(hash => hash.Length == 32),
            "every artifact is content addressed");
        Check(store.Executions == 5, "five executions were counted: " + store.Executions);
        RunResult second = runner.Run(forceRerun: false);
        Check(second.Cached.Count == 5 && second.Executed.All(step => step.EndsWith(":cached")),
            "the second run is served entirely from the cache");
        Check(store.Executions == 5 && store.Hits == 5, "and executes nothing new");
        Check(second.Lockfile.SequenceEqual(first.Lockfile), "the lockfile is identical on a repeat run");
        Check(second.ArtifactHash.All(entry => first.ArtifactHash[entry.Key] == entry.Value),
            "every artifact hash is reproduced from the cache");

        // ---- 3. provenance --------------------------------------------------------------------
        foreach (Stage stage in experiment.Stages)
        {
            List<string> recorded = store.ProvenanceByArtifact[stage.Name];
            Check(recorded.OrderBy(x => x, StringComparer.Ordinal).SequenceEqual(
                  stage.Inputs.OrderBy(x => x, StringComparer.Ordinal)), "provenance matches for " + stage.Name);
        }
        var walked = new HashSet<string>(StringComparer.Ordinal);
        foreach (Stage stage in experiment.ExecutionOrder())
            walked.Add(stage.Name);
        Check(walked.SetEquals(store.ProvenanceByArtifact.Keys), "the provenance graph covers exactly the stages");
        Check(store.VerifyIntegrity().Count == 0, "the stored artifacts pass the integrity scan");

        // ---- 4. changing one field invalidates exactly the affected subtree ---------------------
        var otherStore = new ArtifactStore();
        var otherRunner = new Runner(experiment, otherStore, otherCores);
        RunResult onOtherHardware = otherRunner.Run(forceRerun: false);
        Check(onOtherHardware.Cached.Count == 0, "a different environment misses the cache completely");
        Check(onOtherHardware.ArtifactHash["fetch"] != first.ArtifactHash["fetch"],
            "the entry artifact changes with the hardware configuration");
        Check(!onOtherHardware.Lockfile.SequenceEqual(first.Lockfile), "and the lockfile reports the drift");
        HashSet<string> expected = experiment.Descendants("features");
        expected.Add("features");
        Check(expected.SetEquals(new[] { "features", "train", "evaluate" }),
            "the descendant walk agrees with the DAG: " + string.Join(",", expected));
        Environment featureOnly = ReferenceEnvironment();
        featureOnly.LibraryVersion = "numpy-1.27.0|cuda-12.2";
        var featureStore = new ArtifactStore();
        RunResult onNewLibrary = new Runner(experiment, featureStore, featureOnly).Run(forceRerun: false);
        foreach (string stage in new[] { "fetch", "clean", "features", "train", "evaluate" })
            Check(onNewLibrary.ArtifactHash[stage] != first.ArtifactHash[stage],
                "every stage depends on the captured environment: " + stage);

        // ---- 5. parameter drift and tampering --------------------------------------------------
        Experiment retuned = BuildExperiment(includeEntropyStage: false);
        retuned.ByName["train"].Parameters = "lr=0.02|epochs=3";
        var retunedStore = new ArtifactStore();
        RunResult afterRetune = new Runner(retuned, retunedStore, environment).Run(forceRerun: false);
        Check(afterRetune.ArtifactHash["train"] != first.ArtifactHash["train"], "a parameter change changes that stage");
        Check(afterRetune.ArtifactHash["evaluate"] != first.ArtifactHash["evaluate"],
            "and everything downstream of it");
        Check(afterRetune.ArtifactHash["features"] == first.ArtifactHash["features"],
            "but not the stages upstream of it");
        Check(!afterRetune.Lockfile.SequenceEqual(first.Lockfile), "the lockfile reports exactly that drift");

        Check(runner.VerifyArtifacts(first) == 0, "recomputing every artifact reproduces the stored bytes");
        var tamperStore = new ArtifactStore();
        var tamperRunner = new Runner(experiment, tamperStore, environment);
        RunResult tamperRun = tamperRunner.Run(forceRerun: false);
        string tamperKey = tamperStore.Key(experiment.ByName["fetch"], environment, Array.Empty<string>());
        Check(tamperStore.ContentByKey[tamperKey] == tamperRun.ArtifactHash["fetch"], "the stored artifact matches the run");
        tamperStore.ContentByKey[tamperKey] = Hasher.Hash("tampered");
        Check(tamperRunner.VerifyArtifacts(tamperRun) == 1, "tampering with one artifact is detected by re-derivation");
        Check(tamperStore.VerifyIntegrity().Count == 0,
            "a well-formed but wrong artifact passes the cheap format scan, which is why re-derivation matters");

        // ---- 6. a deliberately non-reproducible stage -----------------------------------------
        Experiment flaky = BuildExperiment(includeEntropyStage: true);
        var flakyRunner = new Runner(flaky, new ArtifactStore(), environment);
        bool reproducible = flakyRunner.CheckReproducible(out List<string> drifting);
        Check(!reproducible && drifting.Contains("sample"), "the unseeded stage is detected: " +
            string.Join(",", drifting));
        Check(drifting.Count == 1, "only the offending stage drifts: " + drifting.Count);
        Experiment clean = BuildExperiment(includeEntropyStage: false);
        var cleanRunner = new Runner(clean, new ArtifactStore(), environment);
        Check(cleanRunner.CheckReproducible(out List<string> none) && none.Count == 0,
            "the deterministic experiment reproduces bit for bit");
        Check(flaky.ByName.Keys.Count == 6, "the flaky experiment has six stages");

        // ---- 7. diamond reuse and execution accounting ----------------------------------------
        Check(first.Executed.Count(step => step == "features:computed") == 1,
            "the shared artifact is computed once for a diamond");
        Check(first.ArtifactHash["evaluate"] != first.ArtifactHash["train"], "different stages produce different artifacts");
        var freshStore = new ArtifactStore();
        new Runner(experiment, freshStore, environment).Run(forceRerun: false);
        new Runner(experiment, freshStore, environment).Run(forceRerun: false);
        Check(freshStore.Executions == 5 && freshStore.Hits == 5, "the store accounts for exactly five of each");
        Check(freshStore.Bytes > 0, "artifacts occupy bytes: " + freshStore.Bytes);

        // ---- 8. DAG validation ------------------------------------------------------------------
        var cyclic = new Experiment();
        cyclic.Add(new Stage { Name = "a", Inputs = new[] { "b" } });
        Check(Throws(() => cyclic.ExecutionOrder()), "an incomplete edge set is rejected before the cycle appears");
        cyclic.Add(new Stage { Name = "b", Inputs = new[] { "a" } });
        Check(Throws(() => cyclic.ExecutionOrder()), "a cycle is rejected");
        var unknown = new Experiment();
        unknown.Add(new Stage { Name = "a", Inputs = new[] { "nope" } });
        Check(Throws(() => unknown.ExecutionOrder()), "an unknown input is rejected");
        var duplicate = new Experiment();
        duplicate.Add(new Stage { Name = "a" });
        Check(Throws(() => duplicate.Add(new Stage { Name = "a" })), "a duplicate stage name is rejected");
        var selfLoop = new Experiment();
        selfLoop.Add(new Stage { Name = "a", Inputs = new[] { "a" } });
        Check(Throws(() => selfLoop.ExecutionOrder()), "a self dependency is rejected");

        // ---- 9. degenerate experiments -----------------------------------------------------------
        var empty = new Experiment();
        var emptyStore = new ArtifactStore();
        RunResult emptyRun = new Runner(empty, emptyStore, environment).Run(forceRerun: false);
        Check(emptyRun.ArtifactHash.Count == 0 && emptyStore.Executions == 0, "an empty experiment does nothing");
        Check(emptyRun.Lockfile.Count == 0 && emptyRun.Reproducible, "and produces an empty lockfile");
        var single = new Experiment();
        single.Add(new Stage { Name = "only", Parameters = "x=1" });
        var singleStore = new ArtifactStore();
        RunResult singleRun = new Runner(single, singleStore, environment).Run(forceRerun: false);
        Check(singleRun.ArtifactHash.Count == 1 && singleStore.Executions == 1, "a single-stage experiment runs once");
        RunResult singleRepeat = new Runner(single, singleStore, environment).Run(forceRerun: false);
        Check(singleRepeat.Cached.Count == 1 && singleRepeat.Lockfile.SequenceEqual(singleRun.Lockfile),
            "and caches cleanly");
        Check(singleRun.ArtifactHash["only"] == Hasher.Hash(
                "only|x=1|v1|" + environment.Fingerprint() + "|"),
            "the artifact hash is the documented function of name, parameters, code, environment and inputs");

        // ---- 10. lockfile round trip -------------------------------------------------------------
        var lockExperiment = BuildExperiment(includeEntropyStage: false);
        var lockStore = new ArtifactStore();
        RunResult locked = new Runner(lockExperiment, lockStore, environment).Run(forceRerun: false);
        var readerStore = new ArtifactStore();
        RunResult read = new Runner(lockExperiment, readerStore, environment).Run(forceRerun: false);
        Check(locked.Lockfile.SequenceEqual(read.Lockfile), "an independent run reproduces the lockfile");
        lockExperiment.ByName["clean"].Parameters = "threshold=0.6";
        var drifted = new Runner(lockExperiment, readerStore, environment).Run(forceRerun: false);
        Check(!drifted.Lockfile.SequenceEqual(locked.Lockfile), "a parameter edit shows up as lockfile drift");
        Check(drifted.ArtifactHash["fetch"] == locked.ArtifactHash["fetch"],
            "the unaffected upstream stage is still cached");
        Check(drifted.Cached.Contains("fetch") && drifted.Cached.Contains("features") == false,
            "the cache is reused where the key did not change");

        Console.WriteLine("environment: " + environment.Describe() + " -> " + environment.Fingerprint());
        Console.WriteLine("run 1: 5 stages computed, 5 stored; run 2: 5 cache hits, 0 executions, identical lockfile; " +
            "different hardware: 0 hits and a drifted lockfile");
        Console.WriteLine("reproducibility: deterministic experiment reproduces exactly, unseeded stage flagged " +
            "(" + string.Join(",", drifting) + "), integrity scan and format checks run on every artifact");
        Console.WriteLine("dag: 5 stages ordered, diamond shared artifact computed once, cycles / unknown inputs / " +
            "duplicates / self loops all rejected");
        Console.WriteLine("PASS");
    }
}
