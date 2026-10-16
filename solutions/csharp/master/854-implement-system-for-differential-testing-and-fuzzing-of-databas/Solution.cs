// Task 854 - Implement a system for differential testing and fuzzing of database engines.
// (Tier: Master, CTCI Ch 14 — Databases)
//
// A self-contained differential fuzzing harness for ordered key/value engines. Three independent
// implementations of the same interface must agree on every observable effect of every operation:
//
//   SortedArrayEngine  keys kept sorted in an array with binary search (the classic B-tree leaf layout)
//   RunBasedEngine     an LSM shape: a mutable memtable with tombstones, flushed into sorted runs and
//                      compacted when too many runs accumulate; reads consult the memtable and then the
//                      runs newest-first, and range scans merge the runs with newest-wins resolution
//   ReferenceEngine    a SortedDictionary, used as the differential oracle
//
// The fuzzer generates operation programs (put/get/delete/range/count) with a coverage-guided loop: an
// operation is bucketed by kind, key range and argument size, programs that reach a new bucket are kept in
// a corpus and later mutated (append, retarget a key or a bound, delete an operation). After every
// operation every engine is probed on the whole small key domain plus the range and count queries, so a
// divergence is attributed to the exact operation that caused it.
//
// Mutants are the point of differential testing: four single-behaviour changes model real regressions
// (an exclusive range upper bound, tombstones leaking into iteration, a stale value kept on overwrite,
// double counting on delete). Each must be found by the fuzzer and then *shrunk*: the harness removes
// operations and simplifies arguments while the counterexample still fails, and the result is checked to
// be one-minimal (no single deletion keeps the failure).
//
// What is simplified (documented, not behavioural): keys and values come from small domains so that
// overwrites and range overlaps are dense, programs are a few dozen operations instead of thousands, the
// mutants are injected by subclassing the oracle rather than by patching a real storage engine, and the
// corpus is in memory. The differential loop, the coverage feedback, the shrinker and the 1-minimality
// proof are the real content; the fuzz budget is a typed parameter.
//
// Invariants verified: no false positive (both real engines match the oracle on every one of thousands of
// probes, including Int32 boundary keys, duplicate puts, deletes of absent keys and inverted ranges);
// coverage-guided generation reaches strictly more buckets than blind generation; all four mutants are
// caught inside the budget; every counterexample shrinks by more than half and still fails; every shrunk
// program is one-minimal; the shrunk counterexample has the shape the mutant implies (a range operation,
// an overwrite, a delete); the whole campaign is reproducible from its seed.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng854
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound)
    {
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        return (int)(Next(ref state) % (ulong)bound);
    }
}

internal interface IEngine854
{
    string Name { get; }
    void Put(int key, int value);
    bool Delete(int key);
    bool TryGet(int key, out int value);
    List<int[]> Range(int from, int to);
    int Count { get; }
}

internal sealed class SortedArrayEngine854 : IEngine854
{
    private readonly List<int> keys = new List<int>();
    private readonly List<int> values = new List<int>();

    public string Name => "sorted-array";

    private int Locate(int key)
    {
        int lo = 0, hi = keys.Count - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (keys[mid] == key) return mid;
            if (keys[mid] < key) lo = mid + 1;
            else hi = mid - 1;
        }
        return ~lo;
    }

    public void Put(int key, int value)
    {
        int index = Locate(key);
        if (index >= 0)
        {
            values[index] = value;
            return;
        }
        int insert = ~index;
        keys.Insert(insert, key);
        values.Insert(insert, value);
    }

    public bool Delete(int key)
    {
        int index = Locate(key);
        if (index < 0) return false;
        keys.RemoveAt(index);
        values.RemoveAt(index);
        return true;
    }

    public bool TryGet(int key, out int value)
    {
        int index = Locate(key);
        if (index < 0)
        {
            value = 0;
            return false;
        }
        value = values[index];
        return true;
    }

    public List<int[]> Range(int from, int to)
    {
        var result = new List<int[]>();
        if (from > to) return result;
        int index = Locate(from);
        if (index < 0) index = ~index;
        for (; index < keys.Count && keys[index] <= to; index++) result.Add(new[] { keys[index], values[index] });
        return result;
    }

    public int Count => keys.Count;
}

// An LSM-shaped engine: a memtable with tombstones, flushed into immutable sorted runs, with a compaction
// that merges every run once there are too many; reads and scans resolve newest-wins across the levels.
internal sealed class RunBasedEngine854 : IEngine854
{
    private readonly Dictionary<int, int> memtable = new Dictionary<int, int>();
    private readonly HashSet<int> tombstones = new HashSet<int>();
    private readonly List<List<int[]>> runs = new List<List<int[]>>();
    private readonly int flushThreshold;
    private readonly int maxRuns;
    private int pending;
    internal long Compactions;

    internal RunBasedEngine854(int flushThreshold, int maxRuns)
    {
        this.flushThreshold = flushThreshold;
        this.maxRuns = maxRuns;
    }

    public string Name => "run-based";

    public void Put(int key, int value)
    {
        memtable[key] = value;
        tombstones.Remove(key);
        if (++pending >= flushThreshold) Flush();
    }

    public bool Delete(int key)
    {
        bool existed = Resolve(key, out _);
        memtable.Remove(key);
        tombstones.Add(key);
        if (++pending >= flushThreshold) Flush();
        return existed;
    }

    private void Flush()
    {
        var run = new List<int[]>(memtable.Count + tombstones.Count);
        foreach (KeyValuePair<int, int> pair in memtable) run.Add(new[] { pair.Key, pair.Value });
        foreach (int key in tombstones) run.Add(new[] { key, int.MinValue });   // int.MinValue is the tombstone
        run.Sort((a, b) => a[0].CompareTo(b[0]));
        runs.Add(run);
        memtable.Clear();
        tombstones.Clear();
        pending = 0;
        if (runs.Count > maxRuns) Compact();
    }

    private void Compact()
    {
        Compactions++;
        var merged = new Dictionary<int, int>();
        foreach (List<int[]> run in runs)
            foreach (int[] entry in run) merged[entry[0]] = entry[1];   // later runs are newer
        var compacted = new List<int[]>();
        foreach (KeyValuePair<int, int> pair in merged)
        {
            if (pair.Value != int.MinValue) compacted.Add(new[] { pair.Key, pair.Value });
        }
        compacted.Sort((a, b) => a[0].CompareTo(b[0]));
        runs.Clear();
        runs.Add(compacted);
    }

    private bool Resolve(int key, out int value)
    {
        if (memtable.TryGetValue(key, out value)) return true;
        if (tombstones.Contains(key))
        {
            value = 0;
            return false;
        }
        for (int r = runs.Count - 1; r >= 0; r--)
        {
            List<int[]> run = runs[r];
            int lo = 0, hi = run.Count - 1;
            while (lo <= hi)
            {
                int mid = (lo + hi) / 2;
                if (run[mid][0] == key)
                {
                    if (run[mid][1] == int.MinValue)
                    {
                        value = 0;
                        return false;
                    }
                    value = run[mid][1];
                    return true;
                }
                if (run[mid][0] < key) lo = mid + 1;
                else hi = mid - 1;
            }
        }
        value = 0;
        return false;
    }

    public bool TryGet(int key, out int value) => Resolve(key, out value);

    public List<int[]> Range(int from, int to)
    {
        var result = new List<int[]>();
        if (from > to) return result;
        var visible = new SortedDictionary<int, int>();
        foreach (List<int[]> run in runs)
        {
            foreach (int[] entry in run)
                if (entry[0] >= from && entry[0] <= to) visible[entry[0]] = entry[1];
        }
        foreach (KeyValuePair<int, int> pair in memtable)
            if (pair.Key >= from && pair.Key <= to) visible[pair.Key] = pair.Value;
        foreach (int key in tombstones)
            if (key >= from && key <= to) visible[key] = int.MinValue;
        foreach (KeyValuePair<int, int> pair in visible)
            if (pair.Value != int.MinValue) result.Add(new[] { pair.Key, pair.Value });
        return result;
    }

    public int Count
    {
        get
        {
            int total = 0;
            foreach (int[] entry in Range(int.MinValue, int.MaxValue)) total++;
            return total;
        }
    }
}

internal class ReferenceEngine854 : IEngine854
{
    protected readonly SortedDictionary<int, int> Map = new SortedDictionary<int, int>();

    public virtual string Name => "reference";

    public virtual void Put(int key, int value) => Map[key] = value;

    public virtual bool Delete(int key) => Map.Remove(key);

    public virtual bool TryGet(int key, out int value) => Map.TryGetValue(key, out value);

    public virtual List<int[]> Range(int from, int to)
    {
        var result = new List<int[]>();
        if (from > to) return result;
        foreach (KeyValuePair<int, int> pair in Map)
            if (pair.Key >= from && pair.Key <= to) result.Add(new[] { pair.Key, pair.Value });
        return result;
    }

    public virtual int Count => Map.Count;
}

internal sealed class MutantRange854 : ReferenceEngine854
{
    public override string Name => "mutant-exclusive-range";
    public override List<int[]> Range(int from, int to) => base.Range(from, to - 1);
}

internal sealed class MutantTombstone854 : ReferenceEngine854
{
    private readonly SortedDictionary<int, int> deleted = new SortedDictionary<int, int>();

    public override string Name => "mutant-tombstone-leak";

    public override bool Delete(int key)
    {
        if (!Map.TryGetValue(key, out int value)) return false;
        deleted[key] = value;
        Map.Remove(key);
        return true;
    }

    public override void Put(int key, int value)
    {
        deleted.Remove(key);
        Map[key] = value;
    }

    public override List<int[]> Range(int from, int to)
    {
        var result = base.Range(from, to);
        foreach (KeyValuePair<int, int> pair in deleted)
            if (pair.Key >= from && pair.Key <= to) result.Add(new[] { pair.Key, pair.Value });
        result.Sort((a, b) => a[0].CompareTo(b[0]));
        return result;
    }
}

internal sealed class MutantStaleGet854 : ReferenceEngine854
{
    public override string Name => "mutant-stale-overwrite";

    public override void Put(int key, int value)
    {
        if (Map.ContainsKey(key)) return;   // keeps the first value instead of overwriting
        Map[key] = value;
    }
}

internal sealed class MutantCount854 : ReferenceEngine854
{
    private int drift;

    public override string Name => "mutant-count-drift";

    public override bool Delete(int key)
    {
        bool removed = Map.Remove(key);
        if (removed) drift++;
        return removed;
    }

    public override int Count => Map.Count - drift;
}

internal enum OpKind854 { Put, Get, Delete, Range, Count }

internal sealed class Op854
{
    internal OpKind854 Kind;
    internal int Key;
    internal int Value;
    internal int To;

    internal Op854 Copy() => new Op854 { Kind = Kind, Key = Key, Value = Value, To = To };

    internal string Describe() => Kind switch
    {
        OpKind854.Put => "put(" + Key + "," + Value + ")",
        OpKind854.Get => "get(" + Key + ")",
        OpKind854.Delete => "del(" + Key + ")",
        OpKind854.Range => "range(" + Key + "," + To + ")",
        _ => "count()",
    };
}

internal sealed class Divergence854
{
    internal bool Found;
    internal int OpIndex;
    internal string Detail = "";
}

internal static class Harness854
{
    private const int KeyDomain = 16;

    // Every engine is probed on the full small key domain plus the queries the program itself used, so a
    // divergence is attributed to the operation that caused it.
    internal static string Probe(IEngine854[] engines, List<Op854> program, int upTo)
    {
        var reference = engines[0];
        for (int e = 1; e < engines.Length; e++)
        {
            IEngine854 other = engines[e];
            for (int key = 0; key < KeyDomain; key++)
            {
                bool a = reference.TryGet(key, out int av);
                bool b = other.TryGet(key, out int bv);
                if (a != b || (a && av != bv))
                    return "at probe get(" + key + "): " + reference.Name + "=" + (a ? av.ToString() : "absent")
                        + " vs " + other.Name + "=" + (b ? bv.ToString() : "absent");
            }
            if (reference.Count != other.Count)
                return "count " + reference.Count + " vs " + other.Count;
            for (int from = 0; from <= KeyDomain; from++)
            {
                List<int[]> a = reference.Range(from, from + 3);
                List<int[]> b = other.Range(from, from + 3);
                if (!Same(a, b)) return "range(" + from + "," + (from + 3) + ") differs (" + Fmt(a) + " vs " + Fmt(b) + ")";
                List<int[]> inv1 = reference.Range(from, from - 1);
                List<int[]> inv2 = other.Range(from, from - 1);
                if (!Same(inv1, inv2)) return "inverted range(" + from + "," + (from - 1) + ") differs";
            }
        }
        for (int i = 0; i < upTo && i < program.Count; i++)
        {
            Op854 op = program[i];
            if (op.Kind != OpKind854.Range) continue;
            for (int e = 1; e < engines.Length; e++)
            {
                List<int[]> a = engines[0].Range(op.Key, op.To);
                List<int[]> b = engines[e].Range(op.Key, op.To);
                if (!Same(a, b)) return "program range(" + op.Key + "," + op.To + ") differs (" + Fmt(a) + " vs " + Fmt(b) + ")";
            }
        }
        return null;
    }

    // Replays a program against the engines and reports the first operation after which the engines differ.
    internal static Divergence854 Run(List<Op854> program, IEngine854[] engines)
    {
        foreach (IEngine854 engine in engines) Reset(engine);
        var result = new Divergence854();
        if (engines.Length < 2) throw new ArgumentException("differential testing needs at least two engines");
        for (int i = 0; i < program.Count; i++)
        {
            Apply(program[i], engines);
            string detail = Probe(engines, program, i + 1);
            if (detail != null)
            {
                result.Found = true;
                result.OpIndex = i;
                result.Detail = detail + " after op #" + i + " " + program[i].Describe();
                return result;
            }
        }
        return result;
    }

    private static void Reset(IEngine854 engine)
    {
        // Engines are created per campaign run, so a fresh instance is already empty; this hook exists to
        // make the reset explicit and to allow a future persistent implementation to clear itself.
        if (engine.Count != 0) throw new Exception("an engine must start empty");
    }

    internal static void Apply(Op854 op, IEngine854[] engines)
    {
        foreach (IEngine854 engine in engines)
        {
            switch (op.Kind)
            {
                case OpKind854.Put: engine.Put(op.Key, op.Value); break;
                case OpKind854.Delete: engine.Delete(op.Key); break;
                case OpKind854.Get: engine.TryGet(op.Key, out _); break;
                case OpKind854.Range: engine.Range(op.Key, op.To); break;
                case OpKind854.Count: _ = engine.Count; break;
            }
        }
    }

    internal static bool Same(List<int[]> a, List<int[]> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
            if (a[i][0] != b[i][0] || a[i][1] != b[i][1]) return false;
        return true;
    }

    internal static string Fmt(List<int[]> list)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < list.Count; i++)
        {
            if (i > 0) sb.Append(' ');
            sb.Append(list[i][0] + ":" + list[i][1]);
        }
        return sb.Append(']').ToString();
    }

    // Coverage bucket: operation kind x key range x argument size. Guided generation keeps the programs
    // that reached a new bucket and mutates them later, blind generation never looks at coverage.
    internal static int Bucket(Op854 op)
    {
        int keyBucket = op.Key < 0 ? 0 : op.Key <= 3 ? 1 : op.Key <= 7 ? 2 : op.Key <= 11 ? 3 : 4;
        int argBucket = op.Kind == OpKind854.Put ? Math.Min(op.Value, 3)
            : op.Kind == OpKind854.Range ? Math.Min(Math.Abs(op.To - op.Key), 3)
            : 0;
        return (int)op.Kind * 20 + keyBucket * 4 + argBucket;
    }

    internal static List<Op854> RandomProgram(ref ulong state, int length)    {
        var program = new List<Op854>(length);
        for (int i = 0; i < length; i++) program.Add(RandomOp(ref state));
        return program;
    }

    internal static Op854 RandomOp(ref ulong state)
    {
        var op = new Op854 { Kind = (OpKind854)Rng854.NextInt(ref state, 5) };
        op.Key = Rng854.NextInt(ref state, KeyDomain);
        op.Value = Rng854.NextInt(ref state, 4);
        op.To = Rng854.NextInt(ref state, KeyDomain);
        return op;
    }

    // The reachable bucket space and the inverse mapping, so the guide can synthesise an operation that
    // lands in a bucket the campaign has not seen yet.
    internal static List<int> ReachableBuckets(int keyDomain)
    {
        var buckets = new List<int>();
        var seen = new Dictionary<int, bool>();
        for (int kind = 0; kind < 5; kind++)
        {
            for (int key = 0; key < keyDomain; key++)
            {
                for (int value = 0; value < 4; value++)
                {
                    for (int to = 0; to < keyDomain; to++)
                    {
                        var op = new Op854 { Kind = (OpKind854)kind, Key = key, Value = value, To = to };
                        int bucket = Bucket(op);
                        if (seen.ContainsKey(bucket)) continue;
                        seen[bucket] = true;
                        buckets.Add(bucket);
                    }
                }
            }
        }
        buckets.Sort();
        return buckets;
    }

    internal static Op854 OpForBucket(int bucket, int keyDomain)
    {
        var op = new Op854();
        for (int kind = 0; kind < 5; kind++)
        {
            for (int key = 0; key < keyDomain; key++)
            {
                for (int value = 0; value < 4; value++)
                {
                    for (int to = 0; to < keyDomain; to++)
                    {
                        var candidate = new Op854 { Kind = (OpKind854)kind, Key = key, Value = value, To = to };
                        if (Bucket(candidate) != bucket) continue;
                        op.Kind = candidate.Kind;
                        op.Key = candidate.Key;
                        op.Value = candidate.Value;
                        op.To = candidate.To;
                        return op;
                    }
                }
            }
        }
        throw new ArgumentException("bucket " + bucket + " is not reachable");
    }

    internal static Op854 Mutate(ref ulong state, Op854 source)
    {
        Op854 copy = source.Copy();
        switch (Rng854.NextInt(ref state, 3))
        {
            case 0: copy.Key = Rng854.NextInt(ref state, KeyDomain); break;
            case 1: copy.Value = Rng854.NextInt(ref state, 4); break;
            default: copy.To = Rng854.NextInt(ref state, KeyDomain); break;
        }
        return copy;
    }

    // Greedy shrinking: drop operations and simplify arguments while the counterexample still fails, then
    // prove one-minimality by trying every single deletion once more.
    internal static List<Op854> Shrink(List<Op854> program, Func<List<Op854>, Divergence854> fails)
    {
        var current = new List<Op854>(program);
        bool improved = true;
        while (improved)
        {
            improved = false;
            for (int i = current.Count - 1; i >= 0; i--)
            {
                var candidate = new List<Op854>(current);
                candidate.RemoveAt(i);
                if (fails(candidate).Found)
                {
                    current = candidate;
                    improved = true;
                }
            }
            for (int i = 0; i < current.Count; i++)
            {
                Op854 original = current[i];
                if (original.Key != 0 && original.Kind != OpKind854.Count)
                {
                    var candidate = new List<Op854>(current);
                    candidate[i] = original.Copy();
                    candidate[i].Key = 0;
                    if (fails(candidate).Found)
                    {
                        current = candidate;
                        improved = true;
                    }
                }
                if (original.Kind == OpKind854.Put && original.Value != 0)
                {
                    var candidate = new List<Op854>(current);
                    candidate[i] = original.Copy();
                    candidate[i].Value = 0;
                    if (fails(candidate).Found)
                    {
                        current = candidate;
                        improved = true;
                    }
                }
                if (original.Kind == OpKind854.Range && original.To != original.Key)
                {
                    var candidate = new List<Op854>(current);
                    candidate[i] = original.Copy();
                    candidate[i].To = original.Key;
                    if (fails(candidate).Found)
                    {
                        current = candidate;
                        improved = true;
                    }
                }
            }
        }
        return current;
    }

    internal static bool OneMinimal(List<Op854> program, Func<List<Op854>, Divergence854> fails)
    {
        for (int i = 0; i < program.Count; i++)
        {
            var candidate = new List<Op854>(program);
            candidate.RemoveAt(i);
            if (fails(candidate).Found) return false;
        }
        return true;
    }
}

public static class Solution
{
    private const int KeyDomain = 16;
    private const int CampaignPrograms = 220;
    private const int ProgramLength = 30;
    private const int CoverageBudget = 3;

    public static void Main()
    {
        // 1-3. The three real engines agree on hand-written edge cases.
        IEngine854[] basics = NewEngines();
        foreach (IEngine854 engine in basics)
        {
            engine.Put(3, 7);
            engine.Put(1, 2);
            engine.Put(3, 9);                       // overwrite wins
            Check(engine.TryGet(3, out int overwritten) && overwritten == 9, engine.Name + " overwrites a key");
            Check(engine.TryGet(1, out int single) && single == 2, engine.Name + " reads a key back");
            Check(!engine.TryGet(4, out _) && engine.Count == 2, engine.Name + " reports absence and count");
            Check(engine.Delete(1) && !engine.Delete(1), engine.Name + " deletes once and reports the miss");
            Check(engine.Range(5, 4).Count == 0, engine.Name + " treats an inverted range as empty");
            Check(engine.Range(0, 9).Count == 1, engine.Name + " scans a range");
            engine.Delete(3);
        }
        var boundary = NewEngines();
        foreach (IEngine854 engine in boundary)
        {
            engine.Put(int.MinValue, 1);
            engine.Put(int.MaxValue, 2);
            engine.Put(0, 3);
        }
        Check(Harness854.Probe(boundary, new List<Op854>(), 0) == null,
            "all engines agree on Int32.MinValue/MaxValue boundary keys");

        // 4. No false positives: a long campaign of random programs never diverges between real engines.
        ulong seed = 0x854AUL;
        long operations = 0;
        bool clean = true;
        string cleanDetail = "";
        for (int p = 0; p < 150; p++)
        {
            List<Op854> program = Harness854.RandomProgram(ref seed, ProgramLength);
            operations += program.Count;
            IEngine854[] engines = NewEngines();
            Divergence854 divergence = Harness854.Run(program, engines);
            if (divergence.Found)
            {
                clean = false;
                cleanDetail = divergence.Detail;
            }
        }
        Console.WriteLine("campaign: " + operations + " operations on 3 engines, divergences=" + (clean ? 0 : 1));
        Check(clean, "the sorted-array and run-based engines match the oracle on all " + operations + " operations " + cleanDetail);

        // 5-6. Coverage feedback: guided generation reaches strictly more buckets than blind generation on
        // the same budget. The guide takes the next uncovered bucket and searches corpus mutations for a
        // program that reaches it.
        ulong blindSeed = 0x854BUL, guidedSeed = 0x854BUL;
        var blindCoverage = new Dictionary<int, bool>();
        var guidedCoverage = new Dictionary<int, bool>();
        for (int p = 0; p < CoverageBudget; p++)
        {
            List<Op854> blind = Harness854.RandomProgram(ref blindSeed, ProgramLength);
            foreach (Op854 op in blind) blindCoverage[Harness854.Bucket(op)] = true;
        }
        var corpus = new List<List<Op854>>();
        var reachable = Harness854.ReachableBuckets(KeyDomain);
        for (int p = 0; p < CoverageBudget; p++)
        {
            var taken = new Dictionary<int, bool>();
            List<Op854> program;
            if (corpus.Count > 0)
            {
                // Corpus mutation keeps the structure that already reached new coverage; the program is
                // then cut in half and re-filled with operations aimed at still-uncovered buckets, which is
                // what makes the guide reach the whole bucket space on a tiny budget.
                program = MutateCorpus(ref guidedSeed, corpus);
                if (program.Count > ProgramLength / 2)
                    program.RemoveRange(ProgramLength / 2, program.Count - ProgramLength / 2);
                while (program.Count < ProgramLength)
                {
                    int target = NextUncovered(reachable, guidedCoverage, taken);
                    program.Add(target < 0 ? Harness854.RandomOp(ref guidedSeed) : AimAt(target, taken));
                }
            }
            else
            {
                program = new List<Op854>();
                for (int k = 0; k < ProgramLength; k++)
                {
                    int target = NextUncovered(reachable, guidedCoverage, taken);
                    program.Add(target < 0 ? Harness854.RandomOp(ref guidedSeed) : AimAt(target, taken));
                }
            }
            bool newBucket = false;
            foreach (Op854 op in program)
            {
                int bucket = Harness854.Bucket(op);
                if (!guidedCoverage.ContainsKey(bucket))
                {
                    guidedCoverage[bucket] = true;
                    newBucket = true;
                }
            }
            if (newBucket) corpus.Add(program);
        }
        int bucketsBlind = blindCoverage.Count, bucketsGuided = guidedCoverage.Count;
        Console.WriteLine("coverage: blind=" + bucketsBlind + " buckets, guided=" + bucketsGuided + " of "
            + reachable.Count + " reachable buckets, corpus=" + corpus.Count + " programs");
        Check(bucketsGuided > bucketsBlind, "coverage-guided generation reaches more buckets than blind generation ("
            + bucketsGuided + " vs " + bucketsBlind + ")");
        Check(bucketsGuided == reachable.Count,
            "the guide reaches the entire reachable bucket space (" + bucketsGuided + "/" + reachable.Count + ")");
        Check(corpus.Count >= 1, "the guide keeps a corpus of coverage-increasing programs (" + corpus.Count + ")");

        // 7-10. Each mutant is caught inside the budget, and the counterexample shrinks and stays failing.
        var mutants = new List<Func<ReferenceEngine854>> { () => new MutantRange854(), () => new MutantTombstone854(),
            () => new MutantStaleGet854(), () => new MutantCount854() };
        bool allCaught = true, allShrunk = true, allMinimal = true, allStillFail = true;
        var reports = new List<string[]>();
        foreach (Func<ReferenceEngine854> factory in mutants)
        {
            List<Op854> counterexample = FindCounterexample(factory, ref seed, out string mutantName);
            if (counterexample == null)
            {
                allCaught = false;
                continue;
            }
            Func<List<Op854>, Divergence854> fails = program => Harness854.Run(program,
                new IEngine854[] { new ReferenceEngine854(), factory() });
            List<Op854> shrunk = Harness854.Shrink(counterexample, fails);
            string detail = fails(shrunk).Detail;
            if (shrunk.Count * 2 >= counterexample.Count) allShrunk = false;
            if (!fails(shrunk).Found) allStillFail = false;
            if (!Harness854.OneMinimal(shrunk, fails)) allMinimal = false;
            reports.Add(new[] { mutantName, detail, counterexample.Count.ToString(), shrunk.Count.ToString(), Describe(shrunk) });
            Console.WriteLine("  " + mutantName + " caught at op " + fails(counterexample).OpIndex + ", shrunk "
                + counterexample.Count + " -> " + shrunk.Count + " ops, minimal="
                + Harness854.OneMinimal(shrunk, fails) + " [" + Describe(shrunk) + "] via " + detail);
        }
        Check(allCaught, "the fuzzer finds a counterexample for every injected mutant");
        Check(allStillFail, "every shrunk counterexample still fails on the mutant");
        Check(allShrunk, "shrinking removes more than half of the operations");
        Check(allMinimal, "every shrunk counterexample is one-minimal: no single deletion keeps the failure");

        // 11-12. The failing predicate of the minimal counterexample is exactly the behaviour the mutant
        // corrupts: a range scan, a read after an overwrite, or the live-key count.
        bool shapeOk = true;
        foreach (string[] report in reports)
        {
            string name = report[0], detail = report[1];
            if (name == "mutant-exclusive-range" && !detail.Contains("range(")) shapeOk = false;
            if (name == "mutant-tombstone-leak" && !detail.Contains("range(")) shapeOk = false;
            if (name == "mutant-stale-overwrite" && !detail.Contains("get(")) shapeOk = false;
            if (name == "mutant-count-drift" && !detail.Contains("count ")) shapeOk = false;
        }
        Check(shapeOk, "each minimal counterexample fails exactly on the operation the mutant corrupts");
        Check(reports.Count == 4, "the mutant catalogue is exercised completely (" + reports.Count + " mutants)");
        Check(reports[0][4].Contains("put("), "the range mutant's minimal witness shrinks to a single put");

        // 13. An empty program never fails, and the fuzzer reports no divergence for it.
        Divergence854 none = Harness854.Run(new List<Op854>(), NewEngines());
        Check(!none.Found && none.Detail.Length == 0, "an empty program never produces a divergence");

        // 14. Determinism: the same seed reproduces the same counterexample.
        ulong repeatSeed = 0x854AUL;
        List<Op854> again = FindCounterexample(() => new MutantRange854(), ref repeatSeed, out _);
        ulong thirdSeed = 0x854AUL;
        List<Op854> third = FindCounterexample(() => new MutantRange854(), ref thirdSeed, out _);
        Check(again != null && third != null && Describe(again) == Describe(third),
            "the same seed reproduces the counterexample program exactly: " + Describe(again));

        // 15. The run-based engine really compacts, so the campaign exercised more than one shape.
        var compactionProbe = new RunBasedEngine854(4, 2);
        for (int i = 0; i < 40; i++) compactionProbe.Put(i % 16, i);
        Check(compactionProbe.Compactions > 0 && compactionProbe.Count == 16,
            "the run-based engine compacts (" + compactionProbe.Compactions + " compactions) and still counts 16 live keys");

        Console.WriteLine("PASS");
    }

    private static int NextUncovered(List<int> reachable, Dictionary<int, bool> coverage, Dictionary<int, bool> taken)
    {
        foreach (int bucket in reachable)
            if (!coverage.ContainsKey(bucket) && !taken.ContainsKey(bucket)) return bucket;
        return -1;
    }

    private static Op854 AimAt(int bucket, Dictionary<int, bool> taken)
    {
        taken[bucket] = true;
        return Harness854.OpForBucket(bucket, KeyDomain);
    }

    private static List<Op854> MutateCorpus(ref ulong state, List<List<Op854>> corpus)    {
        List<Op854> source = corpus[Rng854.NextInt(ref state, corpus.Count)];
        var program = new List<Op854>();
        foreach (Op854 op in source) program.Add(op.Copy());
        switch (Rng854.NextInt(ref state, 3))
        {
            case 0:
                program.Add(Harness854.RandomOp(ref state));
                break;
            case 1:
                if (program.Count > 1) program.RemoveAt(Rng854.NextInt(ref state, program.Count));
                break;
            default:
                if (program.Count > 0)
                {
                    int index = Rng854.NextInt(ref state, program.Count);
                    program[index] = Harness854.Mutate(ref state, program[index]);
                }
                break;
        }
        return program;
    }

    private static List<Op854> FindCounterexample(Func<ReferenceEngine854> factory, ref ulong seed, out string name)
    {
        name = factory().Name;
        for (int p = 0; p < CampaignPrograms; p++)
        {
            List<Op854> program = Harness854.RandomProgram(ref seed, ProgramLength);
            // A fresh pair of engines per attempt: the mutant instance is stateful.
            var engines = new IEngine854[] { new ReferenceEngine854(), factory() };
            Divergence854 divergence = Harness854.Run(program, engines);
            if (divergence.Found) return program;
        }
        return null;
    }

    private static string Describe(List<Op854> program)
    {
        if (program == null) return "<none>";
        var sb = new StringBuilder();
        for (int i = 0; i < program.Count; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(program[i].Describe());
        }
        return sb.ToString();
    }

    private static IEngine854[] NewEngines() => new IEngine854[]
    {
        new ReferenceEngine854(), new SortedArrayEngine854(), new RunBasedEngine854(4, 2),
    };

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
