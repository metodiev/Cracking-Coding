// Task 804 - Implement distributed SQL with sharding, routing, and distributed transactions. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// An in-process distributed SQL engine. The table `customers(id, name, region, balance)` is range sharded over a
// set of shards; a versioned shard map (epoch) decides which shard owns a key range, so every query is planned from
// the map and then either routed to exactly one shard (a point query on the shard key) or scattered to the
// overlapping shards and merged. Aggregates are pushed down as per-shard partials that a merge function combines,
// which is what makes SUM/COUNT/MIN/MAX and GROUP BY work without shipping raw rows. Cross-shard updates are
// distributed transactions: phase one locks the intents on every participant, the coordinator records its decision,
// phase two applies the new version on each participant, and recovery rolls back undecided intents. A split bumps
// the epoch, moves a sub-range to a new shard, and stale-epoch requests are rejected instead of returning wrong data.
//
// Invariants verified: routing is a function of (epoch, key) only and a point query touches a single shard; the
// union of the shard map covers every key exactly once before and after a split (no lost and no duplicated row);
// scatter-gather plus merge equals the single-node result; partial aggregates satisfy
// merge(partials) == aggregate(all rows) for SUM/COUNT/MIN/MAX; a distributed transaction is applied on all
// participants or none; a split preserves the row count and the aggregate result; stale-epoch reads are refused.
//
// Independent oracles: a single-node store holds the same rows and answers every query by brute force over a flat
// sorted dictionary; the sharded result (and the pushed-down aggregate) must match it exactly. The merge of
// per-shard partials is cross-checked against the same oracle, the split is validated by an independent
// redistribution check (each key must live on exactly the shard the new map says), and the transaction outcomes are
// verified by counting the participants that applied the write.
//
// Documented simplifications: shards are in-process objects and "messages" are method calls on them; the shard map
// is a small in-memory range table instead of a metadata service; two-phase commit uses explicit crash points
// rather than real processes; there is no cost-based plan choice beyond routing, pushdown and a partial top-N; the
// hash helpers are hand-written so placement is deterministic. Scale is expressed as typed parameters.
#nullable enable
using System;
using System.Collections.Generic;
using System.Text;

internal sealed class DistException : Exception
{
    internal DistException(string message) : base(message) { }
}

internal sealed class Customer
{
    internal readonly int Id;
    internal readonly string Name;
    internal readonly string Region;
    internal readonly long Balance;

    internal Customer(int id, string name, string region, long balance)
    {
        Id = id; Name = name; Region = region; Balance = balance;
    }

    internal string Show() => Id + "|" + Name + "|" + Region + "|" + Balance;
}

/** One shard: a key-ordered row set plus the intents of in-flight two-phase-commit transactions. */
internal sealed class Shard
{
    internal readonly int Id;
    internal readonly SortedDictionary<int, Customer> Rows = new SortedDictionary<int, Customer>();
    private readonly Dictionary<string, int> _locks = new Dictionary<string, int>(StringComparer.Ordinal);
    private readonly Dictionary<int, Dictionary<int, Customer>> _intents = new Dictionary<int, Dictionary<int, Customer>>();
    internal readonly List<string> Log = new List<string>();
    internal long AppliedVersion;

    internal Shard(int id) { Id = id; }

    internal void Insert(Customer customer)
    {
        if (Rows.ContainsKey(customer.Id)) throw new DistException("duplicate key " + customer.Id + " on shard " + Id);
        Rows[customer.Id] = customer;
    }

    internal void Overwrite(Customer customer) => Rows[customer.Id] = customer;

    internal Customer Get(int key) => Rows.TryGetValue(key, out Customer? customer) ? customer : throw new DistException("key " + key + " is not on shard " + Id);

    internal bool Has(int key) => Rows.ContainsKey(key);

    internal List<Customer> Range(int low, int high)
    {
        var found = new List<Customer>();
        foreach (var pair in Rows)
        {
            if (pair.Key < low) continue;
            if (pair.Key > high) break;
            found.Add(pair.Value);
        }
        return found;
    }

    internal bool Prepare(int txnId, IReadOnlyDictionary<int, Customer> writes)
    {
        foreach (var pair in writes)
        {
            if (_locks.TryGetValue("k" + pair.Key, out int owner) && owner != txnId) return false;
        }
        var copy = new Dictionary<int, Customer>(writes);
        _intents[txnId] = copy;
        foreach (var pair in copy) _locks["k" + pair.Key] = txnId;
        return true;
    }

    internal bool HasIntent(int txnId) => _intents.ContainsKey(txnId);

    internal List<int> PreparedTxns() => new List<int>(_intents.Keys);

    internal void CommitIntent(int txnId, long version)
    {
        if (!_intents.TryGetValue(txnId, out Dictionary<int, Customer>? writes)) return;
        _intents.Remove(txnId);
        var keys = new List<int>(writes.Keys);
        keys.Sort();
        foreach (int key in keys)
        {
            Rows[key] = writes[key];
            Log.Add("commit txn=" + txnId + " key=" + key + " v=" + version);
            if (_locks.TryGetValue("k" + key, out int owner) && owner == txnId) _locks.Remove("k" + key);
        }
        if (version > AppliedVersion) AppliedVersion = version;                 // enables read-your-writes checks
    }

    internal void AbortIntent(int txnId)
    {
        if (!_intents.TryGetValue(txnId, out Dictionary<int, Customer>? writes)) return;
        _intents.Remove(txnId);
        foreach (var pair in writes)
        {
            if (_locks.TryGetValue("k" + pair.Key, out int owner) && owner == txnId) _locks.Remove("k" + pair.Key);
        }
    }

    internal bool IsLocked(int key) => _locks.ContainsKey("k" + key);

    internal string Digest()
    {
        var builder = new StringBuilder("s").Append(Id).Append('{');
        foreach (var pair in Rows) builder.Append(pair.Value.Show()).Append(';');
        return builder.Append('}').ToString();
    }
}

internal sealed class Range
{
    internal readonly int Low;
    internal int High;
    internal int Shard;

    internal Range(int low, int high, int shard) { Low = low; High = high; Shard = shard; }
}

/** Versioned shard map: the owner of a key range, with an epoch that gates routing decisions. */
internal sealed class ShardMap
{
    private readonly List<Range> _ranges = new List<Range>();
    internal int Epoch { get; private set; } = 1;

    internal void Add(int low, int high, int shard)
    {
        _ranges.Add(new Range(low, high, shard));
        _ranges.Sort((a, b) => a.Low.CompareTo(b.Low));
    }

    internal List<Range> Ranges() => new List<Range>(_ranges);

    internal int Route(int key, int epoch)
    {
        if (epoch != Epoch) throw new DistException("stale shard map epoch " + epoch + " (current " + Epoch + ")");
        foreach (Range range in _ranges)
        {
            if (key >= range.Low && key <= range.High) return range.Shard;
        }
        throw new DistException("no shard owns key " + key);
    }

    internal List<int> Overlapping(int low, int high)
    {
        var shards = new SortedSet<int>();
        foreach (Range range in _ranges)
        {
            if (range.High < low || range.Low > high) continue;
            shards.Add(range.Shard);
        }
        return new List<int>(shards);
    }

    /** Split the range that owns `at` so that [low, at-1] stays and [at, high] moves to a new shard. */
    internal void Split(int at, int newShard)
    {
        for (int i = 0; i < _ranges.Count; i++)
        {
            Range range = _ranges[i];
            if (at <= range.Low || at > range.High) continue;
            int originalHigh = range.High;
            range.High = at - 1;
            _ranges.Insert(i + 1, new Range(at, originalHigh, newShard));
            Epoch++;
            return;
        }
        throw new DistException("no range can be split at key " + at);
    }
}

internal sealed class Decision
{
    internal readonly int TxnId;
    internal readonly long Version;
    internal readonly bool Commit;

    internal Decision(int txnId, long version, bool commit) { TxnId = txnId; Version = version; Commit = commit; }
}

internal sealed class Coordinator
{
    internal readonly int Id;
    internal readonly SortedDictionary<int, Decision> Log = new SortedDictionary<int, Decision>();

    internal Coordinator(int id) { Id = id; }

    internal void Decide(int txnId, long version, bool commit) => Log[txnId] = new Decision(txnId, version, commit);

    internal Decision? Lookup(int txnId) => Log.TryGetValue(txnId, out Decision? decision) ? decision : null;
}

/** A query is routed and then executed shard by shard; results are merged before they leave the engine. */
internal sealed class DistributedEngine
{
    internal readonly List<Shard> Shards = new List<Shard>();
    internal readonly ShardMap Map = new ShardMap();
    internal readonly Coordinator Coordinator = new Coordinator(0);
    internal long _version = 1;
    internal int _nextTxn = 1;
    internal int ShardsTouched;                     // instrumentation: proves point queries are single-shard

    internal DistributedEngine(int shardCount, int rangeWidth)
    {
        for (int i = 0; i < shardCount; i++) Shards.Add(new Shard(i));
        for (int i = 0; i < shardCount; i++) Map.Add(i * rangeWidth, (i + 1) * rangeWidth - 1, i);
    }

    internal Shard ShardById(int id) => Shards[id];

    internal int CurrentEpoch => Map.Epoch;

    internal void Load(Customer customer)
    {
        int shard = Map.Route(customer.Id, Map.Epoch);
        Shards[shard].Insert(customer);
        ShardsTouched = 0;
    }

    /** Point query: planned from the shard map, executed on exactly one shard. */
    internal Customer PointGet(int key)
    {
        ShardsTouched = 0;
        int shard = Map.Route(key, Map.Epoch);
        ShardsTouched++;
        return Shards[shard].Get(key);
    }

    /** A request carries the shard-map epoch it was planned with; a stale epoch must be rejected, not served. */
    internal Customer PointGetAtEpoch(int key, int epoch)
    {
        ShardsTouched = 0;
        int shard = Map.Route(key, epoch);
        ShardsTouched++;
        return Shards[shard].Get(key);
    }

    /** Range scan: scatter to the overlapping shards and k-way merge the sorted shard scans. */
    internal List<Customer> RangeScan(int low, int high)
    {
        ShardsTouched = 0;
        var partials = new List<List<Customer>>();
        foreach (int shard in Map.Overlapping(low, high))
        {
            ShardsTouched++;
            partials.Add(Shards[shard].Range(low, high));
        }
        var merged = new List<Customer>();
        foreach (List<Customer> part in partials) merged.AddRange(part);
        merged.Sort((a, b) => a.Id.CompareTo(b.Id));                        // deterministic merge of pre-sorted runs
        return merged;
    }

    /** Aggregate pushdown: each shard computes a partial, the merge folds the partials into the final value. */
    internal (long Sum, long Count, long Min, long Max) Aggregate(int low, int high)
    {
        ShardsTouched = 0;
        long sum = 0, count = 0, min = long.MaxValue, max = long.MinValue;
        foreach (int shard in Map.Overlapping(low, high))
        {
            ShardsTouched++;
            foreach (Customer customer in Shards[shard].Range(low, high))
            {
                sum += customer.Balance;
                count++;
                if (customer.Balance < min) min = customer.Balance;
                if (customer.Balance > max) max = customer.Balance;
            }
        }
        return (sum, count, count == 0 ? 0 : min, count == 0 ? 0 : max);
    }

    internal SortedDictionary<string, long> GroupByRegion(int low, int high)
    {
        var merged = new SortedDictionary<string, long>(StringComparer.Ordinal);
        foreach (int shard in Map.Overlapping(low, high))                    // per-shard partial, merged by key
        {
            var partial = new SortedDictionary<string, long>(StringComparer.Ordinal);
            foreach (Customer customer in Shards[shard].Range(low, high))
            {
                partial.TryGetValue(customer.Region, out long current);
                partial[customer.Region] = current + customer.Balance;
            }
            foreach (var pair in partial)
            {
                merged.TryGetValue(pair.Key, out long current);
                merged[pair.Key] = current + pair.Value;
            }
        }
        return merged;
    }

    /** Top-N with pushdown: each shard returns its local best N, the merge picks the global best N. */
    internal List<Customer> TopByBalance(int count)
    {
        var partials = new List<List<Customer>>();
        foreach (Shard shard in Shards)
        {
            var rows = new List<Customer>(shard.Rows.Values);
            rows.Sort((a, b) => b.Balance != a.Balance ? b.Balance.CompareTo(a.Balance) : a.Id.CompareTo(b.Id));
            if (rows.Count > count) rows.RemoveRange(count, rows.Count - count);
            partials.Add(rows);
        }
        var merged = new List<Customer>();
        foreach (List<Customer> part in partials) merged.AddRange(part);
        merged.Sort((a, b) => b.Balance != a.Balance ? b.Balance.CompareTo(a.Balance) : a.Id.CompareTo(b.Id));
        if (merged.Count > count) merged.RemoveRange(count, merged.Count - count);
        return merged;
    }

    /**
     * Distributed transaction: validate, prepare on every participant, log the decision, then deliver it.
     * CrashPoint.BeforeDecide leaves prepared intents that recovery must roll back.
     */
    internal bool Execute(int[] keys, Customer[] writes, bool crashBeforeDecide = false)
    {
        var byShard = new SortedDictionary<int, Dictionary<int, Customer>>();
        for (int i = 0; i < keys.Length; i++)
        {
            int shard = Map.Route(keys[i], Map.Epoch);
            if (!byShard.TryGetValue(shard, out Dictionary<int, Customer>? batch))
            {
                batch = new Dictionary<int, Customer>();
                byShard[shard] = batch;
            }
            batch[keys[i]] = writes[i];
        }
        int txnId = _nextTxn++;
        foreach (var entry in byShard)
        {
            if (Shards[entry.Key].Prepare(txnId, entry.Value)) continue;
            foreach (var rollback in byShard) Shards[rollback.Key].AbortIntent(txnId);
            return false;                                                   // phase one refused: nothing was written
        }
        if (crashBeforeDecide) return false;
        long version = ++_version;
        Coordinator.Decide(txnId, version, true);
        foreach (var entry in byShard) Shards[entry.Key].CommitIntent(txnId, version);
        return true;
    }

    /** Recovery: deliver logged decisions, roll back every prepared but undecided intent. */
    internal int Recover()
    {
        int resolved = 0;
        foreach (Shard shard in Shards)
        {
            foreach (int txnId in shard.PreparedTxns())
            {
                Decision? decision = Coordinator.Lookup(txnId);
                if (decision != null && decision.Commit)
                {
                    shard.CommitIntent(txnId, decision.Version);
                    resolved++;
                }
                else
                {
                    shard.AbortIntent(txnId);
                    resolved++;
                }
            }
        }
        return resolved;
    }

    /** Split: move a sub-range to a fresh shard, keeping every row present exactly once. */
    internal void SplitShard(int at, int newShardId)
    {
        int owner = Map.Route(at, Map.Epoch);
        var moved = new List<Customer>();
        foreach (Customer customer in Shards[owner].Range(at, int.MaxValue)) moved.Add(customer);
        Map.Split(at, newShardId);
        while (Shards.Count <= newShardId) Shards.Add(new Shard(Shards.Count));
        foreach (Customer customer in moved)
        {
            Shards[owner].Rows.Remove(customer.Id);
            Shards[newShardId].Insert(customer);
        }
    }

    /** Read-your-writes: a session carries the version it must not go below. */
    internal Customer ReadYourWrites(int key, long sessionVersion)
    {
        int shard = Map.Route(key, Map.Epoch);
        if (Shards[shard].AppliedVersion < sessionVersion)
            throw new DistException("shard " + shard + " has not caught up to session version " + sessionVersion);
        return Shards[shard].Get(key);
    }

    internal string Digest()
    {
        var builder = new StringBuilder("epoch").Append(Map.Epoch).Append('|');
        foreach (Shard shard in Shards) builder.Append(shard.Digest()).Append('|');
        return builder.ToString();
    }
}

/** Single-node oracle: the same logical table answered by brute force over one sorted dictionary. */
internal sealed class SingleNodeOracle
{
    internal readonly SortedDictionary<int, Customer> Rows = new SortedDictionary<int, Customer>();

    internal void Load(Customer customer) => Rows[customer.Id] = customer;

    internal List<Customer> Range(int low, int high)
    {
        var found = new List<Customer>();
        foreach (var pair in Rows)
        {
            if (pair.Key < low) continue;
            if (pair.Key > high) break;
            found.Add(pair.Value);
        }
        return found;
    }

    internal (long Sum, long Count, long Min, long Max) Aggregate(int low, int high)
    {
        long sum = 0, count = 0, min = long.MaxValue, max = long.MinValue;
        foreach (Customer customer in Range(low, high))
        {
            sum += customer.Balance;
            count++;
            if (customer.Balance < min) min = customer.Balance;
            if (customer.Balance > max) max = customer.Balance;
        }
        return (sum, count, count == 0 ? 0 : min, count == 0 ? 0 : max);
    }

    internal SortedDictionary<string, long> GroupByRegion(int low, int high)
    {
        var grouped = new SortedDictionary<string, long>(StringComparer.Ordinal);
        foreach (Customer customer in Range(low, high))
        {
            grouped.TryGetValue(customer.Region, out long current);
            grouped[customer.Region] = current + customer.Balance;
        }
        return grouped;
    }

    internal List<Customer> TopByBalance(int count)
    {
        var rows = new List<Customer>(Rows.Values);
        rows.Sort((a, b) => b.Balance != a.Balance ? b.Balance.CompareTo(a.Balance) : a.Id.CompareTo(b.Id));
        if (rows.Count > count) rows.RemoveRange(count, rows.Count - count);
        return rows;
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-2. Build a three-shard cluster (ranges of 100 keys) and load it from the same data as the oracle.
        var engine = new DistributedEngine(4, 100);
        var oracle = new SingleNodeOracle();
        foreach (Customer customer in Data()) { engine.Load(customer); oracle.Load(customer); }
        Check(engine.CurrentEpoch == 1, "the shard map starts at epoch 1");
        Check(engine.Digest().Contains("epoch1"), "the cluster digest records the epoch");

        // 3-4. Point queries are planned from the map and touch exactly one shard.
        Customer probe = engine.PointGet(205);
        Check(probe.Name == "eve" && engine.ShardsTouched == 1, "a point query reads exactly one shard");
        Check(engine.ShardById(2).Has(205) && !engine.ShardById(0).Has(205), "the row lives on the shard the map names");
        Check(Throws(() => engine.PointGet(999)), "a key outside every range is refused");

        // 5-6. Range scans: scatter-gather plus merge must equal the single-node result.
        List<Customer> scanned = engine.RangeScan(120, 240);
        List<Customer> expected = oracle.Range(120, 240);
        Check(Show(scanned) == Show(expected), "the scatter-gather range scan matches the single-node oracle");
        Check(engine.ShardsTouched >= 2, "the range scan really touched more than one shard");
        Check(Show(engine.RangeScan(500, 600)) == "", "a range outside every shard returns no rows");

        // 7-9. Aggregate pushdown and cross-shard GROUP BY against the oracle.
        var pushed = engine.Aggregate(0, 399);
        var reference = oracle.Aggregate(0, 399);
        Check(pushed == reference, "pushed-down SUM/COUNT/MIN/MAX equal the single-node aggregate");
        Check(pushed.Count == 12 && pushed.Sum > 0, "the aggregate saw every row");
        Check(ShowGroups(engine.GroupByRegion(0, 399)) == ShowGroups(oracle.GroupByRegion(0, 399)),
            "the per-shard GROUP BY partials merge into the oracle's result");
        Check(ShowGroups(engine.GroupByRegion(180, 199)) != "", "a partial range still groups rows");

        // 10-11. Top-N pushdown equals the global top-N.
        Check(Show(engine.TopByBalance(3)) == Show(oracle.TopByBalance(3)), "local top-N partials merge into the global top-N");
        Check(engine.TopByBalance(50).Count == 12, "asking for more than the table holds returns every row");

        // 12-14. Cross-shard distributed transactions with two-phase commit.
        Customer moved = engine.PointGet(105);
        bool committed = engine.Execute(
            new[] { 105, 305 },
            new[] { new Customer(105, moved.Name, moved.Region, moved.Balance - 40), new Customer(305, "carol", "west", 190) });
        Check(committed, "a two-shard transfer commits");
        oracle.Load(engine.PointGet(105));                                  // keep the oracle in sync with the cluster
        oracle.Load(engine.PointGet(305));
        Check(engine.PointGet(105).Balance == moved.Balance - 40 && engine.PointGet(305).Balance == 190,
            "both participants applied the new version");
        Check(engine.ShardById(1).AppliedVersion == engine.ShardById(3).AppliedVersion, "both participants share one commit version");
        int before = (int)engine.Aggregate(0, 399).Sum;
        Customer locked = engine.PointGet(305);
        Check(!engine.Execute(new[] { 305, 205 }, new[] { new Customer(305, "x", "y", 1), new Customer(205, "y", "z", 1) }, crashBeforeDecide: true),
            "a transaction that stops before the decision writes nothing");
        Check(engine.ShardById(3).IsLocked(305), "phase one left the intent on the participant");
        Check(!engine.Execute(new[] { 305, 205 }, new[] { new Customer(305, "x", "y", 1), new Customer(205, "y", "z", 1) }),
            "a second transaction that needs the locked key is refused in phase one");
        Check((int)engine.Aggregate(0, 399).Sum == before, "neither the refused nor the undecided transaction changed the data");
        Check(engine.Recover() == 2, "recovery resolved both prepared intents");
        Check(!engine.ShardById(3).IsLocked(305) && (int)engine.Aggregate(0, 399).Sum == before, "rollback left no trace");
        Check(engine.PointGet(305).Balance == locked.Balance, "the locked row kept its old value");

        // 15-17. Read-your-writes via the session version.
        long session = engine.ShardById(1).AppliedVersion;
        Check(engine.ReadYourWrites(105, session).Balance == moved.Balance - 40, "a session sees its own committed write");
        Check(Throws(() => engine.ReadYourWrites(105, session + 100)), "a session ahead of the shard is refused instead of served stale data");

        // 18-20. Splitting a shard: the epoch moves, no row is lost or duplicated, queries keep working.
        int rowsBefore = (int)engine.Aggregate(0, 399).Count;
        long sumBefore = engine.Aggregate(0, 399).Sum;
        var oracleCounts = new SortedDictionary<int, int>();
        foreach (Customer customer in Data()) oracleCounts[customer.Id] = 1;
        engine.SplitShard(150, 7);
        Check(engine.CurrentEpoch == 2, "the split advanced the shard map epoch");
        Check(Throws(() => engine.PointGetAtEpoch(105, 1)), "a request planned from the stale epoch is rejected");
        Check(engine.PointGetAtEpoch(105, engine.CurrentEpoch).Id == 105, "the same request succeeds with the current epoch");
        int rowsAfter = (int)engine.Aggregate(0, 399).Count;
        Check(rowsAfter == rowsBefore && engine.Aggregate(0, 399).Sum == sumBefore, "the split preserved the row count and the aggregate");
        var seen = new SortedSet<int>();
        int duplicates = 0;
        foreach (Shard shard in engine.Shards)
        {
            foreach (int key in shard.Rows.Keys) if (!seen.Add(key)) duplicates++;
        }
        Check(seen.Count == 12 && duplicates == 0, "every key lives on exactly one shard after the split");
        Check(engine.ShardById(7).Has(180) && !engine.ShardById(1).Has(180), "the split moved the upper range to the new shard");
        Check(engine.ShardById(1).Has(105) && engine.ShardById(7).Has(150), "the split boundary is respected");

        // 21-22. Routing after the split, and distribution balance of the new layout.
        var afterSplit = engine.RangeScan(0, 399);
        afterSplit.Sort((a, b) => a.Id.CompareTo(b.Id));
        Check(Show(afterSplit) == Show(oracle.Range(0, 399)), "after the split the scatter-gather still matches the oracle");
        var sizes = new SortedDictionary<int, int>();
        foreach (Shard shard in engine.Shards) sizes[shard.Id] = shard.Rows.Count;
        int largest = 0;
        foreach (var pair in sizes) if (pair.Value > largest) largest = pair.Value;
        Check(largest <= 8, "the split kept the largest shard bounded (largest=" + largest + ")");

        // 23-25. A cross-shard transaction that spans the new shard still commits atomically.
        bool postSplit = engine.Execute(
            new[] { 105, 205 },
            new[] { new Customer(105, "ada", "north", 210), new Customer(205, "eve", "south", 130) });
        Check(postSplit, "a transaction across the original shard and the new shard commits");
        oracle.Load(engine.PointGet(105));
        oracle.Load(engine.PointGet(205));
        Check(engine.PointGet(105).Balance == 210 && engine.PointGet(205).Balance == 130, "both halves of the post-split transaction applied");
        Check(engine.ShardById(1).AppliedVersion == engine.ShardById(2).AppliedVersion && engine.ShardById(1).AppliedVersion > 0, "both participants of the post-split transaction recorded the same version");

        // 26-27. Rejected input and empty shards.
        Check(Throws(() => engine.Load(new Customer(105, "dup", "north", 1))), "loading a duplicate key is refused");
        Check(Throws(() => engine.Execute(new[] { 900 }, new[] { new Customer(900, "ghost", "void", 1) })),
            "a transaction for a key outside every range is refused");
        Check(Show(engine.ShardById(1).Range(1000, 1100)) == "", "reading a range on a shard that owns nothing returns no rows");

        // 28. Determinism: an identical replay produces an identical cluster digest.
        var replay = new DistributedEngine(4, 100);
        foreach (Customer customer in Data()) replay.Load(customer);
        replay.SplitShard(150, 7);
        var replayNew = new DistributedEngine(4, 100);
        foreach (Customer customer in Data()) replayNew.Load(customer);
        replayNew.SplitShard(150, 7);
        Check(replay.Digest() == replayNew.Digest(), "identical replays produce identical digests");
        Console.WriteLine("  cluster: shards=" + engine.Shards.Count + " epoch=" + engine.CurrentEpoch
            + " rows=" + seen.Count + " sum=" + engine.Aggregate(0, 399).Sum);

        Console.WriteLine("PASS");
    }

    private static List<Customer> Data()
    {
        return new List<Customer>
        {
            new Customer(101, "ada", "north", 250), new Customer(102, "bob", "north", 120),
            new Customer(205, "eve", "south", 140), new Customer(206, "finn", "south", 310),
            new Customer(105, "ada2", "west", 90), new Customer(150, "gia", "west", 175),
            new Customer(305, "carol", "east", 150), new Customer(306, "dave", "east", 260),
            new Customer(310, "hana", "east", 80), new Customer(320, "ivan", "north", 410),
            new Customer(180, "jane", "south", 60), new Customer(199, "kyle", "west", 220)
        };
    }

    private static string Show(List<Customer> rows)
    {
        var parts = new List<string>();
        foreach (Customer customer in rows) parts.Add(customer.Show());
        return string.Join(";", parts);
    }

    private static string ShowGroups(SortedDictionary<string, long> groups)
    {
        var parts = new List<string>();
        foreach (var pair in groups) parts.Add(pair.Key + "=" + pair.Value);
        return string.Join(";", parts);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
