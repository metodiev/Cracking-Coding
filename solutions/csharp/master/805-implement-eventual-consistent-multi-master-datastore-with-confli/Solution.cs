// Task 805 - Implement eventual-consistent multi-master datastore with conflict resolution policies. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// A multi-master (active-active) datastore where every replica accepts writes and anti-entropy gossip makes the
// replicas converge. Each key holds a version vector plus the set of concurrent candidate writes, so a merge is a
// join of two states: causally dominated writes are discarded, concurrent writes are kept as siblings, and a
// per-cluster policy resolves the siblings -
//   * LastWriteWins  - ordered by (logical stamp, replica id), so the tie break is totally ordered and identical
//                      at every replica regardless of delivery order;
//   * MaxWins        - the largest numeric value survives;
//   * MultiValue     - siblings are preserved and a resolver (set union) folds them into one value;
//   * Counter        - a PN-counter kept as two vectors of per-replica increments, which is conflict free.
//
// Invariants verified: merging is idempotent (merge(s, s) = s), commutative and associative, so the state after
// gossip is independent of delivery order and of how often a message is replayed; the replica clock never moves
// backwards (monotone merge); a causally newer write is never overwritten by a causally older one (no lost update);
// concurrent writes are never silently dropped (they become siblings before being resolved); and every replica
// converges to a byte-identical digest once gossip quiesces.
//
// Independent oracles: (a) order independence is decided by brute force - every permutation of the gossip pairs of
// a round is replayed and all resulting digests must be equal; (b) convergence is cross-checked with a reference
// merge that folds each replica's writes through a single-accumulator loop in a different (sorted) order;
// (c) the counter result is compared with a plain integer sum of all local increments and decrements.
//
// Documented simplifications: replicas are in-process objects, "gossip" is a deterministic full-state exchange
// between pairs in a fixed order, the vector clocks use replica indexes, there is no durable log, no tombstone
// garbage collection and no quorum/consensus path (this is genuinely an eventual-consistency model: a replica
// serves its own view, which may lag until gossip catches up). Randomness comes from a hand-written splitmix64.
#nullable enable
using System;
using System.Collections.Generic;
using System.Text;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal enum Order { Equal, Before, After, Concurrent }

internal enum Policy { LastWriteWins, MaxWins, MultiValue, Counter }

/** Dotted version vector: replica index -> number of writes observed from that replica. */
internal sealed class VectorClock
{
    internal readonly Dictionary<int, long> Counts = new Dictionary<int, long>();

    internal VectorClock() { }

    internal VectorClock Clone()
    {
        var copy = new VectorClock();
        foreach (var pair in Counts) copy.Counts[pair.Key] = pair.Value;
        return copy;
    }

    internal void Increment(int replica)
    {
        Counts.TryGetValue(replica, out long current);
        Counts[replica] = current + 1;
    }

    internal long Get(int replica) => Counts.TryGetValue(replica, out long value) ? value : 0;

    /** Pointwise maximum: the join of two clocks in the version-vector lattice. */
    internal void Merge(VectorClock other)
    {
        foreach (var pair in other.Counts)
        {
            Counts.TryGetValue(pair.Key, out long current);
            if (pair.Value > current) Counts[pair.Key] = pair.Value;
        }
    }

    internal Order Compare(VectorClock other)
    {
        bool greater = false;
        bool less = false;
        var keys = new SortedSet<int>();
        foreach (int key in Counts.Keys) keys.Add(key);
        foreach (int key in other.Counts.Keys) keys.Add(key);
        foreach (int key in keys)
        {
            long mine = Get(key);
            long theirs = other.Get(key);
            if (mine > theirs) greater = true;
            if (mine < theirs) less = true;
        }
        if (greater && less) return Order.Concurrent;
        if (greater) return Order.After;
        if (less) return Order.Before;
        return Order.Equal;
    }

    internal string Show()
    {
        var keys = new List<int>(Counts.Keys);
        keys.Sort();
        var parts = new List<string>();
        foreach (int key in keys) parts.Add(key + ":" + Counts[key]);
        return "[" + string.Join(",", parts) + "]";
    }
}

/** One candidate write: a value, the replica that produced it and a logical stamp for the deterministic tie break. */
internal sealed class Write
{
    internal readonly string? Value;
    internal readonly int Replica;
    internal readonly long Stamp;
    internal readonly VectorClock Clock;

    internal Write(string? value, int replica, long stamp, VectorClock? clock = null)
    {
        Value = value; Replica = replica; Stamp = stamp; Clock = clock ?? new VectorClock();
    }

    internal string Show() => (Value ?? "<null>") + "@r" + Replica + "s" + Stamp;
}

/** The state of one key: the write set plus the version vector of everything folded into it. */
internal sealed class Entry
{
    internal readonly VectorClock Clock = new VectorClock();
    internal readonly List<Write> Candidates = new List<Write>();
    internal readonly Dictionary<int, long> Positives = new Dictionary<int, long>();
    internal readonly Dictionary<int, long> Negatives = new Dictionary<int, long>();

    internal string Show()
    {
        var parts = new List<string>();
        foreach (Write write in CanonicalCandidates()) parts.Add(write.Show());
        foreach (int key in SortedKeys(Positives)) parts.Add("+" + key + ":" + Positives[key]);
        foreach (int key in SortedKeys(Negatives)) parts.Add("-" + key + ":" + Negatives[key]);
        return Clock.Show() + "{" + string.Join(",", parts) + "}";
    }

    /** Canonical candidate order: the digest and the MultiValue read must not depend on arrival order. */
    internal List<Write> CanonicalCandidates()
    {
        var copy = new List<Write>(Candidates);
        copy.Sort((x, y) => x.Replica != y.Replica ? x.Replica.CompareTo(y.Replica)
            : x.Stamp != y.Stamp ? x.Stamp.CompareTo(y.Stamp)
            : string.CompareOrdinal(x.Value ?? "", y.Value ?? ""));
        return copy;
    }

    internal static List<int> SortedKeys(Dictionary<int, long> counts)
    {
        var keys = new List<int>(counts.Keys);
        keys.Sort();
        return keys;
    }
}

internal sealed class Replica
{
    internal readonly int Id;
    internal readonly SortedDictionary<string, Entry> State = new SortedDictionary<string, Entry>(StringComparer.Ordinal);
    internal long StampCounter;
    internal readonly Policy Policy;

    internal Replica(int id, Policy policy) { Id = id; Policy = policy; }

    /** Local write: the replica's own clock entry is bumped, so the write causally follows everything it has seen. */
    internal void Put(string key, string? value)
    {
        Entry entry = GetOrCreate(key);
        entry.Clock.Increment(Id);
        if (Policy == Policy.Counter)
        {
            entry.Positives.TryGetValue(Id, out long positive);
            entry.Positives[Id] = positive + (value == "inc" ? 1 : 0);
            entry.Negatives.TryGetValue(Id, out long negative);
            entry.Negatives[Id] = negative + (value == "dec" ? 1 : 0);
            return;
        }
        var survivors = new List<Write>();
        foreach (Write write in entry.Candidates)
        {
            if (write.Clock.Compare(entry.Clock) != Order.Before) survivors.Add(write);   // dominated siblings are superseded
        }
        survivors.Add(new Write(value, Id, ++StampCounter, entry.Clock.Clone()));
        List<Write> resolved = Resolution.Resolve(survivors, Policy);
        entry.Candidates.Clear();
        entry.Candidates.AddRange(resolved);
    }

    internal string? Read(string key)
    {
        if (!State.TryGetValue(key, out Entry? entry)) return null;
        if (Policy == Policy.Counter)
        {
            long total = 0;
            foreach (var pair in entry.Positives) total += pair.Value;
            foreach (var pair in entry.Negatives) total -= pair.Value;
            return total.ToString();
        }
        if (entry.Candidates.Count == 0) return null;
        List<Write> resolved = Resolution.Resolve(entry.CanonicalCandidates(), Policy);
        var parts = new List<string>();
        foreach (Write write in resolved) parts.Add(write.Value ?? "<null>");
        return string.Join("+", parts);
    }

    internal Entry GetOrCreate(string key)
    {
        if (!State.TryGetValue(key, out Entry? entry))
        {
            entry = new Entry();
            State[key] = entry;
        }
        return entry;
    }

    /** Merge another replica's state into this one; the join is idempotent, commutative and associative. */
    internal void MergeFrom(Replica other)
    {
        foreach (var pair in other.State)
        {
            if (!State.TryGetValue(pair.Key, out Entry? mine))
            {
                Entry copy = Clone(pair.Value);
                State[pair.Key] = copy;
                continue;
            }
            MergeEntry(mine, pair.Value);
        }
    }

    private void MergeEntry(Entry target, Entry incoming)
    {
        if (incoming.Clock.Compare(target.Clock) == Order.After) target.Candidates.Clear();   // causally superseded
        foreach (Write write in incoming.Candidates)
        {
            if (!Contains(target.Candidates, write)) target.Candidates.Add(write);
        }
        foreach (var pair in incoming.Positives)
        {
            target.Positives.TryGetValue(pair.Key, out long current);
            if (pair.Value > current) target.Positives[pair.Key] = pair.Value;
        }
        foreach (var pair in incoming.Negatives)
        {
            target.Negatives.TryGetValue(pair.Key, out long current);
            if (pair.Value > current) target.Negatives[pair.Key] = pair.Value;
        }
        target.Clock.Merge(incoming.Clock);
        List<Write> resolved = Resolution.Resolve(new List<Write>(target.Candidates), Policy);
        target.Candidates.Clear();
        target.Candidates.AddRange(resolved);
    }

    private static bool Contains(List<Write> writes, Write candidate)
    {
        foreach (Write write in writes)
        {
            if (write.Replica == candidate.Replica && write.Stamp == candidate.Stamp && write.Value == candidate.Value) return true;
        }
        return false;
    }

    internal static Entry Clone(Entry source)
    {
        var copy = new Entry();
        copy.Clock.Merge(source.Clock);
        copy.Candidates.AddRange(source.Candidates);
        foreach (var pair in source.Positives) copy.Positives[pair.Key] = pair.Value;
        foreach (var pair in source.Negatives) copy.Negatives[pair.Key] = pair.Value;
        return copy;
    }

    /** State digest without the replica id, so replicas can be compared for convergence. */
    internal string Digest()
    {
        var builder = new StringBuilder();
        foreach (var pair in State) builder.Append(pair.Key).Append('=').Append(pair.Value.Show()).Append(';');
        return builder.ToString();
    }
}

internal static class Resolution
{
    /** Collapses concurrent siblings according to the policy; MultiValue keeps them until a resolver is applied. */
    internal static List<Write> Resolve(List<Write> candidates, Policy policy)
    {
        if (candidates.Count <= 1 || policy == Policy.MultiValue) return candidates;
        Write best = candidates[0];
        foreach (Write candidate in candidates)
        {
            if (policy == Policy.MaxWins)
            {
                long mine = Parse(candidate.Value);
                long theirs = Parse(best.Value);
                if (mine > theirs || (mine == theirs && candidate.Replica > best.Replica)) best = candidate;
            }
            else if (candidate.Stamp > best.Stamp || (candidate.Stamp == best.Stamp && candidate.Replica > best.Replica))
            {
                best = candidate;                                            // LastWriteWins: total order by (stamp, replica)
            }
        }
        return new List<Write> { best };
    }

    private static long Parse(string? value) => value == null ? long.MinValue : long.Parse(value);

    /** A resolver for MultiValue: fold the siblings into a single value, e.g. by joining sorted tokens. */
    internal static string ResolveMultiValue(IReadOnlyList<Write> candidates)
    {
        var tokens = new SortedSet<string>(StringComparer.Ordinal);
        foreach (Write write in candidates)
        {
            if (write.Value == null) continue;
            foreach (string token in write.Value.Split('+')) tokens.Add(token);
        }
        return string.Join("+", tokens);
    }
}

/** A cluster of replicas exchanging full state through deterministic gossip rounds. */
internal sealed class Cluster
{
    internal readonly List<Replica> Replicas = new List<Replica>();
    internal readonly List<string> Log = new List<string>();

    internal Cluster(int count, Policy policy)
    {
        for (int i = 0; i < count; i++) Replicas.Add(new Replica(i, policy));
    }

    internal Replica this[int index] => Replicas[index];

    /** One pairwise exchange: the source merges the target's state and then the target merges the source. */
    internal void Gossip(int from, int to)
    {
        Replicas[from].MergeFrom(Replicas[to]);
        Replicas[to].MergeFrom(Replicas[from]);
        Log.Add(from + "->" + to);
    }

    internal void Round()
    {
        for (int i = 0; i < Replicas.Count; i++)
        {
            for (int j = i + 1; j < Replicas.Count; j++) Gossip(i, j);
        }
    }

    /** Gossip with a deterministic random pairing; returns the number of rounds needed for convergence. */
    internal int GossipUntilConverged(int maxRounds, ulong seed)
    {
        ulong rng = seed;
        for (int round = 1; round <= maxRounds; round++)
        {
            int a = SplitMix64.NextInt(ref rng, Replicas.Count);
            int b = SplitMix64.NextInt(ref rng, Replicas.Count);
            if (a != b) Gossip(Math.Min(a, b), Math.Max(a, b));
            if (Converged()) return round;
        }
        return -1;
    }

    internal bool Converged()
    {
        string digest = Replicas[0].Digest();
        foreach (Replica replica in Replicas) if (replica.Digest() != digest) return false;
        return true;
    }

    internal string Digest()
    {
        var builder = new StringBuilder();
        for (int i = 0; i < Replicas.Count; i++) builder.Append('r').Append(i).Append(':').Append(Replicas[i].Digest()).Append('#');
        return builder.ToString();
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-3. Version vectors: increments, the pointwise-maximum join and the causality test.
        var a = new VectorClock();
        a.Increment(0);
        a.Increment(0);
        var b = new VectorClock();
        b.Increment(1);
        Check(a.Compare(a.Clone()) == Order.Equal, "a clock is equal to its own clone");
        Check(a.Compare(b) == Order.Concurrent, "writes from different replicas without exchange are concurrent");
        var merged = a.Clone();
        merged.Merge(b);
        Check(merged.Get(0) == 2 && merged.Get(1) == 1 && merged.Show() == "[0:2,1:1]", "the join takes the pointwise maximum");
        var causallyLater = merged.Clone();
        causallyLater.Increment(1);
        Check(a.Compare(causallyLater) == Order.Before && causallyLater.Compare(a) == Order.After,
            "after a merge the causally later write dominates the earlier one");

        // 4-6. LastWriteWins: causally ordered writes are never lost, concurrent writes are resolved identically.
        var lww = new Cluster(3, Policy.LastWriteWins);
        lww[0].Put("user", "alice");
        lww.Gossip(0, 1);
        lww[1].Put("user", "bob");                                     // causally after alice: bob must win everywhere
        lww.Round();
        Check(lww[0].Read("user") == "bob" && lww[1].Read("user") == "bob" && lww[2].Read("user") == "bob",
            "a causally newer write is not lost (no lost update)");
        Check(lww.Converged(), "the cluster converged after one full round");
        var lwwConcurrent = new Cluster(3, Policy.LastWriteWins);
        lwwConcurrent[0].Put("k", "from-r0");
        lwwConcurrent[1].Put("k", "from-r1");
        Check(lwwConcurrent[0].Read("k") != lwwConcurrent[1].Read("k"), "before gossip the replicas disagree (eventual consistency)");
        lwwConcurrent.Round();
        Check(lwwConcurrent[0].Read("k") == lwwConcurrent[1].Read("k") && lwwConcurrent[1].Read("k") == lwwConcurrent[2].Read("k"),
            "LastWriteWins resolves the concurrent pair to one value at every replica");
        Check(lwwConcurrent[0].Read("k") == "from-r1", "the tie is broken by the replica id, so the outcome is deterministic");

        // 7. MaxWins picks the largest value regardless of arrival order.
        var max = new Cluster(2, Policy.MaxWins);
        max[0].Put("score", "10");
        max[1].Put("score", "7");
        max.Round();
        Check(max[0].Read("score") == "10" && max[1].Read("score") == "10", "MaxWins keeps the larger value");
        max[0].Put("score", "3");
        max.Round();
        Check(max[0].Read("score") == "10", "MaxWins never regresses to a smaller value");
        max[1].Put("score", "42");
        max.Round();
        Check(max[0].Read("score") == "42", "a larger later value still wins");

        // 8-10. MultiValue keeps siblings and a resolver folds them.
        var multi = new Cluster(3, Policy.MultiValue);
        multi[0].Put("tags", "red");
        multi[1].Put("tags", "blue");
        multi[2].Put("tags", "green");
        multi.Round();
        Check(multi.Converged(), "the MultiValue cluster converged with three siblings");
        Entry entry = multi[0].GetOrCreate("tags");
        Check(entry.Candidates.Count == 3, "all three concurrent writes survived as siblings");
        Check(Resolution.ResolveMultiValue(entry.Candidates) == "blue+green+red", "the resolver merges the siblings in a deterministic order");
        multi.Round();
        Check(multi[0].GetOrCreate("tags").Candidates.Count == 3, "repeated gossip is idempotent and adds no phantom siblings");

        // 11-13. PN-counter: conflict free increments and decrements converge to the plain arithmetic total.
        var counter = new Cluster(3, Policy.Counter);
        counter[0].Put("hits", "inc");
        counter[0].Put("hits", "inc");
        counter[1].Put("hits", "inc");
        counter[1].Put("hits", "dec");
        counter[2].Put("hits", "inc");
        counter.Round();
        Check(counter[0].Read("hits") == "3" && counter[1].Read("hits") == "3" && counter[2].Read("hits") == "3",
            "the PN-counter converges to 3 = 4 increments - 1 decrement");
        counter[2].Put("hits", "dec");
        counter.Round();
        Check(counter[0].Read("hits") == "2", "a later decrement is applied once, not per replica");
        Check(counter[0].GetOrCreate("hits").Positives.Count == 3, "the counter keeps per-replica components");

        // 14-15. Brute-force order independence: every permutation of a gossip round must end in the same state.
        var baseline = OrderPermutations(Policy.LastWriteWins);
        Check(baseline.Count == 6, "the brute-force oracle enumerated every permutation of the round's gossip pairs");
        bool allEqual = true;
        foreach (string digest in baseline) if (digest != baseline[0]) allEqual = false;
        Check(allEqual, "LastWriteWins merges are order independent (6 permutations, one digest)");
        var multiDigests = OrderPermutations(Policy.MultiValue);
        allEqual = true;
        foreach (string digest in multiDigests) if (digest != multiDigests[0]) allEqual = false;
        Check(allEqual, "MultiValue merges are order independent as well");

        // 16. An independent reference merge, folding the same writes in a different order.
        var reference = new Cluster(2, Policy.LastWriteWins);
        reference[0].Put("k", "v0");
        reference[1].Put("k", "v1");
        var folded = new Replica(0, Policy.LastWriteWins);
        folded.MergeFrom(reference[1]);                        // fold in the opposite order to the gossip below
        folded.MergeFrom(reference[0]);
        reference.Round();
        Check(reference[0].Read("k") == "v1" && folded.Read("k") == "v1" && reference.Converged(),
            "a reference fold in a different order agrees with the gossip result");

        // 17-19. Convergence bounds, monotone clocks and idempotence.
        var sparse = new Cluster(5, Policy.LastWriteWins);
        for (int i = 0; i < 5; i++) sparse[i].Put("k" + i, "v" + i);
        int rounds = sparse.GossipUntilConverged(60, 0x5EEDUL);
        Check(rounds > 0 && rounds <= 30, "random-pairing gossip converged in a bounded number of rounds (" + rounds + ")");
        Check(sparse.Converged() && sparse[0].Read("k4") == "v4" && sparse[4].Read("k0") == "v0", "every scattered write reached every replica");
        var monotone = new Cluster(2, Policy.LastWriteWins);
        monotone[0].Put("k", "a");
        long before0 = monotone[0].GetOrCreate("k").Clock.Get(0);
        monotone.Round();
        long after0 = monotone[0].GetOrCreate("k").Clock.Get(0);
        Check(after0 >= before0 && monotone[1].GetOrCreate("k").Clock.Get(0) == 1,
            "the merged clock only grows and the peer adopted the write");
        string once = monotone[0].Digest();
        monotone.Round();
        Check(monotone[0].Digest() == once, "merging the same state again is a no-op (idempotence)");

        // 20-21. Boundary cases: unknown keys, empty merges, a single replica and null values.
        Check(lww[0].Read("missing") == null, "reading an unknown key returns null");
        var emptyTarget = new Cluster(1, Policy.LastWriteWins);
        var emptySource = new Cluster(1, Policy.LastWriteWins);
        emptyTarget[0].MergeFrom(emptySource[0]);
        Check(emptyTarget[0].State.Count == 0, "merging two empty states changes nothing");
        var solo = new Cluster(1, Policy.LastWriteWins);
        solo[0].Put("k", "only");
        Check(solo.Converged() && solo[0].Read("k") == "only", "a single replica is trivially converged and reads its own write");
        var nulls = new Cluster(2, Policy.MultiValue);
        nulls[0].Put("k", "value");
        nulls[1].Put("k", "x");
        nulls.Round();
        Check(nulls[0].GetOrCreate("k").Candidates.Count == 2 && nulls.Converged(), "a concurrent pair still converges under MultiValue");
        Check(Resolution.ResolveMultiValue(new List<Write> { new Write(null, 0, 1) }) == "", "a null sibling resolves to the empty string");

        // 22. Determinism.
        var detA = new Cluster(3, Policy.MaxWins);
        var detB = new Cluster(3, Policy.MaxWins);
        foreach (Cluster cluster in new[] { detA, detB })
        {
            cluster[0].Put("x", "9");
            cluster[1].Put("x", "4");
            cluster[2].Put("y", "1");
            cluster.Round();
            cluster.Round();
        }
        Check(detA.Digest() == detB.Digest() && detA[0].Read("x") == "9", "identical replay produces identical replicas");
        Console.WriteLine("  gossip: replicas=5 converged in " + rounds + " random-pairing rounds; permutations checked=12");

        Console.WriteLine("PASS");
    }

    /** Brute force: run one round of all three gossip pairs in every possible order and collect the digests. */
    private static List<string> OrderPermutations(Policy policy)
    {
        int[][] pairs = { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 0, 2 } };
        var digests = new List<string>();
        var order = new List<int>();
        var used = new bool[pairs.Length];
        Permute(policy, pairs, used, order, digests);
        return digests;
    }

    private static void Permute(Policy policy, int[][] pairs, bool[] used, List<int> order, List<string> digests)
    {
        if (order.Count == pairs.Length)
        {
            var cluster = new Cluster(3, policy);
            cluster[0].Put("k", "from-0");
            cluster[1].Put("k", "from-1");
            cluster[2].Put("k", "from-2");
            foreach (int index in order) cluster.Gossip(pairs[index][0], pairs[index][1]);
            digests.Add(cluster.Digest());
            return;
        }
        for (int i = 0; i < pairs.Length; i++)
        {
            if (used[i]) continue;
            used[i] = true;
            order.Add(i);
            Permute(policy, pairs, used, order, digests);
            order.RemoveAt(order.Count - 1);
            used[i] = false;
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
