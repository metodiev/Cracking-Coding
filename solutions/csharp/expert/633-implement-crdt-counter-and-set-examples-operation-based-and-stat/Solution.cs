// Task 633 - Implement CRDT counter and set examples (operation-based and state-based). (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// Two flavours of CRDT are implemented side by side.
//
// STATEFUL (convergent / CvRDT): replicas exchange their whole state and merge with a join that is
// commutative, associative and idempotent, which makes the result independent of delivery order.
//   * G-Counter : state = one count per replica, merge = componentwise max, value = the sum.
//   * PN-Counter: two G-Counters, one for increments and one for decrements; the value is their
//     difference, so a PN-Counter can go negative.
//   * OR-Set    : each element owns a set of unique add-tags and the state also carries the set of
//     removed tags. Merging unions both sets, and an element is present while it still has a tag
//     that was never removed. An element therefore survives a concurrent remove (add-wins),
//     because the remover can only tombstone the tags it had actually observed.
//
// OPERATION-BASED (commutative / CmRDT): replicas broadcast operations and every operation carries
// the sender's causal context, so an operation that arrives before one it depends on is buffered
// until the dependency is applied. The counter below applies the operations in causal order, which
// is exactly the delivery condition a CmRDT requires.
//
// What is simplified (documented per the authoring contract): single process, no threads or
// sockets, and a deterministic in-memory exchange - "replica" is just a state object and delivery
// order is scripted or brute-forced. Verified invariants: the merge laws hold for every permutation
// of the replicas' states, replicas converge to the same value/set, concurrent add-and-remove
// resolves add-wins, remove only cancels observed adds, and the op-based counter never applies an
// operation out of causal order.
using System;
using System.Collections.Generic;
using System.Text;

#nullable enable

internal sealed class GCounter
{
    public readonly int[] Counts;

    public GCounter(int replicaCount) { Counts = new int[replicaCount]; }

    private GCounter(int[] counts) { Counts = counts; }

    public GCounter Clone() => new GCounter((int[])Counts.Clone());

    public void Increment(int replicaId, int amount = 1) => Counts[replicaId] += amount;

    public int Value
    {
        get
        {
            int total = 0;
            foreach (int count in Counts) total += count;
            return total;
        }
    }

    public void Merge(GCounter other)
    {
        for (int i = 0; i < Counts.Length; i++) if (other.Counts[i] > Counts[i]) Counts[i] = other.Counts[i];
    }

    public string Describe() => "(" + string.Join(",", Counts) + ")";
}

internal sealed class PnCounter
{
    public readonly GCounter Increments;
    public readonly GCounter Decrements;

    public PnCounter(int replicaCount)
    {
        Increments = new GCounter(replicaCount);
        Decrements = new GCounter(replicaCount);
    }

    private PnCounter(GCounter increments, GCounter decrements)
    {
        Increments = increments;
        Decrements = decrements;
    }

    public PnCounter Clone() => new PnCounter(Increments.Clone(), Decrements.Clone());
    public void Increment(int replicaId) => Increments.Increment(replicaId);
    public void Decrement(int replicaId) => Decrements.Increment(replicaId);
    public int Value => Increments.Value - Decrements.Value;

    public void Merge(PnCounter other)
    {
        Increments.Merge(other.Increments);
        Decrements.Merge(other.Decrements);
    }

    public string Describe() => Increments.Describe() + "-" + Decrements.Describe() + "=" + Value;
}

internal sealed class OrSetReplica
{
    private readonly Dictionary<string, HashSet<string>> adds = new Dictionary<string, HashSet<string>>();
    private readonly HashSet<string> removed = new HashSet<string>();
    private int counter;

    public readonly string Name;
    public int AddsMade;
    public int RemovesMade;
    public int TagsPruned;

    public OrSetReplica(string name) { Name = name; }

    public void Add(string element)
    {
        if (!adds.TryGetValue(element, out HashSet<string>? tags))
        {
            tags = new HashSet<string>();
            adds[element] = tags;
        }
        tags.Add(Name + "#" + (++counter));
        AddsMade++;
    }

    /// <summary>Observed-remove: only the tags this replica has actually seen are tombstoned.</summary>
    public void Remove(string element)
    {
        if (!adds.TryGetValue(element, out HashSet<string>? tags)) return;
        foreach (string tag in new List<string>(tags)) removed.Add(tag);
        RemovesMade++;
        Prune();
    }

    public bool Contains(string element)
    {
        if (!adds.TryGetValue(element, out HashSet<string>? tags)) return false;
        foreach (string tag in tags) if (!removed.Contains(tag)) return true;
        return false;
    }

    public List<string> Elements()
    {
        var present = new List<string>();
        foreach (string element in adds.Keys) if (Contains(element)) present.Add(element);
        present.Sort(StringComparer.Ordinal);
        return present;
    }

    public string Describe() => string.Join(",", Elements());

    public void Merge(OrSetReplica other)
    {
        foreach (KeyValuePair<string, HashSet<string>> kv in other.adds)
        {
            if (!adds.TryGetValue(kv.Key, out HashSet<string>? tags))
            {
                tags = new HashSet<string>();
                adds[kv.Key] = tags;
            }
            tags.UnionWith(kv.Value);
        }
        removed.UnionWith(other.removed);
        Prune();
    }

    private void Prune()
    {
        foreach (KeyValuePair<string, HashSet<string>> kv in adds)
        {
            int before = kv.Value.Count;
            kv.Value.RemoveWhere(tag => removed.Contains(tag));
            TagsPruned += before - kv.Value.Count;
        }
    }

    public OrSetReplica Clone()
    {
        var copy = new OrSetReplica(Name) { counter = counter };
        foreach (KeyValuePair<string, HashSet<string>> kv in adds) copy.adds[kv.Key] = new HashSet<string>(kv.Value);
        copy.removed.UnionWith(removed);
        return copy;
    }

    public int TagCount()
    {
        int total = 0;
        foreach (HashSet<string> tags in adds.Values) total += tags.Count;
        return total + removed.Count;
    }
}

internal sealed class CounterOp
{
    public readonly int From;
    public readonly int Seq;
    public readonly int Delta;
    public readonly int[] Context;      // the sender's applied-vector when the op was created

    public CounterOp(int from, int seq, int delta, int[] context)
    {
        From = from;
        Seq = seq;
        Delta = delta;
        Context = context;
    }
}

internal sealed class OpCounterReplica
{
    private readonly int[] applied;      // one slot per replica: how many ops were applied
    private readonly List<CounterOp> buffer = new List<CounterOp>();
    private int nextSeq = 1;

    public readonly int Id;
    public int Value;
    public int BufferedOps;
    public int AppliedOps;

    public OpCounterReplica(int id, int replicaCount)
    {
        Id = id;
        applied = new int[replicaCount];
    }

    public CounterOp LocalOp(int delta)
    {
        var op = new CounterOp(Id, nextSeq, delta, (int[])applied.Clone());
        nextSeq++;
        Apply(op);
        return op;
    }

    public bool Deliver(CounterOp op)
    {
        if (CanApply(op))
        {
            Apply(op);
            DrainBuffer();
            return true;
        }
        buffer.Add(op);
        BufferedOps++;
        return false;
    }

    private bool CanApply(CounterOp op)
    {
        if (op.Seq != applied[op.From] + 1) return false;
        for (int j = 0; j < applied.Length; j++)
        {
            if (j == op.From || j == Id) continue;
            if (op.Context[j] > applied[j]) return false;
        }
        return true;
    }

    private void Apply(CounterOp op)
    {
        applied[op.From] = op.Seq;
        Value += op.Delta;
        AppliedOps++;
    }

    private void DrainBuffer()
    {
        bool progress = true;
        while (progress)
        {
            progress = false;
            for (int i = 0; i < buffer.Count; i++)
            {
                if (!CanApply(buffer[i])) continue;
                CounterOp op = buffer[i];
                buffer.RemoveAt(i);
                Apply(op);
                progress = true;
                break;
            }
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int NextSeed(int seed) => (seed * 1103515245 + 12345) & 0x7FFFFFFF;

    private static IEnumerable<int[]> Permutations(int[] values)
    {
        if (values.Length <= 1)
        {
            yield return values;
            yield break;
        }
        for (int i = 0; i < values.Length; i++)
        {
            var rest = new int[values.Length - 1];
            int k = 0;
            for (int j = 0; j < values.Length; j++) if (j != i) rest[k++] = values[j];
            foreach (int[] tail in Permutations(rest))
            {
                var result = new int[values.Length];
                result[0] = values[i];
                Array.Copy(tail, 0, result, 1, tail.Length);
                yield return result;
            }
        }
    }

    private static void GCounterMergeLaws()
    {
        var states = new List<GCounter>();
        int seed = 4321;
        for (int i = 0; i < 4; i++)
        {
            var counter = new GCounter(4);
            for (int step = 0; step < 3; step++)
            {
                seed = NextSeed(seed);
                counter.Increment(seed % 4, 1 + seed % 5);
            }
            states.Add(counter);
        }

        foreach (int[] order in Permutations(new[] { 0, 1, 2, 3 }))
        {
            var merged = new GCounter(4);
            foreach (int index in order) merged.Merge(states[index]);
            var reference = new GCounter(4);
            for (int i = 0; i < 4; i++) reference.Merge(states[i]);
            Check(merged.Describe() == reference.Describe(), "merge order " + string.Join(",", order)
                + " gives " + merged.Describe() + ", expected " + reference.Describe());
        }

        var once = new GCounter(4);
        foreach (GCounter state in states) once.Merge(state);
        int value = once.Value;
        foreach (GCounter state in states) once.Merge(state);       // merge everything again
        Check(once.Value == value, "merging twice changes nothing (idempotence)");
        var perReplica = new int[4];
        foreach (GCounter state in states)
            for (int i = 0; i < 4; i++) perReplica[i] = Math.Max(perReplica[i], state.Counts[i]);
        int expected = 0;
        foreach (int count in perReplica) expected += count;
        Check(value == expected, "the G-Counter value is the sum of the maxima: " + value);
        Console.WriteLine("g-counter: value=" + value + " " + once.Describe());
    }

    private static void PnCounterIncrementsAndDecrements()
    {
        var r1 = new PnCounter(3);
        var r2 = new PnCounter(3);
        var r3 = new PnCounter(3);
        for (int i = 0; i < 7; i++) { r1.Increment(0); r2.Increment(1); }
        for (int i = 0; i < 5; i++) { r2.Decrement(1); r3.Decrement(2); }
        for (int i = 0; i < 4; i++) r3.Increment(2);
        Check(r1.Value == 7 && r2.Value == 2 && r3.Value == -1, "local values before the merge");

        var merged = r1.Clone();
        merged.Merge(r2);
        merged.Merge(r3);
        Check(merged.Value == 7 + 2 - 1, "the merged value is the sum of the replicas: " + merged.Value);
        var other = r3.Clone();
        other.Merge(r1);
        other.Merge(r2);
        Check(other.Value == merged.Value, "a different merge order gives the same value");
        Check(merged.Value == 8, "18 increments and 10 decrements leave 8, got " + merged.Value);

        var onlyDecrements = new PnCounter(1);
        for (int i = 0; i < 3; i++) onlyDecrements.Decrement(0);
        Check(onlyDecrements.Value == -3, "a PN-Counter can go negative, got " + onlyDecrements.Value);
        Check(new PnCounter(2).Value == 0, "a fresh counter is zero");
        Console.WriteLine("pn-counter: " + merged.Describe());
    }

    private static void OrSetAddWinsOverConcurrentRemove()
    {
        var r1 = new OrSetReplica("R1");
        var r2 = new OrSetReplica("R2");
        r1.Add("x");
        r2.Merge(r1);                       // R2 has now observed R1's tag
        Check(r2.Contains("x"), "the merged replica sees the element");
        r2.Remove("x");                     // R2 removes what it observed
        r1.Add("x");                        // concurrently, R1 adds a tag R2 never saw
        Check(r2.Contains("x") == false, "R2's own view has the element removed");
        r2.Merge(r1);
        Check(r2.Contains("x"), "the concurrent add wins after the merge");

        var r3 = new OrSetReplica("R3");
        r3.Add("y");
        r3.Remove("y");
        Check(!r3.Contains("y"), "remove cancels the observed add");
        r3.Add("y");
        Check(r3.Contains("y"), "a fresh add after a remove brings the element back");
        r3.Merge(r2);
        Check(r3.Contains("y") && r3.Contains("x"), "the merged replica has both elements");
        Check(r3.Elements().Count == 2, "exactly two elements: " + r3.Describe());
        Console.WriteLine("or-set: " + r3.Describe() + " (tags=" + r3.TagCount() + ")");
    }

    private static void OrSetMergesAreOrderIndependent()
    {
        var r1 = new OrSetReplica("R1");
        var r2 = new OrSetReplica("R2");
        var r3 = new OrSetReplica("R3");
        r1.Add("a");
        r1.Add("b");
        r2.Merge(r1);
        r2.Remove("a");
        r3.Add("c");
        r3.Remove("c");
        r3.Add("b");

        var states = new List<OrSetReplica> { r1.Clone(), r2.Clone(), r3.Clone() };
        string? reference = null;
        foreach (int[] order in Permutations(new[] { 0, 1, 2 }))
        {
            var merged = new OrSetReplica("M");
            foreach (int index in order) merged.Merge(states[index]);
            reference ??= merged.Describe();
            Check(merged.Describe() == reference, "merge order " + string.Join(",", order) + " gives "
                + merged.Describe() + ", expected " + reference);
        }
        Check(reference == "b", "the converged set is {b}: " + reference);

        var again = new OrSetReplica("M");
        foreach (OrSetReplica state in states) again.Merge(state);
        string once = again.Describe();
        foreach (OrSetReplica state in states) again.Merge(state);
        Check(again.Describe() == once, "re-merging is idempotent");
        Check(again.TagCount() > 0, "the state still holds tags: " + again.TagCount());
    }

    private static void OrSetRandomizedConvergence()
    {
        int seed = 777001;
        string[] elements = { "x", "y", "z" };
        for (int trial = 0; trial < 8; trial++)
        {
            var replicas = new List<OrSetReplica> { new OrSetReplica("A"), new OrSetReplica("B"), new OrSetReplica("C") };
            for (int step = 0; step < 40; step++)
            {
                seed = NextSeed(seed);
                int actor = seed % replicas.Count;
                seed = NextSeed(seed);
                string element = elements[seed % elements.Length];
                seed = NextSeed(seed);
                if (seed % 3 == 0 && replicas.Count > 1)
                {
                    int source = (actor + 1) % replicas.Count;
                    replicas[actor].Merge(replicas[source]);          // a gossip exchange
                }
                else if (seed % 2 == 0) replicas[actor].Add(element);
                else replicas[actor].Remove(element);
            }
            var merged = replicas[0].Clone();
            merged.Merge(replicas[1]);
            merged.Merge(replicas[2]);
            var other = replicas[2].Clone();
            other.Merge(replicas[0]);
            other.Merge(replicas[1]);
            Check(merged.Describe() == other.Describe(), "trial " + trial + ": merge orders disagree: "
                + merged.Describe() + " vs " + other.Describe());
            // A full exchange must give every replica exactly the merged set (absorption).
            foreach (OrSetReplica replica in replicas) replica.Merge(merged);
            foreach (OrSetReplica replica in replicas)
            {
                Check(replica.Describe() == merged.Describe(), "trial " + trial + ": replica " + replica.Name
                    + " shows " + replica.Describe() + " after the exchange, expected " + merged.Describe());
            }
        }
    }

    private static void OpBasedCounterRequiresCausalOrder()
    {
        const int replicas = 3;
        var nodes = new OpCounterReplica[replicas];
        for (int i = 0; i < replicas; i++) nodes[i] = new OpCounterReplica(i, replicas);

        CounterOp first = nodes[0].LocalOp(1);
        CounterOp second = nodes[0].LocalOp(1);          // depends on the first (same sender, FIFO)
        CounterOp third = nodes[1].LocalOp(1);
        CounterOp fourth = nodes[1].LocalOp(-1);         // depends on third

        Check(nodes[2].Deliver(second) == false, "the second op cannot be delivered before the first");
        Check(nodes[2].BufferedOps == 1, "it is buffered");
        Check(nodes[2].Value == 0, "nothing was applied yet");
        Check(nodes[2].Deliver(first), "the first op is deliverable");
        Check(nodes[2].Value == 2, "the buffered op was released: " + nodes[2].Value);
        Check(nodes[2].Deliver(third), "an op from another replica is deliverable");
        Check(nodes[2].Deliver(fourth), "so is its follow-up");
        Check(nodes[2].Value == 2, "1+1+1-1 = 2, got " + nodes[2].Value);
        Check(nodes[2].AppliedOps == 4, "four ops applied in total");

        // An op that depends on another replica's op must wait for it.
        var dependent = nodes[2];
        CounterOp crossOp = dependent.LocalOp(5);        // carries the context {1,2,2}
        var fresh = new OpCounterReplica(2, replicas);
        Check(fresh.Deliver(crossOp) == false, "the cross-replica op waits for its dependencies");
        Check(fresh.Deliver(first) && fresh.Deliver(second), "the dependencies are delivered");
        Check(fresh.Deliver(third) && fresh.Deliver(fourth), "including the other replica's ops");
        Check(fresh.Value == 7, "1+1+1-1+5 = 7, got " + fresh.Value);
        Check(fresh.AppliedOps == 5, "every operation applied exactly once");
        Console.WriteLine("op-counter: value=" + fresh.Value + " buffered=" + fresh.BufferedOps);
    }

    public static void Main()
    {
        GCounterMergeLaws();
        PnCounterIncrementsAndDecrements();
        OrSetAddWinsOverConcurrentRemove();
        OrSetMergesAreOrderIndependent();
        OrSetRandomizedConvergence();
        OpBasedCounterRequiresCausalOrder();
        Console.WriteLine("PASS");
    }
}
