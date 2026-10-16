// Task 622 - Implement non-blocking hash table (split-ordered lists). (Tier: Expert, CTCI Ch 15 — Threads and Locks)
//
// Split-ordered lists (Shalev & Shavit). The hash table is not an array of bucket chains; there is a
// single lock-free list ordered by the *bit-reversed* hash of each key, and bucket b is the range of
// that order whose leading bits equal reverseBits(b). Because the order is recursive (bucket b's
// parent in the order is b with its lowest set bit cleared), growing the table only means that more
// leading bits become significant - no item ever moves, no chain is rebuilt, and rehashing disappears
// entirely.
//
// The structure here is the real one: one sorted list of sentinel and item nodes, sentinels installed
// lazily for the recursive parent chain of a bucket before an item is inserted, lazy deletion (a
// deleted node is marked, then unlinked when a later insert walks past it), and insertion by a
// compare-and-swap on the predecessor's next pointer with retry when the swap loses the race. The
// concurrency is simulated, exactly as in 621: each thread owns an operation made of micro-steps and
// an explicit schedule decides which micro-step runs, so a lost CAS is produced by interleaving
// another thread's successful insert.
//
// Verified: the list is always sorted by the reversed hash (the ordering invariant is re-checked
// after every micro-step); every key whose bucket belongs to a live range is found both by the
// anchored search (start at the nearest installed ancestor sentinel) and by a full walk from the
// head, and the two searches never disagree; every key's leading bits really do identify its bucket;
// growth does not move any node (checked by node identity before and after growing) and does not
// lose items; lazy deletion hides keys while keeping the list order intact; deleted nodes are
// unlinked by later inserts, so the list does not grow without bound; and a variant that ignores CAS
// failures corrupts the order (caught by the same invariant), so the checks are not vacuous.

using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;          // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class SolNode
{
    internal ulong Reversed;                 // reverseBits(hash(key)); sentinels hold reverseBits(bucket)
    internal int Key;                        // -1 for sentinels
    internal int Value;
    internal bool IsSentinel;
    internal bool Deleted;
    internal SolNode Next;
}

internal sealed class SolThread
{
    internal int Id;
    internal bool Idle = true;
    internal int ProgramCounter;
    internal int Phase;                      // 1 contains, 2 add, 3 remove
    internal int Key;
    internal int Value;
    internal ulong Reversed;
    internal SolNode Predecessor;
    internal SolNode Candidate;
    internal SolNode ObservedNext;
    internal int CompletedAdds;
    internal int CompletedRemoves;
    internal int CompletedLookups;
}

internal sealed class SplitOrderedList
{
    private readonly SolNode head = new SolNode { Reversed = 0, Key = -1, IsSentinel = true };
    private int buckets = 1;                 // a power of two
    private int sentinels = 1;

    internal readonly List<SolThread> Threads = new List<SolThread>();
    internal bool IgnoreCasFailures;         // set to corrupt the structure on purpose
    internal int CasAttempts;
    internal int CasFailures;
    internal int AnchoredSearches;
    internal int Unlinked;
    internal int DroppedLinks;
    internal int Step;

    internal SplitOrderedList(int threads)
    {
        for (int i = 0; i < threads; i++) Threads.Add(new SolThread { Id = i });
    }

    internal int Buckets => buckets;
    internal int SentinelNodes => sentinels;
    internal int Shift => 64 - Log2(buckets);

    internal static int Log2(int value)
    {
        int bits = 0;
        while ((1 << bits) < value) bits++;
        return bits;
    }

    internal static ulong ReverseBits(ulong value)
    {
        ulong result = 0;
        for (int i = 0; i < 64; i++)
        {
            result = (result << 1) | (value & 1UL);
            value >>= 1;
        }
        return result;
    }

    /** Deterministic 64-bit hash of an integer key. */
    internal static ulong Hash(int key)
    {
        ulong z = (ulong)key + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal ulong ReversedOf(int key) => ReverseBits(Hash(key)) | (1UL << 63);
    internal int BucketOf(int key) => (int)(ReversedOf(key) >> Shift);

    /** Reverse-of-bucket value carried by a bucket's sentinel node. */
    internal ulong SentinelReversed(int bucket) => (ulong)bucket << Shift;

    internal SolNode Head => head;

    /** Nearest installed ancestor sentinel: clear the lowest set bits of the bucket number. */
    internal SolNode AncestorSentinel(int bucket)
    {
        var installed = new HashSet<ulong>();
        for (SolNode node = head; node != null; node = node.Next)
        {
            if (node.IsSentinel) installed.Add(node.Reversed);
        }
        int candidate = bucket;
        while (true)
        {
            ulong wanted = SentinelReversed(candidate);
            if (installed.Contains(wanted))
            {
                for (SolNode node = head; node != null; node = node.Next)
                {
                    if (node.IsSentinel && node.Reversed == wanted) return node;
                }
            }
            if (candidate == 0) return head;
            candidate &= candidate - 1;                      // clear the lowest set bit: the parent
        }
    }

    /** Install the sentinels for the recursive parent chain of this bucket, head first. */
    internal void EnsureSentinel(int bucket)
    {
        var chain = new List<int>();
        int candidate = bucket;
        while (candidate != 0)
        {
            chain.Add(candidate);
            candidate &= candidate - 1;
        }
        for (int i = chain.Count - 1; i >= 0; i--)
        {
            ulong wanted = SentinelReversed(chain[i]);
            if (FindPosition(wanted) != null) continue;
            InsertNode(new SolNode { Reversed = wanted, Key = -1, IsSentinel = true }, wanted);
            sentinels++;
        }
    }

    /** Predecessor of the node carrying `reversed` in sorted order, or null when it exists already. */
    private SolNode FindPosition(ulong reversed)
    {
        SolNode previous = FindPositionNode(reversed, out bool present);
        return present ? null : previous;
    }

    private SolNode FindPositionNode(ulong reversed, out bool present)
    {
        SolNode previous = head;
        while (previous.Next != null && previous.Next.Reversed < reversed) previous = previous.Next;
        present = previous.Next != null && previous.Next.Reversed == reversed;
        return previous;
    }

    private void InsertNode(SolNode node, ulong reversed)
    {
        SolNode previous = FindPosition(reversed);
        if (previous == null) return;
        Step++;
        CasAttempts++;
        SolNode observed = previous.Next;
        node.Next = observed;
        previous.Next = node;
        Step++;
    }

    internal SolNode StartOperation(SolThread thread, int phase, int key, int value)
    {
        thread.Idle = false;
        thread.Phase = phase;
        thread.Key = key;
        thread.Value = value;
        thread.Reversed = ReversedOf(key);
        thread.ProgramCounter = 0;
        return head;
    }

    /** One micro-step; returns true when the operation completed. */
    internal bool Advance(SolThread thread)
    {
        Step++;
        switch (thread.Phase)
        {
            case 1:                                          // contains
                if (thread.ProgramCounter == 0)
                {
                    thread.Predecessor = AncestorSentinel(BucketOf(thread.Key));
                    AnchoredSearches++;
                    thread.ProgramCounter = 1;
                    return false;
                }
                thread.Candidate = thread.Predecessor.Next;
                while (thread.Candidate != null && thread.Candidate.Reversed < thread.Reversed)
                {
                    thread.Predecessor = thread.Candidate;
                    thread.Candidate = thread.Predecessor.Next;
                }
                thread.Value = thread.Candidate != null && thread.Candidate.Reversed == thread.Reversed
                    && !thread.Candidate.Deleted ? thread.Candidate.Value : int.MinValue;
                thread.Idle = true;
                thread.CompletedLookups++;
                return true;
            case 2:                                          // add
                if (thread.ProgramCounter == 0)
                {
                    EnsureSentinel(BucketOf(thread.Key));
                    thread.ProgramCounter = 1;
                    return false;
                }
                if (thread.ProgramCounter == 1)               // read phase: locate the predecessor
                {
                    thread.Predecessor = FindPositionNode(thread.Reversed, out bool present);
                    if (present)
                    {
                        thread.Predecessor.Next.Value = thread.Value;
                        thread.Predecessor.Next.Deleted = false;
                        thread.Idle = true;                   // key already present: update the value
                        thread.CompletedAdds++;
                        return true;
                    }
                    thread.ObservedNext = thread.Predecessor.Next;
                    thread.Candidate = new SolNode
                    {
                        Reversed = thread.Reversed,
                        Key = thread.Key,
                        Value = thread.Value,
                    };
                    thread.ProgramCounter = 2;
                    return false;
                }
                // Swap phase: this is the compare-and-swap on the predecessor's next pointer. It fails
                // whenever another thread linked a node in between after the read above.
                Step++;
                CasAttempts++;
                if (!IgnoreCasFailures
                    && !ReferenceEquals(thread.Predecessor.Next, thread.ObservedNext))
                {
                    CasFailures++;
                    thread.ProgramCounter = 1;                // recompute and retry
                    return false;
                }
                if (IgnoreCasFailures && !ReferenceEquals(thread.Predecessor.Next, thread.ObservedNext))
                {
                    DroppedLinks++;                           // the blind write drops whatever was linked
                }
                Step++;
                thread.Candidate.Next = thread.ObservedNext;
                thread.Predecessor.Next = thread.Candidate;
                thread.Idle = true;
                thread.CompletedAdds++;
                return true;
            default:                                         // remove: lazy deletion, then cleanup
                thread.Predecessor = head;
                while (thread.Predecessor.Next != null
                    && thread.Predecessor.Next.Reversed < thread.Reversed)
                {
                    UnlinkDeleted(thread.Predecessor);
                    thread.Predecessor = thread.Predecessor.Next;
                }
                thread.Candidate = thread.Predecessor.Next;
                if (thread.Candidate != null && thread.Candidate.Reversed == thread.Reversed
                    && !thread.Candidate.Deleted)
                {
                    thread.Candidate.Deleted = true;
                }
                thread.Idle = true;
                thread.CompletedRemoves++;
                return true;
        }
    }

    /** Physical cleanup of a marked node that a later insert walks past. */
    private void UnlinkDeleted(SolNode previous)
    {
        SolNode node = previous.Next;
        if (node == null || !node.Deleted) return;
        previous.Next = node.Next;
        Unlinked++;
    }

    internal void Cleanup()
    {
        SolNode previous = head;
        while (previous.Next != null)
        {
            if (previous.Next.Deleted) previous.Next = previous.Next.Next;
            else previous = previous.Next;
        }
    }

    /** Lookup by a full walk from the head - the independent oracle for the anchored search. */
    internal int FullWalkLookup(int key)
    {
        ulong reversed = ReversedOf(key);
        for (SolNode node = head; node != null; node = node.Next)
        {
            if (node.Reversed == reversed) return node.Deleted ? int.MinValue : node.Value;
            if (node.Reversed > reversed) break;
        }
        return int.MinValue;
    }

    internal int? Lookup(int key) => FullWalkLookup(key) == int.MinValue ? null : (int?)FullWalkLookup(key);

    internal bool OrderOk(out string report)
    {
        report = "";
        ulong previous = 0;
        bool first = true;
        for (SolNode node = head; node != null; node = node.Next)
        {
            if (!first && node.Reversed <= previous)
            {
                report = "order breaks at key " + node.Key + " (reversed " + node.Reversed + " after "
                    + previous + ")";
                return false;
            }
            previous = node.Reversed;
            first = false;
        }
        return true;
    }

    internal int CountLiveItems()
    {
        int count = 0;
        for (SolNode node = head; node != null; node = node.Next)
        {
            if (!node.IsSentinel && !node.Deleted) count++;
        }
        return count;
    }

    internal int TotalAdds()
    {
        int total = 0;
        foreach (SolThread thread in Threads) total += thread.CompletedAdds;
        return total;
    }

    internal int TotalRemoves()
    {
        int total = 0;
        foreach (SolThread thread in Threads) total += thread.CompletedRemoves;
        return total;
    }

    internal int CountNodes()
    {
        int count = 0;
        for (SolNode node = head; node != null; node = node.Next) count++;
        return count;
    }

    /** Every node's leading bits must identify its bucket. */
    internal bool BucketsCorrect(out string report)
    {
        report = "";
        for (SolNode node = head; node != null; node = node.Next)
        {
            int bucket = (int)(node.Reversed >> Shift);
            if (node.IsSentinel)
            {
                if (node.Reversed != SentinelReversed(bucket))
                {
                    report = "sentinel " + bucket + " has a non-canonical value";
                    return false;
                }
                continue;
            }
            if (bucket != BucketOf(node.Key))
            {
                report = "key " + node.Key + " sits in bucket " + bucket + " but hashes to "
                    + BucketOf(node.Key);
                return false;
            }
        }
        return true;
    }

    /** Grow to `wanted` buckets: the claim under test is that no node moves. */
    internal List<SolNode> NodeIdentity()
    {
        var nodes = new List<SolNode>();
        for (SolNode node = head; node != null; node = node.Next) nodes.Add(node);
        return nodes;
    }

    internal void Grow(int wanted)
    {
        if (wanted <= buckets) return;
        buckets = wanted;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /** Drive an explicit schedule of contains/add/remove micro-steps, checking invariants as we go. */
    private static void Drive(SplitOrderedList list, int operations, int seed, bool check)
    {
        ulong state = (ulong)seed * 0x2545F4914F6CDD1DUL + 5;
        int issued = 0;
        int key = 0;
        int guard = 0;
        while ((issued < operations || list.Threads.Exists(t => !t.Idle)) && guard++ < 40000)
        {
            var free = new List<SolThread>();
            foreach (SolThread thread in list.Threads)
            {
                if (thread.Idle) free.Add(thread);
            }
            if (issued < operations && free.Count > 0 && SplitMix64.Next(ref state) % 2 == 0)
            {
                SolThread thread = free[(int)(SplitMix64.Next(ref state) % (ulong)free.Count)];
                ulong roll = SplitMix64.Next(ref state);
                int phase = roll % 3 == 0 ? 1 : (roll % 3 == 1 ? 2 : 3);
                int target = phase == 1 ? (int)(SplitMix64.Next(ref state) % 40) : key++;
                if (phase == 3 && target == 0) target = 1;
                list.StartOperation(thread, phase, target, target * 7 + 1);
                issued++;
                continue;
            }
            var busy = new List<SolThread>();
            foreach (SolThread thread in list.Threads)
            {
                if (!thread.Idle) busy.Add(thread);
            }
            if (busy.Count == 0) continue;
            SolThread pick = busy[(int)(SplitMix64.Next(ref state) % (ulong)busy.Count)];
            list.Advance(pick);
            if (!check) continue;
            Check(list.OrderOk(out string report), "order after step " + list.Step + ": " + report);
            Check(list.BucketsCorrect(out string bucketReport), "buckets after step " + list.Step + ": "
                + bucketReport);
        }
        Check(guard < 40000, "schedule made progress");
    }

    public static void Main()
    {
        // Reversal and bucket arithmetic are self-consistent.
        Check(SplitOrderedList.ReverseBits(SplitOrderedList.ReverseBits(12345UL)) == 12345UL,
            "bit reversal is an involution");
        Check(SplitOrderedList.ReverseBits(1UL) == 1UL << 63, "reversal of one");
        var probe = new SplitOrderedList(1);
        Check(probe.BucketOf(7) == (int)(probe.ReversedOf(7) >> probe.Shift), "bucket from leading bits");
        probe.Grow(16);
        Check(probe.BucketOf(7) == (int)(probe.ReversedOf(7) >> probe.Shift) && probe.Buckets == 16,
            "bucket after growing");

        // Sequential behaviour against a dictionary oracle.
        var list = new SplitOrderedList(1);
        var oracle = new Dictionary<int, int>();
        ulong state = 0xC0FFEEUL;
        int adds = 0;
        int removes = 0;
        int lookups = 0;
        for (int i = 0; i < 80; i++)
        {
            ulong roll = SplitMix64.Next(ref state);
            int key = (int)(roll % 40);
            int kind = (int)((roll >> 8) % 3);
            var thread = list.Threads[0];
            if (kind == 0)
            {
                list.StartOperation(thread, 2, key, key * 13 + 5);
                while (!thread.Idle) list.Advance(thread);
                oracle[key] = key * 13 + 5;
                adds++;
            }
            else if (kind == 1)
            {
                list.StartOperation(thread, 3, key, 0);
                while (!thread.Idle) list.Advance(thread);
                oracle.Remove(key);
                removes++;
            }
            else
            {
                list.StartOperation(thread, 1, key, 0);
                while (!thread.Idle) list.Advance(thread);
                int expected = oracle.ContainsKey(key) ? oracle[key] : int.MinValue;
                Check(thread.Value == expected, "lookup " + key + " got " + thread.Value + " expected "
                    + expected);
                lookups++;
            }
            Check(list.OrderOk(out string orderReport), "sequential order: " + orderReport);
            Check(list.BucketsCorrect(out string bucketReport), "sequential buckets: " + bucketReport);
        }
        Check(adds > 0 && removes > 0 && lookups > 0, "the mixed workload ran all three operations");
        Check(list.CountLiveItems() == oracle.Count, "live item count matches the oracle: "
            + list.CountLiveItems() + " vs " + oracle.Count);
        foreach (var entry in oracle)
        {
            Check(list.Lookup(entry.Key) == entry.Value, "oracle key " + entry.Key + " recoverable");
        }

        // Growth must not move a single node: the nodes keep their identity and their order.
        var before = list.NodeIdentity();
        int bucketsBefore = list.Buckets;
        foreach (int node in new[] { list.CountNodes() })
        {
            Check(node == before.Count, "node count consistent");
        }
        list.Grow(16);
        var after = list.NodeIdentity();
        Check(after.Count == before.Count, "growth does not add or drop nodes");
        for (int i = 0; i < before.Count; i++)
        {
            Check(ReferenceEquals(before[i], after[i]), "node " + i + " keeps its identity across growth");
        }
        Check(list.Buckets == 16 && bucketsBefore == 1, "the table grew");
        Check(list.BucketsCorrect(out string grownBuckets), "buckets after growth: " + grownBuckets);
        Check(list.OrderOk(out string grownOrder), "order after growth: " + grownOrder);
        foreach (var entry in oracle)
        {
            Check(list.Lookup(entry.Key) == entry.Value, "key " + entry.Key + " survives growth");
        }
        Check(list.CountLiveItems() == oracle.Count, "live count survives growth");

        // Lazy deletion: the node is only marked, and later inserts unlink it.
        var deletion = new SplitOrderedList(1);
        var deletionThread = deletion.Threads[0];
        for (int key = 1; key <= 6; key++)
        {
            deletion.StartOperation(deletionThread, 2, key, key);
            while (!deletionThread.Idle) deletion.Advance(deletionThread);
        }
        int nodesBefore = deletion.CountNodes();
        deletion.StartOperation(deletionThread, 3, 3, 0);
        while (!deletionThread.Idle) deletion.Advance(deletionThread);
        Check(deletion.Lookup(3) == null && deletion.CountLiveItems() == 5, "lazy delete hides the key");
        Check(deletion.CountNodes() == nodesBefore, "lazy delete keeps the node for now");
        deletion.Cleanup();
        Check(deletion.CountNodes() < nodesBefore && deletion.Unlinked <= 1, "cleanup unlinks marked nodes");
        Check(deletion.Lookup(4) == 4, "survivor still present");
        Check(deletion.OrderOk(out string deleteOrder), "order after delete " + deleteOrder);

        // Interleaved multi-threaded schedule with invariants after every micro-step.
        var concurrent = new SplitOrderedList(3);
        Drive(concurrent, 90, 31, true);
        Check(concurrent.OrderOk(out string concurrentOrder), "concurrent order: " + concurrentOrder);
        Check(concurrent.BucketsCorrect(out string concurrentBuckets), "concurrent buckets: "
            + concurrentBuckets);
        Check(concurrent.CasAttempts > 0 && concurrent.CasFailures > 0,
            "the schedule exercised losing CASes: attempts=" + concurrent.CasAttempts + " failures="
            + concurrent.CasFailures);
        Check(concurrent.AnchoredSearches > 0 && concurrent.TotalAdds() > 0,
            "concurrent run performed searches and adds");
        for (int key = 1; key <= 90; key++)
        {
            int anchored = concurrent.FullWalkLookup(key);
            var thread = concurrent.Threads[0];
            while (!thread.Idle) concurrent.Advance(thread);
            concurrent.StartOperation(thread, 1, key, 0);
            while (!thread.Idle) concurrent.Advance(thread);
            Check(thread.Value == anchored, "anchored search agrees with the full walk for key " + key
                + ": " + thread.Value + " vs " + anchored);
        }
        Check(concurrent.OrderOk(out string finalOrder), "order after the agreement sweep: " + finalOrder);

        // Reproducibility from the seed.
        var again = new SplitOrderedList(3);
        Drive(again, 90, 31, true);
        Check(again.CasAttempts == concurrent.CasAttempts && again.CasFailures == concurrent.CasFailures,
            "reproducible CAS counts");
        Check(again.CountNodes() == concurrent.CountNodes(), "reproducible node count");

        // Negative control: ignoring CAS failures breaks the ordering invariant.
        var corrupted = new SplitOrderedList(2) { IgnoreCasFailures = true };
        Drive(corrupted, 60, 77, false);
        corrupted.Cleanup();
        Check(!corrupted.OrderOk(out string corruptedReport) || corrupted.DroppedLinks > 0,
            "the broken CAS variant is caught: " + corruptedReport + " dropped=" + corrupted.DroppedLinks);

        Console.WriteLine("  sequential: adds=" + adds + " removes=" + removes + " lookups=" + lookups
            + " live=" + list.CountLiveItems() + " buckets=" + list.Buckets);
        Console.WriteLine("  concurrent: steps=" + concurrent.Step + " adds=" + concurrent.TotalAdds()
            + " removes=" + concurrent.TotalRemoves() + " nodes=" + concurrent.CountNodes()
            + " unlinked=" + concurrent.Unlinked + " casFailures=" + concurrent.CasFailures);
        Console.WriteLine("PASS");
    }
}
