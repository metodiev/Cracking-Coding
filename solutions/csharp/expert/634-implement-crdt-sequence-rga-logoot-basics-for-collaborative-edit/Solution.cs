// Task 634 - Implement CRDT sequence (RGA / Logoot) basics for collaborative editing. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// RGA (Replicated Growable Array) represents the document as a tree of characters: every inserted
// character carries a unique id and the id of the character it was typed after, so an insertion is
// a new child of that reference. The visible order is the depth-first traversal of that tree, where
// the children of a node are ordered by a total order on ids (higher counter first, then replica
// name), which is what makes the order the SAME on every replica no matter in which order the
// operations arrive. Deletions are tombstones rather than removals - a later insertion may still
// reference a deleted character, so the node has to stay in the tree. An operation whose reference
// has not arrived yet is buffered and retried after each successful integration (causal delivery).
//
// Logoot is included as the position-based alternative: a character's position is a list of
// (digit, site) elements compared lexicographically with the missing tail treated as the minimum,
// so a position can always be allocated strictly between two neighbours by splitting the digit
// range and falling back to the site dimension. Positions are allocated with a deterministic PRNG
// over a large space, exactly as Logoot does, so that two concurrent insertions at the same place
// almost never collide and the order stays dense.
//
// What is simplified (documented per the authoring contract): a single process with scripted and
// permuted delivery orders instead of a network, no non-interleaving/intention preservation
// guarantee (plain RGA does interleave concurrent runs, which is why later algorithms exist), and
// Logoot's allocation is a scaled-down digit/site space. Verified invariants: replicas converge to
// the same text for EVERY delivery order, tombstones keep deleted references usable, buffered
// operations are released once their reference arrives, and Logoot positions preserve the intended
// order and stay strictly between their neighbours.
using System;
using System.Collections.Generic;
using System.Text;

#nullable enable

internal sealed class RgaChar
{
    public readonly string Id;          // "counter@replica"
    public readonly string? RefId;      // the character this one was typed after (null = head)
    public readonly char Value;
    public readonly int Counter;
    public readonly string Replica;
    public bool Deleted;

    public RgaChar(int counter, string replica, string? refId, char value)
    {
        Counter = counter;
        Replica = replica;
        Id = counter + "@" + replica;
        RefId = refId;
        Value = value;
    }
}

internal sealed class RgaOp
{
    public bool IsInsert;
    public RgaChar? Character;
    public string TargetId = "";
}

internal sealed class RgaReplica
{
    private readonly List<RgaChar> items = new List<RgaChar>();
    private readonly Dictionary<string, RgaChar> byId = new Dictionary<string, RgaChar>();
    private readonly List<RgaOp> deferred = new List<RgaOp>();
    private int counter;

    public readonly string Name;
    public int BufferedOps;
    public int IntegratedOps;

    public RgaReplica(string name) { Name = name; }

    public string Text()
    {
        var sb = new StringBuilder();
        foreach (RgaChar item in items) if (!item.Deleted) sb.Append(item.Value);
        return sb.ToString();
    }

    public int Length => items.Count;

    public RgaOp LocalInsert(string? refId, char value)
    {
        var character = new RgaChar(++counter, Name, refId, value);
        Integrate(character);
        return new RgaOp { IsInsert = true, Character = character };
    }

    public RgaOp LocalDelete(string targetId)
    {
        if (byId.TryGetValue(targetId, out RgaChar? target)) target.Deleted = true;
        return new RgaOp { IsInsert = false, TargetId = targetId };
    }

    public RgaOp InsertAt(int visibleIndex, char value)
    {
        string? refId = null;
        int seen = 0;
        foreach (RgaChar item in items)
        {
            if (item.Deleted) continue;
            if (seen == visibleIndex) break;
            refId = item.Id;
            seen++;
        }
        return LocalInsert(refId, value);
    }

    public string IdAt(int visibleIndex)
    {
        int seen = 0;
        foreach (RgaChar item in items)
        {
            if (item.Deleted) continue;
            if (seen == visibleIndex) return item.Id;
            seen++;
        }
        throw new Exception("no visible character at index " + visibleIndex);
    }

    /// <summary>Apply a remote operation, buffering it while its reference is missing.</summary>
    public bool Apply(RgaOp op)
    {
        if (op.IsInsert)
        {
            RgaChar character = op.Character!;
            if (byId.ContainsKey(character.Id)) return true;                       // already integrated
            if (character.RefId != null && !byId.ContainsKey(character.RefId))
            {
                if (!deferred.Contains(op)) { deferred.Add(op); BufferedOps++; }
                return false;
            }
            Integrate(character);
        }
        else
        {
            if (!byId.ContainsKey(op.TargetId))
            {
                if (!deferred.Contains(op)) { deferred.Add(op); BufferedOps++; }
                return false;
            }
            byId[op.TargetId].Deleted = true;
            IntegratedOps++;
        }
        DrainDeferred();
        return true;
    }

    private void DrainDeferred()
    {
        bool progress = true;
        while (progress)
        {
            progress = false;
            for (int i = 0; i < deferred.Count; i++)
            {
                RgaOp op = deferred[i];
                if (op.IsInsert)
                {
                    RgaChar character = op.Character!;
                    if (character.RefId != null && !byId.ContainsKey(character.RefId)) continue;
                    if (byId.ContainsKey(character.Id)) { deferred.RemoveAt(i); progress = true; break; }
                    Integrate(character);
                }
                else
                {
                    if (!byId.ContainsKey(op.TargetId)) continue;
                    byId[op.TargetId].Deleted = true;
                    IntegratedOps++;
                }
                deferred.RemoveAt(i);
                progress = true;
                break;
            }
        }
    }

    /// <summary>Depth-first integration: after the reference, past every sibling that outranks us.</summary>
    private void Integrate(RgaChar incoming)
    {
        // Each replica keeps its own copy: operations must never share mutable character objects.
        var character = new RgaChar(incoming.Counter, incoming.Replica, incoming.RefId, incoming.Value);
        int start = 0;
        if (character.RefId != null) start = items.IndexOf(byId[character.RefId]) + 1;
        int i = start;
        while (i < items.Count)
        {
            RgaChar candidate = items[i];
            bool inSubtree = character.RefId == null ? candidate.RefId == null : IsDescendantOf(candidate, character.RefId);
            if (!inSubtree) break;
            if (candidate.RefId == character.RefId && RanksBefore(candidate, character))
            {
                i++;
                while (i < items.Count && IsDescendantOf(items[i], candidate.Id)) i++;
                continue;
            }
            break;
        }
        items.Insert(i, character);
        byId[character.Id] = character;
        IntegratedOps++;
    }

    private bool IsDescendantOf(RgaChar candidate, string ancestorId)
    {
        string? walk = candidate.RefId;
        while (walk != null)
        {
            if (walk == ancestorId) return true;
            walk = byId[walk].RefId;
        }
        return false;
    }

    /// <summary>Total order on siblings: the higher counter first, then the lower replica name.</summary>
    private static bool RanksBefore(RgaChar a, RgaChar b)
    {
        if (a.Counter != b.Counter) return a.Counter > b.Counter;
        return string.CompareOrdinal(a.Replica, b.Replica) < 0;
    }
}

internal sealed class LogootPosition
{
    public readonly List<int[]> Elements = new List<int[]>();     // (digit, site)

    public const int Base = 1 << 20;
    public const int MaxSite = 1 << 20;

    public static LogootPosition Start()
    {
        var start = new LogootPosition();
        start.Elements.Add(new[] { 0, 0 });
        return start;
    }

    public static LogootPosition End()
    {
        var end = new LogootPosition();
        end.Elements.Add(new[] { Base - 1, MaxSite });
        return end;
    }

    /// <summary>Lexicographic on (digit, site); a missing element counts as the minimum (0, 0).</summary>
    public static int Compare(LogootPosition a, LogootPosition b)
    {
        int length = Math.Max(a.Elements.Count, b.Elements.Count);
        for (int i = 0; i < length; i++)
        {
            int[] left = i < a.Elements.Count ? a.Elements[i] : new[] { 0, 0 };
            int[] right = i < b.Elements.Count ? b.Elements[i] : new[] { 0, 0 };
            if (left[0] != right[0]) return left[0] < right[0] ? -1 : 1;
            if (left[1] != right[1]) return left[1] < right[1] ? -1 : 1;
        }
        return 0;
    }

    /// <summary>Allocate a position strictly between p and q (p must be strictly less than q).</summary>
    public static LogootPosition Between(LogootPosition p, LogootPosition q, ref long seed)
    {
        var result = new LogootPosition();
        int i = 0;
        while (true)
        {
            int[] low = i < p.Elements.Count ? p.Elements[i] : new[] { 0, 0 };
            int[] high = i < q.Elements.Count ? q.Elements[i] : new[] { Base - 1, MaxSite };
            if (low[0] > high[0] || (low[0] == high[0] && low[1] > high[1]))
                throw new Exception("the interval is empty: p is not below q");
            if (high[0] - low[0] >= 2)
            {
                int digit = low[0] + 1 + (int)(Next(ref seed) % (uint)(high[0] - low[0] - 1));
                result.Elements.Add(new[] { digit, 1 + (int)(Next(ref seed) % MaxSite) });
                return result;
            }
            if (high[0] - low[0] == 1)
            {
                // Same digit as the lower bound with a larger site keeps us below the upper digit.
                int site = low[1] + 1 + (int)(Next(ref seed) % 4096);
                result.Elements.Add(new[] { low[0], site });
                return result;
            }
            if (high[1] - low[1] >= 2)
            {
                int site = low[1] + 1 + (int)(Next(ref seed) % (uint)(high[1] - low[1] - 1));
                result.Elements.Add(new[] { low[0], site });
                return result;
            }
            // The interval is too tight at this level: copy the lower bound and go one level deeper.
            result.Elements.Add(new[] { low[0], low[1] });
            i++;
            if (i > 24) throw new Exception("position space exhausted");
        }
    }

    private static long Next(ref long seed)
    {
        seed = unchecked(seed * 6364136223846793005L + 1442695040888963407L);
        return (long)((ulong)seed >> 33);
    }

    public string Describe()
    {
        var sb = new StringBuilder();
        foreach (int[] element in Elements) sb.Append('(').Append(element[0]).Append(':').Append(element[1]).Append(')');
        return sb.ToString();
    }
}

internal sealed class LogootChar
{
    public LogootPosition Position = LogootPosition.Start();
    public char Value;
    public bool Deleted;
}

internal sealed class LogootReplica
{
    private readonly List<LogootChar> items = new List<LogootChar>();
    private long seed;

    public readonly string Name;
    public LogootReplica(string name, long seed)
    {
        Name = name;
        this.seed = seed;
    }

    public string Text()
    {
        var sb = new StringBuilder();
        foreach (LogootChar item in items) if (!item.Deleted) sb.Append(item.Value);
        return sb.ToString();
    }

    public LogootPosition InsertAt(int visibleIndex, char value)
    {
        LogootPosition left = visibleIndex == 0 ? LogootPosition.Start() : VisibleAt(visibleIndex - 1).Position;
        LogootPosition right = visibleIndex >= VisibleCount() ? LogootPosition.End() : VisibleAt(visibleIndex).Position;
        LogootPosition position = LogootPosition.Between(left, right, ref seed);
        var character = new LogootChar { Position = position, Value = value };
        int insertAt = 0;
        while (insertAt < items.Count && LogootPosition.Compare(items[insertAt].Position, position) < 0) insertAt++;
        items.Insert(insertAt, character);
        return position;
    }

    public void InsertWithPosition(LogootPosition position, char value)
    {
        int insertAt = 0;
        while (insertAt < items.Count && LogootPosition.Compare(items[insertAt].Position, position) < 0) insertAt++;
        items.Insert(insertAt, new LogootChar { Position = position, Value = value });
    }

    private int VisibleCount()
    {
        int count = 0;
        foreach (LogootChar item in items) if (!item.Deleted) count++;
        return count;
    }

    private LogootChar VisibleAt(int visibleIndex)
    {
        int seen = 0;
        foreach (LogootChar item in items)
        {
            if (item.Deleted) continue;
            if (seen == visibleIndex) return item;
            seen++;
        }
        throw new Exception("no visible character at index " + visibleIndex);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int NextSeed(int seed) => (seed * 1103515245 + 12345) & 0x7FFFFFFF;

    private static List<int[]> Permutations(int count)
    {
        var result = new List<int[]>();
        var current = new int[count];
        for (int i = 0; i < count; i++) current[i] = i;
        Permute(current, 0, result);
        return result;
    }

    private static void Permute(int[] values, int index, List<int[]> output)
    {
        if (index == values.Length)
        {
            output.Add((int[])values.Clone());
            return;
        }
        for (int i = index; i < values.Length; i++)
        {
            (values[index], values[i]) = (values[i], values[index]);
            Permute(values, index + 1, output);
            (values[index], values[i]) = (values[i], values[index]);
        }
    }

    private static void LocalTyping()
    {
        var replica = new RgaReplica("A");
        replica.LocalInsert(null, 'h');
        replica.LocalInsert(replica.IdAt(0), 'i');
        Check(replica.Text() == "hi", "typed locally, got " + replica.Text());
        RgaOp op = replica.InsertAt(1, 'X');
        Check(replica.Text() == "hXi", "insert in the middle, got " + replica.Text());
        Check(op.IsInsert, "an insert operation was produced");
        replica.LocalDelete(replica.IdAt(1));
        Check(replica.Text() == "hi", "the tombstoned character disappears");
        Check(replica.Length == 3, "but the tombstone stays in the sequence");
        Check(replica.Apply(op), "re-applying an already integrated op is a no-op");
        Check(replica.Text() == "hi", "and does not duplicate the character");
    }

    private static void ConcurrentTypingConverges()
    {
        // Two replicas type different words starting from the same base text.
        var baseText = new RgaReplica("base");
        RgaOp baseOp = baseText.InsertAt(0, '|');
        var a = new RgaReplica("A");
        var b = new RgaReplica("B");
        foreach (RgaReplica target in new[] { a, b }) target.Apply(baseOp);

        var ops = new List<RgaOp>();
        ops.Add(a.InsertAt(1, 'x'));
        ops.Add(a.InsertAt(2, 'y'));
        ops.Add(b.InsertAt(0, 'p'));
        ops.Add(b.InsertAt(1, 'q'));
        foreach (RgaOp op in ops)
        {
            Check(a.Apply(op), "replica A accepted " + Describe(op));
            Check(b.Apply(op), "replica B accepted " + Describe(op));
        }
        Check(a.Text() == b.Text(), "both replicas converged: " + a.Text() + " vs " + b.Text());
        Check(a.Text().Length == 5, "five characters, got " + a.Text());
        Check(a.Text().Contains("x") && a.Text().Contains("y") && a.Text().Contains("p") && a.Text().Contains("q"),
            "every character survived: " + a.Text());
    }

    private static string Describe(RgaOp op) => op.IsInsert ? "insert " + op.Character!.Value : "delete " + op.TargetId;

    private static void DeliveryOrderDoesNotMatter()
    {
        // Produce a fixed set of operations, then apply them in many different orders.
        var source = new RgaReplica("S");
        var ops = new List<RgaOp>
        {
            source.InsertAt(0, 'a'),
            source.InsertAt(1, 'b'),
            source.InsertAt(2, 'c'),
            source.InsertAt(1, 'X'),
            source.LocalDelete(source.IdAt(0)),
        };
        Check(source.Text() == "Xbc", "the source text after the local edits, got " + source.Text());

        string? reference = null;
        int orders = 0;
        var subset = new List<RgaOp>(ops);
        foreach (int[] order in Permutations(subset.Count))
        {
            var replica = new RgaReplica("R");
            foreach (int index in order) replica.Apply(subset[index]);
            reference ??= replica.Text();
            Check(replica.Text() == reference, "order " + string.Join(",", order) + " gives " + replica.Text()
                + ", expected " + reference);
            orders++;
        }
        Check(reference == "Xbc", "every order produces the intended text: " + reference);
        Check(orders == 120, "all 120 permutations were tried");
    }

    private static void TombstoneKeepsTheReferenceUsable()
    {
        var a = new RgaReplica("A");
        RgaOp first = a.InsertAt(0, 'a');
        RgaOp second = a.InsertAt(1, 'b');
        var b = new RgaReplica("B");
        b.Apply(first);
        b.Apply(second);
        RgaOp deleteFirst = b.LocalDelete(b.IdAt(0));
        Check(b.Text() == "b", "B deleted the first character");
        // A inserts a character that references the deleted one, concurrently.
        RgaOp afterDeleted = a.InsertAt(1, 'Z');
        Check(a.Text() == "aZb", "A typed after the (still alive for it) first character, got " + a.Text()
            + " items=" + a.Length);

        Check(a.Apply(deleteFirst), "A learns about the deletion");
        Check(b.Apply(afterDeleted), "B learns about the insertion into a tombstone position");
        Check(a.Text() == b.Text(), "both replicas converge: " + a.Text() + " vs " + b.Text());
        Check(a.Text() == "Zb", "the insertion survives the tombstone of its reference: " + a.Text());
        Check(a.Length == 3 && b.Length == 3, "three nodes including the tombstone");
    }

    private static void MissingReferencesAreBuffered()
    {
        var source = new RgaReplica("S");
        RgaOp first = source.InsertAt(0, 'a');
        RgaOp second = source.InsertAt(1, 'b');       // references the first
        var replica = new RgaReplica("R");
        Check(replica.Apply(second) == false, "an operation whose reference is missing is buffered");
        Check(replica.BufferedOps == 1, "one buffered operation");
        Check(replica.Text() == "", "nothing visible yet");
        Check(replica.Apply(first), "the dependency arrives");
        Check(replica.Text() == "ab", "the buffered operation was released in the right order: " + replica.Text());
        Check(replica.Length == 2, "both characters integrated");
        Check(replica.Apply(second) == true, "re-delivering is idempotent");
        Check(replica.Length == 2, "still two characters");
    }

    private static void LogootPositionsStayOrderedAndDense()
    {
        long seed = 20260116;
        var positions = new List<LogootPosition> { LogootPosition.Start(), LogootPosition.End() };
        var values = new List<char> { '\u0000', 'z' };
        for (int step = 0; step < 400; step++)
        {
            int slot = (int)((ulong)(seed * 1103515245L + 12345L) % (ulong)(positions.Count - 1));
            seed = unchecked(seed * 6364136223846793005L + 1442695040888963407L);
            LogootPosition inserted = LogootPosition.Between(positions[slot], positions[slot + 1], ref seed);
            Check(LogootPosition.Compare(positions[slot], inserted) < 0, "the new position is above its left neighbour");
            Check(LogootPosition.Compare(inserted, positions[slot + 1]) < 0, "the new position is below its right neighbour");
            positions.Insert(slot + 1, inserted);
            values.Insert(slot + 1, 'a');
        }
        for (int i = 0; i + 1 < positions.Count; i++)
        {
            Check(LogootPosition.Compare(positions[i], positions[i + 1]) < 0,
                "strictly increasing at " + i + ": " + positions[i].Describe() + " then " + positions[i + 1].Describe());
        }
        Check(positions.Count == 402, "every insertion succeeded, count=" + positions.Count);
        Check(LogootPosition.Compare(LogootPosition.Start(), LogootPosition.End()) < 0, "start is below end");
        Check(LogootPosition.Compare(LogootPosition.Start(), LogootPosition.Start()) == 0, "positions compare equal to themselves");
        Console.WriteLine("logoot: 400 dense insertions, longest position has "
            + LongestPosition(positions) + " elements");
    }

    private static int LongestPosition(List<LogootPosition> positions)
    {
        int longest = 0;
        foreach (LogootPosition position in positions) longest = Math.Max(longest, position.Elements.Count);
        return longest;
    }

    private static void LogootReplicasConvergeOnOrder()
    {
        var a = new LogootReplica("A", 11);
        var b = new LogootReplica("B", 22);
        LogootPosition p0 = a.InsertAt(0, 'H');
        LogootPosition p1 = a.InsertAt(1, 'i');
        LogootPosition p2 = b.InsertAt(0, 'Y');
        LogootPosition p3 = b.InsertAt(1, 'o');
        // Each replica learns the other's characters with the positions that came with them.
        a.InsertWithPosition(p2, 'Y');
        a.InsertWithPosition(p3, 'o');
        b.InsertWithPosition(p0, 'H');
        b.InsertWithPosition(p1, 'i');
        Check(a.Text() == b.Text(), "the position order is identical on both replicas: " + a.Text() + " vs " + b.Text());
        Check(a.Text().Length == 4, "all four characters are present: " + a.Text());
        Check(a.Text().StartsWith("Y") || a.Text().StartsWith("H"),
            "the concurrent runs keep their internal order: " + a.Text());

        var typed = new LogootReplica("C", 33);
        typed.InsertAt(0, 'a');
        typed.InsertAt(1, 'b');
        typed.InsertAt(1, 'X');
        Check(typed.Text() == "aXb", "a local insert lands in the intended place: " + typed.Text());
        typed.InsertAt(0, '0');
        Check(typed.Text() == "0aXb", "inserting at the head works too: " + typed.Text());
        Console.WriteLine("logoot replicas: " + typed.Text());
    }

    public static void Main()
    {
        LocalTyping();
        ConcurrentTypingConverges();
        DeliveryOrderDoesNotMatter();
        TombstoneKeepsTheReferenceUsable();
        MissingReferencesAreBuffered();
        LogootPositionsStayOrderedAndDense();
        LogootReplicasConvergeOnOrder();
        Console.WriteLine("PASS");
    }
}
