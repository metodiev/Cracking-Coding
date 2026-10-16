// Task 666 - Implement conflict-free replicated JSON structure (complex CRDT). (Tier: Expert, CTCI Ch 9 - Sys)
//
// A JSON document as a state-based CRDT tree. A node is one of three things: a last-writer-wins register
// (payload plus a stamp), a map (field name -> child node), or an LWW-element set (per element the highest add
// stamp and the highest remove stamp, an element being present when its add stamp is the greater). Because
// every node carries *all* the state needed to resolve a conflict, merging is a pure recursive function:
// registers keep the higher stamp, maps merge field by field (keys that exist on only one side are copied),
// sets keep the higher stamp per element and per direction, and a kind conflict at one field is resolved by the
// birth stamp of the conflicting nodes. The three CRDT laws - commutativity, associativity and idempotence of
// merge - are therefore properties of the implementation, and the file tests them rather than asserts them.
// What is simplified: stamps are a Lamport counter plus a replica id rather than a vector clock, the payload is
// a string (no integers, floats or nesting inside registers), there is no garbage collection of remove stamps
// or tombstones, and arrays are modelled as sets so they are unordered and duplicate-free. Verified: register
// last-writer-wins including the counter/replica tie-break; independent fields of one map both survive; a
// remove only kills an add it outranks, and a later re-add resurrects the element; a tombstone survives a merge
// with a replica that only saw the add; a kind conflict resolves the same way from either side; merging does
// not mutate its inputs; an empty node is an identity; renderings are canonical and order-independent; and a
// 200-operation / 3-replica randomized workload merged in all six orders produces one identical document,
// whose merge is also idempotent. Deterministic splitmix64 input, no clock.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

/// <summary>A Lamport stamp: counter first, replica id as the deterministic tie-break.</summary>
internal sealed class Stamp : IComparable<Stamp>
{
    internal long Counter;
    internal int Replica;

    internal Stamp(long counter, int replica)
    {
        Counter = counter;
        Replica = replica;
    }

    internal static Stamp Zero { get; } = new Stamp(0, 0);

    public int CompareTo(Stamp? other)
    {
        if (other is null) return 1;
        int byCounter = Counter.CompareTo(other.Counter);
        return byCounter != 0 ? byCounter : Replica.CompareTo(other.Replica);
    }

    public override string ToString() => Counter + "@" + Replica;
}

internal enum NodeKind { Register, Map, Set }

internal sealed class CrdtNode
{
    internal NodeKind Kind;
    internal Stamp Birth = Stamp.Zero;
    internal string Value = string.Empty;
    internal Stamp ValueStamp = Stamp.Zero;
    internal SortedDictionary<string, CrdtNode> Fields = new SortedDictionary<string, CrdtNode>(StringComparer.Ordinal);
    internal SortedDictionary<string, Stamp> Adds = new SortedDictionary<string, Stamp>(StringComparer.Ordinal);
    internal SortedDictionary<string, Stamp> Removes = new SortedDictionary<string, Stamp>(StringComparer.Ordinal);

    internal static CrdtNode Register(string value, Stamp stamp) =>
        new CrdtNode { Kind = NodeKind.Register, Birth = stamp, Value = value, ValueStamp = stamp };

    internal static CrdtNode Map(Stamp birth) => new CrdtNode { Kind = NodeKind.Map, Birth = birth };

    internal static CrdtNode Set(Stamp birth) => new CrdtNode { Kind = NodeKind.Set, Birth = birth };

    internal CrdtNode Clone()
    {
        var copy = new CrdtNode { Kind = Kind, Birth = Birth, Value = Value, ValueStamp = ValueStamp };
        foreach (var field in Fields) copy.Fields[field.Key] = field.Value.Clone();
        foreach (var add in Adds) copy.Adds[add.Key] = add.Value;
        foreach (var remove in Removes) copy.Removes[remove.Key] = remove.Value;
        return copy;
    }

    /// <summary>Returns the child map at `field`, creating (and waiting to be filled) one if needed.</summary>
    internal CrdtNode At(string field, Stamp birth) => Field(field, NodeKind.Map, birth);

    /// <summary>Returns the child of the requested shape, creating it if the field is still empty.</summary>
    internal CrdtNode Field(string field, NodeKind expected, Stamp stamp)
    {
        if (Kind != NodeKind.Map) throw new InvalidOperationException("only maps hold fields");
        if (Fields.TryGetValue(field, out CrdtNode? child))
        {
            if (child.Kind != expected)
                throw new InvalidOperationException("field '" + field + "' already holds a " + child.Kind);
            return child;
        }
        CrdtNode created = expected switch
        {
            NodeKind.Register => Register(string.Empty, stamp),
            NodeKind.Set => Set(stamp),
            _ => Map(stamp),
        };
        Fields[field] = created;
        return created;
    }

    internal CrdtNode RegisterField(string field, Stamp stamp) => Field(field, NodeKind.Register, stamp);

    internal CrdtNode SetField(string field, Stamp stamp) => Field(field, NodeKind.Set, stamp);

    internal void Write(string field, string value, Stamp stamp)
    {
        if (Kind != NodeKind.Map) throw new InvalidOperationException("only maps hold fields");
        if (Fields.TryGetValue(field, out CrdtNode? child))
        {
            if (child.Kind != NodeKind.Register)
                throw new InvalidOperationException("field '" + field + "' already holds a " + child.Kind);
            child.SetRegister(value, stamp);
            return;
        }
        Fields[field] = Register(value, stamp);
    }

    /// <summary>
    /// Writes a register value. A field's shape is fixed by its first write: writing a scalar where a map or a
    /// set already lives is a local programming error, not something a merge can repair.
    /// </summary>
    internal void SetRegister(string value, Stamp stamp)
    {
        if (Kind != NodeKind.Register) throw new InvalidOperationException("this field already holds a " + Kind);
        if (stamp.CompareTo(ValueStamp) > 0)
        {
            Value = value;
            ValueStamp = stamp;
        }
    }

    internal void Add(string element, Stamp stamp)
    {
        if (Kind != NodeKind.Set) throw new InvalidOperationException("this field is not a set");
        if (!Adds.TryGetValue(element, out Stamp? current) || stamp.CompareTo(current) > 0) Adds[element] = stamp;
    }

    internal void Remove(string element, Stamp stamp)
    {
        if (Kind != NodeKind.Set) throw new InvalidOperationException("this field is not a set");
        if (element.Length == 0) throw new ArgumentException("an element name cannot be empty");
        if (!Removes.TryGetValue(element, out Stamp? current) || stamp.CompareTo(current) > 0) Removes[element] = stamp;
    }

    internal bool Contains(string element)
    {
        if (Kind != NodeKind.Set) return false;
        if (!Adds.TryGetValue(element, out Stamp? add)) return false;
        return !Removes.TryGetValue(element, out Stamp? remove) || add.CompareTo(remove) > 0;
    }

    internal List<string> Elements()
    {
        var present = new List<string>();
        if (Kind != NodeKind.Set) return present;
        foreach (var add in Adds) if (Contains(add.Key)) present.Add(add.Key);
        return present;
    }

    /// <summary>Pure recursive merge: neither side is modified and the result is the join of both.</summary>
    internal static CrdtNode Merge(CrdtNode a, CrdtNode b)
    {
        if (a.Kind != b.Kind) return a.Birth.CompareTo(b.Birth) >= 0 ? a.Clone() : b.Clone();
        switch (a.Kind)
        {
            case NodeKind.Register:
                return a.ValueStamp.CompareTo(b.ValueStamp) >= 0
                    ? Register(a.Value, a.ValueStamp)
                    : Register(b.Value, b.ValueStamp);
            case NodeKind.Set:
                var set = Set(a.Birth.CompareTo(b.Birth) >= 0 ? a.Birth : b.Birth);
                foreach (var add in a.Adds) Bump(set.Adds, add.Key, add.Value);
                foreach (var add in b.Adds) Bump(set.Adds, add.Key, add.Value);
                foreach (var remove in a.Removes) Bump(set.Removes, remove.Key, remove.Value);
                foreach (var remove in b.Removes) Bump(set.Removes, remove.Key, remove.Value);
                return set;
            default:
                var map = Map(a.Birth.CompareTo(b.Birth) >= 0 ? a.Birth : b.Birth);
                foreach (var field in a.Fields) map.Fields[field.Key] = field.Value.Clone();
                foreach (var field in b.Fields)
                {
                    map.Fields[field.Key] = map.Fields.TryGetValue(field.Key, out CrdtNode? existing)
                        ? Merge(existing, field.Value)
                        : field.Value.Clone();
                }
                return map;
        }
    }

    private static void Bump(SortedDictionary<string, Stamp> target, string key, Stamp stamp)
    {
        if (!target.TryGetValue(key, out Stamp? current) || stamp.CompareTo(current) > 0) target[key] = stamp;
    }

    /// <summary>Canonical rendering: map keys sorted, set elements sorted, so equal states render identically.</summary>
    internal string Render()
    {
        switch (Kind)
        {
            case NodeKind.Register:
                return "\"" + Value + "\"";
            case NodeKind.Set:
                var builder = new StringBuilder("[");
                var present = Elements();
                for (int i = 0; i < present.Count; i++)
                {
                    if (i > 0) builder.Append(',');
                    builder.Append(present[i]);
                }
                return builder.Append(']').ToString();
            default:
                var json = new StringBuilder("{");
                bool first = true;
                foreach (var field in Fields)
                {
                    if (!first) json.Append(',');
                    first = false;
                    json.Append('"').Append(field.Key).Append("\":").Append(field.Value.Render());
                }
                return json.Append('}').ToString();
        }
    }

    internal string Fingerprint()
    {
        var builder = new StringBuilder();
        Describe(builder);
        return builder.ToString();
    }

    private void Describe(StringBuilder builder)
    {
        builder.Append(Kind).Append('(').Append(Birth).Append(')');
        if (Kind == NodeKind.Register) builder.Append('=').Append(ValueStamp).Append(':').Append(Value);
        if (Kind == NodeKind.Set)
        {
            builder.Append('{');
            foreach (var add in Adds) builder.Append('+').Append(add.Key).Append('@').Append(add.Value).Append(';');
            foreach (var remove in Removes) builder.Append('-').Append(remove.Key).Append('@').Append(remove.Value).Append(';');
            builder.Append('}');
        }
        if (Kind == NodeKind.Map)
        {
            builder.Append('[');
            foreach (var field in Fields) { builder.Append(field.Key).Append(':'); field.Value.Describe(builder); builder.Append(','); }
            builder.Append(']');
        }
    }
}

/// <summary>One replica: a document plus its own Lamport clock.</summary>
internal sealed class Replica
{
    internal int Id;
    internal long Clock;
    internal CrdtNode Root;

    internal Replica(int id)
    {
        Id = id;
        Root = CrdtNode.Map(new Stamp(++Clock, id));
    }

    internal Stamp Next() => new Stamp(++Clock, Id);

    internal void Write(string field, string value)
    {
        CrdtNode node = Root.RegisterField(field, Next());
        node.SetRegister(value, Next());
    }

    internal void Add(string field, string element) => Root.SetField(field, Next()).Add(element, Next());

    internal void Remove(string field, string element) => Root.SetField(field, Next()).Remove(element, Next());

    /// <summary>Returns the nested map at `field`, creating it if the field is still empty.</summary>
    internal CrdtNode Nested(string field) => Root.At(field, Next());

    /// <summary>Absorbs another replica's state and advances the clock past everything it saw.</summary>
    internal void MergeIn(CrdtNode other)
    {
        Root = CrdtNode.Merge(Root, other);
        Clock = Math.Max(Clock, Math.Max(other.Birth.Counter,
            Math.Max(other.ValueStamp.Counter, MaxStamp(other))));
    }

    private static long MaxStamp(CrdtNode node)
    {
        long best = 0;
        foreach (var add in node.Adds) best = Math.Max(best, add.Value.Counter);
        foreach (var remove in node.Removes) best = Math.Max(best, remove.Value.Counter);
        foreach (var field in node.Fields) best = Math.Max(best, MaxStamp(field.Value));
        return best;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static string RenderAll(IReadOnlyList<CrdtNode> nodes)
    {
        CrdtNode merged = nodes[0].Clone();
        for (int i = 1; i < nodes.Count; i++) merged = CrdtNode.Merge(merged, nodes[i]);
        return merged.Render();
    }

    public static void Main()
    {
        // ---- registers: last writer wins --------------------------------------------------------
        {
            var left = new Replica(1);
            var right = new Replica(2);
            left.Write("title", "draft");
            right.Write("title", "final");
            right.Clock = left.Clock + 5;
            right.Write("title", "final");
            var merged = CrdtNode.Merge(left.Root, right.Root);
            Check(merged.Render() == "{\"title\":\"final\"}",
                "the higher stamp wins regardless of merging direction: " + merged.Render());
            var reversed = CrdtNode.Merge(right.Root, left.Root);
            Check(reversed.Render() == merged.Render(), "the merge is commutative for registers");
            Check(left.Root.Render() == "{\"title\":\"draft\"}" && right.Root.Render() == "{\"title\":\"final\"}",
                "merging did not mutate either input");

            var first = new Replica(1);
            var second = new Replica(2);
            first.Write("k", "one");
            second.Write("k", "two");
            string a = CrdtNode.Merge(first.Root, second.Root).Render();
            string b = CrdtNode.Merge(second.Root, first.Root).Render();
            Check(a == b, "equal counters fall back to the replica id, so both directions agree: " + a);
            Check(a == "{\"k\":\"two\"}", "the higher replica id wins the tie: " + a);
        }

        // ---- maps: independent fields survive, the same field conflicts --------------------------
        {
            var left = new Replica(1);
            var right = new Replica(2);
            left.Write("a", "L");
            right.Write("b", "R");
            var merged = CrdtNode.Merge(left.Root, right.Root);
            Check(merged.Render() == "{\"a\":\"L\",\"b\":\"R\"}",
                "fields written on different replicas both survive: " + merged.Render());
            right.Clock = left.Clock + 3;
            right.Write("a", "R2");
            var merged2 = CrdtNode.Merge(left.Root, right.Root);
            Check(merged2.Render() == "{\"a\":\"R2\",\"b\":\"R\"}",
                "a newer write to the same field wins while the other field is untouched: " + merged2.Render());

            var nestedLeft = new Replica(1);
            var nestedRight = new Replica(2);
            nestedLeft.Nested("profile").Write("city", "Oslo", nestedLeft.Next());
            nestedRight.Nested("profile").SetField("tags", nestedRight.Next()).Add("tall", nestedRight.Next());
            var nestedMerged = CrdtNode.Merge(nestedLeft.Root, nestedRight.Root);
            Check(nestedMerged.Render() == "{\"profile\":{\"city\":\"Oslo\",\"tags\":[tall]}}",
                "nested maps merge structurally: " + nestedMerged.Render());
        }

        // ---- sets: add / remove ordering and resurrection ---------------------------------------
        {
            var replica = new Replica(1);
            replica.Add("tags", "alpha");
            Stamp addStamp = replica.Root.SetField("tags", Stamp.Zero).Adds["alpha"];
            replica.Remove("tags", "alpha");
            Check(replica.Root.SetField("tags", Stamp.Zero).Render() == "[]", "a fresh add then remove hides the element");

            var staleRemove = new Replica(2);
            staleRemove.Root.SetField("tags", Stamp.Zero).Add("beta", new Stamp(addStamp.Counter + 10, 2));
            staleRemove.Root.SetField("tags", Stamp.Zero).Remove("beta", new Stamp(addStamp.Counter + 1, 2));
            Check(staleRemove.Root.SetField("tags", Stamp.Zero).Contains("beta"),
                "a remove that is older than the add cannot kill the element");

            var reAdd = new Replica(3);
            var set = reAdd.Root.SetField("tags", Stamp.Zero);
            set.Add("gamma", new Stamp(5, 3));
            set.Remove("gamma", new Stamp(6, 3));
            Check(!set.Contains("gamma"), "a newer remove hides the element");
            set.Add("gamma", new Stamp(9, 3));
            Check(set.Contains("gamma"), "a later add resurrects it (remove-wins only until the next add)");

            var onlyAdds = new Replica(4);
            onlyAdds.Root.SetField("tags", Stamp.Zero).Add("gamma", new Stamp(7, 4));
            Check(CrdtNode.Merge(onlyAdds.Root, reAdd.Root).Render() == "{\"tags\":[gamma]}",
                "the merged state keeps the resurrection: " + CrdtNode.Merge(onlyAdds.Root, reAdd.Root).Render());
            Check(CrdtNode.Merge(reAdd.Root, onlyAdds.Root).Render() == "{\"tags\":[gamma]}",
                "and it does so from either direction");

            var tombstone = new Replica(5);
            tombstone.Add("tags", "delta");
            var deleter = new Replica(6);
            deleter.Root.SetField("tags", Stamp.Zero).Add("delta", new Stamp(4, 1));
            deleter.Root.SetField("tags", Stamp.Zero).Remove("delta", new Stamp(9, 6));
            var afterMerge = CrdtNode.Merge(tombstone.Root, deleter.Root);
            Check(afterMerge.Render() == "{\"tags\":[]}",
                "a tombstone outranks the add even though the other replica never saw the remove: "
                + afterMerge.Render());
        }

        // ---- shape conflicts are rejected locally, and resolved defensively at merge time -------------
        {
            var register = new Replica(1);
            register.Write("x", "scalar");
            var map = new Replica(2);
            map.Nested("x").Write("inner", "nested", map.Next());

            string forward = CrdtNode.Merge(register.Root, map.Root).Render();
            string backward = CrdtNode.Merge(map.Root, register.Root).Render();
            Check(forward == backward,
                "two replicas that chose different shapes for one field still merge deterministically: " + forward);
            Check(forward == "{\"x\":\"scalar\"}" || forward == "{\"x\":{\"inner\":\"nested\"}}",
                "the winner is one of the two states, never a mixture: " + forward);

            bool refused = false;
            try { register.Write("x", "again"); register.Nested("x"); }
            catch (InvalidOperationException) { refused = true; }
            Check(refused, "writing one shape over another locally is refused rather than silently reinterpreted");
            refused = false;
            try { map.Add("x", "element"); }
            catch (InvalidOperationException) { refused = true; }
            Check(refused, "adding a set element to a map field is refused locally");

            var empty = CrdtNode.Map(Stamp.Zero);
            var populated = new Replica(3);
            populated.Write("k", "v");
            Check(CrdtNode.Merge(empty, populated.Root).Render() == populated.Root.Render(),
                "an empty map is the identity of merge");
            Check(CrdtNode.Merge(populated.Root, populated.Root).Render() == populated.Root.Render(),
                "merging a state with itself changes nothing (idempotence)");

            var unordered = new Replica(4);
            unordered.Add("s", "b");
            unordered.Add("s", "a");
            var sameSet = new Replica(5);
            sameSet.Add("s", "a");
            sameSet.Add("s", "b");
            Check(CrdtNode.Merge(unordered.Root, sameSet.Root).Render()
                    == CrdtNode.Merge(sameSet.Root, unordered.Root).Render(),
                "set rendering does not depend on insertion order");
            Check(unordered.Root.Render().StartsWith("{") && unordered.Root.Render().EndsWith("}"),
                "the rendered document is canonical JSON-ish text: " + unordered.Root.Render());
        }

        // ---- a randomized 3-replica workload merged in all six orders ----------------------------
        {
            ulong state = 0x5A5A1234DEAD5678UL;
            var replicas = new List<Replica> { new Replica(1), new Replica(2), new Replica(3) };
            // Each field has one fixed shape on every replica; the shapes differ between fields.
            var registers = new[] { "title", "owner", "body" };
            var sets = new[] { "tags", "meta" };
            int writes = 0, adds = 0, removes = 0, nestedWrites = 0;
            for (int step = 0; step < 200; step++)
            {
                Replica replica = replicas[Rng.NextInt(ref state, replicas.Count)];
                if (replica.Id != 1 && Rng.NextInt(ref state, 4) == 0)
                {
                    // Occasionally catch the clock up, as if this replica had heard gossip.
                    Replica other = replicas[Rng.NextInt(ref state, replicas.Count)];
                    replica.Clock = Math.Max(replica.Clock, other.Clock + Rng.NextInt(ref state, 5));
                }
                int choice = Rng.NextInt(ref state, 10);
                if (choice < 4)
                {
                    replica.Write(registers[Rng.NextInt(ref state, registers.Length)], "v" + replica.Id + "-" + step);
                    writes++;
                }
                else if (choice < 7)
                {
                    replica.Add(sets[Rng.NextInt(ref state, sets.Length)], "e" + Rng.NextInt(ref state, 6));
                    adds++;
                }
                else if (choice < 9)
                {
                    replica.Remove(sets[Rng.NextInt(ref state, sets.Length)], "e" + Rng.NextInt(ref state, 6));
                    removes++;
                }
                else
                {
                    CrdtNode profile = replica.Nested("profile");
                    profile.Write("note", "n" + replica.Id + "-" + step, replica.Next());
                    nestedWrites++;
                }
            }

            var nodes = new List<CrdtNode> { replicas[0].Root, replicas[1].Root, replicas[2].Root };
            var renders = new HashSet<string>();
            foreach (List<int> order in new[]
                     {
                         new List<int> { 0, 1, 2 }, new List<int> { 0, 2, 1 }, new List<int> { 1, 0, 2 },
                         new List<int> { 1, 2, 0 }, new List<int> { 2, 0, 1 }, new List<int> { 2, 1, 0 },
                     })
            {
                CrdtNode merged = nodes[order[0]].Clone();
                for (int i = 1; i < order.Count; i++) merged = CrdtNode.Merge(merged, nodes[order[i]]);
                renders.Add(merged.Render());
            }
            Check(renders.Count == 1, "all six merge orders produce one document: " + string.Join(" | ", renders));
            string consensus = RenderAll(nodes);
            Check(renders.Contains(consensus), "the helper merge agrees with the explicit orders");

            var assembled = CrdtNode.Merge(CrdtNode.Merge(nodes[0], nodes[1]), nodes[2]);
            var reassociated = CrdtNode.Merge(nodes[0], CrdtNode.Merge(nodes[1], nodes[2]));
            Check(assembled.Render() == reassociated.Render(), "merge is associative on the real workload");
            Check(CrdtNode.Merge(assembled, assembled).Render() == assembled.Render(),
                "merge is idempotent on the real workload");
            Check(writes > 40 && adds > 20 && removes > 5 && nestedWrites > 5,
                "the workload exercised writes (" + writes + "), adds (" + adds + "), removes (" + removes
                + ") and nested writes (" + nestedWrites + ")");

            // A fourth replica that receives the merged state and then merges again must not change it.
            var latecomer = new Replica(9);
            latecomer.MergeIn(assembled);
            latecomer.Write("title", "late");
            CrdtNode withLate = CrdtNode.Merge(assembled, latecomer.Root);
            latecomer.Clock += 100;
            latecomer.Write("owner", "latecomer");
            CrdtNode final = CrdtNode.Merge(withLate, latecomer.Root);
            Check(final.Render().Contains("latecomer"),
                "a late-joining replica's newest writes are visible after the merge: " + final.Render());
            Check(CrdtNode.Merge(final, assembled).Render().Contains("latecomer"),
                "re-merging an older state cannot roll back the newcomer's write");
            Console.WriteLine("writes=" + writes + " adds=" + adds + " removes=" + removes + " nested="
                + nestedWrites + " document=" + final.Render());
        }

        Console.WriteLine("PASS");
    }
}
