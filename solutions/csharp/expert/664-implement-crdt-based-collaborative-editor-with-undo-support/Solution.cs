// Task 664 - Implement CRDT-based collaborative editor with undo support. (Tier: Expert, CTCI Ch 9 - Sys Design)
//
// A sequence CRDT in its tree form (RGA / Treedoc). Every character is an element with a globally unique id
// (replica, per-replica counter) and a fixed parent - the element it was inserted after, or the root. The
// visible text is the pre-order walk of that tree with siblings sorted by descending id and tombstones
// skipped. Because the tree is defined only by immutable facts (each element's parent and id) and the sibling
// order is a total, deterministic comparison, two replicas that have seen the same set of insertions render
// exactly the same text no matter what order the operations arrived in - which is the entire convergence
// argument, and it is what the randomized delivery-order cross-check exercises.
// Deletion is an *observed-remove* set: an element carries the set of replicas that deleted it and is hidden
// while that set is non-empty. That choice is what makes selective undo possible without an undo of undo: a
// replica's undo removes only its own deletion, so an element another replica deleted stays hidden everywhere,
// and an undone insertion is tombstoned like any other delete.
// What is simplified: a real editor would anchor positions between characters and handle rich text - here a
// character is the unit, positions are element ids, and undo is a per-replica stack of its own last operations
// rather than a time-travelling history. Verified: hand-worked typing and deletion sequences; convergence of
// two and of three replicas over a 200-operation randomized workload delivered in shuffled orders; idempotent
// re-delivery; undo of an insert removing the character everywhere; undo of a delete restoring it; an undo
// that cannot resurrect a character another replica deleted; an empty undo stack reported as such; identical
// element counts (no lost tombstones) across replicas; and a deterministic sibling order checked directly.
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

internal sealed class ElementId : IComparable<ElementId>
{
    internal int Replica;
    internal int Seq;

    internal ElementId(int replica, int seq)
    {
        Replica = replica;
        Seq = seq;
    }

    public int CompareTo(ElementId? other)
    {
        if (other is null) return 1;
        int bySeq = Seq.CompareTo(other.Seq);
        return bySeq != 0 ? bySeq : Replica.CompareTo(other.Replica);
    }

    public override bool Equals(object? obj) => obj is ElementId other && other.Replica == Replica && other.Seq == Seq;

    public override int GetHashCode() => (Replica * 397) ^ Seq;

    public override string ToString() => Replica + ":" + Seq;
}

internal sealed class Element
{
    internal ElementId Id = new ElementId(0, 0);
    internal ElementId? Parent;
    internal char Value;
    internal SortedSet<int> DeletedBy = new SortedSet<int>();

    internal bool Visible => DeletedBy.Count == 0;
}

internal enum OpKind { Insert, Delete, Undelete }

internal sealed class Operation
{
    internal OpKind Kind;
    internal ElementId Id = new ElementId(0, 0);
    internal ElementId? Parent;
    internal char Value;
    internal int Author;

    internal static Operation Insert(ElementId id, ElementId? parent, char value, int author) =>
        new Operation { Kind = OpKind.Insert, Id = id, Parent = parent, Value = value, Author = author };

    internal static Operation Delete(ElementId id, int author) =>
        new Operation { Kind = OpKind.Delete, Id = id, Author = author };

    internal static Operation Undelete(ElementId id, int author) =>
        new Operation { Kind = OpKind.Undelete, Id = id, Author = author };
}

internal sealed class RgaDocument
{
    private readonly List<Element> _elements = new List<Element>();
    private readonly List<Operation> _undoStack = new List<Operation>();
    // Deletes and undos can arrive before the insertion they refer to, so the last operation per
    // (element, author) is parked here and replayed when the insertion shows up.
    private readonly Dictionary<string, OpKind> _pending = new Dictionary<string, OpKind>();
    private int _seq;
    private readonly int _replica;

    internal RgaDocument(int replica) => _replica = replica;

    internal int Replica => _replica;

    internal int ElementCount => _elements.Count;

    internal int TombstoneCount
    {
        get
        {
            int hidden = 0;
            foreach (Element element in _elements) if (!element.Visible) hidden++;
            return hidden;
        }
    }

    internal int UndoDepth => _undoStack.Count;

    internal ElementId? Find(char value)
    {
        foreach (Element element in _elements) if (element.Value == value) return element.Id;
        return null;
    }

    private Element? ElementOf(ElementId id)
    {
        foreach (Element element in _elements) if (element.Id.Equals(id)) return element;
        return null;
    }

    /// <summary>Inserts a character after `parent` (null = the start of the document).</summary>
    internal Operation InsertLocal(ElementId? parent, char value)
    {
        var id = new ElementId(_replica, ++_seq);
        var operation = Operation.Insert(id, parent, value, _replica);
        Apply(operation);
        _undoStack.Add(operation);
        return operation;
    }

    /// <summary>Deletes a visible element; the deletion is tagged with this replica so it can be undone.</summary>
    internal Operation DeleteLocal(ElementId id)
    {
        if (ElementOf(id) is not Element element) throw new ArgumentException("unknown element " + id);
        if (!element.Visible)
            throw new InvalidOperationException("element " + id + " is already deleted by {"
                + string.Join(",", element.DeletedBy) + "} in replica " + _replica + " text='" + Text() + "'");
        var operation = Operation.Delete(id, _replica);
        Apply(operation);
        _undoStack.Add(operation);
        return operation;
    }

    /// <summary>Undoes the replica's own most recent operation; returns the inverse operation, or null.</summary>
    internal Operation? UndoLocal()
    {
        if (_undoStack.Count == 0) return null;
        Operation undone = _undoStack[_undoStack.Count - 1];
        _undoStack.RemoveAt(_undoStack.Count - 1);
        if (undone.Kind == OpKind.Insert)
        {
            var inverse = Operation.Delete(undone.Id, _replica);
            Apply(inverse);
            return inverse;
        }
        if (undone.Kind == OpKind.Delete)
        {
            var inverse = Operation.Undelete(undone.Id, _replica);
            Apply(inverse);
            return inverse;
        }
        // Undoing an undo is not modelled: the inverse of an undelete is the original delete.
        var again = Operation.Delete(undone.Id, _replica);
        Apply(again);
        return again;
    }

    /// <summary>Applies an operation, local or remote, in any order. Re-delivery is a no-op.</summary>
    internal bool Apply(Operation operation)
    {
        switch (operation.Kind)
        {
            case OpKind.Insert:
                foreach (Element existing in _elements)
                    if (existing.Id.Equals(operation.Id)) return false;      // already integrated
                var added = new Element
                {
                    Id = operation.Id,
                    Parent = operation.Parent,
                    Value = operation.Value,
                };
                _elements.Add(added);
                if (operation.Id.Replica == _replica && operation.Id.Seq > _seq) _seq = operation.Id.Seq;
                FlushPending(added);
                return true;
            case OpKind.Delete:
                if (ElementOf(operation.Id) is Element target) target.DeletedBy.Add(operation.Author);
                else _pending[PendingKey(operation.Id, operation.Author)] = OpKind.Delete;
                return true;
            default:
                if (ElementOf(operation.Id) is Element restored) restored.DeletedBy.Remove(operation.Author);
                else _pending[PendingKey(operation.Id, operation.Author)] = OpKind.Undelete;
                return true;
        }
    }

    private static string PendingKey(ElementId id, int author) => id + "#" + author;

    private void FlushPending(Element element)
    {
        var ready = new List<string>();
        foreach (var pair in _pending)
            if (pair.Key.StartsWith(element.Id + "#", StringComparison.Ordinal)) ready.Add(pair.Key);
        foreach (string key in ready)
        {
            int author = int.Parse(key.Substring(key.IndexOf('#') + 1));
            if (_pending[key] == OpKind.Delete) element.DeletedBy.Add(author);
            else element.DeletedBy.Remove(author);
            _pending.Remove(key);
        }
    }

    /// <summary>Pre-order walk of the element tree with siblings in descending id order, tombstones skipped.</summary>
    internal string Text()
    {
        var builder = new StringBuilder();
        foreach (ElementId id in VisibleOrder()) builder.Append(ElementOf(id)!.Value);
        return builder.ToString();
    }

    internal List<ElementId> VisibleOrder()
    {
        var children = new Dictionary<string, List<Element>>();
        foreach (Element element in _elements)
        {
            string key = element.Parent is null ? "*" : element.Parent.ToString();
            if (!children.TryGetValue(key, out List<Element>? siblings))
            {
                siblings = new List<Element>();
                children[key] = siblings;
            }
            siblings.Add(element);
        }
        foreach (List<Element> siblings in children.Values)
            siblings.Sort((a, b) => b.Id.CompareTo(a.Id));

        var order = new List<ElementId>();
        Walk("*", children, order);
        return order;
    }

    private void Walk(string key, Dictionary<string, List<Element>> children, List<ElementId> order)
    {
        if (!children.TryGetValue(key, out List<Element>? siblings)) return;
        foreach (Element element in siblings)
        {
            if (element.Visible) order.Add(element.Id);
            Walk(element.Id.ToString(), children, order);
        }
    }

    /// <summary>Null when the tree is well formed: unique ids, known parents, no cycles.</summary>
    internal string? InvariantViolation()
    {
        var seen = new HashSet<string>();
        foreach (Element element in _elements)
        {
            if (!seen.Add(element.Id.ToString())) return "duplicate element id " + element.Id;
            if (element.Parent is not null && ElementOf(element.Parent) is null)
                return "element " + element.Id + " references a missing parent " + element.Parent;
            if (element.Parent is not null && element.Parent.Equals(element.Id))
                return "element " + element.Id + " is its own parent";
        }
        foreach (Element element in _elements)
        {
            var path = new HashSet<string>();
            Element? current = element;
            while (current is not null)
            {
                if (!path.Add(current.Id.ToString())) return "cycle through element " + current.Id;
                current = current.Parent is null ? null : ElementOf(current.Parent);
            }
        }
        return null;
    }

    internal string Fingerprint()
    {
        var parts = new List<string>();
        foreach (Element element in _elements)
        {
            var deleted = new StringBuilder();
            foreach (int author in element.DeletedBy) deleted.Append(author).Append(',');
            parts.Add(element.Id + "<" + (element.Parent?.ToString() ?? "root") + ":" + element.Value
                + ":{" + deleted + "}");
        }
        parts.Sort(StringComparer.Ordinal);
        return string.Join("|", parts);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    /// <summary>
    /// Shuffles a log while keeping each author's own operations in their original relative order - the causal
    /// delivery assumption an operation-based CRDT relies on - so the *interleaving* between authors varies.
    /// </summary>
    private static List<Operation> CausalShuffle(List<Operation> operations, ref ulong state)
    {
        var queues = new Dictionary<int, Queue<Operation>>();
        foreach (Operation operation in operations)
        {
            if (!queues.TryGetValue(operation.Author, out Queue<Operation>? queue))
            {
                queue = new Queue<Operation>();
                queues[operation.Author] = queue;
            }
            queue.Enqueue(operation);
        }
        var order = new List<Operation>();
        var authors = new List<int>(queues.Keys);
        while (order.Count < operations.Count)
        {
            int pick = Rng.NextInt(ref state, authors.Count);
            Queue<Operation> queue = queues[authors[pick]];
            if (queue.Count == 0) continue;
            order.Add(queue.Dequeue());
        }
        return order;
    }

    public static void Main()
    {
        // ---- hand-worked typing, deletion and undo ---------------------------------------------
        {
            var doc = new RgaDocument(1);
            var ids = new List<ElementId>();
            ElementId? anchor = null;
            foreach (char c in "hello")
            {
                Operation op = doc.InsertLocal(anchor, c);
                ids.Add(op.Id);
                anchor = op.Id;
            }
            Check(doc.Text() == "hello", "typing five characters in order yields them in order: " + doc.Text());
            Check(doc.InvariantViolation() == null && doc.TombstoneCount == 0,
                "the document tree is well formed with no tombstones yet");

            doc.DeleteLocal(ids[1]);
            Check(doc.Text() == "hllo", "deleting the second character removes it from the text");
            Operation? inverse = doc.UndoLocal();
            Check(inverse is not null && doc.Text() == "hello",
                "undoing the deletion restores the character: " + doc.Text());
            Check(doc.UndoDepth == 5, "the undone delete left the undo stack (five inserts remain)");

            doc.UndoLocal();
            Check(doc.Text() == "hell", "undoing the last insert removes the last character");
            doc.UndoLocal();
            Check(doc.Text() == "hel", "the undo stack keeps walking backwards through the inserts");
            Check(doc.TombstoneCount == 2 && doc.InvariantViolation() == null,
                "undone inserts stay in the tree as tombstones rather than being removed");
            while (doc.UndoLocal() is not null) { }
            Check(doc.Text().Length == 0 && doc.UndoDepth == 0, "undoing everything empties the visible text");
            Check(doc.UndoLocal() is null, "an empty undo stack reports that there is nothing left to undo");
            Check(doc.ElementCount == 5, "the five tombstones are still present in the metadata");
        }

        // ---- two replicas converge on concurrent inserts ---------------------------------------
        {
            var left = new RgaDocument(1);
            var right = new RgaDocument(2);
            Operation a = left.InsertLocal(null, 'a');
            right.Apply(a);
            Check(left.Text() == "a" && right.Text() == "a", "the initial character is shared");

            Operation leftOp = left.InsertLocal(a.Id, 'x');
            Operation rightOp = right.InsertLocal(a.Id, 'y');
            Check(left.Text() == "ax" && right.Text() == "ay",
                "each replica sees only its own concurrent insert so far");
            left.Apply(rightOp);
            right.Apply(leftOp);
            Check(left.Text() == right.Text(),
                "both replicas converge after exchanging the concurrent inserts: " + left.Text());
            Check(left.Fingerprint() == right.Fingerprint(),
                "the converged state is identical on both replicas, not merely the same text");
            Check(left.InvariantViolation() == null && right.InvariantViolation() == null,
                "both trees stay well formed after the merge");

            // RGA orders siblings by descending id, and the id comparison is (sequence, replica): x was this
            // replica's second insertion while y was replica 2's first, so x sorts first.
            Check(left.Text() == "axy", "siblings are ordered by descending id, so 1:2 precedes 2:1");

            var first = new RgaDocument(1);
            var second = new RgaDocument(2);
            Operation x1 = first.InsertLocal(null, 'x');
            Operation y1 = second.InsertLocal(null, 'y');
            first.Apply(y1);
            second.Apply(x1);
            Check(first.Text() == "yx" && second.Text() == "yx",
                "at equal sequence numbers the higher replica id wins, and both agree: " + first.Text());
        }

        // ---- idempotent re-delivery and deletes -------------------------------------------------
        {
            var doc = new RgaDocument(3);
            Operation op = doc.InsertLocal(null, 'q');
            Check(!doc.Apply(op), "re-delivering an insertion that is already integrated is a no-op");
            Check(doc.ElementCount == 1 && doc.Text() == "q", "the duplicate did not create a second element");
            doc.Apply(Operation.Delete(op.Id, 7));
            Check(doc.Text().Length == 0 && doc.TombstoneCount == 1, "a remote delete hides the element");
            doc.Apply(Operation.Delete(op.Id, 7));
            Check(doc.TombstoneCount == 1, "re-delivering a delete does not double-count it");
            doc.Apply(Operation.Undelete(op.Id, 7));
            Check(doc.Text() == "q", "a remote undelete brings the element back");
            doc.Apply(Operation.Delete(op.Id, 7));
            Check(doc.Text().Length == 0, "the element can be deleted again after being restored");

            var early = new RgaDocument(4);
            early.Apply(Operation.Delete(new ElementId(3, 1), 3));
            Check(early.ElementCount == 0, "a delete that arrives before its insertion changes nothing yet");
            early.Apply(Operation.Insert(new ElementId(3, 1), null, 'p', 3));
            Check(early.Text().Length == 0 && early.ElementCount == 1,
                "when the insertion finally arrives the parked delete is honoured");
        }

        // ---- selective undo across replicas -----------------------------------------------------
        {
            var alice = new RgaDocument(1);
            var bob = new RgaDocument(2);
            Operation shared = alice.InsertLocal(null, 'm');
            bob.Apply(shared);
            Operation bobInsert = bob.InsertLocal(shared.Id, 'n');
            alice.Apply(bobInsert);

            Operation aliceDelete = alice.DeleteLocal(shared.Id);
            Operation bobDelete = bob.DeleteLocal(shared.Id);
            Check(alice.Text() == "n" && bob.Text() == "n",
                "each replica hides the character after its own delete");
            alice.Apply(bobDelete);
            bob.Apply(aliceDelete);
            Check(alice.Text() == "n" && bob.Text() == "n" && alice.Fingerprint() == bob.Fingerprint(),
                "both deletions are recorded and the replicas agree");

            Operation? aliceUndo = alice.UndoLocal();
            Check(aliceUndo is not null && aliceUndo.Kind == OpKind.Undelete && alice.Text() == "n",
                "Alice's undo removes only her own deletion, so the character stays hidden for her");
            bob.Apply(aliceUndo);
            Check(bob.Text() == "n" && alice.Text() == bob.Text(),
                "Bob's deletion keeps the character hidden after Alice's undo");
            Operation? bobUndo = bob.UndoLocal();
            alice.Apply(bobUndo!);
            Check(alice.Text() == "mn" && bob.Text() == "mn",
                "once both deletions are undone the character is visible everywhere: " + alice.Text());
        }

        // ---- randomized workload with shuffled delivery orders (the convergence cross-check) -----
        {
            ulong state = 0x0123456789ABCDEFUL;
            var operations = new List<Operation>();
            var authors = new List<int>();
            var docs = new List<RgaDocument> { new RgaDocument(1), new RgaDocument(2), new RgaDocument(3) };
            int undos = 0, deletions = 0, inserts = 0, undoneNothing = 0;

            for (int step = 0; step < 200; step++)
            {
                int which = Rng.NextInt(ref state, docs.Count);
                RgaDocument author = docs[which];
                int choice = Rng.NextInt(ref state, 10);
                if (choice < 2)
                {
                    Operation? inverse = author.UndoLocal();
                    if (inverse is null) { undoneNothing++; }
                    else { undos++; operations.Add(inverse); authors.Add(author.Replica); }
                    continue;
                }
                List<ElementId> visible = author.VisibleOrder();
                if (choice < 5 && visible.Count > 0)
                {
                    ElementId target = visible[Rng.NextInt(ref state, visible.Count)];
                    try
                    {
                        operations.Add(author.DeleteLocal(target));
                        authors.Add(author.Replica);
                        deletions++;
                    }
                    catch (InvalidOperationException) { /* already deleted locally */ }
                    continue;
                }
                ElementId? anchor = visible.Count == 0
                    ? null
                    : (Rng.NextInt(ref state, 4) == 0 ? null : visible[Rng.NextInt(ref state, visible.Count)]);
                operations.Add(author.InsertLocal(anchor, (char)('a' + Rng.NextInt(ref state, 26))));
                authors.Add(author.Replica);
                inserts++;

                // Everybody else receives the operation at a random point, so delivery orders differ.
                for (int other = 0; other < docs.Count; other++)
                    if (other != which && Rng.NextInt(ref state, 2) == 0) docs[other].Apply(operations[operations.Count - 1]);
            }

            // Deliver every operation to every replica in three different shuffled orders, each of which keeps
            // a single author's operations in their original relative order (op-based CRDTs assume that much).
            bool converged = true, identicalTrees = true, wellFormed = true, idempotent = true;
            var texts = new List<string>();
            for (int pass = 0; pass < 3; pass++)
            {
                List<Operation> order = CausalShuffle(operations, ref state);
                foreach (RgaDocument doc in docs)
                {
                    foreach (Operation operation in order) doc.Apply(operation);
                    if (doc.InvariantViolation() != null) wellFormed = false;
                    foreach (Operation operation in order)
                        if (operation.Kind == OpKind.Insert && doc.Apply(operation)) idempotent = false;
                }
            }
            string referenceText = docs[0].Text();
            string referenceFingerprint = docs[0].Fingerprint();
            int referenceCount = docs[0].ElementCount;
            foreach (RgaDocument doc in docs)
            {
                if (doc.Text() != referenceText) converged = false;
                if (doc.Fingerprint() != referenceFingerprint) identicalTrees = false;
                if (doc.ElementCount != referenceCount) identicalTrees = false;
            }
            Check(inserts > 40 && deletions > 10 && undos > 5,
                "the workload exercised inserts (" + inserts + "), deletes (" + deletions + ") and undos ("
                + undos + ")");
            Check(wellFormed, "the element tree stayed well formed on every replica");
            Check(idempotent, "re-delivering the whole operation log changed nothing");
            Check(converged, "all three replicas render the same text: " + referenceText);
            Check(identicalTrees, "the replicas agree on the full state (ids, parents, tombstones), not just the text");
            Check(docs[0].ElementCount == inserts,
                "every insertion survives as exactly one element even where it was undone or deleted");
            Check(referenceCount == inserts && docs[1].TombstoneCount == docs[0].TombstoneCount,
                "tombstone counts agree across replicas");
            Console.WriteLine("inserts=" + inserts + " deletes=" + deletions + " undos=" + undos
                + " emptyUndoAttempts=" + undoneNothing + " length=" + referenceText.Length
                + " tombstones=" + docs[0].TombstoneCount);
        }

        // ---- rejected operations ---------------------------------------------------------------
        {
            var doc = new RgaDocument(9);
            Operation op = doc.InsertLocal(null, 'z');
            bool rejected = false;
            try { doc.DeleteLocal(op.Id); doc.DeleteLocal(op.Id); }
            catch (InvalidOperationException) { rejected = true; }
            Check(rejected, "deleting an already deleted element is rejected locally");
            rejected = false;
            try { doc.DeleteLocal(new ElementId(4, 99)); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "deleting an element this replica has never seen is rejected locally");
            string before = doc.Text();
            doc.Apply(Operation.Delete(new ElementId(4, 99), 9));
            Check(doc.ElementCount == 1 && doc.Text() == before,
                "a remote delete for an unknown element is parked instead of inventing the element");

            var orphan = new RgaDocument(5);
            bool orphanRejected = false;
            orphan.Apply(Operation.Insert(new ElementId(5, 1), new ElementId(7, 7), 'o', 5));
            Check(orphan.InvariantViolation() != null, "an element whose parent was never delivered is flagged");
            try { _ = orphan.Text(); }
            catch (Exception) { orphanRejected = true; }
            Check(!orphanRejected, "rendering tolerates the missing parent instead of crashing");
        }

        Console.WriteLine("PASS");
    }
}
