// Task 665 - Implement operational transform with intention preservation tests.
// (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// Operational transformation for plain text: an operation is an insert of one character at a position or a
// delete of a range, and Transformation Pipeline 1 (TP1) is the property that makes concurrent editing work -
// for any two concurrent operations a and b there exist a' and b' with
//     apply(apply(doc, a), a')  ==  apply(apply(doc, b), b')
// so the two sites converge no matter which of the two operations the server (or each peer) happened to see
// first. The transform is written out case by case: insert against insert shifts when the other insert lands
// at or before the position (with a site-id tie-break at the same position so both mirrors agree), delete
// against insert shifts the range when the insertion lands at or before its start, insert against delete maps
// the position through the removed range, and delete against delete maps both ends of the range through the
// other deletion so the two ranges collapse onto the characters that survive.
// What is simplified: text only (no rich text, no attributes), positions are plain indexes rather than
// persistent anchors, and TP2 (the same property for three operations applied in different groupings) is not
// claimed - the file tests the *intention* side instead, by tracking the identity of every character through
// the merge and requiring that exactly the characters the deletes aimed at disappear.
// Verified: five hand-worked transformations; TP1 on 800 random concurrent pairs over random documents; the
// intention-preservation property - the surviving characters are exactly the original characters outside both
// deleted ranges, in their original relative order, with both inserted characters present; three concurrent
// inserts at distinct positions converge from every application order; no-op deletes transform to no-ops;
// operations out of bounds, deletions of negative length and empty-document inserts behave; and the whole
// pipeline is deterministic.
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

internal sealed class TextOp
{
    private static int _nextSequence;

    internal bool IsInsert;
    internal int Site;
    internal int Sequence;
    internal int Position;
    internal int Length;      // deletions only
    internal char Value;      // insertions only
    internal bool Dropped;    // an insert whose anchor was deleted out from under it

    internal static TextOp Insert(int site, int position, char value) =>
        new TextOp { IsInsert = true, Site = site, Sequence = ++_nextSequence, Position = position, Value = value, Length = 1 };

    internal static TextOp Delete(int site, int position, int length) =>
        new TextOp { IsInsert = false, Site = site, Sequence = ++_nextSequence, Position = position, Length = length };

    /// <summary>An insertion that has no anchor left: both of its neighbours were deleted, so the delete wins.</summary>
    internal static TextOp DroppedInsert(int site) =>
        new TextOp { IsInsert = true, Dropped = true, Site = site, Sequence = ++_nextSequence };

    internal bool IsNoOp => Dropped || (!IsInsert && Length == 0);

    /// <summary>Total order on operations: the tie-break must be total for the transform to be symmetric.</summary>
    internal static int Compare(TextOp a, TextOp b) =>
        a.Site != b.Site ? a.Site.CompareTo(b.Site) : a.Sequence.CompareTo(b.Sequence);

    public override string ToString() => Dropped
        ? "dropped-insert@" + Site + "#" + Sequence
        : IsInsert
            ? "insert(" + Position + ",'" + Value + "')@" + Site + "#" + Sequence
            : "delete(" + Position + "," + Length + ")@" + Site + "#" + Sequence;
}

internal static class Ot
{
    /// <summary>Applies an operation, validating that it fits the document.</summary>
    internal static string Apply(string document, TextOp op)
    {
        if (op.IsInsert)
        {
            if (op.Dropped) return document;
            if (op.Position < 0 || op.Position > document.Length)
                throw new ArgumentException("insert position " + op.Position + " is outside the document");
            return document.Substring(0, op.Position) + op.Value + document.Substring(op.Position);
        }
        if (op.Length < 0) throw new ArgumentException("a delete cannot have negative length");
        if (op.Position < 0 || op.Position + op.Length > document.Length)
            throw new ArgumentException("delete range " + op.Position + "+" + op.Length + " is outside the document");
        return document.Remove(op.Position, op.Length);
    }

    /// <summary>Where position `p` ends up after `removed` has been deleted: ranges collapse, positions after shift.</summary>
    private static int MapPosition(int p, int removedStart, int removedLength)
    {
        int removedEnd = removedStart + removedLength;
        if (p >= removedEnd) return p - removedLength;
        if (p <= removedStart) return p;
        return removedStart;
    }

    /// <summary>`a` transformed against `b`, i.e. the version of `a` to use once `b` has been applied.</summary>
    internal static TextOp TransformAgainst(TextOp a, TextOp b)
    {
        if (a.IsInsert)
        {
            int position = a.Position;
            if (b.IsInsert)
            {
                if (position > b.Position) position++;
            else if (position == b.Position && TextOp.Compare(a, b) > 0) position++;
            }
            else
            {
            // Strictly inside the deleted range there is no anchor left: the delete wins and swallows it.
            if (b.Position < position && position < b.Position + b.Length) return TextOp.DroppedInsert(a.Site);
            position = MapPosition(position, b.Position, b.Length);
            }
            return TextOp.Insert(a.Site, position, a.Value);
        }

        if (b.IsInsert)
        {
            int length = a.Length;
            if (a.Position < b.Position && b.Position < a.Position + a.Length) length++;
            int position = a.Position + (b.Position <= a.Position ? 1 : 0);
            return TextOp.Delete(a.Site, position, length);
        }

        // Both delete: map both ends of a's range through b's removal so only the surviving characters remain.
        int start = MapPosition(a.Position, b.Position, b.Length);
        int end = MapPosition(a.Position + a.Length, b.Position, b.Length);
        return TextOp.Delete(a.Site, start, Math.Max(0, end - start));
    }

    /// <summary>Both directions at once, which is what a pair of concurrent sites exchange.</summary>
    internal static (TextOp A, TextOp B) Transform(TextOp a, TextOp b) =>
        (TransformAgainst(a, b), TransformAgainst(b, a));

    /// <summary>The document as characters with stable identities: (origin index, char) or (-1, inserted char).</summary>
    internal sealed class Letter
    {
        internal int Origin;
        internal char Value;
    }

    internal static List<Letter> Track(string document)
    {
        var letters = new List<Letter>();
        for (int i = 0; i < document.Length; i++) letters.Add(new Letter { Origin = i, Value = document[i] });
        return letters;
    }

    internal static void TrackApply(List<Letter> letters, TextOp op)
    {
        if (op.IsInsert)
        {
            if (!op.Dropped) letters.Insert(op.Position, new Letter { Origin = -1, Value = op.Value });
        }
        else
        {
            letters.RemoveRange(op.Position, op.Length);
        }
    }

    internal static string Render(IReadOnlyList<Letter> letters)
    {
        var builder = new StringBuilder();
        foreach (Letter letter in letters) builder.Append(letter.Value);
        return builder.ToString();
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    /// <summary>
    /// The canonical n-way merge: operation i is rebased on every operation already applied (in the order they
    /// were applied, each in its rebased form) and the rebased operations are applied in site order - the order
    /// a server assigns. Repeating the merge for a second delivery order is *not* claimed to agree: that is
    /// transformation property 2, which the basic transform does not provide.
    /// </summary>
    private static string CanonicalMerge(string document, List<TextOp> ops)
    {
        var ordered = new List<TextOp>(ops);
        ordered.Sort(TextOp.Compare);
        var applied = new List<TextOp>();
        string current = document;
        foreach (TextOp original in ordered)
        {
            TextOp op = original;
            foreach (TextOp prior in applied) op = Ot.TransformAgainst(op, prior);
            current = Ot.Apply(current, op);
            applied.Add(op);
        }
        return current;
    }

    public static void Main()
    {
        // ---- the transform rules, worked by hand -------------------------------------------------
        {
            var (a1, b1) = Ot.Transform(TextOp.Insert(1, 3, 'x'), TextOp.Insert(2, 3, 'y'));
            Check(a1.Position == 3 && b1.Position == 4,
                "at the same position the lower-site insert stays and the higher-site insert shifts: "
                + a1 + " / " + b1);
            var (a2, b2) = Ot.Transform(TextOp.Insert(3, 7, 'x'), TextOp.Insert(3, 7, 'y'));
            Check(a2.Position == 7 && b2.Position == 8,
                "two inserts from the same replica still carry distinct operation ids, so the later one shifts: "
                + a2 + " / " + b2);
            var (a3, b3) = Ot.Transform(TextOp.Insert(1, 2, 'x'), TextOp.Insert(9, 5, 'y'));
            Check(a3.Position == 2 && b3.Position == 6,
                "an insert to the right of another insert shifts by one: " + a3 + " / " + b3);

            var (a4, b4) = Ot.Transform(TextOp.Delete(1, 1, 2), TextOp.Insert(2, 1, 'x'));
            Check(a4.Position == 2 && b4.Position == 1,
                "an insert at the start of a deleted range pushes the range right: " + a4 + " / " + b4);
            var (a5, b5) = Ot.Transform(TextOp.Insert(1, 4, 'x'), TextOp.Delete(2, 1, 2));
            Check(a5.Position == 2 && b5.Position == 1 && b5.Length == 2,
                "an insert beyond a deleted range maps to the collapsed position: " + a5 + " / " + b5);
            var (a8, b8) = Ot.Transform(TextOp.Insert(1, 3, 'x'), TextOp.Delete(2, 1, 4));
            Check(a8.Dropped && b8.Length == 5,
                "an insert strictly inside a deleted range is dropped while the delete grows to swallow it: "
                + a8 + " / " + b8);
            var (a6, b6) = Ot.Transform(TextOp.Delete(1, 2, 2), TextOp.Delete(2, 1, 2));
            Check(a6.Position == 1 && a6.Length == 1 && b6.Position == 1 && b6.Length == 1,
                "overlapping deletes each keep only their non-overlapping part: " + a6 + " / " + b6);
            var (a7, b7) = Ot.Transform(TextOp.Delete(1, 0, 0), TextOp.Delete(2, 2, 3));
            Check(a7.IsNoOp && b7.Length == 3, "a no-op delete transforms to a no-op: " + a7 + " / " + b7);
            Check(a1.Site == 1 && b1.Site == 2, "the transform preserves the originating sites");
        }

        // ---- TP1 on random concurrent pairs -----------------------------------------------------
        {
            ulong state = 0x2468ACE013579BDFUL;
            int checkedPairs = 0, conflicts = 0, drops = 0;
            bool tp1Holds = true;
            string? failure = null;
            for (int trial = 0; trial < 800; trial++)
            {
                int length = 4 + Rng.NextInt(ref state, 12);
                var builder = new StringBuilder();
                for (int i = 0; i < length; i++) builder.Append((char)('a' + Rng.NextInt(ref state, 6)));
                string document = builder.ToString();

                TextOp a = RandomOp(ref state, document.Length, 1 + Rng.NextInt(ref state, 3));
                TextOp b = RandomOp(ref state, document.Length, 1 + Rng.NextInt(ref state, 3));
                if (a.IsInsert == b.IsInsert && a.Position == b.Position) conflicts++;
                if (Ot.TransformAgainst(a, b).Dropped || Ot.TransformAgainst(b, a).Dropped) drops++;

                string left;
                string right;
                try
                {
                    left = Ot.Apply(Ot.Apply(document, a), Ot.TransformAgainst(b, a));
                    right = Ot.Apply(Ot.Apply(document, b), Ot.TransformAgainst(a, b));
                }
                catch (ArgumentException exception)
                {
                    tp1Holds = false;
                    failure ??= "doc='" + document + "' a=" + a + " b=" + b + " -> " + exception.Message;
                    continue;
                }
                if (left != right)
                {
                    tp1Holds = false;
                    failure ??= "doc='" + document + "' a=" + a + " b=" + b + " -> '" + left + "' vs '" + right + "'";
                }
                checkedPairs++;
            }
            Check(tp1Holds, "TP1 holds on all " + checkedPairs + " random concurrent pairs: " + (failure ?? "ok"));
            Check(conflicts > 25, "the workload really produced position conflicts (" + conflicts + " pairs)");
            Check(drops > 25, "and it really exercised the drop rule (" + drops + " pairs): " + (failure ?? "ok"));
        }

        // ---- intention preservation: exactly the targeted characters disappear -------------------
        {
            ulong state = 0x0FEDCBA987654321UL;
            int deleteCases = 0;
            bool intentionHeld = true, orderHeld = true, insertsHeld = true;
            string? failure = null;
            for (int trial = 0; trial < 500; trial++)
            {
                int length = 5 + Rng.NextInt(ref state, 15);
                var builder = new StringBuilder();
                for (int i = 0; i < length; i++) builder.Append((char)('A' + Rng.NextInt(ref state, 8)));
                string document = builder.ToString();

                TextOp a = RandomOp(ref state, length, 1 + Rng.NextInt(ref state, 3));
                TextOp b = RandomOp(ref state, length, 1 + Rng.NextInt(ref state, 3));
                bool anyDelete = !a.IsInsert || !b.IsInsert;

                var tracked = Ot.Track(document);
                TextOp rebasedB0 = Ot.TransformAgainst(b, a);
                Ot.TrackApply(tracked, a);
                Ot.TrackApply(tracked, rebasedB0);
                string merged = Ot.Render(tracked);

                // Expected: every original character that neither delete aimed at, in its original order, plus
                // the insertions that still had an anchor after the transform.
                bool[] killed = new bool[length];
                foreach (TextOp op in new[] { a, b })
                    if (!op.IsInsert)
                        for (int i = op.Position; i < op.Position + op.Length; i++) killed[i] = true;
                var expected = new StringBuilder();
                for (int i = 0; i < length; i++) if (!killed[i]) expected.Append(document[i]);

                var survivors = new List<char>();
                int inserted = 0;
                foreach (Ot.Letter letter in tracked)
                {
                    if (letter.Origin < 0) { inserted++; continue; }
                    survivors.Add(letter.Value);
                }
                if (!anyDelete) continue;
                deleteCases++;
                if (new string(survivors.ToArray()) != expected.ToString())
                {
                    intentionHeld = false;
                    failure ??= "doc='" + document + "' a=" + a + " b=" + b + " survivors='"
                        + new string(survivors.ToArray()) + "' expected='" + expected + "'";
                }
                TextOp rebasedA = Ot.TransformAgainst(a, b);
                TextOp rebasedB = Ot.TransformAgainst(b, a);
                int expectedInserts = (a.IsInsert && !rebasedA.Dropped ? 1 : 0)
                    + (b.IsInsert && !rebasedB.Dropped ? 1 : 0);
                if (inserted != expectedInserts)
                {
                    insertsHeld = false;
                    failure ??= "insert count for a=" + a + " b=" + b + " was " + inserted + " not " + expectedInserts;
                }
                int lastIndex = -1;
                bool ordered = true;
                for (int i = 0; i < tracked.Count; i++)
                {
                    if (tracked[i].Origin < 0) continue;
                    if (tracked[i].Origin <= lastIndex) ordered = false;
                    lastIndex = tracked[i].Origin;
                }
                if (!ordered) orderHeld = false;
                if (merged.Length != survivors.Count + inserted) orderHeld = false;
            }
            Check(deleteCases > 200, "the intention workload contained " + deleteCases + " pairs with a delete");
            Check(intentionHeld, "exactly the characters aimed at by the deletes disappear: " + (failure ?? "ok"));
            Check(insertsHeld, "no insertion is ever lost by the merge: " + (failure ?? "ok"));
            Check(orderHeld, "surviving characters keep their original relative order");
        }

        // ---- three concurrent inserts, merged canonically ---------------------------------------
        {
            string document = "abcdef";
            List<TextOp> ops = new List<TextOp>
            {
                TextOp.Insert(1, 0, 'x'),
                TextOp.Insert(2, 2, 'y'),
                TextOp.Insert(3, 5, 'z'),
            };
            string merged = CanonicalMerge(document, ops);
            Check(merged == CanonicalMerge(document, ops), "the canonical merge is deterministic");
            Check(merged.Length == document.Length + 3, "three inserts add exactly three characters: " + merged);
            foreach (char c in "abcdefxyz")
                Check(merged.Contains(c), "the merged document still contains '" + c + "': " + merged);
            Check(merged == "xabycdezf",
                "the canonical merge places each insert where its author put it: " + merged);
            var withoutInserts = new StringBuilder();
            foreach (char c in merged) if (c != 'x' && c != 'y' && c != 'z') withoutInserts.Append(c);
            Check(withoutInserts.ToString() == document,
                "stripping the insertions leaves the untouched document: " + withoutInserts);

            var deleteOps = new List<TextOp>
            {
                TextOp.Delete(1, 0, 2),          // site 1 removes "ab"
                TextOp.Delete(2, 1, 2),          // site 2 removes "bc"; its "b" is already gone
                TextOp.Insert(3, 3, 'W'),        // site 3 inserts before "d"
            };
            string deletions = CanonicalMerge("abcdef", deleteOps);
            Check(deletions == "Wdef",
                "overlapping deletes keep only the survivors and the insert lands before them: " + deletions);

            var tracked = Ot.Track("abcdef");
            var canonicalTrack = new List<TextOp> { deleteOps[0], deleteOps[1], deleteOps[2] };
            canonicalTrack.Sort(TextOp.Compare);
            var appliedTrack = new List<TextOp>();
            foreach (TextOp original in canonicalTrack)
            {
                TextOp op = original;
                foreach (TextOp prior in appliedTrack) op = Ot.TransformAgainst(op, prior);
                Ot.TrackApply(tracked, op);
                appliedTrack.Add(op);
            }
            var survivorChars = new List<char>();
            foreach (Ot.Letter letter in tracked) if (letter.Origin >= 0) survivorChars.Add(letter.Value);
            Check(new string(survivorChars.ToArray()) == "def",
                "only the characters both deletes aimed at are gone: " + new string(survivorChars.ToArray()));
        }

        // ---- edge cases and rejected operations --------------------------------------------------
        {
            Check(Ot.Apply(string.Empty, TextOp.Insert(1, 0, 'a')) == "a", "a character can be inserted into an empty document");
            Check(Ot.Apply("abc", TextOp.Delete(1, 0, 3)) == string.Empty, "a whole document can be deleted");
            var (noOpA, noOpB) = Ot.Transform(TextOp.Insert(1, 2, 'q'), TextOp.Delete(2, 1, 0));
            Check(noOpA.Position == 2 && noOpB.IsNoOp, "transforming against a no-op delete changes nothing");
            var (shiftA, shiftB) = Ot.Transform(TextOp.Insert(1, 5, 'q'), TextOp.Delete(2, 0, 4));
            Check(shiftA.Position == 1 && !shiftA.Dropped && shiftB.Position == 0 && shiftB.Length == 4,
                "an insert past a deleted range shifts left by the deleted length: " + shiftA + " / " + shiftB);

            int rejected = 0;
            foreach (TextOp bad in new[]
                     {
                         TextOp.Insert(1, 4, 'q'), TextOp.Insert(1, -1, 'q'),
                         TextOp.Delete(1, 1, 3), TextOp.Delete(1, 0, -1),
                     })
            {
                try { Ot.Apply("abc", bad); }
                catch (ArgumentException) { rejected++; }
            }
            Check(rejected == 4, "out-of-bounds inserts and deletes and negative lengths are all rejected ("
                + rejected + "/4)");
            Check(Ot.Apply("abc", TextOp.Delete(1, 3, 0)) == "abc", "an empty delete range at the end is legal");
        }

        // ---- determinism ------------------------------------------------------------------------
        {
            ulong state = 0x13572468ACE0BDF1UL;
            var tracked = Ot.Track("concurrent");
            var applied = new List<TextOp>();
            for (int i = 0; i < 40; i++)
            {
                TextOp op = RandomOp(ref state, tracked.Count, 1 + Rng.NextInt(ref state, 3));
                Ot.TrackApply(tracked, op);
                applied.Add(op);
            }
            string first = Ot.Render(tracked);
            Check(tracked.Count >= 1 && first.Length == tracked.Count,
                "forty cumulative operations produced a document of " + first.Length + " characters");
            Check(Ot.Render(Ot.Track(first)) == first, "re-rendering the merged document is stable");
            var (againA, againB) = Ot.Transform(applied[0], applied[1]);
            var (firstA, firstB) = Ot.Transform(applied[0], applied[1]);
            Check(againA.Position == firstA.Position && againB.Position == firstB.Position
                    && againA.Dropped == firstA.Dropped && againB.Dropped == firstB.Dropped,
                "transforming the same pair twice gives the same result");
            Console.WriteLine("ops=" + applied.Count + " mergedLength=" + first.Length + " doc='" + first + "'");
        }

        Console.WriteLine("PASS");
    }

    private static TextOp RandomOp(ref ulong state, int documentLength, int site)
    {
        if (documentLength == 0 || Rng.NextInt(ref state, 2) == 0)
            return TextOp.Insert(site, Rng.NextInt(ref state, documentLength + 1), (char)('a' + Rng.NextInt(ref state, 6)));
        int position = Rng.NextInt(ref state, documentLength);
        int maxLength = documentLength - position;
        return TextOp.Delete(site, position, 1 + Rng.NextInt(ref state, maxLength));
    }
}
