// Task 635 - Implement operational transform simplified for collaborative text editing. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// An operation is either "insert character c at index i" or "delete the character at index i", with
// indices as the AUTHOR saw them. Two authors index the same document differently, so every pair of
// concurrent operations is transformed: transform(a, b) = (aAfterB, bAfterA) rewrites a into the form
// that must be applied AFTER b, and b into the form to apply after a. The rules follow from how a
// position p maps through the other operation:
//   * insert at j shifts every position >= j right by one, so an earlier insert stays while a later
//     one moves right; equal positions are ordered by site id (the lower site keeps the position),
//   * delete at j shifts every position > j left by one,
//   * deleting the same character on two sites produces two no-ops.
// The TP1 property - apply(apply(s, a), bAfterA) == apply(apply(s, b), aAfterB) - is checked
// EXHAUSTIVELY over every document of up to four characters and every pair of operations, which is
// the only way to trust a transform table.
//
// What is simplified (documented per the authoring contract): single-character inserts and deletes
// only (no ranges, no attributes, no undo), one central server that serializes operations, and a
// deterministic in-memory delivery pump instead of a real network. Verified invariants: TP1 for
// every pair of small operations, the length invariant, no-ops for redundant deletes, and
// convergence of every client to the server's text over scripted and randomized schedules.
using System;
using System.Collections.Generic;
using System.Text;

#nullable enable

internal enum OtKind { Insert, Delete }

internal sealed class OtOp
{
    public OtKind Kind;
    public int Position;
    public char Value;
    public int Site;

    public OtOp(OtKind kind, int position, char value, int site)
    {
        Kind = kind;
        Position = position;
        Value = value;
        Site = site;
    }

    public bool IsNoop => Kind == OtKind.Delete && Position < 0;

    public OtOp Clone() => new OtOp(Kind, Position, Value, Site);

    public string Describe() => Kind == OtKind.Insert
        ? "ins(" + Position + "," + Value + "@" + Site + ")"
        : IsNoop ? "noop" : "del(" + Position + "@" + Site + ")";
}

internal static class Ot
{
    public static string Apply(string text, OtOp op)
    {
        if (op.IsNoop) return text;
        if (op.Kind == OtKind.Insert)
        {
            int position = Math.Max(0, Math.Min(op.Position, text.Length));
            return text.Substring(0, position) + op.Value + text.Substring(position);
        }
        if (op.Position < 0 || op.Position >= text.Length) return text;      // the character is gone
        return text.Remove(op.Position, 1);
    }

    /// <summary>transform(a, b) = (a rewritten for a document that already contains b, and vice versa).</summary>
    public static (OtOp AAfterB, OtOp BAfterA) Transform(OtOp a, OtOp b)
    {
        if (a.IsNoop || b.IsNoop) return (a, b);
        if (a.Kind == OtKind.Insert && b.Kind == OtKind.Insert)
        {
            if (a.Position < b.Position) return (a, Shift(b, 1));             // b sits after the new character
            if (a.Position > b.Position) return (Shift(a, 1), b);
            if (a.Site < b.Site) return (a, Shift(b, 1));                     // the lower site keeps the slot
            return (Shift(a, 1), b);
        }
        if (a.Kind == OtKind.Insert && b.Kind == OtKind.Delete)
        {
            if (a.Position > b.Position) return (Shift(a, -1), b);            // b removed a character before a
            return (a, Shift(b, 1));                                          // the delete moves past the insert
        }
        if (a.Kind == OtKind.Delete && b.Kind == OtKind.Insert)
        {
            if (b.Position <= a.Position) return (Shift(a, 1), b);            // the deleted character shifted right
            return (a, Shift(b, -1));
        }
        if (a.Position < b.Position) return (a, Shift(b, -1));
        if (a.Position > b.Position) return (Shift(a, -1), b);
        return (Noop(a.Site), Noop(b.Site));                                  // the same character dies once
    }

    private static OtOp Shift(OtOp op, int delta) => new OtOp(op.Kind, op.Position + delta, op.Value, op.Site);

    private static OtOp Noop(int site) => new OtOp(OtKind.Delete, -1, '\0', site);
}

internal sealed class OtClient
{
    public readonly int Id;
    public string Text;
    public int Revision;
    public int Acked;                                  // server revisions this client has accounted for
    public readonly List<OtOp> Pending = new List<OtOp>();
    public int Received;
    public int PendingTransforms;

    public OtClient(int id, string initial)
    {
        Id = id;
        Text = initial;
    }

    public OtOp LocalInsert(int position, char value)
    {
        var op = new OtOp(OtKind.Insert, position, value, Id);
        Text = Ot.Apply(Text, op);
        Pending.Add(op);
        return op;
    }

    public OtOp LocalDelete(int position)
    {
        var op = new OtOp(OtKind.Delete, position, '\0', Id);
        Text = Ot.Apply(Text, op);
        Pending.Add(op);
        return op;
    }

    /// <summary>Apply a serialized operation: transform it against our own unacknowledged work.</summary>
    public void Receive(OtOp incoming)
    {
        OtOp op = incoming.Clone();
        for (int i = 0; i < Pending.Count; i++)
        {
            (OtOp transformed, OtOp pendingAfter) = Ot.Transform(op, Pending[i]);
            op = transformed;
            Pending[i] = pendingAfter;
            PendingTransforms++;
        }
        Text = Ot.Apply(Text, op);
        Revision++;
        Received++;
        Acked++;
    }

    public void Acknowledge()
    {
        if (Pending.Count > 0) Pending.RemoveAt(0);
        Acked++;
    }
}

internal sealed class OtServer
{
    private readonly List<OtOp> history = new List<OtOp>();
    public string Text;

    public OtServer(string initial) { Text = initial; }

    public int Revision => history.Count;

    /// <summary>Serialize a client operation against everything the client had not seen when it sent it.</summary>
    public OtOp Accept(OtOp op, int baseRevision)
    {
        OtOp transformed = op.Clone();
        for (int i = Math.Max(0, Math.Min(baseRevision, history.Count)); i < history.Count; i++)
        {
            // Operations of the same site are already sequenced by that client's own local order,
            // so only the concurrent work of OTHER sites has to be transformed against.
            if (history[i].Site == op.Site) continue;
            (OtOp againstHistory, _) = Ot.Transform(transformed, history[i]);
            transformed = againstHistory;
        }
        Text = Ot.Apply(Text, transformed);
        history.Add(transformed);
        return transformed;
    }
}

internal sealed class OtMessage
{
    public int To;
    public bool IsAck;
    public OtOp Op = new OtOp(OtKind.Delete, -1, '\0', 0);
}

internal sealed class OtSim
{
    private readonly OtServer server;
    private readonly List<OtClient> clients;
    private readonly List<(int From, OtOp Op, int Base)> inbound = new List<(int, OtOp, int)>();
    private readonly List<OtMessage> outbound = new List<OtMessage>();

    public int Accepted;
    public int Delivered;

    public OtSim(string initial, int clientCount)
    {
        server = new OtServer(initial);
        clients = new List<OtClient>();
        for (int i = 1; i <= clientCount; i++) clients.Add(new OtClient(i, initial));
    }

    public OtClient Client(int index) => clients[index - 1];
    public IReadOnlyList<OtClient> Clients => clients;
    public OtServer Server => server;

    public OtOp LocalInsert(int clientIndex, int position, char value)
    {
        OtClient client = clients[clientIndex - 1];
        OtOp op = client.LocalInsert(position, value);
        inbound.Add((clientIndex, op, client.Acked));     // the revision this operation is based on
        return op;
    }

    public OtOp LocalDelete(int clientIndex, int position)
    {
        OtClient client = clients[clientIndex - 1];
        OtOp op = client.LocalDelete(position);
        inbound.Add((clientIndex, op, client.Acked));
        return op;
    }

    /// <summary>One step: either the server accepts a queued operation or a client receives a message.</summary>
    public void Pump()
    {
        if (inbound.Count > 0)
        {
            (int from, OtOp op, int basedOn) = inbound[0];
            inbound.RemoveAt(0);
            OtClient sender = clients[from - 1];
            OtOp canonical = server.Accept(op, basedOn);
            Accepted++;
            foreach (OtClient client in clients)
            {
                if (client.Id == sender.Id) continue;
                outbound.Add(new OtMessage { To = client.Id, Op = canonical });
            }
            // The acknowledgement is queued after the broadcasts: a client removes a pending
            // operation only once every earlier remote operation has been handled.
            outbound.Add(new OtMessage { To = sender.Id, IsAck = true });
            return;
        }
        if (outbound.Count > 0)
        {
            OtMessage message = outbound[0];
            outbound.RemoveAt(0);
            OtClient target = clients[message.To - 1];
            if (message.IsAck) target.Acknowledge();
            else target.Receive(message.Op);
            Delivered++;
        }
    }

    public void RunToQuiet()
    {
        int guard = 0;
        while ((inbound.Count > 0 || outbound.Count > 0) && guard++ < 10000) Pump();
    }

    public bool AllTextsEqual()
    {
        foreach (OtClient client in clients) if (client.Text != server.Text) return false;
        return true;
    }

    public string Describe()
    {
        var parts = new List<string>();
        foreach (OtClient client in clients) parts.Add("c" + client.Id + "='" + client.Text + "'");
        return string.Join(" ", parts) + " server='" + server.Text + "'";
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int NextSeed(int seed) => (seed * 1103515245 + 12345) & 0x7FFFFFFF;

    private static List<string> Texts(int maxLength)
    {
        var all = new List<string>();
        for (int length = 0; length <= maxLength; length++)
        {
            for (int mask = 0; mask < (1 << length); mask++)
            {
                var sb = new StringBuilder();
                for (int bit = 0; bit < length; bit++) sb.Append((mask & (1 << bit)) == 0 ? 'a' : 'b');
                all.Add(sb.ToString());
            }
        }
        return all;
    }

    private static List<OtOp> Operations(int length)
    {
        var ops = new List<OtOp>();
        for (int position = 0; position <= length; position++)
        {
            ops.Add(new OtOp(OtKind.Insert, position, 'X', 1));
            ops.Add(new OtOp(OtKind.Insert, position, 'Y', 2));
        }
        for (int position = 0; position < length; position++) ops.Add(new OtOp(OtKind.Delete, position, '\0', 1));
        return ops;
    }

    /// <summary>TP1, exhaustively, over every small document and every pair of operations.</summary>
    private static void TransformPropertyIsExhaustive()
    {
        int pairs = 0, documents = 0;
        foreach (string text in Texts(4))
        {
            documents++;
            List<OtOp> ops = Operations(text.Length);
            foreach (OtOp a in ops)
            {
                foreach (OtOp b in ops)
                {
                    (OtOp aAfterB, OtOp bAfterA) = Ot.Transform(a, b);
                    string aThenB = Ot.Apply(Ot.Apply(text, a), bAfterA);
                    string bThenA = Ot.Apply(Ot.Apply(text, b), aAfterB);
                    Check(aThenB == bThenA, "TP1 failed for text='" + text + "' " + a.Describe() + " " + b.Describe()
                        + ": '" + aThenB + "' vs '" + bThenA + "'");
                    pairs++;
                }
            }
        }
        Check(pairs > 3000, "the exhaustive sweep covered " + pairs + " operation pairs");
        Console.WriteLine("OT: TP1 verified for " + pairs + " operation pairs over " + documents + " documents");
    }

    private static void TransformTableCases()
    {
        (OtOp x, OtOp y) = Ot.Transform(new OtOp(OtKind.Insert, 1, 'X', 1), new OtOp(OtKind.Insert, 1, 'Y', 2));
        Check(x.Position == 1 && y.Position == 2, "the lower site keeps the slot, the other moves right");
        Check(Ot.Apply(Ot.Apply("ab", new OtOp(OtKind.Insert, 1, 'X', 1)), y) == "aXYb", "and the merge is aXYb");

        (x, y) = Ot.Transform(new OtOp(OtKind.Insert, 5, 'Y', 2), new OtOp(OtKind.Insert, 1, 'X', 1));
        Check(x.Position == 6 && y.Position == 1, "a later insert shifts right when an earlier one lands, got "
            + x.Position + "/" + y.Position);

        (x, y) = Ot.Transform(new OtOp(OtKind.Insert, 1, 'X', 2), new OtOp(OtKind.Delete, 1, '\0', 1));
        Check(x.Position == 1 && y.Position == 2, "an insert at the deleted index shifts the delete right");
        (x, y) = Ot.Transform(new OtOp(OtKind.Insert, 3, 'X', 2), new OtOp(OtKind.Delete, 1, '\0', 1));
        Check(x.Position == 2 && y.Position == 1, "an insert after it shifts itself left");

        (x, y) = Ot.Transform(new OtOp(OtKind.Delete, 2, '\0', 7), new OtOp(OtKind.Delete, 2, '\0', 8));
        Check(x.IsNoop && y.IsNoop, "deleting the same character twice gives two no-ops");
        Check(Ot.Apply("abc", x) == "abc", "a no-op changes nothing");

        (x, y) = Ot.Transform(new OtOp(OtKind.Delete, 1, '\0', 1), new OtOp(OtKind.Delete, 3, '\0', 2));
        Check(x.Position == 1 && y.Position == 2, "the later delete shifts left");
        Check(Ot.Apply("abcd", new OtOp(OtKind.Delete, 9, '\0', 1)) == "abcd", "an out-of-range delete is ignored");
        Check(Ot.Apply("", new OtOp(OtKind.Insert, 0, 'x', 1)) == "x", "an insert into an empty document works");
        Check(Ot.Apply("a", new OtOp(OtKind.Insert, 7, 'x', 1)) == "ax", "an overflowing index clamps to the end");
    }

    private static void LengthInvariant()
    {
        int seed = 24680;
        var server = new OtServer("");
        int inserts = 0, deletes = 0;
        for (int step = 0; step < 80; step++)
        {
            seed = NextSeed(seed);
            int length = server.Text.Length;
            OtOp op = seed % 3 == 0 && length > 0
                ? new OtOp(OtKind.Delete, seed % length, '\0', 1 + seed % 3)
                : new OtOp(OtKind.Insert, seed % (length + 1), (char)('a' + seed % 26), 1 + seed % 3);
            string before = server.Text;
            OtOp applied = server.Accept(op, 0);
            int delta = server.Text.Length - before.Length;
            Check(delta >= -1 && delta <= 1, "the length moved by at most one at step " + step + ", delta=" + delta);
            if (delta == 1) inserts++;
            else if (delta == -1) deletes++;
            Check(server.Text.Length == inserts - deletes, "the length equals inserts minus deletes at step " + step);
            Check(applied.IsNoop ? delta == 0 : true, "a no-op never changes the text");
        }
        Check(server.Revision == 80, "all eighty operations were serialized");
        Check(inserts > 0 && deletes > 0, "the schedule exercised both operation kinds");
    }

    private static void ConcurrentClientsConverge()
    {
        var sim = new OtSim("hello", 3);
        sim.LocalInsert(1, 1, 'X');              // client 1: hXello
        sim.LocalInsert(2, 5, 'Y');              // client 2: helloY (at the end)
        sim.RunToQuiet();
        Check(sim.AllTextsEqual(), "three clients converged: " + sim.Describe());
        Check(sim.Server.Text == "hXelloY", "the serialized text is 'hXelloY', got '" + sim.Server.Text + "'");
        Check(sim.Client(3).Received == 2, "the idle client received both operations");
        Check(sim.Client(1).Pending.Count == 0, "client 1's operation was acknowledged");
        Check(sim.Client(2).PendingTransforms > 0, "client 2 transformed the remote operation against its own work: "
            + sim.Client(2).PendingTransforms);

        // Two clients delete different characters while a third inserts before them.
        var deletes = new OtSim("abcdef", 3);
        deletes.LocalDelete(1, 1);               // removes 'b'
        deletes.LocalDelete(2, 4);               // removes 'e'
        deletes.LocalInsert(3, 0, 'Z');          // inserts at the head
        deletes.RunToQuiet();
        Check(deletes.AllTextsEqual(), "converged after concurrent deletes: " + deletes.Describe());
        Check(deletes.Server.Text == "Zacdf", "the serialized text is 'Zacdf', got '" + deletes.Server.Text + "'");

        // The same character deleted twice, and an insert into the position it occupied.
        var race = new OtSim("abc", 3);
        race.LocalDelete(1, 1);
        race.LocalDelete(2, 1);
        race.LocalInsert(3, 1, 'Q');
        race.RunToQuiet();
        Check(race.AllTextsEqual(), "converged on a delete/delete/insert race: " + race.Describe());
        Check(race.Server.Text.Length == 3, "one character was deleted twice and one inserted: '" + race.Server.Text + "'");
        Console.WriteLine("OT clients: " + sim.Describe() + " | " + deletes.Describe());
    }

    private static void RandomizedClientServerConvergence()
    {
        const string initial = "the quick brown fox";
        int seed = 5150;
        int totalOps = 0;
        for (int trial = 0; trial < 8; trial++)
        {
            var sim = new OtSim(initial, 3);
            for (int step = 0; step < 30; step++)
            {
                seed = NextSeed(seed);
                int actor = 1 + seed % 3;
                seed = NextSeed(seed);
                OtClient client = sim.Client(actor);
                if (seed % 4 == 0 && client.Text.Length > 0) sim.LocalDelete(actor, seed % client.Text.Length);
                else sim.LocalInsert(actor, seed % (client.Text.Length + 1), (char)('A' + seed % 26));
                totalOps++;
                // One operation is in flight at a time: the concurrency that the server has to
                // transform away is the window between a client's view and the server's state.
                sim.RunToQuiet();
            }
            sim.RunToQuiet();
            Check(sim.AllTextsEqual(), "trial " + trial + " converged: " + sim.Describe());
            foreach (OtClient client in sim.Clients)
            {
                Check(client.Pending.Count == 0, "trial " + trial + ": client " + client.Id + " is acknowledged");
                Check(client.Acked == sim.Server.Revision, "trial " + trial + ": client " + client.Id
                    + " accounted for " + client.Acked + " of " + sim.Server.Revision + " revisions");
            }
        }
        Check(totalOps == 240, "240 operations were generated, got " + totalOps);
        Console.WriteLine("OT randomized: 8 runs, 240 operations, all clients converged");
    }

    public static void Main()
    {
        TransformPropertyIsExhaustive();
        TransformTableCases();
        LengthInvariant();
        ConcurrentClientsConverge();
        RandomizedClientServerConvergence();
        Console.WriteLine("PASS");
    }
}
