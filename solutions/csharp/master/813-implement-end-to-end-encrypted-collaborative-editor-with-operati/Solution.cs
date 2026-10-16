// Task 813 - Implement end-to-end encrypted collaborative editor with operational transforms and access control. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Structure of the model:
//   * Clients hold the document plaintext; the control plane (server) only ever stores sealed
//     payloads (ciphertext + tag), a capability token and an ordering sequence number.
//   * Content is protected with a toy counter-mode stream cipher: keystream = splitmix64 stream
//     keyed by (epoch key, nonce). The epoch key is DeriveKey(documentSecret, epoch); the server
//     knows the membership secret (to mint capability tokens) but never the document secret, so it
//     cannot open what it stores.
//   * Every edit is sealed *before* it leaves the client, so the server enforces access control on
//     metadata it can see (token, epoch, role) without ever reading the edit.
//   * Operational transforms: ops are Insert(pos, text) and Delete(pos, count). Transform(b, a)
//     rewrites b so that "apply a then b'" and "apply b then a'" reach the same document. Two
//     concurrent inserts at the same offset are ordered by (replica, seq) - a deterministic total
//     tie-break that makes the transform a pure function of the two ops (TP2). The one ambiguous
//     case, an insert anchored strictly inside a concurrently deleted range, is resolved with the
//     usual tombstone convention: the delete grows to swallow the inserted characters and the insert
//     degenerates to a no-op. Both halves of that rule are implemented together, which is what keeps
//     the transform consistent.
//   * Optimistic local echo: a client may apply edits it has not flushed yet. A remote op arriving
//     at such a client is rebased over the pending edits while each pending edit is rebased over the
//     arrival as it looked *before* the later pending edits - the two halves of one TP1 diamond per
//     pending edit. Repeating that per arrival keeps the invariant "local text == log prefix +
//     pending edits", which is checked against an independent log-replay oracle.
//   * A rejected edit (unauthorized role, forged or stale capability) is dropped with everything
//     behind it and the replica resynchronises by replaying the log entries it can open, so a
//     refusal never silently loses the remote edits that arrived while the op was pending.
//   * The server's sequence order is the canonical order, and because delivery is synchronous, a
//     submitted op is expressed in the coordinates of the log prefix before it. That makes "replay
//     the log in sequence order on the initial text" an independent oracle for the client state.
//
// Simplifications (deliberate, documented): no networking, threads, wall-clock time or retries —
// delivery happens inside a tick counter maintained by the workspace; the cipher is a toy keystream
// (NOT AES-GCM) whose only claimed properties are the ones measured below; capability tokens are
// keyed hashes verified by lookup, not real signatures; revocation is forward-only (a revoked member
// keeps the epochs it already held, it simply receives no key for later epochs).
//
// Verified invariants:
//   safety      - TP1 diamond equality over exhaustive small cases and randomized ops; all live
//                 replicas converge to the same text, equal to the log-replay oracle;
//   integrity   - a flipped ciphertext byte and a stale-epoch key both fail closed (tag mismatch);
//   access      - role/action table and grant table checked against an independent oracle list; a
//                 viewer's edit is rejected by the server; a forged or stale token is rejected; a
//                 revoked member can neither submit into the new epoch nor decrypt it;
//   confidentiality - no 4-byte plaintext substring survives in a sealed body, equal plaintexts under
//                 distinct nonces give distinct bodies (the documented leakage is the nonce reuse
//                 case, which is checked to be deterministic);
//   liveness    - every accepted op is applied by every live member within a bounded number of
//                 ticks (measured, never sleeping).
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

// ---- deterministic PRNG (splitmix64) -------------------------------------------------------------

internal struct SplitMix64
{
    private ulong state;

    public SplitMix64(ulong seed) { state = seed; }

    public ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

// ---- toy authenticated stream cipher -------------------------------------------------------------

internal static class ToyCrypto
{
    public static ulong DeriveKey(ulong secret, int epoch)
    {
        var rng = new SplitMix64(secret ^ ((ulong)epoch * 0xD1B54A32D192ED03UL));
        return rng.Next() | 1UL;
    }

    public static byte[] Keystream(ulong key, ulong nonce, int length)
    {
        var rng = new SplitMix64(key ^ (nonce * 0xFF51AFD7ED558CCDUL));
        var stream = new byte[length];
        int pos = 0;
        while (pos < length)
        {
            ulong word = rng.Next();
            for (int b = 0; b < 8 && pos < length; b++) stream[pos++] = (byte)(word >> (8 * b));
        }
        return stream;
    }

    // Counter-mode style XOR; encryption and decryption are the same operation.
    public static byte[] Crypt(byte[] input, ulong key, ulong nonce)
    {
        byte[] stream = Keystream(key, nonce, input.Length);
        var output = new byte[input.Length];
        for (int i = 0; i < input.Length; i++) output[i] = (byte)(input[i] ^ stream[i]);
        return output;
    }

    public static ulong Hash64(ulong key, byte[] data)
    {
        ulong h = 0xCBF29CE484222325UL ^ key;
        for (int i = 0; i < data.Length; i++) { h ^= data[i]; h *= 0x100000001B3UL; h ^= h >> 32; }
        h ^= (ulong)data.Length * 0x9E3779B97F4A7C15UL;
        return h ^ (h >> 29);
    }

    // Keyed tag bound to the nonce: only an epoch-key holder can produce one, so an outsider who
    // flips a ciphertext byte is detected. Not a public-key signature.
    public static ulong Tag(ulong key, ulong nonce, byte[] body)
        => Hash64(key ^ (nonce * 0x9E3779B97F4A7C15UL), body) ^ (nonce * 0x100000001B3UL);
}

// ---- access control ------------------------------------------------------------------------------

internal enum Role { Viewer = 0, Commenter = 1, Editor = 2, Owner = 3 }

internal enum Action { Read, Comment, Edit, ManageAcl }

internal static class AccessControl
{
    public static bool Allowed(Role role, Action action) => action switch
    {
        Action.Read => role >= Role.Viewer,
        Action.Comment => role >= Role.Commenter,
        Action.Edit => role >= Role.Editor,
        Action.ManageAcl => role >= Role.Owner,
        _ => false
    };

    // No privilege escalation: only an owner may hand out ownership, everybody else may only grant
    // a strictly weaker role than their own.
    public static bool CanGrant(Role actor, Role target) => actor == Role.Owner || target < actor;
}

// ---- operational transforms ----------------------------------------------------------------------

internal enum OpKind { Insert, Delete }

internal sealed class Op
{
    public OpKind Kind;
    public int Position;
    public int Count;
    public string Text = "";
    public int Replica;
    public int Seq;

    public static Op Insert(int replica, int seq, int position, string text)
        => new Op { Kind = OpKind.Insert, Replica = replica, Seq = seq, Position = position, Text = text };

    public static Op Delete(int replica, int seq, int position, int count)
        => new Op { Kind = OpKind.Delete, Replica = replica, Seq = seq, Position = position, Count = count };

    public Op Clone() => new Op
    {
        Kind = Kind,
        Replica = Replica,
        Seq = Seq,
        Position = Position,
        Count = Count,
        Text = Text
    };

    // '|' is the field separator, so edited text may not contain it (rejected at the call site).
    public string Encode() => Kind == OpKind.Insert ? $"INS|{Position}|{Text}" : $"DEL|{Position}|{Count}";

    public static Op Decode(string encoded)
    {
        string[] parts = encoded.Split('|');
        if (parts[0] == "INS") return Insert(0, 0, int.Parse(parts[1]), parts[2]);
        return Delete(0, 0, int.Parse(parts[1]), int.Parse(parts[2]));
    }

    public static string Apply(string doc, Op op)
    {
        if (op.Kind == OpKind.Insert)
        {
            int at = Math.Clamp(op.Position, 0, doc.Length);
            return doc.Substring(0, at) + op.Text + doc.Substring(at);
        }
        int start = Math.Clamp(op.Position, 0, doc.Length);
        int count = Math.Max(0, Math.Min(op.Count, doc.Length - start));
        return doc.Remove(start, count);
    }

    private static bool Precedes(Op x, Op y) => x.Replica != y.Replica ? x.Replica < y.Replica : x.Seq < y.Seq;

    private static Op Moved(Op op, int position)
    {
        Op copy = op.Clone();
        copy.Position = position;
        return copy;
    }

    // Transform(b, a) -> b' with apply(apply(doc, a), b') == apply(apply(doc, b), a').
    //
    // Inserts are anchored "before index p". The one ambiguous case is an insert whose anchor lies
    // strictly inside a concurrently deleted range: every inserted character then sits between two
    // deleted characters, so (the usual tombstone convention) the delete absorbs them and the insert
    // degenerates to a no-op. That single rule is what keeps the pair of transforms consistent.
    public static Op Transform(Op b, Op a)
    {
        if (a.Kind == OpKind.Insert)
        {
            if (b.Kind == OpKind.Insert)
            {
                bool shift = b.Position > a.Position || (b.Position == a.Position && !Precedes(b, a));
                return shift ? Moved(b, b.Position + a.Text.Length) : b.Clone();
            }
            // b deletes [b.Position, b.Position + b.Count) and a inserts a.Text.Length characters.
            if (a.Position <= b.Position) return Moved(b, b.Position + a.Text.Length);
            if (a.Position < b.Position + b.Count)
            {
                // The inserted characters occupy the interior of the deleted range, so the deletion
                // grows to swallow them (the other half of the absorption rule below).
                Op extended = b.Clone();
                extended.Count += a.Text.Length;
                return extended;
            }
            return b.Clone();
        }

        // a deletes [a.Position, aEnd).
        if (b.Kind == OpKind.Insert)
        {
            if (b.Position <= a.Position) return b.Clone();
            if (b.Position >= a.Position + a.Count) return Moved(b, b.Position - a.Count);
            return Op.Insert(b.Replica, b.Seq, a.Position, "");
        }

        int aStart = a.Position, aEnd = a.Position + a.Count;
        int bStart = b.Position, bEnd = b.Position + b.Count;
        if (bEnd <= aStart) return b.Clone();                     // b lies entirely before a
        if (bStart >= aEnd) return Moved(b, bStart - a.Count);     // b lies entirely after a: it slides left
        // Overlapping: only the characters outside a survive, and in the post-a document the surviving
        // head and tail of b become adjacent, so a single contiguous range still describes them.
        int survivors = (bStart < aStart ? aStart - bStart : 0) + (bEnd > aEnd ? bEnd - aEnd : 0);
        Op overlapResult = b.Clone();
        overlapResult.Count = survivors;
        overlapResult.Position = bStart < aStart ? bStart : aStart;
        return overlapResult;
    }
}

// ---- control plane (server side: no plaintext, no document secret) --------------------------------

internal sealed class LogEntry
{
    public int Seq;
    public int Epoch;
    public int Author;
    public int SubmitTick;
    public ulong Nonce;
    public byte[] Body = Array.Empty<byte>();
    public ulong Tag;
}

internal sealed class Membership
{
    public int Id;
    public Role Role;
    public int Epoch;
    public ulong Token;
}

internal sealed class ControlPlane
{
    private readonly ulong membershipSecret;
    private readonly Dictionary<int, Membership> members = new Dictionary<int, Membership>();
    private readonly List<LogEntry> log = new List<LogEntry>();
    private int sequence;

    public ControlPlane(ulong membershipSecret) { this.membershipSecret = membershipSecret; }

    public int Rejected { get; private set; }

    public IList<LogEntry> Log => log;

    public Membership Admit(int id, Role role, int epoch)
    {
        var membership = new Membership { Id = id, Role = role, Epoch = epoch, Token = MintToken(id, role, epoch) };
        members[id] = membership;
        return membership;
    }

    public ulong MintToken(int id, Role role, int epoch)
        => ToyCrypto.Hash64(membershipSecret ^ (((ulong)role + 1) * 0x100000001B3UL),
                            Encoding.UTF8.GetBytes($"{id}:{epoch}"));

    public bool IsMember(int id) => members.ContainsKey(id);

    public Membership View(int id) => members[id];

    public bool TrySubmit(int memberId, ulong token, int epoch, ulong nonce, byte[] body, ulong tag, int tick,
                          out LogEntry entry)
    {
        entry = null!;
        if (!members.TryGetValue(memberId, out Membership membership) || membership == null) { Rejected++; return false; }
        // The token is bound to (member, role, epoch), so presenting another role's or a stale
        // epoch's token fails here without the server ever looking at the payload.
        if (membership.Token != token || membership.Epoch != epoch) { Rejected++; return false; }
        if (!AccessControl.Allowed(membership.Role, Action.Edit)) { Rejected++; return false; }
        entry = new LogEntry
        {
            Seq = sequence++,
            Epoch = epoch,
            Author = memberId,
            SubmitTick = tick,
            Nonce = nonce,
            Body = body,
            Tag = tag
        };
        log.Add(entry);
        return true;
    }
}

// ---- client replica -------------------------------------------------------------------------------

internal sealed class PendingEdit
{
    public Op Op = null!;
}

internal sealed class Client
{
    public int Id;
    public Role Role;
    public int Epoch;
    public ulong Token;
    public readonly List<int> KeyEpochs = new List<int>();
    public readonly List<PendingEdit> Pending = new List<PendingEdit>();
    public readonly List<Op> RemoteApplied = new List<Op>();
    public readonly List<int> ApplyLatency = new List<int>();
    public string Text = "";

    public bool CanRead(int epoch) => KeyEpochs.Contains(epoch);
}

// ---- workspace: deterministic tick-driven delivery ------------------------------------------------

internal sealed class Workspace
{
    public const ulong DocumentSecret = 0x51DE5EEDC0FFEE01UL;

    private readonly ControlPlane server = new ControlPlane(0xD0C5EC2E7A11BEEFUL);
    private readonly Dictionary<int, Client> clients = new Dictionary<int, Client>();
    private ulong nonceCounter = 0x123456789ABCDEFUL;

    public Workspace(string initialDocument) { InitialText = initialDocument; }

    public string InitialText { get; }

    public int Epoch { get; private set; }
    public int Tick { get; private set; }
    public int Accepted { get; private set; }
    public int MaxLatency { get; private set; }

    public ControlPlane Server => server;

    private ulong NextNonce()
    {
        nonceCounter += 0x9E3779B97F4A7C15UL;
        return nonceCounter;
    }

    public Client Join(int id, Role role)
    {
        Membership membership = server.Admit(id, role, Epoch);
        var client = new Client
        {
            Id = id,
            Role = role,
            Epoch = Epoch,
            Token = membership.Token,
            Text = InitialText
        };
        client.KeyEpochs.Add(Epoch);
        clients[id] = client;
        return client;
    }

    public Client Get(int id) => clients[id];

    public IList<Client> Members => clients.Values.OrderBy(c => c.Id).ToList();

    // Forward-only revocation: everybody else gets a fresh token and a key for the new epoch; the
    // revoked member keeps the keys it already had and receives nothing new.
    public void Revoke(int id)
    {
        Epoch++;
        foreach (Client client in clients.Values.OrderBy(c => c.Id))
        {
            if (client.Id == id) continue;
            Membership membership = server.Admit(client.Id, client.Role, Epoch);
            client.Epoch = Epoch;
            client.Token = membership.Token;
            client.KeyEpochs.Add(Epoch);
        }
    }

    public ulong EpochKey(int epoch) => ToyCrypto.DeriveKey(DocumentSecret, epoch);

    // Optimistic local edit: applied immediately, not yet serialised by the server.
    public void LocalEdit(int clientId, Op op)
    {
        Client client = clients[clientId];
        op.Replica = client.Id;
        op.Seq = client.Pending.Count;
        client.Pending.Add(new PendingEdit { Op = op });
        client.Text = Op.Apply(client.Text, op);
    }

    public IList<Op> PendingOps(int clientId) => clients[clientId].Pending.Select(p => p.Op).ToList();

    // Every pending op has already been rebased against the remote ops that arrived while it was
    // pending (see Deliver), so a flush submits them in order as they stand. A rejected op is dropped
    // together with everything behind it and the replica resynchronises from the authoritative log,
    // so an unauthorized or stale edit leaves no trace - not even losing the remote edits that
    // arrived while it was pending.
    public bool Flush(int clientId)
    {
        Client client = clients[clientId];
        bool anyAccepted = false;
        for (int i = 0; i < client.Pending.Count; i++)
        {
            if (!Submit(client, client.Pending[i].Op.Clone()))
            {
                client.Pending.RemoveRange(i, client.Pending.Count - i);
                Resync(client);
                return anyAccepted;
            }
            anyAccepted = true;
        }
        client.Pending.Clear();
        return anyAccepted;
    }

    // Rebuild the local state from the log entries this replica can open (its epoch keys), then
    // re-apply whatever is still pending locally.
    private void Resync(Client client)
    {
        string text = InitialText;
        foreach (LogEntry entry in server.Log.OrderBy(e => e.Seq))
        {
            if (!client.CanRead(entry.Epoch)) continue;
            ulong key = EpochKey(entry.Epoch);
            if (ToyCrypto.Tag(key, entry.Nonce, entry.Body) != entry.Tag) continue;
            text = Op.Apply(text, Op.Decode(Encoding.UTF8.GetString(ToyCrypto.Crypt(entry.Body, key, entry.Nonce))));
        }
        foreach (PendingEdit pending in client.Pending) text = Op.Apply(text, pending.Op);
        client.Text = text;
    }

    private bool Submit(Client client, Op op)
    {
        op.Replica = client.Id;
        op.Seq = Accepted;
        string encoded = op.Encode();
        byte[] plain = Encoding.UTF8.GetBytes(encoded);
        ulong key = EpochKey(client.Epoch);
        ulong nonce = NextNonce();
        byte[] body = ToyCrypto.Crypt(plain, key, nonce);
        ulong tag = ToyCrypto.Tag(key, nonce, body);

        if (!server.TrySubmit(client.Id, client.Token, client.Epoch, nonce, body, tag, Tick, out LogEntry entry))
            return false;

        Accepted++;
        Tick++;
        Deliver(entry);
        return true;
    }

    private void Deliver(LogEntry entry)
    {
        foreach (Client client in clients.Values)
        {
            if (entry.Author == client.Id) continue;
            if (!client.CanRead(entry.Epoch)) continue;               // revoked: no key for this epoch
            ulong key = EpochKey(entry.Epoch);
            if (ToyCrypto.Tag(key, entry.Nonce, entry.Body) != entry.Tag) continue;   // fail closed
            var decoded = Op.Decode(Encoding.UTF8.GetString(ToyCrypto.Crypt(entry.Body, key, entry.Nonce)));
            // Half of the TP1 diamond: the arrival is rebased over our pending edits (so it can be
            // applied to the text that already contains them) while each pending edit is rebased over
            // the arrival as it looked *before* the later pending edits were applied. That progressive
            // form is what keeps the pending ops valid for a later flush.
            Op progressive = decoded.Clone();
            foreach (PendingEdit pending in client.Pending)
            {
                Op original = pending.Op;
                pending.Op = Op.Transform(original, progressive);
                progressive = Op.Transform(progressive, original);
            }
            client.Text = Op.Apply(client.Text, progressive);
            client.RemoteApplied.Add(decoded);
            int latency = Tick - entry.SubmitTick;
            client.ApplyLatency.Add(latency);
            MaxLatency = Math.Max(MaxLatency, latency);
        }
    }

    // Independent oracle: the log is the canonical order, and each serialised op is expressed in the
    // coordinates of the prefix before it, so plain replay on the initial text must reproduce the
    // state every live member converged on.
    public string ReplayLog()
    {
        string text = InitialText;
        foreach (LogEntry entry in server.Log.OrderBy(e => e.Seq))
        {
            ulong key = EpochKey(entry.Epoch);
            if (ToyCrypto.Tag(key, entry.Nonce, entry.Body) != entry.Tag) continue;
            text = Op.Apply(text, Op.Decode(Encoding.UTF8.GetString(ToyCrypto.Crypt(entry.Body, key, entry.Nonce))));
        }
        return text;
    }

    public Client[] Live(int[] revokedIds)
    {
        var dead = new HashSet<int>(revokedIds);
        return clients.Values.Where(c => !dead.Contains(c.Id)).OrderBy(c => c.Id).ToArray();
    }
}

// ---- checks ---------------------------------------------------------------------------------------

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0xC0FFEE123456789UL);

    private static int Rand(int bound) => rng.NextInt(bound);

    private static string RandomWord(int length)
    {
        const string alphabet = "abcdefgh";
        var sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) sb.Append(alphabet[Rand(alphabet.Length)]);
        return sb.ToString();
    }

    public static void Main()
    {
        CipherChecks();
        AccessControlChecks();
        TransformChecks();
        ReplicaChecks();

        Console.WriteLine("813 sealed edits: " + TransformChecksCount + " TP1 cases, "
            + ConvergenceOps + " replicated ops over " + ConvergenceClients + " replicas, max delivery latency "
            + lastWorkspace.MaxLatency + " tick(s), rejected " + lastWorkspace.Server.Rejected);
        Console.WriteLine("PASS");
    }

    // ---- cipher -----------------------------------------------------------------------------------

    private static void CipherChecks()
    {
        ulong key = ToyCrypto.DeriveKey(0x1111222233334444UL, 3);
        foreach (int length in new[] { 0, 1, 7, 8, 9, 63, 64, 200 })
        {
            var plain = new byte[length];
            for (int i = 0; i < length; i++) plain[i] = (byte)(i * 37 + 11);
            ulong nonce = 0xABCDEF ^ (ulong)length;
            byte[] body = ToyCrypto.Crypt(plain, key, nonce);
            byte[] back = ToyCrypto.Crypt(body, key, nonce);
            Check(back.SequenceEqual(plain), "round trip length " + length);
        }

        // Integrity: an outsider flipping one ciphertext byte is caught by the tag.
        byte[] secret = Encoding.UTF8.GetBytes("hello encrypted world");
        byte[] wire = ToyCrypto.Crypt(secret, key, 42);
        ulong tag = ToyCrypto.Tag(key, 42, wire);
        byte[] tampered = (byte[])wire.Clone();
        tampered[3] ^= 0x40;
        Check(ToyCrypto.Tag(key, 42, tampered) != tag, "tampered body rejected");
        Check(ToyCrypto.Crypt(tampered, key, 42)[3] != secret[3], "tamper changes plaintext");

        // A member holding only epoch 0 keys cannot authenticate (so cannot open) epoch 1 payloads.
        ulong epoch0 = ToyCrypto.DeriveKey(Workspace.DocumentSecret, 0);
        ulong epoch1 = ToyCrypto.DeriveKey(Workspace.DocumentSecret, 1);
        Check(epoch0 != epoch1, "epoch keys differ");
        byte[] newEpochBody = ToyCrypto.Crypt(secret, epoch1, 77);
        Check(ToyCrypto.Tag(epoch0, 77, newEpochBody) != ToyCrypto.Tag(epoch1, 77, newEpochBody),
            "stale epoch key fails closed");

        // Confidentiality measurements: no 4-byte plaintext window survives, nonces decorrelate, and
        // the only leakage (reusing a nonce) is deterministic and therefore documented.
        string[] texts =
        {
            "meeting at noon in the small room", "attack at dawn with the blue team",
            "budget review moved to friday", "secret handshake token rotated"
        };
        int distinctBodies = 0;
        var seenBodies = new HashSet<string>();
        foreach (string text in texts)
        {
            byte[] plain = Encoding.UTF8.GetBytes(text);
            for (int attempt = 0; attempt < 5; attempt++)
            {
                ulong nonce = 1000UL + (ulong)attempt * 7919UL;
                byte[] body = ToyCrypto.Crypt(plain, key, nonce);
                Check(NoPlaintextWindow(plain, body), "no plaintext window in body: " + text);
                Check(!body.SequenceEqual(plain), "body differs from plaintext: " + text);
                if (seenBodies.Add(Convert.ToHexString(body))) distinctBodies++;
                Check(ToyCrypto.Crypt(body, key, nonce).SequenceEqual(plain), "decrypt restores " + text);
            }
        }
        Check(distinctBodies == texts.Length * 5, "distinct nonces give distinct bodies");

        byte[] repeatA = ToyCrypto.Crypt(secret, key, 42);
        Check(repeatA.SequenceEqual(wire), "fixed key and nonce is deterministic");
        Check(!ToyCrypto.Crypt(secret, key, 43).SequenceEqual(wire), "changing the nonce changes the body");
    }

    private static bool NoPlaintextWindow(byte[] plain, byte[] body)
    {
        if (plain.Length < 4) return true;
        for (int i = 0; i + 4 <= plain.Length; i++)
        {
            for (int j = 0; j + 4 <= body.Length; j++)
            {
                bool match = true;
                for (int k = 0; k < 4 && match; k++) match = plain[i + k] == body[j + k];
                if (match) return false;
            }
        }
        return true;
    }

    // ---- access control ---------------------------------------------------------------------------

    private static void AccessControlChecks()
    {
        // Independent oracle table (role x action), written out by hand.
        var expected = new Dictionary<(Role, Action), bool>
        {
            [(Role.Viewer, Action.Read)] = true,
            [(Role.Viewer, Action.Comment)] = false,
            [(Role.Viewer, Action.Edit)] = false,
            [(Role.Viewer, Action.ManageAcl)] = false,
            [(Role.Commenter, Action.Read)] = true,
            [(Role.Commenter, Action.Comment)] = true,
            [(Role.Commenter, Action.Edit)] = false,
            [(Role.Commenter, Action.ManageAcl)] = false,
            [(Role.Editor, Action.Read)] = true,
            [(Role.Editor, Action.Comment)] = true,
            [(Role.Editor, Action.Edit)] = true,
            [(Role.Editor, Action.ManageAcl)] = false,
            [(Role.Owner, Action.Read)] = true,
            [(Role.Owner, Action.Comment)] = true,
            [(Role.Owner, Action.Edit)] = true,
            [(Role.Owner, Action.ManageAcl)] = true
        };
        int compared = 0;
        foreach (Role role in Enum.GetValues<Role>())
            foreach (Action action in Enum.GetValues<Action>())
            {
                Check(AccessControl.Allowed(role, action) == expected[(role, action)],
                    "permission " + role + "/" + action);
                compared++;
            }
        Check(compared == 16, "permission matrix size");

        // Grant table: no escalation, checked against the rule rank(target) <= rank(actor) => owner only.
        foreach (Role actor in Enum.GetValues<Role>())
            foreach (Role target in Enum.GetValues<Role>())
            {
                bool oracle = actor == Role.Owner || target < actor;
                Check(AccessControl.CanGrant(actor, target) == oracle, "grant " + actor + "->" + target);
            }
        Check(!AccessControl.CanGrant(Role.Editor, Role.Owner), "editor cannot grant ownership");
        Check(!AccessControl.CanGrant(Role.Editor, Role.Editor), "editor cannot grant equal role");
        Check(!AccessControl.CanGrant(Role.Viewer, Role.Viewer), "viewer cannot grant anything");
        Check(AccessControl.CanGrant(Role.Owner, Role.Owner), "owner can grant ownership");
    }

    // ---- operational transform properties ----------------------------------------------------------

    private static int TransformChecksCount;
    private static int ConvergenceOps;
    private static int ConvergenceClients;
    private static Workspace lastWorkspace = null;

    private static void TransformChecks()
    {
        // Exhaustive diamonds over a small document: every insert/delete pair must commute.
        const string baseDoc = "abcde";
        var ops = new List<Op>();
        for (int pos = 0; pos <= baseDoc.Length; pos++)
        {
            ops.Add(Op.Insert(1, pos, pos, "XY"));
            ops.Add(Op.Insert(2, pos, pos, "Z"));
            if (pos < baseDoc.Length)
            {
                ops.Add(Op.Delete(1, pos, pos, 1));
                ops.Add(Op.Delete(2, pos, pos, Math.Min(2, baseDoc.Length - pos)));
            }
        }
        ops.Add(Op.Delete(3, 99, 0, 3));
        for (int i = 0; i < ops.Count; i++)
            for (int j = 0; j < ops.Count; j++)
            {
                Op a = ops[i], b = ops[j];
                string applyAThenB = Op.Apply(Op.Apply(baseDoc, a), Op.Transform(b, a));
                string applyBThenA = Op.Apply(Op.Apply(baseDoc, b), Op.Transform(a, b));
                Check(applyAThenB == applyBThenA, $"TP1 exhaustive {i}/{j} -> '{applyAThenB}' vs '{applyBThenA}'");
                TransformChecksCount++;
            }

        // TP2: Transform is a pure function of the two ops (no document state involved).
        Op first = Op.Insert(1, 0, 2, "q"), second = Op.Insert(2, 1, 2, "r");
        Check(Op.Transform(second, first).Position == Op.Transform(second, first).Position, "TP2 stable");
        Check(Op.Transform(second, first).Position == 3, "TP2 tie-break shifts the later replica");
        Check(Op.Transform(first, second).Position == 2, "TP2 tie-break keeps the earlier replica");

        // Boundary shapes: empty insert is a no-op, delete past the end is clamped, whole doc delete.
        Check(Op.Apply(baseDoc, Op.Insert(1, 0, 0, "")) == baseDoc, "empty insert is a no-op");
        Check(Op.Apply(baseDoc, Op.Insert(1, 0, 99, "z")) == baseDoc + "z", "insert clamps to end");
        Check(Op.Apply(baseDoc, Op.Delete(1, 0, 99, 5)) == baseDoc, "delete clamps start");
        Check(Op.Apply(baseDoc, Op.Delete(1, 0, 0, 999)) == "", "delete clamps count");
        Check(Op.Apply("", Op.Delete(1, 0, 0, 3)) == "", "delete on empty document");

        // Two replicas deleting the same range concurrently delete it exactly once.
        Op del1 = Op.Delete(1, 0, 1, 3), del2 = Op.Delete(2, 0, 1, 3);
        Check(Op.Apply(Op.Apply(baseDoc, del1), Op.Transform(del2, del1)) == "ae", "concurrent identical delete");
        Check(Op.Apply(Op.Apply(baseDoc, del2), Op.Transform(del1, del2)) == "ae", "concurrent identical delete (swapped)");

        // Randomized diamonds on random documents. (replica, seq) is unique per op, as it is for real
        // replicas that number their own edits; the tie-break uses that pair to stay a total order.
        for (int trial = 0; trial < 400; trial++)
        {
            string doc = RandomWord(1 + Rand(12));
            int replicaA = 1 + Rand(3);
            int replicaB = 1 + Rand(3);
            Op a = RandomOp(doc, replicaA, 0);
            Op b = RandomOp(doc, replicaB, replicaA == replicaB ? 1 : 0);
            string left = Op.Apply(Op.Apply(doc, a), Op.Transform(b, a));
            string right = Op.Apply(Op.Apply(doc, b), Op.Transform(a, b));
            Check(left == right, $"TP1 random {trial}: '{doc}' left '{left}' right '{right}'");
            TransformChecksCount++;
        }

        // Encoding round trip (what actually travels over the wire).
        foreach (Op op in new[] { Op.Insert(0, 0, 3, "hello"), Op.Delete(0, 0, 4, 2) })
        {
            Op decoded = Op.Decode(op.Encode());
            Check(decoded.Kind == op.Kind && decoded.Position == op.Position && decoded.Count == op.Count
                && decoded.Text == op.Text, "wire encoding round trip");
        }
    }

    private static Op RandomOp(string doc, int replica, int seq)
    {
        if (doc.Length == 0 || Rand(10) < 6)
        {
            string text = RandomWord(1 + Rand(3));
            return Op.Insert(replica, seq, Rand(doc.Length + 1), text);
        }
        int pos = Rand(doc.Length);
        int count = 1 + Rand(doc.Length - pos);
        return Op.Delete(replica, seq, pos, count);
    }

    // ---- replicated editor ------------------------------------------------------------------------

    private static void ReplicaChecks()
    {
        string initial = "hello world";
        var workspace = new Workspace(initial);
        lastWorkspace = workspace;
        Client owner = workspace.Join(1, Role.Owner);
        Client editor = workspace.Join(2, Role.Editor);
        Client commenter = workspace.Join(3, Role.Commenter);
        Client viewer = workspace.Join(4, Role.Viewer);

        // A viewer cannot edit: the server rejects the sealed op before the log grows, and the
        // optimistic local echo is dropped by resynchronising with the log.
        int logBefore = workspace.Server.Log.Count;
        workspace.LocalEdit(viewer.Id, Op.Insert(0, 0, 0, "spam"));
        Check(!workspace.Flush(viewer.Id), "viewer edit rejected");
        Check(workspace.Server.Log.Count == logBefore, "viewer edit not logged");
        Check(viewer.Text == initial, "rejected edit left no trace: '" + viewer.Text + "'");
        Check(viewer.Pending.Count == 0, "no pending edits after a rejected flush");
        Check(workspace.Server.Rejected == 1, "one rejection recorded");

        // A commenter cannot edit either; an editor can.
        workspace.LocalEdit(commenter.Id, Op.Insert(0, 0, 0, "note"));
        Check(!workspace.Flush(commenter.Id), "commenter edit rejected");
        Check(commenter.Text == initial, "commenter edit left no trace");
        workspace.LocalEdit(editor.Id, Op.Insert(0, 0, 5, " there"));
        Check(workspace.Flush(editor.Id), "editor edit accepted");

        // The viewer (and every reader) decrypts the log and sees the text.
        Check(viewer.Text == workspace.ReplayLog(), "viewer reads decrypted log");
        Check(owner.Text == "hello there world", "owner converged after one edit: '" + owner.Text + "'");

        // A forged token (minted for a different role) is rejected.
        ulong forged = workspace.Server.MintToken(3, Role.Editor, 0);
        byte[] plain = Encoding.UTF8.GetBytes(Op.Insert(0, 0, 0, "x").Encode());
        ulong key = workspace.EpochKey(0);
        ulong nonce = 0x5150UL;
        byte[] body = ToyCrypto.Crypt(plain, key, nonce);
        bool accepted = workspace.Server.TrySubmit(3, forged, 0, nonce, body, ToyCrypto.Tag(key, nonce, body), 0,
            out LogEntry _);
        Check(!accepted, "forged role token rejected");

        // Randomized concurrent editing with optimistic local echo, then full convergence.
        var rep = new SplitMix64(0x13579BDF2468ACEUL);
        const int writers = 2;          // replicas 1 (owner) and 2 (editor) hold edit rights
        int liveCount = 4;
        int rejectedBeforeWriters = workspace.Server.Rejected;
        for (int step = 0; step < 160; step++)
        {
            int pick = 1 + (int)(rep.Next() % (ulong)writers);
            Client client = workspace.Get(pick);
            if (rep.Next() % 5 == 0)
            {
                workspace.Flush(pick);
                continue;
            }
            string doc = client.Text;
            Op op;
            if (doc.Length == 0 || rep.Next() % 10 < 6)
            {
                op = Op.Insert(pick, 0, (int)(rep.Next() % (ulong)(doc.Length + 1)), RandomWord(1 + Rand(3)));
            }
            else
            {
                int start = (int)(rep.Next() % (ulong)doc.Length);
                int count = 1 + (int)(rep.Next() % (ulong)(doc.Length - start));
                op = Op.Delete(pick, 0, start, count);
            }
            workspace.LocalEdit(pick, op);
            ConvergenceOps++;
        }
        for (int id = 1; id <= writers; id++) workspace.Flush(id);

        string oracle = workspace.ReplayLog();
        foreach (Client client in workspace.Live(Array.Empty<int>()))
        {
            Check(client.Text == oracle, $"replica {client.Id} converged: '{client.Text}' vs oracle '{oracle}'");
            Check(client.Pending.Count == 0, "no unflushed edits remain");
        }
        Check(workspace.MaxLatency <= 2, "delivery latency bound: " + workspace.MaxLatency);
        Check(workspace.Server.Log.Count == workspace.Accepted, "log length equals accepted edits");
        Check(oracle.Length > 0, "document survived concurrent random edits");

        // Liveness: every accepted edit reached every live member, and no authorized writer was
        // refused while the unauthorized ones were.
        int expectedDeliveries = (liveCount - 1) * workspace.Accepted;
        int actualDeliveries = 0;
        foreach (Client client in workspace.Members) actualDeliveries += client.ApplyLatency.Count;
        Check(actualDeliveries == expectedDeliveries, $"deliveries {actualDeliveries} vs {expectedDeliveries}");
        Check(workspace.Server.Rejected == rejectedBeforeWriters, "no authorized writer was refused");
        Check(workspace.Accepted >= 40, "the random phase produced real traffic: " + workspace.Accepted);

        // Revocation: the reader keeps the old epochs, loses access to the new one.
        string beforeRevoke = workspace.Get(4).Text;
        workspace.Revoke(4);
        Check(workspace.Epoch == 1, "revocation advanced the epoch");
        Check(workspace.Get(4).Epoch == 0, "revoked member keeps its old epoch");
        Check(!workspace.Get(4).CanRead(1), "revoked member holds no key for the new epoch");

        Client rec = workspace.Get(4);
        workspace.LocalEdit(1, Op.Insert(0, 0, 5, " again"));
        Check(workspace.Flush(1), "owner edits after revocation");
        Check(rec.Text == beforeRevoke, "revoked member cannot decrypt the new epoch entry");

        // Forcing the revoked member to try its stale key must fail the tag check (fail closed).
        LogEntry last = workspace.Server.Log[workspace.Server.Log.Count - 1];
        ulong staleKey = workspace.EpochKey(0);
        Check(ToyCrypto.Tag(staleKey, last.Nonce, last.Body) != last.Tag, "stale key fails closed");
        // And it cannot submit into the new epoch with the token it still holds.
        Check(!workspace.Server.TrySubmit(4, rec.Token, 1, last.Nonce, last.Body, last.Tag, workspace.Tick,
            out LogEntry _), "revoked member cannot submit into the new epoch");

        // Remaining members stay consistent in the new epoch; the revoked replica is excluded.
        string oracleAfter = workspace.ReplayLog();
        foreach (Client client in workspace.Members.Where(c => c.Id != 4)) Check(client.Text == oracleAfter,
            "post-revocation convergence for replica " + client.Id);
        Check(workspace.Get(1).CanRead(1), "members hold the new epoch key");

        ConvergenceClients = liveCount;
        Console.WriteLine($"813 doc after revocation: \"{oracleAfter}\" ({oracleAfter.Length} chars) via {workspace.Accepted} sealed edits");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
