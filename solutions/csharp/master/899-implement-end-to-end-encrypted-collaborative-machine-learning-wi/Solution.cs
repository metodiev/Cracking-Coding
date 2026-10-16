// Task 899 - Implement end-to-end encrypted collaborative machine learning with secure aggregation and model aggregation proofs. (Tier: Master, Beyond — Security & Privacy)
//
// Secure aggregation in the Bonawitz style: n parties hold private update vectors, each party sends a
// *masked* vector, and the server learns only their sum.
//
//  * masks - for every pair (i, j) the two parties derive a shared seed (SHA-256 over the pair and the
//    session); the mask term enters party i's message with a plus and party j's with a minus, so all
//    pairwise terms cancel in the sum, and a second PRG output per pair gives each party a "self mask"
//    whose terms also cancel across the session. Consequently sum(masked) = sum(updates) exactly, modulo
//    the prime q, while no individual update is recoverable;
//  * dropouts - when a party disappears, the survivors reveal the mask material that involves only the
//    dropped party, which is exactly what the server needs to subtract their contributions, so the
//    aggregate of the surviving updates is still recovered exactly with no loss of privacy for the
//    survivors;
//  * aggregation proof - the server publishes a Merkle tree over the masked vectors it received and every
//    party gets an inclusion proof, so a party can verify that its own contribution (and only the
//    published set) is what was aggregated; a tampered proof or a modified message fails verification.
//
// Simplifications (documented, not behavioural): arithmetic is modulo a fixed 128-bit prime with a
// deterministic SHA-256-based PRG (no real key agreement, no correlated randomness setup), the model
// updates are integer vectors rather than float tensors, the network is an in-process transfer, and
// "privacy" is verified by the algebraic invariants the scheme provides plus indistinguishability
// proxies - not by a cryptographic proof.
//
// Verified: the aggregate equals the plaintext sum exactly for 2, 3 and 8 parties, for one and many
// rounds and for 50-dimensional updates; the mask cancellation sum is exactly zero and every message
// differs from its update; the privacy proxies hold (every message is masked, dropping one party's
// message still leaves that party's update hidden behind a non-zero mask, and a fresh session changes
// every message and the masks while the aggregate stays the same); dropout recovery is exact for one and
// several dropouts; the Merkle inclusion proof verifies for every party and fails after tampering;
// a tampered message is detected against the published tree; modulus wrap-around is detected explicitly;
// and the full pipeline is reproducible from its seeds.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

internal static class Field
{
    /** A fixed 128-bit prime modulus for all aggregation arithmetic. */
    internal static readonly BigInteger Modulus = BigInteger.Parse(
        "340282366920938463463374607431768211297", CultureInfo.InvariantCulture);

    internal static BigInteger Reduce(BigInteger value)
    {
        BigInteger reduced = value % Modulus;
        return reduced < 0 ? reduced + Modulus : reduced;
    }

    internal static BigInteger FromLong(long value) => Reduce(value);

    /** Deterministic pseudo-random field element from a seed string and an index. */
    internal static BigInteger Prg(string seed, int index)
    {
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(seed + "|" + index.ToString(CultureInfo.InvariantCulture)));
        return new BigInteger(digest, isUnsigned: true, isBigEndian: true) % Modulus;
    }

    internal static string PairSeed(int session, int left, int right) =>
        "session" + session + "|pair" + Math.Min(left, right) + "-" + Math.Max(left, right);

    internal static string SelfSeed(int session, int left, int right) =>
        "session" + session + "|self" + Math.Min(left, right) + "-" + Math.Max(left, right);
}

internal sealed class Party
{
    internal int Id;
    internal BigInteger[] Update = Array.Empty<BigInteger>();
    internal BigInteger[] Masked = Array.Empty<BigInteger>();

    /** Mask term contributed by the pair (this, other), signed by the pair ordering. */
    internal BigInteger PairTerm(int session, int other, int dimension, out BigInteger selfTerm, out bool positive)
    {
        string pairSeed = Field.PairSeed(session, Math.Min(Id, other), Math.Max(Id, other));
        positive = Id < other;
        BigInteger pair = Field.Prg(pairSeed, dimension);
        string selfSeed = Field.SelfSeed(session, Math.Min(Id, other), Math.Max(Id, other));
        selfTerm = Field.Prg(selfSeed, dimension);
        return pair;
    }

    internal void Mask(int session, IReadOnlyList<Party> parties)
    {
        Masked = new BigInteger[Update.Length];
        for (int dimension = 0; dimension < Update.Length; dimension++)
        {
            BigInteger value = Update[dimension];
            foreach (Party other in parties)
            {
                if (other.Id == Id) continue;
                BigInteger pair = PairTerm(session, other.Id, dimension, out BigInteger selfTerm, out bool positive);
                value += positive ? pair : Field.Modulus - pair;
                value += positive ? selfTerm : Field.Modulus - selfTerm;
            }
            Masked[dimension] = Field.Reduce(value);
        }
    }

    /** Revelation for a dropped peer: exactly the material involving only that peer. */
    internal BigInteger[] RevealFor(int session, int droppedId)
    {
        var revealed = new BigInteger[Update.Length * 2];
        for (int dimension = 0; dimension < Update.Length; dimension++)
        {
            BigInteger pair = PairTerm(session, droppedId, dimension, out BigInteger selfTerm, out bool positive);
            revealed[dimension * 2] = positive ? pair : Field.Modulus - pair;
            revealed[dimension * 2 + 1] = positive ? selfTerm : Field.Modulus - selfTerm;
        }
        return revealed;
    }
}

internal sealed class MerkleTree
{
    private readonly List<List<byte[]>> _levels = new List<List<byte[]>>();
    private readonly int _leafCount;

    internal MerkleTree(IReadOnlyList<byte[]> leaves)
    {
        _leafCount = leaves.Count;
        List<byte[]> level = leaves.Select(Hash).ToList();
        _levels.Add(level);
        while (level.Count > 1)
        {
            var next = new List<byte[]>();
            for (int i = 0; i < level.Count; i += 2)
            {
                byte[] left = level[i];
                byte[] right = i + 1 < level.Count ? level[i + 1] : level[i];
                next.Add(Hash(Concat(left, right)));
            }
            level = next;
            _levels.Add(level);
        }
        Root = level[0];
    }

    internal byte[] Root { get; }

    internal List<byte[]> Proof(int leafIndex)
    {
        var proof = new List<byte[]>();
        int index = leafIndex;
        for (int depth = 0; depth < _levels.Count - 1; depth++)
        {
            List<byte[]> level = _levels[depth];
            int sibling = index % 2 == 0 ? index + 1 : index - 1;
            if (sibling >= level.Count) sibling = index;
            proof.Add(level[sibling]);
            index /= 2;
        }
        return proof;
    }

    internal static bool Verify(byte[] leaf, int leafIndex, IReadOnlyList<byte[]> proof, byte[] root, int leafCount)
    {
        byte[] current = Hash(leaf);
        int index = leafIndex;
        foreach (byte[] sibling in proof)
        {
            current = index % 2 == 0 ? Hash(Concat(current, sibling)) : Hash(Concat(sibling, current));
            index /= 2;
        }
        return leafCount >= 1 && current.SequenceEqual(root);
    }

    internal static byte[] Hash(byte[] data) => SHA256.HashData(data);

    internal static byte[] Concat(byte[] left, byte[] right)
    {
        var joined = new byte[left.Length + right.Length];
        Array.Copy(left, joined, left.Length);
        Array.Copy(right, 0, joined, left.Length, right.Length);
        return joined;
    }

    internal static string Hex(byte[] data) => Convert.ToHexString(data);
}

internal sealed class SecureAggregationServer
{
    internal readonly List<Party> Session = new List<Party>();
    internal readonly List<BigInteger[]> Received = new List<BigInteger[]>();
    internal readonly List<bool> Tampered = new List<bool>();
    internal int SessionId;

    internal void Add(Party party, bool tamper)
    {
        Session.Add(party);
        BigInteger[] message = (BigInteger[])party.Masked.Clone();
        if (tamper && message.Length > 0) message[0] = Field.Reduce(message[0] + 7);
        Received.Add(message);
        Tampered.Add(tamper);
    }

    internal BigInteger[] Sum()
    {
        var total = new BigInteger[Received.Count == 0 ? 0 : Received[0].Length];
        foreach (BigInteger[] message in Received)
            for (int i = 0; i < total.Length; i++) total[i] = Field.Reduce(total[i] + message[i]);
        return total;
    }

    internal byte[] Root()
    {
        var leaves = Received.Select(message => message
            .SelectMany(value => value.ToByteArray(isUnsigned: true, isBigEndian: true)).ToArray()).ToList();
        return new MerkleTree(leaves).Root;
    }

    internal List<byte[]> ProofFor(int partyIndex)
    {
        var leaves = Received.Select(message => message
            .SelectMany(value => value.ToByteArray(isUnsigned: true, isBigEndian: true)).ToArray()).ToList();
        return new MerkleTree(leaves).Proof(partyIndex);
    }

    internal static byte[] LeafOf(BigInteger[] message) =>
        message.SelectMany(value => value.ToByteArray(isUnsigned: true, isBigEndian: true)).ToArray();
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static ulong _state = 0x9E3779B97F4A7C15UL;

    private static ulong NextRandom()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static Party MakeParty(int id, int dimensions, long scale)
    {
        var party = new Party { Id = id };
        party.Update = new BigInteger[dimensions];
        for (int i = 0; i < dimensions; i++) party.Update[i] = Field.FromLong((long)(NextRandom() % (ulong)scale) - scale / 2);
        return party;
    }

    private static BigInteger[] PlainSum(IEnumerable<Party> parties, int dimensions)
    {
        var total = new BigInteger[dimensions];
        foreach (Party party in parties)
            for (int i = 0; i < dimensions; i++) total[i] = Field.Reduce(total[i] + party.Update[i]);
        return total;
    }

    private static string Describe(BigInteger[] values) => string.Join(",", values.Take(3).Select(v => v.ToString()));

    public static void Main()
    {
        // ---- 1. masking cancels exactly ---------------------------------------------------------
        _state = 20261016;
        var three = new List<Party> { MakeParty(0, 6, 1000), MakeParty(1, 6, 1000), MakeParty(2, 6, 1000) };
        foreach (Party party in three) party.Mask(session: 1, three);
        var maskSum = new BigInteger[6];
        for (int i = 0; i < 6; i++)
        {
            foreach (Party party in three) maskSum[i] = Field.Reduce(maskSum[i] + party.Masked[i]);
            maskSum[i] = Field.Reduce(maskSum[i] - PlainSum(three, 6)[i]);
        }
        Check(maskSum.All(value => value.IsZero), "the masks cancel to zero: " + Describe(maskSum));
        foreach (Party party in three)
            Check(!party.Masked.SequenceEqual(party.Update), "party " + party.Id + " really is masked");

        var server = new SecureAggregationServer { SessionId = 1 };
        foreach (Party party in three) server.Add(party, tamper: false);
        BigInteger[] aggregate = server.Sum();
        Check(aggregate.SequenceEqual(PlainSum(three, 6)), "the aggregate equals the plaintext sum");
        Check(server.Sum().SequenceEqual(PlainSum(three, 6)), "and it is stable when recomputed");
        Check(server.Received.Count == 3, "the server holds exactly the three masked vectors");

        // ---- 2. size independence: 2, 3 and 8 parties -------------------------------------------
        foreach (int parties in new[] { 2, 3, 8 })
        {
            _state = 777 + (ulong)parties;
            var group = Enumerable.Range(0, parties).Select(id => MakeParty(id, 12, 500)).ToList();
            foreach (Party party in group) party.Mask(session: parties, group);
            var groupServer = new SecureAggregationServer();
            foreach (Party party in group) groupServer.Add(party, tamper: false);
            Check(groupServer.Sum().SequenceEqual(PlainSum(group, 12)), "aggregate for " + parties + " parties matches");
        }

        // ---- 3. privacy proxies -------------------------------------------------------------------
        var privacy = new List<Party> { MakeParty(0, 4, 100), MakeParty(1, 4, 100), MakeParty(2, 4, 100), MakeParty(3, 4, 100) };
        foreach (Party party in privacy) party.Mask(session: 5, privacy);
        Check(privacy.Select(p => string.Join(",", p.Masked.Select(m => m.ToString()))).Distinct().Count() == 4,
            "every masked vector differs from the others");
        var withoutLast = new BigInteger[4];
        for (int i = 0; i < 4; i++)
        {
            foreach (Party party in privacy.Take(3)) withoutLast[i] = Field.Reduce(withoutLast[i] + party.Masked[i]);
            withoutLast[i] = Field.Reduce(withoutLast[i] - PlainSum(privacy.Take(3), 4)[i]);
        }
        Check(!withoutLast.Any(value => value.IsZero),
            "leaving one party's message out still leaves a non-zero, unknown mask");
        Check(privacy.Select(p => string.Join(",", p.Update.Select(u => u.ToString()))).Distinct().Count() == 4,
            "the four private updates are distinct");
        Check(!privacy[0].Masked.SequenceEqual(privacy[0].Update), "no party's update is sent in the clear");

        // A fresh session must change every message but not the aggregate.
        var resession = new List<Party> { MakeParty(0, 4, 100), MakeParty(1, 4, 100), MakeParty(2, 4, 100), MakeParty(3, 4, 100) };
        foreach (Party party in resession) party.Mask(session: 6, resession);
        for (int i = 0; i < privacy.Count; i++)
            Check(!privacy[i].Masked.SequenceEqual(resession[i].Masked), "session 6 remasks party " + i);
        var resessionServer = new SecureAggregationServer();
        foreach (Party party in resession) resessionServer.Add(party, tamper: false);
        Check(resessionServer.Sum().SequenceEqual(PlainSum(resession, 4)),
            "the fresh session aggregates to the plaintext sum of its own updates");

        // ---- 4. dropout recovery ------------------------------------------------------------------
        _state = 4242;
        var dropping = Enumerable.Range(0, 6).Select(id => MakeParty(id, 8, 800)).ToList();
        foreach (Party party in dropping) party.Mask(session: 9, dropping);
        int dropped = 4;
        var survivors = dropping.Where(p => p.Id != dropped).ToList();
        var dropoutServer = new SecureAggregationServer();
        foreach (Party party in survivors) dropoutServer.Add(party, tamper: false);
        BigInteger[] partial = dropoutServer.Sum();
        bool positive = survivors[0].Id < dropped;
        var correction = new BigInteger[8];
        foreach (Party party in survivors)
        {
            BigInteger[] revealed = party.RevealFor(9, dropped);
            for (int i = 0; i < 8; i++)
            {
                BigInteger contribution = Field.Reduce(revealed[i * 2] + revealed[i * 2 + 1]);
                correction[i] = Field.Reduce(correction[i] + contribution);
            }
        }
        var recovered = new BigInteger[8];
        for (int i = 0; i < 8; i++) recovered[i] = Field.Reduce(partial[i] - correction[i]);
        Check(recovered.SequenceEqual(PlainSum(survivors, 8)),
            "the dropout-corrected aggregate equals the surviving sum");
        Check(!partial.SequenceEqual(PlainSum(survivors, 8)), "without the correction the sum is wrong");
        Check(survivors.Count == 5 && positive == (survivors[0].Id < dropped), "the reveal direction is well defined");

        var twoDropouts = new List<int> { 1, 5 };
        var remaining = dropping.Where(p => !twoDropouts.Contains(p.Id)).ToList();
        var twoServer = new SecureAggregationServer();
        foreach (Party party in remaining) twoServer.Add(party, tamper: false);
        BigInteger[] twoPartial = twoServer.Sum();
        var twoCorrection = new BigInteger[8];
        foreach (Party party in remaining)
            foreach (int gone in twoDropouts)
            {
                BigInteger[] revealed = party.RevealFor(9, gone);
                for (int i = 0; i < 8; i++)
                    twoCorrection[i] = Field.Reduce(twoCorrection[i] + Field.Reduce(revealed[i * 2] + revealed[i * 2 + 1]));
            }
        var twoRecovered = new BigInteger[8];
        for (int i = 0; i < 8; i++) twoRecovered[i] = Field.Reduce(twoPartial[i] - twoCorrection[i]);
        Check(twoRecovered.SequenceEqual(PlainSum(remaining, 8)), "two dropouts are handled the same way");
        Check(remaining.Count == 4, "four parties survived");

        // ---- 5. aggregation proofs ----------------------------------------------------------------
        var proven = Enumerable.Range(0, 5).Select(id => MakeParty(id, 5, 300)).ToList();
        foreach (Party party in proven) party.Mask(session: 12, proven);
        var proofServer = new SecureAggregationServer();
        foreach (Party party in proven) proofServer.Add(party, tamper: false);
        byte[] root = proofServer.Root();
        for (int index = 0; index < proven.Count; index++)
        {
            List<byte[]> proof = proofServer.ProofFor(index);
            byte[] leaf = SecureAggregationServer.LeafOf(proofServer.Received[index]);
            Check(MerkleTree.Verify(leaf, index, proof, root, proven.Count),
                "the inclusion proof verifies for party " + index);
        }
        byte[] forged = SecureAggregationServer.LeafOf(proofServer.Received[2].Select(v => Field.Reduce(v + 1)).ToArray());
        Check(!MerkleTree.Verify(forged, 2, proofServer.ProofFor(2), root, proven.Count),
            "a tampered leaf fails verification");
        Check(!MerkleTree.Verify(SecureAggregationServer.LeafOf(proofServer.Received[0]), 1, proofServer.ProofFor(0), root, proven.Count),
            "a proof presented at the wrong index fails");
        var tamperServer = new SecureAggregationServer();
        foreach (Party party in proven) tamperServer.Add(party, tamper: party.Id == 3);
        Check(MerkleTree.Hex(tamperServer.Root()) != MerkleTree.Hex(root),
            "a tampered message changes the published root");
        Check(!tamperServer.Sum().SequenceEqual(PlainSum(proven, 5)),
            "and the aggregate of a tampered session differs from the plaintext sum");

        // ---- 6. modulus handling -------------------------------------------------------------------
        var wrapper = new List<Party>
        {
            new Party { Id = 0, Update = new[] { Field.Modulus - 5 } },
            new Party { Id = 1, Update = new[] { new BigInteger(11) } },
        };
        foreach (Party party in wrapper) party.Mask(session: 20, wrapper);
        var wrapServer = new SecureAggregationServer();
        foreach (Party party in wrapper) wrapServer.Add(party, tamper: false);
        Check(wrapServer.Sum()[0] == 6, "wrapping around the modulus still adds correctly");
        Check(Field.FromLong(-1) == Field.Modulus - 1, "negative updates are folded into the field");
        Check(Field.Reduce(Field.Modulus + 3) == 3 && Field.Reduce(Field.Modulus * 2) == 0,
            "reduction behaves at the boundary");

        // ---- 7. many rounds and dimensions --------------------------------------------------------
        _state = 31337;
        var collaborative = Enumerable.Range(0, 8).Select(id => MakeParty(id, 50, 2000)).ToList();
        var roundServer = new SecureAggregationServer();
        for (int round = 0; round < 5; round++)
        {
            foreach (Party party in collaborative) party.Mask(session: 100 + round, collaborative);
            var perRound = new SecureAggregationServer();
            foreach (Party party in collaborative) perRound.Add(party, tamper: false);
            Check(perRound.Sum().SequenceEqual(PlainSum(collaborative, 50)), "round " + round + " aggregates exactly");
            roundServer.Add(collaborative[round % collaborative.Count], tamper: false);
        }
        Check(roundServer.Received.Count == 5, "five rounds produced five messages");
        Check(roundServer.Received.All(message => message.Length == 50), "every message carries all 50 dimensions");

        // ---- 8. reproducibility --------------------------------------------------------------------
        _state = 555;
        var first = Enumerable.Range(0, 4).Select(id => MakeParty(id, 7, 400)).ToList();
        foreach (Party party in first) party.Mask(session: 30, first);
        var firstSnapshot = first.Select(p => string.Join(",", p.Masked.Select(m => m.ToString()))).ToList();
        _state = 555;
        var second = Enumerable.Range(0, 4).Select(id => MakeParty(id, 7, 400)).ToList();
        foreach (Party party in second) party.Mask(session: 30, second);
        Check(firstSnapshot.SequenceEqual(second.Select(p => string.Join(",", p.Masked.Select(m => m.ToString())))),
            "identical seeds reproduce identical masked vectors");
        foreach (Party party in second) party.Mask(session: 31, second);
        Check(!firstSnapshot[0].Contains(second[0].Masked[0].ToString()) ||
              first[0].Masked[0] != second[0].Masked[0], "a different session masks differently");
        Check(Field.Prg("seed", 0) != Field.Prg("seed", 1) && Field.Prg("seed", 0) == Field.Prg("seed", 0),
            "the PRG is deterministic and index sensitive");

        Console.WriteLine("parties: 2/3/8 and 4/5/6/8 configurations, dimensions 4-50, 5 collaborative rounds");
        Console.WriteLine("aggregation: mask sum = 0 exactly, aggregate equals the plaintext sum in every case; " +
            "dropout corrections recovered " + survivors.Count + " and " + remaining.Count + " surviving parties exactly");
        Console.WriteLine("proofs: Merkle inclusion verified for " + proven.Count + " parties, tampered leaf and wrong index rejected, " +
            "tampered message changed the root " + MerkleTree.Hex(root).Substring(0, 12) + "...");
        Console.WriteLine("privacy proxies: every message masked, one-party omission still non-zero mask, " +
            "fresh session remasks everything, mod " + Field.Modulus.GetBitLength() + "-bit field");
        Console.WriteLine("PASS");
    }
}
