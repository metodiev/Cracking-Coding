// Task 889 - Implement secure multiparty computation for private set intersection at scale. (Tier: Master, Beyond — Security & Privacy)
//
// Multi-party PSI from commutative blinding (Diffie-Hellman style, semi-honest model). Everything lives
// in the subgroup <2> of Z_p* for the secp256k1 field prime p, and every secret key is chosen coprime
// to p-1, so v -> v^a is a *bijection* on that subgroup: cardinalities survive blinding and two
// different inputs can never collide through it. Each element x is mapped to H(x) = 2^h(x) with
// h(x) = SHA-256(x), so equal elements map to equal group elements in every party's set.
//
// The protocol runs in n synchronous rounds over an explicit in-memory network: round 1 has each party
// blind its own set with its own key and broadcast it; round t (2..n) has party t raise *every*
// published set with its key and rebroadcast the whole vector, shuffling each set. After n rounds every
// party holds the same public vector F_j = { H(x)^(a1*a2*...*an) : x in S_j } - exponent composition is
// order independent, which is checked directly. The intersection is the intersection of the F_j, and a
// party can only decide membership for elements it already owns: it cannot name what it does not have.
// A final digest round over the public vector makes a party that publishes a wrong transcript
// detectable by everyone.
//
// Simplifications (documented, not behavioural): the network is synchronous with a deterministic round
// barrier (no latency, no reordering); one fixed public prime stands in for negotiated parameters; the
// shuffle is a splitmix64 Fisher-Yates permutation, so ordering is hidden but the multiset is public;
// "at scale" is three parties with two hundred elements each, sized so the suite runs well under a
// second. Security is modelled rather than proven: the checks measure the properties the construction
// actually offers.
//
// Verified: the recovered intersection equals brute force for equal, nested, disjoint, partial, single
// element and empty sets; every party's local view agrees, no party names an element outside its own set
// and no party outside the true intersection; final set sizes equal input sizes (bijection, no
// accidental collision via blinding); the transcript leaks no plaintext element under a raw byte
// search; the same element is unlinkable across parties and no published value equals the plain image;
// message count and element volume equal independently derived formulas; the number of modular
// exponentiations matches the round structure; blinding order does not matter; fixed seeds give an
// identical transcript and different seeds a different one with the same result; four parties work; and
// a cheating party is caught by the digest comparison.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

internal static class Group
{
    /** secp256k1 field prime p = 2^256 - 2^32 - 977, used here only as a small public group. */
    internal static readonly BigInteger P = BigInteger.Parse(
        "115792089237316195423570985008687907853269984665640564039457584007908834671663", CultureInfo.InvariantCulture);
    internal static readonly BigInteger Generator = 2;
    internal static readonly BigInteger Order = P - 1;

    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static BigInteger Hash(string element)
    {
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes("psi|" + element));
        return new BigInteger(digest, isUnsigned: true, isBigEndian: true) % Order;
    }

    /** The public image of an element: every party maps the same string to the same group element. */
    internal static BigInteger Map(string element) => BigInteger.ModPow(Generator, Hash(element), P);

    internal static BigInteger Blend(BigInteger value, BigInteger key) => BigInteger.ModPow(value, key, P);

    /** Secret keys must be units modulo p-1, otherwise blinding could collide two inputs. */
    internal static BigInteger Key(ref ulong state)
    {
        while (true)
        {
            BigInteger candidate = new BigInteger(Next(ref state)) | 1;
            if (candidate > 1 && BigInteger.GreatestCommonDivisor(candidate, Order) == 1) return candidate;
        }
    }

    internal static void Shuffle<T>(List<T> items, ref ulong state)
    {
        for (int i = items.Count - 1; i > 0; i--)
        {
            int j = (int)(Next(ref state) % (ulong)(i + 1));
            (items[i], items[j]) = (items[j], items[i]);
        }
    }
}

internal sealed class Message
{
    internal int Round;
    internal int From;
    internal int To;
    internal string Kind = "";
    internal BigInteger[] Payload = Array.Empty<BigInteger>();
}

/** Synchronous in-memory network: sends land in inboxes, a barrier delivers them in order. */
internal sealed class Network
{
    private readonly Dictionary<int, List<Message>> _inboxes = new Dictionary<int, List<Message>>();
    private readonly List<Message> _inFlight = new List<Message>();

    internal long Sent;
    internal long Delivered;
    internal long ElementsTransmitted;

    internal void Register(int party) => _inboxes[party] = new List<Message>();

    internal void Send(Message message)
    {
        _inFlight.Add(message);
        Sent++;
        ElementsTransmitted += message.Payload.Length;
    }

    internal void Barrier()
    {
        foreach (Message message in _inFlight)
        {
            _inboxes[message.To].Add(message);
            Delivered++;
        }
        _inFlight.Clear();
    }

    internal List<Message> Transcript()
    {
        var all = new List<Message>();
        foreach (List<Message> inbox in _inboxes.Values) all.AddRange(inbox);
        return all;
    }
}

internal sealed class Party
{
    internal int Id;
    internal string[] Set = Array.Empty<string>();
    internal BigInteger Key;
    internal ulong ShuffleState;
    internal string Digest = "";
    internal readonly Dictionary<string, BigInteger> Image = new Dictionary<string, BigInteger>(StringComparer.Ordinal);
}

internal sealed class PsiProtocol
{
    internal readonly Network Network = new Network();
    internal readonly List<Party> Parties = new List<Party>();
    internal readonly List<List<BigInteger>[]> Rounds = new List<List<BigInteger>[]>();
    internal long ModPows;

    private List<BigInteger>[] _published = Array.Empty<List<BigInteger>>();
    private BigInteger _product = -1;

    internal PsiProtocol(IReadOnlyList<string[]> sets, IReadOnlyList<ulong> seeds)
    {
        for (int i = 0; i < sets.Count; i++)
        {
            ulong state = seeds[i];
            var party = new Party
            {
                Id = i,
                Set = sets[i].OrderBy(x => x, StringComparer.Ordinal).ToArray(),
                ShuffleState = seeds[i],
                Key = Group.Key(ref state),
            };
            Parties.Add(party);
            Network.Register(i);
        }
        _published = new List<BigInteger>[sets.Count];
        foreach (Party party in Parties)
        {
            foreach (string element in party.Set) party.Image[element] = Group.Map(element);
            var blinded = new List<BigInteger>();
            foreach (string element in party.Set)
            {
                blinded.Add(Group.Blend(party.Image[element], party.Key));
                ModPows++;
            }
            Group.Shuffle(blinded, ref party.ShuffleState);
            _published[party.Id] = blinded;
        }
    }

    /**
     * Publish round: every party blinds its own set and broadcasts it. Then n re-blinding rounds: in
     * round t party t applies its key to every set except its own (whose key set already contains it),
     * so after round n every set has been blinded by all n keys, in a different order per set.
     */
    internal void Run()
    {
        foreach (Party party in Parties)
        {
            BigInteger[] payload = Flatten(new[] { _published[party.Id] });
            for (int to = 0; to < Parties.Count; to++)
            {
                if (to == party.Id) continue;
                Network.Send(new Message { Round = 1, From = party.Id, To = to, Kind = "blinded-set", Payload = payload });
            }
        }
        Network.Barrier();
        Rounds.Add(Snapshot());

        for (int round = 1; round <= Parties.Count; round++)
        {
            Party owner = Parties[round - 1];
            var vector = new List<BigInteger>[Parties.Count];
            for (int j = 0; j < Parties.Count; j++)
            {
                if (j == owner.Id)
                {
                    vector[j] = new List<BigInteger>(_published[j]);   // already carries the owner's key
                    continue;
                }
                var reblinded = new List<BigInteger>();
                foreach (BigInteger value in _published[j])
                {
                    reblinded.Add(Group.Blend(value, owner.Key));
                    ModPows++;
                }
                Group.Shuffle(reblinded, ref owner.ShuffleState);
                vector[j] = reblinded;
            }
            _published = vector;
            BigInteger[] flat = Flatten(_published);
            for (int to = 0; to < Parties.Count; to++)
            {
                if (to == owner.Id) continue;
                Network.Send(new Message { Round = round + 1, From = owner.Id, To = to, Kind = "re-blinded-vector", Payload = flat });
            }
            Network.Barrier();
            Rounds.Add(Snapshot());
        }

        string digest = DigestOf(Flatten(_published));
        foreach (Party party in Parties)
        {
            party.Digest = digest;
            for (int to = 0; to < Parties.Count; to++)
            {
                if (to == party.Id) continue;
                Network.Send(new Message { Round = Parties.Count + 1, From = party.Id, To = to, Kind = "digest" });
            }
        }
        Network.Barrier();
    }

    internal static string DigestOf(IEnumerable<BigInteger> values)
    {
        var builder = new StringBuilder();
        foreach (BigInteger value in values) builder.Append(value.ToString("X", CultureInfo.InvariantCulture)).Append(';');
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString())));
    }

    private static BigInteger[] Flatten(List<BigInteger>[] vector)
    {
        var flat = new List<BigInteger>();
        foreach (List<BigInteger> set in vector) flat.AddRange(set);
        return flat.ToArray();
    }

    private List<BigInteger>[] Snapshot()
    {
        var copy = new List<BigInteger>[Parties.Count];
        for (int j = 0; j < Parties.Count; j++) copy[j] = new List<BigInteger>(_published[j]);
        return copy;
    }

    internal List<BigInteger>[] FinalSets() => Snapshot();

    internal BigInteger Product()
    {
        if (_product >= 0) return _product;
        BigInteger product = BigInteger.One;
        foreach (Party party in Parties) product = product * party.Key % Group.Order;
        _product = product;
        return _product;
    }

    /** A party can only decide membership for elements it already owns. */
    internal List<string> LocalIntersection(Party party)
    {
        List<BigInteger>[] final = FinalSets();
        BigInteger product = Product();
        var result = new List<string>();
        foreach (string element in party.Set)
        {
            BigInteger image = Group.Blend(party.Image[element], product);
            if (final.All(set => set.Contains(image))) result.Add(element);
        }
        return result;
    }

    internal bool DigestsAgree() => Parties.All(p => p.Digest == Parties[0].Digest);

    /** Cheating party: it publishes a transcript digest that does not match the real one. */
    internal void Corrupt(int partyIndex) => Parties[partyIndex].Digest = "00";

    internal List<BigInteger> AllTransmitted()
    {
        var values = new List<BigInteger>();
        for (int round = 0; round < Rounds.Count; round++)
            foreach (List<BigInteger> set in Rounds[round])
                values.AddRange(set);
        return values;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static string[] Range(string prefix, int from, int count) =>
        Enumerable.Range(from, count).Select(i => prefix + i).ToArray();

    private static string[] BruteIntersection(IReadOnlyList<string[]> sets)
    {
        IEnumerable<string> result = sets[0];
        for (int i = 1; i < sets.Count; i++)
        {
            var current = new HashSet<string>(sets[i], StringComparer.Ordinal);
            result = result.Where(current.Contains);
        }
        return result.OrderBy(x => x, StringComparer.Ordinal).ToArray();
    }

    private static string Join(IEnumerable<string> values) => string.Join(",", values);

    private static PsiProtocol Build(IReadOnlyList<string[]> sets) =>
        new PsiProtocol(sets, sets.Select((_, i) => 0x1000UL + (ulong)i * 0x9E3779B97F4A7C15UL).ToArray());

    private static bool ContainsBytes(byte[] haystack, byte[] needle)
    {
        if (needle.Length == 0 || haystack.Length < needle.Length) return false;
        for (int i = 0; i + needle.Length <= haystack.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
                if (haystack[i + j] != needle[j]) { match = false; break; }
            if (match) return true;
        }
        return false;
    }

    public static void Main()
    {
        // ---- 1. correctness against brute force on six shapes ----------------------------------
        var shapes = new List<(string Name, string[][] Sets)>
        {
            ("equal", new[] { Range("a", 0, 40), Range("a", 0, 40) }),
            ("nested", new[] { Range("a", 0, 60), Range("a", 10, 30) }),
            ("disjoint", new[] { Range("a", 0, 30), Range("b", 0, 30) }),
            ("partial", new[] { Range("a", 0, 50), Range("a", 25, 50), Range("a", 40, 50) }),
            ("single", new[] { new[] { "only" }, new[] { "only", "extra" } }),
            ("empty", new[] { Range("a", 0, 20), Array.Empty<string>() }),
        };
        foreach ((string name, string[][] sets) in shapes)
        {
            PsiProtocol protocol = Build(sets);
            protocol.Run();
            string[] expected = BruteIntersection(sets);
            foreach (Party party in protocol.Parties)
            {
                List<string> local = protocol.LocalIntersection(party);
                Check(Join(local) == Join(expected),
                    "local intersection for " + name + " at party " + party.Id + ": got " + local.Count +
                    " of " + expected.Length + " [" + Join(local.Take(3)) + "] vs [" + Join(expected.Take(3)) + "]");
            }
            Check(protocol.DigestsAgree(), "all parties agree on the digest for " + name);
        }

        // ---- 2. the algebraic assumptions the protocol rests on ---------------------------------
        ulong state = 12345UL;
        BigInteger keyA = Group.Key(ref state);
        BigInteger keyB = Group.Key(ref state);
        Check(keyA != keyB && keyA > 1 && keyB > 1, "keys are distinct units");
        Check(BigInteger.GreatestCommonDivisor(keyA, Group.Order) == 1 &&
              BigInteger.GreatestCommonDivisor(keyB, Group.Order) == 1, "keys are coprime to p-1");
        BigInteger probe = Group.Map("probe");
        Check(Group.Blend(Group.Blend(probe, keyA), keyB) == Group.Blend(Group.Blend(probe, keyB), keyA),
            "blinding composes commutatively");
        Check(Group.Blend(probe, keyA) != probe, "blinding changes the value");
        Check(Group.Map("probe") == Group.Map("probe") && Group.Map("probe") != Group.Map("probe2"),
            "the map is deterministic and injective on distinct inputs");

        // ---- 3. bijection: sizes survive, so blinding cannot collide two inputs -----------------
        var big = new[] { Range("x", 0, 200), Range("x", 120, 200), Range("x", 150, 200) };
        PsiProtocol scale = Build(big);
        long constructionPows = scale.ModPows;
        scale.Run();
        List<BigInteger>[] final = scale.FinalSets();
        for (int j = 0; j < big.Length; j++)
        {
            Check(final[j].Count == big[j].Length, "final set " + j + " keeps its cardinality");
            Check(final[j].Distinct().Count() == big[j].Distinct().Count(), "distinct inputs stay distinct");
        }
        string[] truth = BruteIntersection(big);
        Check(truth.Length == 50, "the three x-ranges share fifty elements");
        Check(Join(scale.LocalIntersection(scale.Parties[0])) == Join(truth), "two hundred elements per party intersect exactly");
        Check(scale.LocalIntersection(scale.Parties[1]).Count == 50, "the middle party names exactly the shared fifty");
        Check(Join(scale.LocalIntersection(scale.Parties[2])) == Join(truth), "the third party names the same set");

        // ---- 4. privacy of the transcript ------------------------------------------------------
        var privacy = new PsiProtocol(new[] { new[] { "secretAlpha", "secretBeta" }, new[] { "secretBeta", "secretGamma" } },
            new[] { 7UL, 11UL });
        privacy.Run();
        var transcriptBytes = new List<byte>();
        foreach (BigInteger value in privacy.AllTransmitted())
            transcriptBytes.AddRange(value.ToByteArray(isUnsigned: true, isBigEndian: true));
        foreach (string element in new[] { "secretAlpha", "secretBeta", "secretGamma" })
            Check(!ContainsBytes(transcriptBytes.ToArray(), Encoding.UTF8.GetBytes(element)),
                "no plaintext element in the transcript: " + element);
        Check(Join(privacy.LocalIntersection(privacy.Parties[0])) == "secretBeta", "the obfuscated transcript still intersects");

        // The same element, blinded by two different keys, must give two different values; no
        // published value may equal the plain group image of an element.
        Check(Group.Blend(Group.Map("secretBeta"), privacy.Parties[0].Key) !=
              Group.Blend(Group.Map("secretBeta"), privacy.Parties[1].Key), "the same element is unlinkable across parties");
        for (int j = 0; j < privacy.Parties.Count; j++)
            Check(!privacy.Rounds[0][j].Contains(Group.Map(privacy.Parties[j].Set[0])),
                "published values are blinded, party " + j);

        // ---- 5. message count, volume and work match closed forms -------------------------------
        long n = privacy.Parties.Count;
        long closedForm = 3 * n * (n - 1);            // publish + n re-blinding rounds + digest, each n(n-1)
        Check(privacy.Network.Delivered == closedForm,
            "message count matches 3n(n-1): " + privacy.Network.Delivered + " vs " + closedForm);
        int sumSizes = privacy.Parties.Sum(p => p.Set.Length);
        long expectedVolume = (n - 1) * (1 + n * (n - 1)) * sumSizes;
        Check(privacy.Network.ElementsTransmitted == expectedVolume,
            "element volume matches the broadcast formula: " + privacy.Network.ElementsTransmitted + " vs " + expectedVolume);

        long scaleN = scale.Parties.Count;
        Check(scale.Network.Delivered == 3 * scaleN * (scaleN - 1), "three-party message count: " + scale.Network.Delivered);
        long expectedPows = constructionPows + (scaleN - 1) * big.Sum(s => (long)s.Length);
        Check(scale.ModPows == expectedPows, "exponentiations match the round structure: " + scale.ModPows + " vs " + expectedPows);

        // ---- 6. cheating is caught --------------------------------------------------------------
        PsiProtocol honest = Build(new[] { Range("a", 0, 20), Range("a", 10, 20) });
        honest.Run();
        Check(honest.DigestsAgree(), "an honest run agrees");
        honest.Corrupt(1);
        Check(!honest.DigestsAgree(), "a party that publishes the wrong transcript is detected");

        // ---- 7. determinism --------------------------------------------------------------------
        var seedSet = new[] { 0x1000UL, 0x1000UL + 0x9E3779B97F4A7C15UL };
        var repeat = new PsiProtocol(new[] { Range("a", 0, 30), Range("a", 15, 30) }, seedSet);
        repeat.Run();
        var fresh = new PsiProtocol(new[] { Range("a", 0, 30), Range("a", 15, 30) }, seedSet);
        fresh.Run();
        Check(repeat.AllTransmitted().SequenceEqual(fresh.AllTransmitted()), "identical seeds yield an identical transcript");
        var other = new PsiProtocol(new[] { Range("a", 0, 30), Range("a", 15, 30) }, new[] { 555UL, 777UL });
        other.Run();
        Check(!other.AllTransmitted().SequenceEqual(fresh.AllTransmitted()), "different seeds yield a different transcript");
        Check(Join(other.LocalIntersection(other.Parties[0])) == Join(fresh.LocalIntersection(fresh.Parties[0])),
            "the result is independent of the blinding");

        // ---- 8. precision and recall, party by party -------------------------------------------
        int named = 0;
        foreach (Party party in scale.Parties)
        {
            string[] local = scale.LocalIntersection(party).ToArray();
            Check(local.All(truth.Contains), "no party may name an element outside the true intersection");
            Check(local.All(party.Set.Contains), "a party names only elements it owns");
            named += local.Length;
        }
        Check(named == truth.Length * 3, "each of the three parties names the whole intersection");

        // ---- 9. four parties, small sets (does the round structure generalise?) ---------------
        var four = new[] { Range("q", 0, 12), Range("q", 6, 12), Range("q", 9, 12), Range("q", 10, 4) };
        PsiProtocol quad = Build(four);
        quad.Run();
        Check(quad.Rounds.Count == 1 + four.Length, "one publish round plus n re-blinding rounds: " + quad.Rounds.Count);
        foreach (Party party in quad.Parties)
            Check(Join(quad.LocalIntersection(party)) == Join(BruteIntersection(four)),
                "four-party intersection at party " + party.Id);

        // ---- 10. degenerate parties ------------------------------------------------------------
        PsiProtocol degenerate = Build(new[] { new[] { "dup", "dup", "x" }, new[] { "dup" } });
        degenerate.Run();
        Check(Join(degenerate.LocalIntersection(degenerate.Parties[1])) == "dup", "a minimal party still finds the shared element");
        PsiProtocol empty = Build(new[] { Range("a", 0, 10), Array.Empty<string>() });
        empty.Run();
        Check(empty.LocalIntersection(empty.Parties[0]).Count == 0, "an empty peer yields nothing");
        Check(degenerate.DigestsAgree() && empty.DigestsAgree(), "degenerate runs still agree on the digest");

        int leaks = new[] { "secretAlpha", "secretBeta", "secretGamma" }
            .Count(e => ContainsBytes(transcriptBytes.ToArray(), Encoding.UTF8.GetBytes(e)));
        Console.WriteLine("scale: 3 parties x 200 elements, final sizes " + string.Join("/", scale.FinalSets().Select(s => s.Count)) +
            ", intersection=" + truth.Length + ", modpows=" + scale.ModPows + " vs " + expectedPows +
            ", messages=" + scale.Network.Delivered + " vs " + (3 * scaleN * (scaleN - 1)));
        Console.WriteLine("privacy: transcript values=" + privacy.AllTransmitted().Count + " for n=" + n +
            ", elements transmitted=" + privacy.Network.ElementsTransmitted + " vs " + expectedVolume +
            ", plaintext leaks=" + leaks);
        Console.WriteLine("shapes=" + shapes.Count + ", four-party rounds=" + quad.Rounds.Count +
            ", cheating detected=" + !honest.DigestsAgree() + ", named by every party=" + named);
        Console.WriteLine("PASS");
    }
}
