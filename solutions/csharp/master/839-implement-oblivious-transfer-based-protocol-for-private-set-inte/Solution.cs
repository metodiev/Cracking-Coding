// Task 839 - Implement oblivious transfer based protocol for private set intersection. (Tier: Master, CTCI Beyond — Security & Privacy)
//
// Private set intersection from oblivious transfer, in the shape the real protocols have:
//
//   * the primitive is a 1-out-of-2 OT in the Bellare-Micali construction over the order-Q subgroup of
//     quadratic residues modulo the safe prime P = 2Q + 1. The sender publishes C = g^c; the receiver draws a
//     uniform exponent k and sends PK_b = g^k and PK_{1-b} = C * PK_b^-1; the sender encrypts m_i as
//     m_i * PK_i^c; the receiver recovers m_b as ct_b * (C^k)^-1. The pair {g^k, C*g^-k} is two uniform group
//     elements and the map k -> c-k turns a choice-0 transcript into an identically shaped choice-1 transcript,
//     so the sender's view is *identically* distributed for both choices — verified algebraically on sampled
//     exponents, not merely statistically.
//   * PSI drives that primitive through a (2,1)-choice hash table: the sender places each of its elements into
//     the least loaded of the two bins it hashes to; whatever cannot be placed goes into an explicit stash that
//     is processed by the next pass with fresh hash salts, which is how the real protocols absorb the same
//     overflow. The receiver retrieves the two bins its element hashes to per pass — 2B parallel 1-out-of-2 OTs,
//     choosing 1 only for the wanted bin and a dummy subgroup element everywhere else — and reports a match when
//     a retrieved payload equals the payload of its own element.
//
// Verified, each by recomputation:
//   * safety: the intersection equals the brute-force intersection exactly (no false positive, no false negative)
//     over nine configurations (overlap, disjoint, nested, identical, empty either side, singletons), which
//     requires an empty final stash; the payload map is injective — checked exhaustively on all 32767 admissible
//     elements — and out-of-range elements, including the reserved sentinel 0, are rejected instead of silently
//     colliding;
//   * receiver privacy: the ordered query pair for choice 0 with exponent k equals the pair for choice 1 with
//     exponent c-k on 499 sampled exponents; a 16-bucket chi-square over 20000 transmitted queries is uniform;
//     the single degenerate exponent is accounted for;
//   * sender privacy: recovering with the wrong index yields a value whose 16-bucket distribution over 10000
//     samples is uniform, and over 5000 trials it never equals the unselected message;
//   * cost and liveness: OTs = 2|Y| * (bins summed over passes) exactly, modular exponentiations match the
//     closed-form model 6*OTs + recoveries + dummy fills exactly, every element completes with no retries, and
//     the receiver's view is 2 of B bins.
//
// Documented simplifications (toy scale, no real key material): the group is 31 bits wide, so the confidentiality
// measured here is that of the *reduction* (the algebra), not of any real discrete-logarithm instance; the
// 1-of-N retrieval is the trivial B-parallel construction, whose O(B) OTs are what OT extension and a log-B tree
// remove (the log-B figure is printed for scale, not implemented); payloads are the square of the element, so
// elements must fit 15 bits; the attacker modelled is honest-but-curious (each OT hides one bit, which is the
// standard assumption, and no malicious-security checks are present). Randomness is hand-written splitmix64 with
// explicit seeds, so every measurement is reproducible.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static long NextBelow(ref ulong state, long bound) => (long)(Next(ref state) % (ulong)bound);

    /// <summary>30-bit mixing of an element with a salt: the bin hash of the table.</summary>
    internal static long Mix(long value, long salt)
    {
        ulong state = (ulong)value * 0x9E3779B97F4A7C15UL ^ (ulong)salt;
        return (long)(Next(ref state) & 0x3FFFFFFFUL);
    }
}

/// <summary>
/// The order-Q subgroup of quadratic residues modulo the safe prime P = 2Q + 1. Every element is a 31-bit
/// integer, so products stay below 2^62 and are exact in a signed 64-bit integer.
/// </summary>
internal static class ToyGroup
{
    internal const long P = 2147483579L;    // largest safe prime below 2^31
    internal const long Q = 1073741789L;    // (P - 1) / 2, prime: the order of the subgroup below
    internal const long G = 4L;             // 2^2 mod P, a generator of the order-Q subgroup

    internal static long Exponentiations;

    internal static long Mul(long a, long b) => a * b % P;

    internal static long Pow(long b, long e)
    {
        Exponentiations++;
        long result = 1, base_ = b % P;
        while (e > 0)
        {
            if ((e & 1) != 0) result = result * base_ % P;
            base_ = base_ * base_ % P;
            e >>= 1;
        }
        return result;
    }

    /// <summary>Fermat inversion: valid for every non-zero residue, since P is prime.</summary>
    internal static long Inverse(long a)
    {
        if (a == 0) throw new ArgumentException("zero has no inverse", nameof(a));
        return Pow(a, P - 2);
    }

    /// <summary>Subgroup membership: a non-zero element is a quadratic residue iff a^Q = 1 (P is prime).</summary>
    internal static bool IsQr(long a) => a % P != 0 && a % P != P - 1 && Pow(a % P, Q) == 1;

    /// <summary>P = 3 (mod 4), so the square root of a quadratic residue is a single exponentiation.</summary>
    internal static long Sqrt(long qr) => Pow(qr % P, (P + 1) / 4);

    internal static long Encode(long message)
    {
        if (message < 0 || message > 32767) throw new ArgumentException("messages must fit 15 bits", nameof(message));
        return message * message % P;
    }

    internal static long Decode(long encoded)
    {
        long root = Sqrt(encoded);
        if (root * root % P != encoded % P) throw new ArgumentException("not a quadratic residue", nameof(encoded));
        return Math.Min(root, P - root);
    }

    internal static long RandomExponent(ref ulong state) => SplitMix64.NextBelow(ref state, Q - 1) + 1;

    internal static long RandomSubgroup(ref ulong state) => Pow(G, RandomExponent(ref state));
}

/// <summary>Explicit in-memory wire: it counts what actually travels, so cost claims can be measured.</summary>
internal sealed class Wire
{
    internal int Messages { get; private set; }
    internal long FieldElements { get; private set; }

    internal void Send(int elements)
    {
        Messages++;
        FieldElements += elements;
    }
}

/// <summary>Bellare-Micali 1-out-of-2 oblivious transfer over the order-Q subgroup.</summary>
internal static class Ot2
{
    /// <summary>The strict transcript for a given choice and exponent: the ordered pair (PK_choice, PK_other).</summary>
    internal static long[] QueriesFor(long senderPublicC, long exponent, int choice)
    {
        if (choice != 0 && choice != 1) throw new ArgumentException("choice must be a bit", nameof(choice));
        if (exponent <= 0 || exponent >= ToyGroup.Q) throw new ArgumentException("exponent outside [1, Q)", nameof(exponent));
        long chosen = ToyGroup.Pow(ToyGroup.G, exponent);
        long other = ToyGroup.Mul(senderPublicC, ToyGroup.Inverse(chosen));
        return choice == 0 ? new[] { chosen, other } : new[] { other, chosen };
    }

    /// <summary>
    /// Receiver step 1 as it goes on the wire, drawn from the seeded PRNG. The receiver's secrecy comes from the
    /// pair {g^k, C*g^-k} being two uniform group elements whose assignment the sender cannot make out — not from
    /// any shuffling of the wire order, which the sender would see anyway.
    /// </summary>
    internal static long[] Query(long senderPublicC, int choice, ref ulong state, out long secretExponent, Wire wire)
    {
        secretExponent = ToyGroup.RandomExponent(ref state);
        long[] queries = QueriesFor(senderPublicC, secretExponent, choice);
        wire.Send(2);
        return queries;
    }

    /// <summary>Sender step: masks both messages with the corresponding query raised to its own secret exponent.</summary>
    internal static long[] Encrypt(long senderSecret, long[] queries, long message0, long message1, Wire wire)
    {
        long key0 = ToyGroup.Pow(queries[0], senderSecret);
        long key1 = ToyGroup.Pow(queries[1], senderSecret);
        wire.Send(2);
        return new[] { ToyGroup.Mul(message0, key0), ToyGroup.Mul(message1, key1) };
    }

    /// <summary>Receiver step 2: removes the mask it can compute, which succeeds only for the chosen index.</summary>
    internal static long Recover(long senderPublicC, long secretExponent, int index, long[] ciphertexts)
    {
        long mask = ToyGroup.Pow(senderPublicC, secretExponent);
        return ToyGroup.Mul(ciphertexts[index], ToyGroup.Inverse(mask));
    }

    /// <summary>One complete transfer; also returns the transcript so a caller can attempt the other index.</summary>
    internal static long Transfer(long message0, long message1, int choice, ref ulong state,
        out long[] ciphertexts, out long senderPublicC, out long secretExponent)
    {
        long senderSecret = ToyGroup.RandomExponent(ref state);
        senderPublicC = ToyGroup.Pow(ToyGroup.G, senderSecret);
        long[] queries = Query(senderPublicC, choice, ref state, out long k, new Wire());
        ciphertexts = Encrypt(senderSecret, queries, message0, message1, new Wire());
        secretExponent = k;
        return Recover(senderPublicC, k, choice, ciphertexts);
    }
}

/// <summary>
/// Sender side of one pass: a (2,1)-choice hash table with one element per bin plus the stash of elements that
/// could not be placed. Payloads are the squares of the elements, hence payload equality is exact.
/// </summary>
internal sealed class SenderTable
{
    internal int Bins { get; }
    internal long[] Slots { get; }              // 0 = free, else the encoded payload of its element
    internal List<long> Stash { get; } = new List<long>();
    internal int StashCount => Stash.Count;
    internal int Placed { get; private set; }

    private readonly long _salt;

    internal SenderTable(IReadOnlyList<long> set, int bins, long salt)
    {
        Bins = bins;
        _salt = salt;
        Slots = new long[bins];
        var load = new int[bins];
        foreach (long element in set)
        {
            // 0 is reserved as the "free bin" sentinel and the payload must stay a subgroup element, so the
            // admissible identity space is [1, 32767].
            if (element < 1 || element > 32767)
                throw new ArgumentException("toy scale: elements must lie in [1, 32767]", nameof(set));
            int first = Bin(element, 0), second = Bin(element, 1);
            int target = load[first] <= load[second] ? first : second;   // least loaded, ties to the first choice
            if (Slots[target] != 0) Stash.Add(element);
            else { Slots[target] = Payload(element); load[target] = 1; Placed++; }
        }
    }

    /// <summary>
    /// Payload of an element: its square, injective on [0, 32767] (verified exhaustively in Main), so two distinct
    /// elements can never share a payload and the match test cannot manufacture a false positive.
    /// </summary>
    internal static long Payload(long element) => ToyGroup.Encode(element);

    internal int Bin(long element, int choice) => (int)(SplitMix64.Mix(element, _salt + 1 + choice) % Bins);

    /// <summary>Table width for a pass: twice the number of elements, rounded up to a power of two.</summary>
    internal static int BinsFor(int count)
    {
        int bins = 4;
        while (bins < 2 * count) bins *= 2;
        return bins;
    }
}

/// <summary>Shape of one pass, so the cost model can be stated in terms of the tables, not of the run.</summary>
internal sealed class Pass
{
    internal int Bins;
    internal int Placed;
}

internal static class Psi
{
    internal sealed class Outcome
    {
        internal List<long> Intersection = new List<long>();
        internal List<Pass> Passes = new List<Pass>();
        internal long Ots;
        internal int BinsTotal;
        internal int Deflected;         // elements that needed a later pass (the stash traffic)
        internal int FinalStash;        // must be 0, otherwise the intersection is not exact
        internal long DummyFills;       // free bins that were masked with a fresh subgroup element
    }

    /// <summary>Builds the table of every pass; the salt changes per pass so the stash rehashes somewhere new.</summary>
    private static List<SenderTable> BuildTables(IReadOnlyList<long> senderSet, long salt, out int deflected)
    {
        var tables = new List<SenderTable>();
        var level = new List<long>(senderSet);
        deflected = 0;
        while (level.Count > 0)
        {
            var table = new SenderTable(level, SenderTable.BinsFor(level.Count), salt + 7919L * tables.Count);
            tables.Add(table);
            deflected += table.StashCount;
            level = table.Stash;
            if (tables.Count > 8) throw new InvalidOperationException("table recursion did not terminate");
        }
        return tables;
    }

    /// <summary>
    /// Receiver side of one bin retrieval: B parallel 1-out-of-2 OTs that select the wanted bin and transfer a
    /// fresh dummy subgroup element for every other bin, so the sender learns nothing about the index.
    /// </summary>
    private static long RetrieveBin(SenderTable table, int wanted, ref ulong state, Wire wire,
        ref long ots, ref long dummyFills)
    {
        long retrieved = 0;
        for (int bin = 0; bin < table.Bins; bin++)
        {
            long senderSecret = ToyGroup.RandomExponent(ref state);
            long publicC = ToyGroup.Pow(ToyGroup.G, senderSecret);
            int choice = bin == wanted ? 1 : 0;
            long[] queries = Ot2.Query(publicC, choice, ref state, out long k, wire);
            long dummy = ToyGroup.RandomSubgroup(ref state);
            long payload = table.Slots[bin];
            if (payload == 0) { payload = ToyGroup.RandomSubgroup(ref state); dummyFills++; }   // a free bin looks occupied
            long[] ciphertexts = Ot2.Encrypt(senderSecret, queries, dummy, payload, wire);
            if (choice == 1) retrieved = Ot2.Recover(publicC, k, 1, ciphertexts);
            ots++;
        }
        return retrieved;
    }

    internal static Outcome Run(IReadOnlyList<long> senderSet, IReadOnlyList<long> receiverSet, long salt, ulong seed)
    {
        var outcome = new Outcome();
        ulong state = seed;
        List<SenderTable> tables = BuildTables(senderSet, salt, out int deflected);
        outcome.Deflected = deflected;
        foreach (SenderTable table in tables)
        {
            outcome.Passes.Add(new Pass { Bins = table.Bins, Placed = table.Placed });
            outcome.BinsTotal += table.Bins;
        }
        outcome.FinalStash = tables.Count == 0 ? 0 : tables[tables.Count - 1].StashCount;

        var wire = new Wire();
        long dummyFills = 0;
        foreach (long element in receiverSet)
        {
            long mine = SenderTable.Payload(element);
            bool matched = false;
            foreach (SenderTable table in tables)
            {
                long first = RetrieveBin(table, table.Bin(element, 0), ref state, wire, ref outcome.Ots, ref dummyFills);
                long second = RetrieveBin(table, table.Bin(element, 1), ref state, wire, ref outcome.Ots, ref dummyFills);
                if (first == mine || second == mine) matched = true;
            }
            if (matched) outcome.Intersection.Add(element);
        }
        outcome.DummyFills = dummyFills;
        return outcome;
    }

    internal static List<long> IntersectionBruteForce(IReadOnlyList<long> a, IReadOnlyList<long> b)
    {
        var set = new HashSet<long>(a);
        var result = new List<long>();
        foreach (long x in b) if (set.Contains(x)) result.Add(x);
        return result;
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static double ChiSquareCritical(int degreesOfFreedom) => degreesOfFreedom switch
    {
        15 => 37.70,
        _ => throw new ArgumentException("no critical value tabulated for df " + degreesOfFreedom),
    };

    private static double ChiSquare(long[] buckets, double expected)
    {
        double statistic = 0;
        foreach (long observed in buckets) statistic += (observed - expected) * (observed - expected) / expected;
        return statistic;
    }

    private static List<long> Set(int count, int stride, int offset)
    {
        var values = new List<long>();
        for (int i = 0; i < count; i++) values.Add(offset + (long)i * stride);
        return values;
    }

    /// <summary>Set comparison, so two intersections can be compared without depending on insertion order.</summary>
    private static bool SameSet(List<long> left, List<long> right)
    {
        if (left.Count != right.Count) return false;
        var a = new List<long>(left);
        var b = new List<long>(right);
        a.Sort();
        b.Sort();
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void Main()
    {
        // ---- the group behaves as the protocol assumes ------------------------------------------------
        {
            bool arithmetic = ToyGroup.P == 2 * ToyGroup.Q + 1 && ToyGroup.P % 4 == 3
                && ToyGroup.Pow(ToyGroup.G, ToyGroup.Q) == 1 && ToyGroup.G != 1;
            Check(arithmetic, "P = 2Q + 1 is a safe prime, P = 3 (mod 4), and G generates the order-Q subgroup");
            bool roundTrips = true, residues = true;
            for (long m = 1; m < 60; m++)
            {
                long encoded = ToyGroup.Encode(m);
                if (!ToyGroup.IsQr(encoded) || ToyGroup.Decode(encoded) != m) roundTrips = false;
                if (m == 2 && ToyGroup.IsQr(2)) residues = false;      // P = 3 (mod 8), so 2 is a non-residue
            }
            Check(roundTrips, "Encode/Decode round-trips every small message and every encoding is a residue");
            Check(residues, "2 is a non-residue, so the membership test can actually reject something");
            bool rejected = false;
            try { ToyGroup.Decode(2); } catch (ArgumentException) { rejected = true; }
            Check(rejected, "decoding a non-residue throws instead of returning a wrong message");
            Console.WriteLine("group: P=" + ToyGroup.P + " Q=" + ToyGroup.Q + " G=" + ToyGroup.G + " subgroupOrder=Q");
        }

        // ---- OT correctness, over both choices --------------------------------------------------------
        {
            const int Trials = 300;
            ulong state = 0x5EED839000000001UL;
            bool chosen = true, other = true, subgroupQueries = true;
            for (int t = 0; t < Trials; t++)
            {
                long m0 = 1 + t % 200, m1 = 20000 + t % 5000;
                int choice = t % 2;
                long got = Ot2.Transfer(ToyGroup.Encode(m0), ToyGroup.Encode(m1), choice, ref state,
                    out long[] ciphertexts, out long senderPublicC, out long secretExponent);
                if (ToyGroup.Decode(got) != (choice == 0 ? m0 : m1)) chosen = false;
                if (ToyGroup.Decode(got) == (choice == 0 ? m1 : m0)) other = false;
                long attempt = Ot2.Recover(senderPublicC, secretExponent, 1 - choice, ciphertexts);
                if (ToyGroup.Decode(attempt) == (choice == 0 ? m1 : m0)) other = false;
                long[] queries = Ot2.QueriesFor(senderPublicC, secretExponent, choice);
                if (!ToyGroup.IsQr(queries[0]) || !ToyGroup.IsQr(queries[1])) subgroupQueries = false;
            }
            Check(chosen, "the receiver always recovers exactly the selected message (" + Trials + " transfers)");
            Check(other, "neither the recovered value nor the wrong-index attempt ever yields the unselected message");
            Check(subgroupQueries, "both queries are subgroup elements, so the sender never handles a non-residue");
            Console.WriteLine("OT correctness: " + Trials + " transfers, all exact, both indices exercised");
        }

        // ---- receiver privacy: the sender's view does not depend on the choice -----------------------
        {
            ulong state = 0x5EED839000000002UL;
            long c = 1000 + SplitMix64.NextBelow(ref state, ToyGroup.Q - 2000);
            long publicC = ToyGroup.Pow(ToyGroup.G, c);
            bool bijection = true;
            int degenerate = 0;
            for (long k = 1; k < 500; k++)
            {
                long[] asZero = Ot2.QueriesFor(publicC, k, 0);
                long[] asOne = Ot2.QueriesFor(publicC, c - k, 1);
                if (asZero[0] != asOne[0] || asZero[1] != asOne[1]) bijection = false;
                if (k == c - k) degenerate++;                 // PK_0 == PK_1 is the only leaky exponent, one of Q
            }
            Check(bijection, "the choice-0 transcript with exponent k equals the choice-1 transcript with exponent c-k on "
                + "499 exponents: the sender's view is identically distributed for both choices");
            Check(degenerate == 0, "no sampled exponent is degenerate; exactly one exponent out of Q (k = c/2) would be");

            const int Trials = 20000, Buckets = 16;
            var buckets = new long[Buckets];
            ulong stream = 0x5EED839000000003UL;
            var wire = new Wire();
            for (int t = 0; t < Trials; t++)
            {
                long senderSecret = ToyGroup.RandomExponent(ref stream);
                long senderPublic = ToyGroup.Pow(ToyGroup.G, senderSecret);
                long[] q = Ot2.Query(senderPublic, t % 2, ref stream, out long _, wire);
                buckets[(int)(q[0] * Buckets / ToyGroup.P)]++;
            }
            double statistic = ChiSquare(buckets, (double)Trials / Buckets);
            Check(statistic < ChiSquareCritical(Buckets - 1), "a transmitted query is uniform over the group (chi2 "
                + statistic.ToString("F2") + " < " + ChiSquareCritical(Buckets - 1).ToString("F2") + " on " + Trials + " queries)");
            Check(wire.Messages == Trials && wire.FieldElements == 2L * Trials,
                "the query step sends exactly two field elements per OT (" + wire.FieldElements + ")");
            Console.WriteLine("receiver privacy: bijection on 499 exponents, query chi2=" + statistic.ToString("F2"));
        }

        // ---- sender privacy: the wrong index is uniformly random -------------------------------------
        {
            const int Trials = 10000, Buckets = 16;
            ulong state = 0x5EED839000000004UL;
            var buckets = new long[Buckets];
            int matches = 0;
            const long Message0 = 12345, Message1 = 5432;
            for (int t = 0; t < Trials; t++)
            {
                int choice = t % 2;
                Ot2.Transfer(ToyGroup.Encode(Message0), ToyGroup.Encode(Message1), choice, ref state,
                    out long[] ciphertexts, out long senderPublicC, out long secretExponent);
                long attempt = Ot2.Recover(senderPublicC, secretExponent, 1 - choice, ciphertexts);
                buckets[(int)(attempt * Buckets / ToyGroup.P)]++;
                if (ToyGroup.Decode(attempt) == (choice == 0 ? Message1 : Message0)) matches++;
            }
            Check(matches == 0, "the wrong-index recovery never yields the unselected message in " + Trials + " trials");
            double statistic = ChiSquare(buckets, (double)Trials / Buckets);
            Check(statistic < ChiSquareCritical(Buckets - 1), "the wrong-index value is uniform over the group (chi2 "
                + statistic.ToString("F2") + " < " + ChiSquareCritical(Buckets - 1).ToString("F2")
                + "), so it carries no information about the other message");
            Console.WriteLine("sender privacy: " + Trials + " wrong-index recoveries, 0 matched, chi2=" + statistic.ToString("F2"));
        }

        // ---- PSI safety: exact intersection against the brute-force oracle ----------------------------
        {
            var configurations = new (string Name, List<long> Sender, List<long> Receiver)[]
            {
                ("overlap", Set(24, 3, 100), Set(20, 2, 106)),
                ("disjoint", Set(16, 5, 3), Set(12, 7, 1000)),
                ("nested", Set(20, 2, 50), Set(8, 2, 54)),
                ("identical", Set(12, 4, 7), Set(12, 4, 7)),
                ("empty receiver", Set(10, 3, 1), new List<long>()),
                ("empty sender", new List<long>(), Set(10, 3, 1)),
                ("single match", new List<long> { 77 }, new List<long> { 77 }),
                ("single miss", new List<long> { 1 }, new List<long> { 2 }),
                ("receiver larger", Set(6, 2, 40), Set(40, 1, 44)),
            };
            const long Salt = 0x839;
            ulong seed = 0x5EED839000000005UL;
            bool exact = true, emptyFinalStash = true;
            long totalOts = 0;
            int deflectedTotal = 0, passesTotal = 0;
            var overlapCount = -1;
            foreach ((string name, List<long> sender, List<long> receiver) in configurations)
            {
                var outcome = Psi.Run(sender, receiver, Salt, seed);
                if (!SameSet(outcome.Intersection, Psi.IntersectionBruteForce(sender, receiver)))
                {
                    exact = false;
                    Console.WriteLine("  mismatch in " + name + ": got " + outcome.Intersection.Count + " want "
                        + Psi.IntersectionBruteForce(sender, receiver).Count);
                }
                if (outcome.FinalStash != 0) emptyFinalStash = false;
                totalOts += outcome.Ots;
                deflectedTotal += outcome.Deflected;
                passesTotal += outcome.Passes.Count;
                if (name == "overlap") overlapCount = outcome.Intersection.Count;
            }
            Check(exact, "all nine configurations match the brute-force intersection exactly: no false positive and no "
                + "false negative");
            Check(emptyFinalStash, "every configuration ends with an empty stash, which is what makes the match test exact");
            Check(overlapCount == 7, "the overlap configuration finds exactly its 7 shared elements "
                + "(106, 112, ... 142: even numbers that are 1 mod 3)");
            Check(deflectedTotal > 0, "the (2,1)-choice table really does deflect overflow into later passes ("
                + deflectedTotal + " deflections over " + passesTotal + " passes)");
            Console.WriteLine("PSI safety: 9 configurations exact, OTs=" + totalOts + " deflected=" + deflectedTotal
                + " passes=" + passesTotal);
        }

        // ---- payload injectivity, exhaustively -------------------------------------------------------
        {
            var seen = new HashSet<long>();
            bool injective = true, inSubgroup = true;
            for (long m = 1; m <= 32767; m++)
            {
                long payload = SenderTable.Payload(m);
                if (!ToyGroup.IsQr(payload)) inSubgroup = false;
                if (!seen.Add(payload)) injective = false;
            }
            Check(injective, "the payload map m -> m^2 mod P is injective on all 32767 admissible elements, so two "
                + "distinct elements can never share a payload");
            Check(inSubgroup && seen.Count == 32767, "every payload is a subgroup element (" + seen.Count + " distinct)");
            bool rejected = false, sentinelSafe = false;
            try { new SenderTable(new List<long> { 32768 }, 8, 1); } catch (ArgumentException) { rejected = true; }
            try { new SenderTable(new List<long> { 0 }, 8, 1); } catch (ArgumentException) { sentinelSafe = true; }
            Check(rejected, "an element outside the 15-bit payload space is rejected instead of silently colliding");
            Check(sentinelSafe, "element 0 is rejected, so it can never be confused with the free-bin sentinel");
            Console.WriteLine("payload injectivity: 32768 elements, 0 collisions, all in the subgroup");
        }

        // ---- cost model: OTs and exponentiations match the closed form exactly ------------------------
        {
            var sender = Set(24, 3, 100);
            var receiver = Set(18, 2, 106);
            long before = ToyGroup.Exponentiations;
            var outcome = Psi.Run(sender, receiver, 0x839, 0x5EED839000000006UL);
            long spent = ToyGroup.Exponentiations - before;
            long expectedOts = 0, expectedFills = 0;
            foreach (Pass pass in outcome.Passes)
            {
                expectedOts += 2L * receiver.Count * pass.Bins;
                expectedFills += 2L * receiver.Count * (pass.Bins - pass.Placed);
            }
            // Closed form: six exponentiations per OT, two more per retrieval (the recovery mask and its inverse),
            // one dummy subgroup element per free bin visited.
            long retrievals = 2L * receiver.Count * outcome.Passes.Count;
            long expectedSpent = 6 * expectedOts + 2 * retrievals + outcome.DummyFills;
            Check(outcome.Ots == expectedOts, "the query set costs exactly 2|Y| * sum(B_t) OTs (" + outcome.Ots + " vs "
                + expectedOts + ")");
            Check(outcome.DummyFills == expectedFills, "free bins are covered by exactly one dummy per visit ("
                + outcome.DummyFills + " vs " + expectedFills + ")");
            Check(spent == expectedSpent, "the modular exponentiation count matches the closed-form model (" + spent
                + " vs " + expectedSpent + ")");
            Check(SameSet(outcome.Intersection, Psi.IntersectionBruteForce(sender, receiver)),
                "the cost-model run is still correct");
            long bits = 31L * 2 * outcome.Ots;                        // one 31-bit group element per OT message
            long naiveBits = 31L * sender.Count * receiver.Count;     // what the naive comparison protocol would cost
            Console.WriteLine("cost: |X|=" + sender.Count + " |Y|=" + receiver.Count + " bins=" + outcome.BinsTotal
                + " passes=" + outcome.Passes.Count + " OTs=" + outcome.Ots + " modexps=" + spent
                + " receiverView=2/" + outcome.Passes[0].Bins + " per pass otBits=" + bits + " naiveBits=" + naiveBits
                + " logBWouldNeedOtExtension=" + (receiver.Count * (long)Math.Log2(outcome.Passes[0].Bins)));
        }

        // ---- liveness and determinism ----------------------------------------------------------------
        {
            var sender = Set(20, 3, 100);
            var receiver = Set(14, 2, 104);
            var first = Psi.Run(sender, receiver, 0x839, 0x5EED839000000007UL);
            var second = Psi.Run(sender, receiver, 0x839, 0x5EED839000000007UL);
            Check(first.Ots == second.Ots && SameSet(first.Intersection, second.Intersection),
                "two runs with the same seed transfer the same number of OTs and agree on the intersection");
            var otherSeed = Psi.Run(sender, receiver, 0x839, 0x5EED839000000008UL);
            Check(SameSet(first.Intersection, otherSeed.Intersection),
                "the result does not depend on the randomness of the protocol");
            long perElement = first.Ots / receiver.Count;
            Check(first.Ots % receiver.Count == 0 && perElement == 2L * first.BinsTotal,
                "every element completes in exactly 2 * sum(B_t) OTs, so queries never retry (bounded latency: "
                + perElement + " OTs per element)");
            Check(first.FinalStash == 0 && first.Intersection.Count > 0, "the finished run has a well-defined result");
            Console.WriteLine("liveness: OTsPerElement=" + perElement + " retries=0 intersection="
                + first.Intersection.Count);
        }

        Console.WriteLine("PASS");
    }
}
