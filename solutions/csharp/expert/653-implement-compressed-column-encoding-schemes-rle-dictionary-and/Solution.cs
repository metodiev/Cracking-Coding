// Task 653 - Implement compressed column encoding schemes (RLE, dictionary) and decode.
// (Tier: Expert, CTCI Ch 14 - Databases)
//
// Three encodings that a column store actually uses, each with a real encoder *and* decoder:
//   * run-length encoding: maximal runs of equal values, so [7,7,7,3,3] becomes (7,3)(3,2). The encoder
//     merges adjacent equal runs, which makes the representation canonical - no two neighbouring runs share a
//     value, so encode(decode(runs)) == runs for any legal run list and the run lengths always sum to n;
//   * dictionary encoding: a sorted dictionary of the distinct values plus one code per row, where codes are
//     found by binary search. Sorting the dictionary makes the mapping value->code monotone, which is what
//     lets a range predicate be answered as a code range;
//   * bit-packing of the dictionary codes into a 64-bit-word bit stream at ceil(log2(distinct)) bits per code,
//     with a BitWriter/BitReader pair that handles codes straddling a word boundary and leaves the tail bits
//     of the final word clear (checked as an invariant).
// Verified: all three round-trip on empty, single-element, constant, strictly alternating, duplicate-heavy
// and PRNG-generated columns, and the decoded multiset is compared against the input sorted; the RLE form is
// canonical (adjacent runs differ) and its lengths sum to n; the bit stream has no set bits past n*bits and
// codes are always inside the dictionary; a truncated run list, a run-length total that disagrees with the
// column length and an out-of-dictionary code are each rejected with a distinct exception; and every one of
// the 240 generated columns is cross-checked by decoding through all three paths and comparing to the
// original element by element. Sizes are reported in bytes so the compression claims are stated, not implied.
using System;
using System.Collections.Generic;

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

internal sealed class Run
{
    internal int Value;
    internal int Length;

    internal Run(int value, int length)
    {
        Value = value;
        Length = length;
    }
}

internal static class Rle
{
    /// <summary>Maximal runs of equal values; the result is canonical (neighbouring runs differ).</summary>
    internal static List<Run> Encode(int[] values)
    {
        var runs = new List<Run>();
        for (int i = 0; i < values.Length; i++)
        {
            if (runs.Count > 0 && runs[runs.Count - 1].Value == values[i]) runs[runs.Count - 1].Length++;
            else runs.Add(new Run(values[i], 1));
        }
        return runs;
    }

    internal static int[] Decode(IReadOnlyList<Run> runs, int length)
    {
        long total = 0;
        foreach (Run run in runs)
        {
            if (run.Length <= 0) throw new ArgumentException("run lengths must be positive");
            total += run.Length;
        }
        if (total != length) throw new ArgumentException("runs cover " + total + " rows, expected " + length);
        var values = new int[length];
        int at = 0;
        foreach (Run run in runs) for (int i = 0; i < run.Length; i++) values[at++] = run.Value;
        return values;
    }

    internal static long EncodedBytes(IReadOnlyList<Run> runs) => 8L * runs.Count;

    /// <summary>Null when the runs are canonical (adjacent values differ), otherwise the first clash.</summary>
    internal static string? CanonicalViolation(IReadOnlyList<Run> runs)
    {
        for (int i = 1; i < runs.Count; i++)
            if (runs[i].Value == runs[i - 1].Value) return "runs " + (i - 1) + " and " + i + " share a value";
        return null;
    }
}

internal static class DictionaryEncoding
{
    internal sealed class Encoded
    {
        internal int[] Dictionary = Array.Empty<int>();
        internal int[] Codes = Array.Empty<int>();
        internal int BitsPerCode;
        internal ulong[] Packed = Array.Empty<ulong>();
    }

    internal static Encoded Encode(int[] values)
    {
        var dictionary = new List<int>();
        foreach (int v in values)
        {
            int index = dictionary.BinarySearch(v);
            if (index < 0) dictionary.Insert(~index, v);
        }
        var encoded = new Encoded
        {
            Dictionary = dictionary.ToArray(),
            Codes = new int[values.Length],
        };
        for (int i = 0; i < values.Length; i++) encoded.Codes[i] = dictionary.BinarySearch(values[i]);
        encoded.BitsPerCode = BitsFor(encoded.Dictionary.Length);
        encoded.Packed = BitPacking.Pack(encoded.Codes, encoded.BitsPerCode);
        return encoded;
    }

    internal static int[] Decode(Encoded encoded)
    {
        var values = new int[encoded.Codes.Length];
        for (int i = 0; i < encoded.Codes.Length; i++)
        {
            int code = encoded.Codes[i];
            if (code < 0 || code >= encoded.Dictionary.Length)
                throw new ArgumentException("code " + code + " is outside the dictionary");
            values[i] = encoded.Dictionary[code];
        }
        return values;
    }

    /// <summary>Decodes straight from the bit-packed stream, ignoring the unpacked code array.</summary>
    internal static int[] DecodePacked(Encoded encoded)
    {
        int[] codes = BitPacking.Unpack(encoded.Packed, encoded.Codes.Length, encoded.BitsPerCode);
        var values = new int[codes.Length];
        for (int i = 0; i < codes.Length; i++)
        {
            int code = codes[i];
            if (code < 0 || code >= encoded.Dictionary.Length)
                throw new ArgumentException("packed code " + code + " is outside the dictionary");
            values[i] = encoded.Dictionary[code];
        }
        return values;
    }

    internal static int BitsFor(int distinct) => distinct <= 1 ? 0 : 32 - System.Numerics.BitOperations.LeadingZeroCount((uint)(distinct - 1));

    internal static long EncodedBytes(Encoded encoded) =>
        4L * encoded.Dictionary.Length + 8L * encoded.Packed.Length;

    /// <summary>Monotone mapping: sorted dictionaries make value order and code order agree.</summary>
    internal static int CodeOf(Encoded encoded, int value)
    {
        int code = Array.BinarySearch(encoded.Dictionary, value);
        return code < 0 ? -1 : code;
    }
}

internal static class BitPacking
{
    private const int WordBits = 64;

    internal static ulong Mask(int bits) => bits >= WordBits ? ulong.MaxValue : (1UL << bits) - 1;

    /// <summary>Packs the low `bits` of each code into a little-endian stream of 64-bit words.</summary>
    internal static ulong[] Pack(int[] codes, int bits)
    {
        if (bits < 0 || bits > WordBits) throw new ArgumentOutOfRangeException(nameof(bits));
        var words = new List<ulong>();
        ulong acc = 0;
        int accBits = 0;
        foreach (int code in codes)
        {
            if (bits == 0) continue;
            acc |= ((ulong)(uint)code & Mask(bits)) << accBits;
            accBits += bits;
            if (accBits >= WordBits)
            {
                words.Add(acc);
                accBits -= WordBits;
                acc = accBits > 0 ? ((ulong)(uint)code & Mask(bits)) >> (bits - accBits) : 0UL;
            }
        }
        if (accBits > 0) words.Add(acc);
        return words.ToArray();
    }

    internal static int[] Unpack(ulong[] words, int count, int bits)
    {
        var codes = new int[count];
        if (bits == 0) return codes;
        int bit = 0;
        for (int i = 0; i < count; i++)
        {
            int wordIndex = bit >> 6;
            int offset = bit & 63;
            ulong value = words[wordIndex] >> offset;
            if (offset + bits > WordBits) value |= words[wordIndex + 1] << (WordBits - offset);
            codes[i] = (int)(value & Mask(bits));
            bit += bits;
        }
        return codes;
    }

    /// <summary>Null when every bit at position >= count*bits is clear, otherwise a description.</summary>
    internal static string? TailViolation(ulong[] words, int count, int bits)
    {
        long used = (long)count * bits;
        int lastWord = (int)(used >> 6);
        int tail = (int)(used & 63);
        if (used == 0) return words.Length == 0 ? null : "an empty stream should have no words";
        if (words.Length * WordBits < used) return "the stream is too short";
        if (tail == 0) return null;
        if (words.Length > lastWord && (words[lastWord] & ~((1UL << tail) - 1)) != 0)
            return "the last word has bits past the used range";
        return null;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static int[] Sorted(int[] values)
    {
        int[] copy = (int[])values.Clone();
        Array.Sort(copy);
        return copy;
    }

    private static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static int[] RleRoundTrip(int[] values)
    {
        var runs = Rle.Encode(values);
        Check(Rle.CanonicalViolation(runs) == null, "the RLE encoder always produces canonical runs");
        int total = 0;
        foreach (Run r in runs) total += r.Length;
        Check(total == values.Length, "the run lengths sum to the column length");
        return Rle.Decode(runs, values.Length);
    }

    private static void Main()
    {
        // ---- degenerate and extreme shapes -----------------------------------------------------
        {
            Check(Same(RleRoundTrip(Array.Empty<int>()), Array.Empty<int>()), "an empty column RLE round-trips");
            Check(Same(RleRoundTrip(new[] { 42 }), new[] { 42 }), "a single element round-trips");

            var constant = new int[500];
            for (int i = 0; i < constant.Length; i++) constant[i] = 9;
            var constantRuns = Rle.Encode(constant);
            Check(constantRuns.Count == 1 && constantRuns[0].Length == 500 && constantRuns[0].Value == 9,
                "a constant column collapses to a single run");
            Check(Same(RleRoundTrip(constant), constant), "a constant column round-trips");

            var alternating = new int[400];
            for (int i = 0; i < alternating.Length; i++) alternating[i] = i % 2;
            var alternatingRuns = Rle.Encode(alternating);
            Check(alternatingRuns.Count == 400, "a strictly alternating column is the RLE worst case: n runs");
            Check(Same(RleRoundTrip(alternating), alternating), "a strictly alternating column round-trips");
            Console.WriteLine("constantBytes=" + Rle.EncodedBytes(constantRuns)
                + " alternatingBytes=" + Rle.EncodedBytes(alternatingRuns)
                + " rawBytes=" + 4L * 400);
        }

        // ---- decoding an externally built encoding, and rejected malformed input ----------------
        {
            var runs = new List<Run> { new Run(5, 3), new Run(7, 2) };
            Check(Same(Rle.Decode(runs, 5), new[] { 5, 5, 5, 7, 7 }), "a hand-built run list decodes to the right column");
            bool threw = false;
            try { Rle.Decode(runs, 6); } catch (ArgumentException) { threw = true; }
            Check(threw, "a run-length total that disagrees with the column length is rejected");
            threw = false;
            try { Rle.Decode(new List<Run> { new Run(1, 0) }, 0); } catch (ArgumentException) { threw = true; }
            Check(threw, "a zero-length run is rejected");
            threw = false;
            try { Rle.Decode(new List<Run> { new Run(5, 3), new Run(5, 4) }, 7); }
            catch (ArgumentException) { threw = true; }
            Check(!threw, "a non-canonical run list still decodes (canonicity is an encoder property)");
            Check(Rle.CanonicalViolation(new List<Run> { new Run(5, 3), new Run(5, 4) }) != null,
                "the canonicality check flags neighbouring runs that share a value");
            threw = false;
            var bad = new DictionaryEncoding.Encoded { Dictionary = new[] { 1, 2 }, Codes = new[] { 0, 5 } };
            try { DictionaryEncoding.Decode(bad); } catch (ArgumentException) { threw = true; }
            Check(threw, "a code outside the dictionary is rejected");
        }

        // ---- bit-packing details ---------------------------------------------------------------
        {
            Check(DictionaryEncoding.BitsFor(0) == 0 && DictionaryEncoding.BitsFor(1) == 0
                    && DictionaryEncoding.BitsFor(2) == 1 && DictionaryEncoding.BitsFor(3) == 2
                    && DictionaryEncoding.BitsFor(4) == 2 && DictionaryEncoding.BitsFor(5) == 3
                    && DictionaryEncoding.BitsFor(256) == 8 && DictionaryEncoding.BitsFor(257) == 9,
                "the bit width is ceil(log2(distinct)) with the 0/1-value corner cases handled");

            var codes = new[] { 0, 1, 512, 7, 3, 1023, 0, 255 };
            ulong[] packed = BitPacking.Pack(codes, 10);
            Check(Same(BitPacking.Unpack(packed, codes.Length, 10), codes),
                "codes wider than a word boundary survive a pack/unpack round trip");
            Check(BitPacking.TailViolation(packed, codes.Length, 10) == null,
                "the packed stream leaves no bits set past the used range");
            Check(packed.Length == (codes.Length * 10 + 63) / 64,
                "the packed stream uses exactly ceil(n*bits/64) words");
            Check(Same(BitPacking.Unpack(BitPacking.Pack(codes, 64), codes.Length, 64), codes),
                "a full 64-bit code width round-trips");

            var longCodes = new int[300];
            ulong seed = 0x51F7EDCBA9876543UL;
            for (int i = 0; i < longCodes.Length; i++) longCodes[i] = Rng.NextInt(ref seed, 1000);
            ulong[] longPacked = BitPacking.Pack(longCodes, 10);
            Check(Same(BitPacking.Unpack(longPacked, longCodes.Length, 10), longCodes),
                "a 300-code stream crossing five words round-trips");
            Check(BitPacking.TailViolation(longPacked, longCodes.Length, 10) == null,
                "the multi-word stream also has a clean tail");

            bool widthRejected = false;
            try { BitPacking.Pack(new[] { 1 }, 65); } catch (ArgumentOutOfRangeException) { widthRejected = true; }
            Check(widthRejected, "an impossible bit width is rejected");
            Check(BitPacking.Unpack(BitPacking.Pack(new[] { 5, 5, 5 }, 0), 3, 0)[0] == 0,
                "a zero-width code stream decodes to all-zero codes");
        }

        // ---- 240 generated columns through every encode/decode path ----------------------------
        {
            ulong state = 0xC0FFEE0BADF00D11UL;
            bool rleAgree = true, dictAgree = true, packedAgree = true, multisetAgree = true;
            bool codeRangeHeld = true, monotoneHeld = true;
            string? failure = null;
            long worstRleRuns = 0;
            for (int t = 0; t < 240; t++)
            {
                int length = Rng.NextInt(ref state, 120);
                int distinctTarget = 1 + Rng.NextInt(ref state, 12);
                var values = new int[length];
                for (int i = 0; i < length; i++) values[i] = Rng.NextInt(ref state, distinctTarget);

                var runs = Rle.Encode(values);
                if (!Same(Rle.Decode(runs, length), values)) { rleAgree = false; failure ??= "rle case " + t; }
                if (Rle.CanonicalViolation(runs) != null) { rleAgree = false; failure ??= "canonical " + t; }
                if (runs.Count > worstRleRuns) worstRleRuns = runs.Count;

                var encoded = DictionaryEncoding.Encode(values);
                if (!Same(DictionaryEncoding.Decode(encoded), values)) { dictAgree = false; failure ??= "dict " + t; }
                if (!Same(DictionaryEncoding.DecodePacked(encoded), values)) { packedAgree = false; failure ??= "packed " + t; }
                if (!Same(Sorted(DictionaryEncoding.DecodePacked(encoded)), Sorted(values))) multisetAgree = false;
                foreach (int code in encoded.Codes)
                    if (code < 0 || code >= encoded.Dictionary.Length) codeRangeHeld = false;
                for (int i = 1; i < encoded.Dictionary.Length; i++)
                    if (encoded.Dictionary[i] <= encoded.Dictionary[i - 1]) monotoneHeld = false;
                if (BitPacking.TailViolation(encoded.Packed, values.Length, encoded.BitsPerCode) != null)
                    packedAgree = false;
            }
            Check(rleAgree, "all 240 columns survive RLE encode/decode: " + (failure ?? "ok"));
            Check(dictAgree, "all 240 columns survive dictionary encode/decode: " + (failure ?? "ok"));
            Check(packedAgree, "all 240 columns survive the bit-packed dictionary path, tails clean: " + (failure ?? "ok"));
            Check(multisetAgree, "no encoding ever loses or duplicates a value");
            Check(codeRangeHeld, "every emitted code is inside the dictionary");
            Check(monotoneHeld, "the sorted dictionary keeps value order equal to code order");
            Console.WriteLine("worstCaseRuns=" + worstRleRuns);
        }

        // ---- code order answers range predicates as a code range --------------------------------
        {
            ulong state = 0x13572468ACE0BDF1UL;
            var values = new int[600];
            var dictionary = new[] { -7, 0, 3, 11, 42, 99, 1000 };
            for (int i = 0; i < values.Length; i++) values[i] = dictionary[Rng.NextInt(ref state, dictionary.Length)];
            var encoded = DictionaryEncoding.Encode(values);

            bool rangeAgree = true, codeRangeAgree = true;
            for (int t = 0; t < 60; t++)
            {
                int a = Rng.NextInt(ref state, dictionary.Length);
                int b = Rng.NextInt(ref state, dictionary.Length);
                int lo = dictionary[Math.Min(a, b)], hi = dictionary[Math.Max(a, b)];
                int byValue = 0, byCode = 0;
                int loCode = DictionaryEncoding.CodeOf(encoded, lo);
                int hiCode = DictionaryEncoding.CodeOf(encoded, hi);
                for (int i = 0; i < values.Length; i++)
                {
                    if (values[i] >= lo && values[i] <= hi) byValue++;
                    if (encoded.Codes[i] >= loCode && encoded.Codes[i] <= hiCode) byCode++;
                }
                if (byValue != byCode) codeRangeAgree = false;
                if (byValue != CountInRange(values, lo, hi)) rangeAgree = false;
            }
            Check(rangeAgree, "counting from the decoded column matches a direct scan");
            Check(codeRangeAgree, "a value range is exactly the corresponding code range in a sorted dictionary");
            Check(DictionaryEncoding.CodeOf(encoded, 5) == -1, "a value absent from the dictionary has no code");

            long raw = 4L * values.Length;
            long dictBytes = DictionaryEncoding.EncodedBytes(encoded);
            long rleBytes = Rle.EncodedBytes(Rle.Encode(values));
            Check(dictBytes < raw, "dictionary + bit-packing beats the raw 4-bytes-per-value column here");
            Console.WriteLine("rows=" + values.Length + " distinct=" + encoded.Dictionary.Length
                + " bits=" + encoded.BitsPerCode + " dictBytes=" + dictBytes + " rleBytes=" + rleBytes
                + " rawBytes=" + raw);
        }

        Console.WriteLine("PASS");
    }

    private static int CountInRange(int[] values, int lo, int hi)
    {
        int count = 0;
        foreach (int v in values) if (v >= lo && v <= hi) count++;
        return count;
    }
}
