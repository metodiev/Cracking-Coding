// Task 389 - Implement Huffman coding for characters frequency. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;
using System.Text;

internal sealed class HuffNode
{
    public readonly long Freq;
    public readonly int Id;
    public readonly char Ch;
    public readonly bool Leaf;
    public readonly HuffNode? Left;
    public readonly HuffNode? Right;

    public HuffNode(long freq, int id, char ch, bool leaf, HuffNode? left, HuffNode? right)
    {
        Freq = freq;
        Id = id;
        Ch = ch;
        Leaf = leaf;
        Left = left;
        Right = right;
    }
}

public static class Solution
{
    private static Dictionary<char, int> CharacterFrequencies(string text)
    {
        var frequencies = new Dictionary<char, int>();
        foreach (char ch in text)
        {
            frequencies.TryGetValue(ch, out int count);
            frequencies[ch] = count + 1;
        }
        return frequencies;
    }

    /// <summary>
    /// Greedy min-heap merge with the documented deterministic tie-break: nodes ordered by
    /// (frequency, creation id); first popped node becomes the left child (bit '0'), the
    /// second the right child (bit '1').
    /// </summary>
    private static Dictionary<char, string> HuffmanCodes(Dictionary<char, int> frequencies)
    {
        var codes = new Dictionary<char, string>();
        if (frequencies.Count == 0) return codes;
        var chars = new List<char>(frequencies.Keys);
        chars.Sort();
        var heap = new PriorityQueue<HuffNode, (long, int)>();
        int id = 0;
        foreach (char ch in chars)
            heap.Enqueue(new HuffNode(frequencies[ch], id++, ch, true, null, null),
                    (frequencies[ch], id - 1));
        int nextId = id;
        while (heap.Count > 1)
        {
            var left = heap.Dequeue();
            var right = heap.Dequeue();
            long combined = left.Freq + right.Freq;
            heap.Enqueue(new HuffNode(combined, nextId, '\0', false, left, right), (combined, nextId));
            nextId++;
        }
        Walk(heap.Peek(), "", codes);
        return codes;
    }

    private static void Walk(HuffNode node, string prefix, Dictionary<char, string> codes)
    {
        if (node.Leaf)
        {
            codes[node.Ch] = prefix.Length == 0 ? "0" : prefix;
            return;
        }
        Walk(node.Left!, prefix + "0", codes);
        Walk(node.Right!, prefix + "1", codes);
    }

    private static long WeightedLength(Dictionary<char, string> codes, Dictionary<char, int> frequencies)
    {
        long total = 0;
        foreach (var entry in codes) total += (long)frequencies[entry.Key] * entry.Value.Length;
        return total;
    }

    private static string HuffmanEncode(string text, Dictionary<char, string> codes)
    {
        var bits = new StringBuilder();
        foreach (char ch in text) bits.Append(codes[ch]);
        return bits.ToString();
    }

    private static string HuffmanDecode(string bits, Dictionary<char, string> codes)
    {
        var reverse = new Dictionary<string, char>();
        foreach (var entry in codes) reverse[entry.Value] = entry.Key;
        var outBuilder = new StringBuilder();
        var current = new StringBuilder();
        foreach (char bit in bits)
        {
            current.Append(bit);
            if (reverse.TryGetValue(current.ToString(), out char ch))
            {
                outBuilder.Append(ch);
                current.Clear();
            }
        }
        if (current.Length > 0) throw new InvalidOperationException("bits end with an incomplete code");
        return outBuilder.ToString();
    }

    private static bool IsPrefixFree(Dictionary<char, string> codes)
    {
        foreach (string code in codes.Values)
            foreach (string other in codes.Values)
                if (code != other && other.StartsWith(code, StringComparison.Ordinal)) return false;
        return true;
    }

    /// <summary>sum 2^(-len) == 1 for a complete binary tree; the single-symbol table "0" is a lone leaf.</summary>
    private static bool KraftSumOk(Dictionary<char, string> codes)
    {
        if (codes.Count == 0) return true;
        if (codes.Count == 1)
        {
            foreach (string code in codes.Values) return code == "0";
        }
        int longest = 0;
        foreach (string code in codes.Values) longest = Math.Max(longest, code.Length);
        long sum = 0;
        foreach (string code in codes.Values) sum += 1L << (longest - code.Length);
        return sum == 1L << longest;
    }

    private static int CeilLog2(int value)
    {
        int bits = 0;
        while ((1 << bits) < value) bits++;
        return bits;
    }

    private static int seed = 2718;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void Main()
    {
        // hand-verified small example: a:2, b:3, c:4 -> {c: "0", a: "10", b: "11"}
        string text = "aabbbcccc";
        var freq = CharacterFrequencies(text);
        Check(freq['a'] == 2 && freq['b'] == 3 && freq['c'] == 4, "frequencies");
        var codes = HuffmanCodes(freq);
        Check(codes.Count == 3 && codes['a'] == "10" && codes['b'] == "11" && codes['c'] == "0",
                "codes aabbbcccc");
        long weighted = WeightedLength(codes, freq);
        Check(weighted == 14 && weighted == 2 * 2 + 3 * 2 + 4 * 1, "weighted 14");
        string bits = HuffmanEncode(text, codes);
        Check(bits == "10101111110000" && bits.Length == weighted, "encoded bits");
        Check(HuffmanDecode(bits, codes) == text, "round trip");

        // single distinct character has the single code "0"
        var freq1 = CharacterFrequencies("aaaa");
        var codes1 = HuffmanCodes(freq1);
        Check(codes1.Count == 1 && codes1['a'] == "0", "single symbol code");
        Check(WeightedLength(codes1, freq1) == 4, "single symbol weight");
        string bits1 = HuffmanEncode("aaaa", codes1);
        Check(bits1 == "0000" && HuffmanDecode(bits1, codes1) == "aaaa", "single round trip");

        // empty text
        var emptyCodes = HuffmanCodes(new Dictionary<char, int>());
        Check(emptyCodes.Count == 0, "empty codes");
        Check(HuffmanEncode("", emptyCodes).Length == 0 && HuffmanDecode("", emptyCodes).Length == 0,
                "empty round trip");

        // second hand-verified structure, built straight from frequencies
        var bigFreq = new Dictionary<char, int>
        {
            ['a'] = 35, ['b'] = 17, ['c'] = 17, ['d'] = 16, ['e'] = 15,
        };
        var bigCodes = HuffmanCodes(bigFreq);
        Check(bigCodes.Count == 5 && bigCodes['a'] == "0" && bigCodes['e'] == "100"
                && bigCodes['d'] == "101" && bigCodes['b'] == "110" && bigCodes['c'] == "111",
                "five-symbol codes");
        Check(WeightedLength(bigCodes, bigFreq) == 230
                && WeightedLength(bigCodes, bigFreq) == 35 * 1 + (17 + 17 + 16 + 15) * 3,
                "five-symbol weight");

        // prefix-free property and the Kraft equality (a complete binary tree)
        Check(IsPrefixFree(codes) && KraftSumOk(codes), "prefix free and kraft small");
        Check(IsPrefixFree(codes1) && KraftSumOk(codes1), "prefix free and kraft single");
        Check(IsPrefixFree(bigCodes) && KraftSumOk(bigCodes), "prefix free and kraft big");

        // decoding rejects a dangling prefix (documented error)
        bool threw = false;
        try { HuffmanDecode("1", codes); }
        catch (InvalidOperationException) { threw = true; }
        Check(threw, "incomplete bits rejected");

        // LCG strings: round trips, bit length equals the weighted path length
        for (int trial = 0; trial < 120; trial++)
        {
            int length = NextSeed() % 80;
            int alphabetSize = 1 + NextSeed() % 6;
            var builder = new StringBuilder();
            for (int i = 0; i < length; i++)
                builder.Append((char)('a' + NextSeed() % alphabetSize));
            string sample = builder.ToString();
            var sampleFreq = CharacterFrequencies(sample);
            var sampleCodes = HuffmanCodes(sampleFreq);
            string sampleBits = HuffmanEncode(sample, sampleCodes);
            long sampleWeight = WeightedLength(sampleCodes, sampleFreq);
            Check(HuffmanDecode(sampleBits, sampleCodes) == sample, "LCG round trip " + trial);
            Check(sampleBits.Length == sampleWeight, "bit length " + trial);
            Check(IsPrefixFree(sampleCodes) && KraftSumOk(sampleCodes), "LCG prefix " + trial);
            if (length > 0)
            {
                var distinct = new HashSet<char>(sample);
                Check(sampleWeight <= (long)length * Math.Max(1, CeilLog2(distinct.Count)),
                        "fixed-length bound " + trial);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
