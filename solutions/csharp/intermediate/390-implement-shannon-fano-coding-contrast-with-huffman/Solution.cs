// Task 390 - Implement Shannon-Fano coding (contrast with Huffman). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal sealed class SfHuffNode
{
    public readonly long Freq;
    public readonly int Id;
    public readonly char Ch;
    public readonly bool Leaf;
    public readonly SfHuffNode? Left;
    public readonly SfHuffNode? Right;

    public SfHuffNode(long freq, int id, char ch, bool leaf, SfHuffNode? left, SfHuffNode? right)
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
    /// Top-down split: frequency-descending order (ties by character), cut where the halves'
    /// total-frequency difference is minimal (earliest split on ties), '0' left, '1' right.
    /// </summary>
    private static Dictionary<char, string> ShannonFanoCodes(Dictionary<char, int> frequencies)
    {
        var codes = new Dictionary<char, string>();
        if (frequencies.Count == 0) return codes;
        var items = frequencies.ToList();
        items.Sort((x, y) => x.Value != y.Value
            ? y.Value.CompareTo(x.Value)
            : x.Key.CompareTo(y.Key));
        ShannonFanoRec(items, "", codes);
        return codes;
    }

    private static void ShannonFanoRec(List<KeyValuePair<char, int>> part, string prefix,
            Dictionary<char, string> codes)
    {
        if (part.Count == 1)
        {
            codes[part[0].Key] = prefix.Length == 0 ? "0" : prefix;
            return;
        }
        long total = 0;
        foreach (var entry in part) total += entry.Value;
        int bestSplit = 1;
        long bestDiff = long.MaxValue;
        long leftSum = 0;
        for (int k = 1; k < part.Count; k++)
        {
            leftSum += part[k - 1].Value;
            long diff = Math.Abs(total - 2 * leftSum);
            if (diff < bestDiff)
            {
                bestDiff = diff;
                bestSplit = k;
            }
        }
        ShannonFanoRec(part.GetRange(0, bestSplit), prefix + "0", codes);
        ShannonFanoRec(part.GetRange(bestSplit, part.Count - bestSplit), prefix + "1", codes);
    }

    /// <summary>Local Huffman re-implementation (same tie-break as task 389).</summary>
    private static Dictionary<char, string> HuffmanCodes(Dictionary<char, int> frequencies)
    {
        var codes = new Dictionary<char, string>();
        if (frequencies.Count == 0) return codes;
        var chars = new List<char>(frequencies.Keys);
        chars.Sort();
        var heap = new PriorityQueue<SfHuffNode, (long, int)>();
        int id = 0;
        foreach (char ch in chars)
        {
            int nodeId = id++;
            heap.Enqueue(new SfHuffNode(frequencies[ch], nodeId, ch, true, null, null),
                    (frequencies[ch], nodeId));
        }
        int nextId = id;
        while (heap.Count > 1)
        {
            var left = heap.Dequeue();
            var right = heap.Dequeue();
            long combined = left.Freq + right.Freq;
            heap.Enqueue(new SfHuffNode(combined, nextId, '\0', false, left, right),
                    (combined, nextId));
            nextId++;
        }
        HuffmanWalk(heap.Peek(), "", codes);
        return codes;
    }

    private static void HuffmanWalk(SfHuffNode node, string prefix, Dictionary<char, string> codes)
    {
        if (node.Leaf)
        {
            codes[node.Ch] = prefix.Length == 0 ? "0" : prefix;
            return;
        }
        HuffmanWalk(node.Left!, prefix + "0", codes);
        HuffmanWalk(node.Right!, prefix + "1", codes);
    }

    private static long WeightedLength(Dictionary<char, string> codes, Dictionary<char, int> frequencies)
    {
        long total = 0;
        foreach (var entry in codes) total += (long)frequencies[entry.Key] * entry.Value.Length;
        return total;
    }

    private static string Encode(string text, Dictionary<char, string> codes)
    {
        var bits = new StringBuilder();
        foreach (char ch in text) bits.Append(codes[ch]);
        return bits.ToString();
    }

    private static string Decode(string bits, Dictionary<char, string> codes)
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

    private static int seed = 8642;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void Main()
    {
        // tie example: aabbbcccc -> both codecs cost 14 bits
        var tieFreq = new Dictionary<char, int> { ['a'] = 2, ['b'] = 3, ['c'] = 4 };
        var sf = ShannonFanoCodes(tieFreq);
        var huff = HuffmanCodes(tieFreq);
        Check(sf.Count == 3 && sf['c'] == "0" && sf['b'] == "10" && sf['a'] == "11",
                "shannon-fano aabbbcccc");
        Check(huff.Count == 3 && huff['c'] == "0" && huff['a'] == "10" && huff['b'] == "11",
                "huffman aabbbcccc");
        Check(WeightedLength(sf, tieFreq) == 14, "sf weight tie");
        Check(WeightedLength(huff, tieFreq) == 14, "huffman weight tie");
        Check(WeightedLength(sf, tieFreq) == WeightedLength(huff, tieFreq), "documented tie");
        foreach (var codes in new[] { sf, huff })
        {
            Check(IsPrefixFree(codes) && KraftSumOk(codes), "prefix/kraft tie");
            Check(Decode(Encode("aabbbcccc", codes), codes) == "aabbbcccc", "round trip tie");
        }

        // strict example: Huffman 230 < Shannon-Fano 231
        var bigFreq = new Dictionary<char, int>
        {
            ['a'] = 35, ['b'] = 17, ['c'] = 17, ['d'] = 16, ['e'] = 15,
        };
        var sfBig = ShannonFanoCodes(bigFreq);
        var huffBig = HuffmanCodes(bigFreq);
        Check(sfBig.Count == 5 && sfBig['a'] == "00" && sfBig['b'] == "01" && sfBig['c'] == "10"
                && sfBig['d'] == "110" && sfBig['e'] == "111", "shannon-fano big");
        Check(huffBig.Count == 5 && huffBig['a'] == "0" && huffBig['e'] == "100"
                && huffBig['d'] == "101" && huffBig['b'] == "110" && huffBig['c'] == "111",
                "huffman big");
        Check(WeightedLength(sfBig, bigFreq) == 231, "sf weight big");
        Check(WeightedLength(huffBig, bigFreq) == 230, "huffman weight big");
        Check(WeightedLength(huffBig, bigFreq) < WeightedLength(sfBig, bigFreq), "strictly better");
        Check(sfBig['a'].Length == 2 && huffBig['a'].Length == 1, "a code lengths");
        foreach (var codes in new[] { sfBig, huffBig })
        {
            Check(IsPrefixFree(codes) && KraftSumOk(codes), "prefix/kraft big");
        }

        // edge cases
        Check(ShannonFanoCodes(new Dictionary<char, int>()).Count == 0, "sf empty");
        Check(HuffmanCodes(new Dictionary<char, int>()).Count == 0, "huffman empty");
        var single = new Dictionary<char, int> { ['a'] = 4 };
        Check(ShannonFanoCodes(single)['a'] == "0" && HuffmanCodes(single)['a'] == "0",
                "single symbol");
        Check(Decode("", new Dictionary<char, string>()).Length == 0, "decode empty");
        Check(Encode("", new Dictionary<char, string>()).Length == 0, "encode empty");

        // LCG strings: both codecs round trip, prefix-free, and Huffman <= Shannon-Fano
        int tieCount = 0;
        for (int trial = 0; trial < 120; trial++)
        {
            int length = NextSeed() % 80;
            int alphabetSize = 1 + NextSeed() % 6;
            var builder = new StringBuilder();
            for (int i = 0; i < length; i++)
                builder.Append((char)('a' + NextSeed() % alphabetSize));
            string sample = builder.ToString();
            var freq = CharacterFrequencies(sample);
            var sfCodes = ShannonFanoCodes(freq);
            var huffCodes = HuffmanCodes(freq);
            foreach (var codes in new[] { sfCodes, huffCodes })
            {
                Check(IsPrefixFree(codes) && KraftSumOk(codes), "LCG prefix " + trial);
                string bits = Encode(sample, codes);
                Check(Decode(bits, codes) == sample, "LCG round trip " + trial);
                Check(bits.Length == WeightedLength(codes, freq), "LCG weight " + trial);
            }
            long sfWeight = WeightedLength(sfCodes, freq);
            long huffWeight = WeightedLength(huffCodes, freq);
            Check(huffWeight <= sfWeight, "huffman <= shannon-fano " + trial);
            if (huffWeight == sfWeight) tieCount++;
        }
        Check(tieCount > 0, "ties are observed among the samples");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
