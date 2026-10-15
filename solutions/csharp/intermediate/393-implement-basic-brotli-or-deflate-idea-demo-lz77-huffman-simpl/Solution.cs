// Task 393 - Implement a basic DEFLATE/Brotli idea demo (LZ77 + Huffman, simplified). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;
using System.Linq;

public class HuffNode
{
    public int Weight, Seq;
    public int? ByteValue;
    public HuffNode? Left, Right;
    public HuffNode(int weight, int seq, int? byteValue) { Weight = weight; Seq = seq; ByteValue = byteValue; }
}

public static class Solution
{
    /// <summary>Returns tokens: {0, byte, 0} literals and {1, length, distance} matches.</summary>
    public static List<int[]> Lz77Tokens(string text, int window)
    {
        int n = text.Length;
        var tokens = new List<int[]>();
        int i = 0;
        while (i < n)
        {
            int bestLen = 0, bestDist = 0;
            for (int dist = 1; dist <= Math.Min(window, i); dist++)
            {
                int length = 0;
                while (i + length < n && text[i + length] == text[i - dist + length] && length < 255) length++;
                if (length > bestLen)
                {
                    bestLen = length;
                    bestDist = dist;
                }
            }
            if (bestLen >= 3)
            {
                tokens.Add(new[] { 1, bestLen, bestDist });
                i += bestLen;
            }
            else
            {
                tokens.Add(new[] { 0, text[i], 0 });
                i++;
            }
        }
        return tokens;
    }

    public static string Lz77Decompress(List<int[]> tokens)
    {
        var outBytes = new List<int>();
        foreach (var token in tokens)
        {
            if (token[0] == 0) outBytes.Add(token[1]);
            else for (int k = 0; k < token[1]; k++) outBytes.Add(outBytes[outBytes.Count - token[2]]);
        }
        return new string(outBytes.Select(b => (char)b).ToArray());
    }

    public static List<int> TokensToBytes(List<int[]> tokens)
    {
        var stream = new List<int>();
        foreach (var token in tokens)
        {
            stream.Add(token[0]);
            stream.Add(token[1]);
            if (token[0] == 1) stream.Add(token[2]);
        }
        return stream;
    }

    public static List<int[]> BytesToTokens(List<int> stream)
    {
        var tokens = new List<int[]>();
        int i = 0;
        while (i < stream.Count)
        {
            if (stream[i] == 0)
            {
                tokens.Add(new[] { 0, stream[i + 1], 0 });
                i += 2;
            }
            else
            {
                tokens.Add(new[] { 1, stream[i + 1], stream[i + 2] });
                i += 3;
            }
        }
        return tokens;
    }

    public static HuffNode BuildHuffman(SortedDictionary<int, int> freq)
    {
        var work = new List<HuffNode>();
        int seq = 0;
        foreach (var entry in freq) work.Add(new HuffNode(entry.Value, seq++, entry.Key));
        while (work.Count > 1)
        {
            work.Sort((a, b) => a.Weight != b.Weight ? a.Weight - b.Weight : a.Seq - b.Seq);
            var x = work[0];
            var y = work[1];
            work.RemoveRange(0, 2);
            work.Add(new HuffNode(x.Weight + y.Weight, seq++, null) { Left = x, Right = y });
        }
        return work[0];
    }

    public static void CollectCodes(HuffNode root, string prefix, SortedDictionary<int, string> outCodes)
    {
        if (root.ByteValue != null)
        {
            outCodes[root.ByteValue.Value] = prefix;
            return;
        }
        CollectCodes(root.Left!, prefix + "0", outCodes);
        CollectCodes(root.Right!, prefix + "1", outCodes);
    }

    /// <summary>Returns (bits, code table) for the given data.</summary>
    public static (string Bits, SortedDictionary<int, string> Codes) HuffmanEncode(List<int> data)
    {
        if (data.Count == 0) return ("", new SortedDictionary<int, string>());
        var freq = new SortedDictionary<int, int>();
        foreach (int byteValue in data) freq[byteValue] = freq.TryGetValue(byteValue, out int c) ? c + 1 : 1;
        var codes = new SortedDictionary<int, string>();
        CollectCodes(BuildHuffman(freq), "", codes);
        var bits = new System.Text.StringBuilder();
        foreach (int byteValue in data) bits.Append(codes[byteValue]);
        return (bits.ToString(), codes);
    }

    public static List<int> HuffmanDecode(string bits, SortedDictionary<int, string> codes)
    {
        var reverse = new Dictionary<string, int>();
        foreach (var entry in codes) reverse[entry.Value] = entry.Key;
        var outBytes = new List<int>();
        var current = new System.Text.StringBuilder();
        foreach (char bit in bits)
        {
            current.Append(bit);
            if (reverse.TryGetValue(current.ToString(), out int byteValue))
            {
                outBytes.Add(byteValue);
                current.Clear();
            }
        }
        if (current.Length != 0) throw new Exception("truncated code stream");
        return outBytes;
    }

    public static bool IsPrefixFree(SortedDictionary<int, string> codes)
    {
        var table = codes.Values.ToList();
        for (int i = 0; i < table.Count; i++)
            for (int j = 0; j < table.Count; j++)
                if (i != j && table[j].StartsWith(table[i], StringComparison.Ordinal)) return false;
        return true;
    }

    private static int seed = 3131;

    private static List<string> LcgStrings(string alphabet, int length, int count)
    {
        var strings = new List<string>();
        for (int k = 0; k < count; k++)
        {
            var chars = new char[length];
            for (int i = 0; i < length; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                chars[i] = alphabet[seed % alphabet.Length];
            }
            strings.Add(new string(chars));
        }
        return strings;
    }

    private static string JoinTokens(List<int[]> tokens) =>
        string.Join(";", tokens.Select(t => t[0] + "," + t[1] + "," + t[2]));

    public static void Main()
    {
        Check(JoinTokens(Lz77Tokens("abcabcabc", 8)) == "0,97,0;0,98,0;0,99,0;1,6,3", "hand-derived tokens");
        Check(JoinTokens(Lz77Tokens("aaaaaaaa", 8)) == "0,97,0;1,7,1", "run of a");
        Check(JoinTokens(Lz77Tokens(string.Concat(Enumerable.Repeat("abc", 10)), 8)) == "0,97,0;0,98,0;0,99,0;1,27,3", "abc x10");
        Check(Lz77Tokens("", 8).Count == 0, "empty input");
        Check(JoinTokens(Lz77Tokens("ab", 8)) == "0,97,0;0,98,0", "two literals");

        Check(Lz77Decompress(new List<int[]> { new[] { 0, 97, 0 }, new[] { 0, 98, 0 }, new[] { 0, 99, 0 }, new[] { 1, 6, 3 } }) == "abcabcabc", "decompress abc");
        Check(Lz77Decompress(new List<int[]> { new[] { 0, 97, 0 }, new[] { 1, 7, 1 } }) == "aaaaaaaa", "decompress a-run");

        Check(string.Join(",", TokensToBytes(new List<int[]> { new[] { 0, 97, 0 }, new[] { 1, 6, 3 } })) == "0,97,1,6,3", "serialize");
        Check(JoinTokens(BytesToTokens(new List<int> { 0, 97, 1, 6, 3 })) == "0,97,0;1,6,3", "deserialize");

        var samples = new List<string> { "abcabcabc", "aaaaaaaa", string.Concat(Enumerable.Repeat("abc", 10)), "", "ab", "aabaabaa" };
        samples.AddRange(LcgStrings("abc", 18, 30));
        foreach (string s in samples)
        {
            Check(Lz77Decompress(Lz77Tokens(s, 8)) == s, "lz77 round trip " + s);
            Check(JoinTokens(BytesToTokens(TokensToBytes(Lz77Tokens(s, 8)))) == JoinTokens(Lz77Tokens(s, 8)), "token round trip " + s);
        }

        foreach (string s in samples)
        {
            var (bits, codes) = HuffmanEncode(TokensToBytes(Lz77Tokens(s, 8)));
            Check(IsPrefixFree(codes), "prefix-free " + s);
            int weighted = TokensToBytes(Lz77Tokens(s, 8)).Sum(byteValue => codes[byteValue].Length);
            Check(bits.Length == weighted, "weighted length " + s);
            Check(Lz77Decompress(BytesToTokens(HuffmanDecode(bits, codes))) == s, "pipeline round trip " + s);
        }

        foreach (string text in new[] { "abcabcabc", "aaaaaaaa", string.Concat(Enumerable.Repeat("abc", 10)) })
        {
            var (bits, _) = HuffmanEncode(TokensToBytes(Lz77Tokens(text, 8)));
            Check(bits.Length < 8 * text.Length, "compressed " + text);
        }

        Check(JoinTokens(Lz77Tokens("abcabcabc", 2)) ==
              "0,97,0;0,98,0;0,99,0;0,97,0;0,98,0;0,99,0;0,97,0;0,98,0;0,99,0", "window too small for a match");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
