// Task 391 - Implement adaptive Huffman (outline, conceptual). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;
using System.Linq;

public class HuffNode
{
    public int Weight, Seq;
    public string? Ch;
    public HuffNode? Left, Right;
    public HuffNode(int weight, int seq, string? ch) { Weight = weight; Seq = seq; Ch = ch; }
}

public static class Solution
{
    /// <summary>Builds the Huffman tree; ties broken by (weight, creation order).</summary>
    public static HuffNode BuildTree(Dictionary<string, int> freq, List<string> alphabet)
    {
        var work = new List<HuffNode>();
        int seq = 0;
        foreach (string ch in alphabet) work.Add(new HuffNode(freq[ch], seq++, ch));
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

    public static void CollectCodes(HuffNode root, string prefix, Dictionary<string, string> outCodes)
    {
        if (root.Ch != null)
        {
            outCodes[root.Ch] = prefix;
            return;
        }
        CollectCodes(root.Left!, prefix + "0", outCodes);
        CollectCodes(root.Right!, prefix + "1", outCodes);
    }

    public static string AdaptiveEncode(string text, string alphabet)
    {
        var freq = new Dictionary<string, int>();
        var symbols = new List<string>();
        foreach (char c in alphabet)
        {
            freq[c.ToString()] = 1;
            symbols.Add(c.ToString());
        }
        var bits = new System.Text.StringBuilder();
        foreach (char c in text)
        {
            string ch = c.ToString();
            if (!freq.ContainsKey(ch)) throw new ArgumentException("symbol outside the alphabet: " + ch);
            var codes = new Dictionary<string, string>();
            CollectCodes(BuildTree(freq, symbols), "", codes);
            bits.Append(codes[ch]);
            freq[ch]++;
        }
        return bits.ToString();
    }

    public static string AdaptiveDecode(string bits, string alphabet)
    {
        var freq = new Dictionary<string, int>();
        var symbols = new List<string>();
        foreach (char c in alphabet)
        {
            freq[c.ToString()] = 1;
            symbols.Add(c.ToString());
        }
        var outText = new System.Text.StringBuilder();
        int i = 0;
        while (i < bits.Length)
        {
            var node = BuildTree(freq, symbols);
            while (node.Ch == null)
            {
                node = bits[i] == '0' ? node.Left! : node.Right!;
                i++;
            }
            outText.Append(node.Ch);
            freq[node.Ch]++;
        }
        return outText.ToString();
    }

    private static int seed = 6060;

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

    public static void Main()
    {
        Check(AdaptiveEncode("a", "abc") == "10", "first symbol code");
        Check(AdaptiveEncode("aaabbc", "abc") == "1001000110", "hand-derived stream");
        Check(AdaptiveEncode("aaa", "abc") == "1001", "three repeats");
        Check(AdaptiveEncode("", "abc") == "", "empty input");
        Check(AdaptiveDecode("1001000110", "abc") == "aaabbc", "decode the hand-derived stream");

        var samples = new List<string> { "a", "abc", "aaabbc", "abababab", "cccbaaa" };
        samples.AddRange(LcgStrings("abc", 24, 40));
        foreach (string s in samples)
        {
            Check(AdaptiveDecode(AdaptiveEncode(s, "abc"), "abc") == s, "round trip " + s);
        }

        var finalFreq = new Dictionary<string, int> { ["a"] = 41, ["b"] = 2, ["c"] = 2 };
        var codes = new Dictionary<string, string>();
        CollectCodes(BuildTree(finalFreq, new List<string> { "a", "b", "c" }), "", codes);
        Check(codes["a"] == "1" && codes["b"].Length == 2 && codes["c"].Length == 2,
              "frequent symbol gets the short code");

        bool threw = false;
        try { AdaptiveEncode("d", "abc"); }
        catch (ArgumentException) { threw = true; }
        Check(threw, "out-of-alphabet symbol rejected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
