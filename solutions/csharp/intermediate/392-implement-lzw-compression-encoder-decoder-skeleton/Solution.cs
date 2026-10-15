// Task 392 - Implement LZW compression (encoder/decoder skeleton). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>alphabet: its order fixes the dictionary ids.</summary>
    public static List<int> LzwEncode(string text, string alphabet)
    {
        var dictionary = new Dictionary<string, int>();
        for (int i = 0; i < alphabet.Length; i++) dictionary[alphabet[i].ToString()] = i;
        int nextCode = dictionary.Count;
        var codes = new List<int>();
        string current = "";
        foreach (char c in text)
        {
            string ch = c.ToString();
            string candidate = current + ch;
            if (dictionary.ContainsKey(candidate))
            {
                current = candidate;
            }
            else
            {
                if (current.Length == 0) throw new ArgumentException("symbol outside the alphabet: " + ch);
                codes.Add(dictionary[current]);
                dictionary[candidate] = nextCode++;
                current = ch;
                if (!dictionary.ContainsKey(ch)) throw new ArgumentException("symbol outside the alphabet: " + ch);
            }
        }
        if (current.Length > 0) codes.Add(dictionary[current]);
        return codes;
    }

    public static string LzwDecode(List<int> codes, string alphabet)
    {
        var dictionary = new Dictionary<int, string>();
        for (int i = 0; i < alphabet.Length; i++) dictionary[i] = alphabet[i].ToString();
        int nextCode = dictionary.Count;
        if (codes.Count == 0) return "";
        int first = codes[0];
        if (!dictionary.ContainsKey(first)) throw new ArgumentException("invalid first code");
        string previous = dictionary[first];
        var outText = new System.Text.StringBuilder(previous);
        for (int k = 1; k < codes.Count; k++)
        {
            int code = codes[k];
            string entry;
            if (dictionary.TryGetValue(code, out var known)) entry = known;
            else if (code == nextCode) entry = previous + previous[0];   // the cSc rule
            else throw new ArgumentException("invalid code: " + code);
            outText.Append(entry);
            dictionary[nextCode] = previous + entry[0];
            nextCode++;
            previous = entry;
        }
        return outText.ToString();
    }

    private static int seed = 5150;

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

    private static string Join(List<int> codes) => string.Join(",", codes);

    public static void Main()
    {
        Check(Join(LzwEncode("abababa", "ab")) == "0,1,2,4", "abababa");
        Check(Join(LzwEncode("aaaa", "ab")) == "0,2,0", "aaaa");
        Check(Join(LzwEncode("abcabcabcabc", "abc")) == "0,1,2,3,5,4,6", "abc cycle");
        Check(LzwEncode("", "ab").Count == 0, "empty input");
        Check(Join(LzwEncode("a", "ab")) == "0", "single symbol");

        Check(LzwDecode(new List<int> { 0, 1, 2, 4 }, "ab") == "abababa", "decode abababa");
        Check(LzwDecode(new List<int> { 0, 2, 0 }, "ab") == "aaaa", "decode aaaa");
        Check(LzwDecode(new List<int> { 0, 1, 2, 3, 5, 4, 6 }, "abc") == "abcabcabcabc", "decode cycle");
        Check(LzwDecode(new List<int>(), "ab") == "", "decode empty");
        Check(LzwDecode(new List<int> { 0 }, "ab") == "a", "decode single");

        var samples = new List<string> { "abababa", "aaaa", "", "a" };
        samples.AddRange(LcgStrings("ab", 30, 40));
        foreach (string s in samples) Check(LzwDecode(LzwEncode(s, "ab"), "ab") == s, "round trip " + s);
        foreach (string s in LcgStrings("abcd", 40, 40)) Check(LzwDecode(LzwEncode(s, "abcd"), "abcd") == s, "round trip " + s);

        string repetitive = string.Concat(Enumerable.Repeat("ab", 20));
        Check(LzwEncode(repetitive, "ab").Count < repetitive.Length, "repetitive text shrinks");

        var csc = LzwEncode("abababab", "ab");
        Check(LzwDecode(csc, "ab") == "abababab", "cSc round trip");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
