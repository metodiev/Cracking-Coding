// Task 316 - Repeated DNA sequences of length 10 (hashing). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public static class Solution
{
    public const int DnaLength = 10;
    public const string Bases = "ACGT";

    /// <summary>
    /// All k-letter substrings appearing more than once, sorted; empty list if none. Each base is
    /// packed into 2 bits, so the rolling key in [0, 4^k) is collision-free and O(1) per step.
    /// </summary>
    public static List<string> RepeatedDnaSequences(string s, int k)
    {
        var result = new List<string>();
        if (k <= 0 || s.Length < k) return result;
        long mask = (1L << (2 * k)) - 1;                 // keeps only the last k bases
        var seen = new HashSet<long>();
        var repeated = new SortedSet<string>();
        long key = 0;
        for (int i = 0; i < s.Length; i++)
        {
            key = ((key << 2) | Bases.IndexOf(s[i])) & mask;
            if (i >= k - 1)
            {
                if (!seen.Add(key)) repeated.Add(s.Substring(i - k + 1, k));
            }
        }
        result.AddRange(repeated);
        return result;
    }

    /// <summary>Counting map over every substring; the independent oracle.</summary>
    public static List<string> RepeatedDnaNaive(string s, int k)
    {
        var result = new List<string>();
        if (k <= 0 || s.Length < k) return result;
        var counts = new SortedDictionary<string, int>();
        for (int i = 0; i + k <= s.Length; i++)
        {
            string word = s.Substring(i, k);
            counts[word] = counts.TryGetValue(word, out int count) ? count + 1 : 1;
        }
        foreach (var entry in counts)
        {
            if (entry.Value > 1) result.Add(entry.Key);
        }
        return result;
    }

    /// <summary>Occurrences counted with overlaps (non-overlapping counting would undercount repeats).</summary>
    public static int OverlappingCount(string s, string word)
    {
        int count = 0;
        for (int i = 0; i + word.Length <= s.Length; i++)
        {
            if (string.CompareOrdinal(s.Substring(i, word.Length), word) == 0) count++;
        }
        return count;
    }

    private static int seed = 3160;

    private static string LcgDna(int length)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            sb.Append(Bases[seed % 4]);
        }
        return sb.ToString();
    }

    private static string Join(List<string> values) => string.Join(",", values);

    public static void Main()
    {
        string classic = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT";
        Check(Join(RepeatedDnaSequences(classic, DnaLength)) == "AAAAACCCCC,CCCCCAAAAA", "classic");
        Check(Join(RepeatedDnaNaive(classic, DnaLength)) == "AAAAACCCCC,CCCCCAAAAA", "classic oracle");

        Check(RepeatedDnaSequences("ACGTACGTAC", DnaLength).Count == 0, "no repeat");
        Check(Join(RepeatedDnaSequences("AAAAAAAAAAA", DnaLength)) == "AAAAAAAAAA", "eleven A");
        Check(RepeatedDnaSequences("AAAAAAAAAA", DnaLength).Count == 0, "ten A");
        Check(RepeatedDnaSequences("ACGT", DnaLength).Count == 0, "short");
        Check(RepeatedDnaSequences("", DnaLength).Count == 0, "empty");
        Check(Join(RepeatedDnaSequences("ACGTACGTAC", 4)) == "ACGT,CGTA,GTAC", "k=4");
        Check(RepeatedDnaSequences("ACGTACGTAC", 0).Count == 0, "k=0");
        Check(RepeatedDnaSequences("ACGTACGTAC", 20).Count == 0, "k too large");

        var samples = new List<string> { classic, "AAAAAAAAAAAA", "ACGTACGTACGTACGTACGT" };
        for (int i = 0; i < 40; i++) samples.Add(LcgDna(40));
        for (int i = 0; i < 20; i++) samples.Add(LcgDna(20));

        foreach (string s in samples)
        {
            var found = RepeatedDnaSequences(s, DnaLength);
            Check(Join(found) == Join(RepeatedDnaNaive(s, DnaLength)), "cross-check " + s);
            Check(Join(found) == Join(found.OrderBy(word => word, StringComparer.Ordinal).ToList()), "sorted " + s);
            Check(found.Distinct().Count() == found.Count, "unique " + s);
            foreach (string word in found)                  // every report really repeats
            {
                Check(word.Length == DnaLength && OverlappingCount(s, word) > 1, "repeat " + s + " " + word);
            }
        }

        // The result is exactly the set of 10-mers occurring at least twice
        for (int i = 0; i < 20; i++)
        {
            string s = LcgDna(30);
            var expect = new List<string>();
            for (int start = 0; start + DnaLength <= s.Length; start++)
            {
                string word = s.Substring(start, DnaLength);
                if (OverlappingCount(s, word) > 1 && !expect.Contains(word)) expect.Add(word);
            }
            expect.Sort(StringComparer.Ordinal);
            Check(Join(RepeatedDnaSequences(s, DnaLength)) == Join(expect), "brute " + s);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
