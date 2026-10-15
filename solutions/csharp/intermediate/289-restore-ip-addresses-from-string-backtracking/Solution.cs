// Task 289 - Restore IP addresses from a digit string. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static string digits = "";
    private static List<string> parts = new List<string>();
    private static List<string> results = new List<string>();

    /// <summary>
    /// Parts are extended left to right, trying segment lengths 1, 2, 3 in that order, so
    /// the emitted addresses are deterministic. A segment is valid when it is 1-3 digits,
    /// at most 255, and has no leading zero unless it is exactly "0".
    /// </summary>
    public static List<string> RestoreIpAddresses(string s)
    {
        if (s.Length < 4 || s.Length > 12) return new List<string>();
        digits = s;
        parts = new List<string>();
        results = new List<string>();
        Backtrack(0);
        return results;
    }

    private static void Backtrack(int index)
    {
        if (parts.Count == 4)
        {
            if (index == digits.Length) results.Add(string.Join(".", parts));
            return;
        }
        for (int length = 1; length <= 3; length++)
        {
            if (index + length > digits.Length) break;
            string segment = digits.Substring(index, length);
            if (!IsValidSegment(segment)) continue;
            parts.Add(segment);
            Backtrack(index + length);
            parts.RemoveAt(parts.Count - 1);
        }
    }

    public static bool IsValidSegment(string segment)
    {
        if (segment.Length == 0 || segment.Length > 3) return false;
        if (segment[0] == '0') return segment == "0";
        return int.Parse(segment) <= 255;
    }

    /// <summary>Oracle: try every triple of cut positions and validate the four segments.</summary>
    public static List<string> BruteForceIpAddresses(string s)
    {
        var found = new List<string>();
        int n = s.Length;
        for (int i = 1; i < Math.Min(4, n); i++)
        {
            for (int j = i + 1; j < Math.Min(i + 4, n); j++)
            {
                for (int k = j + 1; k < Math.Min(j + 4, n); k++)
                {
                    var candidate = new[]
                    {
                        s.Substring(0, i), s.Substring(i, j - i), s.Substring(j, k - j), s.Substring(k),
                    };
                    if (candidate.All(IsValidSegment)) found.Add(string.Join(".", candidate));
                }
            }
        }
        found.Sort(StringComparer.Ordinal);
        return found;
    }

    public static bool IsValidIp(string text, string s)
    {
        var segments = text.Split('.');
        if (segments.Length != 4) return false;
        if (!segments.All(IsValidSegment)) return false;
        return string.Join("", segments) == s;
    }

    private static int seed = 13579;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void Main()
    {
        Check(RestoreIpAddresses("25525511135").SequenceEqual(new[] { "255.255.11.135", "255.255.111.35" }), "classic");
        Check(RestoreIpAddresses("0000").SequenceEqual(new[] { "0.0.0.0" }), "zeros");
        Check(RestoreIpAddresses("1111").SequenceEqual(new[] { "1.1.1.1" }), "ones");
        Check(RestoreIpAddresses("010010").SequenceEqual(new[] { "0.10.0.10", "0.100.1.0" }), "leading zero rules");
        Check(RestoreIpAddresses("101023").SequenceEqual(
            new[] { "1.0.10.23", "1.0.102.3", "10.1.0.23", "10.10.2.3", "101.0.2.3" }), "101023");
        Check(RestoreIpAddresses("255255255255").SequenceEqual(new[] { "255.255.255.255" }), "max address");

        Check(RestoreIpAddresses("111").Count == 0, "too short");
        Check(RestoreIpAddresses("").Count == 0, "empty");
        Check(RestoreIpAddresses("1111111111111").Count == 0, "too long");
        Check(RestoreIpAddresses("256256256256").Count == 0, "segments over 255");
        Check(RestoreIpAddresses("111111111111111111111").Count == 0, "way too long");

        for (int trial = 0; trial < 200; trial++)
        {
            int length = 1 + NextSeed() % 14;
            var builder = new System.Text.StringBuilder();
            for (int i = 0; i < length; i++) builder.Append((char)('0' + NextSeed() % 10));
            string s = builder.ToString();
            var found = RestoreIpAddresses(s);
            var sorted = new List<string>(found);
            sorted.Sort(StringComparer.Ordinal);
            Check(sorted.SequenceEqual(BruteForceIpAddresses(s)), "oracle " + s);
            var unique = new HashSet<string>();
            foreach (string text in found)
            {
                Check(IsValidIp(text, s), "valid " + s + " " + text);
                Check(unique.Add(text), "duplicate " + s);
            }
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
