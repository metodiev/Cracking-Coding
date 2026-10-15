// Task 003 - Count vowels and consonants in a string. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static int[] CountVowelsAndConsonants(string s)
    {
        int vowels = 0, consonants = 0;
        foreach (var c in s.ToLowerInvariant())
        {
            if (!char.IsLetter(c)) continue;
            if ("aeiou".IndexOf(c) >= 0) vowels++; else consonants++;
        }
        return new[] { vowels, consonants };
    }

    public static void Main()
    {
        Check(CountVowelsAndConsonants("hello")[0] == 2 && CountVowelsAndConsonants("hello")[1] == 3, "hello");
        Check(CountVowelsAndConsonants("")[0] == 0, "empty");
        Check(CountVowelsAndConsonants("AEIOU")[0] == 5, "AEIOU");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
