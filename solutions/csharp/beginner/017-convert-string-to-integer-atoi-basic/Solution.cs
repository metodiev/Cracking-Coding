// Task 017 - Convert string to integer (atoi) basic. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static int Atoi(string s)
    {
        if (s.Length == 0) throw new ArgumentException("empty input");
        int i = 0, sign = 1;
        if (s[0] == '+' || s[0] == '-')
        {
            sign = s[0] == '-' ? -1 : 1;
            i = 1;
        }
        if (i == s.Length) throw new ArgumentException("no digits");
        int value = 0;
        for (; i < s.Length; i++)
        {
            char c = s[i];
            if (c < '0' || c > '9') throw new ArgumentException("invalid char: " + c);
            value = value * 10 + (c - '0');
        }
        return sign * value;
    }

    public static void Main()
    {
        Check(Atoi("42") == 42, "42");
        Check(Atoi("-123") == -123, "-123");
        Check(Atoi("+7") == 7, "+7");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
