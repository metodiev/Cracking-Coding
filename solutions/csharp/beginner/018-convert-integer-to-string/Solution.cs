// Task 018 - Convert integer to string (manual itoa). (Tier: Beginner, CTCI Ch 1)
using System;
using System.Text;

public static class Solution
{
    public static string Itoa(int n)
    {
        if (n == 0) return "0";
        bool negative = n < 0;
        long value = Math.Abs((long)n);
        var digits = new StringBuilder();
        while (value > 0)
        {
            digits.Append((char)('0' + value % 10));
            value /= 10;
        }
        var chars = digits.ToString().ToCharArray();
        Array.Reverse(chars);
        return (negative ? "-" : "") + new string(chars);
    }

    public static void Main()
    {
        Check(Itoa(0) == "0", "0");
        Check(Itoa(123) == "123", "123");
        Check(Itoa(-456) == "-456", "-456");
        Check(Itoa(int.MinValue) == "-2147483648", "min int");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
