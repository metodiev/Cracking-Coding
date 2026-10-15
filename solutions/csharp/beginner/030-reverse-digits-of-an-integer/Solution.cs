// Task 030 - Reverse digits of an integer. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static int ReverseDigits(int n)
    {
        long result = 0;
        while (n != 0)
        {
            result = result * 10 + n % 10;
            n /= 10;
        }
        return result < int.MinValue || result > int.MaxValue ? 0 : (int)result;
    }

    public static void Main()
    {
        Check(ReverseDigits(123) == 321, "123");
        Check(ReverseDigits(-123) == -321, "-123");
        Check(ReverseDigits(0) == 0, "0");
        Check(ReverseDigits(1534236469) == 0, "overflow clamps to 0");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
