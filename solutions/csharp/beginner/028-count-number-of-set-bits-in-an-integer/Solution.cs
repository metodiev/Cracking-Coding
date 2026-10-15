// Task 028 - Count number of set bits in an integer. (Tier: Beginner, CTCI Ch 5)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static int CountSetBits(int x)
    {
        int count = 0;
        while (x != 0) { x &= x - 1; count++; }
        return count;
    }

    public static void Main()
    {
        Check(CountSetBits(0) == 0, "0");
        Check(CountSetBits(255) == 8, "255");
        Check(CountSetBits(0b101101) == 4, "0b101101");
        Check(CountSetBits(-1) == 32, "all bits (-1)");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
