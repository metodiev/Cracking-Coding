// Task 029 - Check if integer is power of two. (Tier: Beginner, CTCI Ch 5)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static bool IsPowerOfTwo(int x)
    {
        return x > 0 && (x & (x - 1)) == 0;
    }

    public static void Main()
    {
        Check(IsPowerOfTwo(1) && IsPowerOfTwo(1024), "powers");
        Check(!IsPowerOfTwo(0) && !IsPowerOfTwo(12), "non-powers");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
