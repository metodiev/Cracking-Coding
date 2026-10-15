// Task 041 - Check if brackets are balanced (single type). (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;
using System.Text;



public static class Solution
{
    public static bool IsBalanced(string s)
    {
        int depth = 0;
        foreach (var c in s)
        {
            if (c == '(') depth++;
            else if (c == ')') { depth--; if (depth < 0) return false; }
        }
        return depth == 0;
    }

    public static void Main()
    {
        Check(IsBalanced("((()))"), "nested");
        Check(IsBalanced("()()()"), "sequential");
        Check(!IsBalanced("(()"), "unclosed");
        Check(!IsBalanced(")("), "bad order");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
