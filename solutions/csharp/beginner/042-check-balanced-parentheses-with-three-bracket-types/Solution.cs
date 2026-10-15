// Task 042 - Check balanced parentheses with three bracket types. (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;
using System.Text;



public static class Solution
{
    public static bool IsBalanced(string s)
    {
        var pairs = new Dictionary<char, char> { [')'] = '(', [']'] = '[', ['}'] = '{' };
        var stack = new Stack<char>();
        foreach (var c in s)
        {
            if (c == '(' || c == '[' || c == '{') stack.Push(c);
            else if (pairs.ContainsKey(c))
            {
                if (stack.Count == 0 || stack.Pop() != pairs[c]) return false;
            }
        }
        return stack.Count == 0;
    }

    public static void Main()
    {
        Check(IsBalanced("{[()]}"), "nested mixed");
        Check(IsBalanced("()[]{}"), "sequential");
        Check(!IsBalanced("{[(])}"), "crossed");
        Check(!IsBalanced(")"), "stray close");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
