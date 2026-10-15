// Task 080 - Implement string builder/concatenation efficient. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;

public static class Solution
{
    private class StringBuilderSimple
    {
        private readonly List<string> parts = new List<string>();

        public StringBuilderSimple Append(object text)
        {
            parts.Add(text.ToString() ?? "");
            return this;
        }

        public string Build() => string.Join("", parts);

        public int Length
        {
            get
            {
                int total = 0;
                foreach (var p in parts) total += p.Length;
                return total;
            }
        }
    }

    public static void Main()
    {
        var sb = new StringBuilderSimple();
        sb.Append("Hello").Append(", ").Append("world").Append("!").Append(42);
        Check(sb.Build() == "Hello, world!42", "mixed appends");
        Check(sb.Length == 15, "length, got " + sb.Length);
        Check(new StringBuilderSimple().Build() == "", "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
