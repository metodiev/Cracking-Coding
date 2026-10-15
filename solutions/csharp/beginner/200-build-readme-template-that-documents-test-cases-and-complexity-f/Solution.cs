// Task 200 - Build README template that documents test cases and complexity. (Tier: Beginner, CTCI Beyond)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static void Main()
    {
        string twoSum = RenderSection("Two Sum", "Beginner", "O(n)", "O(n)",
                new List<(string, string)> { ("1, 2", "3"), ("[]", "0") });
        Check(twoSum == string.Join("\n",
                "## Two Sum", "", "**Tier:** Beginner", "",
                "| Complexity | Value |", "| --- | --- |", "| Time | O(n) |", "| Space | O(n) |", "",
                "| Input | Expected |", "| --- | --- |",
                "| `1, 2` | `3` |", "| `[]` | `0` |"), "Two Sum section");

        string escaped = RenderSection("Escape | Demo", "Beginner", "O(1)", "O(1)",
                new List<(string, string)> { ("a | b", "c\nd"), ("x\ny", "z") });
        Check(escaped == string.Join("\n",
                "## Escape | Demo", "", "**Tier:** Beginner", "",
                "| Complexity | Value |", "| --- | --- |", "| Time | O(1) |", "| Space | O(1) |", "",
                "| Input | Expected |", "| --- | --- |",
                "| `a \\| b` | `c d` |", "| `x y` | `z` |"), "cell escaping");

        string empty = RenderSection("Empty Cases", "Beginner", "O(1)", "O(1)",
                new List<(string, string)>());
        Check(empty == string.Join("\n",
                "## Empty Cases", "", "**Tier:** Beginner", "",
                "| Complexity | Value |", "| --- | --- |", "| Time | O(1) |", "| Space | O(1) |", "",
                "| Input | Expected |", "| --- | --- |"), "no test cases");

        Check(!twoSum.EndsWith("\n") && !escaped.EndsWith("\n") && !empty.EndsWith("\n"),
                "no trailing newline");
        Console.WriteLine("PASS");
    }

    private static string RenderSection(string name, string tier, string timeComplexity,
                                        string spaceComplexity, IReadOnlyList<(string Input, string Expected)> testCases)
    {
        var lines = new List<string>
        {
            "## " + name,
            "",
            "**Tier:** " + tier,
            "",
            "| Complexity | Value |",
            "| --- | --- |",
            "| Time | " + Cell(timeComplexity) + " |",
            "| Space | " + Cell(spaceComplexity) + " |",
            "",
            "| Input | Expected |",
            "| --- | --- |",
        };
        foreach (var (input, expected) in testCases)
        {
            lines.Add("| `" + Cell(input) + "` | `" + Cell(expected) + "` |");
        }
        return string.Join("\n", lines);
    }

    private static string Cell(string value)
    {
        return value.Replace("|", "\\|").Replace("\n", " ");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
