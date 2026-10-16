// Task 953 - Implement contributor guide and templates for new problems, tests, and solutions.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The contributor workflow is a tiny template engine plus the rules the guide states: one folder per
// problem named `<3-digit number>-<slug>` holding exactly one solution file per language, a lower-case
// dash slug, and a header that names the task.  Templates carry `{{placeholder}}` markers; rendering
// refuses unknown or unterminated markers, and a linter then rejects whatever a careless contributor
// would slip in (leftover markers, tabs, trailing whitespace, no final newline).
//
// Verified: rendering and lint for all four language templates, a round-trip extraction of
// number/slug/title back out of the rendered file (independent oracle), rejection of missing, unknown
// and unterminated placeholders, slug and path boundaries (007 vs 1000), and verbatim titles that
// contain braces.  Simplification: nothing touches the disk - the "files" are strings in memory.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }
    /** Returns the exception message of a failing action, or "" when it succeeds. */
    private static string Fails(Action action)
    {
        try
        {
            action();
            return "";
        }
        catch (Exception ex)
        {
            return ex.Message;
        }
    }
    private static readonly string[] Languages = { "python", "java", "javascript", "csharp" };
    private static readonly Dictionary<string, string> Templates = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["python"] = @"'''Task {{number}} - {{title}}
tier {{tier}}, {{chapter}}
path: {{path}}
'''
def main():
    print('PASS')
if __name__ == '__main__':
    main()
",
        ["java"] = @"// Task {{number}} - {{title}}
// tier {{tier}}, {{chapter}}
// path: {{path}}
public class Solution
{
    public static void main(String[] args)
    {
        System.out.println(""PASS"");
    }
}
",
        ["javascript"] = @"""use strict"";
// Task {{number}} - {{title}}
// tier {{tier}}, {{chapter}}
// path: {{path}}
function main() {
  console.log(""PASS"");
}
main();
",
        ["csharp"] = @"// Task {{number}} - {{title}}
// tier {{tier}}, {{chapter}}
// path: {{path}}
using System;
public static class Solution
{
    public static void Main()
    {
        Console.WriteLine(""PASS"");
    }
}
",
    };
    private static string Extension(string language) => language == "python" ? "py" : language == "java" ? "java" :
        language == "javascript" ? "js" : language == "csharp" ? "cs" : throw new Exception("unknown language: " + language);

    private static string SolutionFile(string language) => language == "python" ? "solution.py" : language == "java" ? "Solution.java" :
        language == "javascript" ? "solution.js" : language == "csharp" ? "Solution.cs" : throw new Exception("unknown language: " + language);
    private static string PathFor(string number, string slug, string tier, string language) =>
        "solutions/" + language + "/" + tier + "/" + number + "-" + slug + "/" + SolutionFile(language);
    private static List<string> PlanPaths(int number, string slug, string tier)
    {
        string nn = number.ToString("D3", CultureInfo.InvariantCulture);
        var paths = new List<string>();
        foreach (string language in Languages) paths.Add(PathFor(nn, slug, tier, language));
        return paths;
    }    private static SortedSet<string> Placeholders(string template)
    {
        var found = new SortedSet<string>(StringComparer.Ordinal);
        int cursor = 0;
        while (true)
        {
            int open = template.IndexOf("{{", cursor, StringComparison.Ordinal);
            if (open < 0) return found;
            int close = template.IndexOf("}}", open + 2, StringComparison.Ordinal);
            if (close < 0) throw new Exception("unterminated placeholder in template at " + open);
            found.Add(template.Substring(open + 2, close - open - 2));
            cursor = close + 2;
        }
    }
    private static string Render(string template, Dictionary<string, string> values)
    {
        foreach (string key in Placeholders(template))
        {
            if (!values.TryGetValue(key, out string replacement))
                throw new Exception("missing placeholder: " + key);
            template = template.Replace("{{" + key + "}}", replacement);
        }
        if (template.Contains("{{", StringComparison.Ordinal) || template.Contains("}}", StringComparison.Ordinal)) throw new Exception("leftover marker after render");
        return template;
    }
    /** Values shared by every language template. */
    private static Dictionary<string, string> Values(int number, string slug, string title, string tier, string chapter)
    {
        return new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["number"] = number.ToString("D3", CultureInfo.InvariantCulture), ["slug"] = slug, ["title"] = title, ["tier"] = tier, ["chapter"] = chapter,
        };
    }
    private static Dictionary<string, string> LangValues(Dictionary<string, string> baseValues, string language)
    {
        var values = new Dictionary<string, string>(baseValues, StringComparer.Ordinal);
        values["lang"] = language; values["ext"] = Extension(language);
        values["path"] = PathFor(baseValues["number"], baseValues["slug"], baseValues["tier"], language);
        return values;
    }
    private static List<string> Lint(string text)
    {
        var problems = new List<string>();
        if (text.Contains("{{", StringComparison.Ordinal) || text.Contains("}}", StringComparison.Ordinal))
            problems.Add("leftover placeholder marker");
        if (text.Length == 0 || text[text.Length - 1] != '\n') problems.Add("missing final newline");
        string[] lines = text.Split('\n');
        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i].Contains("\t", StringComparison.Ordinal)) problems.Add("tab on line " + (i + 1).ToString(CultureInfo.InvariantCulture));
            if (lines[i].Length > 0 && lines[i][lines[i].Length - 1] == ' ') problems.Add("trailing space on line " + (i + 1).ToString(CultureInfo.InvariantCulture));
        }
        return problems;
    }
    private static bool ValidateSlug(string candidate)
    {
        if (candidate.Length < 3 || candidate.Length > 64) return false;
        if (candidate[0] == '-' || candidate[candidate.Length - 1] == '-') return false;
        char previous = '\0';
        foreach (char ch in candidate)
        {
            bool ok = (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '-';
            if (!ok) return false;
            if (ch == '-' && previous == '-') return false;
            previous = ch;
        }
        return true;
    }
    /** Independent oracle: recover the problem identity from the rendered file text alone. */
    private static (int Number, string Slug, string Title) Extract(string text)
    {
        string path = null!;
        string title = null!;
        foreach (string rawLine in text.Split('\n'))
        {
            string line = rawLine.Trim();
            int p = line.IndexOf("path:", StringComparison.Ordinal);
            if (p >= 0) path = line.Substring(p + 5).Trim();
            int t = line.IndexOf("Task ", StringComparison.Ordinal);
            if (t >= 0)
            {
                int dash = line.IndexOf(" - ", t + 5, StringComparison.Ordinal);
                if (dash > 0) title = line.Substring(dash + 3).Trim();
            }
        }
        if (path == null || title == null) throw new Exception("extraction failed: no path or title marker");
        string[] parts = path.Split('/');
        string folder = parts[parts.Length - 2];
        int separator = folder.IndexOf('-', StringComparison.Ordinal);
        return (int.Parse(folder.Substring(0, separator), CultureInfo.InvariantCulture), folder.Substring(separator + 1), title);
    }
    public static void Main()
    {
        const int number = 42;
        const string slug = "reverse-a-linked-list";
        const string title = "Reverse a linked list";
        Dictionary<string, string> values = Values(number, slug, title, "intermediate", "Ch 2 - Linked Lists");
        var rendered = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string language in Languages)
        {
            string text = Render(Templates[language], LangValues(values, language));
            rendered[language] = text;
            Check(!text.Contains("{{", StringComparison.Ordinal), language + " template renders with no leftover marker");
            Check(text.Contains("042-reverse-a-linked-list"), language + " template embeds the numbered folder name");
            Check(Lint(text).Count == 0, language + " template lints clean");
        }
        Check(rendered.Count == Languages.Length, "all four language templates render");
        foreach (string language in Languages)
        {
            (int extractedNumber, string extractedSlug, string extractedTitle) = Extract(rendered[language]);
            Check(extractedNumber == number && extractedSlug == slug && extractedTitle == title,
                "round-trip extraction of the " + language + " template recovers 042/reverse-a-linked-list/title");
        }
        var incomplete = new Dictionary<string, string>(values, StringComparer.Ordinal);
        incomplete.Remove("chapter");
        Check(Fails(() => Render(Templates["python"], incomplete)).Contains("chapter"),
            "a missing placeholder is rejected and the offending key is named");
        Check(Fails(() => Render("x {{author}} y", values)).Contains("author"), "an unknown placeholder is rejected");
        Check(Fails(() => Render("x {{number", values)).Contains("unterminated"), "an unterminated placeholder is rejected");
        Check(Lint("line with {{oops}}\n").Count > 0, "the linter catches a leftover marker that bypassed rendering");
        Check(Lint("tab\there\n").Count > 0 && Lint("no newline").Count > 0, "the linter catches tabs and a missing final newline");
        List<string> plan = PlanPaths(number, slug, "intermediate");
        Check(plan.Count == 4 && plan[3] == "solutions/csharp/intermediate/042-reverse-a-linked-list/Solution.cs",
            "the plan places four files in identically named per-language folders");
        Check(PlanPaths(7, "two-sum", "beginner")[0] == "solutions/python/beginner/007-two-sum/solution.py",
            "path numbers are zero padded to three digits");
        Check(PlanPaths(1000, "big-problem", "master")[1] == "solutions/java/master/1000-big-problem/Solution.java",
            "numbers above 999 are not truncated");
        Check(ValidateSlug("two-sum") && ValidateSlug("fizz-buzz-412"), "lower-case dash slugs are accepted");
        Check(!ValidateSlug("Two-Sum") && !ValidateSlug("two--sum") && !ValidateSlug("-two-sum") && !ValidateSlug("two-sum-") && !ValidateSlug("ab"),
            "upper case, doubled dashes, leading/trailing dashes and too-short slugs are rejected");
        const string trickyTitle = "Compare {a} & <b> carefully";
        var trickyValues = LangValues(Values(3, "compare-tokens", trickyTitle, "expert", "Ch 4 - Trees"), "csharp");
        string trickyText = Render(Templates["csharp"], trickyValues);
        Check(trickyText.Contains(trickyTitle), "braces, ampersands and angle brackets in the title survive verbatim");
        Check(Render(Templates["csharp"], trickyValues) == trickyText, "rendering the same template twice is byte-identical");
        (_, _, string trickyExtracted) = Extract(trickyText);
        Check(trickyExtracted == trickyTitle, "the round-trip oracle preserves a title containing braces");
        Console.WriteLine("templates: " + Languages.Length + " languages, " + Templates["csharp"].Split('\n').Length + "-line csharp skeleton, all lint clean");
        Console.WriteLine("plan: " + string.Join(", ", plan));
        Console.WriteLine("markers: missing/unknown/unterminated placeholders all rejected; slug rules enforced");
        Console.WriteLine("PASS");
    }
}