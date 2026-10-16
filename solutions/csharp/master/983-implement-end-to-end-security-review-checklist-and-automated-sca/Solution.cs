// Task 983 - Implement end-to-end security review checklist and automated scans for repository code.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// A scan rule set (regex per vulnerability class with a severity) runs over every line of the
// repository snapshot, findings are ordered by path, line and rule so reviews are reproducible, and a
// release gate blocks the build when any high-severity finding survives without a written waiver. The
// checklist part requires every mandatory review item to be signed off independently of the scan.
//
// Verified: each rule fires on a crafted vulnerable line, a clean snapshot produces zero findings, the
// finding order is deterministic, the gate blocks on high severity but passes with a waiver or with
// only medium findings, an unsigned checklist item blocks the release, the empty snapshot boundary,
// and an independent string-based scanner that reproduces the regex scanner's counts.
// Simplification: rules are regexes over text lines with a fixed severity table; there is no
// data-flow analysis, no dependency or container scanning and no secret entropy check.
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Rule
    {
        public readonly string Id;
        public readonly string Severity;
        public readonly string Pattern;
        public readonly Regex Expression;

        public Rule(string id, string severity, string pattern)
        {
            Id = id;
            Severity = severity;
            Pattern = pattern;
            Expression = new Regex(pattern, RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);
        }
    }

    private sealed class CodeFile
    {
        public readonly string Path;
        public readonly string[] Lines;

        public CodeFile(string path, params string[] lines)
        {
            Path = path;
            Lines = lines;
        }
    }

    private sealed class Finding
    {
        public readonly string Rule;
        public readonly string Severity;
        public readonly string Path;
        public readonly int Line;

        public Finding(string rule, string severity, string path, int line)
        {
            Rule = rule;
            Severity = severity;
            Path = path;
            Line = line;
        }
    }

    private static readonly Rule[] Rules =
    {
        new Rule("hardcoded-secret", "high", "(password|apikey|api_key|secret)\\s*=\\s*\"[^\"]+\""),
        new Rule("sql-concatenation", "high", "\"(select|insert|update|delete)[^\"]*\"\\s*\\+"),
        new Rule("tls-verification-disabled", "high", "(servercertificatevalidationcallback|verify\\s*=\\s*false|insecureskipverify)"),
        new Rule("weak-hash", "medium", "(md5|sha1)\\b"),
        new Rule("unsafe-eval", "medium", "(\\beval\\(|dangerouslysetinnerhtml)"),
    };

    private static readonly string[] MandatoryChecklist =
    {
        "threat-model-reviewed",
        "dependency-scan-clean",
        "secrets-rotation-verified",
        "least-privilege-iam",
    };

    private static int SeverityRank(string severity)
        => severity == "critical" ? 3 : severity == "high" ? 2 : severity == "medium" ? 1 : 0;

    private static List<Finding> Scan(List<CodeFile> code)
    {
        var findings = new List<Finding>();
        foreach (CodeFile file in code)
            for (int i = 0; i < file.Lines.Length; i++)
                foreach (Rule rule in Rules)
                    if (rule.Expression.IsMatch(file.Lines[i]))
                        findings.Add(new Finding(rule.Id, rule.Severity, file.Path, i + 1));
        findings.Sort((a, b) =>
        {
            int byPath = string.CompareOrdinal(a.Path, b.Path);
            if (byPath != 0) return byPath;
            if (a.Line != b.Line) return a.Line.CompareTo(b.Line);
            return string.CompareOrdinal(a.Rule, b.Rule);
        });
        return findings;
    }

    private sealed class Gate
    {
        public bool Passed;
        public int Blocking;
        public string Reason = "";
    }

    private static Gate Release(List<Finding> findings, List<string> waivedRules, HashSet<string> checklist)
    {
        var gate = new Gate();
        foreach (Finding finding in findings)
            if (SeverityRank(finding.Severity) >= 2 && !waivedRules.Contains(finding.Rule)) gate.Blocking++;
        if (gate.Blocking > 0)
        {
            gate.Reason = gate.Blocking + " high severity finding(s) unresolved";
            return gate;
        }
        foreach (string item in MandatoryChecklist)
            if (!checklist.Contains(item))
            {
                gate.Reason = "checklist item not signed off: " + item;
                return gate;
            }
        gate.Passed = true;
        gate.Reason = "no blocking findings and the checklist is complete";
        return gate;
    }

    private static HashSet<string> SignedChecklist()
        => new HashSet<string>(MandatoryChecklist, StringComparer.Ordinal);

    public static void Main()
    {
        var vulnerable = new List<CodeFile>
        {
            new CodeFile("src/db.cs",
                "var sql = \"select * from users where id = \" + id;",
                "var digest = MD5.Create();",
                "Console.WriteLine(sql);"),
            new CodeFile("src/auth.cs",
                "var password = \"hunter2\";",
                "ServicePointManager.ServerCertificateValidationCallback = (a, b, c, d) => true;"),
            new CodeFile("web/app.js",
                "element.innerHTML = dangerouslySetInnerHTML(userInput);",
                "var hash = sha1(data);"),
        };
        List<Finding> findings = Scan(vulnerable);
        Check(findings.Count == 6, "the vulnerable snapshot produces six findings (got " + findings.Count + ")");
        var rulesSeen = new HashSet<string>(StringComparer.Ordinal);
        foreach (Finding finding in findings) rulesSeen.Add(finding.Rule);
        Check(rulesSeen.Count == Rules.Length, "all five rules fire on the crafted snapshot");

        // Deterministic ordering: sorted by path, then line, then rule id.
        bool ordered = true;
        for (int i = 1; i < findings.Count; i++)
        {
            Finding previous = findings[i - 1], current = findings[i];
            int byPath = string.CompareOrdinal(previous.Path, current.Path);
            if (byPath > 0) ordered = false;
            else if (byPath == 0 && previous.Line > current.Line) ordered = false;
            else if (byPath == 0 && previous.Line == current.Line
                     && string.CompareOrdinal(previous.Rule, current.Rule) > 0) ordered = false;
        }
        Check(ordered, "findings are reported in a stable path, line, rule order");

        // Oracle: a string-based scanner independent of the regex engine must count the same hits.
        int independent = 0;
        foreach (CodeFile file in vulnerable)
            foreach (string line in file.Lines)
            {
                string lower = line.ToLowerInvariant();
                if ((lower.Contains("password =") || lower.Contains("apikey") || lower.Contains("secret ="))
                    && lower.Contains("\"")) independent++;
                if (lower.Contains("\"select") && lower.Contains("+")) independent++;
                if (lower.Contains("servercertificatevalidationcallback") || lower.Contains("insecureskipverify")) independent++;
                if (lower.Contains("md5") || lower.Contains("sha1")) independent++;
                if (lower.Contains("eval(") || lower.Contains("dangerouslysetinnerhtml")) independent++;
            }
        Check(independent == findings.Count, "an independent substring scanner agrees on the finding count ("
            + independent + " vs " + findings.Count + ")");

        // False-positive guard: a clean snapshot stays clean.
        var clean = new List<CodeFile>
        {
            new CodeFile("src/safe.cs",
                "var sql = \"select * from users where id = @id\";",
                "using (var connection = new SqlConnection(connectionString)) { }",
                "var digest = SHA256.Create();",
                "if (!File.Exists(path)) return;"),
        };
        Check(Scan(clean).Count == 0, "a parameterised, SHA256-only snapshot produces no findings");

        // Gate behaviour: block on high severity, allow a waiver, and re-block on the checklist.
        Gate blocked = Release(findings, new List<string>(), SignedChecklist());
        Check(!blocked.Passed && blocked.Blocking == 3 && blocked.Reason.Contains("high severity"),
            "three high severity findings block the release (got " + blocked.Blocking + ")");
        Check(Release(findings, new List<string> { "hardcoded-secret" }, SignedChecklist()).Blocking == 2,
            "waiving one rule suppresses only that rule's findings");
        var highWaivers = new List<string> { "hardcoded-secret", "sql-concatenation", "tls-verification-disabled" };
        Check(Release(findings, highWaivers, SignedChecklist()).Passed,
            "waiving every high severity rule lets the release through");
        Gate partial = Release(findings, highWaivers,
            new HashSet<string>(new[] { "threat-model-reviewed" }, StringComparer.Ordinal));
        Check(!partial.Passed && partial.Reason.Contains("dependency-scan-clean"),
            "an incomplete checklist blocks the release even with every high finding waived");

        // Boundary: an empty repository, and a waiver that must not suppress other rules.
        Check(Scan(new List<CodeFile>()).Count == 0, "an empty snapshot produces no findings");
        Check(Release(new List<Finding>(), new List<string>(), SignedChecklist()).Passed,
            "a finding-free release passes the gate once the checklist is signed");
        Check(Release(new List<Finding>(), new List<string>(), new HashSet<string>(StringComparer.Ordinal)).Passed == false,
            "the checklist is mandatory even with zero findings");
        var onlyMedium = new List<Finding> { new Finding("weak-hash", "medium", "src/db.cs", 2) };
        Check(Release(onlyMedium, new List<string>(), SignedChecklist()).Passed,
            "medium findings alone do not block the release");
        Check(Release(findings, new List<string> { "hardcoded-secret" }, SignedChecklist()).Blocking == 2,
            "waiving a single high rule leaves the other two blocking");

        Console.WriteLine("scan: " + findings.Count + " findings over " + Rules.Length + " rules, "
            + rulesSeen.Count + " rules fired");
        Console.WriteLine("gate: blocked by " + blocked.Blocking + " high findings -> '" + blocked.Reason + "'");
        Console.WriteLine("clean snapshot: " + Scan(clean).Count + " findings, empty snapshot: " + Scan(new List<CodeFile>()).Count);
        Console.WriteLine("PASS");
    }
}
