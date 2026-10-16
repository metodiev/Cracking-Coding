// Task 997 - Implement ethics section generator for problems with privacy/security implications.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Problem tags map onto ethics risk categories (personal data, security, bias, dual use, medical), each
// category demands a fixed set of clauses, and combinations add cross-cutting clauses (personal data plus
// security demands data minimisation for any exploit traces). The generator emits the mandatory clauses plus
// the optional ones the policy keeps enabled, always in a stable sorted order so the same tag set yields the
// same section, and refuses unknown tags and any attempt to suppress a mandatory clause. Verified: the clause
// set against a brute-force recomputation from the tables (oracle), the combination clause appearing only when
// both categories are present, the unknown-tag and suppression refusals, order independence, and the
// no-tags and all-tags boundaries.
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }
    private static string Error(Action action)
    {
        try { action(); return ""; }
        catch (Exception ex) { return ex.Message; }
    }

    private sealed class Clause
    {
        public readonly string Id, Text;
        public readonly bool Mandatory;
        public Clause(string id, string text, bool mandatory) { Id = id; Text = text; Mandatory = mandatory; }
    }

    private static readonly Dictionary<string, string> TagToCategory = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["pii"] = "pii", ["logs"] = "pii", ["telemetry"] = "pii",
        ["exploit"] = "security", ["crypto"] = "security", ["auth"] = "security",
        ["ranking"] = "bias", ["scoring"] = "bias",
        ["dual-use"] = "dual-use", ["malware"] = "dual-use",
        ["health"] = "medical", ["genetics"] = "medical",
    };

    private static readonly Dictionary<string, List<string>> CategoryClauses = new Dictionary<string, List<string>>(StringComparer.Ordinal)
    {
        ["pii"] = new List<string> { "pii-de-identify", "pii-consent" },
        ["security"] = new List<string> { "security-disclosure", "security-safe-harbour" },
        ["bias"] = new List<string> { "bias-audit" },
        ["dual-use"] = new List<string> { "dual-use-review" },
        ["medical"] = new List<string> { "medical-advice" },
    };

    private static readonly List<Clause> BaseClauses = new List<Clause>
    {
        new Clause("integrity", "Solutions are offered for study; behaviour that could harm others is out of scope.", true),
        new Clause("citations", "Editorial claims cite the verified checks of this repository.", false),
    };

    private static readonly Dictionary<string, Clause> ClauseTable = new Dictionary<string, Clause>(StringComparer.Ordinal)
    {
        ["pii-de-identify"] = new Clause("pii-de-identify", "Any dataset must be de-identified before publication.", true),
        ["pii-consent"] = new Clause("pii-consent", "Personal data is used only with the consent recorded by the task.", true),
        ["security-disclosure"] = new Clause("security-disclosure", "Vulnerabilities found in practice are reported responsibly.", false),
        ["security-safe-harbour"] = new Clause("security-safe-harbour", "Exploit code stays inside the sandbox and never addresses production.", false),
        ["bias-audit"] = new Clause("bias-audit", "Ranking and scoring rules are audited for disparate impact.", true),
        ["dual-use-review"] = new Clause("dual-use-review", "Dual-use techniques are reviewed before publication.", true),
        ["medical-advice"] = new Clause("medical-advice", "Nothing here is medical advice.", true),
        ["pii-security-minimise"] = new Clause("pii-security-minimise", "After a security exercise, personal data in traces is minimised and purged.", true),
    };

    private static List<string> Categories(List<string> tags)
    {
        var categories = new List<string>();
        foreach (string tag in tags)
        {
            if (!TagToCategory.TryGetValue(tag, out string category)) throw new Exception("unknown-tag:" + tag);
            if (!categories.Contains(category)) categories.Add(category);
        }
        categories.Sort(StringComparer.Ordinal);
        return categories;
    }

    /** Clauses demanded by a tag set, before any policy filtering; the oracle recomputes the same set. */
    private static List<string> Required(List<string> tags)
    {
        List<string> categories = Categories(tags);
        var required = new List<string>();
        foreach (string category in categories)
            foreach (string clause in CategoryClauses[category])
                if (!required.Contains(clause)) required.Add(clause);
        if (categories.Contains("pii") && categories.Contains("security")) required.Add("pii-security-minimise");
        required.Sort(StringComparer.Ordinal);
        return required;
    }

    private sealed class Policy
    {
        public readonly HashSet<string> Suppressed = new HashSet<string>(StringComparer.Ordinal);
        public Policy(params string[] suppressed) { foreach (string id in suppressed) Suppressed.Add(id); }
    }

    private static string Section(List<string> tags, Policy policy)
    {
        List<string> required = Required(tags);
        foreach (string id in required)
        {
            if (!policy.Suppressed.Contains(id)) continue;
            if (ClauseTable[id].Mandatory) throw new Exception("clause-suppressed:" + id);
        }
        foreach (Clause clause in BaseClauses)
            if (clause.Mandatory && policy.Suppressed.Contains(clause.Id))
                throw new Exception("clause-suppressed:" + clause.Id);
        var emitted = new List<string>();
        foreach (string id in required) if (!policy.Suppressed.Contains(id)) emitted.Add(id);
        bool hasRisks = required.Count > 0;
        foreach (Clause clause in BaseClauses)
            if (clause.Mandatory || (hasRisks && !policy.Suppressed.Contains(clause.Id))) emitted.Add(clause.Id);
        emitted.Sort(StringComparer.Ordinal);
        var text = new StringBuilder("## ethics\n");
        foreach (string id in emitted)
            text.Append("- ").Append(id).Append(": ").Append(ClauseTable.ContainsKey(id) ? ClauseText(id) : BaseText(id)).Append('\n');
        return text.ToString();
    }

    private static string ClauseText(string id) => ClauseTable[id].Text;
    private static string BaseText(string id)
    {
        foreach (Clause clause in BaseClauses) if (clause.Id == id) return clause.Text;
        throw new Exception("unknown-clause:" + id);
    }

    public static void Main()
    {
        var tags = new List<string> { "logs", "exploit", "ranking", "small" };
        // Oracle: rebuild the requirement set by brute force over the tables instead of calling Required.
        var brute = new List<string>();
        foreach (string category in new List<string> { "bias", "pii", "security" })
            foreach (string clause in CategoryClauses[category]) if (!brute.Contains(clause)) brute.Add(clause);
        brute.Add("pii-security-minimise");
        brute.Sort(StringComparer.Ordinal);
        bool unknownRefused = false;
        try { Required(tags); } catch (Exception ex) { unknownRefused = ex.Message == "unknown-tag:small"; }
        Check(unknownRefused && string.Join(",", Required(tags.GetRange(0, 3))) == string.Join(",", brute),
              "the clause set matches the brute-force rebuild and an unknown tag is refused (oracle)");

        string section = Section(tags.GetRange(0, 3), new Policy());
        Check(section.Contains("pii-security-minimise") && section.Contains("pii-de-identify") && section.Contains("bias-audit")
              && section.Contains("integrity"),
              "personal data plus security add the minimisation clause alongside the category clauses");

        Check(!Section(new List<string> { "logs" }, new Policy()).Contains("pii-security-minimise")
              && !Section(new List<string> { "exploit" }, new Policy()).Contains("pii-security-minimise")
              && Section(new List<string> { "exploit" }, new Policy()).Contains("security-disclosure"),
              "the combination clause appears only when both categories are present");

        Check(Error(() => Section(new List<string> { "pii" }, new Policy("pii-de-identify"))) == "clause-suppressed:pii-de-identify"
              && Error(() => Section(new List<string> { "auth" }, new Policy("integrity"))) == "clause-suppressed:integrity",
              "suppressing a mandatory clause is refused");

        string withOptional = Section(new List<string> { "pii" }, new Policy());
        string withoutOptional = Section(new List<string> { "pii" }, new Policy("citations", "security-disclosure"));
        Check(withOptional.Contains("citations") && !withoutOptional.Contains("citations")
              && withoutOptional.Contains("pii-consent") && Section(new List<string> { "pii" }, new Policy()) == withOptional,
              "an optional clause can be dropped while mandatory ones stay, and rendering is deterministic");

        Check(Section(new List<string> { "telemetry", "logs" }, new Policy()) == Section(new List<string> { "logs", "telemetry" }, new Policy())
              && Section(new List<string> { "logs" }, new Policy()).Split('\n').Length
                 == Section(new List<string> { "logs" }, new Policy("citations")).Split('\n').Length + 1,
              "duplicate and reordered tags render identically and dropping an optional clause removes exactly one line");

        var all = new List<string>();
        foreach (string tag in TagToCategory.Keys) all.Add(tag);
        string everything = Section(all, new Policy());
        List<string> required = Required(all);
        bool complete = true;
        foreach (string id in required) if (!everything.Contains(ClauseText(id))) complete = false;
        Check(complete && required.Count == 8 && Section(new List<string>(), new Policy()).Contains("integrity")
              && Section(new List<string>(), new Policy()).Split('\n').Length == 3
              && !Section(new List<string>(), new Policy()).Contains("citations"),
              "the all-tags and no-tags boundaries are complete and minimal");

        Console.WriteLine("categories: " + CategoryClauses.Count + ", clauses: " + ClauseTable.Count
                          + ", required for the sample: " + required.Count);
        Console.WriteLine("PASS");
    }
}
