// Task 967 - Implement accessibility features for lesson content (screen-reader friendly).
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Lesson content is modelled as a block stream (headings, paragraphs, images, links, code) and audited the
// way an accessibility checker would: exactly one h1, no skipped heading levels, meaningful alt text on
// every image, link text that describes its destination, and a language on every code block.  Colour pairs
// are scored with the WCAG relative-luminance formula, and the screen-reader outline is derived from the
// blocks so a reader can be tested against it.
//
// Verified: black on white is exactly 21:1 and the two published grey boundaries score 4.54 and 4.48 (an
// independent oracle for the formula), contrast is symmetric and never below 1, the 4.5/3.0 thresholds are
// inclusive, heading skips, placeholder alt text, vague link text and unlabelled code are all reported,
// a clean document is clean, the empty and duplicate-h1 boundaries hold, and the outline matches an
// independently projected rendering.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    private sealed class Block
    {
        public string Kind = "paragraph";
        public int Level;
        public string Text = "";
        public string Alt = "";
        public string Href = "";
    }

    private static Block Heading(int level, string text) => new Block { Kind = "heading", Level = level, Text = text };
    private static Block Paragraph(string text) => new Block { Kind = "paragraph", Text = text };
    private static Block Image(string alt) => new Block { Kind = "image", Alt = alt };
    private static Block Link(string text, string href) => new Block { Kind = "link", Text = text, Href = href };
    private static Block Code(string language) => new Block { Kind = "code", Text = language };

    private static readonly string[] Announced = { "heading", "image", "link" };
    private static readonly string[] PlaceholderAlt = { "", "image", "photo", "picture", "icon", "graphic" };
    private static readonly string[] VagueLink = { "click here", "here", "link", "read more", "more" };

    private static List<string> Audit(List<Block> blocks)
    {
        var issues = new List<string>();
        int h1 = 0, previousLevel = 0;
        foreach (Block block in blocks)
        {
            if (block.Kind == "heading")
            {
                if (block.Level == 1) h1++;
                if (previousLevel > 0 && block.Level > previousLevel + 1) issues.Add("heading skips from h" + previousLevel + " to h" + block.Level + " at '" + block.Text + "'");
                previousLevel = block.Level;
            }
            else if (block.Kind == "image")
            {
                if (Array.IndexOf(PlaceholderAlt, block.Alt.Trim().ToLowerInvariant()) >= 0) issues.Add("image has no meaningful alt text");
            }
            else if (block.Kind == "link")
            {
                if (Array.IndexOf(VagueLink, block.Text.Trim().ToLowerInvariant()) >= 0) issues.Add("link text '" + block.Text + "' does not describe the destination");
                if (block.Href.Trim().Length == 0) issues.Add("link with an empty href");
            }
            else if (block.Kind == "code" && block.Text.Trim().Length == 0) issues.Add("code block without a language");
        }
        if (h1 != 1) issues.Add("document has " + h1 + " h1 headings, expected exactly 1");
        return issues;
    }

    private static double Channel(int value)
    {
        double c = value / 255.0;
        return c <= 0.03928 ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);
    }

    private static double Luminance(int r, int g, int b) => 0.2126 * Channel(r) + 0.7152 * Channel(g) + 0.0722 * Channel(b);

    private static double Contrast(int[] foreground, int[] background)
    {
        double a = Luminance(foreground[0], foreground[1], foreground[2]);
        double b = Luminance(background[0], background[1], background[2]);
        return (Math.Max(a, b) + 0.05) / (Math.Min(a, b) + 0.05);
    }

    private static bool MeetsWcagAa(double ratio, bool largeText) => ratio >= (largeText ? 3.0 : 4.5);

    /** What a screen reader announces, in document order. */
    private static List<string> Outline(List<Block> blocks)
    {
        var outline = new List<string>();
        foreach (Block block in blocks)
        {
            if (block.Kind == "heading") outline.Add("h" + block.Level + ": " + block.Text);
            else if (block.Kind == "image") outline.Add("image: " + block.Alt);
            else if (block.Kind == "link") outline.Add("link: " + block.Text);
        }
        return outline;
    }

    /** Independent projection: filter which kinds are announced first, then render each entry. */
    private static List<string> OutlineOracle(List<Block> blocks)
    {
        var announced = new List<Block>();
        foreach (Block block in blocks) { if (Array.IndexOf(Announced, block.Kind) >= 0) announced.Add(block); }
        var rendered = new List<string>();
        foreach (Block block in announced)
        {
            switch (block.Kind)
            {
                case "heading": rendered.Add("h" + block.Level + ": " + block.Text); break;
                case "image": rendered.Add("image: " + block.Alt); break;
                default: rendered.Add("link: " + block.Text); break;
            }
        }
        return rendered;
    }

    private static int CountMatching(List<string> issues, string prefix)
    {
        int count = 0;
        foreach (string issue in issues) { if (issue.StartsWith(prefix, StringComparison.Ordinal)) count++; }
        return count;
    }

    private static int[] Rgb(int r, int g, int b) => new[] { r, g, b };

    public static void Main()
    {
        double blackOnWhite = Contrast(Rgb(0, 0, 0), Rgb(255, 255, 255));
        Check(Math.Abs(blackOnWhite - 21.0) < 1e-9, "black on white is exactly 21:1 (" + blackOnWhite.ToString("F9", CultureInfo.InvariantCulture) + ")");

        // Published WCAG reference values: #767676 is the lightest grey passing 4.5 on white, #777777 the first failing one.
        double pass = Contrast(Rgb(118, 118, 118), Rgb(255, 255, 255));
        double fail = Contrast(Rgb(119, 119, 119), Rgb(255, 255, 255));
        Check(Math.Abs(pass - 4.54) < 0.01 && Math.Abs(fail - 4.48) < 0.01,
            "the published grey boundaries score 4.54 (pass) and 4.48 (fail): " + pass.ToString("F4", CultureInfo.InvariantCulture) + " / " + fail.ToString("F4", CultureInfo.InvariantCulture));
        Check(MeetsWcagAa(pass, false) && !MeetsWcagAa(fail, false), "the 4.5 threshold admits #767676 and rejects #777777");
        Check(MeetsWcagAa(3.0, true) && !MeetsWcagAa(2.999, true), "the large-text threshold of 3.0 is inclusive at exactly 3.0");
        int[] same = Rgb(7, 7, 7);
        Check(Math.Abs(Contrast(Rgb(40, 40, 40), Rgb(255, 255, 255)) - Contrast(Rgb(255, 255, 255), Rgb(40, 40, 40))) < 1e-12 && Math.Abs(Contrast(same, same) - 1.0) < 1e-12,
            "contrast is symmetric and identical colours give exactly 1:1");

        var skipped = new List<Block> { Heading(1, "Lesson"), Heading(3, "Jump"), Paragraph("body"), Image("recursion call tree"), Link("View the solution", "solution-951.html") };
        Check(Audit(skipped).Exists(i => i.StartsWith("heading skips", StringComparison.Ordinal)), "a heading level skip from h1 to h3 is reported");
        Check(Audit(new List<Block> { Heading(1, "Lesson"), Heading(2, "Step"), Heading(3, "Detail") }).Count == 0,
            "an unbroken h1 -> h2 -> h3 hierarchy is clean");

        var badAlt = new List<Block> { Heading(1, "Images"), Image("image"), Image("") };
        Check(badAlt.Count > 0 && CountMatching(Audit(badAlt), "image has no meaningful") == 2,
            "placeholder and empty alt text are both reported");
        Check(Audit(new List<Block> { Heading(1, "Images"), Image("call stack after push(4)") }).Count == 0, "a descriptive alt text passes");

        var badLinks = new List<Block> { Heading(1, "Links"), Link("click here", "solution-952.html"), Link("See the proof", "") };
        List<string> linkIssues = Audit(badLinks);
        Check(linkIssues.Exists(i => i.Contains("does not describe", StringComparison.Ordinal)) && linkIssues.Exists(i => i.Contains("empty href", StringComparison.Ordinal)),
            "vague link text and an empty href are both reported");
        Check(Audit(new List<Block> { Heading(1, "Links"), Link("Read the complexity proof", "proof.html") }).Count == 0, "a descriptive link with a target passes");

        Check(Audit(new List<Block> { Heading(1, "Code"), Code("") }).Exists(i => i.Contains("without a language", StringComparison.Ordinal)),
            "a code block without a language is reported");
        Check(Audit(new List<Block> { Heading(1, "Code"), Code("csharp"), Paragraph("Console.WriteLine(1);") }).Count == 0, "a labelled code block passes");

        var clean = new List<Block>
        {
            Heading(1, "Sorting"), Paragraph("Sorting puts elements in order."), Heading(2, "Merge sort"),
            Code("csharp"), Image("merge sort recursion tree"), Link("Open problem 951", "solution-951.html"), Heading(2, "Complexity"), Paragraph("O(n log n)."),
        };
        Check(Audit(clean).Count == 0, "a well-formed lesson produces no accessibility issues");

        Check(Audit(new List<Block>()).Count == 1, "an empty document reports exactly one issue (no h1)");
        Check(Audit(new List<Block> { Heading(1, "A"), Heading(1, "B") }).Exists(i => i.Contains("expected exactly 1", StringComparison.Ordinal)),
            "two h1 headings are reported as a single-title violation");

        List<string> outline = Outline(clean);
        Check(outline.Count == 5 && string.Join("|", outline) == string.Join("|", OutlineOracle(clean)),
            "the screen-reader outline matches the independent projection (" + outline.Count + " entries)");
        Check(outline[0] == "h1: Sorting" && outline[1] == "h2: Merge sort" && outline[4] == "h2: Complexity",
            "the outline preserves document order: " + string.Join(" -> ", outline));
        Check(string.Join("|", Outline(clean)) == string.Join("|", Outline(clean)) && Audit(clean).Count == 0,
            "auditing and outlining the same lesson twice is deterministic");

        Console.WriteLine("contrast: black/white " + blackOnWhite.ToString("F4", CultureInfo.InvariantCulture) + ", #767676 " + pass.ToString("F4", CultureInfo.InvariantCulture) +
            ", #777777 " + fail.ToString("F4", CultureInfo.InvariantCulture) + " (WCAG AA needs 4.5 or 3.0 for large text)");
        Console.WriteLine("outline: " + string.Join(" -> ", outline));
        Console.WriteLine("audit: heading/alt/link/code/title rules exercised; clean lesson has " + Audit(clean).Count + " issues");
        Console.WriteLine("PASS");
    }
}
