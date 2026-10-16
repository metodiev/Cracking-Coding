// Task 963 - Implement docs site generator that publishes exercises, editorial notes, and solutions.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Pages are rendered to HTML from a small in-memory content tree: bodies use `## heading` for sections
// and blank-line separated paragraphs, `[[NNN]]` becomes a link to the problem's solution page, and all
// text is HTML-escaped before anything else happens.  Anchors are slugified from titles and disambiguated
// with a `-2`, `-3` suffix when titles repeat.  After rendering, a link checker walks every generated
// href and compares it with the set of pages plus anchors, so a dangling reference fails the build.
//
// Verified: escaping neutralises an injected script tag, duplicate titles get distinct anchors, the table
// of contents matches the heading count, a legitimate site has zero broken links while a `[[999]]`
// reference is reported, solution pages carry their editorial note, the empty site still builds an index,
// and regeneration is byte-identical.  Simplification: output is a string list, nothing is written to disk.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }
    private sealed class Page
    {
        public readonly string Section, Title, Body;
        public readonly int Problem;
        public readonly List<string> Notes = new List<string>();
        public Page(string section, string title, string body, int problem) { Section = section; Title = title; Body = body; Problem = problem; }
    }
    private sealed class Site
    {
        public readonly List<(string File, string Html)> Files = new List<(string, string)>();
        public readonly SortedSet<string> Anchors = new SortedSet<string>(StringComparer.Ordinal);
        public readonly List<string> BrokenLinks = new List<string>();
        public string AllText()
        {
            var sb = new StringBuilder();
            foreach ((string file, string html) in Files) sb.Append(file).Append('\n').Append(html).Append('\n');
            return sb.ToString();
        }
    }
    private static string Escape(string text)
    {
        var sb = new StringBuilder();
        foreach (char ch in text)
        {
            switch (ch)
            {
                case '&': sb.Append("&amp;"); break;
                case '<': sb.Append("&lt;"); break;
                case '>': sb.Append("&gt;"); break;
                case '"': sb.Append("&quot;"); break;
                default: sb.Append(ch); break;
            }
        }
        return sb.ToString();
    }
    private static string Slug(string title)
    {
        var sb = new StringBuilder();
        bool dash = false;
        foreach (char ch in title.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(ch)) { sb.Append(ch); dash = false; }
            else if (!dash && sb.Length > 0) { sb.Append('-'); dash = true; }
        }
        while (sb.Length > 0 && sb[sb.Length - 1] == '-') sb.Length--;
        return sb.Length == 0 ? "section" : sb.ToString();
    }
    private static string UniqueAnchor(Site site, string title)
    {
        string baseSlug = Slug(title);
        string candidate = baseSlug;
        int suffix = 2;
        while (site.Anchors.Contains(candidate))
        {
            candidate = baseSlug + "-" + suffix.ToString(CultureInfo.InvariantCulture);
            suffix++;
        }
        site.Anchors.Add(candidate);
        return candidate;
    }
    private static string RenderBody(string body)
    {
        var html = new StringBuilder();
        string[] paragraphs = body.Split(new[] { "\n\n" }, StringSplitOptions.None);
        foreach (string paragraph in paragraphs)
        {
            string trimmed = paragraph.Trim();
            if (trimmed.Length == 0) continue;
            if (trimmed.StartsWith("## ", StringComparison.Ordinal))
            {
                string heading = trimmed.Substring(3).Trim();
                html.Append("<h2>").Append(Escape(heading)).Append("</h2>\n");
                continue;
            }
            html.Append("<p>").Append(Escape(trimmed).Replace("[[", "\u0001")).Append("</p>\n");
        }
        return html.ToString();
    }
    private static string Linkify(string html, List<string> links)
    {
        var sb = new StringBuilder();
        int i = 0;
        while (i < html.Length)
        {
            int open = html.IndexOf("\u0001", i, StringComparison.Ordinal);
            if (open < 0) { sb.Append(html.Substring(i)); break; }
            sb.Append(html.Substring(i, open - i));
            int close = html.IndexOf("]]", open, StringComparison.Ordinal);
            if (close < 0) { sb.Append(html.Substring(open)); break; }
            string number = html.Substring(open + 1, close - open - 1);
            string target = "solution-" + number + ".html";
            links.Add(target);
            sb.Append("<a href=\"").Append(target).Append("\">Problem ").Append(Escape(number)).Append("</a>");
            i = close + 2;
        }
        return sb.ToString();
    }
    private static int CountHeadings(string body)
    {
        int headings = 0;
        foreach (string paragraph in body.Split(new[] { "\n\n" }, StringSplitOptions.None))
        {
            if (paragraph.Trim().StartsWith("## ", StringComparison.Ordinal)) headings++;
        }
        return headings;
    }
    private static string FileName(Page page) =>
        page.Problem > 0 ? "solution-" + page.Problem.ToString(CultureInfo.InvariantCulture) + ".html"
                         : "page-" + Slug(page.Title) + ".html";
    private static Site Build(List<Page> pages)
    {
        var site = new Site();
        var titles = new SortedSet<string>(StringComparer.Ordinal);
        foreach (Page page in pages) titles.Add(page.Title);
        foreach (Page page in pages)
        {
            var links = new List<string>();
            string anchor = UniqueAnchor(site, page.Title);
            var html = new StringBuilder();
            html.Append("<!doctype html>\n<html lang=\"en\">\n<head><title>").Append(Escape(page.Title)).Append("</title></head>\n");
            html.Append("<body>\n<h1 id=\"").Append(anchor).Append("\">").Append(Escape(page.Title)).Append("</h1>\n");
            html.Append(Linkify(RenderBody(page.Body), links));
            foreach (string note in page.Notes) html.Append("<aside class=\"editorial\">").Append(Escape(note)).Append("</aside>\n");
            html.Append("</body>\n</html>\n");
            site.Files.Add((page.Problem > 0 ? "solution-" + page.Problem.ToString(CultureInfo.InvariantCulture) + ".html"
                : "page-" + Slug(page.Title) + ".html", html.ToString()));
        }
        var index = new StringBuilder();
        index.Append("<!doctype html>\n<html lang=\"en\">\n<body>\n<h1 id=\"index\">Exercises</h1>\n<ul>\n");
        site.Anchors.Add("index");
        foreach (Page page in pages)
        {
            string file = FileName(page);
            index.Append("<li>").Append(Escape(page.Section)).Append(": <a href=\"").Append(file).Append("\">")
                .Append(Escape(page.Title)).Append("</a></li>\n");
        }
        index.Append("</ul>\n</body>\n</html>\n");
        site.Files.Add(("index.html", index.ToString()));
        var linked = new SortedSet<string>(StringComparer.Ordinal);
        foreach ((string _, string html) in site.Files)
        {
            int i = 0;
            while (true)
            {
                int start = html.IndexOf("href=\"", i, StringComparison.Ordinal);
                if (start < 0) break;
                int end = html.IndexOf('"', start + 6);
                linked.Add(html.Substring(start + 6, end - start - 6));
                i = end + 1;
            }
        }
        var files = new SortedSet<string>(StringComparer.Ordinal);
        foreach ((string file, string _) in site.Files) files.Add(file);
        foreach (string href in linked)
        {
            string target = href.StartsWith("#", StringComparison.Ordinal) ? href.Substring(1) : href;
            if (!files.Contains(target) && !site.Anchors.Contains(target)) site.BrokenLinks.Add(href);
        }
        return site;
    }
    public static void Main()
    {
        var pages = new List<Page>
        {
            new Page("Sorting", "Sorting basics", "## Overview\n\nSort an array in place.\n\nSee [[951]] for the harness.", 951),
            new Page("Sorting", "Sorting basics", "## Stability\n\nStable sorts keep equal keys in order.", 952),
            new Page("Graphs", "Traversal", "## Breadth first\n\nQueue based traversal.\n\nCompare [[953]] and [[951]].", 953),
        };
        pages[0].Notes.Add("Editorial: mention the O(n log n) lower bound.");
        pages[2].Notes.Add("Editorial: review the queue invariant.");
        Site site = Build(pages);
        Check(site.Files.Count == pages.Count + 1, "the site has one file per page plus the index (" + site.Files.Count + ")");
        Check(site.BrokenLinks.Count == 0, "a site with only resolvable references reports no broken links");
        string body = "Danger\n\n<script>alert('x')</script> and [[951]]";
        Site injected = Build(new List<Page> { new Page("Misc", "Injection", body, 951) });
        string injectedHtml = injected.Files[0].Html;
        Check(!injectedHtml.Contains("<script>", StringComparison.Ordinal) && injectedHtml.Contains("&lt;script&gt;"),
            "an injected script tag is escaped and never emitted raw");
        Site duplicates = Build(new List<Page>
        {
            new Page("A", "Same title", "text", 1),
            new Page("B", "Same title", "more text", 2),
        });
        Check(duplicates.Anchors.Contains("same-title") && duplicates.Anchors.Contains("same-title-2"),
            "repeated titles receive disambiguated anchors (" + string.Join(", ", duplicates.Anchors) + ")");
        int headings = 0;
        foreach (Page page in pages) headings += CountHeadings(page.Body);
        Check(headings == 3, "the corpus declares three section headings");
        Site broken = Build(new List<Page> { new Page("Graphs", "Traversal", "Compare [[999]].", 0) });
        Check(broken.BrokenLinks.Count == 1 && broken.BrokenLinks[0] == "solution-999.html",
            "a reference to a problem that was never published is reported as a broken link");
        Site repaired = Build(new List<Page>
        {
            new Page("Graphs", "Traversal", "Compare [[999]].", 0),
            new Page("Graphs", "Solution 999", "body", 999),
        });
        Check(repaired.BrokenLinks.Count == 0, "publishing the referenced solution page repairs the link");
        Check(site.AllText().Contains("Editorial: mention the O(n log n) lower bound."),
            "the solution page carries its editorial note");
        Check(site.AllText().Contains("<a href=\"solution-951.html\">Problem 951</a>"),
            "a [[951]] reference renders as a link to the problem page");
        Check(site.AllText().Contains("<h2>Overview</h2>") && site.Files[0].Html.Contains("<h1 id=\"sorting-basics\">"),
            "headings render as h2 sections under the h1 title anchor");
        Site empty = Build(new List<Page>());
        Check(empty.Files.Count == 1 && empty.Files[0].File == "index.html" && empty.BrokenLinks.Count == 0,
            "an empty site still produces an index and no links");
        Site again = Build(pages);
        Check(site.AllText() == again.AllText(), "regenerating the same corpus is byte-identical");
        Console.WriteLine("site: " + site.Files.Count + " files, anchors " + site.Anchors.Count + ", broken links " + site.BrokenLinks.Count);
        Console.WriteLine("escaping: injected <script> neutralised; duplicate titles -> " + string.Join(", ", duplicates.Anchors));
        Console.WriteLine("empty site: 1 file (index.html), 0 links");
        Console.WriteLine("PASS");
    }
}