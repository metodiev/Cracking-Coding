// Task 096 - Simple URL parser (extract protocol, host). (Tier: Beginner, CTCI Ch 7)
// Protocol is lowercased; absent path becomes "/"; port defaults to 80 (http) or
// 443 (https), 0 for other schemes; absent query/fragment are "".
using System;

public static class Solution
{
    public static (string Protocol, string Host, int Port, string Path, string Query, string Fragment) ParseUrl(string url)
    {
        int sep = url.IndexOf("://", StringComparison.Ordinal);
        if (sep <= 0) throw new ArgumentException("URL must start with <scheme>://");
        string protocol = url.Substring(0, sep).ToLowerInvariant();
        string rest = url.Substring(sep + 3);

        string fragment = "";
        int hash = rest.IndexOf('#');
        if (hash != -1)
        {
            fragment = rest.Substring(hash + 1);
            rest = rest.Substring(0, hash);
        }

        string query = "";
        int q = rest.IndexOf('?');
        if (q != -1)
        {
            query = rest.Substring(q + 1);
            rest = rest.Substring(0, q);
        }

        string path = "/";
        int slash = rest.IndexOf('/');
        if (slash != -1)
        {
            path = rest.Substring(slash);
            rest = rest.Substring(0, slash);
        }

        string host = rest;
        int port = protocol == "http" ? 80 : protocol == "https" ? 443 : 0;
        int colon = rest.IndexOf(':');
        if (colon != -1)
        {
            host = rest.Substring(0, colon);
            port = int.Parse(rest.Substring(colon + 1));
        }
        return (protocol, host, port, path, query, fragment);
    }

    public static void Main()
    {
        Check(ParseUrl("https://example.com/path?q=1#frag") == ("https", "example.com", 443, "/path", "q=1", "frag"), "full https");
        Check(ParseUrl("http://example.com") == ("http", "example.com", 80, "/", "", ""), "no path");
        Check(ParseUrl("HTTP://example.com/x") == ("http", "example.com", 80, "/x", "", ""), "uppercase scheme");
        Check(ParseUrl("http://example.com:8080/a/b?x=1&y=2#top") == ("http", "example.com", 8080, "/a/b", "x=1&y=2", "top"), "explicit port");
        Check(ParseUrl("https://host?q=1") == ("https", "host", 443, "/", "q=1", ""), "query without path");
        Check(ParseUrl("https://host/#f") == ("https", "host", 443, "/", "", "f"), "fragment only");
        Check(ParseUrl("https://example.com:8443") == ("https", "example.com", 8443, "/", "", ""), "port without path");
        Check(ParseUrl("ftp://files.example.com/pub") == ("ftp", "files.example.com", 0, "/pub", "", ""), "unknown scheme");

        bool threw = false;
        try
        {
            ParseUrl("example.com/no-scheme");
        }
        catch (ArgumentException)
        {
            threw = true;
        }
        Check(threw, "missing scheme throws");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
