/** Task 096 - Simple URL parser (extract protocol, host). (Tier: Beginner, CTCI Ch 7)

 * Protocol is lowercased; absent path becomes "/"; port defaults to 80 (http) or
 * 443 (https), 0 for other schemes; absent query/fragment are "".
 */
public class Solution {
    static class Url {
        final String protocol, host, path, query, fragment;
        final int port;

        Url(String protocol, String host, int port, String path, String query, String fragment) {
            this.protocol = protocol;
            this.host = host;
            this.port = port;
            this.path = path;
            this.query = query;
            this.fragment = fragment;
        }
    }

    static Url parseUrl(String url) {
        int sep = url.indexOf("://");
        if (sep <= 0) throw new IllegalArgumentException("URL must start with <scheme>://");
        String protocol = url.substring(0, sep).toLowerCase();
        String rest = url.substring(sep + 3);

        String fragment = "";
        int hash = rest.indexOf('#');
        if (hash != -1) {
            fragment = rest.substring(hash + 1);
            rest = rest.substring(0, hash);
        }

        String query = "";
        int q = rest.indexOf('?');
        if (q != -1) {
            query = rest.substring(q + 1);
            rest = rest.substring(0, q);
        }

        String path = "/";
        int slash = rest.indexOf('/');
        if (slash != -1) {
            path = rest.substring(slash);
            rest = rest.substring(0, slash);
        }

        String host = rest;
        int port = protocol.equals("http") ? 80 : protocol.equals("https") ? 443 : 0;
        int colon = rest.indexOf(':');
        if (colon != -1) {
            host = rest.substring(0, colon);
            port = Integer.parseInt(rest.substring(colon + 1));
        }
        return new Url(protocol, host, port, path, query, fragment);
    }

    public static void main(String[] args) {
        checkUrl(parseUrl("https://example.com/path?q=1#frag"),
                "https", "example.com", 443, "/path", "q=1", "frag", "full https");
        checkUrl(parseUrl("http://example.com"),
                "http", "example.com", 80, "/", "", "", "no path");
        checkUrl(parseUrl("HTTP://example.com/x"),
                "http", "example.com", 80, "/x", "", "", "uppercase scheme");
        checkUrl(parseUrl("http://example.com:8080/a/b?x=1&y=2#top"),
                "http", "example.com", 8080, "/a/b", "x=1&y=2", "top", "explicit port");
        checkUrl(parseUrl("https://host?q=1"),
                "https", "host", 443, "/", "q=1", "", "query without path");
        checkUrl(parseUrl("https://host/#f"),
                "https", "host", 443, "/", "", "f", "fragment only");
        checkUrl(parseUrl("https://example.com:8443"),
                "https", "example.com", 8443, "/", "", "", "port without path");
        checkUrl(parseUrl("ftp://files.example.com/pub"),
                "ftp", "files.example.com", 0, "/pub", "", "", "unknown scheme");

        boolean threw = false;
        try {
            parseUrl("example.com/no-scheme");
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "missing scheme throws");
        System.out.println("PASS");
    }

    static void checkUrl(Url u, String protocol, String host, int port, String path,
                         String query, String fragment, String label) {
        check(u.protocol.equals(protocol) && u.host.equals(host) && u.port == port
                && u.path.equals(path) && u.query.equals(query) && u.fragment.equals(fragment), label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
