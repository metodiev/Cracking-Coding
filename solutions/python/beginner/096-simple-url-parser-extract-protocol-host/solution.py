"""Task 096 - Simple URL parser (extract protocol, host). (Tier: Beginner, CTCI Ch 7)

Splits <protocol>://<host>[:port]<path>?<query>#<fragment>.
Choices: protocol is lowercased; an absent path becomes "/"; the port defaults to
80 for http and 443 for https (0 for any other scheme); absent query/fragment are "".
"""

DEFAULT_PORTS = {"http": 80, "https": 443}


def parse_url(url):
    sep = url.find("://")
    if sep <= 0:
        raise ValueError("URL must start with <scheme>://")
    protocol = url[:sep].lower()
    rest = url[sep + 3:]

    fragment = ""
    if "#" in rest:
        rest, fragment = rest.split("#", 1)

    query = ""
    if "?" in rest:
        rest, query = rest.split("?", 1)

    path = "/"
    slash = rest.find("/")
    if slash != -1:
        rest, path = rest[:slash], rest[slash:]

    host = rest
    port = DEFAULT_PORTS.get(protocol, 0)
    colon = rest.find(":")
    if colon != -1:
        host = rest[:colon]
        port = int(rest[colon + 1:])

    return {
        "protocol": protocol,
        "host": host,
        "port": port,
        "path": path,
        "query": query,
        "fragment": fragment,
    }


def main():
    assert parse_url("https://example.com/path?q=1#frag") == {
        "protocol": "https", "host": "example.com", "port": 443,
        "path": "/path", "query": "q=1", "fragment": "frag",
    }
    assert parse_url("http://example.com") == {
        "protocol": "http", "host": "example.com", "port": 80,
        "path": "/", "query": "", "fragment": "",
    }
    assert parse_url("HTTP://example.com/x") == {
        "protocol": "http", "host": "example.com", "port": 80,
        "path": "/x", "query": "", "fragment": "",
    }
    assert parse_url("http://example.com:8080/a/b?x=1&y=2#top") == {
        "protocol": "http", "host": "example.com", "port": 8080,
        "path": "/a/b", "query": "x=1&y=2", "fragment": "top",
    }
    assert parse_url("https://host?q=1")["path"] == "/"
    assert parse_url("https://host?q=1")["query"] == "q=1"
    assert parse_url("https://host/#f")["fragment"] == "f"
    assert parse_url("https://example.com:8443")["port"] == 8443
    assert parse_url("ftp://files.example.com/pub")["port"] == 0
    assert parse_url("ftp://files.example.com/pub")["path"] == "/pub"

    try:
        parse_url("example.com/no-scheme")
        raise AssertionError("expected ValueError for missing scheme")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
