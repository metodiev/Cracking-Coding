/** Task 096 - Simple URL parser (extract protocol, host). (Tier: Beginner, CTCI Ch 7)

 * Protocol is lowercased; absent path becomes "/"; port defaults to 80 (http) or
 * 443 (https), 0 for other schemes; absent query/fragment are "".
 */
"use strict";

const DEFAULT_PORTS = { http: 80, https: 443 };

function parseUrl(url) {
  const sep = url.indexOf("://");
  if (sep <= 0) throw new Error("URL must start with <scheme>://");
  const protocol = url.slice(0, sep).toLowerCase();
  let rest = url.slice(sep + 3);

  let fragment = "";
  const hash = rest.indexOf("#");
  if (hash !== -1) {
    fragment = rest.slice(hash + 1);
    rest = rest.slice(0, hash);
  }

  let query = "";
  const q = rest.indexOf("?");
  if (q !== -1) {
    query = rest.slice(q + 1);
    rest = rest.slice(0, q);
  }

  let path = "/";
  const slash = rest.indexOf("/");
  if (slash !== -1) {
    path = rest.slice(slash);
    rest = rest.slice(0, slash);
  }

  let host = rest;
  let port = protocol in DEFAULT_PORTS ? DEFAULT_PORTS[protocol] : 0;
  const colon = rest.indexOf(":");
  if (colon !== -1) {
    host = rest.slice(0, colon);
    port = Number(rest.slice(colon + 1));
  }
  return { protocol, host, port, path, query, fragment };
}

module.exports = { parseUrl };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(parseUrl("https://example.com/path?q=1#frag"),
    { protocol: "https", host: "example.com", port: 443, path: "/path", query: "q=1", fragment: "frag" });
  assert.deepStrictEqual(parseUrl("http://example.com"),
    { protocol: "http", host: "example.com", port: 80, path: "/", query: "", fragment: "" });
  assert.deepStrictEqual(parseUrl("HTTP://example.com/x"),
    { protocol: "http", host: "example.com", port: 80, path: "/x", query: "", fragment: "" });
  assert.deepStrictEqual(parseUrl("http://example.com:8080/a/b?x=1&y=2#top"),
    { protocol: "http", host: "example.com", port: 8080, path: "/a/b", query: "x=1&y=2", fragment: "top" });
  assert.deepStrictEqual(parseUrl("https://host?q=1"),
    { protocol: "https", host: "host", port: 443, path: "/", query: "q=1", fragment: "" });
  assert.deepStrictEqual(parseUrl("https://host/#f"),
    { protocol: "https", host: "host", port: 443, path: "/", query: "", fragment: "f" });
  assert.deepStrictEqual(parseUrl("https://example.com:8443"),
    { protocol: "https", host: "example.com", port: 8443, path: "/", query: "", fragment: "" });
  assert.deepStrictEqual(parseUrl("ftp://files.example.com/pub"),
    { protocol: "ftp", host: "files.example.com", port: 0, path: "/pub", query: "", fragment: "" });
  assert.throws(() => parseUrl("example.com/no-scheme"), /scheme/);
  console.log("PASS");
}
