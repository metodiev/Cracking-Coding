/** Task 684 - Implement longest-prefix-match lookup optimized (binary trie). (Tier: Expert, CTCI Beyond - Networking, OS & Real-Time)
 *
 * A routing table is stored in a binary (bitwise) trie over the 32 bits of an IPv4 address.
 * Every node owns one bit position and carries the route whose prefix ends there. A lookup
 * address determines exactly one root path, so longest-prefix match is a single descent that
 * remembers the deepest node holding a route - no backtracking, O(W) with W = 32.
 *
 * Two optimisations are implemented and cross-checked against the plain descent:
 *   * a first-byte fast path (stride 8): 256 cached entry points, each recording the node
 *     reached after the top 8 bits plus the best route seen within those 8 bits, so a lookup
 *     starts 8 levels down;
 *   * `nodeCount()`, which demonstrates the reason tries replace linear scans: shared
 *     prefixes occupy one chain, so the structure is bounded by 1 + sum(prefix lengths)
 *     regardless of how the address space is populated.
 *
 * What is simplified: real routers walk pipelined hardware stages or TCAM banks; here the
 * trie is a pointer structure with one port per node. The invariant verified is exact
 * agreement with a brute-force linear scan of every installed prefix (deepest prefix wins,
 * /0 default route and /32 host routes included) for hand-built and pseudo-random tables.
 */
import java.util.*;

/** One installed route: network address, prefix length and egress port. */
class Route {
    final int network;
    final int length;
    final int port;

    Route(int network, int length, int port) {
        this.network = network;
        this.length = length;
        this.port = port;
    }
}

/** Binary longest-prefix-match trie with an 8-bit-stride fast path. */
class BinaryTrie {
    static final class Node {
        Node zero;
        Node one;
        boolean hasRoute;
        int port;
    }

    private final Node root = new Node();
    private int nodeCount = 1;
    private int routeCount;
    private boolean fastPathValid;
    private final Node[] fastEntry = new Node[256];
    private final int[] fastPort = new int[256];

    static int mask(int length) {
        return length == 0 ? 0 : -1 << (32 - length);
    }

    void insert(int network, int length, int port) {
        if (length < 0 || length > 32) {
            throw new IllegalArgumentException("prefix length out of range: " + length);
        }
        Node cur = root;
        for (int i = 0; i < length; i++) {
            int bit = (network >>> (31 - i)) & 1;
            if (bit == 0) {
                if (cur.zero == null) {
                    cur.zero = new Node();
                    nodeCount++;
                }
                cur = cur.zero;
            } else {
                if (cur.one == null) {
                    cur.one = new Node();
                    nodeCount++;
                }
                cur = cur.one;
            }
        }
        if (!cur.hasRoute) routeCount++;
        cur.hasRoute = true;
        cur.port = port;
        fastPathValid = false;
    }

    boolean hasRoute(int network, int length) {
        Node cur = root;
        for (int i = 0; i < length; i++) {
            cur = ((network >>> (31 - i)) & 1) == 0 ? cur.zero : cur.one;
            if (cur == null) return false;
        }
        return cur.hasRoute;
    }

    int lookup(int address) {
        Node cur = root;
        int best = cur.hasRoute ? cur.port : -1;
        for (int i = 0; i < 32; i++) {
            cur = ((address >>> (31 - i)) & 1) == 0 ? cur.zero : cur.one;
            if (cur == null) break;
            if (cur.hasRoute) best = cur.port;
        }
        return best;
    }

    private void buildFastPath() {
        Arrays.fill(fastEntry, null);
        Arrays.fill(fastPort, -1);
        for (int firstByte = 0; firstByte < 256; firstByte++) {
            Node cur = root;
            int best = root.hasRoute ? root.port : -1;
            boolean complete = true;
            for (int i = 0; i < 8; i++) {
                cur = ((firstByte >>> (7 - i)) & 1) == 0 ? cur.zero : cur.one;
                if (cur == null) {
                    // The address diverges from the trie inside the top byte: no node beyond
                    // the divergence can match, so the cached value is already the answer.
                    complete = false;
                    break;
                }
                if (cur.hasRoute) best = cur.port;
            }
            fastEntry[firstByte] = complete ? cur : null;
            fastPort[firstByte] = best;
        }
        fastPathValid = true;
    }

    int lookupFast(int address) {
        if (!fastPathValid) buildFastPath();
        int firstByte = (address >>> 24) & 0xFF;
        Node cur = fastEntry[firstByte];
        int best = fastPort[firstByte];
        for (int i = 8; cur != null && i < 32; i++) {
            cur = ((address >>> (31 - i)) & 1) == 0 ? cur.zero : cur.one;
            if (cur == null) break;
            if (cur.hasRoute) best = cur.port;
        }
        return best;
    }

    int nodeCount() {
        return nodeCount;
    }

    int routeCount() {
        return routeCount;
    }
}

public class Solution {
    /** splitmix64: deterministic, fixed seed, no java.util.Random. */
    private static long prngState = 0x2545F4914F6CDD1DL;

    static long nextLong() {
        prngState += 0x9E3779B97F4A7C15L;
        long z = prngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), (long) bound);
    }

    static int ip(String dotted) {
        String[] parts = dotted.split("\\.");
        int out = 0;
        for (String part : parts) out = (out << 8) | Integer.parseInt(part);
        return out;
    }

    static String ipText(int address) {
        return ((address >>> 24) & 0xFF) + "." + ((address >>> 16) & 0xFF) + "."
                + ((address >>> 8) & 0xFF) + "." + (address & 0xFF);
    }

    /**
     * Brute-force oracle: deepest matching prefix, or -1 when nothing matches. Equal prefixes
     * are compared with >= so a re-inserted prefix overwrites the earlier port, exactly like
     * the trie does.
     */
    static int naiveLookup(List<Route> routes, int address) {
        int bestLength = -1;
        int bestPort = -1;
        for (Route r : routes) {
            if ((address & BinaryTrie.mask(r.length)) == (r.network & BinaryTrie.mask(r.length))
                    && r.length >= bestLength) {
                bestLength = r.length;
                bestPort = r.port;
            }
        }
        return bestPort;
    }

    static Route route(String dotted, int length, int port) {
        return new Route(ip(dotted) & BinaryTrie.mask(length), length, port);
    }

    static void installAll(BinaryTrie trie, List<Route> routes) {
        for (Route r : routes) trie.insert(r.network, r.length, r.port);
    }

    public static void main(String[] args) {
        List<Route> table = new ArrayList<>(List.of(
                route("0.0.0.0", 0, 1),
                route("10.0.0.0", 8, 2),
                route("10.1.0.0", 16, 3),
                route("10.1.2.0", 24, 4),
                route("192.168.0.0", 16, 5),
                route("192.168.7.7", 32, 6)));
        BinaryTrie trie = new BinaryTrie();
        installAll(trie, table);

        check(trie.lookup(ip("10.1.2.9")) == 4, "longest prefix wins (/24)");
        check(trie.lookup(ip("10.1.3.9")) == 3, "next shorter prefix wins (/16)");
        check(trie.lookup(ip("10.9.9.9")) == 2, "back off to /8");
        check(trie.lookup(ip("8.8.8.8")) == 1, "default /0 route");
        check(trie.lookup(ip("192.168.7.7")) == 6, "host route /32 beats /16");
        check(trie.lookup(ip("192.168.7.8")) == 5, "neighbour address falls back to /16");

        check(trie.lookupFast(ip("10.1.2.9")) == 4 && trie.lookupFast(ip("192.168.7.7")) == 6
                && trie.lookupFast(ip("8.8.8.8")) == 1, "fast path agrees on spot values");

        BinaryTrie empty = new BinaryTrie();
        check(empty.lookup(ip("10.0.0.1")) == -1 && empty.lookupFast(ip("1.2.3.4")) == -1
                && empty.nodeCount() == 1, "empty table misses and keeps only the root");

        BinaryTrie noDefault = new BinaryTrie();
        noDefault.insert(ip("172.16.0.0"), 12, 9);
        check(noDefault.lookup(ip("172.16.5.5")) == 9 && noDefault.lookup(ip("172.32.5.5")) == -1,
                "miss without a default route");

        BinaryTrie overwrite = new BinaryTrie();
        overwrite.insert(ip("10.0.0.0"), 8, 2);
        overwrite.insert(ip("10.0.0.0"), 8, 7);
        check(overwrite.lookup(ip("10.1.1.1")) == 7 && overwrite.routeCount() == 1,
                "re-insert replaces the port in place");

        check(trie.nodeCount() <= 1 + 0 + 8 + 16 + 24 + 16 + 32,
                "shared prefixes occupy one chain each");
        boolean rejected = false;
        try {
            trie.insert(0, 33, 1);
        } catch (IllegalArgumentException e) {
            rejected = true;
        }
        check(rejected, "prefix length 33 rejected");

        for (Route r : table) {
            check(trie.hasRoute(r.network, r.length), "installed prefix present " + ipText(r.network));
        }

        for (int firstByte = 0; firstByte < 256; firstByte++) {
            int probe = firstByte << 24;
            check(trie.lookupFast(probe) == naiveLookup(table, probe), "fast path byte " + firstByte);
        }

        int mismatches = 0;
        for (int trial = 0; trial < 6; trial++) {
            List<Route> random = new ArrayList<>();
            BinaryTrie rt = new BinaryTrie();
            for (int i = 0; i < 60; i++) {
                int length = new int[]{0, 4, 8, 12, 16, 20, 24, 28, 32}[nextInt(9)];
                int network = (int) nextLong() & BinaryTrie.mask(length);
                int port = 1 + nextInt(200);
                Route r = new Route(network, length, port);
                random.add(r);
                rt.insert(r.network, r.length, r.port);
            }
            for (int i = 0; i < 200; i++) {
                int address = (int) nextLong();
                int expected = naiveLookup(random, address);
                if (rt.lookup(address) != expected) mismatches++;
                if (rt.lookupFast(address) != expected) mismatches++;
            }
        }
        check(mismatches == 0, "600 pseudo-random prefixes vs linear scan oracle");

        int dense = 1 << 20 | 1 << 10;
        check(mismatches == 0 && trie.lookup(dense) == naiveLookup(table, dense), "invariant holds");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
