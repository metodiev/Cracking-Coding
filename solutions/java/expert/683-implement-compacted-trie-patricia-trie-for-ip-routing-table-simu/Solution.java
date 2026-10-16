/** Task 683 - Implement compacted trie (Patricia trie) for IP routing table simulation. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * A compressed binary trie over 32-bit prefixes: every node carries the bit index it tests, so the bits
 * between a node and its child are implicit and a lookup walks at most 33 nodes instead of 32 levels per
 * bit. A node may also hold the route whose length equals its bit index, which is exactly a longest-prefix
 * match table: descending from the root, each node whose own prefix contains the query updates the answer,
 * and because prefix length equals the node's bit index and increases down the tree, the last match is the
 * longest one. Insertion forks the trie at the first differing bit (and lifts the existing subtree under a
 * new ancestor when the new prefix is the shorter one); deletion clears the route and splices the node out
 * when it is left with at most one child.
 *
 * Invariants checked after every update: a node's prefix length equals its bit index, bit indices strictly
 * increase along every root-to-leaf path, the keys of a subtree agree with the node's representative on all
 * bits above the node's bit index, and the node count stays within twice the number of routes. Correctness
 * is measured against a brute-force linear scan of the route table: every route is found by an exact
 * lookup with its own value, and the longest-prefix answer for hundreds of addresses (including a
 * randomised insert/delete stream) must match the scan in both prefix and next hop.
 */
import java.util.*;

final class Prefix {
    final int addr;              // network address, host bits cleared
    final int len;               // prefix length in bits, 0..32

    Prefix(int addr, int len) {
        this.len = len;
        this.addr = mask(addr, len);
    }

    static int mask(int addr, int len) {
        if (len <= 0) return 0;
        return len >= 32 ? addr : addr & (0xFFFFFFFF << (32 - len));
    }

    static Prefix parse(String cidr) {
        int slash = cidr.indexOf('/');
        if (slash <= 0 || slash == cidr.length() - 1) return null;
        String[] octets = cidr.substring(0, slash).split("\\.", -1);
        if (octets.length != 4) return null;
        int value = 0;
        for (String octet : octets) {
            if (octet.isEmpty() || octet.length() > 3) return null;
            for (char c : octet.toCharArray()) if (c < '0' || c > '9') return null;
            int part = Integer.parseInt(octet);
            if (part > 255) return null;
            value = (value << 8) | part;
        }
        int len;
        try {
            len = Integer.parseInt(cidr.substring(slash + 1));
        } catch (NumberFormatException e) {
            return null;
        }
        if (len < 0 || len > 32) return null;
        return new Prefix(value, len);
    }

    boolean contains(int ip) {
        return len == 0 || (ip & mask(-1, len)) == addr;
    }

    String key() { return toString(); }

    public String toString() {
        return ((addr >>> 24) & 0xFF) + "." + ((addr >>> 16) & 0xFF) + "." + ((addr >>> 8) & 0xFF) + "."
                + (addr & 0xFF) + "/" + len;
    }
}

final class PatriciaNode {
    final int bit;               // the bit index this node tests
    final int repr;              // representative address: the subtree agrees with it above |bit|
    Prefix prefix;               // route of length |bit| stored here, or null
    int value;
    PatriciaNode left, right;

    PatriciaNode(int bit, Prefix prefix, int value, int repr) {
        this.bit = bit;
        this.prefix = prefix;
        this.value = value;
        this.repr = repr;
    }

    boolean isLeaf() { return left == null && right == null; }
}

final class PatriciaTrie {
    PatriciaNode root;
    final Map<String, Integer> routes = new TreeMap<>();     // canonical prefix -> next hop (the oracle)
    int lookups, hops;

    static int bitAt(int addr, int index) { return (addr >>> (31 - index)) & 1; }

    /** Index of the most significant bit where a and b differ, or 32 when they are equal. */
    static int firstDifferingBit(int a, int b) {
        int diff = a ^ b;
        return diff == 0 ? 32 : Integer.numberOfLeadingZeros(diff);
    }

    void insert(Prefix p, int value) {
        routes.put(p.key(), value);
        if (root == null) root = new PatriciaNode(p.len, p, value, p.addr);
        else root = insert(root, p, value);
    }

    static PatriciaNode insert(PatriciaNode node, Prefix p, int value) {
        int diff = firstDifferingBit(node.repr, p.addr);
        if (diff >= node.bit) {                                  // p follows this subtree as far as node.bit
            if (p.len == node.bit) {
                node.prefix = p;
                node.value = value;
                return node;
            }
            if (p.len < node.bit) {                              // p is an ancestor of the whole subtree
                PatriciaNode lift = new PatriciaNode(p.len, p, value, node.repr);
                if (bitAt(node.repr, p.len) == 1) lift.right = node;
                else lift.left = node;
                return lift;
            }
            boolean branch = bitAt(p.addr, node.bit) == 1;        // p.len > node.bit: descend
            PatriciaNode child = branch ? node.right : node.left;
            if (child == null) {
                PatriciaNode leaf = new PatriciaNode(p.len, p, value, p.addr);
                if (branch) node.right = leaf;
                else node.left = leaf;
                return node;
            }
            PatriciaNode updated = insert(child, p, value);
            if (branch) node.right = updated;
            else node.left = updated;
            return node;
        }
        if (p.len <= diff) {                                     // p is an ancestor, the subtree hangs below
            PatriciaNode lift = new PatriciaNode(p.len, p, value, node.repr);
            if (bitAt(node.repr, p.len) == 1) lift.right = node;
            else lift.left = node;
            return lift;
        }
        PatriciaNode fork = new PatriciaNode(diff, null, 0, node.repr);
        PatriciaNode leaf = new PatriciaNode(p.len, p, value, p.addr);
        if (bitAt(node.repr, diff) == 1) {
            fork.right = node;
            fork.left = leaf;
        } else {
            fork.left = node;
            fork.right = leaf;
        }
        return fork;
    }

    boolean delete(Prefix p) {
        if (!routes.containsKey(p.key())) return false;
        routes.remove(p.key());
        root = delete(root, p);
        return true;
    }

    static PatriciaNode delete(PatriciaNode node, Prefix p) {
        if (node == null) return null;
        if (node.bit == p.len && node.prefix != null && node.prefix.addr == p.addr) {
            if (node.left != null && node.right != null) {
                node.prefix = null;                              // becomes a pure fork node
                return node;
            }
            return node.left != null ? node.left : node.right;    // splice out
        }
        if (node.bit >= 32) return node;
        boolean branch = bitAt(p.addr, node.bit) == 1;
        if (branch) node.right = delete(node.right, p);
        else node.left = delete(node.left, p);
        if (node.prefix != null) return node;
        if (node.left == null && node.right == null) return null;      // dangling fork: drop it
        if (node.left == null) return node.right;                      // single child: splice through
        if (node.right == null) return node.left;
        return node;
    }

    /** Longest prefix match, or the pair (null, -1) when nothing matches. */
    Prefix lookup(int ip, int[] valueOut) {
        lookups++;
        PatriciaNode cur = root;
        Prefix best = null;
        int bestValue = -1;
        while (cur != null) {
            hops++;
            if (cur.prefix != null && cur.prefix.contains(ip)) {
                best = cur.prefix;
                bestValue = cur.value;
            }
            if (cur.bit >= 32) break;
            cur = bitAt(ip, cur.bit) == 1 ? cur.right : cur.left;
        }
        valueOut[0] = bestValue;
        return best;
    }

    /** Exact route lookup by canonical prefix. */
    Integer lookupExact(Prefix p) {
        PatriciaNode cur = root;
        while (cur != null) {
            if (cur.bit == p.len && cur.prefix != null && cur.prefix.addr == p.addr) return cur.value;
            if (cur.bit >= 32) break;
            cur = bitAt(p.addr, cur.bit) == 1 ? cur.right : cur.left;
        }
        return null;
    }

    /** Brute force: scan the whole table and keep the longest matching prefix. */
    Prefix bruteForce(int ip, int[] valueOut) {
        Prefix best = null;
        for (Map.Entry<String, Integer> entry : routes.entrySet()) {
            Prefix candidate = Prefix.parse(entry.getKey());
            if (!candidate.contains(ip)) continue;
            if (best == null || candidate.len > best.len) {
                best = candidate;
                valueOut[0] = entry.getValue();
            }
        }
        return best;
    }

    int countNodes() { return countNodes(root); }

    static int countNodes(PatriciaNode node) { return node == null ? 0 : 1 + countNodes(node.left) + countNodes(node.right); }

    /** Returns a description of the first structural violation, or null when the trie is well formed. */
    String checkStructure() {
        return checkStructure(root, -1);
    }

    static String checkStructure(PatriciaNode node, int parentBit) {
        if (node == null) return null;
        if (node.bit <= parentBit) return "bit index " + node.bit + " does not increase past " + parentBit;
        if (node.prefix != null && node.prefix.len != node.bit)
            return "prefix " + node.prefix + " sits on a node testing bit " + node.bit;
        if (node.prefix == null && node.isLeaf()) return "empty leaf at bit " + node.bit;
        if (node.bit > 0) {
            int mask = Prefix.mask(-1, node.bit);
            for (Prefix p : subtreePrefixes(node))
                if ((p.addr & mask) != (node.repr & mask))
                    return "prefix " + p + " breaks the shared prefix of node " + node.bit;
        }
        String left = checkStructure(node.left, node.bit), right = checkStructure(node.right, node.bit);
        return left != null ? left : right;
    }

    static List<Prefix> subtreePrefixes(PatriciaNode node) {
        List<Prefix> out = new ArrayList<>();
        collect(node, out);
        return out;
    }

    static void collect(PatriciaNode node, List<Prefix> out) {
        if (node == null) return;
        if (node.prefix != null) out.add(node.prefix);
        collect(node.left, out);
        collect(node.right, out);
    }
}

public class Solution {
    static long state = 0x1B0000L;

    static long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static int ip(String dotted) {
        String[] parts = dotted.split("\\.");
        int value = 0;
        for (String part : parts) value = (value << 8) | Integer.parseInt(part);
        return value;
    }

    /** Trie answer must equal the brute-force scan, in prefix and in next hop. */
    static void checkLookup(PatriciaTrie trie, int address, String label) {
        int[] trieValue = {-1}, bruteValue = {-1};
        Prefix triePrefix = trie.lookup(address, trieValue);
        Prefix brutePrefix = trie.bruteForce(address, bruteValue);
        String expected = brutePrefix == null ? "none" : brutePrefix + "->" + bruteValue[0];
        String got = triePrefix == null ? "none" : triePrefix + "->" + trieValue[0];
        if (!expected.equals(got)) throw new AssertionError("FAILED: " + label + " at " + address + ": " + got + " vs " + expected);
    }

    public static void main(String[] args) {
        check(Prefix.parse("10.1.2.3/8").toString().equals("10.0.0.0/8"),
                "host bits are cleared when a prefix is parsed: " + Prefix.parse("10.1.2.3/8"));
        check(Prefix.parse("0.0.0.0/0").toString().equals("0.0.0.0/0"), "the default route parses");
        check(Prefix.parse("192.168.1.255/32").toString().equals("192.168.1.255/32"), "a host route keeps all bits");
        check(Prefix.parse("10.0.0.0/33") == null && Prefix.parse("10.0.0/8") == null
                        && Prefix.parse("10.0.0.256/8") == null && Prefix.parse("10.0.0.0/x") == null
                        && Prefix.parse("10.0.0.0") == null,
                "malformed prefixes are rejected");
        check(new Prefix(ip("10.1.2.3"), 8).contains(ip("10.255.0.1"))
                        && !new Prefix(ip("10.1.2.3"), 8).contains(ip("11.0.0.1")),
                "contains() respects the mask");

        PatriciaTrie trie = new PatriciaTrie();
        trie.insert(Prefix.parse("10.0.0.0/8"), 1);
        trie.insert(Prefix.parse("10.1.0.0/16"), 2);
        trie.insert(Prefix.parse("10.1.2.0/24"), 3);
        trie.insert(Prefix.parse("192.168.0.0/16"), 4);
        int[] value = {-1};
        check(trie.lookup(ip("10.1.2.77"), value) != null && value[0] == 3, "the /24 wins inside its range");
        check(trie.lookup(ip("10.1.9.9"), value) != null && value[0] == 2, "the /16 wins between the /16 and /24");
        check(trie.lookup(ip("10.200.0.1"), value) != null && value[0] == 1, "the /8 wins outside the /16");
        check(trie.lookup(ip("11.0.0.1"), value) == null && value[0] == -1, "no route matches 11.0.0.0/8");
        check(trie.lookup(ip("192.168.44.1"), value) != null && value[0] == 4, "the unrelated /16 still works");
        check(trie.checkStructure() == null && trie.countNodes() <= 2 * trie.routes.size(),
                "structure is well formed with " + trie.countNodes() + " nodes for " + trie.routes.size() + " routes");
        check(trie.lookupExact(Prefix.parse("10.1.0.0/16")) == 2 && trie.lookupExact(Prefix.parse("10.0.0.0/9")) == null,
                "exact lookup finds stored routes only");

        trie.insert(Prefix.parse("0.0.0.0/0"), 99);
        check(trie.lookup(ip("8.8.8.8"), value) != null && value[0] == 99, "the default route catches the rest");
        check(trie.lookup(ip("10.1.2.1"), value) != null && value[0] == 3, "the default does not beat a longer match");
        check(trie.checkStructure() == null, "the default route keeps the trie well formed");
        checkLookup(trie, ip("10.1.2.1"), "default-plus-specific");

        int before = trie.countNodes();
        trie.insert(Prefix.parse("10.1.0.0/16"), 42);
        check(trie.lookupExact(Prefix.parse("10.1.0.0/16")) == 42 && trie.countNodes() == before,
                "re-inserting a route only updates its next hop");
        trie.insert(Prefix.parse("10.1.0.0/16"), 2);

        for (int i = 0; i < 40; i++) {
            int addr = (int) (nextLong() >>> 32);
            trie.insert(new Prefix(addr, 8 + (int) ((nextLong() >>> 40) % 25)), (int) ((nextLong() >>> 50) % 900) + 100);
        }
        check(trie.checkStructure() == null, "structure holds after 40 random routes");
        check(trie.countNodes() <= 2 * trie.routes.size(), "node count stays compact: " + trie.countNodes()
                + " nodes for " + trie.routes.size() + " routes");
        for (Map.Entry<String, Integer> entry : trie.routes.entrySet()) {
            Prefix stored = Prefix.parse(entry.getKey());
            check(trie.lookupExact(stored) != null && trie.lookupExact(stored).equals(entry.getValue()),
                    "every stored route is findable exactly: " + stored);
        }
        for (int i = 0; i < 300; i++) checkLookup(trie, (int) (nextLong() >>> 32), "random address " + i);
        for (int i = 0; i < 40; i++) checkLookup(trie, ip("10.1." + i + "." + (i * 3 % 256)), "near the nested routes");

        PatriciaTrie stream = new PatriciaTrie();
        stream.insert(Prefix.parse("172.16.0.0/12"), 7);
        stream.insert(Prefix.parse("172.16.4.0/22"), 8);
        check(stream.lookupExact(Prefix.parse("172.16.4.0/22")) == 8, "the nested prefix was inserted");
        check(stream.delete(Prefix.parse("172.16.4.0/22")), "deleting an existing route reports success");
        check(stream.lookupExact(Prefix.parse("172.16.4.0/22")) == null, "the deleted route is gone");
        check(stream.lookup(ip("172.16.4.9"), value) != null && value[0] == 7, "the covering route takes over");
        check(!stream.delete(Prefix.parse("172.16.4.0/22")) && !stream.delete(Prefix.parse("9.9.9.0/24")),
                "deleting a missing route reports failure");
        check(stream.checkStructure() == null, "the trie stays well formed after deletion");

        int operations = 0, structureErrors = 0, lookupMismatch = 0;
        PatriciaTrie random = new PatriciaTrie();
        List<Prefix> pool = new ArrayList<>();
        for (int i = 0; i < 120; i++) {
            long r = nextLong();
            boolean remove = !pool.isEmpty() && (r & 3) == 0;
            if (remove) {
                Prefix victim = pool.remove((int) ((r >>> 5) % pool.size()));
                random.delete(victim);
            } else {
                Prefix p = new Prefix((int) ((r >>> 7) % 200) * 0x01010101, 8 + (int) ((r >>> 11) % 21));
                random.insert(p, (int) ((r >>> 17) % 500) + 1);
                if (!pool.contains(p)) pool.add(p);
            }
            operations++;
            if (random.checkStructure() != null) structureErrors++;
            if (operations % 10 == 0) {
                for (int k = 0; k < 25; k++) {
                    int address = (int) (nextLong() >>> 32);
                    int[] a = {-1}, b = {-1};
                    Prefix trieAnswer = random.lookup(address, a);
                    Prefix bruteAnswer = random.bruteForce(address, b);
                    String left = trieAnswer == null ? "none" : trieAnswer + ":" + a[0];
                    String right = bruteAnswer == null ? "none" : bruteAnswer + ":" + b[0];
                    if (!left.equals(right)) lookupMismatch++;
                }
            }
        }
        check(operations == 120 && structureErrors == 0, "120 insert/delete operations kept the invariants");
        check(lookupMismatch == 0, "25 lookups every 10 operations matched the brute-force table (" + lookupMismatch + " mismatches)");
        check(random.countNodes() <= 2 * random.routes.size() + 1, "node count still compact: "
                + random.countNodes() + " nodes for " + random.routes.size() + " routes");

        PatriciaTrie wiped = new PatriciaTrie();
        wiped.insert(Prefix.parse("10.0.0.0/8"), 1);
        wiped.insert(Prefix.parse("10.0.0.0/16"), 2);
        wiped.insert(Prefix.parse("10.0.0.0/24"), 3);
        check(wiped.countNodes() == 3, "nested prefixes of one address chain into three nodes, got " + wiped.countNodes());
        wiped.delete(Prefix.parse("10.0.0.0/8"));
        check(wiped.checkStructure() == null && wiped.lookupExact(Prefix.parse("10.0.0.0/16")) == 2,
                "deleting the outermost prefix keeps the rest reachable");
        check(wiped.lookup(ip("10.0.0.9"), value) != null && value[0] == 3, "longest match still applies");
        wiped.delete(Prefix.parse("10.0.0.0/16"));
        wiped.delete(Prefix.parse("10.0.0.0/24"));
        check(wiped.countNodes() == 0 && wiped.lookup(ip("10.0.0.1"), value) == null,
                "an empty table has no nodes and no matches");

        PatriciaTrie again = new PatriciaTrie();
        PatriciaTrie first = new PatriciaTrie();
        for (int i = 0; i < 30; i++) {
            Prefix p = new Prefix((int) (nextLong() >>> 32), 8 + (int) ((nextLong() >>> 40) % 17));
            first.insert(p, i + 1);
            again.insert(p, i + 1);
        }
        check(first.countNodes() == again.countNodes(), "the same inserts build the same trie size");
        for (int i = 0; i < 50; i++) {
            int address = (int) (nextLong() >>> 32);
            int[] a = {-1}, b = {-1};
            Prefix p1 = first.lookup(address, a), p2 = again.lookup(address, b);
            check((p1 == null) == (p2 == null) && a[0] == b[0], "lookups are deterministic");
        }
        check(first.lookups > 0 && first.hops <= 33 * first.lookups, "lookups walk at most 33 nodes: "
                + first.hops + " hops for " + first.lookups + " lookups");

        System.out.println("nested routes : 10.0.0.0/8 -> 1, /16 -> 2, /24 -> 3 resolved by longest match");
        System.out.println("structure     : " + trie.countNodes() + " nodes for " + trie.routes.size()
                + " routes, all invariants hold; worst-case hops bounded by 33");
        System.out.println("random stream : " + operations + " insert/delete operations, 0 structure errors, "
                + lookupMismatch + " lookup mismatches");
        System.out.println("PASS");
    }
}
