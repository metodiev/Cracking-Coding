/** Task 470 - Implement consistent hashing ring (virtual nodes). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Consistent-hashing ring with VNODES = 64 virtual nodes per physical node. Ring
 * positions come from the FIXED LCG hash (documented, deterministic): fold over
 * bytes with h = (h*75 + 74 + ch) mod 65537, then two LCG avalanche steps
 * h = (h*75 + 74) mod 65537 (the plain fold alone clusters badly - measured shares
 * 0.02..0.48; the avalanched hash balances). Points are sorted by (position, node
 * label); a key hashes with the same function and is owned by the first vnode
 * clockwise at or after its position (wrap-around).
 *
 * Checks on 4 physical nodes x 64 vnodes and 4096 keys:
 *   determinism (identical rebuild), bisect lookup equals a brute-force clockwise
 *   scan on a key sample, balance within the documented [0.18, 0.32] band (measured
 *   0.220..0.270), adding "node-e" moves exactly 716 keys (0.175 <= 0.30 documented)
 *   with every moved key going to "node-e", and removing "node-b" moves exactly its
 *   1034 keys (0.252 <= 0.30) with no other key changing owner.
 */
import java.util.*;

class ConsistentHashRing {
    static final int VNODES = 64;
    static final int MODULUS = 65537;

    static class Point {
        final long pos;
        final String node;
        final int idx;

        Point(long pos, String node, int idx) {
            this.pos = pos;
            this.node = node;
            this.idx = idx;
        }
    }

    private final Point[] points;

    static int lcgHash(String text) {
        int h = 0;
        for (int i = 0; i < text.length(); i++) h = (h * 75 + 74 + text.charAt(i)) % MODULUS;
        h = (h * 75 + 74) % MODULUS;
        h = (h * 75 + 74) % MODULUS;
        return h;
    }

    ConsistentHashRing(String[] nodes) {
        List<Point> list = new ArrayList<>();
        for (String node : nodes)
            for (int i = 0; i < VNODES; i++) list.add(new Point(lcgHash(node + "#" + i), node, i));
        list.sort((a, b) -> {
            if (a.pos != b.pos) return Long.compare(a.pos, b.pos);
            int cmp = a.node.compareTo(b.node);
            return cmp != 0 ? cmp : Integer.compare(a.idx, b.idx);
        });
        points = list.toArray(new Point[0]);
    }

    int distinctPositions() {
        Set<Long> seen = new HashSet<>();
        for (Point point : points) seen.add(point.pos);
        return seen.size();
    }

    String owner(String key) {
        long pos = lcgHash(key);
        int lo = 0, hi = points.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (points[mid].pos < pos) lo = mid + 1;
            else hi = mid;
        }
        if (lo == points.length) lo = 0; // wrap around
        return points[lo].node;
    }

    String ownerBrute(String key) {
        long pos = lcgHash(key);
        Point best = null;
        for (Point point : points) {
            if (point.pos >= pos
                    && (best == null || point.pos < best.pos
                        || (point.pos == best.pos && point.node.compareTo(best.node) < 0)
                        || (point.pos == best.pos && point.node.equals(best.node) && point.idx < best.idx))) {
                best = point;
            }
        }
        if (best == null) best = points[0];
        return best.node;
    }
}

public class Solution {
    public static void main(String[] args) {
        String[] nodes = {"node-a", "node-b", "node-c", "node-d"};
        String[] keys = new String[4096];
        for (int i = 0; i < keys.length; i++) keys[i] = "k" + i;

        ConsistentHashRing ring = new ConsistentHashRing(nodes);
        Map<String, String> first = assign(ring, keys);
        check(first.equals(assign(new ConsistentHashRing(nodes), keys)), "deterministic");
        check(ring.distinctPositions() > nodes.length * ConsistentHashRing.VNODES / 2, "positions spread");

        for (int i = 0; i < keys.length; i += 79)
            check(ring.owner(keys[i]).equals(ring.ownerBrute(keys[i])), "bisect equals brute at " + i);

        Map<String, Integer> shares = new HashMap<>();
        for (String owner : first.values()) shares.merge(owner, 1, Integer::sum);
        for (String node : nodes) {
            double share = shares.get(node) / (double) keys.length;
            check(share >= 0.18 && share <= 0.32, node + " share " + share);
        }

        String[] withE = {"node-a", "node-b", "node-c", "node-d", "node-e"};
        Map<String, String> afterAdd = assign(new ConsistentHashRing(withE), keys);
        List<String> moved = new ArrayList<>();
        for (String key : keys) if (!afterAdd.get(key).equals(first.get(key))) moved.add(key);
        check(moved.size() <= 0.30 * keys.length, "add movement " + moved.size());
        for (String key : moved) check(afterAdd.get(key).equals("node-e"), "moved to node-e");

        String[] withoutB = {"node-a", "node-c", "node-d"};
        Map<String, String> afterRemove = assign(new ConsistentHashRing(withoutB), keys);
        List<String> movedRemove = new ArrayList<>();
        for (String key : keys) if (!afterRemove.get(key).equals(first.get(key))) movedRemove.add(key);
        check(movedRemove.size() <= 0.30 * keys.length, "remove movement " + movedRemove.size());
        for (String key : keys) {
            if (!first.get(key).equals("node-b")) check(afterRemove.get(key).equals(first.get(key)), "untouched " + key);
        }
        for (String key : movedRemove) check(first.get(key).equals("node-b"), "only node-b keys move");

        check(moved.size() == 716 && movedRemove.size() == 1034, "measured counts");
        System.out.println("PASS");
    }

    private static Map<String, String> assign(ConsistentHashRing ring, String[] keys) {
        Map<String, String> owners = new HashMap<>();
        for (String key : keys) owners.put(key, ring.owner(key));
        return owners;
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
