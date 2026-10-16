/** Task 685 - Implement fast IP routing simulation with TCAM-like behavior (bitwise). (Tier: Expert, CTCI Beyond - Networking, OS & Real-Time)
 *
 * A TCAM stores entries as (value, mask) pairs and compares a lookup key against every entry in
 * parallel, returning the port of the highest-priority (lowest index) match. The bitwise kernel
 * of that behaviour is modelled here:
 *
 *   matched(i, key)  := ((key ^ value_i) & mask_i) == 0
 *   matchVector(key) := one bit per entry, bit i set iff entry i matches
 *   lookup(key)      := entry of the lowest set bit (a hardware priority encoder, modelled by
 *                       Long.numberOfTrailingZeros over the 64-entry vector)
 *
 * IP prefixes are installed as ternary rules (network bits kept, host bits masked out), which is
 * why a TCAM routes in one lookup regardless of table size: longest-prefix match becomes pure
 * priority order. The mask may be non-contiguous - something a binary trie cannot express - and
 * that case is exercised explicitly.
 *
 * What is simplified: a real TCAM has one comparator per cell and answers in one cycle; the
 * parallel match is emulated with a 64-bit vector, and update cost is approximated by the number
 * of entries shifted when a rule is inserted or deleted. Verified: the vector lookup returns
 * exactly the same port as a brute-force linear scan of the entries in priority order, packet
 * counters stay consistent with the packets sent, and shadowed rules are detected.
 */
import java.util.*;

/** One ternary rule: match when (key & mask) == value. */
class TcamEntry {
    final int value;
    final int mask;
    final int port;

    TcamEntry(int value, int mask, int port) {
        this.value = value & mask;
        this.mask = mask;
        this.port = port;
    }
}

/** Priority-ordered ternary table with a bitwise (TCAM-style) match vector. */
class Tcam {
    static final int MAX_ENTRIES = 64;

    private final List<TcamEntry> entries = new ArrayList<>();
    private long updates;
    private long shiftedCells;

    int add(int value, int mask, int port) {
        if (entries.size() == MAX_ENTRIES) {
            throw new IllegalStateException("TCAM full (" + MAX_ENTRIES + " entries)");
        }
        TcamEntry candidate = new TcamEntry(value, mask, port);
        for (TcamEntry e : entries) {
            if (e.value == candidate.value && e.mask == candidate.mask) {
                throw new IllegalArgumentException("duplicate ternary rule");
            }
        }
        entries.add(candidate);
        updates++;
        return entries.size() - 1;
    }

    int remove(int index) {
        TcamEntry removed = entries.remove(index);
        // Every entry after the hole is rewritten one cell higher.
        shiftedCells += entries.size() - index;
        updates++;
        return removed.port;
    }

    /** Priority-preserving insert: entries at and after `index` move one cell down. */
    int insertAt(int index, int value, int mask, int port) {
        if (entries.size() == MAX_ENTRIES) {
            throw new IllegalStateException("TCAM full (" + MAX_ENTRIES + " entries)");
        }
        TcamEntry candidate = new TcamEntry(value, mask, port);
        shiftedCells += entries.size() - index;
        entries.add(index, candidate);
        updates++;
        return index;
    }

    long matchVector(int key) {
        long vector = 0L;
        for (int i = 0; i < entries.size(); i++) {
            TcamEntry e = entries.get(i);
            if (((key ^ e.value) & e.mask) == 0) vector |= 1L << i;
        }
        return vector;
    }

    /** Hardware priority encoder: the lowest-index matching entry wins. */
    int lookup(int key) {
        long vector = matchVector(key);
        if (vector == 0L) return -1;
        return entries.get(Long.numberOfTrailingZeros(vector)).port;
    }

    /** Independent reference: linear scan honouring priority order. */
    int lookupByScan(int key) {
        for (TcamEntry e : entries) {
            if (((key ^ e.value) & e.mask) == 0) return e.port;
        }
        return -1;
    }

    int size() {
        return entries.size();
    }

    long updates() {
        return updates;
    }

    long shiftedCells() {
        return shiftedCells;
    }

    TcamEntry entry(int index) {
        return entries.get(index);
    }

    /**
     * Exact coverage test between two ternary rules: `earlier` covers `target` when every bit
     * that `earlier` fixes is fixed identically in `target`, and `target` does not leave any of
     * those bits free. In that case every key matching `target` also matches `earlier`.
     */
    boolean covers(int earlierIndex, int targetIndex) {
        TcamEntry earlier = entries.get(earlierIndex);
        TcamEntry target = entries.get(targetIndex);
        return (target.value & earlier.mask) == earlier.value
                && (target.mask & earlier.mask) == earlier.mask;
    }

    /** A rule is dead when an earlier rule covers its whole match space. */
    boolean isShadowed(int index) {
        for (int j = 0; j < index; j++) {
            if (covers(j, index)) return true;
        }
        return false;
    }
}

public class Solution {
    private static long prngState = 0x0123456789ABCDEFL;

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
        int out = 0;
        for (String part : dotted.split("\\.")) out = (out << 8) | Integer.parseInt(part);
        return out;
    }

    static int prefixMask(int length) {
        return length == 0 ? 0 : -1 << (32 - length);
    }

    /** Builds the routing table used by the forwarding simulation, most specific first. */
    static Tcam routingTable() {
        Tcam tcam = new Tcam();
        tcam.add(ip("10.1.2.0") & prefixMask(24), prefixMask(24), 4);   // priority 0
        tcam.add(ip("10.1.0.0") & prefixMask(16), prefixMask(16), 3);
        tcam.add(ip("10.0.0.0") & prefixMask(8), prefixMask(8), 2);
        tcam.add(ip("192.168.0.0") & prefixMask(16), prefixMask(16), 5);
        tcam.add(0, 0, 1);                                             // default route
        return tcam;
    }

    public static void main(String[] args) {
        Tcam tcam = routingTable();
        check(tcam.size() == 5, "table installed");
        check(tcam.lookup(ip("10.1.2.7")) == 4, "most specific prefix first");
        check(tcam.lookup(ip("10.1.9.7")) == 3, "second priority prefix");
        check(tcam.lookup(ip("10.200.0.1")) == 2, "less specific prefix");
        check(tcam.lookup(ip("192.168.4.4")) == 5, "independent /16");
        check(tcam.lookup(ip("8.8.8.8")) == 1, "default route (mask 0) matches anything");
        check(tcam.lookup(8) == 1 && tcam.lookup(0xFFFFFFFF) == 1, "wildcard extremes");

        Tcam nonContiguous = new Tcam();
        nonContiguous.add(ip("10.0.1.0") & 0xFFFF00FF, 0xFFFF00FF, 11);
        check(nonContiguous.lookup(ip("10.0.99.0")) == 11, "non-contiguous mask matches");
        check(nonContiguous.lookup(ip("10.0.99.5")) == -1, "non-contiguous mask rejects");

        Tcam miss = new Tcam();
        miss.add(ip("172.16.0.0") & prefixMask(12), prefixMask(12), 9);
        check(miss.lookup(ip("172.16.1.1")) == 9 && miss.lookup(ip("172.32.1.1")) == -1,
                "no match returns the drop port");
        check(miss.matchVector(ip("172.32.1.1")) == 0L, "empty match vector on miss");

        boolean duplicateRejected = false;
        try {
            tcam.add(ip("10.0.0.0") & prefixMask(8), prefixMask(8), 42);
        } catch (IllegalArgumentException e) {
            duplicateRejected = true;
        }
        check(duplicateRejected, "duplicate ternary rule rejected");

        Tcam small = new Tcam();
        for (int i = 0; i < Tcam.MAX_ENTRIES; i++) {
            small.add(i << 24, prefixMask(8), i + 1);
        }
        boolean fullRejected = false;
        try {
            small.add(0x7F000000, prefixMask(24), 99);
        } catch (IllegalStateException e) {
            fullRejected = true;
        }
        check(fullRejected, "65th entry rejected (vector is 64 bits wide)");

        int vectorVsScan = 0;
        long matchedEntries = 0;
        for (int i = 0; i < 4000; i++) {
            int key = (int) nextLong();
            int viaVector = tcam.lookup(key);
            int viaScan = tcam.lookupByScan(key);
            if (viaVector != viaScan) vectorVsScan++;
            matchedEntries += Integer.bitCount((int) tcam.matchVector(key));
        }
        check(vectorVsScan == 0, "4000 keys: priority encoder equals linear scan");
        check(matchedEntries >= 4000, "every key matches at least the default route");

        Tcam randomTable = new Tcam();
        List<int[]> installed = new ArrayList<>();
        for (int i = 0; i < 48; i++) {
            int length = new int[]{1, 4, 8, 12, 16, 20, 24, 28, 32}[nextInt(9)];
            int mask = prefixMask(length);
            int network = ((int) nextLong() & mask);
            // Skip duplicates so the TCAM invariant (no two equal rules) is preserved.
            boolean clash = false;
            for (int[] rule : installed) {
                if (rule[0] == (network & mask) && rule[1] == mask) clash = true;
            }
            if (clash) continue;
            installed.add(new int[]{network & mask, mask});
            randomTable.add(network & mask, mask, i + 1);
        }
        int randomMismatch = 0;
        for (int i = 0; i < 3000; i++) {
            int key = (int) nextLong();
            if (randomTable.lookup(key) != randomTable.lookupByScan(key)) randomMismatch++;
        }
        check(randomMismatch == 0, "pseudo-random ternary rules vs linear scan");

        // Forwarding simulation: send packets, track per-port counters and byte totals.
        long[] portPackets = new long[8];
        long[] portBytes = new long[8];
        long packets = 5000, dropped = 0;
        Tcam forwarding = routingTable();
        for (long i = 0; i < packets; i++) {
            int destination;
            switch (nextInt(4)) {
                case 0:  destination = 0x0A010200 | nextInt(256);       break; // 10.1.2.0/24
                case 1:  destination = 0x0A010000 | nextInt(1 << 16);   break; // 10.1.0.0/16
                case 2:  destination = 0x0A000000 | nextInt(1 << 24);   break; // 10.0.0.0/8
                default: destination = 0xC0A80000 | nextInt(1 << 16);   break; // 192.168.0.0/16
            }
            int port = forwarding.lookup(destination);
            if (port < 0) {
                dropped++;
            } else {
                portPackets[port]++;
                portBytes[port] += 64 + nextInt(1400);
            }
        }
        long totalForwarded = 0;
        for (int p = 1; p < 8; p++) totalForwarded += portPackets[p];
        check(totalForwarded + dropped == packets, "forwarded + dropped accounts for all packets");
        check(dropped == 0, "default route means nothing is dropped");
        check(portPackets[2] + portPackets[3] + portPackets[4] + portPackets[5] == packets,
                "10/8 and 192.168/16 traffic lands on their own ports");
        check(portPackets[2] > 0 && portPackets[3] > 0 && portPackets[4] > 0 && portPackets[5] > 0,
                "every routed port carried packets and bytes");

        // Update cost accounting and post-update correctness. Rules are kept most specific first,
        // which is the priority order a router must program into the TCAM.
        Tcam updateCost = new Tcam();
        updateCost.add(ip("10.1.0.0") & prefixMask(16), prefixMask(16), 3);
        updateCost.add(ip("10.0.0.0") & prefixMask(8), prefixMask(8), 2);
        updateCost.add(0, 0, 1);
        check(updateCost.lookup(ip("10.1.2.2")) == 3, "most specific rule wins");
        check(updateCost.lookup(ip("10.9.9.9")) == 2 && updateCost.lookup(ip("9.9.9.9")) == 1,
                "fallbacks behave");
        int removed = updateCost.remove(1);
        check(removed == 2 && updateCost.shiftedCells() == 1, "delete shifts one lower cell");
        check(updateCost.lookup(ip("10.1.2.2")) == 3 && updateCost.lookup(ip("10.9.9.9")) == 1,
                "table is correct after the delete");
        updateCost.insertAt(0, ip("10.1.2.0") & prefixMask(24), prefixMask(24), 7);
        check(updateCost.shiftedCells() == 3, "priority insert shifts the two existing cells");
        check(updateCost.lookup(ip("10.1.2.2")) == 7, "new more-specific rule takes over");
        check(updateCost.lookup(ip("10.1.9.9")) == 3, "unrelated traffic in 10.1/16 unaffected");
        check(updateCost.lookup(ip("10.9.0.1")) == 1, "traffic that no longer matches falls back");
        check(updateCost.updates() == 5, "update counter counts adds, the delete and the insert");

        // Shadowing analysis: a rule is dead when an earlier rule covers its whole space.
        Tcam shadowed = new Tcam();
        shadowed.add(ip("10.2.0.0") & prefixMask(16), prefixMask(16), 2);
        shadowed.add(ip("10.3.0.0") & prefixMask(16), prefixMask(16), 3);
        shadowed.add(ip("10.0.0.0") & prefixMask(8), prefixMask(8), 1);
        shadowed.add(ip("10.2.5.0") & prefixMask(24), prefixMask(24), 4);
        check(shadowed.isShadowed(3), "10.2.5/24 shadowed by the earlier 10.2/16");
        check(!shadowed.isShadowed(2), "the broad 10/8 still carries 10.9 traffic");
        check(!shadowed.isShadowed(0) && !shadowed.isShadowed(1), "the leading rules are live");
        check(shadowed.covers(0, 3) && shadowed.covers(2, 3) && !shadowed.covers(0, 2),
                "coverage follows the prefix containment relation");
        check(shadowed.lookup(ip("10.2.5.7")) == 2, "shadowed rule is indeed never selected");
        check(shadowed.lookup(ip("10.3.9.9")) == 3 && shadowed.lookup(ip("10.9.9.9")) == 1,
                "the remaining rules still match");

        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
