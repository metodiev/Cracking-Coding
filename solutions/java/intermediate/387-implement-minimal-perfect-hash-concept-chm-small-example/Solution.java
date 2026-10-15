/** Task 387 - Implement minimal perfect hash concept (CHM) small example. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class Mphf {
    final int n;
    final String[] table;
    final int[] g;
    final int[] displacement;
    final long base2;

    Mphf(int n, String[] table, int[] g, int[] displacement, long base2) {
        this.n = n;
        this.table = table;
        this.g = g;
        this.displacement = displacement;
        this.base2 = base2;
    }

    /** Slot of a member key in [0, n); -1 when the first-level bucket stayed empty. */
    int lookup(String key) {
        int bucket = (int) (Solution.polyHash(key, Solution.BASE1) % n);
        int shift = displacement[bucket];
        if (shift < 0) return -1;
        int sigma = (int) ((Solution.polyHash(key, base2) + shift) % n);
        if (g[sigma] < 0) return -1;
        return g[sigma];
    }
}

public class Solution {
    static final long HASH_MOD = 2147483647L;
    static final long BASE1 = 131;
    static final long BASE2 = 1000003;
    static final String[] KEYS = { "alice", "bob", "carol", "dave", "erin", "frank", "grace",
            "heidi", "ivan", "judy", "karl", "laura" };
    static final String[] MORE_KEYS = { "kiwi", "mango", "melon", "olive", "peach", "plum",
            "quince", "raspberry" };

    static long polyHash(String text, long base) {
        long value = 0;
        for (int i = 0; i < text.length(); i++)
            value = (value * base + text.charAt(i)) % HASH_MOD;
        return value;
    }

    /**
     * Hash-displace build: buckets by h1 mod n; each bucket gets the smallest shift whose
     * sigmas (h2 + shift) mod n are distinct and unused; final slots are consecutive in
     * increasing sigma order. Retries the whole build with base2 + attempt on failure.
     */
    @SuppressWarnings("unchecked")
    static Mphf buildMphf(String[] keys) {
        int n = keys.length;
        for (int attempt = 0; attempt < 100; attempt++) {
            long base2 = BASE2 + attempt;
            List<String>[] buckets = new List[n];
            for (int b = 0; b < n; b++) buckets[b] = new ArrayList<>();
            for (String key : keys) buckets[(int) (polyHash(key, BASE1) % n)].add(key);
            Integer[] order = new Integer[n];
            for (int b = 0; b < n; b++) order[b] = b;
            Arrays.sort(order, (x, y) -> buckets[x].size() != buckets[y].size()
                    ? Integer.compare(buckets[y].size(), buckets[x].size())
                    : Integer.compare(x, y));
            String[] table = new String[n];
            int[] g = new int[n];
            int[] displacement = new int[n];
            Arrays.fill(g, -1);
            Arrays.fill(displacement, -1);
            int nextSlot = 0;
            boolean feasible = true;
            for (int bucket : order) {
                List<String> members = buckets[bucket];
                if (members.isEmpty()) continue;
                int chosen = -1;
                int[] sigmas = new int[members.size()];
                for (int shift = 0; shift < n && chosen < 0; shift++) {
                    boolean[] used = new boolean[n];
                    boolean good = true;
                    for (int i = 0; i < members.size() && good; i++) {
                        int sigma = (int) ((polyHash(members.get(i), base2) + shift) % n);
                        if (used[sigma] || g[sigma] != -1) good = false;
                        used[sigma] = true;
                        sigmas[i] = sigma;
                    }
                    if (good) chosen = shift;
                }
                if (chosen < 0) {
                    feasible = false;
                    break;
                }
                displacement[bucket] = chosen;
                Integer[] idx = new Integer[members.size()];
                for (int i = 0; i < idx.length; i++) idx[i] = i;
                final int[] finalSigmas = sigmas;
                Arrays.sort(idx, (x, y) -> Integer.compare(finalSigmas[x], finalSigmas[y]));
                for (int rank = 0; rank < idx.length; rank++) {
                    int sigma = sigmas[idx[rank]];
                    g[sigma] = nextSlot + rank;
                    table[nextSlot + rank] = members.get(idx[rank]);
                }
                nextSlot += members.size();
            }
            if (feasible) return new Mphf(n, table, g, displacement, base2);
        }
        throw new IllegalStateException("no displacement assignment found");
    }

    private static String joinInts(int[] values) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) builder.append(',');
            builder.append(values[i]);
        }
        return builder.toString();
    }

    public static void main(String[] args) {
        int n = KEYS.length;
        check(n == 12, "twelve keys");
        check(new HashSet<>(Arrays.asList(KEYS)).size() == n, "keys are distinct");

        // deterministic hash sanity (values pin the polynomial hash)
        check(polyHash("alice", BASE1) == 893813729L, "hash alice base 131");
        check(polyHash("alice", BASE2) == 76280623L, "hash alice base 1000003");
        check(polyHash("alice", BASE1) % n == 5, "alice bucket 5");
        check(polyHash("bob", BASE1) % n == 1 && polyHash("laura", BASE1) % n == 1, "bucket 1 pair");
        check(polyHash("frank", BASE1) % n == 8 && polyHash("judy", BASE1) % n == 8
                && polyHash("karl", BASE1) % n == 8, "bucket 8 triple");
        int largest = 0;
        for (int b = 0; b < n; b++) {
            int count = 0;
            for (String key : KEYS) if (polyHash(key, BASE1) % n == b) count++;
            largest = Math.max(largest, count);
        }
        check(largest == 3, "largest bucket has 3 keys");

        Mphf mphf = buildMphf(KEYS);
        check(mphf.base2 == BASE2, "first attempt already succeeds");
        check(String.join(",", mphf.table).equals(
                "frank,judy,karl,laura,bob,ivan,grace,heidi,dave,alice,erin,carol"), "expected table");
        check(joinInts(mphf.g).equals("7,0,5,9,10,11,1,2,3,4,6,8"), "g table");
        check(joinInts(mphf.displacement).equals("-1,1,7,0,5,8,5,-1,0,-1,6,-1"), "displacements");
        for (String entry : mphf.table) check(entry != null, "no empty table slot");
        for (int entry : mphf.g) check(entry >= 0, "no empty g slot");

        // minimal perfect hash: every key maps to its own slot in [0, n), no collisions
        boolean[] seen = new boolean[n];
        for (String key : KEYS) {
            int slot = mphf.lookup(key);
            check(slot >= 0 && slot < n, "slot in range " + key);
            check(mphf.table[slot].equals(key), "table[slot] = " + key);
            check(!seen[slot], "slot not reused " + key);
            seen[slot] = true;
        }
        for (boolean used : seen) check(used, "all slots used");

        // determinism: rebuilding yields the identical structure
        Mphf again = buildMphf(KEYS);
        check(Arrays.equals(again.table, mphf.table) && Arrays.equals(again.g, mphf.g)
                && Arrays.equals(again.displacement, mphf.displacement)
                && again.base2 == mphf.base2, "deterministic rebuild");

        // non-member keys are out of scope: any in-range answer is meaningless by construction
        int stranger = mphf.lookup("stranger");
        check(stranger == -1 || (stranger >= 0 && stranger < n), "non-member answer in range or -1");

        // a second key set needs one hash retry (documented) and still builds a bijection
        Mphf second = buildMphf(MORE_KEYS);
        check(second.base2 == BASE2 + 2, "attempt 2 succeeds for the second set");
        check(String.join(",", second.table).equals(
                "plum,kiwi,peach,melon,olive,mango,raspberry,quince"), "second table");
        check(joinInts(second.displacement).equals("0,0,0,-1,1,4,-1,-1"), "second displacements");
        boolean[] seen3 = new boolean[MORE_KEYS.length];
        for (String key : MORE_KEYS) {
            int slot = second.lookup(key);
            check(slot >= 0 && slot < MORE_KEYS.length, "second slot in range " + key);
            check(second.table[slot].equals(key), "second table[slot] = " + key);
            check(!seen3[slot], "second slot not reused " + key);
            seen3[slot] = true;
        }
        for (boolean used : seen3) check(used, "second all slots used");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
