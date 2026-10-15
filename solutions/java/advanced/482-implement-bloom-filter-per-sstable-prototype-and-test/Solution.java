/** Task 482 - Implement bloom filter per SSTable prototype and test. (Tier: Advanced, CTCI Ch 14)
 *
 * Prototype of an LSM-style lookup: each SSTable owns a bloom filter sized to 10 bits per key with
 * k = 7 double-hashed probes, and a sorted entry list for the exact confirmation step. Hashing is
 * FNV-1a 32-bit; probe j uses (h1 + j * h2) % m with h2 = fnv1a(key + "!") | 1 (odd, so the probe
 * sequence never collapses). Lookup consults every table's filter first (false positives only cost
 * an exact binary search, never a wrong answer).
 *
 * Documented false-positive estimate (bits m, keys n, probes k): p = (1 - e^(-k*n/m))^k.
 * With n = 150/180/210, m = 10n, k = 7 that is p ~ 0.0082 per table; for the combined union of
 * tables the estimate is 1 - prod(1 - p_t) ~ 0.0244. The test pins the exact (deterministic)
 * false-positive counts of the probe sets and also asserts the measured rate stays within a
 * documented 3x-theory bound, plus zero false negatives for every inserted key.
 */
import java.util.*;

class BloomFilter {
    final int m;
    final int k;
    final byte[] words;
    int count;

    BloomFilter(int bits, int hashes) {
        m = bits;
        k = hashes;
        words = new byte[(bits + 7) / 8];
    }

    static long fnv1a(String text) {
        long h = 2166136261L;
        for (int i = 0; i < text.length(); i++) {
            h ^= text.charAt(i);
            h = (h * 16777619L) & 0xFFFFFFFFL;
        }
        return h;
    }

    long[] probes(String key) {
        long h1 = fnv1a(key);
        long h2 = fnv1a(key + "!") | 1L;
        long[] idx = new long[k];
        for (int j = 0; j < k; j++) idx[j] = (h1 + j * h2) % m;
        return idx;
    }

    void add(String key) {
        for (long idx : probes(key)) words[(int) (idx >> 3)] |= 1 << (idx & 7);
        count++;
    }

    boolean maybeContains(String key) {
        for (long idx : probes(key)) {
            if ((words[(int) (idx >> 3)] & (1 << (idx & 7))) == 0) return false;
        }
        return true;
    }
}

class SSTable {
    final List<String> entries;
    final BloomFilter bloom;

    SSTable(List<String> entries, int bitsPerKey, int hashes) {
        this.entries = new ArrayList<>(entries);
        Collections.sort(this.entries);
        this.bloom = new BloomFilter(bitsPerKey * this.entries.size(), hashes);
    }

    SSTable build(List<String> keys) {
        for (String key : keys) bloom.add(key);
        return this;
    }

    boolean containsExact(String key) {
        int lo = 0, hi = entries.size() - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            int cmp = entries.get(mid).compareTo(key);
            if (cmp == 0) return true;
            if (cmp < 0) lo = mid + 1;
            else hi = mid - 1;
        }
        return false;
    }

    boolean contains(String key) {
        return bloom.maybeContains(key) && containsExact(key);
    }
}

class Store {
    final List<SSTable> tables;

    Store(List<SSTable> tables) {
        this.tables = tables;
    }

    boolean get(String key) {
        for (SSTable table : tables) if (table.contains(key)) return true;
        return false;
    }

    boolean bloomAny(String key) {
        for (SSTable table : tables) if (table.bloom.maybeContains(key)) return true;
        return false;
    }
}

public class Solution {
    private static int seed = 482;

    private static int[] lcgValues(int count, int limit) {
        int[] values = new int[count];
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % limit;
        }
        return values;
    }

    private static double theoreticalFpRate(int keys, int bits, int hashes) {
        return Math.pow(1.0 - Math.exp(-hashes * (double) keys / bits), hashes);
    }

    private static List<SSTable> buildStore(int[] sizes, int startSeed) {
        List<SSTable> tables = new ArrayList<>();
        seed = startSeed;
        for (int i = 0; i < sizes.length; i++) {
            int size = sizes[i];
            List<String> keys = new ArrayList<>();
            List<String> one = new ArrayList<>();
            for (int j = 0; j < size; j++) one.add("t" + i + ":" + j);
            int[] rotation = new int[1];
            seed = (seed * 75 + 74) % 65537;
            rotation[0] = seed % size;
            keys.addAll(one.subList(rotation[0], size));
            keys.addAll(one.subList(0, rotation[0]));
            tables.add(new SSTable(keys, 10, 7).build(keys));
        }
        return tables;
    }

    private static int popcount(byte[] bytes) {
        int total = 0;
        for (byte b : bytes) total += Integer.bitCount(b & 0xFF);
        return total;
    }

    public static void main(String[] args) {
        BloomFilter tiny = new BloomFilter(64, 4);
        tiny.add("alpha");
        check(tiny.maybeContains("alpha"), "tiny contains");
        check(!new BloomFilter(64, 4).maybeContains("alpha"), "empty rejects");

        int[] sizes = {150, 180, 210};
        List<SSTable> tables = buildStore(sizes, 482);
        Store store = new Store(tables);
        for (int i = 0; i < 3; i++) {
            check(tables.get(i).entries.size() == sizes[i], "table size " + i);
            check(tables.get(i).bloom.m == sizes[i] * 10, "bloom bits " + i);
            check(tables.get(i).bloom.k == 7, "bloom hashes " + i);
        }

        int presentQueries = 0;
        for (SSTable table : tables) {
            for (String key : table.entries) {
                check(table.bloom.maybeContains(key), "no false negative " + key);
                check(store.get(key), "store finds " + key);
                presentQueries++;
            }
        }
        check(presentQueries == 540, "present query count");

        Set<String> unionSet = new TreeSet<>();
        for (SSTable table : tables) unionSet.addAll(table.entries);
        List<String> union = new ArrayList<>(unionSet);
        seed = 9001;
        for (int value : lcgValues(200, 540)) {
            String key = union.get(value);
            check(store.get(key) && store.bloomAny(key), "present probe " + key);
        }
        List<String> absent = new ArrayList<>();
        seed = 9002;
        for (int value : lcgValues(240, 65537)) absent.add("absent:" + value);
        for (String key : absent) {
            check(!unionSet.contains(key), "absent build " + key);
            check(!store.get(key), "absent lookup " + key);
        }

        int[] perTableFp = new int[3];
        for (int i = 0; i < 3; i++) {
            for (String key : absent) if (tables.get(i).bloom.maybeContains(key)) perTableFp[i]++;
        }
        int combinedFp = 0;
        for (String key : absent) if (store.bloomAny(key)) combinedFp++;
        // Pinned from the deterministic construction; identical in all four languages.
        check(Arrays.equals(perTableFp, new int[]{2, 1, 1}), Arrays.toString(perTableFp));
        check(combinedFp == 4, "combined fp " + combinedFp);

        double[] pTable = new double[3];
        double pUnionComplement = 1.0;
        for (int i = 0; i < 3; i++) {
            SSTable t = tables.get(i);
            pTable[i] = theoreticalFpRate(sizes[i], t.bloom.m, t.bloom.k);
            pUnionComplement *= 1.0 - pTable[i];
            check(perTableFp[i] <= 3 * pTable[i] * absent.size() + 3, "theory bound " + i);
            check(perTableFp[i] / (double) absent.size() <= 0.05, "rate bound " + i);
            check(Math.round(pTable[i] * 10000) == 82, "theoretical p " + i);
        }
        double pUnion = 1.0 - pUnionComplement;
        check(combinedFp <= 3 * pUnion * absent.size() + 3, "union bound");
        check(combinedFp / (double) absent.size() <= 0.05, "union rate bound");
        check(Math.round(pUnion * 10000) == 244, "theoretical union " + pUnion);

        int[] setBits = new int[3];
        for (int i = 0; i < 3; i++) setBits[i] = popcount(tables.get(i).bloom.words);
        check(Arrays.equals(setBits, new int[]{768, 905, 1052}), Arrays.toString(setBits));
        for (int i = 0; i < 3; i++) check(setBits[i] < tables.get(i).bloom.m, "not saturated " + i);
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
