/** Task 374 - Bitset (bit array) with fast set operations for a small universe. (Tier: Intermediate, CTCI Ch 5) */
import java.util.*;

/**
 * A fixed universe of n_bits (here <= 256) is packed into 32-bit words.  Every
 * operation touches only whole words, so add/remove/contains are O(1), count() is
 * O(words) using a popcount, and union/intersection are O(words) instead of
 * O(universe).  Iteration extracts the lowest set bit of each word in turn, which
 * yields indices in strictly ascending order without scanning empty stretches.
 *
 * Everything is cross-checked against the plain set oracle (TreeSet) on deterministic
 * LCG universes: membership of every index, count() against the oracle size, ascending
 * iteration against the sorted oracle, and union/intersection against the oracle set
 * operations.  Boundary indices 0, 31, 32 and n_bits-1 are asserted explicitly.
 *
 * The implementation only uses 32-bit words, so the sign quirks of each language are
 * handled identically: words are treated as unsigned bit patterns (Integer.bitCount
 * and Integer.numberOfTrailingZeros work on the raw 32-bit pattern).
 */

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
    private int seed;

    Lcg(int seed) {
        this.seed = seed;
    }

    int next(int modulus) {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

/** Bitset over indices 0 .. bits-1, backed by 32-bit words. */
class IntBitset {
    static final int UNIVERSE = 256;

    final int bits;
    final int[] words;

    IntBitset(int bits) {
        if (bits <= 0) throw new IllegalArgumentException("universe must be positive");
        this.bits = bits;
        this.words = new int[(bits + 31) / 32];
    }

    void add(int index) {
        validate(index);
        words[index >>> 5] |= 1 << (index & 31);
    }

    void remove(int index) {
        validate(index);
        words[index >>> 5] &= ~(1 << (index & 31));
    }

    boolean contains(int index) {
        validate(index);
        return ((words[index >>> 5] >>> (index & 31)) & 1) == 1;
    }

    int count() {
        int total = 0;
        for (int word : words) total += Integer.bitCount(word);
        return total;
    }

    IntBitset union(IntBitset other) {
        checkSameUniverse(other);
        IntBitset result = new IntBitset(bits);
        for (int i = 0; i < words.length; i++) result.words[i] = words[i] | other.words[i];
        return result;
    }

    IntBitset intersection(IntBitset other) {
        checkSameUniverse(other);
        IntBitset result = new IntBitset(bits);
        for (int i = 0; i < words.length; i++) result.words[i] = words[i] & other.words[i];
        return result;
    }

    /** All members as indices in ascending order. */
    int[] iterate() {
        int[] members = new int[count()];
        int at = 0;
        for (int wi = 0; wi < words.length; wi++) {
            int word = words[wi];
            while (word != 0) {
                int low = word & -word;
                members[at++] = (wi << 5) + Integer.numberOfTrailingZeros(word);
                word ^= low;
            }
        }
        return members;
    }

    private void validate(int index) {
        if (index < 0 || index >= bits) throw new IllegalArgumentException("index " + index);
    }

    private void checkSameUniverse(IntBitset other) {
        if (bits != other.bits) throw new IllegalArgumentException("universe size mismatch");
    }
}

public class Solution {
    static int[] sortedMembers(TreeSet<Integer> oracle) {
        int[] result = new int[oracle.size()];
        int at = 0;
        for (int value : oracle) result[at++] = value;
        return result;
    }

    public static void main(String[] args) {
        // --- boundary behaviour ----------------------------------------------
        IntBitset empty = new IntBitset(IntBitset.UNIVERSE);
        check(empty.count() == 0, "empty count");
        check(empty.iterate().length == 0, "empty iterate");

        IntBitset boundary = new IntBitset(IntBitset.UNIVERSE);
        for (int index : new int[]{0, 31, 32, 33, 63, 64, 255}) boundary.add(index);
        check(Arrays.equals(boundary.iterate(), new int[]{0, 31, 32, 33, 63, 64, 255}), "boundary iterate");
        check(boundary.count() == 7, "boundary count");
        check(boundary.contains(0) && boundary.contains(255), "boundary contains");
        check(!boundary.contains(1) && !boundary.contains(254), "boundary non-members");
        boundary.add(0);
        check(boundary.count() == 7, "add is idempotent");
        boundary.remove(1);
        check(boundary.count() == 7, "remove absent is a no-op");
        boundary.remove(0);
        check(!boundary.contains(0) && boundary.count() == 6, "remove works");

        // --- explicit small example -------------------------------------------
        IntBitset first = new IntBitset(64);
        IntBitset second = new IntBitset(64);
        for (int index : new int[]{1, 5, 40, 63}) first.add(index);
        for (int index : new int[]{5, 7, 40, 41}) second.add(index);
        check(Arrays.equals(first.iterate(), new int[]{1, 5, 40, 63}), "first iterate");
        check(Arrays.equals(second.iterate(), new int[]{5, 7, 40, 41}), "second iterate");
        check(Arrays.equals(first.union(second).iterate(), new int[]{1, 5, 7, 40, 41, 63}), "union");
        check(Arrays.equals(first.intersection(second).iterate(), new int[]{5, 40}), "intersection");
        check(first.union(second).count() == 6, "union count");
        check(first.intersection(second).count() == 2, "intersection count");
        check(new IntBitset(64).union(first).count() == 4, "union with empty");
        check(new IntBitset(64).intersection(first).count() == 0, "intersection with empty");

        IntBitset full = new IntBitset(IntBitset.UNIVERSE);
        for (int index = 0; index < IntBitset.UNIVERSE; index++) full.add(index);
        check(full.count() == IntBitset.UNIVERSE, "full count");
        check(full.iterate().length == IntBitset.UNIVERSE, "full iterate size");
        IntBitset spark = new IntBitset(IntBitset.UNIVERSE);
        for (int index : new int[]{1, 5, 40, 63, 255}) spark.add(index);
        check(full.union(spark).count() == IntBitset.UNIVERSE, "full union stays full");
        check(full.intersection(spark).count() == 5, "full intersection");
        check(new IntBitset(IntBitset.UNIVERSE).union(spark).count() == 5, "empty union spark");

        // --- LCG cross-checks against the plain set oracle --------------------
        Lcg lcg = new Lcg(37413);
        for (int trial = 0; trial < 40; trial++) {
            TreeSet<Integer> oracleA = new TreeSet<>();
            TreeSet<Integer> oracleB = new TreeSet<>();
            IntBitset bitsA = new IntBitset(IntBitset.UNIVERSE);
            IntBitset bitsB = new IntBitset(IntBitset.UNIVERSE);

            for (int step = 0; step < 60; step++) {
                int index = lcg.next(IntBitset.UNIVERSE);
                int action = lcg.next(IntBitset.UNIVERSE);
                if (action % 3 == 0) {
                    bitsA.remove(index);
                    oracleA.remove(index);
                } else {
                    bitsA.add(index);
                    oracleA.add(index);
                }
                index = lcg.next(IntBitset.UNIVERSE);
                action = lcg.next(IntBitset.UNIVERSE);
                if (action % 3 == 0) {
                    bitsB.remove(index);
                    oracleB.remove(index);
                } else {
                    bitsB.add(index);
                    oracleB.add(index);
                }
            }

            for (int index = 0; index < IntBitset.UNIVERSE; index++) {
                check(bitsA.contains(index) == oracleA.contains(index), "membership A " + trial);
                check(bitsB.contains(index) == oracleB.contains(index), "membership B " + trial);
            }
            check(bitsA.count() == oracleA.size(), "count A " + trial);
            check(bitsB.count() == oracleB.size(), "count B " + trial);
            check(Arrays.equals(bitsA.iterate(), sortedMembers(oracleA)), "iterate A " + trial);
            check(Arrays.equals(bitsB.iterate(), sortedMembers(oracleB)), "iterate B " + trial);

            TreeSet<Integer> unionOracle = new TreeSet<>(oracleA);
            unionOracle.addAll(oracleB);
            TreeSet<Integer> intersectionOracle = new TreeSet<>(oracleA);
            intersectionOracle.retainAll(oracleB);
            IntBitset union = bitsA.union(bitsB);
            IntBitset intersection = bitsA.intersection(bitsB);
            check(union.count() == unionOracle.size(), "union count " + trial);
            check(intersection.count() == intersectionOracle.size(), "intersection count " + trial);
            check(Arrays.equals(union.iterate(), sortedMembers(unionOracle)), "union iterate " + trial);
            check(Arrays.equals(intersection.iterate(), sortedMembers(intersectionOracle)),
                    "intersection iterate " + trial);
            for (int index = 0; index < IntBitset.UNIVERSE; index += 7) {
                check(union.contains(index) == unionOracle.contains(index), "union membership " + trial);
                check(intersection.contains(index) == intersectionOracle.contains(index),
                        "intersection membership " + trial);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
