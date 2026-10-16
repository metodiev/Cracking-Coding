/** Task 652 - Implement bitmap index and fast AND/OR operations. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * A bitmap index maps every value of a low-cardinality column to the set of row ids holding that
 * value, stored as a packed long[] (64 rows per word, java.util.BitSet kept out of the hot path so
 * word level behaviour is observable). Set combination is done word-at-a-time:
 *
 *   AND / OR / ANDNOT  - one pass over the words, cardinality via Long.bitCount
 *   intersectAll       - operands sorted by ascending cardinality, short circuiting as soon as the
 *                        accumulator is empty, and skipping zero words entirely
 *
 * Invariants verified: the OR of every value bitmap is exactly the full row mask; AND and OR agree
 * with both a brute-force row scan and an independent java.util.BitSet reference; AND is commutative
 * and associative; ANDNOT(a,b) = card(a) - card(a AND b); an absent value yields an empty bitmap and
 * the sparse short circuit touches zero words; rows past the last word boundary (63/64/65) and the
 * degenerate 0- and 1-row columns are exercised.
 */
import java.util.*;

class LongBitmap {
    final int bits;
    final long[] words;

    LongBitmap(int bits) {
        this.bits = bits;
        this.words = new long[(bits + 63) >>> 6];
    }

    void set(int row) {
        words[row >>> 6] |= 1L << (row & 63);
    }

    boolean get(int row) {
        return row >= 0 && row < bits && (words[row >>> 6] & (1L << (row & 63))) != 0;
    }

    int cardinality() {
        int total = 0;
        for (long word : words) total += Long.bitCount(word);
        return total;
    }

    boolean isEmpty() {
        for (long word : words) if (word != 0) return false;
        return true;
    }

    int[] rows() {
        int[] out = new int[cardinality()];
        int at = 0;
        for (int row = 0; row < bits; row++) if (get(row)) out[at++] = row;
        return out;
    }

    static void requireSameLength(long[] a, long[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("bitmaps differ in word count");
    }

    /** Word-parallel AND. counters[0] counts the words that needed a real AND + popcount. */
    static int and(long[] a, long[] b, int[] counters) {
        requireSameLength(a, b);
        int total = 0, touched = 0;
        for (int i = 0; i < a.length; i++) {
            long word = a[i];
            if (word == 0) continue;
            touched++;
            total += Long.bitCount(word & b[i]);
        }
        counters[0] = touched;
        return total;
    }

    static int or(long[] a, long[] b, int[] counters) {
        requireSameLength(a, b);
        int total = 0, touched = 0;
        for (int i = 0; i < a.length; i++) {
            long merged = a[i] | b[i];
            if (merged != 0) touched++;
            total += Long.bitCount(merged);
        }
        counters[0] = touched;
        return total;
    }

    static int andNot(long[] a, long[] b) {
        requireSameLength(a, b);
        int total = 0;
        for (int i = 0; i < a.length; i++) total += Long.bitCount(a[i] & ~b[i]);
        return total;
    }
}

class BitmapIndex {
    final int rows;
    final TreeMap<Integer, LongBitmap> byValue = new TreeMap<>();

    BitmapIndex(int[] column) {
        rows = column.length;
        for (int row = 0; row < column.length; row++) {
            byValue.computeIfAbsent(column[row], value -> new LongBitmap(rows)).set(row);
        }
    }

    /** Empty bitmap when the value does not occur - an equality against an absent value matches no row. */
    LongBitmap bitmapOf(int value) {
        LongBitmap bitmap = byValue.get(value);
        return bitmap == null ? new LongBitmap(rows) : bitmap;
    }

    LongBitmap allRows() {
        LongBitmap merged = new LongBitmap(rows);
        for (LongBitmap bitmap : byValue.values()) {
            for (int i = 0; i < merged.words.length; i++) merged.words[i] |= bitmap.words[i];
        }
        return merged;
    }

    LongBitmap valueUnion(int[] values) {
        LongBitmap merged = new LongBitmap(rows);
        for (int value : values) {
            LongBitmap bitmap = bitmapOf(value);
            for (int i = 0; i < merged.words.length; i++) merged.words[i] |= bitmap.words[i];
        }
        return merged;
    }

    int countEqual(int value) {
        return bitmapOf(value).cardinality();
    }

    int countOr(int[] values) {
        return valueUnion(values).cardinality();
    }

    int countAnd(int[] values, int[] counters) {
        List<LongBitmap> operands = new ArrayList<>();
        for (int value : values) operands.add(bitmapOf(value));
        return intersectAll(operands, counters);
    }

    /** Intersection with ascending-cardinality ordering and empty-accumulator short circuit. */
    static int intersectAll(List<LongBitmap> operands, int[] counters) {
        if (operands.isEmpty()) throw new IllegalArgumentException("need at least one bitmap");
        for (LongBitmap bitmap : operands) LongBitmap.requireSameLength(bitmap.words, operands.get(0).words);
        List<LongBitmap> ordered = new ArrayList<>(operands);
        ordered.sort(Comparator.comparingInt(LongBitmap::cardinality));
        long[] accumulator = ordered.get(0).words.clone();
        int touched = 0;
        for (int k = 1; k < ordered.size(); k++) {
            if (isEmptyWords(accumulator)) break;
            long[] other = ordered.get(k).words;
            for (int i = 0; i < accumulator.length; i++) {
                if (accumulator[i] == 0) continue;
                touched++;
                accumulator[i] &= other[i];
            }
        }
        counters[0] = touched;
        int total = 0;
        for (long word : accumulator) total += Long.bitCount(word);
        return total;
    }

    private static boolean isEmptyWords(long[] words) {
        for (long word : words) if (word != 0) return false;
        return true;
    }

    int countAndNot(int a, int b) {
        return LongBitmap.andNot(bitmapOf(a).words, bitmapOf(b).words);
    }

    /** Rows whose value lies in [lo, hi] - a union over the (small) value domain. */
    int countBetween(int lo, int hi) {
        if (lo > hi) throw new IllegalArgumentException("lo > hi");
        LongBitmap merged = new LongBitmap(rows);
        for (Map.Entry<Integer, LongBitmap> entry : byValue.entrySet()) {
            if (entry.getKey() >= lo && entry.getKey() <= hi) {
                for (int i = 0; i < merged.words.length; i++) merged.words[i] |= entry.getValue().words[i];
            }
        }
        return merged.cardinality();
    }
}

public class Solution {
    private static long state = 0x2545F4914F6CDD1DL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    static int[] randomColumn(int size, int domain) {
        int[] column = new int[size];
        for (int i = 0; i < size; i++) column[i] = nextInt(domain);
        return column;
    }

    static int bruteEquals(int[] column, int value) {
        int count = 0;
        for (int v : column) if (v == value) count++;
        return count;
    }

    static BitSet bruteSet(int[] column, int value) {
        BitSet set = new BitSet(column.length);
        for (int row = 0; row < column.length; row++) if (column[row] == value) set.set(row);
        return set;
    }

    public static void main(String[] args) {
        // --- degenerate columns -------------------------------------------------
        BitmapIndex empty = new BitmapIndex(new int[0]);
        check(empty.countEqual(4) == 0, "empty index: equality");
        check(empty.countOr(new int[]{1, 2, 3}) == 0, "empty index: union");
        check(empty.countBetween(0, 99) == 0, "empty index: range");
        check(empty.allRows().isEmpty(), "empty index: row mask is empty");

        BitmapIndex single = new BitmapIndex(new int[]{9});
        check(single.countEqual(9) == 1 && single.countEqual(8) == 0, "single row: equality");
        check(single.allRows().cardinality() == 1, "single row: full mask");
        check(!single.bitmapOf(9).get(1), "single row: out of range read is false");

        // --- word boundary rows (63 / 64 / 65) ---------------------------------
        int[] boundary = new int[65];
        for (int row = 0; row < boundary.length; row++) boundary[row] = row % 5;
        BitmapIndex boundaryIndex = new BitmapIndex(boundary);
        for (int value = 0; value < 5; value++) {
            check(boundaryIndex.countEqual(value) == bruteEquals(boundary, value), "boundary equality " + value);
        }
        check(boundaryIndex.allRows().cardinality() == 65, "boundary full mask");
        check(boundaryIndex.bitmapOf(4).get(64) && !boundaryIndex.bitmapOf(0).get(64), "row 64 in second word");
        check(!boundaryIndex.bitmapOf(1).get(63) && boundaryIndex.bitmapOf(3).get(63),
                "row 63 belongs to value 3");

        // --- constant column ----------------------------------------------------
        int[] constant = new int[40];
        Arrays.fill(constant, 7);
        BitmapIndex constantIndex = new BitmapIndex(constant);
        check(constantIndex.countEqual(7) == 40, "constant column: owning bitmap");
        check(constantIndex.countAnd(new int[]{7, 8}, new int[1]) == 0, "constant column: AND with absent");
        check(constantIndex.countBetween(7, 7) == 40, "constant column: range");

        // --- AND / OR / ANDNOT against brute force and BitSet ------------------
        int[] column = randomColumn(200, 32);
        BitmapIndex index = new BitmapIndex(column);
        check(index.byValue.size() == distinctCount(column), "one bitmap per distinct value");
        int[] counters = new int[1];
        int checkedPairs = 0;
        for (int a = 0; a < 32; a++) {
            for (int b = 0; b < 32; b++) {
                BitSet reference = bruteSet(column, a);
                reference.and(bruteSet(column, b));
                int got = BitmapIndex.intersectAll(
                        List.of(index.bitmapOf(a), index.bitmapOf(b)), counters);
                check(got == reference.cardinality(), "AND " + a + "&" + b + " -> " + got);
                BitSet union = bruteSet(column, a);
                union.or(bruteSet(column, b));
                check(index.countOr(new int[]{a, b}) == union.cardinality(), "OR " + a + "|" + b);
                int andNot = index.countAndNot(a, b);
                check(andNot == index.countEqual(a) - got, "ANDNOT " + a + "\\" + b);
                checkedPairs++;
            }
        }
        check(checkedPairs == 1024, "exhaustive pair sweep ran");

        // --- invariants: full mask, commutativity, associativity, duplicates ---
        check(index.allRows().cardinality() == column.length, "OR of every value bitmap is the full mask");
        for (int value : new int[]{0, 3, 17, 31}) {
            check(index.countEqual(value) == bruteEquals(column, value), "equality vs brute force " + value);
        }
        int[] countersA = new int[1], countersB = new int[1];
        check(index.countAnd(new int[]{5, 11, 23}, countersA) == index.countAnd(new int[]{23, 11, 5}, countersB),
                "AND is commutative");
        check(index.countAnd(new int[]{5, 11, 23}, countersA)
                == index.countAnd(new int[]{11, 5, 5, 23, 23}, countersB), "AND ignores duplicates");
        int leftAssociated = index.countAnd(new int[]{0, 5, 11, 23}, countersA);
        BitSet lowHalf = bruteSet(column, 0);
        lowHalf.and(bruteSet(column, 5));
        BitSet highHalf = bruteSet(column, 11);
        highHalf.and(bruteSet(column, 23));
        lowHalf.and(highHalf);
        check(leftAssociated == lowHalf.cardinality(), "AND associativity");
        check(index.countAnd(new int[]{3}, countersA) == index.countEqual(3), "single operand intersection");

        // --- sparse short circuit ---------------------------------------------
        check(index.countAnd(new int[]{999, 5}, countersA) == 0, "absent value empties the intersection");
        check(countersA[0] == 0, "absent value short circuits with zero word operations, touched " + countersA[0]);
        check(index.bitmapOf(999).isEmpty(), "absent value bitmap is empty");
        int[] touched = new int[1];
        LongBitmap.and(index.bitmapOf(999).words, index.bitmapOf(5).words, touched);
        check(touched[0] == 0, "empty left operand skips every word, touched " + touched[0]);

        // --- ranges and rejected input ----------------------------------------
        for (int lo = 0; lo < 8; lo++) {
            for (int hi = lo; hi < 32; hi += 7) {
                int expected = 0;
                for (int v : column) if (v >= lo && v <= hi) expected++;
                check(index.countBetween(lo, hi) == expected, "range [" + lo + "," + hi + "]");
            }
        }
        boolean rejected = false;
        try {
            index.countBetween(9, 4);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "lo > hi must be rejected");
        rejected = false;
        try {
            LongBitmap.requireSameLength(new long[2], new long[3]);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "mismatched bitmap widths must be rejected");

        // --- wider sweep on differently shaped columns ------------------------
        int[] shapes = {0, 1, 2, 63, 64, 65, 130};
        for (int shape : shapes) {
            int[] data = randomColumn(shape, 7);
            BitmapIndex shapeIndex = new BitmapIndex(data);
            for (int value = 0; value < 7; value++) {
                check(shapeIndex.countEqual(value) == bruteEquals(data, value),
                        "shape " + shape + " value " + value);
                check(Arrays.equals(shapeIndex.bitmapOf(value).rows(),
                        sortedRows(bruteSet(data, value))), "shape " + shape + " rows " + value);
            }
            check(shapeIndex.allRows().cardinality() == shape, "shape " + shape + " full mask");
        }

        System.out.println("PASS");
    }

    static int[] sortedRows(BitSet set) {
        int[] out = new int[set.cardinality()];
        int at = 0;
        for (int row = set.nextSetBit(0); row >= 0; row = set.nextSetBit(row + 1)) out[at++] = row;
        return out;
    }

    static int distinctCount(int[] column) {
        Set<Integer> seen = new HashSet<>();
        for (int value : column) seen.add(value);
        return seen.size();
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
