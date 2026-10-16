/** Task 653 - Implement compressed column encoding schemes (RLE, dictionary) and decode. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * Two column-store encodings, each with a full encode / decode round trip:
 *
 *   RunLength   - maximal runs of equal values stored as (value, length) pairs. Cheap for sorted or
 *                 clustered columns; a column with k equal-value runs costs 2k ints instead of n.
 *   Dictionary  - a distinct-value table plus one small code per row (first-appearance order, so the
 *                 codes are dense: 0..distinct-1). Cheap for low-cardinality columns whatever the order.
 *
 * Decoding is lazy where it matters: valueAt(row) binary-searches the run containing the row, and
 * decodeRange(from, to) materialises only the requested window. The two schemes compose - RLE over a
 * dictionary code column is the classic "compressed column" pipeline and is checked to decode back to
 * the original column.
 *
 * Verified: decode(encode(x)) == x for empty / single / constant / alternating / clustered / pseudo
 * random columns, run counts against the number of value changes, positive and exhaustive run
 * lengths, dictionary size against the true distinct count, codes inside [0, distinct), window
 * decodes against the source column, and rejection of an out-of-range dictionary code.
 */
import java.util.*;

class RunLength {
    final int[] values;
    final int[] lengths;
    final int runs;

    private RunLength(int[] values, int[] lengths, int runs) {
        this.values = values;
        this.lengths = lengths;
        this.runs = runs;
    }

    static RunLength encode(int[] column) {
        if (column.length == 0) return new RunLength(new int[0], new int[0], 0);
        int runCount = 1;
        for (int i = 1; i < column.length; i++) if (column[i] != column[i - 1]) runCount++;
        int[] values = new int[runCount];
        int[] lengths = new int[runCount];
        int run = 0;
        values[0] = column[0];
        lengths[0] = 1;
        for (int i = 1; i < column.length; i++) {
            if (column[i] == column[i - 1]) {
                lengths[run]++;
            } else {
                run++;
                values[run] = column[i];
                lengths[run] = 1;
            }
        }
        return new RunLength(values, lengths, runCount);
    }

    int[] decode() {
        int[] out = new int[size()];
        int at = 0;
        for (int run = 0; run < runs; run++) {
            for (int k = 0; k < lengths[run]; k++) out[at++] = values[run];
        }
        return out;
    }

    int size() {
        int total = 0;
        for (int run = 0; run < runs; run++) total += lengths[run];
        return total;
    }

    /** Index of the run containing `row`; binary search over the prefix sums of the run lengths. */
    private int runOf(int row) {
        if (row < 0 || row >= size()) throw new IndexOutOfBoundsException("row " + row);
        int lo = 0, hi = runs - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (prefixEnd(mid) <= row) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    private int prefixEnd(int run) {
        int end = 0;
        for (int i = 0; i <= run; i++) end += lengths[i];
        return end;
    }

    int valueAt(int row) {
        return values[runOf(row)];
    }

    int[] decodeRange(int from, int to) {
        if (from < 0 || to > size() || from > to) throw new IndexOutOfBoundsException(from + ".." + to);
        int[] out = new int[to - from];
        int at = 0;
        for (int run = 0; run < runs && at < out.length; run++) {
            int runStart = prefixEnd(run) - lengths[run];
            int runEnd = prefixEnd(run);
            for (int row = Math.max(from, runStart); row < Math.min(to, runEnd); row++) out[at++] = values[run];
        }
        return out;
    }
}

class DictionaryEncoding {
    final int[] dictionary;   // distinct values, first-appearance order
    final int[] codes;        // one code per row, 0..dictionary.length-1

    private DictionaryEncoding(int[] dictionary, int[] codes) {
        this.dictionary = dictionary;
        this.codes = codes;
    }

    static DictionaryEncoding encode(int[] column) {
        Map<Integer, Integer> codeOf = new LinkedHashMap<>();
        int[] codes = new int[column.length];
        for (int row = 0; row < column.length; row++) {
            Integer code = codeOf.get(column[row]);
            if (code == null) {
                code = codeOf.size();
                codeOf.put(column[row], code);
            }
            codes[row] = code;
        }
        int[] dictionary = new int[codeOf.size()];
        for (Map.Entry<Integer, Integer> entry : codeOf.entrySet()) dictionary[entry.getValue()] = entry.getKey();
        return new DictionaryEncoding(dictionary, codes);
    }

    int decodeCode(int code) {
        if (code < 0 || code >= dictionary.length) throw new IllegalArgumentException("code out of range: " + code);
        return dictionary[code];
    }

    int[] decode() {
        int[] out = new int[codes.length];
        for (int row = 0; row < codes.length; row++) out[row] = decodeCode(codes[row]);
        return out;
    }

    int[] decodeRange(int from, int to) {
        if (from < 0 || to > codes.length || from > to) throw new IndexOutOfBoundsException(from + ".." + to);
        int[] out = new int[to - from];
        for (int row = from; row < to; row++) out[row - from] = decodeCode(codes[row]);
        return out;
    }

    /** Codes are dense: the first appearance of every distinct value gets the next free code. */
    boolean codesAreDense() {
        boolean[] seen = new boolean[dictionary.length];
        int next = 0;
        for (int code : codes) {
            if (code < 0 || code >= dictionary.length) return false;
            if (!seen[code]) {
                if (code != next) return false;
                seen[code] = true;
                next++;
            }
        }
        return next == dictionary.length;
    }
}

public class Solution {
    private static long state = 0x41C64E6DA5A1F1B3L;

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

    static int distinctCount(int[] column) {
        Set<Integer> seen = new HashSet<>();
        for (int value : column) seen.add(value);
        return seen.size();
    }

    static int expectedRuns(int[] column) {
        if (column.length == 0) return 0;
        int runs = 1;
        for (int i = 1; i < column.length; i++) if (column[i] != column[i - 1]) runs++;
        return runs;
    }

    static int[] sortedColumn(int[] column) {
        int[] copy = column.clone();
        Arrays.sort(copy);
        return copy;
    }

    public static void main(String[] args) {
        int[] shapes = {0, 1, 2, 3, 17};

        // --- degenerate columns -------------------------------------------------
        int[] empty = new int[0];
        check(RunLength.encode(empty).decode().length == 0, "RLE empty round trip");
        check(RunLength.encode(empty).runs == 0, "RLE empty has no runs");
        check(DictionaryEncoding.encode(empty).dictionary.length == 0, "dictionary empty");
        check(DictionaryEncoding.encode(empty).codes.length == 0, "dictionary empty codes");

        int[] single = {42};
        check(Arrays.equals(RunLength.encode(single).decode(), single), "RLE single row");
        check(RunLength.encode(single).runs == 1, "RLE single run");
        check(DictionaryEncoding.encode(single).dictionary.length == 1, "dictionary single entry");

        int[] constant = new int[500];
        Arrays.fill(constant, -3);
        RunLength constantRuns = RunLength.encode(constant);
        check(constantRuns.runs == 1 && constantRuns.lengths[0] == 500, "constant column is a single run");
        check(Arrays.equals(constantRuns.decode(), constant), "constant column round trip");
        check(2 * constantRuns.runs < constant.length / 10, "constant column compresses to a few ints");
        check(Arrays.equals(constantRuns.decodeRange(120, 130), Arrays.copyOfRange(constant, 120, 130)),
                "constant column window decode");

        int[] alternating = new int[64];
        for (int i = 0; i < alternating.length; i++) alternating[i] = i % 2 == 0 ? 1 : 0;
        check(RunLength.encode(alternating).runs == 64, "alternating column has one run per row");
        check(Arrays.equals(RunLength.encode(alternating).decode(), alternating), "alternating round trip");

        // --- run count formula and run bookkeeping -----------------------------
        for (int shape : shapes) {
            int[] column = randomColumn(shape, 4);
            RunLength runs = RunLength.encode(column);
            check(runs.runs == expectedRuns(column), "run count for size " + shape);
            check(Arrays.equals(runs.decode(), column), "RLE round trip for size " + shape);
            int total = 0;
            for (int run = 0; run < runs.runs; run++) {
                check(runs.lengths[run] > 0, "run lengths are positive");
                total += runs.lengths[run];
                if (run > 0) check(runs.values[run] != runs.values[run - 1], "adjacent runs differ");
            }
            check(total == shape, "run lengths sum to the column size");
        }

        // --- clustering really drives the RLE win ------------------------------
        int[] shuffled = randomColumn(400, 3);
        int[] clustered = sortedColumn(shuffled);
        int shuffledRuns = RunLength.encode(shuffled).runs;
        int clusteredRuns = RunLength.encode(clustered).runs;
        check(clusteredRuns == distinctCount(clustered),
                "sorted column has one run per distinct value, got " + clusteredRuns);
        check(clusteredRuns < shuffledRuns, "clustering reduces runs: " + clusteredRuns + " < " + shuffledRuns);
        check(Arrays.equals(RunLength.encode(clustered).decode(), clustered), "clustered round trip");

        // --- dictionary encoding ---------------------------------------------
        int[] column = randomColumn(300, 6);
        DictionaryEncoding dictionary = DictionaryEncoding.encode(column);
        check(dictionary.dictionary.length == distinctCount(column), "dictionary size is the distinct count");
        check(dictionary.codes.length == column.length, "one code per row");
        check(Arrays.equals(dictionary.decode(), column), "dictionary round trip");
        check(dictionary.codesAreDense(), "codes are dense and in first-appearance order");
        for (int code : dictionary.codes) check(code >= 0 && code < dictionary.dictionary.length, "code in range");
        for (int row = 0; row < column.length; row++) {
            check(dictionary.decodeCode(dictionary.codes[row]) == column[row], "code decodes at row " + row);
        }

        // first-appearance order: dictionary[k] is the value first seen in row k
        int[] firstSeenRow = new int[dictionary.dictionary.length];
        Arrays.fill(firstSeenRow, -1);
        for (int row = 0; row < column.length; row++) {
            int code = dictionary.codes[row];
            if (firstSeenRow[code] < 0) firstSeenRow[code] = row;
        }
        boolean increasing = true;
        for (int k = 1; k < firstSeenRow.length; k++) if (firstSeenRow[k - 1] >= firstSeenRow[k]) increasing = false;
        check(increasing, "first appearance rows increase with the code");

        // --- window decoding against the source column -------------------------
        for (int shape : new int[]{1, 40, 300}) {
            int[] data = randomColumn(shape, 5);
            RunLength runs = RunLength.encode(data);
            DictionaryEncoding dict = DictionaryEncoding.encode(data);
            for (int from = 0; from <= shape; from += Math.max(1, shape / 7)) {
                int to = Math.min(shape, from + Math.max(1, shape / 5));
                int[] expected = Arrays.copyOfRange(data, from, to);
                check(Arrays.equals(runs.decodeRange(from, to), expected), "RLE window " + from + ".." + to);
                check(Arrays.equals(dict.decodeRange(from, to), expected), "dictionary window " + from + ".." + to);
            }
            for (int row = 0; row < shape; row += Math.max(1, shape / 9)) {
                check(runs.valueAt(row) == data[row], "RLE valueAt row " + row);
            }
        }

        // --- pipelines and rejected input --------------------------------------
        int[] pipelineSource = sortedColumn(randomColumn(240, 5));
        DictionaryEncoding stage1 = DictionaryEncoding.encode(pipelineSource);
        RunLength stage2 = RunLength.encode(stage1.codes);
        int[] codesBack = stage2.decode();
        check(Arrays.equals(codesBack, stage1.codes), "RLE of dictionary codes round trips");
        int[] pipelineBack = new int[codesBack.length];
        for (int row = 0; row < codesBack.length; row++) pipelineBack[row] = stage1.decodeCode(codesBack[row]);
        check(Arrays.equals(pipelineBack, pipelineSource), "dictionary + RLE pipeline decodes the column");
        check(stage1.dictionary.length + stage1.codes.length + 2 * stage2.runs < 3 * pipelineSource.length,
                "pipeline is smaller than one int per value per stage");

        boolean rejected = false;
        try {
            dictionary.decodeCode(dictionary.dictionary.length);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "out-of-range dictionary code must be rejected");
        rejected = false;
        try {
            RunLength.encode(pipelineSource).decodeRange(5, 4);
        } catch (IndexOutOfBoundsException expected) {
            rejected = true;
        }
        check(rejected, "inverted decode window must be rejected");
        rejected = false;
        try {
            RunLength.encode(pipelineSource).valueAt(pipelineSource.length);
        } catch (IndexOutOfBoundsException expected) {
            rejected = true;
        }
        check(rejected, "valueAt past the end must be rejected");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
