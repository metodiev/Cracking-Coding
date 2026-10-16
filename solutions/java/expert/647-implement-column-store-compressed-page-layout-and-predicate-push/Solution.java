/** Task 647 - Implement column-store compressed page layout and predicate pushdown simulation. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * A column file is a sequence of fixed-size pages, one per column, and every column is stored twice so
 * the two compressed layouts can be compared:
 *
 *   RLE         - each page keeps a list of (value, runLength) runs, so a page holding one repeated
 *                 value costs a single run;
 *   dictionary  - each page keeps the column's sorted dictionary plus one code per row, so a predicate
 *                 is evaluated by comparing codes against the dictionary (no per-row value decode).
 *
 * Each page additionally carries a zone map (min, max). Predicate pushdown works in two stages: a page
 * whose zone map does not intersect the predicate's range is skipped without being touched at all, and
 * the surviving pages are filtered *in their encoded form* - whole runs are accepted or rejected with
 * one comparison, and dictionary pages resolve the matching code range once per page and then only
 * compare integers. Only the matching row ids are produced; the payload columns are materialised later
 * ("late materialisation"), and the check compares that materialised result with the naive row store.
 *
 * Simplified (deliberately): fixed page sizes with no bit-packing, no delta/frame-of-reference
 * encoding, no per-page dictionaries (one dictionary per column), no compression-ratio accounting and
 * single-column zone maps.
 *
 * Verified: both decoders reproduce the original column exactly; the dictionary is exactly the distinct
 * value set and every zone map is the true per-page min/max (both re-derived by brute force); pushdown
 * returns exactly the same row ids as a naive row-by-row scan for nine different predicates including
 * empty and full ranges; the number of decoded pages equals the number of pages whose zone map
 * intersects the range (an independent recomputation); a selective range really skips pages, and the
 * encoded evaluation really does fewer comparisons than the naive scan; and the degenerate cases
 * (empty column, single row, out-of-range predicate) behave.
 */
import java.util.*;

class ColumnStore {

    static final class Column {
        final String name;
        final int[] values;
        final int pageRows, pageCount;
        final List<List<int[]>> rlePages = new ArrayList<>();
        final int[] dictionary;
        final Map<Integer, Integer> codeOf = new HashMap<>();
        final List<int[]> codePages = new ArrayList<>();
        final List<int[]> zoneMaps = new ArrayList<>();

        Column(String name, int[] values, int pageRows) {
            this.name = name;
            this.values = values.clone();
            this.pageRows = pageRows;
            this.pageCount = (values.length + pageRows - 1) / pageRows;
            TreeSet<Integer> distinct = new TreeSet<>();
            for (int value : values) distinct.add(value);
            dictionary = new int[distinct.size()];
            int index = 0;
            for (int value : distinct) {
                dictionary[index] = value;
                codeOf.put(value, index++);
            }
            for (int page = 0; page < pageCount; page++) {
                int from = page * pageRows;
                int to = Math.min(from + pageRows, values.length);
                List<int[]> runs = new ArrayList<>();
                int[] codes = new int[to - from];
                int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
                for (int i = from; i < to; i++) {
                    int value = values[i];
                    codes[i - from] = codeOf.get(value);
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                    if (!runs.isEmpty() && runs.get(runs.size() - 1)[0] == value) {
                        runs.get(runs.size() - 1)[1]++;
                    } else {
                        runs.add(new int[]{value, 1});
                    }
                }
                rlePages.add(runs);
                codePages.add(codes);
                zoneMaps.add(new int[]{min, max});
            }
        }

        int[] decodeRle() {
            int[] out = new int[values.length];
            int cursor = 0;
            for (List<int[]> runs : rlePages) {
                for (int[] run : runs) {
                    for (int i = 0; i < run[1]; i++) out[cursor++] = run[0];
                }
            }
            return out;
        }

        int[] decodeDictionary() {
            int[] out = new int[values.length];
            int cursor = 0;
            for (int[] codes : codePages) {
                for (int code : codes) out[cursor++] = dictionary[code];
            }
            return out;
        }

        int totalRuns() {
            int runs = 0;
            for (List<int[]> page : rlePages) runs += page.size();
            return runs;
        }
    }

    static final class Stats {
        final List<Integer> rowIds = new ArrayList<>();
        int pagesTotal, pagesSkipped, pagesDecoded;
        long runComparisons, codeComparisons, valueComparisons, rowsEvaluated, rowsResolvedByRun;
        boolean dictionaryEncoded;

        String summary() {
            return (dictionaryEncoded ? "dict" : "rle") + " rows=" + rowIds.size() + " pages " + pagesDecoded
                    + "/" + pagesTotal + " pagesElided=" + pagesSkipped + " rowComparisons=" + valueComparisons
                    + " runComparisons=" + runComparisons + " codeComparisons=" + codeComparisons;
        }

        boolean rowIdsAscending() {
            for (int i = 1; i < rowIds.size(); i++) if (rowIds.get(i) <= rowIds.get(i - 1)) return false;
            return true;
        }
    }

    /** Naive row-store scan: one value comparison per row. */
    static Stats naiveScan(Column column, int low, int high) {
        Stats stats = new Stats();
        stats.pagesTotal = column.pageCount;
        for (int i = 0; i < column.values.length; i++) {
            stats.valueComparisons++;
            if (column.values[i] >= low && column.values[i] <= high) stats.rowIds.add(i);
        }
        stats.rowsEvaluated = column.values.length;
        return stats;
    }

    /** Zone maps + encoded predicate evaluation; only matching row ids are produced. */
    static Stats pushdown(Column column, int low, int high) {
        Stats stats = new Stats();
        stats.dictionaryEncoded = true;
        stats.pagesTotal = column.pageCount;
        int codeLow = 0, codeHigh = column.dictionary.length - 1;
        while (codeLow < column.dictionary.length && column.dictionary[codeLow] < low) codeLow++;
        while (codeHigh >= 0 && column.dictionary[codeHigh] > high) codeHigh--;
        for (int page = 0; page < column.pageCount; page++) {
            int[] zone = column.zoneMaps.get(page);
            if (zone[1] < low || zone[0] > high) {
                stats.pagesSkipped++;
                continue;
            }
            stats.pagesDecoded++;
            if (codeLow > codeHigh) {
                stats.codeComparisons += column.codePages.get(page).length;
                continue;
            }
            int from = page * column.pageRows;
            int[] codes = column.codePages.get(page);
            for (int offset = 0; offset < codes.length; offset++) {
                stats.codeComparisons++;
                if (codes[offset] >= codeLow && codes[offset] <= codeHigh) stats.rowIds.add(from + offset);
            }
        }
        return stats;
    }

    /** The same pushdown, but evaluating whole RLE runs with one comparison per run. */
    static Stats pushdownRuns(Column column, int low, int high) {
        Stats stats = new Stats();
        stats.dictionaryEncoded = false;
        stats.pagesTotal = column.pageCount;
        for (int page = 0; page < column.pageCount; page++) {
            int[] zone = column.zoneMaps.get(page);
            if (zone[1] < low || zone[0] > high) {
                stats.pagesSkipped++;
                continue;
            }
            stats.pagesDecoded++;
            int from = page * column.pageRows;
            int row = from;
            for (int[] run : column.rlePages.get(page)) {
                stats.runComparisons++;
                boolean matches = run[0] >= low && run[0] <= high;
                if (matches) {
                    for (int i = 0; i < run[1]; i++) stats.rowIds.add(row + i);
                    stats.rowsResolvedByRun += run[1];
                }
                stats.valueComparisons++;
                row += run[1];
            }
        }
        return stats;
    }

    static List<Integer> materialise(Column projection, List<Integer> rowIds) {
        List<Integer> out = new ArrayList<>();
        for (int row : rowIds) out.add(projection.values[row]);
        return out;
    }

    static List<Integer> naiveResult(Column column, int low, int high) {
        return naiveScan(column, low, high).rowIds;
    }
}

class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

public class Solution {

    public static void main(String[] args) {
        int[] sorted = new int[256];
        for (int i = 0; i < sorted.length; i++) sorted[i] = i / 8;
        ColumnStore.Column rleColumn = new ColumnStore.Column("ts", sorted, 8);

        check(Arrays.equals(rleColumn.decodeRle(), sorted), "RLE decoder reproduces the column");
        check(Arrays.equals(rleColumn.decodeDictionary(), sorted), "dictionary decoder reproduces the column");
        check(rleColumn.totalRuns() == 32, "one run per page: " + rleColumn.totalRuns() + " runs for 256 rows");
        check(rleColumn.dictionary.length == 32, "dictionary holds the distinct values: "
                + rleColumn.dictionary.length);

        int[] lowCardinality = new int[200];
        SplitMix64 rng = new SplitMix64(13579L);
        for (int i = 0; i < lowCardinality.length; i++) lowCardinality[i] = rng.nextInt(5);
        ColumnStore.Column dictColumn = new ColumnStore.Column("status", lowCardinality, 16);
        check(Arrays.equals(dictColumn.decodeRle(), lowCardinality), "RLE round trip on a random column");
        check(Arrays.equals(dictColumn.decodeDictionary(), lowCardinality), "dictionary round trip");
        check(dictColumn.dictionary.length == 5, "five distinct statuses: " + dictColumn.dictionary.length);

        boolean zonesCorrect = true;
        for (int page = 0; page < dictColumn.pageCount; page++) {
            int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
            for (int i = page * 16; i < Math.min((page + 1) * 16, lowCardinality.length); i++) {
                min = Math.min(min, lowCardinality[i]);
                max = Math.max(max, lowCardinality[i]);
            }
            int[] zone = dictColumn.zoneMaps.get(page);
            if (zone[0] != min || zone[1] != max) zonesCorrect = false;
        }
        check(zonesCorrect, "every zone map is the true per-page min/max");

        int[][] predicates = {{0, 3}, {4, 4}, {7, 11}, {5, 5}, {0, 0}, {100, 200}, {-5, 1}, {30, 31}, {0, 31}};
        for (int[] predicate : predicates) {
            int low = predicate[0], high = predicate[1];
            ColumnStore.Stats naive = ColumnStore.naiveScan(rleColumn, low, high);
            ColumnStore.Stats encoded = ColumnStore.pushdown(rleColumn, low, high);
            ColumnStore.Stats runs = ColumnStore.pushdownRuns(rleColumn, low, high);
            check(naive.rowIds.equals(encoded.rowIds), "dictionary pushdown matches the scan for ["
                    + low + "," + high + "]");
            check(naive.rowIds.equals(runs.rowIds), "RLE pushdown matches the scan for [" + low + "," + high + "]");
        }
        for (int[] predicate : new int[][]{{0, 2}, {3, 3}, {0, 4}, {5, 9}, {-1, -1}, {9, 9}, {0, 99}}) {
            int low = predicate[0], high = predicate[1];
            check(ColumnStore.naiveScan(dictColumn, low, high).rowIds
                            .equals(ColumnStore.pushdown(dictColumn, low, high).rowIds),
                    "low-cardinality pushdown matches for [" + low + "," + high + "]");
        }

        ColumnStore.Stats narrow = ColumnStore.pushdown(rleColumn, 7, 9);
        int overlappingPages = 0;
        for (int[] zone : rleColumn.zoneMaps) if (zone[1] >= 7 && zone[0] <= 9) overlappingPages++;
        check(narrow.pagesDecoded == overlappingPages,
                "decoded pages equal the pages whose zone map overlaps: " + narrow.pagesDecoded
                        + " vs " + overlappingPages);
        check(narrow.pagesSkipped == rleColumn.pageCount - overlappingPages,
                "the remaining pages were skipped: " + narrow.pagesSkipped);
        check(narrow.pagesSkipped > 0 && narrow.pagesDecoded < rleColumn.pageCount,
                "a selective range really skips pages");
        check(narrow.rowIds.size() == 24, "range [7,9] matches three runs of eight: " + narrow.rowIds.size());

        ColumnStore.Stats narrowRuns = ColumnStore.pushdownRuns(rleColumn, 7, 9);
        ColumnStore.Stats narrowNaive = ColumnStore.naiveScan(rleColumn, 7, 9);
        check(narrowRuns.valueComparisons < narrowNaive.valueComparisons,
                "run-level evaluation compares fewer values: " + narrowRuns.valueComparisons + " vs "
                        + narrowNaive.valueComparisons);
        check(narrowRuns.rowsResolvedByRun == 24, "whole runs are accepted without per-row work: "
                + narrowRuns.rowsResolvedByRun);
        check(narrow.rowIdsAscending(), "row ids come back in ascending order");
        check(narrow.codeComparisons == 24, "only the decoded pages are touched row by row: "
                + narrow.codeComparisons);

        ColumnStore.Stats wide = ColumnStore.pushdown(rleColumn, 0, 31);
        check(wide.pagesSkipped == 0 && wide.pagesDecoded == rleColumn.pageCount,
                "a range covering everything skips nothing");
        check(wide.rowIds.size() == sorted.length, "and matches every row");
        ColumnStore.Stats outside = ColumnStore.pushdown(rleColumn, 100, 200);
        check(outside.rowIds.isEmpty() && outside.pagesDecoded == 0,
                "a range outside the zone maps decodes nothing: " + outside.pagesSkipped + " pages elided");

        ColumnStore.Stats filtered = ColumnStore.pushdown(dictColumn, 0, 2);
        long selectiveRows = dictColumn.values.length;
        check(filtered.rowIds.size() > 0 && filtered.rowIds.size() < selectiveRows,
                "the status predicate is selective: " + filtered.rowIds.size() + " of " + selectiveRows);
        check(ColumnStore.materialise(dictColumn, filtered.rowIds).size() == filtered.rowIds.size(),
                "materialised rows line up with the row ids");
        List<Integer> expectedPayloads = new ArrayList<>();
        for (int row : ColumnStore.naiveResult(dictColumn, 0, 2)) expectedPayloads.add(dictColumn.values[row]);
        check(ColumnStore.materialise(dictColumn, filtered.rowIds).equals(expectedPayloads),
                "late materialisation equals the naive projection");
        for (int value : ColumnStore.materialise(dictColumn, filtered.rowIds)) {
            check(value >= 0 && value <= 2, "late materialisation returns only matching payloads");
        }

        ColumnStore.Column empty = new ColumnStore.Column("empty", new int[0], 8);
        check(Arrays.equals(empty.decodeRle(), new int[0]), "an empty column decodes to nothing");
        check(ColumnStore.pushdown(empty, 0, 10).rowIds.isEmpty(), "an empty column matches nothing");
        check(empty.pageCount == 0, "an empty column has no pages");
        ColumnStore.Column single = new ColumnStore.Column("one", new int[]{5}, 8);
        check(Arrays.equals(single.decodeRle(), new int[]{5}) && single.pageCount == 1, "single-row column");
        check(ColumnStore.pushdown(single, 5, 5).rowIds.equals(List.of(0)), "single row matches its own range");
        check(ColumnStore.pushdown(single, 6, 9).rowIds.isEmpty(), "and misses a disjoint range");
        check(ColumnStore.pushdown(single, 5, 5).pagesDecoded == 1, "the single page is decoded");

        boolean rejected = false;
        try {
            new ColumnStore.Column("bad", null, 8);
        } catch (NullPointerException expected2) {
            rejected = true;
        }
        check(rejected, "a null column is rejected");

        System.out.println("RLE pages: " + rleColumn.totalRuns() + " runs for " + sorted.length
                + " rows, dictionary size " + rleColumn.dictionary.length);
        System.out.println("range [7,9]: " + narrow.summary() + " vs naive " + narrowNaive.valueComparisons
                + " comparisons; run mode " + narrowRuns.valueComparisons + " comparisons, "
                + narrowRuns.rowsResolvedByRun + " rows resolved by whole runs");
        System.out.println("low-cardinality predicate [0,2]: " + filtered.summary());
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
