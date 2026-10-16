/** Task 607 - Implement external-memory suffix array construction (disk-based). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * External prefix doubling: ranks live in a file, and each round sorts the suffixes by the pair
 * (rank[i], rank[i + step]) and reassigns ranks. The sort is an external LSD radix sort with 16
 * buckets per pass over the nibbles of both key fields, so every pass is a sequential record stream;
 * shifted keys rank[i + step] are materialised in one sequential pass per round and (position, rank)
 * pairs are re-sorted by position before the rank column is kept. The loop stops as soon as all
 * ranks are distinct; the sorted record file of that round is the suffix array.
 *
 * Documented simplifications of the disk model: "files" are lists of fixed 32-int blocks in memory,
 * reached only through single-block sequential readers/writers, with every block transfer counted; the
 * suffix array is read back only after construction, for verification (counted too), and the peak
 * resident block count is asserted small - the external-memory promise this simulation stands in for.
 * Verified: equality with an in-memory prefix-doubling suffix array and a naive comparison sort on
 * small texts, the permutation and strict order properties, the final rank file, and I/O counters.
 */
import java.util.*;
class SimulatedDisk {
    static final int BLOCK_INTS = 32;                       // 128 byte pages

    final Map<String, List<int[]>> blocks = new LinkedHashMap<>();
    final Map<String, Integer> lengths = new HashMap<>();
    long blockReads, blockWrites;
    int resident, peakResident;

    int size(String name) { return lengths.getOrDefault(name, 0); }

    void delete(String name) { blocks.remove(name); lengths.remove(name); }
    Reader read(String name) { return new Reader(name); }
    Writer write(String name) {                             // truncates, then appends
        blocks.put(name, new ArrayList<>());
        lengths.put(name, 0);
        return new Writer(name);
    }

    int[] readAll(String name) {                            // verification only, still counted
        int[] out = new int[size(name)];
        Reader reader = read(name);
        for (int i = 0; i < out.length; i++) out[i] = reader.next();
        reader.close();
        return out;
    }
    class Reader {
        private final String name;
        private final int[] buffer = new int[BLOCK_INTS];
        private int blockIndex = -1, offset, remaining;

        Reader(String name) {
            this.name = name;
            remaining = size(name);
            resident++;
            peakResident = Math.max(peakResident, resident);
        }

        boolean hasNext() { return remaining > 0; }

        int next() {
            if (remaining <= 0) throw new IllegalStateException("read past end of " + name);
            if (offset == BLOCK_INTS || blockIndex < 0) {
                System.arraycopy(blocks.get(name).get(++blockIndex), 0, buffer, 0, BLOCK_INTS);
                blockReads++;
                offset = 0;
            }
            remaining--;
            return buffer[offset++];
        }

        void close() { resident--; }
    }

    class Writer {
        private final String name;
        private final int[] buffer = new int[BLOCK_INTS];
        private int offset;

        Writer(String name) {
            this.name = name;
            resident++;
            peakResident = Math.max(peakResident, resident);
        }

        void put(int value) {
            buffer[offset++] = value;
            if (offset == BLOCK_INTS) flush(BLOCK_INTS);
        }

        void close() { if (offset > 0) flush(offset); resident--; }

        private void flush(int count) {
            int[] block = new int[BLOCK_INTS];
            System.arraycopy(buffer, 0, block, 0, count);
            blocks.get(name).add(block);
            lengths.put(name, lengths.get(name) + count);
            blockWrites++;
            offset = 0;
        }
    }
}

class ExternalSuffixArray {
    static final int RADIX_BITS = 4;
    static final int RADIX = 1 << RADIX_BITS;

    final SimulatedDisk disk = new SimulatedDisk();
    final String text;
    final int n;
    final int digits;                        // nibbles per key, enough to cover every rank value
    int rounds;
    long algorithmReads, algorithmWrites;
    int peakResidentBlocks;

    ExternalSuffixArray(String text) {
        this.text = text;
        n = text.length();
        digits = Math.max(1, 32 - Integer.numberOfLeadingZeros(Math.max(1, n)) + RADIX_BITS - 1)
                / RADIX_BITS;
        if (n == 0) return;
        SimulatedDisk.Writer textOut = disk.write("text");
        for (int i = 0; i < n; i++) textOut.put(text.charAt(i));
        textOut.close();
        int[] histogram = new int[256];                              // O(sigma) memory, sequential pass
        SimulatedDisk.Reader scan = disk.read("text");
        while (scan.hasNext()) histogram[scan.next()]++;
        scan.close();
        int[] code = new int[256];
        int distinct = 0;
        for (int c = 0; c < 256; c++) if (histogram[c] > 0) code[c] = ++distinct;
        String rankFile = "rank_0";
        SimulatedDisk.Writer ranks = disk.write(rankFile);
        SimulatedDisk.Reader textIn = disk.read("text");
        while (textIn.hasNext()) ranks.put(code[textIn.next()]);
        textIn.close();
        ranks.close();
        if (n == 1) {                                                // one suffix, no round needed
            SimulatedDisk.Writer order = disk.write("order");
            order.put(0); order.put(1); order.put(0);
            order.close();
            snapshot();
            return;
        }
        int step = 1;
        while ((rounds == 0 || distinct < n) && step < n) {
            buildShifted(rankFile, step);
            writeRecords(rankFile);
            radixSort("records", "sorted", 3, new int[]{1, 2});       // fields most significant first
            distinct = assignRanks("sorted");
            radixSort("updates", "byPosition", 2, new int[]{0});
            String nextRank = "rank_" + (rounds + 1);
            writeRanks("byPosition", nextRank);
            disk.delete(rankFile); disk.delete("updates"); disk.delete("byPosition");
            rankFile = nextRank;
            rounds++;
            step <<= 1;
        }
        if (distinct != n) throw new IllegalStateException("ranks did not become distinct");
        snapshot();
    }

    private void snapshot() {
        algorithmReads = disk.blockReads;
        algorithmWrites = disk.blockWrites;
        peakResidentBlocks = disk.peakResident;
    }

    /** shift[i] = rank[i + step], and 0 for the last `step` positions that look past the end. */
    private void buildShifted(String rankFile, int step) {
        SimulatedDisk.Writer out = disk.write("shift");
        SimulatedDisk.Reader in = disk.read(rankFile);
        for (int i = 0; i < step && in.hasNext(); i++) in.next();
        while (in.hasNext()) out.put(in.next());
        in.close();
        for (int i = 0; i < step; i++) out.put(0);
        out.close();
    }

    /** Records (position, rank[position], rank[position + step]) in position order. */
    private void writeRecords(String rankFile) {
        SimulatedDisk.Writer out = disk.write("records");
        SimulatedDisk.Reader ranks = disk.read(rankFile);
        SimulatedDisk.Reader shifted = disk.read("shift");
        for (int i = 0; i < n; i++) { int rank = ranks.next(), shift = shifted.next();
            out.put(i); out.put(rank); out.put(shift); }
        ranks.close(); shifted.close(); out.close();
    }

    /** Ranks rebuilt in sorted order as (position, rank) pairs; returns the number of distinct keys. */
    private int assignRanks(String sortedRecords) {
        SimulatedDisk.Writer out = disk.write("updates");
        SimulatedDisk.Reader in = disk.read(sortedRecords);
        int rank = 0, previousFirst = -1, previousSecond = -1;
        for (int i = 0; i < n; i++) {
            int position = in.next(), first = in.next(), second = in.next();
            if (first != previousFirst || second != previousSecond) {
                rank++; previousFirst = first; previousSecond = second;
            }
            out.put(position); out.put(rank);
        }
        in.close(); out.close();
        return rank;
    }

    /** Keep only the rank column, giving the position-indexed rank file used by the next round. */
    private void writeRanks(String pairs, String rankFile) {        SimulatedDisk.Writer out = disk.write(rankFile);
        SimulatedDisk.Reader in = disk.read(pairs);
        for (int i = 0; i < n; i++) { in.next(); out.put(in.next()); }
        in.close(); out.close();
    }

    /** External LSD radix sort of `width`-int records; each field is consumed nibble by nibble. */
    private void radixSort(String input, String output, int width, int[] fields) {
        String current = input;
        for (int f = fields.length - 1; f >= 0; f--)
            for (int d = 0; d < digits; d++) current = radixPass(current, output, width, fields[f], d * RADIX_BITS);
    }

    private String radixPass(String input, String output, int width, int field, int shift) {
        SimulatedDisk.Writer[] writers = new SimulatedDisk.Writer[RADIX];
        String[] bucketNames = new String[RADIX];
        for (int b = 0; b < RADIX; b++) writers[b] = disk.write(bucketNames[b] = "bucket_" + b);
        SimulatedDisk.Reader in = disk.read(input);
        while (in.hasNext()) {
            int[] record = new int[width];
            for (int k = 0; k < width; k++) record[k] = in.next();
            int bucket = (record[field] >>> shift) & (RADIX - 1);
            for (int k = 0; k < width; k++) writers[bucket].put(record[k]);
        }
        in.close();
        for (SimulatedDisk.Writer writer : writers) writer.close();
        SimulatedDisk.Writer sink = disk.write(output);              // input is fully read by now
        for (int b = 0; b < RADIX; b++) {
            SimulatedDisk.Reader bucketIn = disk.read(bucketNames[b]);
            while (bucketIn.hasNext()) sink.put(bucketIn.next());
            bucketIn.close();
            disk.delete(bucketNames[b]);
        }
        sink.close();
        return output;
    }
    /** The suffix array: positions of the final sorted record file, with the keys skipped. */
    int[] suffixArray() {
        if (n == 0) return new int[0];
        int[] order = new int[n];
        SimulatedDisk.Reader in = disk.read(n == 1 ? "order" : "sorted");
        for (int i = 0; i < n; i++) { order[i] = in.next(); in.next(); in.next(); }
        in.close();
        return order;
    }
    int[] ranks() { return disk.readAll("rank_" + rounds); }

    static int[] doubling(String s) {
        int n = s.length();
        if (n == 0) return new int[0];
        Integer[] sa = new Integer[n];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) { sa[i] = i; rank[i] = s.charAt(i); }
        for (int k = 1; ; k <<= 1) {
            final int step = k;
            Arrays.sort(sa, (a, b) -> {
                if (rank[a] != rank[b]) return Integer.compare(rank[a], rank[b]);
                int ra = a + step < n ? rank[a + step] : -1;
                int rb = b + step < n ? rank[b + step] : -1;
                return Integer.compare(ra, rb);
            });
            int[] next = new int[n];
            for (int i = 1; i < n; i++) {
                int prev = sa[i - 1], cur = sa[i];
                int pa = prev + step < n ? rank[prev + step] : -1;
                int pb = cur + step < n ? rank[cur + step] : -1;
                next[cur] = next[prev] + (rank[prev] != rank[cur] || pa != pb ? 1 : 0);
            }
            System.arraycopy(next, 0, rank, 0, n);
            if (rank[sa[n - 1]] == n - 1) break;
        }
        return Arrays.stream(sa).mapToInt(Integer::intValue).toArray();
    }

    static int[] naive(String s) {
        Integer[] sa = new Integer[s.length()];
        for (int i = 0; i < sa.length; i++) sa[i] = i;
        Arrays.sort(sa, (a, b) -> s.substring(a).compareTo(s.substring(b)));
        return Arrays.stream(sa).mapToInt(Integer::intValue).toArray();
    }
}

public class Solution {
    private static long state = 0x607F17E51DL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static String randomText(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) sb.append(alphabet.charAt((int) ((nextLong() >>> 33) % alphabet.length())));
        return sb.toString();
    }

    /** Construction must reproduce the suffix array, keep a small block working set and move I/O. */
    static void verify(String text) {
        ExternalSuffixArray external = new ExternalSuffixArray(text);
        int[] sa = external.suffixArray();
        check(Arrays.equals(sa, ExternalSuffixArray.doubling(text)), "external SA equals doubling: '" + text + "'");
        if (text.length() <= 32) {
            check(Arrays.equals(sa, ExternalSuffixArray.naive(text)), "external SA equals naive sort");
        }
        boolean[] seen = new boolean[text.length()];
        for (int i = 0; i < sa.length; i++) {
            check(sa[i] >= 0 && sa[i] < text.length() && !seen[sa[i]], "SA is a permutation");
            seen[sa[i]] = true;
            check(i == 0 || text.substring(sa[i - 1]).compareTo(text.substring(sa[i])) < 0,
                    "suffixes strictly increase: '" + text + "'");
        }
        check(text.isEmpty() || (external.algorithmReads > 0 && external.algorithmWrites > 0), "block I/O moved");
        check(external.peakResidentBlocks <= 20 && external.disk.size("text") == text.length(),
                "small working set and text round-trip");
        if (text.length() > 1) {
            int[] ranks = external.ranks(), inverse = new int[text.length()];
            for (int i = 0; i < sa.length; i++) inverse[sa[i]] = i;
            for (int i = 0; i < ranks.length; i++) check(ranks[i] == inverse[i] + 1, "rank file matches order");
        }
    }
    public static void main(String[] args) {
        check(new ExternalSuffixArray("").suffixArray().length == 0
                && Arrays.equals(new ExternalSuffixArray("a").suffixArray(), new int[]{0}), "empty and single");
        check(Arrays.equals(new ExternalSuffixArray("banana").suffixArray(), new int[]{5, 3, 1, 0, 4, 2}),
                "banana SA");
        String[] small = {"", "a", "b", "aa", "ab", "ba", "aaaa", "abab", "banana", "mississippi",
                          "abracadabra", "abcabcabc"};
        for (String text : small) verify(text);
        for (String alphabet : new String[]{"ab", "abc", "abcdefghij"}) {
            for (int trial = 0; trial < 5; trial++) verify(randomText(40, alphabet));
        }
        verify("a".repeat(150)); verify("ab".repeat(64));    // degenerate run, then a periodic text
        verify(randomText(200, "abcdefghijklmnopqrstuvwxyz"));
        verify(randomText(256, "ab"));
        ExternalSuffixArray large = new ExternalSuffixArray(randomText(512, "abcdef"));
        check(Arrays.equals(large.suffixArray(), ExternalSuffixArray.doubling(large.text)), "512 char text");
        System.out.println("rounds / digits     : " + large.rounds + " / " + large.digits);
        System.out.println("blocks read/written : " + large.algorithmReads + " / " + large.algorithmWrites
                + ", peak resident " + large.peakResidentBlocks);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
