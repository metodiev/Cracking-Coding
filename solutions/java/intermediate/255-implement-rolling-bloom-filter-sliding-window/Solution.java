/** Task 255 - Implement rolling bloom filter sliding window. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
public class Solution {
    static final int SLICES = 4;
    static final int BITS_PER_SLICE = 64;
    static final int HASHES = 3;
    // Deterministic expectation for the "loaded" scenario in main (cross-language check).
    static final int EXPECTED_HITS = 4;

    static int fnv1a(String text) {
        int h = 0x811C9DC5;
        for (int i = 0; i < text.length(); i++) {
            h ^= text.charAt(i);
            h *= 0x01000193;
        }
        return h;
    }

    static int fmix32(int x) {
        x ^= x >>> 16;
        x *= 0x85EBCA6B;
        x ^= x >>> 13;
        x *= 0xC2B2AE35;
        x ^= x >>> 16;
        return x;
    }

    static int[] positions(String item, int bits, int hashes) {
        long h1 = fmix32(fnv1a(item)) & 0xFFFFFFFFL;
        long h2 = (fmix32(fnv1a(item + "#")) & 0xFFFFFFFFL) | 1L;
        int[] out = new int[hashes];
        for (int i = 0; i < hashes; i++) out[i] = (int) ((h1 + i * h2) % bits);
        return out;
    }

    /** Every insert writes into ALL slices (OR query); every step clears one slice. */
    static class RollingBloom {
        final boolean[][] slices;
        final int bits;
        final int hashes;
        int current;

        RollingBloom(int sliceCount, int bits, int hashes) {
            this.slices = new boolean[sliceCount][bits];
            this.bits = bits;
            this.hashes = hashes;
        }

        void insert(String item) {
            int[] probes = positions(item, bits, hashes);
            for (boolean[] slice : slices) {
                for (int p : probes) slice[p] = true;
            }
        }

        void step() {
            current = (current + 1) % slices.length;
            slices[current] = new boolean[bits];
        }

        boolean contains(String item) {
            int[] probes = positions(item, bits, hashes);
            for (boolean[] slice : slices) {
                boolean all = true;
                for (int p : probes) {
                    if (!slice[p]) {
                        all = false;
                        break;
                    }
                }
                if (all) return true;
            }
            return false;
        }

        int popcount(int index) {
            int count = 0;
            for (boolean b : slices[index]) if (b) count++;
            return count;
        }
    }

    public static void main(String[] args) {
        RollingBloom rb = new RollingBloom(SLICES, BITS_PER_SLICE, HASHES);
        check(!rb.contains("alpha"), "empty window: guaranteed miss");

        rb.insert("alpha");
        check(rb.contains("alpha"), "just inserted");
        for (int i = 0; i < SLICES; i++) check(rb.popcount(i) == HASHES, "3 distinct probes per slice");

        rb.step();
        check(rb.contains("alpha"), "age 1: still inside the window");
        int zeroed = 0;
        for (int i = 0; i < SLICES; i++) if (rb.popcount(i) == 0) zeroed++;
        check(zeroed == 1, "exactly one slice rotated");

        RollingBloom ghost = new RollingBloom(SLICES, BITS_PER_SLICE, HASHES);
        ghost.insert("solo");
        for (int i = 0; i < SLICES; i++) ghost.step();
        check(!ghost.contains("solo"), "window passed: expired");
        for (int i = 0; i < SLICES; i++) check(ghost.popcount(i) == 0, "all slices cleared");

        RollingBloom keep = new RollingBloom(SLICES, BITS_PER_SLICE, HASHES);
        keep.insert("keep");
        for (int i = 0; i < SLICES - 1; i++) keep.step();
        check(keep.contains("keep"), "age 3: still inside the window");
        keep.step();
        check(!keep.contains("keep"), "age 4: every slice was cleared");

        RollingBloom timeline = new RollingBloom(SLICES, BITS_PER_SLICE, HASHES);
        String[] history = new String[9];
        int[] insertedAt = new int[9];
        int tracked = 0;
        int stepNo = 0;

        history[tracked] = "alpha";
        insertedAt[tracked] = stepNo;
        tracked++;
        timeline.insert("alpha");
        for (int i = 1; i <= 8; i++) {
            timeline.step();
            stepNo++;
            String item = "w" + i;
            history[tracked] = item;
            insertedAt[tracked] = stepNo;
            tracked++;
            timeline.insert(item);
            for (int j = 0; j < tracked; j++) {                 // never a false negative
                if (stepNo - insertedAt[j] <= SLICES - 1) {
                    check(timeline.contains(history[j]), "in-window miss for " + history[j]);
                }
            }
        }
        check(!timeline.contains("alpha"), "8 steps old: fully evicted");

        RollingBloom loaded = new RollingBloom(SLICES, BITS_PER_SLICE, HASHES);
        for (int i = 0; i < 8; i++) loaded.insert("member-" + i);
        for (int i = 0; i < 8; i++) check(loaded.contains("member-" + i), "loaded member " + i);
        int hits = 0;
        for (int i = 0; i < 100; i++) if (loaded.contains("q" + i)) hits++;
        check(hits == EXPECTED_HITS, "loaded false positives agree across languages");
        check(hits >= 1, "false positives do happen: expected, never false negatives");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
