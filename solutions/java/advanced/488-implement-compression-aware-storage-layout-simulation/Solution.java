/** Task 488 - Implement compression-aware storage layout simulation. (Tier: Advanced, CTCI Ch 14)
 *
 * Value blocks (32 non-negative 16-bit ints each; the last block may be shorter) are encoded with
 * the cheaper of two schemes, chosen by exact byte count with a documented tie rule (RLE first):
 * - RLE     : header 2 bytes (mode=0, run count) + 4 bytes per run (value, run length).
 * - BITPACK : header 2 bytes (mode=1, bit width w) + ceil(count * w / 8) bytes, LSB-first packing,
 *             where w = max(1, max bit length of the block's values). For values < 2^31 bit packing
 *             with w <= 31 bits is never worse than a raw 32-bit copy, so raw mode is not needed.
 *
 * "Compression-aware layout": blocks are grouped by similarity - a stable sort by bit width puts
 * similar blocks next to each other, and consecutive blocks sharing a width cost one 2-byte group
 * header for the whole run instead of per-block overhead. storedBytes counts group headers plus
 * per-block payloads only.
 *
 * The test asserts exact round-trips on skewed, mixed and hand blocks, checks the mode actually
 * chosen per block, and bounds stored bytes on skewed data: for 1024 ints in {0..3} the layout
 * stores 322 bytes vs 4096 raw (<= raw/3), matching the pinned exact count.
 */
import java.util.*;

class BlockInfo {
    final int mode;
    final int width; // -1 for RLE
    final byte[] payload;
    final int runs; // -1 for BITPACK

    BlockInfo(int mode, int width, byte[] payload, int runs) {
        this.mode = mode;
        this.width = width;
        this.payload = payload;
        this.runs = runs;
    }
}

class LayoutEntry {
    final int index;
    final int group;
    final BlockInfo info;

    LayoutEntry(int index, int group, BlockInfo info) {
        this.index = index;
        this.group = group;
        this.info = info;
    }
}

class Layout {
    final List<LayoutEntry> entries;
    final int storedBytes;
    final int groups;

    Layout(List<LayoutEntry> entries, int storedBytes, int groups) {
        this.entries = entries;
        this.storedBytes = storedBytes;
        this.groups = groups;
    }
}

public class Solution {
    static final int MODE_RLE = 0;
    static final int MODE_BITPACK = 1;
    static final int HEADER = 2;

    static int bitLength(int value) {
        return 32 - Integer.numberOfLeadingZeros(value);
    }

    static BlockInfo encodeBlock(int[] values) {
        List<int[]> runs = new ArrayList<>();
        for (int value : values) {
            if (!runs.isEmpty() && runs.get(runs.size() - 1)[0] == value) {
                runs.get(runs.size() - 1)[1]++;
            } else {
                runs.add(new int[]{value, 1});
            }
        }
        int rleSize = HEADER + 4 * runs.size();
        int width = 1;
        for (int value : values) width = Math.max(width, bitLength(value));
        int bitpackSize = HEADER + (values.length * width + 7) / 8;
        if (rleSize <= bitpackSize) {
            byte[] payload = new byte[rleSize];
            payload[0] = MODE_RLE;
            payload[1] = (byte) runs.size();
            int at = HEADER;
            for (int[] run : runs) {
                payload[at++] = (byte) (run[0] & 0xFF);
                payload[at++] = (byte) ((run[0] >> 8) & 0xFF);
                payload[at++] = (byte) (run[1] & 0xFF);
                payload[at++] = (byte) ((run[1] >> 8) & 0xFF);
            }
            return new BlockInfo(MODE_RLE, -1, payload, runs.size());
        }
        byte[] payload = new byte[bitpackSize];
        payload[0] = MODE_BITPACK;
        payload[1] = (byte) width;
        int acc = 0, bits = 0, at = HEADER;
        for (int value : values) {
            acc |= value << bits;
            bits += width;
            while (bits >= 8) {
                payload[at++] = (byte) (acc & 0xFF);
                acc >>>= 8;
                bits -= 8;
            }
        }
        if (bits > 0) payload[at++] = (byte) (acc & 0xFF);
        check(at == bitpackSize, "bitpack size " + at + " " + bitpackSize);
        return new BlockInfo(MODE_BITPACK, width, payload, -1);
    }

    static int[] decodeBlock(BlockInfo info, int count) {
        byte[] payload = info.payload;
        if (payload[0] == MODE_RLE) {
            int runCount = payload[1] & 0xFF;
            int[] values = new int[count];
            int at = HEADER, written = 0;
            for (int i = 0; i < runCount; i++) {
                int value = (payload[at] & 0xFF) | ((payload[at + 1] & 0xFF) << 8);
                int run = (payload[at + 2] & 0xFF) | ((payload[at + 3] & 0xFF) << 8);
                at += 4;
                for (int j = 0; j < run; j++) values[written++] = value;
            }
            check(written == count, "rle count");
            return values;
        }
        int width = payload[1] & 0xFF;
        int[] values = new int[count];
        int acc = 0, bits = 0, at = HEADER;
        for (int i = 0; i < count; i++) {
            while (bits < width) {
                acc |= (payload[at] & 0xFF) << bits;
                at++;
                bits += 8;
            }
            values[i] = acc & ((1 << width) - 1);
            acc >>>= width;
            bits -= width;
        }
        return values;
    }

    static Layout buildLayout(List<int[]> blocks) {
        List<BlockInfo> infos = new ArrayList<>();
        for (int[] values : blocks) infos.add(encodeBlock(values));
        Integer[] order = new Integer[blocks.size()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            int cmp = Integer.compare(infos.get(a).width, infos.get(b).width);
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        List<LayoutEntry> entries = new ArrayList<>();
        int stored = 0, groups = 0, previousWidth = Integer.MIN_VALUE;
        for (int index : order) {
            BlockInfo info = infos.get(index);
            if (info.width != previousWidth) {
                groups++;
                stored += HEADER;
                previousWidth = info.width;
            }
            stored += info.payload.length;
            entries.add(new LayoutEntry(index, groups - 1, info));
        }
        return new Layout(entries, stored, groups);
    }

    static List<int[]> decodeLayout(Layout layout, int[] lengths) {
        int[][] blocks = new int[lengths.length][];
        for (LayoutEntry entry : layout.entries) {
            check(blocks[entry.index] == null, "duplicate index");
            blocks[entry.index] = decodeBlock(entry.info, lengths[entry.index]);
        }
        return Arrays.asList(blocks);
    }

    static List<int[]> chunk(int[] values, int size) {
        List<int[]> blocks = new ArrayList<>();
        for (int i = 0; i < values.length; i += size) {
            blocks.add(Arrays.copyOfRange(values, i, Math.min(values.length, i + size)));
        }
        return blocks;
    }

    static int[] lcgValues(int count, int limit) {
        int[] values = new int[count];
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % limit;
        }
        return values;
    }

    private static int seed = 488;

    static boolean sameBlocks(List<int[]> a, List<int[]> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) if (!Arrays.equals(a.get(i), b.get(i))) return false;
        return true;
    }

    public static void main(String[] args) {
        int[] constantValues = new int[32];
        Arrays.fill(constantValues, 7);
        BlockInfo constant = encodeBlock(constantValues);
        check(constant.mode == MODE_RLE && constant.runs == 1, "constant rle");
        check(constant.payload.length == 6, "constant size");
        check(Arrays.equals(decodeBlock(constant, 32), constantValues), "constant round trip");

        BlockInfo zeros = encodeBlock(new int[32]);
        check(zeros.mode == MODE_RLE && zeros.payload.length == 6, "zeros rle");
        check(Arrays.equals(decodeBlock(zeros, 32), new int[32]), "zeros round trip");

        int[] narrowValues = new int[32];
        for (int i = 0; i < 32; i++) narrowValues[i] = i % 4;
        BlockInfo narrow = encodeBlock(narrowValues);
        check(narrow.mode == MODE_BITPACK && narrow.width == 2, "narrow bitpack");
        check(narrow.payload.length == 2 + 8, "narrow size");
        check(Arrays.equals(decodeBlock(narrow, 32), narrowValues), "narrow round trip");

        int[] wideValues = new int[32];
        for (int i = 0; i < 32; i++) wideValues[i] = new int[]{65535, 0, 12345, 65534}[i % 4];
        BlockInfo wide = encodeBlock(wideValues);
        check(wide.mode == MODE_BITPACK && wide.width == 16, "wide bitpack");
        check(wide.payload.length == 2 + 64, "wide size");
        check(Arrays.equals(decodeBlock(wide, 32), wideValues), "wide round trip");

        int[] countingValues = new int[32];
        for (int i = 0; i < 32; i++) countingValues[i] = i + 1;
        BlockInfo counting = encodeBlock(countingValues);
        check(counting.mode == MODE_BITPACK && counting.width == 6, "counting bitpack");
        check(Arrays.equals(decodeBlock(counting, 32), countingValues), "counting round trip");

        seed = 488;
        int[] skewed = lcgValues(1024, 4);
        List<int[]> blocks = chunk(skewed, 32);
        check(blocks.size() == 32, "block count");
        Layout layout = buildLayout(blocks);
        check(layout.groups == 1, "skewed groups");
        check(layout.storedBytes == 322, "skewed stored " + layout.storedBytes);
        int raw = 4 * skewed.length;
        check(layout.storedBytes <= raw / 3, "skewed bound");
        int[] lengths = new int[blocks.size()];
        for (int i = 0; i < blocks.size(); i++) lengths[i] = blocks.get(i).length;
        check(sameBlocks(decodeLayout(layout, lengths), blocks), "skewed round trip");

        seed = 489;
        int[] mixedValues = new int[1024];
        int[] firstHalf = lcgValues(512, 4);
        seed = 490;
        int[] secondHalf = lcgValues(512, 65536);
        System.arraycopy(firstHalf, 0, mixedValues, 0, 512);
        System.arraycopy(secondHalf, 0, mixedValues, 512, 512);
        List<int[]> mixed = chunk(mixedValues, 32);
        check(mixed.size() == 32, "mixed block count");
        layout = buildLayout(mixed);
        check(layout.groups == 2, "mixed groups " + layout.groups);
        check(layout.storedBytes == 16 * 10 + 16 * 66 + 2 * 2, "mixed stored " + layout.storedBytes);
        check(layout.storedBytes <= (4 * mixedValues.length) / 3, "mixed bound");
        lengths = new int[mixed.size()];
        for (int i = 0; i < mixed.size(); i++) lengths[i] = mixed.get(i).length;
        check(sameBlocks(decodeLayout(layout, lengths), mixed), "mixed round trip");
        int previousWidth = Integer.MIN_VALUE, changes = 0, groupCount = 0;
        for (LayoutEntry entry : layout.entries) {
            if (entry.info.width != previousWidth) {
                groupCount++;
                if (previousWidth != Integer.MIN_VALUE) changes++;
                previousWidth = entry.info.width;
            }
        }
        check(changes == groupCount - 1 && groupCount == layout.groups, "grouping contiguous");

        int[] shortValues = new int[33];
        Arrays.fill(shortValues, 5);
        List<int[]> shorts = chunk(shortValues, 32);
        check(shorts.size() == 2 && shorts.get(0).length == 32 && shorts.get(1).length == 1, "short blocks");
        layout = buildLayout(shorts);
        int[] shortLengths = {32, 1};
        check(sameBlocks(decodeLayout(layout, shortLengths), shorts), "short round trip");
        check(layout.groups == 2, "short groups");
        check(layout.storedBytes == 6 + 3 + 2 + 2, "short stored " + layout.storedBytes);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
