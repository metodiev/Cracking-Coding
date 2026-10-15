/** Task 438 - Implement external memory sort outline (multiway merge). (Tier: Advanced, CTCI Ch 10)
 *
 * Simulated external sort with documented block parameters: BLOCK_SIZE = 4 integers per disk block,
 * MEMORY_BLOCKS = 4 blocks of RAM (16 integers per initial run) and FAN_IN = 4 (each merge pass merges
 * groups of up to 4 sorted runs). Phase 1 loads up to one memory worth of integers, sorts them in RAM
 * and writes each initial run back to the simulated disk. Phase 2 keeps one block of every run in
 * memory, picks the next output element with a linear min-scan over the at most FAN_IN = 4 buffers (a
 * heap would take over for larger fan-in) and streams a single output block; every block read and
 * every block written is counted by SimDisk. After the last pass the final run is read back. The
 * counted block transfers are asserted against the documented bound 2 * ceil(n/BLOCK_SIZE) *
 * (1 + passes) - one initial write pass, two transfers per block per merge pass, one final read pass -
 * and the result is checked for equality against a naive in-memory sort (correctness only). The disk
 * also keeps every intermediate run, so the accounting assert totalItems == n * (passes + 1) verifies
 * that no element was dropped or duplicated.
 */
import java.util.*;

class SimDisk {
    final Map<Integer, List<List<Integer>>> runs = new LinkedHashMap<>();
    int nextId;
    int reads;
    int writes;

    int createRun() {
        int runId = nextId++;
        runs.put(runId, new ArrayList<>());
        return runId;
    }

    List<Integer> readBlock(int runId, int index) {
        reads++;
        return runs.get(runId).get(index);
    }

    void appendBlock(int runId, List<Integer> block) {
        writes++;
        runs.get(runId).add(new ArrayList<>(block));
    }

    int blockCount(int runId) {
        return runs.get(runId).size();
    }

    int ioOps() {
        return reads + writes;
    }

    long totalItems() {
        long total = 0;
        for (List<List<Integer>> run : runs.values()) {
            for (List<Integer> block : run) total += block.size();
        }
        return total;
    }

    boolean blocksSizedOk() {
        for (List<List<Integer>> run : runs.values()) {
            for (List<Integer> block : run) {
                if (block.isEmpty() || block.size() > ExternalSort.BLOCK_SIZE) return false;
            }
        }
        return true;
    }
}

class SortResult {
    final List<Integer> data;
    final SimDisk disk;
    final int passes;

    SortResult(List<Integer> data, SimDisk disk, int passes) {
        this.data = data;
        this.disk = disk;
        this.passes = passes;
    }
}

class ExternalSort {
    static final int BLOCK_SIZE = 4;
    static final int MEMORY_BLOCKS = 4;
    static final int FAN_IN = 4;

    static int ioBound(int itemCount, int passes) {
        int blocks = (itemCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return 2 * blocks * (1 + passes);
    }

    static SortResult sort(List<Integer> data) {
        SimDisk disk = new SimDisk();
        int capacity = MEMORY_BLOCKS * BLOCK_SIZE;
        List<Integer> runs = new ArrayList<>();
        for (int start = 0; start < data.size(); start += capacity) {
            List<Integer> chunk = new ArrayList<>(data.subList(start, Math.min(start + capacity, data.size())));
            Collections.sort(chunk);
            int runId = disk.createRun();
            for (int index = 0; index < chunk.size(); index += BLOCK_SIZE) {
                disk.appendBlock(runId, chunk.subList(index, Math.min(index + BLOCK_SIZE, chunk.size())));
            }
            runs.add(runId);
        }
        int passes = 0;
        while (runs.size() > 1) {
            passes++;
            List<Integer> merged = new ArrayList<>();
            for (int start = 0; start < runs.size(); start += FAN_IN) {
                merged.add(mergeGroup(disk, runs.subList(start, Math.min(start + FAN_IN, runs.size()))));
            }
            runs = merged;
        }
        List<Integer> result = new ArrayList<>();
        if (!runs.isEmpty()) {
            int runId = runs.get(0);
            for (int index = 0; index < disk.blockCount(runId); index++) {
                result.addAll(disk.readBlock(runId, index));
            }
        }
        return new SortResult(result, disk, passes);
    }

    private static int mergeGroup(SimDisk disk, List<Integer> group) {
        int outId = disk.createRun();
        int slots = group.size();
        List<List<Integer>> blocks = new ArrayList<>();
        int[] positions = new int[slots];
        int[] indices = new int[slots];
        for (int runId : group) blocks.add(disk.readBlock(runId, 0));
        List<Integer> outBlock = new ArrayList<>();
        while (true) {
            int best = -1;
            for (int slot = 0; slot < slots; slot++) {
                if (indices[slot] < blocks.get(slot).size()) {
                    if (best == -1 || blocks.get(slot).get(indices[slot]) < blocks.get(best).get(indices[best])) {
                        best = slot;
                    }
                }
            }
            if (best == -1) break;
            outBlock.add(blocks.get(best).get(indices[best]));
            indices[best]++;
            if (outBlock.size() == BLOCK_SIZE) {
                disk.appendBlock(outId, outBlock);
                outBlock = new ArrayList<>();
            }
            if (indices[best] == blocks.get(best).size()) {
                positions[best]++;
                if (positions[best] < disk.blockCount(group.get(best))) {
                    blocks.set(best, disk.readBlock(group.get(best), positions[best]));
                    indices[best] = 0;
                }
            }
        }
        if (!outBlock.isEmpty()) disk.appendBlock(outId, outBlock);
        return outId;
    }
}

public class Solution {
    public static void main(String[] args) {
        SortResult result = ExternalSort.sort(new ArrayList<>());
        check(result.data.isEmpty() && result.passes == 0 && result.disk.ioOps() == 0, "empty input");

        result = ExternalSort.sort(new ArrayList<>(Collections.singletonList(7)));
        check(result.data.toString().equals("[7]") && result.passes == 0, "single item");
        check(result.disk.reads == 1 && result.disk.writes == 1, "single item io");

        List<Integer> data = Arrays.asList(5, 1, 4, 2, 8, 3, 6, 7, 0, 9);
        result = ExternalSort.sort(data);
        check(result.data.equals(Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)), "hand sorted");
        check(result.passes == 0, "hand passes");
        check(result.disk.ioOps() <= ExternalSort.ioBound(data.size(), result.passes), "hand io bound");
        check(result.disk.totalItems() == (long) data.size() * (result.passes + 1), "hand accounting");
        check(result.disk.blocksSizedOk(), "hand blocks");

        List<Integer> duplicates = Arrays.asList(-3, 5, 5, 1, 0, -7, 3, 3, 3, -7, 9, 0, -1, 2, 2, 8, -5);
        result = ExternalSort.sort(duplicates);
        List<Integer> sortedDuplicates = new ArrayList<>(duplicates);
        Collections.sort(sortedDuplicates);
        check(result.data.equals(sortedDuplicates), "duplicates sorted");
        check(result.passes == 1, "duplicates passes " + result.passes);
        check(result.disk.ioOps() <= ExternalSort.ioBound(duplicates.size(), result.passes), "duplicates io bound");
        check(result.disk.totalItems() == (long) duplicates.size() * (result.passes + 1), "duplicates accounting");
        check(result.disk.blocksSizedOk(), "duplicates blocks");

        List<Integer> big = new ArrayList<>();
        int seed = 4380;
        for (int i = 0; i < 2000; i++) {
            seed = (seed * 75 + 74) % 65537;
            big.add(seed % 1000);
        }
        result = ExternalSort.sort(big);
        List<Integer> naive = new ArrayList<>(big);
        Collections.sort(naive);
        check(result.data.equals(naive), "big sorted");
        check(result.passes == 4, "big passes " + result.passes);
        check(result.disk.ioOps() <= ExternalSort.ioBound(big.size(), result.passes), "big io bound");
        check(result.disk.totalItems() == (long) big.size() * (result.passes + 1), "big accounting");
        check(result.disk.blocksSizedOk(), "big blocks");
        int maxBlock = 0;
        for (List<List<Integer>> run : result.disk.runs.values()) {
            for (List<Integer> block : run) maxBlock = Math.max(maxBlock, block.size());
        }
        check(maxBlock == ExternalSort.BLOCK_SIZE, "full blocks present");

        List<Integer> reversed = new ArrayList<>();
        for (int i = 0; i < 200; i++) reversed.add(200 - i);
        result = ExternalSort.sort(reversed);
        Collections.sort(reversed);
        check(result.data.equals(reversed), "reversed sorted");
        check(result.disk.ioOps() <= ExternalSort.ioBound(reversed.size(), result.passes), "reversed io bound");
        check(result.passes > 0, "reversed passes");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
