/** Task 721 - Implement checkpoint compression and incremental checkpointing. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Long training runs checkpoint the model regularly, but a full checkpoint of every step is wasteful: two
 * consecutive steps of a modern optimizer differ in a small fraction of the parameters. This solution keeps
 * a *full* checkpoint every `fullInterval` steps and, in between, a *delta* that lists only the changed
 * blocks of the model - a block being the smallest write granularity. A restore walks back to the newest
 * full checkpoint at or before the requested step and replays the deltas forward, so any stored step is
 * reconstructible. The block list of a delta is then run-length encoded, because an optimizer update
 * usually dirties a run of neighbouring blocks rather than scattered ones.
 *
 * Invariants verified:
 *   - reconstruct  restore(step) equals the exact model state at that step, for every stored step, against
 *                  an independent shadow copy of the whole history
 *   - minimality   a delta holds exactly the blocks that actually differ from the previous stored version -
 *                  checked by re-deriving the difference independently from the shadow history
 *   - coding       the run-length code round-trips (encode then decode returns the same block list) and
 *                  never grows: the encoded form is at most one run per block plus the header
 *   - economy      the delta scheme stores far fewer blocks and bytes than checkpointing every step, and a
 *                  step that changed nothing costs exactly zero blocks
 *
 * Simplified: the model is an integer block array rather than floating-point tensors, and the byte count is
 * a varint-style estimate (one byte per run boundary, four bytes per value) rather than a real file format.
 * The verified property is the one that matters: bit-exact reconstruction of every checkpointed state.
 */
import java.util.*;

public class Solution {
    /** The changed blocks of one step: parallel arrays of block keys and their new contents. */
    static final class Delta {
        final int step;
        final int[] keys;          // tensor * blocks + block, ascending
        final int[][] values;      // values[k] is the block contents for keys[k]
        Delta(int step, int[] keys, int[][] values) {
            this.step = step;
            this.keys = keys;
            this.values = values;
        }
        int size() { return keys.length; }
    }

    /** A checkpoint chain: full snapshots every `fullInterval` steps, deltas in between. */
    static final class CheckpointStore {
        final int tensors;
        final int blocks;
        final int blockSize;
        final int fullInterval;
        private final List<Integer> steps = new ArrayList<>();
        private final List<int[][]> fulls = new ArrayList<>();     // null for a delta entry
        private final List<Delta> deltas = new ArrayList<>();      // null for a full entry
        private int[][] previous;
        long storedBlocks;
        long fullEquivalentBlocks;

        CheckpointStore(int tensors, int blocks, int blockSize, int fullInterval) {
            if (tensors <= 0 || blocks <= 0 || blockSize <= 0 || fullInterval <= 0) {
                throw new IllegalArgumentException("empty checkpoint layout");
            }
            this.tensors = tensors;
            this.blocks = blocks;
            this.blockSize = blockSize;
            this.fullInterval = fullInterval;
        }

        int storedSteps() { return steps.size(); }

        /** Stores step `step`; the store must receive steps in strictly increasing order. */
        void store(int step, int[][] state) {
            if (!steps.isEmpty() && step <= steps.get(steps.size() - 1)) {
                throw new IllegalArgumentException("steps must increase: " + step);
            }
            if (state.length != tensors) throw new IllegalArgumentException("wrong tensor count");
            boolean full = steps.isEmpty() || step % fullInterval == 0;
            if (full) {
                fulls.add(copyOf(state));
                deltas.add(null);
                storedBlocks += (long) tensors * blocks;
            } else {
                fulls.add(null);
                deltas.add(diff(step, previous, state));
                storedBlocks += deltas.get(deltas.size() - 1).size();
            }
            fullEquivalentBlocks += (long) tensors * blocks;
            previous = copyOf(state);
            steps.add(step);
        }

        /** Number of blocks the delta scheme stored for this step. */
        int storedBlocksAt(int step) {
            int index = steps.indexOf(step);
            if (index < 0) throw new IllegalArgumentException("no checkpoint at step " + step);
            Delta delta = deltas.get(index);
            return delta == null ? tensors * blocks : delta.size();
        }

        /** Rebuilds the exact state at `step` by replaying the chain from the newest full checkpoint. */
        int[][] restore(int step) {
            int index = steps.indexOf(step);
            if (index < 0) throw new IllegalArgumentException("no checkpoint at step " + step);
            int base = index;
            while (fulls.get(base) == null) base--;
            int[][] state = copyOf(fulls.get(base));
            for (int k = base + 1; k <= index; k++) apply(state, deltas.get(k));
            return state;
        }

        /** Blocks copied while restoring `step`: the replay cost, reported by the checks. */
        long restoreCost(int step) {
            int index = steps.indexOf(step);
            if (index < 0) throw new IllegalArgumentException("no checkpoint at step " + step);
            int base = index;
            while (fulls.get(base) == null) base--;
            long cost = (long) tensors * blocks;
            for (int k = base + 1; k <= index; k++) cost += deltas.get(k).size();
            return cost;
        }

        /** Total bytes of the stored chain under a simple varint estimate. */
        long storedBytes() {
            long bytes = 0;
            for (int i = 0; i < steps.size(); i++) {
                if (fulls.get(i) != null) {
                    bytes += (long) tensors * blocks * (1 + (long) blockSize * 4);
                } else {
                    int[] runs = runsOf(deltas.get(i).keys);
                    bytes += runs.length * 4L + (long) deltas.get(i).size() * (long) blockSize * 4;
                }
            }
            return bytes;
        }

        private Delta diff(int step, int[][] before, int[][] after) {
            List<Integer> keyList = new ArrayList<>();
            List<int[]> valueList = new ArrayList<>();
            for (int t = 0; t < tensors; t++) {
                for (int b = 0; b < blocks; b++) {
                    int base = b * blockSize;
                    boolean changed = false;
                    for (int i = 0; i < blockSize; i++) {
                        if (before[t][base + i] != after[t][base + i]) { changed = true; break; }
                    }
                    if (changed) {
                        int[] block = new int[blockSize];
                        System.arraycopy(after[t], base, block, 0, blockSize);
                        keyList.add(t * blocks + b);
                        valueList.add(block);
                    }
                }
            }
            int[] keys = new int[keyList.size()];
            for (int i = 0; i < keys.length; i++) keys[i] = keyList.get(i);
            return new Delta(step, keys, valueList.toArray(new int[0][]));
        }

        private void apply(int[][] state, Delta delta) {
            for (int k = 0; k < delta.size(); k++) {
                int key = delta.keys[k];
                int tensor = key / blocks;
                int base = (key % blocks) * blockSize;
                System.arraycopy(delta.values[k], 0, state[tensor], base, blockSize);
            }
        }

    }

    /** Run-length encoding of the sorted block keys: start and length pairs, in two parallel arrays. */
    static int[] runsOf(int[] keys) {
        List<Integer> runs = new ArrayList<>();
        int i = 0;
        while (i < keys.length) {
            int start = keys[i];
            int length = 1;
            while (i + length < keys.length && keys[i + length] == start + length) length++;
            runs.add(start);
            runs.add(length);
            i += length;
        }
        int[] flat = new int[runs.size()];
        for (int k = 0; k < flat.length; k++) flat[k] = runs.get(k);
        return flat;
    }

    /** Inverse of runsOf: expands the run list back into the ascending key list. */
    static int[] expandRuns(int[] runs) {
        if (runs.length % 2 != 0) throw new IllegalArgumentException("odd run list");
        int total = 0;
        for (int k = 1; k < runs.length; k += 2) total += runs[k];
        int[] keys = new int[total];
        int at = 0;
        for (int k = 0; k < runs.length; k += 2) {
            for (int i = 0; i < runs[k + 1]; i++) keys[at++] = runs[k] + i;
        }
        return keys;
    }

    private static long rngState = 0x0123456789ABCDEFL;

    /** splitmix64 - deterministic, no Math.random. */
    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static int[][] copyOf(int[][] state) {
        int[][] clone = new int[state.length][];
        for (int t = 0; t < state.length; t++) clone[t] = state[t].clone();
        return clone;
    }

    public static void main(String[] args) {
        final int tensors = 4;
        final int blocks = 8;
        final int blockSize = 4;
        final int fullInterval = 5;
        final int stepsTotal = 40;

        // --- rejected inputs (negative checks) --------------------------------
        CheckpointStore store = new CheckpointStore(tensors, blocks, blockSize, fullInterval);
        boolean rejectedEmpty = false;
        boolean rejectedStep = false;
        try { new CheckpointStore(0, 2, 2, 1); } catch (IllegalArgumentException e) { rejectedEmpty = true; }
        try { store.restore(3); } catch (IllegalArgumentException e) { rejectedStep = true; }
        check(rejectedEmpty && rejectedStep,
                "an empty layout and a restore of a never-stored step must both be rejected");

        // --- generate a plausible training run --------------------------------
        int[][] state = new int[tensors][blocks * blockSize];
        List<int[][]> history = new ArrayList<>();
        for (int step = 0; step < stepsTotal; step++) {
            if (step > 0) {
                int dirtyBlocks = step % 7 == 0 ? 0 : 1 + nextInt(3);      // every seventh step is idle
                for (int d = 0; d < dirtyBlocks; d++) {
                    int t = nextInt(tensors);
                    int b = nextInt(blocks);
                    for (int i = 0; i < blockSize; i++) {
                        state[t][b * blockSize + i] = step * 100 + i + nextInt(1000);
                    }
                }
            }
            // The store keeps its own copy; the shadow keeps every version for the oracle.
            history.add(copyOf(state));
            store.store(step, copyOf(state));
        }
        check(history.size() == stepsTotal && store.storedSteps() == stepsTotal,
                "every step must be stored, stored=" + store.storedSteps());

        // --- reconstruction against the shadow history ------------------------
        for (int step = 0; step < stepsTotal; step++) {
            check(Arrays.deepEquals(store.restore(step), history.get(step)),
                    "restore(" + step + ") must equal the model state at that step");
        }
        check(Arrays.deepEquals(store.restore(0), history.get(0)), "the first checkpoint is a full copy");
        check(store.storedBlocksAt(0) == tensors * blocks, "step 0 must store every block");
        for (int step = 1; step < stepsTotal; step++) {
            if (step % fullInterval == 0) {
                check(store.storedBlocksAt(step) == tensors * blocks,
                        "step " + step + " is a full checkpoint by construction");
            }
        }

        // --- minimality: the delta holds exactly the blocks that differ --------
        for (int step = 1; step < stepsTotal; step++) {
            if (step % fullInterval == 0) continue;
            int[][] before = history.get(step - 1);
            int[][] after = history.get(step);
            int changed = 0;
            for (int t = 0; t < tensors; t++) {
                for (int b = 0; b < blocks; b++) {
                    for (int i = 0; i < blockSize; i++) {
                        if (before[t][b * blockSize + i] != after[t][b * blockSize + i]) {
                            changed++;
                            break;
                        }
                    }
                }
            }
            check(store.storedBlocksAt(step) == changed,
                    "delta at step " + step + " must hold " + changed + " blocks, stored "
                            + store.storedBlocksAt(step));
            if (step % 7 == 0) {
                check(changed == 0 && store.storedBlocksAt(step) == 0,
                        "a step that changed nothing costs zero blocks");
            }
        }

        // --- coding round-trip and no growth ---------------------------------
        int[][] patterns = {
                {},
                {7},
                {0, 1, 2, 3, 4, 5, 6, 7, 8},
                {0, 2, 4, 6, 8, 10},
                {5, 6, 7, 20, 21, 33},
        };
        for (int[] keys : patterns) {
            int[] runs = runsOf(keys);
            check(Arrays.equals(expandRuns(runs), keys), "run-length round trip on " + Arrays.toString(keys));
            check(runs.length <= 2 * keys.length + 2, "the code never expands beyond one run per key");
        }
        boolean rejectedOdd = false;
        try { expandRuns(new int[]{1, 2, 3}); } catch (IllegalArgumentException e) { rejectedOdd = true; }
        check(rejectedOdd && runsOf(new int[]{1, 2, 3, 4, 5}).length == 2
                        && runsOf(new int[]{1, 2, 3, 4, 5})[1] == 5,
                "a malformed run list is rejected and one contiguous run encodes as a single pair");

        // --- economy and restore cost -----------------------------------------
        check(store.storedBlocks < store.fullEquivalentBlocks,
                "the chain must store fewer blocks than full checkpoints: " + store.storedBlocks + " vs "
                        + store.fullEquivalentBlocks);
        long fullBytes = (long) tensors * blocks * (1 + (long) blockSize * 4) * stepsTotal;
        check(store.storedBytes() * 3 < fullBytes, "the chain must be at least three times smaller in bytes: "
                + store.storedBytes() + " vs " + fullBytes);
        long lastCost = store.restoreCost(stepsTotal - 1);
        check(lastCost < 2L * tensors * blocks,
                "replaying to the last step stays within two model copies, cost=" + lastCost);
        for (int step = 1; step < stepsTotal; step++) {
            if (step % fullInterval == 0) continue;
            check(store.restoreCost(step) - store.restoreCost(step - 1) == store.storedBlocksAt(step),
                    "each replay adds exactly its delta at step " + step);
        }
        check(store.storedBytes() > 0, "the byte model is non-trivial");

        // --- determinism ------------------------------------------------------
        CheckpointStore replay = new CheckpointStore(tensors, blocks, blockSize, fullInterval);
        for (int step = 0; step < stepsTotal; step++) replay.store(step, history.get(step));
        check(replay.storedBlocks == store.storedBlocks && replay.storedBytes() == store.storedBytes(),
                "the same history must produce the same chain");
        check(Arrays.deepEquals(replay.restore(stepsTotal - 3), history.get(stepsTotal - 3)),
                "replaying the same history restores the same state");
        boolean rejectedRewind = false;
        try { replay.store(0, history.get(0)); } catch (IllegalArgumentException e) { rejectedRewind = true; }
        check(rejectedRewind, "a non-increasing step must be rejected");

        System.out.println("checkpoint chain over " + stepsTotal + " steps of " + tensors + " tensors, full "
                + "checkpoint every " + fullInterval + " steps");
        System.out.println("blocks stored " + store.storedBlocks + " against " + store.fullEquivalentBlocks
                + " for full checkpoints every step (" + (100 * store.storedBlocks / store.fullEquivalentBlocks)
                + "%)");
        System.out.println("bytes " + store.storedBytes() + " against " + fullBytes
                + "; restore cost of the last step " + lastCost + " blocks");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
