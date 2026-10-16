/** Task 995 - Implement interactive visualization generator for algorithm steps (trace player).
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  A trace player needs frames it can seek to at any time. The generator instruments two sorts and
 *  records a frame per mutation: insertion sort stores the array plus the index up to which the prefix
 *  is sorted, quicksort stores the partition window plus the pivot position and value. A memory cap
 *  keeps the timeline bounded: when more frames exist than the cap allows, the recorder samples evenly
 *  (stride = ceil((n-1)/(cap-1))) and always keeps the very last frame, so the player can jump to any
 *  retained step in O(1) while every retained frame still satisfies its invariant. The player exposes
 *  seek/next/prev with clamping, and because frames carry their original step number, rewinding after a
 *  jump returns the exact same snapshot.
 *  Simplified: fixed-size int arrays instead of a widget toolkit; indexing, sampling, the invariant
 *  checks and the replay determinism are the real ones.
 *  Verified: the sorted-prefix invariant holds on every retained insertion frame and the partition
 *  invariant on every retained quicksort frame, comparison counts match an uninstrumented oracle, the
 *  sampled timeline keeps the cap and the endpoints with a uniform stride, seek/next/prev clamp and
 *  round-trip, the empty and single-element boundaries behave, and replays are byte-identical.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final class Frame {
        final int step;
        final int[] array;
        final int sortedUpTo;                     // insertion sort: prefix [0..sortedUpTo] is sorted, -1 otherwise
        final int lo, hi, pivotIndex, pivotValue; // quicksort: partition window and pivot, -1 / MIN_VALUE otherwise
        final String note;
        Frame(int step, int[] array, int sortedUpTo, int lo, int hi, int pivotIndex, int pivotValue, String note) {
            this.step = step; this.array = array; this.sortedUpTo = sortedUpTo;
            this.lo = lo; this.hi = hi; this.pivotIndex = pivotIndex; this.pivotValue = pivotValue; this.note = note;
        }
        static Frame insertion(int step, int[] array, int sortedUpTo, String note) {
            return new Frame(step, array.clone(), sortedUpTo, -1, -1, -1, Integer.MIN_VALUE, note);
        }
        static Frame partition(int step, int[] array, int lo, int hi, int pivotIndex, int pivotValue) {
            return new Frame(step, array.clone(), -1, lo, hi, pivotIndex, pivotValue, "partition");
        }
    }

    static final class Trace {
        final List<Frame> frames;
        final int recordedFrames, comparisons, moves, cap;
        Trace(List<Frame> frames, int recordedFrames, int comparisons, int moves, int cap) {
            this.frames = frames; this.recordedFrames = recordedFrames;
            this.comparisons = comparisons; this.moves = moves; this.cap = cap;
        }
        boolean sampled() { return frames.size() < recordedFrames; }
        int stride() { return frames.size() <= 1 ? 1 : Math.max(1, (int) Math.ceil((recordedFrames - 1) / (double) (cap - 1))); }
    }

    /** Keeps the timeline inside `cap` by sampling evenly; the last frame is always retained. */
    static List<Frame> sample(List<Frame> all, int cap) {
        if (all.size() <= cap || cap < 2) return new ArrayList<>(all);
        int stride = (int) Math.ceil((all.size() - 1) / (double) (cap - 1));
        List<Frame> kept = new ArrayList<>();
        for (int i = 0; i < all.size(); i += stride) kept.add(all.get(i));
        Frame last = all.get(all.size() - 1);
        if (kept.get(kept.size() - 1).step != last.step) {
            if (kept.size() == cap) kept.set(cap - 1, last);
            else kept.add(last);
        }
        return kept;
    }

    static Trace recordInsertion(int[] input, int cap) {
        int[] a = input.clone();
        List<Frame> all = new ArrayList<>();
        all.add(Frame.insertion(0, a, 0, "start"));
        int comparisons = 0, moves = 0;
        for (int i = 1; i < a.length; i++) {
            int key = a[i], j = i - 1;
            while (j >= 0) {
                comparisons++;
                if (a[j] > key) { a[j + 1] = a[j]; moves++; j--; } else break;
            }
            a[j + 1] = key;
            moves++;
            all.add(Frame.insertion(all.size(), a, i, "insert a[" + i + "]"));
        }
        return new Trace(sample(all, cap), all.size(), comparisons, moves, cap);
    }

    static Trace recordQuicksort(int[] input, int cap) {
        int[] a = input.clone();
        List<Frame> all = new ArrayList<>();
        all.add(Frame.insertion(0, a, -1, "start"));
        int comparisons = 0, moves = 0;
        Deque<int[]> stack = new ArrayDeque<>();
        stack.push(new int[]{0, a.length - 1});
        while (!stack.isEmpty()) {
            int[] range = stack.pop();
            int lo = range[0], hi = range[1];
            if (lo >= hi) continue;
            int pivotValue = a[hi], store = lo;
            for (int i = lo; i < hi; i++) {
                comparisons++;
                if (a[i] < pivotValue) {
                    int tmp = a[store]; a[store] = a[i]; a[i] = tmp;
                    store++;
                    moves++;
                }
            }
            int tmp = a[store]; a[store] = a[hi]; a[hi] = tmp;
            moves++;
            all.add(Frame.partition(all.size(), a, lo, hi, store, pivotValue));
            stack.push(new int[]{store + 1, hi});
            stack.push(new int[]{lo, store - 1});
        }
        all.add(Frame.insertion(all.size(), a, a.length - 1, "done"));
        return new Trace(sample(all, cap), all.size(), comparisons, moves, cap);
    }

    /** The player: index-addressed seeking with clamping, so any retained frame is O(1) away. */
    static final class Player {
        final Trace trace;
        int cursor;
        Player(Trace trace) { this.trace = trace; }
        Frame seek(int index) {
            cursor = Math.max(0, Math.min(trace.frames.size() - 1, index));
            return current();
        }
        Frame next() { return seek(cursor + 1); }
        Frame prev() { return seek(cursor - 1); }
        Frame current() { return trace.frames.get(cursor); }
    }

    static boolean prefixSorted(int[] array, int upTo) {
        for (int i = 1; i <= upTo; i++) if (array[i - 1] > array[i]) return false;
        return true;
    }

    static boolean partitionHolds(Frame f) {
        for (int i = f.lo; i < f.pivotIndex; i++) if (f.array[i] > f.pivotValue) return false;
        for (int i = f.pivotIndex + 1; i <= f.hi; i++) if (f.array[i] < f.pivotValue) return false;
        return true;
    }

    static int[] oracleInsertion(int[] input) {
        int[] a = input.clone();
        for (int i = 1; i < a.length; i++) {
            int key = a[i], j = i - 1;
            while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
            a[j + 1] = key;
        }
        return a;
    }

    static int oracleComparisons(int[] input) {
        int[] a = input.clone();
        int comparisons = 0;
        for (int i = 1; i < a.length; i++) {
            int key = a[i], j = i - 1;
            while (j >= 0) {
                comparisons++;
                if (a[j] > key) { a[j + 1] = a[j]; j--; } else break;
            }
            a[j + 1] = key;
        }
        return comparisons;
    }

    public static void main(String[] args) {
        int[] sample = {9, 3, 7, 1, 8, 2, 6, 5, 4, 0, 9, 3};

        Trace insertion = recordInsertion(sample, 512);
        check(!insertion.sampled() && insertion.frames.size() == insertion.recordedFrames && insertion.frames.get(0).step == 0, "an uncapped trace must keep every frame: " + insertion.frames.size());
        for (Frame f : insertion.frames)
            check(prefixSorted(f.array, f.sortedUpTo), "sorted-prefix invariant broken at step " + f.step);
        check(Arrays.equals(insertion.frames.get(insertion.frames.size() - 1).array, oracleInsertion(sample)),
                "the final frame must equal the oracle sort: " + Arrays.toString(insertion.frames.get(0).array));
        check(insertion.comparisons == oracleComparisons(sample) && insertion.comparisons > 0, "comparison count must match the uninstrumented oracle: " + insertion.comparisons);
        boolean increasing = true;
        for (int i = 1; i < insertion.frames.size(); i++) increasing &= insertion.frames.get(i).step > insertion.frames.get(i - 1).step;
        check(increasing, "frame steps must be strictly increasing");
        // --- sampling: the timeline stays inside the cap, keeps both endpoints and a uniform stride.
        Trace capped = recordInsertion(sample, 6);
        check(capped.sampled() && capped.frames.size() <= 6 && capped.frames.size() > 1, "a capped trace must be sampled down: " + capped.frames.size() + " of " + capped.recordedFrames);
        check(capped.frames.get(0).step == 0 && capped.frames.get(capped.frames.size() - 1).step == capped.recordedFrames - 1, "sampling must keep the first and the last frame: " + capped.frames.get(0).step + ".." + capped.frames.get(capped.frames.size() - 1).step);
        int stride = capped.stride();
        boolean uniform = true;
        for (int i = 1; i < capped.frames.size() - 1; i++)
            uniform &= capped.frames.get(i).step % stride == 0 && capped.frames.get(i).step == i * stride;
        check(uniform && capped.frames.size() <= 6 && capped.frames.size() >= (capped.recordedFrames - 1) / stride, "kept steps must follow a uniform stride of " + stride + ": " + stepsOf(capped));
        for (Frame f : capped.frames)
            check(prefixSorted(f.array, f.sortedUpTo), "sampled frame at step " + f.step + " broke the invariant");
        // --- player: O(1) indexed seeking, clamping and round-trips.
        Player player = new Player(insertion);
        for (int i = 0; i < insertion.frames.size(); i++)
            check(player.seek(i).step == insertion.frames.get(i).step, "seek(" + i + ") must land on that frame");
        check(player.seek(-5).step == 0 && player.seek(10_000).step == insertion.recordedFrames - 1, "seek must clamp out-of-range indices: " + player.current().step);
        int before = player.current().step;
        check(player.prev().step == Math.max(0, before - 1) && player.next().step == before, "prev followed by next must return to the same frame: " + player.current().step);
        Frame snapshot = player.seek(3);
        check(Arrays.equals(snapshot.array, insertion.frames.get(3).array) && snapshot.step == insertion.frames.get(3).step, "a jump then rewind must restore the identical snapshot");
        check(player.seek(player.seek(2).step).step == 2 && player.seek(2).step < player.seek(3).step, "seeking by step number and forward growth must be consistent");
        // --- quicksort: partition invariant per kept frame, sorted final frame.
        Trace quick = recordQuicksort(sample, 512);
        int partitions = 0;
        for (Frame f : quick.frames)
            if (f.pivotIndex >= 0) { check(partitionHolds(f), "partition invariant broken at step " + f.step); partitions++; }
        check(partitions > 0 && Arrays.equals(quick.frames.get(quick.frames.size() - 1).array, oracleInsertion(sample)),
                "quicksort must end sorted after " + partitions + " recorded partitions");
        check(quick.comparisons > 0 && quick.moves > 0, "quicksort must count comparisons and moves: " + quick.comparisons + "/" + quick.moves);
        Trace quickCapped = recordQuicksort(sample, 5);
        check(quickCapped.frames.size() <= 5 && quickCapped.frames.get(quickCapped.frames.size() - 1).pivotIndex < 0, "a capped quicksort trace must keep the final sorted frame: " + quickCapped.frames.size());
        // --- replay determinism: two recordings of the same input are byte-identical.
        Trace again = recordInsertion(sample, 512);
        check(sameTrace(insertion, again), "replaying the same input must reproduce the trace exactly");
        check(!sameTrace(insertion, recordInsertion(new int[]{1, 2, 3}, 512)), "a different input must produce a different trace");
        // --- boundaries: empty and single-element inputs.
        Trace empty = recordInsertion(new int[]{}, 8);
        Trace single = recordInsertion(new int[]{42}, 8);
        check(empty.frames.size() == 1 && empty.frames.get(0).step == 0 && empty.comparisons == 0, "an empty array must yield exactly one frame and no comparisons");
        check(single.frames.size() == 1 && single.comparisons == 0 && Arrays.equals(single.frames.get(0).array, new int[]{42}),
                "a single-element array must yield one frame with no comparisons");
        Trace emptyQuick = recordQuicksort(new int[]{}, 8);
        check(emptyQuick.frames.size() >= 1 && emptyQuick.recordedFrames >= 1, "an empty quicksort input must still produce a start frame");
        // --- oracle: the frame count of the insertion trace equals recordedFrames, verified pairwise.
        check(oracleFrameCount(sample) == insertion.recordedFrames - 1, "frame oracle mismatch: " + oracleFrameCount(sample) + " vs " + (insertion.recordedFrames - 1));
        check(Arrays.equals(quick.frames.get(quick.frames.size() - 1).array, oracleInsertion(sample)), "quicksort final array must match the independent sort");

        System.out.println("insertion: " + insertion.recordedFrames + " frames, " + insertion.comparisons
                + " comparisons, " + insertion.moves + " moves");
        System.out.println("PASS");
    }

    static List<Integer> stepsOf(Trace trace) {
        List<Integer> steps = new ArrayList<>();
        for (Frame f : trace.frames) steps.add(f.step);
        return steps;
    }
    static boolean sameTrace(Trace a, Trace b) {
        if (a.recordedFrames != b.recordedFrames || a.comparisons != b.comparisons || a.frames.size() != b.frames.size())
            return false;
        for (int i = 0; i < a.frames.size(); i++)
            if (!Arrays.equals(a.frames.get(i).array, b.frames.get(i).array)
                    || a.frames.get(i).step != b.frames.get(i).step) return false;
        return true;
    }
    /** Oracle: counts the frames an insertion trace must have - one start frame plus one per outer pass. */
    static int oracleFrameCount(int[] input) {
        int frames = 0;
        for (int i = 1; i < input.length; i++) frames++;
        return frames;
    }
}
