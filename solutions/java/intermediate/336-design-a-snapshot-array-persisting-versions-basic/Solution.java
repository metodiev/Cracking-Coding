/** Task 336 - Design a snapshot array (persisting versions basic). (Tier: Intermediate, CTCI Ch 7) */
import java.util.*;

class SnapshotArray {
    // history[i] holds (snapId, value) pairs with strictly increasing snap ids; the entry for
    // snap 0 defaults to 0. get() binary-searches the last write with snapId <= requested.
    private final int length;
    private final List<List<int[]>> history = new ArrayList<>();
    private int snapCount = 0;

    SnapshotArray(int length) {
        if (length < 1) throw new IllegalArgumentException("length must be >= 1");
        this.length = length;
        for (int i = 0; i < length; i++) {
            List<int[]> entries = new ArrayList<>();
            entries.add(new int[]{0, 0});
            history.add(entries);
        }
    }

    void set(int index, int value) {
        if (index < 0 || index >= length) throw new IllegalArgumentException("index out of range");
        List<int[]> entries = history.get(index);
        int[] last = entries.get(entries.size() - 1);
        if (last[0] == snapCount) {
            last[1] = value;                       // overwrite the current (unsealed) write
        } else {
            entries.add(new int[]{snapCount, value});
        }
    }

    int snap() {
        return snapCount++;
    }

    int get(int index, int snapId) {
        if (index < 0 || index >= length) throw new IllegalArgumentException("index out of range");
        if (snapId < 0 || snapId >= snapCount) throw new IllegalArgumentException("unknown snapshot id");
        List<int[]> entries = history.get(index);
        int lo = 0, hi = entries.size() - 1;
        while (lo < hi) {                          // last entry with snapId <= requested
            int mid = (lo + hi + 1) / 2;
            if (entries.get(mid)[0] <= snapId) lo = mid;
            else hi = mid - 1;
        }
        return entries.get(lo)[1];
    }
}

class NaiveSnapshotArray {
    private final int[] current;
    private final List<int[]> versions = new ArrayList<>();

    NaiveSnapshotArray(int length) {
        current = new int[length];
    }

    void set(int index, int value) {
        current[index] = value;
    }

    int snap() {
        versions.add(current.clone());
        return versions.size() - 1;
    }

    int get(int index, int snapId) {
        return versions.get(snapId)[index];
    }
}

public class Solution {
    private static int seed = 3141;

    public static void main(String[] args) {
        // Canonical LeetCode sequence.
        SnapshotArray arr = new SnapshotArray(3);
        arr.set(0, 5);
        check(arr.snap() == 0, "first snap id");
        arr.set(0, 6);
        check(arr.get(0, 0) == 5, "old value survives");
        check(arr.snap() == 1, "second snap id");
        check(arr.get(0, 0) == 5, "snapshot 0 value");
        check(arr.get(0, 1) == 6, "snapshot 1 value");
        check(arr.get(1, 0) == 0 && arr.get(2, 1) == 0, "untouched indices are zero");

        // Writes before the first snap are visible in snapshot 0; later writes are not.
        SnapshotArray arr2 = new SnapshotArray(2);
        arr2.set(1, 7);
        arr2.set(1, 9);                            // same snapshot 0: overwrites 7
        check(arr2.snap() == 0, "snap 0");
        check(arr2.get(1, 0) == 9, "overwrite before snap");
        arr2.set(1, 11);
        check(arr2.get(1, 0) == 9, "unsealed write invisible");
        check(arr2.snap() == 1, "snap 1");
        check(arr2.get(1, 1) == 11, "new value visible in snapshot 1");

        // Snapshots with no writes in between repeat the previous value.
        SnapshotArray arr3 = new SnapshotArray(1);
        arr3.set(0, 1);
        int first = arr3.snap();
        int second = arr3.snap();
        arr3.set(0, 2);
        int third = arr3.snap();
        check(first == 0 && second == 1 && third == 2, "snap ids");
        check(arr3.get(0, 0) == 1 && arr3.get(0, 1) == 1 && arr3.get(0, 2) == 2, "repeated values");

        boolean threw = false;
        try {
            arr3.get(0, 3);
        } catch (IllegalArgumentException expected) {
            threw = true;
        }
        check(threw, "unknown snapshot raises");

        // LCG operation sequences against the full-copy oracle.
        int length = 8;
        SnapshotArray fast = new SnapshotArray(length);
        NaiveSnapshotArray naive = new NaiveSnapshotArray(length);
        int snapshots = 0;
        for (int step = 0; step < 3000; step++) {
            seed = (seed * 75 + 74) % 65537;
            int choice = seed % 10;
            seed = (seed * 75 + 74) % 65537;
            if (choice < 5) {
                int index = seed % length;
                seed = (seed * 75 + 74) % 65537;
                int value = seed % 1000;
                fast.set(index, value);
                naive.set(index, value);
            } else if (choice < 7) {
                check(fast.snap() == naive.snap(), "random snap " + step);
                snapshots++;
            } else if (snapshots > 0) {
                int index = seed % length;
                seed = (seed * 75 + 74) % 65537;
                int snapId = seed % snapshots;
                check(fast.get(index, snapId) == naive.get(index, snapId), "random get " + step);
            }
            if (snapshots > 0) {
                int latest = snapshots - 1;
                for (int probe = 0; probe < length; probe++) {
                    check(fast.get(probe, latest) == naive.get(probe, latest), "probe " + step);
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
