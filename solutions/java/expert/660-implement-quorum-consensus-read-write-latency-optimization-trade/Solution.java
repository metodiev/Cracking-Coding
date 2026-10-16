/** Task 660 - Implement quorum consensus read/write latency optimization tradeoffs. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Quorum replication over n replicas with per-replica one-way latencies. A write is sent to a set of
 * W replicas and completes when the slowest of them acknowledges, so the fastest choice is the W
 * replicas with the smallest latency and the completion time is the W-th smallest latency (proved by
 * brute force over every W-subset in the checks). A read works symmetrically with R replicas and
 * returns the value carrying the highest version among the answers.
 *
 * The consistency/latency tradeoff: with W + R > n every write quorum intersects every read quorum,
 * so a read issued after a write completed always observes it (linearizable); with W + R <= n there
 * are disjoint quorums and a read can return a stale value. Latency is minimized at the minimal
 * quorum W + R = n + 1, which is shown to beat every larger configuration for the given latency
 * table.
 *
 * Verified: brute-force agreement that the W-th-smallest latency is the optimal quorum completion
 * time; latency monotone in W and R; the minimal quorum is optimal among all safe configurations
 * (exhaustive); quorum intersection holds for every W+R > n and fails for every W+R <= n (exhaustive
 * over all subset pairs); the simulated read returns the highest version present in its quorum; a
 * stale read is demonstrated on a disjoint quorum; n = 1 degenerates to a single replica; and
 * invalid quorum sizes and latencies are rejected.
 */
import java.util.*;

final class QuorumConfig {
    final int n;
    final int w;
    final int r;
    final int[] latency;         // one-way latency per replica

    QuorumConfig(int[] latency, int w, int r) {
        this.n = latency.length;
        if (n < 1) throw new IllegalArgumentException("need at least one replica");
        if (w < 1 || w > n) throw new IllegalArgumentException("write quorum out of range: " + w);
        if (r < 1 || r > n) throw new IllegalArgumentException("read quorum out of range: " + r);
        for (int value : latency) if (value < 0) throw new IllegalArgumentException("negative latency");
        this.latency = latency.clone();
        this.w = w;
        this.r = r;
    }

    boolean linearizable() { return w + r > n; }

    /** The `count` replicas with the smallest latency, ties broken by index. */
    int[] fastest(int count) {
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> latency[a] != latency[b] ? Integer.compare(latency[a], latency[b])
                : Integer.compare(a, b));
        int[] out = new int[count];
        for (int i = 0; i < count; i++) out[i] = order[i];
        return out;
    }

    int completion(int count) {
        int[] chosen = fastest(count);
        int worst = 0;
        for (int replica : chosen) worst = Math.max(worst, latency[replica]);
        return worst;
    }

    int writeCompletion() { return completion(w); }

    int readCompletion() { return completion(r); }

    int totalCompletion() { return writeCompletion() + readCompletion(); }

    /** Brute force: the smallest achievable max latency over any quorum of the given size. */
    static int bestCompletion(int[] latency, int count) {
        int best = Integer.MAX_VALUE;
        int n = latency.length;
        for (int mask = 0; mask < (1 << n); mask++) {
            if (Integer.bitCount(mask) != count) continue;
            int worst = 0;
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) worst = Math.max(worst, latency[i]);
            best = Math.min(best, worst);
        }
        return best;
    }

    static List<int[]> subsets(int n, int size) {
        List<int[]> out = new ArrayList<>();
        for (int mask = 0; mask < (1 << n); mask++) {
            if (Integer.bitCount(mask) != size) continue;
            int[] set = new int[size];
            int at = 0;
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) set[at++] = i;
            out.add(set);
        }
        return out;
    }

    static boolean intersects(int[] a, int[] b) {
        Set<Integer> set = new HashSet<>();
        for (int value : a) set.add(value);
        for (int value : b) if (set.contains(value)) return true;
        return false;
    }

    /** True when some write quorum can miss a given read quorum. */
    boolean somePairIsDisjoint() {
        for (int[] writeQuorum : subsets(n, w)) {
            for (int[] readQuorum : subsets(n, r)) {
                if (!intersects(writeQuorum, readQuorum)) return true;
            }
        }
        return false;
    }

    boolean everyPairIntersects() { return !somePairIsDisjoint(); }
}

/** Replicated store: versions grow monotonically, a reader returns the newest version it can reach. */
final class QuorumStore {
    final int[] latency;
    final int[] appliedVersion;      // per replica
    final int[] appliedValue;
    long clock;

    QuorumStore(int[] latency) {
        this.latency = latency.clone();
        this.appliedVersion = new int[latency.length];
        this.appliedValue = new int[latency.length];
    }

    /** Write through the given quorum; returns the completion time and the version written. */
    int[] write(int[] quorum, int value) {
        int version = (int) (++clock);
        int completion = 0;
        for (int replica : quorum) {
            appliedVersion[replica] = version;
            appliedValue[replica] = value;
            completion = Math.max(completion, latency[replica]);
        }
        return new int[]{completion, version};
    }

    /** Read from the given quorum: the value carrying the highest version reachable. */
    int[] read(int[] quorum) {
        int bestVersion = -1;
        int bestValue = -1;
        int completion = 0;
        for (int replica : quorum) {
            if (appliedVersion[replica] > bestVersion) {
                bestVersion = appliedVersion[replica];
                bestValue = appliedValue[replica];
            }
            completion = Math.max(completion, latency[replica]);
        }
        return new int[]{bestValue, bestVersion, completion};
    }

    boolean replicaHas(int replica, int version) {
        return appliedVersion[replica] >= version;
    }
}

public class Solution {
    static int[] latencyTable() { return new int[]{3, 1, 7, 2}; }

    public static void main(String[] args) {
        int[] latency = latencyTable();
        int n = latency.length;

        // --- optimal quorum completion is the k-th smallest latency ------------
        for (int size = 1; size <= n; size++) {
            QuorumConfig config = new QuorumConfig(latency, size, 1);
            check(config.completion(size) == QuorumConfig.bestCompletion(latency, size),
                    "the " + size + "-th smallest latency is optimal for a quorum of " + size);
        }
        check(QuorumConfig.bestCompletion(latency, 1) == 1, "single replica quorum costs the cheapest replica");
        check(QuorumConfig.bestCompletion(latency, 4) == 7, "the full quorum costs the slowest replica");

        // --- latency monotonicity ---------------------------------------------
        int previousWrite = 0;
        int previousRead = 0;
        for (int size = 1; size <= n; size++) {
            QuorumConfig config = new QuorumConfig(latency, size, size);
            check(config.writeCompletion() >= previousWrite, "write latency grows with the quorum size");
            check(config.readCompletion() >= previousRead, "read latency grows with the quorum size");
            previousWrite = config.writeCompletion();
            previousRead = config.readCompletion();
        }

        // --- the minimal safe quorum minimizes the total latency ---------------
        QuorumConfig minimal = null;
        int bestTotal = Integer.MAX_VALUE;
        for (int w = 1; w <= n; w++) {
            for (int r = 1; r <= n; r++) {
                QuorumConfig config = new QuorumConfig(latency, w, r);
                if (!config.linearizable()) continue;
                if (config.totalCompletion() < bestTotal) {
                    bestTotal = config.totalCompletion();
                    minimal = config;
                }
            }
        }
        check(minimal != null, "some safe configuration exists");
        check(minimal.w + minimal.r == n + 1, "the cheapest safe configuration uses the minimal quorum, got w="
                + minimal.w + " r=" + minimal.r);
        check(bestTotal == 5, "the optimal total latency is 5 for this table, got " + bestTotal);
        QuorumConfig example = new QuorumConfig(latency, 2, 3);
        check(example.totalCompletion() == bestTotal && example.linearizable(),
                "w=2 r=3 is a minimal quorum with the optimal latency");
        QuorumConfig oversized = new QuorumConfig(latency, 2, 4);
        check(oversized.linearizable() && oversized.totalCompletion() > bestTotal,
                "a larger quorum is safe but slower: " + oversized.totalCompletion());

        // --- quorum intersection, exhaustive ----------------------------------
        int safeCount = 0, unsafeCount = 0;
        for (int w = 1; w <= n; w++) {
            for (int r = 1; r <= n; r++) {
                QuorumConfig config = new QuorumConfig(latency, w, r);
                if (config.linearizable()) {
                    check(config.everyPairIntersects(), "W+R>n must force every quorum pair to intersect");
                    safeCount++;
                } else {
                    check(config.somePairIsDisjoint(), "W+R<=n must leave a disjoint pair");
                    unsafeCount++;
                }
            }
        }
        check(safeCount == 10 && unsafeCount == 6, "16 configurations split 10 safe / 6 unsafe");

        // --- simulated read through an intersecting quorum sees the write ------
        QuorumConfig safe = new QuorumConfig(latency, 3, 2);
        QuorumStore store = new QuorumStore(latency);
        int[] write = store.write(safe.fastest(safe.w), 42);
        check(write[0] == safe.writeCompletion(), "the write completes at the quorum latency");
        int[] read = store.read(safe.fastest(safe.r));
        check(read[0] == 42 && read[1] == write[1], "an intersecting read observes the write");
        check(read[2] == safe.readCompletion(), "the read completes at its quorum latency");

        // --- a disjoint quorum can read stale data -----------------------------
        QuorumConfig unsafe = new QuorumConfig(latency, 2, 2);
        QuorumStore unsafeStore = new QuorumStore(latency);
        int[] writeQuorum = {0, 1};
        int[] readQuorum = {2, 3};
        check(!QuorumConfig.intersects(writeQuorum, readQuorum), "the two quorums are disjoint");
        int[] unsafeWrite = unsafeStore.write(writeQuorum, 99);
        check(unsafeWrite[0] == 3, "a write to replicas 0 and 1 completes in 3, got " + unsafeWrite[0]);
        check(unsafe.somePairIsDisjoint(), "the unsafe configuration really has disjoint quorums");
        int[] staleRead = unsafeStore.read(readQuorum);
        check(staleRead[1] < unsafeWrite[1], "the disjoint read misses the write (stale read)");
        check(staleRead[0] == 0, "and it reports the initial value, got " + staleRead[0]);
        int[] freshRead = unsafeStore.read(writeQuorum);
        check(freshRead[0] == 99 && freshRead[1] == unsafeWrite[1], "reading the write quorum sees the value");

        // --- versions only move forward ---------------------------------------
        QuorumStore sequence = new QuorumStore(latency);
        QuorumConfig all = new QuorumConfig(latency, n, n);
        int lastVersion = 0;
        for (int value = 1; value <= 6; value++) {
            int[] result = sequence.write(all.fastest(all.w), value * 11);
            check(result[1] > lastVersion, "versions strictly increase");
            lastVersion = result[1];
            int[] observed = sequence.read(all.fastest(all.r));
            check(observed[0] == value * 11, "the latest write is visible through the full quorum");
            boolean allReplicas = true;
            for (int replica = 0; replica < n; replica++) if (!sequence.replicaHas(replica, lastVersion)) allReplicas = false;
            check(allReplicas, "a full quorum updates every replica");
        }

        // --- different latency tables -----------------------------------------
        int[][] tables = {{1, 1, 1}, {5, 4, 3, 2, 1}, {10, 2, 8, 6}};
        for (int[] table : tables) {
            int size = table.length;
            QuorumConfig best = null;
            int bestSum = Integer.MAX_VALUE;
            for (int w = 1; w <= size; w++) {
                for (int r = 1; r <= size; r++) {
                    QuorumConfig config = new QuorumConfig(table, w, r);
                    if (config.linearizable() && config.totalCompletion() < bestSum) {
                        bestSum = config.totalCompletion();
                        best = config;
                    }
                }
            }
            check(best != null && best.w + best.r == size + 1,
                    "minimal quorums are optimal for table of size " + size);
            for (int w = 1; w <= size; w++) {
                QuorumConfig check = new QuorumConfig(table, w, 1);
                check(check.writeCompletion() == QuorumConfig.bestCompletion(table, w),
                        "quorum completion is optimal for size " + w);
            }
            QuorumConfig safeConfig = new QuorumConfig(table, (size + 1) / 2, size + 1 - (size + 1) / 2);
            check(safeConfig.everyPairIntersects(), "the balanced quorum intersects for size " + size);
        }

        // --- n = 1 degenerates -------------------------------------------------
        QuorumConfig single = new QuorumConfig(new int[]{4}, 1, 1);
        check(single.linearizable(), "a single replica is trivially linearizable");
        check(single.writeCompletion() == 4 && single.readCompletion() == 4, "one replica, one latency");
        QuorumStore singleStore = new QuorumStore(new int[]{4});
        singleStore.write(single.fastest(1), 7);
        check(singleStore.read(single.fastest(1))[0] == 7, "single replica read after write");

        // --- rejected input ----------------------------------------------------
        boolean rejected = false;
        try {
            new QuorumConfig(latency, 0, 3);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a zero write quorum must be rejected");
        rejected = false;
        try {
            new QuorumConfig(latency, 5, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a write quorum larger than the replica set must be rejected");
        rejected = false;
        try {
            new QuorumConfig(latency, 2, 0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a zero read quorum must be rejected");
        rejected = false;
        try {
            new QuorumConfig(new int[]{1, -1}, 1, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a negative replica latency must be rejected");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
