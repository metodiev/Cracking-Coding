/** Task 355 - Implement epoch-based reclamation idea for concurrent data structure safe delete. (Tier: Intermediate, CTCI Ch 15) */
import java.util.*;

public class Solution {
    static final int EPOCHS = 3;
    static final int SCRIPT_STEPS = 120;

    /**
     * Epoch-based reclamation (EBR) sketch: a global epoch counter, per-thread pinning
     * (pin() records the current epoch, unpin() clears it) and three retirement buckets
     * keyed by epoch % EPOCHS. Documented free rule: an object retired in epoch r may be
     * freed only when no thread is pinned in an epoch <= r - i.e. r <= minPinnedEpoch - 1,
     * and with no pinned threads every retired object is safe. collect() first tries to
     * advance the epoch (an advance requires every pinned thread to sit in the current
     * epoch), then drains every bucket entry the rule allows and keeps the rest on the
     * deferred list. Real EBR frees the bucket from two epochs back; this conservative
     * rule is strictly safer. A thread id of -1 in pinEpochs means "not pinned".
     */
    static class EbrManager {
        final int[] pinEpochs;
        final List<List<int[]>> buckets = new ArrayList<>();   // entries: {objectId, retirementEpoch}
        final List<int[]> freeLog = new ArrayList<>();         // entries: {objectId, retirementEpoch, epochAtFree}
        int globalEpoch = 0;
        int violations = 0;
        int deferredEntries = 0;
        int advances = 0;

        EbrManager(int numThreads) {
            pinEpochs = new int[numThreads];
            Arrays.fill(pinEpochs, -1);
            for (int i = 0; i < EPOCHS; i++) buckets.add(new ArrayList<>());
        }

        void pin(int thread) {
            if (pinEpochs[thread] != -1) throw new AssertionError("thread already pinned: " + thread);
            pinEpochs[thread] = globalEpoch;
        }

        void unpin(int thread) {
            if (pinEpochs[thread] == -1) throw new AssertionError("thread is not pinned: " + thread);
            pinEpochs[thread] = -1;
        }

        boolean isPinned(int thread) { return pinEpochs[thread] != -1; }

        void retire(int thread, int objectId) {
            if (pinEpochs[thread] == -1) throw new AssertionError("retire requires a pinned thread");
            buckets.get(globalEpoch % EPOCHS).add(new int[]{objectId, globalEpoch});
        }

        /** Advance only when every pinned thread sits in the current epoch. */
        boolean tryAdvance() {
            for (int epoch : pinEpochs) {
                if (epoch != -1 && epoch != globalEpoch) return false;
            }
            globalEpoch++;
            advances++;
            return true;
        }

        int minPinnedEpoch() {
            int min = Integer.MAX_VALUE;
            for (int epoch : pinEpochs) if (epoch != -1) min = Math.min(min, epoch);
            return min == Integer.MAX_VALUE ? -1 : min;
        }

        /** Drain every safe deferred object; returns the objects freed by this call. */
        List<Integer> collect() {
            tryAdvance();
            int minPinned = minPinnedEpoch();
            int safeEpoch = minPinned == -1 ? -1 : minPinned - 1;
            boolean everythingSafe = minPinned == -1;
            List<Integer> freed = new ArrayList<>();
            for (List<int[]> bucket : buckets) {
                List<int[]> kept = new ArrayList<>();
                for (int[] entry : bucket) {
                    int objectId = entry[0];
                    int retiredEpoch = entry[1];
                    if (everythingSafe || retiredEpoch <= safeEpoch) {
                        recordFree(objectId, retiredEpoch);
                        freed.add(objectId);
                    } else {
                        kept.add(entry);
                    }
                }
                deferredEntries += kept.size();
                bucket.clear();
                bucket.addAll(kept);
            }
            return freed;
        }

        /**
         * The safety invariant, checked at the moment of every free: no thread may be
         * pinned in an epoch <= the object's retirement epoch.
         */
        void recordFree(int objectId, int retiredEpoch) {
            for (int epoch : pinEpochs) {
                if (epoch != -1 && epoch <= retiredEpoch) violations++;
            }
            freeLog.add(new int[]{objectId, retiredEpoch, globalEpoch});
        }
    }

    public static void main(String[] args) {
        // Phase 1: retirement in epoch 0; the pin that predates an advance blocks the free.
        EbrManager mgr = new EbrManager(3);
        mgr.pin(0);
        mgr.retire(0, 1);                                     // object 1 retired in epoch 0
        check(mgr.collect().isEmpty(), "object 1 must not be freed while thread 0 pins epoch 0");

        mgr.pin(1);                                           // thread 1 pins the current epoch 1
        check(mgr.collect().isEmpty(), "old pin (epoch 0) keeps object 1 deferred");
        check(mgr.violations == 0, "no free while a thread pinned an old epoch");

        mgr.unpin(0);
        check(mgr.collect().equals(List.of(1)), "object 1 frees once no thread pins an epoch <= 0");
        check(mgr.violations == 0, "freeing object 1 respected the pin rule");

        // Phase 2: a thread pinned in the retirement epoch itself blocks the free.
        mgr.pin(0);                                           // thread 0 pins epoch 2
        mgr.retire(0, 2);                                     // object 2 retired in epoch 2
        check(mgr.collect().isEmpty(), "object 2 deferred while thread 1 still pins epoch 1");
        mgr.unpin(1);
        check(mgr.collect().isEmpty(), "object 2 still deferred: thread 0 pins its retirement epoch 2");
        mgr.unpin(0);
        check(mgr.collect().equals(List.of(2)), "object 2 frees at quiescence");
        check(mgr.violations == 0, "freeing object 2 respected the pin rule");

        // Phase 3: LCG-driven interleaving of pin / unpin / retire / collect across 3 threads.
        mgr = new EbrManager(3);
        long seed = 4242;
        int nextObject = 100;
        List<Integer> retired = new ArrayList<>();
        for (int step = 0; step < SCRIPT_STEPS; step++) {
            seed = (seed * 75 + 74) % 65537;
            int thread = (int) (seed % 3);
            seed = (seed * 75 + 74) % 65537;
            int action = (int) (seed % 4);
            if (action == 0 && !mgr.isPinned(thread)) {
                mgr.pin(thread);
            } else if (action == 1 && mgr.isPinned(thread)) {
                mgr.unpin(thread);
            } else if (action >= 2 && mgr.isPinned(thread)) {
                mgr.retire(thread, nextObject);
                retired.add(nextObject);
                nextObject++;
            }
            mgr.collect();
            check(mgr.violations == 0, "free-of-pinned violation at step " + step);
        }

        check(retired.size() >= 10, "the interleaving retired a useful number of objects");
        check(mgr.deferredEntries > 0, "old pins actually deferred some objects");

        // Quiescence: unpin everything, then the deferred list must drain completely.
        for (int thread = 0; thread < 3; thread++) {
            if (mgr.isPinned(thread)) mgr.unpin(thread);
        }
        mgr.collect();
        for (List<int[]> bucket : mgr.buckets) check(bucket.isEmpty(), "deferred lists drained");

        List<Integer> freedIds = new ArrayList<>();
        for (int[] entry : mgr.freeLog) freedIds.add(entry[0]);
        List<Integer> sortedFreed = new ArrayList<>(freedIds);
        Collections.sort(sortedFreed);
        List<Integer> sortedRetired = new ArrayList<>(retired);
        Collections.sort(sortedRetired);
        check(sortedFreed.equals(sortedRetired), "every retired object freed exactly once");
        check(new HashSet<>(freedIds).size() == freedIds.size(), "no double frees");
        check(mgr.violations == 0, "the whole run respected the pin rule");
        check(mgr.advances >= 1, "epochs actually advanced");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
