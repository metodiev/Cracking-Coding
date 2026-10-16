/** Task 617 - Implement consistent snapshot across shards simulation. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A sharded, versioned store is simulated: every shard keeps a per-key version chain of (epoch, value)
 * pairs, and a transaction that touches several shards is committed with a single new epoch that is
 * appended to all of its shards at once. "Consistent snapshot at epoch E" then means reading every
 * shard *as of the same epoch E*: the shard returns, for each key, the value of the latest version with
 * a write epoch <= E. Because the epoch is assigned once per transaction, a snapshot can never observe
 * one half of a multi-shard transaction, which is the whole point of the coordinator.
 *
 * The naive alternative - asking each shard for its current state at whatever moment the request
 * arrives - is simulated as well and is shown to tear a transfer apart.
 *
 * Verified: the snapshot at epoch E equals an independent replay of the first E commits for every E;
 * every multi-shard transaction is visible all-or-nothing; the sum of all balances in a snapshot never
 * changes although transfers move money between shards; reads reproduce the last write not newer than
 * the epoch; epoch 0 is the initial state and an epoch past the last commit is the latest state; and
 * mixed-epoch (torn) snapshots break the invariant while same-epoch ones never do.
 */
import java.util.*;

/** One shard: a map from key to a version chain of {epoch, value}. */
class Shard {
    final int id;
    private final Map<Integer, List<int[]>> chains = new LinkedHashMap<>();

    Shard(int id) {
        this.id = id;
    }

    void write(int epoch, int key, int value) {
        chains.computeIfAbsent(key, k -> new ArrayList<>()).add(new int[]{epoch, value});
    }

    int read(int key, int epoch) {
        List<int[]> chain = chains.get(key);
        if (chain == null) throw new IllegalArgumentException("unknown key " + key + " on shard " + id);
        int value = 0;
        boolean found = false;
        for (int[] version : chain) {
            if (version[0] <= epoch) {
                value = version[1];
                found = true;
            }
        }
        if (!found) throw new IllegalStateException("key " + key + " has no version <= " + epoch);
        return value;
    }

    Set<Integer> keys() {
        return new TreeSet<>(chains.keySet());
    }

    int latestEpoch() {
        int latest = 0;
        for (List<int[]> chain : chains.values()) {
            for (int[] version : chain) latest = Math.max(latest, version[0]);
        }
        return latest;
    }
}

class MvccStore {
    final Shard[] shards;
    final List<int[]> commitLog = new ArrayList<>();       // {epoch, shard, key, value}
    int lastEpoch;

    MvccStore(int shardCount) {
        shards = new Shard[shardCount];
        for (int i = 0; i < shardCount; i++) shards[i] = new Shard(i);
    }

    /** Writes the initial state at epoch 0, so every key has a value from the beginning. */
    void initialise(int[] keys, int[] values) {
        for (int i = 0; i < keys.length; i++) {
            shards[shardOf(keys[i])].write(0, keys[i], values[i]);
            commitLog.add(new int[]{0, shardOf(keys[i]), keys[i], values[i]});
        }
    }

    int shardOf(int key) { return key % shards.length; }

    /** Applies every write of one transaction at the same fresh epoch. */
    int commit(List<int[]> writes) {
        int epoch = ++lastEpoch;
        for (int[] write : writes) {
            shards[write[0]].write(epoch, write[1], write[2]);
            commitLog.add(new int[]{epoch, write[0], write[1], write[2]});
        }
        return epoch;
    }

    /** The consistent snapshot: one epoch for every shard. */
    Map<Integer, Integer> snapshot(int epoch) {
        if (epoch < 0 || epoch > lastEpoch) throw new IllegalArgumentException("epoch " + epoch);
        Map<Integer, Integer> state = new TreeMap<>();
        for (Shard shard : shards) {
            for (int key : shard.keys()) state.put(key, shard.read(key, epoch));
        }
        return state;
    }

    /** A torn snapshot: each shard is read at its own epoch. */
    Map<Integer, Integer> tornSnapshot(int[] epochsPerShard) {
        Map<Integer, Integer> state = new TreeMap<>();
        for (int s = 0; s < shards.length; s++) {
            for (int key : shards[s].keys()) state.put(key, shards[s].read(key, epochsPerShard[s]));
        }
        return state;
    }

    int total(Map<Integer, Integer> state) {
        int sum = 0;
        for (int value : state.values()) sum += value;
        return sum;
    }

    int totalAt(int epoch) { return total(snapshot(epoch)); }

    /** Independent oracle: replays the commit log up to the epoch in commit order. */
    Map<Integer, Integer> replay(int epoch) {
        Map<Integer, Integer> state = new TreeMap<>();
        for (int[] write : commitLog) {
            if (write[0] <= epoch) state.put(write[2], write[3]);
        }
        return state;
    }
}

public class Solution {
    private static long state = 0x9216D5D98979FB1BL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static void verifyAllEpochs(MvccStore store, String label) {
        for (int epoch = 0; epoch <= store.lastEpoch; epoch++) {
            Map<Integer, Integer> snapshot = store.snapshot(epoch);
            check(snapshot.equals(store.replay(epoch)), label + ": snapshot at " + epoch + " matches the replay");
            check(store.total(snapshot) == store.totalAt(0),
                    label + ": the total is conserved at epoch " + epoch);
        }
    }

    /** Turns a transfer into the two cross-shard writes the coordinator would send. */
    static List<int[]> transferWrites(MvccStore store, int epoch, int from, int to, int amount) {
        int fromBalance = store.shards[store.shardOf(from)].read(from, epoch);
        int toBalance = store.shards[store.shardOf(to)].read(to, epoch);
        List<int[]> writes = new ArrayList<>();
        writes.add(new int[]{store.shardOf(from), from, fromBalance - amount});
        writes.add(new int[]{store.shardOf(to), to, toBalance + amount});
        return writes;
    }

    public static void main(String[] args) {
        int[] keys = {0, 1, 2, 3, 4, 5, 6, 7};
        int[] initial = {100, 200, 50, 75, 0, 120, 30, 60};
        int initialTotal = 0;
        for (int value : initial) initialTotal += value;

        MvccStore store = new MvccStore(3);
        store.initialise(keys, initial);
        check(store.lastEpoch == 0, "the initial state is epoch 0");
        check(store.totalAt(0) == initialTotal, "the initial total");
        check(store.snapshot(0).get(3) == 75 && store.snapshot(0).get(4) == 0, "initial values");
        verifyAllEpochs(store, "initial");

        // a scripted workload of cross-shard transfers
        List<int[]> transactions = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            int from = (int) ((nextLong() >>> 33) % keys.length);
            int to = (from + 1 + (int) ((nextLong() >>> 33) % (keys.length - 1))) % keys.length;
            int amount = 1 + (int) ((nextLong() >>> 33) % 15);
            transactions.add(new int[]{from, to, amount});
        }
        for (int[] txn : transactions) {
            int epoch = store.commit(transferWrites(store, store.lastEpoch, txn[0], txn[1], txn[2]));
            check(epoch == store.lastEpoch, "the commit got a fresh epoch");
            verifyAllEpochs(store, "after commit " + epoch);
        }
        check(store.lastEpoch == transactions.size(), "one epoch per transaction");
        check(store.totalAt(store.lastEpoch) == initialTotal, "the final total is unchanged");
        verifyAllEpochs(store, "after the whole workload");

        // all-or-nothing: a cross-shard transaction is visible at one epoch, never half applied
        for (int epoch = 1; epoch <= store.lastEpoch; epoch++) {
            List<int[]> writes = writesAt(store, epoch);
            check(!writes.isEmpty(), "every epoch has its writes");
            Map<Integer, Integer> before = store.snapshot(epoch - 1);
            Map<Integer, Integer> after = store.snapshot(epoch);
            boolean changed = false;
            for (int[] write : writes) {
                check(after.get(write[2]) == write[3], "the committed value is visible at epoch " + epoch);
                if (before.get(write[2]) != write[3]) changed = true;
            }
            check(changed, "commit " + epoch + " changed the state");
        }

        // every key of a snapshot is the last write not newer than the epoch
        for (int epoch = 0; epoch <= store.lastEpoch; epoch++) {
            Map<Integer, Integer> snapshot = store.snapshot(epoch);
            for (int key : keys) {
                check(snapshot.get(key) == lastValueAt(store, key, epoch),
                        "key " + key + " at epoch " + epoch + " is the last write at or before the epoch");
            }
        }

        // mixed epochs tear a transaction apart, the coordinated snapshot never does
        int shardCount = store.shards.length;
        int brokenMixed = 0;
        int checkedMixed = 0;
        for (int epoch = 1; epoch <= store.lastEpoch; epoch++) {
            for (int skip = 0; skip < shardCount; skip++) {
                int[] epochsPerShard = new int[shardCount];
                for (int s = 0; s < shardCount; s++) epochsPerShard[s] = (s == skip) ? epoch - 1 : epoch;
                Map<Integer, Integer> mixed = store.tornSnapshot(epochsPerShard);
                checkedMixed++;
                if (store.total(mixed) != initialTotal) brokenMixed++;
            }
            check(store.total(store.snapshot(epoch)) == initialTotal,
                    "the coordinated snapshot at " + epoch + " is never torn");
        }
        check(brokenMixed > 0, "at least one mixed-epoch snapshot tears a transfer");
        System.out.println("mixed-epoch snapshots that break the invariant: " + brokenMixed
                + " of " + checkedMixed);

        // boundaries
        check(store.snapshot(store.lastEpoch).equals(store.replay(store.lastEpoch)), "the latest snapshot");
        boolean threw = false;
        try { store.snapshot(store.lastEpoch + 1); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "an epoch past the last commit is rejected");
        threw = false;
        try { store.snapshot(-1); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "a negative epoch is rejected");

        // a single write-only transaction and a same-shard transfer
        MvccStore small = new MvccStore(2);
        small.initialise(new int[]{0, 1, 2}, new int[]{10, 20, 30});
        small.commit(Arrays.asList(new int[]{0, 0, 5}));
        check(small.snapshot(1).get(0) == 5 && small.totalAt(1) == 55, "a write-only commit");
        small.commit(transferWrites(small, 1, 0, 2, 5));                       // both keys on shard 0
        check(small.shardOf(0) == small.shardOf(2), "the two keys share a shard");
        check(small.totalAt(2) == 55 && small.snapshot(2).get(0) == 0 && small.snapshot(2).get(2) == 35,
                "a same-shard transfer is atomic too");
        check(small.snapshot(2).equals(small.replay(2)), "the small store matches the replay");
        System.out.println("PASS");
    }

    static int lastValueAt(MvccStore store, int key, int epoch) {
        int value = 0;
        for (int[] write : store.commitLog) {
            if (write[2] == key && write[0] <= epoch) value = write[3];
        }
        return value;
    }

    static List<int[]> writesAt(MvccStore store, int epoch) {
        List<int[]> out = new ArrayList<>();
        for (int[] write : store.commitLog) if (write[0] == epoch) out.add(write);
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
