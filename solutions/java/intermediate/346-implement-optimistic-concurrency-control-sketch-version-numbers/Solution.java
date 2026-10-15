/**
 * Task 346 - Optimistic concurrency control sketch with version numbers. (Tier: Intermediate, CTCI Ch 15)
 *
 * Versioned records: a write commits only if the expected version is still current, otherwise the
 * caller re-reads and retries.  This is a single-threaded, fully scripted simulation of the
 * interleaving (deterministic in every language): two clients read version 1, A commits, B's stale
 * commit is rejected, B re-reads and commits on top of A; a second script with three clients and a
 * round-robin scheduler shows retry-until-success and rules out lost updates.  Production OCC would
 * compare-and-swap the version field atomically.
 */
import java.util.*;

class VersionedStore {
    private final Map<String, long[]> records = new HashMap<>();   // key -> {value, version}

    void create(String key, long value) {
        if (records.containsKey(key)) throw new IllegalArgumentException("key already exists");
        records.put(key, new long[]{value, 1});                    // new records start at version 1
    }

    long[] read(String key) {
        long[] record = records.get(key);
        if (record == null) throw new IllegalArgumentException("no such key: " + key);
        return record.clone();
    }

    boolean commit(String key, long newValue, long expectedVersion) {
        long[] record = records.get(key);
        if (record == null) return false;
        if (record[1] != expectedVersion) return false;            // stale read - the caller retries
        record[0] = newValue;
        record[1] = expectedVersion + 1;
        return true;
    }
}

/** A client with a read snapshot; attempt() commits or reports a conflict. */
class OccClient {
    final String name;
    private final VersionedStore store;
    private final String key;
    long[] snapshot;
    int retries = 0;

    OccClient(String name, VersionedStore store, String key) {
        this.name = name;
        this.store = store;
        this.key = key;
    }

    void read() {
        snapshot = store.read(key);
    }

    boolean attempt(long newValue) {
        boolean ok = store.commit(key, newValue, snapshot[1]);
        if (!ok) retries++;
        return ok;
    }
}

public class Solution {
    public static void main(String[] args) {
        VersionedStore store = new VersionedStore();
        store.create("counter", 10);

        // scripted interleaving: A and B both read version 1 before anyone writes
        OccClient a = new OccClient("A", store, "counter");
        OccClient b = new OccClient("B", store, "counter");
        a.read();
        b.read();
        check(a.snapshot[0] == 10 && a.snapshot[1] == 1, "A read (10, 1)");
        check(b.snapshot[0] == 10 && b.snapshot[1] == 1, "B read (10, 1)");

        check(a.attempt(11), "A commits first");
        check(store.read("counter")[0] == 11 && store.read("counter")[1] == 2, "store now (11, 2)");
        check(!b.attempt(11), "B's stale commit is rejected");
        check(b.retries == 1, "B counted one retry");
        check(store.read("counter")[0] == 11 && store.read("counter")[1] == 2, "rejected write changed nothing");

        b.read();
        check(b.snapshot[0] == 11 && b.snapshot[1] == 2, "B re-read (11, 2)");
        check(b.attempt(12), "B retries and wins");
        check(a.retries == 0 && b.retries == 1, "retry counts A 0, B 1");
        check(store.read("counter")[0] == 12 && store.read("counter")[1] == 3, "final state (12, 3): 10 + 1 + 1");

        check(!store.commit("counter", 99, 1), "stale commit long after is still rejected");
        check(store.read("counter")[0] == 12, "value unchanged");
        check(!store.commit("missing", 1, 1), "unknown key never commits");
        try {
            store.create("counter", 0);
            throw new AssertionError("expected IllegalArgumentException on duplicate create");
        } catch (IllegalArgumentException expected) {
            // documented behaviour
        }

        // three clients, deterministic round-robin scheduler, retry until success
        store.create("total", 0);
        List<OccClient> clients = new ArrayList<>();
        for (String name : new String[]{"X", "Y", "Z"}) clients.add(new OccClient(name, store, "total"));
        for (OccClient client : clients) client.read();            // all read version 1 before anyone writes
        List<OccClient> pending = new ArrayList<>(clients);
        int rounds = 0;
        while (!pending.isEmpty()) {
            rounds++;
            List<OccClient> stillPending = new ArrayList<>();
            for (OccClient client : pending) {
                if (!client.attempt(client.snapshot[0] + 1)) {
                    client.read();                                 // conflict -> refresh, retry later
                    stillPending.add(client);
                }
            }
            pending = stillPending;
        }
        check(rounds == 3, "X, then Y, then Z");
        check(clients.get(0).retries == 0 && clients.get(1).retries == 1 && clients.get(2).retries == 2,
                "round-robin retry counts [0, 1, 2]");
        check(store.read("total")[0] == 3 && store.read("total")[1] == 4, "no lost update: 0 + 3 increments");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
