/** Task 476 - Implement snapshot isolation simulator for transactions (conceptual). (Tier: Advanced, CTCI Ch 14 — Databases) */
import java.util.*;

class Txn {
    final String id;
    final int snapshot;
    final Map<String, Integer> writes = new LinkedHashMap<>();
    String state = "active";

    Txn(String id, int snapshot) {
        this.id = id;
        this.snapshot = snapshot;
    }
}

class MVCCStore {
    final Map<String, List<int[]>> versions = new LinkedHashMap<>(); // key -> [(ts, value)]
    int commitCounter = 0;
    int nextTid = 0;

    MVCCStore(Map<String, Integer> initial) {
        for (Map.Entry<String, Integer> e : initial.entrySet()) appendCommitted(e.getKey(), e.getValue());
    }

    private void appendCommitted(String key, int value) {
        commitCounter++;
        versions.computeIfAbsent(key, k -> new ArrayList<>()).add(new int[]{commitCounter, value});
    }

    Txn begin() {
        nextTid++;
        return new Txn("T" + nextTid, commitCounter);
    }

    Integer read(Txn txn, String key) {
        if (txn.writes.containsKey(key)) return txn.writes.get(key); // read-your-writes
        Integer best = null;
        for (int[] v : versions.getOrDefault(key, Collections.emptyList())) {
            if (v[0] <= txn.snapshot) best = v[1];                    // state as of BEGIN
        }
        return best;
    }

    void write(Txn txn, String key, int value) {
        txn.writes.put(key, value);
    }

    boolean commit(Txn txn) {
        for (String key : txn.writes.keySet()) {
            for (int[] v : versions.getOrDefault(key, Collections.emptyList())) {
                if (v[0] > txn.snapshot) {                            // committed after BEGIN
                    txn.state = "aborted";
                    return false;
                }
            }
        }
        commitCounter++;
        int ts = commitCounter;
        for (Map.Entry<String, Integer> e : txn.writes.entrySet()) {
            versions.computeIfAbsent(e.getKey(), k -> new ArrayList<>()).add(new int[]{ts, e.getValue()});
        }
        txn.state = "committed";
        return true;
    }

    Map<String, Integer> materialize() {
        Map<String, Integer> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<int[]>> e : versions.entrySet()) {
            int latest = 0;
            for (int[] v : e.getValue()) latest = v[1];
            out.put(e.getKey(), latest);
        }
        return out;
    }
}

public class Solution {
    static final List<String> KEYS = Arrays.asList("a", "b", "c", "d");

    static Map<String, Integer> initial() {
        Map<String, Integer> m = new LinkedHashMap<>();
        m.put("a", 0);
        m.put("b", 1);
        m.put("c", 2);
        m.put("d", 3);
        return m;
    }

    static int lcg(int seed) {
        return (seed * 75 + 74) % 65537;
    }

    static int[][] randomScript(int count, int seed) {
        int[][] script = new int[count][];
        for (int i = 0; i < count; i++) {
            seed = lcg(seed);
            int action = seed % 5;
            seed = lcg(seed);
            int pick = seed;
            seed = lcg(seed);
            int keyIndex = seed % KEYS.size();
            seed = lcg(seed);
            int value = seed % 100;
            script[i] = new int[]{action, pick, keyIndex, value};
        }
        return script;
    }

    static void runRandomPhase() {
        MVCCStore store = new MVCCStore(initial());
        Map<String, Integer> committed = initial();
        Map<String, Integer> versionTs = new LinkedHashMap<>();
        int ts = 1;
        for (String key : initial().keySet()) versionTs.put(key, ts++);
        int counter = store.commitCounter;
        check(counter == 4, "initial commit counter");
        Map<String, Txn> liveTxns = new LinkedHashMap<>();
        Map<String, Map<String, Integer>> liveSnapshots = new LinkedHashMap<>();
        Map<String, Map<String, Integer>> liveWrites = new LinkedHashMap<>();
        int nextTid = 0;
        for (int[] op : randomScript(60, 97531)) {
            int action = op[0];
            int pick = op[1];
            String key = KEYS.get(op[2]);
            int value = op[3];
            if (action == 0 || liveTxns.isEmpty()) {
                nextTid++;
                Txn txn = store.begin();
                check(txn.snapshot == counter, "snapshot pinned at BEGIN");
                liveTxns.put(txn.id, txn);
                liveSnapshots.put(txn.id, new LinkedHashMap<>(committed));
                liveWrites.put(txn.id, new LinkedHashMap<>());
            } else {
                List<String> ids = new ArrayList<>(liveTxns.keySet());
                Collections.sort(ids);
                String tid = ids.get(pick % ids.size());
                Txn txn = liveTxns.get(tid);
                if (action == 1 || action == 4) {
                    Integer got = store.read(txn, key);
                    Map<String, Integer> writes = liveWrites.get(tid);
                    Integer expected = writes.containsKey(key) ? writes.get(key) : liveSnapshots.get(tid).get(key);
                    check(Objects.equals(got, expected), "random read " + tid + " " + key + ": " + got + " vs " + expected);
                } else if (action == 2) {
                    store.write(txn, key, value);
                    liveWrites.get(tid).put(key, value);
                } else {
                    boolean success = true;
                    for (String k : txn.writes.keySet()) {
                        if (versionTs.getOrDefault(k, -1) > txn.snapshot) success = false;
                    }
                    boolean got = store.commit(txn);
                    check(got == success, "random commit " + tid + ": " + got + " vs " + success);
                    if (got) {
                        counter = store.commitCounter;
                        for (Map.Entry<String, Integer> e : txn.writes.entrySet()) {
                            versionTs.put(e.getKey(), counter);
                            committed.put(e.getKey(), e.getValue());
                        }
                    }
                    liveTxns.remove(tid);
                    liveSnapshots.remove(tid);
                    liveWrites.remove(tid);
                }
            }
            check(store.materialize().equals(committed), "materialized state matches oracle");
            check(store.commitCounter == counter, "commit counter matches oracle");
        }
        for (String tid : liveTxns.keySet()) {
            Txn txn = liveTxns.get(tid);
            for (String key : KEYS) {
                Integer got = store.read(txn, key);
                Map<String, Integer> writes = liveWrites.get(tid);
                Integer expected = writes.containsKey(key) ? writes.get(key) : liveSnapshots.get(tid).get(key);
                check(Objects.equals(got, expected), "leftover txn read");
            }
        }
        check(store.materialize().equals(committed), "final materialized state matches oracle");
    }

    public static void main(String[] args) {
        MVCCStore store = new MVCCStore(initial()); // versions a@1, b@2, c@3, d@4
        Txn t1 = store.begin();                     // snapshot 4
        store.write(t1, "a", 99);
        Txn t2 = store.begin();                     // snapshot 4
        Txn t3 = store.begin();                     // snapshot 4
        check(store.read(t2, "a") == 0, "no dirty read of t1's uncommitted 99");
        store.write(t2, "a", 100);
        check(store.commit(t1), "first committer wins");
        check(!store.commit(t2), "write-write conflict must abort the loser");
        check(store.read(t3, "a") == 0, "snapshot pinned before t1's commit");
        check(store.read(t3, "a") == 0, "repeatable read");
        check(store.commit(t3), "read-only txn commits");
        Txn t4 = store.begin();
        check(store.read(t4, "a") == 99, "fresh snapshot sees the commit");

        Txn t5 = store.begin();
        store.write(t5, "d", 6);
        check(store.read(t5, "d") == 6, "read-your-writes");
        check(store.commit(t5), "t5 commits");
        Txn t6 = store.begin();
        check(store.read(t6, "d") == 6, "t6 sees committed d");

        Txn t7 = store.begin();
        Txn t8 = store.begin();
        check(store.read(t7, "d") == 6 && store.read(t8, "d") == 6, "both read d=6");
        store.write(t7, "d", 7);
        store.write(t8, "d", 8);
        check(store.commit(t7), "t7 wins");
        check(!store.commit(t8), "lost update must be prevented");
        check(store.read(store.begin(), "d") == 7, "d stays 7");
        Map<String, Integer> expected = new LinkedHashMap<>();
        expected.put("a", 99);
        expected.put("b", 1);
        expected.put("c", 2);
        expected.put("d", 7);
        check(store.materialize().equals(expected), "materialized state");

        // Write skew (documented SI anomaly): disjoint write sets both commit.
        Map<String, Integer> docs0 = new LinkedHashMap<>();
        docs0.put("x", 0);
        docs0.put("y", 0);
        MVCCStore docs = new MVCCStore(docs0);
        Txn t9 = docs.begin();
        Txn t10 = docs.begin();
        check(docs.read(t9, "x") == 0 && docs.read(t9, "y") == 0, "t9 snapshot");
        check(docs.read(t10, "x") == 0 && docs.read(t10, "y") == 0, "t10 snapshot");
        docs.write(t9, "x", 1);
        docs.write(t10, "y", 1);
        check(docs.commit(t9) && docs.commit(t10), "write skew is allowed under SI");
        Map<String, Integer> docsFinal = new LinkedHashMap<>();
        docsFinal.put("x", 1);
        docsFinal.put("y", 1);
        check(docs.materialize().equals(docsFinal), "the on-call invariant is broken");
        Txn fresh = docs.begin();
        check(docs.read(fresh, "x") == 1 && docs.read(fresh, "y") == 1, "fresh snapshot");

        runRandomPhase();
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
