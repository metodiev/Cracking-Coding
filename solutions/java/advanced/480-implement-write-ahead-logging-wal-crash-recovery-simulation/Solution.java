/** Task 480 - Implement write-ahead logging (WAL) + crash recovery simulation. (Tier: Advanced, CTCI Ch 14 — Databases) */
import java.util.*;

class RecoveryResult {
    final Map<String, Integer> state;
    final int replayed;

    RecoveryResult(Map<String, Integer> state, int replayed) {
        this.state = state;
        this.replayed = replayed;
    }
}

class WALEngine {
    Map<String, Integer> state = new LinkedHashMap<>();
    List<Object[]> wal = new ArrayList<>(); // {lsn, op, key, value}
    int lsn = 0;
    int durableLsn = 0;
    Map<String, Integer> checkpointState = null;
    int checkpointLsn = 0;
    int checkpointCount = 0;
    int crashCount = 0;
    int lastReplayCount = 0;

    void set(String key, int value) {
        lsn++;
        wal.add(new Object[]{lsn, "set", key, value});
        state.put(key, value);
    }

    void delete(String key) {
        lsn++;
        wal.add(new Object[]{lsn, "del", key, null});
        state.remove(key);
    }

    void flush() {
        durableLsn = lsn;
    }

    void checkpointNow() {
        flush();
        checkpointState = new LinkedHashMap<>(state);
        checkpointLsn = lsn;
        checkpointCount++;
    }

    String walText(String tornTail) {
        StringBuilder sb = new StringBuilder();
        for (Object[] record : wal) {
            int recordLsn = (Integer) record[0];
            String op = (String) record[1];
            String key = (String) record[2];
            Integer value = (Integer) record[3];
            sb.append(recordLsn).append('|').append(op).append('|').append(key).append('|');
            if (value != null) sb.append(value);
            sb.append('\n');
        }
        if (tornTail != null) sb.append(tornTail);
        return sb.toString();
    }

    Map<String, Integer> crash() {
        return crash(null);
    }

    Map<String, Integer> crash(String tornTail) {
        String onDisk = walText(tornTail);
        int durable = durableLsn;
        RecoveryResult result = recover(onDisk, durable, checkpointState, checkpointLsn);
        state = result.state;
        lastReplayCount = result.replayed;
        lsn = durable;
        wal = new ArrayList<>(wal.subList(0, Math.min(durable, wal.size()))); // truncate past watermark
        durableLsn = durable;
        crashCount++;
        return state;
    }

    static RecoveryResult recover(String walText, int durableLsn, Map<String, Integer> ckptState, int ckptLsn) {
        Map<String, Integer> state = ckptState == null ? new LinkedHashMap<>() : new LinkedHashMap<>(ckptState);
        int snapshotLsn = ckptState == null ? 0 : ckptLsn;
        int replayed = 0;
        for (String line : walText.split("\n", -1)) {
            if (line.isEmpty()) continue;
            String[] parts = line.split("\\|", -1);
            if (parts.length != 4) break; // torn record: stop, drop it
            int recordLsn = Integer.parseInt(parts[0]);
            if (recordLsn <= snapshotLsn) continue;
            if (recordLsn > durableLsn) break; // never made durable: lost
            if (parts[1].equals("set")) state.put(parts[2], Integer.parseInt(parts[3]));
            else state.remove(parts[2]);
            replayed++;
        }
        return new RecoveryResult(state, replayed);
    }
}

public class Solution {
    static int lcg(int seed) {
        return (seed * 75 + 74) % 65537;
    }

    static int[] runRandom() {
        WALEngine eng = new WALEngine();
        Map<String, Integer> oracleLive = new LinkedHashMap<>();
        Map<String, Integer> oracleDurable = new LinkedHashMap<>();
        int crashes = 0;
        int seed = 8642;
        for (int i = 0; i < 60; i++) {
            seed = lcg(seed);
            int action = seed % 10;
            seed = lcg(seed);
            String key = "k" + seed % 12;
            seed = lcg(seed);
            int value = seed % 1000;
            if (action < 5) {
                eng.set(key, value);
                oracleLive.put(key, value);
            } else if (action < 7) {
                eng.delete(key);
                oracleLive.remove(key);
            } else if (action == 7) {
                eng.flush();
                oracleDurable = new LinkedHashMap<>(oracleLive);
            } else if (action == 8) {
                eng.checkpointNow();
                oracleDurable = new LinkedHashMap<>(oracleLive);
            } else {
                crashes++;
                eng.crash();
                check(eng.state.equals(oracleDurable), "random crash " + i + " recovered the durable prefix");
                oracleLive = new LinkedHashMap<>(oracleDurable);
            }
        }
        check(crashes >= 1, "random workload crashed");
        check(eng.checkpointCount >= 1, "random workload checkpointed");
        eng.checkpointNow();
        oracleDurable = new LinkedHashMap<>(oracleLive);
        check(eng.state.equals(oracleLive) && eng.state.equals(oracleDurable), "settled state");
        eng.crash();
        check(eng.state.equals(oracleLive), "a crash with nothing in flight is idempotent");
        return new int[]{crashes, eng.checkpointCount};
    }

    public static void main(String[] args) {
        // Direct parser checks: replay stops at the torn record, watermark and snapshot respected.
        String text = "1|set|a|1\n2|set|b|2\n3|del|a|\n4|set|c|3\n5|set|d";
        RecoveryResult r1 = WALEngine.recover(text, 4, null, 0);
        check(r1.state.equals(map("b", 2, "c", 3)) && r1.replayed == 4, "parser watermark");
        RecoveryResult r2 = WALEngine.recover(text, 2, null, 0);
        check(r2.state.equals(map("a", 1, "b", 2)) && r2.replayed == 2, "earlier watermark");
        Map<String, Integer> snap = new LinkedHashMap<>();
        snap.put("z", 9);
        RecoveryResult r3 = WALEngine.recover(text, 5, snap, 3);
        check(r3.state.equals(map("z", 9, "c", 3)) && r3.replayed == 1, "snapshot skips LSNs <= 3");

        // Checkpoint + durable suffix replay, then a torn tail beyond the watermark.
        WALEngine eng = new WALEngine();
        for (int i = 1; i <= 5; i++) eng.set("k" + i, i);
        check(eng.lsn == 5 && eng.durableLsn == 0, "writes buffered");
        eng.checkpointNow();
        check(eng.durableLsn == 5 && eng.checkpointCount == 1, "checkpoint implies durability");
        for (int i = 6; i <= 10; i++) eng.set("k" + i, i);
        eng.flush();
        eng.set("k11", 11);
        eng.delete("k3");
        eng.set("k12", 12);
        check(eng.lsn == 13, "three in-flight ops");
        eng.crash("14|set|k13");
        Map<String, Integer> expected = new LinkedHashMap<>();
        for (int i = 1; i <= 10; i++) expected.put("k" + i, i);
        check(eng.state.equals(expected), "recovered = committed prefix k1..k10");
        check(eng.lastReplayCount == 5, "only the five post-checkpoint durable records replay");
        check(eng.lsn == 10 && eng.durableLsn == 10, "watermark restored");

        // The recovered engine keeps logging after the watermark.
        eng.set("k100", 100);
        check(eng.lsn == 11, "lsn continues");
        eng.flush();
        eng.crash();
        expected.put("k100", 100);
        check(eng.state.equals(expected), "second recovery");
        check(eng.lastReplayCount == 6, "six replayed records");

        // Unflushed writes are lost: only the committed prefix survives.
        WALEngine eng2 = new WALEngine();
        eng2.set("x", 1);
        eng2.flush();
        eng2.set("x", 2);
        eng2.crash();
        check(eng2.state.equals(map("x", 1)), "unflushed write lost");

        // Recovery from a bare WAL without any checkpoint.
        WALEngine eng3 = new WALEngine();
        for (int i = 0; i < 4; i++) eng3.set("a" + i, i);
        eng3.flush();
        eng3.crash();
        check(eng3.state.equals(map("a0", 0, "a1", 1, "a2", 2, "a3", 3)), "bare WAL replay");
        check(eng3.lastReplayCount == 4, "bare WAL replayed all four");

        int[] randomResult = runRandom();
        check(randomResult[0] == 9, "random workload crash count (a==9 rolls at 5,9,15,24,27,32,47,51,55)");
        check(randomResult[1] == 11, "ten a==8 checkpoints plus the final settle checkpoint");
        System.out.println("PASS");
    }

    static Map<String, Integer> map(Object... pairs) {
        Map<String, Integer> out = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) out.put((String) pairs[i], (Integer) pairs[i + 1]);
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
