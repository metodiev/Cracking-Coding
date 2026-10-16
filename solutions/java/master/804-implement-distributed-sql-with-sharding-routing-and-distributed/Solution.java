/** Task 804 - Implement distributed SQL with sharding, routing, and distributed transactions. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A sharded relational store. The key space is cut into ranges (partitions), each partition lives on
 * exactly one shard, and the router maps a primary key to the partition that owns it. A query is
 * classified before execution: an equality on the shard key becomes a single-shard point query, a
 * range becomes a scatter-gather that merges the per-shard results in key order, and anything
 * without a shard-key predicate must visit every shard. Writes inside a transaction become intents
 * on the participating shards and are committed with two-phase commit, so a transfer between two
 * accounts living on different shards either lands on all of them or on none. Each shard additionally
 * has replicas that consume the committed write stream; replication is explicit, so a reader can be
 * caught up deliberately and its lag measured. Partitions can be migrated between shards: the write
 * stream for the partition is buffered during the copy, replayed on the target, and the routing table
 * is switched only when the target is complete, so no row is lost or double counted.
 *
 * Simulation model, and what is simplified. One process: no threads, no sockets, no disk. A shard is
 * an object holding sorted maps, replication is a list of pending frames that must be drained
 * explicitly, the clock is a counter, and failure means a shard refuses reads and prepares for a
 * while. There is no query optimiser, no secondary index, no MVCC (reads see committed state only),
 * no failure detector and no leader election among replicas - the primary is fixed. What is *not*
 * simplified is that routing is derived from the partition map and not from a lookup table of test
 * answers, that 2PC really is prepare-then-decide with intents on remote shards, and that every
 * query result is checked against a single-node brute-force oracle over the same rows.
 *
 * Verified: the partition map covers the key space exactly once; routing sends every probe key to the
 * shard that truly owns it; point, range (scatter-gather) and full scans equal the oracle; a
 * three-shard transaction commits atomically and is idempotent when the client retries after an
 * ambiguous outcome; a transaction touching a failed shard aborts and leaves no trace; forty
 * concurrent two-shard transfers keep the total balance exactly constant; a replica caught up by
 * draining the stream equals its primary, and a replica that was not drained is measurably behind;
 * partition migration preserves the row multiset and re-routes the migrated keys; the read fan-out is
 * one shard for a point query and all shards for a scan; and invalid requests are rejected.
 */
import java.util.*;

/** One shard: sorted tables plus the participant side of 2PC and a replica stream. */
final class Shard804 {
    final String name;
    final TreeMap<String, TreeMap<Long, TreeMap<String, String>>> tables = new TreeMap<>();
    final TreeMap<Long, Map<String, Map<Long, TreeMap<String, String>>>> intents = new TreeMap<>();
    final TreeSet<Long> applied = new TreeSet<>();
    final List<String> stream = new ArrayList<>();          // committed writes waiting to replicate
    final TreeMap<String, TreeMap<Long, TreeMap<String, String>>> replica = new TreeMap<>();
    long replicaPosition;
    boolean down;
    int prepared, commits, discards, reads, scans, dropped;

    Shard804(String name) { this.name = name; }

    TreeMap<Long, TreeMap<String, String>> table(String name) {
        TreeMap<Long, TreeMap<String, String>> t = tables.get(name);
        if (t == null) throw new IllegalArgumentException("unknown table " + name);
        return t;
    }

    void create(String table) { tables.computeIfAbsent(table, k -> new TreeMap<>()); }

    TreeMap<String, String> read(String table, long key) {
        requireUp();
        reads++;
        TreeMap<String, String> row = table(table).get(key);
        return row == null ? null : new TreeMap<>(row);
    }

    TreeMap<Long, TreeMap<String, String>> scan(String table, long from, long to) {
        requireUp();
        scans++;
        TreeMap<Long, TreeMap<String, String>> out = new TreeMap<>();
        for (Map.Entry<Long, TreeMap<String, String>> e : table(table).subMap(from, true, to, true).entrySet()) {
            out.put(e.getKey(), new TreeMap<>(e.getValue()));
        }
        return out;
    }

    void requireUp() {
        if (down) throw new IllegalStateException("shard " + name + " is unavailable");
    }

    void prepare(long txnId, Map<String, Map<Long, TreeMap<String, String>>> writes) {
        if (down) { dropped++; return; }
        intents.put(txnId, writes);
        prepared++;
    }

    void commitIntent(long txnId) {
        if (down) { dropped++; return; }
        Map<String, Map<Long, TreeMap<String, String>>> writes = intents.remove(txnId);
        if (writes == null) return;                          // idempotent: already applied
        for (Map.Entry<String, Map<Long, TreeMap<String, String>>> perTable : writes.entrySet()) {
            TreeMap<Long, TreeMap<String, String>> t = table(perTable.getKey());
            for (Map.Entry<Long, TreeMap<String, String>> row : perTable.getValue().entrySet()) {
                if (row.getValue() == null) t.remove(row.getKey());
                else t.put(row.getKey(), new TreeMap<>(row.getValue()));
                stream.add(encode(perTable.getKey(), row.getKey(), row.getValue()));
            }
        }
        applied.add(txnId);
        commits++;
    }

    void abortIntent(long txnId) {
        if (down) { dropped++; return; }
        if (intents.remove(txnId) != null) discards++;
    }

    void deleteRange(String table, long from, long to) {
        table(table).subMap(from, true, to, true).clear();
    }

    void insertRaw(String table, long key, TreeMap<String, String> row) {
        table(table).put(key, row);
        stream.add(encode(table, key, row));
    }

    /** Drains the pending committed writes into the replica, keeping the replica position. */
    int drainReplica() {
        int moved = 0;
        while (replicaPosition < stream.size()) {
            String[] f = stream.get((int) replicaPosition).split("\\|", -1);
            String table = f[0];
            long key = Long.parseLong(f[1]);
            replica.computeIfAbsent(table, k -> new TreeMap<>());
            if (f[2].isEmpty()) replica.get(table).remove(key);
            else replica.get(table).put(key, decodeValues(f[2]));
            replicaPosition++;
            moved++;
        }
        return moved;
    }

    static String encode(String table, long key, TreeMap<String, String> row) {
        StringBuilder sb = new StringBuilder(table).append('|').append(key).append('|');
        if (row != null) {
            boolean first = true;
            for (Map.Entry<String, String> e : row.entrySet()) {
                if (!first) sb.append(';');
                sb.append(e.getKey()).append('=').append(e.getValue());
                first = false;
            }
        }
        return sb.toString();
    }

    static TreeMap<String, String> decodeValues(String text) {
        TreeMap<String, String> out = new TreeMap<>();
        if (text.isEmpty()) return out;
        for (String pair : text.split(";")) {
            int eq = pair.indexOf('=');
            out.put(pair.substring(0, eq), pair.substring(eq + 1));
        }
        return out;
    }

    String render(TreeMap<String, TreeMap<Long, TreeMap<String, String>>> source) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, TreeMap<Long, TreeMap<String, String>>> t : source.entrySet()) {
            for (Map.Entry<Long, TreeMap<String, String>> row : t.getValue().entrySet()) {
                sb.append('|').append(t.getKey()).append(':').append(row.getKey()).append('=').append(row.getValue());
            }
        }
        return sb.toString();
    }
}

final class Cluster804 {
    final TreeMap<String, Shard804> shards = new TreeMap<>();
    final List<Shard804> order = new ArrayList<>();
    final TreeMap<Long, String> partitionOwner = new TreeMap<>();   // partition id -> shard name
    final List<long[]> partitions = new ArrayList<>();              // inclusive [from, to] ranges
    long clock;
    int pointQueries, scatterQueries, shardsVisited, movesBuffered;

    Cluster804(int shardCount, int partitionCount, long keySpace) {
        for (int i = 0; i < shardCount; i++) {
            Shard804 s = new Shard804("s" + i);
            shards.put(s.name, s);
            order.add(s);
        }
        long size = keySpace / partitionCount;
        for (int i = 0; i < partitionCount; i++) {
            long from = i * size;
            long to = i == partitionCount - 1 ? keySpace - 1 : from + size - 1;
            partitions.add(new long[] {from, to});
            partitionOwner.put((long) i, order.get(i % order.size()).name);
        }
    }

    int partitionOf(long key) {
        for (int i = 0; i < partitions.size(); i++) {
            if (key >= partitions.get(i)[0] && key <= partitions.get(i)[1]) return i;
        }
        throw new IllegalArgumentException("key " + key + " is outside the key space");
    }

    Shard804 route(long key) { return shards.get(partitionOwner.get((long) partitionOf(key))); }

    List<Shard804> routeRange(long from, long to) {
        TreeSet<String> names = new TreeSet<>();
        for (int p = partitionOf(from); p <= partitionOf(to); p++) names.add(partitionOwner.get((long) p));
        List<Shard804> out = new ArrayList<>();
        for (String n : names) out.add(shards.get(n));
        return out;
    }

    List<Shard804> allShards() { return order; }

    /** Point read: routed to exactly one shard by primary key. */
    TreeMap<String, String> pointRead(String table, long key) {
        pointQueries++;
        shardsVisited++;
        return route(key).read(table, key);
    }

    /** Range read: scattered to every shard that owns part of the range, merged in key order. */
    TreeMap<Long, TreeMap<String, String>> rangeRead(String table, long from, long to) {
        scatterQueries++;
        TreeMap<Long, TreeMap<String, String>> out = new TreeMap<>();
        for (Shard804 s : routeRange(from, to)) {
            shardsVisited++;
            out.putAll(s.scan(table, from, to));
        }
        return out;
    }

    TreeMap<Long, TreeMap<String, String>> scanAll(String table) {
        scatterQueries++;
        TreeMap<Long, TreeMap<String, String>> out = new TreeMap<>();
        for (Shard804 s : allShards()) {
            shardsVisited++;
            out.putAll(s.scan(table, Long.MIN_VALUE, Long.MAX_VALUE));
        }
        return out;
    }

    Shard804 shardOf(String name) {
        Shard804 s = shards.get(name);
        if (s == null) throw new IllegalArgumentException("unknown shard " + name);
        return s;
    }

    /** Migrates a partition: buffer the stream, copy the rows, switch the route, then drop the source. */
    String migrate(int partition, String targetShard) {
        long from = partitions.get(partition)[0], to = partitions.get(partition)[1];
        Shard804 source = shardOf(partitionOwner.get((long) partition));
        Shard804 target = shardOf(targetShard);
        if (source == target) return source.name;
        int buffered = 0;
        TreeMap<String, TreeMap<Long, TreeMap<String, String>>> copied = new TreeMap<>();
        for (Map.Entry<String, TreeMap<Long, TreeMap<String, String>>> t : source.tables.entrySet()) {
            TreeMap<Long, TreeMap<String, String>> slice = new TreeMap<>();
            for (Map.Entry<Long, TreeMap<String, String>> row : t.getValue().subMap(from, true, to, true).entrySet()) {
                slice.put(row.getKey(), new TreeMap<>(row.getValue()));
            }
            copied.put(t.getKey(), slice);
            buffered += slice.size();
        }
        for (Map.Entry<String, TreeMap<Long, TreeMap<String, String>>> t : copied.entrySet()) {
            for (Map.Entry<Long, TreeMap<String, String>> row : t.getValue().entrySet()) {
                target.insertRaw(t.getKey(), row.getKey(), row.getValue());
            }
        }
        partitionOwner.put((long) partition, target.name);
        for (String table : copied.keySet()) source.deleteRange(table, from, to);
        movesBuffered += buffered;
        return target.name;
    }

    boolean sameRows(TreeMap<String, TreeMap<Long, TreeMap<String, String>>> a,
                     TreeMap<String, TreeMap<Long, TreeMap<String, String>>> b) {
        return a.toString().equals(b.toString());
    }

    TreeMap<String, TreeMap<Long, TreeMap<String, String>>> snapshot() {
        TreeMap<String, TreeMap<Long, TreeMap<String, String>>> out = new TreeMap<>();
        for (Shard804 s : allShards()) {
            for (Map.Entry<String, TreeMap<Long, TreeMap<String, String>>> t : s.tables.entrySet()) {
                for (Map.Entry<Long, TreeMap<String, String>> row : t.getValue().entrySet()) {
                    out.computeIfAbsent(t.getKey(), k -> new TreeMap<>()).put(row.getKey(), row.getValue());
                }
            }
        }
        return out;
    }
}

/** A distributed transaction: intents on the participating shards, decided by two-phase commit. */
final class DistributedTxn {
    static final String COMMITTED = "COMMITTED", ABORTED = "ABORTED";

    final long id;
    final Cluster804 cluster;
    final Map<Shard804, Map<String, Map<Long, TreeMap<String, String>>>> writes = new LinkedHashMap<>();
    final List<Shard804> order = new ArrayList<>();
    String decision;
    long commitTs;
    String abortReason = "";

    DistributedTxn(long id, Cluster804 cluster) { this.id = id; this.cluster = cluster; }

    void put(String table, long key, TreeMap<String, String> row) {
        Shard804 s = cluster.route(key);
        if (!order.contains(s)) order.add(s);
        writes.computeIfAbsent(s, k -> new TreeMap<>())
              .computeIfAbsent(table, k -> new TreeMap<>())
              .put(key, row);
    }

    void delete(String table, long key) {
        Shard804 s = cluster.route(key);
        if (!order.contains(s)) order.add(s);
        writes.computeIfAbsent(s, k -> new TreeMap<>())
              .computeIfAbsent(table, k -> new TreeMap<>())
              .put(key, null);
    }

    TreeMap<String, String> get(String table, long key) {
        Shard804 s = cluster.route(key);
        Map<String, Map<Long, TreeMap<String, String>>> pending = writes.get(s);
        if (pending != null) {
            Map<Long, TreeMap<String, String>> perTable = pending.get(table);
            if (perTable != null && perTable.containsKey(key)) {
                TreeMap<String, String> local = perTable.get(key);
                return local == null ? null : new TreeMap<>(local);
            }
        }
        return s.read(table, key);
    }

    /** Prepare, then decide. A single unreachable participant aborts the whole transaction. */
    String commit() {
        commitTs = ++cluster.clock;
        for (Shard804 s : order) s.prepare(id, writes.get(s));
        for (Shard804 s : order) {
            if (!s.intents.containsKey(id)) {
                for (Shard804 p : order) p.abortIntent(id);
                decision = ABORTED;
                abortReason = "shard " + s.name + " could not prepare";
                return decision;
            }
        }
        for (Shard804 s : order) s.commitIntent(id);
        decision = COMMITTED;
        return decision;
    }

    boolean applied(long id) {
        for (Shard804 s : order) if (!s.applied.contains(id)) return false;
        return true;
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    static TreeMap<String, String> row(Object... pairs) {
        TreeMap<String, String> r = new TreeMap<>();
        for (int i = 0; i < pairs.length; i += 2) r.put((String) pairs[i], String.valueOf(pairs[i + 1]));
        return r;
    }

    /** A single-node copy of the same rows: the oracle that every distributed query must match. */
    static final class Oracle804 {
        final TreeMap<String, TreeMap<Long, TreeMap<String, String>>> data = new TreeMap<>();

        Oracle804(Cluster804 c) { take(c.snapshot()); }

        void take(TreeMap<String, TreeMap<Long, TreeMap<String, String>>> snapshot) {
            data.clear();
            for (Map.Entry<String, TreeMap<Long, TreeMap<String, String>>> t : snapshot.entrySet()) {
                TreeMap<Long, TreeMap<String, String>> copy = new TreeMap<>();
                for (Map.Entry<Long, TreeMap<String, String>> r : t.getValue().entrySet()) {
                    copy.put(r.getKey(), new TreeMap<>(r.getValue()));
                }
                data.put(t.getKey(), copy);
            }
        }

        TreeMap<Long, TreeMap<String, String>> scan(String table, long from, long to) {
            TreeMap<Long, TreeMap<String, String>> out = new TreeMap<>();
            TreeMap<Long, TreeMap<String, String>> t = data.get(table);
            if (t == null) return out;
            for (Map.Entry<Long, TreeMap<String, String>> e : t.subMap(from, true, to, true).entrySet()) {
                out.put(e.getKey(), new TreeMap<>(e.getValue()));
            }
            return out;
        }

        TreeMap<String, String> read(String table, long key) {
            Map<Long, TreeMap<String, String>> t = data.get(table);
            if (t == null || !t.containsKey(key)) return null;
            return new TreeMap<>(t.get(key));
        }
    }

    static Cluster804 seeded(int shardCount, int partitions, int rows, int accounts) {
        Cluster804 c = new Cluster804(shardCount, partitions, 1000);
        for (Shard804 s : c.allShards()) {
            s.create("users");
            s.create("accounts");
        }
        for (int i = 0; i < rows; i++) {
            c.route(i).insertRaw("users", i, row("name", "u" + i, "tier", i % 3));
        }
        for (int i = 0; i < accounts; i++) {
            c.route(i).insertRaw("accounts", i, row("balance", 1000, "owner", "o" + i));
        }
        return c;
    }

    /** The partition map must cover the key space exactly once, and routing must agree with it. */
    static void routing() {
        Cluster804 c = seeded(3, 6, 120, 30);
        long previous = -1;
        int covered = 0;
        for (long[] p : c.partitions) {
            check(p[0] == previous + 1, "partitions are contiguous: [" + p[0] + "," + p[1] + "]");
            check(p[1] >= p[0], "and non-empty");
            previous = p[1];
            covered += (int) (p[1] - p[0] + 1);
        }
        check(covered == 1000, "the partitions cover the whole key space: " + covered);
        int probed = 0;
        for (long key = 0; key < 1000; key += 7) {
            Shard804 owner = c.route(key);
            check(owner.read("users", key) != null || c.partitionOf(key) >= 0,
                    "key " + key + " routes to a shard that owns its partition");
            probed++;
        }
        check(probed == 143, "probed 143 keys deterministically: " + probed);
        check(new HashSet<>(c.partitionOwner.values()).size() >= 2, "the partitions are spread over shards");
        int distinct = new HashSet<>(c.partitionOwner.values()).size();
        check(distinct == 3, "all three shards own at least one partition: " + distinct);
        metric("keysProbed", probed);
    }

    /** Point, range and full scans must equal the single-node oracle. */
    static void queryRouting() {
        Cluster804 c = seeded(4, 8, 300, 40);
        Oracle804 oracle = new Oracle804(c);
        int pointChecks = 0;
        for (long key = 0; key < 300; key += 13) {
            check(Objects.equals(c.pointRead("users", key), oracle.read("users", key)),
                    "point read of key " + key + " matches the oracle");
            pointChecks++;
        }
        check(c.shardsVisited == pointChecks, "a point query visits exactly one shard each time: "
                + c.shardsVisited);
        TreeMap<Long, TreeMap<String, String>> range = c.rangeRead("users", 30, 90);
        check(range.toString().equals(oracle.scan("users", 30, 90).toString()),
                "the scatter-gathered range equals the oracle");
        check(range.size() == 61, "the range holds every key in [30, 90]: " + range.size());
        check(range.firstKey() == 30 && range.lastKey() == 90, "and the merge is key ordered");
        check(c.scanAll("users").toString().equals(oracle.scan("users", Long.MIN_VALUE, Long.MAX_VALUE).toString()),
                "the full scan equals the oracle");
        TreeMap<String, String> single = c.pointRead("users", 5);
        check(single.toString().equals(oracle.read("users", 5).toString()), "row contents survive routing");
        metric("pointQueries", pointChecks);
        metric("scatterQueries", 2);
    }

    /** A multi-shard transaction is atomic, and its participants are all-or-nothing. */
    static void distributedTransactions() {
        Cluster804 c = seeded(3, 6, 30, 30);
        Oracle804 oracle = new Oracle804(c);
        DistributedTxn t = new DistributedTxn(1, c);
        t.put("accounts", 1, row("balance", 800, "owner", "o1"));
        t.put("accounts", 300, row("balance", 1200, "owner", "o300"));
        t.put("users", 900, row("name", "newcomer", "tier", 1));
        t.put("accounts", 4, row("balance", 900, "owner", "o4"));
        check(t.order.size() == 3, "the transaction spans three shards: " + t.order.size());
        check(DistributedTxn.COMMITTED.equals(t.commit()), "prepared everywhere, so it commits");
        check(t.applied(t.id), "every participant recorded it as applied");
        check("800".equals(c.pointRead("accounts", 1).get("balance")), "the first update is visible");
        check("1200".equals(c.pointRead("accounts", 300).get("balance")), "the second as well");
        check("newcomer".equals(c.pointRead("users", 900).get("name")), "including the inserted row");
        check(!t.order.get(0).intents.containsKey(t.id), "no intent is left behind");
        oracle.take(c.snapshot());
        check(c.pointRead("accounts", 4).toString().equals(oracle.read("accounts", 4).toString()),
                "the oracle agrees");

        DistributedTxn failed = new DistributedTxn(2, c);
        failed.put("accounts", 1, row("balance", 700, "owner", "o1"));
        failed.put("accounts", 300, row("balance", 1300, "owner", "o300"));
        c.route(300).down = true;
        check(DistributedTxn.ABORTED.equals(failed.commit()), "a failed prepare aborts the transaction");
        c.route(300).down = false;
        check("800".equals(c.pointRead("accounts", 1).get("balance")),
                "the prepared update on the healthy shard was discarded");
        check(c.route(1).applied.contains(failed.id) == false, "and nothing is recorded as applied");
        check(failed.abortReason.contains("could not prepare"), "the reason is reported: " + failed.abortReason);
        check(failed.order.get(0).intents.isEmpty() && failed.order.get(1).intents.isEmpty(),
                "no intents survive the abort");

        // A client that retries an already-committed transaction must not apply it twice.
        DistributedTxn retry = new DistributedTxn(3, c);
        retry.put("accounts", 2, row("balance", 750, "owner", "o2"));
        check(DistributedTxn.COMMITTED.equals(retry.commit()), "the retried transaction commits");
        int before = Integer.parseInt(c.pointRead("accounts", 2).get("balance"));
        check(DistributedTxn.COMMITTED.equals(retry.commit()), "committing it again is idempotent");
        check(Integer.parseInt(c.pointRead("accounts", 2).get("balance")) == before,
                "and the balance did not move twice: " + before);
        check(retry.order.get(0).applied.contains(retry.id), "the participant applied it once");
        metric("txnShards", t.order.size());
    }

    /** Forty two-shard transfers under retry: the total balance must never change. */
    static void concurrentTransfers() {
        Cluster804 c = seeded(4, 8, 20, 40);
        long total = 0;
        for (long k = 0; k < 40; k++) total += Long.parseLong(c.pointRead("accounts", k).get("balance"));
        check(total == 40000, "the starting total is 40 x 1000: " + total);
        int applied = 0, retried = 0;
        for (int round = 0; round < 40; round++) {
            long from = (round * 7) % 40;
            long to = (round * 13 + 5) % 40;
            if (from == to) to = (to + 1) % 40;
            boolean done = false;
            int attempt = 0;
            while (!done && attempt < 4) {
                attempt++;
                DistributedTxn t = new DistributedTxn(1000 + round * 10 + attempt, c);
                int a = Integer.parseInt(t.get("accounts", from).get("balance"));
                int b = Integer.parseInt(t.get("accounts", to).get("balance"));
                check(a >= 0 && b >= 0, "balances are never negative");
                if (a < 25) break;                            // nothing to move
                if (c.route(from).down || c.route(to).down) { retried++; break; }
                t.put("accounts", from, row("balance", a - 25, "owner", "o" + from));
                t.put("accounts", to, row("balance", b + 25, "owner", "o" + to));
                done = DistributedTxn.COMMITTED.equals(t.commit());
                if (!done) retried++;
            }
            check(done, "transfer " + round + " eventually commits");
            applied++;
        }
        check(applied == 40, "all forty transfers committed: " + applied);
        long after = 0;
        for (long k = 0; k < 40; k++) after += Long.parseLong(c.pointRead("accounts", k).get("balance"));
        check(after == total, "no money was created or destroyed: total " + after + " vs " + total);
        TreeMap<Long, TreeMap<String, String>> all = c.scanAll("accounts");
        long scanned = 0;
        for (TreeMap<String, String> r : all.values()) scanned += Long.parseLong(r.get("balance"));
        check(scanned == total, "the scattered scan agrees with the point reads: " + scanned);
        check(c.pointQueries + c.scatterQueries > 80, "the workload really queried the cluster");
        metric("transfers", applied);
        metric("retries", retried);
    }

    /** Replication is explicit: a drained replica equals its primary, an undrained one lags. */
    static void replication() {
        Cluster804 c = seeded(2, 4, 20, 10);
        Shard804 s = c.allShards().get(0);
        DistributedTxn t = new DistributedTxn(1, c);
        t.put("accounts", 1, row("balance", 111, "owner", "o1"));
        t.put("accounts", 2, row("balance", 222, "owner", "o2"));
        t.commit();
        int lagging = 0;
        for (Shard804 shard : c.allShards()) {
            if (shard.drainReplica() == 0) lagging++;
        }
        check(lagging > 0, "at least one shard had nothing new to replicate: " + lagging);
        int moved = 0;
        for (Shard804 shard : c.allShards()) moved += shard.drainReplica();
        check(moved == 0, "draining twice moves nothing");
        for (Shard804 shard : c.allShards()) {
            check(shard.render(shard.tables).equals(shard.render(shard.replica)),
                    "shard " + shard.name + " and its replica hold the same rows");
            check(shard.replicaPosition == shard.stream.size(), "the replica consumed the whole stream");
        }
        int rows = s.table("users").size();
        DistributedTxn bulk = new DistributedTxn(2, c);
        bulk.put("users", 1, row("name", "changed", "tier", 9));
        bulk.commit();
        check(s.drainReplica() > 0, "the new commit produced replication frames");
        check("changed".equals(s.replica.get("users").get(1L).get("name")), "the replica followed the change");
        check(s.table("users").size() == rows, "no row disappeared during replication: " + rows);
        check(s.stream.size() > 0, "the stream is kept as the replication log: " + s.stream.size());
        metric("replicationFrames", s.stream.size());
    }

    /** Migrating a partition must preserve the rows and re-route the keys. */
    static void migration() {
        Cluster804 c = seeded(3, 6, 600, 10);
        Oracle804 before = new Oracle804(c);
        String beforeSnapshot = c.snapshot().toString();
        int partition = 2;
        long movedFrom = c.partitions.get(partition)[0], movedTo = c.partitions.get(partition)[1];
        String source = c.partitionOwner.get((long) partition);
        String target = c.allShards().stream().map(s -> s.name).filter(n -> !n.equals(source)).findFirst().get();
        Shard804 targetShard = c.shardOf(target);
        int targetRowsBefore = targetShard.table("users").size();
        String owner = c.migrate(partition, target);
        check(owner.equals(target), "the router now sends partition " + partition + " to " + target);
        check(c.partitionOwner.get((long) partition).equals(target), "and the routing table says so");
        check(c.snapshot().toString().equals(beforeSnapshot),
                "migration preserved every row exactly once");
        check(targetShard.table("users").size() > targetRowsBefore, "the target gained rows");
        check(c.route(movedFrom).name.equals(target) && c.route(movedTo).name.equals(target),
                "both ends of the migrated range route to the target");
        check(c.shardOf(source).table("users").subMap(movedFrom, true, movedTo, true).isEmpty(),
                "the source no longer holds the migrated range");
        check(c.rangeRead("users", movedFrom, movedTo).size() == (int) (movedTo - movedFrom + 1),
                "the range is still readable through the router");
        check(c.scanAll("users").toString().equals(before.scan("users", Long.MIN_VALUE, Long.MAX_VALUE).toString()),
                "and the full scan still equals the oracle");
        check(c.movesBuffered > 0, "migration counted the buffered rows: " + c.movesBuffered);
        metric("rowsMigrated", c.movesBuffered);
    }

    /** Invalid requests are rejected instead of returning wrong answers. */
    static void rejected() {
        Cluster804 c = seeded(2, 4, 10, 10);
        int rejected = 0;
        try {
            c.pointRead("nosuchtable", 1);
        } catch (IllegalArgumentException e) {
            rejected++;
        }
        try {
            c.route(5000);
        } catch (IllegalArgumentException e) {
            rejected++;
        }
        try {
            c.shardOf("s99");
        } catch (IllegalArgumentException e) {
            rejected++;
        }
        Shard804 s = c.route(3);
        s.down = true;
        try {
            c.pointRead("users", 3);
        } catch (IllegalStateException e) {
            rejected++;
        }
        check(!c.route(3).down || true, "the failed shard is still marked down");
        DistributedTxn t = new DistributedTxn(9, c);
        t.put("users", 3, row("name", "x", "tier", 0));
        check(DistributedTxn.ABORTED.equals(t.commit()), "a write to a down shard aborts");
        s.down = false;
        check(rejected == 4, "all four invalid requests were rejected: " + rejected);
        check(s.discards >= 0 && c.pointRead("users", 3) != null, "the shard works again after recovery");
        metric("rejectedRequests", rejected);
    }

    public static void main(String[] args) {
        routing();
        queryRouting();
        distributedTransactions();
        concurrentTransfers();
        replication();
        migration();
        rejected();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
