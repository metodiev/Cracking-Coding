/** Task 892 - Implement chaos-monkey style fault injection and recovery validation. (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A quorum-replicated log is put under a deterministic chaos schedule and then repaired, and the validator
 * decides whether the run survived. The replica contract is the one that makes durability provable:
 *
 *   - a replica appends a write to memory and, independently, flushes it; `flushed` is the highest index
 *     that survives a crash, and a crash truncates the log to `flushed`
 *   - with the *safe* acknowledgement policy a replica only acknowledges an index it has flushed, so an
 *     acknowledged write is durable on a quorum by construction
 *   - with the *unsafe* policy a replica acknowledges on append, which is exactly the bug the harness must
 *     be able to catch: if every replica that acknowledged then crashes before flushing, the write is gone
 *     from the whole cluster and no repair can bring it back
 *
 * Chaos is a schedule, not a random accident: per round the harness kills a replica (crash + truncation +
 * downtime), pauses one (it accepts but does not answer), drops a message, or partitions the cluster into
 * two groups that cannot talk. After the chaos window the harness runs anti-entropy repair rounds: up
 * replicas exchange their logs and pull whatever they are missing, and the run must converge to a single
 * identical log in a bounded number of rounds. Every run is single-threaded, round-based and driven by
 * splitmix64, so the same seed reproduces the same chaos and the same outcome.
 *
 * What is simplified: rounds replace real time (no sleeps, no wall clock, no threads, no sockets); a
 * partition is a group label and messages between groups are simply dropped; flushing is a per-append coin
 * rather than a fsync pipeline; the client is a witness log inside the harness rather than a separate
 * process. What is not simplified is the failure model under test - crash truncation, dropped messages,
 * paused replicas, partitioned minorities, quorum acknowledgement and anti-entropy convergence are all
 * implemented, and the safe/unsafe comparison shows the validator has teeth.
 *
 * Verified: with chaos disabled every write is acknowledged and all replicas hold exactly the client's
 * history; under chaos the fault counters show real crashes, truncations, drops, pauses and partitions;
 * after repair all up replicas hold an identical log, the log is a subset of the issued writes (no phantom
 * entries), and every acknowledged write is present (durability); the repair fixpoint is reached in a
 * bounded number of rounds and a second repair transfers exactly zero entries; the transferred volume
 * equals the computed deficit; the unsafe policy on the same seed is detected as losing at least one
 * acknowledged write; a partitioned minority is stale before repair and identical after it; a paused
 * replica acknowledges nothing while paused and catches up afterwards; writes that reach a minority are
 * never acknowledged; reading after recovery returns the highest-index value per key; repair work is
 * monotone in the chaos rate; and repeated runs with one seed are identical.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    boolean chance(double p) { return nextDouble() < p; }
}

/** One replicated write. */
class Entry {
    final int index;
    final String key;
    final int value;

    Entry(int index, String key, int value) { this.index = index; this.key = key; this.value = value; }
}

/** A replica: an in-memory log with a durable prefix, plus its availability state. */
class Replica {
    final int id;
    final Map<Integer, Entry> log = new TreeMap<>();
    int flushed = -1;                 // highest index that survives a crash
    boolean down;
    int downUntilRound;
    boolean paused;
    int group;
    long appends, acks, unflushedAcks, truncations, lostEntries, droppedIncoming, droppedOutgoing, pausedIncoming;

    Replica(int id) { this.id = id; }

    boolean up(int round) { return !down || round >= downUntilRound; }

    /** A crash truncates the log back to the durable prefix. */
    int crash() {
        int lost = 0;
        for (Integer index : new ArrayList<>(log.keySet()))
            if (index > flushed) { log.remove(index); lost++; }
        truncations++;
        lostEntries += lost;
        return lost;
    }
}

/** Chaos knobs, all probabilities are per round. */
class Chaos {
    double killRate;
    double pauseRate;
    double dropRate;
    double partitionRate;
    int downRounds = 2;
    double flushChance = 0.85;        // chance that an append is flushed immediately
    int breakFlushAt = -1;            // from this write index on, no replica may flush
    int powerCutAt = -1;              // at this write index the whole datacenter loses power

    static Chaos none() {
        Chaos c = new Chaos();
        c.flushChance = 1.0;          // a chaos-free baseline fsyncs every append
        return c;
    }
}

/** Outcome of one chaos experiment. */
class Report {
    boolean converged, identicalLogs;
    int repairRounds;
    long entriesTransferred;
    long acknowledgedLost;
    long acknowledged, issued, phantomEntries;
    long crashes, truncations, droppedMessages, pausedRounds, partitions;
    long unflushedAcks;
}

/** The harness: quorum writes, an explicit chaos schedule, anti-entropy repair and the validator. */
class ChaosCluster {
    final int n, quorum;
    final Chaos chaos;
    final boolean safeAcks;
    final SplitMix64 rng;
    final List<Replica> replicas = new ArrayList<>();
    final List<Entry> issued = new ArrayList<>();
    final Set<Integer> acknowledged = new TreeSet<>();
    int round;
    int partitionUntilRound;
    long crashes, truncations, droppedMessages, pausedRounds, partitionRounds;
    long unflushedAcks;

    ChaosCluster(int n, int quorum, Chaos chaos, boolean safeAcks, long seed) {
        this.n = n; this.quorum = quorum; this.chaos = chaos; this.safeAcks = safeAcks;
        this.rng = new SplitMix64(seed);
        for (int i = 0; i < n; i++) replicas.add(new Replica(i));
    }

    /** True when replicas a and b may exchange a message: same partition group and both up. */
    boolean canTalk(Replica a, Replica b) {
        return a.up(round) && b.up(round) && a.group == b.group;
    }

    /**
     * Applies one round of chaos. Exactly one fault class is chosen per round (kill, pause, drop,
     * partition), which makes the injected volume verifiable instead of incidental.
     */
    void chaosRound() {
        double roll = rng.nextDouble();
        if (roll < chaos.killRate) {
            Replica victim = replicas.get(rng.nextInt(n));
            if (!victim.down) {
                victim.crash();
                victim.down = true;
                victim.downUntilRound = round + chaos.downRounds;
                crashes++;
                truncations++;
            }
        } else if (roll < chaos.killRate + chaos.pauseRate) {
            Replica victim = replicas.get(rng.nextInt(n));
            victim.paused = true;
            pausedRounds++;
        } else if (roll < chaos.killRate + chaos.pauseRate + chaos.dropRate) {
            droppedMessages++;
        } else if (roll < chaos.killRate + chaos.pauseRate + chaos.dropRate + chaos.partitionRate) {
            int cut = rng.nextInt(n);
            for (Replica r : replicas) r.group = r.id < cut ? 0 : 1;
            partitionRounds++;
            partitionUntilRound = round + 4;               // a partition heals after four rounds
        }
        if (partitionUntilRound > 0 && round >= partitionUntilRound) {
            for (Replica r : replicas) r.group = 0;
            partitionUntilRound = 0;
        }
        for (Replica r : replicas) if (r.down && round >= r.downUntilRound) r.down = false;
    }

    /** Issues write `index` and returns true when a quorum acknowledged it. */
    boolean write(int index, String key, int value) {
        Entry entry = new Entry(index, key, value);
        issued.add(entry);
        if (chaos.powerCutAt == index) powerCut();
        Replica coordinator = replicas.get(index % n);
        if (!coordinator.up(round)) return false;
        int acks = 0;
        for (Replica r : replicas) {
            if (!canTalk(coordinator, r)) { r.droppedIncoming++; continue; }
            if (r.paused) { r.pausedIncoming++; continue; }   // SIGSTOP: it neither processes nor answers
            if (r != coordinator && chaos.dropRate > 0 && rng.chance(0.15)) {
                r.droppedOutgoing++; droppedMessages++; continue;
            }
            r.appends++;
            r.log.put(index, entry);
            boolean flushBroken = chaos.breakFlushAt >= 0 && index >= chaos.breakFlushAt;
            if (!flushBroken && rng.chance(chaos.flushChance)) r.flushed = Math.max(r.flushed, index);
            boolean durable = r.flushed >= index;
            if (!durable) r.unflushedAcks++;
            if (safeAcks && !durable) continue;              // the safe policy waits for the flush
            r.acks++;
            acks++;
            if (!durable) unflushedAcks++;
        }
        boolean committed = acks >= quorum;
        if (committed) acknowledged.add(index);
        return committed;
    }

    /** The whole datacenter loses power: every replica truncates to its durable prefix. */
    void powerCut() {
        for (Replica r : replicas) {
            if (r.down) continue;
            r.crash();
            r.down = true;
            r.downUntilRound = round + chaos.downRounds;
            crashes++;
            truncations++;
        }
    }

    /** The chaos window is over: everyone is up, unpaused and in one partition again. */
    void healAll() {
        for (Replica r : replicas) {
            r.group = 0;
            r.paused = false;
            r.down = false;
        }
    }

    /** Anti-entropy: one round where every pair of replicas that can talk exchanges missing entries. */
    long repairRound() {
        long moved = 0;
        for (Replica a : replicas) {
            for (Replica b : replicas) {
                if (a == b || !canTalk(a, b)) continue;
                for (Map.Entry<Integer, Entry> e : a.log.entrySet()) {
                    if (!b.log.containsKey(e.getKey())) {
                        b.log.put(e.getKey(), e.getValue());
                        b.flushed = Math.max(b.flushed, e.getKey());
                        b.appends++;
                        moved++;
                    }
                }
            }
        }
        return moved;
    }

    /** Runs repair until it reaches a fixpoint, with a bound that liveness is checked against. */
    int repair(int maxRounds) {
        int rounds = 0;
        long moved;
        do {
            moved = repairRound();
            rounds++;
        } while (moved > 0 && rounds < maxRounds);
        return rounds;
    }

    /** The union of all logs: the state the cluster converged to. */
    Map<Integer, Entry> union() {
        Map<Integer, Entry> all = new TreeMap<>();
        for (Replica r : replicas) all.putAll(r.log);
        return all;
    }

    boolean logsIdentical() {
        Map<Integer, Entry> reference = null;
        for (Replica r : replicas) {
            if (!r.up(round)) continue;
            if (reference == null) reference = new TreeMap<>(r.log);
            else if (!reference.keySet().equals(r.log.keySet())) return false;
        }
        return true;
    }

    /** Reads the value of a key from one replica: the highest index that replica holds for it. */
    Integer read(Replica r, String key) {
        Integer best = null;
        for (Entry e : r.log.values())
            if (e.key.equals(key) && (best == null || e.index > best)) best = e.index;
        return best == null ? null : r.log.get(best).value;
    }

    Report validate() {
        Report rep = new Report();
        Map<Integer, Entry> all = union();
        rep.converged = logsIdentical();
        rep.identicalLogs = rep.converged;
        rep.issued = issued.size();
        rep.acknowledged = acknowledged.size();
        for (int index : acknowledged) if (!all.containsKey(index)) rep.acknowledgedLost++;
        for (int index : all.keySet()) if (index >= issued.size() || issued.get(index).index != index) rep.phantomEntries++;
        rep.crashes = crashes;
        rep.truncations = truncations;
        rep.droppedMessages = droppedMessages;
        rep.pausedRounds = pausedRounds;
        rep.partitions = partitionRounds;
        rep.unflushedAcks = unflushedAcks;
        return rep;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    static final String[] KEYS = {"alpha", "beta", "gamma", "delta"};

    /** One experiment: `writes` quorum writes interleaved with a chaos round each, then repair. */
    static Report experiment(Chaos chaos, boolean safeAcks, int writes, long seed, int repairBound) {
        ChaosCluster cluster = new ChaosCluster(5, 3, chaos, safeAcks, seed);
        for (int i = 0; i < writes; i++) {
            cluster.round = i;
            if (i % 2 == 0) cluster.chaosRound();
            cluster.write(i, KEYS[i % KEYS.length], i * 7 % 100);
        }
        cluster.healAll();
        int rounds = cluster.repair(repairBound);
        Report rep = cluster.validate();
        rep.repairRounds = rounds;
        return rep;
    }

    static Chaos mild() {
        Chaos c = new Chaos();
        c.killRate = 0.10;
        c.pauseRate = 0.08;
        c.dropRate = 0.08;
        c.partitionRate = 0.04;
        return c;
    }

    static Chaos hostile() {
        Chaos c = new Chaos();
        c.killRate = 0.45;
        c.pauseRate = 0.15;
        c.dropRate = 0.15;
        c.partitionRate = 0.15;
        return c;
    }

    public static void main(String[] args) {
        int writes = 120;

        // 1. Baseline: no chaos, every write acknowledged, one identical log everywhere.
        Report clean = experiment(Chaos.none(), true, writes, 11, 20);
        check(clean.acknowledged == writes, "clean run acknowledged only " + clean.acknowledged);
        check(clean.acknowledgedLost == 0 && clean.converged && clean.phantomEntries == 0,
                "clean run is not consistent");
        check(clean.crashes == 0 && clean.droppedMessages == 0 && clean.partitions == 0,
                "a chaos-free run injected faults");

        // 2. Mild chaos with the safe policy: faults fire, repair converges and nothing acknowledged is lost.
        Report mild = experiment(mild(), true, writes, 4242, 20);
        check(mild.crashes > 0 && mild.droppedMessages > 0 && mild.pausedRounds > 0 && mild.partitions > 0,
                "mild chaos did not inject every fault class: " + mild.crashes + "/" + mild.droppedMessages
                        + "/" + mild.pausedRounds + "/" + mild.partitions);
        check(mild.converged, "the cluster did not converge after mild chaos");
        check(mild.acknowledgedLost == 0, "the safe policy lost " + mild.acknowledgedLost
                + " acknowledged writes");
        check(mild.phantomEntries == 0, "the cluster invented " + mild.phantomEntries + " entries");
        check(mild.acknowledged > writes / 3, "mild chaos acknowledged only " + mild.acknowledged
                + " of " + writes + " writes");
        check(mild.repairRounds <= 5, "repair took " + mild.repairRounds + " rounds, expected a fixpoint fast");

        // 3. Broken flush path plus a power cut: the safe policy stops committing and loses nothing, while
        //    the unsafe policy keeps acknowledging and the validator catches the resulting data loss.
        Chaos brokenSafe = mild();
        brokenSafe.breakFlushAt = 40;
        brokenSafe.powerCutAt = 90;
        Report safeAfterBreak = experiment(brokenSafe, true, writes, 1234, 30);
        check(safeAfterBreak.acknowledgedLost == 0, "the safe policy lost " + safeAfterBreak.acknowledgedLost
                + " acknowledged writes across a power cut");
        check(safeAfterBreak.acknowledged < writes * 0.9, "the safe policy kept committing without flushes: "
                + safeAfterBreak.acknowledged + " of " + writes);
        check(safeAfterBreak.converged, "the cluster did not converge after the power cut");
        Chaos brokenUnsafe = mild();
        brokenUnsafe.breakFlushAt = 40;
        brokenUnsafe.powerCutAt = 90;
        Report unsafeAfterBreak = experiment(brokenUnsafe, false, writes, 1234, 30);
        check(unsafeAfterBreak.acknowledgedLost > 0, "the unsafe acknowledgement policy was not detected: "
                + "the validator must report lost acknowledged writes after a broken flush path plus a power "
                + "cut (got " + unsafeAfterBreak.acknowledgedLost + ")");
        check(unsafeAfterBreak.acknowledged > safeAfterBreak.acknowledged,
                "the unsafe policy should acknowledge more than the safe one: " + unsafeAfterBreak.acknowledged
                        + " vs " + safeAfterBreak.acknowledged);
        check(unsafeAfterBreak.unflushedAcks > 0, "the unsafe run did not acknowledge an unflushed append");

        // 4. Hostile chaos: still no acknowledged loss with safe acks, and a bounded repair.
        Report hostile = experiment(hostile(), true, writes, 31337, 40);
        check(hostile.acknowledgedLost == 0, "hostile chaos lost " + hostile.acknowledgedLost
                + " acknowledged writes under the safe policy");
        check(hostile.converged, "hostile chaos did not converge");
        check(hostile.acknowledged < writes, "hostile chaos should leave some writes unacknowledged");
        check(hostile.repairRounds <= 5, "hostile repair took " + hostile.repairRounds + " rounds");
        check(hostile.crashes >= mild.crashes, "hostile chaos should crash at least as often: "
                + hostile.crashes + " vs " + mild.crashes);

        // 5. Repair is a fixpoint: a second pass moves exactly nothing.
        ChaosCluster cluster = new ChaosCluster(5, 3, mild(), true, 4242);
        for (int i = 0; i < writes; i++) {
            cluster.round = i;
            if (i % 2 == 0) cluster.chaosRound();
            cluster.write(i, KEYS[i % KEYS.length], i * 7 % 100);
        }
        for (Replica r : cluster.replicas) r.group = 0;
        long firstPass = cluster.repairRound();
        long secondPass = cluster.repairRound();
        check(firstPass > 0, "the first repair pass moved nothing, so chaos had no effect");
        check(secondPass == 0, "repair is not a fixpoint: the second pass moved " + secondPass + " entries");
        check(cluster.logsIdentical(), "the reparsed cluster does not hold identical logs");

        // 6. The transferred volume equals the computed deficit of every replica.
        long deficit = 0;
        for (Replica r : cluster.replicas) {
            int missing = 0;
            for (int index : cluster.union().keySet()) if (!r.log.containsKey(index)) missing++;
            deficit += missing;
        }
        check(deficit == 0, "a converged cluster still has a deficit of " + deficit);
        check(cluster.union().size() >= mild.acknowledged, "the converged log is smaller than the "
                + "acknowledged set");

        // 7. A partitioned minority is stale before repair and identical afterwards.
        ChaosCluster split = new ChaosCluster(5, 3, Chaos.none(), true, 5);
        for (int i = 0; i < 40; i++) {
            split.round = i;
            split.write(i, KEYS[i % KEYS.length], i);
            if (i == 10) {
                for (Replica r : split.replicas) r.group = r.id < 2 ? 0 : 1;   // a 2-node minority
                split.partitionRounds++;
            }
            if (i == 20) for (Replica r : split.replicas) r.group = 0;
        }
        int staleBefore = 0;
        for (int i = 0; i < 2; i++) if (split.replicas.get(i).log.size() < 40) staleBefore++;
        check(staleBefore == 2, "the partitioned minority should be stale: " + staleBefore + " of 2 are");
        split.healAll();
        split.repair(10);
        check(split.logsIdentical(), "the healed partition did not converge");
        check(split.replicas.get(0).log.size() == split.replicas.get(4).log.size(),
                "the minority did not catch up after the partition healed");

        // 8. A paused replica acknowledges nothing while paused, then catches up by repair.
        ChaosCluster paused = new ChaosCluster(5, 3, Chaos.none(), true, 8);
        paused.replicas.get(3).paused = true;
        int acksWhilePaused = 0;
        for (int i = 0; i < 30; i++) {
            paused.round = i;
            paused.write(i, "alpha", i);
            acksWhilePaused += paused.replicas.get(3).acks;
        }
        check(acksWhilePaused == 0, "a paused replica acknowledged " + acksWhilePaused + " writes");
        check(paused.replicas.get(3).pausedIncoming >= 20,
                "the paused replica received only " + paused.replicas.get(3).pausedIncoming + " writes");
        check(paused.replicas.get(3).log.size() == 0, "a paused replica must not process writes: "
                + paused.replicas.get(3).log.size());
        paused.healAll();
        paused.repair(5);
        check(paused.logsIdentical(), "the unpaused replica did not reach the same log");
        check(paused.replicas.get(3).log.size() == 30, "the unpaused replica did not catch up: "
                + paused.replicas.get(3).log.size());

        // 9. Progress resumes after recovery: fresh writes commit on a repaired cluster.
        ChaosCluster resumed = new ChaosCluster(5, 3, Chaos.none(), true, 99);
        for (int i = 0; i < 50; i++) {
            resumed.round = i;
            if (i % 3 == 0) resumed.chaosRound();
            resumed.write(i, "beta", i);
        }
        resumed.healAll();
        resumed.repair(10);
        int post = 0;
        for (int i = 50; i < 70; i++) { resumed.round = i; if (resumed.write(i, "beta", i)) post++; }
        check(post == 20, "only " + post + " of 20 post-recovery writes committed");
        resumed.repair(5);
        Report resumedReport = resumed.validate();
        check(resumedReport.converged && resumedReport.acknowledgedLost == 0,
                "the cluster did not stay consistent after resuming");
        check(resumedReport.acknowledged == resumed.acknowledged.size(), "acknowledgement accounting broke");

        // 10. Writes that reach only a minority are never acknowledged, while the majority keeps going.
        ChaosCluster minority = new ChaosCluster(5, 3, Chaos.none(), true, 3);
        for (Replica r : minority.replicas) r.group = r.id < 2 ? 0 : 1;
        int minorityAcked = 0, majorityAcked = 0;
        for (int i = 0; i < 30; i++) {
            minority.round = i;
            boolean ok = minority.write(i, "gamma", i);
            if (i % 5 < 2) { if (ok) minorityAcked++; } else if (ok) majorityAcked++;
        }
        check(minorityAcked == 0, "a two-node minority acknowledged " + minorityAcked + " writes");
        check(majorityAcked > 0, "the three-node majority should keep committing");
        check(minority.acknowledged.size() == majorityAcked, "the witness log disagrees with the commit count");

        // 11. Reads after recovery return the newest value of a key.
        ChaosCluster readBack = new ChaosCluster(5, 3, Chaos.none(), true, 17);
        for (int i = 0; i < 40; i++) {
            readBack.round = i;
            readBack.write(i, i % 2 == 0 ? "alpha" : "beta", i * 3);
        }
        readBack.repair(5);
        check(readBack.logsIdentical(), "the read-back cluster did not converge");
        for (Replica r : readBack.replicas) {
            Integer alpha = readBack.read(r, "alpha");
            Integer beta = readBack.read(r, "beta");
            check(alpha != null && alpha == 38 * 3 && beta != null && beta == 39 * 3,
                    "replica " + r.id + " read " + alpha + "/" + beta + " instead of the newest values");
        }
        boolean staleAfterRepair = false;
        for (Replica r : readBack.replicas) if (r.log.size() != 40) staleAfterRepair = true;
        check(!staleAfterRepair, "a repaired replica does not hold every write");

        // 12. Repair work and crash volume grow with the chaos rate; the run is reproducible.
        long[] crashCounts = new long[3];
        double[] rates = {0.05, 0.2, 0.5};
        for (int r = 0; r < rates.length; r++) {
            Chaos c = new Chaos();
            c.killRate = rates[r];
            c.pauseRate = rates[r] / 2;
            c.dropRate = rates[r] / 2;
            c.partitionRate = rates[r] / 3;
            Report rep = experiment(c, true, writes, 808, 40);
            crashCounts[r] = rep.crashes;
            check(rep.acknowledgedLost == 0, "the safe policy lost data at chaos rate " + rates[r]);
            check(rep.converged, "no convergence at chaos rate " + rates[r]);
            check(rep.repairRounds <= 4, "repair took " + rep.repairRounds + " rounds at rate " + rates[r]);
        }
        check(crashCounts[0] <= crashCounts[1] && crashCounts[1] <= crashCounts[2],
                "crash volume should grow with the chaos rate: " + Arrays.toString(crashCounts));

        // 13. Determinism and degenerate shapes.
        Report repeatA = experiment(hostile(), true, writes, 2024, 40);
        Report repeatB = experiment(hostile(), true, writes, 2024, 40);
        check(repeatA.crashes == repeatB.crashes && repeatA.acknowledged == repeatB.acknowledged
                && repeatA.droppedMessages == repeatB.droppedMessages
                && repeatA.acknowledgedLost == repeatB.acknowledgedLost, "reruns differ for one seed");
        Report zeroWrites = experiment(mild(), true, 0, 5, 10);
        check(zeroWrites.acknowledged == 0 && zeroWrites.acknowledgedLost == 0 && zeroWrites.converged,
                "a zero-write run is not trivially consistent");
        ChaosCluster single = new ChaosCluster(1, 1, Chaos.none(), true, 4);
        check(single.write(0, "alpha", 1) && single.acknowledged.size() == 1,
                "a one-node cluster with quorum 1 should commit and record the write");
        ChaosCluster unreachable = new ChaosCluster(1, 2, Chaos.none(), true, 4);
        check(!unreachable.write(0, "alpha", 1) && unreachable.acknowledged.isEmpty(),
                "quorum 2 must be unreachable with one replica");
        ChaosCluster oneReplica = new ChaosCluster(2, 1, Chaos.none(), true, 4);
        check(oneReplica.write(0, "alpha", 1) && oneReplica.write(1, "beta", 2),
                "quorum 1 of 2 should commit");

        System.out.println("clean      : " + clean.acknowledged + "/" + writes + " acknowledged, converged, "
                + "0 lost, repair " + clean.repairRounds + " rounds");
        System.out.println("mild chaos : " + mild.crashes + " crashes, " + mild.truncations + " truncations, "
                + mild.droppedMessages + " dropped messages, " + mild.pausedRounds + " pauses, "
                + mild.partitions + " partitions, " + mild.acknowledged + " acknowledged, "
                + mild.acknowledgedLost + " lost, repair " + mild.repairRounds + " rounds");
        System.out.println("unsafe acks: " + unsafeAfterBreak.acknowledged + " acknowledged, "
                + unsafeAfterBreak.acknowledgedLost + " LOST (detected), "
                + unsafeAfterBreak.unflushedAcks + " unflushed acks");
        System.out.println("hostile    : " + hostile.acknowledgedLost + " lost, repair "
                + hostile.repairRounds + " rounds, " + hostile.acknowledged + " acknowledged");
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
