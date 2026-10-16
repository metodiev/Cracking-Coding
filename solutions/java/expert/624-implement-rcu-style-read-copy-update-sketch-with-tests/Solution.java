/** Task 624 - Implement RCU-style read-copy-update sketch with tests. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * Read-copy-update in a nutshell: readers never lock and never block a writer. A writer that
 * changes a shared structure builds a *copy* (sharing the untouched suffix of the chain by
 * object identity), publishes the new root with one atomic store, and only then reclaims the
 * nodes it dropped. That reclamation is deferred by one *grace period*: it must wait until
 * every reader that was inside a read-side critical section when the new version was
 * published has passed a *quiescent state* (left the critical section).
 *
 * This file is a sketch without real threads: "readers" and "writers" are step machines
 * interleaved by an in-memory scheduler whose order comes from a fixed-seed PRNG, and a
 * quiescent state is simply a reader reporting that it finished its walk. Time is a logical
 * step counter, so "reclaim happened after the reader went quiescent" is a decidable
 * predicate rather than a wall-clock hope.
 *
 * Two deliberate simplifications, called out because they are not what the technique is
 * about: (a) the published root is a plain field written by one writer at a time, no
 * CAS-versioned pointer, because the point of the sketch is the grace period, not publication
 * racing; (b) "freeing" is modelled by a `reclaimed` flag - a reader that touches a reclaimed
 * node is counted as a use-after-free, which is exactly what the grace period must prevent.
 *
 * Verified: (1) structural sharing - an update copies only the changed prefix and the suffix
 * keeps its object identity; (2) atomicity - every reader snapshot equals one published
 * version exactly; (3) grace period - each reclaim happens strictly after the publish and
 * only once every reader active at publish time went quiescent, and no reader ever observed a
 * reclaimed node; (4) accounting - every node ever created is either live in the chain or
 * reclaimed exactly once; (5) with reclamation made eager (no grace period) the same scripted
 * interleaving *does* produce an observed reclaimed node - the negative control.
 */
import java.util.*;

/** Immutable value node: an update never mutates one, it builds a replacement. */
class RcuNode {
    final int key;
    final int value;
    final RcuNode next;
    boolean reclaimed;

    RcuNode(int key, int value, RcuNode next) {
        this.key = key;
        this.value = value;
        this.next = next;
    }
}

/** Shared state: the published root, a logical clock and the reclamation ledger. */
class RcuWorld {
    RcuNode head;
    int step;
    final boolean eager;
    final boolean[] pending;              // reader -> still owes a quiescent state
    final List<List<Integer>> versions = new ArrayList<>();
    int created;                          // every node ever allocated in this world
    int reclaims;
    int nodesReclaimed;
    int doubleReclaims;
    int observedReclaimed;                // a reader touched a reclaimed node
    boolean reclaimNotAfterPublish;

    RcuWorld(boolean eager, int readers) {
        this.eager = eager;
        this.pending = new boolean[readers];
    }

    int pendingCount() {
        int count = 0;
        for (boolean p : pending) if (p) count++;
        return count;
    }

    void recordVersion() {
        versions.add(keysOf(head));
    }

    boolean isVersion(List<Integer> snapshot) {
        return versions.contains(snapshot);
    }

    static List<Integer> keysOf(RcuNode head) {
        List<Integer> keys = new ArrayList<>();
        for (RcuNode cur = head; cur != null; cur = cur.next) keys.add(cur.key);
        return keys;
    }

    int liveCount() {
        return keysOf(head).size();
    }

    /** Reclaim dropped nodes; `safeStep` is the clock reading at which the grace period ended. */
    void reclaim(List<RcuNode> retired, int safeStep, int publishStep) {
        reclaims++;
        if (safeStep <= publishStep) reclaimNotAfterPublish = true;
        for (RcuNode node : retired) {
            if (node.reclaimed) {
                doubleReclaims++;
                continue;
            }
            node.reclaimed = true;
            nodesReclaimed++;
        }
    }
}

class RcuReader {
    final int id;
    boolean active;
    int startedAt = -1;
    int quiescentAt = -1;
    List<Integer> seen = new ArrayList<>();
    final List<List<Integer>> snapshots = new ArrayList<>();

    RcuReader(int id) {
        this.id = id;
    }
}

abstract class RcuFrame {
    boolean done;

    abstract void step();
}

/** Reader: enter a critical section, walk the version it captured, then go quiescent. */
class ReaderFrame extends RcuFrame {
    final RcuWorld world;
    final RcuReader reader;
    final int criticalSections;
    int completed = 1;
    RcuNode cur;
    int stage;

    ReaderFrame(RcuWorld world, RcuReader reader, int criticalSections) {
        this.world = world;
        this.reader = reader;
        this.criticalSections = criticalSections;
    }

    void step() {
        switch (stage) {
            case 0:
                reader.active = true;
                reader.startedAt = world.step;
                reader.seen = new ArrayList<>();
                cur = world.head;                 // one atomic load of the root
                stage = 1;
                break;
            case 1:
                if (cur == null) {
                    stage = 2;
                    break;
                }
                if (cur.reclaimed) world.observedReclaimed++;
                reader.seen.add(cur.key);
                cur = cur.next;
                break;
            case 2:
                reader.active = false;
                reader.quiescentAt = world.step;  // quiescent state
                world.pending[reader.id] = false;
                reader.snapshots.add(reader.seen);
                if (completed == criticalSections) {
                    done = true;
                } else {
                    completed++;
                    stage = 0;
                }
                break;
            default:
                done = true;
        }
    }
}

/** Writer: copy-on-write publish, then defer reclamation until the grace period ends. */
class WriterFrame extends RcuFrame {
    final RcuWorld world;
    final RcuReader[] readers;
    final int key;
    final int newValue;                  // -1 deletes
    final List<RcuNode> retired = new ArrayList<>();
    final List<RcuReader> activeAtPublish = new ArrayList<>();
    int publishStep = -1;
    int waited;
    int stage;
    boolean published;

    WriterFrame(RcuWorld world, RcuReader[] readers, int key, int newValue) {
        this.world = world;
        this.readers = readers;
        this.key = key;
        this.newValue = newValue;
    }

    void step() {
        switch (stage) {
            case 0:
                buildAndPublish();
                if (!published) {
                    done = true;
                    break;
                }
                stage = 1;
                break;
            case 1:
                if (!world.eager && world.pendingCount() > 0) {
                    waited++;                             // synchronize_rcu() blocks here
                    break;
                }
                if (!world.eager) {
                    for (RcuReader r : activeAtPublish) {
                        Solution.check(r.quiescentAt >= publishStep, "grace period: reader " + r.id
                                + " quiescent at " + r.quiescentAt + " must be >= publish step "
                                + publishStep);
                    }
                }
                world.reclaim(retired, world.step, publishStep);
                done = true;
                break;
            default:
                done = true;
        }
    }

    /** Build the new chain (sharing the untouched suffix), publish it, list dropped nodes. */
    void buildAndPublish() {
        List<Integer> keys = new ArrayList<>();
        Map<Integer, RcuNode> byKey = new HashMap<>();
        Map<Integer, Integer> values = new HashMap<>();
        for (RcuNode cur = world.head; cur != null; cur = cur.next) {
            keys.add(cur.key);
            byKey.put(cur.key, cur);
            values.put(cur.key, cur.value);
        }
        boolean existed = byKey.containsKey(key);
        if (newValue == -1) keys.remove(Integer.valueOf(key));
        else if (!existed) {
            keys.add(key);
            Collections.sort(keys);
        } else values.put(key, newValue);

        Set<RcuNode> kept = Collections.newSetFromMap(new IdentityHashMap<>());
        RcuNode newHead = null;
        for (int i = keys.size() - 1; i >= 0; i--) {
            int k = keys.get(i);
            Integer v = newValue != -1 && k == key ? Integer.valueOf(newValue) : values.get(k);
            RcuNode old = byKey.get(k);
            // Reuse an old node only when it is unchanged *and* its successor is already part
            // of the new chain - that is what makes the untouched suffix shared.
            if (old != null && v.equals(old.value) && old.next == newHead) {
                newHead = old;
            } else {
                newHead = new RcuNode(k, v, newHead);
                world.created++;
            }
            kept.add(newHead);
        }
        for (RcuNode cur = world.head; cur != null; cur = cur.next) {
            if (!kept.contains(cur)) retired.add(cur);
        }
        if (newHead == world.head) return;            // nothing changed: no publish at all
        published = true;
        world.head = newHead;                         // single atomic publish
        world.recordVersion();
        publishStep = world.step;
        for (RcuReader r : readers) {
            world.pending[r.id] = r.active;
            if (r.active) activeAtPublish.add(r);
        }
    }
}

public class Solution {
    static long seed = 0x9E3779B97F4A7C15L;

    static int nextInt(int bound) {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Math.floorMod(z, bound);
    }

    static void run(List<RcuFrame> frames, RcuWorld world) {
        int ticks = 0;
        while (true) {
            List<RcuFrame> live = new ArrayList<>();
            for (RcuFrame f : frames) if (!f.done) live.add(f);
            if (live.isEmpty()) return;
            for (int i = live.size() - 1; i > 0; i--) {
                int j = nextInt(i + 1);
                RcuFrame tmp = live.get(i);
                live.set(i, live.get(j));
                live.set(j, tmp);
            }
            for (RcuFrame f : live) f.step();
            world.step++;
            check(++ticks < 20000, "schedule terminates");
        }
    }

    /** One scripted micro-step plus one clock tick. */
    static void tick(RcuWorld world, RcuFrame frame) {
        frame.step();
        world.step++;
    }

    static RcuWorld worldWith(int[] keys, boolean eager, int readers) {
        RcuWorld world = new RcuWorld(eager, readers);
        RcuNode head = null;
        for (int i = keys.length - 1; i >= 0; i--) {
            head = new RcuNode(keys[i], i, head);
            world.created++;
        }
        world.head = head;
        world.recordVersion();
        return world;
    }

    static RcuNode nodeWith(RcuNode head, int key) {
        for (RcuNode cur = head; cur != null; cur = cur.next) if (cur.key == key) return cur;
        return null;
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    public static void main(String[] args) {
        copyOnWriteBasics();
        structuralSharing();
        eagerNegativeControl();
        scheduledGracePeriod();
        System.out.println("PASS");
    }

    /** Insert/delete semantics, no-op detection and reclamation accounting. */
    static void copyOnWriteBasics() {
        RcuWorld world = worldWith(new int[0], false, 1);
        check(world.versions.size() == 1 && world.versions.get(0).isEmpty(), "empty version");

        WriterFrame ins = new WriterFrame(world, new RcuReader[0], 5, 5);
        tick(world, ins);
        check(RcuWorld.keysOf(world.head).equals(Arrays.asList(5)), "insert into empty list");
        tick(world, ins);
        check(world.reclaims == 1 && world.nodesReclaimed == 0, "nothing dropped yet");
        check(ins.waited == 0, "no reader to wait for");

        WriterFrame dup = new WriterFrame(world, new RcuReader[0], 5, 5);
        tick(world, dup);
        check(world.versions.size() == 2, "same-value write publishes no new version");
        check(dup.done, "no-op writer finishes immediately");

        RcuNode only = world.head;
        WriterFrame del = new WriterFrame(world, new RcuReader[0], 5, -1);
        tick(world, del);
        tick(world, del);
        check(world.head == null && world.versions.size() == 3, "delete the last element");
        check(world.reclaims == 2 && world.nodesReclaimed == 1 && only.reclaimed,
                "the unlinked node is reclaimed once");

        WriterFrame missing = new WriterFrame(world, new RcuReader[0], 9, -1);
        tick(world, missing);
        check(world.versions.size() == 3 && world.head == null,
                "delete of a missing key is a no-op");

        RcuWorld multi = worldWith(new int[]{1, 3, 5, 7}, false, 0);
        WriterFrame add = new WriterFrame(multi, new RcuReader[0], 4, 4);
        tick(multi, add);
        tick(multi, add);
        check(RcuWorld.keysOf(multi.head).equals(Arrays.asList(1, 3, 4, 5, 7)), "sorted insert");
        WriterFrame delHead = new WriterFrame(multi, new RcuReader[0], 1, -1);
        tick(multi, delHead);
        tick(multi, delHead);
        check(RcuWorld.keysOf(multi.head).equals(Arrays.asList(3, 4, 5, 7)), "delete head");
        WriterFrame delTail = new WriterFrame(multi, new RcuReader[0], 7, -1);
        tick(multi, delTail);
        tick(multi, delTail);
        check(RcuWorld.keysOf(multi.head).equals(Arrays.asList(3, 4, 5)), "delete tail");
        check(multi.nodesReclaimed + multi.liveCount() == multi.created,
                "every created node is live or reclaimed");
        check(multi.doubleReclaims == 0 && !multi.reclaimNotAfterPublish, "reclamation ledger clean");
    }

    /** An update copies the changed prefix and keeps the untouched suffix's identity. */
    static void structuralSharing() {
        RcuWorld world = worldWith(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, false, 0);
        RcuNode oldHead = world.head;
        RcuNode oldThree = nodeWith(oldHead, 3);
        RcuNode oldFour = nodeWith(oldHead, 4);
        WriterFrame w = new WriterFrame(world, new RcuReader[0], 3, 99);
        tick(world, w);
        check(w.retired.size() == 3, "3 prefix nodes dropped: " + w.retired.size());
        tick(world, w);
        check(nodeWith(world.head, 4) == oldFour, "suffix node 4 keeps its identity");
        check(nodeWith(world.head, 3) != oldThree && nodeWith(world.head, 3).value == 99,
                "the changed key is a fresh node with the new value");
        check(world.head.next != oldHead.next, "the prefix was copied, not mutated in place");
        check(world.head != oldHead && oldThree.value == 2 && RcuWorld.keysOf(oldHead).size() == 8,
                "old version left untouched (immutable nodes)");
        check(world.nodesReclaimed == 3 && world.observedReclaimed == 0, "prefix reclaimed only");
        check(nodeWith(world.head, 8) != null && !nodeWith(world.head, 8).reclaimed,
                "shared suffix never reclaimed");
    }

    /** Reclaiming without a grace period: the reader walks straight into freed memory. */
    static void eagerNegativeControl() {
        check(observeAfterUpdate(false) == 0, "RCU hides the dropped version from a live reader");
        check(observeAfterUpdate(true) > 0, "eager reclamation exposes the dropped version");
    }

    static int observeAfterUpdate(boolean eager) {
        RcuWorld world = worldWith(new int[]{1, 2, 3, 4, 5}, eager, 1);
        RcuReader reader = new RcuReader(0);
        ReaderFrame rf = new ReaderFrame(world, reader, 1);
        tick(world, rf);                              // enter the critical section
        tick(world, rf);                              // read key 1
        WriterFrame wf = new WriterFrame(world, new RcuReader[]{reader}, 4, 40);
        tick(world, wf);                              // publish
        for (int i = 0; i < 12 && !(rf.done && wf.done); i++) {
            tick(world, rf);
            tick(world, wf);
        }
        check(rf.done && wf.done, "both frames finished");
        check(reader.seen.contains(1) && reader.seen.size() >= 2, "reader captured the old version");
        if (!eager) check(wf.waited > 0, "writer had to wait for the reader");
        return world.observedReclaimed;
    }

    /** Random-but-reproducible interleaving: readers must never see a reclaimed node. */
    static void scheduledGracePeriod() {
        RcuWorld world = worldWith(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, false, 2);
        RcuReader[] readers = {new RcuReader(0), new RcuReader(1)};
        ReaderFrame r0 = new ReaderFrame(world, readers[0], 3);
        ReaderFrame r1 = new ReaderFrame(world, readers[1], 3);
        WriterFrame w0 = new WriterFrame(world, readers, 11, 110);
        WriterFrame w1 = new WriterFrame(world, readers, 9, -1);
        for (int i = 0; i < 3; i++) {                 // put both readers inside a section first
            tick(world, r0);
            tick(world, r1);
        }
        check(readers[0].active && readers[1].active, "both readers active at publish time");

        run(new ArrayList<>(Arrays.asList(r0, r1, w0, w1)), world);

        check(world.reclaims == 2, "both writers reclaimed once");
        check(w0.waited > 0 && w1.waited > 0, "both writers waited for a grace period: "
                + w0.waited + "/" + w1.waited);
        check(world.observedReclaimed == 0, "no reader observed a reclaimed node");
        check(world.doubleReclaims == 0 && !world.reclaimNotAfterPublish,
                "no node reclaimed twice and never before its publish");
        check(world.nodesReclaimed + world.liveCount() == world.created,
                "created == live + reclaimed (" + world.created + " vs " + world.nodesReclaimed
                        + " + " + world.liveCount() + ")");
        check(world.versions.size() == 3, "initial + two published versions");
        check(RcuWorld.keysOf(world.head).equals(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 10, 11)),
                "final version: " + RcuWorld.keysOf(world.head));

        for (RcuReader r : readers) {
            check(r.snapshots.size() == 3, "three critical sections per reader");
            for (List<Integer> snap : r.snapshots) {
                check(world.isVersion(snap), "each snapshot equals a published version: " + snap);
                List<Integer> sorted = new ArrayList<>(snap);
                Collections.sort(sorted);
                check(sorted.equals(snap), "snapshot is a sorted version: " + snap);
            }
        }
        for (RcuNode cur = world.head; cur != null; cur = cur.next) {
            check(!cur.reclaimed, "live chain holds no reclaimed node (key " + cur.key + ")");
        }
    }
}
