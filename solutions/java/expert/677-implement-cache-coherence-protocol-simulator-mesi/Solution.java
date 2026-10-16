/** Task 677 - Implement cache coherence protocol simulator (MESI). (Tier: Expert, CTCI Beyond — Computer Architecture)
 *
 * A snooping-bus MESI model: N private caches over A addresses plus a shared memory, with the four
 * states Invalid / Shared / Exclusive / Modified. Reads that miss issue BusRd (a dirty copy is flushed
 * to memory and downgraded to Shared, clean-shared copies stay shared, and the requester ends Exclusive
 * when it is the only holder), writes that hit in Modified or Exclusive are silent, and any other write
 * issues BusRdX with a flush of a dirty copy followed by invalidation of every other copy.
 *
 * Verified after every single operation: the SWMR invariant (at most one Modified and then all others
 * Invalid, at most one Exclusive and then all others Invalid), the coherence value invariant (every valid
 * copy and, when no cache is Modified, memory itself holds the last written value) and the sequential
 * model (every read returns exactly the last value written by any processor), plus the bus accounting
 * identities readMisses == BusRd, writeMisses + writeHitUpgrades == BusRdX, invalidations == the number
 * of independently observed transitions into Invalid, and that a dirty line is always flushed before its
 * copy is dropped (memory is up to date whenever nothing is Modified). Workloads: directed state-machine
 * walks, a deterministic pseudo-random sequence and a write ping-pong that shows the ownership cost.
 */
import java.util.*;

final class MesiCore {
    static final int I = 0, S = 1, E = 2, M = 3;

    final int id;
    final int[] state;
    final int[] data;
    int readHits, writeHits, readMisses, writeMisses, silentUpgrades, upgradeRdX;

    MesiCore(int id, int addresses) {
        this.id = id;
        state = new int[addresses];
        data = new int[addresses];
    }

    String stateName(int addr) {
        return switch (state[addr]) {
            case I -> "I";
            case S -> "S";
            case E -> "E";
            default -> "M";
        };
    }
}

final class Coherence {
    final List<MesiCore> cores = new ArrayList<>();
    final int[] memory;
    final int addresses;
    int busRd, busRdX, flushes, invalidations, silentWrites, reads, writes;
    final int[] lastWritten;

    Coherence(int cores, int addresses, int initialValue) {
        this.addresses = addresses;
        memory = new int[addresses];
        lastWritten = new int[addresses];
        Arrays.fill(memory, initialValue);
        Arrays.fill(lastWritten, initialValue);
        for (int i = 0; i < cores; i++) this.cores.add(new MesiCore(i, addresses));
    }

    int cpuRead(int core, int addr) {
        MesiCore me = cores.get(core);
        reads++;
        if (me.state[addr] != MesiCore.I) {
            me.readHits++;
            return me.data[addr];
        }
        me.readMisses++;
        busRd++;
        boolean dirtyFound = false, shared = false;
        for (MesiCore other : cores) {
            if (other == me || other.state[addr] == MesiCore.I) continue;
            shared = true;
            if (other.state[addr] == MesiCore.M) {                  // flush and downgrade the dirty owner
                memory[addr] = other.data[addr];
                other.state[addr] = MesiCore.S;
                flushes++;
                dirtyFound = true;
            } else if (other.state[addr] == MesiCore.E) {
                other.state[addr] = MesiCore.S;                     // clean exclusive becomes shared
            }
        }
        int value = memory[addr];
        me.data[addr] = value;
        me.state[addr] = shared || dirtyFound ? MesiCore.S : MesiCore.E;
        return value;
    }

    void cpuWrite(int core, int addr, int value) {
        MesiCore me = cores.get(core);
        writes++;
        if (me.state[addr] == MesiCore.M || me.state[addr] == MesiCore.E) {
            if (me.state[addr] == MesiCore.E) me.silentUpgrades++;
            me.state[addr] = MesiCore.M;
            me.data[addr] = value;
            me.writeHits++;
            silentWrites++;
            lastWritten[addr] = value;
            return;
        }
        if (me.state[addr] == MesiCore.S) {                          // upgrade: invalidate the sharers
            me.writeHits++;
            me.upgradeRdX++;
            busRdX++;
            for (MesiCore other : cores)
                if (other != me && other.state[addr] != MesiCore.I) {
                    other.state[addr] = MesiCore.I;
                    invalidations++;
                }
            me.state[addr] = MesiCore.M;
            me.data[addr] = value;
            lastWritten[addr] = value;
            return;
        }
        me.writeMisses++;                                            // write miss: flush then invalidate
        busRdX++;
        for (MesiCore other : cores) {
            if (other == me) continue;
            if (other.state[addr] == MesiCore.M) {
                memory[addr] = other.data[addr];
                flushes++;
            }
            if (other.state[addr] != MesiCore.I) {
                other.state[addr] = MesiCore.I;
                invalidations++;
            }
        }
        me.state[addr] = MesiCore.M;
        me.data[addr] = value;
        lastWritten[addr] = value;
    }

    /** Number of Invalid states observed in a snapshot, used to check the invalidation counter. */
    int[] snapshotStates(int addr) {
        int[] snap = new int[cores.size()];
        for (int i = 0; i < cores.size(); i++) snap[i] = cores.get(i).state[addr];
        return snap;
    }

    int newInvalidations(int addr, int[] before) {
        int count = 0;
        for (int i = 0; i < cores.size(); i++)
            if (before[i] != MesiCore.I && cores.get(i).state[addr] == MesiCore.I) count++;
        return count;
    }

    /** Returns a description of the first violated invariant, or null when everything holds. */
    String checkInvariants() {
        for (int addr = 0; addr < addresses; addr++) {
            int modified = 0, exclusive = 0, valid = 0;
            for (MesiCore c : cores) {
                if (c.state[addr] == MesiCore.M) modified++;
                if (c.state[addr] == MesiCore.E) exclusive++;
                if (c.state[addr] != MesiCore.I) {
                    valid++;
                    if (c.data[addr] != lastWritten[addr])
                        return "stale copy in cache " + c.id + " at " + addr + ": " + c.data[addr]
                                + " != " + lastWritten[addr];
                }
            }
            if (modified > 1) return "two Modified copies at " + addr;
            if (modified == 1 && valid != 1) return "Modified line still shared at " + addr;
            if (exclusive > 1) return "two Exclusive copies at " + addr;
            if (exclusive == 1 && valid != 1) return "Exclusive line still shared at " + addr;
            if (modified == 0 && memory[addr] != lastWritten[addr])
                return "memory stale at " + addr + " while nothing is Modified";
        }
        return null;
    }

    boolean busAccountingOk() {
        int rd = 0, rdx = 0;
        for (MesiCore c : cores) {
            rd += c.readMisses;
            rdx += c.writeMisses + c.upgradeRdX;            // Exclusive upgrades are silent, Shared upgrades are not
        }
        return rd == busRd && rdx == busRdX;
    }

    String stats() {
        return "BusRd=" + busRd + " BusRdX=" + busRdX + " flushes=" + flushes + " invalidations=" + invalidations
                + " silentWrites=" + silentWrites;
    }
}

public class Solution {
    static long state = 0x4D35123L;

    static long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    public static void main(String[] args) {
        Coherence one = new Coherence(1, 2, 0);
        check(one.cpuRead(0, 0) == 0 && one.cores.get(0).stateName(0).equals("E"),
                "a single reader ends Exclusive, got " + one.cores.get(0).stateName(0));
        one.cpuWrite(0, 0, 7);
        check(one.cores.get(0).stateName(0).equals("M") && one.busRdX == 0 && one.silentWrites == 1,
                "an Exclusive write upgrades silently, got " + one.stats());
        check(one.cpuRead(0, 0) == 7 && one.busRd == 1, "reading the Modified line is a hit with no bus traffic");
        check(one.checkInvariants() == null && one.busAccountingOk(), "single-cache invariants hold");

        Coherence two = new Coherence(2, 1, 0);
        two.cpuRead(0, 0);
        two.cpuRead(1, 0);
        check(two.cores.get(0).stateName(0).equals("S") && two.cores.get(1).stateName(0).equals("S")
                        && two.busRd == 2 && two.flushes == 0,
                "the second reader shares the clean line, got " + two.stats() + " states "
                        + two.cores.get(0).stateName(0) + "/" + two.cores.get(1).stateName(0));
        two.cpuWrite(0, 0, 5);
        check(two.cores.get(0).stateName(0).equals("M") && two.cores.get(1).stateName(0).equals("I")
                        && two.busRdX == 1 && two.invalidations == 1,
                "a write into Shared takes ownership and invalidates the sharer, got " + two.stats());
        check(two.cpuRead(1, 0) == 5 && two.flushes == 1 && two.cores.get(0).stateName(0).equals("S"),
                "the dirty copy is flushed on the remote read, got " + two.stats());
        check(two.checkInvariants() == null && two.busAccountingOk(), "two-cache invariants hold");

        Coherence hw = new Coherence(3, 1, 0);
        hw.cpuRead(0, 0);
        hw.cpuRead(1, 0);
        hw.cpuRead(2, 0);
        hw.cpuWrite(2, 0, 42);
        check(hw.cores.get(0).stateName(0).equals("I") && hw.cores.get(1).stateName(0).equals("I")
                        && hw.cores.get(2).stateName(0).equals("M") && hw.invalidations == 2,
                "a write to a three-way shared line invalidates both sharers, got " + hw.stats());
        check(hw.cpuRead(0, 0) == 42 && hw.memory[0] == 42 && hw.flushes == 1,
                "the value is visible after the flush: read " + hw.cpuRead(0, 0));
        hw.cpuWrite(2, 0, 100);
        check(hw.cores.get(2).stateName(0).equals("M") && hw.cpuRead(1, 0) == 100,
                "the newest value wins and the owner stays Modified after the flush");
        check(hw.checkInvariants() == null && hw.busAccountingOk(), "three-cache invariants hold");

        Coherence walk = new Coherence(3, 4, 11);
        walk.cpuRead(1, 2);
        int[] before = walk.snapshotStates(2);
        walk.cpuWrite(2, 2, 20);
        check(walk.newInvalidations(2, before) == 1 && walk.invalidations == 1,
                "the invalidation counter matches the independently observed S->I transition");
        check(walk.cpuRead(0, 3) == 11 && walk.cores.get(0).stateName(3).equals("E"),
                "a different address is fetched independently and stays Exclusive");
        walk.cpuWrite(0, 3, 12);
        check(walk.cores.get(0).stateName(3).equals("M") && walk.cpuRead(2, 3) == 12,
                "the second address behaves like a private line until it is shared");
        check(walk.checkInvariants() == null && walk.busAccountingOk(), "walk invariants hold");

        Coherence rnd = new Coherence(4, 5, 0);
        int ops = 600, violations = 0, readValueErrors = 0, invalidationsObserved = 0;
        for (int i = 0; i < ops; i++) {
            long r = nextLong();
            int core = (int) ((r >>> 33) % 4);
            int addr = (int) ((r >>> 17) % 5);
            boolean isWrite = (r & 1) == 0;
            int[] snap = rnd.snapshotStates(addr);
            if (isWrite) rnd.cpuWrite(core, addr, (int) ((r >>> 7) % 1000) + 1);
            else if (rnd.cpuRead(core, addr) != rnd.lastWritten[addr]) readValueErrors++;
            invalidationsObserved += rnd.newInvalidations(addr, snap);
            if (rnd.checkInvariants() != null) violations++;
        }
        check(violations == 0, "no invariant is ever violated across " + ops + " random operations");
        check(readValueErrors == 0, "every read returned the last written value (" + readValueErrors + " errors)");
        check(rnd.invalidations == invalidationsObserved,
                "invalidation accounting: " + rnd.invalidations + " vs " + invalidationsObserved + " observed");
        check(rnd.busAccountingOk(), "bus transactions match the per-cache miss counters: " + rnd.stats());
        check(rnd.busRd > 0 && rnd.busRdX > 0 && rnd.flushes > 0,
                "the random workload exercised all three bus events: " + rnd.stats());

        Coherence ping = new Coherence(2, 1, 0);
        ping.cpuWrite(0, 0, 1);
        for (int i = 1; i <= 20; i++) ping.cpuWrite(i % 2, 0, i + 1);
        check(ping.busRdX == 21 && ping.invalidations == 20,
                "write ping-pong costs one ownership transfer per write and invalidates the previous owner, got "
                        + ping.stats());
        check(ping.checkInvariants() == null && ping.cpuRead(1, 0) == 21,
                "the ping-pong value is coherent: " + ping.cpuRead(1, 0));
        int pingWrites = ping.writes;
        check(pingWrites == 21, "the ping-pong performed 21 writes, got " + pingWrites);

        Coherence quiet = new Coherence(2, 1, 9);
        quiet.cpuRead(0, 0);
        quiet.cpuRead(0, 0);
        quiet.cpuRead(0, 0);
        check(quiet.busRd == 1 && quiet.cores.get(0).readHits == 2,
                "repeated local reads touch the bus once, got " + quiet.stats());
        check(quiet.cpuRead(1, 0) == 9 && quiet.flushes == 0 && quiet.busRdX == 0,
                "sharing a clean line needs no flush or invalidation");

        Coherence again = new Coherence(4, 5, 0);
        Coherence first = new Coherence(4, 5, 0);
        for (int i = 0; i < 200; i++) {
            long r = nextLong();
            int core = (int) ((r >>> 33) % 4), addr = (int) ((r >>> 17) % 5);
            if ((r & 1) == 0) { first.cpuWrite(core, addr, i + 1); again.cpuWrite(core, addr, i + 1); }
            else { first.cpuRead(core, addr); again.cpuRead(core, addr); }
        }
        check(first.busRd == again.busRd && first.busRdX == again.busRdX && first.flushes == again.flushes,
                "the simulator is deterministic");

        System.out.println("directed walks : " + hw.stats());
        System.out.println("random 600 ops : " + rnd.stats() + " with 0 invariant violations");
        System.out.println("write ping-pong: " + ping.busRdX + " transfers for " + pingWrites + " writes");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
