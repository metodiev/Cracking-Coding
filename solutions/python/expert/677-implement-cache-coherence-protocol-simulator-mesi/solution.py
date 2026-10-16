"""Task 677 - Implement cache coherence protocol simulator (MESI).  (Tier: Expert, Beyond — Networking, OS & Real-Time)

A snooping MESI simulator: N cores each with a private cache of whole lines, a shared
memory, and a bus whose transactions every other cache snoops.

Transitions implemented
  * read miss            -> BusRd; any peer in M writes the line back and goes S, peers in
                            E go S, and the requester ends in E (nobody else had a copy) or S;
  * write hit in E or M  -> silent, stays M;
  * write hit in S       -> BusUpgr: peers invalidate, the line becomes M;
  * write miss           -> BusRdX: peers write back if dirty and invalidate, the requester
                            installs the line in M;
  * eviction of a dirty line -> write back to memory.

After *every* access the simulation is checked against the coherence invariants: at most one
cache may hold a line in M, an M or E copy excludes every other copy (single-writer), every
copy of a line holds the same data, and that data equals the most recently written value -
the last one is tracked in a shadow `expected` map that a correct protocol must always agree
with, which is what makes a read hit that returns stale data impossible to miss.

The scripted checks pin the state transitions, the write-back accounting, the fact that a
read hit generates no bus traffic, and the single-writer transition; a 300-operation random
sweep re-verifies the invariants after every step and reports the traffic statistics.

Simplified: no cache capacity/eviction policy (eviction is explicit), no store buffer or
out-of-order effects, and the bus is modelled as a synchronous snoop broadcast.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Line:
    __slots__ = ("state", "data")

    def __init__(self, state, data):
        self.state = state
        self.data = data


class MesiSystem:
    """A small MESI system; every access re-verifies the coherence invariants."""

    def __init__(self, cores=2, addresses=(0x100, 0x200), memory=None):
        if cores < 1:
            raise ValueError("at least one core is required")
        self.core_count = cores
        self.addresses = set(addresses)
        self.caches = [{} for _ in range(cores)]
        self.memory = {addr: 0 for addr in self.addresses}
        if memory:
            for addr, value in memory.items():
                if addr not in self.addresses:
                    raise ValueError("unknown address %r" % (addr,))
                self.memory[addr] = value
        self.expected = dict(self.memory)      # the most recently written value per line
        self.stats = {"reads": 0, "writes": 0, "hits": 0, "misses": 0, "bus_rd": 0,
                      "bus_rdx": 0, "bus_upgr": 0, "writebacks": 0, "invalidations": 0}
        self.checks = 0

    def _require(self, addr):
        if addr not in self.addresses:
            raise ValueError("address %r does not exist" % (addr,))

    def _snoop_bus_read(self, requester, addr):
        """BusRd: a dirty peer flushes, clean-exclusive peers downgrade; report sharing."""
        shared = False
        for i, cache in enumerate(self.caches):
            line = cache.get(addr)
            if i == requester or line is None:
                continue
            shared = True
            if line.state == "M":
                self.memory[addr] = line.data
                self.stats["writebacks"] += 1
                line.state = "S"
            elif line.state == "E":
                line.state = "S"
        return shared

    def _snoop_invalidate(self, requester, addr):
        for i, cache in enumerate(self.caches):
            line = cache.get(addr)
            if i == requester or line is None:
                continue
            if line.state == "M":
                self.memory[addr] = line.data
                self.stats["writebacks"] += 1
            del cache[addr]
            self.stats["invalidations"] += 1

    def read(self, core, addr):
        self._require(addr)
        self.stats["reads"] += 1
        cache = self.caches[core]
        line = cache.get(addr)
        if line is not None:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1
            self.stats["bus_rd"] += 1
            shared = self._snoop_bus_read(core, addr)
            line = Line("S" if shared else "E", self.memory[addr])
            cache[addr] = line
        value = line.data
        self.verify()
        return value

    def write(self, core, addr, value):
        self._require(addr)
        self.stats["writes"] += 1
        cache = self.caches[core]
        line = cache.get(addr)
        if line is not None and line.state in ("E", "M"):
            self.stats["hits"] += 1
        elif line is not None:                      # S: upgrade in place
            self.stats["hits"] += 1
            self.stats["bus_upgr"] += 1
            self._snoop_invalidate(core, addr)
        else:
            self.stats["misses"] += 1
            self.stats["bus_rdx"] += 1
            self._snoop_invalidate(core, addr)
            line = Line("M", value)
            cache[addr] = line
        line.state = "M"
        line.data = value
        self.expected[addr] = value
        self.verify()

    def evict(self, core, addr):
        """Drop a line; a dirty line must be written back first."""
        self._require(addr)
        cache = self.caches[core]
        line = cache.get(addr)
        if line is None:
            self.verify()
            return False
        if line.state == "M":
            self.memory[addr] = line.data
            self.stats["writebacks"] += 1
        del cache[addr]
        self.verify()
        return True

    def line_state(self, core, addr):
        line = self.caches[core].get(addr)
        return "I" if line is None else line.state

    def verify(self):
        """Coherence invariants; raises as soon as one is broken."""
        self.checks += 1
        for addr in self.addresses:
            copies = [(i, self.caches[i][addr]) for i in range(self.core_count)
                      if addr in self.caches[i]]
            dirty = [i for i, line in copies if line.state == "M"]
            check(len(dirty) <= 1, "two caches hold %r dirty: %r" % (addr, dirty))
            check(not (dirty and len(copies) > 1),
                  "a dirty copy of %r must exclude every other copy" % (addr,))
            exclusive = [i for i, line in copies if line.state == "E"]
            check(not (exclusive and len(copies) > 1),
                  "an exclusive copy of %r must exclude every other copy" % (addr,))
            check(all(line.data == copies[0][1].data for _, line in copies),
                  "caches disagree about the data of %r" % (addr,))
            check(all(line.data == self.expected[addr] for _, line in copies),
                  "a cache holds a stale copy of %r" % (addr,))
            check(all(line.state in ("S", "E", "M") for _, line in copies),
                  "unknown cache state for %r" % (addr,))

    def resident(self, core, addr):
        return addr in self.caches[core]


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_workload(seed, operations=300, cores=3, addresses=4):
    """A deterministic stream of reads, writes and evictions."""
    state = seed
    stream = []
    for _ in range(operations):
        state, raw = splitmix64(state)
        core = raw % cores
        state, raw = splitmix64(state)
        addr = 0x100 * (1 + raw % addresses)
        state, raw = splitmix64(state)
        kind = raw % 10
        if kind < 4:
            stream.append(("read", core, addr, None))
        elif kind < 9:
            stream.append(("write", core, addr, (raw // 10) % 97))
        else:
            stream.append(("evict", core, addr, None))
    return stream


def main():
    # ---- scripted transitions --------------------------------------------------------
    system = MesiSystem(2, (0x100, 0x200), {0x100: 42})
    check(system.line_state(0, 0x100) == "I", "caches start empty")
    check(system.read(0, 0x100) == 42, "a read miss must return the memory value")
    check(system.line_state(0, 0x100) == "E",
          "an unshared read miss leaves the line exclusive: %s" % system.line_state(0, 0x100))
    check(system.read(1, 0x100) == 42 and system.line_state(1, 0x100) == "S",
          "the second reader gets a shared copy")
    check(system.line_state(0, 0x100) == "S", "the first copy downgrades to shared")
    check(system.stats["bus_rd"] == 2, "each cold read costs one bus read")

    hits_before = system.stats["hits"]
    check(system.read(0, 0x100) == 42, "a read hit returns the cached value")
    check(system.stats["hits"] == hits_before + 1 and system.stats["bus_rd"] == 2,
          "a read hit must not generate bus traffic")

    system.write(1, 0x100, 7)
    check(system.line_state(1, 0x100) == "M", "a write hit in S upgrades to modified")
    check(system.line_state(0, 0x100) == "I", "the upgrade invalidates the other copy")
    check(system.stats["bus_upgr"] == 1 and system.stats["invalidations"] == 1,
          "the upgrade is one bus transaction and one invalidation")
    check(system.read(0, 0x100) == 7, "the reader must see the newest value")
    check(system.stats["writebacks"] == 1, "the dirty peer must flush on the read miss")
    check(system.line_state(1, 0x100) == "S", "the flushed peer downgrades to shared")

    system.write(0, 0x100, 9)
    check(system.line_state(0, 0x100) == "M" and system.line_state(1, 0x100) == "I",
          "a write hit in S invalidates the peer")
    check(system.memory[0x100] == 7, "memory stays stale until the dirty line is flushed")
    check(system.evict(0, 0x100) and system.memory[0x100] == 9
          and system.stats["writebacks"] == 2,
          "evicting a dirty line writes it back")
    check(system.read(1, 0x100) == 9, "after the write-back the value survives")
    check(system.evict(1, 0x100) and system.stats["writebacks"] == 2,
          "evicting a clean line writes nothing back")
    check(not system.evict(1, 0x100), "evicting a line that is not present is a no-op")
    check(system.evict(0, 0x200) is False, "nothing to evict for an untouched line")

    # ---- a single core is trivially coherent -----------------------------------------
    solo = MesiSystem(1, (0x100,), {0x100: 1})
    solo.read(0, 0x100)
    for _ in range(20):
        solo.write(0, 0x100, 5)
    check(solo.stats["misses"] == 1 and solo.stats["invalidations"] == 0
          and solo.stats["bus_upgr"] == 0 and solo.stats["writebacks"] == 0,
          "one core never needs the bus after the cold miss: %r" % (solo.stats,))
    check(solo.line_state(0, 0x100) == "M", "the single copy stays modified")

    # ---- ping-pong -------------------------------------------------------------------
    ping = MesiSystem(2, (0x100,), {0x100: 0})
    for i in range(12):
        ping.write(i % 2, 0x100, i + 1)
    check(ping.stats["invalidations"] >= 11,
          "every cross-core write must invalidate the other copy: %r" % (ping.stats,))
    check(ping.stats["writebacks"] >= 10, "each hand-over flushes the dirty line")
    check(ping.read(1, 0x100) == 12, "the last writer's value is what everyone sees")

    # ---- random sweep with invariants after every operation --------------------------
    state = 0xC0FFEE
    for label, cores in (("2 cores", 2), ("3 cores", 3), ("4 cores", 4)):
        system = MesiSystem(cores, tuple(0x100 * (1 + i) for i in range(4)),
                            **{"memory": {0x100: 3}})
        operations = reads = writes = 0
        for kind, core, addr, value in random_workload(state, 300, cores, 4):
            operations += 1
            if kind == "read":
                reads += 1
                observed = system.read(core, addr)
                check(observed == system.expected[addr],
                      "read %s returned a stale value: %r" % (label, observed))
            elif kind == "write":
                writes += 1
                system.write(core, addr, value)
            else:
                system.evict(core, addr)
        check(system.checks >= operations, "the invariants must be re-checked after every access")
        check(system.stats["reads"] == reads and system.stats["writes"] == writes,
              "every operation must be counted exactly once (%s)" % label)
        state += 7919

    sweep = MesiSystem(3, tuple(0x100 * (1 + i) for i in range(4)), **{"memory": {0x100: 3}})
    for kind, core, addr, value in random_workload(0xC0FFEE, 300, 3, 4):
        if kind == "read":
            sweep.read(core, addr)
        elif kind == "write":
            sweep.write(core, addr, value)
        else:
            sweep.evict(core, addr)
    check(sweep.stats["writebacks"] > 0 and sweep.stats["invalidations"] > 0,
          "the sweep must exercise flushing and invalidation: %r" % (sweep.stats,))
    repeated = MesiSystem(3, tuple(0x100 * (1 + i) for i in range(4)),
                          **{"memory": {0x100: 3}})
    for kind, core, addr, value in random_workload(0xC0FFEE, 300, 3, 4):
        if kind == "read":
            repeated.read(core, addr)
        elif kind == "write":
            repeated.write(core, addr, value)
        else:
            repeated.evict(core, addr)
    check(repeated.stats == sweep.stats and repeated.memory == sweep.memory,
          "the simulation must be deterministic")

    # ---- error handling ---------------------------------------------------------------
    rejected = 0
    for bad in (lambda: MesiSystem(0, (0x100,)),
                lambda: MesiSystem(1, (0x100,), {0x200: 1}),
                lambda: system.read(0, 0x999),
                lambda: system.write(0, 0x999, 1),
                lambda: system.evict(0, 0x999)):
        try:
            bad()
        except ValueError:
            rejected += 1
    check(rejected == 5, "invalid systems and addresses must be rejected (%d of 5)" % rejected)

    print("ping-pong: %r" % (ping.stats,))
    print("sweep (3 cores, 4 lines, 300 ops): %r" % (sweep.stats,))
    print("PASS")


if __name__ == "__main__":
    main()
