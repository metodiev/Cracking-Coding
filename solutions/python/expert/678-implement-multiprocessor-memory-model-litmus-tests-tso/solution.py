"""Task 678 - Implement multiprocessor memory model litmus tests (TSO).  (Tier: Expert, Beyond — Networking, OS & Real-Time)

An operational model of Total Store Order and of sequential consistency, plus exhaustive
enumeration of every architected interleaving.

Machine model
  * each thread has a program, private registers and (under TSO) a FIFO store buffer;
  * a store is issued into the buffer and becomes globally visible only when it reaches the
    buffer head and is *committed* to memory; a load forwards the youngest buffered store to
    the same address, otherwise it reads memory;
  * because a thread executes its instructions in program order while stores are committed
    later, TSO preserves store->store (FIFO buffer), load->load and load->store order, but
    relaxes store->load - exactly the relaxation store buffering litmus tests expose;
  * a `fence` stalls the thread until its buffer drains;
  * SC is the same machine with `use_buffers=False` (stores become visible immediately).

`explore` performs an exhaustive search over all interleavings of thread steps and buffer
commits, and returns the set of observed register outcomes. Two litmus tests are used:
store buffering (T0: x=1; r1=y / T1: y=1; r2=x) and message passing (T0: x=1; y=1 /
T1: r1=y; r2=x). Verified results: under TSO store buffering produces all four outcomes
including (0,0) while SC produces only three; message passing forbids (1,0) under both;
and adding a fence to store buffering makes TSO agree with SC exactly. The SC enumerator is
cross-checked against a fully independent permutation oracle that executes every legal total
order of the instructions one at a time.

Simplified: two addresses, stores of the constant 1, no atomic read-modify-write
instructions, no speculation-induced reordering beyond the FIFO store buffer.
"""

from itertools import permutations


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def validate(programs):
    for program in programs:
        if not program:
            raise ValueError("every thread needs at least one instruction")
        for op in program:
            if op[0] == "store" and len(op) != 3:
                raise ValueError("a store needs an address and a value")
            if op[0] == "load" and len(op) != 3:
                raise ValueError("a load needs an address and a register")
            if op[0] not in ("store", "load", "fence"):
                raise ValueError("unknown instruction %r" % (op,))
    return programs


def mem_get(memory, addr):
    for name, value in memory:
        if name == addr:
            return value
    return 0


def mem_set(memory, addr, value):
    entries = dict(memory)
    entries[addr] = value
    return tuple(sorted(entries.items()))


def outcome_of(registers):
    """Canonical outcome: every thread's registers as sorted (name, value) pairs."""
    entries = []
    for regs in registers:
        entries.extend(regs.items() if isinstance(regs, dict) else regs)
    return tuple(sorted(entries))


def successors(programs, state, use_buffers):
    """Every legal next step: one instruction of one thread, or one buffer commit."""
    pcs, registers, buffers, memory = state
    steps = []
    for i, program in enumerate(programs):
        if pcs[i] >= len(program):
            continue
        op = program[pcs[i]]
        if op[0] == "fence":
            if buffers[i]:
                continue                     # the thread stalls until the buffer drains
            new_pcs = list(pcs)
            new_pcs[i] += 1
            steps.append((tuple(new_pcs), registers, buffers, memory))
        elif op[0] == "store":
            address, value = op[1], op[2]
            new_pcs = list(pcs)
            new_pcs[i] += 1
            if use_buffers:
                new_buffers = list(buffers)
                new_buffers[i] = buffers[i] + ((address, value),)
                steps.append((tuple(new_pcs), registers, tuple(new_buffers), memory))
            else:
                steps.append((tuple(new_pcs), registers, buffers,
                              mem_set(memory, address, value)))
        else:
            address, register = op[1], op[2]
            value = None
            if use_buffers:                  # store-to-load forwarding, youngest first
                for buffered_address, buffered_value in reversed(buffers[i]):
                    if buffered_address == address:
                        value = buffered_value
                        break
            if value is None:
                value = mem_get(memory, address)
            new_pcs = list(pcs)
            new_pcs[i] += 1
            new_registers = list(registers)
            new_registers[i] = tuple(sorted(dict(registers[i], **{register: value}).items()))
            steps.append((tuple(new_pcs), tuple(new_registers), buffers, memory))
    if use_buffers:
        for i in range(len(programs)):
            if buffers[i]:
                address, value = buffers[i][0]
                new_buffers = list(buffers)
                new_buffers[i] = buffers[i][1:]
                steps.append((pcs, registers, tuple(new_buffers), mem_set(memory, address, value)))
    return steps


def explore(programs, use_buffers=True, limit=100000):
    """Exhaustive interleaving search. Returns (outcomes, visited_states)."""
    validate(programs)
    addresses = sorted({op[1] for program in programs for op in program if op[0] != "fence"})
    start = (tuple(0 for _ in programs), tuple(() for _ in programs),
             tuple(() for _ in programs), tuple((addr, 0) for addr in addresses))
    visited = set()
    stack = [start]
    outcomes = set()
    while stack:
        state = stack.pop()
        if state in visited:
            continue
        if len(visited) >= limit:
            raise AssertionError("the state space exceeds the search limit")
        visited.add(state)
        pcs = state[0]
        if all(pcs[i] >= len(programs[i]) for i in range(len(programs))):
            outcomes.add(outcome_of(state[1]))
            continue
        stack.extend(successors(programs, state, use_buffers))
    return outcomes, len(visited)


def permutation_outcomes(programs):
    """Independent SC oracle: execute every total order of the instructions that respects
    each thread's program order, atomically, one instruction at a time."""
    flat = [(i, k) for i, program in enumerate(programs) for k in range(len(program))]
    outcomes = set()
    for order in permutations(flat):
        position = {item: index for index, item in enumerate(order)}
        if any(position[(i, k)] > position[(i, k + 1)]
               for i, program in enumerate(programs) for k in range(len(program) - 1)):
            continue
        memory = {}
        registers = [{} for _ in programs]
        for i, k in order:
            op = programs[i][k]
            if op[0] == "store":
                memory[op[1]] = op[2]
            elif op[0] == "load":
                registers[i][op[2]] = memory.get(op[1], 0)
        outcomes.add(outcome_of(registers))
    return outcomes


def store_buffering(fences=False):
    """T0: x=1; r1=y   T1: y=1; r2=x"""
    thread0 = [("store", "x", 1), ("fence",), ("load", "y", "r1")] if fences else \
              [("store", "x", 1), ("load", "y", "r1")]
    thread1 = [("store", "y", 1), ("fence",), ("load", "x", "r2")] if fences else \
              [("store", "y", 1), ("load", "x", "r2")]
    return [thread0, thread1]


def message_passing(fences=False):
    """T0: x=1; y=1   T1: r1=y; r2=x"""
    thread0 = [("store", "x", 1), ("fence",), ("store", "y", 1)] if fences else \
              [("store", "x", 1), ("store", "y", 1)]
    thread1 = [("load", "y", "r1"), ("fence",), ("load", "x", "r2")] if fences else \
              [("load", "y", "r1"), ("load", "x", "r2")]
    return [thread0, thread1]


def pairs(outcomes):
    return {tuple(value for _, value in outcome) for outcome in outcomes}


def main():
    # ---- store buffering --------------------------------------------------------------
    sb_tso, sb_tso_states = explore(store_buffering(), use_buffers=True)
    sb_sc, sb_sc_states = explore(store_buffering(), use_buffers=False)
    check(pairs(sb_tso) == {(0, 0), (0, 1), (1, 0), (1, 1)},
          "TSO store buffering must produce all four outcomes: %r" % (sorted(pairs(sb_tso)),))
    check((0, 0) in pairs(sb_tso),
          "the classic relaxation: both loads may read the old value under TSO")
    check((0, 0) not in pairs(sb_sc),
          "sequential consistency forbids (0,0) in store buffering")
    check(pairs(sb_sc) == {(0, 1), (1, 0), (1, 1)},
          "SC allows exactly the three other outcomes: %r" % (sorted(pairs(sb_sc)),))
    check(sb_sc == permutation_outcomes(store_buffering()),
          "the operational SC model must agree with the permutation oracle")
    check(sb_tso_states > sb_sc_states,
          "the store buffer adds states (%d vs %d)" % (sb_tso_states, sb_sc_states))

    # ---- message passing --------------------------------------------------------------
    mp_tso, _ = explore(message_passing(), use_buffers=True)
    mp_sc, _ = explore(message_passing(), use_buffers=False)
    check((1, 0) not in pairs(mp_tso),
          "message passing must never observe the flag without the data under TSO")
    check((0, 0) in pairs(mp_tso) and (1, 1) in pairs(mp_tso),
          "the other message-passing outcomes stay reachable: %r" % (sorted(pairs(mp_tso)),))
    check(pairs(mp_tso) == pairs(mp_sc),
          "TSO preserves store->store order, so message passing matches SC exactly")
    check(mp_sc == permutation_outcomes(message_passing()),
          "the SC message-passing model must agree with the permutation oracle")

    # ---- fences restore sequential consistency ----------------------------------------
    sb_fenced, _ = explore(store_buffering(fences=True), use_buffers=True)
    check(pairs(sb_fenced) == pairs(sb_sc),
          "a fence between the store and the load restores SC for store buffering: %r"
          % (sorted(pairs(sb_fenced)),))
    check((0, 0) not in pairs(sb_fenced), "the fenced store buffering test forbids (0,0)")
    mp_fenced, _ = explore(message_passing(fences=True), use_buffers=True)
    check(pairs(mp_fenced) == pairs(mp_tso),
          "TSO already orders the two stores, so fencing message passing changes nothing: %r"
          % (sorted(pairs(mp_fenced)),))
    check((1, 0) not in pairs(mp_fenced), "the mixed outcome stays forbidden with fences")
    check(pairs(mp_fenced) <= pairs(mp_tso),
          "fences can never create new outcomes")

    # ---- store-to-load forwarding -----------------------------------------------------
    forwarded, _ = explore([[("store", "x", 1), ("load", "x", "r1")]], use_buffers=True)
    check(pairs(forwarded) == {(1,)},
          "a thread must see its own buffered store: %r" % (pairs(forwarded),))
    stale, _ = explore([[("store", "x", 1), ("fence",), ("load", "x", "r1")]],
                       use_buffers=True)
    check(pairs(stale) == {(1,)}, "the value is the same after the fence")

    two_threads, _ = explore([[("store", "x", 1)], [("load", "x", "r1")]], use_buffers=True)
    check(pairs(two_threads) == {(0,), (1,)},
          "a buffered store is visible to the other thread only after the commit: %r"
          % (sorted(pairs(two_threads)),))
    two_threads_sc, _ = explore([[("store", "x", 1)], [("load", "x", "r1")]],
                                use_buffers=False)
    check(pairs(two_threads_sc) == {(0,), (1,)},
          "the same two outcomes exist under SC: %r" % (sorted(pairs(two_threads_sc)),))

    # ---- load->load order is preserved under TSO -------------------------------------
    load_order = [[("store", "x", 1), ("store", "y", 1)], [("load", "y", "r1"), ("load", "x", "r2")]]
    observed, _ = explore(load_order, use_buffers=True)
    check((1, 0) not in pairs(observed),
          "a thread must not observe the flag's store while missing the older data store: %r"
          % (sorted(pairs(observed)),))
    check((0, 0) in pairs(observed) and (1, 1) in pairs(observed),
          "the consistent observations stay reachable")

    # ---- boundary cases, errors and determinism --------------------------------------
    trivial, states = explore([[("load", "z", "r1")]])
    check(pairs(trivial) == {(0,)} and states >= 1, "an untouched address reads zero")
    none, _ = explore([[("fence",)], [("fence",)]])
    check(pairs(none) == {()}, "two threads with nothing to observe have one outcome")
    rejected = 0
    for bad in (lambda: explore([[]]),
                lambda: explore([[("frob", "x")]]),
                lambda: explore([[("store", "x")]]),
                lambda: explore([[("load", "x")]]),
                lambda: explore([("store", "x", 1)])):
        try:
            bad()
        except (ValueError, IndexError, TypeError):
            rejected += 1
    check(rejected == 5, "malformed programs must be rejected (%d of 5)" % rejected)
    check(explore(store_buffering(), use_buffers=True) == (sb_tso, sb_tso_states),
          "the exploration must be deterministic")
    check(permutation_outcomes(store_buffering()) == permutation_outcomes(store_buffering()),
          "the permutation oracle must be deterministic")
    limited = False
    try:
        explore(store_buffering(), use_buffers=True, limit=2)
    except AssertionError:
        limited = True
    check(limited, "a too small search limit must abort loudly")

    print("store buffering: TSO %r (%d states) vs SC %r (%d states)"
          % (sorted(pairs(sb_tso)), sb_tso_states, sorted(pairs(sb_sc)), sb_sc_states))
    print("message passing: TSO %r; fenced store buffering %r"
          % (sorted(pairs(mp_tso)), sorted(pairs(sb_fenced))))
    print("PASS")


if __name__ == "__main__":
    main()
