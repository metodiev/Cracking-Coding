"""Task 753 - Implement a control-flow integrity (CFI) checker skeleton.  (Tier: Expert, Beyond - Security & Privacy)

CFI checks every *indirect* control transfer against a statically computed set of legal targets.
This skeleton models a small program as basic blocks with four kinds of indirect transfer:

    ('icall_ind', slot, ret)   indirect call through a function pointer; the pointer's declared
                               type is looked up in SLOT_SIGNATURES, so the *call site* has an
                               expected type label "int,int->int" and so on
    ('ijmp', table)            indirect jump through a switch jump table (a declared target set)
    ('ret',)                   return: legal only to the address on the run-time return stack
    ('call', f, ret)           a *direct* call, whose target is a constant (still verified)

Policies, coarse to fine:
    none    - any function may be called indirectly                       (no protection)
    coarse  - only functions whose address is taken anywhere in the image  (blocklists the rest)
    fine    - additionally the callee's type label must equal the call site's expected label

The checker walks a *trace* of transfers, recomputes the shadow return stack (push the site's
declared return address on a call, require the popped top on every return) and reports the first
violation with its step index.  Legal sets are computed once per site and memoised; the trace comes
from a deterministic interpreter that only ever makes legal transfers, so a clean trace is the
positive control.  The invariant that gets verified exhaustively: for every indirect site, every
block in the image and every policy, the checker accepts exactly the targets that an independently
recomputed legality predicate accepts - i.e. the checker and the static target sets agree on the
whole state space, not just on the traces the interpreter happens to produce.

Simplified versus production CFI (LLVM CFI, Windows CFG): the program is a block list, not a binary;
type labels are canonical signature strings instead of vtable hash labels; there is no landing-pad
layout, no exception unwinding and no virtual-dispatch special case; return protection is the
classic shadow stack, and the frame layout is not modelled - only which value may be jumped to.
"""
MASK64 = (1 << 64) - 1

SIGNATURES = {
    "main": "int->int",
    "add": "int,int->int",
    "sub": "int,int->int",
    "log_i": "int->void",
    "log_s": "str->void",
    "gadget": "->void",
}

# block id -> (function, instruction list, terminator); 'mov' takes a function's address
BLOCKS = {
    0: ("main", [("mov", "fp", "add")], ("icall_ind", "fp", 1)),
    1: ("main", [("mov", "cb", "log_s")], ("icall_ind", "cb", 2)),
    2: ("main", [], ("jcc", 3, 6)),
    3: ("main", [], ("ijmp", "switch_a")),
    4: ("main", [], ("jmp", 6)),
    5: ("main", [], ("jmp", 6)),
    6: ("main", [("mov", "guarded", "log_i")], ("call", "add", 7)),
    7: ("main", [], ("halt",)),
    10: ("add", [], ("ret",)),
    20: ("sub", [], ("ret",)),
    30: ("log_i", [], ("ret",)),
    40: ("log_s", [], ("ret",)),
    50: ("gadget", [], ("ret",)),
}
ENTRIES = {name: block for block, (name, _, _) in BLOCKS.items() if block in (0, 10, 20, 30, 40, 50)}
SLOT_SIGNATURES = {"fp": "int,int->int", "cb": "int->void", "guarded": "int->void"}
JUMP_TABLES = {"switch_a": (4, 5)}
POLICIES = ("none", "coarse", "fine")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class ImageError(ValueError):
    """Raised when the block image is inconsistent (unknown target, empty legal set, ...)."""


class Image:
    """A tiny block image plus the static facts CFI needs: entries, labels, target sets."""

    def __init__(self, blocks, entries, slot_signatures, jump_tables, signatures, entry_func="main"):
        self.terminators = {}
        self.func_of_block = {}
        self.address_taken = set()
        for block, (func, instrs, term) in blocks.items():
            if func not in signatures:
                raise ImageError("unknown function %r for block %d" % (func, block))
            self.terminators[block] = term
            self.func_of_block[block] = func
            for instr in instrs:
                if instr[0] != "mov" or len(instr) != 3:
                    raise ImageError("malformed instruction %r" % (instr,))
                if instr[2] not in signatures:
                    raise ImageError("taking the address of unknown function %r" % (instr[2],))
                self.address_taken.add(instr[2])
        self.entries = dict(entries)
        self.entry_of = {name: block for block, name in
                         ((b, f) for b, f in self.func_of_block.items() if b in entries.values())}
        self.slot_signatures = dict(slot_signatures)
        self.jump_tables = {name: tuple(targets) for name, targets in jump_tables.items()}
        self.signatures = dict(signatures)
        self.entry = self.entries[entry_func]
        self._validate()

    # --------------------------------------------------------------- static validation
    def _validate(self):
        for name, block in self.entries.items():
            if self.func_of_block.get(block) != name:
                raise ImageError("entry of %r is not a block of %r" % (name, name))
        for name, targets in self.jump_tables.items():
            for target in targets:
                if target not in self.terminators:
                    raise ImageError("jump table %r targets unknown block %d" % (name, target))
        for block, term in self.terminators.items():
            if term[0] in ("jmp", "jcc"):
                for target in term[1:]:
                    if target not in self.terminators:
                        raise ImageError("block %d branches to unknown block %d" % (block, target))
            elif term[0] == "ijmp":
                if term[1] not in self.jump_tables:
                    raise ImageError("block %d uses unknown jump table %r" % (block, term[1]))
            elif term[0] in ("call", "icall_ind"):
                if term[0] == "call" and term[1] not in self.entries:
                    raise ImageError("block %d calls unknown function %r" % (block, term[1]))
                if term[0] == "icall_ind" and term[1] not in self.slot_signatures:
                    raise ImageError("block %d calls through undeclared slot %r" % (block, term[1]))
                if term[-1] not in self.terminators:
                    raise ImageError("block %d returns to unknown block %d" % (block, term[-1]))
            elif term[0] not in ("ret", "halt"):
                raise ImageError("unknown terminator %r" % (term[0],))
            if term[0] == "icall_ind":                 # every site needs a non-empty target set
                for policy in POLICIES:
                    if not self.legal_functions(block, policy):
                        raise ImageError("block %d has no legal callee under %s" % (block, policy))

    # --------------------------------------------------------------------- CFI target sets
    def expected_label(self, block):
        return self.slot_signatures[self.terminators[block][1]]

    def legal_functions(self, block, policy):
        """Functions this indirect call site may transfer to, per policy granularity."""
        if policy == "none":
            return frozenset(self.entries)
        if policy == "coarse":
            return frozenset(self.address_taken)
        return frozenset(f for f in self.address_taken
                         if self.signatures[f] == self.expected_label(block))

    def legal_blocks(self, block, policy):
        return frozenset(self.entries[f] for f in self.legal_functions(block, policy))

    def indirect_sites(self):
        return tuple(sorted(b for b, t in self.terminators.items() if t[0] == "icall_ind"))


def interpret(image, policy, seed, max_steps=200):
    """Deterministic interpreter: emits a trace of (block, kind, target) transfers."""
    state = seed & MASK64
    block = image.entry
    stack = []
    trace = []
    depths = []
    while len(trace) < max_steps:
        term = image.terminators[block]
        kind = term[0]
        depths.append(len(stack))
        if kind == "halt":
            trace.append((block, kind, None))
            break
        if kind == "ret":
            if not stack:
                trace.append((block, kind, None))
                break
            target = stack.pop()
        elif kind == "jmp":
            target = term[1]
        elif kind == "jcc":
            state, r = splitmix64(state)
            target = term[1] if r % 2 else term[2]
        elif kind == "ijmp":
            state, r = splitmix64(state)
            target = image.jump_tables[term[1]][r % len(image.jump_tables[term[1]])]
        else:                                     # call / icall_ind
            if kind == "call":
                callee = term[1]                  # a direct call: the target is a constant
            else:
                legal = sorted(image.legal_functions(block, policy))
                state, r = splitmix64(state)
                callee = legal[r % len(legal)]
            trace.append((block, kind, image.entry_of[callee]))
            stack.append(term[-1])
            block = image.entry_of[callee]
            continue
        trace.append((block, kind, target))
        block = target
    return trace, depths


def check_trace(image, trace, policy):
    """First CFI violation as (index, reason, depths) or (-1, None, depths) when the trace is clean."""
    stack = []
    depths = []
    for index, step in enumerate(trace):
        if not isinstance(step, tuple) or len(step) != 3:
            raise ImageError("malformed trace step %r" % (step,))
        block, kind, target = step
        if block not in image.terminators:
            raise ImageError("trace step %d names unknown block %r" % (index, block))
        term = image.terminators[block]
        depths.append(len(stack))
        if term[0] != kind:
            return index, "step %d: site %d executed as %r but is %r" % (index, block, kind, term[0]), depths
        if term[0] == "jmp":
            if target != term[1]:
                return index, "step %d: direct jump to %r, expected %d" % (index, target, term[1]), depths
        elif term[0] == "jcc":
            if target not in (term[1], term[2]):
                return index, "step %d: conditional branch to %r" % (index, target), depths
        elif term[0] == "ijmp":
            if target not in image.jump_tables[term[1]]:
                return index, "step %d: jump table %r has no case %d" % (index, term[1], target), depths
        elif term[0] == "call":
            expected = image.entry_of[term[1]]
            if target != expected:
                return index, "step %d: direct call to block %r, expected %d" % (index, target, expected), depths
            stack.append(term[-1])
        elif term[0] == "icall_ind":
            callee = image.func_of_block.get(target)
            if callee is None or target != image.entry_of.get(callee):
                return index, "step %d: indirect target %r is not a function entry" % (index, target), depths
            if callee not in image.legal_functions(block, policy):
                return index, ("step %d: %s CFI rejects %s at call site %d (expected %s)"
                               % (index, policy, callee, block, image.expected_label(block))), depths
            stack.append(term[-1])
        elif term[0] == "ret":
            if not stack:
                return index, "step %d: return with an empty shadow stack" % index, depths
            if stack[-1] != target:
                return index, ("step %d: return to %r does not match the pushed address %d"
                               % (index, target, stack[-1])), depths
            stack.pop()
        else:                                     # halt
            if target is not None:
                return index, "step %d: halt carries a target" % index, depths
    return -1, None, depths


def patch(trace, index, target):
    """Retarget a single transfer - the hijack an attacker would attempt."""
    mutated = list(trace)
    block, kind, _ = mutated[index]
    mutated[index] = (block, kind, target)
    return mutated


def first_step(trace, kind):
    for index, step in enumerate(trace):
        if step[1] == kind:
            return index
    return -1


def independent_legal(image, site, policy):
    """Recompute the legal callee set from the image without using the memoised cache."""
    term = image.terminators[site]
    slot_label = image.slot_signatures[term[1]]
    if policy == "fine":
        return tuple(sorted(e for f, e in image.entry_of.items()
                            if f in image.address_taken and image.signatures[f] == slot_label))
    if policy == "coarse":
        return tuple(sorted(image.entries[f] for f in image.address_taken))
    return tuple(sorted(image.entries.values()))


def main():
    image = Image(BLOCKS, ENTRIES, SLOT_SIGNATURES, JUMP_TABLES, SIGNATURES)

    # 1. static facts: which functions are address-taken, and the per-site target sets
    check(image.address_taken == {"add", "log_i", "log_s"}, "address-taken set from 'mov' instructions")
    check("sub" not in image.address_taken and "gadget" not in image.address_taken,
          "sub and gadget are never address-taken")
    check(image.legal_functions(0, "fine") == frozenset({"add"}), "site 0 expects int,int->int")
    check(image.legal_functions(1, "fine") == frozenset({"log_i"}), "site 1 expects int->void")
    check(image.legal_functions(0, "coarse") == frozenset({"add", "log_i", "log_s"}),
          "coarse CFI allows every address-taken function")
    check(image.legal_functions(0, "none") == frozenset(SIGNATURES), "the 'none' policy allows all")
    for site in image.indirect_sites():
        for a, b in (("fine", "coarse"), ("coarse", "none")):
            check(image.legal_functions(site, a) <= image.legal_functions(site, b),
                  "%s targets must be a subset of %s targets at site %d" % (a, b, site))

    # 2. malformed images are rejected before any check can pass vacuously
    broken = dict(BLOCKS)
    broken[3] = ("main", [], ("ijmp", "switch_missing"))
    no_match = dict(SLOT_SIGNATURES)
    no_match["fp"] = "float->float"          # no address-taken function has this signature
    for blocks, slots in ((broken, SLOT_SIGNATURES), (BLOCKS, no_match)):
        try:
            Image(blocks, ENTRIES, slots, JUMP_TABLES, SIGNATURES)
            raise AssertionError("an inconsistent image must be rejected")
        except ImageError:
            pass

    # 3. positive control: interpreter traces are legal, and the shadow stack mirrors the call depth
    fine_traces, coarse_traces = [], []
    for seed in range(20):
        trace, depths = interpret(image, "fine", 1000 + seed)
        fine_traces.append(trace)
        index, reason, checked_depths = check_trace(image, trace, "fine")
        check(index == -1, "fine CFI must accept its own trace: %s" % reason)
        check(checked_depths == depths, "the checker's shadow stack must mirror the interpreter's")
        check(len(trace) >= 8, "a trace must exercise several transfers")
        check(trace[-1][1] == "halt" and depths[-1] == 0,
              "a legit run must return to the top level with a balanced shadow stack")
        coarse_trace, _ = interpret(image, "coarse", 2000 + seed)
        coarse_traces.append(coarse_trace)

    # 4. negative controls: the classic hijacks, at known step indices
    legit = fine_traces[0]
    call_index = first_step(legit, "icall_ind")
    check(call_index >= 0, "the trace must contain an indirect call")
    # (a) target a function whose address was never taken - both granularities must reject it
    for policy in ("coarse", "fine"):
        index, reason, _ = check_trace(image, patch(legit, call_index, image.entry_of["sub"]), policy)
        check(index == call_index and "rejects" in reason,
              "%s CFI must reject a call to a non-address-taken function" % policy)
    # (b) type confusion: an address-taken function with the wrong signature - only fine CFI sees it
    confusion = patch(legit, call_index, image.entry_of["log_s"])
    index, reason, _ = check_trace(image, confusion, "fine")
    check(index == call_index and "expected int,int->int" in reason,
          "fine CFI must reject the type-confused callee")
    check(check_trace(image, confusion, "coarse")[0] == -1,
          "coarse CFI is blind to the type confusion (its known weakness)")
    # (c) an indirect call into the middle of a function, even though the block exists
    index, reason, _ = check_trace(image, patch(legit, call_index, 4), "coarse")
    check(index == call_index and "not a function entry" in reason,
          "an indirect branch into a non-entry block must be rejected")
    # (d) jump table, (e) corrupted direct branch, (f) corrupted direct call
    for kind, target, expected in (("ijmp", 10, "jump table"), ("jmp", 7, "direct jump"),
                                   ("call", image.entry_of["sub"], "direct call")):
        site = first_step(legit, kind)
        index, reason, _ = check_trace(image, patch(legit, site, target), "coarse")
        check(index == site and expected in reason, "a corrupted %s must be caught" % kind)

    # 5. return-address hijack: coarse and fine CFI cannot help, the shadow stack must
    ret_index = first_step(legit, "ret")
    check(ret_index >= 0, "the trace must contain a return")
    index, reason, _ = check_trace(image, patch(legit, ret_index, 3), "coarse")
    check(index == ret_index and "does not match the pushed address" in reason,
          "a ROP-style return hijack must be caught by the shadow stack")
    truncated = legit[:ret_index + 1] + [(image.entry_of["add"], "ret", 7)]
    index, reason, _ = check_trace(image, truncated, "fine")
    check(index == len(truncated) - 1 and "empty shadow stack" in reason,
          "a return without a matching call must be rejected")

    # 6. exhaustive soundness: the checker agrees with an independently recomputed predicate on
    #    every (indirect site, block, policy) combination - not just on interpreter traces
    agreement = 0
    for site in image.indirect_sites():
        for policy in POLICIES:
            legal = set(independent_legal(image, site, policy))
            for block in sorted(image.terminators):
                single = [(site, "icall_ind", block)]
                index, _, _ = check_trace(image, single, policy)
                check((index == -1) == (block in legal),
                      "checker/predicate disagree: site %d, policy %s, block %d" % (site, policy, block))
                agreement += 1

    # 7. coarse CFI really is weaker: its own traces contain type-confused calls
    confused = [t for t in coarse_traces
                if any(step[1] == "icall_ind"
                       and image.func_of_block[step[2]] not in image.legal_functions(step[0], "fine")
                       for step in t)]
    check(confused, "coarse target sets must sometimes pick a wrong-signature callee")
    for trace in confused:
        index, _, _ = check_trace(image, trace, "coarse")
        check(index == -1, "coarse CFI must accept its own trace")
        index, reason, _ = check_trace(image, trace, "fine")
        check(index >= 0, "fine CFI must reject the type-confused coarse trace")
    # and policy monotonicity over those traces: nothing fine accepts may be refused by coarse
    for trace in coarse_traces:
        fine_index, _, _ = check_trace(image, trace, "fine")
        coarse_index, _, _ = check_trace(image, trace, "coarse")
        none_index, _, _ = check_trace(image, trace, "none")
        if fine_index == -1:
            check(coarse_index == -1 and none_index == -1, "policy inclusions must be monotone")

    # 8. granularity statistics and boundary behaviour
    coarse_sizes = [len(image.legal_functions(s, "coarse")) for s in image.indirect_sites()]
    fine_sizes = [len(image.legal_functions(s, "fine")) for s in image.indirect_sites()]
    check(sum(fine_sizes) < sum(coarse_sizes),
          "fine-grained CFI must shrink the average target set: %s vs %s" % (fine_sizes, coarse_sizes))
    check(check_trace(image, [], "fine")[0] == -1, "an empty trace is trivially clean")
    check(check_trace(image, [], "fine")[2] == [], "an empty trace has an empty depth trace")
    for malformed in ([(99, "halt", None)], ["nonsense"]):
        try:
            check_trace(image, malformed, "fine")
            raise AssertionError("a malformed trace must be rejected: %r" % (malformed,))
        except ImageError:
            pass

    # 9. determinism of both the interpreter and the checker
    for seed in (1000, 1007):
        again, _ = interpret(image, "fine", seed)
        check(again == fine_traces[seed - 1000], "interpretation must be deterministic")
    check(check_trace(image, legit, "fine") == check_trace(image, legit, "fine"),
          "verdicts must be deterministic")

    print("753 sites=%d address_taken=%s" % (len(image.indirect_sites()), sorted(image.address_taken)))
    print("753 legal set sizes: coarse=%s fine=%s (mean %.2f -> %.2f)"
          % (coarse_sizes, fine_sizes, sum(coarse_sizes) / len(coarse_sizes),
             sum(fine_sizes) / len(fine_sizes)))
    print("753 traces: %d fine-clean, %d coarse traces with a type-confused call"
          % (len(fine_traces), len(confused)))
    print("753 checker/predicate agreement on %d (site, policy, block) cases" % agreement)
    print("PASS")


if __name__ == "__main__":
    main()
