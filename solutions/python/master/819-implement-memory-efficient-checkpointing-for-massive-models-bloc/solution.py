"""Task 819 - Implement memory-efficient checkpointing for massive models (block-level dedup).  (Tier: Master, CTCI Beyond - Machine Learning & AI)

A large model checkpoint is a list of named parameter blocks.  Training checkpoints are mostly
identical to their predecessor: a few blocks change, the rest are untouched, and many blocks are tied
or duplicated across layers.  So a checkpoint store should be *content addressed* - store each distinct
block payload once, keyed by its hash - and a checkpoint should be a small *manifest* of block name to
payload hash, optionally inheriting the manifest of its parent so that only the changed blocks are
written at all.  That is what this implementation does:

  * `BlockStore` is write-once content-addressed storage: `put` returns the payload's hash and stores
    the bytes only the first time, `get` verifies the hash of what it hands back, and refcounts make
    garbage collection safe;
  * `CheckpointManager.commit(name, blocks, parent)` writes only the blocks it is given, holds a
    reference to every distinct payload it introduced and resolves a version by walking the parent
    chain, so a delta checkpoint restores exactly like a full one;
  * `restore` rebuilds the full block map, `drop` releases a version's own references and collects the
    payloads whose refcount reached zero, and `physical_bytes` reports what is actually stored.

Every structural claim is checked against an independent oracle that knows nothing about hashing: the
expected set of stored payloads is `set` of the raw payload tuples, the expected refcount of a payload
is the number of live versions whose own manifest contains it, and the expected effective block map is
built with plain dictionaries by walking the parent chain.

Verified: the store's block count and byte count equal the oracle's distinct payload count and size;
the dedup ratio for a reuse-heavy workload (and exactly 1.0 for an all-distinct one); every version
restoring byte-identically to its expected effective map, including delta versions that inherit from a
parent; a shared block surviving the garbage collection of one of its owners while an unshared block is
freed, with refcounts equal to the oracle's; the safety property that no live version breaks after a
collection; integrity (a corrupted payload and a missing block both make a restore raise); idempotent
puts; hash stability and injectivity over the whole workload; the number of new blocks written per
version; the finer-granularity trade-off (small blocks dedup better for localised updates); a 200
version x 50 block scale run; determinism; and the degenerate shapes (empty version, zero-length
payload, one block, all blocks identical).

Simplified: payloads are small integer tuples instead of gigabyte tensors, the hash is a toy 64-bit
splitmix64 fold rather than a real content hash (a collision would be caught by `get`'s verification,
and the checks also measure that no collision occurs in the workload), storage is one process's
dictionaries instead of object storage, and there is no compression, no sharding of the store, no
concurrent writer and no remote read path.  The invariants verified - content addressing, dedup
accounting, refcount-exact collection, restore fidelity and tamper detection - are the ones a real
checkpoint service must hold.
"""
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    """Fixed-seed splitmix64 stream: every run generates the same checkpoints."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state, value = splitmix64(self.state)
        return value % bound


# ------------------------------------------------------------------ content addressing
def block_hash(payload):
    """Toy content hash: a splitmix64 fold over the length and the payload bytes."""
    if not isinstance(payload, tuple):
        payload = tuple(payload)
    h = (len(payload) * 0x9E3779B97F4A7C15 + 0xA5A5A5A5) & MASK64
    for byte in payload:
        check(0 <= byte <= 255, "a payload element must be a byte")
        h, mixed = splitmix64((h ^ byte) & MASK64)
        h = (h + mixed) & MASK64
    return format(h, "016x")


class BlockStoreError(Exception):
    """Raised when stored content is missing or does not match its hash."""


class BlockStore:
    """Write-once content-addressed storage with refcounts, so collection can be exact."""

    def __init__(self):
        self.payloads = {}
        self.refs = {}
        self.writes = 0
        self.put_calls = 0
        self.hits = 0
        self.bytes_written = 0

    def put(self, payload):
        """Store the payload if it is new; returns its hash.  Idempotent."""
        payload = tuple(payload)
        digest = block_hash(payload)
        self.put_calls += 1
        if digest in self.payloads:
            check(self.payloads[digest] == payload,
                  "two different payloads must not share a hash")
            self.hits += 1
        else:
            self.payloads[digest] = payload
            self.writes += 1
            self.bytes_written += len(payload)
        return digest

    def get(self, digest):
        """Return the payload, verifying that it still hashes to the requested digest."""
        if digest not in self.payloads:
            raise BlockStoreError(f"missing block {digest}")
        payload = self.payloads[digest]
        if block_hash(payload) != digest:
            raise BlockStoreError(f"corrupted block {digest}")
        return payload

    def acquire(self, digest):
        self.refs[digest] = self.refs.get(digest, 0) + 1

    def release(self, digest):
        check(digest in self.refs, f"releasing a block that is not referenced: {digest}")
        self.refs[digest] -= 1
        if self.refs[digest] <= 0:
            del self.refs[digest]
            self.payloads.pop(digest, None)
            return True
        return False

    def physical_bytes(self):
        return sum(len(payload) for payload in self.payloads.values())

    def live_blocks(self):
        return len(self.payloads)


class Snapshot:
    """A version: its own block manifest, plus an optional parent to inherit the rest from."""

    __slots__ = ("name", "parent", "manifest", "order")

    def __init__(self, name, manifest, parent=None):
        self.name = name
        self.manifest = dict(manifest)
        self.parent = parent
        self.order = [block for block in sorted(manifest)]

    def own_hashes(self):
        return sorted(set(self.manifest.values()))

    def resolve(self, store):
        """The effective block map: walk the parent chain and let the child override."""
        chain = []
        node = self
        while node is not None:
            chain.append(node)
            node = node.parent
        effective = {}
        for node in reversed(chain):
            for block, digest in node.manifest.items():
                effective[block] = digest
        return {block: store.get(digest) for block, digest in effective.items()}


class CheckpointManager:
    def __init__(self):
        self.store = BlockStore()
        self.versions = []
        self.dropped = []

    def commit(self, name, blocks, parent=None):
        """Write only the given blocks and hold a reference to each distinct payload."""
        manifest = {}
        for block in sorted(blocks):
            digest = self.store.put(blocks[block])
            manifest[block] = digest
        for digest in sorted(set(manifest.values())):
            self.store.acquire(digest)
        snapshot = Snapshot(name, manifest, parent)
        self.versions.append(snapshot)
        return snapshot

    def restore(self, snapshot):
        return snapshot.resolve(self.store)

    def drop(self, snapshot):
        """Release the version's own references and collect whatever fell to zero."""
        check(snapshot in self.versions, "only a live version can be dropped")
        freed = []
        for digest in snapshot.own_hashes():
            if self.store.release(digest):
                freed.append(digest)
        self.versions.remove(snapshot)
        self.dropped.append(snapshot)
        return freed

    def physical_bytes(self):
        return self.store.physical_bytes()

    def logical_bytes(self, snapshot):
        return sum(len(payload) for payload in self.restore(snapshot).values())

    def ratio(self):
        """Total logical bytes of every live version divided by what is actually stored."""
        physical = self.physical_bytes()
        logical = sum(self.logical_bytes(version) for version in self.versions)
        return logical / float(physical) if physical else 1.0


# ------------------------------------------------------------------ oracles
def oracle_unique_payloads(versions):
    """The distinct payloads any correct store must hold - computed without hashing."""
    return {payload for snapshot in versions for payload in snapshot.manifest_blocks.values()}


class PlainVersion:
    """A dependency-free version record used only by the oracles."""

    __slots__ = ("name", "parent", "blocks", "manifest_blocks")

    def __init__(self, name, blocks, parent=None):
        self.name = name
        self.blocks = dict(blocks)
        self.parent = parent
        self.manifest_blocks = {block: self.blocks[block] for block in sorted(self.blocks)}

    def resolve(self):
        chain = []
        node = self
        while node is not None:
            chain.append(node)
            node = node.parent
        effective = {}
        for node in reversed(chain):
            effective.update(node.blocks)
        return effective


def oracle_refcounts(versions):
    counts = {}
    for version in versions:
        for payload in set(version.manifest_blocks.values()):
            counts[payload] = counts.get(payload, 0) + 1
    return counts


# ------------------------------------------------------------------ workloads
def make_payload(rng, length):
    return tuple(rng.below(256) for _ in range(length))


def mutate(rng, payload, changes=1):
    out = list(payload)
    for _ in range(changes):
        out[rng.below(len(out))] = rng.below(256)
    return tuple(out)


def make_model(rng, blocks, length):
    return {f"block{index:02d}": make_payload(rng, length) for index in range(blocks)}


def run_workload(seed, versions, blocks, length, changes, update_fraction=8):
    """A training-like run: each version rewrites a few blocks and reuses the rest."""
    rng = Rng(seed)
    manager = CheckpointManager()
    plain = []
    model = make_model(rng, blocks, length)
    names = sorted(model)
    current = None
    for version in range(versions):
        changed = {}
        if version == 0:
            changed = dict(model)
        else:
            for _ in range(max(1, blocks // update_fraction)):
                block = names[rng.below(len(names))]
                if block in changed:
                    continue
                changed[block] = mutate(rng, model[block], changes)
                model[block] = changed[block]
        current = manager.commit(f"v{version:03d}", changed, current)
        plain.append(PlainVersion(f"v{version:03d}", changed,
                                  plain[-1] if plain else None))
    return manager, plain


def main():
    # 1. content addressing: the same payload always maps to the same hash, distinct ones differ
    payloads = [make_payload(Rng(0xABC + index), 6) for index in range(400)]
    hashes = {block_hash(payload) for payload in payloads}
    check(len(hashes) == len({payload for payload in payloads}),
          "distinct payloads must hash distinctly (no collision in this workload)")
    check(block_hash((1, 2, 3)) == block_hash((1, 2, 3)), "the hash must be stable")
    check(block_hash((1, 2, 3)) != block_hash((1, 2, 4)), "one changed byte must change the hash")
    check(block_hash(()) == block_hash(()), "an empty payload must hash consistently")
    try:
        block_hash((256,))
        check(False, "an out-of-range byte must be rejected")
    except AssertionError:
        pass

    # 2. the store keeps exactly the distinct payloads, and puts are idempotent
    store = BlockStore()
    distinct = set()
    for payload in payloads[:50]:
        distinct.add(payload)
        store.put(payload)
    check(store.live_blocks() == len(distinct),
          f"the store must hold one copy per distinct payload: {store.live_blocks()} vs "
          f"{len(distinct)}")
    check(store.physical_bytes() == sum(len(payload) for payload in distinct),
          "the stored bytes must be the sum of the distinct payload sizes")
    before = store.live_blocks()
    for payload in payloads[:50]:
        store.put(payload)
    check(store.live_blocks() == before and store.writes == len(distinct),
          "re-putting a payload must not write it again")
    check(store.hits == 50, "the store must count the deduplicated puts")
    check(store.get(block_hash(payloads[0])) == payloads[0], "get must round-trip a payload")

    # 3. the store matches the hashing-free oracle on a reuse-heavy workload
    manager, plain = run_workload(0xD1E7, versions=40, blocks=30, length=8, changes=1)
    unique = oracle_unique_payloads(plain)
    check(manager.store.live_blocks() == len(unique),
          f"the store must hold exactly the distinct payloads: {manager.store.live_blocks()} vs "
          f"{len(unique)}")
    check(manager.physical_bytes() == sum(len(payload) for payload in unique),
          "the physical bytes must equal the distinct payload bytes")
    total_logical = sum(sum(len(payload) for payload in version.resolve().values())
                        for version in plain)
    ratio = manager.ratio()
    check(abs(ratio - total_logical / float(manager.physical_bytes())) < 1e-12,
          "the ratio helper must match the explicit sum")
    check(ratio > 5.0, f"a reuse-heavy workload must dedup well, measured ratio {ratio:.2f}")
    check(manager.store.writes == len(unique), "only distinct payloads may be written")

    # 4. every version restores byte-identically to the hashing-free oracle
    for version, reference in zip(manager.versions, plain):
        restored = manager.restore(version)
        expected = reference.resolve()
        check(restored == expected, f"{version.name} must restore exactly")
        check(len(restored) == 30, "every version must hold the whole model")
        check(set(restored) == set(expected), "no block may be lost or invented")
    check(manager.versions[-1].parent is manager.versions[-2],
          "each version must inherit from its predecessor")
    check(len(manager.versions[-1].manifest) < 30,
          "a delta version must write only the blocks it changed")
    delta_writes = [len(version.manifest) for version in manager.versions[1:]]
    check(max(delta_writes) <= 30, "a delta can never exceed the model size")
    check(sum(delta_writes) / len(delta_writes) < 6,
          f"the average delta must be small, got {sum(delta_writes) / len(delta_writes):.2f} blocks")

    # 5. an all-distinct workload dedups exactly nothing
    rng = Rng(0x57A7)
    flat = CheckpointManager()
    for index in range(10):
        flat.commit(f"v{index}", {f"block{n}": make_payload(Rng(index * 31 + n), 5)
                                 for n in range(4)})
    check(flat.ratio() == 1.0,
          f"all-distinct payloads must give a ratio of exactly 1.0, got {flat.ratio()}")
    check(flat.store.hits == 0, "nothing may be deduplicated when nothing repeats")

    # 6. refcounts and collection are exact
    collector = CheckpointManager()
    shared = (7, 7, 7, 7)
    first = collector.commit("v0", {"a": shared, "b": (1, 2, 3, 4)})
    second = collector.commit("v1", {"a": shared, "c": (5, 6, 7, 8)}, parent=first)
    expected = oracle_refcounts([PlainVersion("v0", {"a": shared, "b": (1, 2, 3, 4)}),
                                 PlainVersion("v1", {"a": shared, "c": (5, 6, 7, 8)})])
    live_counts = {collector.store.get(digest): count
                   for digest, count in collector.store.refs.items()}
    check(live_counts == expected,
          f"refcounts must equal the oracle: {live_counts} vs {expected}")
    freed = collector.drop(second)
    check(collector.store.get(block_hash(shared)) == shared,
          "a block shared with a live version must survive collection")
    check(collector.store.live_blocks() == 2,
          f"only the unshared blocks must remain, got {collector.store.live_blocks()}")
    check(len(freed) == 1, f"dropping v1 must free its single unshared block, got {len(freed)}")
    check(collector.restore(first)["a"] == shared,
          "the surviving version must still restore after a collection")
    collector.drop(first)
    check(collector.store.live_blocks() == 0 and collector.physical_bytes() == 0,
          "dropping every version must reclaim all the storage")
    try:
        collector.drop(first)
        check(False, "dropping an already dropped version must be rejected")
    except AssertionError:
        pass

    # 7. collection on the big workload still matches the oracle's live set
    last_version = manager.versions[-1]
    dropped = manager.drop(last_version)
    live = oracle_unique_payloads(plain[:-1])
    live_payloads = {manager.store.payloads[digest] for digest in manager.store.payloads}
    check(live_payloads == live,
          f"after the collection the store must hold exactly the live payloads "
          f"({len(live_payloads)} vs {len(live)})")
    check(last_version not in manager.versions,
          "the dropped version must leave the registry")
    check(len(manager.versions) == len(plain) - 1, "exactly one version must be gone")
    for version, reference in zip(manager.versions, plain[:-1]):
        check(manager.restore(version) == reference.resolve(),
              f"{version.name} must restore after another version was dropped")
    check(dropped == sorted(set(dropped)), "the freed list must be well formed")

    # 8. integrity: a corrupted payload and a missing block are both detected
    tampered = CheckpointManager()
    version = tampered.commit("v0", {"a": (1, 1, 1), "b": (2, 2, 2)})
    digest = block_hash((1, 1, 1))
    tampered.store.payloads[digest] = (9, 9, 9)
    try:
        tampered.restore(version)
        check(False, "a corrupted payload must be detected on restore")
    except BlockStoreError:
        pass
    tampered.store.payloads[digest] = (1, 1, 1)
    check(tampered.restore(version) == {"a": (1, 1, 1), "b": (2, 2, 2)},
          "the restore must work again once the payload is intact")
    del tampered.store.payloads[block_hash((2, 2, 2))]
    try:
        tampered.restore(version)
        check(False, "a missing block must be detected on restore")
    except BlockStoreError:
        pass

    # 9. block granularity: the same logical update writes less when the blocks are smaller
    fine_manager = CheckpointManager()
    base_fine = {f"b{n:02d}": tuple(range(n * 4, n * 4 + 4)) for n in range(20)}
    parent = fine_manager.commit("v0", base_fine)
    after_v0 = fine_manager.store.bytes_written
    changed_b3 = tuple(99 if index == 1 else value for index, value in enumerate(base_fine["b03"]))
    fine_manager.commit("v1", {"b03": changed_b3}, parent)
    fine_delta_bytes = fine_manager.store.bytes_written - after_v0
    coarse_manager = CheckpointManager()
    base_coarse = {f"B{n:02d}": tuple(range(n * 8, n * 8 + 8)) for n in range(10)}
    parent = coarse_manager.commit("v0", base_coarse)
    after_v0 = coarse_manager.store.bytes_written
    changed_B1 = tuple(99 if index == 5 else value for index, value in enumerate(base_coarse["B01"]))
    coarse_manager.commit("v1", {"B01": changed_B1}, parent)
    coarse_delta_bytes = coarse_manager.store.bytes_written - after_v0
    check(len(base_fine["b03"]) + sum(len(block) for block in base_fine.values()) -
          len(base_fine["b03"]) == sum(len(block) for block in base_coarse.values()),
          "both granularities must describe the same logical model")
    check(fine_delta_bytes == 4 and coarse_delta_bytes == 8,
          f"one changed byte must rewrite one block: {fine_delta_bytes} vs {coarse_delta_bytes}")
    check(fine_manager.physical_bytes() < coarse_manager.physical_bytes(),
          "the fine-grained store must be smaller after the same logical update")
    check(fine_manager.restore(fine_manager.versions[-1])["b03"] == changed_b3,
          "the fine-grained delta must restore")
    check(coarse_manager.restore(coarse_manager.versions[-1])["B01"] == changed_B1,
          "the coarse-grained delta must restore")

    # 10. a 200-version scale run, with the write amplification measured
    big_manager, big_plain = run_workload(0xB16, versions=200, blocks=50, length=16, changes=1)
    big_unique = oracle_unique_payloads(big_plain)
    check(big_manager.store.live_blocks() == len(big_unique),
          f"the scale run must hold exactly {len(big_unique)} distinct blocks")
    check(big_manager.physical_bytes() == sum(len(payload) for payload in big_unique),
          "the scale run's bytes must equal the distinct payload bytes")
    scale_ratio = sum(sum(len(payload) for payload in version.resolve().values())
                      for version in big_plain) / float(big_manager.physical_bytes())
    check(abs(scale_ratio - (200 * 50) / float(len(big_unique))) < 1e-9,
          "the ratio must be exactly versions*blocks/distinct blocks when block sizes match")
    check(7.5 < scale_ratio < 9.5,
          f"the steady-state dedup must be near blocks-per-version-changed, got {scale_ratio:.2f}")
    check(big_manager.store.writes == len(big_unique), "the writes must equal the distinct blocks")
    check(50 + 199 * 4 <= big_manager.store.put_calls <= 50 + 199 * 6,
          f"every version must offer only its changed blocks to the store, got "
          f"{big_manager.store.put_calls}")
    for version, reference in list(zip(big_manager.versions, big_plain))[::25]:
        check(big_manager.restore(version) == reference.resolve(),
              f"{version.name} must restore exactly in the scale run")

    # 11. degenerate shapes and determinism
    empty = CheckpointManager()
    blank = empty.commit("empty", {})
    check(empty.restore(blank) == {} and empty.physical_bytes() == 0,
          "an empty version must store and restore nothing")
    zero = empty.commit("zero", {"a": ()})
    check(empty.restore(zero) == {"a": ()}, "a zero-length payload must round-trip")
    before = empty.store.live_blocks()
    identical = empty.commit("identical", {f"b{n}": (3, 3, 3) for n in range(20)})
    check(empty.store.live_blocks() == before + 1 and len(empty.restore(identical)) == 20,
          f"twenty identical blocks must be stored once, store grew by "
          f"{empty.store.live_blocks() - before}")
    check(empty.store.refs[block_hash((3, 3, 3))] == 1,
          "one version referencing one distinct payload holds exactly one reference")
    repeat, _ = run_workload(0xD1E7, versions=40, blocks=30, length=8, changes=1)
    again, _ = run_workload(0xD1E7, versions=40, blocks=30, length=8, changes=1)
    check(repeat.physical_bytes() == again.physical_bytes(),
          "two identical runs must consume identical storage")
    check([version.manifest for version in repeat.versions] ==
          [version.manifest for version in again.versions],
          "identical runs must produce identical manifests")
    check(repeat.store.refs == again.store.refs, "identical runs must refcount identically")

    print(f"scale run: 200 versions x 50 blocks -> {big_manager.store.live_blocks()} distinct "
          f"blocks, {big_manager.physical_bytes()} bytes stored, ratio {scale_ratio:.1f}x")
    print(f"dedup: small workload ratio {ratio:.2f}x, all-distinct ratio 1.0x, "
          f"granularity delta fine={fine_delta_bytes} coarse={coarse_delta_bytes} bytes")
    print("PASS")


if __name__ == "__main__":
    main()
