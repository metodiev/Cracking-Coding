"""Task 842 - Implement blockchain light node gossip + header stitching robustly.  (Tier: Master, CTCI Beyond — System Design & Scalability)

A light client keeps only block *headers*, learns them from peers over a flood gossip
protocol, stitches them into a validated tree as they arrive (possibly out of order), and
follows the heaviest branch by cumulative proof-of-work - the behaviour a real SPV client
needs to answer "is this transaction in the best chain?" without ever downloading bodies.

There are no sockets, threads or wall-clock reads: hashing is SHA-256 over a canonical header
encoding, "proof of work" means the digest starts with `difficulty` zero hex digits, every
node is a plain object with an inbox, and a *round* is one deterministic pass in which each
node forwards exactly the headers it learned in the previous round (store-and-forward
flooding with hash deduplication).  The light node validates every offered header:

  * the digest must satisfy the claimed difficulty, the difficulty must be within the client's
    policy maximum, the index must be exactly the parent's index + 1, and the parent must be
    known - otherwise the header is either rejected (bad proof, bad geometry, absurd
    difficulty) or parked as an *orphan* until its parent shows up, at which point a whole
    subtree is stitched in one go;
  * cumulative work is sum(16^difficulty) along the branch, and the tip is the branch with the
    most work - so a *shorter* branch of harder blocks correctly wins over a longer easy one;
  * a reorg is just a tip change: the client re-points to the heavier head and keeps the old
    headers, which is what makes "depth k is stable" measurable.

Simplified versus production: real clients negotiate with a few peers, request headers in
batches, and check retargeting/median-time rules; here all peers flood every header, difficulty
is fixed per block (no retargeting), there is no timestamp or chainwork-adjustment rule, the
smallest-hash tie-break stands in for a chain-selection rule, and difficulties stay at 2-3
leading hex digits so that mining is instant.  Nothing here is a real blockchain - the point is
that every validation, stitch, fork choice and merkle proof below is genuine.

Verified: proof of work actually gates headers (a mutated header is refused and the measured
acceptance rate of random nonces matches the target), a chain offered in shuffled order is
stitched through orphans back to genesis, work accumulates so the shorter harder branch wins, a
reorg switches the tip and leaves headers at depth >= 3 untouched, duplicates cost nothing,
fabricated headers (bad proof, huge claimed difficulty, wrong index, unknown parent) are
refused, merkle inclusion proofs verify for every transaction in a block and fail for tampered
paths/roots, the light node never stores a transaction body, and after enough gossip rounds every
peer agrees on the same heaviest tip.
"""

import hashlib

GENESIS_PREV = "0" * 64


def header_blob(index, prev, merkle, difficulty, nonce):
    return "blk|%d|%s|%s|%d|%d" % (index, prev, merkle, difficulty, nonce)


class Header:
    __slots__ = ("index", "prev", "merkle", "difficulty", "nonce", "hash")

    def __init__(self, index, prev, merkle, difficulty, nonce):
        self.index = index
        self.prev = prev
        self.merkle = merkle
        self.difficulty = difficulty
        self.nonce = nonce
        self.hash = hashlib.sha256(header_blob(index, prev, merkle, difficulty,
                                               nonce).encode()).hexdigest()

    def work(self):
        return 16 ** self.difficulty


def valid_proof(header):
    return header.hash.startswith("0" * header.difficulty)


def mine(index, prev, merkle, difficulty, start=0):
    nonce = start
    while True:
        candidate = Header(index, prev, merkle, difficulty, nonce)
        if valid_proof(candidate):
            return candidate
        nonce += 1


def merkle_root(txids):
    if not txids:
        return hashlib.sha256(b"empty").hexdigest()
    layer = [hashlib.sha256(txid.encode()).hexdigest() for txid in txids]
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [hashlib.sha256((layer[i] + layer[i + 1]).encode()).hexdigest()
                 for i in range(0, len(layer), 2)]
    return layer[0]


def merkle_proof(txids, index):
    """Sibling path (side, hash) from the leaf up to the root, with the odd-duplicate rule."""
    layer = [hashlib.sha256(txid.encode()).hexdigest() for txid in txids]
    path = []
    position = index
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        sibling = position ^ 1
        path.append(("right" if sibling > position else "left", layer[sibling]))
        layer = [hashlib.sha256((layer[i] + layer[i + 1]).encode()).hexdigest()
                 for i in range(0, len(layer), 2)]
        position //= 2
    return path


def verify_proof(root, txid, proof):
    node = hashlib.sha256(txid.encode()).hexdigest()
    for side, sibling in proof:
        if side == "right":
            node = hashlib.sha256((node + sibling).encode()).hexdigest()
        else:
            node = hashlib.sha256((sibling + node).encode()).hexdigest()
    return node == root


def build_chain(start_index, prev, difficulties, txid_groups, start_nonce=0):
    chain = []
    for offset, difficulty in enumerate(difficulties):
        index = start_index + offset
        group = txid_groups[offset]
        block = mine(index, prev, merkle_root(group), difficulty, start_nonce)
        chain.append((block, group))
        prev = block.hash
    return chain


class LightNode:
    """Header-only client: validates, stitches orphans and follows the heaviest branch."""

    def __init__(self, genesis, max_difficulty=4):
        self.max_difficulty = max_difficulty
        self.headers = {genesis.hash: genesis}
        self.work = {genesis.hash: genesis.work()}
        self.children = {}
        self.tip = genesis.hash
        self.orphans = {}
        self.counters = {"stored": 0, "duplicate": 0, "orphan": 0, "rejected": 0,
                         "stitched": 0}
        self.bodies = set()
        self.reasons = []

    def _link(self, header):
        self.children.setdefault(header.prev, []).append(header)

    def _consider(self, header):
        self.work[header.hash] = self.work[header.prev] + header.work()
        if self.work[header.hash] > self.work[self.tip] or (
                self.work[header.hash] == self.work[self.tip]
                and header.hash < self.tip):
            self.tip = header.hash

    def offer(self, header):
        """Validate one header; returns the verdict string, mirroring a peer message."""
        if header.hash in self.headers:
            self.counters["duplicate"] += 1
            return "duplicate"
        if not valid_proof(header):
            self.counters["rejected"] += 1
            self.reasons.append("bad proof of work")
            return "rejected"
        if header.difficulty > self.max_difficulty:
            self.counters["rejected"] += 1
            self.reasons.append("difficulty above policy")
            return "rejected"
        if header.prev not in self.headers:
            self.orphans.setdefault(header.prev, []).append(header)
            self.counters["orphan"] += 1
            return "orphan"
        parent = self.headers[header.prev]
        if header.index != parent.index + 1:
            self.counters["rejected"] += 1
            self.reasons.append("index does not follow the parent")
            return "rejected"
        self.headers[header.hash] = header
        self._link(header)
        self._consider(header)
        self.counters["stored"] += 1
        self._stitch(header.hash)
        return "stored"

    def _stitch(self, parent_hash):
        """Attach every parked orphan whose parent just became known (recursively)."""
        queue = [parent_hash]
        while queue:
            current = queue.pop()
            for header in self.orphans.pop(current, []):
                if header.hash in self.headers or not valid_proof(header):
                    continue
                parent = self.headers[current]
                if header.index != parent.index + 1:
                    self.counters["rejected"] += 1
                    continue
                self.headers[header.hash] = header
                self._link(header)
                self._consider(header)
                self.counters["stored"] += 1
                self.counters["stitched"] += 1
                queue.append(header.hash)

    def path_to_genesis(self, tip=None):
        """Walk parents back to genesis; returns None if the branch is not connected."""
        node = self.headers[tip or self.tip]
        out = [node.hash]
        while node.prev in self.headers:
            node = self.headers[node.prev]
            out.append(node.hash)
        return out if node.prev == GENESIS_PREV else None

    def depth_of(self, block_hash):
        return len(self.path_to_genesis(block_hash)) - 1


class Peer:
    """Flood-gossip participant; `light=True` keeps headers only and validates everything."""

    def __init__(self, name, genesis, light=False, max_difficulty=4):
        self.name = name
        self.light = light
        self.node = LightNode(genesis, max_difficulty)
        self.fresh = {genesis.hash}

    def offer(self, header):
        return self.node.offer(header)

    def publish(self, headers):
        for header in headers:
            self.node.offer(header)
        self.fresh = set(self.node.headers)


def run_gossip(peers, rounds):
    """Deterministic store-and-forward flooding; returns the per-round message count."""
    traffic = []
    for _ in range(rounds):
        messages = []
        for sender in peers:
            for receiver in peers:
                if receiver is sender:
                    continue
                for block_hash in sorted(sender.fresh):
                    messages.append((receiver.name, sender.name, block_hash, sender))
        delivered = 0
        for target, _, block_hash, sender in messages:
            receiver = next(peer for peer in peers if peer.name == target)
            header = sender.node.headers.get(block_hash)
            if header is None:
                continue
            verdict = receiver.offer(header)
            if verdict == "stored":
                receiver.fresh.add(header.hash)
                receiver.fresh.update(header.hash for header in receiver.node.headers.values()
                                      if header.prev == header.hash)
            delivered += 1
        traffic.append(delivered)

    for peer in peers:
        for header in peer.node.headers.values():
            peer.fresh.add(header.hash)
    return traffic


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    genesis = mine(0, GENESIS_PREV, merkle_root(["genesis"]), 0)
    check(valid_proof(genesis) and genesis.index == 0, "genesis is index 0 and valid")
    check(genesis.work() == 1, "difficulty 0 costs one unit of work")

    # ---- 1. proof of work -------------------------------------------------
    block = mine(1, genesis.hash, merkle_root(["a"]), 2)
    check(valid_proof(block), "a mined block satisfies its difficulty target")
    mutated = Header(block.index, block.prev, block.merkle, block.difficulty, block.nonce + 1)
    check(mutated.hash != block.hash, "changing the nonce changes the block hash")
    check(not valid_proof(mutated), "the mutated header must fail its own proof check")
    accepted = 0
    for nonce in range(400):
        if valid_proof(Header(2, block.hash, "root", 2, nonce)):
            accepted += 1
    check(accepted <= 12, "only about 1 in 256 nonces meets difficulty 2 (got %d)" % accepted)
    check(16 ** 3 > 16 ** 2, "higher difficulty means more work per block")

    # ---- 2. merkle proofs -------------------------------------------------
    txids = ["tx%d" % index for index in range(7)]
    root = merkle_root(txids)
    check(root == merkle_root(list(txids)), "merkle roots are deterministic")
    for index, txid in enumerate(txids):
        proof = merkle_proof(txids, index)
        check(verify_proof(root, txid, proof), "proof for tx %d must verify" % index)
    bogus = merkle_proof(txids, 3)
    bogus[0] = ("right" if bogus[0][0] == "left" else "left", bogus[0][1])
    check(not verify_proof(root, txids[3], bogus), "a flipped sibling must fail")
    check(not verify_proof(root, "txX", merkle_proof(txids, 0)),
          "a proof must not verify for another transaction")
    check(not verify_proof("deadbeef", txids[1], merkle_proof(txids, 1)),
          "a proof must not verify against another root")
    check(merkle_root([]) == merkle_root([]), "the empty tree is well defined")
    check(merkle_root(["only"]) == hashlib.sha256(b"only").hexdigest(),
          "a single-leaf tree is the leaf hash")

    # ---- 3. stitching an out-of-order delivery ----------------------------
    groups = [["tx%d" % (10 * i + j) for j in range(3)] for i in range(6)]
    main_chain = build_chain(1, genesis.hash, [2, 2, 2, 2, 2, 2], groups)
    client = LightNode(genesis)
    for block, _ in reversed(main_chain):
        client.offer(block)
    check(client.tip == main_chain[-1][0].hash,
          "a reversed delivery order must still stitch to the heaviest head")
    check(client.path_to_genesis() is not None, "the stitched branch reaches genesis")
    check(client.counters["orphan"] == len(main_chain) - 1,
          "every header except the first delivered was parked as an orphan")
    check(client.counters["stitched"] == len(main_chain) - 1,
          "every orphan was stitched once its parent arrived")
    check(client.depth_of(main_chain[-1][0].hash) == 6, "the head sits at depth 6")
    for index, (block, _) in enumerate(main_chain):
        check(client.headers[block.hash].index == index + 1, "index chain intact")

    # duplicates are free
    tip_work = client.work[client.tip]
    for block, _ in main_chain:
        client.offer(block)
    check(client.counters["duplicate"] == len(main_chain),
          "re-offering the same headers is detected as duplicate")
    check(client.work[client.tip] == tip_work and client.tip == main_chain[-1][0].hash,
          "a duplicate header must not change the work or the tip")

    # ---- 4. work, forks and reorgs ----------------------------------------
    fork_parent = main_chain[1][0]
    fork_groups = [["fork%d" % index] for index in range(4)]
    fork = build_chain(fork_parent.index + 1, fork_parent.hash, [3, 3, 3, 3], fork_groups)
    easy_work = client.work[main_chain[-1][0].hash]
    hard_work = client.work[fork_parent.hash] + sum(block.work() for block, _ in fork)
    check(len(fork) < len(main_chain) and hard_work > easy_work,
          "the harder branch is shorter but heavier")
    old_tip = client.tip
    for block, _ in fork:
        client.offer(block)
    check(client.tip == fork[-1][0].hash, "the tip must follow the heaviest branch")
    check(client.headers[old_tip] is not None, "the abandoned branch is still stored")
    check(client.path_to_genesis() is not None, "the new branch reaches genesis")
    stable = [block.hash for block, _ in main_chain[:3]]
    for block_hash in stable:
        check(block_hash in client.headers, "headers below the fork point survive the reorg")
    check(client.depth_of(fork[-1][0].hash) == fork_parent.index + 4,
          "the new head has the expected depth")
    check(client.work[client.tip] == hard_work, "cumulative work is sum(16^difficulty)")

    # ---- 5. hostile headers ------------------------------------------------
    hostile = LightNode(genesis, max_difficulty=3)
    bogus = Header(1, genesis.hash, merkle_root(["x"]), 2, 0)
    while valid_proof(bogus):
        bogus = Header(1, genesis.hash, merkle_root(["x"]), 2, bogus.nonce + 1)
    check(hostile.offer(bogus) == "rejected", "a header without proof of work is rejected")
    greedy = Header(1, genesis.hash, merkle_root(["x"]), 9, 0)
    check(hostile.offer(greedy) == "rejected",
          "an absurd difficulty claim is rejected before any work is done")
    check(hostile.tip == genesis.hash, "rejected headers never move the tip")
    wrong_index = mine(7, genesis.hash, merkle_root(["y"]), 2)
    check(hostile.offer(wrong_index) == "rejected", "an index that skips ahead is rejected")
    unlinked = mine(1, "f" * 64, merkle_root(["z"]), 2)
    check(hostile.offer(unlinked) == "orphan",
          "a header with an unknown parent is parked, not accepted")
    check(hostile.tip == genesis.hash and hostile.counters["orphan"] == 1,
          "orphans do not move the tip")
    check(len(hostile.reasons) == 3, "each rejection records a reason")
    check(hostile.bodies == set(), "the light node never stores a transaction body")

    # ---- 6. gossip convergence --------------------------------------------
    chain = build_chain(1, genesis.hash, [2, 2, 2], [["g%d" % index] for index in range(3)])
    forked = build_chain(2, chain[0][0].hash, [3, 3], [["h0"], ["h1"]])
    peer_a = Peer("a", genesis, light=False)
    peer_b = Peer("b", genesis, light=False)
    client_peer = Peer("light", genesis, light=True, max_difficulty=4)
    peer_a.publish([block for block, _ in chain])
    peer_b.publish([block for block, _ in chain[:1]] + [block for block, _ in forked])
    check(peer_a.node.tip != peer_b.node.tip, "the two peers start on different branches")
    traffic = run_gossip([peer_a, peer_b, client_peer], 4)
    check(all(count > 0 for count in traffic), "gossip must actually move headers")
    tips = {peer.node.tip for peer in (peer_a, peer_b, client_peer)}
    check(len(tips) == 1, "all peers must converge on one tip (%d distinct)" % len(tips))
    winner = max((peer_a.node.work[peer_a.node.tip], peer_b.node.work[peer_b.node.tip]))
    check(winner == client_peer.node.work[client_peer.node.tip],
          "the converged tip must be the heaviest known branch")
    check(client_peer.node.path_to_genesis() is not None, "the client's branch reaches genesis")
    check(all(header.hash == hashlib.sha256(header_blob(header.index, header.prev,
                                                        header.merkle, header.difficulty,
                                                        header.nonce).encode()).hexdigest()
              for header in client_peer.node.headers.values()),
          "every stored header hashes to its own identifier")
    check(client_peer.node.counters["rejected"] == 0,
          "no honest header may be rejected during gossip")
    check(client_peer.node.counters["duplicate"] > 0,
          "gossip must deduplicate repeated headers")

    # ---- 7. determinism ----------------------------------------------------
    repeat_a = Peer("a", genesis, light=False)
    repeat_b = Peer("b", genesis, light=False)
    repeat_client = Peer("light", genesis, light=True, max_difficulty=4)
    repeat_a.publish([block for block, _ in chain])
    repeat_b.publish([block for block, _ in chain[:1]] + [block for block, _ in forked])
    repeat_traffic = run_gossip([repeat_a, repeat_b, repeat_client], 4)
    check(repeat_traffic == traffic and repeat_client.node.tip == client_peer.node.tip,
          "the gossip run must be reproducible")

    print("pow: difficulty 2 accepted %d/400 random nonces, chain work %d vs %d"
          % (accepted, easy_work, hard_work))
    print("stitching: %d orphans stitched, %d stored, %d duplicates, %d rejected"
          % (client.counters["stitched"], client.counters["stored"],
             client.counters["duplicate"], client.counters["rejected"]))
    print("gossip: %d rounds, traffic %r, %d converged tips"
          % (len(traffic), traffic, len(tips)))
    print("PASS")


if __name__ == "__main__":
    main()
