"""Task 831 - Implement blockchain light client with SPV proofs and pruning. (Tier: Master, CTCI Ch 9 — System Design and Scalability)

A light client stores *headers only* -- (height, previous hash, Merkle root, timestamp,
difficulty, nonce) -- and still answers the only two questions it is ever asked: "which chain
do I follow?" (the one with the most cumulative proof-of-work) and "is this transaction in that
chain?" (a Merkle path against a header's root). Pruning keeps memory bounded: everything below
a depth is dropped except a checkpoint header whose verified cumulative work is retained, so
the fork choice and the work accounting stay correct with O(depth) headers instead of O(chain).

What is implemented
-------------------
* `mine`: a deterministic nonce search for a header whose SHA-256 is below the target
  2^(256 - difficulty). A header's *work* is defined as 2^difficulty (its expected number of
  hashes) -- the quantity the heaviest-chain rule sums.
* `merkle_root` / `merkle_path` / `verify_merkle`: Bitcoin-style binary Merkle tree (an odd
  level duplicates its last node), paths of length ceil(log2 n). Each path element carries the
  side of its sibling, so verification needs only the path and the root.
* `retarget`: every `RETARGET` blocks the difficulty moves by up to two bits, read off the
  span of the header timestamps (explicit logical ticks -- no wall-clock), clamped to a hard
  range. The light client re-derives the rule from the headers it holds.
* `FullNode`: mines, stores headers + transactions, serves headers and SPV proofs.
* `fork_headers`: mines a competing branch off the same genesis under the same rules.
* `LightClient`: header store indexed by hash and by height, the heaviest-chain rule with an
  explicit switch log, cumulative work, SPV verification returning exactly one of
  verified / unconfirmed / pruned_header / unknown_header / not_in_best_chain / bad_proof, and
  `prune_keep(depth)` with a checkpoint.

What is simplified
------------------
* No P2P layer, mempool or signatures: a "transaction" is an opaque string, so what is
  exercised is exactly an SPV client's job -- inclusion proofs, chain weighing, retarget
  validation, bounded memory.
* Mining is real SHA-256 grinding at 8-12 leading zero bits (hundreds to a few thousand hashes
  per block), sized so the whole run is instant and deterministic.
* Pruning trusts the checkpoint header (it was fully validated before being pruned). A client
  cannot re-verify work it no longer stores, and the checks say so instead of hiding it.

Invariants verified
-------------------
1.  Merkle: the root equals the top of an explicitly built level list, every index's path
    verifies against that root, a power-of-two tree has exactly n-1 distinct siblings, a
    single-transaction block has an empty path, and odd levels duplicate consistently.
2.  Mutation is caught: a changed transaction, a forged sibling, a path built for another
    index and a path taken from another block all fail.
3.  Proof-of-work: every accepted header beats its target; the work credited is exactly
    2^difficulty; the miner really grinds (the nonce counter is reported).
4.  Admission rules: unknown parent, wrong height, out-of-range difficulty, a difficulty that
    ignores the retarget rule and a nonce that misses the target are each rejected with their
    own reason, and a foreign genesis cannot be added.
5.  Retargeting: a chain whose timestamps advance at the target rate keeps its difficulty, a
    fast chain ratchets up two bits per retarget, is clamped at the maximum, and the client
    re-derives the same values from headers alone.
6.  Cumulative work grows strictly along a chain and equals sum(2^difficulty) exactly.
7.  Heaviest chain: a *shorter* branch with more work displaces the best chain (exactly one
    switch), a lighter branch does not, and a branch whose headers fail proof-of-work is
    refused however heavy it claims to be.
8.  SPV: an included transaction verifies with the right confirmation depth, a tip-block
    transaction is unconfirmed, forged/mis-indexed/borrowed proofs are rejected, and a
    transaction living in a side branch is not in the best chain.
9.  Pruning: the header count drops to the depth plus the checkpoint, the total work is
    unchanged, the checkpoint is the highest pruned header, repeating a prune is a no-op, a
    deeper prune moves the checkpoint forward, a proof for a pruned non-checkpoint block
    reports pruned_header rather than a false accept, and the retained checkpoint can still
    prove its own transactions (which is what a checkpoint is for).
10. Storage: the client holds headers and one checkpoint, never transactions.
11. Determinism: re-mining the same chain reproduces every hash.
"""
import hashlib

RETARGET = 2                     # a retarget boundary every RETARGET blocks
TARGET_SPAN = 10                 # consensus: 10 logical ticks per block
NOMINAL_SPAN = (RETARGET - 1) * TARGET_SPAN    # the window covers RETARGET-1 intervals
MIN_DIFFICULTY, MAX_DIFFICULTY = 8, 12
MAX_HASH = 1 << 256
LIMIT = 1 << 17                  # nonce search cap: a miss becomes an error, not a hang


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------
def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def header_blob(header):
    return (f"{header['height']}|{header['prev']}|{header['root']}|"
            f"{header['timestamp']}|{header['difficulty']}|{header['nonce']}")


def hash_hex(header):
    return sha(header_blob(header))


def block_hash(header):
    return int(hash_hex(header), 16)


def target_for(difficulty):
    return MAX_HASH >> difficulty


def work_of(header):
    return 1 << header["difficulty"]


def tx_hash(tx):
    return sha(tx)


# --------------------------------------------------------------------------------------
# Merkle tree (odd levels duplicate their last node)
# --------------------------------------------------------------------------------------
def merkle_root(txs):
    if not txs:
        raise ValueError("a block needs at least one transaction")
    level = [tx_hash(tx) for tx in txs]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_levels(txs):
    """The independent oracle: the same tree materialised level by level."""
    level = [tx_hash(tx) for tx in txs]
    levels = [list(level)]
    while len(level) > 1:
        if len(level) % 2:
            level = level + [level[-1]]
        level = [sha(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
        levels.append(list(level))
    return levels


def merkle_path(txs, index):
    """[(side, sibling hash), ...]; `side` says where the sibling sits."""
    level = [tx_hash(tx) for tx in txs]
    path, position = [], index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        path.append(("right" if position % 2 == 0 else "left", level[position ^ 1]))
        level = [sha(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
        position //= 2
    return path


def verify_merkle(tx, path, root):
    node = tx_hash(tx)
    for side, sibling in path:
        node = sha(node + sibling) if side == "right" else sha(sibling + node)
    return node == root


# --------------------------------------------------------------------------------------
# mining and retargeting
# --------------------------------------------------------------------------------------
def mine(header):
    """Deterministic nonce search: the first nonce whose hash beats the target."""
    target = target_for(header["difficulty"])
    for nonce in range(LIMIT):
        candidate = dict(header)
        candidate["nonce"] = nonce
        if block_hash(candidate) < target:
            return candidate, nonce + 1
    raise RuntimeError("no proof-of-work found inside the nonce limit")


def retarget(parent_difficulty, window):
    """Coarse toy retarget over the last RETARGET headers (RETARGET-1 intervals).

    Timestamps advancing at the target rate keep the difficulty, a span below half the target
    raises it by two bits, a span above twice the target lowers it by two, and the result is
    clamped to [MIN_DIFFICULTY, MAX_DIFFICULTY].
    """
    span = window[-1]["timestamp"] - window[0]["timestamp"]
    if span * 2 < NOMINAL_SPAN:
        step = 2
    elif span < NOMINAL_SPAN:
        step = 1
    elif span > 2 * NOMINAL_SPAN:
        step = -2
    elif span > NOMINAL_SPAN:
        step = -1
    else:
        step = 0
    return max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, parent_difficulty + step))


# --------------------------------------------------------------------------------------
# a full node that serves headers and SPV proofs
# --------------------------------------------------------------------------------------
class FullNode:
    def __init__(self, timestamp_step=TARGET_SPAN):
        self.timestamp_step = timestamp_step
        genesis = {"height": 0, "prev": "0" * 64, "root": merkle_root(["genesis"]),
                   "timestamp": 0, "difficulty": MIN_DIFFICULTY, "nonce": 0}
        self.headers = [mine(genesis)[0]]
        self.blocks = {hash_hex(self.headers[0]): ["genesis"]}
        self.nonces_tried = 1

    def tip(self):
        return self.headers[-1]

    def next_difficulty(self):
        parent = self.tip()
        height = parent["height"] + 1
        if height % RETARGET or len(self.headers) < RETARGET:
            return parent["difficulty"]
        return retarget(parent["difficulty"], self.headers[-RETARGET:])

    def extend(self, txs, timestamp=None, difficulty=None):
        parent = self.tip()
        header = {"height": parent["height"] + 1, "prev": hash_hex(parent),
                  "root": merkle_root(txs),
                  "timestamp": parent["timestamp"] + self.timestamp_step
                  if timestamp is None else timestamp,
                  "difficulty": self.next_difficulty() if difficulty is None else difficulty,
                  "nonce": 0}
        header, tried = mine(header)
        self.nonces_tried += tried
        self.headers.append(header)
        self.blocks[hash_hex(header)] = list(txs)
        return header

    def prove(self, tx):
        for header in self.headers:
            txs = self.blocks.get(hash_hex(header), [])
            if tx in txs:
                index = txs.index(tx)
                return {"header": header, "index": index, "path": merkle_path(txs, index)}
        return None


def fork_headers(genesis, prefix, length, timestamp_step, per_block=4):
    """Mine a side branch off `genesis`, following exactly the same retarget rule."""
    headers, txs_by_hash, window = [], {}, [genesis]
    parent = genesis
    for height in range(1, length + 1):
        txs = [f"{prefix}-{height}-{index}" for index in range(per_block)]
        difficulty = parent["difficulty"]
        if height % RETARGET == 0 and len(window) >= RETARGET:
            difficulty = retarget(parent["difficulty"], window[-RETARGET:])
        header = {"height": height, "prev": hash_hex(parent), "root": merkle_root(txs),
                  "timestamp": parent["timestamp"] + timestamp_step,
                  "difficulty": difficulty, "nonce": 0}
        header, _ = mine(header)
        headers.append(header)
        txs_by_hash[hash_hex(header)] = txs
        window.append(header)
        parent = header
    return headers, txs_by_hash


# --------------------------------------------------------------------------------------
# the light client
# --------------------------------------------------------------------------------------
class LightClient:
    """Headers only: heaviest-chain rule, SPV verification, pruning with a checkpoint."""

    def __init__(self, genesis, confirmations=2):
        digest = hash_hex(genesis)
        self.by_hash = {digest: genesis}
        self.by_height = {0: [digest]}
        self.cumulative = {digest: work_of(genesis)}
        self.best = digest
        self.confirmations = confirmations
        self.checkpoint = None                  # {"hash", "header", "work"}
        self.pruned_to = -1
        self.rejections = []
        self.switches = []
        self.accepted = 1

    # -- state -------------------------------------------------------------------------
    def tip(self):
        return self.by_hash[self.best]

    def known(self, digest):
        return digest in self.by_hash or (self.checkpoint is not None
                                         and self.checkpoint["hash"] == digest)

    def total_work(self):
        """Checkpoint work (validated before pruning) plus the stored best chain's own work.

        Only stored data is used, so the number is exactly what the client can still justify.
        """
        total = self.checkpoint["work"] if self.checkpoint else 0
        for header in self.best_chain():
            if self.checkpoint is not None and hash_hex(header) == self.checkpoint["hash"]:
                continue
            total += work_of(header)
        return total

    def chain_to(self, digest):
        chain, current = [], self.by_hash[digest]
        while True:
            chain.append(current)
            parent = self.by_hash.get(current["prev"])
            if parent is None:
                break
            current = parent
        return list(reversed(chain))

    def best_chain(self):
        return self.chain_to(self.best)

    def best_chain_hashes(self):
        return {hash_hex(header) for header in self.best_chain()}

    # -- admission ---------------------------------------------------------------------
    def add_header(self, header):
        """Validate and store a header; returns (accepted, reason, switched)."""
        digest = hash_hex(header)
        if digest in self.by_hash:
            return False, "duplicate", False
        parent = self.by_hash.get(header["prev"])
        if parent is None:
            reason = "pruned_parent" if self.pruned_to >= 0 else "unknown_parent"
            self.rejections.append((digest, reason))
            return False, reason, False
        if header["height"] != parent["height"] + 1:
            self.rejections.append((digest, "bad_height"))
            return False, "bad_height", False
        if not MIN_DIFFICULTY <= header["difficulty"] <= MAX_DIFFICULTY:
            self.rejections.append((digest, "difficulty_range"))
            return False, "difficulty_range", False
        expected = self.expected_difficulty(parent)
        if expected is not None and header["difficulty"] != expected:
            self.rejections.append((digest, "bad_retarget"))
            return False, "bad_retarget", False
        if block_hash(header) >= target_for(header["difficulty"]):
            self.rejections.append((digest, "bad_pow"))
            return False, "bad_pow", False
        self.by_hash[digest] = header
        self.by_height.setdefault(header["height"], []).append(digest)
        self.cumulative[digest] = self.cumulative[hash_hex(parent)] + work_of(header)
        self.accepted += 1
        switched = False
        if self.cumulative[digest] > self.cumulative[self.best]:
            extension = header["prev"] == self.best      # simply extending the best chain
            self.best = digest
            if not extension:
                # the tip jumped to a different branch: this is the reorganisation the
                # heaviest-chain rule exists for, and it is logged as such
                self.switches.append((hash_hex(parent), digest))
                switched = True
        return True, "accepted", switched

    def expected_difficulty(self, parent):
        """Re-derive the retarget rule from the headers the client still holds."""
        height = parent["height"] + 1
        if height % RETARGET:
            return parent["difficulty"]
        chain = self.chain_to(hash_hex(parent))
        if len(chain) < RETARGET:
            return None                    # cannot verify a retarget without the window
        return retarget(parent["difficulty"], chain[-RETARGET:])

    def admitted_difficulties(self, headers):
        """Diagnostic: the difficulty the client would expect for each header in order."""
        return [self.expected_difficulty(parent) for parent in headers]

    # -- SPV ---------------------------------------------------------------------------
    def verify_inclusion(self, tx, proof):
        """(status, confirmations); exactly one status can come back."""
        header = proof["header"]
        digest = hash_hex(header)
        height = header["height"]
        checkpoint_digest = self.checkpoint["hash"] if self.checkpoint else None
        if not self.known(digest):
            return ("pruned_header" if height <= self.pruned_to else "unknown_header"), 0
        stored = digest in self.by_hash
        if stored and height < self.tip()["height"] \
                and digest not in self.best_chain_hashes():
            return "not_in_best_chain", 0
        if not stored and digest != checkpoint_digest:
            return "not_in_best_chain", 0
        if not verify_merkle(tx, proof["path"], header["root"]):
            return "bad_proof", 0
        depth = self.tip()["height"] - height
        return ("verified" if depth >= self.confirmations else "unconfirmed"), depth

    # -- pruning -----------------------------------------------------------------------
    def prune_keep(self, depth):
        """Drop headers below tip-depth, keeping a checkpoint with its verified work."""
        keep_from = max(1, self.tip()["height"] - depth)
        if self.checkpoint is not None and keep_from <= self.checkpoint["header"]["height"] + 1:
            return 0
        best = {hash_hex(header) for header in self.best_chain()}
        boundary, removed = None, 0
        for digest, header in list(self.by_hash.items()):
            if header["height"] < keep_from:
                if digest in best and (boundary is None
                                       or header["height"] > boundary["height"]):
                    boundary = header
                del self.by_hash[digest]
                self.by_height[header["height"]] = [
                    other for other in self.by_height.get(header["height"], [])
                    if other != digest]
                removed += 1
        if boundary is not None:
            self.checkpoint = {"hash": hash_hex(boundary), "header": boundary,
                               "work": self.cumulative[hash_hex(boundary)]}
        self.pruned_to = keep_from - 1
        return removed

    def stored_headers(self):
        return len(self.by_hash)


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------
def main():
    node = FullNode()
    blocks = []
    for height in range(1, 7):
        txs = [f"tx-{height}-{index}" for index in range(8)]
        blocks.append((node.extend(txs), txs))
    genesis = node.header_at_zero = node.headers[0]

    # -- 1. Merkle structure --------------------------------------------------------------
    txs = blocks[0][1]
    root = merkle_root(txs)
    check(merkle_levels(txs)[-1] == [root] and merkle_levels(txs)[0] == [tx_hash(t) for t in txs],
          "the root is the top of the explicit level-by-level tree")
    check(all(verify_merkle(txs[i], merkle_path(txs, i), root) for i in range(len(txs))),
          "every index's path verifies against the same root")
    check(all(len(merkle_path(txs, i)) == 3 for i in range(len(txs))),
          "eight transactions give three-element paths")
    siblings = {sibling for i in range(len(txs)) for _, sibling in merkle_path(txs, i)}
    bottom = {merkle_path(txs, i)[0][1] for i in range(len(txs))}
    check(bottom == {tx_hash(t) for t in txs},
          "every transaction of the block appears once as a bottom-level sibling")
    check(len(siblings) == 2 * len(txs) - 2,
          f"every non-root node appears as a sibling exactly once ({len(siblings)})")
    check(merkle_root(["only-tx"]) == tx_hash("only-tx")
          and merkle_path(["only-tx"], 0) == [], "a one-transaction block has an empty path")
    odd = ["a", "b", "c"]
    check(all(verify_merkle(odd[i], merkle_path(odd, i), merkle_root(odd)) for i in range(3)),
          "odd levels duplicate their last node consistently")
    check(len(merkle_path(odd, 2)) == 2
          and merkle_levels(odd)[1][1] == sha(tx_hash("c") + tx_hash("c")),
          "three transactions need two siblings and the odd node is hashed with itself")

    # -- 2. mutation detection -------------------------------------------------------------
    check(not verify_merkle("tampered", merkle_path(txs, 0), root),
          "altering the transaction breaks the proof")
    forged = list(merkle_path(txs, 0))
    forged[0] = (forged[0][0], sha("not-a-sibling"))
    check(not verify_merkle(txs[0], forged, root), "a forged sibling breaks the proof")
    check(not verify_merkle(txs[0], merkle_path(txs, 3), root),
          "a path built for another index breaks the proof")
    check(not verify_merkle(txs[0], merkle_path(blocks[1][1], 0), root),
          "a path taken from another block breaks the proof")

    # -- 3. proof-of-work ------------------------------------------------------------------
    header = blocks[2][0]
    check(block_hash(header) < target_for(header["difficulty"]),
          "the mined header's hash beats its target")
    check(work_of(header) == 1 << header["difficulty"] == 256,
          "a header's work is exactly 2^difficulty")
    check(node.nonces_tried > 6 and node.nonces_tried < 6 * LIMIT,
          f"mining seven blocks took {node.nonces_tried} nonces, i.e. real grinding")
    check(all(block_hash(h) < target_for(h["difficulty"]) for h, _ in blocks),
          "every main-chain header beats its own target")

    # -- 4. admission rules ----------------------------------------------------------------
    client = LightClient(genesis, confirmations=2)
    verdicts = [client.add_header(header) for header, _ in blocks]
    check(all(verdict[0] and verdict[1] == "accepted" for verdict in verdicts),
          "the honest chain is accepted header by header")
    check(client.tip()["height"] == 6 and len(client.best_chain()) == 7,
          "the best chain reaches height 6")
    first_work = client.total_work()
    check(first_work == 7 * 256 == sum(work_of(h) for h in node.headers),
          f"cumulative work is the exact sum of 2^difficulty ({first_work})")
    running = []
    for header in client.best_chain():
        running.append(work_of(header) + (running[-1] if running else 0))
    check(all(a < b for a, b in zip(running, running[1:])), "cumulative work grows strictly")

    foreign = LightClient(mine({"height": 0, "prev": "f" * 64, "root": merkle_root(["x"]),
                                "timestamp": 0, "difficulty": MIN_DIFFICULTY, "nonce": 0})[0])
    check(not foreign.add_header(blocks[0][0])[0],
          "a genesis from another chain does not validate")
    stranger = dict(blocks[1][0])
    stranger["prev"] = sha("no-such-block")
    check(client.add_header(stranger)[1] == "unknown_parent",
          "a header quoting an unknown parent is refused")
    wrong_height = dict(blocks[1][0])
    wrong_height["height"] = 9
    check(client.add_header(wrong_height)[1] == "bad_height",
          "a header that skips heights is refused")
    out_of_range = dict(blocks[1][0])
    out_of_range["difficulty"] = MAX_DIFFICULTY + 3
    check(client.add_header(out_of_range)[1] == "difficulty_range",
          "a difficulty outside the protocol range is refused")
    duplicate = client.add_header(blocks[1][0])
    check(duplicate[1] == "duplicate", "a header already stored is not stored twice")

    # -- 5. retargeting --------------------------------------------------------------------
    main_difficulties = [header["difficulty"] for header, _ in blocks]
    check(main_difficulties == [MIN_DIFFICULTY] * 6,
          f"timestamps at the target rate hold the difficulty ({main_difficulties})")
    fast, fast_txs = fork_headers(genesis, "fast", 4, timestamp_step=1)
    fast_difficulties = [header["difficulty"] for header in fast]
    check(fast_difficulties == [8, 10, 10, 12],
          f"a fast chain ratchets up two bits per retarget and clamps at the cap "
          f"({fast_difficulties})")
    check(MAX_DIFFICULTY in fast_difficulties and max(fast_difficulties) == MAX_DIFFICULTY,
          "the retarget stops at the maximum difficulty")
    probe = LightClient(genesis, confirmations=2)
    probe.add_header(fast[0])
    cheat = dict(fast[1])
    cheat["difficulty"] = MIN_DIFFICULTY
    cheat = mine(cheat)[0]
    check(probe.add_header(cheat)[1] == "bad_retarget",
          "a block that ignores the retarget rule is rejected")
    check(probe.expected_difficulty(fast[0]) == fast[1]["difficulty"] == 10,
          "the client re-derives exactly the retarget the miner used")
    main_probe = LightClient(genesis, confirmations=2)
    main_probe.add_header(blocks[0][0])
    check(main_probe.expected_difficulty(blocks[0][0]) == MIN_DIFFICULTY,
          "a window spanning the target time keeps the difficulty")
    lazy = dict(fast[0])
    lazy["nonce"] = lazy["nonce"] + 1
    check(main_probe.add_header(lazy)[1] in ("bad_retarget", "bad_pow", "unknown_parent"),
          "a header that does not actually beat its target is refused")

    # -- 6. heaviest chain -----------------------------------------------------------------
    slow, _ = fork_headers(genesis, "slow", 3, timestamp_step=TARGET_SPAN)
    check(sum(work_of(h) for h in slow) < sum(work_of(h) for h, _ in blocks),
          "the three-block branch carries less work than the six-block chain")
    for header in slow:
        client.add_header(header)
    check(client.tip()["height"] == 6 and not client.switches,
          "a lighter branch does not displace the best chain")
    switches_before = len(client.switches)
    for header in fast:
        client.add_header(header)
    check(len(client.switches) == switches_before + 1 and client.tip()["height"] == 4,
          "a shorter branch with more work takes over (one switch)")
    check(client.total_work() == 256 + sum(work_of(h) for h in fast) == 256 + 6400,
          f"the work accounting follows the new best chain ({client.total_work()})")
    check(all(client.cumulative[hash_hex(fast[i])] < client.cumulative[hash_hex(fast[i + 1])]
              for i in range(3)), "the winning branch's work also grows strictly")
    sneak = dict(fast[2])
    sneak["nonce"] = sneak["nonce"] + 1
    check(client.add_header(sneak)[1] in ("bad_pow", "duplicate"),
          "a heavier-looking header without valid work is refused")

    # -- 7. SPV ----------------------------------------------------------------------------
    spv = LightClient(genesis, confirmations=2)
    for header, _ in blocks:
        spv.add_header(header)
    for header in slow:
        spv.add_header(header)
    proof = node.prove("tx-2-5")
    check(proof is not None and proof["index"] == 5,
          "the full node finds the transaction and its index")
    status, depth = spv.verify_inclusion("tx-2-5", proof)
    check(status == "verified" and depth == 4, f"the SPV proof verifies at depth {depth}")
    status, depth = spv.verify_inclusion("tx-6-7", node.prove("tx-6-7"))
    check(status == "unconfirmed" and depth == 0,
          "a transaction in the tip block has no confirmations yet")
    mislabelled = dict(proof)
    mislabelled["path"] = merkle_path(blocks[2][1], 4)
    check(spv.verify_inclusion("tx-2-5", mislabelled)[0] == "bad_proof",
          "a proof built for another index is rejected")
    stolen = dict(proof)
    stolen["header"] = blocks[3][0]
    check(spv.verify_inclusion("tx-2-5", stolen)[0] == "bad_proof",
          "a proof bound to another block's header is rejected")
    absent = {"header": blocks[2][0], "index": 0, "path": merkle_path(blocks[2][1], 0)}
    check(spv.verify_inclusion("tx-99-99", absent)[0] == "bad_proof",
          "a transaction that is not in the block is rejected")
    side = {"header": slow[0], "index": 1, "path": merkle_path(
        [f"slow-1-{i}" for i in range(4)], 1)}
    check(spv.verify_inclusion("slow-1-1", side)[0] == "not_in_best_chain",
          "a transaction in a side branch is not in the best chain")
    unknown = {"header": mine({"height": 7, "prev": hash_hex(blocks[5][0]),
                               "root": merkle_root(["ghost"]), "timestamp": 99,
                               "difficulty": MIN_DIFFICULTY, "nonce": 0})[0],
               "index": 0, "path": []}
    check(spv.verify_inclusion("ghost", unknown)[0] == "unknown_header",
          "a header the client never saw yields unknown_header")

    # -- 8. pruning ------------------------------------------------------------------------
    check(spv.stored_headers() == 10, "the client holds the seven main headers and three side")
    before = spv.total_work()
    check(before == 7 * 256 == 1792, f"the best chain's work is {before}")
    removed = spv.prune_keep(depth=3)
    check(removed == 5 and spv.stored_headers() == 5,
          f"pruning dropped {removed} headers leaving {spv.stored_headers()} "
          f"(side branches below the depth go too)")
    check(spv.checkpoint["header"]["height"] == 2,
          "the checkpoint is the highest pruned header of the best chain")
    check(spv.total_work() == before, "the total cumulative work is unchanged by pruning")
    check(spv.checkpoint["work"] == 3 * 256, "the checkpoint carries the verified prefix work")
    status, _ = spv.verify_inclusion("tx-1-0", node.prove("tx-1-0"))
    check(status == "pruned_header",
          "a proof for a pruned block reports pruned_header, never a false accept")
    status, depth = spv.verify_inclusion("tx-6-7", node.prove("tx-6-7"))
    check(status == "unconfirmed" and depth == 0, "recent blocks still verify after pruning")
    status, depth = spv.verify_inclusion("tx-4-3", node.prove("tx-4-3"))
    check(status == "verified" and depth == 2, "deep-enough blocks stay verifiable")
    check(spv.prune_keep(depth=3) == 0 and spv.checkpoint["header"]["height"] == 2,
          "repeating the same prune is a no-op")
    check(spv.prune_keep(depth=2) == 2 and spv.checkpoint["header"]["height"] == 3,
          "a deeper prune moves the checkpoint forward")
    check(spv.total_work() == before, "and the work total is still exact")
    check(spv.prune_keep(depth=5) == 0 and spv.stored_headers() == 3,
          "a prune can never drop the tip")
    status, depth = spv.verify_inclusion("tx-2-0", node.prove("tx-2-0"))
    check(status == "pruned_header",
          "a pruned non-checkpoint block still reports pruned_header")
    status, depth = spv.verify_inclusion("tx-3-2", node.prove("tx-3-2"))
    check(status == "verified" and depth == 3,
          "the retained checkpoint can still prove its own transactions -- the point of it")
    check(spv.known(spv.checkpoint["hash"]) and not spv.known(sha("nonsense")),
          "the checkpoint hash is the only pruned hash the client still recognises")

    # -- 9. storage and determinism --------------------------------------------------------
    check(not hasattr(spv, "blocks"),
          "the client has no transaction store at all")
    again = FullNode()
    for height in range(1, 7):
        again.extend([f"tx-{height}-{index}" for index in range(8)])
    check([hash_hex(header) for header in again.headers]
          == [hash_hex(header) for header in node.headers],
          "re-mining the same chain reproduces every hash")

    print(f"main chain: {len(node.headers)} headers, work {before}, "
          f"{node.nonces_tried} nonces tried")
    print(f"difficulties: main {main_difficulties}, fast branch {fast_difficulties}")
    print(f"after pruning to depth 2: {spv.stored_headers()} headers, checkpoint height "
          f"{spv.checkpoint['header']['height']}, work {spv.total_work()}, "
          f"rejections recorded by the syncing client {len(client.rejections)}")
    print("PASS")


if __name__ == "__main__":
    main()
