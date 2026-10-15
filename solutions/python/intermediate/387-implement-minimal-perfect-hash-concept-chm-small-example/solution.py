"""Task 387 - Implement minimal perfect hash concept (CHM) small example. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

Hash-displace construction in the spirit of the CHM scheme, simplified (no ranked bit array).
Level 1 splits the n keys into buckets by h1(k) mod n. Level 2 assigns each bucket a
displacement shift, chosen as the smallest value in [0, n) for which
sigma(k) = (h2(k) + shift) mod n are distinct and still unused, then fills the bucket's
consecutive final slots in increasing sigma order through the g table:
f(k) = g[sigma(k)]. Buckets are processed by decreasing size, then increasing bucket index
(documented deterministic tie-break); if some bucket cannot be placed, the whole build is
retried with h2 base shifted by one (documented). Since the bucket sizes sum to n, a
successful build is minimal: f is a bijection from the key set onto [0, n).
Non-member keys are out of scope -- f still returns a slot in [0, n) but with no guarantee.
"""

HASH_MOD = 2147483647
BASE1 = 131
BASE2 = 1000003

KEYS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan",
        "judy", "karl", "laura"]
MORE_KEYS = ["kiwi", "mango", "melon", "olive", "peach", "plum", "quince", "raspberry"]


def poly_hash(text, base):
    value = 0
    for ch in text:
        value = (value * base + ord(ch)) % HASH_MOD
    return value


def build_mphf(keys):
    """Return (table, g, displacement, base2).

    table[slot] = key, g[sigma] = slot, displacement[bucket] = shift (None for empty buckets).
    """
    n = len(keys)
    for attempt in range(100):
        base2 = BASE2 + attempt
        buckets = [[] for _ in range(n)]
        for key in keys:
            buckets[poly_hash(key, BASE1) % n].append(key)
        order = sorted(range(n), key=lambda b: (-len(buckets[b]), b))
        table = [None] * n
        g = [None] * n
        displacement = [None] * n
        next_slot = 0
        feasible = True
        for bucket in order:
            members = buckets[bucket]
            if not members:
                continue
            chosen = None
            for shift in range(n):
                sigmas = sorted((poly_hash(key, base2) + shift) % n for key in members)
                if len(set(sigmas)) != len(sigmas):
                    continue
                if any(g[sigma] is not None for sigma in sigmas):
                    continue
                chosen = shift
                break
            if chosen is None:
                feasible = False
                break
            displacement[bucket] = chosen
            pairs = sorted(((poly_hash(key, base2) + chosen) % n, key) for key in members)
            for rank, (sigma, key) in enumerate(pairs):
                g[sigma] = next_slot + rank
                table[next_slot + rank] = key
            next_slot += len(members)
        if feasible:
            return table, g, displacement, base2
    raise RuntimeError("no displacement assignment found")


def mphf_lookup(table, g, displacement, base2, key):
    """Slot of a member key in [0, n); -1 for keys whose first-level bucket stayed empty.
    Non-member keys are out of scope: any non-negative answer is meaningless."""
    n = len(table)
    bucket = poly_hash(key, BASE1) % n
    shift = displacement[bucket]
    if shift is None:
        return -1
    sigma = (poly_hash(key, base2) + shift) % n
    if g[sigma] is None:
        return -1
    return g[sigma]


def main():
    n = len(KEYS)
    assert n == 12 and len(set(KEYS)) == n

    # deterministic hash sanity (values pin the polynomial hash)
    assert poly_hash("alice", BASE1) == 893813729
    assert poly_hash("alice", BASE2) == 76280623
    bucket_of = lambda k: poly_hash(k, BASE1) % n
    assert bucket_of("alice") == 5
    assert bucket_of("bob") == 1 and bucket_of("laura") == 1
    assert bucket_of("frank") == 8 and bucket_of("judy") == 8 and bucket_of("karl") == 8
    assert max(len([k for k in KEYS if bucket_of(k) == b]) for b in range(n)) == 3

    table, g, displacement, base2 = build_mphf(KEYS)
    assert base2 == BASE2, "first attempt already succeeds for this key set"
    assert table == ["frank", "judy", "karl", "laura", "bob", "ivan", "grace", "heidi",
                     "dave", "alice", "erin", "carol"]
    assert g == [7, 0, 5, 9, 10, 11, 1, 2, 3, 4, 6, 8]
    assert displacement == [None, 1, 7, 0, 5, 8, 5, None, 0, None, 6, None]
    assert all(slot is not None for slot in table) and all(slot is not None for slot in g)

    # minimal perfect hash: every key maps to its own slot in [0, n), no collisions
    slots = {}
    for key in KEYS:
        slot = mphf_lookup(table, g, displacement, base2, key)
        assert 0 <= slot < n, key
        assert table[slot] == key, key
        slots[key] = slot
    assert len(set(slots.values())) == n
    assert sorted(slots.values()) == list(range(n))

    # determinism: rebuilding yields the identical structure
    table2, g2, displacement2, base22 = build_mphf(KEYS)
    assert table2 == table and g2 == g and displacement2 == displacement and base22 == base2

    # non-member keys are out of scope: any in-range answer is meaningless by construction
    stranger = mphf_lookup(table, g, displacement, base2, "stranger")
    assert stranger == -1 or 0 <= stranger < n

    # a second key set needs one hash retry (documented) and still builds a bijection
    table3, g3, displacement3, base23 = build_mphf(MORE_KEYS)
    assert base23 == BASE2 + 2, "attempts 0 and 1 fail for this set, attempt 2 succeeds"
    assert table3 == ["plum", "kiwi", "peach", "melon", "olive", "mango", "raspberry", "quince"]
    assert displacement3 == [0, 0, 0, None, 1, 4, None, None]
    slots3 = {}
    for key in MORE_KEYS:
        slot = mphf_lookup(table3, g3, displacement3, base23, key)
        assert 0 <= slot < len(MORE_KEYS)
        assert table3[slot] == key, key
        slots3[key] = slot
    assert sorted(slots3.values()) == list(range(len(MORE_KEYS)))


if __name__ == "__main__":
    main()
    print("PASS")
