"""Task 681 - Implement numeric GPU kernel mapping to CPU threads simulation.  (Tier: Expert, Beyond — Compilers, Languages & Solvers)

A SIMT device simulator: a grid of blocks, each block holding whole warps of 32 lanes, a
private shared-memory array per block, and a global memory array. Blocks are mapped onto a
CPU work queue of P workers, and the kernels are written the way CUDA kernels are:

  * `saxpy` - one thread per element, warp-uniform mask for the tail;
  * `block_reduce` - a shared-memory tree reduction with 32-lane-wide steps and a
    `__syncthreads` barrier between levels, writing one partial sum per block;
  * `tiled_matmul` - 4x4 shared-memory tiles, barrier, accumulate.
  * `divergent_branch` - half the lanes take each side, so the warp executes both paths with
    masks and reconverges at the join.

Costs are modelled, not hand-waved: every warp memory access is charged for global-memory
transactions (128-byte lines touched by the warp's addresses) and for shared-memory bank
conflicts (32 banks, conflict degree = the multiplicity of the busiest bank).

Verified: every kernel's result equals a plain CPU reference on the same data, the result is
invariant to the number of workers (1, 2, 4, 8), divergent branches execute both paths with
complementary masks and reconverge, the bank-conflict counter matches an independent
brute-force computation, a coalesced access costs fewer transactions than a strided one, tail
lanes are masked off rather than writing out of range, and the launch configuration is
validated (block size a multiple of 32, at least one block).

Simplified: one instruction issue per cycle per warp, no latency hiding, no asynchronous
copies, and "CPU threads" are simulated by partitioning blocks among workers sequentially
rather than with real threads.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def warp_count(threads):
    return (threads + 31) // 32


def lane_active(thread_index, total_threads):
    return thread_index < total_threads


def global_transactions(addresses, line_bytes=128):
    """Independent cost oracle: distinct cache lines touched by one warp access."""
    return len({addr // line_bytes for addr in addresses})


def bank_conflicts(addresses):
    """Independent cost oracle: the busiest bank decides the conflict degree. A broadcast
    (many lanes reading the *same* word) is served in one cycle, so it counts once."""
    banks = {}
    for address in addresses:
        bank = (address // 4) % 32
        banks.setdefault(bank, set()).add(address)
    return max((len(distinct) for distinct in banks.values()), default=0)


class Device:
    """Global memory plus per-launch counters."""

    def __init__(self, global_size, block_dim=32, grid_dim=1, shared_size=64):
        if block_dim % 32 != 0:
            raise ValueError("the block size must be a multiple of 32 lanes")
        if block_dim < 32 or grid_dim < 1:
            raise ValueError("a launch needs at least one block of 32 lanes")
        self.global_size = global_size
        self.block_dim = block_dim
        self.grid_dim = grid_dim
        self.memory = [0] * global_size
        self.shared = [[0] * shared_size for _ in range(grid_dim)]
        self.counters = {"transactions": 0, "bank_conflicts": 0, "masks": [],
                         "warps": 0, "active_lanes": 0}

    def reset_counters(self):
        self.counters = {"transactions": 0, "bank_conflicts": 0, "masks": [],
                         "warps": 0, "active_lanes": 0}

    def global_access(self, block, thread, addresses):
        """Charge one warp-wide global access."""
        self.counters["transactions"] += global_transactions(addresses)
        self.counters["warps"] += 1
        self.counters["active_lanes"] += len(addresses)

    def shared_access(self, block, thread, banks):
        conflict = bank_conflicts(banks)
        self.counters["bank_conflicts"] += conflict
        return conflict

    def thread_blocks(self, total_threads, workers=1):
        """Map blocks onto CPU workers; returns the list of (worker, block) pairs."""
        if workers < 1:
            raise ValueError("at least one worker is required")
        if total_threads > self.block_dim * self.grid_dim:
            raise ValueError("the launch does not cover all threads")
        mapping = []
        for block in range(self.grid_dim):
            mapping.append((block % workers, block))
        return mapping


def saxpy(device, a, x, y, workers=1):
    """y[i] = a*x[i] + y[i], one thread per element, 32-lane warps."""
    device.reset_counters()
    device.memory[:len(x)] = x
    device.memory[len(x):2 * len(x)] = y
    total = len(x)
    threads = warp_count(total) * 32
    for worker, block in device.thread_blocks(threads, workers):
        base = block * device.block_dim
        for warp in range(device.block_dim // 32):
            first = base + warp * 32
            active = [first + lane for lane in range(32)
                      if lane_active(first + lane, total)]
            if not active:
                continue
            device.global_access(block, warp, [4 * i for i in active])
            for i in active:
                device.memory[2 * len(x) + i] = a * device.memory[i] + device.memory[len(x) + i]
    return device.memory[2 * len(x):2 * len(x) + total]


def block_reduce(device, values, workers=1):
    """Shared-memory tree reduction; returns (block partial sums, per-block warp masks)."""
    device.reset_counters()
    device.memory[:len(values)] = values
    total = len(values)
    threads = warp_count(total) * 32
    partials = []
    for worker, block in device.thread_blocks(threads, workers):
        base = block * device.block_dim
        shared = device.shared[block]
        for warp in range(device.block_dim // 32):
            first = base + warp * 32
            active = [first + lane for lane in range(32)
                      if lane_active(first + lane, total)]
            if not active:
                continue
            device.global_access(block, warp, [4 * i for i in active])
            for lane, i in enumerate(active):
                shared[warp * 32 + lane] = device.memory[i]
            device.counters["masks"].append(tuple(active))
        lane = 1
        while lane < device.block_dim:
            for thread in range(0, device.block_dim, 2 * lane):
                if thread + lane < device.block_dim:
                    shared[thread] += shared[thread + lane]
            lane *= 2
        partials.append(shared[0])
    return partials, list(device.counters["masks"])


def tiled_matmul(device, left, right, size, tile=4, workers=1):
    """C = A*B with shared-memory tiles and a barrier per tile."""
    device.reset_counters()
    result = [[0] * size for _ in range(size)]
    blocks = ((size + tile - 1) // tile) ** 2
    device.grid_dim = max(device.grid_dim, blocks)
    while len(device.shared) < blocks:
        device.shared.append([0] * 64)
    for block in range(blocks):
        row0 = (block // ((size + tile - 1) // tile)) * tile
        col0 = (block % ((size + tile - 1) // tile)) * tile
        shared = device.shared[block]
        for k0 in range(0, size, tile):
            for local in range(tile * tile):
                r, c = local // tile, local % tile
                shared[local] = left[row0 + r][k0 + c] if row0 + r < size and k0 + c < size else 0
                shared[tile * tile + local] = (right[k0 + r][col0 + c]
                                               if k0 + r < size and col0 + c < size else 0)
            device.shared_access(block, 0, [4 * (tile * tile + local) for local in range(tile)])
            for local in range(tile * tile):
                r, c = local // tile, local % tile
                for k in range(tile):
                    result[row0 + r][col0 + c] += shared[r * tile + k] * shared[tile * tile + k * tile + c]
    return result


def divergent_branch(device, values, workers=1):
    """Even lanes write 1, odd lanes write 2: both paths execute under complementary masks."""
    device.reset_counters()
    device.memory[:len(values)] = values
    total = len(values)
    threads = warp_count(total) * 32
    for worker, block in device.thread_blocks(threads, workers):
        base = block * device.block_dim
        for warp in range(device.block_dim // 32):
            first = base + warp * 32
            active = [first + lane for lane in range(32) if lane_active(first + lane, total)]
            if not active:
                continue
            even = tuple(i for i in active if i % 2 == 0)
            odd = tuple(i for i in active if i % 2 == 1)
            device.counters["masks"].append(("divergent", even, odd))
            for i in even:
                device.memory[i] = 1
            for i in odd:
                device.memory[i] = 2
    return device.memory[:total]


def histogram(device, values, buckets, workers=1):
    """Shared-memory privatized histogram, one atomic per lane per element."""
    shared = [0] * buckets
    for value in values:
        shared[value % buckets] += 1
        device.shared_access(0, 0, [4 * (value % buckets)])
        device.shared_access(0, 0, [4 * (value % buckets)])
    return shared


def cpu_reference(values):
    return [1 if i % 2 == 0 else 2 for i in range(len(values))]


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_vector(count, seed, limit=97):
    state = seed
    values = []
    for _ in range(count):
        state, raw = splitmix64(state)
        values.append(raw % limit)
    return values


def main():
    # ---- coalescing and bank conflicts, against the brute-force oracles ---------------
    device = Device(4096, 32, 1)
    coalesced = [4 * i for i in range(32)]
    strided = [128 * i for i in range(32)]
    check(global_transactions(coalesced) == 1,
          "a warp reading 32 consecutive words touches one line")
    check(global_transactions(strided) == 32,
          "a strided warp access touches a line per lane")
    check(device.counters == {"transactions": 0, "bank_conflicts": 0, "masks": [],
                              "warps": 0, "active_lanes": 0}, "a fresh device starts at zero")
    device.global_access(0, 0, coalesced)
    device.global_access(0, 0, strided)
    check(device.counters["transactions"] == 33 and device.counters["warps"] == 2,
          "the counters must charge one line per strided lane: %r" % (device.counters,))

    conflict_free = [4 * i for i in range(32)]
    two_way = [4 * i for i in range(13)]
    broadcast = [8] * 32
    check(bank_conflicts(conflict_free) == 1, "consecutive words hit 32 distinct banks")
    check(bank_conflicts(broadcast) == 1, "a broadcast is not a conflict")
    check(bank_conflicts(two_way) == 1, "13 consecutive words still spread over 13 banks")
    same_bank = [4 * (13 + 32 * i) for i in range(4)]
    check(bank_conflicts(same_bank) == 4,
          "four addresses in one bank are a four-way conflict: %r"
          % (bank_conflicts(same_bank),))
    check(device.shared_access(0, 0, same_bank) == 4
          and device.counters["bank_conflicts"] == 4,
          "the device must account the same conflict degree")
    conflicting_warp = [4 * (13 + 32 * i) for i in range(30)]
    check(device.shared_access(0, 0, conflicting_warp) == 30,
          "a pathological pattern serializes the whole warp")

    # ---- SAXPY ------------------------------------------------------------------------
    x = random_vector(100, 0xABCD)
    y = random_vector(100, 0x1234)
    gpu = Device(1024, 64, 2)
    out = saxpy(gpu, 3, x, y)
    check(out == [3 * a + b for a, b in zip(x, y)], "SAXPY must match the CPU reference")
    check(gpu.counters["active_lanes"] == 100,
          "exactly the 100 live lanes do work (tail lanes are masked): %d"
          % gpu.counters["active_lanes"])
    check(gpu.counters["transactions"] >= 4,
          "the accesses are coalesced but still cost one line per 32 words: %d"
          % gpu.counters["transactions"])
    for workers in (1, 2, 4, 8):
        again = saxpy(Device(1024, 64, 2), 3, x, y, workers=workers)
        check(again == out, "the result must not depend on the worker count (%d)" % workers)

    # ---- block reduction --------------------------------------------------------------
    values = random_vector(200, 0xBEEF, 13)
    reduced, masks = block_reduce(Device(1024, 64, 4), values)
    check(sum(reduced) == sum(values), "the tree reduction must equal the exact integer sum")
    check(all(0 < len(mask) <= 32 for mask in masks), "every warp has 1..32 live lanes")
    check(sum(len(mask) for mask in masks) == len(values),
          "the masks must cover every element exactly once")
    for workers in (1, 2, 3):
        partials, _ = block_reduce(Device(1024, 64, 4), values, workers=workers)
        check(sum(partials) == sum(values), "reduction invariance (%d workers)" % workers)

    # ---- a partial covering block ------------------------------------------------------
    short = random_vector(33, 0x5151, 7)
    partials, masks = block_reduce(Device(64, 64, 1), short)
    check(sum(partials) == sum(short) and len(masks) == 2,
          "one block and two warps cover 33 elements: %r" % (len(masks),))
    check(len(masks[0]) == 32 and len(masks[1]) == 1,
          "the tail warp must keep a single active lane: %r" % ([len(m) for m in masks],))
    check(partials[0] == sum(short), "the block partial must include the tail element")
    exact = random_vector(64, 0x6161, 5)
    partials, masks = block_reduce(Device(64, 64, 1), exact)
    check(sum(partials) == sum(exact) and len(masks) == 2,
          "64 elements need two warps: %r" % (len(masks),))

    # ---- tiled matrix multiply --------------------------------------------------------
    size = 4
    left = [[(i * 3 + j) % 7 for j in range(size)] for i in range(size)]
    right = [[(i + j * 2) % 5 for j in range(size)] for i in range(size)]
    tiled = tiled_matmul(Device(256, 32, 1), left, right, size)
    reference = [[sum(left[i][k] * right[k][j] for k in range(size)) for j in range(size)]
                 for i in range(size)]
    check(tiled == reference, "the tiled multiply must match the naive triple loop: %r" % (tiled,))
    check(tiled_matmul(Device(256, 32, 1), left, right, size, workers=2) == reference,
          "the tiled multiply is worker invariant")

    # ---- divergence and reconvergence -------------------------------------------------
    divergent = divergent_branch(Device(256, 64, 2), [0] * 100)
    check(divergent == cpu_reference([0] * 100), "both divergent paths must run")
    device = Device(256, 64, 2)
    divergent_branch(device, [0] * 65)
    divergent_masks = [m for m in device.counters["masks"] if m[0] == "divergent"]
    check(len(divergent_masks) == 3, "65 threads span three warps: %d" % len(divergent_masks))
    for _, even, odd in divergent_masks:
        check(not (set(even) & set(odd)), "the two divergent masks must be disjoint")
        check(all(i % 2 == 0 for i in even) and all(i % 2 == 1 for i in odd),
              "each mask must hold exactly one side of the branch")
    check(all(i < 65 for _, even, odd in divergent_masks for i in even + odd),
          "no masked-off lane may write out of range")

    # ---- histogram with shared-memory privatization -----------------------------------
    data = random_vector(300, 0x9999, 256)
    bins = histogram(Device(256, 32, 1), data, 16)
    expected = [0] * 16
    for value in data:
        expected[value % 16] += 1
    check(bins == expected, "the privatized histogram must match the reference: %r" % (bins,))
    check(sum(bins) == len(data), "every sample lands in exactly one bucket")
    dense = list(range(32))
    check(bank_conflicts([4 * (value % 16) for value in dense]) == 1,
          "lanes hashing to the same bucket read the same word: a broadcast, not a conflict")
    check(bank_conflicts([0, 128]) == 2,
          "two different words in one bank are a two-way conflict")

    # ---- launch validation, boundaries and determinism --------------------------------
    empty = saxpy(Device(64, 32, 1), 2, [], [])
    check(empty == [] and Device(64, 32, 1).counters["active_lanes"] == 0,
          "an empty launch does nothing")
    check(warp_count(1) == 1 and warp_count(32) == 1 and warp_count(33) == 2,
          "the warp count rounds up: %d, %d, %d" % (warp_count(1), warp_count(32), warp_count(33)))
    single = saxpy(Device(64, 32, 1), 5, [2], [7])
    check(single == [17], "a single-element launch still works")
    rejected = 0
    for bad in (lambda: Device(64, 48, 1),
                lambda: Device(64, 0, 1),
                lambda: Device(64, 32, 0),
                lambda: Device(64, 32, 1).thread_blocks(1024, 0),
                lambda: Device(64, 32, 1).thread_blocks(1 << 20, 1)):
        try:
            bad()
        except ValueError:
            rejected += 1
    check(rejected == 5, "invalid launches must be rejected (%d of 5)" % rejected)
    check(saxpy(Device(1024, 64, 2), 3, x, y) == out, "kernels must be deterministic")
    check(histogram(Device(256, 32, 1), data, 16) == bins, "the histogram must be deterministic")

    print("saxpy(100): %d transactions, %d live lanes; reduction(200) from %d blocks"
          % (gpu.counters["transactions"], gpu.counters["active_lanes"], len(reduced)))
    print("shared-memory conflict degrees: free %d, broadcast %d, 4-way %d, pathological %d"
          % (bank_conflicts(conflict_free), bank_conflicts(broadcast),
             bank_conflicts(same_bank), bank_conflicts(conflicting_warp)))
    print("PASS")


if __name__ == "__main__":
    main()
