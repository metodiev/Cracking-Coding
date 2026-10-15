"""Task 438 - Implement external memory sort outline (multiway merge). (Tier: Advanced, CTCI Ch 10)

Simulated external sort with documented block parameters: BLOCK_SIZE = 4 integers per disk block,
MEMORY_BLOCKS = 4 blocks of RAM (16 integers per initial run) and FAN_IN = 4 (each merge pass merges
groups of up to 4 sorted runs). Phase 1 loads up to one memory worth of integers, sorts them in RAM
(one block write per block of the run) and writes each run back to the simulated disk. Phase 2 keeps
one block of every run in memory with a linear min-scan over the at most FAN_IN = 4 buffers (a heap
would take over for larger fan-in) and streams a single output block; every block read and every
block written is counted by SimDisk. After the last pass the final run is read back. The
counted block transfers are asserted against the documented bound 2 * ceil(n/BLOCK_SIZE) *
(1 + passes) - one initial write pass, two transfers per block per merge pass, one final read pass -
and the result is checked for equality against a naive in-memory sort (correctness only). The disk
also keeps every intermediate run, so the accounting assert total_items == n * (passes + 1) verifies
no element was dropped or duplicated.
"""

import math

BLOCK_SIZE = 4
MEMORY_BLOCKS = 4
FAN_IN = 4


class SimDisk:
    def __init__(self):
        self.runs = {}
        self.next_id = 0
        self.reads = 0
        self.writes = 0

    def create_run(self):
        run_id = self.next_id
        self.next_id += 1
        self.runs[run_id] = []
        return run_id

    def read_block(self, run_id, index):
        self.reads += 1
        return self.runs[run_id][index]

    def append_block(self, run_id, block):
        self.writes += 1
        self.runs[run_id].append(block)

    def block_count(self, run_id):
        return len(self.runs[run_id])

    def io_ops(self):
        return self.reads + self.writes

    def total_items(self):
        return sum(len(block) for run in self.runs.values() for block in run)

    def blocks_sized_ok(self):
        for run in self.runs.values():
            for block in run:
                if not 1 <= len(block) <= BLOCK_SIZE:
                    return False
        return True


def external_sort(data):
    disk = SimDisk()
    capacity = MEMORY_BLOCKS * BLOCK_SIZE
    runs = []
    for start in range(0, len(data), capacity):
        chunk = sorted(data[start:start + capacity])
        run_id = disk.create_run()
        for index in range(0, len(chunk), BLOCK_SIZE):
            disk.append_block(run_id, chunk[index:index + BLOCK_SIZE])
        runs.append(run_id)
    passes = 0
    while len(runs) > 1:
        passes += 1
        merged = []
        for start in range(0, len(runs), FAN_IN):
            merged.append(_merge_group(disk, runs[start:start + FAN_IN]))
        runs = merged
    result = []
    if runs:
        run_id = runs[0]
        for index in range(disk.block_count(run_id)):
            result.extend(disk.read_block(run_id, index))
    return result, disk, passes


def _merge_group(disk, group):
    out_id = disk.create_run()
    blocks = [disk.read_block(run_id, 0) for run_id in group]
    positions = [0] * len(group)
    indices = [0] * len(group)
    out_block = []
    while True:
        best = -1
        for slot in range(len(group)):
            if indices[slot] < len(blocks[slot]):
                if best == -1 or blocks[slot][indices[slot]] < blocks[best][indices[best]]:
                    best = slot
        if best == -1:
            break
        out_block.append(blocks[best][indices[best]])
        indices[best] += 1
        if len(out_block) == BLOCK_SIZE:
            disk.append_block(out_id, out_block)
            out_block = []
        if indices[best] == len(blocks[best]):
            positions[best] += 1
            if positions[best] < disk.block_count(group[best]):
                blocks[best] = disk.read_block(group[best], positions[best])
                indices[best] = 0
    if out_block:
        disk.append_block(out_id, out_block)
    return out_id


def io_bound(item_count, passes):
    blocks = math.ceil(item_count / BLOCK_SIZE)
    return 2 * blocks * (1 + passes)


def main():
    result, disk, passes = external_sort([])
    assert result == [] and passes == 0 and disk.io_ops() == 0

    result, disk, passes = external_sort([7])
    assert result == [7] and passes == 0
    assert disk.reads == 1 and disk.writes == 1

    data = [5, 1, 4, 2, 8, 3, 6, 7, 0, 9]
    result, disk, passes = external_sort(data)
    assert result == sorted(data) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert passes == 0
    assert disk.io_ops() <= io_bound(len(data), passes)
    assert disk.total_items() == len(data) * (passes + 1)
    assert disk.blocks_sized_ok()

    duplicates = [-3, 5, 5, 1, 0, -7, 3, 3, 3, -7, 9, 0, -1, 2, 2, 8, -5]
    result, disk, passes = external_sort(duplicates)
    assert result == sorted(duplicates)
    assert passes == 1
    assert disk.io_ops() <= io_bound(len(duplicates), passes), disk.io_ops()
    assert disk.total_items() == len(duplicates) * (passes + 1)
    assert disk.blocks_sized_ok()

    data = []
    seed = 4380
    for _ in range(2000):
        seed = (seed * 75 + 74) % 65537
        data.append(seed % 1000)
    result, disk, passes = external_sort(data)
    naive = sorted(data)
    assert result == naive
    assert passes == 4
    assert disk.io_ops() <= io_bound(len(data), passes), (disk.io_ops(), io_bound(len(data), passes))
    assert disk.total_items() == len(data) * (passes + 1)
    assert disk.blocks_sized_ok()
    assert max(len(block) for run in disk.runs.values() for block in run) == BLOCK_SIZE

    reversed_data = list(range(200))
    reversed_data.reverse()
    result, disk, passes = external_sort(reversed_data)
    assert result == sorted(reversed_data)
    assert disk.io_ops() <= io_bound(len(reversed_data), passes)
    assert passes > 0


if __name__ == "__main__":
    main()
    print("PASS")
