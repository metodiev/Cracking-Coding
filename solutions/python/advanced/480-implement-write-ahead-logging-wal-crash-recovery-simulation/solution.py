"""Task 480 - Implement write-ahead logging (WAL) + crash recovery simulation. (Tier: Advanced, CTCI Ch 14 — Databases)

Write-ahead log engine: every set/delete appends a record "lsn|op|key|value" to the log and
applies it in memory; fsync (flush) advances a durable watermark; checkpoints store a state
snapshot at the current durable LSN. A crash throws away the in-memory state (and any log tail
beyond the watermark), so recovery = latest checkpoint snapshot + replay of durable records
after the snapshot LSN, stopping at the first torn record. Asserted: the recovered state always
equals the committed (durable) prefix, unflushed writes are lost, torn tails are ignored,
repeated crashes are idempotent, and a randomized workload with interleaved flush/checkpoint/
crash actions matches a hand-rolled oracle at every crash.
"""


class WALEngine:
    def __init__(self):
        self.state = {}
        self.wal = []              # [(lsn, op, key, value|None)]
        self.lsn = 0
        self.durable_lsn = 0
        self.checkpoint = None     # (snapshot dict, lsn)
        self.checkpoint_count = 0
        self.crash_count = 0
        self.last_replay_count = 0

    def set(self, key, value):
        self.lsn += 1
        self.wal.append((self.lsn, "set", key, value))
        self.state[key] = value

    def delete(self, key):
        self.lsn += 1
        self.wal.append((self.lsn, "del", key, None))
        self.state.pop(key, None)

    def flush(self):
        self.durable_lsn = self.lsn

    def checkpoint_now(self):
        self.flush()
        self.checkpoint = (dict(self.state), self.lsn)
        self.checkpoint_count += 1

    def wal_text(self, torn_tail=None):
        lines = ["%d|%s|%s|%s" % (lsn, op, key, "" if value is None else value)
                 for lsn, op, key, value in self.wal]
        text = "\n".join(lines)
        if lines:
            text += "\n"
        if torn_tail:
            text += torn_tail
        return text

    def crash(self, torn_tail=None):
        on_disk = self.wal_text(torn_tail)
        durable = self.durable_lsn
        state, replayed = recover(on_disk, durable, self.checkpoint)
        self.state = state
        self.last_replay_count = replayed
        self.lsn = durable
        self.wal = self.wal[:durable]       # restart truncates beyond the durable watermark
        self.durable_lsn = durable
        self.crash_count += 1
        return state


def recover(wal_text, durable_lsn, checkpoint):
    if checkpoint is None:
        state = {}
        snapshot_lsn = 0
    else:
        state = dict(checkpoint[0])
        snapshot_lsn = checkpoint[1]
    replayed = 0
    for line in wal_text.split("\n"):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 4:                 # torn record: stop, drop it
            break
        lsn = int(parts[0])
        if lsn <= snapshot_lsn:
            continue                        # already captured by the checkpoint
        if lsn > durable_lsn:
            break                           # never made durable: lost
        op, key, value = parts[1], parts[2], parts[3]
        if op == "set":
            state[key] = int(value)
        else:
            state.pop(key, None)
        replayed += 1
    return state, replayed


def lcg(seed):
    return (seed * 75 + 74) % 65537


def run_random():
    eng = WALEngine()
    oracle_live = {}
    oracle_durable = {}
    crashes = 0
    seed = 8642
    for i in range(60):
        seed = lcg(seed)
        action = seed % 10
        seed = lcg(seed)
        key = "k%d" % (seed % 12)
        seed = lcg(seed)
        value = seed % 1000
        if action < 5:
            eng.set(key, value)
            oracle_live[key] = value
        elif action < 7:
            eng.delete(key)
            oracle_live.pop(key, None)
        elif action == 7:
            eng.flush()
            oracle_durable = dict(oracle_live)
        elif action == 8:
            eng.checkpoint_now()
            oracle_durable = dict(oracle_live)
        else:
            crashes += 1
            eng.crash()
            assert eng.state == oracle_durable, (i, eng.state, oracle_durable)
            oracle_live = dict(oracle_durable)
    assert crashes >= 1
    assert eng.checkpoint_count >= 1
    eng.checkpoint_now()
    oracle_durable = dict(oracle_live)
    assert eng.state == oracle_live == oracle_durable
    eng.crash()
    assert eng.state == oracle_live, "a crash with nothing in flight is idempotent"
    return crashes, eng.checkpoint_count


def main():
    # Direct parser checks: replay stops at the torn record, watermark and snapshot respected.
    text = "1|set|a|1\n2|set|b|2\n3|del|a|\n4|set|c|3\n5|set|d"
    state, replayed = recover(text, 4, None)
    assert state == {"b": 2, "c": 3} and replayed == 4
    state, replayed = recover(text, 2, None)
    assert state == {"a": 1, "b": 2} and replayed == 2
    state, replayed = recover(text, 5, ({"z": 9}, 3))
    assert state == {"z": 9, "c": 3} and replayed == 1, "snapshot skips LSNs <= 3"

    # Checkpoint + durable suffix replay, then a torn tail beyond the watermark.
    eng = WALEngine()
    for i in range(1, 6):
        eng.set("k%d" % i, i)
    assert eng.lsn == 5 and eng.durable_lsn == 0
    eng.checkpoint_now()
    assert eng.durable_lsn == 5 and eng.checkpoint_count == 1
    for i in range(6, 11):
        eng.set("k%d" % i, i)
    eng.flush()
    eng.set("k11", 11)
    eng.delete("k3")
    eng.set("k12", 12)
    assert eng.lsn == 13
    eng.crash(torn_tail="14|set|k13")
    assert eng.state == {"k%d" % i: i for i in range(1, 11)}, "recovered = committed prefix k1..k10"
    assert eng.last_replay_count == 5, "only the five post-checkpoint durable records replay"
    assert eng.lsn == 10 and eng.durable_lsn == 10

    # The recovered engine keeps logging after the watermark.
    eng.set("k100", 100)
    assert eng.lsn == 11
    eng.flush()
    eng.crash()
    expected = {"k%d" % i: i for i in range(1, 11)}
    expected["k100"] = 100
    assert eng.state == expected
    assert eng.last_replay_count == 6

    # Unflushed writes are lost: only the committed prefix survives.
    eng2 = WALEngine()
    eng2.set("x", 1)
    eng2.flush()
    eng2.set("x", 2)
    eng2.crash()
    assert eng2.state == {"x": 1}

    # Recovery from a bare WAL without any checkpoint.
    eng3 = WALEngine()
    for i in range(4):
        eng3.set("a%d" % i, i)
    eng3.flush()
    eng3.crash()
    assert eng3.state == {"a0": 0, "a1": 1, "a2": 2, "a3": 3}
    assert eng3.last_replay_count == 4

    crashes, checkpoints = run_random()
    assert crashes == 9, "random workload crash count (a==9 rolls at 5,9,15,24,27,32,47,51,55)"
    assert checkpoints == 11, "ten a==8 checkpoints plus the final settle checkpoint"


if __name__ == "__main__":
    main()
    print("PASS")
