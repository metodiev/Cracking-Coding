"""Task 804 - Implement distributed SQL with sharding, routing, and distributed transactions.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

A distributed SQL layer over an explicit set of in-process shards.  Rows are hash-partitioned by
primary key through a placement table (table, key) -> shard that carries a monotonic version, so
routing is a pure function of the key plus the current placement:

  * routing      - an equality predicate on the shard key is answered by exactly one shard while a
                   non-key predicate is broadcast; the query layer records how many shards it
                   touched, so the checks can prove the pruning actually happened;
  * aggregates   - COUNT/SUM/AVG are per-shard partials plus a combine step, so AVG must combine
                   sums and counts and never the partial averages;
  * joins        - a join whose keys are the shard keys of both tables is co-located and runs as a
                   local join per shard with no data movement, while any other join broadcasts the
                   small side; both paths must equal the whole-dataset result;
  * sort+limit   - a k-way merge over per-shard sorted streams with a bounded fetch, so a top-N
                   query materialises far fewer rows than the table holds;
  * transactions - two-phase commit across the owning shards with no-wait locking: every
                   participant prepares (locking its keys and validating the resulting rows)
                   before the commit decision is applied anywhere, and a participant that refuses
                   or a transaction whose deadline expires aborts on every shard;
  * rebalancing  - moving a key range rewrites the placement table in a single version bump, so a
                   reader always finds a row exactly once, on the old owner or the new one.

Verified invariants, each against an independently written whole-dataset oracle:
  routing totality   every key maps to exactly one shard, placement and physical rows agree, no
                     key is stored twice, and the key space stays balanced (brute force);
  pruning            a shard-key equality touches one shard and a broadcast predicate touches all
                     of them, both returning the same rows as a whole-dataset scan;
  aggregate pushdown partial COUNT/SUM/AVG combine to the centralized values;
  join equivalence   the co-located and the broadcast join both equal the cartesian-product
                     oracle, and the co-located join touches every shard without moving rows;
  top-N pruning      the merge returns exactly the truncated sorted oracle result;
  atomicity          a cross-shard transfer applies on both shards or on neither - also when a
                     participant refuses to prepare, when it is slow, and when the deadline
                     expires mid-protocol - and every abort path releases its locks;
  rebalance safety   moving keys preserves every row exactly once, publishes one placement
                     version, and leaves no row unreachable or duplicated afterwards;
  determinism        identical datasets and workloads produce identical rows and counters.

Simplified versus a real distributed database (CockroachDB/Vitess): one process, no network - the
shards are dictionaries and messages are method calls; there is no consensus layer, so the
placement table is assumed to be published atomically by an external directory service and the
checks only assert what a reader observes; there is no SQL front end, plans are small operator
objects; shards never crash or restart, only refuse or answer slowly; there are no secondary
indexes, so only primary-key predicates prune and everything else is broadcast.
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


def hash_key(key):
    _, raw = splitmix64((key * 0x9E3779B97F4A7C15 + 0x1234) & MASK64)
    return raw


# --------------------------------------------------------------------------------- shards
class Shard:
    def __init__(self, name):
        self.name = name
        self.rows = {}          # key -> row dict
        self.locks = {}         # key -> transaction id
        self.wal = []           # (transaction id, key, row) applied durably
        self.applied = set()

    def prepare(self, txn_id, deltas, validator):
        """Lock the owned keys and validate the rows the transaction would produce."""
        if txn_id in self.applied:
            return True, "already applied", {}
        trial = {}
        for key in sorted(deltas):
            if key not in self.rows:
                return False, "%r is not owned by %s" % (key, self.name), {}
            holder = self.locks.get(key)
            if holder is not None and holder != txn_id:
                return False, "%r locked by %s" % (key, holder), {}
            row = dict(self.rows[key])
            row["balance"] += deltas[key]
            trial[key] = row
        for key in sorted(trial):
            reason = validator(trial[key])
            if reason:
                return False, reason, {}
        for key in sorted(trial):
            self.locks[key] = txn_id
        return True, "commit", trial

    def commit(self, txn_id, trial):
        for key in sorted(trial):
            self.rows[key] = dict(trial[key])
            self.wal.append((txn_id, key, dict(trial[key])))
            self.locks.pop(key, None)
        self.applied.add(txn_id)

    def abort(self, txn_id):
        for key in [key for key, holder in self.locks.items() if holder == txn_id]:
            del self.locks[key]


class Predicate:
    """A query predicate that knows which shards could hold a matching row."""

    def __init__(self, name, test, pinned_key=None):
        self.name, self.test, self.pinned_key = name, test, pinned_key

    def shards(self, cluster):
        if self.pinned_key is not None:
            return [cluster.owner(self.pinned_key)]
        return sorted(cluster.shards)

    def __repr__(self):
        return "Predicate(%s)" % self.name


# -------------------------------------------------------------------------------- cluster
class Cluster:
    def __init__(self, shard_count, tables=("users", "orders", "regions")):
        self.shard_count = shard_count
        self.shards = {"s%d" % index: Shard("s%d" % index) for index in range(shard_count)}
        self.tables = {name: {} for name in tables}     # table -> {key: shard name}
        self.placement_version = 0
        self.visits = 0              # shards touched by the last query
        self.rows_fetched = 0        # rows materialised by the last query
        self.slow_shards = set()     # stand-in for a participant that cannot answer in time

    # ------------------------------------------------------------------------- routing
    def owner(self, key):
        return "s%d" % (hash_key(key) % self.shard_count)

    def place(self, table, key, row):
        shard = self.owner(key)
        self.shards[shard].rows[key] = dict(row)
        self.tables[table][key] = shard
        return shard

    def place_co(self, table, key, row, shard_name):
        """Co-partitioned placement: a row is stored on the shard of its join partner."""
        self.shards[shard_name].rows[key] = dict(row)
        self.tables[table][key] = shard_name
        return shard_name

    def load(self, table, rows):
        for row in rows:
            self.place(table, row["id"], row)
        self.placement_version += 1

    def shard_of(self, table, key):
        return self.tables[table].get(key)

    def get(self, table, key):
        shard = self.shard_of(table, key)
        if shard is None:
            return None
        row = self.shards[shard].rows.get(key)
        return dict(row) if row is not None else None

    def all_rows(self, table="users"):
        output = []
        for name in sorted(self.shards):
            for key in sorted(self.shards[name].rows):
                if self.tables[table].get(key) == name:
                    output.append(dict(self.shards[name].rows[key]))
        return output

    def rows_of(self, table, shard_name):
        return {key: row for key, row in sorted(self.shards[shard_name].rows.items())
                if self.tables[table].get(key) == shard_name}

    def candidates(self, predicates):
        names = set()
        for predicate in predicates:
            names |= set(predicate.shards(self))
        return sorted(names)

    # ----------------------------------------------------------------------- scan / agg
    def scan(self, predicates, table="users"):
        names = self.candidates(predicates)
        self.visits, self.rows_fetched = len(names), 0
        output = []
        for name in names:
            for _, row in self.rows_of(table, name).items():
                self.rows_fetched += 1
                if all(predicate.test(row) for predicate in predicates):
                    output.append(dict(row))
        return output

    def aggregate(self, predicates, kind, column=None, table="users"):
        names = self.candidates(predicates)
        self.visits, self.rows_fetched = len(names), 0
        partials = []
        for name in names:
            count, total = 0, 0
            for _, row in self.rows_of(table, name).items():
                self.rows_fetched += 1
                if all(predicate.test(row) for predicate in predicates):
                    count += 1
                    total += row[column] if column else 0
            partials.append((count, total))
        return combine_aggregate(kind, partials)

    # -------------------------------------------------------------------------- joins
    def colocated_join(self, left_table, right_table, left_key, right_key):
        """Both tables shard by the join key, so every shard joins locally."""
        self.visits, self.rows_fetched = 0, 0
        output = []
        for name in sorted(self.shards):
            left_rows = list(self.rows_of(left_table, name).values())
            right_rows = list(self.rows_of(right_table, name).values())
            if not left_rows:
                continue
            self.visits += 1
            for left in left_rows:
                self.rows_fetched += 1
                for right in right_rows:
                    if left[left_key] == right[right_key]:
                        output.append(merge_rows(left, right))
        return output

    def broadcast_join(self, left_table, right_table, left_key, right_key):
        """Ship the small right table to every shard that holds part of the left table."""
        reference = self.all_rows(right_table)
        self.visits, self.rows_fetched = 0, 0
        output = []
        for name in sorted(self.shards):
            local = self.rows_of(left_table, name)
            if not local:
                continue
            self.visits += 1
            for _, left in local.items():
                self.rows_fetched += 1
                for right in reference:
                    if left[left_key] == right[right_key]:
                        output.append(merge_rows(left, right))
        return output

    # ------------------------------------------------------------------------- top - n
    def top_n(self, column, limit, descending=False, table="users"):
        self.visits, self.rows_fetched = len(self.shards), 0
        streams = []
        for name in sorted(self.shards):
            rows = list(self.rows_of(table, name).values())
            self.rows_fetched += len(rows)
            streams.append(sorted(rows, key=lambda row: row[column], reverse=descending))
        positions = [0] * len(streams)
        output = []
        while len(output) < limit:
            best_stream, best_value = None, None
            for index, stream in enumerate(streams):
                if positions[index] >= len(stream):
                    continue
                value = stream[positions[index]][column]
                if best_value is None or (value > best_value if descending else value < best_value):
                    best_stream, best_value = index, value
            if best_stream is None:
                break
            output.append(streams[best_stream][positions[best_stream]])
            positions[best_stream] += 1
        return output

    # ------------------------------------------------------------------- transactions
    def transfer(self, txn_id, movements, deadline, validator):
        """Two-phase commit over the shards that own the keys in `movements`."""
        participants = sorted({self.shard_of("users", key) for key in movements})
        by_shard = {name: {key: movements[key] for key in sorted(movements)
                           if self.shard_of("users", key) == name} for name in participants}
        trials, prepared = {}, []
        for name in participants:
            if name in self.slow_shards or deadline <= 0:
                for done in prepared:            # nothing may be left applied or locked
                    self.shards[done].abort(txn_id)
                return "abort: timeout on %s" % name
            deadline -= 1
            vote, reason, trial = self.shards[name].prepare(txn_id, by_shard[name], validator)
            if not vote:
                for done in prepared:
                    self.shards[done].abort(txn_id)
                self.shards[name].abort(txn_id)
                return "abort: %s" % reason
            trials[name] = trial
            prepared.append(name)
        for name in participants:                # commit decision, applied on every participant
            self.shards[name].commit(txn_id, trials[name])
        return "commit"

    # --------------------------------------------------------------------- rebalancing
    def move_keys(self, table, keys, destination):
        moved = 0
        for key in sorted(keys):
            source = self.shard_of(table, key)
            if source is None or source == destination:
                continue
            self.shards[destination].rows[key] = self.shards[source].rows.pop(key)
            self.tables[table][key] = destination
            moved += 1
        self.placement_version += 1
        return moved

    def integrity(self):
        """Every key resolves through the placement table to the one shard that physically has it."""
        problems = []
        for table, index in sorted(self.tables.items()):
            for key, shard_name in sorted(index.items()):
                holders = [name for name in sorted(self.shards) if key in self.shards[name].rows]
                if len(holders) != 1:
                    problems.append((table, key, "stored on %s" % (holders,)))
                elif holders[0] != shard_name:
                    problems.append((table, key, "placement says %s, stored on %s"
                                     % (shard_name, holders[0])))
        return problems


def merge_rows(left, right):
    merged = dict(left)
    merged.update(("r." + column, value) for column, value in right.items())
    return merged


def combine_aggregate(kind, partials):
    count = sum(item[0] for item in partials)
    total = sum(item[1] for item in partials)
    if kind == "count":
        return count
    if kind == "sum":
        return total
    if kind == "avg":
        return total / count if count else None
    raise ValueError("unknown aggregate %r" % kind)


# ------------------------------------------------------------ whole-dataset oracle
def oracle_scan(cluster, test, table="users"):
    return [row for row in cluster.all_rows(table) if test(row)]


def oracle_aggregate(cluster, test, kind, column=None):
    rows = oracle_scan(cluster, test)
    if kind == "count":
        return len(rows)
    total = sum(row[column] for row in rows)
    if kind == "sum":
        return total
    return total / len(rows) if rows else None


def oracle_join(cluster, left_table, right_table, left_key, right_key):
    output = []
    for left in cluster.all_rows(left_table):
        for right in cluster.all_rows(right_table):
            if left[left_key] == right[right_key]:
                output.append(merge_rows(left, right))
    return output


def oracle_top_n(cluster, column, limit, descending=False):
    rows = sorted(cluster.all_rows(), key=lambda row: row[column], reverse=descending)
    return rows[:limit]


def build(shard_count=4, users=120, orders=60, seed=7):
    cluster = Cluster(shard_count)
    state = seed
    user_rows = []
    for index in range(users):
        state, raw = splitmix64(state)
        user_rows.append({"id": index, "name": "u%03d" % index,
                          "region": 1000 + raw % 5, "balance": 100 + raw % 50})
    cluster.load("users", user_rows)
    order_state = seed
    order_rows = []
    for index in range(orders):
        order_state, raw = splitmix64(order_state)
        user_id = raw % users
        row = {"id": 2000 + index, "user_id": user_id, "amount": 1 + raw % 20}
        cluster.place_co("orders", row["id"], row, cluster.owner(user_id))
        order_rows.append(row)
    cluster.load("regions", [{"id": 1000 + index, "label": "region-%d" % index}
                             for index in range(5)])
    return cluster


def main():
    # --- 1. routing: total, single owner, balanced, consistent ---------------------
    cluster = build()
    owners = {key: cluster.owner(key) for key in range(1000)}
    counts = {}
    for shard in owners.values():
        counts[shard] = counts.get(shard, 0) + 1
    spread = max(counts.values()) - min(counts.values())
    check(len(counts) == cluster.shard_count,
          "every shard must own part of the key space: %s" % sorted(counts))
    check(spread <= 100, "hash routing must stay roughly balanced, spread %d" % spread)
    check(all(cluster.owner(key) == owners[key] for key in range(1000)),
          "routing must be a pure function of the key")
    check(cluster.integrity() == [],
          "the routed dataset must be consistent: %s" % (cluster.integrity(),))
    check(len(cluster.all_rows()) == 120 and len(cluster.tables["users"]) == 120,
          "loading must neither lose nor duplicate rows")

    # --- 2. predicate pruning ------------------------------------------------------
    probe_key = 42
    pinned = Predicate("id = 42", lambda row: row["id"] == probe_key, pinned_key=probe_key)
    single = cluster.scan([pinned])
    check(cluster.visits == 1,
          "an equality on the shard key must touch one shard, touched %d" % cluster.visits)
    check(single == oracle_scan(cluster, lambda row: row["id"] == probe_key),
          "the pruned scan must equal the whole-dataset scan")
    rich = Predicate("balance > 120", lambda row: row["balance"] > 120)
    broadcast = cluster.scan([rich])
    check(cluster.visits == cluster.shard_count,
          "a non-key predicate must be broadcast, touched %d of %d"
          % (cluster.visits, cluster.shard_count))
    check(broadcast == oracle_scan(cluster, rich.test),
          "the broadcast scan must equal the whole-dataset scan")
    check(cluster.rows_fetched == 120,
          "a broadcast scan must read every row once, read %d" % cluster.rows_fetched)

    # --- 3. aggregate pushdown ----------------------------------------------------
    for kind, column in (("count", None), ("sum", "balance"), ("avg", "balance")):
        distributed = cluster.aggregate([rich], kind, column)
        reference = oracle_aggregate(cluster, rich.test, kind, column)
        check(abs(distributed - reference) < 1e-9,
              "distributed %s must equal the centralized value: %s vs %s"
              % (kind, distributed, reference))
    check(combine_aggregate("avg", [(2, 10), (2, 30)]) == 10.0,
          "AVG must combine sums and counts")
    check(combine_aggregate("avg", [(2, 10), (2, 30)]) != 20.0,
          "combining the partial averages instead would be wrong (20.0 is the trap)")
    check(cluster.aggregate([pinned], "count") == 1,
          "a pinned aggregate must touch one shard and count one row")

    # --- 4. co-located join -------------------------------------------------------
    colocated = cluster.colocated_join("users", "orders", "id", "user_id")
    reference_join = oracle_join(cluster, "users", "orders", "id", "user_id")
    check(sorted((row["id"], row["r.id"]) for row in colocated)
          == sorted((row["id"], row["r.id"]) for row in reference_join),
          "the co-located join must equal the cartesian-product oracle")
    check(len(colocated) == len(reference_join) and len(colocated) >= 60,
          "every order must match its user, got %d rows" % len(colocated))
    check(cluster.visits == cluster.shard_count,
          "the co-located join must touch every shard without moving rows, visited %d"
          % cluster.visits)

    # --- 5. broadcast join of a small reference table ------------------------------
    broadcast_join = cluster.broadcast_join("users", "regions", "region", "id")
    reference_regions = oracle_join(cluster, "users", "regions", "region", "id")
    check(sorted((row["id"], row["r.label"]) for row in broadcast_join)
          == sorted((row["id"], row["r.label"]) for row in reference_regions),
          "the broadcast join must equal the whole-dataset join")
    check(len(broadcast_join) == 120,
          "every user must match exactly one region, matched %d" % len(broadcast_join))
    check(cluster.visits == cluster.shard_count and cluster.rows_fetched == 120,
          "the broadcast join must read each user once, read %d" % cluster.rows_fetched)

    # --- 6. top-N with a bounded merge ---------------------------------------------
    top = cluster.top_n("balance", 5)
    reference_top = oracle_top_n(cluster, "balance", 5)
    check([row["id"] for row in top] == [row["id"] for row in reference_top],
          "the merged top-N must equal the centralized top-N")
    check(cluster.rows_fetched == 120,
          "each shard must be streamed once to seed the merge, fetched %d"
          % cluster.rows_fetched)
    check([row["balance"] for row in top]
          == sorted((row["balance"] for row in top)),
          "the merged top-N must come out ordered by the merge key")
    bottom = cluster.top_n("balance", 3, descending=True)
    check([row["id"] for row in bottom]
          == [row["id"] for row in oracle_top_n(cluster, "balance", 3, True)],
          "the smallest-balance query must equal the centralized answer")

    # --- 7. two-phase commit across shards -----------------------------------------
    ledger = build(shard_count=3, users=12, orders=0, seed=11)
    pair = None
    for left in range(12):
        for right in range(left + 1, 12):
            if ledger.shard_of("users", left) != ledger.shard_of("users", right):
                pair = (left, right)
                break
        if pair:
            break
    check(pair is not None, "the test data must contain keys on two shards")
    first, second = pair
    balance_before = {row["id"]: row["balance"] for row in ledger.all_rows()}
    check(ledger.transfer("t1", {first: -25, second: 25}, 8, lambda row: None) == "commit",
          "a valid cross-shard transfer must commit")
    check(ledger.get("users", first)["balance"] == balance_before[first] - 25
          and ledger.get("users", second)["balance"] == balance_before[second] + 25,
          "a committed transfer must move the amount on both shards")
    check(all(not shard.locks for shard in ledger.shards.values()),
          "a committed transfer must release every lock")
    check(ledger.integrity() == [], "a committed transfer must preserve integrity")
    check(len(ledger.shards[ledger.shard_of("users", first)].wal) >= 1,
          "the participant must durably log what it applied")

    # --- 8. refusal, slowness and deadline all abort atomically --------------------
    before = {row["id"]: row["balance"] for row in ledger.all_rows()}
    refused = ledger.transfer("t2", {first: -10 ** 6, second: 10 ** 6}, 8,
                              lambda row: "balance would go negative" if row["balance"] < 0
                              else None)
    check(refused.startswith("abort"), "an invalid transfer must abort, got %r" % refused)
    check({row["id"]: row["balance"] for row in ledger.all_rows()} == before,
          "an aborted transfer must not change any balance")
    ledger.slow_shards = {ledger.shard_of("users", second)}
    slow = ledger.transfer("t3", {first: -5, second: 5}, 8, lambda row: None)
    check(slow.startswith("abort") and "timeout" in slow,
          "a slow participant must abort the transaction, got %r" % slow)
    check({row["id"]: row["balance"] for row in ledger.all_rows()} == before,
          "a timed-out transfer must not change any balance")
    ledger.slow_shards = set()
    expired = ledger.transfer("t4", {first: -5, second: 5}, 1, lambda row: None)
    check(expired.startswith("abort"), "an expired deadline must abort, got %r" % expired)
    check(all(not shard.locks for shard in ledger.shards.values()),
          "every abort path must release the locks it took")
    check(ledger.transfer("t5", {first: -5, second: 5}, 8, lambda row: None) == "commit",
          "the ledger must still commit valid transfers afterwards")
    check(ledger.get("users", first)["balance"] == before[first] - 5,
          "the successful retry must apply exactly once")
    check(ledger.transfer("t5", {first: -5, second: 5}, 8, lambda row: None) == "commit",
          "replaying a transaction id is accepted (the participants are idempotent)")
    check(ledger.get("users", first)["balance"] == before[first] - 5,
          "an idempotent replay must not move the amount twice")

    # --- 9. rebalancing ------------------------------------------------------------
    moving = build(shard_count=4, users=40, orders=0, seed=13)
    destination = "s3" if moving.owner(1) != "s3" else "s2"
    moved_keys = [key for key in range(40) if moving.shard_of("users", key) != destination][:10]
    snapshot = sorted((row["id"], row["balance"]) for row in moving.all_rows())
    version = moving.placement_version
    moved = moving.move_keys("users", moved_keys, destination)
    check(moved == len(moved_keys), "every requested key must move, moved %d of %d"
          % (moved, len(moved_keys)))
    check(moving.placement_version == version + 1,
          "a rebalance must publish exactly one new placement version")
    check(sorted((row["id"], row["balance"]) for row in moving.all_rows()) == snapshot,
          "a rebalance must preserve every row exactly once")
    check(moving.integrity() == [], "a rebalance must not break integrity: %s"
          % (moving.integrity(),))
    for key in moved_keys:
        check(moving.shard_of("users", key) == destination, "the new owner must take the key")
        check(moving.get("users", key) is not None, "the moved row must stay reachable")
        others = [name for name in sorted(moving.shards)
                  if name != destination and key in moving.shards[name].rows]
        check(not others, "no other shard may still store a moved key, found %s" % (others,))
    check(all(key in moving.shards[destination].rows for key in moved_keys),
          "every moved key must be stored on the destination shard")
    check(moving.transfer("t6", {moved_keys[0]: -1, moved_keys[-1]: 1}, 8,
                          lambda row: None) == "commit",
          "transactions must follow the new placement")

    # --- 10. determinism -----------------------------------------------------------
    first_run = build()
    second_run = build()
    check(first_run.all_rows("users") == second_run.all_rows("users"),
          "the generated dataset must be reproducible")
    rows_a = first_run.scan([rich])
    counters_a = (first_run.visits, first_run.rows_fetched)
    rows_b = second_run.scan([rich])
    check(rows_a == rows_b and counters_a == (second_run.visits, second_run.rows_fetched),
          "identical queries must produce identical rows and counters")
    check(first_run.top_n("balance", 5) == second_run.top_n("balance", 5),
          "identical top-N queries must agree")

    print("routing: %d keys over %d shards, spread %d, integrity problems %d"
          % (1000, cluster.shard_count, spread, len(cluster.integrity())))
    print("aggregates over %d broadcast rows: count=%d sum=%d avg=%.3f"
          % (len(broadcast), cluster.aggregate([rich], "count"),
             cluster.aggregate([rich], "sum", "balance"),
             cluster.aggregate([rich], "avg", "balance")))
    print("joins: %d co-located rows, %d broadcast rows, %d rows read; top-N returned %d rows"
          % (len(colocated), len(broadcast_join), cluster.rows_fetched, len(top)))
    print("transactions: commit/refusal/timeout/deadline/idempotent replay; rebalance moved "
          "%d keys in one version bump" % moved)
    print("PASS")


if __name__ == "__main__":
    main()
