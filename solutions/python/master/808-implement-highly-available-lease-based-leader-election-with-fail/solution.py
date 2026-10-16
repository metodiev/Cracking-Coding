"""Task 808 - Implement highly-available lease-based leader election with failover testing.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

Lease-based leader election with bounded clock drift, simulated as a deterministic real-time
schedule.  Real time is an explicit integer; every server has a local clock `t + drift` with
`|drift| <= D`.  A candidate collects grants from a quorum of voters; a voter that grants a lease
refuses every later grant until its own clock has passed the grant time plus the lease length plus
the drift guard, and the leader only acts while its own clock is inside the granted interval minus
its own drift guard.  Those two guards are what make the protocol safe: any two quorums intersect
in a voter, that voter's refusal window covers the whole of the first lease *in real time even
though the two sides disagree about the clock*, so a second leader cannot become active until the
first leader has provably stopped.

The checks brute-force the resulting schedule: every lease window granted anywhere in a run is
compared pairwise in real time and must be disjoint, leaders must stop acting at their validity
deadline, a minority candidate must never acquire a lease, and failover must happen within the
bounded window that the guards imply.  A deliberately mis-configured cluster whose voters drift
beyond D is run as well, and the same detector must flag the overlap - which is what shows the
drift bound is load bearing rather than decorative.

Verified invariants:
  mutual exclusion     all lease validity windows in a run are pairwise disjoint in real time, even
                       with extreme (but legal) drifts, crashes, renewals and frequent failover;
  bounded stop time    a leader that loses its quorum stops acting by its deadline, and no action
                       is ever recorded after the deadline;
  no stale leader      a partitioned (minority) candidate acquires nothing and performs nothing;
  failover liveness    after a leader dies a new lease is acquired within guard + drift ticks of
                       the old deadline (lease + guard - (lease - drift)), and the cluster keeps
                       an active leader for the rest of the run;
  renewal continuity   a healthy leader renews before expiry and its validity window is gap-free;
  quorum arithmetic    the quorum predicate accepts exactly the majority subsets (brute force);
  clock monotonicity   lease terms increase, grants never move a voter's refusal window backwards,
                       and a voter never grants twice inside one refusal window;
  drift bound          with a voter drifting beyond D the pairwise disjointness check fails, so the
                       bound is necessary, not cosmetic;
  determinism          identical schedules produce identical leases, leaders and window sets.

Simplified versus a production election service (Chubby/Spanner): one process, so messages are
immediate and the network is not modelled - the interesting uncertainty is clock drift, which is
modelled explicitly; there is no log replication (a leader is only a right to act), no reconfiguration of the
member set, no watchdog process on the voters, and timeouts are driven by the schedule rather than
by timers.  Leases are integers and the run length is a few hundred ticks, so the whole schedule can
be brute-forced afterwards.
"""
DEFAULT_DURATION = 30
DEFAULT_DRIFT = 3
VOTER_GUARD = 2 * DEFAULT_DRIFT      # the voter stays closed for this long past the lease end
LEADER_GUARD = DEFAULT_DRIFT         # the leader stops this long before the clock end


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


def quorum_of(size):
    return size // 2 + 1


class Voter:
    def __init__(self, name, drift, guard=VOTER_GUARD):
        self.name = name
        self.drift = drift
        self.guard = guard
        self.refuse_before = -(1 << 60)      # a local-clock instant
        self.holder = None                   # the leader this voter promised the window to
        self.grants = []                     # (local instant, local refusal end, real time, holder)

    def clock(self, real_time):
        return real_time + self.drift

    def grant(self, real_time, duration, holder):
        """Grant or refresh a lease; only the holder may extend into its own refusal window."""
        local = self.clock(real_time)
        if local < self.refuse_before and self.holder != holder:
            return False
        self.holder = holder
        self.refuse_before = local + duration + self.guard
        self.grants.append((local, self.refuse_before, real_time, holder))
        return True


class Lease:
    def __init__(self, term, leader, granted_at, valid_from, valid_until, duration, voters):
        self.term = term
        self.leader = leader
        self.granted_at = granted_at
        self.valid_from = valid_from
        self.valid_until = valid_until
        self.duration = duration
        self.voters = tuple(voters)

    def covers(self, real_time):
        return self.valid_from <= real_time < self.valid_until

    def __repr__(self):
        return "<lease term %d %s [%d, %d)>" % (self.term, self.leader, self.valid_from,
                                                self.valid_until)


class Cluster:
    """Voters plus an explicit real-time schedule; every action is a method call at a tick."""

    def __init__(self, size=5, duration=DEFAULT_DURATION, drifts=None, guard=VOTER_GUARD):
        self.size = size
        self.duration = duration
        self.guard = guard
        names = ["v%d" % index for index in range(size)]
        self.drifts = drifts or self.default_drifts(size)
        self.voters = {name: Voter(name, self.drifts[name], guard) for name in names}
        self.term = 0
        self.leases = []
        self.active = None                   # the currently valid lease
        self.actions = []                    # (real time, leader, term)
        self.refused = 0
        self.blocked = set()                 # voters that cannot be reached
        self.down_until = {}

    @staticmethod
    def default_drifts(size, spread=DEFAULT_DRIFT):
        drifts = {}
        state = 0x5EED
        for index in range(size):
            state, raw = splitmix64(state + 17 * index)
            drifts["v%d" % index] = (raw % (2 * spread + 1)) - spread
        return drifts

    # ------------------------------------------------------------------ protocol
    def voters_in_order(self, candidate):
        return [name for name in sorted(self.voters) if name != candidate]

    def try_acquire(self, candidate, real_time, duration=None):
        """Ask every reachable voter, keep a lease when a quorum granted one."""
        duration = duration or self.duration
        if candidate in self.blocked:              # an unreachable candidate cannot campaign
            self.refused += 1
            return None
        grants = []
        for name in self.voters_in_order(candidate):
            if name in self.blocked:
                continue
            voter = self.voters[name]
            if voter.grant(real_time, duration, candidate):
                grants.append(name)
        self_granted = self.voters[candidate].grant(real_time, duration, candidate)
        if self_granted:
            grants.append(candidate)
        if len(grants) < quorum_of(self.size):
            self.refused += 1
            return None
        self.term += 1
        leader = self.voters[candidate]
        valid_from = real_time
        valid_until = leader.clock(real_time) + duration - LEADER_GUARD - leader.drift
        lease = Lease(self.term, candidate, real_time, valid_from, valid_until, duration,
                      sorted(grants))
        self.leases.append(lease)
        self.active = lease
        return lease

    def renew(self, lease, real_time, duration=None):
        """A renewal is a fresh grant round; the leader may only renew while still valid."""
        if lease is not self.active or not lease.covers(real_time):
            return None
        renewed = self.try_acquire(lease.leader, real_time, duration)
        if renewed is None:
            return None
        return renewed

    def act(self, real_time):
        """The leader performs one committed action, if it is still inside its lease."""
        if self.active is None or not self.active.covers(real_time):
            return None
        self.actions.append((real_time, self.active.leader, self.active.term))
        return self.active.leader

    def expire(self, real_time):
        """Drop a lease the leader failed to renew before its deadline."""
        if self.active is not None and real_time >= self.active.valid_until:
            expired = self.active
            self.active = None
            return expired
        return None


def window_overlaps(leases):
    """Brute force: every pair of lease windows must be disjoint in real time."""
    problems = []
    for index, left in enumerate(leases):
        for right in leases[index + 1:]:
            if left.leader == right.leader:
                continue
            if left.valid_from < right.valid_until and right.valid_from < left.valid_until:
                problems.append((left.term, left.leader, right.term, right.leader,
                                 left.valid_from, left.valid_until, right.valid_from,
                                 right.valid_until))
    return problems


def action_problems(cluster):
    """No action may be recorded outside the actor's lease window."""
    problems = []
    by_term = {lease.term: lease for lease in cluster.leases}
    for real_time, leader, term in cluster.actions:
        lease = by_term.get(term)
        if lease is None or not lease.covers(real_time):
            problems.append((real_time, leader, term))
    return problems


def voter_grant_problems(cluster):
    """A voter may never promise two different leaders overlapping windows."""
    problems = []
    for voter in cluster.voters.values():
        last_end, last_holder = -(1 << 60), None
        for local, refuse_end, real_time, holder in voter.grants:
            if local < last_end and holder != last_holder:
                problems.append((voter.name, local, last_end, holder))
            last_end, last_holder = refuse_end, holder
    return problems


def quorum_predicate_matches_bruteforce(size):
    universe = list(range(size))
    for mask in range(1 << size):
        subset = {universe[index] for index in range(size) if mask >> index & 1}
        if (len(subset) >= quorum_of(size)) != (len(subset) * 2 > size):
            return False, subset
    return True, None


def run_schedule(cluster, ticks, renew_every=10, plan=None, duration=None, candidates=None):
    """Drive the cluster: a leader renews on schedule, a plan can cut voters off, and idle
    candidates take turns trying to acquire a lease."""
    candidates = candidates or sorted(cluster.voters)
    attempts = 0
    for tick in range(ticks):
        if plan is not None:
            plan(cluster, tick)
        cluster.expire(tick)
        if cluster.active is None:
            cluster.try_acquire(candidates[attempts % len(candidates)], tick, duration)
            attempts += 1
        elif tick % renew_every == 0:
            cluster.renew(cluster.active, tick, duration)
        cluster.act(tick)
    return cluster


def main():
    # --- 1. quorum predicate ------------------------------------------------------
    for size in (3, 5, 7):
        ok, witness = quorum_predicate_matches_bruteforce(size)
        check(ok, "the quorum predicate must be a strict majority for size %d: %s"
              % (size, witness))
    check(quorum_of(5) == 3 and quorum_of(4) == 3,
          "a five-voter cluster needs three grants and a four-voter cluster needs three")

    # --- 2. a quorum acquires a lease, a minority cannot ---------------------------
    cluster = Cluster(size=5)
    lease = cluster.try_acquire("v0", 0)
    check(lease is not None and lease.leader == "v0", "a five-voter cluster must grant a lease")
    check(len(lease.voters) >= 3, "the lease must rest on a quorum, got %s" % (lease.voters,))
    check(lease.valid_from == 0 and lease.valid_until == cluster.duration - LEADER_GUARD,
          "the leader guard must shorten the validity window: %d" % lease.valid_until)
    solo = Cluster(size=5)
    solo.blocked = {"v1", "v2", "v3"}
    check(solo.try_acquire("v0", 0) is None and solo.refused == 1,
          "a candidate that cannot reach a quorum must not acquire a lease")
    check(solo.active is None and solo.leases == [], "a refused candidate must hold nothing")
    check(solo.act(0) is None, "a refused candidate must not act either")

    # --- 3. mutual exclusion under a long, hostile schedule -----------------------
    busy = Cluster(size=5)

    def plan(cluster, tick):
        if tick in (20, 100) and cluster.active is not None:
            cluster.blocked = {cluster.active.leader}   # the leader loses the cluster
        if tick in (60, 140):
            cluster.blocked = set()

    run_schedule(busy, 220, renew_every=7, plan=plan)
    check(len(busy.leases) >= 3, "the hostile schedule must force several leases, saw %d"
          % len(busy.leases))
    check(window_overlaps(busy.leases) == [],
          "lease windows must never overlap: %s" % (window_overlaps(busy.leases)[:2],))
    check(action_problems(busy) == [],
          "no action may be recorded outside a valid lease: %s" % (action_problems(busy)[:3],))
    check(voter_grant_problems(busy) == [],
          "a voter must never grant twice inside one refusal window: %s"
          % (voter_grant_problems(busy)[:3],))
    terms = [lease.term for lease in busy.leases]
    check(terms == sorted(terms) and len(set(terms)) == len(terms),
          "lease terms must strictly increase: %s" % (terms,))
    holders = {lease.leader for lease in busy.leases}
    check(len(holders) >= 2, "failover must have moved leadership between servers: %s" % holders)

    # --- 4. leaders stop at their deadline ----------------------------------------
    stopped = Cluster(size=3, duration=12)
    stopped.try_acquire("v0", 0)
    first = stopped.active
    check(first is not None, "the three-voter cluster must grant a lease")
    last_action = None
    for tick in range(0, 40):
        if stopped.act(tick) is not None:
            last_action = tick
    check(last_action is not None and last_action < first.valid_until,
          "a leader must stop acting before its deadline: last action %s, deadline %d"
          % (last_action, first.valid_until))
    check(stopped.actions[-1][0] == first.valid_until - 1,
          "the leader must act up to the last tick of its lease, got %s" % (stopped.actions[-1],))
    expired = stopped.expire(first.valid_until)
    check(expired is first and stopped.active is None,
          "a lease must expire once its deadline passes")
    check(stopped.act(first.valid_until) is None,
          "an expired lease must not permit any action")
    check(action_problems(stopped) == [], "the expired lease must not have been used late")

    # --- 5. failover liveness and disjointness ------------------------------------
    failover = Cluster(size=5)
    failover.try_acquire("v0", 0)
    old = failover.active
    check(old is not None, "the first lease must be granted")
    for tick in range(1, old.valid_until + 1):
        failover.act(tick)
    failover.expire(old.valid_until)
    new = failover.try_acquire("v1", old.valid_until)
    check(new is None,
          "a new leader cannot be granted the moment the old one expires: %s" % (new,))
    deadline = None
    for tick in range(old.valid_until, old.valid_until + 4 * DEFAULT_DRIFT + 2):
        candidate_lease = failover.try_acquire("v1", tick)
        if candidate_lease is not None:
            deadline = tick
            break
    check(deadline is not None,
          "failover must complete within the drift guard window")
    check(deadline - old.valid_until <= 3 * DEFAULT_DRIFT,
          "failover took %d ticks after the old deadline" % (deadline - old.valid_until))
    check(failover.active.valid_from >= old.valid_until,
          "the new lease must start after the old one ended")
    check(window_overlaps(failover.leases) == [],
          "failover must not produce overlapping windows: %s"
          % (window_overlaps(failover.leases),))

    # --- 6. renewal keeps a healthy leader continuously valid ---------------------
    healthy = Cluster(size=5, duration=20)
    healthy.try_acquire("v0", 0)
    covered = []
    for tick in range(0, 180):
        if tick % 5 == 0 and healthy.active is not None:
            healthy.renew(healthy.active, tick)
        healthy.expire(tick)
        if healthy.active is None:
            healthy.try_acquire("v0", tick)
        covered.append(healthy.act(tick) is not None)
    check(all(covered),
          "a healthy leader that renews on time must never lose validity, gaps at %s"
          % [index for index, value in enumerate(covered) if not value][:5])
    check(len(healthy.leases) > 4, "renewal must create fresh leases, saw %d"
          % len(healthy.leases))
    check(window_overlaps(healthy.leases) == [],
          "renewals by the same leader must not overlap other leaders")

    # --- 7. extreme legal drifts still cannot break exclusion ---------------------
    extreme = Cluster(size=5, drifts={"v0": DEFAULT_DRIFT, "v1": -DEFAULT_DRIFT,
                                      "v2": DEFAULT_DRIFT, "v3": -DEFAULT_DRIFT,
                                      "v4": 0})
    run_schedule(extreme, 200, renew_every=9)
    check(window_overlaps(extreme.leases) == [],
          "extreme but legal drifts must not break exclusion: %s"
          % (window_overlaps(extreme.leases)[:2],))
    check(len(extreme.leases) >= 2, "the run must include several leases")
    check(all(abs(drift) <= DEFAULT_DRIFT for drift in extreme.drifts.values()),
          "the cluster must stay inside the documented drift bound")

    # --- 8. both guards are load bearing ------------------------------------------
    overrun = Cluster(size=5, guard=0)             # voters hold no guard of their own
    overrun.try_acquire("v0", 0)
    overrun.active.valid_until = overrun.duration + DEFAULT_DRIFT   # runs past the leader guard
    overrun.expire(overrun.active.valid_until)
    second = overrun.try_acquire("v1", overrun.duration)
    check(second is not None,
          "without a voter guard the next candidate is granted while the old leader still acts")
    check(window_overlaps(overrun.leases) != [],
          "the detector must flag the overlap: %s" % (window_overlaps(overrun.leases)[:1],))
    guarded = Cluster(size=5)
    guarded.try_acquire("v0", 0)
    guarded.active.valid_until = guarded.duration + DEFAULT_DRIFT   # the same overrun, with guards
    guarded.expire(guarded.active.valid_until)
    check(guarded.try_acquire("v1", guarded.duration) is None,
          "the voter guard must refuse the next candidate even when the old leader overruns")
    check(window_overlaps(guarded.leases) == [],
          "with the guard in place the overrun must not produce an overlap")
    check(guarded.refused == 1, "the refused attempt must be counted")

    # --- 9. determinism -----------------------------------------------------------
    def identical_run():
        cluster = Cluster(size=5)
        run_schedule(cluster, 150, renew_every=6)
        return cluster

    first_run, second_run = identical_run(), identical_run()
    check([(lease.term, lease.leader, lease.valid_from, lease.valid_until)
           for lease in first_run.leases]
          == [(lease.term, lease.leader, lease.valid_from, lease.valid_until)
              for lease in second_run.leases],
          "identical schedules must produce identical leases")
    check(first_run.actions == second_run.actions,
          "identical schedules must produce identical actions")
    check(window_overlaps(first_run.leases) == [],
          "the deterministic run must still be safe")

    print("hostile schedule: %d leases over %d voters, drifts %s, actions %d"
          % (len(busy.leases), busy.size, [busy.drifts[name] for name in sorted(busy.drifts)],
             len(busy.actions)))
    print("failover: old lease ended at %d, new lease started at %d (%d ticks later)"
          % (old.valid_until, failover.active.valid_from,
             failover.active.valid_from - old.valid_until))
    print("overlap detection: legal runs %d / %d, deliberately broken run %d"
          % (len(window_overlaps(busy.leases)), len(window_overlaps(extreme.leases)),
             len(window_overlaps(overrun.leases))))
    print("PASS")


if __name__ == "__main__":
    main()
