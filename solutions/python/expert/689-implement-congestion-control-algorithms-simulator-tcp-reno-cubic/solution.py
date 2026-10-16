"""Task 689 - Implement congestion control algorithms simulator (TCP Reno/CUBIC).  (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)

Two congestion controllers are driven by the same deterministic event schedule so that their
reactions can be compared directly:

* **Reno** (RFC 5681): slow start doubles cwnd per RTT until ssthresh, congestion avoidance adds one
  MSS per RTT, a triple duplicate ACK halves the window and stays in fast recovery (cwnd = cwnd/2),
  and a timeout drops to one MSS and re-enters slow start.
* **CUBIC** (RFC 8312): after a loss it remembers W_max and follows the cubic function
  W_cubic(t) = C*(t - K)^3 + W_max with C = 0.4, beta = 0.7 and K = cbrt(W_max*(1 - beta)/C), so the
  window starts at W_max*beta, grows quickly while it is far below W_max (concave region), flattens
  exactly at W_max (t = K) and then probes above it (convex region).  Where the cubic curve falls
  below the TCP-friendly estimate W_est(t) = W_max*beta + 3(1 - beta)/(1 + beta) * t/RTT, CUBIC
  follows W_est instead, which is the "friendly region" of the RFC.

Simplifications: one flow, no queue model, no SACK, no fast-convergence adjustment of W_max, no RTT
measurement jitter, loss events are given as an explicit schedule, and time advances one RTT per
round (so a "round" is RTT seconds).  Verified: Reno's exact doubling/linear phases and the classic
0.75*W_max sawtooth average; the Mathis throughput 1.22/sqrt(loss rate); CUBIC's K closed form,
W_cubic(K) == W_max, the concave/convex split and the 0.7 multiplicative decrease; cwnd >= 1 and
ssthresh >= 2 after every single event; CUBIC's higher utilisation on one schedule; and CUBIC's
RTT-fairness against Reno's throughput advantage for the shorter-RTT flow.
"""



MIN_CWND = 1.0                                  # one MSS
MIN_SSTHRESH = 2.0                              # RFC 5681: ssthresh must never fall below two MSS
TRIPLE_DUP, TIMEOUT = "triple_dup_ack", "timeout"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class RenoFlow:
    """RFC 5681: slow start / congestion avoidance / fast recovery / timeout."""

    def __init__(self, rtt=1.0, cwnd=1.0, ssthresh=64.0):
        self.rtt = rtt
        self.cwnd = cwnd
        self.ssthresh = ssthresh
        self.time = 0.0
        self.losses = 0

    def on_ack_round(self):
        """One RTT of ACKs: double below ssthresh, then +1 MSS per RTT."""
        if self.cwnd < self.ssthresh:
            self.cwnd = min(2.0 * self.cwnd, self.ssthresh)
        else:
            self.cwnd += 1.0
        self.time += self.rtt

    def on_loss(self, kind):
        self.losses += 1
        self.ssthresh = max(self.cwnd / 2.0, MIN_SSTHRESH)
        if kind == TRIPLE_DUP:
            self.cwnd = self.ssthresh                       # fast recovery, no slow start
        else:
            self.cwnd = MIN_CWND                            # timeout: restart from one MSS
        self.cwnd = max(self.cwnd, MIN_CWND)


class CubicFlow:
    """RFC 8312: the cubic growth function plus the TCP-friendly region."""

    C = 0.4                                             # MSS / second^3
    BETA = 0.7                                          # multiplicative decrease factor

    def __init__(self, rtt=1.0, cwnd=1.0, ssthresh=64.0):
        self.rtt = rtt
        self.cwnd = cwnd
        self.ssthresh = ssthresh
        self.time = 0.0
        self.losses = 0
        self.w_max = cwnd
        self.k = 0.0
        self.epoch_start = 0.0
        self.in_epoch = False

    def cubic_target(self, t_epoch):
        """W_cubic(t) = C*(t - K)^3 + W_max, in MSS."""
        return self.C * (t_epoch - self.k) ** 3 + self.w_max

    def friendly_target(self, t_epoch):
        """W_est(t) = W_max*beta + 3(1 - beta)/(1 + beta) * t/RTT — the AIMD-friendly floor."""
        return (self.w_max * self.BETA
                + 3.0 * (1.0 - self.BETA) / (1.0 + self.BETA) * (t_epoch / self.rtt))

    def epoch_k(self):
        """K = cbrt(W_max*(1 - beta)/C): the time the cubic curve needs to climb back to W_max."""
        return (self.w_max * (1.0 - self.BETA) / self.C) ** (1.0 / 3.0)

    def start_epoch(self):
        self.epoch_start = self.time
        self.k = self.epoch_k()
        self.in_epoch = True

    def on_ack_round(self):
        self.time += self.rtt
        if self.cwnd < self.ssthresh:                    # slow start, exactly as Reno's
            self.cwnd = min(2.0 * self.cwnd, self.ssthresh)
            return
        if not self.in_epoch:
            self.start_epoch()
        t_epoch = self.time - self.epoch_start
        target = self.cubic_target(t_epoch + self.rtt)   # one RTT ahead, as the RFC evaluates per ACK
        friendly = self.friendly_target(t_epoch)
        if target < friendly:
            self.cwnd = friendly                         # TCP-friendly region
        else:
            self.cwnd += (target - self.cwnd) / max(self.cwnd, MIN_CWND)
        self.cwnd = max(self.cwnd, MIN_CWND)

    def on_loss(self, kind):
        self.losses += 1
        self.w_max = self.cwnd
        self.ssthresh = max(self.cwnd * self.BETA, MIN_SSTHRESH)
        if kind == TRIPLE_DUP:
            self.cwnd = self.ssthresh
            self.start_epoch()
        else:
            self.cwnd = MIN_CWND                         # timeout: slow start again
        self.cwnd = max(self.cwnd, MIN_CWND)


def simulate(flow, rounds, loss_rounds, kind=TRIPLE_DUP):
    """Drive a flow for `rounds` RTTs; a loss event replaces the ACK round it is scheduled on."""
    trace = [flow.cwnd]
    for round_number in range(1, rounds + 1):
        if round_number in loss_rounds:
            flow.on_loss(kind)
        else:
            flow.on_ack_round()
        trace.append(flow.cwnd)
    return trace


def simulate_seconds(flow, seconds, loss_times, kind=TRIPLE_DUP):
    """Same driver but budgeted in seconds, so flows with different RTTs can be compared fairly."""
    rounds = int(seconds / flow.rtt)
    trace = [flow.cwnd]
    for round_number in range(1, rounds + 1):
        if round_number * flow.rtt in loss_times:
            flow.on_loss(kind)
        else:
            flow.on_ack_round()
        trace.append(flow.cwnd)
    return trace


def average(trace):
    return sum(trace) / len(trace)


def make_prng(seed):
    """splitmix64: the lossy-link experiment must be identical on every run."""
    state = seed & 0xFFFFFFFFFFFFFFFF

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    return nxt


def mathis_throughput(loss_rate):
    """The classic 1.22/sqrt(p) MSS per RTT figure for AIMD."""
    return 1.22 / (loss_rate ** 0.5)


def invariants(flow):
    return (MIN_CWND <= flow.cwnd, flow.ssthresh >= MIN_SSTHRESH, flow.losses >= 0)


def main():
    reno = RenoFlow(ssthresh=64.0)
    trace = simulate(reno, 7, set())
    check(trace[:7] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], "Reno doubles the window per RTT")
    check(trace[7] == 65.0, "at ssthresh slow start hands over to congestion avoidance")
    check(simulate(RenoFlow(ssthresh=64.0), 9, set())[9] == 67.0, "congestion avoidance adds 1 MSS/RTT")

    slow = RenoFlow(cwnd=1.0, ssthresh=64.0)
    slow.cwnd = 40.0
    slow.on_loss(TRIPLE_DUP)
    check(slow.cwnd == 20.0 and slow.ssthresh == 20.0, "a triple duplicate ACK halves the window")
    check(slow.cwnd >= slow.ssthresh, "fast recovery leaves cwnd at the new ssthresh")
    slow.on_ack_round()
    check(slow.cwnd == 21.0, "and congestion avoidance continues immediately after it")
    timeouted = RenoFlow()
    timeouted.cwnd = 40.0
    timeouted.on_loss(TIMEOUT)
    check(timeouted.cwnd == 1.0 and timeouted.ssthresh == 20.0, "a timeout restarts from one MSS")
    small = RenoFlow()
    small.cwnd = 3.0
    small.on_loss(TIMEOUT)
    check(small.ssthresh == MIN_SSTHRESH, "ssthresh never drops below two MSS")

    sawtooth = RenoFlow(cwnd=1.0, ssthresh=16.0)
    teeth = simulate(sawtooth, 60, {30, 51})
    check(all(value >= MIN_CWND for value in teeth), "cwnd is always at least one MSS")
    check(teeth[29] == 41.0, "the first loss happens at the top of a full ramp")
    check(teeth[30] == 20.5, "the loss halves the window to W_max/2")
    cycle = teeth[30:51]
    check(cycle == sorted(cycle), "cwnd rises monotonically inside a congestion-avoidance cycle")
    check(cycle[0] == 20.5 and cycle[-1] == 40.5, "the cycle runs from W_max/2 back up to W_max")
    check(abs(average(cycle) - 0.75 * 40.5) / 40.5 < 0.02,
          "the classic average of a sawtooth cycle is 0.75*W_max")
    check(teeth[51] == 20.25, "the second loss halves it again")

    prng = make_prng(0xA11CE)
    for loss_rate in (0.004, 0.01, 0.04):
        flow = RenoFlow(cwnd=1.0, ssthresh=64.0)
        trace = [flow.cwnd]
        for _ in range(4000):
            # every packet of the current window may be dropped, so the chance per RTT is p*cwnd
            per_round = min(1.0, loss_rate * flow.cwnd)
            if (prng() % 1000000) / 1000000.0 < per_round:
                flow.on_loss(TRIPLE_DUP)
            else:
                flow.on_ack_round()
            trace.append(flow.cwnd)
            cwnd_ok, ssthresh_ok, _ = invariants(flow)
            check(cwnd_ok and ssthresh_ok, "Reno keeps its window bounds under random losses")
        measured = average(trace)
        predicted = mathis_throughput(loss_rate)
        check(abs(measured - predicted) / predicted < 0.25,
              f"Reno at p={loss_rate}: {measured:.2f} MSS/RTT vs Mathis {predicted:.2f}")

    cubic = CubicFlow(cwnd=1.0, ssthresh=64.0)
    for _ in range(200):
        cubic.on_ack_round()
    high = cubic.cwnd
    cubic.on_loss(TRIPLE_DUP)
    check(abs(cubic.w_max - high) < 1e-9, "CUBIC remembers W_max as the window at the loss")
    check(abs(cubic.cwnd - CubicFlow.BETA * high) < 1e-9, "the multiplicative decrease is 0.7")
    check(abs(cubic.cubic_target(0.0) - CubicFlow.BETA * high) < 1e-9,
          "W_cubic(0) equals the post-loss window W_max*beta")
    check(abs(cubic.k - cubic.epoch_k()) < 1e-12, "K is the closed form cbrt(W_max(1-beta)/C)")
    check(abs(cubic.cubic_target(cubic.k) - high) < 1e-9, "W_cubic(K) is back at W_max exactly")

    step = 0.5
    early = [cubic.cubic_target(t) for t in (cubic.k - 2 * step, cubic.k - step, cubic.k)]
    late = [cubic.cubic_target(t) for t in (cubic.k, cubic.k + step, cubic.k + 2 * step)]
    check(early[1] - early[0] > early[2] - early[1], "the cubic curve is concave below K")
    check(late[1] - late[0] < late[2] - late[1], "and convex above K")
    check(cubic.cubic_target(cubic.k + 1.0) > high, "above K the window probes past W_max")

    cubic2 = CubicFlow(cwnd=1.0, ssthresh=20.0)
    simulate(cubic2, 200, {5, 60})
    check(cubic2.losses == 2, "both scheduled losses were seen")
    check(MIN_CWND <= cubic2.cwnd, "CUBIC never drops below one MSS")
    reno2 = RenoFlow(cwnd=1.0, ssthresh=20.0)
    reno_trace = simulate(reno2, 200, {5, 60})
    cubic_trace = simulate(CubicFlow(cwnd=1.0, ssthresh=20.0), 200, {5, 60})
    check(average(cubic_trace) > average(reno_trace),
          f"CUBIC keeps a fuller pipe: {average(cubic_trace):.2f} vs {average(reno_trace):.2f} MSS")
    check(40.0 < average(reno_trace) < 90.0, "and Reno's average is in the expected band")

    reno_long = RenoFlow(rtt=2.0, cwnd=50.0, ssthresh=1.0)
    reno_short = RenoFlow(rtt=1.0, cwnd=50.0, ssthresh=1.0)
    losses = {float(t) for t in range(20, 200, 20)}
    simulate_seconds(reno_long, 200, losses)
    simulate_seconds(reno_short, 200, losses)
    reno_ratio = reno_long.cwnd / reno_short.cwnd
    cubic_long = CubicFlow(rtt=2.0, cwnd=50.0, ssthresh=1.0)
    cubic_short = CubicFlow(rtt=1.0, cwnd=50.0, ssthresh=1.0)
    simulate_seconds(cubic_long, 200, losses)
    simulate_seconds(cubic_short, 200, losses)
    cubic_ratio = cubic_long.cwnd / cubic_short.cwnd
    check(reno_ratio < 0.6, f"Reno starves the long-RTT flow (ratio {reno_ratio:.2f})")
    check(cubic_ratio > 0.8, f"CUBIC stays RTT-fair (ratio {cubic_ratio:.2f})")
    check(cubic_ratio > reno_ratio, "CUBIC is the fairer of the two")

    empty = simulate(CubicFlow(), 0, set())
    check(len(empty) == 1 and empty[0] == 1.0, "a zero-round run does nothing")
    check(mathis_throughput(0.25) < mathis_throughput(0.0025), "Mathis throughput falls with loss")

    print(f"Reno sawtooth: cycle from {cycle[0]:.1f} to {cycle[-1]:.1f}, average "
          f"{average(cycle):.2f} MSS (0.75*W_max = {0.75 * 40.5:.2f})")
    print(f"CUBIC epoch: W_max {high:.2f} -> {cubic.cwnd:.2f} after the loss, K = {cubic.k:.2f} s")
    print(f"RTT fairness after 200 s: Reno ratio {reno_ratio:.2f}, CUBIC ratio {cubic_ratio:.2f}")
    print("PASS")


if __name__ == "__main__":
    main()
