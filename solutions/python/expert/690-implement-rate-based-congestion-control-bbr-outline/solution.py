"""Task 690 - Implement rate-based congestion control (BBR outline).  (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)

BBR is a *rate-based* controller: instead of filling the bottleneck until a packet is lost, it paces
at an estimate of the bottleneck bandwidth scaled by a state-dependent gain, so the standing queue
stays near zero.  This file implements the outline faithfully at fluid granularity; every mechanism
the RFC describes is present:

  * **estimates** — BtlBw is the maximum delivery rate sampled over the last 10 rounds, RTprop is the
    minimum round trip over a 10 s window, and every decision reads those two numbers.
  * **the state machine** — STARTUP (gain 2.885, exit after three rounds without 25% growth), DRAIN
    (gain 1/2.885, exit once the flight size falls to the BDP), PROBE_BW (the eight-step gain cycle
    1.25, 0.75, then six 1.0 steps), and PROBE_RTT (the window collapses to four packets for 200 ms
    whenever RTprop has not been refreshed for 10 s).
  * **the bottleneck** — a fluid droptail queue: forward up to capacity, buffer the rest, drop the
    overflow, and let a packet's delay be RTprop + queue/capacity, so the queueing delay is directly
    observable.

Simplifications (a scaled model, not a stub): no packet framing, no SACK or retransmission timers —
a dropped packet is resent immediately and only consumes capacity — one flow, no receive-rate model,
and a round trip is taken as RTprop because round counting is coarse in a fluid model.  Verified: the
gain constants; the STARTUP -> DRAIN -> PROBE_BW sequence; convergence of BtlBw and RTprop to the
true bottleneck values; PROBE_RTT collapsing the flight size to four packets and returning every
10 s; exact fluid conservation (offered == forwarded + dropped + queued); and — the point of the
design — BBR's mean round trip staying near RTprop while a loss-based sender on an identical
bottleneck sits deep in the buffer, with both still saturating the link.
"""

from collections import deque, namedtuple

STARTUP, DRAIN, PROBE_BW, PROBE_RTT = "startup", "drain", "probe_bw", "probe_rtt"
STARTUP_GAIN = 2.885                                # BBR's slow-start gain (ln 2 + 1)
DRAIN_GAIN = 1.0 / STARTUP_GAIN
PROBE_BW_GAINS = (1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
CWND_GAIN = 2.0                                     # BBR keeps two BDPs of window by default
MIN_CWND = 4.0                                      # PROBE_RTT's four-packet window
BW_WINDOW_ROUNDS = 10                               # BtlBw = max delivery rate over ten rounds
RT_WINDOW = 10000                                   # RTprop = min round trip over ten seconds
PROBE_RTT_DURATION = 200                            # PROBE_RTT stays for 200 ticks (200 ms)
STARTUP_STALL_ROUNDS = 3                            # exit after three rounds without 25% growth

Sample = namedtuple("Sample", "round rate")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Bottleneck:
    """Fluid droptail link: `capacity` packets per tick, standing queue bounded by `buffer`."""

    def __init__(self, capacity, buffer, rtprop):
        self.capacity = capacity
        self.buffer = buffer
        self.rtprop = rtprop
        self.queue = 0.0
        self.offered_total = 0.0
        self.forwarded_total = 0.0
        self.dropped_total = 0.0
        self.overflow_ticks = 0

    def transmit(self, offered):
        """Fluid FIFO server.  Returns (forwarded now, dropped, round trip an arriving packet sees).

        Data is served in order at the link rate, so what is forwarded *this* tick is the oldest
        data in the queue and its ACK comes back one propagation delay later - the queueing delay is
        already baked into how long that data waited.  A packet arriving now, on the other hand,
        waits behind the whole standing queue, which is the round trip reported to the sender."""
        self.offered_total += offered
        self.queue += offered
        forwarded = min(self.queue, self.capacity)
        self.queue -= forwarded
        dropped = 0.0
        if self.queue > self.buffer:
            dropped = self.queue - self.buffer
            self.queue = self.buffer
            self.overflow_ticks += 1
        self.forwarded_total += forwarded
        self.dropped_total += dropped
        return forwarded, dropped, self.rtprop + self.queue / self.capacity

    def conservation_error(self):
        """Everything offered is either forwarded, dropped or still sitting in the queue."""
        return abs(self.offered_total - (self.forwarded_total + self.dropped_total + self.queue))


class BbrSender:
    """The four-state BBR outline, driven once per tick of `link`."""

    def __init__(self, link):
        self.link = link
        self.state = STARTUP
        self.state_since = 0
        self.cwnd = MIN_CWND
        self.inflight = 0.0
        self.delivered = 0.0
        self.offered = 0.0
        self.dropped = 0.0
        self.bw_samples = []
        self.btl_bw = 0.0
        self.rtt_samples = []
        self.rt_prop = float(link.rtprop)
        self.rtprop_stamp = 0
        self.rtt = float(link.rtprop)
        self.round = 0
        self.next_round_delivered = MIN_CWND
        self.bw_at_round_start = 0.0
        self.stall_rounds = 0
        self.gain_index = 0
        self.pending = {}
        self.state_log = [(0, STARTUP)]
        self.probe_rtt_events = 0
        self.probe_rtt_min_reached = 0
        self.probe_reached_min = False
        self.max_inflight_in_probe = 0.0
        self.max_inflight = 0.0
        self.delivered_log = deque()

    # ---- estimates ---------------------------------------------------------------------------
    def delivery_rate(self):
        """BtlBw: the highest delivery rate sampled within the last ten rounds."""
        oldest = self.round - BW_WINDOW_ROUNDS
        rates = [sample.rate for sample in self.bw_samples if sample.round > oldest]
        return max(rates) if rates else 0.0
    def windowed_rtprop(self, tick):
        """RTprop: the minimum observed round trip inside the ten-second window."""
        cutoff = tick - RT_WINDOW
        self.rtt_samples = [item for item in self.rtt_samples if item[0] >= cutoff]
        return min(item[1] for item in self.rtt_samples) if self.rtt_samples else self.rt_prop

    def pacing_gain(self):
        if self.state == STARTUP:
            return STARTUP_GAIN
        if self.state == DRAIN:
            return DRAIN_GAIN
        if self.state == PROBE_BW:
            return PROBE_BW_GAINS[self.gain_index]
        return 1.0

    def pacing_rate(self):
        """BBR paces at BtlBw * gain; before the first estimate exists it uses the initial window
        over RTprop, exactly as bbr_init_pacing_rate_from_rtt() does."""
        base = max(self.btl_bw, MIN_CWND / max(self.rt_prop, 1e-9))
        return base * self.pacing_gain()

    def flight_size(self):
        """cwnd = cwnd_gain * BDP, with the BDP taken at the *minimum* round trip, not the current."""
        return max(MIN_CWND, max(self.btl_bw, 1e-9) * self.rt_prop * CWND_GAIN)

    # ---- state machine -----------------------------------------------------------------------
    def change_state(self, state, tick):
        self.state = state
        self.state_since = tick
        if state == PROBE_BW:
            self.gain_index = 0
        elif state == PROBE_RTT:
            self.probe_rtt_events += 1
            self.max_inflight_in_probe = 0.0
            self.probe_reached_min = False
        self.state_log.append((tick, state))

    def on_new_round(self, tick):
        self.round += 1
        self.btl_bw = max(self.btl_bw, self.delivery_rate())
        if self.state == STARTUP:
            grew = self.btl_bw >= 1.25 * max(self.bw_at_round_start, 1e-9)
            self.stall_rounds = 0 if grew else self.stall_rounds + 1
            if self.stall_rounds >= STARTUP_STALL_ROUNDS:
                self.change_state(DRAIN, tick)
        elif self.state == PROBE_BW:
            self.gain_index = (self.gain_index + 1) % len(PROBE_BW_GAINS)
        self.bw_at_round_start = self.btl_bw

    def update_state(self, tick):
        bdp = max(self.btl_bw, 1e-9) * self.rt_prop
        if self.state == PROBE_RTT:
            if tick - self.state_since >= PROBE_RTT_DURATION:
                self.change_state(PROBE_BW, tick)
                self.rtprop_stamp = tick            # the probe refreshed the estimate
        elif self.state == DRAIN and self.inflight <= bdp:
            self.change_state(PROBE_BW, tick)
        elif tick - self.rtprop_stamp >= RT_WINDOW:
            self.change_state(PROBE_RTT, tick)

    # ---- the tick ----------------------------------------------------------------------------
    def sample_delivery_rate(self, tick):
        """BBR samples the delivery rate over an ACK interval; here one interval is the round trip,
        because acks in a fluid queue arrive in bursts when the queue drains.  A sample is the data
        delivered over the last RTprop ticks divided by that interval."""
        self.delivered_log.append((tick, self.delivered))
        interval = max(1, int(round(self.rt_prop)))
        cutoff = tick - interval - 1
        while self.delivered_log and self.delivered_log[0][0] < cutoff:
            self.delivered_log.popleft()
        if self.delivered_log and tick - self.delivered_log[0][0] >= interval:
            past_tick, past_delivered = self.delivered_log[0]
            span = max(1, tick - past_tick)
            self.bw_samples.append(Sample(self.round, (self.delivered - past_delivered) / span))
            oldest = self.round - BW_WINDOW_ROUNDS
            self.bw_samples = [s for s in self.bw_samples if s.round > oldest]

    def step(self, tick):
        acked = self.pending.pop(tick, 0.0)
        if acked > 0:
            self.delivered += acked
            self.inflight = max(0.0, self.inflight - acked)
        self.sample_delivery_rate(tick)
        if self.delivered >= self.next_round_delivered:
            self.next_round_delivered = self.delivered + self.inflight + 1e-9
            self.on_new_round(tick)
        self.update_state(tick)

        window = MIN_CWND if self.state == PROBE_RTT else self.flight_size()
        allowed = min(self.pacing_rate(), max(0.0, window - self.inflight))
        forwarded, dropped, delay = self.link.transmit(allowed)
        self.offered += allowed
        self.inflight += allowed - dropped         # a dropped packet leaves the network immediately
        self.dropped += dropped
        self.max_inflight = max(self.max_inflight, self.inflight)
        if forwarded > 0:
            key = int(tick + self.link.rtprop)     # the ACK follows the propagation delay only
            self.pending[key] = self.pending.get(key, 0.0) + forwarded

        self.rtt = self.link.rtprop + self.link.queue / self.link.capacity
        self.rtt_samples.append((tick, self.rtt))
        windowed = self.windowed_rtprop(tick)
        if abs(windowed - self.rt_prop) > 1e-12:
            self.rt_prop = windowed
            self.rtprop_stamp = tick
        self.btl_bw = max(self.btl_bw, self.delivery_rate())
        if self.state == PROBE_RTT:
            self.max_inflight_in_probe = max(self.max_inflight_in_probe, self.inflight)
            if not self.probe_reached_min and self.inflight <= MIN_CWND + 1e-6:
                self.probe_reached_min = True
                self.probe_rtt_min_reached += 1
        return allowed


class LossBasedSender:
    """A minimal window-driven AIMD sender on its own bottleneck: it keeps increasing until the
    droptail buffer overflows, which is exactly the standing queue BBR avoids."""

    def __init__(self, link, cwnd=MIN_CWND):
        self.link = link
        self.cwnd = cwnd
        self.inflight = 0.0
        self.pending = {}
        self.max_inflight = 0.0
        self.delays = []

    def step(self, tick):
        acked = self.pending.pop(tick, 0.0)
        if acked > 0:
            self.inflight = max(0.0, self.inflight - acked)
            self.cwnd += acked / max(self.cwnd, 1.0)   # one MSS per RTT in congestion avoidance
        allowed = max(0.0, self.cwnd - self.inflight)
        forwarded, dropped, delay = self.link.transmit(allowed)
        self.inflight += allowed - dropped
        self.max_inflight = max(self.max_inflight, self.inflight)
        if dropped > 0:
            self.cwnd = max(self.cwnd / 2.0, 2.0)       # multiplicative decrease on loss
        if forwarded > 0:
            key = int(tick + self.link.rtprop)
            self.pending[key] = self.pending.get(key, 0.0) + forwarded
        self.delays.append(delay)
        return allowed


def run_bbr(ticks, capacity=10.0, buffer=200.0, rtprop=20):
    link = Bottleneck(capacity, buffer, rtprop)
    sender = BbrSender(link)
    for tick in range(ticks):
        sender.step(tick)
    return link, sender


def run_loss_based(ticks, capacity=10.0, buffer=200.0, rtprop=20):
    link = Bottleneck(capacity, buffer, rtprop)
    sender = LossBasedSender(link)
    for tick in range(ticks):
        sender.step(tick)
    return link, sender


def mean(values):
    return sum(values) / len(values)


def main():
    check(STARTUP_GAIN == 2.885, "STARTUP uses the 2.885 gain")
    check(abs(DRAIN_GAIN - 1.0 / 2.885) < 1e-12, "DRAIN uses the reciprocal of the STARTUP gain")
    check(len(PROBE_BW_GAINS) == 8 and PROBE_BW_GAINS[0] == 1.25 and PROBE_BW_GAINS[1] == 0.75,
          "PROBE_BW cycles the eight documented gains")
    check(abs(mean(PROBE_BW_GAINS) - 1.0) < 1e-12,
          "the PROBE_BW cycle is balanced: the 1.25 probe is paid for by the 0.75 drain")

    probe_link = Bottleneck(10.0, 25.0, 20)
    probe_link.transmit(30.0)
    check(probe_link.queue == 20.0, "the queue holds what the link could not forward this tick")
    check(abs(probe_link.dropped_total) < 1e-9, "and nothing is dropped while the buffer has room")
    probe_link.transmit(25.0)
    check(probe_link.dropped_total == 10.0 and probe_link.queue == 25.0,
          "the overflow above the buffer is dropped")
    check(abs(probe_link.transmit(0.0)[2] - 21.5) < 1e-9,
          "a packet's delay is RTprop plus the standing queue over the capacity")
    check(probe_link.conservation_error() < 1e-9, "fluid conservation holds on the bottleneck")
    check(Bottleneck(10.0, 25.0, 20).transmit(0.0)[2] == 20.0,
          "an empty queue shows the propagation delay alone")

    link, sender = run_bbr(30000)
    states = [state for _, state in sender.state_log]
    check(states[:3] == [STARTUP, DRAIN, PROBE_BW],
          f"the flow runs STARTUP -> DRAIN -> PROBE_BW, saw {states[:3]}")
    check(PROBE_RTT in states, "PROBE_RTT is reached during a thirty-second run")
    check(sender.probe_rtt_events >= 2, f"{sender.probe_rtt_events} PROBE_RTT events in 30 s")
    check(abs(sender.btl_bw - link.capacity) / link.capacity < 0.05,
          f"BtlBw converged to {sender.btl_bw:.2f} against a capacity of {link.capacity}")
    check(abs(sender.rt_prop - link.rtprop) < 1.0,
          f"RTprop converged to {sender.rt_prop:.2f} against {link.rtprop}")
    check(sender.probe_rtt_min_reached >= 2,
          "every PROBE_RTT actually drains the flight size down to four packets")
    probe_windows = []
    for index, (tick, state) in enumerate(sender.state_log):
        if state == PROBE_RTT:
            probe_windows.append(tick)
        elif state == PROBE_BW and sender.state_log[index - 1][1] == PROBE_RTT:
            probe_windows[-1] = tick - probe_windows[-1]
    check(probe_windows and all(window >= PROBE_RTT_DURATION for window in probe_windows),
          f"each PROBE_RTT lasts at least 200 ms: {probe_windows[:4]}")
    check(link.conservation_error() < 1e-6, "the whole run conserves fluid exactly")
    check(link.queue <= link.buffer + 1e-9, "the bottleneck queue never exceeds the buffer")
    check(link.dropped_total < 0.01 * link.forwarded_total,
          f"BBR barely loses anything: {link.dropped_total:.3f} packets dropped")
    check(link.forwarded_total >= 0.90 * link.capacity * 29999,
          f"BBR keeps the link busy: {link.forwarded_total / (29999 * link.capacity):.3f} of capacity")

    reno_link, reno = run_loss_based(30000)
    bbr_mean = mean([sample for _, sample in sender.rtt_samples])
    bbr_queue = bbr_mean - link.rtprop
    reno_queue = mean(reno.delays) - reno_link.rtprop
    check(bbr_mean < link.rtprop * 1.35,
          f"BBR's mean round trip is {bbr_mean:.2f} against a propagation delay of {link.rtprop}")
    check(bbr_queue < 2.0, f"BBR keeps only {bbr_queue:.2f} ticks of standing queue")
    check(reno_queue > 5.0, f"a loss-based sender keeps {reno_queue:.2f} ticks of standing queue")
    check(reno_queue > 8.0 * bbr_queue,
          f"the same bottleneck costs the loss-based sender {reno_queue / max(bbr_queue, 1e-9):.1f}x"
          " BBR's queueing delay")
    check(reno_link.overflow_ticks > 0, "the loss-based sender really overflows the buffer")
    check(reno_link.forwarded_total >= 0.85 * reno_link.capacity * 29999,
          "it also saturates the link, so the difference is delay rather than throughput")
    check(sender.max_inflight <= sender.flight_size() * 1.05 + MIN_CWND,
          "the flight size never runs away from the cwnd")

    once_link, once = run_bbr(3000)
    twice_link, twice = run_bbr(3000)
    check((once_link.offered_total, once.state_log, once.btl_bw, once.probe_rtt_events,
           once.probe_rtt_min_reached)
          == (twice_link.offered_total, twice.state_log, twice.btl_bw, twice.probe_rtt_events,
              twice.probe_rtt_min_reached),
          "the simulation is deterministic")
    check(once.state in (PROBE_BW, PROBE_RTT), "a three-second run has already left STARTUP")
    check(run_bbr(50)[1].state_log[0] == (0, STARTUP), "a flow always starts in STARTUP")

    print(f"BBR: BtlBw {sender.btl_bw:.2f} pkt/tick, RTprop {sender.rt_prop:.2f} ticks, "
          f"{sender.probe_rtt_events} PROBE_RTT events, mean RTT {bbr_mean:.2f}")
    print(f"loss-based sender: {reno_queue:.2f} ticks of queueing delay "
          f"(mean RTT {reno_queue + reno_link.rtprop:.2f}) on the same bottleneck")
    print(f"link utilisation: BBR {link.forwarded_total / (29999 * link.capacity):.3f}, "
          f"loss-based {reno_link.forwarded_total / (29999 * reno_link.capacity):.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
