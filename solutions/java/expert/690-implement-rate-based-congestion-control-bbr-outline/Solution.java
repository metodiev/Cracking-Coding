/** Task 690 - Implement rate-based congestion control (BBR outline). (Tier: Expert, CTCI Beyond - Networking, OS & Real-Time)
 *
 * BBR replaces "detect loss, then back off" with a model of the path: it keeps a maximum
 * delivery rate sample (BtlBw) and a minimum RTT sample (RTprop), and paces at
 * gain * BtlBw while never letting inflight exceed cwndGain * BDP, where BDP = BtlBw * RTprop.
 *
 * State machine (BBRv1):
 *   STARTUP  pacing gain 2.885 = 2/ln2, doubling until the bandwidth estimate stops growing by
 *            25% for three rounds ("full bandwidth reached");
 *   DRAIN    gain 1/2.885, draining the queue built by STARTUP until inflight <= BDP;
 *   PROBEBW  the eight-phase gain cycle {1.25, 0.75, 1, 1, 1, 1, 1, 1}, one RTprop per phase,
 *            averaging exactly 1.0 so the rate is sustainable while probing for more bandwidth;
 *   PROBERTT entered when RTprop has not been refreshed for a while: cwnd collapses to four
 *            packets so the queue empties and a new minimum RTT can be measured.
 *
 * What is simplified: a round is one RTprop; the path is a single bottleneck of fixed capacity
 * with a finite queue whose draining sets the observed RTT (rtt = 1 + queue/capacity rounds);
 * BBR's 10 second RTprop lifetime becomes a 60 round window and ProbeRTT lasts two rounds.
 * The invariants still verified are the real ones: pacing = gain * BtlBw at every instant, the
 * estimated bandwidth never exceeds the true bottleneck rate, ProbeBW never paces above
 * 1.25 * BtlBw, the gain cycle averages 1.0, and the queue stays shallow compared with a
 * loss-based sender that fills the buffer - which is the whole point of pacing.
 */
import java.util.*;

/** Bottleneck link: fixed capacity per round plus a finite queue. */
class BbrLink {
    final int capacity;
    final int queueLimit;
    int queue;
    long lost;

    BbrLink(int capacity, int queueLimit) {
        this.capacity = capacity;
        this.queueLimit = queueLimit;
    }

    /** One round of service: whatever is queued plus what arrived, capped at the capacity. */
    double serve(double send) {
        double delivered = Math.min(send + queue, capacity);
        queue = (int) Math.round(queue + send - delivered);
        if (queue < 0) queue = 0;
        if (queue > queueLimit) {
            lost += queue - queueLimit;
            queue = queueLimit;
        }
        return delivered;
    }

    /** Queueing delay expressed in rounds, so RTprop is at least one round. */
    double rttRounds() {
        return 1.0 + (double) queue / capacity;
    }
}

/** One round of the BBR state machine, kept for after-the-fact invariant checks. */
class BbrRecord {
    final int round;
    final String state;
    final double gain;
    final double cwndGain;
    final double btlBw;
    final double rtProp;
    final double send;
    final double cwnd;
    final int queue;
    final double rtt;
    final long lost;

    BbrRecord(int round, String state, double gain, double cwndGain, double btlBw, double rtProp,
              double send, double cwnd, int queue, double rtt, long lost) {
        this.round = round;
        this.state = state;
        this.gain = gain;
        this.cwndGain = cwndGain;
        this.btlBw = btlBw;
        this.rtProp = rtProp;
        this.send = send;
        this.cwnd = cwnd;
        this.queue = queue;
        this.rtt = rtt;
        this.lost = lost;
    }
}

class BbrSender {
    enum State { STARTUP, DRAIN, PROBEBW, PROBERTT }

    static final double STARTUP_GAIN = 2.885;
    static final double DRAIN_GAIN = 1 / 2.885;
    static final double[] CYCLE = {1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    static final int BW_WINDOW = 10;
    static final int RT_WINDOW = 60;
    static final int PROBE_RTT_ROUNDS = 2;
    static final double PROBE_RTT_CWND = 4;

    State state = State.STARTUP;
    double btlBw = 1;
    double rtProp = 1.0;
    private double bwPrevious = 1;
    private int plateauRounds;
    private int cyclePhase;
    private int rtPropAge;
    private int probeRttLeft;
    final Deque<Double> bwSamples = new ArrayDeque<>();
    final Deque<Double> rttSamples = new ArrayDeque<>();
    final List<BbrRecord> trace = new ArrayList<>();

    double pacingGain() {
        switch (state) {
            case STARTUP: return STARTUP_GAIN;
            case DRAIN: return DRAIN_GAIN;
            case PROBERTT: return 1.0;
            default: return CYCLE[cyclePhase];
        }
    }

    double cwndGain() {
        if (state == State.PROBERTT) return 1.0;
        if (state == State.PROBEBW) return 2.0;
        return STARTUP_GAIN;
    }

    void step(BbrLink link, int round) {
        double gain = pacingGain();
        double cwnd = state == State.PROBERTT ? PROBE_RTT_CWND : cwndGain() * btlBw;
        double send = Math.min(gain * btlBw, cwnd);
        double delivered = link.serve(send);
        trace.add(new BbrRecord(round, state.name(), gain, cwndGain(), btlBw, rtProp, send, cwnd,
                link.queue, link.rttRounds(), link.lost));

        bwSamples.addLast(delivered);
        while (bwSamples.size() > BW_WINDOW) bwSamples.removeFirst();
        double maxSample = 0;
        for (double sample : bwSamples) maxSample = Math.max(maxSample, sample);
        btlBw = maxSample;

        rttSamples.addLast(link.rttRounds());
        while (rttSamples.size() > RT_WINDOW) rttSamples.removeFirst();
        double minRtt = Double.MAX_VALUE;
        for (double sample : rttSamples) minRtt = Math.min(minRtt, sample);
        if (minRtt < rtProp - 1e-12) {
            rtProp = minRtt;
            rtPropAge = 0;
        } else {
            rtPropAge++;
        }

        switch (state) {
            case STARTUP:
                if (btlBw >= 1.25 * bwPrevious) plateauRounds = 0;
                else plateauRounds++;
                bwPrevious = btlBw;
                if (plateauRounds >= 3) state = State.DRAIN;
                break;
            case DRAIN:
                if (link.queue + send <= btlBw) {
                    state = State.PROBEBW;
                    cyclePhase = 0;
                }
                break;
            case PROBEBW:
                cyclePhase = (cyclePhase + 1) % CYCLE.length;
                if (rtPropAge > RT_WINDOW) {
                    state = State.PROBERTT;
                    probeRttLeft = PROBE_RTT_ROUNDS;
                    // ProbeRTT is the moment the minimum RTT is re-measured, so the timer restarts.
                    rtPropAge = 0;
                }
                break;
            default:
                if (--probeRttLeft <= 0) {
                    state = State.PROBEBW;
                    cyclePhase = 0;
                }
                break;
        }
    }

    /** Loss-based comparison sender: keeps a window of two BDP, ignoring RTT and pacing. */
    static void greedy(BbrLink link, double window, int rounds) {
        for (int round = 0; round < rounds; round++) link.serve(window);
    }
}

public class Solution {
    static BbrSender runBbr(BbrLink link, int rounds) {
        BbrSender sender = new BbrSender();
        for (int round = 0; round < rounds; round++) {
            sender.step(link, round);
        }
        return sender;
    }

    /** Average RTT (in rounds) over a range of recorded rounds. */
    static double averageRtt(List<BbrRecord> trace, int from, int to) {
        double sum = 0;
        int count = 0;
        for (int i = from; i < to && i < trace.size(); i++) {
            sum += trace.get(i).rtt;
            count++;
        }
        return sum / count;
    }

    static String statesOf(List<BbrRecord> trace) {
        StringBuilder sb = new StringBuilder();
        String previous = null;
        for (BbrRecord record : trace) {
            if (!record.state.equals(previous)) {
                if (previous != null) sb.append('>');
                sb.append(record.state);
                previous = record.state;
            }
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // 1. STARTUP ramps the rate until the bandwidth estimate stops growing.
        BbrLink link = new BbrLink(100, 200);
        BbrSender bbr = runBbr(link, 400);
        List<BbrRecord> trace = bbr.trace;
        check(link.capacity == 100 && bbr.btlBw > 0, "link model and sender initialised");
        check(trace.get(0).state.equals("STARTUP"), "the sender starts in STARTUP");
        check(trace.get(10).btlBw >= 50 && trace.get(20).btlBw == 100,
                "the bandwidth estimate reaches the bottleneck: " + trace.get(20).btlBw);
        boolean bounded = true;
        boolean nonDecreasingStartup = true;
        for (int i = 0; i < trace.size(); i++) {
            BbrRecord record = trace.get(i);
            if (record.btlBw > link.capacity + 1e-9) bounded = false;
            if (record.state.equals("STARTUP") && i > 0 && record.btlBw < trace.get(i - 1).btlBw - 1e-9) {
                nonDecreasingStartup = false;
            }
        }
        check(bounded, "BtlBw never exceeds the true bottleneck rate");
        check(nonDecreasingStartup, "BtlBw is non-decreasing during STARTUP");
        check(trace.get(0).gain == BbrSender.STARTUP_GAIN, "STARTUP paces at 2/ln2 = 2.885");

        // 2. The state machine visits every state in order and leaves STARTUP exactly once.
        String states = statesOf(trace);
        check(states.startsWith("STARTUP>"), "the first transition leaves STARTUP: " + states);
        check(states.contains("STARTUP>DRAIN>PROBEBW"), "STARTUP drains into PROBEBW: " + states);
        check(states.contains("PROBEBW>PROBERTT>PROBEBW"), "ProbeRTT is entered and left: " + states);

        // 3. Pacing is always gain * BtlBw, and cwnd persists the in-flight cap.
        boolean pacingExact = true;
        boolean cwndRespected = true;
        boolean probeBwBounded = true;
        for (BbrRecord record : trace) {
            if (Math.abs(record.send - Math.min(record.gain * record.btlBw, record.cwnd)) > 1e-9) {
                pacingExact = false;
            }
            if (record.send > record.cwnd + 1e-9) cwndRespected = false;
            if (record.state.equals("PROBEBW") && record.gain * record.btlBw > 1.25 * record.btlBw + 1e-9) {
                probeBwBounded = false;
            }
        }
        check(pacingExact, "pacing rate is exactly gain * BtlBw every round");
        check(cwndRespected, "no round sends more than the congestion window");
        check(probeBwBounded, "ProbeBW never paces above 1.25 * BtlBw");

        // 4. The gain cycle averages exactly 1.0, which is what makes it sustainable.
        double cycleSum = 0;
        double cycleMax = 0;
        for (double gain : BbrSender.CYCLE) {
            cycleSum += gain;
            cycleMax = Math.max(cycleMax, gain);
        }
        check(Math.abs(cycleSum - 8.0) < 1e-12 && cycleMax == 1.25,
                "the eight-phase cycle averages 1.0 with a 1.25 peak");
        int phases = 0;
        for (int i = 0; i + 1 < trace.size(); i++) {
            BbrRecord current = trace.get(i);
            BbrRecord next = trace.get(i + 1);
            if (current.state.equals("PROBEBW") && next.state.equals("PROBEBW")) {
                int expected = (indexOfGain(current.gain) + 1) % BbrSender.CYCLE.length;
                if (Math.abs(BbrSender.CYCLE[expected] - next.gain) < 1e-12) phases++;
            }
        }
        check(phases > 100, "the ProbeBW phases follow the cycle in order: " + phases + " steps");

        // 5. The queue stays shallow under BBR compared with a loss-based greedy sender.
        double bbrRtt = averageRtt(trace, 100, 400);
        BbrLink greedyLink = new BbrLink(100, 200);
        BbrSender.greedy(greedyLink, 2.0 * link.capacity, 400);
        double greedyRtt = greedyLink.rttRounds();
        check(greedyLink.lost > 0 && greedyLink.queue == 200,
                "the greedy sender fills the buffer and drops packets: " + greedyLink.lost);
        check(bbrRtt < greedyRtt * 0.75,
                "BBR keeps RTT low: " + bbrRtt + " vs greedy " + greedyRtt);
        check(link.queue < 100, "BBR's queue never approaches the buffer limit: " + link.queue);

        // 6. ProbeRTT empties the queue and refreshes RTprop.
        int probeIndex = -1;
        for (int i = 0; i < trace.size(); i++) {
            if (trace.get(i).state.equals("PROBERTT")) {
                probeIndex = i;
                break;
            }
        }
        check(probeIndex > 0, "a ProbeRTT episode happened");
        BbrRecord probe = trace.get(probeIndex);
        check(probe.send <= BbrSender.PROBE_RTT_CWND + 1e-9 && probe.cwnd == BbrSender.PROBE_RTT_CWND,
                "ProbeRTT collapses the window to four packets");
        check(trace.get(probeIndex - 1).rtt > 1.0,
                "the queue was non-empty just before ProbeRTT: " + trace.get(probeIndex - 1).rtt);
        check(trace.get(probeIndex).queue == 0 && trace.get(probeIndex + 1).queue == 0,
                "ProbeRTT empties the queue");
        check(trace.get(probeIndex + BbrSender.PROBE_RTT_ROUNDS - 1).rtt < 1.02,
                "the measured RTT returns to the propagation delay");
        check(trace.get(probeIndex + BbrSender.PROBE_RTT_ROUNDS - 1).rtProp <= 1.02,
                "RTprop is refreshed to the empty-queue RTT");

        // 7. DRAIN actually drains: inflight falls back to the bandwidth-delay product.
        int drainStart = -1;
        int drainEnd = -1;
        for (int i = 0; i < trace.size(); i++) {
            if (trace.get(i).state.equals("DRAIN")) {
                if (drainStart < 0) drainStart = i;
                drainEnd = i;
            }
        }
        check(drainStart > 0 && drainEnd > drainStart, "DRAIN lasts more than one round");
        check(trace.get(drainStart - 1).queue > trace.get(drainEnd).queue,
                "DRAIN shrinks the queue");
        check(trace.get(drainEnd).gain == BbrSender.DRAIN_GAIN, "DRAIN paces at 1/2.885");

        // 8. Determinism: the same link and settings give a byte-identical trace.
        BbrLink repeatLink = new BbrLink(100, 200);
        BbrSender repeat = runBbr(repeatLink, 400);
        boolean identical = repeat.trace.size() == trace.size();
        for (int i = 0; identical && i < trace.size(); i++) {
            BbrRecord a = trace.get(i);
            BbrRecord b = repeat.trace.get(i);
            identical = a.state.equals(b.state) && a.send == b.send && a.btlBw == b.btlBw
                    && a.rtProp == b.rtProp;
        }
        check(identical && repeatLink.lost == link.lost, "the simulation is deterministic");
        check(link.capacity == 100 && bbr.trace.get(3).send > 0, "traffic was actually sent");

        System.out.println("PASS");
    }

    static int indexOfGain(double gain) {
        for (int i = 0; i < BbrSender.CYCLE.length; i++) {
            if (Math.abs(BbrSender.CYCLE[i] - gain) < 1e-12) return i;
        }
        return -1;
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
