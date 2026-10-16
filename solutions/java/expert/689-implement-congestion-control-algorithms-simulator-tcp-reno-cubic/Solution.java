/** Task 689 - Implement congestion control algorithms simulator (TCP Reno/CUBIC). (Tier: Expert, CTCI Beyond - Networking, OS & Real-Time)
 *
 * Rounds are RTTs. Each round the flow sends floor(cwnd) segments into a bottleneck that can
 * carry `capacity` of them; the surplus is dropped, and the drop is classified by severity:
 *
 *   * a loss that destroys at least half of the window makes the sender time out and collapse
 *     to one segment;
 *   * anything smaller still yields three duplicate ACKs, so it is recovered by fast
 *     retransmit and the window drops to ssthresh instead of one.
 *
 * Reno: slow start doubles the window (capped at ssthresh), congestion avoidance adds one
 * segment per RTT, and a loss sets ssthresh = cwnd/2.
 *
 * CUBIC: after a loss the target wMax is the pre-loss window, the window restarts at
 * beta * wMax (beta = 0.7) and then follows
 *     W(t) = C (t - K)^3 + wMax,   K = cbrt(wMax (1 - beta) / C),  C = 0.4,
 * clipped below by the TCP-friendly linear estimate wMax*beta + [3(1-beta)/(1+beta)] * t so
 * CUBIC is never less aggressive than Reno. CUBIC also applies fast convergence (an identical
 * loss gets a smaller target than the previous one). Before the first loss CUBIC behaves like
 * TCP (slow start, then additive increase), because the cubic function needs a reference wMax.
 *
 * Verified: the exact slow-start / additive-increase sequences, the packing 3/4 W_max average
 * window of an AIMD sawtooth (the classic result, checked to within 5%), timeouts only on
 * heavy losses, CUBIC's convex region and TCP-friendly floor, CUBIC's large goodput advantage
 * on a high-BDP path, the capacity-1 degenerate link, and window/packet accounting.
 */
import java.util.*;

/** Recorded trajectory of one sender. */
class FlowTrace {
    final String algorithm;
    final double[] window;
    final double[] ssthresh;
    final long packetsSent;
    final long lostPackets;
    final int fastRetransmits;
    final int timeouts;
    final int firstLossRound;

    FlowTrace(String algorithm, double[] window, double[] ssthresh, long packetsSent,
              long lostPackets, int fastRetransmits, int timeouts, int firstLossRound) {
        this.algorithm = algorithm;
        this.window = window;
        this.ssthresh = ssthresh;
        this.packetsSent = packetsSent;
        this.lostPackets = lostPackets;
        this.fastRetransmits = fastRetransmits;
        this.timeouts = timeouts;
        this.firstLossRound = firstLossRound;
    }

    long goodput() {
        return packetsSent - lostPackets;
    }

    double averageWindow(int from) {
        double sum = 0;
        int count = 0;
        for (int i = from; i < window.length; i++) {
            sum += window[i];
            count++;
        }
        return sum / count;
    }

    double maxWindow(int from) {
        double best = 0;
        for (int i = from; i < window.length; i++) best = Math.max(best, window[i]);
        return best;
    }
}

/** One long-lived TCP flow against a fixed bottleneck. */
class TcpFlow {
    static final double CUBIC_C = 0.4;
    static final double CUBIC_BETA = 0.7;

    final String algorithm;
    double cwnd = 1;
    double ssthresh;
    double wMax;
    double epoch;
    long packetsSent;
    long lostPackets;
    int fastRetransmits;
    int timeouts;
    int firstLossRound = -1;

    TcpFlow(String algorithm, double initialSsthresh) {
        this.algorithm = algorithm;
        this.ssthresh = initialSsthresh;
    }

    void grow() {
        if (cwnd < ssthresh) {
            // Slow start: one ACK per segment doubles the window, clamped on exiting.
            cwnd = Math.min(cwnd * 2, ssthresh);
            return;
        }
        if (algorithm.equals("cubic")) {
            if (wMax <= 0) {
                cwnd += 1;
                return;
            }
            epoch += 1;
            double k = Math.cbrt(wMax * (1 - CUBIC_BETA) / CUBIC_C);
            double cubic = CUBIC_C * Math.pow(epoch - k, 3) + wMax;
            double friendly = wMax * CUBIC_BETA
                    + (3 * (1 - CUBIC_BETA) / (1 + CUBIC_BETA)) * epoch;
            cwnd = Math.max(cubic, friendly);
        } else {
            cwnd += 1;
        }
    }

    void onLoss(boolean fastRetransmit) {
        double lost = cwnd;
        if (algorithm.equals("cubic")) {
            double target = lost;
            if (wMax > 0 && lost < wMax) target = lost * (1 + CUBIC_BETA) / 2;
            wMax = target;
            ssthresh = Math.max(lost * CUBIC_BETA, 2);
            epoch = 0;
        } else {
            ssthresh = Math.max(lost / 2.0, 2);
        }
        cwnd = fastRetransmit ? ssthresh : 1;
    }

    FlowTrace run(int rounds, int capacity) {
        int[] schedule = new int[rounds];
        Arrays.fill(schedule, capacity);
        return run(schedule);
    }

    /** Runs with a per-round bottleneck capacity, which lets a transient loss open a free path. */
    FlowTrace run(int[] capacityPerRound) {
        int rounds = capacityPerRound.length;
        double[] window = new double[rounds];
        double[] thresholds = new double[rounds];
        for (int round = 0; round < rounds; round++) {
            window[round] = cwnd;
            thresholds[round] = ssthresh;
            int send = (int) Math.floor(cwnd + 1e-9);
            packetsSent += send;
            int excess = Math.max(0, send - capacityPerRound[round]);
            if (excess > 0) {
                lostPackets += excess;
                boolean timeout = 2 * excess >= send;
                if (firstLossRound < 0) firstLossRound = round;
                if (timeout) {
                    timeouts++;
                } else {
                    fastRetransmits++;
                }
                onLoss(!timeout);
            } else {
                grow();
            }
        }
        return new FlowTrace(algorithm, window, thresholds, packetsSent, lostPackets,
                fastRetransmits, timeouts, firstLossRound);
    }
}

public class Solution {
    static double sum(double[] values) {
        double total = 0;
        for (double value : values) total += value;
        return total;
    }

    /** Rounds from `from` until the window first reaches `target`; -1 when it never does. */
    static int roundsToReach(FlowTrace trace, int from, double target) {
        for (int i = from; i < trace.window.length; i++) {
            if (trace.window[i] >= target) return i - from;
        }
        return -1;
    }

    /** Checks cwnd >= 1 and ssthresh >= 2 on every round of a run. */
    static boolean saneEverywhere(FlowTrace trace) {
        for (int i = 0; i < trace.window.length; i++) {
            if (trace.window[i] < 1 || trace.ssthresh[i] < 2) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        // 1. Slow start doubles the window, then congestion avoidance adds one per RTT.
        FlowTrace slow = new TcpFlow("reno", 16).run(12, 1000);
        check(slow.window[0] == 1 && slow.window[1] == 2 && slow.window[2] == 4
                && slow.window[3] == 8 && slow.window[4] == 16, "slow start doubles per RTT");
        check(slow.window[5] == 17 && slow.window[6] == 18 && slow.window[7] == 19,
                "congestion avoidance adds one segment per RTT");
        check(slow.lostPackets == 0 && slow.timeouts == 0 && slow.fastRetransmits == 0,
                "a bottleneck wider than the window never drops a segment");
        check(slow.packetsSent == (long) sum(slow.window), "sent segments equal the window sum");

        // 2. AIMD sawtooth: the average window of a cycle is 3/4 of its peak (classic result).
        FlowTrace saw = new TcpFlow("reno", 60).run(400, 60);
        check(saw.fastRetransmits > 3, "the sawtooth repeats: " + saw.fastRetransmits + " losses");
        check(saw.timeouts == 0, "a one-segment overshoot is always a fast retransmit");
        double peak = saw.maxWindow(20);
        double average = saw.averageWindow(20);
        check(Math.abs(average / peak - 0.75) < 0.05,
                "AIMD average window is 3/4 of the peak: " + average / peak);
        check(peak > 60 && peak < 63, "the loss happens just above the bottleneck: " + peak);
        double firstLossWindow = saw.window[saw.firstLossRound];
        check(Math.abs(saw.ssthresh[saw.firstLossRound + 1] - firstLossWindow / 2) < 1e-9,
                "fast retransmit halves the window into ssthresh");
        check(Math.abs(saw.window[saw.firstLossRound + 1] - firstLossWindow / 2) < 1e-9,
                "the window restarts at the halved value");
        check(saneEverywhere(saw), "window stays at least one segment, ssthresh at least two");
        check(saw.lostPackets * 10 < saw.packetsSent, "loss rate stays below ten percent");

        // 3. The Reno sawtooth after the first loss is exactly +1 per round.
        int flips = 0;
        for (int i = saw.firstLossRound + 1; i + 1 < 40; i++) {
            double delta = saw.window[i + 1] - saw.window[i];
            if (Math.abs(delta - 1) > 1e-9) flips++;
        }
        check(flips == 0, "congestion avoidance slope is exactly one segment per RTT");

        // 4. A heavy loss (most of the window) is a timeout, not a fast retransmit.
        FlowTrace starved = new TcpFlow("reno", 4).run(40, 1);
        check(starved.timeouts > 5 && starved.fastRetransmits == 0,
                "a window far above the capacity times out");
        check(starved.goodput() == 40, "the degenerate link still carries one segment per round");
        check(starved.maxWindow(0) >= 2 && starved.ssthresh[5] == 2 && saneEverywhere(starved),
                "ssthresh is pinned at two on a one-segment link");

        // 5. CUBIC on a high-BDP path: far more goodput and a much faster recovery.
        FlowTrace renoBig = new TcpFlow("reno", 200).run(300, 200);
        FlowTrace cubicBig = new TcpFlow("cubic", 200).run(300, 200);
        check(cubicBig.goodput() > renoBig.goodput(),
                "CUBIC out-delivers Reno: " + cubicBig.goodput() + " vs " + renoBig.goodput());
        double renoTarget = 0.95 * renoBig.window[renoBig.firstLossRound];
        double cubicTarget = 0.95 * cubicBig.window[cubicBig.firstLossRound];
        int renoRecovery = roundsToReach(renoBig, renoBig.firstLossRound + 1, renoTarget);
        int cubicRecovery = roundsToReach(cubicBig, cubicBig.firstLossRound + 1, cubicTarget);
        check(renoRecovery > 50 && cubicRecovery > 0, "both recover: " + renoRecovery + "/" + cubicRecovery);
        check(cubicRecovery * 2 <= renoRecovery,
                "CUBIC rebuilds its window at least twice as fast: " + cubicRecovery + " vs " + renoRecovery);
        check(Math.abs(cubicBig.ssthresh[cubicBig.firstLossRound + 1]
                - 0.7 * cubicBig.window[cubicBig.firstLossRound]) < 1e-9,
                "CUBIC backs off to beta = 0.7 of the window");

        // 6. The cubic curve is convex past wMax and never below the TCP-friendly estimate.
        //    A short congested phase sets wMax, then the link is free so the shape is visible.
        int[] schedule = new int[120];
        Arrays.fill(schedule, 100000);
        for (int i = 0; i < 10; i++) schedule[i] = 100;
        FlowTrace cubic = new TcpFlow("cubic", 100).run(schedule);
        int first = cubic.firstLossRound;
        check(first > 0 && cubic.algorithm.equals("cubic"), "cubic run has a loss epoch");
        double wMax = cubic.window[first];
        int inflection = -1;
        for (int i = first + 1; i < cubic.window.length; i++) {
            if (cubic.window[i] > wMax) {
                inflection = i;
                break;
            }
        }
        check(inflection > 0, "the window passes wMax again");
        boolean convex = true;
        for (int i = inflection; i + 2 < inflection + 8; i++) {
            double d1 = cubic.window[i + 1] - cubic.window[i];
            double d2 = cubic.window[i + 2] - cubic.window[i + 1];
            if (d2 < d1 - 1e-9) convex = false;
        }
        check(convex, "past the inflection the increments grow (cubic shape)");
        check(cubic.window[inflection + 6] - cubic.window[inflection + 5] > 20,
                "the growth is far from linear: "
                        + (cubic.window[inflection + 6] - cubic.window[inflection + 5]));
        boolean friendlyFloor = true;
        for (int t = 2; t <= 9; t++) {
            // window[first + t] is the value after t-1 cubic epochs.
            double floor = wMax * TcpFlow.CUBIC_BETA
                    + (3 * (1 - TcpFlow.CUBIC_BETA) / (1 + TcpFlow.CUBIC_BETA)) * (t - 1);
            if (cubic.window[first + t] < floor - 1e-9) friendlyFloor = false;
        }
        check(friendlyFloor, "CUBIC never drops below the TCP-friendly linear floor");
        check(saneEverywhere(cubic), "cubic invariants hold on every round");

        // 7. Determinism and the shared slow-start behaviour of both algorithms.
        FlowTrace repeat = new TcpFlow("cubic", 200).run(300, 200);
        boolean identical = repeat.window.length == cubicBig.window.length;
        for (int i = 0; identical && i < repeat.window.length; i++) {
            if (repeat.window[i] != cubicBig.window[i]) identical = false;
        }
        check(identical && repeat.packetsSent == cubicBig.packetsSent,
                "the simulation is deterministic");
        check(cubicBig.window[0] == 1 && cubicBig.window[1] == 2 && cubicBig.window[2] == 4,
                "CUBIC shares the initial slow start with Reno");

        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
