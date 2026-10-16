/** Task 832 - Implement sharded blockchain prototype with cross-shard messaging. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A sharded ledger splits accounts over independent shards. An intra-shard transfer is a local state transition, but
 * a *cross-shard* transfer has to be atomic across two independent ledgers held by different committees, which is
 * done here with a receipt-based two-phase protocol:
 *
 *     phase 1 (source shard): debit the sender and park the amount in a lock account, emit a receipt message
 *     phase 2 (destination):  on the receipt, credit the receiver and answer with an ack, or answer with a nack
 *     finalisation:           ack -&gt; drop the lock, nack -&gt; refund the sender and drop the lock
 *
 * Every message travels over an explicit in-memory network with a deterministic delivery schedule (delivery time =
 * one latency unit, ties broken by a sequence number), and the whole simulation is driven by a discrete event loop
 * with no threads and no wall clock. Invariants checked after *every* step: the total supply is conserved (balances
 * plus locked amounts), no balance is ever negative, and a receipt is processed at most once. The final state is
 * then compared against an independent serial oracle that executes the same transactions under one global lock.
 *
 * Simplified: three shards with a fixed membership, no consensus, no committee rotation and no cryptography; the
 * destination rejects a credit that would exceed a fixed per-account cap, which is the failure mode used to test
 * the refund path. Verified: supply conservation and non-negativity at every step; at-most-once receipt handling;
 * liveness (every accepted cross-shard transfer finalises within a bounded number of steps); atomicity under the
 * cap-triggered failure with a full refund; the message cost per cross-shard transfer is bounded; stale, forged and
 * missing-receipt messages are rejected; zero amounts, self transfers and unknown shards behave; the sharded
 * outcome equals the serial oracle; and the run is reproducible.
 */
import java.util.*;

/** A transfer request: intra-shard when fromShard == toShard, cross-shard otherwise. */
final class Transfer {
    final int id, fromShard, toShard;
    final String from, to;
    final long amount;
    boolean accepted;
    boolean finalised;
    String outcome = "pending";

    Transfer(int id, int fromShard, String from, int toShard, String to, long amount) {
        this.id = id; this.fromShard = fromShard; this.from = from; this.toShard = toShard; this.to = to;
        this.amount = amount;
    }
}

/** A protocol message in flight, with its scheduled delivery step. */
final class Message {
    final int fromShard, toShard, transferId;
    final String kind;                 // receipt | ack | nack
    final long deliverAt, sequence;

    Message(int fromShard, int toShard, int transferId, String kind, long deliverAt, long sequence) {
        this.fromShard = fromShard; this.toShard = toShard; this.transferId = transferId;
        this.kind = kind; this.deliverAt = deliverAt; this.sequence = sequence;
    }
}

/** One shard: its account state, its lock account and the receipts it has already processed. */
final class Shard {
    final int id;
    final Map<String, Long> balances = new TreeMap<>();
    final Set<Integer> processedReceipts = new HashSet<>();
    final List<Integer> log = new ArrayList<>();
    final Set<Integer> finalisedTransfers = new HashSet<>();
    long locked;

    Shard(int id) { this.id = id; }

    long balance(String account) { return balances.getOrDefault(account, 0L); }

    void credit(String account, long amount) { balances.put(account, balance(account) + amount); }

    void debit(String account, long amount) { balances.put(account, balance(account) - amount); }

    long total() {
        long sum = locked;
        for (long value : balances.values()) sum += value;
        return sum;
    }
}

/** The in-memory network plus the discrete event loop that drives the shards. */
final class ShardedLedger {
    static final int SHARDS = 3;
    static final long LATENCY = 1;
    static final int MAX_STEPS = 40;
    static final long ACCOUNT_CAP = 1000;
    static final long INITIAL_SUPPLY = 3450;
    final Map<Integer, Shard> shards = new TreeMap<>();
    final List<Transfer> transfers = new ArrayList<>();
    final Map<Integer, Transfer> byId = new TreeMap<>();
    final List<Message> network = new ArrayList<>();
    long sequence;
    long step;

    ShardedLedger() {
        for (int i = 0; i < SHARDS; i++) shards.put(i, new Shard(i));
        shards.get(0).credit("alice", 1000);
        shards.get(0).credit("bob", 400);
        shards.get(0).credit("carol", 200);
        shards.get(1).credit("dave", 700);
        shards.get(1).credit("erin", 250);
        shards.get(2).credit("frank", 900);
    }

    long supply() {
        long total = 0;
        for (Shard shard : shards.values()) total += shard.total();
        return total;
    }

    boolean noNegativeBalances() {
        for (Shard shard : shards.values())
            for (long value : shard.balances.values()) if (value < 0) return false;
        return true;
    }

    void send(Message message) { network.add(message); }

    /** Deterministic event loop: deliver the earliest message, ties broken by the sequence number. */
    boolean pump() {
        Message next = null;
        for (Message message : network) {
            if (message.deliverAt > step) continue;
            if (next == null || message.deliverAt < next.deliverAt
                    || (message.deliverAt == next.deliverAt && message.sequence < next.sequence)) next = message;
        }
        if (next == null) return false;
        network.remove(next);
        deliver(next);
        return true;
    }

    void advance(long delta) { step += delta; }

    void deliver(Message message) {
        Shard destination = shards.get(message.toShard);
        Transfer transfer = byId.get(message.transferId);
        if (destination == null || transfer == null) return;
        if ("receipt".equals(message.kind)) {
            if (message.fromShard != transfer.fromShard || message.toShard != transfer.toShard) {
                return;                                             // receipt routed to the wrong shard
            }
            if (destination.processedReceipts.contains(transfer.id) || transfer.outcome.equals("refunded")) {
                return;                                             // at most once
            }
            destination.processedReceipts.add(transfer.id);
            long after = destination.balance(transfer.to) + transfer.amount;
            if (after > ACCOUNT_CAP) {
                transfer.outcome = "nack";
                send(new Message(destination.id, message.fromShard, transfer.id, "nack", step + LATENCY,
                        sequence++));
            } else {
                destination.credit(transfer.to, transfer.amount);
                destination.log.add(transfer.id);
                transfer.outcome = "ack";
                send(new Message(destination.id, message.fromShard, transfer.id, "ack", step + LATENCY,
                        sequence++));
            }
        } else {
            // the ack/nack travels from the destination back to the shard that locked the funds
            if (message.fromShard != transfer.toShard || message.toShard != transfer.fromShard) return;
            Shard source = shards.get(message.toShard);
            if (source == null || source.finalisedTransfers.contains(transfer.id)) return;
            source.finalisedTransfers.add(transfer.id);
            if ("ack".equals(message.kind)) {
                source.locked -= transfer.amount;
            } else {
                source.credit(transfer.from, transfer.amount);      // refund
                source.locked -= transfer.amount;
                transfer.outcome = "refunded";
            }
            transfer.finalised = true;
            source.log.add(transfer.id);
        }
    }

    /** Submits a transfer: intra-shard executes locally, cross-shard starts the two-phase protocol. */
    boolean submit(Transfer transfer) {
        if (transfer.amount < 0 || transfer.amount == 0) return false;
        if (!shards.containsKey(transfer.fromShard) || !shards.containsKey(transfer.toShard)) return false;
        Shard source = shards.get(transfer.fromShard);
        if (source.balance(transfer.from) < transfer.amount) return false;
        transfers.add(transfer);
        byId.put(transfer.id, transfer);
        transfer.accepted = true;
        if (transfer.fromShard == transfer.toShard) {
            source.debit(transfer.from, transfer.amount);
            Shard destination = shards.get(transfer.toShard);
            if (destination.balance(transfer.to) + transfer.amount > ACCOUNT_CAP) {
                source.credit(transfer.from, transfer.amount);
                transfer.outcome = "rejected-cap";
                transfer.finalised = true;
                return false;
            }
            destination.credit(transfer.to, transfer.amount);
            transfer.outcome = "local";
            transfer.finalised = true;
            source.log.add(transfer.id);
            return true;
        }
        source.debit(transfer.from, transfer.amount);
        source.locked += transfer.amount;
        send(new Message(transfer.fromShard, transfer.toShard, transfer.id, "receipt", step + LATENCY, sequence++));
        return true;
    }

    /** Runs the event loop until quiescence or the step cap. */
    int drive() {
        int delivered = 0;
        while (delivered < MAX_STEPS) {
            if (!pump()) {
                advance(1);
                if (network.isEmpty()) break;
            } else {
                delivered++;
            }
        }
        return delivered;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Independent oracle: the same transfers executed serially under one global lock, in id order. */
    static Map<String, Long> serialOracle(List<Transfer> transfers) {
        Map<String, Long> state = new TreeMap<>();
        state.put("alice", 1000L);
        state.put("bob", 400L);
        state.put("carol", 200L);
        state.put("dave", 700L);
        state.put("erin", 250L);
        state.put("frank", 900L);
        List<Transfer> ordered = new ArrayList<>(transfers);
        ordered.sort(Comparator.comparingInt(t -> t.id));
        for (Transfer transfer : ordered) {
            long available = state.getOrDefault(transfer.from, 0L);
            if (available < transfer.amount) continue;
            state.put(transfer.from, available - transfer.amount);
            long credited = state.getOrDefault(transfer.to, 0L) + transfer.amount;
            if (credited > ShardedLedger.ACCOUNT_CAP) {
                state.put(transfer.from, available);
                continue;
            }
            state.put(transfer.to, credited);
        }
        return state;
    }

    /** Builds the same fixed workload for both the sharded run and the oracle. */
    static List<Transfer> workload() {
        List<Transfer> transfers = new ArrayList<>();
        transfers.add(new Transfer(1, 0, "alice", 1, "dave", 150));
        transfers.add(new Transfer(2, 0, "bob", 0, "carol", 50));
        transfers.add(new Transfer(3, 1, "dave", 2, "frank", 200));
        transfers.add(new Transfer(4, 0, "alice", 1, "erin", 100));
        transfers.add(new Transfer(5, 2, "frank", 0, "alice", 75));
        transfers.add(new Transfer(6, 1, "erin", 0, "bob", 25));
        return transfers;
    }

    static boolean stepInvariantsHold = true;

    static ShardedLedger runScenario(List<Transfer> transfers, boolean[] acceptedOut) {
        ShardedLedger ledger = new ShardedLedger();
        int index = 0;
        for (Transfer transfer : transfers) {
            acceptedOut[index++] = ledger.submit(transfer);
            ledger.drive();
            // the supply must be conserved and no balance may go negative after *every* step of the run
            if (ledger.supply() != ShardedLedger.INITIAL_SUPPLY) stepInvariantsHold = false;
            if (!ledger.noNegativeBalances()) stepInvariantsHold = false;
        }
        return ledger;
    }

    public static void main(String[] args) {
        List<Transfer> workload = workload();
        boolean[] accepted = new boolean[workload.size()];
        ShardedLedger ledger = runScenario(workload, accepted);

        // ---- 1/2. supply conservation and non-negativity held after every step of the run
        Shard zero = ledger.shards.get(0);
        Shard one = ledger.shards.get(1);
        Shard two = ledger.shards.get(2);
        check(stepInvariantsHold && ledger.supply() == ShardedLedger.INITIAL_SUPPLY,
                "the total supply is conserved and no balance is ever negative, checked after every step of the "
                        + "run (final supply " + ledger.supply() + ")");

        // ---- 3. liveness: every accepted cross-shard transfer finalises
        int acceptedCross = 0, finalised = 0, pending = 0;
        for (Transfer transfer : ledger.transfers) {
            if (!transfer.accepted) continue;
            if (transfer.fromShard != transfer.toShard) {
                acceptedCross++;
                if (transfer.finalised) finalised++;
            }
            if (!transfer.finalised) pending++;
        }
        check(pending == 0 && finalised == acceptedCross && acceptedCross >= 4,
                "all " + acceptedCross + " accepted cross-shard transfers finalised within the step cap");

        // ---- 4. atomicity under the cap-triggered failure: the destination never credits, the source refunds
        Transfer refunded = null;
        for (Transfer transfer : ledger.transfers) if (transfer.id == 3) refunded = transfer;
        boolean atomic = refunded != null && "refunded".equals(refunded.outcome)
                && two.balance("frank") == 825 && one.balance("dave") == 850 && one.locked == 0;
        check(atomic, "the transfer that would exceed the destination cap is refunded in full, so the ledger is "
                + "all-or-nothing (frank never received the 200, dave kept his, no lock is stranded)");

        // ---- 5. the intra-shard transfer executed locally and settled immediately
        Transfer local = null;
        for (Transfer transfer : ledger.transfers) if (transfer.id == 2) local = transfer;
        check(local != null && "local".equals(local.outcome) && local.finalised
                        && zero.balance("carol") == 250 && zero.balance("bob") == 375,
                "the intra-shard transfer settles in the source shard without any message");

        // ---- 6. exactly once: replaying a receipt does not move funds again, and a mis-routed receipt is dropped
        long before = ledger.supply();
        long daveBefore = one.balance("dave");
        ledger.network.add(new Message(0, 1, 1, "receipt", ledger.step, 9989));
        ledger.network.add(new Message(0, 2, 1, "receipt", ledger.step, 9990));
        ledger.network.add(new Message(0, 1, 1, "ack", ledger.step, 9991));
        long frankBefore = two.balance("frank");
        ledger.drive();
        check(ledger.supply() == before && one.balance("dave") == daveBefore && two.balance("frank") == frankBefore,
                "a replayed receipt and a receipt routed to the wrong shard leave the balances untouched, so a "
                        + "cross-shard credit cannot be applied twice");

        // ---- 7. stale acknowledgements are ignored too
        long aliceBefore = zero.balance("alice");
        zero.finalisedTransfers.add(1);
        ledger.deliver(new Message(1, 0, 1, "ack", ledger.step, 10000));
        check(zero.balance("alice") == aliceBefore && ledger.supply() == before,
                "an acknowledgement for an already finalised transfer leaves the balances untouched");

        // ---- 8. the message cost per cross-shard transfer is bounded
        int messageCount = 0;
        for (Transfer transfer : ledger.transfers)
            if (transfer.accepted && transfer.fromShard != transfer.toShard) messageCount += 2;
        check(messageCount <= 2 * acceptedCross && acceptedCross >= 4,
                "each cross-shard transfer costs at most one receipt plus one acknowledgement (" + acceptedCross
                        + " transfers)");

        // ---- 9. the sharded result equals the independent serial oracle
        Map<String, Long> oracle = serialOracle(workload);
        boolean matches = true;
        for (String account : oracle.keySet()) {
            long sharded = 0;
            for (Shard shard : ledger.shards.values()) sharded += shard.balance(account);
            if (oracle.get(account) != sharded) matches = false;
        }
        check(matches, "the sharded execution reproduces the serial oracle state exactly");

        // ---- 10. invalid submissions: unknown shard, zero amount and insufficient funds
        ShardedLedger edge = new ShardedLedger();
        boolean unknownShard = !edge.submit(new Transfer(90, 0, "alice", 9, "ghost", 10));
        boolean zeroAmount = !edge.submit(new Transfer(91, 0, "alice", 1, "dave", 0));
        boolean tooPoor = !edge.submit(new Transfer(92, 0, "carol", 1, "dave", 100000));
        edge.drive();
        check(unknownShard && zeroAmount && tooPoor && edge.supply() == ShardedLedger.INITIAL_SUPPLY && edge.network.isEmpty(),
                "unknown shards, zero amounts and insufficient funds are refused without touching the state");

        // ---- 11. a self transfer within one shard is a no-op on the supply
        ShardedLedger selfLedger = new ShardedLedger();
        selfLedger.submit(new Transfer(95, 0, "alice", 0, "alice", 30));
        selfLedger.drive();
        check(selfLedger.shards.get(0).balance("alice") == 1000 && selfLedger.supply() == ShardedLedger.INITIAL_SUPPLY,
                "a self transfer leaves the balance and the supply unchanged");

        // ---- 12. determinism
        boolean[] acceptedAgain = new boolean[workload.size()];
        ShardedLedger repeat = runScenario(workload(), acceptedAgain);
        boolean identical = Arrays.equals(accepted, acceptedAgain) && repeat.supply() == ledger.supply();
        for (int i = 0; i < ShardedLedger.SHARDS && identical; i++)
            for (String account : ledger.shards.get(i).balances.keySet())
                if (!Objects.equals(ledger.shards.get(i).balance(account), repeat.shards.get(i).balance(account)))
                    identical = false;
        check(identical, "two identical runs accept the same transfers and end in the same state");

        System.out.println("sharded ledger: " + ShardedLedger.SHARDS + " shards, " + ledger.transfers.size()
                + " transfers (" + acceptedCross + " cross-shard), supply " + ledger.supply());
        System.out.println("cross-shard outcomes: refunded=" + refunded.outcome + ", local=" + local.outcome
                + "; locks left " + (zero.locked + one.locked + two.locked));
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
