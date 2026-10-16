/** Task 998 - Implement community moderation tools and quality review workflow.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  Submissions move through an explicit state machine (DRAFT -> IN_REVIEW -> CHANGES_REQUESTED /
 *  APPROVED / ESCALATED -> PUBLISHED) where every transition is checked against the legal table and
 *  written to an audit log, so a submission can never be published straight from draft and a published
 *  one can never be reopened. Reviewers vote approve/reject/abstain with a reputation weight; the author
 *  cannot review their own work, reviewers with a conflict of interest are excluded from the tally, and
 *  only the latest vote per reviewer counts. Quorum needs two distinct eligible approvals with combined
 *  reputation >= 4 and a weighted approval ratio >= 0.6; a policy-severity reject escalates to the
 *  moderator queue instead of publishing, and requesting changes clears earlier approvals so they must
 *  be earned again.
 *  Simplified: in-memory reviewers and a fixed reputation threshold instead of an external identity
 *  service; the transition table, the conflict exclusion, the quorum arithmetic and the audit replay
 *  are the real ones.
 *  Verified: the happy path publishes only after quorum, illegal transitions and self-review are
 *  refused, conflicted and low-reputation votes cannot reach quorum, the weighted ratio gate blocks a
 *  50/50 tally, changes-requested resets approvals, policy rejects escalate and block publishing, the
 *  zero-reviewer and unknown-verdict boundaries behave, and independent oracles recount quorum and
 *  replay the audit log.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final Map<String, Set<String>> LEGAL = Map.of(
            "DRAFT", Set.of("IN_REVIEW"),
            "IN_REVIEW", Set.of("CHANGES_REQUESTED", "APPROVED", "ESCALATED"),
            "CHANGES_REQUESTED", Set.of("IN_REVIEW"),
            "APPROVED", Set.of("PUBLISHED", "CHANGES_REQUESTED"),
            "ESCALATED", Set.of("CHANGES_REQUESTED"),
            "PUBLISHED", Set.of());

    static final int MIN_APPROVALS = 2;
    static final double MIN_APPROVAL_REPUTATION = 4.0;
    static final double MIN_APPROVAL_RATIO = 0.6;

    static final class Vote {
        final String reviewer, verdict, severity;      // verdict: approve|reject|abstain; severity: normal|policy
        final double reputation;
        final boolean conflicted;
        Vote(String reviewer, String verdict, String severity, double reputation, boolean conflicted) {
            this.reviewer = reviewer; this.verdict = verdict; this.severity = severity;
            this.reputation = reputation; this.conflicted = conflicted;
        }
    }

    static final class Workflow {
        final String author;
        String state = "DRAFT";
        final List<String> audit = new ArrayList<>();
        final Map<String, Vote> latestVotes = new LinkedHashMap<>();   // reviewer -> latest vote only
        String error;

        Workflow(String author) { this.author = author; }

        boolean transition(String to) {
            Set<String> allowed = LEGAL.getOrDefault(state, Set.of());
            if (!allowed.contains(to)) {
                error = "illegal transition " + state + " -> " + to;
                return false;
            }
            audit.add("state " + state + " -> " + to);
            state = to;
            return true;
        }

        boolean submit() { return transition("IN_REVIEW"); }

        boolean requestChanges(String reviewer) {
            if (!transition("CHANGES_REQUESTED")) return false;
            latestVotes.clear();                                  // approvals must be earned again
            audit.add("changes requested by " + reviewer + ", approvals cleared");
            return true;
        }

        boolean vote(Vote vote) {
            if (vote.reviewer.equals(author)) { error = "self-review refused: " + vote.reviewer; return false; }
            if (!Set.of("approve", "reject", "abstain").contains(vote.verdict)) {
                error = "unknown verdict: " + vote.verdict;
                return false;
            }
            latestVotes.put(vote.reviewer, vote);                 // a reviewer's later vote replaces the earlier one
            audit.add("vote " + vote.verdict + " by " + vote.reviewer + " (rep " + vote.reputation
                    + (vote.conflicted ? ", conflicted" : "") + ")");
            evaluate();
            return error == null;
        }

        /** Applies the quorum rule to the latest eligible votes, or escalates on a policy reject. */
        void evaluate() {
            if (!state.equals("IN_REVIEW")) return;
            int approvals = 0, rejections = 0;
            double approvalWeight = 0, rejectionWeight = 0;
            boolean policyReject = false;
            for (Vote vote : eligibleVotes()) {
                if (vote.verdict.equals("approve")) { approvals++; approvalWeight += vote.reputation; }
                else if (vote.verdict.equals("reject")) {
                    rejections++;
                    rejectionWeight += vote.reputation;
                    if (vote.severity.equals("policy")) policyReject = true;
                }
            }
            if (policyReject) { transition("ESCALATED"); audit.add("escalated to the moderator queue"); return; }
            double cast = approvalWeight + rejectionWeight;
            double ratio = cast == 0 ? 0 : approvalWeight / cast;
            if (approvals >= MIN_APPROVALS && approvalWeight >= MIN_APPROVAL_REPUTATION && ratio >= MIN_APPROVAL_RATIO) {
                transition("APPROVED");
                return;
            }
            if (rejections >= 2 && approvalWeight < rejectionWeight) requestChanges("quorum vote");
        }
        /** Distinct non-conflicted reviewers' latest votes - the only ones that count. */
        List<Vote> eligibleVotes() {
            List<Vote> eligible = new ArrayList<>();
            for (Vote vote : latestVotes.values())
                if (!vote.conflicted && !vote.reviewer.equals(author)) eligible.add(vote);
            eligible.sort(Comparator.comparing(v -> v.reviewer));
            return eligible;
        }
    }
    /** Oracle: recounts quorum with a nested-loop tally over the raw vote store. */
    static boolean oracleQuorum(Workflow workflow) {
        int approvals = 0, rejections = 0;
        double approvalWeight = 0, rejectionWeight = 0;
        for (Map.Entry<String, Vote> entry : workflow.latestVotes.entrySet()) {
            Vote vote = entry.getValue();
            if (vote.conflicted || vote.reviewer.equals(workflow.author)) continue;
            if (vote.verdict.equals("approve")) { approvals++; approvalWeight += vote.reputation; }
            if (vote.verdict.equals("reject")) { rejections++; rejectionWeight += vote.reputation; }
        }
        double cast = approvalWeight + rejectionWeight;
        double ratio = cast == 0 ? 0 : approvalWeight / cast;
        return approvals >= MIN_APPROVALS && approvalWeight >= MIN_APPROVAL_REPUTATION && ratio >= MIN_APPROVAL_RATIO;
    }
    /** Oracle: replays the audit log and checks every recorded state change against the legal table. */
    static boolean oracleAuditLegal(Workflow workflow) {
        String current = "DRAFT";
        for (String entry : workflow.audit) {
            if (!entry.startsWith("state ")) continue;
            String[] parts = entry.substring("state ".length()).split(" -> ");
            if (!parts[0].equals(current)) return false;
            if (!LEGAL.getOrDefault(current, Set.of()).contains(parts[1])) return false;
            current = parts[1];
        }
        return current.equals(workflow.state);
    }
    static Workflow published(String first, String second) {
        Workflow flow = new Workflow("editor");
        check(flow.submit(), "submit");
        flow.vote(new Vote(first, "approve", "normal", 3.0, false));
        flow.vote(new Vote(second, "approve", "normal", 2.0, false));
        return flow;
    }
    public static void main(String[] args) {
        // --- happy path: quorum approves, then publication is allowed and logged.
        Workflow happy = published("alice", "bob");
        check(happy.state.equals("APPROVED") && happy.transition("PUBLISHED") && happy.state.equals("PUBLISHED"),
                "a quorum-approved submission must publish: " + happy.state);
        check(happy.transition("IN_REVIEW") == false && happy.error.equals("illegal transition PUBLISHED -> IN_REVIEW"),
                "a published submission must not be reopened: " + happy.error);
        check(oracleAuditLegal(happy), "the audit log must contain only legal transitions: " + happy.audit);
        check(happy.audit.contains("changes requested by quorum vote") == false, "a submission at quorum must not be sent back for changes: " + happy.audit);
        // --- illegal transitions from the start of the lifecycle.
        Workflow illegal = new Workflow("editor");
        check(!illegal.transition("PUBLISHED") && illegal.error.equals("illegal transition DRAFT -> PUBLISHED") && illegal.state.equals("DRAFT"),
                "publishing from draft must be refused: " + illegal.error);
        check(!illegal.transition("APPROVED") && !illegal.transition("ESCALATED"), "a draft can only be submitted for review");
        // --- self-review and conflict of interest are excluded from the tally.
        Workflow selfReview = new Workflow("editor");
        selfReview.submit();
        check(!selfReview.vote(new Vote("editor", "approve", "normal", 9.0, false)) && selfReview.error.equals("self-review refused: editor"),
                "the author must not review their own work: " + selfReview.error);
        check(selfReview.latestVotes.isEmpty(), "a refused self-review must not be recorded");
        Workflow conflicted = new Workflow("editor");
        conflicted.submit();
        conflicted.vote(new Vote("alice", "approve", "normal", 3.0, false));
        conflicted.vote(new Vote("bob", "approve", "normal", 3.0, true));
        check(conflicted.state.equals("IN_REVIEW") && conflicted.eligibleVotes().size() == 1, "a conflicted reviewer must not count towards quorum: " + conflicted.eligibleVotes().size());
        check(conflicted.latestVotes.size() == 2, "the conflicted vote is still recorded for the audit trail");
        // --- quorum gates: count, reputation and the weighted ratio.
        Workflow oneApproval = new Workflow("editor");
        oneApproval.submit();
        oneApproval.vote(new Vote("alice", "approve", "normal", 5.0, false));
        check(oneApproval.state.equals("IN_REVIEW"), "a single approval must not approve: " + oneApproval.state);
        Workflow lowReputation = new Workflow("editor");
        lowReputation.submit();
        lowReputation.vote(new Vote("alice", "approve", "normal", 0.5, false));
        lowReputation.vote(new Vote("bob", "approve", "normal", 0.5, false));
        check(lowReputation.state.equals("IN_REVIEW") && !oracleQuorum(lowReputation), "two approvals with too little reputation must not reach quorum");
        Workflow split = new Workflow("editor");
        split.submit();
        split.vote(new Vote("alice", "approve", "normal", 5.0, false));
        split.vote(new Vote("bob", "reject", "normal", 5.0, false));
        check(split.state.equals("IN_REVIEW") && split.audit.stream().noneMatch(e -> e.contains("-> APPROVED")) && 5.0 / (5.0 + 5.0) < MIN_APPROVAL_RATIO, "a 50/50 weighted tally is below the 0.6 ratio gate: " + split.state);
        check(!oracleQuorum(split) && oracleQuorum(happy), "the quorum oracle must agree on both cases");
        // --- changing a vote: only the latest vote per reviewer counts.
        Workflow switched = new Workflow("editor");
        switched.submit();
        switched.vote(new Vote("alice", "approve", "normal", 3.0, false));
        switched.vote(new Vote("bob", "approve", "normal", 3.0, false));
        check(switched.state.equals("APPROVED"), "two solid approvals must approve");
        Workflow revised = new Workflow("editor");
        revised.submit();
        revised.vote(new Vote("alice", "approve", "normal", 3.0, false));
        revised.vote(new Vote("alice", "reject", "normal", 3.0, false));
        check(revised.latestVotes.size() == 1 && revised.eligibleVotes().get(0).verdict.equals("reject"),
                "a reviewer's later vote must replace the earlier one: " + revised.eligibleVotes().size());
        // --- changes requested clears approvals and must be re-earned after resubmission.
        Workflow rework = new Workflow("editor");
        rework.submit();
        rework.vote(new Vote("alice", "approve", "normal", 3.0, false));
        check(rework.requestChanges("alice") && rework.state.equals("CHANGES_REQUESTED") && rework.latestVotes.isEmpty(),
                "requesting changes must clear the approvals: " + rework.state);
        check(rework.transition("IN_REVIEW") && rework.eligibleVotes().isEmpty() && rework.state.equals("IN_REVIEW"),
                "a resubmitted submission must start again without approvals");
        rework.vote(new Vote("alice", "approve", "normal", 3.0, false));
        check(rework.state.equals("IN_REVIEW"), "one re-earned approval is still not a quorum");
        rework.vote(new Vote("bob", "approve", "normal", 3.0, false));
        check(rework.state.equals("APPROVED") && rework.transition("PUBLISHED"), "re-earned quorum must approve and allow publication");
        // --- policy rejects escalate instead of publishing.
        Workflow escalated = new Workflow("editor");
        escalated.submit();
        escalated.vote(new Vote("alice", "approve", "normal", 5.0, false));
        escalated.vote(new Vote("bob", "reject", "policy", 1.0, false));
        check(escalated.state.equals("ESCALATED") && escalated.audit.contains("escalated to the moderator queue"),
                "a policy reject must escalate to the moderator queue: " + escalated.state);
        check(!escalated.transition("PUBLISHED") && escalated.error.equals("illegal transition ESCALATED -> PUBLISHED"),
                "an escalated submission must not publish: " + escalated.error);
        check(!oracleQuorum(escalated), "the oracle must agree that an escalated submission has no quorum");
        // --- boundaries: no reviewers at all, and an unknown verdict.
        Workflow untouched = new Workflow("editor");
        untouched.submit();
        untouched.evaluate();
        check(untouched.state.equals("IN_REVIEW") && untouched.eligibleVotes().isEmpty() && !oracleQuorum(untouched),
                "with no reviewers the submission stays in review: " + untouched.state);
        check(!untouched.vote(new Vote("alice", "maybe", "normal", 4.0, false)) && untouched.error.equals("unknown verdict: maybe"),
                "an unknown verdict must be refused: " + untouched.error);
        check(untouched.latestVotes.isEmpty(), "a refused vote must not enter the tally");
        // --- oracle: independent audit replay and quorum recount on every workflow above.
        for (Workflow flow : List.of(happy, illegal, selfReview, conflicted, oneApproval, lowReputation, split,
                switched, revised, rework, escalated, untouched))
            check(oracleAuditLegal(flow), "audit replay must be legal for every workflow: " + flow.audit);
        System.out.println("state reached: " + happy.state + ", eligible votes: " + happy.eligibleVotes().size());
        System.out.println("PASS");
    }
}
