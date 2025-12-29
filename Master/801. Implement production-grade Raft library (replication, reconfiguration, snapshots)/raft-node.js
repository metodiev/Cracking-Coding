/**
 * raft-node.js
 * Node class for Raft consensus, state machine, and RPC handlers
 */

class RaftNode {
    constructor({ id, peers }) {
        this.id = id;
        this.peers = peers; // Array of peer node IDs
        this.state = "Follower"; // Follower, Candidate, Leader
        this.currentTerm = 0;
        this.votedFor = null;
        this.log = []; // Array of { term, command }
        this.commitIndex = 0;
        this.lastApplied = 0;

        // Heartbeat / election timeout timers
        this.electionTimeout = null;
        this.heartbeatInterval = null;
    }

    start() {
        console.log(`Node ${this.id} starting as ${this.state}`);
        this.resetElectionTimeout();
    }

    resetElectionTimeout() {
        // TODO: implement randomized election timeout
    }

    handleRequestVoteRPC(rpc) {
        // TODO: implement RequestVote RPC handling
    }

    handleAppendEntriesRPC(rpc) {
        // TODO: implement AppendEntries RPC handling (log replication)
    }

    appendCommand(command) {
        if (this.state !== "Leader") {
            console.log("Cannot append command: not leader");
            return;
        }
        this.log.push({ term: this.currentTerm, command });
        console.log(`Appended command to log:`, command);
        // TODO: replicate to followers
    }

    addNode(nodeId) {
        console.log(`Adding node ${nodeId} to cluster`);
        // TODO: handle joint consensus reconfiguration
    }

    removeNode(nodeId) {
        console.log(`Removing node ${nodeId} from cluster`);
        // TODO: handle safe removal
    }
}

// Export
module.exports = { RaftNode };

// Example Usage
if (require.main === module) {
    const node1 = new RaftNode({ id: 1, peers: [2,3] });
    node1.start();
    node1.appendCommand({ action: "set", key: "x", value: 42 });
}
