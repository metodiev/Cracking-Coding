/**
 * raft-log.js
 * Log management and snapshot handling for Raft nodes
 */

class RaftLog {
    constructor() {
        this.entries = []; // Array of { term, command }
        this.snapshot = null; // Snapshot of state machine
        this.lastIncludedIndex = 0;
        this.lastIncludedTerm = 0;
    }

    append(entry) {
        this.entries.push(entry);
    }

    getEntry(index) {
        if (index <= this.lastIncludedIndex) {
            console.log("Entry is in snapshot");
            return null;
        }
        return this.entries[index - this.lastIncludedIndex - 1];
    }

    createSnapshot(stateMachineState, lastIncludedIndex, lastIncludedTerm) {
        this.snapshot = stateMachineState;
        this.lastIncludedIndex = lastIncludedIndex;
        this.lastIncludedTerm = lastIncludedTerm;
        this.entries = this.entries.slice(lastIncludedIndex);
        console.log("Snapshot created at index", lastIncludedIndex);
    }

    restoreSnapshot(snapshot, lastIncludedIndex, lastIncludedTerm) {
        this.snapshot = snapshot;
        this.lastIncludedIndex = lastIncludedIndex;
        this.lastIncludedTerm = lastIncludedTerm;
        this.entries = [];
        console.log("Snapshot restored");
    }
}

module.exports = { RaftLog };
