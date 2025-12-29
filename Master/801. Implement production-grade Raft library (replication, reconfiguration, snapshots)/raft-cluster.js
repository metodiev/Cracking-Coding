/**
 * raft-cluster.js
 * Cluster management and reconfiguration
 */

const { RaftNode } = require("./raft-node");

class RaftCluster {
    constructor(nodes = []) {
        this.nodes = new Map(); // Map nodeId -> RaftNode
        nodes.forEach(node => this.nodes.set(node.id, node));
    }

    addNode(node) {
        if (this.nodes.has(node.id)) {
            console.log(`Node ${node.id} already exists`);
            return;
        }
        this.nodes.set(node.id, node);
        console.log(`Node ${node.id} added to cluster`);
        // TODO: handle joint consensus if cluster is running
    }

    removeNode(nodeId) {
        if (!this.nodes.has(nodeId)) {
            console.log(`Node ${nodeId} does not exist`);
            return;
        }
        this.nodes.delete(nodeId);
        console.log(`Node ${nodeId} removed from cluster`);
        // TODO: handle joint consensus removal
    }

    getLeader() {
        for (const node of this.nodes.values()) {
            if (node.state === "Leader") return node;
        }
        return null;
    }

    startAllNodes() {
        for (const node of this.nodes.values()) {
            node.start();
        }
    }
}

module.exports = { RaftCluster };

// Example Usage
if (require.main === module) {
    const node1 = new RaftNode({ id: 1, peers: [2] });
    const node2 = new RaftNode({ id: 2, peers: [1] });
    const cluster = new RaftCluster([node1, node2]);
    cluster.startAllNodes();
}
