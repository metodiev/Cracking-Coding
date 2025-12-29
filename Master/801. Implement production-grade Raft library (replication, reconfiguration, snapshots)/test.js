/**
 * test.js
 * Unit & integration tests for Raft library
 */

const { RaftNode } = require("./raft-node");
const { RaftCluster } = require("./raft-cluster");
const { RaftLog } = require("./raft-log");

console.log("=== Raft Node Tests ===");
const node1 = new RaftNode({ id: 1, peers: [2,3] });
node1.start();
node1.appendCommand({ action: "set", key: "x", value: 42 });

console.log("\n=== Raft Log Tests ===");
const log = new RaftLog();
log.append({ term: 1, command: { action: "set", key: "y", value: 100 } });
log.createSnapshot({ y: 100 }, 0, 1);

console.log("\n=== Raft Cluster Tests ===");
const node2 = new RaftNode({ id: 2, peers: [1,3] });
const node3 = new RaftNode({ id: 3, peers: [1,2] });
const cluster = new RaftCluster([node1, node2, node3]);
cluster.startAllNodes();
console.log("Leader node:", cluster.getLeader()?.id ?? "No leader yet");

// TODO: Add simulated RPCs, election, replication, reconfiguration tests
