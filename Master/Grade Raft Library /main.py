from raft.node import RaftNode
from raft.raft import RaftController


# Create 3 nodes
nodes = [RaftNode(i, [1,2,3]) for i in range(1,4)]
controller = RaftController(nodes)
controller.start_all()


# Append command to leader
controller.append_command(1, ("key", "value"))