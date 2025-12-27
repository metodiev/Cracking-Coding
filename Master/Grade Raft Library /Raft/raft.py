from .node import RaftNode
from .network import RPCNetwork


class RaftController:
    def __init__(self, nodes):
        self.nodes = {n.node_id: n for n in nodes}
        self.network = RPCNetwork(self.nodes)


    def start_all(self):
        for node in self.nodes.values():
            node.start()


    def append_command(self, leader_id, command):
        leader = self.nodes[leader_id]
        leader.append_command(command)