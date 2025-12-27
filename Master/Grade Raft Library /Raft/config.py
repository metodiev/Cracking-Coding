class ClusterConfig:
    def __init__(self, nodes):
        self.nodes = set(nodes)


    def add_node(self, node_id):
        self.nodes.add(node_id)


    def remove_node(self, node_id):
        self.nodes.remove(node_id)