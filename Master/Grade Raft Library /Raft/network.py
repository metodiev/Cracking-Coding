class RPCNetwork:
    def __init__(self, nodes):
     self.nodes = nodes # node_id -> RaftNode


    def send_append_entries(self, leader_id, target_id, entries):
    # Send AppendEntries RPC
         pass


    def send_request_vote(self, candidate_id, target_id):
    # Send RequestVote RPC
        pass    