import asyncio
from .log import LogEntry
from .snapshot import Snapshot


class RaftNode:
    def __init__(self, node_id, cluster):
        self.node_id = node_id
        self.cluster = cluster # list of node_ids
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = -1
        self.last_applied = -1
        self.state = 'follower' # 'follower', 'candidate', 'leader'
        self.next_index = {}
        self.match_index = {}
        self.snapshot = None


def start(self):
# Start event loop or timers
    pass


def append_command(self, command):
    if self.state != 'leader':
        raise Exception("Not leader")
        entry = LogEntry(len(self.log), self.current_term, command)
    self.log.append(entry)
    self.replicate_log()


def replicate_log(self):
# Send AppendEntries RPCs to followers
    pass


def create_snapshot(self):
    self.snapshot = Snapshot(self.log, self.commit_index)