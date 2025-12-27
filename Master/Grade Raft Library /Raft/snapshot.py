class Snapshot:
    def __init__(self, log, commit_index):
        self.state = self.apply_log(log, commit_index)
        self.last_index = commit_index


    def apply_log(self, log, commit_index):
        # Apply committed log entries to generate state snapshot
        state = {}
        for entry in log[:commit_index+1]:
            # Assume commands are simple key-value sets
            if isinstance(entry.command, tuple):
                key, value = entry.command
                state[key] = value
        return state