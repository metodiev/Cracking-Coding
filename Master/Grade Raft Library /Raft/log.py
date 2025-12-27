class LogEntry:
    def __init__(self, index, term, command):
        self.index = index
        self.term = term
        self.command = command