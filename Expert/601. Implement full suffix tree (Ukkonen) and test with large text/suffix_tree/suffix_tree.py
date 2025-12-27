from .node import Node
from .end import End

class SuffixTree:
    def __init__(self, text):
        self.text = text
        self.root = Node()
        self.root.suffix_link = self.root
        self.active_node = self.root
        self.active_edge = -1
        self.active_length = 0
        self.remaining = 0
        self.leaf_end = End(-1)
        self.build()

    def edge_length(self, node):
        return node.end.value - node.start + 1

    def build(self):
        for i, ch in enumerate(self.text):
            self.leaf_end.value = i
            self.remaining += 1
            last_new_node = None

            while self.remaining > 0:
                if self.active_length == 0:
                    self.active_edge = i

                edge_char = self.text[self.active_edge]

                if edge_char not in self.active_node.children:
                    leaf = Node()
                    leaf.start = i
                    leaf.end = self.leaf_end
                    self.active_node.children[edge_char] = leaf

                    if last_new_node:
                        last_new_node.suffix_link = self.active_node
                        last_new_node = None
                else:
                    next_node = self.active_node.children[edge_char]
                    if self.active_length >= self.edge_length(next_node):
                        self.active_edge += self.edge_length(next_node)
                        self.active_length -= self.edge_length(next_node)
                        self.active_node = next_node
                        continue

                    if self.text[next_node.start + self.active_length] == ch:
                        self.active_length += 1
                        if last_new_node:
                            last_new_node.suffix_link = self.active_node
                        break

                    split = Node()
                    split.start = next_node.start
                    split.end = End(next_node.start + self.active_length - 1)

                    self.active_node.children[edge_char] = split

                    leaf = Node()
                    leaf.start = i
                    leaf.end = self.leaf_end

                    split.children[ch] = leaf
                    next_node.start += self.active_length
                    split.children[self.text[next_node.start]] = next_node

                    if last_new_node:
                        last_new_node.suffix_link = split
                    last_new_node = split

                self.remaining -= 1
                if self.active_node == self.root and self.active_length > 0:
                    self.active_length -= 1
                    self.active_edge = i - self.remaining + 1
                else:
                    self.active_node = self.active_node.suffix_link