class PolicyTree:
    def __init__(self, data_):
        self.data = []
        if data_:
            self.data.append(data_)
        
        self.subtrees = {}

    def add_subtree(self,key,subtree):
        self.subtrees[key] = subtree   

    def print_trees(self, indent=0):
        print(" " * indent + "Decision Rule: " +str(self.data[0]) + ", value: " +str(self.data[1]))
        for key, subtree in self.subtrees.items():
            print("" * (indent + 2) + "└─ "+ f"belief : {key.value}")
            subtree.print_trees(indent + 5)
    