class PolicyTree:
    def __init__(self, DR,value):
        self.DR = DR
        self.value = value
        self.subtrees = {}

    def add_subtree(self,key,subtree):
        self.subtrees[key] = subtree   

    def print_trees(self, indent=0):
        print(" " * indent + "Decision Rule: " +str(self.DR) + ", value: " +str(self.value))
        for key, subtree in self.subtrees.items():
            print(" " * (indent + 2) + "└─ "+ f"belief : {key}")
            subtree.print_trees(indent + 5)
    
    def subtree(self,belief_id,joint_action,joint_observation):
        for stored_belief_id,subtree in self.subtrees.items():
            if stored_belief_id == belief_id:
                return subtree
        return PolicyTree(None,None)