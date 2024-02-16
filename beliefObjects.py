from constant import Constants
import numpy as np
import sys
import utilities as Utilities
from belief import Belief
CONSTANT = Constants.get_instance()



class BeliefSpace:
    def __init__(self,horizon,initial_belief,density, limit):
        self.density = density
        self.limit = limit
        self.horizon = horizon

        #initialize initial belief object
        if type(initial_belief)!=Belief: self.initial_belief = Belief(initial_belief,None,None)
        else : self.initial_belief = initial_belief

        #initialize network
        self.network = BeliefNetwork(horizon,self)

        #initialize belief dictionary that keeps mapping of beliefs to belief_ids
        self.belief_dictionary = {0:self.initial_belief}

        #initialize time_index_table dictionary that stores beliefs at every timestep as a set (so as to only keep unique ids)
        self.time_index_table = {}
        for timestep in range(horizon+2):
            self.time_index_table[timestep] = set()

        #add intial belief at timstep zero 
        self.time_index_table[0].add(0)
        self.id = 1
    
    def reset(self):


        #initialize belief dictionary that keeps mapping of beliefs to belief_ids
        self.belief_dictionary = {0:self.initial_belief}

        #initialize time_index_table dictionary that stores beliefs at every timestep as a set (so as to only keep unique ids)
        self.time_index_table = {}
        for timestep in range(self.horizon+2):
            self.time_index_table[timestep] = set()
        
        #initialize network
        self.network = BeliefNetwork(self.horizon,self)

        #add intial belief at timstep zero 
        self.time_index_table[0].add(0)
        self.id = 1
        return self

    def set_density(self,density):
        self.density = density
        return
    
    def get_belief(self,belief_id):
        return self.belief_dictionary[belief_id]

    def find_belief_id(self, belief):
        for belief_id, belief_in_network in self.belief_dictionary.items():
            if np.array_equal(belief.value, belief_in_network.value):
                return belief_id
        else :
            _ ,belief_id = self.get_closest_belief(belief)
        return belief_id

    def get_inital_belief(self):
        """returns beief object at timestep 0, i.e. belief object with index 0"""
        return self.belief_dictionary[0]


    def distance(self,belief):
        """function to check if a new belief point is "sufficiently different from other points in the bag of beliefs(uses density value set in class )"""
        if len(self.belief_dictionary)<=1: return True
        min_belief = min(self.belief_dictionary.values(), key=lambda stored_belief: np.linalg.norm(stored_belief.value-belief.value))
        min_magnitude = np.linalg.norm(min_belief.value-belief.value)


        return min_magnitude > self.density
        # check what happens if there are no stored beliefs at timestep
    
    def get_closest_belief(self,belief):
        """ returns belief and belief_id at timestep t that is closest in distance to the input belief """
        if not self.distance(belief) :
            max = np.inf
            closest_belief_id, min_belief_value = min(self.belief_dictionary.items(), key=lambda stored_belief: np.linalg.norm(stored_belief[1].value - belief.value))
            if closest_belief_id!=None: return self.get_belief(closest_belief_id),closest_belief_id
            else : 
                print("err0r : no belief found")
                print(min_belief_value)
                print(closest_belief_id)

                sys.exit()
        else : 
            print("New belief encountered, improper sampling!")
            sys.exit()
        
    
    def belief_size(self):
        """return number of beliefs in bag"""
        return len(self.belief_dictionary)

    
    def add_new_belief_in_bag(self,belief,timestep):
        belief.set_id(self.id)
        self.belief_dictionary[self.id]= belief
        self.time_index_table[timestep].add(self.id)


    def expansion(self):
        """populates self.belief_state table by expanding belief tree to all branches so long as it satisfies density sufficiency"""
        self.reset()
        print(self.time_index_table)
        for timestep in range(0,self.horizon):
            for current_belief_id in self.time_index_table[timestep]:
                    for joint_action in CONSTANT.JOINT_ACTIONS:
                        for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                            if Utilities.observation_probability(joint_observation,self.get_belief(current_belief_id),joint_action):
                                original_next_belief = self.belief_dictionary[current_belief_id].next_belief(joint_action,joint_observation)
                                if self.distance(original_next_belief) and len(self.belief_dictionary) < self.limit:
                                    self.add_new_belief_in_bag(original_next_belief,timestep+1)
                                    self.network.add_new_connection(timestep,current_belief_id,joint_action,joint_observation,self.id)
                                    self.id+=1
                                elif len(self.belief_dictionary) < self.limit:
                                    closest_belief,closest_belief_index = self.get_closest_belief(original_next_belief)
                                    self.network.add_new_connection(timestep,current_belief_id,joint_action,joint_observation,closest_belief_index)
                                    self.time_index_table[timestep+1].add(closest_belief_index)
                                    # print(f"original_next_belief =  {original_next_belief.value}, closest_belief= {closest_belief.value}")


        print(f"\tbelief expansion done, belief space size = {self.belief_size()}\n")
        # print(f"{[self.time_index_table[index] for index in range(len(self.time_index_table))]}")
        # self.network.print_network()

    
       

class BeliefNetwork(BeliefSpace):
    def __init__(self,horizon, belief_space) :
        """    Mapping of belief tree using belief_ids and joint_action , joint_observation branches """
        """    belief_id,joint_action,joint_observation ->   list(next_belief_ids)                    """
        self.horizon = horizon
        self.network = {}
        self.belief_space = belief_space

   
    def add_new_connection(self,timestep,belief_id,joint_action,joint_observation,next_belief_id):
        if timestep not in self.network.keys():
            self.network[timestep] = {}
        if belief_id not in self.network[timestep].keys(): 
            self.network[timestep][belief_id]= {joint_action:{joint_observation:None}}
        if joint_action not in self.network[timestep][belief_id]:
            self.network[timestep][belief_id][joint_action] = {}
        self.network[timestep][belief_id][joint_action][joint_observation] = next_belief_id
        return
    

    def existing_next_belief_id(self,timestep,belief_id,joint_action,joint_observation):
        if timestep==self.horizon: return None
        if belief_id in self.network[timestep].keys():
            if joint_action in self.network[timestep][belief_id].keys():
                if joint_observation in self.network[timestep][belief_id][joint_action].keys() and Utilities.observation_probability(joint_observation,self.belief_space.get_belief(belief_id),joint_action):
                    next_belief_id = self.network[timestep][belief_id][joint_action][joint_observation]
                    if next_belief_id!= None : return next_belief_id

                else : return None
        return None

    def print_network(self):
        for timstep in self.network.keys():
            print(f"timetep {timstep}")
            for belief_id in self.network[timstep].keys():
                print(f"  ∟ belief {belief_id}")
                for joint_action in self.network[timstep][belief_id]:
                    for joint_observation in self.network[timstep][belief_id][joint_action]:
                        print(f"      ∟ action {joint_action}, observation {joint_observation} : belief {self.network[timstep][belief_id][joint_action][joint_observation]}")
