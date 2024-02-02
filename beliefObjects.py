from constant import Constants
import numpy as np
import sys
from utilities import Utilities

CONSTANT = Constants.get_instance()
utilities = Utilities(CONSTANT)



class BeliefNetwork:
    def __init__(self,horizon) :
        self.network = {}
        # for timestep in range(horizon):
        #     self.network[timestep] = {}

    
    def add_new_connection(self,belief_id,joint_action,joint_observation,next_belief_id):
        if belief_id not in self.network: 
            self.network[belief_id]={joint_action:{joint_observation:None}}
        if joint_action not in self.network[belief_id]:
            self.network[belief_id][joint_action] = {}
        if joint_observation not in self.network[belief_id][joint_action]:
            self.network[belief_id][joint_action][joint_observation] = None
        self.network[belief_id][joint_action][joint_observation] = next_belief_id
    
    def existing_next_belief_id(self,belief_id,joint_action,joint_observation):
        if belief_id in self.network.keys():
            if joint_action in self.network[belief_id].keys():
                if joint_observation in self.network[belief_id][joint_action].keys():
                    return self.network[belief_id][joint_action][joint_observation]
        return None

      
    def print_network(self):
        for belief_id in self.network.keys():
            print(f"belief {belief_id}")
            for joint_action in self.network[belief_id]:
                for joint_observation in self.network[belief_id][joint_action]:
                    print(f"  ∟ action {joint_action}, observation {joint_observation} : belief {self.network[belief_id][joint_action][joint_observation]}")





class BeliefSpace:
    def __init__(self,horizon,initial_belief,density, limit):
        self.density = density
        self.limit = limit
        self.horizon = horizon
        self.belief_dictionary = {}
        self.network = BeliefNetwork(horizon)
        if type(initial_belief)!=Belief: self.initial_belief = Belief(initial_belief,None,None)
        else : self.initial_belief = initial_belief
        self.belief_dictionary[0] = Belief(initial_belief,None,None)
        self.time_index_table = {}
        for timestep in range(horizon+2):
            self.time_index_table[timestep] = set()
        self.time_index_table[0].add(0)

        
        self.id = 1
      
    
    def get_belief(self,id):
        return self.belief_dictionary[id]

    def get_inital_belief(self):
        return self.belief_dictionary[0]


    def distance(self,belief):
        """function to check if a new belief point is "sufficiently different from other points in the bag of beliefs """
        if len(self.belief_dictionary)<=1: return True
        min_belief = min(self.belief_dictionary.values(), key=lambda stored_belief: np.linalg.norm(stored_belief.value-belief.value))
        min_magnitude = np.linalg.norm(min_belief.value-belief.value)
        return min_magnitude > self.density
        # check what happens if there are no stored beliefs at timestep
    
    def get_closest_belief(self,belief):
        """ returns belief state at timestep t that is closest in distance to the input belief """
        max = np.inf
        closest_belief = None
        # random.sample(beliefs_t, len(beliefs_t))
        for id, belief_t in self.belief_dictionary.items():
            distance = np.abs(np.linalg.norm(np.array(belief.value) - np.array(belief_t.value)))
            if distance<max: 
                closest_belief = belief_t
                closest_belief_id = id
                max = distance
        if closest_belief: return closest_belief,closest_belief_id
        else : 
            print("err0r : no belief found")
            sys.exit()
        

    def belief_size(self):
        return len(self.belief_dictionary)

    def expansion(self):
        """populates self.belief_state table"""
        for timestep in range(self.horizon+1):
            for current_belief_id in self.time_index_table[timestep]:
                    for joint_action in CONSTANT.JOINT_ACTIONS:
                        for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                            n_belief = self.belief_dictionary[current_belief_id].next_belief(joint_action,joint_observation)
                            #check if next belief is not all zeros
                            if n_belief:
                                if self.distance(n_belief) and len(self.belief_dictionary) < self.limit:
                                    self.belief_dictionary[self.id]= n_belief
                                    self.network.add_new_connection(current_belief_id,joint_action,joint_observation,self.id)
                                    self.time_index_table[timestep+1].add(self.id)
                                    self.id+=1
                                else:
                                    n_belief,next_belief_index = self.get_closest_belief(n_belief)
                                    self.network.add_new_connection(current_belief_id,joint_action,joint_observation,next_belief_index)
                                    self.time_index_table[timestep+1].add(next_belief_index)


    
        print(f"\tbelief expansion done, belief space size = {self.belief_size()}\n")
        # for timestep in range(self.horizon+2):
        #     print(f"{self.time_index_table[timestep]}")
        # sys.exit()
        # print(f"belief_id at each timestep : {self.time_index_table}\n")
        # print(f"network : {self.network.print_network()} ")
       
       
       
    
         

class Belief:
    def __init__(self,value,action_label,observation_label,id=None):
        self.value = value 
        self.action_label = action_label
        self.observation_label = observation_label
        self.id = id

    def set_id(self,id):
        self.id=id
        return

    def next_belief(self,joint_DR,joint_observation):
        """function to calculate next belief based on current belief, DR/joint action , and observation"""
        # returns the value of b1
        next_belief_value= np.zeros(len(self.value))

        if type(joint_observation) != int :
            joint_observation = self.PROBLEM.joint_observations.index(joint_observation)


        if type(joint_DR) == int: # if joint_DR enterred as a deterministic action 
            # print(f"{CONSTANT.TRANSITION_FUNCTION} and {CONSTANT.OBSERVATION_FUNCTION}")
            for next_state in CONSTANT.STATES:
                value = 0
                for state in CONSTANT.STATES:
                    value += self.value[state] * CONSTANT.TRANSITION_FUNCTION[joint_DR][state][next_state]  * CONSTANT.OBSERVATION_FUNCTION[joint_DR][state][joint_observation]

                next_belief_value[next_state]+=value    
        else:   # if joint_DR is a decision rule
            for next_state in CONSTANT.STATES:
                value = 0
                for state in CONSTANT.STATES:
                    for joint_action in CONSTANT.JOINT_ACTIONS:
                        value += self.value[state] * joint_DR[joint_action] * CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state]  * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation]
                next_belief_value[next_state]+=value

        if np.sum(next_belief_value) ==0 :
            return None
        next_belief_value = utilities.normalize(next_belief_value)

        if np.sum(next_belief_value)<= 1.001 and np.sum(next_belief_value)> 0.99999:
            return  Belief(next_belief_value,joint_DR,joint_observation)
        else:
            print("err0r : belief doesn not sum up to 1\n")
            print(f"current belief: \n{self.value}")
            print(f"next belief :\n{next_belief_value}")
            print(f"sum : {np.sum(next_belief_value)}")
            sys.exit()
        