from constant import Constants
import numpy as np
import sys
import utilities as Utilities
CONSTANT = Constants.get_instance()



class BeliefNetwork:
    def __init__(self,horizon) :
        """    Mapping of belief tree using belief_ids and joint_action , joint_observation branches """
        """    belief_id,joint_action,joint_observation ->   list(next_belief_ids)                    """
        self.network = {}
   
    def add_new_connection(self,belief_id,joint_action,joint_observation,next_belief_id):
        if belief_id not in self.network: 
            self.network[belief_id] = {joint_action:{joint_observation:None}}
        if joint_action not in self.network[belief_id]:
            self.network[belief_id][joint_action] = {}
        if joint_observation not in self.network[belief_id][joint_action]:
            self.network[belief_id][joint_action][joint_observation] = None
        self.network[belief_id][joint_action][joint_observation] = next_belief_id
        return
    
    def existing_next_belief_id(self,belief_id,joint_action,joint_observation):
        if belief_id in self.network.keys():
            if joint_action in self.network[belief_id].keys():
                if joint_observation in self.network[belief_id][joint_action].keys() :
                    if self.network[belief_id][joint_action][joint_observation]!= None : return self.network[belief_id][joint_action][joint_observation]
                    else : 
                        print("no belief found")
                        sys.exit()
                else : 
                    print("unexplored branch")
                    sys.exit()

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

        #initialize initial belief object
        if type(initial_belief)!=Belief: self.initial_belief = Belief(initial_belief,None,None)
        else : self.initial_belief = initial_belief

        #initialize network
        self.network = BeliefNetwork(horizon)

        #initialize belief dictionary that keeps mapping of beliefs to belief_ids
        self.belief_dictionary = {}
        self.belief_dictionary[0] = self.initial_belief

        #initialize time_index_table dictionary that stores beliefs at every timestep as a set (so as to only keep unique ids)
        self.time_index_table = {}
        for timestep in range(horizon+2):
            self.time_index_table[timestep] = set()

        #add intial belief at timstep zero 
        self.time_index_table[0].add(0)
        self.id = 1
      
    
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
            closest_belief = None
            # random.sample(beliefs_t, len(beliefs_t))
            # for id, stored_belief in self.belief_dictionary.items():
            #     distance = np.linalg.norm(belief.value - stored_belief.value)
            #     if distance<max: 
            #         closest_belief = stored_belief
            #         closest_belief_id = id
            #         max = distance
            min_belief = min(self.belief_dictionary.values(), key=lambda stored_belief: np.linalg.norm(stored_belief.value-belief.value))
            closest_belief_id = self.find_belief_id(min_belief)
            if closest_belief_id!=None: return closest_belief,closest_belief_id
            else : 
                print("err0r : no belief found")
                print(min_belief.value)
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
        self.time_index_table[timestep+1].add(self.id)
        self.id+=1


    def expansion(self):
        """populates self.belief_state table by expanding belief tree to all branches so long as it satisfies density sufficiency"""
        for timestep in range(self.horizon+1):
            for current_belief_id in self.time_index_table[timestep]:
                    for joint_action in CONSTANT.JOINT_ACTIONS:
                        for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                            n_belief = self.belief_dictionary[current_belief_id].next_belief(joint_action,joint_observation)
                            if n_belief:
                                if self.distance(n_belief) and len(self.belief_dictionary) < self.limit:
                                    self.network.add_new_connection(current_belief_id,joint_action,joint_observation,self.id)
                                    self.add_new_belief_in_bag(n_belief,timestep)
                                elif len(self.belief_dictionary) < self.limit:
                                    closest_belief,closest_belief_index = self.get_closest_belief(n_belief)
                                    self.network.add_new_connection(current_belief_id,joint_action,joint_observation,closest_belief_index)
                                    self.time_index_table[timestep+1].add(closest_belief_index)
    
        print(f"\tbelief expansion done, belief space size = {self.belief_size()}\n")
      
       
       
       
    
         

class Belief:
    def __init__(self,value,action_label,observation_label,id=None):
        self.value = value 
        self.action_label = action_label
        self.observation_label = observation_label
        self.id = id

    def set_id(self,id):
        self.id=id
        return
    
    def __eq__(self, other):
        return (isinstance(other, Belief) and 
                self.action_label == other.action_label and 
                self.observation_label == other.observation_label and self.id == other.id)

    def __hash__(self):
        # Hash only using certain unique attributes
        return hash((self.action_label, self.observation_label,self.id))

    def next_belief(self,joint_DR,joint_observation):
        """function to calculate next belief based on current belief, compatible with probabilistic joint_DR/ deterministc joint action , and observation"""
        # returns the value of b1
        next_belief_value= np.zeros(len(self.value))

        if type(joint_observation) != int :
            joint_observation = self.PROBLEM.joint_observations.index(joint_observation)


        if type(joint_DR) == int: # if joint_DR enterred as a deterministic action 
            # b_next(x') = \sum_{x} += b_current[x] * TRANSITION_MATRIX(u_j,x',z_j,x)
            for next_state in CONSTANT.STATES:
                value = 0
                for state in CONSTANT.STATES:
                    value += self.value[state] * CONSTANT.TRANSITION_FUNCTION[joint_DR][state][next_state]  * CONSTANT.OBSERVATION_FUNCTION[joint_DR][state][joint_observation]
                next_belief_value[next_state]+=value  

        else:   # if joint_DR is a decision rule
            # b_next(x') = \sum_{x} \sum_(u_j) += b_current[x] * a(u_j) TRANSITION_MATRIX(u_j,x',z_j,x)
            for next_state in CONSTANT.STATES:
                value = 0
                for state in CONSTANT.STATES:
                    for joint_action in CONSTANT.JOINT_ACTIONS:
                        value += self.value[state] * joint_DR[joint_action] * CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state]  * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation]
                next_belief_value[next_state]+=value

        if np.sum(next_belief_value) == 0 :
            return None
        
        # if not none normalize
        next_belief_value = Utilities.normalize(next_belief_value)

        if np.sum(next_belief_value)<= 1.001 and np.sum(next_belief_value)> 0.99999:
            return  Belief(next_belief_value,joint_DR,joint_observation)
        else:
            print("err0r : belief doesn not sum up to 1\n")
            print(f"current belief: \n{self.value}")
            print(f"next belief :\n{next_belief_value}")
            print(f"sum : {np.sum(next_belief_value)}")
            sys.exit()
        