##  CLASS DEFINITIONS
import numpy as np
import sys 
from decpomdp import DecPOMDP
from constant import Constants
from utilities import *
import random
import gc
gc.enable()

CONSTANT =  Constants.get_instance()
utilities = Utilities(CONSTANT)
PROBLEM = CONSTANT.PROBLEM




class AlphaVector:
    def __init__(self,DR,vector1,vector2,problem,sota=False):
        self.DR = DR
        self.sota = sota
        self.vectors = [vector1,vector2]
        self.problem = problem
        
        
    def get_value(self,belief):
        return np.dot(belief.value,self.vectors[0]),np.dot(belief.value,self.vectors[1])

    def print_vector(self):
        print(self.vectors)

    def set_value(self,agent,hidden_state,value):
        self.vectors[agent][hidden_state] = value

 
   

    
################################################################################################

class DecisionRule:
    def __init__(self,a1,a2,aj):
        self.individual = {0:a1,1:a2}
        self.joint = aj

################################################################################################

class BetaVector:
    def __init__(self,two_d_vector_0,two_d_vector_1,problem):
        self.problem = problem
        self.two_d_vectors = [two_d_vector_0,two_d_vector_1]

    def get_alpha_vector(self, payoff, game_type, DR, sota=False):
        vectors = np.zeros((2,len(CONSTANT.STATES)))

        if game_type == "zerosum" and sota == True:
            u_1_best = 0
            u_1_best_value = -np.inf
            for u_1 in CONSTANT.ACTIONS[1]:
                u_1_value = 0
                
                for u_0 in CONSTANT.ACTIONS[0]:
                    u_1_value += DR.individual[0][u_0] * payoff[self.problem.get_joint_action(u_0, u_1)]

                if u_1_best_value < u_1_value:
                   u_1_best = u_1
                   u_1_best_value = u_1_value 

            for state in CONSTANT.STATES:
                vectors[0][state] = 0
                for u_0 in CONSTANT.ACTIONS[0]:
                    vectors[0][state] += DR.individual[0][u_0] * self.two_d_vectors[0][self.problem.get_joint_action(u_0, u_1_best)][state]

            return AlphaVector(DR,vectors[0],vectors[0], self.problem,sota)

        else:    
            for state in CONSTANT.STATES:
                vectors[0][state] = 0
                vectors[1][state] = 0
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    vectors[0][state] += DR.joint[joint_action] * self.two_d_vectors[0][joint_action][state]
                    vectors[1][state] += DR.joint[joint_action] * self.two_d_vectors[1][joint_action][state]

        return AlphaVector(DR,vectors[0],vectors[1], self.problem, sota)

        
    def print_vector(self):
        print(self.two_d_vectors)



################################################################################################

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
    
        

################################################################################################
            
class ValueFunction:
    def __init__(self,horizon, initial_belief,problem,belief_space,sota=False):
        self.horizon = horizon
        self.vector_sets = {}
        self.problem=problem
        self.point_value_fn = {}
        self.belief_space = belief_space
        self.sota=sota
        self.initial_belief = initial_belief
        for timestep in range(horizon+2):
            self.point_value_fn[timestep] = {}
            for agent in range(2):
                self.vector_sets[timestep] = []
                self.point_value_fn[timestep][agent] = {}
                for belief_index in self.belief_space.time_index_table[timestep]:
                    self.point_value_fn[timestep][agent][belief_index] = 0
            #     print(f"value function at timestep {timestep} : {self.point_value_fn[timestep]}")
        # sys.exit()


        vector = np.zeros(len(CONSTANT.STATES))
        self.add_alpha_vector(AlphaVector(0,vector,vector,problem,self.sota),horizon)
    
    def add_alpha_vector(self,alpha,timestep):
        self.vector_sets[timestep].append(alpha)

    def pruning(self,timestep):
        self.vector_sets[timestep] = set(self.vector_sets[timestep])
    
    def get_alpha_vector(self, belief,timestep):
        max = -np.inf
        max_alpha = 0
        for alpha in self.vector_sets[timestep]:
            value = alpha.get_value(belief)
            if value>max:
                max = value
                max_alpha = alpha
        return max,max_alpha
    
    def point_backup(self,belief_space,belief_id,timestep,game_type):

        #initialize payoff matrix for linear program
        belief = belief_space.belief_dictionary[belief_id]
        payoffs = {}
        value = 0
        # calculate values for payoff values
        for agent in range(0,2):
            #get proper rewards
            reward = CONSTANT.REWARDS[game_type][agent]
            payoffs[agent] = np.zeros(len(CONSTANT.JOINT_ACTIONS))
            if game_type!="stackelberg" and agent==1 and self.sota==True :
                payoffs[agent] = payoffs[0]
                break
            for joint_action in CONSTANT.JOINT_ACTIONS:
                for state in CONSTANT.STATES :
                    # payoff = b(x)r(x,u)
                    payoffs[agent][joint_action] += reward[joint_action][state] * belief.value[state]
                
                    if game_type == "stackelberg"  and agent==1 and self.sota==True : 
                        continue #for the blind opponent of the stackelberg games

                    # payoff += \sum_z Pr(z|b,u)*V_{t+1}(T(b,u,z))
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        observation_probability = utilities.observation_probability(joint_observation,belief,joint_action)
                        if observation_probability > 0:  
                            next_belief_id = belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                            # if joint_action==0 and timestep==0:
                            #     print(f"for action LISTEN-LISTEN : Pr(z|b,u) = {utilities.observation_probability(joint_observation,belief,joint_action)} , next optimal value = {self.point_value_fn[timestep+1][next_belief_id]}  ")
                            #     print(f" Value function at timestep {timestep+1} : {self.point_value_fn[timestep+1]}, and next belief id = {next_belief_id}")
                            payoffs[agent][joint_action] += utilities.observation_probability(joint_observation,belief,joint_action) * self.point_value_fn[timestep+1][agent][next_belief_id]
                            
        # Get optimal DR for payoff matrix using linear program
        if self.sota==False :
            leader_value, DR , DR0 , DR1 = utilities.LP(payoffs[0],payoffs[1])
              # save value of leader from lienar program in value function indexed by belief ID
            self.point_value_fn[timestep][0][belief_id] = leader_value
            self.point_value_fn[timestep][1][belief_id] = np.dot(DR,payoffs[1])

        else:
            leader_value, DR, DR0, DR1 = utilities.sota_strategy(payoffs[0],payoffs[1],game_type)
            self.point_value_fn[timestep][0][belief_id] = leader_value
      
        return 
    
        
    
    def backup(self,belief,belief_id,timestep,gametype):
        max = -np.inf
        max_alpha = None

        for alpha in self.vector_sets[timestep+1]:
            
            value , _alpha = alpha.solve(belief,gametype)
            if type(alpha) != AlphaVector or type(_alpha) != AlphaVector or alpha.sota!=self.sota:
                print("ERROR ALPHA VECTOR MISMATCH")
            if value>max:
                max = value
                max_alpha = _alpha
        if max_alpha == None:
            print(f"time : {timestep}")
            print(f"max_alpha = {max_alpha}")
            print(f"size : {len(self.vector_sets[timestep+1])}")
            return
        self.add_alpha_vector(max_alpha,timestep)

   
    def get_alpha_mappings(self,belief_id,timestep):
        #initialize
        belief = self.belief_space.belief_dictionary[belief_id]
        alpha_mappings = {}
        for  joint_action in CONSTANT.JOINT_ACTIONS:
            alpha_mappings[joint_action] = {}
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                alpha_mappings[joint_action][joint_observation] = None
       
        #loop over actions and observations 
        for joint_action in CONSTANT.JOINT_ACTIONS:
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                if utilities.observation_probability(joint_observation,belief,joint_action):
                    max = -np.inf
                    for alpha in self.vector_sets[timestep+1]:
                        next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                        if alpha.get_value(self.belief_space.belief_dictionary[next_belief_id])[0] > max:
                            max = alpha.get_value(self.belief_space.belief_dictionary[next_belief_id])[0]
                            alpha_mappings[joint_action][joint_observation] = alpha
                            if timestep ==0 :
                                print(f"Alpha mapping timestep {timestep} , joint action = {joint_action} , joint observation = {joint_observation}, \nvector =  {alpha.vectors}\n")

        return alpha_mappings


    def backup2(self,belief,belief_id,timestep,gametype):

        alpha_mappings = self.get_alpha_mappings(belief_id,timestep)

        value , _alpha = self.solve(belief,alpha_mappings,gametype)
       
       

        if _alpha == None:
            print(f"time : {timestep}")
            print(f"max_alpha = {_alpha}")
            print(f"size : {len(self.vector_sets[timestep+1])}")
            return
        self.add_alpha_vector(_alpha,timestep)

    def get_beta_two_d_vector(self,alpha_mappings,game_type):
        global CONSTANT
        two_d_vectors = {}

        for agent in range(0,2):
            reward = self.problem.REWARDS[game_type][agent]
            two_d_vectors[agent] = np.zeros((len(CONSTANT.JOINT_ACTIONS),(len(CONSTANT.STATES))))
            # To improve efficiency in the cases of zerosum and cooperative games, we only need the first player payoffs, so we can skip the second player payoffs and provide the same for both players.  
            if game_type!="stackelberg" and agent==1 and self.sota==True :
                return BetaVector(two_d_vectors[0],two_d_vectors[0],self.problem)
                
            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    two_d_vectors[agent][joint_action][state] = reward[joint_action][state]
                    
                    if game_type == "stackelberg"  and agent==1 and self.sota==True : 
                        continue #for the blind strategy of the stackelberg games
                    
                    for next_state in CONSTANT.STATES:
                        for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                            two_d_vectors[agent][joint_action][state]+= CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state] * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation]* alpha_mappings[joint_action][joint_observation].vectors[agent][state]
                    
        # note : u can filter the zero probabilites out of the vector to reduce computational 
 
        return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)
    
    
    def payoff_function(self,belief,alpha_mappings,game_type):
        beta = self.get_beta_two_d_vector(alpha_mappings,game_type)
        payoffs = {}
        for agent in range(0,2):
            payoffs[agent] = np.zeros(len(CONSTANT.JOINT_ACTIONS))
            for joint_action in CONSTANT.JOINT_ACTIONS:
                for state in CONSTANT.STATES:
                    payoffs[agent][joint_action] += belief.value[state] * beta.two_d_vectors[agent][joint_action][state]
        # print(f"payoffs:\nLeader:\n\t{payoffs[0]},\nFollower:\n{payoffs[1]}")
        return payoffs, beta
    

    
    def solve(self,belief,alpha_mappings,game_type):
        payoffs, beta = self.payoff_function(belief,alpha_mappings,game_type)
        
        if self.sota==False :
            value , DR , DR0 , DR1 = utilities.LP(payoffs[0],payoffs[1])
        
        else:
            value, DR, DR0, DR1 = utilities.sota_strategy(payoffs[0],payoffs[1],game_type)
        
        alpha = beta.get_alpha_vector(payoffs[0],game_type,DecisionRule(DR0,DR1,DR), self.sota)
        # print( f"Game {game_type}  ::  Original: {value}  --  Reconstructed: {alpha.get_value(belief)}   --  belief {belief.value}  -- DR {DR}" )
        return value, alpha

 
    
    def get_values_initial_belief(self):
        values_leader = -np.inf
        values_follower = -np.inf
    
        for alpha in self.vector_sets[0]:
            leader, follower = alpha.get_value(self.initial_belief)
            if( leader > values_leader ):
                values_leader = leader
                values_follower = follower

        return values_leader, values_follower
    


################################################################################################

class PBVI:
    def __init__(self,problem,horizon,density,gametype,limit,sota=False):
        self.sota = sota
        self.belief_space = BeliefSpace(horizon,problem.b0,density,limit)
        self.policies = [PolicyTree(None),PolicyTree(None)]
        self.value_function = None
        self.gametype = gametype
        self.problem = problem
        self.intitial_belief = Belief(problem.b0,None,None)
        self.horizon = horizon
        self.density = density
        self.limit = limit

    def backward_induction(self):
        for timestep in range(self.horizon-1,-1,-1):
            
            for belief_id in self.belief_space.time_index_table[timestep]:
                belief = self.belief_space.belief_dictionary[belief_id]
                self.value_function.backup2(belief,belief_id,timestep,self.gametype)
            print(f"\tbackup at timestep {timestep} done")

        print("\tbackward induction done")

    def point_backward_induction(self):
        for timestep in range(self.horizon,-1,-1):
            for belief_id in self.belief_space.time_index_table[timestep]:
                self.value_function.point_backup(self.belief_space,belief_id,timestep,self.gametype)
            print(f"\tbackup at timestep {timestep} done")

    
    def point_based_solve(self):
        """point based implementation without the use of alpha vectors"""
        self.belief_space.expansion()
        self.value_function = ValueFunction(self.horizon,self.intitial_belief,self.problem,self.belief_space,sota=self.sota)

        print(f"size of belief space = {self.belief_space.belief_size()}")
        self.point_backward_induction()
        print(f"\n\nvalue function : ")
        for timestep in range(self.horizon+1):
            print(f"value at {timestep}, agent = 0, values: {self.value_function.point_value_fn[timestep][0]} ,  agent = 1, values: {self.value_function.point_value_fn[timestep][1]}")

        return self.value_function
           
    def solve(self,iterations,decay):
        self.belief_space.expansion()
        self.value_function = ValueFunction(self.horizon,self.intitial_belief,self.problem,self.belief_space,sota=self.sota)

        for _ in range(iterations):
            print(f"iteration : {_}")
            self.backward_induction()
            self.density /= decay #hyperparameter
        return self.value_function.get_values_initial_belief()
   
    def tree_extraction(self,belief, agent,timestep):
        global utilities

        # edge case at last horizon
        if timestep > self.horizon : return PolicyTree(None)

        #initialize policy and DR_bt
        policy = PolicyTree(None)
        DR = DecisionRule(None,None,None)

        max = -np.inf

        # extract decision rule of agent from value function at current belief
        for alpha in self.value_function.vector_sets[timestep]:
            value = alpha.get_value(belief)
            if (max<value[agent]) :
                max = value[agent]
                DR = alpha.DR

        #for all agent actions
        for u_agent in CONSTANT.ACTIONS[agent]:
            #check if probability of action is not 0
            if DR.individual[agent][u_agent] > 0:

                #get actions of the other agent
                for u_not_agent in CONSTANT.ACTIONS[int(not agent)]:

                    joint_action = CONSTANT.PROBLEM.get_joint_action(u_agent,u_not_agent)
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        #get next belief of joint action and observation
                        belief_next = belief.next_belief(joint_action,joint_observation)
                        #create subtree for next belief
                        if belief_next and self.belief_space.distance(belief_next,timestep):
                            subtree = self.tree_extraction(belief_next,agent,timestep+1)
                            policy.add_subtree(belief_next,subtree)
                        # else:print("no further viable beliefs")
        policy.data.append(DR.individual[agent])
        policy.data.append(max)
        return policy
    
############################################################################################################     

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
        # print(f"network : ")
       
       
    
         

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
        