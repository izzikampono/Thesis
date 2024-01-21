##  CLASS DEFINITIONS
import numpy as np
from decpomdp import DecPOMDP
from constant import Constants
from utilities import *
import random

CONSTANT = None
utilities = None
PROBLEM =None

def set_problem(prob):
    global CONSTANT,utilities,PROBLEM
    PROBLEM = prob
    CONSTANT = Constants(prob)
    utilities = Utilities(CONSTANT)
    print(f"problem set to {CONSTANT.NAME}")
    return




class AlphaVector:
    def __init__(self,DR,vector1,vector2,sota=False):
        self.DR = DR
        self.sota = sota
        self.vectors = [vector1,vector2]
        
        
    def get_value(self,belief):
        return np.dot(belief.value,self.vectors[0]),np.dot(belief.value,self.vectors[1])

    def print_vector(self):
        print(self.vectors)

    def set_value(self,agent,hidden_state,value):
        self.vectors[agent][hidden_state] = value

    def get_beta_two_d_vector(self,game_type):
        global REWARDS
        two_d_vectors = {}

        for agent in range(0,2):
            reward = CONSTANT.REWARDS[game_type][agent]
            two_d_vectors[agent] = np.zeros((len(CONSTANT.JOINT_ACTIONS),(len(CONSTANT.STATES))))
            # To improve efficiency in the cases of zerosum and cooperative games, we only need the first player payoffs, so we can skip the second player payoffs and provide the same for both players.  
            if game_type!="stackelberg" and agent==1 and self.sota==True :
                two_d_vectors[agent]=two_d_vectors[0]
                return BetaVector(two_d_vectors[0],two_d_vectors[1])
                
            for hidden_state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    two_d_vectors[agent][joint_action][hidden_state] = reward[joint_action][hidden_state]
                    
                    if game_type == "stackelberg"  and agent==1 and self.sota==True : 
                        continue #for the blind strategy of the stackelberg games
                    
                    for next_hidden_state in CONSTANT.STATES:
                        for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                            two_d_vectors[agent][joint_action][hidden_state]+= CONSTANT.TRANSITION_FUNCTION[joint_action][hidden_state][next_hidden_state] * CONSTANT.OBSERVATION_FUNCTION[joint_action][hidden_state][joint_observation]* self.vectors[agent][next_hidden_state]
                    
        # note : u can filter the zero probabilites out of the vector to reduce computational 

        return BetaVector(two_d_vectors[0],two_d_vectors[1])
    def payoff_function(self,belief,game_type):

        beta = self.get_beta_two_d_vector(game_type)


        payoffs = {}
        for agent in range(0,2):
            payoffs[agent] = np.zeros(len(CONSTANT.JOINT_ACTIONS))
            for joint_action in CONSTANT.JOINT_ACTIONS:
                for hidden_state in CONSTANT.STATES:
                    payoffs[agent][joint_action] += belief.value[hidden_state] * beta.two_d_vectors[agent][joint_action][hidden_state]
        return payoffs,beta
    

    def solve(self,belief,game_type):
        payoffs ,beta = self.payoff_function(belief,game_type)
        if self.sota==False :
            value , DR , DR0 , DR1 = utilities.LP(payoffs[0],payoffs[1])
        else:
            value, DR, DR0, DR1 = utilities.sota_strategy(payoffs[0],payoffs[1],game_type)
        alpha = beta.get_alpha_vector2(DecisionRule(DR0,DR1,DR),belief,sota2=self.sota)
        return value, alpha

    
    
################################################################################################

class DecisionRule:
    def __init__(self,a1,a2,aj):
        self.individual = {0:a1,1:a2}
        self.joint = aj

################################################################################################

class BetaVector:
    def __init__(self,two_d_vector_0,two_d_vector_1):
        self.two_d_vectors = [two_d_vector_0,two_d_vector_1]

    def get_alpha_vector(self,DR,sota2=False):

        vectors = np.zeros((2,len(CONSTANT.STATES)))
        for x in CONSTANT.STATES:
            vectors[0][x] = 0
            vectors[1][x] = 0
            for u in CONSTANT.JOINT_ACTIONS:
                joint_action_probability = DR.joint[u]
                vectors[0][x] += joint_action_probability * self.two_d_vectors[0][x][u]
                vectors[1][x] += joint_action_probability * self.two_d_vectors[1][x][u]

        return AlphaVector(DR,vectors[0],vectors[1],sota=sota2)
    
    def get_alpha_vector2(self,DR,belief,sota2=False):

        vectors = np.zeros((2,len(CONSTANT.STATES)))

        for agent in range(2):
            max = -np.inf
            for joint_action in CONSTANT.JOINT_ACTIONS:
                value=0
                for state in CONSTANT.STATES:
                    value += belief.value[state] * self.two_d_vectors[agent][joint_action][state]
                if value> max :
                    argmax = self.two_d_vectors[agent][joint_action]
                    max = value
            vectors[agent] = argmax
        
        alpha_vector = np.zeros((2,len(CONSTANT.STATES)))
        for agent in range(2):
            for u in CONSTANT.JOINT_ACTIONS:
                joint_action_probability = DR.joint[u]
                for x in CONSTANT.STATES:
                    alpha_vector[agent][x] += joint_action_probability * vectors[agent][x]


        
        return AlphaVector(DR,alpha_vector[0],alpha_vector[1],sota=sota2)


        
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
    def __init__(self,horizon, initial_belief,sota=False):
        self.horizon = horizon
        self.vector_sets = {}
        self.sota=sota
        self.initial_belief = Belief(np.array(initial_belief),None,None)
        for timestep in range(horizon+2):
            self.vector_sets[timestep] = []
        vector = np.zeros(len(CONSTANT.STATES))
        self.add_alpha_vector(AlphaVector(0,vector,vector,self.sota),horizon+1)
    
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
    
    def backup(self,belief,timestep,gametype):
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

        
    
    # def tree_extraction(self,belief, agent,timestep):
    #     global utilities
    #     # edge case at last horizon
    #     if timestep > self.horizon : return PolicyTree(None)

    #     #initialize policy and DR_bt
    #     policy = PolicyTree(None)
    #     DR = DecisionRule(None,None,None)

    #     max = -np.inf

    #     # extract decision rule of agent from value function at current belief
    #     for alpha in self.vector_sets[timestep]:
    #         value = alpha.get_value(belief)
    #         if (max<value[agent]) :
    #             max = value[agent]
    #             DR = alpha.DR
    #     print(f"max value at timestep {timestep} = {max}")
    #     for u_agent in CONSTANT.ACTIONS[agent]:
    #         #if probability of action is not 0
    #         if DR.individual[agent][u_agent] > 0:

    #             #get actions of the other agent
    #             for u_not_agent in CONSTANT.ACTIONS[int(not agent)]:

    #                 joint_action = CONSTANT.PROBLEM.get_joint_action(u_agent,u_not_agent)
    #                 for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
    #                     belief_next = belief.next_belief(joint_action,joint_observation)
    #                     subtree = self.tree_extraction(belief_next,agent,timestep+1)
    #                     policy.add_subtree(belief_next,subtree)
    #     policy.data.append(DR.individual[agent])
    #     policy.data.append(max)
    #     return policy
    
   
    
    def get_values_initial_belief(self):
        values_leader,values_follower = [],[]
        for alpha in self.vector_sets[0]:
            value_leader,value_follower = alpha.get_value(self.initial_belief)
            values_leader.append(value_leader)
            values_follower.append(value_follower)
        return values_leader,values_follower
    


################################################################################################

class PBVI:
    def __init__(self,problem,horizon,density,gametype,sota=False):
        self.sota = sota
        self.value_function = ValueFunction(horizon,problem.b0,sota=self.sota)
        self.belief_space = BeliefSpace(horizon,problem.b0,density)
        self.policies = [PolicyTree(None),PolicyTree(None)]
        self.gametype = gametype
        self.problem = problem
        self.horizon = horizon
        self.density = density



    def backward_induction(self):
        for timestep in range(self.horizon,-1,-1):
            for belief in self.belief_space.belief_states[timestep]:
                self.value_function.backup(belief,timestep,self.gametype)
            print(f"\tbackup at timestep {timestep} done")

        print("\tbackward induction done")
           
    def solve(self,iterations,decay):
        self.belief_space.expansion()
        for _ in range(1,iterations+1):
            print(f"iteration : {_}")
            self.backward_induction()
            self.density /= decay #hyperparameter
        initial_belief = self.belief_space.get_inital_belief()
        self.policies[0] = self.tree_extraction(initial_belief,agent=0,timestep = 0)    
        self.policies[1] = self.tree_extraction(initial_belief,agent=1,timestep =0)  
        return self.policies   
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
                        if self.belief_space.distance(belief_next,timestep):
                            subtree = self.tree_extraction(belief_next,agent,timestep+1)
                            policy.add_subtree(belief_next,subtree)
                        # else:print("no further viable beliefs")
        policy.data.append(DR.individual[agent])
        policy.data.append(max)
        return policy
    
############################################################################################################     

class BeliefSpace:
    def __init__(self,horizon,initial_belief,density):
        self.density = density
        self.belief_states = {} 
        for timestep in range(horizon+1):
            self.belief_states[timestep] = []
        self.belief_states[0].append(Belief(initial_belief,None,None))
        self.horizon = horizon
    def get_inital_belief(self):
        return self.belief_states[0][0]


    def distance(self,belief,timestep):
        """function to check if a new belief point is "sufficiently different from other points in the bag of beliefs """
        if len(self.belief_states[timestep])<=0: return True
        belief_states = np.array(self.belief_states[timestep])
        min_belief = min(belief_states, key=lambda stored_belief: np.linalg.norm(stored_belief.value-belief.value))
        min_magnitude = np.linalg.norm(min_belief.value-belief.value)
        return min_magnitude > self.density
        # check what happens if there are no stored beliefs at timestep
    
    def get_closest_belief(self,belief,timestep):
        """ returns belief state at timestep t that is closest in distance to the input belief """
        max = -np.inf
        # random.sample(beliefs_t, len(beliefs_t))
        for belief_t in self.belief_states[timestep].keys():
            distance = np.abs(np.linalg.norm(np.array(belief.value) - np.array(belief_t.value)))
            if distance<=max: 
                next = belief_t
                max = distance
        if next: return next
        else : print("err0r : no belief found")
        

    def belief_size(self):
        size = 0
        for timestep in range(self.horizon+1):
            size+=len(self.belief_states[timestep])
        return size

    def expansion(self):
        """populates self.belief_state table"""
        for timestep in range(1,self.horizon+1):
            for previous_belief in self.belief_states[timestep-1]:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        belief = previous_belief.next_belief(joint_action,joint_observation)
                        if self.distance(belief,timestep):
                            self.belief_states[timestep].append(belief)
                            # print(f"belief point added at timestep {timestep}: {belief}")
        print("\tbelief expansion done")  
    
         

class Belief:
    def __init__(self,value,action_label,observation_label):
        self.value = value 
        self.action_label = action_label
        self.observation_label = observation_label

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
            return Belief(next_belief_value,joint_DR,joint_observation)
        next_belief_value = utilities.normalize(next_belief_value)

        if np.sum(next_belief_value)<= 1.001 and np.sum(next_belief_value)> 0.99999:
            return  Belief(next_belief_value,joint_DR,joint_observation)
        else:
            print("err0r : belief doesn not sum up to 1\n")
            print(f"current belief: \n{self.value}")
            print(f"next belief :\n{next_belief_value}")
            print(f"sum : {np.sum(next_belief_value)}")
            sys.exit()
        return np.array(next_belief)   