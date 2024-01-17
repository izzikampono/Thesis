##  CLASS DEFINITIONS
import numpy as np
from decpomdp import DecPOMDP
from constant import Constants
import random

problem = None
utilities = None

def set_problem(constant,utils):
    global problem,utilities
    problem = constant
    utilities = utils
    print(f"problem set to problem={constant.NAME}")
    return




class AlphaVector:
    def __init__(self,DR,vector1,vector2,sota=False):
        self.DR = DR
        self.sota = sota
        self.vectors = [vector1,vector2]
        
        
    def get_value(self,belief):
        return np.dot(belief,self.vectors[0]),np.dot(belief,self.vectors[1])

    def print_vector(self):
        print(self.vectors)

    def set_value(self,agent,hidden_state,value):
        self.vectors[agent][hidden_state] = value

    # TODO  To improve efficiency ... in the cases of zerosum and cooperative games, we only need the first player payoffs, so we can skip the second player payoffs and provide the same for both players.  
    def get_beta_two_d_vector(self,game_type):
        global REWARDS
        two_d_vectors = {}

        for agent in range(0,2):
            reward = problem.REWARDS[game_type][agent]
            two_d_vectors[agent] = np.zeros((len(problem.STATES),len(problem.JOINT_ACTIONS)))
            for hidden_state in problem.STATES:
                for joint_action in problem.JOINT_ACTIONS:
                    two_d_vectors[agent][hidden_state][joint_action] = reward[joint_action][hidden_state]
                    
                    if game_type == "stackelberg" and self.sota==True and agent==1 : 
                        continue #for the blind strategy of the stackelberg games
                    
                    for next_hidden_state in problem.STATES:
                        for joint_observation in problem.JOINT_OBSERVATIONS:
                            two_d_vectors[agent][hidden_state][joint_action] += problem.TRANSITION_FUNCTION[joint_action][hidden_state][next_hidden_state] * problem.OBSERVATION_FUNCTION[joint_action][hidden_state][joint_observation]* self.vectors[agent][next_hidden_state]
                    
        # note : u can filter the zero probabilites out of the vector to reduce computational 

        return BetaVector(two_d_vectors[0],two_d_vectors[1])

    def payoff_function(self,belief,game_type):

        beta = self.get_beta_two_d_vector(game_type)
        payoffs = {}
        for agent in range(0,2):
            payoffs[agent] = np.zeros(len(problem.JOINT_ACTIONS))
            for joint_action in problem.JOINT_ACTIONS:
                for hidden_state in problem.STATES:
                    payoffs[agent][joint_action] += belief[hidden_state] * beta.two_d_vectors[agent][hidden_state][joint_action]
            
        return payoffs,beta

    def solve(self,belief,game_type):
        payoffs ,beta = self.payoff_function(belief,game_type)
        if self.sota==False :
            value , DR , DR0 , DR1 = utilities.LP(payoffs[0],payoffs[1])
        else:
            # print("HERE IN SOTA IF STATEMENT")
            value, DR, DR0, DR1 = utilities.sota_strategy(payoffs[0],payoffs[1],game_type)
        alpha = beta.get_alpha_vector(DecisionRule(DR0,DR1,DR))
        if (alpha.sota !=self.sota):
            alpha.sota = self.sota
        return value, alpha

    
    
################################################################################################

class DecisionRule:
    def __init__(self,a1,a2,aj):
        self.agents = {0:a1,1:a2}
        self.joint = aj

################################################################################################

class BetaVector:
    def __init__(self,two_d_vector_0,two_d_vector_1):
        self.two_d_vectors = [two_d_vector_0,two_d_vector_1]

    def get_alpha_vector(self,DR,sota2=False):

        vectors = np.zeros((2,len(problem.STATES)))
        for x in problem.STATES:
            vectors[0][x] = 0
            vectors[1][x] = 0
            for u in problem.JOINT_ACTIONS:
                joint_action_probability = DR.joint[u]
                vectors[0][x] += joint_action_probability * self.two_d_vectors[0][x][u]
                vectors[1][x] += joint_action_probability * self.two_d_vectors[1][x][u]

        return AlphaVector(DR,vectors[0],vectors[1],sota=sota2)

        
    def print_vector(self):
        print(self.two_d_vectors)



################################################################################################

class PolicyTree:
    def __init__(self, data):
        self.data = data
        self.subtrees = {}

    def add_subtree(self,key,subtree):
        self.subtrees[tuple(key)] = subtree   

    def print_trees(self, indent=0):
        print(" " * indent + str(self.data))
        for key, subtree in self.subtrees.items():
            print(" " * (indent + 2) + f"belief : {key}")
            subtree.print_trees(indent + 4)
    
        

################################################################################################
            
class ValueFunction:
    def __init__(self,horizon, initial_belief,sota=False):
        self.horizon = horizon
        self.vector_sets = {}
        self.sota=sota
        self.initial_belief = initial_belief
        for timestep in range(horizon+1):
            self.vector_sets[timestep] = []
        vector = np.zeros(len(problem.STATES))
        self.add_alpha_vector(AlphaVector(None,vector,vector,self.sota),horizon)
    
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
                print("ERROR")
            if value>max:
                max = value
                max_alpha = _alpha
        if max_alpha == None:
            print(f"time : {timestep}")
            print(f"max_alpha = {max_alpha}")
            print(f"size : {len(self.vector_sets[timestep+1])}")
            return
        self.add_alpha_vector(max_alpha,timestep)

        
    
    def tree_extraction(self,belief, agent,timestep):

        # edge case at last horizon
        if timestep == self.horizon : return PolicyTree(None)

        #initialize policy and DR_bt
        policy = PolicyTree(None)
        DR = DecisionRule(None,None,None)

        max = -np.inf

        # extract decision rule of agent from value function at current belief
        for alpha in self.vector_sets[timestep]:
            value = alpha.get_value(belief)
            if (max<value[agent]) :
                max = value[agent]
                DR = alpha.DR

        for u_agent in problem.ACTIONS[agent]:
            #if probability of action is not 0
            if DR.agents[agent][u_agent] > 0:

                #get actions of the other agent
                for u_not_agent in problem.ACTIONS[int(not agent)]:

                    joint_action = problem.PROBLEM.get_joint_action(u_agent,u_not_agent)
                    for joint_observation in problem.JOINT_OBSERVATIONS:
                        belief_next = utilities.next_belief(belief,joint_action,joint_observation)
                        subtree = self.tree_extraction(belief_next,agent,timestep+1)
                        policy.add_subtree(belief_next,subtree)
                    
        policy.data = DR.agents[agent]
        return policy
    
   
    
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
           
    def solve(self,iterations,decay):
        for _ in range(iterations):
            self.belief_space.expansion()
            self.backward_induction()
            self.density /= decay #hyperparameter
        self.policies[0] = self.value_function.tree_extraction(self.problem.b0,0,0)    
        self.policies[1] = self.value_function.tree_extraction(self.problem.b0,1,0)  
        return self.policies   
    
############################################################################################################     

class BeliefSpace:
    def __init__(self,horizon,initial_belief,density):
        self.density = density
        self.belief_states = {}
        for timestep in range(horizon+1):
            self.belief_states[timestep] = []
        self.belief_states[0] = [initial_belief]
        self.horizon = horizon


    def distance(self,belief,timestep):
        """function to check if a new belief point is "sufficiently different from other points in the bag of beliefs """
        if len(self.belief_states[timestep])<=0: return True
        belief_states = np.array(self.belief_states[timestep])
        min_belief = min(belief_states, key=lambda stored_belief: np.linalg.norm(stored_belief-belief))
        min_magnitude = np.linalg.norm(min_belief-belief)
        return min_magnitude > self.density
        # check what happens if there are no stored beliefs at timestep
    
    def get_closest_belief(self,belief,timestep):
        """ returns belief state at timestep t that is closest in distance to the input belief """
        max = -np.inf
        # random.sample(beliefs_t, len(beliefs_t))
        for belief_t in self.belief_states[timestep].keys():
            distance = np.abs(np.linalg.norm(np.array(belief) - np.array(belief_t)))
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
        for timestep in range(1,self.horizon):
            for previous_belief in self.belief_states[timestep-1]:
                for joint_action in problem.JOINT_ACTIONS:
                    for joint_observation in problem.JOINT_OBSERVATIONS:
                        belief = utilities.next_belief(previous_belief,joint_action,joint_observation)
                        if self.distance(belief,timestep):
                            self.belief_states[timestep].append(belief)
                          

                        