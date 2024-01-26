##  CLASS DEFINITIONS

import numpy as np
from constant import Constants
from utilities import *
import random
import gc

gc.enable()

CONSTANT = None
utilities = None
PROBLEM = None

def set_problem(prob):
    global CONSTANT,utilities,PROBLEM
    PROBLEM = prob
    CONSTANT = Constants(prob)
    utilities = Utilities(CONSTANT)
    # print(f"PROBLEM SET T0 ==> {CONSTANT.NAME}")
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

    def get_beta_future_value(self,agent,joint_action,joint_observation):
        value = np.zeros(len(CONSTANT.STATES))
        for next_state in CONSTANT.STATES:
            value += CONSTANT.TRANSITION_FUNCTION[joint_action][next_state] * CONSTANT.OBSERVATION_FUNCTION[joint_action][joint_observation]* self.vectors[agent][next_state]
        return value
            
    # note : u can filter the zero probabilites out of the vector to reduce computational 
   
    

    
    
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
        for joint_action in CONSTANT.JOINT_ACTIONS:
            if DR.joint[joint_action]>0:
                vectors[CONSTANT.LEADER] += DR.joint[joint_action] * self.two_d_vectors[CONSTANT.LEADER][joint_action]
                vectors[CONSTANT.FOLLOWER] += DR.joint[joint_action] * self.two_d_vectors[CONSTANT.FOLLOWER][joint_action]

        return AlphaVector(DR,vectors[0],vectors[1],sota=sota2)
    
        
    def payoff_function(self,belief,game_type):

        payoffs = {}
        for agent in range(0,2):
            payoffs[agent] = np.zeros(len(CONSTANT.JOINT_ACTIONS))
            # if statement for efficiency, since Linear Program only uses first player payoff for cooperative and zerosum games
            if game_type!="stackelberg" and agent==CONSTANT.FOLLOWER:
                payoffs[CONSTANT.FOLLOWER] = payoffs[CONSTANT.LEADER]
                return payoffs
            for joint_action in CONSTANT.JOINT_ACTIONS:
                payoffs[agent][joint_action] = np.linalg.norm(belief.value * self.two_d_vectors[agent][joint_action])
        return payoffs
    
    def solve(self,belief,game_type,sota):
        payoffs = self.payoff_function(belief,game_type)
        if sota==False :
            value , DR , DR0 , DR1 = utilities.LP(payoffs[0],payoffs[1])
        else:
            value, DR, DR0, DR1 = utilities.sota_strategy(payoffs[0],payoffs[1],game_type)
        alpha = self.get_alpha_vector(DecisionRule(DR0,DR1,DR),sota)
        return value, alpha
    
        
    def print_vector(self):
        print(self.two_d_vectors)



################################################################################################

class PolicyTree:
    def __init__(self, data_):
        self.DR = None
        self.value = None
        if data_:
            self.data.append(data_)
        
        self.subtrees = {}

    def add_subtree(self,key,subtree):
        if subtree != PolicyTree(None):
            self.subtrees[key] = subtree   

    def print_trees(self, indent=0):
        print(" " * indent + "Decision Rule: " +str(self.data[0]) + ", value: " +str(self.data[1]))
        for key, subtree in self.subtrees.items():
            if (subtree.value):
                print(" " * (indent + 2) + "└─ "+ f"belief : {key.value}")
                subtree.print_trees(indent + 5)
        return
    
    def next(self,joint_action,joint_observation):
        for belief in self.subtrees.keys():
            if belief.action_label == joint_action and belief.observation_label==joint_observation:
                return self.subtrees[belief]
        return None
    def max_DR(self,belief):
        max = -np.inf
        for belief_t,subtree in self.subtrees.items():
            if np.array_equal(belief.value,belief_t.value) and subtree.value>max:
                max = subtree.value
                max_DR = subtree.DR 
                branch = subtree
        if max<=-np.inf:
            print("no branch found")
            return
        self = branch
        return max_DR
        

    
        

################################################################################################
            
class ValueFunction:
    def __init__(self,horizon, belief_space,sota=False):
        self.horizon = horizon
        self.vector_sets = {}
        self.sota=sota
        self.beliefs = belief_space
        for timestep in range(horizon+2):
            self.vector_sets[timestep] = []
        vector = np.zeros(len(CONSTANT.STATES))
        self.add_alpha_vector(AlphaVector(0,vector,vector,self.sota),horizon+1)
    
    def add_alpha_vector(self,alpha,timestep):
        self.vector_sets[timestep].append(alpha)

    def pruning(self,timestep):
        self.vector_sets[timestep] = set(self.vector_sets[timestep])
    
    def get_max_alpha_vector(self, belief,timestep):
        max = -np.inf
        max_alpha = None
        for alpha in self.vector_sets[timestep]:
            leader_value,follower_value = alpha.get_value(belief)
            # print(f"alpha vector value at timestep {timestep} t : {value}")
            if leader_value>max:
                max = leader_value
                max_alpha = alpha
        return max,max_alpha
    
    def create_beta_vector(self,belief,timestep,gametype):
        two_d_vectors = {}
        for agent in range(0,2):
        #get reward matrix for agent
            reward = CONSTANT.REWARDS[gametype][agent]

            #initialize beta vector space (by action space first)
            two_d_vectors[agent] = np.zeros((len(CONSTANT.JOINT_ACTIONS),len(CONSTANT.STATES)))

            # To improve efficiency in the cases of zerosum and cooperative games, we only need the first player payoffs, so we can skip the second player payoffs and provide the same for both players.  
            if gametype!="stackelberg" and agent==CONSTANT.FOLLOWER and self.sota==True :
                two_d_vectors[agent]=two_d_vectors[0]
                return BetaVector(two_d_vectors[0],two_d_vectors[1])
                
            for joint_action in CONSTANT.JOINT_ACTIONS:
                two_d_vectors[agent][joint_action] += reward[joint_action]
                # if statement for blind agents
                if gametype=="stackelberg" and agent==CONSTANT.FOLLOWER and self.sota==True:
                    continue
                for joint_observation in CONSTANT.JOINT_OBSERVATIONS:                        
                    next_belief = belief.next_belief(joint_action,joint_observation)
                    _, max_alpha = self.get_max_alpha_vector(next_belief,timestep+1)
                    two_d_vectors[agent][joint_action] += max_alpha.get_beta_future_value(agent,joint_action,joint_observation)

        return BetaVector(two_d_vectors[0],two_d_vectors[1])
    

    def backup(self,belief,timestep,gametype):
        beta = self.create_beta_vector(belief,timestep,gametype)
        value , alpha = beta.solve(belief,gametype,self.sota)
        self.add_alpha_vector(alpha,timestep)

                       
        # note : u can filter the zero probabilites out of the vector to reduce computational 

    
    def get_values_initial_belief(self):
        values_leader,values_follower = [],[]
        for alpha in self.vector_sets[0]:
            value_leader,value_follower = alpha.get_value(self.beliefs.initial_belief)
            values_leader.append(value_leader)
            values_follower.append(value_follower)
        if len(values_leader)<=1 or len(values_follower)<1:
            return value_leader,value_follower
        else: return np.average(value_leader),np.average(value_follower)
    


################################################################################################

class PBVI:
    def __init__(self,problem,horizon,density,gametype,sota=False):
        set_problem(problem)
        self.sota = sota
        self.belief_space = BeliefSpace(horizon,problem.b0,density)
        self.value_function = ValueFunction(horizon,self.belief_space,sota=self.sota)

        self.policy = []
        self.gametype = gametype
        self.problem = problem
        self.horizon = horizon
        self.density = density

        self.leader_b0_values = []
        self.follower_b0_values = []


    def backward_induction(self):
        for timestep in range(self.horizon,-1,-1):
            for belief in self.belief_space.belief_states[timestep]:
                self.value_function.backup(belief,timestep,self.gametype)
            print(f"\tbackup at timestep {timestep} done")

        print("\tbackward induction done")
    
    def extract_policies(self,belief_space):
        self.policy.append(self.tree_extraction(self.belief_space.initial_belief,agent = 0,timestep = 0,belief_space = belief_space))
        self.policy.append(self.tree_extraction(self.belief_space.initial_belief,agent = 1,timestep = 0,belief_space = belief_space))
        print(f"policy extraction done")

   
    def DP(self, leader_policy, follower_policy,belief = None):
        #check if theres a DR
        if leader_policy==None or follower_policy==None:return 0

        #if belief is NONE, we set it to initial belief
        if not belief : belief =self.belief_space.initial_belief

        #initialize value and reward
        value = 0 
        reward = CONSTANT.REWARDS[self.gametype][CONSTANT.LEADER]
        #get reward component of value function by weighting reward with DR of both players and current belief
        for state in CONSTANT.STATES:
            for leader_action in CONSTANT.ACTIONS[CONSTANT.LEADER]:
                for follower_action in CONSTANT.ACTIONS[CONSTANT.FOLLOWER]:
                    joint_action = self.problem.get_joint_action(leader_action,follower_action)
                    value += belief.value[state]*leader_policy.DR[leader_action]*follower_policy.DR[follower_action]*reward[joint_action][state]
        #get future component of value function by getting optimal value of "next belief" weighted by probability of observation of subsequent "next belief"
        for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
            for leader_action in CONSTANT.ACTIONS[CONSTANT.LEADER]:
                for follower_action in CONSTANT.ACTIONS[CONSTANT.FOLLOWER]:
                    value+=utilities.observation_probability(joint_observation,belief,joint_action) * leader_policy.DR[leader_action] * follower_policy.DR[follower_action] * self.DP(leader_policy.next(leader_action,joint_observation),follower_policy.next(follower_action,joint_observation),belief.next_belief(joint_action,joint_observation))
        return value
           
    def solve(self,iterations,growth):
        self.belief_space.reset()
        for _ in range(1,iterations+1):
            print(f"iteration : {_}")
            self.belief_space.expansion()
            self.backward_induction()
            leader_value , follower_value = self.value_function.get_values_initial_belief()
            self.density *= growth #hyperparameter
            self.leader_b0_values.append(leader_value) 
            self.follower_b0_values.append(follower_value) 
        print(f"    extracting policy...")
        initial_belief = self.belief_space.get_inital_belief()
        self.extract_policies(self.belief_space)
        return self.policy , self.leader_b0_values,self.follower_b0_values
    
    def build_comparison_matrix(self,policy_comparison_matrix,policies):
        sota = False
        for gametype in ["cooperative","zerosum","stackelberg"]:
            strong_leader_strong_follower = policies[gametype][int(sota)][CONSTANT.LEADER].value
            weak_leader_weak_follower = policies[gametype][int(not sota)][CONSTANT.LEADER].value
            strong_leader_weak_follower = self.DP(leader_policy=policies[gametype][int(sota)][CONSTANT.LEADER],follower_policy=policies[gametype][int(not sota)][CONSTANT.FOLLOWER])
            weak_leader_strong_follower = self.DP(leader_policy=policies[gametype][int(not sota)][CONSTANT.LEADER],follower_policy=policies[gametype][int(sota)][CONSTANT.FOLLOWER])
            policy_comparison_matrix[gametype] = np.array([[strong_leader_strong_follower,strong_leader_weak_follower],[weak_leader_strong_follower,weak_leader_weak_follower]])
        print("Calculated comparison matrix")
        return
    
    def tree_extraction(self,belief, agent,timestep,belief_space):
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
                    if DR.joint[joint_action]>0:
                        for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                            #get next belief of joint action and observation
                            belief_next = belief.next_belief(joint_action,joint_observation)
                            #create subtree for next belief
                            # if belief_space.distance(belief_next,timestep):
                            subtree = self.tree_extraction(belief_next,agent,timestep+1,belief_space)
                            policy.add_subtree(belief_next,subtree)
        
        if max > -np.inf : 
            policy.DR = DR.individual[agent]
            policy.value = max

        return policy
    
############################################################################################################     

class BeliefSpace:
    def __init__(self,horizon,initial_belief,density):
        self.density = density
        self.belief_states = {} 
        if type(initial_belief)!=Belief:self.initial_belief=Belief(initial_belief,None,None)
        else : self.initial_belief = initial_belief
        for timestep in range(horizon+1):
            self.belief_states[timestep] = []
        self.belief_states[0].append(self.initial_belief)
        self.horizon = horizon

    def get_inital_belief(self):
        return self.belief_states[0][0]
    
    def reset(self):
        for timestep in range(self.horizon+1):
            self.belief_states[timestep] = []
        self.belief_states[0].append(self.initial_belief)
        return




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
        print("\tbelief expansion done")  
    
         

class Belief:
    def __init__(self,value,action_label,observation_label):
        self.value = value 
        self.action_label = action_label
        self.observation_label = observation_label

    def next_belief(self,joint_action,joint_observation):
        """function to calculate next belief based on current belief, DR/joint action , and observation"""
        # returns the value of b1
        next_belief_value= np.zeros(len(CONSTANT.STATES))

        if type(joint_observation) != int :
            joint_observation = CONSTANT.PROBLEM.joint_observations.index(joint_observation)


        if type(joint_action) == int: # if joint_DR enterred as a deterministic action 
            for next_state in CONSTANT.STATES:
                next_belief_value[next_state]+= np.linalg.norm(self.value * CONSTANT.TRANSITION_FUNCTION[joint_action][next_state]  * CONSTANT.OBSERVATION_FUNCTION[joint_action][joint_observation])

        else:   # if joint_DR is a decision rule
            for next_state in CONSTANT.STATES:
                value = 0
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    value += self.value * joint_action[joint_action] * CONSTANT.TRANSITION_FUNCTION[joint_action][next_state]  * CONSTANT.OBSERVATION_FUNCTION[joint_action][joint_observation]
                next_belief_value[next_state]+=value
        
        if np.sum(next_belief_value) ==0 :
            return Belief(next_belief_value,joint_action,joint_observation)
        next_belief_value = utilities.normalize(next_belief_value)

        if np.sum(next_belief_value)<= 1.001 and np.sum(next_belief_value)> 0.99999:
            return  Belief(next_belief_value,joint_action,joint_observation)
        else:
            print("err0r : belief doesn not sum up to 1\n")
            print(f"current belief: \n{self.value}")
            print(f"next belief :\n{next_belief_value}")
            print(f"sum : {np.sum(next_belief_value)}")
            sys.exit()
        return np.array(next_belief)   