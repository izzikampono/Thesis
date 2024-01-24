import numpy as np
import pandas as pd
import random
from decpomdp import DecPOMDP
import matplotlib.pyplot as plt
from docplex.cp.model import CpoModel as Model
from docplex.cp.config import context
import sys


horizon = None
problem = DecPOMDP("dectiger", 1,horizon=1)
states = None
actions = None
joint_actions = None
joint_observations = None
transition_function = None
observation_function = None



def set_problem(problem_,horizon_):
    global horizon,problem,states,actions,joint_actions,joint_observations,transition_function,observation_function
    horizon = horizon_
    problem = problem_
    states = [i for i in range(len(problem.states))]
    actions = [[i for i in range(len(problem.actions[0]))],[j for j in range(len(problem.actions[1]))]]
    joint_actions = [i for i in range(len(problem.joint_actions))]
    joint_observations = [i for i in range(len(problem.joint_observations))]
    transition_function = problem.transition_fn
    problem.reset()



def print_nested_dict(d, depth=0, parent_key=None, last_child=False):
    if parent_key is not None:
        prefix = "  " * (depth - 1) + ("└─ " if last_child else "├─ ")
        print(prefix + str(parent_key) + ":")
    
    for idx, (key, value) in enumerate(d.items()):
        is_last = idx == len(d) - 1
        if isinstance(value, dict):
            print_nested_dict(value, depth + 1, key, last_child=is_last)
        else:
            prefix = "  " * depth + ("└─ " if is_last else "├─ ")
            print(prefix + str(key) + ": " + str(value))

def generate_probability_distribution(length):
    #generate random probabilities
    probabilities = np.random.rand(length)

    #normalize to make the sum equal to 1
    probabilities /= probabilities.sum()

    return probabilities

#function to generate individual decision rule 
def generate_sample_actions(n):
    samples=[]
    for j in range(n):
        samples.append(generate_probability_distribution(3))
    return np.array(samples)

#function to generate joint decision rule 
def generate_sample_joint_actions(n):
    samples=[]
    l=0
    for j in range(n):
        samples.append(generate_probability_distribution(9))
    return np.array(samples)

#function to generate sample beliefs
def generate_sample_belief(n):
    samples=[]
    for j in range(n):
        samples.append(generate_probability_distribution(2))
    return np.array(samples)

def initialize_rewards():
    #Competitive reward matrix indexed by joint actions
    comp_reward_1 = np.reshape([ [-1,25,-10,-50,-50,-60,20,45,20],
                  [-1,-10,25,20,20,45,-50,-60,-50]],
                  (9,2) 
    )
    comp_reward_2 = np.reshape([ [-1,-50,20,25,-50,45,-10,-60,20],
                  [-1,20,-50,-10,20,-60,25,45,-50]],
                   (9,2)

    )
    rewards = { "cooperative" : [problem.reward_fn_sa,problem.reward_fn_sa],
                "zerosum" : [problem.reward_fn_sa,problem.reward_fn_sa*-1],
                "stackelberg" :[comp_reward_1,comp_reward_2]
                }
    return rewards

rewards = initialize_rewards()

def normalize(vector):
    vector = np.array(vector) / np.sum(vector)
    return vector


def  observation_probability(joint_observation,belief,joint_action):
    """function to calculate prob of an observation given a belief and joint action uj"""
    sum=0
    for state in states:
        for next_state in states:
                sum += belief[state]  * transition_function[joint_action][state][next_state] * observation_function[joint_action][state][joint_observation]
    return sum



last_solution = None


def LP(Q1,Q2):
    
    milp = Model("tiger problem")
    # if (last_solution):milp.mip_start_sol(last_solution)
    if (last_solution):milp.set_starting_point(last_solution, complete_vars=True
)
    a1_0 , a1_1, a1_2 = milp.continuous_var_list(3,name = ["a1_0","a1_1","a1_2"],ub=1,lb=0)
    a2_0 , a2_1, a2_2 = milp.binary_var_list(3, name = ["a2_0","a2_1","a2_2"],ub=1)
    a0_00, a0_01 ,a0_02 ,a0_10 ,a0_11 ,a0_12,a0_20,a0_21,a0_22 = milp.continuous_var_list(9,name  = [f"a0_{i}{j}" for i in actions[0] for j in actions[1]],ub=1,lb=0)
    
    joint_DR = [a0_00,a0_01,a0_02,a0_10,a0_11,a0_12,a0_20,a0_21,a0_22]
    player1_DR = [a1_0 , a1_1, a1_2]
    player2_DR = [ a2_0 , a2_1, a2_2]



    # define objective function 
    obj_fn = 0
    for idx,a_ii in enumerate(joint_DR):
            obj_fn += Q1[idx]  * a_ii
    milp.maximize(obj_fn)

    # define constraints 

    # define lhs of linear equivalence expression equal to V^2(b0,a1,a2) :

    lhs = 0
    for idx,a_ii in enumerate(joint_DR):
            lhs += Q2[idx] * a_ii 

    # define rhs of linear equivalence expression equal to V^2(b0,a1,a2)
    for u2 in actions[1]:    
        rhs = 0       
        for a1_i in player1_DR:
                rhs += Q2[u2] * a1_i 
                u2 += 3 # used to select index of joint actions where u2 is consistent
        milp.add_constraint(lhs>=rhs)
        


    #add seperability constraints
    milp.add_constraint(a0_00 + a0_01 + a0_02 == a1_0)
    milp.add_constraint(a0_10 + a0_11 + a0_12 == a1_1)
    milp.add_constraint(a0_20 + a0_21 + a0_22 == a1_2)
    milp.add_constraint(a0_00 + a0_10 + a0_20 == a2_0)
    milp.add_constraint(a0_01 + a0_11 + a0_21 == a2_1)
    milp.add_constraint(a0_02 + a0_12 + a0_22 == a2_2)
    milp.add_constraint(a0_00 + a0_01 + a0_02+ a0_10+ a0_11 + a0_12+ a0_20+ a0_21 + a0_22 ==1)

    sol = milp.solve()
    last_solution = sol
    # print(f"value solution = {milp.solution.get_objective_value()}")
    return milp.solution.get_objective_value(),milp.solution.get_values(joint_DR), milp.solution.get_values(player1_DR), milp.solution.get_values(player2_DR)

def get_joint_DR(DR0,DR1):
    DR=np.zeros(len(joint_actions))
    for leader_action in actions[0]:
        for follower_action in actions[1]:
            DR[problem.get_joint_action(leader_action,follower_action)]= DR0[leader_action] * DR1[follower_action]

    return DR


def sota_strategy(P1,P2, game_type):
    if game_type=="zerosum":
        value0 , DR0 =  zerosum_lp_leader(P1)
        value1 , DR1 =  zerosum_lp_follower(P1)
        DR = get_joint_DR(DR0,DR1)
        print(f"sota_strategy  ::  value{0}={value0} and value{1}={value1}")
        return value0, DR,DR0,DR1
    if game_type=="stackelberg":
        return LP(P1,P2)
    if game_type=="cooperative":
        max = -np.inf
        optimal_joint_action = None
        for joint_action in joint_actions:
            if max<P1[joint_action]:
                max = P1[joint_action]
                optimal_joint_action = joint_action
        DR_joint =  np.identity(len(joint_actions))[optimal_joint_action]
        action_0, action_1 = problem.get_seperate_action(optimal_joint_action)
        DR0 = np.identity(len(actions[0]))[action_0]
        DR1 = np.identity(len(actions[1]))[action_1]
        return None,DR_joint,DR0,DR1


def zerosum_lp_leader(payoff):
    "linear program for SOTA of zerosum game"
    milp = Model("tiger problem")

    #initialize linear program variables
    DR = []
    V = []
    for action in actions[0]:
        DR.append(milp.continuous_var(name=f"a{0}_{action}",ub=1,lb=0))
    V = milp.continuous_var(name="V",ub=float('inf'),lb=float('-inf'))

    # define objective function 
    milp.maximize(V)

    # define constraints 
    for opponent_action in actions[1]:    
        rhs = 0   

        for agent_action, agent_action_probability in enumerate(DR):
            rhs += payoff[problem.get_joint_action(agent_action,opponent_action)] * agent_action_probability
        
        milp.add_constraint(V<=rhs)


    #add sum-to-one constraint
    value = 0
    for agent_action_probability in DR:
        value += agent_action_probability
    
    milp.add_constraint(value == 1)

    #solve and export 
    sol = milp.solve()
    milp.export_as_lp(f"zerosum_lp_{0}")

    print(f"Linear program solved :{(sol!=None)}")
    return milp.solution.get_objective_value(),milp.solution.get_values(DR)


def zerosum_lp_follower(payoff):
    "linear program for SOTA of zerosum game"
    milp = Model("tiger problem")

    #initialize linear program variables
    DR = []
    V = []
    for action in actions[1]:
        DR.append(milp.continuous_var(name=f"a{1}_{action}",ub=1,lb=0))
    V = milp.continuous_var(name="V",ub=float('inf'),lb=float('-inf'))

    # define objective function 
    milp.minimize(V)

    # define constraints 
    for opponent_action in actions[0]:    
        rhs = 0   

        for agent_action, agent_action_probability in enumerate(DR):
            rhs += payoff[problem.get_joint_action(opponent_action,agent_action)] * agent_action_probability
        
        milp.add_constraint(V>=rhs)


    #add sum-to-one constraint
    value = 0
    for agent_action_probability in DR:
        value += agent_action_probability
    
    milp.add_constraint(value == 1)

    #solve and export 
    sol = milp.solve()
    milp.export_as_lp(f"zerosum_lp_{1}")

    print(f"Linear program solved :{(sol!=None)}")
    return milp.solution.get_objective_value(),milp.solution.get_values(DR)




# Define Q function for blind attackers(player 2)
def Q2_blind(bt,u):
    """subroutine Q value for blind agent that only uses reward to approx Q value"""
    reward = rewards["stackelberg"][1]
    sum = 0
    for x in states :
        sum += bt[x] * reward[u][x]
    return sum
 

##########################################################################################################################################################################################################################################
#####################################################################################################################


    
    
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

        vectors = np.zeros((2,len(states)))
        for x in states:
            vectors[0][x] = 0
            vectors[1][x] = 0
            for u in joint_actions:
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
        global states
        self.horizon = horizon
        self.vector_sets = {}
        self.sota=sota
        self.initial_belief = initial_belief
        for timestep in range(horizon+1):
            self.vector_sets[timestep] = []
        vector = np.zeros(len(states))
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
    

                        
