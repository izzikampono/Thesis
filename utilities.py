from warnings import catch_warnings
import numpy as np
import pandas as pd
import random
import warnings
from decpomdp import DecPOMDP
import matplotlib.pyplot as plt
from docplex.mp.model import Model
# import cplex
import sys
import subprocess
from constant import Constants


CONSTANT = Constants.get_instance()
PROBLEM = CONSTANT.PROBLEM
    


def install_dependencies():
    try:
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install dependencies. {e}")
    return

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

def normalize(vector):
    warnings.filterwarnings("error", category=RuntimeWarning)

    try:
        vector = np.array(vector) / np.sum(vector)
        return vector
    except RuntimeWarning as rw:
        print(f"RuntimeWarning: {rw}")
        print(f"cannot normalize vector V: {vector}")
        sys.exit


def  observation_probability(joint_observation,belief,joint_action):
    """function to calculate prob of an observation given a belief and joint action uj"""
    sum=0
    for state in CONSTANT.STATES:
        for next_state in CONSTANT.STATES:
                sum += belief.value[state]  * CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state] * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation]
    return sum




def MILP(beta,belief):
    """returns DR of joint and follower in the form of DR(action|state)"""

    milp = Model(f"{CONSTANT.PROBLEM.name} problem")

    #initalize MILP variables 
    leader_DR = milp.continuous_var_list(len(CONSTANT.ACTIONS[0]),name = [f"a0_u{i}" for i in CONSTANT.ACTIONS[0]],ub=1,lb=0)
    follower_DR = {}
    joint_DR = {}
    for state in CONSTANT.STATES:
        follower_DR[state] = milp.binary_var_list(len(CONSTANT.ACTIONS[1]), name = [f"a1_u{i}_x{state}" for i in CONSTANT.ACTIONS[1]])
        joint_DR[state] = milp.continuous_var_list(len(CONSTANT.JOINT_ACTIONS),name  = [f"aj_u{i}_x{state}" for i in CONSTANT.JOINT_ACTIONS],ub=1,lb=0)


    # objective function :: maximize V^1(b,a^1,a^2)
    obj_fn = 0
    for state in CONSTANT.STATES:
        for joint_action, joint_action_probability in enumerate(joint_DR[state]):
                obj_fn += belief.value[state] * beta.two_d_vectors[0][state][joint_action]  * joint_action_probability 
    milp.maximize(obj_fn)

    # Constraints (subject to) :: 

    # define lhs of linear equivalence expression equal to V^2(x,a1,a2) - without the belief value 
    lhs = {}
    for state in CONSTANT.STATES:
        lhs[state] = 0
        for follower_action,follower_action_probability in enumerate(follower_DR[state]):
            for leader_action, leader_action_probability in enumerate(leader_DR):
                joint_action = CONSTANT.PROBLEM.get_joint_action(leader_action,follower_action)
                lhs[state] += beta.two_d_vectors[1][state][joint_action]  * joint_DR[state][joint_action]

    # define rhs of linear equivalence expression equal to  V^2(x,a1,u2) using loop
    # for every state and follower action, we add the constraint V^2(x,a1,a2) >= V^2(x,a1,u2) to the milp
    for state in CONSTANT.STATES:
        for follower_action,follower_action_probability in enumerate(follower_DR[state]):
            rhs = 0
            for leader_action, leader_action_probability in enumerate(leader_DR):
                joint_action = CONSTANT.PROBLEM.get_joint_action(leader_action,follower_action)
                rhs += beta.two_d_vectors[1][state][joint_action] * leader_action_probability
            milp.add_constraint(lhs[state]>=rhs)


    ## sum to 1 constraint for all Decision Rules
    milp.add_constraint(milp.sum(leader_DR)==1)
    for state in CONSTANT.STATES:
        milp.add_constraint(milp.sum(follower_DR[state])==1)
        milp.add_constraint(milp.sum(joint_DR[state])==1)

    ## seperability constraints for joint_DRs and singular_DRs 
    for state in CONSTANT.STATES:
        for leader_action, leader_action_probability in enumerate(leader_DR):
            sum = 0
            for follower_action,follower_action_probability in enumerate(follower_DR[state]):
                joint_action = PROBLEM.get_joint_action(leader_action,follower_action)
                sum += joint_DR[state][joint_action]
            milp.add_constraint(sum == leader_action_probability)
    for state in CONSTANT.STATES:
        for follower_action,follower_action_probability in enumerate(follower_DR[state]):
            sum = 0
            for leader_action, leader_action_probability in enumerate(leader_DR):
                joint_action = PROBLEM.get_joint_action(leader_action,follower_action)
                sum += joint_DR[state][joint_action]
            milp.add_constraint(sum == follower_action_probability)

    sol = milp.solve()
    milp.export_as_lp(f"New_Stackelberg_LP")

    if sol:
        joint_solution = {}
        follower_solution = {}
        leader_solution = milp.solution.get_value_list(leader_DR)
        for state in CONSTANT.STATES:
            joint_solution[state] = np.array(milp.solution.get_value_list(joint_DR[state]))
            follower_solution[state] = np.array(milp.solution.get_value_list(follower_DR[state])) 
        return milp.solution.get_objective_value(),joint_solution,leader_solution,follower_solution
    print("No solution found for MILP\n")

    sys.exit()





def LP(Q1,Q2):

    milp = Model(f"{CONSTANT.PROBLEM.name} problem")
    leader_DR = milp.continuous_var_list(len(CONSTANT.ACTIONS[0]),name = [f"a0_{i}" for i in CONSTANT.ACTIONS[1]],ub=1,lb=0)
    follower_DR = milp.binary_var_list(len(CONSTANT.ACTIONS[1]), name = [f"a1_{i}" for i in CONSTANT.ACTIONS[0]])
    joint_DR = milp.continuous_var_list(len(CONSTANT.JOINT_ACTIONS),name  = [f"aj_{i}" for i in CONSTANT.JOINT_ACTIONS],ub=1,lb=0)
    


    # define objective function 
    obj_fn = 0
    for joint_action,joint_action_probability in enumerate(joint_DR):
            obj_fn += Q1[joint_action]  * joint_action_probability
    milp.maximize(obj_fn)

    # define constraints 

    # define lhs of linear equivalence expression equal to V^2(b0,a1,a2) :

    lhs = 0
    for joint_action,joint_action_probability in enumerate(joint_DR):
        lhs += Q2[joint_action] * joint_action_probability 

    # define rhs of linear equivalence expression equal to V^2(b0,a1,a2)
    for follower_action in CONSTANT.ACTIONS[1]:    
        rhs = 0       
        for leader_action,leader_action_probability in enumerate(leader_DR):
                joint_action = PROBLEM.get_joint_action(leader_action,follower_action)
                rhs += Q2[joint_action] * leader_action_probability
        milp.add_constraint(lhs>=rhs)
    

    ## add seperability constraints of joint_DRs and singular_DRs 
    joint_sum = 0
    for leader_action in CONSTANT.ACTIONS[0]:
        value = 0
        for follower_action in CONSTANT.ACTIONS[1]:   
            joint_action = PROBLEM.get_joint_action(leader_action,follower_action)
            value+=joint_DR[joint_action]
            joint_sum+=joint_DR[joint_action]
        milp.add_constraint(value==leader_DR[leader_action])
    milp.add_constraint(joint_sum==1)

    for follower_action in CONSTANT.ACTIONS[0]:
        value = 0
        for leader_action in CONSTANT.ACTIONS[1]:   
            joint_action = PROBLEM.get_joint_action(leader_action,follower_action)
            value+=joint_DR[joint_action]
        milp.add_constraint(value==follower_DR[follower_action])


    sol = milp.solve()
    milp.export_as_lp(f"Stackelberg__old_LP")
    # print(f"value solution = {milp.solution.get_objective_value()}")
    return milp.solution.get_objective_value(), milp.solution.get_value_list(joint_DR),milp.solution.get_value_list(leader_DR), milp.solution.get_value_list(follower_DR)



def get_joint_DR(DR0,DR1):
    DR=np.zeros(len(CONSTANT.JOINT_ACTIONS))
    for leader_action in CONSTANT.ACTIONS[0]:
        for follower_action in CONSTANT.ACTIONS[1]:
            DR[PROBLEM.get_joint_action(leader_action,follower_action)]= DR0[leader_action] * DR1[follower_action]

    return DR


def sota_strategy(beta,belief, game_type):
    payoffs = payoff_function(belief,beta)
    if game_type=="zerosum":
        return zerosum_sota(beta,belief)
    if game_type=="stackelberg":
        return stackelberg_alpha_sota(beta,belief)
    if game_type=="cooperative":
        return cooperative_sota(beta,belief)
    
def tabular_sota_strategy(beta,belief, game_type):
    if game_type=="zerosum":
        return zerosum_sota(beta,belief)
    if game_type=="stackelberg":
        return stackelberg_tabular_sota(beta,belief)
    if game_type=="cooperative":
        return cooperative_sota(beta,belief)
    
def stackelberg_alpha_sota(beta,belief):
    """returns value by  \sum over state += b(x) a(u^j) beta(x,u)"""
    leader_value , DR, DR0,DR1  = MILP(beta,belief)
    follower_value = 0
    for state in CONSTANT.STATES:
        for joint_action, joint_action_probability in enumerate(DR[state]):
            follower_value += belief.value[state] * beta.two_d_vectors[1][state][joint_action]  * joint_action_probability 
    return (leader_value,follower_value),DR,DR0,DR1

    
def stackelberg_tabular_sota(beta,belief):
    """returns value by  \sum over state += b(x) a(u^j) beta(x,u)"""
    leader_value , DR, DR0,DR1  = MILP(beta,belief)
    follower_value = 0
    for state in CONSTANT.STATES:
        for joint_action, joint_action_probability in enumerate(DR[state]):
            follower_value += belief.value[state] * beta.two_d_vectors[1][state][joint_action]  * joint_action_probability 
    return (leader_value,follower_value),DR,DR0,DR1




def zerosum_sota(beta,belief):
    """returns DRs in the form of a(u^j)"""
    payoffs = payoff_function(belief,beta)
    leader_value , DR0 =  zerosum_lp_leader(payoffs[0])
    follower_value , DR1 =  zerosum_lp_follower(payoffs[0])
    DR = get_joint_DR(DR0,DR1)

    return (leader_value,follower_value), DR, DR0,DR1

def cooperative_sota(beta,belief):
    payoffs = payoff_function(belief,beta)
    max_value = -np.inf
    optimal_joint_action = None
    for joint_action in CONSTANT.JOINT_ACTIONS:
        if max_value<payoffs[0][joint_action]:
            max_value = payoffs[0][joint_action]
            optimal_joint_action = joint_action
    DR_joint =  np.identity(len(CONSTANT.JOINT_ACTIONS))[optimal_joint_action]

    #seperate joint decision rule
    action_0, action_1 = PROBLEM.get_seperate_action(optimal_joint_action)
    DR0 = np.identity(len(CONSTANT.ACTIONS[0]))[action_0]
    DR1 = np.identity(len(CONSTANT.ACTIONS[1]))[action_1]
    return (max_value,max_value),DR_joint,DR0,DR1




 
def payoff_function(belief,beta):
        payoffs = {}
        for agent in range(0,2):
            payoffs[agent] = np.zeros(len(CONSTANT.JOINT_ACTIONS))
            for joint_action in CONSTANT.JOINT_ACTIONS:
                for state in CONSTANT.STATES:
                    payoffs[agent][joint_action] += belief.value[state] * beta.two_d_vectors[agent][state][joint_action]
        return payoffs

def zerosum_lp_leader(payoff):
    "linear program for SOTA of zerosum game"
    lp = Model(f"{PROBLEM} problem")

    #initialize linear program variables
    DR =  lp.continuous_var_list(len(CONSTANT.ACTIONS[0]),name = [f"a0_{i}" for i in CONSTANT.ACTIONS[1]],ub=1,lb=0)
    V = lp.continuous_var(name="V",ub=float('inf'),lb=float('-inf'))

    # define objective function 
    lp.maximize(V)

    # define constraints 
    for opponent_action in CONSTANT.ACTIONS[1]:    
        rhs = 0   
        for agent_action, agent_action_probability in enumerate(DR):
            rhs += payoff[PROBLEM.get_joint_action(agent_action,opponent_action)] * agent_action_probability
        
        lp.add_constraint(V<=rhs)


    #add sum-to-one constraint
    value = 0
    for agent_action_probability in DR:
        value += agent_action_probability
    
    lp.add_constraint(value == 1)

    #solve and export 
    sol = lp.solve()
    lp.export_as_lp(f"zerosum_lp_leader")

    # print(f"Linear program solved :{(sol!=None)}")
    return lp.solution.get_objective_value(),lp.solution.get_values(DR)


def zerosum_lp_follower(payoff):
    "linear program for SOTA of zerosum game"
    milp = Model(f"{PROBLEM.name} problem")

    #initialize linear program variables
    DR = []
    V = []
    for action in CONSTANT.ACTIONS[1]:
        DR.append(milp.continuous_var(name=f"a{1}_{action}",ub=1,lb=0))
    V = milp.continuous_var(name="V",ub=float('inf'),lb=float('-inf'))

    # define objective function 
    milp.minimize(V)

    # define constraints 
    for opponent_action in CONSTANT.ACTIONS[0]:    
        rhs = 0   

        for agent_action, agent_action_probability in enumerate(DR):
            rhs += payoff[PROBLEM.get_joint_action(opponent_action,agent_action)] * agent_action_probability
        
        milp.add_constraint(V>=rhs)


    #add sum-to-one constraint
    value = 0
    for agent_action_probability in DR:
        value += agent_action_probability
    
    milp.add_constraint(value == 1)

    #solve and export 
    sol = milp.solve()
    milp.export_as_lp(f"zerosum_lp_follower")

    # print(f"Linear program solved :{(sol!=None)}")
    return milp.solution.get_objective_value(), milp.solution.get_values(DR)



##########################################################################################################################################################################################################################################
#######################################################################################################################  CLASS DEFINITIONS  ##############################################################################################


