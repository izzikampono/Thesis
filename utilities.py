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
from decisionRule import DecisionRule
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


def normalize(vector):
    """function to normalize a vector"""
    warnings.filterwarnings("error", category=RuntimeWarning)
    try:
        vector = np.array(vector) / np.sum(vector)
        return vector
    except RuntimeWarning as rw:
        print(f"RuntimeWarning: {rw}")
        print(f"cannot normalize vector V: {vector}")
        sys.exit()

def exponential_decrease(start, stop, num):
    "function to generate exponentially decreasing values to be used for density values of the belief space"
    if num == 1:
        return np.array([start])
    else:
        return start * np.power(8, -np.linspace(0, np.log10(start/stop), num))


def  observation_probability(joint_observation,belief,joint_action):
    """function to calculate probability of an observation given a belief and joint action"""
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
        return milp.solution.get_objective_value(),DecisionRule(leader_solution,follower_solution,joint_solution)
    print("No solution found for MILP\n")

    sys.exit()


def get_joint_DR(DR0,DR1):
    joint_DR = {}
    for state in CONSTANT.STATES:
        joint_DR[state] = np.zeros((len(CONSTANT.JOINT_ACTIONS)))
        for leader_action,leader_action_probability in enumerate(DR0):
            for follower_action,follower_action_probability in enumerate(DR1[state]):
                joint_DR[state][PROBLEM.get_joint_action(leader_action,follower_action)] = leader_action_probability * follower_action_probability
    return joint_DR


def sota_strategy(belief,beta, game_type):
    if game_type=="zerosum":
        return zerosum_sota(belief,beta)
    if game_type=="stackelberg":
        return stackelberg_sota(belief,beta)
    if game_type=="cooperative":
        return new_cooperative_sota(belief,beta)
    
    
def stackelberg_sota(belief,beta):
    """returns value by  \sum over state += b(x) a(u^j) beta(x,u)"""
    leader_value , decision_rule  = MILP(beta,belief)
    follower_value = extract_follower_value(belief,decision_rule,beta)
    return leader_value,follower_value,decision_rule
  

def zerosum_sota(belief,beta):
    """returns DRs in the form of a(u^j)"""
    leader_value , DR0 =  new_zerosum_lp_leader(belief,beta)
    follower_value , DR1 =  new_zerosum_lp_follower(belief,beta)
    DR = get_joint_DR(DR0,DR1)
    return leader_value,follower_value, DecisionRule(DR0,DR1,DR)

def extract_follower_value(belief,DR,beta):
    "function to extract follower value from MILP since the docplex model cannot extract follower value"
    tabular_follower_value = 0
    for state in CONSTANT.STATES:
        for joint_action, joint_action_probability in enumerate(DR.joint[state]):
            tabular_follower_value += belief.value[state] * beta.two_d_vectors[1][state][joint_action]  * joint_action_probability
    return tabular_follower_value


def new_zerosum_lp_leader(belief,beta):
    "linear program for SOTA of zerosum game"
    lp = Model(f"{PROBLEM} problem")

    #initialize linear program variables
    DR = lp.continuous_var_list(len(CONSTANT.ACTIONS[0]),name = [f"a0_{i}" for i in CONSTANT.ACTIONS[0]],ub=1,lb=0)
    V = lp.continuous_var_list(len(CONSTANT.STATES) ,name=[f"V_{state}" for state in CONSTANT.STATES],ub=float('inf'),lb=float('-inf'))

    # define objective function 
    obj_fn = 0
    for state,V_state in enumerate(V) : 
        obj_fn += belief.value[state] * V_state
    lp.maximize(obj_fn)

    # define constraints 
    for state,state_value in enumerate(V):
        for opponent_action in CONSTANT.ACTIONS[1]:    
            lhs = 0   
            for leader_action, leader_action_probability in enumerate(DR):
                lhs += beta.two_d_vectors[0][state][PROBLEM.get_joint_action(leader_action,opponent_action)] * leader_action_probability
            lp.add_constraint(lhs>=state_value)

    #add sum-to-one constraint
    lp.add_constraint(lp.sum(DR) == 1)

    #solve and export 
    sol = lp.solve()
    lp.export_as_lp(f"zerosum_lp_leader")

    # print(f"Linear program solved :{(sol!=None)}")
    return lp.solution.get_objective_value(),lp.solution.get_value_list(DR)

def new_cooperative_sota(belief,beta):
    best_leader_action = 0
    best_follower_value = -np.inf 
    best_follower_decisoin_rule = {}

    # get follower best response for each state 
    for leader_action in CONSTANT.ACTIONS[0]:
        
        follower_value = 0
        follower_decisoin_rule = {}
        for state in CONSTANT.STATES:
            max = -np.inf
            for follower_action in CONSTANT.ACTIONS[1]:
                value = beta.two_d_vectors[0][state][PROBLEM.get_joint_action(leader_action,follower_action)]
                if value > max:
                    max = value
                    follower_decisoin_rule[state] = follower_action
            follower_value += belief.value[state] * max

        if follower_value > best_follower_value:
            best_leader_action = leader_action
            best_follower_value = follower_value
            best_follower_decisoin_rule = follower_decisoin_rule


    # get binary DR of follower a(u|x)
    follower_DR = {}
    for state in CONSTANT.STATES:
        follower_DR[state] = np.identity(len(CONSTANT.ACTIONS[1]))[best_follower_decisoin_rule[state]]
  
    leader_DR = np.identity(len(CONSTANT.ACTIONS[1]))[best_leader_action]

    joint_DR = get_joint_DR(leader_DR,follower_DR)
    
    return best_follower_value, best_follower_value, DecisionRule(leader_DR,follower_DR,joint_DR)


def new_zerosum_lp_follower(belief,beta):
    "linear program for follower of SOTA zerosum game"
    lp = Model(f"{PROBLEM.name} problem")

    #initialize linear program variables
    V = lp.continuous_var(name="V",ub=float('inf'),lb=float('-inf'))
    DR = {}
    for state in CONSTANT.STATES:
        DR[state] = lp.continuous_var_list(len(CONSTANT.ACTIONS[1]), name = [f"a1_u{i}_x{state}" for i in CONSTANT.ACTIONS[1]])
       
    # define objective function 
    lp.minimize(V)

    # define constraints 
    # constraint for every x and u^2 : \sum_{x} \sum_{u^1} += b(x) a(u^2) beta(b,x,u^1,u^2)  <= V
    for leader_action in CONSTANT.ACTIONS[0]:
        rhs = 0
        for state in CONSTANT.STATES:
            for follower_action, follower_action_probability in enumerate(DR[state]):            
                rhs += belief.value[state] * beta.two_d_vectors[0][state][PROBLEM.get_joint_action(leader_action,follower_action)] * follower_action_probability
        lp.add_constraint(V>=rhs)


    #add sum-to-one constraint
    for state in CONSTANT.STATES:
        lp.add_constraint(lp.sum(DR[state]) == 1)

    #solve and export 
    sol = lp.solve()
    if sol:
        lp.export_as_lp(f"zerosum_lp_follower")
        follower_DR = {}
        for state in CONSTANT.STATES:
            follower_DR[state] = lp.solution.get_value_list(DR[state])
        return lp.solution.get_objective_value(),  follower_DR 
    else: 
        print("CANNOT SOLVE ZEROSUM FOLLOWER LINEAR PROGRAM ")
        sys.exit()

##########################################################################################################################################################################################################################################
#######################################################################################################################  CLASS DEFINITIONS  ##############################################################################################


