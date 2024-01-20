from warnings import catch_warnings
import numpy as np
import pandas as pd
import random
import warnings
from decpomdp import DecPOMDP
import matplotlib.pyplot as plt
from docplex.mp.model import Model
import sys
import subprocess
from constant import Constants

class Utilities:
    def __init__(self,constant:Constants) :
        self.constant = constant
        self.HORIZON = constant.HORIZON
        self.PROBLEM = constant.PROBLEM
        self.STATES = [i for i in range(len(constant.PROBLEM.states))]
        self.ACTIONS = [[i for i in range(len(constant.PROBLEM.actions[0]))],[j for j in range(len(self.PROBLEM.actions[1]))]]
        self.JOINT_ACTIONS = [i for i in range(len(constant.PROBLEM.joint_actions))]
        self.JOINT_OBSERVATIONS = [i for i in range(len(constant.PROBLEM.joint_observations))]
        self.TRANSITION_FUNCTION = self.PROBLEM.transition_fn
        self.OBSERVATION_FUNCTION = self.PROBLEM.observation_fn
        self.REWARDS = constant.initialize_rewards()
        self.PROBLEM.reset()



    def install_dependencies(self,):
        try:
            subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to install dependencies. {e}")

    def print_nested_dict(self,d, depth=0, parent_key=None, last_child=False):
        if parent_key is not None:
            prefix = "  " * (depth - 1) + ("└─ " if last_child else "├─ ")
            print(prefix + str(parent_key) + ":")
        
        for idx, (key, value) in enumerate(d.items()):
            is_last = idx == len(d) - 1
            if isinstance(value, dict):
                self.print_nested_dict(value, depth + 1, key, last_child=is_last)
            else:
                prefix = "  " * depth + ("└─ " if is_last else "├─ ")
                print(prefix + str(key) + ": " + str(value))

    def generate_probability_distribution(self,length):
        #generate random probabilities
        probabilities = np.random.rand(length)

        #normalize to make the sum equal to 1
        probabilities /= probabilities.sum()

        return probabilities

    #function to generate individual decision rule 
    def generate_sample_actions(self,n):
        samples=[]
        for j in range(n):
            samples.append(self.generate_probability_distribution(3))
        return np.array(samples)

    #function to generate joint decision rule 
    def generate_sample_joint_actions(self,n):
        samples=[]
        l=0
        for j in range(n):
            samples.append(self.generate_probability_distribution(9))
        return np.array(samples)

    #function to generate sample beliefs
    def generate_sample_belief(self,n):
        samples=[]
        for j in range(n):
            samples.append(self.generate_probability_distribution(2))
        return np.array(samples)

    def normalize(self,vector):
        warnings.filterwarnings("error", category=RuntimeWarning)

        try:
            vector = np.array(vector) / np.sum(vector)
            return vector
        except RuntimeWarning as rw:
            print(f"RuntimeWarning: {rw}")
            print(f"cannot normalize vector V: {vector}")
            sys.exit


    def  observation_probability(self,joint_observation,belief,joint_action):
        """function to calculate prob of an observation given a belief and joint action uj"""
        sum=0
        for state in self.STATES:
            for next_state in self.STATES:
                    sum += belief[state]  * self.TRANSITION_FUNCTION[joint_action][state][next_state] * self.OBSERVATION_FUNCTION[joint_action][state][joint_observation]
        return sum


    

    def LP(self,Q1,Q2):
    
        milp = Model(f"{self.PROBLEM.name} problem")
        leader_DR = milp.continuous_var_list(len(self.ACTIONS[0]),name = [f"a0_{i}" for i in self.ACTIONS[1]],ub=1,lb=0)
        follower_DR = milp.binary_var_list(len(self.ACTIONS[1]), name = [f"a1_{i}" for i in self.ACTIONS[0]])
        joint_DR = milp.continuous_var_list(len(self.JOINT_ACTIONS),name  = [f"aj_{i}" for i in self.JOINT_ACTIONS],ub=1,lb=0)
        


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
        for follower_action in self.ACTIONS[1]:    
            rhs = 0       
            for leader_action,leader_action_probability in enumerate(leader_DR):
                    joint_action = self.PROBLEM.get_joint_action(leader_action,follower_action)
                    rhs += Q2[joint_action] * leader_action_probability
            milp.add_constraint(lhs>=rhs)
        

        ## add seperability constraints of joint_DRs and singular_DRs 
        joint_sum = 0
        for leader_action in self.ACTIONS[0]:
            value = 0
            for follower_action in self.ACTIONS[1]:   
                joint_action = self.PROBLEM.get_joint_action(leader_action,follower_action)
                value+=joint_DR[joint_action]
                joint_sum+=joint_DR[joint_action]
            milp.add_constraint(value==leader_DR[leader_action])
        milp.add_constraint(joint_sum==1)

        for follower_action in self.ACTIONS[0]:
            value = 0
            for leader_action in self.ACTIONS[1]:   
                joint_action = self.PROBLEM.get_joint_action(leader_action,follower_action)
                value+=joint_DR[joint_action]
            milp.add_constraint(value==follower_DR[follower_action])




        sol = milp.solve()
        milp.export_as_lp(f"Stackelberg_LP")
        # print(f"value solution = {milp.solution.get_objective_value()}")
        return milp.solution.get_objective_value(),milp.solution.get_values(joint_DR), milp.solution.get_values(leader_DR), milp.solution.get_values(follower_DR)

    def get_joint_DR(self,DR0,DR1):
        DR=np.zeros(len(self.JOINT_ACTIONS))
        for leader_action in self.ACTIONS[0]:
            for follower_action in self.ACTIONS[1]:
                DR[self.PROBLEM.get_joint_action(leader_action,follower_action)]= DR0[leader_action] * DR1[follower_action]

        return DR


    def sota_strategy(self,P1,P2, game_type):
        if game_type=="zerosum":
            value0 , DR0 =  self.zerosum_lp_leader(P1)
            value1 , DR1 =  self.zerosum_lp_follower(P1)
            DR = self.get_joint_DR(DR0,DR1)
            # print(f"sota_strategy  ::  value{0}={value0} and value{1}={value1}")
            return value0, DR,DR0,DR1
        if game_type=="stackelberg":
            return self.LP(P1,P2)
        if game_type=="cooperative":
            max = -np.inf
            optimal_joint_action = None
            for joint_action in self.JOINT_ACTIONS:
                if max<P1[joint_action]:
                    max = P1[joint_action]
                    optimal_joint_action = joint_action
            DR_joint =  np.identity(len(self.JOINT_ACTIONS))[optimal_joint_action]
            action_0, action_1 = self.PROBLEM.get_seperate_action(optimal_joint_action)
            DR0 = np.identity(len(self.ACTIONS[0]))[action_0]
            DR1 = np.identity(len(self.ACTIONS[1]))[action_1]
            return max,DR_joint,DR0,DR1


    def zerosum_lp_leader(self,payoff):
        "linear program for SOTA of zerosum game"
        milp = Model("tiger problem")

        #initialize linear program variables
        DR = []
        V = []
        for action in self.ACTIONS[0]:
            DR.append(milp.continuous_var(name=f"a{0}_{action}",ub=1,lb=0))
        V = milp.continuous_var(name="V",ub=float('inf'),lb=float('-inf'))

        # define objective function 
        milp.maximize(V)

        # define constraints 
        for opponent_action in self.ACTIONS[1]:    
            rhs = 0   

            for agent_action, agent_action_probability in enumerate(DR):
                rhs += payoff[self.PROBLEM.get_joint_action(agent_action,opponent_action)] * agent_action_probability
            
            milp.add_constraint(V<=rhs)


        #add sum-to-one constraint
        value = 0
        for agent_action_probability in DR:
            value += agent_action_probability
        
        milp.add_constraint(value == 1)

        #solve and export 
        sol = milp.solve()
        milp.export_as_lp(f"zerosum_lp_{0}")

        # print(f"Linear program solved :{(sol!=None)}")
        return milp.solution.get_objective_value(),milp.solution.get_values(DR)


    def zerosum_lp_follower(self,payoff):
        "linear program for SOTA of zerosum game"
        milp = Model("tiger problem")

        #initialize linear program variables
        DR = []
        V = []
        for action in self.ACTIONS[1]:
            DR.append(milp.continuous_var(name=f"a{1}_{action}",ub=1,lb=0))
        V = milp.continuous_var(name="V",ub=float('inf'),lb=float('-inf'))

        # define objective function 
        milp.minimize(V)

        # define constraints 
        for opponent_action in self.ACTIONS[0]:    
            rhs = 0   

            for agent_action, agent_action_probability in enumerate(DR):
                rhs += payoff[self.PROBLEM.get_joint_action(opponent_action,agent_action)] * agent_action_probability
            
            milp.add_constraint(V>=rhs)


        #add sum-to-one constraint
        value = 0
        for agent_action_probability in DR:
            value += agent_action_probability
        
        milp.add_constraint(value == 1)

        #solve and export 
        sol = milp.solve()
        milp.export_as_lp(f"zerosum_lp_{1}")

        # print(f"Linear program solved :{(sol!=None)}")
        return milp.solution.get_objective_value(),milp.solution.get_values(DR)




    # Define Q function for blind attackers(player 2)
    def Q2_blind(self,bt,u):
        """subroutine Q value for blind agent that only uses reward to approx Q value"""
        reward = self.REWARDS["stackelberg"][1]
        sum = 0
        for x in self.STATES :
            sum += bt[x] * reward[u][x]
        return sum
    

    ##########################################################################################################################################################################################################################################
    #######################################################################################################################  CLASS DEFINITIONS
   

