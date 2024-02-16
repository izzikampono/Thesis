##  CLASS DEFINITIONS
import numpy as np
import sys 
from decpomdp import DecPOMDP
from vector import AlphaVector,BetaVector
from policyTree import PolicyTree
from constant import Constants
from beliefObjects import Belief,BeliefNetwork,BeliefSpace
from tabularValueFunction import TabularValueFunction
from decisionRule import DecisionRule
import utilities as Utilities
import random
import time
import gc
gc.enable()

CONSTANT =  Constants.get_instance()
PROBLEM = CONSTANT.PROBLEM



class tabularPBVI:
    def __init__(self,problem,horizon,density,gametype,limit,sota=False):
        self.sota = sota
        self.belief_space = BeliefSpace(horizon,problem.b0,density,limit)
        self.policies = [PolicyTree(DR=None,value=None),PolicyTree(DR=None,value=None)]
        self.value_function = None
        self.gametype = gametype
        self.problem = problem
        self.initial_belief = Belief(problem.b0,None,None)
        self.horizon = horizon
        self.density = density
        self.limit = limit

    def backward_induction(self):
        #start loop from  horizon-1
        for timestep in range(self.horizon-1,-1,-1):
            print(f"========================= backup at timestep {timestep} =========================== ")
            print()
            for belief_id in self.belief_space.time_index_table[timestep]:
                self.value_function.backup(belief_id,timestep,self.gametype)

            print("========== Backup done, veryfing calulations for next timestep backup ==========")

    def solve(self,iterations):
        "solve function that uses a fixed density"

        # expand belief space and initialize Value Function
        print(f"\t\t\t Solving {self.gametype} {self.problem.name} GAME WITH SOTA {self.sota} {self.horizon} ")
        self.value_function = TabularValueFunction (self.horizon,self.initial_belief,self.problem,self.belief_space,sota=self.sota)
        values = []
        times = []
        start_time = time.time()
        self.belief_space.expansion()
        for _ in range(1,iterations+1):
            print(f"iteration : {_}")
            self.backward_induction()
            times.append(time.time()-start_time)
            values.append(self.value_function.get_tabular_value_at_belief(belief_id=0,timestep=0))


        # terminal result printing
        print(f"\n\n\n\n\n================================================= END OF {self.gametype} GAME WITH SOTA {self.sota} ======================================================================")
        print(f"\n\t\t\tpoint value at initial belief  {self.value_function.get_tabular_value_at_belief(belief_id=0,timestep=0)}")
        print(f"\n\n==========================================================================================================================================================================")
        return values, times , _
    
    def solve_sampled_densities(self,iterations,min_density):
        np.random.seed(20)
        # initialize Value Function
        print(f"\t\t\t Solving {self.gametype} {self.problem.name} GAME WITH SOTA {self.sota} {self.horizon} ")


        self.value_function = TabularValueFunction(self.horizon,self.initial_belief,self.problem,self.belief_space,sota=self.sota)
        values = []
        times = []
        belief_sizes = []
        densities = sorted(np.linspace(1/len(CONSTANT.STATES),min_density,iterations), reverse=True)

        start_time = time.time()
        for _ in range(iterations):
            print(f"iteration : {_+1} , density = {densities[_]}")
            #solving
            self.belief_space.set_density(densities[_])
            self.belief_space.expansion()
            self.backward_induction()

            # record measurements
            times.append(time.time()-start_time)
            values.append(self.value_function.get_tabular_value_at_belief(belief_id=0,timestep=0))
            belief_sizes.append(self.belief_space.belief_size())

            #increase density size
            self.belief_space.set_density(self.belief_space.density)
            if self.belief_space.density>0.6:
                self.belief_space.density=0.6
            
        # terminal result printing
        print("\n\n\n\n\n=========================================================== END ======================================================================")
        print(f"\npoint value at initial belief  {self.value_function.get_tabular_value_at_belief(belief_id=0,timestep=0)}")
        return values, times, densities , belief_sizes , _


    def extract_leader_policy(self,belief_id,timestep):

        # edge case at last horizon
        if timestep == self.horizon : return PolicyTree(None,None)

        #initialize policy and DR_bt
        policy = PolicyTree(None,None)

        # extract decision rule of agent from value function at current belief
        alpha = self.value_function.vector_sets[timestep][belief_id]

        #for all agent actions
        for leader_action in CONSTANT.ACTIONS[0]:
            #check if probability of action is not 0
            if alpha.DR.individual[0][leader_action] > 0:

                #get actions of the other agent
                for follower_action in CONSTANT.ACTIONS[1]:

                    joint_action = CONSTANT.PROBLEM.get_joint_action(leader_action,follower_action)
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        #get next belief of joint action and observation
                        next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                        #create subtree for next belief
                        if next_belief_id:
                            subtree = self.extract_leader_policy(next_belief_id,timestep+1)
                            policy.add_subtree(next_belief_id,subtree)
        policy.DR = alpha.DR.individual[0]
        policy.value = max
        return policy
    
    def extract_follower_policy(self,belief_id,timestep):

        # edge case at last horizon
        if timestep == self.horizon : return PolicyTree(None,None)

        #initialize policy and DR_bt
        policy = PolicyTree(None,None)
        alpha = self.value_function.vector_sets[timestep][belief_id]

        # extract decision rule of agent from value function at current belief
        belief= self.belief_space.get_belief(belief_id)
            

        if alpha.get_value(belief)==0: return PolicyTree(None)

        #for all follower actions
        for follower_action in CONSTANT.ACTIONS[1]:
            #check if probability of action is not 0

            #sum over states of follower DR a2(u|x) -> a2(u)
            follower_DR = np.zeros(len(CONSTANT.ACTIONS[1]))
            [follower_DR.__iadd__(np.array(DR) * belief.value[state]) for state,DR in alpha.DR.individual[1].items()]
            
            if follower_DR[follower_action]> 0:

                #get actions of the leader agent
                for leader_action in CONSTANT.ACTIONS[0]:

                    joint_action = CONSTANT.PROBLEM.get_joint_action(leader_action,follower_action)

                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        
                        # get the existing next belief_id in network using current belief_id, joint action and observation
                        next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                        
                        #create subtree for next belief
                        if next_belief_id:
                            subtree = self.extract_follower_policy(next_belief_id,timestep+1)
                            policy.add_subtree(self.belief_space.get_belief(next_belief_id),subtree)
        policy.DR = alpha.DR.individual[1]
        policy.value = alpha.get_value(belief)
        return policy
    
    def DP(self,belief_id,timestep,leader_tree,follower_tree):
        """recursive function to get joint value from individual policy trees, traverses individuals policy trees in parallel"""
        # edge case
        if timestep == self.horizon: return (0,0) 
        if leader_tree.DR == None or follower_tree.DR==None : return (0,0)
        print(f"belief od = {belief_id} at timestep {timestep}")


        #initialize value
        values = []
        belief = self.belief_space.get_belief(belief_id)

        # get V(b) recursively, \sum_{x} \sum{u_joint} += b(x) a1(u1) a2(u2) + \sum_{z} += Pr(z|b,u_joint) * V(TRANSITION(b,u_joint,z))
        
        for agent in range(2):
            value = 0
            reward = CONSTANT.REWARDS["stackelberg"][agent]
            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    leader_action, follower_action = PROBLEM.get_seperate_action(joint_action)
                    value += belief.value[state] * leader_tree.DR[leader_action] * follower_tree.DR[state][follower_action] * reward[joint_action][state]
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation) 
                        if next_belief_id and timestep<self.horizon:
                            value +=  Utilities.observation_probability(joint_observation,belief,joint_action) * self.DP(next_belief_id, timestep+1, leader_tree.subtree(next_belief_id,joint_action,joint_observation) , follower_tree.subtree(next_belief_id,joint_action,joint_observation))[agent]
            values.append(value)
        return values
    

