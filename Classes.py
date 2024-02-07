##  CLASS DEFINITIONS
import numpy as np
import sys 
from decpomdp import DecPOMDP
from vector import AlphaVector,BetaVector
from policyTree import PolicyTree
from constant import Constants
from beliefObjects import Belief,BeliefNetwork,BeliefSpace
from valueFunction import ValueFunction,DecisionRule
import utilities as Utilities
import random
import gc
gc.enable()

CONSTANT =  Constants.get_instance()
PROBLEM = CONSTANT.PROBLEM




################################################################################################

class PBVI:
    def __init__(self,problem,horizon,density,gametype,limit,sota=False):
        self.sota = sota
        self.belief_space = BeliefSpace(horizon,problem.b0,density,limit)
        self.policies = [PolicyTree(DR=None,value=None),PolicyTree(DR=None,value=None)]
        self.value_function = None
        self.gametype = gametype
        self.problem = problem
        self.intitial_belief = Belief(problem.b0,None,None)
        self.horizon = horizon
        self.density = density
        self.limit = limit

    def backward_induction(self):
        #start loop from  horizon-1
        for timestep in range(self.horizon,-1,-1):
            print(f"========== backup at timestep {timestep} ========== ")

            for belief_id in self.belief_space.time_index_table[timestep]:
                self.value_function.backup(belief_id,timestep,self.gametype)



            ## check solutions
            for belief_id in self.belief_space.time_index_table[timestep]:
                belief = self.belief_space.belief_dictionary[belief_id]
                tabular_value = self.value_function.get_tabular_value_at_belief(belief_id,timestep)
                max_plane_value = self.value_function.get_max_plane_values_at_belief(belief,timestep)

                if np.abs(tabular_value[0]- max_plane_value[0])>0.01 or np.abs(tabular_value[1]- max_plane_value[1])>0.01:
                    print(f"\n\nDifference found at timestep {timestep} , belief id = {belief_id}, tabular value : {tabular_value}, max plane value : {max_plane_value}\n")
                    point_beta = self.value_function.tabular_beta(belief_id,timestep,self.gametype)
                    alpha_mappings = self.value_function.get_alpha_mappings(belief_id,timestep)
                    beta = self.value_function.max_plane_beta(alpha_mappings,self.gametype)

                    if self.sota == False:
                        point_leader_value, DR = Utilities.MILP(point_beta,belief)
                        point_follower_value = Utilities.extract_follower_value(belief,DR,point_beta)

                        alpha_leader_value, DR = Utilities.MILP(beta,belief)
                        alpha_follower_value = Utilities.extract_follower_value(belief,DR,beta)

                    else:
                        point_leader_value,point_follower_value, DR = Utilities.sota_strategy(belief,point_beta,self.gametype)
                        alpha_leader_value,alpha_follower_value, DR = Utilities.sota_strategy(belief,beta,self.gametype)
                    print(f"Linear Program Results :")
                    print(f"point based solution value : {point_leader_value},{point_follower_value} , alpha vector value : {alpha_leader_value,alpha_follower_value}")

                    sys.exit()
                


        # print("\tbackward induction done")


           
    def solve(self,iterations,decay):

        # expand belief space and initialize Value Function
        self.belief_space.expansion()
        self.value_function = ValueFunction(self.horizon,self.intitial_belief,self.problem,self.belief_space,sota=self.sota)

        # backward induction 
        for _ in range(iterations):
            # print(f"iteration : {_}")
            self.backward_induction()
            self.density /= decay #hyperparameter


        # terminal result printing
        print("\n\n\n\n\n=========================================================== END ======================================================================")
        print(f"\npoint value at initial belief  {self.value_function.get_tabular_value_at_belief(belief_id=0,timestep=0)}")
        print(f"alphavectors value at inital belief (V0,V1) : {self.value_function.get_max_plane_values_at_belief(self.intitial_belief,timestep=0)}\n\n")
        # print(f"\n\nvalue function : ")
        # for timestep in range(self.horizon+1):
        #     print(f"value at {timestep}, agent = 0, values: {self.value_function.point_value_fn[timestep][0]} ,  agent = 1, values: {self.value_function.point_value_fn[timestep][1]}")

              
        return self.value_function.get_tabular_value_at_belief(belief_id=0,timestep=0), self.value_function.get_max_plane_values_at_belief(self.intitial_belief,timestep=0)
              
   
    def tree_extraction(self,belief, agent,timestep):

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
    
