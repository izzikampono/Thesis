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
import time
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

            ## check solutions
            print(f"{(belief_id,self.belief_space.get_belief(belief_id)) in self.belief_space.time_index_table[timestep]}")
            for belief_id in self.belief_space.time_index_table[timestep]:
                belief = self.belief_space.get_belief(belief_id)
                print(f"{(belief_id,belief.value)}")

                tabular_value = self.value_function.get_tabular_value_at_belief(belief_id,timestep)
                max_alpha, max_alpha_value = self.value_function.get_max_alpha(belief,timestep)
                if np.abs(tabular_value[0]- max_alpha_value[0])>0.1 or np.abs(tabular_value[1]- max_alpha_value[1])>0.1:
                    print(f"\n\nDifference found for {self.gametype} {self.problem.name} game with SOTA = {self.sota}\nat timestep {timestep} , belief id = {belief_id}, tabular value : {tabular_value}, max plane value : {max_alpha_value}\n")
                    tabular_beta = self.value_function.tabular_beta(belief_id,timestep,self.gametype)
                    alpha_mappings = self.value_function.get_alpha_mappings2(belief_id,timestep)
                    beta = self.value_function.max_plane_beta2(belief_id,alpha_mappings,self.gametype)

                    print(f"dealing with belief of value = {belief.value}\n")

                   

                    print("looking into beta vector..")
                    for agent in range(2):
                        for state in CONSTANT.STATES:
                            for joint_action in CONSTANT.JOINT_ACTIONS:
                                if beta.two_d_vectors[agent][state][joint_action] != tabular_beta.two_d_vectors[agent][state][joint_action] :
                            
                                    print(f"\tagent {agent}, beta(x = {state},  u = {joint_action}) , max_plane_beta = {beta.two_d_vectors[agent][state][joint_action]} , tabular_beta = {tabular_beta.two_d_vectors[agent][state][joint_action]} ")
                                    
                                    print("\tlooking into future component of beta..")
                                    reward= CONSTANT.REWARDS[self.gametype][agent][joint_action][state]
                                    print(f"\t\treward  = {reward}")
                                    print(f"\t\tfuture component :  max_plane_beta = {beta.two_d_vectors[agent][state][joint_action]-reward} , tabular_beta = {tabular_beta.two_d_vectors[agent][state][joint_action]-reward}")
                                    sys.exit()
                                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                                        next_belief_id = self.belief_space.network.existing_next_belief_id(timestep,belief_id,joint_action,joint_observation)
                                        print(f"\t\tPr({joint_observation}|b,{joint_action}) = {Utilities.observation_probability(joint_observation,belief,joint_action)} ,Future reward from max_plane {alpha_mappings[agent][joint_action][joint_observation]}, Future reward from point based {self.point_value_fn[timestep+1][next_belief_id].get_value(self.belief_space.get_belief(next_belief_id))[agent]} ")

                                    sys.exit()

                    print("no difference in beta vector found :)")
                    if self.sota == False:
                        point_leader_value, tabular_DR = Utilities.MILP(tabular_beta,belief)
                        point_follower_value = Utilities.extract_follower_value(belief,tabular_DR,tabular_beta)

                        alpha_leader_value, DR = Utilities.MILP(beta,belief)
                        alpha_follower_value = Utilities.extract_follower_value(belief,DR,beta)

                    else:
                        point_leader_value,point_follower_value, tabular_DR = Utilities.sota_strategy(belief,tabular_beta,self.gametype)
                        alpha_leader_value,alpha_follower_value, DR = Utilities.sota_strategy(belief,beta,self.gametype)
                    
                
                    print(f"\nLinear Program Results :")
                    print(f"tabular solution value : {point_leader_value,point_follower_value} max-plane solution value : {alpha_leader_value,alpha_follower_value}\n")
                    
                    max_plane_alpha = beta.get_alpha_vector(belief,self.gametype,DR, self.sota)
                    tabular_alpha = tabular_beta.get_alpha_vector(belief,self.gametype,tabular_DR,self.sota)
                    print(f"\nreconstructed alpha vectors:\n\tmax_plane  vector :{max_plane_alpha.vectors}\n\ttabular vector : {tabular_alpha.vectors}")
                    print(f"\nmax alpha vector that was found for this belief id :\n{max_alpha.vectors}\n")

                    


                    # for alpha in self.value_function.vector_sets[timestep]:
                    #     if alpha.get_value(belief)[0] >= tabular_value[0]:
                    #         print("FOUND BIGGER ALPHA")
                    #         for alpha_state, belief_state in zip(alpha.vectors[0],belief.value):
                    #             print(f"\talpha(x) = {alpha_state} , belief_state(x) = {belief_state} => {alpha_state * belief_state}")

                    #         sys.exit()


                    sys.exit()
                else : 
                    print(f"belief = {self.belief_space.get_belief(belief_id).value}, max-plane vector = {max_alpha.vectors} , value = {max_alpha_value}/{max_alpha.get_value(self.belief_space.get_belief(belief_id))} , tabular alpha = {self.value_function.point_value_fn[timestep][belief_id].vectors}  ,tabular value = {tabular_value}")
                
        sys.exit()
        
    def solve(self,iterations):
        "solve function that uses a fixed density"

        # expand belief space and initialize Value Function
        print(f"\t\t\t Solving {self.gametype} {self.problem.name} GAME WITH SOTA {self.sota} {self.horizon} ")
        self.value_function = ValueFunction(self.horizon,self.initial_belief,self.problem,self.belief_space,sota=self.sota)
        values = []
        times = []
        start_time = time.time()
        self.belief_space.expansion()
        for _ in range(1,iterations+1):
            print(f"iteration : {_}")
            self.backward_induction()
            values.append(self.value_function.get_max_plane_values_at_belief(belief=self.initial_belief,timestep=0))
            times.append(time.time()-start_time)


        # terminal result printing
        print(f"\n\n\n\n\n================================================= END OF {self.gametype} GAME WITH SOTA {self.sota} ======================================================================")
        print(f"\n\t\t\tpoint value at initial belief  {self.value_function.get_tabular_value_at_belief(belief_id=0,timestep=0)}")
        print(f"\t\t\talphavectors value at inital belief (V0,V1) : {self.value_function.get_max_plane_values_at_belief(self.initial_belief,timestep=0)}")
        print(f"\n\n==========================================================================================================================================================================")

        return values,times
    
    def solve_sampled_densities(self,iterations,initial_density):
        np.random.seed(20)
        # initialize Value Function
        print(f"\t\t\t Solving {self.gametype} {self.problem.name} GAME WITH SOTA {self.sota} {self.horizon} ")


        self.value_function = ValueFunction(self.horizon,self.initial_belief,self.problem,self.belief_space,sota=self.sota)
        values = []
        times = []
        belief_sizes = []
        densities = sorted(np.concatenate((np.array([initial_density]), np.random.uniform(initial_density,0.5,iterations-1))))

        start_time = time.time()


        for _ in range(iterations):
            print(f"iteration : {_+1} , density = {densities[_]}")
            #solving
            self.belief_space.set_density(densities[_])
            self.belief_space.expansion()
            self.backward_induction()

            # record measurements
            values.append( self.value_function.get_max_plane_values_at_belief(self.initial_belief,timestep=0))
            times.append(time.time()-start_time)
            belief_sizes.append(self.belief_space.belief_size())

            #increase density size
            self.belief_space.set_density(self.belief_space.density)
            if self.belief_space.density>0.6:
                self.belief_space.density=0.6
            
        # terminal result printing
        print("\n\n\n\n\n=========================================================== END ======================================================================")
        print(f"\npoint value at initial belief  {self.value_function.get_tabular_value_at_belief(belief_id=0,timestep=0)}")
        print(f"alphavectors value at inital belief (V0,V1) : {self.value_function.get_max_plane_values_at_belief(self.initial_belief,timestep=0)}\n\n")
       
              
        return values, times, densities , belief_sizes
               
              
    def extract_leader_policy(self,belief_id,timestep):

        # edge case at last horizon
        if timestep == self.horizon : return PolicyTree(None,None)

        #initialize policy and DR_bt
        policy = PolicyTree(None,None)
        belief = self.belief_space.get_belief(belief_id)

        # extract decision rule of agent from value function at current belief
        max = -np.inf
        # extract decision rule of agent from value function at current belief
        for alpha in self.value_function.vector_sets[timestep]:
            value = alpha.get_value(belief)
            if (max<value[0]) :
                max = value[0]
                DR = alpha.DR
        # print(f"DR at timestep {timestep} , {DR.individual}")
        if value==0: return PolicyTree(None,None)

        #for all agent actions
        for leader_action in CONSTANT.ACTIONS[0]:
            #check if probability of action is not 0
            if DR.individual[0][leader_action] > 0:

                #get actions of the other agent
                for follower_action in CONSTANT.ACTIONS[1]:

                    joint_action = CONSTANT.PROBLEM.get_joint_action(leader_action,follower_action)
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        #get next belief of joint action and observation
                        next_belief_id = self.belief_space.network.existing_next_belief_id(timestep,belief_id,joint_action,joint_observation)
                        #create subtree for next belief
                        if next_belief_id:
                            subtree = self.extract_leader_policy(next_belief_id,timestep+1)
                            policy.add_subtree(self.belief_space.get_belief(next_belief_id),subtree)
        policy.DR = DR.individual[0]
        policy.value = max
        return policy
    
    def extract_follower_policy(self,belief_id,timestep):

        # edge case at last horizon
        if timestep == self.horizon : return PolicyTree(None,None)

        #initialize policy and DR_bt
        policy = PolicyTree(None,None)
        belief = self.belief_space.get_belief(belief_id)
        # extract decision rule of agent from value function at current belief
        max = -np.inf
        # extract decision rule of agent from value function at current belief
        for alpha in self.value_function.vector_sets[timestep]:
            value = alpha.get_value(belief)
            if (max<value[1]) :
                max = value[1]
                DR = alpha.DR

        if value==0: return PolicyTree(None)

        #for all follower actions
        for follower_action in CONSTANT.ACTIONS[1]:
            #check if probability of action is not 0

            #sum over states of follower DR a2(u|x) -> a2(u)
            follower_DR = np.zeros(len(CONSTANT.ACTIONS[1]))
            [follower_DR.__iadd__(np.array(DR) * belief.value[state]) for state,DR in DR.individual[1].items()]
            
            if follower_DR[follower_action]> 0:

                #get actions of the leader agent
                for leader_action in CONSTANT.ACTIONS[0]:

                    joint_action = CONSTANT.PROBLEM.get_joint_action(leader_action,follower_action)

                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        
                        # get the existing next belief_id in network using current belief_id, joint action and observation
                        next_belief_id = self.belief_space.network.existing_next_belief_id(timestep,belief_id,joint_action,joint_observation)
                        
                        #create subtree for next belief
                        if next_belief_id:
                            subtree = self.extract_follower_policy(next_belief_id,timestep+1)
                            policy.add_subtree(self.belief_space.get_belief(next_belief_id),subtree)
        policy.DR = DR.individual[1]
        policy.value = value
        return policy
    

    
    def DP(self,belief_id,timestep,leader_tree,follower_tree):
        """recursive function to get joint value from individual policy trees, traverses individuals policy trees in parallel"""
        # edge case
        if timestep == self.horizon: return 0 
        if leader_tree.DR == None or follower_tree.DR==None : return 0 
        reward = CONSTANT.REWARDS["stackelberg"][0]

        #initialize value
        value = 0
        belief = self.belief_space.get_belief(belief_id)

        # get V(b) recursively, \sum_{x} \sum{u_joint} += b(x) a1(u1) a2(u2) + \sum_{z} += Pr(z|b,u_joint) * V(TRANSITION(b,u_joint,z))
        for state in CONSTANT.STATES:
            for joint_action in CONSTANT.JOINT_ACTIONS:
                leader_action, follower_action = PROBLEM.get_seperate_action(joint_action)
                value += belief.value[state] * leader_tree.DR[leader_action] * follower_tree.DR[state][follower_action] * reward[joint_action][state]
                for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                    if Utilities.observation_probability(joint_observation,belief,joint_action):
                        next_belief_id = self.belief_space.network.existing_next_belief_id(timestep,belief_id,joint_action,joint_observation) 
                        value +=  Utilities.observation_probability(joint_observation,belief,joint_action) * self.DP(next_belief_id, timestep+1, leader_tree.subtree(joint_action,joint_observation) , follower_tree.subtree(joint_action,joint_observation))
        return value
    

    def run_game(self,iterations,decay):
        tabular_value , max_plane_value = self.solve(iterations,decay)
        leader_policy = self.extract_leader_policy(belief_id=0,timestep=0)
        follower_policy = self.extract_follower_policy(belief_id=0,timestep=0)
        return tabular_value , max_plane_value , leader_policy , follower_policy
    

        