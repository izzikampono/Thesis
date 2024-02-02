##  CLASS DEFINITIONS
import numpy as np
import sys 
from decpomdp import DecPOMDP
from alphaVector import AlphaVector
from policyTree import PolicyTree
from constant import Constants
from beliefObjects import Belief,BeliefNetwork,BeliefSpace
from utilities import *
import random
import gc
gc.enable()

CONSTANT =  Constants.get_instance()
utilities = Utilities(CONSTANT)
PROBLEM = CONSTANT.PROBLEM

    
################################################################################################

class DecisionRule:
    def __init__(self,a1,a2,aj):
        self.individual = {0:a1,1:a2}
        self.joint = aj

################################################################################################

class BetaVector:
    def __init__(self,two_d_vector_0,two_d_vector_1,problem):
        self.problem = problem
        self.two_d_vectors = [two_d_vector_0,two_d_vector_1]

    def get_alpha_vector(self, payoff, game_type, DR, sota=False):
        vectors = np.zeros((2,len(CONSTANT.STATES)))

        if game_type == "zerosum" and sota == True:
            u_1_best = 0
            u_1_best_value = -np.inf
            for u_1 in CONSTANT.ACTIONS[1]:
                u_1_value = 0
                
                for u_0 in CONSTANT.ACTIONS[0]:
                    u_1_value += DR.individual[0][u_0] * payoff[self.problem.get_joint_action(u_0, u_1)]

                if u_1_best_value < u_1_value:
                   u_1_best = u_1
                   u_1_best_value = u_1_value 

            for state in CONSTANT.STATES:
                for u_0 in CONSTANT.ACTIONS[0]:
                    vectors[0][state] += DR.individual[0][u_0] * self.two_d_vectors[0][self.problem.get_joint_action(u_0, u_1_best)][state]
                    vectors[1][state] += DR.individual[1][u_0] * self.two_d_vectors[1][self.problem.get_joint_action(u_0, u_1_best)][state]
            # print(f"reconstructed vector : {vectors[0]} , {vectors[1]}")
            return AlphaVector(DR,vectors[0],vectors[1], self.problem,sota)

        else:    
            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    vectors[0][state] += DR.joint[joint_action] * self.two_d_vectors[0][joint_action][state]
                    vectors[1][state] += DR.joint[joint_action] * self.two_d_vectors[1][joint_action][state]
            
            # for u_0 in CONSTANT.ACTIONS[0]:
            #     for u_1 in CONSTANT.ACTIONS[1]:
            #         joint_action = 
            #         DR.individual[0][u_0] * self.two_d_vectors[1][joint_action][state]


        return AlphaVector(DR,vectors[0],vectors[1], self.problem, sota)

        
    def print_vector(self):
        print(self.two_d_vectors)

       
class ValueFunction:
    def __init__(self,horizon, initial_belief,problem,belief_space,sota=False):
        self.horizon = horizon
        self.vector_sets = {}
        self.problem=problem
        self.point_value_fn = {}
        self.belief_space = belief_space
        self.sota=sota
        self.initial_belief = initial_belief
        for timestep in range(horizon+2):
            self.point_value_fn[timestep] = {}
            for agent in range(2):
                self.vector_sets[timestep] = []
                self.point_value_fn[timestep][agent] = {}
                for belief_index in self.belief_space.time_index_table[timestep]:
                    self.point_value_fn[timestep][agent][belief_index] = 0
            #     print(f"value function at timestep {timestep} : {self.point_value_fn[timestep]}")
        # sys.exit()
        vector = np.zeros(len(CONSTANT.STATES))
        self.add_alpha_vector(AlphaVector(0,vector,vector,problem,self.sota),horizon)
    
    def add_alpha_vector(self,alpha,timestep):
        self.vector_sets[timestep].append(alpha)

    def pruning(self,timestep):
        self.vector_sets[timestep] = set(self.vector_sets[timestep])
    
    def get_max_alpha(self,timestep,belief):
        max = -np.inf
        max_alpha = None
        for alpha in self.vector_sets[timestep]:
            leader_value,follower_value = alpha.get_value(belief)
            if leader_value>max:
                max_value = (leader_value,follower_value)
                max_alpha = alpha
        return max_alpha
    
        
    
    
    def point_based_payoffs(self,belief_id,timestep,game_type):

        #initialize payoff matrix for linear program
        belief =self.belief_space.belief_dictionary[belief_id]
        payoffs = {}
        reward = CONSTANT.REWARDS[game_type]
        # calculate values for payoff values
        for agent in range(0,2):
           
            payoffs[agent] = np.zeros(len(CONSTANT.JOINT_ACTIONS))
            if game_type=="cooperative" and agent==1 :
                payoffs[agent] = payoffs[0]
                break
            for joint_action in CONSTANT.JOINT_ACTIONS:
                for state in CONSTANT.STATES :
                    # payoff = b(x)r(x,u)
                    payoffs[agent][joint_action] += reward[agent][joint_action][state] * belief.value[state]
                
                    if game_type == "stackelberg"  and agent==1 and self.sota==True : 
                        continue #for the blind opponent of the stackelberg games

                    # payoff += \sum_z Pr(z|b,u)*V_{t+1}(T(b,u,z))
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        observation_probability = utilities.observation_probability(joint_observation,belief,joint_action)
                        if observation_probability > 0:  
                            next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                            payoffs[agent][joint_action] += utilities.observation_probability(joint_observation,belief,joint_action) * self.point_value_fn[timestep+1][agent][next_belief_id]* belief.value[state]
                            # if timestep==0:print(f"next belief value, for action {joint_action} , observation {joint_observation}, value = {self.point_value_fn[timestep+1][agent][next_belief_id]}")
        # if timestep==0:print(f"point based payoffs :\nLeader\n {payoffs[0]} \nFollower\n {payoffs[1]}\n")

        return payoffs
    

    def point_backup(self,belief_id,timestep,game_type):
        payoffs = self.point_based_payoffs(belief_id,timestep,game_type)
        # if timestep==0:
        #     print(f"point based payoffs :\nLeader\n {payoffs[0]} \nFollower\n {payoffs[1]}\n")

        # Get optimal DR for payoff matrix using linear program
        if self.sota==False :
            leader_value, DR , DR0 , DR1 = utilities.LP(payoffs[0],payoffs[1])
              # save value of leader from lienar program in value function indexed by belief ID
            self.point_value_fn[timestep][0][belief_id] = leader_value
            self.point_value_fn[timestep][1][belief_id] = np.dot(DR,payoffs[1])

        else:
            leader_value, DR, DR0, DR1 = utilities.sota_strategy(payoffs[0],payoffs[1],game_type)
            self.point_value_fn[timestep][0][belief_id] = leader_value
            self.point_value_fn[timestep][1][belief_id] = np.dot(DR,payoffs[1])
        return 
    

   
    def get_alpha_mappings(self,belief_id,timestep):
        #initialize
        belief = self.belief_space.belief_dictionary[belief_id]
        alpha_mappings = {}
        for  joint_action in CONSTANT.JOINT_ACTIONS:
            alpha_mappings[joint_action] = {}
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                alpha_mappings[joint_action][joint_observation] = None
       
        #loop over actions and observations 
        for joint_action in CONSTANT.JOINT_ACTIONS:
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                if utilities.observation_probability(joint_observation,belief,joint_action):
                    max = -np.inf
                    for alpha in self.vector_sets[timestep+1]:
                        next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                        leader_value,follower_value = alpha.get_value(self.belief_space.belief_dictionary[next_belief_id])
                        if  leader_value> max:
                            max = leader_value
                            alpha_mappings[joint_action][joint_observation] = alpha
                            # if timestep==0:print(f" action = {joint_action} , joint observation = {joint_observation}, \n vector value =  {leader_value}, {follower_value}\n")

                else :   alpha_mappings[joint_action][joint_observation] = AlphaVector(None,np.zeros(CONSTANT.STATES),np.zeros(CONSTANT.STATES),self.problem,sota=self.sota)
        # if timestep==0:
        #     print("\nAlpha vectors at timestep 1 : ")
        #     print([i.vectors for i in self.vector_sets[timestep+1]])

            # print("\n\nAlpha mappings at timestep 0:")
            # for joint_action in CONSTANT.JOINT_ACTIONS:
            #     for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
            #         print(f" joint action = {joint_action} , joint observation = {joint_observation}, \nvector =  {alpha_mappings[joint_action][joint_observation].vectors[0]}\n")

        return alpha_mappings


    def backup(self,belief_id,timestep,gametype):
        _alpha = self.solve(belief_id,gametype,timestep)
        # if timestep==0:
        #     print(_alpha.vectors)
        if _alpha == None:
            print(f"time : {timestep}")
            print(f"max_alpha = {_alpha}")
            print(f"size : {len(self.vector_sets[timestep+1])}")
            return
        self.add_alpha_vector(_alpha,timestep)

    def get_blind_two_d_vector(self,game_type):
        reward = self.problem.REWARDS[game_type]
        two_d_vector = np.zeros((len(CONSTANT.JOINT_ACTIONS),(len(CONSTANT.STATES))))
            
        for state in CONSTANT.STATES:
            for joint_action in CONSTANT.JOINT_ACTIONS:
                two_d_vector[joint_action][state] = reward[joint_action][state]
                
        return two_d_vector



    def get_beta_two_d_vector(self,alpha_mappings,game_type):
        global CONSTANT
        two_d_vectors = {}
        reward = self.problem.REWARDS[game_type]

        for agent in range(0,2):
            two_d_vectors[agent] = np.zeros((len(CONSTANT.JOINT_ACTIONS),(len(CONSTANT.STATES))))
            
            if game_type=="cooperative" and agent==1 :
                return BetaVector(two_d_vectors[0],two_d_vectors[0],self.problem)

            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    two_d_vectors[agent][joint_action][state] = reward[agent][joint_action][state]
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                            for next_state in CONSTANT.STATES:
                                two_d_vectors[agent][joint_action][state]+= CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state] * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation]* alpha_mappings[joint_action][joint_observation].vectors[agent][next_state]
                        
        # note : u can filter the zero probabilites out of the vector to reduce computational 

        if self.sota==True and game_type=="stackelberg":
            two_d_vectors[1] = self.get_blind_two_d_vector()


        return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)
    
    
    def payoff_function(self,belief,beta):
        payoffs = {}
        for agent in range(0,2):
            payoffs[agent] = np.zeros(len(CONSTANT.JOINT_ACTIONS))
            for joint_action in CONSTANT.JOINT_ACTIONS:
                for state in CONSTANT.STATES:
                    payoffs[agent][joint_action] += belief.value[state] * beta.two_d_vectors[agent][joint_action][state]
        return payoffs
    

    
    def solve(self,belief_id,game_type,timestep):
        alpha_mappings = self.get_alpha_mappings(belief_id,timestep)
        beta = self.get_beta_two_d_vector(alpha_mappings,game_type)
        payoffs = self.payoff_function(self.belief_space.belief_dictionary[belief_id],beta)



        # if timestep==0:
        #     # print(f"beta :\nLeader\n {beta.two_d_vectors[0]} \nFollower\n {beta.two_d_vectors[1]}\n")
        #     print(f"payoffs :\nLeader\n {payoffs[0]} \nFollower\n {payoffs[1]}\n")



        
        if self.sota==False :
            leader_value , DR , DR0 , DR1 = utilities.LP(payoffs[0],payoffs[1])
            follower_value = np.dot(DR,payoffs[1])
        
        else:
            leader_value, DR, DR0, DR1 = utilities.sota_strategy(payoffs[0],payoffs[1],game_type)
        
        alpha = beta.get_alpha_vector(payoffs[0],game_type,DecisionRule(DR0,DR1,DR), self.sota)
        belief = self.belief_space.belief_dictionary[belief_id]
        print( f"Game {game_type}  ::  Original: {leader_value}  --  Reconstructed: {alpha.get_value(belief)}   --  belief {belief.value}  -- DR {DR}" )
        return alpha

 
    
    def get_values_at_belief(self,timstep,belief):
        values_leader = np.inf
    
        for alpha in self.vector_sets[timstep]:
            leader, follower = alpha.get_value(belief)
            if( leader < values_leader ):
                values_leader = leader
                values_follower = follower

        return values_leader, values_follower
 


################################################################################################

class PBVI:
    def __init__(self,problem,horizon,density,gametype,limit,sota=False):
        self.sota = sota
        self.belief_space = BeliefSpace(horizon,problem.b0,density,limit)
        self.policies = [PolicyTree(None),PolicyTree(None)]
        self.value_function = None
        self.gametype = gametype
        self.problem = problem
        self.intitial_belief = Belief(problem.b0,None,None)
        self.horizon = horizon
        self.density = density
        self.limit = limit

    def backward_induction(self):
        for timestep in range(self.horizon-1,-1,-1):
            for belief_id in self.belief_space.time_index_table[timestep]:
                self.value_function.point_backup(belief_id,timestep,self.gametype)
                self.value_function.backup(belief_id,timestep,self.gametype)



                ## check if they are the same 
            for belief_id in self.belief_space.time_index_table[timestep]:
                belief = self.belief_space.belief_dictionary[belief_id]
                print(f"\n=========== backup for belief id {belief_id} ============ ")
                if np.abs(self.value_function.point_value_fn[timestep][0][belief_id] - self.value_function.get_values_at_belief(timestep,belief)[0]) > 0.01 : 
                    print(f"\n\nat timestep {timestep} , belief id = {belief_id}, point based value : {self.value_function.point_value_fn[timestep][0][belief_id],self.value_function.point_value_fn[timestep][1][belief_id]}, alpha vector value : {self.value_function.get_values_at_belief(timestep,belief)}\n")
                    print([i.vectors for i in self.value_function.vector_sets[timestep]])
                    point_payoffs = self.value_function.point_based_payoffs(belief_id,timestep,self.gametype)
                    alpha_mappings = self.value_function.get_alpha_mappings(belief_id,timestep)
                    beta = self.value_function.get_beta_two_d_vector(alpha_mappings,self.gametype)
                    alpha_payoffs = self.value_function.payoff_function(self.belief_space.belief_dictionary[belief_id],beta)
                    # print("\n LEADER :")
                    # for joint_action in CONSTANT.JOINT_ACTIONS:
                    #     print(f"for action {joint_action}, alpha payoff = {alpha_payoffs[0][joint_action]}, point based payoff = {point_payoffs[0][joint_action]}")                            
                    # print("\n Follower :")

                    # for joint_action in CONSTANT.JOINT_ACTIONS:
                    #     print(f"for action {joint_action}, alpha payoff = {alpha_payoffs[1][joint_action]}, point based payoff = {point_payoffs[1][joint_action]}")                            
                    
                    point_leader_value, DR , DR0 , DR1 = utilities.LP(point_payoffs[0],point_payoffs[1])
                    point_follower_value = np.dot(DR,point_payoffs[1])

                    alpha_leader_value, DR , DR0 , DR1 = utilities.LP(alpha_payoffs[0],alpha_payoffs[1])
                    alpha_follower_value = np.dot(DR,alpha_payoffs[1])

                    # print(f"Linear Program Results :")
                    # print(f"point based solution value : {point_leader_value},{point_follower_value} , alpha vector value : {alpha_leader_value,alpha_follower_value}")


            sys.exit()
                

            # print(f"\t ========== backup at timestep {timestep} done ========== ")

        # print("\tbackward induction done")


           
    def solve(self,iterations,decay):
        self.belief_space.expansion()
        self.value_function = ValueFunction(self.horizon,self.intitial_belief,self.problem,self.belief_space,sota=self.sota)

        for _ in range(iterations):
            # print(f"iteration : {_}")
            self.backward_induction()
            self.density /= decay #hyperparameter



        print("\n\n\n\n\n=========================================================== END ======================================================================")
        print(f"\npoint value at initial belief  {self.value_function.point_value_fn[0][0]},{self.value_function.point_value_fn[0][1]}")
        print(f"alphavectors value at inital belief (V0,V1) : {self.value_function.get_values_at_belief(0,self.intitial_belief)}\n\n")
        # print(f"\n\nvalue function : ")
        # for timestep in range(self.horizon+1):
        #     print(f"value at {timestep}, agent = 0, values: {self.value_function.point_value_fn[timestep][0]} ,  agent = 1, values: {self.value_function.point_value_fn[timestep][1]}")

              
        return self.value_function.get_values_at_belief(0,self.intitial_belief), (self.value_function.point_value_fn[0][0],self.value_function.point_value_fn[0][1])
              
   
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
                        if belief_next and self.belief_space.distance(belief_next,timestep):
                            subtree = self.tree_extraction(belief_next,agent,timestep+1)
                            policy.add_subtree(belief_next,subtree)
                        # else:print("no further viable beliefs")
        policy.data.append(DR.individual[agent])
        policy.data.append(max)
        return policy
    
