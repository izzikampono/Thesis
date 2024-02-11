from constant import Constants
import numpy as np
import sys
import utilities as Utilities
from vector import AlphaVector, BetaVector
from constant import Constants
CONSTANT = Constants.get_instance()
from decisionRule import DecisionRule
import gc
gc.enable()

       
class ValueFunction:

    def __init__(self,horizon, initial_belief,problem,belief_space,sota=False):
        self.horizon = horizon
        self.vector_sets = {}
        self.problem=problem
        self.point_value_fn = {}
        self.belief_space = belief_space
        self.sota=sota
        self.initial_belief = initial_belief

        #initialize tables to store value function of both tabular based and max plane approach 
        for timestep in range(horizon+2):
            self.point_value_fn[timestep] = {}
            self.vector_sets[timestep] = []
            self.point_value_fn[timestep] = {}
            for belief_index in self.belief_space.time_index_table[timestep]:
                self.point_value_fn[timestep][belief_index] = None
                #for last horizon, initial 0 vectors
                if timestep>horizon:
                    vector = np.zeros(len(CONSTANT.STATES))
                    self.point_value_fn[timestep][belief_index] = AlphaVector(0,vector,vector,self.sota)

        #initialize last horizon to have a 0 alphavector 
        vector = np.zeros(len(CONSTANT.STATES))
        self.add_alpha_vector(AlphaVector(0,vector,vector,self.sota),horizon+1)
    

    def add_alpha_vector(self,alpha,timestep):
        self.vector_sets[timestep].append(alpha)


    def add_tabular_alpha_vector(self,alpha,belief_id,timestep):
        self.point_value_fn[timestep][belief_id] = alpha


    def pruning(self,timestep):
        self.vector_sets[timestep] = set(self.vector_sets[timestep])
        # check if this works


    def get_max_alpha(self,belief,timestep):
        """returns alpha vector object that gives the maximum value for a given belief at a certain timestep"""
        max = -np.inf
        max_alpha = None
        for alpha in self.vector_sets[timestep]:
            leader_value,follower_value = alpha.get_value(belief)
            if leader_value>max:
                max_value = (leader_value,follower_value)
                max_alpha = alpha
        return max_alpha,max_value
    

    def get_tabular_value_at_belief(self,belief_id,timestep):
        belief = self.belief_space.get_belief(belief_id)
        return self.point_value_fn[timestep][belief_id].get_value(belief)
    

    def get_max_plane_values_at_belief(self,belief,timestep):
        max_value = -np.inf
    
        for alpha in self.vector_sets[timestep]:
            leader_value, follower_value = alpha.get_value(belief)
            if( leader_value > max_value ) and alpha.sota==self.sota:
                max_value = leader_value
                max_follower_value = follower_value

        return max_value, max_follower_value
        
    
    
    def tabular_beta(self,belief_id,timestep,game_type):

        #initialize beta and choose appropriate reward
        two_d_vectors = {}
        reward = CONSTANT.REWARDS[game_type]
        belief = self.belief_space.get_belief(belief_id)

        #for each agent calculate Beta(belief,state,joint_action) = Reward(state,joint_action) + \sum_{joint_observation} \sum_{next_state} TRANSITION MATRIX(state,next_state,joint_action,joint_observation) * V_{t+1}(T(belief,joint_action,joint_observation))[next_state]
        for agent in range(0,2):
            two_d_vectors[agent] = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))

            if game_type=="stackelberg" and self.sota == True and agent==1 :
                two_d_vectors[1] = self.get_blind_beta(game_type)
                return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)


            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    two_d_vectors[agent][state][joint_action] = reward[agent][joint_action][state]
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        if Utilities.observation_probability(joint_observation,belief,joint_action)>0 and timestep < self.horizon:
                            #calculate next belief and get existing belief id with the same value
                            next_belief_id = self.belief_space.network.existing_next_belief_id(timestep,belief_id,joint_action,joint_observation)
                            two_d_vectors[agent][state][joint_action] += Utilities.observation_probability(joint_observation,belief,joint_action) *  self.point_value_fn[timestep+1][next_belief_id].get_value(self.belief_space.get_belief(next_belief_id))[agent]
        return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)
    

    def max_plane_beta(self,alpha_mappings,game_type):
        global CONSTANT
        two_d_vectors = {}
        reward = self.problem.REWARDS[game_type]

        for agent in range(0,2):
            two_d_vectors[agent] = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))
            
            if game_type=="stackelberg" and self.sota == True and agent==1 :
                two_d_vectors[1] = self.get_blind_beta(game_type)
                return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)

            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    two_d_vectors[agent][state][joint_action] = reward[agent][joint_action][state]
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                            for next_state in CONSTANT.STATES:
                                two_d_vectors[agent][state][joint_action]+= CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state] * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation]* alpha_mappings[joint_action][joint_observation].vectors[agent][next_state]
                        
        # note : u can filter the zero probabilites out of the vector to reduce computational 

        if self.sota==True and game_type=="stackelberg":
            two_d_vectors[1] = self.get_blind_beta(game_type)

        return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)
    
    def max_plane_beta2(self,belief_id,alpha_mappings,game_type):
        global CONSTANT
        two_d_vectors = {}
        reward = self.problem.REWARDS[game_type]
        belief = self.belief_space.get_belief(belief_id)

        for agent in range(0,2):
            two_d_vectors[agent] = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))
            
            if game_type=="stackelberg" and self.sota == True and agent==1 :
                two_d_vectors[1] = self.get_blind_beta(game_type)
                return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)

            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    two_d_vectors[agent][state][joint_action] = reward[agent][joint_action][state]
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        two_d_vectors[agent][state][joint_action]+= Utilities.observation_probability(joint_observation,belief,joint_action) * alpha_mappings[agent][joint_action][joint_observation]
                        
        # note : u can filter the zero probabilites out of the vector to reduce computational 

        if self.sota==True and game_type=="stackelberg":
            two_d_vectors[1] = self.get_blind_beta(game_type)

        return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)


    def get_alpha_mappings2(self,belief_id,timestep):
        #initialize
        belief = self.belief_space.get_belief(belief_id)
        alpha_mappings = {0:{},1:{}}
        for agent in range(2):
            for joint_action in CONSTANT.JOINT_ACTIONS:
                alpha_mappings[agent][joint_action] = {}
                for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                    alpha_mappings[agent][joint_action][joint_observation] = None
        
        #loop over actions and observations 
        for agent in range(2):
            for joint_action in CONSTANT.JOINT_ACTIONS:
                for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                    # if observation probability = 0 , skip and initialize 0 alpha vector for the (action-observation) pair
                    if Utilities.observation_probability(joint_observation,belief,joint_action) and timestep< self.horizon:
                        next_belief_id = self.belief_space.network.existing_next_belief_id(timestep,belief_id,joint_action,joint_observation)
                        max = -np.inf
                        max_alpha = None
                        #loop over all vectors at timstep, to get maximum alpha that can maximize the value w.r.t  belief
                        for alpha in self.vector_sets[timestep+1]:
                            leader_value,follower_value = alpha.get_value(self.belief_space.get_belief(next_belief_id))
                            if  leader_value>= max:
                                max = leader_value
                                max_alpha = alpha
                        
                        alpha_mappings[agent][joint_action][joint_observation] = max_alpha.get_value(self.belief_space.get_belief(next_belief_id))[agent]
                    else : alpha_mappings[agent][joint_action][joint_observation] = 0
        return alpha_mappings
    

  
   
    def get_alpha_mappings(self,belief_id,timestep):
        #initialize
        belief = self.belief_space.get_belief(belief_id)
        alpha_mappings = {}
        for joint_action in CONSTANT.JOINT_ACTIONS:
            alpha_mappings[joint_action] = {}
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                alpha_mappings[joint_action][joint_observation] = None
       
        #loop over actions and observations 
        for joint_action in CONSTANT.JOINT_ACTIONS:
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                # if observation probability = 0 , skip and initialize 0 alpha vector for the (action-observation) pair
                if Utilities.observation_probability(joint_observation,belief,joint_action) and timestep< self.horizon:
                    next_belief_id = self.belief_space.network.existing_next_belief_id(timestep,belief_id,joint_action,joint_observation)
                    max = -np.inf
                    #loop over all vectors at timstep, to get maximum alpha that can maximize the value w.r.t  belief
                    for alpha in self.vector_sets[timestep+1]:
                        leader_value,follower_value = alpha.get_value(belief)
                        if  leader_value> max:
                            max = leader_value
                            alpha_mappings[joint_action][joint_observation] = alpha

                else : alpha_mappings[joint_action][joint_observation] = AlphaVector(None,np.zeros(len(CONSTANT.STATES)),np.zeros(len(CONSTANT.STATES)),sota=self.sota)
        return alpha_mappings
    

    def backup(self,belief_id,timestep,gametype):
        max_plane_alpha, tabular_alpha = self.solve(belief_id,gametype,timestep)
        if max_plane_alpha == None:
            print(f"time : {timestep}")
            print(f"max_alpha = {max_plane_alpha}")
            print(f"size : {len(self.vector_sets[timestep+1])}")
            return
        self.add_alpha_vector(max_plane_alpha,timestep)
        self.add_tabular_alpha_vector(tabular_alpha,belief_id,timestep)


    def get_blind_beta(self,game_type):
        """ build beta vector for blind opponent (only uses current reward without an expectation of future reward """
        reward = self.problem.REWARDS[game_type]
        two_d_vector = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))
        for state in CONSTANT.STATES:
            for joint_action in CONSTANT.JOINT_ACTIONS:
                two_d_vector[state][joint_action] = reward[1][joint_action][state]
        return two_d_vector
    

    def solve(self,belief_id,game_type,timestep):

        # construct alpha mappings () and beta vectors for max plane solution (alpha mapping is mapping of (u,z) -> alpha vector) at a certain belief
        alpha_mappings = self.get_alpha_mappings2(belief_id,timestep)
        max_plane_beta = self.max_plane_beta2(belief_id,alpha_mappings,game_type)

        # construct tabular beta for tabular solution
        tabular_beta = self.tabular_beta(belief_id,timestep,game_type)
        belief = self.belief_space.get_belief(belief_id)

        # solve for optimal DR using linear program using constructed beta and current belief
        if self.sota==False:
            # DR returns joint action probabilities conditioned by state
            max_plane_leader_value , max_plane_DR = Utilities.MILP(max_plane_beta,belief)
            tabular_leader_value , tabular_DR  = Utilities.MILP(tabular_beta,belief)

            # extract tabular follower value
            tabular_follower_value = Utilities.extract_follower_value(belief,tabular_DR,tabular_beta)
            max_plane_follower_value = Utilities.extract_follower_value(belief,max_plane_DR,max_plane_beta)
           
        #if sota = True, use respective sota strategies
        else:
            max_plane_leader_value , max_plane_follower_value , max_plane_DR = Utilities.sota_strategy(belief,max_plane_beta,game_type)
            tabular_leader_value , tabular_follower_value, tabular_DR = Utilities.sota_strategy(belief,tabular_beta,game_type)
          
        # reconstruct alpha vectors
        max_plane_alpha = max_plane_beta.get_alpha_vector(belief,game_type,max_plane_DR, self.sota)
        tabular_alpha = tabular_beta.get_alpha_vector(belief,game_type,tabular_DR,self.sota)
        # print( f"Game {game_type}  ::  max plane LP value: {max_plane_leader_value,max_plane_follower_value}, tabular LP value : {tabular_leader_value,tabular_follower_value}  --  Reconstructed Max plane alpha: {max_plane_alpha.get_value(belief)}, reconstructed tabular alpha : {tabular_alpha.get_value(belief)}  --  belief {belief.value}  -- DR {max_plane_DR}\n" )

        #printing
        if np.abs(max_plane_leader_value-tabular_leader_value)>0.1 :
            print(f"\n\n FOUND DIFFERENCE IN LP for belief ID : {belief_id}! ")

            print( f"Game {game_type}  ::  max plane LP value: {max_plane_leader_value,max_plane_follower_value}, tabular LP value : {tabular_leader_value,tabular_follower_value}  --  Reconstructed Max plane alpha: {max_plane_alpha.get_value(belief)}, reconstructed tabular alpha : {tabular_alpha.get_value(belief)}  --  belief {belief.value}  -- DR {max_plane_DR}\n" )

            print("looking into beta vector..")
            for agent in range(2):
                for state in CONSTANT.STATES:
                    for joint_action in CONSTANT.JOINT_ACTIONS:
                        if max_plane_beta.two_d_vectors[agent][state][joint_action] != tabular_beta.two_d_vectors[agent][state][joint_action] :
                        
                            print(f"\tagent {agent}, beta(x = {state},  u = {joint_action}) , max_plane_beta = {max_plane_beta.two_d_vectors[agent][state][joint_action]} , tabular_beta = {tabular_beta.two_d_vectors[agent][state][joint_action]} ")
                            print("\tlooking into future component of beta..")
                            reward= CONSTANT.REWARDS[game_type][agent][joint_action][state]
                            print(f"\t\treward  = {reward}")
                            print(f"\t\tfuture component :  max_plane_beta = {max_plane_beta.two_d_vectors[agent][state][joint_action]-reward} , tabular_beta = {tabular_beta.two_d_vectors[agent][state][joint_action]-reward}")

                            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                                next_belief_id = self.belief_space.network.existing_next_belief_id(timestep,belief_id,joint_action,joint_observation)
                                print(f"\t\tPr({joint_observation}|b,{joint_action}) = {Utilities.observation_probability(joint_observation,belief,joint_action)} ,Future reward from max_plane {alpha_mappings[agent][joint_action][joint_observation]}, Future reward from point based {self.point_value_fn[timestep+1][next_belief_id].get_value(self.belief_space.get_belief(next_belief_id))[agent]} ")
                              

                            sys.exit()

        # return alpha vectors
        return max_plane_alpha , tabular_alpha

 
