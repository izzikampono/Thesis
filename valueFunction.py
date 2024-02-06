from constant import Constants
import numpy as np
import sys
import utilities as Utilities
from vector import AlphaVector, BetaVector
from constant import Constants
CONSTANT = Constants.get_instance()



class DecisionRule:
    def __init__(self,a1,a2,aj):
        self.individual = {0:a1,1:a2}
        self.joint = aj
       
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

    def get_max_alpha(self,timestep,belief):
        max = -np.inf
        max_alpha = None
        for alpha in self.vector_sets[timestep]:
            leader_value,follower_value = alpha.get_value(belief)
            if leader_value>max:
                max_value = (leader_value,follower_value)
                max_alpha = alpha
        return max_alpha
    
    def get_tabular_value_at_belief(self,belief_id,timestep):
        belief = self.belief_space.get_belief(belief_id)
        return self.point_value_fn[timestep][belief_id].get_value(belief)
    
    def get_max_plane_values_at_belief(self,belief,timestep):
        values_leader = -np.inf
    
        for alpha in self.vector_sets[timestep]:
            leader, follower = alpha.get_value(belief)
            if( leader > values_leader ):
                values_leader = leader
                values_follower = follower

        return values_leader, values_follower
        
    
    
    def tabular_beta(self,belief_id,timestep,game_type):

        #initialize beta and choose appropriate reward
        two_d_vectors = {}
        two_d_vectors[0] = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))
        two_d_vectors[1] = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))
        reward = CONSTANT.REWARDS[game_type]
        belief = self.belief_space.get_belief(belief_id)
        #for each agent calculate Beta(belief,state,joint_action) = Reward(state,joint_action) + \sum_{joint_observation} \sum_{next_state} TRANSITION MATRIX(state,next_state,joint_action,joint_observation) * V_{t+1}(T(belief,joint_action,joint_observation))[next_state]
        for agent in range(0,2):
            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    two_d_vectors[agent][state][joint_action] = reward[agent][joint_action][state]
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                            #calculate next belief and get existing belief id with the same value
                            next_belief = belief.next_belief(joint_action,joint_observation)
                            next_belief_id = self.belief_space.find_belief_id(next_belief)
                            for next_state in CONSTANT.STATES:
                                two_d_vectors[agent][state][joint_action] += CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state] * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation] *  self.point_value_fn[timestep+1][next_belief_id].vectors[agent][next_state]
        return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)
    def max_plane_beta(self,alpha_mappings,game_type):
        global CONSTANT
        two_d_vectors = {}
        reward = self.problem.REWARDS[game_type]

        for agent in range(0,2):
            two_d_vectors[agent] = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))
            
            if game_type=="cooperative" and agent==1 :
                return BetaVector(two_d_vectors[0],two_d_vectors[0],self.problem)

            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    two_d_vectors[agent][state][joint_action] = reward[agent][joint_action][state]
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                            for next_state in CONSTANT.STATES:
                                two_d_vectors[agent][state][joint_action]+= CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state] * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation]* alpha_mappings[joint_action][joint_observation].vectors[agent][next_state]
                        
        # note : u can filter the zero probabilites out of the vector to reduce computational 

        if self.sota==True and game_type=="stackelberg":
            two_d_vectors[1] = self.get_blind_beta()


        return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)
    
  


    def tabular_backup(self,belief_id,timestep,game_type):
        beta = self.tabular_beta(belief_id,timestep,game_type)
        belief = self.belief_space.belief_dictionary[belief_id]
        # if timestep==0:
        #     print(f"point based payoffs :\nLeader\n {payoffs[0]} \nFollower\n {payoffs[1]}\n")

        # Get optimal DR for payoff matrix using linear program
        if self.sota==False :
            leader_value, DR , DR0 , DR1 = Utilities.MILP(beta,belief)
            follower_value=0
            for state in CONSTANT.STATES:
                for joint_action, joint_action_probability in enumerate(DR[state]):
                    follower_value +=  beta.two_d_vectors[1][state][joint_action]  * joint_action_probability 
                self.point_value_fn[timestep][1][belief_id][state] = follower_value
                self.point_value_fn[timestep][0][belief_id][state] = leader_value
        else:
            values, DR, DR0, DR1 = Utilities.tabular_sota_strategy(beta,belief,game_type)
            self.point_value_fn[timestep][0][belief_id] = values[0]
            self.point_value_fn[timestep][0][belief_id] = values[1]

            # follower_value = np.dot(belief.value,DR)
       
        return 
    

   
    def get_alpha_mappings(self,belief_id,timestep):
        #initialize
        belief = self.belief_space.belief_dictionary[belief_id]
        alpha_mappings = {}
        for joint_action in CONSTANT.JOINT_ACTIONS:
            alpha_mappings[joint_action] = {}
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                alpha_mappings[joint_action][joint_observation] = None
       
        #loop over actions and observations 
        for joint_action in CONSTANT.JOINT_ACTIONS:
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                # if observation probability = 0 , skip and initialize 0 alpha vector for the (action-observation) pair
                if Utilities.observation_probability(joint_observation,belief,joint_action) and timestep<= self.horizon:
                    next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                    max = -np.inf
                    #loop over all vectors at timstep, to get maximum alpha that can maximize the value w.r.t  belief
                    for alpha in self.vector_sets[timestep+1]:
                        leader_value,follower_value = alpha.get_value(self.belief_space.get_belief(next_belief_id))
                        if  leader_value> max:
                            max = leader_value
                            alpha_mappings[joint_action][joint_observation] = alpha

                else : alpha_mappings[joint_action][joint_observation] = AlphaVector(None,np.zeros(CONSTANT.STATES),np.zeros(CONSTANT.STATES),sota=self.sota)
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

    def get_blind_beta(self):
        """ build beta vector for blind opponent (only uses current reward without an expectation of future reward """
        reward = self.problem.REWARDS["stackelberg"]
        two_d_vector = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))
        for state in CONSTANT.STATES:
            for joint_action in CONSTANT.JOINT_ACTIONS:
                two_d_vector[state][joint_action] = reward[1][joint_action][state]
                
        return two_d_vector




    
    def solve(self,belief_id,game_type,timestep):

        # construct alpha mappings and beta vectors
        alpha_mappings = self.get_alpha_mappings(belief_id,timestep)
        max_plane_beta = self.max_plane_beta(alpha_mappings,game_type)

        # construct tabular beta
        tabular_beta = self.tabular_beta(belief_id,timestep,game_type)
        belief = self.belief_space.get_belief(belief_id)


        # RUN linear program for both tabular betas and max_plane betas 
        if self.sota==False:
            # DR returns joint action probabilities conditioned by state
            max_plane_leader_value , max_plane_DR , max_plane_DR0 , max_plane_DR1 = Utilities.MILP(max_plane_beta,belief)
            tabular_leader_value , tabular_DR , tabular_DR0 , tabular_DR1 = Utilities.MILP(tabular_beta,belief)

            # extract tabular follower value
            tabular_follower_value = 0
            for state in CONSTANT.STATES:
                for joint_action, joint_action_probability in enumerate(tabular_DR[state]):
                    tabular_follower_value += belief.value[state] * max_plane_beta.two_d_vectors[1][state][joint_action]  * joint_action_probability
          
        #if sota = True, use respective sota strategies
        else:
            max_plane_leader_value, max_plane_DR, max_plane_DR0, max_plane_DR1 = Utilities.sota_strategy(max_plane_beta,belief,game_type)
            tabular_leader_value , tabular_DR , tabular_DR0 , tabular_DR1 = Utilities.MILP(max_plane_beta,self.belief_space.belief_dictionary[belief_id])


        max_plane_alpha = max_plane_beta.get_alpha_vector(belief,game_type,DecisionRule(max_plane_DR0,max_plane_DR1,max_plane_DR), self.sota)
        tabular_alpha = tabular_beta.get_alpha_vector(belief,game_type,DecisionRule(tabular_DR0,tabular_DR1,tabular_DR),self.sota)
        belief = self.belief_space.belief_dictionary[belief_id]
        print( f"Game {game_type}  ::  max plane LP value: {max_plane_leader_value}, tabular LP value : {tabular_leader_value,tabular_follower_value}  --  Reconstructed Max plane alpha: {max_plane_alpha.get_value(belief)}, reconstructed tabular alpha : {tabular_alpha.get_value(belief)}  --  belief {belief.value}  -- DR {max_plane_DR}" )
        return max_plane_alpha , tabular_alpha

 
