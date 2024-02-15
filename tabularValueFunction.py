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

       
class TabularValueFunction:

    def __init__(self,horizon, initial_belief,problem,belief_space,sota=False):
        self.horizon = horizon
        self.vector_sets = {}
        self.problem=problem
        self.point_value_fn = {}
        self.belief_space = belief_space
        self.sota=sota
        self.initial_belief = initial_belief

        #initialize vector_sets as a dictionary to store alpha vector for each belief_id in a given timestep
        for timestep in range(horizon+1):
            self.vector_sets[timestep] = {}
            for belief_id in self.belief_space.time_index_table[timestep]:
                self.vector_sets[timestep][belief_id] = None
                #for last horizon, initial 0 vectors
                if timestep == self.horizon: self.add_alpha_vector(AlphaVector(0,np.zeros(len(CONSTANT.STATES)),np.zeros(len(CONSTANT.STATES)),belief_id,self.sota),self.horizon,belief_id)

    def add_alpha_vector(self,alpha,timestep,belief_id):
        self.vector_sets[timestep][belief_id]=alpha

    def pruning(self,timestep):
        self.vector_sets[timestep] = set(self.vector_sets[timestep])
        # check if this works


    def get_tabular_value_at_belief(self,belief_id,timestep):
        belief = self.belief_space.get_belief(belief_id)
        return self.vector_sets[timestep][belief_id].get_value(belief)
    
    

    def backup(self,belief_id,timestep,gametype):
        tabular_alpha = self.solve(belief_id,gametype,timestep)
        self.add_alpha_vector(tabular_alpha,timestep,belief_id)
        # print(f"belief _id = {belief_id}, max_plane = {max_plane_alpha.vectors} , tabular = {tabular_alpha.vectors}")



    def get_blind_beta(self,game_type):
        """ build beta vector for blind opponent (only uses current reward without an expectation of future reward """
        reward = self.problem.REWARDS[game_type]
        two_d_vector = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))
        for state in CONSTANT.STATES:
            for joint_action in CONSTANT.JOINT_ACTIONS:
                two_d_vector[state][joint_action] = reward[1][joint_action][state]
        return two_d_vector
    

    
    def contruct_beta(self, belief_id, timestep, mapping_belief_to_alpha, game_type):
        #initialize beta and choose appropriate reward
        two_d_vectors = {}
        reward = CONSTANT.REWARDS[game_type]

        for agent in range(0,2):
            two_d_vectors[agent] = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))

            if game_type=="stackelberg" and self.sota == True and agent==1 :
                two_d_vectors[1] = self.get_blind_beta(game_type)
                return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)

            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    two_d_vectors[agent][state][joint_action] = reward[agent][joint_action][state]
                    
                    if timestep >= self.horizon - 1 :
                        break

                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                        if next_belief_id and next_belief_id in mapping_belief_to_alpha.keys():
                            for next_state in CONSTANT.STATES:
                                two_d_vectors[agent][state][joint_action]+= CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state] * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation] * mapping_belief_to_alpha[next_belief_id].vectors[agent][next_state]

        return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)

    def construct_from_tabular_belief_to_alpha_mapping(self, belief_id, timestep):
        tabular_belief_to_alpha_mapping = {}

        for joint_action in CONSTANT.JOINT_ACTIONS:
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                if next_belief_id : 
                    if  timestep == self.horizon : tabular_belief_to_alpha_mapping[next_belief_id] = AlphaVector(None, np.zeros(len(CONSTANT.STATES)), np.zeros(len(CONSTANT.STATES),belief_id))
                    else :tabular_belief_to_alpha_mapping[next_belief_id]= self.vector_sets[timestep][next_belief_id]

        return tabular_belief_to_alpha_mapping


    def solve(self,belief_id,game_type,timestep):


        # construct tabular beta for tabular solution
        tabular_beta = self.contruct_beta(belief_id, timestep, self.construct_from_tabular_belief_to_alpha_mapping(belief_id, timestep+1), game_type)

        belief = self.belief_space.get_belief(belief_id)

        # solve for optimal DR using linear program using constructed beta and current belief
        if self.sota==False:
            # DR returns joint action probabilities conditioned by state
            leader_value , DR  = Utilities.MILP(tabular_beta,belief)

            # extract tabular follower value
            follower_value = Utilities.extract_follower_value(belief,DR,tabular_beta)
           
        #if sota = True, use respective sota strategies
        else:
            leader_value , follower_value, DR = Utilities.sota_strategy(belief,tabular_beta,game_type)
          
        # reconstruct alpha vectors
        tabular_alpha = tabular_beta.get_alpha_vector(belief_id,game_type,DR,self.sota)
        print( f"Game {game_type}  ::   tabular LP value : {leader_value,follower_value}  --  reconstructed tabular alpha : {tabular_alpha.get_value(belief)}  --  belief {belief.value}  -- DR {DR}\n" )

        #printing
        # if np.abs(max_plane_leader_value-tabular_leader_value)>0.1 :
        #     print(f"\n\n FOUND DIFFERENCE IN LP for belief ID : {belief_id}! ")

        #     print( f"Game {game_type}  ::  max plane LP value: {max_plane_leader_value,max_plane_follower_value}, tabular LP value : {tabular_leader_value,tabular_follower_value}  --  Reconstructed Max plane alpha: {max_plane_alpha.get_value(belief)}, reconstructed tabular alpha : {tabular_alpha.get_value(belief)}  --  belief {belief.value}  -- DR {max_plane_DR}\n" )

        #     print("looking into beta vector..")
        #     for agent in range(2):
        #         for state in CONSTANT.STATES:
        #             for joint_action in CONSTANT.JOINT_ACTIONS:
        #                 if np.abs(max_plane_beta.two_d_vectors[agent][state][joint_action]- tabular_beta.two_d_vectors[agent][state][joint_action])>0.01 :
                        
        #                     print(f"\tagent {agent}, beta(x = {state},  u = {joint_action}) , max_plane_beta = {max_plane_beta.two_d_vectors[agent][state][joint_action]} , tabular_beta = {tabular_beta.two_d_vectors[agent][state][joint_action]} ")
        #                     print("\tlooking into future component of beta..")
        #                     reward= CONSTANT.REWARDS[game_type][agent][joint_action][state]
        #                     print(f"\t\treward  = {reward}")
        #                     print(f"\t\tfuture component :  max_plane_beta = {max_plane_beta.two_d_vectors[agent][state][joint_action]-reward} , tabular_beta = {tabular_beta.two_d_vectors[agent][state][joint_action]-reward}")
        #                     sys.exit()
        #                     for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
        #                         next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
        #                         print(f"\t\tPr({joint_observation}|b,{joint_action}) = {Utilities.observation_probability(joint_observation,belief,joint_action)} ,Future reward from max_plane {alpha_mappings[agent][joint_action][joint_observation]}, Future reward from point based {self.point_value_fn[timestep+1][next_belief_id].get_value(self.belief_space.get_belief(next_belief_id))[agent]} ")
                              

                     
        # return alpha vectors
        return  tabular_alpha

 
