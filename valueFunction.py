from constant import Constants
import numpy as np
import sys
import utilities as Utilities
from vector import AlphaVector, BetaVector
from constant import Constants
CONSTANT = Constants.get_instance()
from decisionRule import DecisionRule


       
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
        for timestep in range(horizon+1):
            self.point_value_fn[timestep] = {}
            self.vector_sets[timestep] = []
            self.point_value_fn[timestep] = {}
            for belief_id in self.belief_space.time_index_table[timestep]:
                self.point_value_fn[timestep][belief_id] = None
                #for last horizon, initial 0 vectors
                if timestep==horizon:
                    vector = np.zeros(len(CONSTANT.STATES))
                    self.point_value_fn[timestep][belief_id] = AlphaVector(0,vector,vector,belief_id,self.sota)
                    self.add_alpha_vector(AlphaVector(0,vector,vector,belief_id,self.sota),horizon)

        #initialize last horizon to have a 0 alphavector 
        vector = np.zeros(len(CONSTANT.STATES))
        self.add_alpha_vector(AlphaVector(0,vector,vector,belief_id,self.sota),horizon)
    

    def add_alpha_vector(self,alpha,timestep):
        self.vector_sets[timestep].append(alpha)

    def get_alpha(self,timestep,belief_id):
        vector = np.zeros(len(CONSTANT.STATES))
        if timestep == self.horizon : return AlphaVector(0,vector,vector,belief_id,self.sota)
        for alpha in self.vector_sets[timestep]:
            if alpha.belief_id == belief_id and alpha.belief_id != None:
                return alpha
        print("no alpha found ")

    def add_tabular_alpha_vector(self,alpha,belief_id,timestep):
        self.point_value_fn[timestep][belief_id] = alpha


    def pruning(self,timestep):
        alpha_vectors = [alpha.vectors for alpha in self.vector_sets[timestep]]
        for fixed_leader_vector,follower_vector in alpha_vectors:
            check = 0
            for leader_vector,follower_vector in alpha_vectors:
                for state in CONSTANT.state:
                    if fixed_leader_vector[state]>leader_vector[state] : check+=1
                if check == len(CONSTANT.STATES):
                    leader_vector
                    #remove leader vector
        

        # check if this works


    def get_max_alpha(self,belief,timestep):
        """returns alpha vector object that gives the maximum value for a given belief at a certain timestep"""
        max = -np.inf
        max_alpha = None
        max_value = None
        for alpha in self.vector_sets[timestep]:
            leader_value,follower_value = alpha.get_value(belief)
            if leader_value>max :
                max = leader_value
                max_value = (leader_value,follower_value)
                max_alpha = alpha
        return max_alpha,max_value
    
    def get_max_alpha2(self,previous_belief_id,belief_id,timestep):
        """returns alpha vector object that gives the maximum value for a given belief at a certain timestep"""
        max = -np.inf
        max_alpha = None
        max_value = None
        belief = self.belief_space.get_belief(belief_id)
        for alpha in self.vector_sets[timestep]:
            leader_value,follower_value = alpha.get_value(belief)
            if leader_value>max and self.belief_space.network.check_connection(previous_belief_id,alpha.belief_id):
                max = leader_value
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
            if leader_value > max_value :
                max_value = leader_value
                max_follower_value = follower_value

        return max_value, max_follower_value
        
    
    

    def backup(self,belief_id,timestep,gametype):
        max_plane_alpha, tabular_alpha = self.solve(belief_id,gametype,timestep)
        # print(f"belief _id = {belief_id}, max_plane = {max_plane_alpha.vectors} , tabular = {tabular_alpha.vectors}")
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
    

    
    def contruct_beta(self, belief_id, timestep, mapping_belief_to_alpha, game_type):
        two_d_vectors = {}
        
        for agent in range(0,2):
            # initialize beta and choose appropriate reward
            reward = CONSTANT.REWARDS[game_type][agent]
            two_d_vectors[agent] = np.zeros((len(CONSTANT.STATES),len(CONSTANT.JOINT_ACTIONS)))

            # if statement for the beta vector of blind opponents in SOTA =TRUE stackelberg games
            if game_type=="stackelberg" and self.sota == True and agent==1 :
                two_d_vectors[1] = self.get_blind_beta(game_type)
                return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)

             #main loop to calculate beta values of each (state,action) pair
            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    # beta(x,u) = r(x,u)
                    two_d_vectors[agent][state][joint_action] = reward[joint_action][state]

                    # beta(x,u) += \sum_{z} \sum_{y} DYNAMICS(x,u,z,y) * V_{t+1}(Transition(belief,u,z)) [y]
                    for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                        next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                        if next_belief_id:
                            for next_state in CONSTANT.STATES:
                                two_d_vectors[agent][state][joint_action] += CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state] * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation] * mapping_belief_to_alpha[next_belief_id].vectors[agent][next_state]

        return BetaVector(two_d_vectors[0],two_d_vectors[1],self.problem)

    def construct_from_tabular_belief_to_alpha_mapping(self, belief_id, timestep):
        tabular_belief_to_alpha_mapping = {}

        for joint_action in CONSTANT.JOINT_ACTIONS:
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                if next_belief_id : 
                    if  timestep >= self.horizon-1: 
                        tabular_belief_to_alpha_mapping[next_belief_id] = AlphaVector(None, np.zeros(len(CONSTANT.STATES)), np.zeros(len(CONSTANT.STATES)),next_belief_id)
                    else :tabular_belief_to_alpha_mapping[next_belief_id]= self.point_value_fn[timestep+1][next_belief_id]

        return tabular_belief_to_alpha_mapping

    def construct_from_maxplane_belief_to_alpha_mapping(self, belief_id, timestep):
        maxplane_belief_to_alpha_mapping = {}

        for joint_action in CONSTANT.JOINT_ACTIONS:
            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                if next_belief_id : 
                    if  timestep >= self.horizon-1: maxplane_belief_to_alpha_mapping[next_belief_id] = AlphaVector(None, np.zeros(len(CONSTANT.STATES)), np.zeros(len(CONSTANT.STATES)),next_belief_id)
                    else:
                        alpha, _  = self.get_max_alpha2(belief_id,next_belief_id, timestep+1)
                        maxplane_belief_to_alpha_mapping[next_belief_id] = alpha
        return maxplane_belief_to_alpha_mapping


    def solve(self,belief_id,game_type,timestep):

        # print(f"solving timestep {timestep},belief id = {belief_id} ")
        # construct alpha mappings () and beta vectors for max plane solution (alpha mapping is mapping of (u,z) -> alpha vector) at a certain belief
        max_plane_beta = self.contruct_beta(belief_id, timestep, self.construct_from_maxplane_belief_to_alpha_mapping(belief_id, timestep), game_type)

        # construct tabular beta for tabular solution
        tabular_beta = self.contruct_beta(belief_id, timestep, self.construct_from_tabular_belief_to_alpha_mapping(belief_id, timestep), game_type)

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
        max_plane_alpha = max_plane_beta.get_alpha_vector(belief_id, game_type, max_plane_DR, self.sota)
        tabular_alpha = tabular_beta.get_alpha_vector(belief_id, game_type, tabular_DR, self.sota)
        # print( f"Game {game_type}  ::  max plane LP value: {max_plane_leader_value,max_plane_follower_value}, tabular LP value : {tabular_leader_value,tabular_follower_value}  --  Reconstructed Max plane alpha: {max_plane_alpha.get_value(belief)}, reconstructed tabular alpha : {tabular_alpha.get_value(belief)}  --  belief {belief.value}  -- DR {max_plane_DR}\n" )
       
        #printing
        if np.abs(max_plane_leader_value-tabular_leader_value)>0.1 :
            print(f"\n\n FOUND DIFFERENCE during backuo of belief ID : {belief_id} at timestep {timestep}! ")

            print( f"Game {game_type}  :: \n\tReconstructed Max plane alpha:{max_plane_alpha.vectors} , value ={max_plane_alpha.get_value(belief)}\n\treconstructed tabular alpha : {tabular_alpha.vectors}, value = {tabular_alpha.get_value(belief)}  --  belief {belief.value}  -- DR {max_plane_DR}\n" )
            print("looking into beta vector..\n")
            # print(f"\n\nBETA : \n max_plane:\n{max_plane_beta.two_d_vectors}\ntabular:\n{tabular_beta.two_d_vectors}")
           
            for agent in range(2):
                print(f"agent {agent}")
                for state in CONSTANT.STATES:
                    for joint_action in CONSTANT.JOINT_ACTIONS:
                        if np.abs(max_plane_beta.two_d_vectors[agent][state][joint_action]- tabular_beta.two_d_vectors[agent][state][joint_action])>0.01 :
                            print(f"\n\tbeta(x={state}, u={joint_action}),")
                            reward= CONSTANT.REWARDS[game_type][agent][joint_action][state]
                            print(f"\treward  = {reward} + future component from max_plane = {max_plane_beta.two_d_vectors[agent][state][joint_action]-reward}, from tabular = {tabular_beta.two_d_vectors[agent][state][joint_action]-reward}")
                            alpha_mapping = self.construct_from_maxplane_belief_to_alpha_mapping(belief_id, timestep)
                            tabular_mapping = self.construct_from_tabular_belief_to_alpha_mapping(belief_id,timestep)
                            for joint_observation in CONSTANT.JOINT_OBSERVATIONS:
                                next_belief_id = self.belief_space.network.existing_next_belief_id(belief_id,joint_action,joint_observation)
                                if next_belief_id and next_belief_id!=alpha_mapping[next_belief_id].belief_id: 
                                    # print(f"\t\tPr({joint_observation}|b_id={belief_id}, u={joint_action}) = {Utilities.observation_probability(joint_observation,belief,joint_action)} ,\n\t\tV^t+1(b_next = {next_belief_id}) = from max_plane (alpha built on belief_id {alpha_mapping[next_belief_id].belief_id}) : {alpha_mapping[next_belief_id].get_value(self.belief_space.get_belief(next_belief_id))} from tabular : {tabular_mapping[next_belief_id].get_value(self.belief_space.get_belief(next_belief_id))}  ")
                                    print(f"\n\t\talpha from z = {joint_observation} , V^t+1(b_next = {next_belief_id}) = from max_plane (alpha built on belief_id {alpha_mapping[next_belief_id].belief_id}) :{alpha_mapping[next_belief_id].vectors} , tabular : {tabular_mapping[next_belief_id].vectors}\n\t\t max_plane value :{alpha_mapping[next_belief_id].get_value(self.belief_space.get_belief(next_belief_id))} ,tabular value : {tabular_mapping[next_belief_id].get_value(self.belief_space.get_belief(next_belief_id))}  ")

                break
            # sys.exit() 

                     

        # return alpha vectors
        return max_plane_alpha , tabular_alpha

 
