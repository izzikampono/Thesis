import numpy as np
from constant import Constants
CONSTANT = Constants.get_instance()

class AlphaVector:
    def __init__(self,DR,vector1,vector2,sota=False):
        self.DR = DR
        self.sota = sota
        self.vectors = [vector1,vector2]
        
        
    def get_value(self,belief):
        return np.dot(belief.value,self.vectors[0]),np.dot(belief.value,self.vectors[1])

    def print_vector(self):
        print(self.vectors)

    def set_value(self,agent,hidden_state,value):
        self.vectors[agent][hidden_state] = value


class BetaVector:
    def __init__(self,two_d_vector_0,two_d_vector_1,problem):
        self.problem = problem
        self.two_d_vectors = [two_d_vector_0,two_d_vector_1]

    def get_zerosum_sota_alpha_vector(self,belief,game_type,DR,sota=True):
        """ finds lowest security level of leader """
        vectors = np.zeros((2,len(CONSTANT.STATES)))
        if game_type == "zerosum" and sota == True:

            follower_best_action = None
            follower_best_action_value = -np.inf

            # get best follower action for all posible leader actions
            for follower_action in CONSTANT.ACTIONS[1]:
                follower_action_value = 0
                for state in CONSTANT.STATES:
                    for leader_action,leader_action_probability in enumerate(DR.individual[0]):
                        # value (u^2) += a^1(u^1) * b(x) * Beta^2(x,u^1,u^2)
                        follower_action_value += leader_action_probability * belief.value[state] * self.two_d_vectors[0][state][self.problem.get_joint_action(leader_action, follower_action)]

                    if follower_best_action_value < follower_action_value:
                        follower_best_action = follower_action
                        follower_best_action_value = follower_action_value 

            for state in CONSTANT.STATES:
                for leader_action in CONSTANT.ACTIONS[0]:
                    vectors[0][state] +=  DR.individual[0][leader_action] * self.two_d_vectors[0][state][self.problem.get_joint_action(leader_action, follower_best_action)]
                    vectors[1][state] +=  DR.individual[1][state][follower_best_action] * self.two_d_vectors[1][state][self.problem.get_joint_action(leader_action,follower_best_action)]
            # print(f"reconstructed vector : {vectors[0]} , {vectors[1]}")
            return AlphaVector(DR,vectors[0],vectors[1],sota)


    def get_alpha_vector(self,belief, game_type, DR, sota=False):
        vectors = np.zeros((2,len(CONSTANT.STATES)))

        if game_type == "zerosum" and sota == True:
           return self.get_zerosum_sota_alpha_vector(belief,game_type,DR,sota)
        
        else:    
            for state in CONSTANT.STATES:
                for joint_action in CONSTANT.JOINT_ACTIONS:
                    vectors[0][state] += DR.joint[state][joint_action] * self.two_d_vectors[0][state][joint_action] 
                    vectors[1][state] += DR.joint[state][joint_action] * self.two_d_vectors[1][state][joint_action]
        return AlphaVector(DR,vectors[0],vectors[1], sota)
    
    
        
    def print_vector(self):
        print(self.two_d_vectors)