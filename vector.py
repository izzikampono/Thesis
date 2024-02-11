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

    def get_alpha_vector(self,belief, game_type, DR, sota=False):
        vectors = np.zeros((2,len(CONSTANT.STATES)))
        for state in CONSTANT.STATES:
            for joint_action in CONSTANT.JOINT_ACTIONS:
                vectors[0][state] += DR.joint[state][joint_action] * self.two_d_vectors[0][state][joint_action] 
                vectors[1][state] += DR.joint[state][joint_action] * self.two_d_vectors[1][state][joint_action]
        
        return AlphaVector(DR,vectors[0],vectors[1], sota)
    
    
        
    def print_vector(self):
        print(self.two_d_vectors)