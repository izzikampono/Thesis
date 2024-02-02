import numpy as np
class AlphaVector:
    def __init__(self,DR,vector1,vector2,problem,sota=False):
        self.DR = DR
        self.sota = sota
        self.vectors = [vector1,vector2]
        self.problem = problem
        
        
    def get_value(self,belief):
        return np.dot(belief.value,self.vectors[0]),np.dot(belief.value,self.vectors[1])

    def print_vector(self):
        print(self.vectors)

    def set_value(self,agent,hidden_state,value):
        self.vectors[agent][hidden_state] = value
