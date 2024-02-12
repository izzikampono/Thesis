from constant import Constants
import numpy as np
import sys
import utilities as Utilities
CONSTANT = Constants.get_instance()

class Belief:
    def __init__(self,value,action_label,observation_label,id=None):
        self.value = value 
        self.action_label = action_label
        self.observation_label = observation_label
        self.id = id

    def set_id(self,id):
        self.id=id
        return
    
    def __eq__(self, other):
        return (isinstance(other, Belief) and 
                self.action_label == other.action_label and 
                self.observation_label == other.observation_label and self.id == other.id)

    def __hash__(self):
        # Hash only using certain unique attributes
        return hash((self.action_label, self.observation_label,self.id))

    def next_belief(self,joint_DR,joint_observation):
        """function to calculate next belief based on current belief, compatible with probabilistic joint_DR/ deterministc joint action , and observation"""
        # returns the value of b1
        next_belief_value= np.zeros(len(self.value))

        if type(joint_observation) != int :
            joint_observation = self.PROBLEM.joint_observations.index(joint_observation)


        if type(joint_DR) == int: # if joint_DR enterred as a deterministic action 
            # b_next(x') = \sum_{x} += b_current[x] * TRANSITION_MATRIX(u_j,x',z_j,x)
            for next_state in CONSTANT.STATES:
                value = 0
                for state in CONSTANT.STATES:
                    value += self.value[state] * CONSTANT.TRANSITION_FUNCTION[joint_DR][state][next_state]  * CONSTANT.OBSERVATION_FUNCTION[joint_DR][state][joint_observation]
                next_belief_value[next_state]+=value  

        else:   # if joint_DR is a decision rule
            # b_next(x') = \sum_{x} \sum_(u_j) += b_current[x] * a(u_j) TRANSITION_MATRIX(u_j,x',z_j,x)
            for next_state in CONSTANT.STATES:
                value = 0
                for state in CONSTANT.STATES:
                    for joint_action in CONSTANT.JOINT_ACTIONS:
                        value += self.value[state] * joint_DR[joint_action] * CONSTANT.TRANSITION_FUNCTION[joint_action][state][next_state]  * CONSTANT.OBSERVATION_FUNCTION[joint_action][state][joint_observation]
                next_belief_value[next_state]+=value

        if np.sum(next_belief_value) == 0 :
            return None
        
        # if not none normalize
        next_belief_value = Utilities.normalize(next_belief_value)

        if np.sum(next_belief_value)<= 1.001 and np.sum(next_belief_value)> 0.99999:
            return  Belief(next_belief_value,joint_DR,joint_observation)
        else:
            print("err0r : belief doesn not sum up to 1\n")
            print(f"current belief: \n{self.value}")
            print(f"next belief :\n{next_belief_value}")
            print(f"sum : {np.sum(next_belief_value)}")
            sys.exit()
        