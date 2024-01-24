import numpy as np
from decpomdp import DecPOMDP
import random

class Constants:
   #DEFINE CONSTANTS
    def __init__(self,problem) :
        self.HORIZON = problem.horizon
        self.NAME = problem.name
        self.PROBLEM = problem
        self.STATES = [i for i in range(len(problem.states))]
        self.ACTIONS = [[i for i in range(len(problem.actions[0]))],[j for j in range(len(problem.actions[1]))]]
        self.JOINT_ACTIONS = [i for i in range(len(problem.joint_actions))]
        self.JOINT_OBSERVATIONS = [i for i in range(len(problem.joint_observations))]
        self.TRANSITION_FUNCTION = np.transpose(problem.transition_fn, axes=(0,2,1))
        self.OBSERVATION_FUNCTION =problem.observation_fn.reshape((problem.num_joint_actions,problem.num_joint_observations,problem.num_states))
        self.REWARDS = self.initialize_rewards()
        self.LEADER = 0
        self.FOLLOWER = 1

    def initialize_rewards(self,):
        #Competitive reward matrix indexed by joint actions
        self.REWARDS = { "cooperative" : [self.PROBLEM.reward_fn_sa,self.PROBLEM.reward_fn_sa],
                    "zerosum" : [self.PROBLEM.reward_fn_sa,self.PROBLEM.reward_fn_sa*-1],
                    "stackelberg" :[self.PROBLEM.reward_fn_sa,self.follower_stackelberg_reward()]
                    }
        return self.REWARDS
    
    def follower_stackelberg_reward(self):
        seed_value = 42
        random.seed(seed_value)

        stackelberg_follower_reward = np.zeros(self.PROBLEM.reward_fn_sa.shape)
        min_NUM = int(min([min(row)for row in self.PROBLEM.reward_fn_sa]))
        max_NUM = int(max([max(row) for row in self.PROBLEM.reward_fn_sa]))
        for joint_action in self.JOINT_ACTIONS:
            for state in self.STATES:
                stackelberg_follower_reward[joint_action][state]+=random.randint(min_NUM, max_NUM)
        return stackelberg_follower_reward