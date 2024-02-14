import numpy as np
from decpomdp import DecPOMDP
import random

class Constants:
    _instance = None
    
   #DEFINE CONSTANTS
    def __init__(self,problem) :
        self.HORIZON = problem.horizon
        self.NAME = problem.name
        self.PROBLEM = problem
        self.STATES = [i for i in range(len(problem.states))]
        self.ACTIONS = [[i for i in range(len(problem.actions[0]))],[j for j in range(len(problem.actions[1]))]]
        self.JOINT_ACTIONS = [i for i in range(len(problem.joint_actions))]
        self.JOINT_OBSERVATIONS = [i for i in range(len(problem.joint_observations))]
        self.TRANSITION_FUNCTION = problem.transition_fn
        self.OBSERVATION_FUNCTION = problem.observation_fn
        self.REWARDS = self.initialize_rewards()
        self.PROBLEM.reset()
        self.gametypes = ["cooperative","stackelberg","zerosum"]
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise ValueError("Constants has not been initialized.")
        return cls._instance

    @classmethod
    def initialize(cls, value):
        if cls._instance is not None:
            raise ValueError("Constants has already been initialized.")
        cls._instance = cls(value)

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
        # Generate five random integers between 1 and 10
        for joint_action in self.JOINT_ACTIONS:
            for state in self.STATES:
                stackelberg_follower_reward[joint_action][state]+=random.randint(min_NUM, max_NUM)
        return stackelberg_follower_reward