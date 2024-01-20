import numpy as np
import itertools, os
from parser import read_file, read_count_or_enum, read_field, read_start, read_items, read_transition, read_observation, read_reward,read_rewards

class DecPOMDP:
    def __init__(self, problem, num_players, horizon, observation_histories = False, truncation = np.inf):
        self.name = problem
        self.horizon = horizon
        self.observation_histories = observation_histories
        self.truncation = truncation
        problem = os.path.join("problems", "%s.dpomdp" % (problem))
        raw_data = read_file(problem)
        self.agents = read_count_or_enum(raw_data["agents"][0], "agent")
        self.num_agents = len(self.agents)
        self.num_players = num_players
        self.discount = 1.#float(raw_data["discount"][0])
        self.val_type = read_field(raw_data["values"][0])[0]
        self.states = read_count_or_enum(raw_data["states"][0], "s")
        self.num_states = len(self.states)
        self.b0 = read_start(raw_data, self.states)
        self.actions = read_items(raw_data["actions"][0], [name + "_a" for name in self.agents])
        self.num_actions = [len(self.actions[a]) for a in range(self.num_agents)]
        self.joint_actions = list(itertools.product(*self.actions))
        self.num_joint_actions = len(self.joint_actions)
        self.observations = read_items(raw_data["observations"][0], [name + "_z" for name in self.agents])
        self.num_observations = [len(self.observations[a]) for a in range(self.num_agents)]
        self.joint_observations = list(itertools.product(*self.observations))
        self.num_joint_observations = len(self.joint_observations)
        self.all_histories = [self._generate_all_histories(t) for t in range(self.horizon)]
        self.transition_fn = read_transition(raw_data["T"], self.states, self.actions)
        self.observation_fn = read_observation(raw_data["O"], self.states, self.actions, self.observations)
        self.reward_fn = read_rewards(raw_data["R"], self.states, self.actions, self.observations)
        self.reward_fn_sa = np.zeros((self.num_joint_actions, self.num_states))
        if self.num_players > 1:
            self.reward_fn_sa = np.zeros((self.num_players,self.num_joint_actions, self.num_states))
            for i in range(num_players):
                for ja in range(self.num_joint_actions):
                    for s in range(self.num_states):
                        self.reward_fn_sa[i,ja, s] = sum([self.reward_fn[i,ja, s, s1, jz] * self.transition_fn[ja, s, s1] * self.observation_fn[ja, s1, jz] for s1 in range(self.num_states) for jz in range(self.num_joint_observations)])

        else:
            for ja in range(self.num_joint_actions):
                for s in range(self.num_states):
                    self.reward_fn_sa[ja, s] = sum([self.reward_fn[ja, s, s1, jz] * self.transition_fn[ja, s, s1] * self.observation_fn[ja, s1, jz] for s1 in range(self.num_states) for jz in range(self.num_joint_observations)])
    
        self.action_dictionary={}
        n=0
        for idx1,u1 in enumerate(self.actions[0]):
            for idx2,u2 in enumerate(self.actions[1]):
                self.action_dictionary[f"{idx1}{idx2}"] = n
                n+=1

        self.action_dict = dict((val, key) for key, val in self.action_dictionary.items())

    def get_joint_action(self,u1,u2):
        return self.action_dictionary[f"{u1}{u2}"]
    def get_seperate_action(self,uj):
        action_Join = int(self.action_dict[uj])
        return action_Join//10,action_Join%10
    def reset(self):
        self.state = np.random.choice(self.states, p = self.b0)
        self.n_steps = 0
        return self.b0

    def step(self, action):
        assert action in self.joint_actions
        old_state = self.state
        self.state = np.random.choice(self.states, p = self.transition_fn[self.joint_actions.index(action), self.states.index(self.state)])
        observation = self.joint_observations[np.random.choice(self.num_joint_observations, p = self.observation_fn[self.joint_actions.index(action), self.states.index(self.state)])]
        reward = self.reward_fn[self.joint_actions.index(action), self.states.index(old_state), self.states.index(self.state), self.joint_observations.index(observation)]
        self.n_steps += 1


        return observation, reward, (self.n_steps >= self.horizon), {}
    def step_isolate(self,action,obs):
        action = self.joint_actions[action]
        assert action in self.joint_actions
        old_state = self.state
        new_state = np.random.choice(self.states, p = self.transition_fn[self.joint_actions.index(action), self.states.index(self.state)])

        observation = self.joint_observations[obs]
        reward = self.reward_fn[self.joint_actions.index(action), self.states.index(old_state), self.states.index(new_state), self.joint_observations.index(observation)]
        return reward


    def render(self, mode = "human"):
        pass

    def close(self):
        pass

    # Used to reset the environment to a specific configuration (quite violating RL formulation)
    def set(self, state, n_steps):
        self.state = state
        self.n_steps = n_steps

    def _generate_all_histories(self, t):
        all_product_set = []
        for a in range(self.num_agents):
            if self.observation_histories:
                product_set = [self.observations[a]] * min(t, self.truncation)
            else:
                product_set = [self.actions[a], self.observations[a]] * min(t, self.truncation)
            all_hist_per_agent = list(itertools.product(*product_set))
            all_product_set.append(all_hist_per_agent)
        return list(itertools.product(*all_product_set))

    def expand_history(self, history, action, observation):
        history = list(history)
        for a in range(self.num_agents):
            if self.observation_histories:
                history[a] = history[a] + (observation[a],)
            else:
                history[a] = history[a] + (action[a], observation[a],)
            if len(history[a]) > (2 - self.observation_histories) * self.truncation:
                history[a] = history[a][(-2 + self.observation_histories) * self.truncation:]
        return tuple(history)

    def is_prefix_history(self, history, new_history):
        return np.all([history[a][(2 - self.observation_histories) * (not np.isinf(self.truncation)):] == new_history[a][:-2 + self.observation_histories] for a in range(self.num_agents)])