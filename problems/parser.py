import operator
from functools import reduce
import numpy as np

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def read_file(filename):
    with open(filename, 'r') as f:
        raw_file = '\n'.join(l.strip() for l in f if not l.startswith('#'))
    tokens = ["agents", "discount", "values", "states", "start", "start include", "start exclude", "actions", "observations", "T", "O", "R"]
    raw_data = {tok: [] for tok in tokens}
    splits = []
    last = 0
    for tok in tokens:
        pos = raw_file.find(tok + ":", last)
        while pos > -1:
            splits.append((tok, pos))
            last = pos
            pos = raw_file.find(tok + ":", pos + len(tok))
    for (tok, pos), (_, nxt) in zip(splits[:-1], splits[1:]):raw_data[tok].append(raw_file[pos + len(tok) + 1:nxt])
    (tok, pos) = splits[-1]
    raw_data[tok].append(raw_file[pos + len(tok) + 1:])
    return raw_data

def read_line(raw_str):
    return [field.strip() for field in raw_str.split(":") if field]

def read_field(raw_str):
    return [val.strip(" \n\"\'") for val in raw_str.split()]

def read_count_or_enum(raw_str, prefix):
    try:
        count = int(raw_str)
        return ["{}{}".format(prefix, i) for i in range(count)]
    except ValueError:
        return read_field(raw_str)

def read_id_or_item(raw_str, items):
    try:
        return int(raw_str)
    except ValueError:
        return items.index(raw_str)

def match_item(raw_str, items):
    if raw_str == "*":
        if len(items) > 1:
            for ji in match_item(" ".join("*" * len(items)), items):
                yield ji
        else:
            for i in range(len(items[0])):
                yield i
    else:
        if len(items) > 1:
            left, cur = raw_str.rsplit(maxsplit = 1)
            for ji in match_item(left, items[:-1]):
                for i in match_item(cur, [items[-1]]):
                    yield len(items[-1]) * ji + i
        else:
            yield read_id_or_item(raw_str, items[0])

def read_items(raw_str, prefixes):
    lines = raw_str.splitlines()
    return [read_count_or_enum(line, prefix) for line, prefix in zip(lines[1:], prefixes)]

def read_start(raw_dict, states):
    start = None
    if raw_dict["start"]:
        lines = raw_dict["start"][0].splitlines()
        fields = read_line(lines[0])
        if len(fields) == 0:
            f = read_field(lines[1])
            if f[0] == "uniform":
                start = np.ones(len(states)) / len(states)
            else:
                start = np.array([float(p) for p in f])
        elif len(fields) == 1:
            start = np.zeros(len(states))
            start[read_id_or_item(fields[0], states)] = 1
    elif raw_dict["start include"]:
        included = read_field(raw_dict["start include"][0])
        start = np.zeros(len(states))
        for s_id in map(lambda s:read_id_or_item(s, states), included):
            start[s_id] = 1 / len(included)
    elif raw_dict["start exclude"]:
        excluded = read_field(raw_dict["start exclude"][0])
        start = np.ones( len(states) ) / (len(states) - len(excluded))
        for s_id in map(lambda s: read_id_or_item(s, states), excluded):
            start[s_id] = 0
    return start

def read_transition(raw_list, states, actions):
    n_ja = prod([len(a) for a in actions])
    transition_mat = np.zeros((n_ja, len(states), len(states)))
    for rule in raw_list:
        lines = rule.splitlines()
        fields = read_line(lines[0])
        f = None
        if len(lines) > 1:
            f = read_field(lines[1])
        for ja in match_item(fields[0], actions):
            if len(fields) > 1:
                for s in match_item(fields[1], [states]):
                    if len(fields) > 2:
                        for nxt in match_item(fields[2], [states]):
                            transition_mat[ja, s, nxt] = float(fields[3])
                    elif f[0] == "uniform":
                        transition_mat[ja, s, :] = 1 / len(states)
                    elif f[0] == "identity":
                        transition_mat[ja, s, :] = 0
                        transition_mat[ja, s, s] = 1
                    else:
                        for nxt, p in enumerate(f):
                            transition_mat[ja, s, nxt] = float(p)
            elif f[0] == "uniform":
                transition_mat[ja, :, :] = 1 / len(states)
            elif f[0] == "identity":
                transition_mat[ja, :, :] = np.eye(len(states))
            else:
                for s in range(len(states)):
                    for nxt, p in enumerate(read_field(lines[s + 1])):
                        transition_mat[ja, s, nxt] = float(p)
    return transition_mat

def read_observation(raw_list, states, actions, observations):
    n_ja = prod([len(a) for a in actions])
    n_jz = prod([len(z) for z in observations])
    observation_mat = np.zeros((n_ja, len(states), n_jz))
    for rule in raw_list:
        lines = rule.splitlines()
        fields = read_line(lines[0])
        f = None
        if len(lines) > 1:
            f = read_field(lines[1])
        for ja in match_item(fields[0], actions):
            if len(fields) > 1:
                for nxt in match_item(fields[1], [states]):
                    if len(fields) > 2:
                        for jz in match_item(fields[2], observations):
                            observation_mat[ja, nxt, jz] = float(fields[3])
                    elif f[0] == "uniform":
                        observation_mat[ja, nxt, :] = 1 / n_jz
                    else:
                        for jz, p in enumerate(f):
                            observation_mat[ja, nxt, jz] = float(p)
            elif f[0] == "uniform":
                observation_mat[ja, :, :] = 1 / n_jz
            else:
                for nxt in range(len(states)):
                    for jz, p in enumerate(read_field(lines[nxt + 1])):
                        observation_mat[ja, nxt, jz] = float(p)
    return observation_mat

def read_reward(raw_list, states, actions, observations):
    n_ja = prod([len(a) for a in actions])
    n_jz = prod([len(z) for z in observations])
    reward_mat = np.zeros((n_ja, len(states), len(states), n_jz))
    for rule in raw_list:
        lines = rule.splitlines()
        fields = read_line(lines[0])
        for ja in match_item(fields[0], actions):
            for s in match_item(fields[1], [states]):
                if len(fields) > 2:
                    for nxt in match_item(fields[2], [states]):
                        if len(fields) > 3:
                            for jz in match_item(fields[3], observations):
                                reward_mat[ja, s, nxt, jz] = float(fields[4])
                        else:
                            for jz, r in enumerate(read_field(lines[1])):
                                reward_mat[ja, s, nxt, jz] = float(r)
                else:
                    for nxt in range(len(states)):
                        for jz, r in enumerate(read_field(lines[1 + nxt])):
                            reward_mat[ja, s, nxt, jz] = float(r)
    return reward_mat

def read_rewards(raw_list, states, actions, observations):
    n_ja = prod([len(a) for a in actions])
    n_jz = prod([len(z) for z in observations])
    flag = True
    reward_mat = np.zeros((n_ja, len(states), len(states), n_jz))
    for rule in raw_list:
        lines = rule.splitlines()
        fields = read_line(lines[0])
        for ja in match_item(fields[0], actions):
            for s in match_item(fields[1], [states]):
                if len(fields) > 2:
                    for nxt in match_item(fields[2], [states]):
                        if len(fields) > 3:
                            for jz in match_item(fields[3], observations):
                                if len(fields) > 4:
                                    if len(fields)>5:
                                        player = int(fields[4])
                                        # print(f"test this : {float(fields[5])}")
                                        if flag :
                                            reward_mat = np.zeros((2,n_ja, len(states), len(states), n_jz))
                                            flag = False
                                        reward_mat[player,ja, s, nxt, jz] =float(fields[5])
                                    else: 
                                        reward_mat[ja, s, nxt, jz] = float(fields[4])       
                        else:
                            for jz, r in enumerate(read_field(lines[1])):
                                reward_mat[ja, s, nxt, jz] = float(r)
                else:
                    for nxt in range(len(states)):
                        for jz, r in enumerate(read_field(lines[1 + nxt])):
                            reward_mat[ja, s, nxt, jz] = float(r)
    return reward_mat