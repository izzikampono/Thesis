import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decpomdp import DecPOMDP
import Classes
from constant import Constants
import time
import os
import sys

#example run : experiment.py recycling zerosum 3 1 1

if len(sys.argv) < 2:
    print("err0r : not enough arguments given")
    sys.exit(1)
if len(sys.argv)> 5 :
    file_name = str(sys.argv[1])
    game_type = str(sys.argv[2])
    planning_horizon = int(sys.argv[3])
    num_iterations = int(sys.argv[4])
    sota_ = bool(int(sys.argv[5]))
else : 
    print("not enough arguments")
    sys.exit()


#import problem
problem = DecPOMDP(file_name, 1,horizon=planning_horizon)
Classes.set_problem(problem)

print("GAME INITIATED :")
print(f"game of type {game_type} initiated with SOTA set to = {sota_} with horizon {planning_horizon}")
print(f"game size :\n\t|S| = {len(problem.states)}")
print(f"\t|Z| = {problem.num_joint_observations}\n\t|U| = {problem.num_joint_actions} with |U_i| = {problem.num_actions[0]}")
print(f"intiial_belief : {problem.b0}")
print("\n\n\n")

# solve
def SOLVE(game):
    start_time = time.time()
    policy = game.solve(3,0.9)
    end_time = time.time()
    solve_time = end_time - start_time
    value_fn = game.value_function
    return policy,solve_time,value_fn

def initialize_database():
    database = {"gametype":[],
                "SOTA" : [],
                "horizon": [],
                    "num_iterations" : [],
                    "average_time" : [],
                    "number_of_beliefs" : [],
                    "leader_value_b0":[],
                    "follower_value_b0":[]
                    # "density" = []
                    # "gap":[]
                   
                    }
    return database

def add_to_database(database,horizon,game_type,num_iterations,average_time,num_beliefs,V0_B0,V1_B0,SOTA):
    database["gametype"].append(game_type)
    database["horizon"].append(horizon)
    database["SOTA"].append(SOTA)
    database["num_iterations"].append(num_iterations)
    database["average_time"].append(average_time)
    database["number_of_beliefs"].append(num_beliefs)
    database["leader_value_b0"].append(V0_B0)
    database["follower_value_b0"].append(V1_B0)
    # database["gap"].append(abs(V0_B0-V1_B0))
    # database["density"].append(density)
    return

database = initialize_database()
for sota_ in [True,False]:
    for horizon in range(1,planning_horizon+1):
        print(f"\n===== GAME WITH HORIZON {horizon} , SOTA {sota_} =====")
        game = Classes.PBVI(problem=problem,horizon=horizon,density=0.1,gametype=game_type,sota=sota_)
        policy, time_ , value_fn = SOLVE(game)
        num_beliefs = game.belief_space.belief_size()
        value0,value1= value_fn.get_values_initial_belief()
        add_to_database(database,horizon,game_type,2,time_,num_beliefs,value0,value1,sota_)

database = pd.DataFrame(database)
file_name = f"/Results/{file_name}_{game_type}_{horizon}_experiment_results.csv"
database.to_csv(file_name, index=False)
print(f"RESULTS WRITTEN AS : {file_name}:\n")



