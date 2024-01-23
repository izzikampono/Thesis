import numpy as np
import pandas as pd
from decpomdp import DecPOMDP
import Classes
from constant import Constants
import time
import sys
import gc

gc.enable()

#example run :
# python experiment.py problem=dectiger horizon=10 iter=10

if len(sys.argv) < 2:
    print("err0r : not enough arguments given")
    sys.exit(1)
if len(sys.argv)> 3 :
    file_name = str(sys.argv[1]).split("=")[1]
    planning_horizon = int(sys.argv[2].split("=")[1])
    num_iterations = int(sys.argv[3].split("=")[1])
else : 
    print("not enough arguments")
    sys.exit()

#import problem
problem = DecPOMDP(file_name, 1,horizon=planning_horizon)
Classes.set_problem(problem)
density = 0.1


# solve
def SOLVE(game):
    start_time = time.time()
    policy = game.solve(num_iterations,0.9)
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
                    "follower_value_b0":[],
                    "density" : [],
                    "gap":[],
                    }
    return database

def add_to_database(database,horizon,game_type,num_iterations,average_time,num_beliefs,V0_B0,V1_B0,SOTA,density):
    database["gametype"].append(game_type)
    database["horizon"].append(horizon)
    database["SOTA"].append(SOTA)
    database["num_iterations"].append(num_iterations)
    database["average_time"].append(average_time)
    database["number_of_beliefs"].append(num_beliefs)
    database["leader_value_b0"].append(V0_B0)
    database["follower_value_b0"].append(V1_B0)
    database["gap"].append(np.abs(V1_B0-V0_B0))
    # database["gap"].append(abs(V0_B0-V1_B0))
    database["density"].append(density)
    return

database = initialize_database()
for game_type in ["cooperative","stackelberg","zerosum"]:
    for sota_ in [True,False]:
        for horizon_ in range(1,planning_horizon+1):
            print(f"\n===== GAME of type {game_type} WITH HORIZON {horizon_} , SOTA {sota_} =====")
            problem = DecPOMDP(file_name,horizon = horizon_, num_players=1)
            Classes.set_problem(problem)
            game = Classes.PBVI(problem=problem,horizon=horizon_,density=density,gametype=game_type,sota=sota_)
            policy, time_ , value_fn = SOLVE(game)
            num_beliefs = game.belief_space.belief_size()
            value0,value1= value_fn.get_values_initial_belief()
            add_to_database(database,horizon_,game_type,2,time_,num_beliefs,np.average(value0),np.average(value1),sota_,density)
print("Calculations done... exporting to csv....")
database = pd.DataFrame(database)
file_name = f"{file_name}_{planning_horizon}.csv"
path = "/server_results/"
database.to_csv(path+file_name, index=False)
print(f"RESULTS WRITTEN AS : {file_name}:\n")



