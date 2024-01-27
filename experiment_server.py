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
# python experiment_server.py problem=dectiger horizon=5 iter=1

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




# solve
def SOLVE(game,iterations):
    start_time = time.time()
    policy,leader_values,follower_values,belief_sizes = game.solve(iterations)
    end_time = time.time()
    solve_time = end_time - start_time
    return policy,leader_values,follower_values,solve_time,belief_sizes

def initialize_storage():
    database = {"gametype":[],
                "SOTA" : [],
                "horizon": [],
                    "num_iterations" : [],
                    "average_time" : [],
                    "number_of_beliefs" : [],
                    "ave_leader_value_b0":[],
                    "ave_follower_value_b0":[],
                    "density" : [],
                    "gap":[]
                   
                    }
    policies = {"cooperative" : [] ,"zerosum":[],"stackelberg":[]}
    policy_comparison_matrix = {"cooperative" : [] ,"zerosum":[],"stackelberg":[]}
    return database,policies,policy_comparison_matrix

def add_to_database(database,horizon,SOTA,game_type,num_iterations,average_time,num_beliefs,V0_B0,V1_B0,gap,density):
    sota = {True:"State of the Art" , False:"Stackelberg"}
    database["gametype"].append(game_type)
    database["horizon"].append(horizon)
    database["SOTA"].append(sota[SOTA])
    database["num_iterations"].append(num_iterations)
    database["average_time"].append(average_time)
    database["number_of_beliefs"].append(num_beliefs)
    database["ave_leader_value_b0"].append(V0_B0)
    database["ave_follower_value_b0"].append(V1_B0)
    database["gap"].append(abs(V0_B0-V1_B0))
    database["density"].append(density)
    return

def export_database(database):
    database = pd.DataFrame(database)
    path = "server_results/"
    file= f"{file_name}_{planning_horizon}_{num_iterations}.csv"
    database.to_csv(path+file, index=False)
    return

def initialize_problem():
    problem = DecPOMDP(file_name,horizon = planning_horizon, num_players=1)
    print("GAME DESCRIPTION :")
    print(f"game size :\n\t|S| = {len(problem.states)}")
    print(f"\t|Z| = {problem.num_joint_observations}\n\t|U| = {problem.num_joint_actions} with |U_i| = {problem.num_actions[0]}")
    print(f"intiial_belief : {problem.b0}")
    Classes.set_problem(problem)
    return problem

def export_policy_matrix(policy_comparison_matrix):
    for gametype in ["cooperative","zerosum","stackelberg"]:
        database = pd.DataFrame(policy_comparison_matrix[gametype],columns=["Strong F","Weak F"],index=["Strong L","Weak L"])
        path = "policy_matrix/"
        file= f"{file_name}_{gametype}_{horizon}_{num_iterations}.csv"
        database.to_csv(path+file, index=False)
    return
print("\n\nInitializing problem .... waiting ...")
problem = initialize_problem()
database,policies,policy_comparison_matrix = initialize_storage()

for gametype in ["cooperative","zerosum","stackelberg"]:
    growth = 1.5
    for sota_ in [False,True]:
        for horizon in range(1,planning_horizon+1):
            density = 0.2
            print(f"\n============= {gametype} GAME WITH HORIZON {horizon} , SOTA {sota_} ===========")
            #initialize game with fixed planning horizon
            game = Classes.PBVI(problem=problem.set_horizon(horizon),horizon=horizon,density=density,growth=growth,gametype=gametype,sota=sota_)
            #solve game with num_iterations
            policy,leader_values,follower_values, time_,belief_sizes  = SOLVE(game,num_iterations)
            #add values to database
            for iters in range(num_iterations):
                add_to_database(database,horizon,sota_,gametype,iters+1,time_,belief_sizes[iters],leader_values[iters],follower_values[iters],np.abs(leader_values[iters]-follower_values[iters]),density)
                density = density*growth

        policies[gametype].append(policy)

#compare SOTA an non-SOTA trategy of each gametype
game.build_comparison_matrix(policy_comparison_matrix,policies)

#export
print("Calculations done... exporting to csv....")
export_database(database)
export_policy_matrix(policy_comparison_matrix)





