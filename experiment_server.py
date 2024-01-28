import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decpomdp import DecPOMDP
import Classes
import time
import sys
import gc
gc.enable()
# examplus usage :
# python3 experiment_server.py problem=dectiger horizon=3 iter=3 

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

print("Initializing problem...")
games = ["cooperative","zerosum","stackelberg"]
problem = DecPOMDP(file_name, 1,horizon=planning_horizon)
Classes.set_problem(problem)

print(f"{file_name} Problem description:")
print(f"game size :\n\t|S| = {len(problem.states)}")
print(f"\t|Z| = {problem.num_joint_observations}\n\t|U| = {problem.num_joint_actions} with |U_i| = {problem.num_actions[0]}")
print(f"intiial_belief : {problem.b0}")


def add_to_database(database,horizon,game_type,num_iterations,average_time,num_beliefs,V0_B0,V1_B0,SOTA,density,gap):
    sota = {True:"State of the Art" , False:"Stackelberg"}
    database["gametype"].append(game_type)
    database["horizon"].append(horizon)
    database["SOTA"].append(sota[SOTA])
    database["num_iterations"].append(num_iterations)
    database["average_time"].append(average_time)
    database["number_of_beliefs"].append(num_beliefs)
    database["leader_value"].append(V0_B0)
    database["follower_value"].append(V1_B0)
    database["gap"].append(abs(V0_B0-V1_B0))
    database["density"].append(density)
    return


# solve
def SOLVE(game,database,horizon,gametype):
    start_time = time.time()
    policy , belief_size , densities,values = game.solve(num_iterations,0.6)
    end_time = time.time()
    solve_time = end_time - start_time
    for iterations in range(num_iterations):
        # value0,value1= policy[0][iterations].value , policy[1][iterations].value
        add_to_database(database,horizon,gametype,iterations,solve_time,belief_size[iterations],values[iterations][0],values[iterations][1],sota_,densities[iterations],np.abs(values[iterations][0]-values[iterations][1]))
    return policy

def plots(database):
    Strong_leader = database[database["SOTA"]=="Stackelberg"]
    Strong_leader = Strong_leader[Strong_leader["horizon"]==planning_horizon]
    Weak_leader = database[database["SOTA"]!="Stackelberg"]
    Weak_leader= Weak_leader[Weak_leader["horizon"]==planning_horizon]

    fig, axs = plt.subplots(len(games), figsize=(10, 12),constrained_layout=True)

    for idx,gametype in enumerate(games):
        strong_leader_data = Strong_leader[Strong_leader["gametype"]==gametype]["leader_value"]
        weak_leader_data = Weak_leader[Weak_leader["gametype"]==gametype]["leader_value"]
        
        axs[idx].plot(range(len(strong_leader_data)),strong_leader_data, label='Strong Leader Strong Follower')
        axs[idx].plot(range(len(weak_leader_data)),weak_leader_data, label='Weak Leader Weak Leader')
        axs[idx].set_title(f'{gametype} game'.format(idx+1))
        axs[idx].set_ylabel('Leader Value')
        axs[idx].set_xlabel('Iterations')
        axs[idx].legend()
    # plt.title("Results of PBVI algorithm for {file_name}")
    file_path = f'plots/result_{file_name}_{planning_horizon}_{num_iterations}.png'
    fig.savefig(file_path)
    plt.subplots_adjust(top=0.5,bottom=0.4)
    plt.show()
    print("plots made and exported...")
    return

def initialize_storage():
    database = {"gametype":[],
                "SOTA" : [],
                "horizon": [],
                    "num_iterations" : [],
                    "average_time" : [],
                    "number_of_beliefs" : [],
                    "leader_value":[],
                    "follower_value":[],
                    "density" : [],
                    "gap":[]
                   
                    }
    policies = {"cooperative" : {True:[],False:[]} ,"zerosum":{True:[],False:[]},"stackelberg":{True:[],False:[]}}
    policy_comparison_matrix = {"cooperative" : [] ,"zerosum":[],"stackelberg":[]}
    return database,policies,policy_comparison_matrix


def export_policy_matrix(policy_comparison_matrix,gametype):
    matrix = pd.DataFrame(policy_comparison_matrix[gametype],columns=["Strong Follower","Weak Follower"],index=["Strong Leader","Weak Leader"])
    path = "comparison_matrix/"
    filename = f"{file_name}_{gametype}_{planning_horizon}_{num_iterations}.csv"
    matrix.to_csv(path+filename)
    return


def export_database(database):
    path = "Results/"
    filename = f"{file_name}_{horizon}_experiment_results.csv"
    database.to_csv(path+filename, index=False)
    print(f"file exported as {filename}")

#==================================================================#
# RUN EXPERIMENTS :
database,policies,policy_comparison_matrix = initialize_storage()
start_experiment_time = time.time()
for gametype in ["cooperative","zerosum","stackelberg"]:
    for horizon in range(1,planning_horizon+1):
        for sota_ in [False,True]:
            print(f"\n============= {gametype} GAME WITH HORIZON {horizon} , SOTA {sota_} ===========")
            game = Classes.PBVI(problem=problem,horizon=horizon,density=0.9,gametype=gametype,sota=sota_)
            policies[gametype][sota_] = SOLVE(game,database,horizon,gametype)
    game.build_comparison_matrix(policy_comparison_matrix,policies,gametype,iteration=num_iterations-1)
    export_policy_matrix(policy_comparison_matrix,gametype)
    print("======= END OF EXPERIMENT =========")
end_experiment_time = time.time()
print(f"finished in {end_experiment_time-start_experiment_time} seconds")
database = pd.DataFrame(database)
export_database(database)
plots(database)




