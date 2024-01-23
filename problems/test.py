#imports
import sys
import numpy as np
import pandas as pd
import random
import Classes
from decpomdp import DecPOMDP
from utilities import *
import utilities
import time


# input : file_name , game type  , planning horizon, num iterations,sota(1 or 0)
# sample : 
# python test.py dectiger cooperative 3 3 1
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
PROBLEM = DecPOMDP(file_name, 1,horizon=planning_horizon)
Classes.set_problem(PROBLEM,planning_horizon)
PROBLEM.reset()

print(f"{game_type} initiated with SOTA set to = {sota_}")

# solve
start_time = time.time()
game = Classes.PBVI(PROBLEM,planning_horizon,0.1,game_type,sota=sota_)
policy = game.solve(num_iterations,1)
end_time = time.time()
solve_time = end_time - start_time




# results
print(f"\n{game_type} {file_name} problem with {num_iterations} iterations solved in ", solve_time, "seconds\n")
print(f"value at initial belief (V0,V1) :\n{game.value_function.get_values_initial_belief()}")

print("print policy tree?")
if input("answer (y/n) :") =="y":
    print("\nLEADER POLICY\n")
    policy[0].print_trees()
    print("\nFOLLOWER POLICY\n")
    policy[1].print_trees()
    


# policy[0].print_trees()
# policy[1].print_trees()


