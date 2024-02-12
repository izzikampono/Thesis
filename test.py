#imports
import sys
import numpy as np
import pandas as pd
import random
from decpomdp import DecPOMDP
from constant import Constants
import time


# input : file_name , game type  , planning horizon, num iterations,sota(1 or 0)
# sample : 
# python test.py problem=dectiger gametype=zerosum horizon=2 iter=1 sota=0
if len(sys.argv) < 2:
    print("err0r : not enough arguments given")
    sys.exit(1)
if len(sys.argv)> 5 :
    file_name = str(sys.argv[1]).split("=")[1]
    game_type = str(sys.argv[2]).split("=")[1]
    planning_horizon = int(sys.argv[3].split("=")[1])
    num_iterations = int(sys.argv[4].split("=")[1])
    sota_ = bool(int(sys.argv[5].split("=")[1]))
    # num_points = bool(int(sys.argv[6].split("=")[1]))
else : 
    print("not enough arguments")
    sys.exit()

#import problem
problem = DecPOMDP(file_name,horizon=planning_horizon)
Constants.initialize(problem)
import pbvi


# solve
game = pbvi.PBVI(problem,planning_horizon,0.0000001,game_type,limit=1000,sota=sota_)
values,times = game.solve(num_iterations)
print(f"\nvalues at initial belief:\n {values}\ntime :\n{times}\n")
