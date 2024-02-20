import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gc
from pbvi import PBVI
from tabularPbvi import tabularPBVI
from constant import Constants
gc.enable()
from matplotlib import rc
plt.rcParams["font.family"] = "Arial"


class Experiment():

    def __init__(self,planning_horizon,problem,algorithm="tabular"):
        self.planning_horizon = planning_horizon
        self.problem = problem
        self.database = None
        self.num_iterations = None
        self.algorithm = algorithm #tabular or max_plane
        self.game = None
        self.policies = {"cooperative":{},"stackelberg":{},"zerosum":{}}
        

    def initialize_game(self,density,type,limit,sota):
        if self.algorithm=="max_plane":
            self.game = PBVI(problem=self.problem,horizon=self.planning_horizon,density=density,gametype=type,limit=limit,sota=sota)

        if self.algorithm=="tabular":
            self.game = tabularPBVI(problem=self.problem,horizon=self.planning_horizon,density=density,gametype=type,limit=limit,sota=sota)



    def initialize_database(self):
        database = {"gametype":[],
                    "SOTA" : [],
                    "horizon": [],
                        "iterations" : [],
                        "time" : [],
                        "number_of_beliefs" : [],
                        "values":[],
                        "tabular value":[],
                        "density" : []
                        # "gap":[]
                        }
        self.database=database

    def add_to_database(self,SOTA,horizon,game_type,num_iterations,average_time,num_beliefs,values,tabular_value,density):
        sota = {True:"State of the Art" , False:"Stackelberg"}
        self.database["gametype"].append(game_type)
        self.database["horizon"].append(horizon)
        self.database["SOTA"].append(sota[SOTA])
        self.database["iterations"].append(num_iterations)
        self.database["time"].append(average_time)
        self.database["number_of_beliefs"].append(num_beliefs)
        self.database["values"].append(values)
        self.database["tabular value"].append(tabular_value)
        self.database["density"].append(density)
        return

    def run_experiments(self,iterations,density=0.001):
        stackelberg_policy = {True : {}, False :{}}
        self.initialize_database()


        for type in ["cooperative","zerosum","stackelberg"]:
            for sota in (True,False):
                for horizon in range(1,self.planning_horizon+1):
                    game = PBVI(problem=self.problem,horizon=horizon,density=density,gametype=type,limit=100,sota=sota)
                    values, times, tabular_value = game.solve(iterations)
                    #at the last horizon of the stackelberg game
                    if type=="stackelberg" and horizon==self.planning_horizon:
                        if sota==False: WL_SF = values[-1] #take value at last iteration
                        if sota==True: SL_WF = values[-1]
                        print("\nEXTRACTING STACKELBERG POLICIES ... ")
                    self.extract_policies(type,sota)
                    self.add_to_database(sota,horizon,type,
                                         iterations,times,game.belief_space.belief_size(),
                                         values,tabular_value,density)
        print("calculating stackelberg comparsion matrix...")
        #get policy value from strock stackelberg leader and blind agent
        WL_WF = self.game.DP(belief_id=0,timestep=0,leader_tree=self.policies["stackelberg"][False][0],follower_tree=self.policies["stackelberg"][True][1])
        #get policy val ue from weak stackelberg leader and strong stackelberg follower
        SL_SF = self.game.DP(belief_id=0,timestep=0,leader_tree=self.policies["stackelberg"][True][0],follower_tree=self.policies["stackelberg"][False][1])


        # make stackelberg comparison matrix and save 
        matrix = {"Strong Leader" : {"Strong Follower" : SL_SF , "Blind Follower" : SL_WF}, "Weak Leader" : {"Strong Follower" : WL_SF, "Blind Follower" : WL_WF }}
        matrix = pd.DataFrame(matrix)
        matrix.to_csv(f"comparison_matrix/{self.problem.name}({horizon}).csv", index=False)

        # export database
        self.export_database(f"raw_results/{self.problem.name} ({self.planning_horizon}).csv")

        return self.database, matrix, self.policies
    
    def export_database(self,file_name):
        if type(self.database)!=pd.DataFrame : self.database = pd.DataFrame(self.database)
        self.database.to_csv(f"raw_results/{self.problem.name} ({self.planning_horizon}).csv", index=False)


    def run_experiments_decreasing_density(self,iterations,limit=1000,initial_density=0.0001):
        stackelberg_policy = {True : {}, False :{}}
        self.initialize_database()
        self.num_iterations = iterations

        for type in ["cooperative","zerosum","stackelberg"]:
            for sota in (False,True):
                for horizon in range(1,self.planning_horizon+1):
                    self.initialize_game(initial_density,type,limit,sota)
                    values, times ,densities, belief_sizes , tabular_values = self.game.solve_sampled_densities(iterations,initial_density)
                    #extract value of Strong
                    if type=="stackelberg" and horizon==self.planning_horizon:
                        if sota==False: WL_SF = values[-1] #take value at last iteration
                        if sota==True: SL_WF = values[-1]
                        print("\nEXTRACTING STACKELBERG POLICIES ... ")
                        self.extract_policies(type,sota)
                    self.add_to_database(sota,horizon,type,iterations,times,belief_sizes,values[-1],tabular_values[-1],densities)
        print("calculating stackelberg comparsion matrix...")

        #get policy value from strong stackelberg leader and blind follower
        WL_WF = self.game.DP(belief_id=0,timestep=0,leader_tree=self.policies["stackelberg"][False][0],follower_tree=self.policies["stackelberg"][True][1])
        #get policy value from weak stackelberg leader and strong stackelberg follower
        SL_SF = self.game.DP(belief_id=0,timestep=0,leader_tree=self.policies["stackelberg"][True][0],follower_tree=self.policies["stackelberg"][False][1])


        # make stackelberg comparison matrix 
        matrix = {"Strong Leader" : {"Strong Follower" : SL_SF , "Blind Follower" : SL_WF}, "Weak Leader" : {"Strong Follower" : WL_SF, "Blind Follower" : WL_WF }}
        matrix = pd.DataFrame(matrix)
        matrix.to_csv(f"comparison_matrix/{self.problem.name}({horizon}).csv", index=False)        
        self.export_database(f"raw_results/{self.problem.name}({self.planning_horizon})")
        return self.database, matrix ,self.policies

    def run_single_experiment(self,density,gametype,limit,sota,iterations):
        self.initialize_game(density,gametype,limit,sota)
        values,times,tabular_values = self.game.solve(iterations)
        self.extract_policies(gametype,sota)
        return values,times,tabular_values,self.policies
    
    def extract_policies(self,gametype,sota):
        self.policies[gametype][sota]  = [self.game.extract_leader_policy(0,timestep=0),self.game.extract_follower_policy(0,timestep=0)]
        return



    
    def horizon_value_plot(self):
        bar_width = 0.35
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for id,gametype in enumerate(["cooperative","stackelberg","zerosum"]):
            colors = ['red', 'tan']
            horizons = np.arange(self.planning_horizon,step=1)
            sota_values = []
            non_sota_values = []
            x_labels = []
            for horizon in range(1,self.planning_horizon+1):
                data = self.database[self.database["gametype"]==gametype][self.database["horizon"]==horizon]
                sota_values.append(np.average([values[0] for values in np.array(data["values"][data["SOTA"]=="State of the Art"])[0]]))
                non_sota_values.append(np.average([values[0] for values in np.array(data["values"][data["SOTA"]=="Stackelberg"])[0]]))
                x_labels.append(horizon)
            # plotting
            axs[id].bar(horizons, sota_values, bar_width, label='Stackelberg')
            axs[id].bar(horizons + bar_width, non_sota_values, bar_width, label='State of the art')

            # labels
            axs[id].set_xlabel("Horizon")
            axs[id].set_ylabel('Leader value')
            axs[id].set_title(f"{gametype} game")
            axs[id].set_xticks(horizons+ bar_width / 2,horizons)
            axs[id].set_xticklabels(x_labels)

        fig.suptitle(f"Results for {self.problem.name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"horizon_plot/{self.problem.name} ({self.planning_horizon}).png")
        plt.show(block=False)
        plt.pause(8)
        plt.close('all')

    def plots(self):
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for id,gametype in enumerate(["cooperative","stackelberg","zerosum"]):
            data = self.database[self.database["gametype"]==gametype][self.database["horizon"]==self.planning_horizon]
            x = [i+1 for i in range(0,self.num_iterations)]
            for sota in ["Stackelberg","State of the Art"]:
                y = [values[0] for values in np.array(data["values"][data["SOTA"]==sota])[0]]
                axs[id].plot(x,y,label = sota)
                
                axs[id].set_xlabel("Iterations")
                axs[id].set_ylabel("leader value")
                axs[id].set_title(f"{gametype} game") 
            axs[id].legend()
        fig.suptitle(f"Results for {self.problem.name} with horizon = {self.planning_horizon}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"plots/{self.problem.name} ({self.planning_horizon}).png")

        plt.show(block=False)
        plt.close('all')


    def generate_summary_table(self):
        algorithms = ['State of the Art','PBVI']
        columns = pd.MultiIndex.from_product([algorithms, ['time', 'value', 'iteration']])
       
        # Create an empty DataFrame with the specified columns
        tables = dict.fromkeys(["cooperative","zerosum","stackelberg"])
        for gametype in ["cooperative","zerosum","stackelberg"]:
            game_data = []
            df = pd.DataFrame(columns=columns)
            for horizon in range(self.planning_horizon):
                new_row_data = []
                for SOTA in ["State of the Art","Stackelberg"]:
                    current_data = self.database[self.database["SOTA"]==SOTA][self.database["horizon"]==horizon+1][self.database["gametype"]==gametype]
                    time = current_data["time"].values[0][self.num_iterations-1]
                    value = current_data["values"].values[0][self.num_iterations-1]
                    iteration = current_data["iterations"].values[0]
                    new_row_data = new_row_data + [time,value,iteration]
                game_data.append(new_row_data)
            new_row = pd.DataFrame(game_data, columns=columns)
            df = df.merge(new_row, how='outer')
            df.index = [f"{self.problem.name}({horizon})" for horizon in range(self.planning_horizon)]
            df.to_csv(f"processed_results/{gametype}_{self.problem.name}.csv",index=True)
            tables[gametype]=df
        return tables
    
        
                
