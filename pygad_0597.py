import pandas as pd
from math import pi
import pygad
import time
import tkinter as tk
from multiprocessing import Lock

from main_051 import Simulation
"""
Given the following function:
    y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
"""


# function_inputs = [4, -2, 3.5, 5, -11, -4.7]  # Function inputs.
# desired_output = 44  # Function output.

class GeneticAlgorithm:
    def __init__(self, fitness_params, ga_params):

        self.csv_file_lock = Lock()  # Locks access to the csv file for other processes
        # Number of movable links
        self.number_of_links = fitness_params["Num of movable links"]
        # Fitness parameters
        self.target_xcor = fitness_params["target_xcor"]
        self.max_fitness = fitness_params["max_fitness"]
        self.distance_value = fitness_params["distance_value"]  # Weight of distance in fitness function
        self.time_value = fitness_params["time_value"]  # Weight of time of throw in fitness function
        self.work_sum_value = fitness_params["work_sum_value"]  # Weight of total work sum in fitness function
        self.penalty_col = fitness_params["penalty_col"]  # Penalty for hitting the obstacle
        self.penalty_angle = fitness_params["penalty_angle"]  # Penalty for wrong angle solutions
        self.ga_amount = fitness_params["Num_of_training_instances"]
        # Genetic algorithm parameters
        self.num_generations = ga_params["num_generations"]
        self.num_parents_mating = ga_params["num_parents_mating"]
        self.parent_selection_type = ga_params["parent_selection_type"]
        self.crossover_type = ga_params["crossover_type"]
        self.init_range_low = ga_params["init_range_low"]
        self.init_range_high = ga_params["init_range_high"]
        self.random_mutation_min_val = ga_params["random_mutation_min_val"]
        self.random_mutation_max_val = ga_params["random_mutation_max_val"]
        self.mutation_probability = ga_params["mutation_probability"]
        self.sol_per_pop = ga_params["sol_per_pop"]  # Number of solutions in the population.

        self.displayed_generation = 0  # Displayed in main menu's training progress window
        self.displayed_fitness = 0  # Displayed in main menu's training progress window
        self.training_finished = False

        self.t0 = time.time()
        self.t1 = False  # False means acceptable solution has not been reached
        self.t2 = 0
        self.generation = 0
        self.minimum_desired_fitness_reached = False
        self.minimum_solution = [0 for _ in range(self.number_of_links * 2)]  # 3 element empty list
        self.best_fitness = 0
        self.acceptable_fitness = False  # False means acceptable solution has not been reached
        self.is_set = False

        # List of fitness values
        self.fitness_change = []

    def ga_function(self, ga_queue):
        self.queue = ga_queue
        # tkinter window START
        self.progress_window = tk.Tk()
        self.progress_window.title("GA training progress")
        self.progress_window.config(padx=25, pady=25, bg="white")

        self.progress_label = tk.Label(self.progress_window, text="GA training in progress",
                                       font=("Consolas", 30, "bold"),
                                       bg="red")
        self.progress_label.grid(row=0, column=0)

        self.num_of_generation_label = tk.Label(self.progress_window, text="Current generation: ",
                                                font=("Consolas", 15, "bold"))
        self.num_of_generation_label.grid(row=1, column=0)

        self.fitness_label = tk.Label(self.progress_window, text="Highest achieved fitness so far: ",
                                      font=("Consolas", 15, "bold"))
        self.fitness_label.grid(row=2, column=0)
        # tkinter window STOP

        # GA parameters
        self.num_genes = self.number_of_links * 2
        self.last_fitness = 0
        self.ga_instance = pygad.GA(num_generations=self.num_generations,
                                    num_parents_mating=self.num_parents_mating,
                                    parent_selection_type="sss",
                                    crossover_type="single_point",
                                    sol_per_pop=self.sol_per_pop,
                                    num_genes=self.num_genes,
                                    fitness_func=self.fitness_func,
                                    init_range_low=self.init_range_low,
                                    init_range_high=self.init_range_high,
                                    random_mutation_min_val=self.random_mutation_min_val,
                                    random_mutation_max_val=self.random_mutation_max_val,
                                    mutation_probability=self.mutation_probability,
                                    # parallel_processing=("thread", 20),
                                    on_generation=self.on_generation)
        # Running the GA to optimize the parameters of the function.
        self.ga_instance.run()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution(
            self.ga_instance.last_generation_fitness)
        best_solution = [float(_) for _ in solution]
        print(f"Parameters of the best solution : {best_solution}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")

        if self.ga_instance.best_solution_generation != -1:
            print(f"\nCalculated for {self.number_of_links} movable links.")
            elapsed_time = time.time() - self.t0
            print(f"Elapsed time: {elapsed_time} s.")
            print(f"Best fitness value reached after {self.ga_instance.best_solution_generation} generations.")
            best_solution_time = time.time() - self.t2
            print(f"Best solution reached after: {best_solution_time} seconds.")
            if self.t1 != False:
                minimum_time = round(self.t1 - self.t0)
                print(f"Minimum desired fitness reached after: {minimum_time} seconds.")
                print(f"Fitness value of the acceptable solution = {self.acceptable_fitness}.")
            else:
                minimum_time = False
                print(f"Minimum desired fitness has not been reached.")
            print(f"Minimum desired fitness value reached after: {self.generation} generations.")
            print(f"Acceptable solution: {self.minimum_solution}.")
            # result = Simulation(genetic_solution=solution, ui_flag=True, number_of_links=number_of_links)

            acceptable_solution_distance = -1
            acceptable_solution_time_of_throw = -1
            acceptable_solution_total_work_sum = -1

            # Creating simulations to save achieved distance, time and work sum to csv file
            if self.acceptable_fitness != False:
                minimum_solution_sim = Simulation(genetic_solution=self.minimum_solution,
                                                  ui_flag=False,
                                                  number_of_links=self.number_of_links,
                                                  target_xcor=self.target_xcor)
                acceptable_solution_distance = minimum_solution_sim.error_sum[0]
                acceptable_solution_time_of_throw = minimum_solution_sim.error_sum[2]
                acceptable_solution_total_work_sum = minimum_solution_sim.error_sum[3]

            best_solution_sim = Simulation(genetic_solution=best_solution,
                                           ui_flag=False,
                                           number_of_links=self.number_of_links,
                                           target_xcor=self.target_xcor)

            df = pd.DataFrame(
                {
                    "Num of movable links": self.number_of_links,
                    "target_xcor": self.target_xcor,
                    "Elapsed time": elapsed_time,
                    "Best solution": [best_solution],
                    "Best solution time": best_solution_time,
                    "Fitness value of the best solution": solution_fitness,
                    "Generation of the best solution": self.ga_instance.best_solution_generation,
                    "Best solution distance": best_solution_sim.error_sum[0],
                    "Best solution time of throw": best_solution_sim.error_sum[2],
                    "Best solution total work sum": best_solution_sim.error_sum[3],
                    "Acceptable solution": [self.minimum_solution],
                    "Acceptable solution time": minimum_time,
                    "Fitness value of the acceptable solution": self.acceptable_fitness,
                    "Generation of the acceptable solution": self.generation,
                    "Acceptable solution distance": acceptable_solution_distance,
                    "Acceptable solution time of throw": acceptable_solution_time_of_throw,
                    "Acceptable solution total work sum": acceptable_solution_total_work_sum,
                    "num_generations": self.num_generations,
                    "num_parents_mating": self.num_parents_mating,
                    "parent_selection_type": self.parent_selection_type,
                    "crossover_type": self.crossover_type,
                    "init_range_low": self.init_range_low,
                    "init_range_high": self.init_range_high,
                    "random_mutation_min_val": self.random_mutation_min_val,
                    "random_mutation_max_val": self.random_mutation_max_val,
                    "mutation_probability": self.mutation_probability,
                    "sol_per_pop": self.sol_per_pop,
                    "max_fitness": self.max_fitness,
                    "distance_value": self.distance_value,  # Weight of distance in fitness function
                    "time_value": self.time_value,  # Weight of time of throw in fitness function
                    "work_sum_value": self.work_sum_value,  # Weight of total work sum in fitness function
                    "penalty_col": self.penalty_col,  # Penalty for hitting the obstacle
                    "penalty_angle": self.penalty_angle,  # Penalty for wrong angle solutions
                    "fitness_change": [self.fitness_change],
                    "Num_of_training_instances": self.ga_amount,
                }
            )

            with self.csv_file_lock:
                try:
                    pd.read_csv("solution.csv")
                except FileNotFoundError:
                    print("File not found, creating a new one.")
                    df.to_csv("solution.csv", mode="w", index=False)
                else:
                    df.to_csv("solution.csv", mode="a", header=False, index=False)
                    print("Adding data to csv.")

            current_settings = pd.DataFrame(
                {
                    "Num of movable links": self.number_of_links,
                    "target_xcor": self.target_xcor,
                    "num_generations": self.num_generations,
                    "num_parents_mating": self.num_parents_mating,
                    "parent_selection_type": self.parent_selection_type,
                    "crossover_type": self.crossover_type,
                    "init_range_low": self.init_range_low,
                    "init_range_high": self.init_range_high,
                    "random_mutation_min_val": self.random_mutation_min_val,
                    "random_mutation_max_val": self.random_mutation_max_val,
                    "mutation_probability": self.mutation_probability,
                    "sol_per_pop": self.sol_per_pop,
                    "max_fitness": self.max_fitness,
                    "distance_value": self.distance_value,  # Weight of distance in fitness function
                    "time_value": self.time_value,  # Weight of time of throw in fitness function
                    "work_sum_value": self.work_sum_value,  # Weight of total work sum in fitness function
                    "penalty_col": self.penalty_col,  # Penalty for hitting the obstacle
                    "penalty_angle": self.penalty_angle,  # Penalty for wrong angle solutions
                    "Num_of_training_instances": self.ga_amount,
                }, index=[0]  # This fixes the "ValueError: If using all scalar values, you must pass an index" error
            )
            current_settings.to_json("settings.json")

            self.training_finished = True  # signaling to main menu that training is completed

            # Once training is completed
            self.progress_label.config(text="GA training finished", bg="green")
            # close_win_but = tk.Button(self.progress_window, text="Return", font=("Consolas", 30, "bold"),
            #                           command=lambda: self.close_window(window=self.progress_window))
            # close_win_but.grid(row=3, column=0)
            self.queue.put("FINISHED")
            self.progress_window.destroy()

            self.progress_window.mainloop()

    def close_window(self, window):
        """This method closes passed popup window"""
        window.destroy()

    def fitness_func(self, ga_instance, solution, solution_idx):

        simulation = Simulation(genetic_solution=solution,
                                ui_flag=False,
                                number_of_links=self.number_of_links,
                                target_xcor=self.target_xcor)
        # fitness = 1.0 / (error_sum + 0.0001)

        # return [registered_distance, hit_obstacle, elapsed_time, work_sum]
        # DOUBLE, BOOLEAN VALUE, DOUBLE, DOUBLE

        fitness = self.max_fitness - (self.distance_value * simulation.error_sum[0] +
                                      self.time_value * simulation.error_sum[2] +
                                      self.work_sum_value * simulation.error_sum[3])

        if simulation.error_sum[1]:
            fitness -= self.penalty_col  # Applying penalty for hitting the obstacle

        for i in range(0, self.number_of_links*2, 2):
            if solution[i] > pi/2 or solution[i] < -pi/2:
                fitness -= self.penalty_angle  # Applying penalty for incorrect desired link angles

        if simulation.error_sum[0] <= 10 and not self.is_set:  # Acquiring the acceptable solution
            self.is_set = True
            self.minimum_desired_fitness_reached = True
            self.t1 = time.time()
            self.minimum_solution = [_ for _ in solution]
            self.acceptable_fitness = fitness

        if fitness >= self.best_fitness:  # Recording the time it took to reach the best solution
            self.best_fitness = fitness
            self.t2 = time.time()

        return fitness

    def on_generation(self, ga_instance):

        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

        if ga_instance.generations_completed >= 0:

            if self.minimum_desired_fitness_reached:
                self.minimum_desired_fitness_reached = False
                self.generation = ga_instance.generations_completed

            print(f"Last fitness = {self.last_fitness}")
            print(f"Generation = {ga_instance.generations_completed}")
            print(f"Parameters of the best solution : {solution}")
            # print(f"Best solution = {ga_instance.best_solution}")
            print(f"Fitness = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
            self.fitness_change.append(ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1])
            print(f"Change = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] -
                              self.last_fitness}")
            # Displayed in training progress window
            self.num_of_generation_label.config(text=f"Current generation: {ga_instance.generations_completed}")
            # Displayed in training progress window
            self.fitness_label.config(text=f"Highest achieved fitness so far: "
                                           f"{round((ga_instance.best_solution(
                                               pop_fitness=ga_instance.last_generation_fitness)[1]), 3)}")
            self.progress_window.update()
