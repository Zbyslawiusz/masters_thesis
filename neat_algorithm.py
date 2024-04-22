from __future__ import print_function
import os
import neat
# import visualize
import pickle
import gzip
import uuid
import pandas as pd
import tkinter as tk
import time
from multiprocessing import Lock

from main_neat import Simulation

# 2-input XOR inputs and expected outputs.
# xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# xor_outputs = [(0.0,),     (1.0,),     (1.0,),     (0.0,)]


class NeatAlgorithm:
    def __init__(self, fitness_params, neat_params):
        self.distance_value = fitness_params["distance_value"]  # Weight of distance in fitness function
        self.time_value = fitness_params["time_value"]  # Weight of time of throw in fitness function
        self.work_sum_value = fitness_params["work_sum_value"]  # Weight of total work sum in fitness function
        self.penalty_col = fitness_params["penalty_col"]  # Penalty for hitting the obstacle
        self.penalty_angle = fitness_params["penalty_angle"]  # Penalty for wrong angle solutions
        self.directory = fitness_params["foldername"]  # Unique filename used for config file and trained network

        # self.distance_value = 0.95
        # self.time_value = 0.5
        # self.work_sum_value = 1e-7
        # self.penalty_col = 300
        # self.penalty_angle = 250

        self.searched_values = ["fitness_threshold"]
        self.csv_file_lock = Lock()  # Locks access to the csv file for other processes

        with open("config-feedforward", "r") as file:
            self.content = file.readlines()

        self.values = []

        for line in self.content:
            for value in self.searched_values:
                self.search_function(line, value)

        self.max_fitness = int(self.values[0])
        self.target_xcor = fitness_params["target_xcor"]
        self.number_of_links = fitness_params["Num of movable links"]
        self.gripper_type = fitness_params["gripper_type"]
        self.sol_per_pop = neat_params["sol_per_pop"]  # Starting population
        self.neat_amount = fitness_params["Num_of_training_instances"]
        self.activation_type = neat_params["activation_type"]
        self.num_hidden = neat_params["num_hidden"]

        self.num_generations = neat_params["num_generations"]
        self.training_finished = False

        self.t0 = time.time()

    def neat_start(self, neat_queue):
        self.queue = neat_queue
        # Determine path to configuration file. This path manipulation is
        # here so that the script will run successfully regardless of the
        # current working directory.
        local_dir = os.path.dirname(__file__)
        config_dir = f"{self.directory}/config"
        config_path = os.path.join(local_dir, config_dir)
        print("AAAAAAAAAAAAAAAAA")
        self.run(config_path)

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # for xi, xo in zip(xor_inputs, xor_outputs):
            #     output = net.activate(xi)
            #     genome.fitness -= (output[0] - xo[0]) ** 2
            # nuke_fitness = False

            simulation = Simulation(
                net=net,
                ui_flag=False,
                number_of_links=self.number_of_links,
                target_xcor=self.target_xcor,
                gripper=self.gripper_type
            )
            # fitness = 1.0 / (error_sum + 0.0001)

            # return [registered_distance, hit_obstacle, elapsed_time, work_sum]
            # DOUBLE, BOOLEAN VALUE, DOUBLE, DOUBLE

            genome.fitness = self.max_fitness - (self.distance_value * simulation.error_sum[0] +  # 1600 for 1 link
                                                 self.time_value * simulation.error_sum[2] +
                                                 self.work_sum_value * simulation.error_sum[3])

            if simulation.error_sum[1]:
                genome.fitness -= self.penalty_col  # Applying penalty for hitting the obstacle

    def run(self, config_file):
        print("BBBBBBBBBBBBBB")
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        # Run for up to 300 generations.
        winner = p.run(self.eval_genomes, self.num_generations)

        # Display the winning genome.
        elapsed_time = time.time() - self.t0

        print('\nBest genome:\n{!s}'.format(winner))

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        filename = f"{uuid.uuid4().hex}"
        timestring = time.strftime("%Y-%m-%d---%H-%M-%S")
        filename = '{0}/{1}-{2}'.format(self.directory, filename, timestring)

        with gzip.open(filename, 'w', compresslevel=5) as f:
            pickle.dump(winner_net, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Show output of the most fit genome against training data.
        # print('\nOutput:')
        # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        # for xi, xo in zip(xor_inputs, xor_outputs):
        #     output = winner_net.activate(xi)
        #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

        # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
        # visualize.draw_net(config, winner, True, node_names=node_names)
        # visualize.plot_stats(stats, ylog=False, view=True)
        # visualize.plot_species(stats, view=True)

        # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
        # p = neat.Checkpointer.restore_checkpoint('./neat_checkpoints/neat-checkpoint-4')
        # p.run(eval_genomes, 10)

            best_solution_sim = Simulation(
                net=winner_net,
                ui_flag=False,
                number_of_links=self.number_of_links,
                target_xcor=self.target_xcor,
                gripper=self.gripper_type
            )

        df = pd.DataFrame(
            {
                "Num of movable links": [self.number_of_links],
                "target_xcor": [self.target_xcor],
                "Elapsed time": [elapsed_time],
                "Best trained network": [filename],
                # "Best solution time": best_solution_time,
                "Fitness value of the best solution": [0],
                # "Generation of the best solution": self.ga_instance.best_solution_generation,
                "Best solution distance": [best_solution_sim.error_sum[0]],
                "Best solution time of throw": [best_solution_sim.error_sum[2]],
                "Best solution total work sum": [best_solution_sim.error_sum[3]],
                # "sol_per_pop": self.sol_per_pop,
                "max_fitness": [self.max_fitness],
                "distance_value": [self.distance_value],  # Weight of distance in fitness function
                "time_value": [self.time_value],  # Weight of time of throw in fitness function
                "work_sum_value": [self.work_sum_value],  # Weight of total work sum in fitness function
                "penalty_col": [self.penalty_col],  # Penalty for hitting the obstacle
                "penalty_angle": [self.penalty_angle],  # Penalty for wrong angle solutions
                # "fitness_change": [self.fitness_change],
                "Num_of_training_instances": [self.neat_amount],
                # "Num_of_interpolation_angles": self.interpolation,
                "gripper_type": [self.gripper_type],
                "net_path": [filename],
            }
        )

        with self.csv_file_lock:
            try:
                pd.read_csv("solution_neat.csv")
            except FileNotFoundError:
                print("File not found, creating a new one.")
                df.to_csv("solution_neat.csv", mode="w", index=False)
            else:
                df.to_csv("solution_neat.csv", mode="a", header=False, index=False)
                print("Adding data to csv.")

        current_settings = pd.DataFrame(
            {
                "Num of movable links": self.number_of_links,
                "activation_type": self.activation_type,
                "num_hidden": self.num_hidden,
                "target_xcor": self.target_xcor,
                "num_generations": self.num_generations,
                "sol_per_pop": self.sol_per_pop,
                "max_fitness": self.max_fitness,
                "distance_value": self.distance_value,  # Weight of distance in fitness function
                "time_value": self.time_value,  # Weight of time of throw in fitness function
                "work_sum_value": self.work_sum_value,  # Weight of total work sum in fitness function
                "penalty_col": self.penalty_col,  # Penalty for hitting the obstacle
                "penalty_angle": self.penalty_angle,  # Penalty for wrong angle solutions
                "Num_of_training_instances": self.neat_amount,
                "gripper_type": self.gripper_type,
            }, index=[0]  # This fixes the "ValueError: If using all scalar values, you must pass an index" error
        )
        current_settings.to_json("settings_neat.json")

        self.training_finished = True  # signaling to main menu that training is completed
        self.queue.put("FINISHED")

        # Once training is completed
        # self.progress_label.config(text="GA training finished", bg="green")
        # # close_win_but = tk.Button(self.progress_window, text="Return", font=("Consolas", 30, "bold"),
        # #                           command=lambda: self.close_window(window=self.progress_window))
        # # close_win_but.grid(row=3, column=0)
        # self.queue.put("FINISHED")
        # self.progress_window.destroy()
        #
        # self.progress_window.mainloop()

    def search_function(self, line, value):
        if line.replace(" ", "")[:len(value)] == value:
            value = line.replace(" ", "")[len(value) + 1:-1]
            self.values.append(value)
