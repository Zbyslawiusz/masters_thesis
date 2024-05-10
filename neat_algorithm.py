from __future__ import print_function
import os
import neat
import visualize
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
        self.title = fitness_params["title"]
        self.throw_type = fitness_params["throw_type"]

        self.node_add_prob = neat_params["node_add_prob"]
        self.node_delete_prob = neat_params["node_delete_prob"]
        self.response_max_value = neat_params["response_max_value"]
        self.response_min_value = neat_params["response_min_value"]
        self.weight_mutate_power = neat_params["weight_mutate_power"]
        self.weight_mutate_rate = neat_params["weight_mutate_rate"]
        self.conn_add_prob = neat_params["conn_add_prob"]
        self.conn_delete_prob = neat_params["conn_delete_prob"]

        # self.distance_value = 0.95
        # self.time_value = 0.5
        # self.work_sum_value = 1e-7
        # self.penalty_col = 300
        # self.penalty_angle = 250

        self.searched_values = ["fitness_threshold"]
        self.csv_file_lock = Lock()  # Locks access to the csv file for other processes

        with open(f"./{self.directory}/config", "r") as file:
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
        self.multi_targets = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]

        # For measuring purposes
        self.t0 = time.time()
        self.t1 = False  # False means acceptable solution has not been reached
        self.t2 = 0  # Best solution time
        self.iteration = 0
        self.generation = 0  # Amount of generations that has passed
        self.best_generation = 0  # Generation of the best fitness
        self.acceptable_generation = 0  # Generation of the acceptable fitness
        self.minimum_desired_fitness_reached = False
        self.acceptable_net = None  # Stores the net that achieved acceptable fitness
        self.best_fitness = 0
        self.acceptable_fitness = False  # False means acceptable solution has not been reached
        self.is_set = False  # Whether the acceptable fitness has been reached or not
        # List of fitness values
        self.fitness_change = []

        self.acceptable_solution_distance = 0
        self.acceptable_solution_time_of_throw = 0
        self.acceptable_solution_total_work_sum = 0

        filename = f"{uuid.uuid4().hex}"
        timestring = time.strftime("%Y-%m-%d---%H-%M-%S")
        self.acceptable_filename = '{0}/{1}-{2}-acceptable'.format(self.directory, filename, timestring)

    def neat_start(self, neat_queue):
        self.queue = neat_queue
        # tkinter window START
        self.progress_window = tk.Tk()
        self.progress_window.title("NEAT training progress")
        self.progress_window.config(padx=25, pady=25, bg="white")

        self.progress_label = tk.Label(self.progress_window, text="NEAT training in progress",
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
        # Determine path to configuration file. This path manipulation is
        # here so that the script will run successfully regardless of the
        # current working directory.
        local_dir = os.path.dirname(__file__)
        config_dir = f"{self.directory}/config"
        config_path = os.path.join(local_dir, config_dir)
        # print("AAAAAAAAAAAAAAAAA")
        self.run(config_path)

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # for xi, xo in zip(xor_inputs, xor_outputs):
            #     output = net.activate(xi)
            #     genome.fitness -= (output[0] - xo[0]) ** 2
            # nuke_fitness = False

            # fitness = 1.0 / (error_sum + 0.0001)
            # return [registered_distance, hit_obstacle, elapsed_time, work_sum]
            # DOUBLE, BOOLEAN VALUE, DOUBLE, DOUBLE

            # Fitness function for throwing the ball at a target x coordinate
            if self.throw_type == "target" or self.throw_type == "gimmick":
                simulation = Simulation(
                    net=net,
                    ui_flag=False,
                    number_of_links=self.number_of_links,
                    target_xcor=self.target_xcor,
                    gripper=self.gripper_type,
                    throw_type=self.throw_type
                )

                genome.fitness = self.max_fitness - (self.distance_value * simulation.error_sum[0] +
                                                     self.time_value * simulation.error_sum[2] +
                                                     self.work_sum_value * simulation.error_sum[3])

                if simulation.error_sum[1]:
                    genome.fitness -= self.penalty_col  # Applying penalty for hitting the obstacle
            # Fitness function for throwing the ball as far away as possible
            elif self.throw_type == "far":
                simulation = Simulation(
                    net=net,
                    ui_flag=False,
                    number_of_links=self.number_of_links,
                    target_xcor=self.target_xcor,
                    gripper=self.gripper_type,
                    throw_type=self.throw_type
                )

                genome.fitness = (simulation.error_sum[0] -
                                  (self.time_value * simulation.error_sum[2] +
                                   self.work_sum_value * simulation.error_sum[3]))

            # print(f"\nSIMULATION ERRORS IN NEAT\n"
            #       f"Fitness: {genome.fitness}\n"
            #       f"Distance: {simulation.error_sum[0]}\n"
            #       f"Time of throw: {simulation.error_sum[2]}\n"
            #       f"Total work sum: {simulation.error_sum[3]}\n")

            elif self.throw_type == "multi-target":
                fitnesses = []
                for target in self.multi_targets:
                    simulation = Simulation(
                        net=net,
                        ui_flag=False,
                        number_of_links=self.number_of_links,
                        target_xcor=target,
                        gripper=self.gripper_type,
                        throw_type=self.throw_type
                    )

                    fitness = self.max_fitness - (self.distance_value * simulation.error_sum[0] +
                                                  self.time_value * simulation.error_sum[2] +
                                                  self.work_sum_value * simulation.error_sum[3])

                    # if simulation.error_sum[1]:
                    #     fitness -= self.penalty_col  # Applying penalty for hitting the obstacle

                    fitnesses.append(fitness)
                    print(f"Fitness of {target} target: {fitness}")

                # Calculating mean square of all simulation fitness values
                genome.fitness = 0
                for _ in fitnesses:
                    # genome.fitness += _**2
                    genome.fitness += _
                genome.fitness /= len(fitnesses)
                # genome.fitness **= 0.5
                print(f"Genome fitness: {genome.fitness}")

            # Monitoring best fitness
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness  # Acquiring best fitness value
                self.t2 = time.time()  # Acquiring best fitness time
                self.best_generation = self.generation  # Acquiring best fitness generation
                # print(f"\n-----------------------------------------------------------------------------------------\n"
                #       f"CURRENT BEST FITNESS\n"
                #       f"Current best fitness: {self.best_fitness}\n"
                #       f"Distance: {simulation.error_sum[0]}\n"
                #       f"Time of throw: {simulation.error_sum[2]}\n"
                #       f"Total work sum: {simulation.error_sum[3]}\n"
                #       f"-----------------------------------------------------------------------------------------\n")

            # Monitoring acceptable fitness
            elif self.throw_type != "far":
                if genome.fitness > 0.9 * self.max_fitness and not self.is_set:
                    self.is_set = True
                    self.acceptable_fitness = genome.fitness  # Acquiring acceptable fitness value
                    self.t1 = time.time()  # Acquiring acceptable fitness time
                    self.acceptable_generation = self.generation  # Acquiring acceptable fitness generation
                    # print(f"\n-----------------------------------------------------------------------------------------\n"
                    #       f"ACCEPTABLE SOLUTION\n"
                    #       f"Fitness: {genome.fitness}\n"
                    #       f"Distance: {simulation.error_sum[0]}\n"
                    #       f"Time of throw: {simulation.error_sum[2]}\n"
                    #       f"Total work sum: {simulation.error_sum[3]}\n"
                    #       f"-----------------------------------------------------------------------------------------\n")

                    with gzip.open(self.acceptable_filename, 'w', compresslevel=5) as f:
                        pickle.dump(net, f, protocol=pickle.HIGHEST_PROTOCOL)

                    self.acceptable_solution_distance = simulation.error_sum[0]
                    self.acceptable_solution_time_of_throw = simulation.error_sum[2]
                    self.acceptable_solution_total_work_sum = simulation.error_sum[3]

            if self.iteration % self.sol_per_pop == 0:
                self.generation += 1  # Keeping track of the number of generations
                self.fitness_change.append(self.best_fitness)

            # Displayed in training progress window
            self.num_of_generation_label.config(text=f"Current generation: {self.generation}")
            # Displayed in training progress window
            self.fitness_label.config(
                text=f"Highest achieved fitness so far: {round(self.best_fitness, 3)}"
            )
            self.progress_window.update()

            self.iteration += 1

    def run(self, config_file):
        # print("BBBBBBBBBBBBBB")
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

            best_solution_sim = Simulation(
                net=winner_net,
                ui_flag=False,
                number_of_links=self.number_of_links,
                target_xcor=self.target_xcor,
                gripper=self.gripper_type,
                throw_type=self.throw_type
            )
        fitness = self.max_fitness - (self.distance_value * best_solution_sim.error_sum[0] +
                                      self.time_value * best_solution_sim.error_sum[2] +
                                      self.work_sum_value * best_solution_sim.error_sum[3])

        # print(f"\n-----------------------------------------------------------------------------------------\n"
        #       f"BEST SOLUTION SIM\n"
        #       f"Fitness: {fitness}\n"
        #       f"Distance: {best_solution_sim.error_sum[0]}\n"
        #       f"Time of throw: {best_solution_sim.error_sum[2]}\n"
        #       f"Total work sum: {best_solution_sim.error_sum[3]}\n"
        #       f"-----------------------------------------------------------------------------------------\n")

        # Show output of the most fit genome against training data.
        # print('\nOutput:')
        # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        # for xi, xo in zip(xor_inputs, xor_outputs):
        #     output = winner_net.activate(xi)
        #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        if self.throw_type == "multi-target":
            node_names = {
                -5 - self.number_of_links * 2: "desired ball x cor",
                -4 - self.number_of_links * 2: "ball x cor",
                -3 - self.number_of_links * 2: "ball y cor",
                -2 - self.number_of_links * 2: "ball x velocity",
                -1 - self.number_of_links * 2: "ball y velocity",
            }

        else:
            node_names = {
                -4 - self.number_of_links * 2: "ball x cor",
                -3 - self.number_of_links * 2: "ball y cor",
                -2 - self.number_of_links * 2: "ball x velocity",
                -1 - self.number_of_links * 2: "ball y velocity",
            }
        i = - self.number_of_links * 2
        j = 1
        for _ in range(0, self.number_of_links):
            node_names[i] = f"current angle {j}"
            i += 1
            j += 1
        i = - self.number_of_links
        j = 1
        for _ in range(0, self.number_of_links):
            node_names[i] = f"previous angle {j}"
            i += 1
            j += 1

        j = 1
        for _ in range(0, self.number_of_links):
            node_names[_] = f"motor torque {j}"
            i += 1
            j += 1
        # for key in node_names:
        #     print(f"{key}: {node_names[key]}")

        visualize.draw_net(config, winner, True, node_names=node_names)
        # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

        # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
        # p = neat.Checkpointer.restore_checkpoint('./neat_checkpoints/neat-checkpoint-4')
        # p.run(eval_genomes, 10)

        if self.t1 != False:  # If acceptable fitness has been reached
            minimum_time = round(self.t1 - self.t0)
        else:
            minimum_time = False

        # If acceptable solution is also the best solution...
        if self.acceptable_fitness == self.best_fitness:
            # ...don't simulate it in pymunk because best simulation would only repeat it
            self.acceptable_fitness = False

        df = pd.DataFrame(
            {
                "title": [self.title],
                "throw_type": [self.throw_type],
                "Num of movable links": [self.number_of_links],
                "target_xcor": [self.target_xcor],
                "Elapsed time": [elapsed_time],
                "Best trained network": [filename],
                "Best solution time": [self.t2 - self.t0],
                "Fitness value of the best solution": [self.best_fitness],
                "Generation of the best solution": [self.best_generation],
                "Best solution distance": [best_solution_sim.error_sum[0]],
                "Best solution time of throw": [best_solution_sim.error_sum[2]],
                "Best solution total work sum": [best_solution_sim.error_sum[3]],
                "Acceptable solution time": [minimum_time],
                "Fitness value of the acceptable solution": [self.acceptable_fitness],
                "Generation of the acceptable solution": [self.generation],
                "Acceptable solution distance": [self.acceptable_solution_distance],
                "Acceptable solution time of throw": [self.acceptable_solution_time_of_throw],
                "Acceptable solution total work sum": [self.acceptable_solution_total_work_sum],
                "num_generations": [self.num_generations],  # Total number of generations
                "sol_per_pop": [self.sol_per_pop],  # Initial population
                "max_fitness": [self.max_fitness],  # Highest achievable fitness
                "distance_value": [self.distance_value],  # Weight of distance in fitness function
                "time_value": [self.time_value],  # Weight of time of throw in fitness function
                "work_sum_value": [self.work_sum_value],  # Weight of total work sum in fitness function
                "penalty_col": [self.penalty_col],  # Penalty for hitting the obstacle
                "penalty_angle": [self.penalty_angle],  # Penalty for wrong angle solutions
                "fitness_change": [self.fitness_change],  # List of all recorded genome fitness values
                "Num_of_training_instances": [self.neat_amount],  # How many NEATs were trained at once
                "gripper_type": [self.gripper_type],
                "net_path": [filename],  # Path to the net with the best solution
                "acceptable_net_path": [self.acceptable_filename],  # Path to the net with acceptable solution
                "activation_type": [self.activation_type],
                "num_hidden": [self.num_hidden],
                "node_add_prob": [self.node_add_prob],
                "node_delete_prob": [self.node_delete_prob],
                "response_max_value": [self.response_max_value],
                "response_min_value": [self.response_min_value],
                "weight_mutate_power": [self.weight_mutate_power],
                "weight_mutate_rate": [self.weight_mutate_rate],
                "conn_add_prob": [self.conn_add_prob],
                "conn_delete_prob": [self.conn_delete_prob],
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
                "throw_type": self.throw_type,
                "Num of movable links": self.number_of_links,
                "activation_type": self.activation_type,
                "num_hidden": self.num_hidden,
                "target_xcor": self.target_xcor,
                "num_generations": self.num_generations,
                "sol_per_pop": self.sol_per_pop,  # Initial population
                "max_fitness": self.max_fitness,  # Highest achievable fitness
                "distance_value": self.distance_value,  # Weight of distance in fitness function
                "time_value": self.time_value,  # Weight of time of throw in fitness function
                "work_sum_value": self.work_sum_value,  # Weight of total work sum in fitness function
                "penalty_col": self.penalty_col,  # Penalty for hitting the obstacle
                "penalty_angle": self.penalty_angle,  # Penalty for wrong angle solutions
                "Num_of_training_instances": self.neat_amount,  # How many NEATs were trained at once
                "gripper_type": self.gripper_type,
                "node_add_prob": self.node_add_prob,
                "node_delete_prob": self.node_delete_prob,
                "response_max_value": self.response_max_value,
                "response_min_value": self.response_min_value,
                "weight_mutate_power": self.weight_mutate_power,
                "weight_mutate_rate": self.weight_mutate_rate,
                "conn_add_prob": self.conn_add_prob,
                "conn_delete_prob": self.conn_delete_prob,
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
