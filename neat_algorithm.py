from __future__ import print_function
import os
import neat
import visualize
import pickle
import uuid
import pandas as pd
import tkinter as tk
import time
from multiprocessing import Lock
import matplotlib.pyplot as plt

from main_neat import Simulation


class NeatAlgorithm:
    def __init__(self, fitness_params, neat_params):
        self.distance_value = fitness_params["distance_value"]  # Weight of distance in fitness function
        self.time_value = fitness_params["time_value"]  # Weight of time of throw in fitness function
        self.work_sum_value = fitness_params["work_sum_value"]  # Weight of total work sum in fitness function
        self.penalty_col = fitness_params["penalty_col"]  # Penalty for hitting the obstacle
        # self.penalty_angle = fitness_params["penalty_angle"]  # Penalty for wrong angle solutions
        self.penalty_angle = 0  # Penalty for wrong angle solutions
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

        self.max_fitness = int(float(self.values[0]))
        self.target_xcor = fitness_params["target_xcor"]
        self.number_of_links = fitness_params["Num of movable links"]
        self.gripper_type = fitness_params["gripper_type"]
        self.sol_per_pop = neat_params["sol_per_pop"]  # Starting population
        self.neat_amount = fitness_params["Num_of_training_instances"]
        self.activation_type = neat_params["activation_type"]
        self.num_hidden = neat_params["num_hidden"]  # Specifies amount of hidden NODES not LAYERS

        self.num_generations = neat_params["num_generations"]
        self.training_finished = False
        self.multi_targets = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]
        # self.multi_targets = [_ for _ in range(1500, 2600, 100)]

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
        self.best_fitness = -100_000_000_000
        self.acceptable_fitness = False  # False means acceptable solution has not been reached
        self.is_set = False  # Whether the acceptable fitness has been reached or not
        # List of fitness values
        self.fitness_change = []
        self.elapsed_time = 0  # Total training time

        self.acceptable_solution_distance = 0
        self.acceptable_solution_time_of_throw = 0
        self.acceptable_solution_total_work_sum = 0

        # Filenames for saving results
        self.filename = f"{uuid.uuid4().hex}"
        timestring = time.strftime("%Y-%m-%d---%H-%M-%S")
        self.best_filename = "{0}/{1}-{2}".format(self.directory, self.filename, timestring)
        self.acceptable_filename = "{0}/{1}-{2}-acceptable".format(self.directory, self.filename, timestring)
        self.stats_name = "{0}-stats".format(self.best_filename)

        self.finish_and_save = False  # Flag that signals premature training ending
        self.stop = False

        self.fig, self.ax = plt.subplots()
        # plt.ion()  # Interactive mode on

    def neat_start(self, neat_queue):
        self.queue = neat_queue
        # tkinter window START -----------------------------------------------------------------------------------------
        self.progress_window = tk.Tk()
        self.progress_window.title("NEAT training progress")
        self.progress_window.config(padx=25, pady=25, bg="white")

        self.progress_label = tk.Label(self.progress_window, text="NEAT training in progress",
                                       font=("Consolas", 30, "bold"),
                                       bg="red")
        # self.progress_label.grid(row=0, column=0)
        self.progress_label.pack()

        self.num_of_generation_label = tk.Label(self.progress_window, text="Current generation: ",
                                                font=("Consolas", 15, "bold"))
        # self.num_of_generation_label.grid(row=1, column=0)
        self.num_of_generation_label.pack()

        self.fitness_label = tk.Label(self.progress_window, text="Highest achieved fitness so far: ",
                                      font=("Consolas", 15, "bold"))
        # self.fitness_label.grid(row=2, column=0)
        self.fitness_label.pack()

        self.finish_button = (tk.Button(text="FINISH AND SAVE NETWORK",
                                        font=("Consolas", 15, "bold"),
                                        command=self.finish_training))
        # self.finish_button.grid(row=3, column=0)
        self.finish_button.pack()

        # tkinter window STOP ------------------------------------------------------------------------------------------
        # Determine path to configuration file. This path manipulation is
        # here so that the script will run successfully regardless of the
        # current working directory.
        local_dir = os.path.dirname(__file__)
        config_dir = f"{self.directory}/config"
        config_path = os.path.join(local_dir, config_dir)
        # print("AAAAAAAAAAAAAAAAA")
        self.run(config_path)

    def finish_training(self):
        self.finish_and_save = True

    def draw_plot(self, x, y):
        plt.plot([0], [0])
        plt.clf()
        plt.plot(x, y, marker='o', linestyle='-', color='b')
        # plt.xlabel("Generation")
        # plt.ylabel("Fitness value")
        plt.xlabel("Pokolenie")
        plt.ylabel("Wartosc funkcji dopasowania")
        # plt.title("Fitness vaule plot")
        plt.title("Przebieg wartosci funkcji dopasowania")
        plt.axhline(0, color='gray', linewidth=0.5)  # Dodanie linii zerowej na osi Y
        plt.axvline(0, color='gray', linewidth=0.5)  # Dodanie linii zerowej na osi X
        plt.grid(True)
        plt.show(block=False)  # Do not stop the code until someone closes the plot window

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            # fitness = 1.0 / (error_sum + 0.0001)
            # return [registered_distance, hit_obstacle, elapsed_time, work_sum]
            # DOUBLE, BOOLEAN VALUE, DOUBLE, DOUBLE

            # Fitness function for throwing the ball at a target x coordinate
            if self.throw_type in ("target", "gimmick", "super-gimmick"):
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
                    # print(f"Fitness of {target} target: {fitness}")

                # Calculating mean square of all simulation fitness values
                total_fitness = 0
                for _ in fitnesses:
                    total_fitness += _**2
                genome.fitness = total_fitness**0.5
                genome.fitness *= -1
                # print(f"Total fitness {genome.fitness}/ len(fitness) {len(fitnesses)}")
                # genome.fitness = total_fitness / len(fitnesses)
                print(f"Genome fitness: {genome.fitness}")

            # Ending simulation on demand ------------------------------
            if self.stop:
                self.misc(winner=genome)
                return

            # Monitoring best fitness
            if genome.fitness >= self.best_fitness:
                self.best_genome = genome  # Stores the current best net
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
            if self.throw_type != "far":
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

                    # with gzip.open(self.acceptable_filename, 'w', compresslevel=5) as f:
                    #     pickle.dump(net, f, protocol=pickle.HIGHEST_PROTOCOL)

                    with open(self.acceptable_filename, 'wb') as f:
                        pickle.dump(net, f)

                    self.acceptable_solution_distance = simulation.error_sum[0]
                    self.acceptable_solution_time_of_throw = simulation.error_sum[2]
                    self.acceptable_solution_total_work_sum = simulation.error_sum[3]

            if self.iteration % self.sol_per_pop == 0:
                if (self.generation + 1) <= self.num_generations:
                    self.generation += 1  # Keeping track of the number of generations
                    self.fitness_change.append(self.best_fitness)
                    # Updating plot
                    self.ax.clear()
                    # self.ax.plot([_ for _ in range(len(self.fitness_change))], self.fitness_change)
                    self.ax.plot(self.fitness_change[1:])
                    # self.ax.set_xlabel("Generation")
                    # self.ax.set_ylabel("Fitness value")
                    self.ax.set_xlabel("Pokolenie")
                    self.ax.set_ylabel("Wartosc funkcji dopasowania")
                    plt.draw()
                    plt.pause(0.001)
                    # Drawing plot
                    # self.draw_plot(x=[_ for _ in range(len(self.fitness_change))], y=self.fitness_change)

            # Displayed in training progress window
            self.num_of_generation_label.config(text=f"Current generation: {self.generation}")
            # self.num_of_generation_label.config(text=f"Obecne pokolenie: {self.generation}")
            # Displayed in training progress window
            self.fitness_label.config(
                text=f"Highest achieved fitness so far: {round(self.best_fitness, 3)}"
                # text=f"Najwyzsza osiagnieta wartosc funkcji dopasowania: {round(self.best_fitness, 3)}"
            )
            self.progress_window.update()

            self.iteration += 1

            # Ending training on demand and saving best network so far
            if self.finish_and_save and self.generation > 1 and not self.stop:
                self.stop = True  # Stops recursion of course
                self.elapsed_time = time.time() - self.t0  # Total time of training
                self.misc(winner=self.best_genome)
                return

    def run(self, config_file):
        # print("BBBBBBBBBBBBBB")
        # Load configuration.
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(self.config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        p.add_reporter(self.stats)
        p.add_reporter(neat.Checkpointer(generation_interval=1000))

        # Run for a set amount of generations.
        try:
            winner = p.run(self.eval_genomes, self.num_generations)
        except TypeError:
            return
        # Once training finished
        self.elapsed_time = time.time() - self.t0

        self.misc(winner=winner)

        # --------------------------------------------------------------------------------------------------------------

    def misc(self, winner):
        """Takes care of post-training clarity of data, saving it to file etc."""
        # Display the winning genome.
        print("\nBest genome:\n{!s}".format(winner))

        winner_net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        # filename = f"{uuid.uuid4().hex}"
        # timestring = time.strftime("%Y-%m-%d---%H-%M-%S")
        # filename = "{0}/{1}-{2}".format(self.directory, filename, timestring)
        #
        # stats_name = "{0}-stats".format(filename)
        # Save stats
        with open(self.stats_name, "wb") as f:
            pickle.dump(self.stats, f)

        # with gzip.open(filename, 'w', compresslevel=5) as f:
            # pickle.dump(winner_net, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Save net
        with open(self.best_filename, "wb") as f:
            pickle.dump(winner, f)

            best_solution_sim = Simulation(
                net=winner_net,  # winner_net
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

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        if self.throw_type == "multi-target":  # "desired ball x cor"
            node_names = {
                -5 - self.number_of_links * 2: "ball y velocity",
                -4 - self.number_of_links * 2: "ball x velocity",
                -3 - self.number_of_links * 2: "ball y cor",
                -2 - self.number_of_links * 2: "ball x cor",
                -1 - self.number_of_links * 2: "desired ball x cor",
            }
        else:
            node_names = {
                -4 - self.number_of_links * 2: "ball y velocity",
                -3 - self.number_of_links * 2: "ball x velocity",
                -2 - self.number_of_links * 2: "ball y cor",
                -1 - self.number_of_links * 2: "ball x cor",
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
        if self.throw_type in ("gimmick", "super-gimmick"):
            node_names[self.number_of_links] = "start moving timestamp"
        # Timestamp of robotic gripper opening is always the last output
        if self.gripper_type == "robotic" and not self.throw_type in ("gimmick", "super-gimmick"):
            node_names[self.number_of_links] = "gripper opening timestamp"
        if self.gripper_type == "robotic" and self.throw_type in ("gimmick", "super-gimmick"):
            node_names[self.number_of_links + 1] = "gripper opening timestamp"

        # if self.throw_type == "multi-target":  # "desired ball x cor"
        #     node_names = {
        #         -5 - self.number_of_links * 2: "Predkosc y pilki",
        #         -4 - self.number_of_links * 2: "Predkosc x pilki",
        #         -3 - self.number_of_links * 2: "Wspolrzednia y pilki",
        #         -2 - self.number_of_links * 2: "Wspolrzednia x pilki",
        #         -1 - self.number_of_links * 2: "Zadana wspolrzedna x",
        #     }
        # else:
        #     node_names = {
        #         -4 - self.number_of_links * 2: "Predkosc y pilki",
        #         -3 - self.number_of_links * 2: "Predkosc x pilki",
        #         -2 - self.number_of_links * 2: "Wspolrzednia y pilki",
        #         -1 - self.number_of_links * 2: "Wspolrzednia x pilki",
        #     }
        # i = - self.number_of_links * 2
        # j = 1
        # for _ in range(0, self.number_of_links):
        #     node_names[i] = f"Obecny kat {j}"
        #     i += 1
        #     j += 1
        # i = - self.number_of_links
        # j = 1
        # for _ in range(0, self.number_of_links):
        #     node_names[i] = f"Poprzedni kat {j}"
        #     i += 1
        #     j += 1
        #
        # j = 1
        # for _ in range(0, self.number_of_links):
        #     node_names[_] = f"Moment sily {j}"
        #     i += 1
        #     j += 1
        # if self.throw_type in ("gimmick", "super-gimmick"):
        #     node_names[self.number_of_links] = "Chwila poczatku ruchu"
        # # Timestamp of robotic gripper opening is always the last output
        # if self.gripper_type == "robotic" and not self.throw_type in ("gimmick", "super-gimmick"):
        #     node_names[self.number_of_links] = "Chwila otworzenia chwytaka"
        # if self.gripper_type == "robotic" and self.throw_type in ("gimmick", "super-gimmick"):
        #     node_names[self.number_of_links + 1] = "Chwila otworzenia chwytaka"

        # for key in node_names:
        #     print(f"{key}: {node_names[key]}")

        visualize.draw_net(self.config, winner, True, node_names=node_names)
        # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
        visualize.plot_stats(self.stats, ylog=False, view=True)
        visualize.plot_species(self.stats, view=True)

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
                "Elapsed time": [self.elapsed_time],
                "Best trained network": [self.filename],
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
                "net_path": [self.best_filename],  # Path to the net with the best solution
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

        self.training_finished = True  # Signaling to neat menu that training is completed
        self.queue.put("FINISHED")
        return

        # --------------------------------------------------------------------------------------------------------------

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
        """I only use it once and I don't even have to use it here but that didn't prevent me from wasting 2 hours
         figuring this out"""
        if line.replace(" ", "")[:len(value)] == value:
            value = line.replace(" ", "")[len(value) + 1:-1]
            self.values.append(value)
