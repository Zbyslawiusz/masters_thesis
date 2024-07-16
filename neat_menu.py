import tkinter as tk
import os
import uuid
import time
import gzip
import pickle
import neat
import visualize

import matplotlib.pyplot as plt
import pandas as pd

from main_neat import Simulation


class Menu:
    def __init__(self):

        self.take_neat_params = 0  # Whether to use last used neat params (0) or the ones from the csv file (1) or user input (2)
        self.take_fitness_params = 0  # Whether to use last used fitness params (0) or the ones from the csv file (1) or user input (2)
        self.csv_exists = False
        self.json_exists = False

        self.num_of_solutions = 0
        self.view_cap = 5
        self.currently_viewed = []
        self.highlighted = 0
        # self.main_func()

        self.searched_values = []

        with open("config-feedforward", "r") as file:
            self.content = file.readlines()

        self.values = []

    def search_function(self, line, value):
        if line.replace(" ", "")[:len(value)] == value:
            value = line.replace(" ", "")[len(value) + 1:-1]
            self.values.append(value)

    # @staticmethod  # TypeError: NeatAlgorithm.neat_start() takes 1 positional argument but 2 were given
    def modify_function(self, line, value, desired_value):
        if line.replace(" ", "")[:len(value)] == value:
            pos = line.find("=")
            new_line = line[:pos + 1]
            new_line += f" {desired_value}\n"
            # print(new_line)
            return new_line
        else:
            return line

    def main_func(self, param_queue):
        self.queue = param_queue
        # Creating the main window\
        self.window = tk.Tk()
        self.window.title("Ball Thrower x3000")
        self.window.config(padx=50, pady=50, bg="white")

        self.main_label = tk.Label(text="Simulate throw or begin NEAT training", font=("Consolas", 64, "bold"))
        self.main_label.grid(row=0, column=0, columnspan=3)

        self.info_label = tk.Label(text="", font=("Consolas", 15, "bold"))
        self.info_label.grid(row=1, column=0)

        # Checking if appropriate files exist --------------------------------------------------------------------------
        self.check_files()  # Has to be called after creation of self.info_label
        # --------------------------------------------------------------------------------------------------------------

        # Creating buttons
        # Cycle up button
        self.up_button = (tk.Button(text="Cycle up", command=self.up)
                          .grid(row=1, column=1))
        # Cycle down button
        self.down_button = (tk.Button(text="Cycle down", command=self.down)
                            .grid(row=1, column=2))
        # Start simulation from selected (highlighted) experiment
        self.start_sim_button = (tk.Button(text="START", font=("Consolas", 15, "bold"),
                                           command=lambda: self.start_sim(False))
                                 .grid(row=2, column=1))
        self.start_sim_with_picks_button = (tk.Button(text="START & screenshots", font=("Consolas", 15, "bold"),
                                                      command=lambda: self.start_sim(True))
                                            .grid(row=3, column=1))
        # Close GUI button, shuts down the programme
        self.shutdown_button = (tk.Button(text="SHUTDOWN", font=("Consolas", 15, "bold"), command=self.shutdown)
                                .grid(row=4, column=1))

        # Start fresh training button with a popup window
        self.start_training_button = (tk.Button(text="Start Fresh Training", font=("Consolas", 15, "bold"),
                                                command=lambda: self.new_train_window_func(df_from_file=None,
                                                                                      mode=None))
                                      .grid(row=2, column=2))

        # Start training from selected experiment button with a popup window
        if self.csv_exists:
            self.start_training_selected_button = (tk.Button(text="Start Training From Selected",
                                                             font=("Consolas", 15, "bold"),
                                                             command=lambda: self.new_train_window_func(
                                                                 df_from_file=self.solutions_file,
                                                                 mode="csv"))
                                                   .grid(row=3, column=2))

        # Start training with las used settings button with a popup window
        if self.json_exists:
            self.start_training_selected_button = (tk.Button(text="Start Training With Las Used Settings",
                                                             font=("Consolas", 15, "bold"),
                                                             command=lambda: self.new_train_window_func(
                                                                 df_from_file=self.settings_file,
                                                                 mode="json"))
                                                   .grid(row=4, column=2))

        if self.csv_exists:
            # Creating a list of labels to display contents of the solution.csv file
            if self.num_of_solutions <= self.view_cap:  # Limiting the amount of labels created
                n = self.num_of_solutions
                print(n)
            else:
                n = self.view_cap
                print(n)

            self.currently_viewed = [_ for _ in range(0, n)]  # List that tells which solutions are viewed at the moment
            self.label_list = [tk.Label(text="", font=("Consolas", 10, "bold")) for _ in range(0, n)]  # List o labels

            self.list_refresh()

        self.window.mainloop()  # has to be here at the end because tkinter says so

    def shutdown(self):
        try:
            plt.close(self.fitness_plot[0].figure)
        except AttributeError:
            pass
        finally:
            self.queue.put("SHUTDOWN")
            self.close_window(self.window)

    def check_files(self):
        """Checks existence of settings and solutions files. Displays results on main window's info_label."""
        # Checking for file containing results of experiments
        try:
            self.solutions_file = pd.read_csv("solution_neat.csv")
        except FileNotFoundError:
            # pass
            self.csv_exists = False
        else:
            self.csv_exists = True
            self.num_of_solutions = len(
                self.solutions_file)  # Checking how many recorded experiments are stored in the file
            self.max_index = self.num_of_solutions - 1  # Determining maximum index for a row in a column
        # Checking for file containing las used GA and fitness params
        try:
            self.settings_file = pd.read_json("settings_neat.json")
        except FileNotFoundError:
            # pass
            self.json_exists = False
        else:
            self.json_exists = True

        if not self.csv_exists and not self.json_exists:
            self.info_label.config(text="No solution or settings files found. This is a fresh run of the program.")
        if not self.csv_exists and self.json_exists:
            self.info_label.config(text="No solution file was found. Settings file with last used params found.")
        if self.csv_exists and not self.json_exists:
            self.info_label.config(text="No settings file was found. You can recreate experiments.")
        if self.csv_exists and self.json_exists:
            self.info_label.config(text="Solution and settings files found. You are free to experiment.")

        # Folders for storing simulation screenshots and trained neat models
        paths = ["./Pymunk_pics/Acceptable_sim", "./Pymunk_pics/Best_sim", "./neat-models"]
        # Checking if the specified path exists or not
        for path in paths:
            if not os.path.exists(path):
                # Creating a new directory if it does not exist
                os.makedirs(path)
                print("The new directory is created!")

    def up(self):
        """Cycles list of labels up"""
        if self.highlighted - 1 >= 0:
            self.highlighted -= 1
            if self.highlighted < self.currently_viewed[0]:
                self.currently_viewed = [_ - 1 for _ in self.currently_viewed]
        self.check_files()
        self.list_refresh()
        # print(self.currently_viewed)

    def down(self):
        """Cycles list of labels down"""
        self.list_refresh()
        if self.highlighted + 1 <= self.max_index:
            self.highlighted += 1
            if self.highlighted > self.currently_viewed[-1]:
                self.currently_viewed = [_ + 1 for _ in self.currently_viewed]
        self.check_files()
        self.list_refresh()
        # print(self.currently_viewed)

    def list_refresh(self):
        """Refreshes the list of labels displayed in the main window"""
        # self.check_files()
        self.max_index = self.num_of_solutions - 1  # Determining maximum index for a row in a column
        i = 0
        # print(self.highlighted)
        if self.num_of_solutions <= self.view_cap:  # Limiting the amount of labels created
            n = self.num_of_solutions
            # print(n)
        else:
            n = self.view_cap
            # print(n)
        self.label_list = [tk.Label(text="", font=("Consolas", 10, "bold")) for _ in range(0, n)]  # List o labels
        if len(self.currently_viewed) < n:
            self.currently_viewed = [_ for _ in range(0, n)]  # List that tells which solutions are viewed at the moment
        for label in self.label_list:
            num = self.currently_viewed[i]
            if num == self.highlighted:
                label.config(background="green")
            else:
                label.config(background="gray")
            label.config(text=f"Experiment number {num + 1}. Title: {self.solutions_file['title'][num]}.\n"
                              f"Best fitness: "
                              f"{round(float((self.solutions_file['Fitness value of the best solution'][num])), 3)}/"
                              f"{self.solutions_file['max_fitness'][num]}, type of throw: "
                              f"{self.solutions_file['throw_type'][num]}\n "
                              f"smallest distance error: {round(float((self.solutions_file['Best solution distance'][num])), 3)}, "
                              f"time: {round(float((self.solutions_file['Best solution time of throw'][num])), 3)},\n "
                              f"best work sum: {round(float((self.solutions_file['Best solution total work sum'][num])), 3)}, "
                              f"number or movable links: {round(float((self.solutions_file['Num of movable links'][num])), 3)}, "
                              f"target x coordinate {(self.solutions_file['target_xcor'][num])}")
            label.grid(row=1+i+1, column=0, columnspan=2)

            # label.config(text=f"Experiment number {num + 1}. ")
            # label.grid(row=1 + i + 1, column=0, columnspan=2)

            i += 1

    def start_sim(self, picks):
        """Starts a simulation based on the highlighted solution from the list of labels.
        First simulation depicts the acceptable solution if it exists.
        Second simulation shows the best solution."""
        # Converting strings to lists and then list's contents from strings to floats
        # acceptable_solution = self.solutions_file["Acceptable solution"][self.highlighted][1:-1].split(sep=",")
        # acceptable_solution = [float(_) for _ in acceptable_solution]
        # best_solution = self.solutions_file["Best solution"][self.highlighted][1:-1].split(sep=",")
        # best_solution = [float(_) for _ in best_solution]
        net_path = self.solutions_file["net_path"][self.highlighted]
        # print(net_path)
        config_directory = os.path.dirname(net_path)
        # print(config_directory)
        config_path = os.path.join(config_directory, "config")
        # print(config_path)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        # stats = neat.StatisticsReporter()
        stats_path = "{0}-stats".format(net_path)
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        # Restoring checkpoint of the acceptable solution if it has been reached
        print(self.solutions_file["Fitness value of the acceptable solution"][self.highlighted])
        print(type(self.solutions_file["Fitness value of the acceptable solution"][self.highlighted]))
        if (self.solutions_file["Fitness value of the acceptable solution"][self.highlighted] != False
            and self.solutions_file["throw_type"][self.highlighted] != "far"
                and self.solutions_file["Acceptable solution time"][self.highlighted] != False):
            # with gzip.open(net_path) as f:
            # with gzip.open(net_path, "rb") as f:
            #     net = pickle.load(f)
            with open(net_path, "rb") as f:
                net = pickle.load(f)
            neat_net = neat.nn.FeedForwardNetwork.create(net, config)

            if self.solutions_file["throw_type"][self.highlighted] == "multi-target":
                targets = [1300, 2150, 2900, 3350, 4200, float(self.solutions_file["target_xcor"][self.highlighted])]
                for x_cor in targets:
                    print(f"Acceptable solution time of throw: "
                          f"{self.solutions_file["Acceptable solution time of throw"][self.highlighted]}")
                    minimum_solution_sim = Simulation(
                        net=neat_net,
                        ui_flag=True,
                        number_of_links=int(self.solutions_file["Num of movable links"][self.highlighted]),
                        target_xcor=x_cor,
                        gripper=self.solutions_file["gripper_type"][self.highlighted],
                        time_of_throw=float(self.solutions_file["Acceptable solution time of throw"][self.highlighted]),
                        picks_or_not=picks,
                        sim_type="acceptable",
                        throw_type=self.solutions_file["throw_type"][self.highlighted]
                    )
            else:
                print(f"Acceptable solution time of throw: "
                      f"{self.solutions_file["Acceptable solution time of throw"][self.highlighted]}")
                minimum_solution_sim = Simulation(
                    net=neat_net,
                    ui_flag=True,
                    number_of_links=int(self.solutions_file["Num of movable links"][self.highlighted]),
                    target_xcor=float(self.solutions_file["target_xcor"][self.highlighted]),
                    gripper=self.solutions_file["gripper_type"][self.highlighted],
                    time_of_throw=float(self.solutions_file["Acceptable solution time of throw"][self.highlighted]),
                    picks_or_not=picks,
                    sim_type="acceptable",
                    throw_type=self.solutions_file["throw_type"][self.highlighted]
                )

            number_of_links = int(self.solutions_file["Num of movable links"][self.highlighted])
            self.visualise_net(config=config, net=net, stats=stats, number_of_links=number_of_links)

        # Restoring checkpoint of the best solution
        # with gzip.open(net_path) as f:
        # with gzip.open(net_path, "rb") as f:
        #     net = pickle.load(f)

        with open(net_path, "rb") as f:
            net = pickle.load(f)
        neat_net = neat.nn.FeedForwardNetwork.create(net, config)

        if self.solutions_file["throw_type"][self.highlighted] == "multi-target":
            targets = [1300, 2150, 2900, 3350, 4200, float(self.solutions_file["target_xcor"][self.highlighted])]
            for x_cor in targets:
                print(f"Best solution time of throw: "
                      f"{self.solutions_file["Best solution time of throw"][self.highlighted]}")
                best_solution_sim = Simulation(
                    net=neat_net,
                    ui_flag=True,
                    number_of_links=int(self.solutions_file["Num of movable links"][self.highlighted]),
                    target_xcor=x_cor,
                    gripper=self.solutions_file["gripper_type"][self.highlighted],
                    time_of_throw=float(self.solutions_file["Best solution time of throw"][self.highlighted]),
                    picks_or_not=picks,
                    sim_type="best",
                    throw_type=self.solutions_file["throw_type"][self.highlighted]
                )
        else:
            print(f"Best solution time of throw: "
                  f"{self.solutions_file["Best solution time of throw"][self.highlighted]}")
            best_solution_sim = Simulation(
                net=neat_net,
                ui_flag=True,
                number_of_links=int(self.solutions_file["Num of movable links"][self.highlighted]),
                target_xcor=float(self.solutions_file["target_xcor"][self.highlighted]),
                gripper=self.solutions_file["gripper_type"][self.highlighted],
                time_of_throw=float(self.solutions_file["Best solution time of throw"][self.highlighted]),
                picks_or_not=picks,
                sim_type="best",
                throw_type=self.solutions_file["throw_type"][self.highlighted]
            )

        number_of_links = int(self.solutions_file["Num of movable links"][self.highlighted])
        self.visualise_net(config=config, net=net, stats=stats, number_of_links=number_of_links)

        best_solution_sim = Simulation(
            net=neat_net,
            ui_flag=False,
            number_of_links=int(self.solutions_file["Num of movable links"][self.highlighted]),
            target_xcor=float(self.solutions_file["target_xcor"][self.highlighted]),
            gripper=self.solutions_file["gripper_type"][self.highlighted],
            time_of_throw=float(self.solutions_file["Best solution time of throw"][self.highlighted]),
            picks_or_not=False,
            sim_type="best",
            throw_type=self.solutions_file["throw_type"][self.highlighted]
        )

        generations = [_ for _ in range(0, self.solutions_file["num_generations"][self.highlighted])]
        # Converting list turned into a string into a list of floats
        fitness_change = self.solutions_file["fitness_change"][self.highlighted][1:-1].split(sep=",")
        fitness_change = [float(_) for _ in fitness_change]

        self.fitness_plot = plt.plot([_ for _ in range(len(fitness_change)-1)], fitness_change[1:])
        plt.clf()
        self.fitness_plot = plt.plot([_ for _ in range(len(fitness_change)-1)], fitness_change[1:])
        # plt.xlabel("Generation")
        # plt.ylabel("Fitness value")
        plt.xlabel("Pokolenie")
        plt.ylabel("Wartosc funkcji dopasowania")
        # plt.title("Fitness vaule plot")
        plt.title("Przebieg wartosci funkcji dopasowania")
        plt.show()

        return None

    def new_train_window_func(self, df_from_file, mode):
        """Popup window with user input for all fitness and neat parameters used during training.
        Allows for manually typing values or correcting those used in previous experiments."""
        self.new_train_window = tk.Tk()
        self.new_train_window.title("Specify parameters")
        self.new_train_window.config(padx=25, pady=25, bg="white")

        info_label = (tk.Label(self.new_train_window, text="Fill in the values or modify existing ones",
                               font=("Consolas", 20, "bold")))
        info_label.grid(row=0, column=0, columnspan=4)  # Has to be in new line or else Label.config() will not work

        fitness_label = (tk.Label(self.new_train_window, text="Fitness function params", font=("Consolas", 20, "bold"))
                         .grid(row=1, column=0, columnspan=2))
        neat_label = (tk.Label(self.new_train_window, text="NEAT algorithm params", font=("Consolas", 20, "bold"))
                    .grid(row=1, column=3, columnspan=2))

        left_text_list = ["Num of movable links", "Target x cor: ", "Max fitness: ", "Distance weight: ",
                          "Time weight: ", "Work sum weight: ", "Collision penalty: ",
                          "Num of training instances: ", "'robotic' or 'stiff' gripper: ",
                          "Title: ", "'target', 'far', 'gimmick',\n 'multi-target', 'super-gimmick' throw: "]

        left_label_list = [(tk.Label(self.new_train_window, text=left_text_list[i], font=("Consolas", 15, "bold"))
                            .grid(row=i+2, column=0))
                           for i in range(0, len(left_text_list))]
        # Left label-entry pairs
        # l1 = (tk.Label(text="Target x cor: ",
        #                font=("Consolas", 15, "bold")).grid(row=1, column=0))
        # e1 = (tk.Entry(new_train_window).grid(row=1, column=1))

        left_entry_list = [(tk.Entry(self.new_train_window, font=("Consolas", 15, "bold")))
                           for i in range(0, len(left_text_list))]
        i = 2
        for entry in left_entry_list:
            entry.grid(row=i, column=1)  # if this command is not done separately, entry.insert will not work
            i += 1

        right_text_list = ["Num of generations: ", "Activation type: ", "Num hidden: ", "Solutions per population: ",
                           "Node add prob: ", "Node delete prob: ", "Response max value: ", "Response min value: ",
                           "Weight mutate power: ", "Weight mutate rate: ", "Connection add prob: ",
                           "Connection delete prob: "]

        right_label_list = [(tk.Label(self.new_train_window, text=right_text_list[i], font=("Consolas", 15, "bold"))
                            .grid(row=i+2, column=2))
                            for i in range(0, len(right_text_list))]

        right_entry_list = [(tk.Entry(self.new_train_window, font=("Consolas", 15, "bold")))
                            for i in range(0, len(right_text_list))]
        i = 2
        for entry in right_entry_list:
            entry.grid(row=i, column=3)
            i += 1

        confirm_button = (tk.Button(self.new_train_window, text="CONFIRM AND START TRAINING", font=("Consolas", 30, "bold"),
                                    command=lambda: self.to_confirm_training(left_entry_list, right_entry_list,
                                                                             label=info_label)))
        confirm_button.grid(row=len(right_label_list)+2, column=0, columnspan=4)

        if isinstance(df_from_file, pd.core.frame.DataFrame):  # Checking if there is a file passed to the function
            # print("Got the file")
            # print(type(file))
            num_of_training_instances = 1
            if mode == "csv":
                index = self.highlighted
                num_of_training_instances = df_from_file["Num_of_training_instances"][index]
            elif mode == "json":
                index = 0
                num_of_training_instances = df_from_file["Num_of_training_instances"][index]
            # Filling in default values for entries based on the contents of the passed file
            left_entry_insert = [df_from_file["Num of movable links"][index], df_from_file["target_xcor"][index],
                                 df_from_file["max_fitness"][index], df_from_file["distance_value"][index],
                                 df_from_file["time_value"][index], df_from_file["work_sum_value"][index],
                                 df_from_file["penalty_col"][index],
                                 num_of_training_instances, df_from_file["gripper_type"][index],
                                 "Title", df_from_file["throw_type"][index]]

            for i in range(0, len(left_entry_list)):
                left_entry_list[i].insert(-1, left_entry_insert[i])

            right_entry_insert = [df_from_file["num_generations"][index], df_from_file["activation_type"][index],
                                  df_from_file["num_hidden"][index], df_from_file["sol_per_pop"][index],
                                  df_from_file["node_add_prob"][index], df_from_file["node_delete_prob"][index],
                                  df_from_file["response_max_value"][index], df_from_file["response_min_value"][index],
                                  df_from_file["weight_mutate_power"][index], df_from_file["weight_mutate_rate"][index],
                                  df_from_file["conn_add_prob"][index], df_from_file["conn_delete_prob"][index]]

            for i in range(0, len(right_entry_insert)):
                right_entry_list[i].insert(-1, right_entry_insert[i])

        else:
            print("No file")

        # BACKUP THAT WORKED WITH **kwargs
        # if kwargs["mode"] == "csv":
        #     print("CSV is chosen")
        #     print(type(kwargs["file"]))
        # elif kwargs["mode"] == "json":
        #     print("Json is chosen")
        #     print(type(kwargs["file"]))
        #     print(isinstance(kwargs["file"], pd.core.frame.DataFrame))
        # elif kwargs["mode"] == "fresh":
        #     print(type(kwargs["file"]))

    def to_confirm_training(self, *entry_lists, label):
        """The purpose of this method is to simply revert info_label from new_train_window back to its original state
        preventing it from becoming stuck at 'Wrong type of value encountered. Correct your input.'
        even after values have been corrected."""
        label.config(text="Fill in the values or modify existing ones")
        self.confirm_training(entry_lists[0], entry_lists[1], label=label)

    def confirm_training(self, *entry_lists, label):
        """This method converts values in entries from new_train_window to appropriate types.
        It is here where info_label from new_train_window is changed if the type of value is wrong.
        Converted values are then sent to neat-python to begin training.
        A popup window indicates the progress of training."""
        wrong_text = "Wrong type of value encountered. Please correct your input."
        errors_detected = False

        # Getting values from left entry widgets
        fitness_params = {
            "Num of movable links": entry_lists[0][0].get(),
            "target_xcor": entry_lists[0][1].get(),
            "max_fitness": entry_lists[0][2].get(),
            "distance_value": entry_lists[0][3].get(),
            "time_value": entry_lists[0][4].get(),
            "work_sum_value": entry_lists[0][5].get(),
            "penalty_col": entry_lists[0][6].get(),
            # "penalty_angle": entry_lists[0][7].get(),
            "Num_of_training_instances": entry_lists[0][7].get(),
            "gripper_type": entry_lists[0][8].get(),
            "title": entry_lists[0][9].get(),
            "throw_type": entry_lists[0][10].get(),
        }

        # Getting values from left entry widgets
        neat_params = {
            "num_generations": entry_lists[1][0].get(),
            "activation_type": entry_lists[1][1].get(),
            "num_hidden": entry_lists[1][2].get(),
            "sol_per_pop": entry_lists[1][3].get(),
            "node_add_prob": entry_lists[1][4].get(),
            "node_delete_prob": entry_lists[1][5].get(),
            "response_max_value": entry_lists[1][6].get(),
            "response_min_value": entry_lists[1][7].get(),
            "weight_mutate_power": entry_lists[1][8].get(),
            "weight_mutate_rate": entry_lists[1][9].get(),
            "conn_add_prob": entry_lists[1][10].get(),
            "conn_delete_prob": entry_lists[1][11].get(),
        }

        # Converting values from dictionaries to appropriate types
        i = 0
        for (key, param) in fitness_params.items():
            if i in (0, 2, 7):
                try:
                    fitness_params[key] = int(param)  # To integers
                except ValueError:
                    errors_detected = True
                    label.config(text=wrong_text)
                    fitness_params[key] = "ERROR"
            elif i in (8, 9, 10):  # Gripper type is a string
                pass
            else:
                try:
                    fitness_params[key] = float(param)  # To floats
                except ValueError:
                    errors_detected = True
                    label.config(text=wrong_text)
                    fitness_params[key] = "ERROR"
            i += 1
        # print(fitness_params)

        i = 0
        for (key, param) in neat_params.items():
            if i in (0, 2, 3):
                try:
                    neat_params[key] = int(param)  # To integers
                except ValueError:
                    errors_detected = True
                    label.config(text=wrong_text)
                    neat_params[key] = "ERROR"
            elif i == 1:  # For strings
                pass
            else:
                try:
                    neat_params[key] = float(param)  # To floats
                except ValueError:
                    errors_detected = True
                    label.config(text=wrong_text)
                    neat_params[key] = "ERROR"
            i += 1
        # print(ga_params)

        # TESTING
        # self.queue.put({"fitness_params": fitness_params, "ga_params": ga_params})
        if errors_detected:
            return
        else:
            with open("config-feedforward", "r") as file:
                content = file.readlines()
                print("reading config")

            # if fitness_params["gripper_type"] == "robotic":
            #     num_outputs = fitness_params["Num of movable links"] + 1
            # else:
            #     num_outputs = fitness_params["Num of movable links"]

            modify_values = {
                "fitness_threshold": fitness_params["max_fitness"],
                "pop_size": neat_params["sol_per_pop"],
                "activation_default": neat_params["activation_type"],
                "num_hidden": neat_params["num_hidden"],
                "num_inputs": fitness_params["Num of movable links"] * 2 + 4,
                "num_outputs": fitness_params["Num of movable links"],
                "node_add_prob": neat_params["node_add_prob"],
                "node_delete_prob": neat_params["node_delete_prob"],
                "response_max_value": neat_params["response_max_value"],
                "response_min_value": neat_params["response_min_value"],
                "weight_mutate_power": neat_params["weight_mutate_power"],
                "weight_mutate_rate": neat_params["weight_mutate_rate"],
                "conn_add_prob": neat_params["conn_add_prob"],
                "conn_delete_prob": neat_params["conn_delete_prob"],
            }

            # Correcting max fitness values for gimmick and super-gimmick types of throw
            if fitness_params["throw_type"] == "gimmick":
                fitness_params["max_fitness"] = 1418.75
            if fitness_params["throw_type"] == "super-gimmick":
                fitness_params["max_fitness"] = 1220

            # To allow NEAT to learn up to desired generation
            if fitness_params["throw_type"] == "far":
                modify_values["fitness_threshold"] = 1_000_000_000
            # Correcting max fitness values for gimmick and super-gimmick types of throw
            if fitness_params["throw_type"] == "gimmick":
                modify_values["fitness_threshold"] = 1418.75
            if fitness_params["throw_type"] == "super-gimmick":
                modify_values["fitness_threshold"] = 1220
            # Adding output for gripper claws release timestamp
            if fitness_params["gripper_type"] == "robotic":
                modify_values["num_outputs"] += 1
            # Adding output for timestamp of links starting to move
            if fitness_params["throw_type"] in ("gimmick", "super-gimmick"):
                modify_values["num_outputs"] += 1
            # Adding input for target x cor in case of multi-target type of throw
            if fitness_params["throw_type"] == "multi-target":
                modify_values["num_inputs"] += 1
                modify_values["fitness_threshold"] = 0
                fitness_params["max_fitness"] = 0

            timestring = time.strftime("%Y-%m-%d---%H-%M-%S")
            unique_name = uuid.uuid4().hex
            foldername = "./neat-models/{0}-{1}".format(unique_name, timestring)
            os.mkdir(foldername)
            filename = "{0}/config".format(foldername)
            fitness_params["foldername"] = foldername

            with open(filename, "w") as file:
                i = 0
                for line in content:
                    for value in modify_values:
                        if line.replace(" ", "")[:len(value)] == value:
                            pos = line.find("=")
                            new_line = line[:pos + 1]
                            new_line += f" {modify_values[value]}\n"
                            # print(new_line)
                            content[i] = new_line
                        else:
                            pass
                        # print(line.replace(" ", "")[:len(value)])
                        # if line.replace(" ", "")[:len(value)] == value:
                        #     content[i] = self.modify_function(line, value, modify_values[value])
                    # print(content[i])
                    i += 1
                print(fitness_params)
                print(content)
                # return  # For debugging
                file.writelines(content)

            self.queue.put({"fitness_params": fitness_params, "neat_params": neat_params})
            self.close_window(self.new_train_window)
        #
        #     progress_window.after(1000)
        #
        #     progress_window.mainloop()

    def close_window(self, window):
        """This method closes passed popup window"""
        window.destroy()

    def visualise_net(self, config, net, stats, number_of_links):
        # winner_net = neat.nn.FeedForwardNetwork.create(net, config)
        print('\nBest genome:\n{!s}'.format(net))

        if self.solutions_file["throw_type"][self.highlighted] == "multi-target":
            node_names = {
                -5 - number_of_links * 2: "desired ball x cor",
                -4 - number_of_links * 2: "ball y velocity",
                -3 - number_of_links * 2: "ball x velocity",
                -2 - number_of_links * 2: "ball y cor",
                -1 - number_of_links * 2: "ball x cor",
            }
        else:
            node_names = {
                -4 - number_of_links * 2: "ball y velocity",
                -3 - number_of_links * 2: "ball x velocity",
                -2 - number_of_links * 2: "ball y cor",
                -1 - number_of_links * 2: "ball x cor",
            }
        i = - number_of_links * 2
        j = 1
        for _ in range(0, number_of_links):
            node_names[i] = f"current angle {j}"
            i += 1
            j += 1
        i = - number_of_links
        j = 1
        for _ in range(0, number_of_links):
            node_names[i] = f"previous angle {j}"
            i += 1
            j += 1

        j = 1
        for _ in range(0, number_of_links):
            node_names[_] = f"motor torque {j}"
            i += 1
            j += 1
        if self.solutions_file["throw_type"][self.highlighted] in ("gimmick", "super-gimmick"):
            node_names[number_of_links] = "start moving timestamp"
        # Timestamp of robotic gripper opening is always the last output
        if (self.solutions_file["gripper_type"][self.highlighted] == "robotic" and
                not self.solutions_file["throw_type"][self.highlighted] in ("gimmick", "super-gimmick")):
            node_names[number_of_links] = "gripper opening timestamp"
        if (self.solutions_file["gripper_type"][self.highlighted] == "robotic" and
                self.solutions_file["throw_type"][self.highlighted] in ("gimmick", "super-gimmick")):
            node_names[number_of_links + 1] = "gripper opening timestamp"

        # Visualising the best net parameters
        visualize.draw_net(config, net, True, node_names=node_names)
        # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)


main_menu = Menu()
