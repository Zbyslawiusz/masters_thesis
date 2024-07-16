import tkinter as tk

import matplotlib.pyplot as plt
import pandas as pd

from main_051 import Simulation
from pygad_0597 import GeneticAlgorithm


class Menu:
    def __init__(self):

        self.take_ga_params = 0  # Whether to use last used ga params (0) or the ones from the csv file (1) or user input (2)
        self.take_fitness_params = 0  # Whether to use last used fitness params (0) or the ones from the csv file (1) or user input (2)
        self.csv_exists = False
        self.json_exists = False

        self.num_of_solutions = 0
        self.view_cap = 5
        self.currently_viewed = []
        self.highlighted = 0
        # self.main_func()

    def main_func(self, param_queue):
        self.queue = param_queue
        # Creating the main window\
        self.window = tk.Tk()
        self.window.title("Ball Thrower x2000")
        self.window.config(padx=50, pady=50, bg="white")

        self.main_label = tk.Label(text="Simulate throw or begin GA training", font=("Consolas", 64, "bold"))
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
        self.start_sim_button = (tk.Button(text="START", font=("Consolas", 15, "bold"), command=self.start_sim)
                                 .grid(row=2, column=1))
        # Close GUI button, shuts down the programme
        self.shutdown_button = (tk.Button(text="SHUTDOWN", font=("Consolas", 15, "bold"), command=self.shutdown)
                                .grid(row=3, column=1))

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
        finally:
            self.queue.put("SHUTDOWN")
            self.close_window(self.window)

    def check_files(self):
        """Checks existence of settings and solutions files. Displays results on main window's info_label."""
        # Checking for file containing results of experiments
        try:
            self.solutions_file = pd.read_csv("solution.csv")
        except FileNotFoundError:
            pass
        else:
            self.csv_exists = True
            self.num_of_solutions = len(
                self.solutions_file)  # Checking how many recorded experiments are stored in the file
            self.max_index = self.num_of_solutions - 1  # Determining maximum index for a row in a column
        # Checking for file containing las used GA and fitness params
        try:
            self.settings_file = pd.read_json("settings.json")
        except FileNotFoundError:
            pass
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
        i = 0
        # print(self.highlighted)
        if self.num_of_solutions <= self.view_cap:  # Limiting the amount of labels created
            n = self.num_of_solutions
            print(n)
        else:
            n = self.view_cap
            print(n)
        self.label_list = [tk.Label(text="", font=("Consolas", 10, "bold")) for _ in range(0, n)]  # List o labels
        for label in self.label_list:
            num = self.currently_viewed[i]
            if num == self.highlighted:
                label.config(background="green")
            else:
                label.config(background="gray")
            label.config(text=f"Experiment number {num + 1}. "
                              f"Best fitness: "
                              f"{round(float((self.solutions_file["Fitness value of the best solution"][num])), 3)}/"
                              f"{self.solutions_file["max_fitness"][num]},\n "
                              f"best distance: {round(float((self.solutions_file["Best solution distance"][num])), 3)}, "
                              f"time: {round(float((self.solutions_file["Best solution time of throw"][num])), 3)},\n "
                              f"best work sum: {round(float((self.solutions_file["Best solution total work sum"][num])), 3)}, "
                              f"number or movable links: {round(float((self.solutions_file["Num of movable links"][num])), 3)}, "
                              f"target x coordinate {(self.solutions_file["target_xcor"][num])}")
            label.grid(row=1+i+1, column=0, columnspan=2)
            i += 1

    def start_sim(self):
        """Starts a simulation based on the highlighted solution from the list of labels.
        First simulation depicts the acceptable solution if it exists.
        Second simulation shows the best solution."""
        # Converting strings to lists and then list's contents from strings to floats
        acceptable_solution = self.solutions_file["Acceptable solution"][self.highlighted][1:-1].split(sep=",")
        acceptable_solution = [float(_) for _ in acceptable_solution]
        best_solution = self.solutions_file["Best solution"][self.highlighted][1:-1].split(sep=",")
        best_solution = [float(_) for _ in best_solution]

        if self.solutions_file["Fitness value of the acceptable solution"][self.highlighted] != "False":
            minimum_solution_sim = Simulation(
                genetic_solution=acceptable_solution,
                ui_flag=True,
                number_of_links=int(self.solutions_file["Num of movable links"][self.highlighted]),
                target_xcor=float(self.solutions_file["target_xcor"][self.highlighted])
            )

        best_solution_sim = Simulation(
            genetic_solution=best_solution,
            ui_flag=True,
            number_of_links=int(self.solutions_file["Num of movable links"][self.highlighted]),
            target_xcor=float(self.solutions_file["target_xcor"][self.highlighted])
        )

        generations = [_ for _ in range(0, self.solutions_file["num_generations"][self.highlighted])]
        # Converting list turned into a string into a list of floats
        fitness_change = self.solutions_file["fitness_change"][self.highlighted][1:-1].split(sep=",")
        fitness_change = [float(_) for _ in fitness_change]

        self.fitness_plot = plt.plot(generations, fitness_change)
        plt.clf()
        self.fitness_plot = plt.plot(generations, fitness_change)
        plt.xlabel("Generations number")
        plt.ylabel("Fitness value")
        plt.show()

        return None

    def new_train_window_func(self, df_from_file, mode):
        """Popup window with user input for all fitness and ga parameters used during training.
        Allows for manually typing values or correcting those used in previous experiments."""
        self.new_train_window = tk.Tk()
        self.new_train_window.title("Specify parameters")
        self.new_train_window.config(padx=25, pady=25, bg="white")

        info_label = (tk.Label(self.new_train_window, text="Fill in the values or modify existing ones",
                               font=("Consolas", 20, "bold")))
        info_label.grid(row=0, column=0, columnspan=4)  # Has to be in new line or else Label.config() will not work

        fitness_label = (tk.Label(self.new_train_window, text="Fitness function params", font=("Consolas", 20, "bold"))
                         .grid(row=1, column=0, columnspan=2))
        ga_label = (tk.Label(self.new_train_window, text="Genetic algorithm params", font=("Consolas", 20, "bold"))
                    .grid(row=1, column=3, columnspan=2))

        left_text_list = ["Num of movable links", "Target x cor: ", "Max fitness: ", "Distance weight: ",
                          "Time weight: ", "Work sum weight: ", "Collision penalty: ", "Wrong angle penalty: ",
                          "Num of training instances: "]

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

        right_text_list = ["Num of generations: ", "Num of parents mating: ", "Parent selection type: ",
                           "Crossover type: ", "Init range low: ", "Init range high: ", "Random mutation min val: ",
                           "Random mutation max val: ", "Mutation probability: ", "Solutions per population: "]

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
        confirm_button.grid(row=12, column=0, columnspan=4)

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
                                 df_from_file["penalty_col"][index], df_from_file["penalty_angle"][index],
                                 num_of_training_instances,]

            for i in range(0, len(left_entry_list)):
                left_entry_list[i].insert(-1, left_entry_insert[i])

            right_entry_insert = [df_from_file["num_generations"][index], df_from_file["num_parents_mating"][index],
                                  df_from_file["parent_selection_type"][index], df_from_file["crossover_type"][index],
                                  df_from_file["init_range_low"][index], df_from_file["init_range_high"][index],
                                  df_from_file["random_mutation_min_val"][index], df_from_file["random_mutation_max_val"][index],
                                  df_from_file["mutation_probability"][index], df_from_file["sol_per_pop"][index]]

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
        Converted values are then sent to pygad to begin training.
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
            "penalty_angle": entry_lists[0][7].get(),
            "Num_of_training_instances": entry_lists[0][8].get(),
        }

        # Getting values from left entry widgets
        ga_params = {
            "num_generations": entry_lists[1][0].get(),
            "num_parents_mating": entry_lists[1][1].get(),
            "parent_selection_type": entry_lists[1][2].get(),
            "crossover_type": entry_lists[1][3].get(),
            "init_range_low": entry_lists[1][4].get(),
            "init_range_high": entry_lists[1][5].get(),
            "random_mutation_min_val": entry_lists[1][6].get(),
            "random_mutation_max_val": entry_lists[1][7].get(),
            "mutation_probability": entry_lists[1][8].get(),
            "sol_per_pop": entry_lists[1][9].get(),
        }

        # Converting values from dictionaries to appropriate types
        i = 0
        for (key, param) in fitness_params.items():
            if i in (0, 8):
                try:
                    fitness_params[key] = int(param)
                except ValueError:
                    errors_detected = True
                    label.config(text=wrong_text)
                    fitness_params[key] = "ERROR"
            else:
                try:
                    fitness_params[key] = float(param)
                except ValueError:
                    errors_detected = True
                    label.config(text=wrong_text)
                    fitness_params[key] = "ERROR"
            i += 1
        # print(fitness_params)

        i = 0
        for (key, param) in ga_params.items():
            if i in (0, 1, 9):
                try:
                    ga_params[key] = int(param)
                except ValueError:
                    errors_detected = True
                    label.config(text=wrong_text)
                    ga_params[key] = "ERROR"
            elif i in (2, 3):
                pass
            else:
                try:
                    ga_params[key] = float(param)
                except ValueError:
                    errors_detected = True
                    label.config(text=wrong_text)
                    ga_params[key] = "ERROR"
            i += 1
        # print(ga_params)

        # TESTING
        # self.queue.put({"fitness_params": fitness_params, "ga_params": ga_params})
        if errors_detected:
            return
        else:
            self.queue.put({"fitness_params": fitness_params, "ga_params": ga_params})
            self.close_window(self.new_train_window)
        #
        #     progress_window.after(1000)
        #
        #     progress_window.mainloop()

    def start_training(self, fitness_params, ga_params):
        """Starts training of the genetic algorithm"""
        training_instance = GeneticAlgorithm(fitness_params=fitness_params, ga_params=ga_params)

    # def training_ui(self, *labels, window, info):
    #     labels[0].config(text=f"Current generation: {info[0]}")
    #     labels[1].config(text=f"Highest achieved fitness so far: {info[1]}")
    #     window.update()

    def close_window(self, window):
        """This method closes passed popup window"""
        window.destroy()


main_menu = Menu()
