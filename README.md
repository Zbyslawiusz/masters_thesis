The aim of this project (which is my master's thesis) was to implement machine learning control for throwing an object using a multi-link manipulator in a 2D simulation.
The following reinforcment learning algorithms were used: genetic algorithm from pygad module and NEAT algorithm from neat-python module.
Tkinter module allowed writing a GUI to simplify experimentation process and result analisys. Results and settings are stored in .csv and .json files thanks to pandas module.
Basic multiprocessing has been implemented to allow training multiple instances of chosen algorithm simultaneusly.

Genetic algorithm operates within an open control structure with no feedback. Meanwhile NEAT algorithm produces networks which move links of the manipulator with direct control over torque thanks to feedback from the simulation.

To use the project you have to install graphviz on your operating system. Other modules listed in the files can be easily installed with PyCharm.

The file named arbiter.py launches the genetic algorith app, while neat-arbiter.py fires up GUI for the NEAT algorithm.
Upon fresh run of the programmes and absence of .csv and .json files all training parameters have to be specified manually according to pygad or neat-python documentation.
