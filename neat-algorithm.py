"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
import visualize
import pickle
import gzip
import uuid

import time

from main_neat import Simulation

# 2-input XOR inputs and expected outputs.
# xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# xor_outputs = [(0.0,),     (1.0,),     (1.0,),     (0.0,)]

distance_value = 0.95
time_value = 0.5
work_sum_value = 1e-7
penalty_col = 300
penalty_angle = 250


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # for xi, xo in zip(xor_inputs, xor_outputs):
        #     output = net.activate(xi)
        #     genome.fitness -= (output[0] - xo[0]) ** 2
        nuke_fitness = False

        simulation = Simulation(
            net=net,
            ui_flag=False,
            number_of_links=1,
            target_xcor=1600,
            interpolation=4,
            gripper="robotic"
        )
        # fitness = 1.0 / (error_sum + 0.0001)

        # return [registered_distance, hit_obstacle, elapsed_time, work_sum]
        # DOUBLE, BOOLEAN VALUE, DOUBLE, DOUBLE

        genome.fitness = 1600 - (distance_value * simulation.error_sum[0] +
                                 time_value * simulation.error_sum[2] +
                                 work_sum_value * simulation.error_sum[3] +
                                 simulation.error_sum[4] +
                                 simulation.error_sum[5])

        if simulation.error_sum[1]:
            genome.fitness -= penalty_col  # Applying penalty for hitting the obstacle

        if simulation.error_sum[6]:  # Nuke fitness
            genome.fitness = 0


def run(config_file):
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
    winner = p.run(eval_genomes, 2)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # filename = f"{uuid.uuid4().hex}.png"
    timestring = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = './neat-models/{0}-{1}'.format(uuid.uuid4().hex, timestring)
    print("Saving checkpoint to {0}".format(filename))

    with gzip.open(filename, 'w', compresslevel=5) as f:
        pickle.dump(winner_net, f, protocol=pickle.HIGHEST_PROTOCOL)

    # simulation = Simulation(
    #     net=winner_net,
    #     ui_flag=True,
    #     number_of_links=1,
    #     target_xcor=1600,
    #     interpolation=4,
    #     gripper="robotic"
    # )

    # # Save the winner.
    # with open('winner-ctrnn', 'wb') as f:
    #     pickle.dump(winner_net, f)

    # Show output of the most fit genome against training data.
    print('\nOutput:')
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


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
